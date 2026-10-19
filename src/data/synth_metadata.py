"""
Assigns synthetic video metadata to a flat frame-ID space and writes a DuckDB database.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np


_TAG_VOCAB = [
    "action", "dialogue", "outdoor", "indoor", "night", "day",
    "crowd", "solo", "vehicle", "animal", "water", "fire",
    "music", "silence", "closeup", "wide", "slow", "fast",
    "combat", "comedy",
]
_SCENES_PER_VIDEO_MIN = 3
_SCENES_PER_VIDEO_MAX = 8
_TAGS_PER_FRAME_MAX = 3

@dataclass
class VideoEntry:
    video_id: str
    start_id: int
    end_id: int
    fps: float

class VideoTable:

    def __init__(self, entries: list[VideoEntry]) -> None:
        self._table: dict[str, VideoEntry] = {e.video_id: e for e in entries}
        _sorted = sorted(entries, key=lambda e: e.start_id)
        self._start_ids: list[int] = [e.start_id for e in _sorted]
        self._sorted_entries: list[VideoEntry] = _sorted

    def __len__(self) -> int:
        return len(self._table)

    def get(self, video_id: str) -> Optional[VideoEntry]:
        return self._table.get(video_id)

    def id_range(self, video_id: str) -> Optional[tuple[int, int]]:
        """Returns (start_id inclusive, end_id exclusive) or None if unknown."""
        e = self._table.get(video_id)
        return (e.start_id, e.end_id) if e else None

    def timestamp_to_frame_id(self, video_id: str, ts: float) -> int:
        e = self._table[video_id]
        n = e.end_id - e.start_id
        return e.start_id + min(int(ts * e.fps), n - 1)

    def frame_id_to_video(self, frame_id: int) -> Optional[str]:
        i = bisect.bisect_right(self._start_ids, frame_id) - 1
        if i < 0:
            return None
        e = self._sorted_entries[i]
        return e.video_id if frame_id < e.end_id else None

    def frame_id_to_timestamp(self, frame_id: int, video_id: str) -> float:
        e = self._table[video_id]
        return (frame_id - e.start_id) / e.fps

    def save(self, path: Path) -> None:
        path.write_text(json.dumps([asdict(e) for e in self._table.values()], indent=2))

    @classmethod
    def load(cls, path: Path) -> "VideoTable":
        return cls([VideoEntry(**d) for d in json.loads(path.read_text())])

def build_metadata(
    n: int,
    out_dir: Path,
    frames_per_video: int = 1000,
    fps: float = 1.0,
    seed: int = 42,
) -> VideoTable:
    """
    Builds VideoTable + DuckDB metadata for n frames.
    Writes video_table.json and meta.duckdb to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_videos = (n + frames_per_video - 1) // frames_per_video

    entries: list[VideoEntry] = []
    frame_rows: list[tuple] = []   # (frame_id, video_id, timestamp, scene_id)
    tag_rows: list[tuple] = []     # (frame_id, tag)

    for v in range(n_videos):
        vid = f"video_{v:04d}"
        start = v * frames_per_video
        end = min(start + frames_per_video, n)
        n_frames = end - start
        entries.append(VideoEntry(video_id=vid, start_id=start, end_id=end, fps=fps))

        # Random scene boundaries within this video
        n_scenes = int(rng.integers(_SCENES_PER_VIDEO_MIN, _SCENES_PER_VIDEO_MAX + 1))
        cuts = sorted(rng.integers(0, n_frames, size=n_scenes - 1).tolist())
        cuts = [0] + cuts + [n_frames]
        scene_of: list[str] = []
        for s in range(len(cuts) - 1):
            scene_id = f"scene_{v:04d}_{s:02d}"
            scene_of.extend([scene_id] * (cuts[s + 1] - cuts[s]))

        for fi in range(n_frames):
            frame_id = start + fi
            ts = fi / fps
            frame_rows.append((frame_id, vid, ts, scene_of[fi]))

            n_tags = int(rng.integers(0, _TAGS_PER_FRAME_MAX + 1))
            if n_tags > 0:
                chosen = rng.choice(len(_TAG_VOCAB), size=n_tags, replace=False)
                for idx in chosen:
                    tag_rows.append((frame_id, _TAG_VOCAB[idx]))

    vt = VideoTable(entries)
    vt.save(out_dir / "video_table.json")

    db_path = str(out_dir / "meta.duckdb")
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE OR REPLACE TABLE frames (
            frame_id  INTEGER NOT NULL,
            video_id  VARCHAR NOT NULL,
            timestamp FLOAT   NOT NULL,
            scene_id  VARCHAR NOT NULL
        )
    """)
    con.execute("""
        CREATE OR REPLACE TABLE frame_tags (
            frame_id INTEGER NOT NULL,
            tag      VARCHAR NOT NULL
        )
    """)
    # Sort by (video_id, timestamp) so DuckDB zone maps prune range scans
    frame_rows.sort(key=lambda r: (r[1], r[2]))
    con.executemany("INSERT INTO frames VALUES (?, ?, ?, ?)", frame_rows)
    con.executemany("INSERT INTO frame_tags VALUES (?, ?)", tag_rows)
    con.execute("CREATE INDEX IF NOT EXISTS idx_tags ON frame_tags(tag)")
    con.close()

    return vt

def load_metadata(index_dir: Path) -> tuple[VideoTable, duckdb.DuckDBPyConnection]:
    """Loads VideoTable and opens a read-only DuckDB connection."""
    d = Path(index_dir)
    vt = VideoTable.load(d / "video_table.json")
    con = duckdb.connect(str(d / "meta.duckdb"), read_only=True)
    return vt, con
