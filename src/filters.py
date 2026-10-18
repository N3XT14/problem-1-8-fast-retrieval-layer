"""
Filter resolution: QueryFilters -> candidate frame IDs + FAISS selector.

Returns FilterResult with materialized candidate_ids (for Path B brute-force)
and a FAISS IDSelector (for Path C IVF-filtered search).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import faiss
import numpy as np

from data.synth_metadata import VideoTable
from models import QueryFilters


@dataclass
class FilterResult:
    candidate_ids: Optional[np.ndarray]  # None = no filter (all frames are candidates)
    selector: Any # FAISS IDSelector, None = no filter
    count: int # number of candidate frames
    _buf: Any = field(default=None, repr=False)  # bitmap buffer (must outlive selector)


def _bitmap_selector(ids: np.ndarray, n_total: int) -> tuple[Any, np.ndarray]:
    """Build a bit-packed FAISS IDSelectorBitmap from an array of frame IDs."""
    n_bytes = (n_total + 7) // 8
    buf = np.zeros(n_bytes, dtype=np.uint8)
    if len(ids) > 0:
        np.bitwise_or.at(buf, ids // 8, (np.uint8(1) << (ids % 8).astype(np.uint8)))
    return faiss.IDSelectorBitmap(n_total, faiss.swig_ptr(buf)), buf


def resolve_filters(
    filters: QueryFilters,
    video_table: Optional[VideoTable],
    db_con: Any,
    n_total: int,
) -> FilterResult:
    """
    Resolves QueryFilters into a FilterResult.
    Returns FilterResult(candidate_ids=None, selector=None, count=n_total) when no filters apply.
    """
    ids: Optional[np.ndarray] = None  # None = "all"

    # video_id + timestamp, ID range (arithmetic, no DB)
    if filters.video_id and video_table:
        rng = video_table.id_range(filters.video_id)
        if rng is None:
            return FilterResult(candidate_ids=np.array([], dtype=np.int64), selector=None, count=0)
        lo, hi = rng

        if filters.timestamp_gte is not None:
            lo = video_table.timestamp_to_frame_id(filters.video_id, filters.timestamp_gte)
        if filters.timestamp_lte is not None:
            hi = min(hi, video_table.timestamp_to_frame_id(filters.video_id, filters.timestamp_lte) + 1)

        lo, hi = max(lo, 0), min(hi, n_total)
        ids = np.arange(lo, hi, dtype=np.int64) if lo < hi else np.array([], dtype=np.int64)

    # scene / tag, DuckDB
    if db_con is not None and (filters.scene_id or filters.tags):
        db_ids = _query_db(db_con, filters)
        ids = np.intersect1d(ids, db_ids) if ids is not None else db_ids

    if ids is None:
        return FilterResult(candidate_ids=None, selector=None, count=n_total)

    if len(ids) == 0:
        return FilterResult(candidate_ids=ids, selector=None, count=0)

    sel, buf = _bitmap_selector(ids, n_total)
    return FilterResult(candidate_ids=ids, selector=sel, count=len(ids), _buf=buf)


def _query_db(db_con: Any, filters: QueryFilters) -> np.ndarray:
    """Returns frame_ids matching scene_id and/or tag filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if filters.scene_id:
        clauses.append("f.scene_id = ?")
        params.append(filters.scene_id)

    if filters.tags:
        placeholders = ", ".join("?" for _ in filters.tags)
        clauses.append(f"""
            f.frame_id IN (
                SELECT frame_id FROM frame_tags WHERE tag IN ({placeholders})
            )
        """)
        params.extend(filters.tags)

    where = " AND ".join(clauses)
    sql = f"SELECT f.frame_id FROM frames f WHERE {where}"
    result = db_con.execute(sql, params).fetchnumpy()
    return result["frame_id"].astype(np.int64)
