"""
Diversification: collapses results from the same video + time window into one clip.

Groups results by (video_id, floor(start_ts / window_s)), keeps the highest-scoring
frame per group, and returns the top-k groups by score.

Prevents the common failure where 8 of 10 results are the same 3-second scene
(frame embeddings are near-identical within a shot, so ANN returns them all).
"""

from __future__ import annotations

import math

from models import Result

_WINDOW_S = 5.0   # seconds per diversity bucket


def diversify(
    results: list[Result],
    window_s: float = _WINDOW_S,
    k: int = 10,
) -> list[Result]:
    """Collapses near-duplicate results from the same video+time window, returns top-k."""
    if not results:
        return results

    seen: dict[tuple, Result] = {}

    for r in results:
        bucket = (r.video_id, math.floor(r.start_ts / window_s))
        if bucket not in seen or r.score > seen[bucket].score:
            seen[bucket] = r

    top = sorted(seen.values(), key=lambda r: r.score, reverse=True)
    return top[:k]
