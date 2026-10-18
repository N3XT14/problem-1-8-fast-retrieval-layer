"""
Shared data models.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryFilters:
    video_id: Optional[str] = None
    timestamp_gte: Optional[float] = None   # seconds from video start
    timestamp_lte: Optional[float] = None
    scene_id: Optional[str] = None
    tags: Optional[list[str]] = None


@dataclass
class StructuredQuery:
    raw: str
    semantic: str
    filters: QueryFilters
    top_k: int = 10
    modality_hint: Optional[str] = None  # 'visual' | 'transcript' soft hint to boosts fusion weight


@dataclass
class Result:
    frame_id: int
    video_id: str
    start_ts: float # seconds — start of the returned clip
    end_ts: float # seconds — end of the returned clip (= start_ts for bare frames)
    score: float
    source: str # 'ann' | 'reranked' | 'brute_force'


@dataclass
class StageLatency:
    parse_ms: float = 0.0
    encode_ms: float = 0.0
    filter_ms: float = 0.0
    ann_ms: float = 0.0
    rerank_ms: float = 0.0
    diversify_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class SearchResponse:
    query: StructuredQuery
    results: list[Result]
    latency: StageLatency
    degraded: bool = False
    reason: Optional[str] = None # which stage fell back and why
    ndis: int = 0 # FAISS codes scanned (from cvar.indexIVF_stats)
    nprobe_used: int = 0
