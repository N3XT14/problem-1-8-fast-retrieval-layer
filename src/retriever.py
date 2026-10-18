"""Retriever — IVF-PQ ANN search + exact fp16 rerank. Every fallback sets degraded=True."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import faiss
import numpy as np

from config import SEARCH
from indexer import load_index
from models import Result, SearchResponse, StageLatency, StructuredQuery, QueryFilters

logger = logging.getLogger(__name__)

@contextmanager
def _timer() -> Generator[list, None, None]:
    bucket: list[float] = []
    t0 = time.perf_counter()
    yield bucket
    bucket.append((time.perf_counter() - t0) * 1000.0)

class Retriever:
    """Loads the three stores from index_dir and exposes a search method."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.index, self.fp16_store, self.meta = load_index(self.index_dir)
        self._set_nprobe(SEARCH.BF_MAX)

    def set_nprobe(self, nprobe: int) -> None:
        self._set_nprobe(nprobe)

    def _set_nprobe(self, nprobe: int) -> None:
        iv = faiss.extract_index_ivf(self.index)
        iv.nprobe = nprobe
        self._nprobe = nprobe

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        do_rerank: bool = True,
    ) -> SearchResponse:
        """Search with a pre-embedded, pre-normalised query vector (1, d) or (d,) float32."""
        qv = np.atleast_2d(query_vec).astype(np.float32)
        if qv.shape[1] != self.meta.dim:
            raise ValueError(f"Query dim {qv.shape[1]} != index dim {self.meta.dim}")

        lat = StageLatency()
        degraded = False
        reason: str | None = None
        fetch = k * SEARCH.OVERFETCH

        faiss.cvar.indexIVF_stats.reset()
        with _timer() as t_ann:
            D, I = self.index.search(qv, fetch)
        lat.ann_ms = t_ann[0]
        ndis = int(faiss.cvar.indexIVF_stats.ndis)

        ann_ids = I[0]
        ann_scores = D[0]

        # FAISS fills unused slots with -1 when the index has fewer results than fetch
        valid = ann_ids >= 0
        ann_ids = ann_ids[valid]
        ann_scores = ann_scores[valid]

        if len(ann_ids) == 0:
            degraded, reason = True, "ANN returned no results"
            return self._empty_response(k, lat, degraded, reason, ndis)

        if do_rerank:
            with _timer() as t_rr:
                final_ids, final_scores = self._rerank(qv[0], ann_ids, k)
            lat.rerank_ms = t_rr[0]
            source = "reranked"
        else:
            final_ids = ann_ids[:k]
            final_scores = ann_scores[:k]
            source = "ann"

        lat.total_ms = lat.ann_ms + lat.rerank_ms

        results = [
            Result(
                frame_id=int(fid),
                video_id="",
                start_ts=0.0,
                end_ts=0.0,
                score=float(sc),
                source=source,
            )
            for fid, sc in zip(final_ids, final_scores)
        ]

        sq = StructuredQuery(raw="", semantic="", filters=QueryFilters())
        return SearchResponse(
            query=sq,
            results=results,
            latency=lat,
            degraded=degraded,
            reason=reason,
            ndis=ndis,
            nprobe_used=self._nprobe,
        )

    def _rerank(
        self,
        query_vec: np.ndarray,
        candidate_ids: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact inner-product rerank over fp16 candidate embeddings."""
        sorted_order = np.argsort(candidate_ids)
        sorted_ids = candidate_ids[sorted_order]

        vecs = self.fp16_store[sorted_ids].astype(np.float32)
        scores = vecs @ query_vec

        top_local = np.argpartition(-scores, min(k, len(scores)) - 1)[:k]
        top_local = top_local[np.argsort(-scores[top_local])]
        return sorted_ids[top_local], scores[top_local]

    def search_batch(
        self,
        query_vecs: np.ndarray,
        k: int = 10,
        do_rerank: bool = True,
    ) -> list[SearchResponse]:
        return [
            self.search(qv.reshape(1, -1), k=k, do_rerank=do_rerank)
            for qv in query_vecs
        ]

    def _empty_response(
        self,
        k: int,
        lat: StageLatency,
        degraded: bool,
        reason: str,
        ndis: int,
    ) -> SearchResponse:
        sq = StructuredQuery(raw="", semantic="", filters=QueryFilters())
        return SearchResponse(
            query=sq, results=[], latency=lat,
            degraded=degraded, reason=reason,
            ndis=ndis, nprobe_used=self._nprobe,
        )
