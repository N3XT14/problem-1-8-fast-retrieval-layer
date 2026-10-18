"""
Retriever — three search paths + exact fp16 rerank. Every fallback sets degraded=True.

Path A — no filter:    full IVF-PQ search
Path B — small filter: brute-force exact search on the filtered ID subset
Path C — large filter: IVF-PQ search with a FAISS IDSelector
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import faiss
import numpy as np

from config import SEARCH
from data.synth_metadata import VideoTable
from diversify import diversify
from filters import FilterResult, resolve_filters
from indexer import load_index
from models import QueryFilters, Result, SearchResponse, StageLatency, StructuredQuery

logger = logging.getLogger(__name__)


@contextmanager
def _timer() -> Generator[list, None, None]:
    bucket: list[float] = []
    t0 = time.perf_counter()
    yield bucket
    bucket.append((time.perf_counter() - t0) * 1000.0)


class Retriever:
    """Loads the three stores from index_dir and exposes a search method."""

    def __init__(
        self,
        index_dir: Path,
        video_table: Optional[VideoTable] = None,
        db_con: Optional[Any] = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index, self.fp16_store, self.meta = load_index(self.index_dir)
        self.video_table = video_table
        self.db_con = db_con
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
        filters: Optional[QueryFilters] = None,
    ) -> SearchResponse:
        """Search with a pre-embedded, pre-normalised query vector (1, d) or (d,) float32."""
        qv = np.atleast_2d(query_vec).astype(np.float32)
        if qv.shape[1] != self.meta.dim:
            raise ValueError(f"Query dim {qv.shape[1]} != index dim {self.meta.dim}")

        lat = StageLatency()
        degraded = False
        reason: str | None = None
        fetch = k * SEARCH.OVERFETCH

        # ---- Filter resolution ----
        filter_result: Optional[FilterResult] = None
        if filters and self.video_table is not None:
            with _timer() as t_filter:
                filter_result = resolve_filters(
                    filters, self.video_table, self.db_con, self.meta.n
                )
            lat.filter_ms = t_filter[0]

            if filter_result.count == 0:
                degraded, reason = True, "Filter matched 0 frames"
                return self._empty_response(k, lat, degraded, reason, 0)

        # ---- Choose search path ----
        faiss.cvar.indexIVF_stats.reset()

        if filter_result is None or filter_result.candidate_ids is None:
            # Path A: no filter
            with _timer() as t_ann:
                D, I = self.index.search(qv, fetch)
            lat.ann_ms = t_ann[0]
            ndis = int(faiss.cvar.indexIVF_stats.ndis)
            ann_ids, ann_scores = I[0], D[0]
            valid = ann_ids >= 0
            ann_ids, ann_scores = ann_ids[valid], ann_scores[valid]
            source = "ann"

        elif filter_result.count <= SEARCH.BF_MAX:
            # Path B: small filter — exact brute-force on fp16 subset
            cids = filter_result.candidate_ids
            with _timer() as t_ann:
                vecs = self.fp16_store[cids].astype(np.float32)
                raw_scores = vecs @ qv[0]
                top_n = min(fetch, len(cids))
                top_local = np.argpartition(-raw_scores, top_n - 1)[:top_n]
                top_local = top_local[np.argsort(-raw_scores[top_local])]
            lat.ann_ms = t_ann[0]
            ndis = len(cids)
            ann_ids, ann_scores = cids[top_local], raw_scores[top_local]
            source = "brute_force"
            do_rerank = False  # already exact

        else:
            # Path C: large filter — IVF-PQ with IDSelector
            params = faiss.SearchParametersIVF()
            params.nprobe = self._nprobe
            params.sel = filter_result.selector
            with _timer() as t_ann:
                D, I = self.index.search(qv, fetch, params=params)
            lat.ann_ms = t_ann[0]
            ndis = int(faiss.cvar.indexIVF_stats.ndis)
            ann_ids, ann_scores = I[0], D[0]
            valid = ann_ids >= 0
            ann_ids, ann_scores = ann_ids[valid], ann_scores[valid]
            source = "ann"

        if len(ann_ids) == 0:
            degraded, reason = True, "ANN returned no results"
            return self._empty_response(k, lat, degraded, reason, ndis)

        # ---- Rerank ----
        if do_rerank:
            with _timer() as t_rr:
                final_ids, final_scores = self._rerank(qv[0], ann_ids, k)
            lat.rerank_ms = t_rr[0]
            source = "reranked"
        else:
            final_ids = ann_ids[:k]
            final_scores = ann_scores[:k]

        lat.total_ms = lat.ann_ms + lat.rerank_ms + lat.filter_ms

        results = [
            Result(
                frame_id=int(fid),
                video_id=self._video_id(int(fid)),
                start_ts=self._timestamp(int(fid)),
                end_ts=self._timestamp(int(fid)),
                score=float(sc),
                source=source,
            )
            for fid, sc in zip(final_ids, final_scores)
        ]

        # ---- Diversify (only when metadata is available so timestamps are meaningful) ----
        if self.video_table is not None:
            with _timer() as t_div:
                results = diversify(results, window_s=5.0, k=k)
            lat.diversify_ms = t_div[0]
        lat.total_ms = lat.ann_ms + lat.rerank_ms + lat.filter_ms + lat.diversify_ms

        sq = StructuredQuery(raw="", semantic="", filters=filters or QueryFilters())
        return SearchResponse(
            query=sq, results=results, latency=lat,
            degraded=degraded, reason=reason,
            ndis=ndis, nprobe_used=self._nprobe,
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

    def _video_id(self, frame_id: int) -> str:
        if self.video_table is None:
            return ""
        return self.video_table.frame_id_to_video(frame_id) or ""

    def _timestamp(self, frame_id: int) -> float:
        if self.video_table is None:
            return 0.0
        vid = self._video_id(frame_id)
        if not vid:
            return 0.0
        return self.video_table.frame_id_to_timestamp(frame_id, vid)

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
