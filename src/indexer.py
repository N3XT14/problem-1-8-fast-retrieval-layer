"""
Builds and loads the three stores written to index_dir/:
  index.faiss     — IVF-PQ FAISS index (loaded into RAM at query time)
  embeddings.f16  — fp16 mmap store (NVMe, touched only for rerank candidates)
  meta.json       — IndexMeta
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from config import IndexConfig, Metric

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100_000  # controls peak RAM during add phase

@dataclass
class IndexMeta:
    n: int
    dim: int
    metric: str
    config_name: str
    nlist: int
    pq_m: int
    nbits: int
    normalized: bool
    factory_string: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "IndexMeta":
        return cls(**json.loads(path.read_text()))


def _safe_nlist(n: int, requested_nlist: int) -> int:
    """Reduce nlist if n is too small — FAISS needs ≥ 39 training vectors per cell."""
    max_nlist = max(1, n // 39)
    if requested_nlist <= max_nlist:
        return requested_nlist
    # largest power-of-2 ≤ max_nlist
    safe = 1
    while safe * 2 <= max_nlist:
        safe *= 2
    logger.warning(
        f"Requested nlist={requested_nlist} but n={n} only supports up to "
        f"nlist={max_nlist}. Using nlist={safe}."
    )
    return safe


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def build_index(
    base: np.ndarray,
    out_dir: Path,
    cfg: IndexConfig,
    normalize: bool = True,
    train_sample_multiplier: int = 64,
) -> IndexMeta:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = base.astype(np.float32)
    n, d = base.shape

    if d != cfg.dim:
        raise ValueError(f"Base dim {d} != config dim {cfg.dim}")

    t0 = time.perf_counter()

    if normalize:
        logger.info("L2-normalising base vectors ...")
        base = _l2_normalize(base)

    f16_path = out_dir / "embeddings.f16"
    logger.info(f"Writing fp16 store [actual vectors] at {f16_path}  ({n * d * 2 / 1e6:.1f} MB) ...")
    mm = np.memmap(f16_path, dtype=np.float16, mode="w+", shape=(n, d))
    for start in range(0, n, 50_000):
        mm[start : start + 50_000] = base[start : start + 50_000].astype(np.float16)
    mm.flush()
    del mm

    actual_nlist = _safe_nlist(n, cfg.nlist)
    if actual_nlist != cfg.nlist:
        factory = (
            f"IVF{actual_nlist},PQ{cfg.pq_m}x{cfg.nbits}"
            if actual_nlist <= 4096
            else f"IVF{actual_nlist}_HNSW32,PQ{cfg.pq_m}x{cfg.nbits}"
        )
    else:
        factory = cfg.factory_string

    logger.info(f"Building FAISS index: {factory}  n={n:,}  d={d}")
    metric_flag = faiss.METRIC_INNER_PRODUCT if cfg.metric == Metric.IP else faiss.METRIC_L2
    index = faiss.index_factory(d, factory, metric_flag)

    n_train = min(n, actual_nlist * train_sample_multiplier)
    logger.info(f"Training on {n_train:,} vectors ...")
    t_train = time.perf_counter()
    rng = np.random.default_rng(42)
    index.train(base[rng.choice(n, size=n_train, replace=False)])
    logger.info(f"  Training done in {time.perf_counter() - t_train:.1f}s")

    logger.info(f"Adding {n:,} vectors in batches of {_BATCH_SIZE:,} ...")
    ids = np.arange(n, dtype=np.int64)
    for start in range(0, n, _BATCH_SIZE):
        end = min(start + _BATCH_SIZE, n)
        index.add_with_ids(base[start:end], ids[start:end])
        if (start // _BATCH_SIZE) % 5 == 0:
            logger.info(f"  Added {end:,} / {n:,}")

    index_path = out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    
    logger.info(f"FAISS index written to disk: {index_path}  ({index_path.stat().st_size / 1e6:.1f} MB)")

    meta = IndexMeta(
        n=n, dim=d, metric=cfg.metric.value,
        config_name=cfg.name,
        nlist=actual_nlist, pq_m=cfg.pq_m, nbits=cfg.nbits,
        normalized=normalize,
        factory_string=factory,
    )
    meta.save(out_dir / "meta.json")

    logger.info(f"Index built in {time.perf_counter() - t0:.1f}s, output saved to {out_dir}")
    return meta


def load_index(index_dir: Path) -> tuple[faiss.Index, np.ndarray, IndexMeta]:
    """Returns (faiss_index, fp16_mmap, meta)."""
    index_dir = Path(index_dir)
    meta = IndexMeta.load(index_dir / "meta.json")
    index = faiss.read_index(str(index_dir / "index.faiss"))
    mm = np.memmap(
        index_dir / "embeddings.f16",
        dtype=np.float16,
        mode="r",
        shape=(meta.n, meta.dim),
    )
    return index, mm, meta
