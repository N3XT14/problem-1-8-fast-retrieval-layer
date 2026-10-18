"""
Exact ground-truth computation and recall evaluation.
GT is recomputed for our data slices since the published GT covers only the full 10M set.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config import Metric
from data.formats import read_ibin, write_ibin

logger = logging.getLogger(__name__)

def compute_gt(
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    metric: Metric = Metric.IP,
) -> np.ndarray:
    """Returns (Q, k) int32 — exact neighbor IDs via brute-force IndexFlat."""
    base = base.astype(np.float32)
    queries = queries.astype(np.float32)
    _, d = base.shape

    if metric == Metric.IP:
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(base)
    logger.info(f"Computing GT: {len(queries)} queries × top-{k} over {len(base):,} base vectors")
    _, I = index.search(queries, k)
    return I.astype(np.int32)

def recall_at_k(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
) -> float:
    """Fraction of true top-k neighbors found in predicted. Handles FAISS sentinel -1."""
    n_queries = len(ground_truth)
    correct = 0
    for pred_row, gt_row in zip(predicted, ground_truth):
        pred_set = set(int(x) for x in pred_row[:k] if x >= 0)
        gt_set = set(int(x) for x in gt_row if x >= 0)
        correct += len(pred_set & gt_set)
    return correct / (n_queries * k) if n_queries * k > 0 else 0.0

def save_gt(path: str | Path, gt: np.ndarray) -> None:
    write_ibin(path, gt)

def load_gt(path: str | Path, n_queries: Optional[int] = None) -> np.ndarray:
    gt = read_ibin(path, count=n_queries)
    return gt
