"""
Dataset downloader.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from data.formats import write_fbin

logger = logging.getLogger(__name__)

def _range_bytes(url: str, byte_start: int, byte_end: int, timeout: int = 30) -> bytes:
    """Fetch a byte range from a URL. Raises if server doesn't support Range."""
    headers = {"Range": f"bytes={byte_start}-{byte_end - 1}"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code == 416:
        raise ValueError(f"Range not satisfiable: {url} bytes {byte_start}-{byte_end-1}")
    if resp.status_code not in (200, 206):
        raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")
    if resp.status_code == 200:
        raise ValueError(
            f"Server at {url} returned 200 (full file) instead of 206 (range). "
            "Range requests not supported."
        )
    return resp.content

def fetch_fbin_slice(
    url: str,
    out_path: Path,
    n_vectors: int,
    timeout: int = 60,
) -> Path:
    """Download the first n_vectors from a remote .fbin via HTTP Range request."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching header from {url} ...")
    header_bytes = _range_bytes(url, 0, 8, timeout=timeout)
    remote_n, d = struct.unpack("<ii", header_bytes)
    logger.info(f"  Remote file: n={remote_n:,}, d={d}")

    if n_vectors > remote_n:
        logger.warning(
            f"Requested {n_vectors:,} vectors but remote file has only {remote_n:,}. "
            f"Fetching all {remote_n:,}."
        )
        n_vectors = remote_n

    data_bytes = n_vectors * d * 4
    byte_start = 8
    byte_end = 8 + data_bytes

    logger.info(f"  Fetching {n_vectors:,} vectors ({data_bytes / 1e6:.1f} MB) ...")
    raw = _range_bytes(url, byte_start, byte_end, timeout=timeout)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<ii", n_vectors, d))
        f.write(raw)

    logger.info(f"  Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def make_clustered_vectors(
    n: int,
    dim: int,
    n_clusters: int = 256,
    rng: Optional[np.random.Generator] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Gaussian mixture synthetic vectors. Must be clustered — uniform random is near-equidistant in high dims."""
    if rng is None:
        rng = np.random.default_rng(42)

    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers /= norms

    assignments = rng.integers(0, n_clusters, size=n)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * 0.05
    vectors = centers[assignments] + noise

    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.where(norms == 0, 1.0, norms)

    return vectors


def download_t2i(
    out_dir: Path,
    n_base: int = 100_000,
    n_queries: int = 10_000,
    base_url: Optional[str] = None,
    query_url: Optional[str] = None,
) -> dict[str, Path]:
    """Range-request download with clustered synthetic fallback. Returns {'base': Path, 'queries': Path}."""
    from config import T2I_URLS

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = base_url or T2I_URLS["base_1b"]
    query_url = query_url or T2I_URLS["queries"]

    paths: dict[str, Path] = {}

    for key, url, n, name in [
        ("base",    base_url,  n_base,   "base"),
        ("queries", query_url, n_queries, "queries"),
    ]:
        out_path = out_dir / f"t2i_{name}_{n}.fbin"

        if out_path.exists():
            logger.info(f"  {out_path.name} already exists — skipping download.")
            paths[key] = out_path
            continue

        try:
            fetch_fbin_slice(url, out_path, n)
            paths[key] = out_path
        except Exception as exc:
            logger.warning(
                f"Download failed ({exc}). Generating clustered synthetic fallback "
                f"({n:,} vectors, dim=200)."
            )
            dim = 200
            vecs = make_clustered_vectors(n, dim)
            write_fbin(out_path, vecs)
            paths[key] = out_path
            logger.info(f"  Synthetic fallback saved: {out_path}")

    return paths


