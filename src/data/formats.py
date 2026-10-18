"""
Readers/writers for the big-ann-benchmarks binary formats.

.fbin — float32 vectors:  [int32 n][int32 d][n x d x float32]
.ibin — int32 arrays:     [int32 n][int32 k][n x k x int32]

read_fbin/read_ibin accept start/count for streaming slices.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

def _read_header(path: str | Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        raw = f.read(8)
    if len(raw) < 8:
        raise ValueError(f"File too small to contain a valid header: {path}")
    n, d = struct.unpack("<ii", raw)
    return int(n), int(d)

def fbin_header(path: str | Path) -> tuple[int, int]:
    return _read_header(path)

def ibin_header(path: str | Path) -> tuple[int, int]:
    return _read_header(path)

def read_fbin(
    path: str | Path,
    start: int = 0,
    count: Optional[int] = None,
    dtype: type = np.float32,
) -> np.ndarray:
    """Returns (count, d) ndarray. start/count for streaming slices."""
    path = Path(path)
    n, d = _read_header(path)

    if start >= n:
        raise IndexError(f"start={start} is out of range for file with n={n}")

    actual_count = min(count if count is not None else n, n - start)
    byte_offset = 8 + start * d * 4  # 8-byte header + float32 rows

    with open(path, "rb") as f:
        f.seek(byte_offset)
        raw = f.read(actual_count * d * 4)

    arr = np.frombuffer(raw, dtype=np.float32).reshape(actual_count, d)
    return arr.astype(dtype, copy=False)

def read_ibin(
    path: str | Path,
    start: int = 0,
    count: Optional[int] = None,
) -> np.ndarray:
    """Returns (count, k) int32 ndarray."""
    path = Path(path)
    n, k = _read_header(path)

    if start >= n:
        raise IndexError(f"start={start} is out of range for file with n={n}")

    actual_count = min(count if count is not None else n, n - start)
    byte_offset = 8 + start * k * 4

    with open(path, "rb") as f:
        f.seek(byte_offset)
        raw = f.read(actual_count * k * 4)

    return np.frombuffer(raw, dtype=np.int32).reshape(actual_count, k)

def write_fbin(path: str | Path, vectors: np.ndarray) -> None:
    """vectors must be 2-D (n, d)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, d = vectors.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, d))
        f.write(vectors.astype(np.float32).tobytes())

def write_ibin(path: str | Path, ids: np.ndarray) -> None:
    """ids must be 2-D (n, k)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, k = ids.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, k))
        f.write(ids.astype(np.int32).tobytes())

def verify_fbin(path: str | Path, expected_dim: Optional[int] = None) -> tuple[int, int]:
    """Returns (n, d). Raises if file is truncated or dim mismatches."""
    path = Path(path)
    n, d = _read_header(path)

    if expected_dim is not None and d != expected_dim:
        raise ValueError(f"Expected dim={expected_dim}, got dim={d} in {path}")

    expected_bytes = 8 + n * d * 4
    actual_bytes = path.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"File truncated: expected {expected_bytes} bytes, got {actual_bytes}. "
            f"({path})"
        )

    return n, d
