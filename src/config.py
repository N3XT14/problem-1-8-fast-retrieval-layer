"""
All tunable knobs in one place.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Metric(str, Enum):
    IP = "IP"
    L2 = "L2"

@dataclass(frozen=True)
class IndexConfig:
    name: str
    dim: int
    metric: Metric
    pq_m: int  # PQ subquantizers so width = dim // pq_m
    nlist: int  # IVF Voronoi cells
    nprobe: int = 64 # cells checked at query time
    nbits: int = 8 # bits per PQ code so 8 bits provide 256 centroids per subquantizer

    @property
    def subvector_width(self) -> int:
        return self.dim // self.pq_m

    @property
    def factory_string(self) -> str:
        """IVFQ default and HNSW coarse quantizer for large nlist."""
        if self.nlist <= 4096:
            return f"IVF{self.nlist},PQ{self.pq_m}x{self.nbits}"
        return f"IVF{self.nlist}_HNSW32,PQ{self.pq_m}x{self.nbits}"

    @property
    def bytes_per_vector(self) -> int:
        """RAM for one stored vector (PQ codes only, no ID overhead)."""
        return self.pq_m

    @property
    def faiss_metric(self) -> int:
        import faiss
        return faiss.METRIC_INNER_PRODUCT if self.metric == Metric.IP else faiss.METRIC_L2

    def min_train_vectors(self) -> int:
        return self.nlist * 64


# Yandex Text-to-Image, 200-dim, inner product
# PQ25 → 25 subvectors x 8 dims = 200, subvector width = 8
T2I_100K = IndexConfig(
    name="t2i_100k", dim=200, metric=Metric.IP, pq_m=25, nlist=512,
)
T2I_1M = IndexConfig(
    name="t2i_1m", dim=200, metric=Metric.IP, pq_m=25, nlist=4096,
)
T2I_10M = IndexConfig(
    name="t2i_10m", dim=200, metric=Metric.IP, pq_m=25, nlist=16384,
)

# Registry
DATASET_CONFIG: dict[str, IndexConfig] = {
    "t2i-100k": T2I_100K,
    "t2i-1m":   T2I_1M,
    "t2i-10m":  T2I_10M,
}


# Search / retrieval knobs

@dataclass(frozen=True)
class SearchConfig:
    BF_MAX: int = 50_000
    OVERFETCH: int = 10


SEARCH = SearchConfig()

T2I_URLS: dict[str, str] = {
    "base_1b":  "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin",
    "queries":  "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin",
    "gt_10m":   "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/text2image-10M",
}

BLENDER_URLS: dict[str, str] = {
    "tears_720p": "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
    "tears_subs": "https://download.blender.org/demo/movies/ToS/subtitles/TOS-en.srt",
}
