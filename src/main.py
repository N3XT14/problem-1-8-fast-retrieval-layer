"""
Fast Retrieval Layer — CLI entry point.

Usage:
  python src/main.py download  --dataset t2i  --n 100000  --out data/
  python src/main.py index     --data-dir data/  --out index/t2i-100k/  --config t2i-100k
  python src/main.py benchmark --suite b1  --index-dir index/t2i-100k/  --out benchmark_results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_download(args: argparse.Namespace) -> None:
    from data.download import download_t2i

    logger.info(f"Downloading dataset={args.dataset}  n={args.n:,}  out={args.out}")

    if args.dataset == "t2i":
        paths = download_t2i(
            out_dir=args.out,
            n_base=args.n,
            n_queries=min(args.n // 10, 10_000),
        )
        for key, p in paths.items():
            logger.info(f"  {key}: {p}")

        from data.formats import verify_fbin
        for key, p in paths.items():
            n, d = verify_fbin(p, expected_dim=200)
            logger.info(f"  verified {key}: n={n:,}  dim={d}")

    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    logger.info("Done.")


def cmd_index(args: argparse.Namespace) -> None:
    from data.formats import read_fbin
    from indexer import build_index
    from config import DATASET_CONFIG

    cfg = DATASET_CONFIG.get(args.config)
    if cfg is None:
        logger.error(f"Unknown config '{args.config}'. Choices: {list(DATASET_CONFIG)}")
        sys.exit(1)

    base_files = sorted(args.data_dir.glob("*base*.fbin"))
    if not base_files:
        logger.error(f"No base*.fbin found in {args.data_dir}. Run download first.")
        sys.exit(1)
    base_path = base_files[0]
    logger.info(f"Loading base vectors from {base_path} ...")
    base = read_fbin(base_path)

    build_index(base, args.out, cfg, normalize=True)
    logger.info(f"Index built → {args.out}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    suites = args.suite if args.suite != ["all"] else ["b1", "b2", "b3", "b4", "b5", "b6"]

    if "b1" in suites:
        from bench.recall_latency import run_b1
        query_path = args.data_dir / "queries.fbin"
        if not query_path.exists():
            candidates = list(args.data_dir.glob("*quer*.fbin"))
            if not candidates:
                logger.error(f"No queries file found in {args.data_dir}. Run download first.")
                sys.exit(1)
            query_path = candidates[0]

        gt_path = args.data_dir / "gt.ibin"
        run_b1(
            index_dir=args.index_dir,
            query_path=query_path,
            gt_path=gt_path if gt_path.exists() else None,
            out_dir=args.out,
            dataset_tag=args.index_dir.name,
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrieval",
        description="Extremely Fast Retrieval Layer — 100M+ video frames at sub-second latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    dl = sub.add_parser("download", help="Download or generate a dataset")
    dl.add_argument(
        "--dataset", choices=["t2i"], required=True,
        help="t2i = Yandex Text-to-Image (scale benchmark)",
    )
    dl.add_argument(
        "--n", type=int, default=100_000,
        help="Number of base vectors to fetch (default: 100,000)",
    )
    dl.add_argument(
        "--out", type=Path, default=Path("data/"),
        help="Output directory (default: data/)",
    )

    idx = sub.add_parser("index", help="Build FAISS index + fp16 store + metadata")
    idx.add_argument("--data-dir", type=Path, required=True, help="Directory with .fbin files")
    idx.add_argument("--out", type=Path, default=Path("index/"), help="Index output directory")
    idx.add_argument(
        "--config", default="t2i-100k",
        help="Index config name from config.DATASET_CONFIG (default: t2i-100k)",
    )

    bench = sub.add_parser("benchmark", help="Run benchmark suite")
    bench.add_argument(
        "--suite", nargs="+",
        choices=["b1", "b2", "b3", "b4", "b5", "b6", "all"],
        default=["b1"],
        help="Which benchmark(s) to run (default: b1)",
    )
    bench.add_argument("--index-dir", type=Path, required=True)
    bench.add_argument("--data-dir", type=Path, default=Path("data/"),
                       help="Directory containing queries.fbin and optionally gt.ibin")
    bench.add_argument("--out", type=Path, default=Path("benchmark_results/"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "download":  cmd_download,
        "index":     cmd_index,
        "benchmark": cmd_benchmark,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
