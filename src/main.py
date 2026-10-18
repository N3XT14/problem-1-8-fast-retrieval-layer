"""
Fast Retrieval Layer — CLI entry point.

Usage:
  python src/main.py download  --dataset t2i  --n 100000  --out data/
  python src/main.py index     --data-dir data/  --out index/t2i-100k/  --config t2i-100k
  python src/main.py query     --text "dragons fighting" --index-dir index/t2i-100k/ --data-dir data/
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

    if args.base_file:
        base_path = args.base_file
        if not base_path.exists():
            logger.error(f"Base file not found: {base_path}")
            sys.exit(1)
    else:
        base_files = sorted(args.data_dir.glob("*base*.fbin"))
        if not base_files:
            logger.error(f"No base*.fbin found in {args.data_dir}. Run download first.")
            sys.exit(1)
        base_path = base_files[0]
    logger.info(f"Loading base vectors from {base_path} ...")
    base = read_fbin(base_path)

    build_index(base, args.out, cfg, normalize=True)

    from data.synth_metadata import build_metadata
    n = len(base)
    logger.info(f"Building synthetic video metadata for {n:,} frames ...")
    build_metadata(n=n, out_dir=args.out)
    logger.info(f"Metadata written: {args.out}")


def cmd_query(args: argparse.Namespace) -> None:
    from data.formats import read_fbin
    from data.synth_metadata import load_metadata
    from indexer import load_index
    from query_parser import parse_query
    from retriever import Retriever

    sq = parse_query(args.text, top_k=args.top_k)
    logger.info(f"Query: {sq.raw!r}")
    logger.info(f"  semantic={sq.semantic!r}  filters={sq.filters}")

    # Load metadata if available
    vt, db_con = None, None
    vt_path = args.index_dir / "video_table.json"
    if vt_path.exists():
        vt, db_con = load_metadata(args.index_dir)
        logger.info(f"Metadata loaded: {len(vt)} videos")

    retriever = Retriever(args.index_dir, video_table=vt, db_con=db_con)
    retriever.set_nprobe(64)

    # Use a precomputed query vector from the queries file as the semantic vector.
    # In production this would be replaced by a CLIP/MiniLM text embedding.
    query_files = list(args.data_dir.glob("*quer*.fbin"))
    if not query_files:
        logger.error(f"No queries.fbin found in {args.data_dir}. Run download first.")
        import sys; sys.exit(1)

    queries = read_fbin(query_files[0])
    import numpy as np
    qv = queries[args.query_idx % len(queries)].astype(np.float32)
    norm = np.linalg.norm(qv)
    if norm > 0:
        qv = qv / norm

    resp = retriever.search(qv, k=sq.top_k, do_rerank=True, filters=sq.filters)

    print(f"\n{'─'*65}")
    print(f"Query : {sq.raw!r}")
    print(f"Vector: queries[{args.query_idx}] (stand-in for text encoder output)")
    print(f"Filters: video_id={sq.filters.video_id}  "
          f"ts=[{sq.filters.timestamp_gte}, {sq.filters.timestamp_lte}]  "
          f"tags={sq.filters.tags}")
    print(f"{'─'*65}")
    print(f"{'rank':>4}  {'video_id':>12}  {'timestamp':>10}  {'score':>7}  source")
    print(f"{'─'*65}")
    for i, r in enumerate(resp.results, 1):
        print(f"{i:>4}  {r.video_id:>12}  {r.start_ts:>9.1f}s  {r.score:>7.4f}  {r.source}")
    print(f"{'─'*65}")
    print(f"Latency: filter={resp.latency.filter_ms:.2f}ms  "
          f"ann={resp.latency.ann_ms:.2f}ms  "
          f"rerank={resp.latency.rerank_ms:.2f}ms  "
          f"diversify={resp.latency.diversify_ms:.2f}ms  "
          f"total={resp.latency.total_ms:.2f}ms")
    print(f"ndis={resp.ndis}  nprobe={resp.nprobe_used}  degraded={resp.degraded}\n")

    if db_con:
        db_con.close()


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
    idx.add_argument("--base-file", type=Path, default=None, help="Explicit base vectors file (overrides glob)")
    idx.add_argument("--out", type=Path, default=Path("index/"), help="Index output directory")
    idx.add_argument(
        "--config", default="t2i-100k",
        help="Index config name from config.DATASET_CONFIG (default: t2i-100k)",
    )

    qry = sub.add_parser("query", help="Run a single natural language query end-to-end")
    qry.add_argument("--text", required=True, help="Natural language query string")
    qry.add_argument("--index-dir", type=Path, required=True)
    qry.add_argument("--data-dir", type=Path, default=Path("data/"),
                     help="Directory containing queries.fbin (vector stand-in for text encoder)")
    qry.add_argument("--top-k", type=int, default=10)
    qry.add_argument("--query-idx", type=int, default=0,
                     help="Which precomputed query vector to use as semantic embedding (default: 0)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "download": cmd_download,
        "index":    cmd_index,
        "query":    cmd_query,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
