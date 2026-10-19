"""Run with PYTHONPATH=knowledge/src python -m alphich_kb."""

import argparse
import json
import sys
from pathlib import Path

from .common import KnowledgeError, read_json, read_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare offline bank evidence for RAG")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="Download and verify pinned PDFs and assets")
    fetch_parser.add_argument("--corpus", default="knowledge/corpus.json")
    fetch_parser.add_argument("--assets", default="knowledge/assets.json")
    fetch_parser.add_argument("--pdf-dir", default="data/raw/bank")
    fetch_parser.add_argument("--model-dir", default="data/models")
    extract_parser = commands.add_parser("extract", help="Render PDFs and extract OCR with coordinates")
    extract_parser.add_argument("--corpus", default="knowledge/corpus.json")
    extract_parser.add_argument("--input-dir", default="data/raw/bank")
    extract_parser.add_argument("--output-dir", required=True)
    extract_parser.add_argument("--tessdata-dir", default="data/models/tessdata")
    extract_parser.add_argument("--dpi", type=int, default=240)
    extract_parser.add_argument("--workers", type=int, default=2)
    build_parser = commands.add_parser("build", help="Build token-bounded chunks and reviewed claims")
    build_parser.add_argument("--ocr-dir", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--tokenizer", default="data/models/e5-base/tokenizer.json")
    build_parser.add_argument("--assets", default="knowledge/assets.json")
    build_parser.add_argument("--reviews", default="knowledge/reviews.json")
    build_parser.add_argument("--max-tokens", type=int, default=384)
    search_parser = commands.add_parser("search", help="FTS5 baseline (no embeddings or generation)")
    search_parser.add_argument("--corpus-dir", required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--as-of", required=True)
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--topic", action="append")
    claim_parser = commands.add_parser("claims", help="Select reviewed forecasts by metric and horizon")
    claim_parser.add_argument("--corpus-dir", required=True)
    claim_parser.add_argument("--as-of", required=True)
    claim_parser.add_argument("--metric", required=True)
    claim_parser.add_argument("--horizon", required=True)
    eval_parser = commands.add_parser("evaluate", help="Evaluate the lexical baseline on annotated questions")
    eval_parser.add_argument("--corpus-dir", required=True)
    eval_parser.add_argument("--cases", default="knowledge/eval_cases.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            from .fetch import fetch
            result = fetch(args.corpus, args.assets, args.pdf_dir, args.model_dir)
        elif args.command == "extract":
            from .extract import extract
            result = extract(args.corpus, args.input_dir, args.output_dir, args.tessdata_dir, args.dpi, args.workers)
        elif args.command == "build":
            from .build import build
            expected = next(a["sha256"] for a in read_json(args.assets)["assets"] if a["id"] == "e5_tokenizer")
            result = build(args.ocr_dir, args.output_dir, args.tokenizer, args.reviews, args.max_tokens, expected)
        elif args.command == "search":
            from .search import search
            result = search(args.corpus_dir, args.query, args.as_of, args.limit, args.topic)
        elif args.command == "claims":
            from .common import verify_artifacts
            from .search import select_claims
            verify_artifacts(args.corpus_dir)
            result = select_claims(read_jsonl(Path(args.corpus_dir) / "claims.jsonl"), args.as_of, args.metric, args.horizon)
        else:
            from .evaluate import evaluate
            result = evaluate(args.corpus_dir, args.cases)
    except (KnowledgeError, OSError, ValueError, KeyError) as error:
        print(f"alphich-kb: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
