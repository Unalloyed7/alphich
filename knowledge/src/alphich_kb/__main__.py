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
    encoder_parser = commands.add_parser("fetch-encoder", help="Download the pinned local E5 ONNX model")
    encoder_parser.add_argument("--model-manifest", default="knowledge/embedding-model.json")
    encoder_parser.add_argument("--model-dir", default="data/models/e5-base")
    index_parser = commands.add_parser("index", help="Build an exact E5 index from the prepared corpus")
    index_parser.add_argument("--corpus-dir", required=True)
    index_parser.add_argument("--output-dir", required=True)
    index_parser.add_argument("--batch-size", type=int, default=4)
    retrieve_parser = commands.add_parser("retrieve", help="Hybrid bank evidence retrieval and bounded LLM context")
    retrieve_parser.add_argument("--index-dir", required=True)
    retrieve_parser.add_argument("--query", required=True)
    retrieve_parser.add_argument("--as-of", required=True)
    retrieve_parser.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default="dense")
    retrieve_parser.add_argument("--limit", type=int, default=5)
    retrieve_parser.add_argument("--topic", action="append")
    retrieve_parser.add_argument("--allow-document", action="append")
    retrieve_parser.add_argument("--metric")
    retrieve_parser.add_argument("--horizon")
    retrieve_parser.add_argument("--max-context-chars", type=int, default=16000)
    retrieve_parser.add_argument("--context-only", action="store_true")
    compare_parser = commands.add_parser("compare-retrieval", help="Compare lexical, dense and hybrid on one index")
    compare_parser.add_argument("--index-dir", required=True)
    compare_parser.add_argument("--cases", default="knowledge/retrieval_cases.json")
    news_parser = commands.add_parser("select-news", help="Select company-specific events and plan bank retrieval")
    news_parser.add_argument("--profiles", default="knowledge/fixtures/company_profiles.json")
    news_parser.add_argument("--events", default="knowledge/fixtures/news_events.json")
    news_parser.add_argument("--company-id", required=True)
    news_parser.add_argument("--as-of", required=True)
    news_parser.add_argument("--role", choices=["cfo", "ceo"], default="cfo")
    news_parser.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default="hybrid")
    news_parser.add_argument("--limit", type=int, default=5)
    news_parser.add_argument("--min-grade", type=int, choices=range(6), default=0,
                             help="4 for the curated feed; 0 preserves the retrieval candidate pool")
    news_parser.add_argument("--allow-event", action="append")
    news_parser.add_argument("--bank-index-dir")
    news_parser.add_argument("--bank-event-id")
    news_parser.add_argument("--allow-bank-document", action="append")
    news_compare = commands.add_parser("compare-news", help="Evaluate selection on fictional company/event grades")
    news_compare.add_argument("--profiles", default="knowledge/fixtures/company_profiles.json")
    news_compare.add_argument("--events", default="knowledge/fixtures/news_events.json")
    news_compare.add_argument("--labels", default="knowledge/fixtures/relevance_labels.json")
    news_compare.add_argument("--mode", choices=["lexical", "hybrid"], default="hybrid")
    for child in (index_parser, retrieve_parser, compare_parser, news_parser, news_compare):
        child.add_argument("--model-dir", default="data/models/e5-base")
        child.add_argument("--model-manifest", default="knowledge/embedding-model.json")
        child.add_argument("--threads", type=int, default=2)
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
        elif args.command == "evaluate":
            from .evaluate import evaluate
            result = evaluate(args.corpus_dir, args.cases)
        elif args.command == "fetch-encoder":
            from .common import safe_child
            from .fetch import download
            config = read_json(args.model_manifest)
            for item in config["files"]:
                download(item["url"], safe_child(args.model_dir, item["path"]), item["sha256"])
            result = {"status": "checksums_verified", "model_id": config["model_id"], "revision": config["revision"]}
        elif args.command == "index":
            from .embeddings import E5Encoder
            from .index import build_index
            encoder = E5Encoder(args.model_dir, args.model_manifest, args.threads)
            result = build_index(args.corpus_dir, args.output_dir, encoder, args.batch_size)
        elif args.command == "retrieve":
            from .embeddings import E5Encoder, EncoderUnavailable
            from .retriever import BankRetriever, RetrievalScope
            encoder = None
            if args.mode != "lexical":
                try:
                    encoder = E5Encoder(args.model_dir, args.model_manifest, args.threads)
                except EncoderUnavailable:
                    pass  # The response records lexical_fallback; it is never labelled hybrid.
            retriever = BankRetriever(args.index_dir, encoder)
            allowed = args.allow_document if args.allow_document is not None else retriever.documents.keys()
            scope = RetrievalScope(args.as_of, frozenset(allowed))
            result = retriever.retrieve(args.query, scope, mode=args.mode, limit=args.limit, topics=args.topic,
                                        metric=args.metric, horizon=args.horizon, max_context_chars=args.max_context_chars)
            if args.context_only:
                from .context import context_for_model
                result = context_for_model(result)
        elif args.command == "compare-retrieval":
            from .embeddings import E5Encoder
            from .compare import compare_retrieval
            encoder = E5Encoder(args.model_dir, args.model_manifest, args.threads)
            result = compare_retrieval(args.index_dir, args.cases, encoder)
        elif args.command == "select-news":
            from .personalization.cli import select_from_files
            result = select_from_files(args)
        else:
            from .personalization.cli import load_encoder
            from .personalization.evaluate import evaluate_selection
            result = evaluate_selection(args.profiles, args.events, args.labels, load_encoder(args))
    except (KnowledgeError, OSError, ValueError, KeyError, ImportError) as error:
        print(f"alphich-kb: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
