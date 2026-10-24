"""Offline corpus/pair preparation, weak labeling and CPU model lifecycle."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..common import KnowledgeError, read_json
from .dataset import dataset_summary, import_fixtures, read_dataset, write_dataset, write_private_json


def parser():
    root = argparse.ArgumentParser(description="Company-conditioned relevance ML; shadow baseline")
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate a dataset and print only aggregate counts")
    inspect.add_argument("--input", required=True)
    census = commands.add_parser("companies", help="Inspect all organizer company rows; print only aggregate sectors")
    census.add_argument("--csv", required=True)
    census.add_argument("--output", required=True)
    collection = commands.add_parser("collect", help="Resume a bounded, free public publisher archive collection")
    collection.add_argument("--sources", default="knowledge/relevance-sources.json")
    collection.add_argument("--db", default="data/private/relevance-ml/news-corpus.sqlite3")
    collection.add_argument("--goal", type=int, default=10000)
    collection.add_argument("--max-articles", type=int, default=100)
    collection.add_argument("--max-requests", type=int, default=300)
    collection.add_argument("--max-seconds", type=int, default=600)
    collection.add_argument("--max-bytes", type=int, default=200000000)
    collection.add_argument("--max-source-articles", type=int, default=2500)
    collection.add_argument("--published-after")
    collection.add_argument("--published-before")
    job = commands.add_parser("collect-job", help="Resume a parallel multi-publisher corpus job")
    job.add_argument("--sources", default="knowledge/relevance-sources.json")
    job.add_argument("--run-dir", default="data/private/relevance-ml/corpus-10k")
    job.add_argument("--goal", type=int, default=10000)
    job.add_argument("--workers", type=int, default=4)
    job.add_argument("--max-seconds", type=int, default=3600)
    job.add_argument("--max-requests", type=int, default=20000)
    job.add_argument("--max-bytes", type=int, default=2000000000)
    job.add_argument("--max-source-articles", type=int, default=1500)
    job.add_argument("--rounds", type=int, default=1, help="1..4 bounded resumptions; HTTP/byte/runtime caps apply per round")
    job.add_argument("--published-after")
    job.add_argument("--published-before")
    job_status = commands.add_parser("job-status", help="Read progress of the parallel collection, including while running")
    job_status.add_argument("--run-dir", default="data/private/relevance-ml/corpus-10k")
    export = commands.add_parser("export-links", help="Export publisher links/dates/titles for review, without article bodies")
    link_source = export.add_mutually_exclusive_group(required=True)
    link_source.add_argument("--db")
    link_source.add_argument("--job-dir")
    export.add_argument("--output", required=True)
    status = commands.add_parser("corpus-status", help="Read accepted corpus and persistent queue counts")
    status.add_argument("--db", default="data/private/relevance-ml/news-corpus.sqlite3")
    sample = commands.add_parser("sample", help="Sample unlabeled pairs from articles and all company sectors")
    sample_source = sample.add_mutually_exclusive_group(required=True)
    sample_source.add_argument("--db")
    sample_source.add_argument("--job-dir")
    sample.add_argument("--companies-csv", required=True)
    sample.add_argument("--as-of", required=True)
    sample.add_argument("--role", choices=["cfo", "ceo"], default="cfo")
    sample.add_argument("--pairs-per-article", type=int, default=3)
    sample.add_argument("--limit", type=int, default=30000)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--output", required=True, help="Use .jsonl for large datasets")
    sample.add_argument("--report", required=True)
    old = commands.add_parser("import-fixtures", help="Join provisional historical labels to independent snapshots")
    old.add_argument("--labels", action="append", required=True)
    old.add_argument("--events", required=True)
    old.add_argument("--profiles", required=True)
    old.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare-labels", help="Create auditable teacher requests, without API calls")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--limit", type=int, default=10)
    judge = commands.add_parser("judge", help="Explicit paid teacher batch on the server's shared ledger")
    judge.add_argument("--input", required=True)
    judge.add_argument("--output", required=True, help="New checkpoint directory; resume reuses original batch ID")
    judge.add_argument("--batch-id", required=True)
    judge.add_argument("--limit", type=int, default=1)
    judge.add_argument("--offset", type=int, default=0,
                       help="Skip this many frozen input rows; use a new batch ID for each different slice")
    judge.add_argument("--model-id")
    judge.add_argument("--env-file")
    judge.add_argument("--live", action="store_true")
    judge.add_argument("--shared-ledger", action="store_true", help="Confirm DATABASE_URL is the team's existing shared backend DB")
    judge.add_argument("--dry-run", action="store_true", help="Validate inputs and wire size without dispatch or ledger writes")
    judge.add_argument("--pilot-plan", help="Frozen pilot manifest; only its train split may receive teacher suggestions")
    judge.add_argument("--max-cost-rub", help="Stop BETWEEN calls after this known batch cost; next receipt may exceed it")
    judge.add_argument("--max-seconds", type=int, default=1200)
    merge = commands.add_parser("export-labels", help="Export completed teacher checkpoints as a training dataset")
    merge.add_argument("--batch-dir", required=True)
    merge.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Train a local, versioned JSON logistic-regression artifact")
    train.add_argument("--input", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--allow-teacher", action="store_true")
    train.add_argument("--split", choices=["article", "company"], default="article")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--min-recall", type=float, help="Select threshold on independent human validation only")
    train.add_argument("--iterations", type=int, default=1000)
    train.add_argument("--validation-input", help="Independent human validation, prepared before teacher labeling")
    train.add_argument("--test-input", help="Independent human test; required together with --validation-input")
    train.add_argument("--pilot-plan", help="Bind labeled inputs to the pilot's original pre-label splits")
    predict = commands.add_parser("predict", help="Local shadow predictions; never enqueue paid work")
    predict.add_argument("--input", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", required=True)
    for command in (train, predict):
        command.add_argument("--model-dir", default="data/models/e5-base")
        command.add_argument("--model-manifest", default="knowledge/embedding-model.json")
        command.add_argument("--threads", type=int, default=2)
    return root


def _train(args):
    import numpy as np
    from ..common import artifact_dir, write_json
    from ..embeddings import E5Encoder
    from .evaluation import assert_disjoint_splits, evaluate_predictions, split_rows, _independent_human
    from .features import FeatureBuilder
    from .model import predict, train_model
    rows = read_dataset(args.input)
    validation_input, test_input = getattr(args, "validation_input", None), getattr(args, "test_input", None)
    if bool(validation_input) != bool(test_input):
        raise KnowledgeError("Provide both --validation-input and --test-input for frozen independent evaluation")
    if validation_input:
        splits = {"train": rows, "validation": read_dataset(validation_input), "test": read_dataset(test_input),
                  "warnings": [], "mode": "frozen_independent_human"}
        assert_disjoint_splits(splits["train"], splits["validation"], splits["test"], company_holdout=args.split == "company")
        for name in ("validation", "test"):
            if any(not _independent_human(row) for row in splits[name]):
                raise KnowledgeError("Frozen validation/test requires independently human-reviewed labels, never teacher gold")
        rows = [row for name in ("train", "validation", "test") for row in splits[name]]
    else:
        splits = split_rows(rows, mode=args.split, seed=args.seed)
    pilot_provenance = None
    if getattr(args, "pilot_plan", None):
        if not validation_input:
            raise KnowledgeError("A frozen pilot requires explicit --validation-input and --test-input")
        from .pilot import validate_pilot_splits
        pilot_provenance = validate_pilot_splits(args.pilot_plan, splits)
        splits["warnings"].extend(pilot_provenance["warnings"])
    # Fail before expensive embedding when a split cannot support training.
    for name in ("train", "validation", "test"):
        if not splits[name]:
            raise KnowledgeError(f"Empty {name} split; collect more independent articles/companies")
        if any(not row.get("label") for row in splits[name]):
            raise KnowledgeError("Training/evaluation requires labels; unknown labels must not become negative examples")
    if len({r["label"]["grade"] >= 4 for r in splits["train"] if r.get("label")}) != 2:
        raise KnowledgeError("Training split needs positive and negative labels; collect more examples")
    builder = FeatureBuilder(E5Encoder(args.model_dir, args.model_manifest, threads=args.threads))
    # Embed each pair once, then slice by immutable pair ID; scaler is fitted in train_model on train only.
    all_vectors = builder.transform(rows)
    positions = {r["pair_id"]: i for i, r in enumerate(rows)}
    matrices = {name: all_vectors[np.asarray([positions[r["pair_id"]] for r in splits[name]], dtype=int)]
                for name in ("train", "validation", "test")}
    artifact = train_model(splits["train"], matrices["train"], feature_manifest=builder.manifest,
                           threshold=args.threshold, allow_teacher=args.allow_teacher,
                           validation_rows=splits["validation"], validation_vectors=matrices["validation"],
                           min_recall=args.min_recall, iterations=args.iterations)
    predictions = predict(artifact, matrices["test"], feature_manifest=builder.manifest)
    report = evaluate_predictions(splits["test"], predictions, threshold=artifact["decision"]["threshold"])
    report["estimate_scope"] = ("frozen_independent_human" if validation_input else
                                "held_out_articles" if args.split == "article" else "held_out_companies_and_articles")
    report["dataset_summary"] = dataset_summary(rows)
    if pilot_provenance is not None:
        report["frozen_pilot"] = pilot_provenance
    report["split_summary"] = {"mode": splits.get("mode", args.split), "seed": args.seed, "warnings": splits["warnings"],
                               "counts": {k: len(splits[k]) for k in ("train", "validation", "test")},
                               "pair_ids": {k: [r["pair_id"] for r in splits[k]] for k in ("train", "validation", "test")}}
    # Keep external dataset/split provenance in the separate report; model is content-addressed.
    with artifact_dir(args.output_dir) as stage:
        stage.chmod(0o700)
        write_json(stage / "model.json", artifact)
        write_json(stage / "evaluation.json", report)
        write_json(stage / "features.json", builder.manifest)
        for path in stage.iterdir():
            path.chmod(0o600)
    return {"model_version": artifact["model_version"], "status": "shadow_only",
            "dataset": dataset_summary(rows), "held_out_pairs": report["count"],
            "human_benchmark_eligible": report["human_benchmark_eligible"],
            "evaluation": str(Path(args.output_dir) / "evaluation.json")}


def _judge(args):
    from datetime import datetime, timezone
    dry_run = getattr(args, "dry_run", False)
    if not dry_run and (not args.live or not args.shared_ledger):
        raise KnowledgeError("Paid labeling requires --live --shared-ledger; prepare-labels works offline")
    if not 1 <= args.limit <= 100:
        raise KnowledgeError("Paid batch limit must be 1..100; no unbounded bulk generation")
    offset = getattr(args, "offset", 0)
    if type(offset) is not int or not 0 <= offset <= 200000:
        raise KnowledgeError("Teacher input offset must be 0..200000")
    from app.intelligence.gateway import load_env_file
    from app.intelligence.settings import Settings
    from app.intelligence.provider import Provider
    from app.intelligence.storage import BudgetDay
    from app.core import make_session_factory
    from .api_runner import batch_plan, cost_stop_micros, read_call_ledger, run_api_batch
    if args.env_file:
        load_env_file(args.env_file)
    if getattr(args, "pilot_plan", None):
        from .pilot import validate_pilot_input
        validate_pilot_input(args.pilot_plan, args.input, split="train")
    cost_stop_micros(getattr(args, "max_cost_rub", None))
    from itertools import islice
    from .dataset import iter_dataset
    rows = list(islice(iter_dataset(args.input, limit=offset + args.limit), offset, offset + args.limit))
    settings = Settings.from_env()
    model = args.model_id or settings.analysis_model
    manifest, preflight = batch_plan(rows, batch_id=args.batch_id, model_id=model,
                                     model=settings.models.get(model))
    if dry_run:
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_private_json(destination / "preflight.json", preflight)
        return {"status": "dry_run", "prepared": len(rows), "paid_calls": 0,
                "input_rows_skipped": offset,
                "model_id": model, "wire_bound_checked": preflight["wire_bound_checked"],
                "shared_ledger_touched": False, "preflight": str(destination / "preflight.json")}
    # An accidental local sqlite ledger would bypass the deployed team's shared budget.
    url = os.getenv("DATABASE_URL", "")
    if not url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise KnowledgeError("Use the existing shared PostgreSQL DATABASE_URL on the backend server")
    settings.require_ready()
    if settings.daily_limit_micros > 500_000_000:
        raise KnowledgeError("Labeling must use the team's shared daily cap of at most 500 RUB")
    engine, factory = make_session_factory(url)
    try:
        with factory() as db:
            day = db.get(BudgetDay, datetime.now(timezone.utc).date().isoformat())
            if day is not None and day.limit_micros > 500_000_000:
                raise KnowledgeError("Existing shared ledger cap exceeds 500 RUB; keep current cap, do not reset it")
        provider = Provider(factory, settings)
        return run_api_batch(rows, provider, destination=args.output, batch_id=args.batch_id,
                             model_id=model, ledger_reader=lambda run_id: read_call_ledger(factory, run_id),
                             max_cost_rub=getattr(args, "max_cost_rub", None),
                             max_seconds=getattr(args, "max_seconds", 1200))
    finally:
        engine.dispose()


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "inspect":
            result = dataset_summary(read_dataset(args.input))
        elif args.command == "companies":
            from .sampling import load_company_snapshots
            _, result = load_company_snapshots(args.csv)
            write_private_json(args.output, result)
        elif args.command == "collect":
            from .corpus import collect
            result = collect(args.sources, args.db, max_articles=args.max_articles, max_requests=args.max_requests,
                             max_total_bytes=args.max_bytes, max_runtime_seconds=args.max_seconds,
                             goal=args.goal, max_source_articles=args.max_source_articles,
                             published_after=args.published_after, published_before=args.published_before)
        elif args.command == "collect-job":
            from .collect_job import collect_job, job_summary
            if not 1 <= args.rounds <= 4:
                raise KnowledgeError("Collection rounds must be 1..4")
            def progress(row):
                print(json.dumps({"event": row["event"], "articles": row["total"], "goal": row["goal"],
                                  "new_in_batch": row.get("batch", {}).get("new"),
                                  "requests_used": row.get("requests_used"),
                                  "stop_reason": row.get("stop_reason")}, ensure_ascii=False), flush=True)
            for round_index in range(args.rounds):
                before = job_summary(args.run_dir)["total"] if (Path(args.run_dir) / "job.json").exists() else 0
                result = collect_job(args.sources, args.run_dir, goal=args.goal, max_workers=args.workers,
                                     max_runtime_seconds=args.max_seconds, max_requests=args.max_requests,
                                     max_total_bytes=args.max_bytes, max_source_articles=args.max_source_articles,
                                     published_after=args.published_after, published_before=args.published_before,
                                     progress=progress)
                result["rounds_completed"] = round_index + 1
                if result["total"] >= args.goal or result["total"] <= before or result["stop_reason"] == "queue_exhausted_or_source_quotas":
                    break
        elif args.command == "job-status":
            from .collect_job import job_summary
            result = job_summary(args.run_dir)
        elif args.command == "export-links":
            from .corpus import iter_articles
            from .collect_job import iter_job_articles
            articles = iter_job_articles(args.job_dir) if args.job_dir else iter_articles(args.db)
            links = [{k: article[k] for k in ("article_id", "source_id", "url", "title", "published_at", "fetched_at")}
                     for article in articles]
            write_private_json(args.output, {"schema_version": "alphich-corpus-links-v1", "articles": links})
            result = {"exported": len(links), "output": args.output, "paid_calls": 0}
        elif args.command == "corpus-status":
            from .corpus import corpus_summary
            result = corpus_summary(args.db)
        elif args.command == "sample":
            from .corpus import iter_articles
            from .collect_job import iter_job_articles
            from .sampling import load_company_snapshots, sample_pairs
            companies, census = load_company_snapshots(args.companies_csv)
            # Publisher topics organize sampling; they are not model features or relevance labels.
            articles = []
            source_articles = iter_job_articles(args.job_dir) if args.job_dir else iter_articles(args.db)
            for article in source_articles:
                article["topics"] = article.get("source_metadata", {}).get("topics", [])
                articles.append(article)
            rows, report = sample_pairs(articles, companies, as_of=args.as_of, role=args.role,
                                       pairs_per_article=args.pairs_per_article, limit=args.limit, seed=args.seed)
            write_dataset(args.output, rows, sampling=report, company_census=census)
            write_private_json(args.report, {"sampling": report, "company_census": census, "dataset": dataset_summary(rows)})
            result = {"pairs": len(rows), "articles": report["usable_unique_articles_visited"],
                      "sampled_companies": report["sampled_company_count"], "labels_generated": 0, "paid_calls": 0}
        elif args.command == "import-fixtures":
            rows, quarantine = import_fixtures([read_json(p) for p in args.labels], read_json(args.events), read_json(args.profiles))
            write_dataset(args.output, rows, scope="provisional_teacher_development", quarantine=quarantine)
            result = dict(dataset_summary(rows), quarantined=len(quarantine))
        elif args.command == "prepare-labels":
            from .teacher import prepare_judgement
            if not 1 <= args.limit <= 10000:
                raise KnowledgeError("Request preparation limit must be 1..10000")
            requests = [prepare_judgement(r) for r in read_dataset(args.input, limit=args.limit)]
            write_private_json(args.output, {"schema_version": "alphich-relevance-requests-v1", "requests": requests})
            result = {"prepared": len(requests), "paid_calls": 0, "labels_are": "not_created"}
        elif args.command == "judge":
            result = _judge(args)
        elif args.command == "export-labels":
            from .teacher import validate_checkpoint
            folder = Path(args.batch_dir)
            batch = read_json(folder / "batch.json")
            rows = []
            for path in sorted(folder.glob("pair_*.json")):
                if path.name.endswith(".error.json"):
                    continue
                row = validate_checkpoint(read_json(path), batch)
                rows.append(row)
            if not rows:
                raise KnowledgeError("Batch has no completed labels")
            write_dataset(args.output, rows, batch=batch, labels_are="teacher_proposals_not_human_gold")
            result = dataset_summary(rows)
        elif args.command == "train":
            result = _train(args)
        elif args.command == "predict":
            from ..embeddings import E5Encoder
            from .features import FeatureBuilder
            from .model import predict
            rows = read_dataset(args.input)
            builder = FeatureBuilder(E5Encoder(args.model_dir, args.model_manifest, threads=args.threads))
            predictions = predict(read_json(args.model), builder.transform(rows), feature_manifest=builder.manifest)
            write_private_json(args.output, {"schema_version": "alphich-relevance-predictions-v1",
                                            "predictions": [{"pair_id": r["pair_id"], **p} for r, p in zip(rows, predictions)]})
            result = {"predicted": len(predictions), "paid_calls": 0, "status": "shadow_only"}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        # DB/HTTP exception text may contain connection details. Return bounded
        # errors rather than a credential-bearing traceback from a batch CLI.
        code = getattr(error, "code", None) if type(error).__module__ == "app.intelligence.settings" else None
        message = str(error) if isinstance(error, KnowledgeError) else code or type(error).__name__
        print(json.dumps({"status": "failed", "error": message}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
