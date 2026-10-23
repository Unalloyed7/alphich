"""Prepare a private target-cohort API labeling pilot; no API access is used."""
import argparse
import json

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.pilot import prepare_pilot
from alphich_kb.relevance_ml.sampling import load_company_snapshots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--job-dir")
    source.add_argument("--db")
    parser.add_argument("--companies-csv", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--max-companies", type=int, default=24)
    parser.add_argument("--min-revenue-rub", default="15000000000")
    parser.add_argument("--revenue-period", default="2025")
    parser.add_argument("--max-age-days", type=int, default=45)
    parser.add_argument("--required-company", action="append", default=[])
    parser.add_argument("--roles", nargs="+", choices=("cfo", "ceo"), default=["cfo", "ceo"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        from alphich_kb.relevance_ml.corpus import iter_articles
        from alphich_kb.relevance_ml.collect_job import iter_job_articles
        companies, census = load_company_snapshots(args.companies_csv)
        articles = iter_job_articles(args.job_dir) if args.job_dir else iter_articles(args.db)
        snapshots = []
        for article in articles:
            article["topics"] = article.get("source_metadata", {}).get("topics", [])
            snapshots.append(article)
        result = prepare_pilot(snapshots, companies, args.output_dir, as_of=args.as_of, limit=args.limit,
            max_companies=args.max_companies, min_revenue_rub=args.min_revenue_rub,
            revenue_period=args.revenue_period, max_age_days=args.max_age_days,
            required_companies=args.required_company, roles=args.roles, seed=args.seed,
            source_metadata={"company_csv_sha256": census["source_sha256"],
                             "organizer_sampling_schema": census["schema_version"]})
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except KnowledgeError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
