"""Private, unlabeled API-teacher pilot with a frozen blind holdout.

Amount eligibility describes the organizer's supplied revenue field. It does
not establish the revenue of a particular legal entity or an inferred segment.
No labels, external company enrichment or paid calls are produced here.
"""
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..common import KnowledgeError, artifact_dir, digest, file_hash, read_json
from .dataset import input_hash, instant, read_dataset, validate_pair, write_dataset, write_private_json
from .evaluation import assert_disjoint_splits, split_rows
from .sampling import okved_sector, sample_pairs

PILOT_SCHEMA = "alphich-api-labeling-pilot-v1"
SPLITS = ("train", "validation", "test")


def select_target_companies(companies, *, min_revenue_rub="15000000000", revenue_period="2025",
                            max_companies=24, required_companies=()):
    """Balance eligible sectors; never infer units or bypass eligibility for a demo."""
    try:
        minimum = Decimal(str(min_revenue_rub))
        if not minimum.is_finite() or minimum < 0:
            raise ValueError()
    except (InvalidOperation, ValueError, TypeError):
        raise KnowledgeError("Minimum revenue must be a finite nonnegative RUB amount") from None
    if type(max_companies) is not int or not 1 <= max_companies <= 200:
        raise KnowledgeError("Pilot company limit must be 1..200")
    eligible, excluded, by_sector = {}, Counter(), defaultdict(list)
    for company in companies:
        facts = {f["name"]: f for f in company["facts"]}
        values = {name: fact["value"] for name, fact in facts.items()}
        needed = ("revenue_raw", "revenue_unit", "revenue_currency", "revenue_period", "revenue_scope")
        if any(name not in facts or not facts[name].get("source_ref") for name in needed):
            excluded["missing_revenue_provenance"] += 1
            continue
        if values["revenue_unit"] != "RUB" or values["revenue_currency"] != "RUB":
            excluded["revenue_not_confirmed_rub"] += 1
            continue
        if values["revenue_period"] != revenue_period:
            excluded["different_or_unknown_revenue_period"] += 1
            continue
        try:
            amount = Decimal(str(values["revenue_raw"]))
            if not amount.is_finite() or amount <= 0 or amount == 1000:
                raise ValueError()
        except (InvalidOperation, ValueError, TypeError):
            excluded["invalid_or_placeholder_revenue"] += 1
            continue
        if amount <= minimum:
            excluded["revenue_at_or_below_threshold"] += 1
            continue
        cid = company["company_id"]
        if cid in eligible:
            raise KnowledgeError("Company cohort contains duplicate identifiers")
        eligible[cid] = company
        by_sector[okved_sector(values.get("okved_code"))].append(cid)
    required = sorted(set(required_companies))
    if len(required) > max_companies or any(cid not in eligible for cid in required):
        raise KnowledgeError("Required companies must exist, meet the cohort rule and fit the company limit")
    selected = list(required)
    buckets = {sector: sorted(ids) for sector, ids in sorted(by_sector.items())}
    # Source order is immaterial. A required demo uses its sector's first slot.
    use = Counter(okved_sector(next(f["value"] for f in eligible[cid]["facts"] if f["name"] == "okved_code"))
                  if any(f["name"] == "okved_code" for f in eligible[cid]["facts"]) else "unknown" for cid in selected)
    while len(selected) < min(max_companies, len(eligible)):
        available = {sector: [cid for cid in ids if cid not in selected] for sector, ids in buckets.items()}
        available = {sector: ids for sector, ids in available.items() if ids}
        sector = min(available, key=lambda key: (use[key], key))
        selected.append(available[sector][0])
        use[sector] += 1
    if not selected:
        raise KnowledgeError("No source-bound companies meet the pilot revenue rule")
    cohort = [eligible[cid] for cid in selected]
    return cohort, {"eligible_company_count": len(eligible), "selected_company_count": len(cohort),
        "input_company_count": len(companies), "excluded_counts": dict(sorted(excluded.items())),
        "selected_sector_counts": dict(sorted(use.items())), "required_company_ids": required,
        "rule": {"revenue_rub_strictly_greater_than": str(minimum), "revenue_period": revenue_period},
        "limitations": ["Revenue scope remains the supplied company/group scope, never inferred from amount",
                        "Revenue selection is a sampling cohort, not proof of a bank business segment"]}


def prepare_pilot(articles, companies, output_dir, *, as_of, limit=600, seed=42, roles=("cfo", "ceo"),
                  max_age_days=45, min_revenue_rub="15000000000", revenue_period="2025",
                  max_companies=24, required_companies=(), source_metadata=None):
    """Freeze cluster-disjoint splits before teacher or human labels are available."""
    cutoff = instant(as_of)
    if type(limit) is not int or not 12 <= limit <= 3000:
        raise KnowledgeError("Pilot pair limit must be 12..3000")
    if not roles or len(set(roles)) != len(roles) or any(role not in {"cfo", "ceo"} for role in roles):
        raise KnowledgeError("Pilot roles must be distinct CFO/CEO values")
    if type(max_age_days) is not int or not 1 <= max_age_days <= 730:
        raise KnowledgeError("Pilot publication window must be 1..730 days")
    cohort, cohort_report = select_target_companies(companies, min_revenue_rub=min_revenue_rub,
        revenue_period=revenue_period, max_companies=max_companies, required_companies=required_companies)
    recent, exclusions = [], Counter()
    for article in articles:
        try:
            published = instant(article.get("published_at"))
        except KnowledgeError:
            exclusions["missing_or_invalid_publication_date"] += 1
            continue
        age = (cutoff - published).total_seconds() / 86400
        if age < 0 or age > max_age_days:
            exclusions["outside_publication_window"] += 1
            continue
        recent.append(article)
    base, sampling = sample_pairs(recent, cohort, as_of=as_of, role=roles[0],
        pairs_per_article=3, limit=limit // len(roles), seed=seed)
    # Recalculate the content-addressed ID when changing role; all roles and
    # company contrasts for one article are subsequently joined in one split.
    rows = []
    for row in base:
        for role in roles:
            value = {key: val for key, val in row.items() if key != "pair_id"}
            rows.append(validate_pair({**value, "role": role}))
    if len({r["company"]["company_id"] for r in rows}) < 2:
        raise KnowledgeError("A pilot needs at least two sampled company contexts")
    splits = split_rows(rows, seed=seed, require_labels=False)
    manifest = {"schema_version": PILOT_SCHEMA, "as_of": cutoff.isoformat(), "seed": seed,
        "labels_generated": 0, "paid_calls": 0, "roles": list(roles), "cohort": cohort_report,
        "sampling": sampling, "publication_window_days": max_age_days,
        "article_exclusion_counts": dict(sorted(exclusions.items())),
        "source_metadata": source_metadata or {},
        "split_fingerprint": splits["split_fingerprint"], "splits": {},
        "limitations": ["Topic/sector arms are sampling heuristics, never positive/negative labels",
            "This is a stratified pilot, not a representative production benchmark",
            "Company holdout is not established by the article split",
            "Paraphrased reprints need externally checked cluster IDs",
            "Publication age is a sampling window, not a business grade",
            "Profiles are current supplied snapshots, not historical company states"]}
    with artifact_dir(output_dir) as stage:
        stage.chmod(0o700)
        for name in SPLITS:
            filename = name + ".jsonl"
            write_dataset(stage / filename, splits[name])
            manifest["splits"][name] = {"file": filename, "count": len(splits[name]),
                "pair_ids": [r["pair_id"] for r in splits[name]],
                "input_hashes": {r["pair_id"]: input_hash(r) for r in splits[name]},
                "label_policy": "api_teacher_proposal_then_human_review" if name == "train" else "blind_two_reviewers_and_arbitration"}
        manifest["outputs"] = {name + ".jsonl": file_hash(stage / (name + ".jsonl")) for name in SPLITS}
        manifest["plan_id"] = "pilot_" + digest(manifest)
        write_private_json(stage / "manifest.json", manifest)
    return {"plan_id": manifest["plan_id"], "selected_companies": len(cohort),
        "sampled_companies": sampling["sampled_company_count"],
        "split_counts": {name: len(splits[name]) for name in SPLITS}, "paid_calls": 0,
        "manifest": str(Path(output_dir) / "manifest.json")}


def load_pilot(plan_path):
    """Validate all frozen inputs and their cross-split identity, not just a name."""
    plan_path = Path(plan_path)
    if plan_path.is_symlink():
        raise KnowledgeError("Pilot manifest must not be symlinked")
    manifest = read_json(plan_path)
    identity = {key: val for key, val in manifest.items() if key != "plan_id"}
    if (manifest.get("schema_version") != PILOT_SCHEMA or manifest.get("plan_id") != "pilot_" + digest(identity)
            or set(manifest.get("splits", {})) != set(SPLITS)
            or set(manifest.get("outputs", {})) != {name + ".jsonl" for name in SPLITS}):
        raise KnowledgeError("Invalid immutable API pilot manifest")
    rows = {}
    for name in SPLITS:
        spec = manifest["splits"][name]
        filename = name + ".jsonl"
        path = plan_path.parent / filename
        if spec.get("file") != filename or path.is_symlink() or file_hash(path) != manifest["outputs"][filename]:
            raise KnowledgeError("Pilot input file checksum mismatch")
        rows[name] = read_dataset(path)
        if (spec.get("count") != len(rows[name]) or spec.get("pair_ids") != [r["pair_id"] for r in rows[name]]
                or spec.get("input_hashes") != {r["pair_id"]: input_hash(r) for r in rows[name]}
                or any(r.get("label") for r in rows[name])):
            raise KnowledgeError("Pilot split changed after planning")
    assert_disjoint_splits(rows["train"], rows["validation"], rows["test"])
    return manifest, rows


def validate_pilot_input(plan_path, input_path, *, split="train"):
    manifest, _ = load_pilot(plan_path)
    if split != "train":
        raise KnowledgeError("API teacher must not label the pilot's blind validation/test splits")
    expected = Path(plan_path).parent / manifest["splits"][split]["file"]
    if Path(input_path).is_symlink() or Path(input_path).resolve() != expected.resolve():
        raise KnowledgeError("Teacher input must be the exact frozen pilot train file")
    return manifest


def validate_pilot_splits(plan_path, splits):
    """Bind reviewed subsets to their original inputs and pre-label split.

    Incomplete review is allowed but its coverage is explicit; moving a teacher
    label or a changed company snapshot into a holdout is never allowed.
    """
    manifest, _ = load_pilot(plan_path)
    coverage = {}
    for name in SPLITS:
        expected = manifest["splits"][name]["input_hashes"]
        seen = set()
        for source in splits[name]:
            row = validate_pair(source)
            if row["pair_id"] in seen or expected.get(row["pair_id"]) != input_hash(row):
                raise KnowledgeError("Reviewed input was changed, repeated or moved across frozen pilot splits")
            seen.add(row["pair_id"])
        coverage[name] = {"planned": len(expected), "reviewed_or_labeled": len(seen), "complete": len(seen) == len(expected)}
    assert_disjoint_splits(splits["train"], splits["validation"], splits["test"])
    return {"plan_id": manifest["plan_id"], "coverage": coverage,
            "sampling_scope": "stratified_pilot_not_production_benchmark", "company_holdout": False,
            "limitations": manifest["limitations"],
            "warnings": ["stratified_pilot_is_not_production_benchmark", "company_transfer_not_established",
                *[name + "_frozen_split_incompletely_labeled" for name, row in coverage.items() if not row["complete"]]]}
