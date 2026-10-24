"""Free candidate mining and blind machine agreement; never human gold.

Queues reference exact frozen offsets. Similarity and lexical hints only decide
which pairs to ask about; neither creates a label or authorizes a runtime card.
"""
from __future__ import annotations

import copy
import math
import random
import re
from collections import Counter
from pathlib import Path

from ..common import KnowledgeError, artifact_dir, digest, file_hash, read_json
from .dataset import input_hash, read_dataset, validate_pair, write_dataset, write_private_json
from .teacher import validate_judgement

QUEUE_SCHEMA = "alphich-candidate-queue-v1"
SUPPLEMENT_SCHEMA = "alphich-matched-train-v1"
HARD_NEGATIVE = re.compile(r"законопроект|предлага[ею]т|пилотн|тестир|семинар|вебинар|провели занятие", re.I)


def score_candidates(rows, encoder):
    """Retrieve against activity/confirmed finance, excluding repetitive metadata.

    Full-profile pooling can rank unrelated finance above a company's activity.
    Full article coverage is retained; headline and body each affect priority.
    These are retrieval scores, not exposure/scale/applicability assessments.
    """
    import numpy as np
    from .features import FeatureBuilder
    from ..personalization.facets import sector_terms
    builder = FeatureBuilder(encoder, text_strategy="asymmetric_retrieval")
    articles, scores, details = {}, [], []
    product_text = {
        "loan": "Ключевая ставка Банка России и условия кредитования бизнеса, стоимость финансирования.",
        "deposit": "Ключевая ставка Банка России и ставки по депозитам юридических лиц, размещение ликвидности.",
        "lease": "Условия лизингового финансирования оборудования и транспорта для бизнеса.",
        "factoring": "Факторинг, отсрочка платежей и финансирование оборотного капитала бизнеса.",
    }
    product_hints = {"loan": r"ключев.{0,12}ставк|кредит|за[её]мн|финансирован",
                     "deposit": r"ключев.{0,12}ставк|депозит|вклад|ликвидност",
                     "lease": r"лизинг", "factoring": r"факторинг|отсрочк.{0,18}платеж|дебиторск"}
    for source in rows:
        row = validate_pair(source)
        article = row["article"]
        key = digest([article["title"], article["text"]])
        if key not in articles:
            body = article["title"] + "\n\n" + article["text"]
            chunks = builder._chunks(body, "passage: ")
            pooled, _ = builder._pool(chunks, builder._vectors(chunks, article=True))
            title_chunks = builder._chunks(article["title"], "passage: ")
            title, _ = builder._pool(title_chunks, builder._vectors(title_chunks, article=True))
            articles[key] = pooled, title, len(chunks)
        body_vector, title_vector, chunks = articles[key]
        facts = {f["name"]: f for f in row["company"]["facts"]}
        values = {name: fact["value"] for name, fact in facts.items()}
        industry = values.get("industry") or {}
        name = values.get("okved_name") or industry.get("name") or ""
        code = values.get("okved_code") or industry.get("code")
        if not name:
            name = ("; ".join(sector_terms(code)) or "Деятельность по коду ОКВЭД " + code) if code else ""
        facets = []
        if name:
            facets.append(("activity", "Отраслевые события и условия деятельности: " + name))
        products = values.get("products") or {}
        for product, text in product_text.items():
            if (values.get(product, products.get(product)) is True
                    and re.search(product_hints[product], article["title"], re.I)):
                facets.append((product, text))
        if not facets:
            raise KnowledgeError("Candidate scoring requires confirmed activity or a financial product")
        candidates = []
        for kind, text in facets:
            query_chunks = builder._chunks(text, "query: ")
            vector, _ = builder._pool(query_chunks, builder._vectors(query_chunks))
            score = float(np.clip(.7 * (body_vector @ vector) + .3 * (title_vector @ vector), -1, 1))
            candidates.append((score, kind))
        score, winner = max(candidates)
        scores.append(score)
        details.append({"retrieval_facet": winner, "full_article_chunks": chunks,
                        "article_coverage": "complete", "facets_checked": [kind for kind, _ in facets]})
    return scores, details, {"version": "candidate-topic-retrieval-v3", "encoder_space_id": encoder.space_id,
        "article_strategy": "asymmetric_query_passage; .7 complete-body-pool + .3 headline-pool",
        "company_strategy": "activity and confirmed finance facets; no registration or repeated absence statements",
        "finance_facets": "confirmed product plus matching headline topic; retrieval hint only, never a label",
        "max_across_facets": True, "labels_or_grades_used": False}


def _article_identity(article):
    from urllib.parse import urlsplit, urlunsplit
    url = urlsplit(article["url"])
    normalized_url = urlunsplit((url.scheme.lower(), url.netloc.lower(), url.path.rstrip("/"), url.query, ""))
    return {"article_id": article["article_id"], "cluster_id": article["cluster_id"],
            "content": digest(" ".join(article["text"].casefold().split())), "url": normalized_url}


def _check_holdout(rows, holdout):
    keys = {name: set(item[name] for item in holdout) for name in ("article_id", "cluster_id", "content", "url")}
    for row in rows:
        if any(value in keys[name] for name, value in _article_identity(row["article"]).items()):
            raise KnowledgeError("Supplementary train overlaps the frozen holdout")


def validate_labeling_input(plan_path, input_path):
    """A supplement is separate from, and cannot rewrite, the original pilot."""
    manifest = read_json(plan_path)
    if manifest.get("schema_version") != SUPPLEMENT_SCHEMA:
        from .pilot import validate_pilot_input
        return validate_pilot_input(plan_path, input_path)
    identity = {k: v for k, v in manifest.items() if k != "plan_id"}
    expected = Path(plan_path).parent / "train.jsonl"
    if (Path(plan_path).is_symlink() or Path(input_path).is_symlink()
            or Path(input_path).resolve() != expected.resolve()
            or manifest.get("plan_id") != "matched_" + digest(identity)
            or manifest.get("train_sha256") != file_hash(expected)):
        raise KnowledgeError("Supplementary train manifest or source changed")
    rows = read_dataset(expected)
    if manifest.get("input_hashes") != {r["pair_id"]: input_hash(r) for r in rows} or any(r.get("label") for r in rows):
        raise KnowledgeError("Supplementary train rows changed after freezing")
    _check_holdout(rows, manifest["frozen_holdout"])
    return manifest


def build_matched_train(plan_path, output_dir, encoder, *, excluded=(), seed=42, companies=None, company_source=None,
                        articles=None, article_source=None):
    """Rematch train articles to the whole existing cohort, never held-out articles.

    Select two activity matches and one random contrast per article, both roles.
    All targets remain unlabeled; random contrasts can also be relevant.
    """
    from .pilot import load_pilot
    pilot, splits = load_pilot(plan_path)
    if companies is None:
        companies = {r["company"]["company_id"]: r["company"] for r in splits["train"]}
    else:
        from .pilot import select_target_companies
        selected, _ = select_target_companies(companies, max_companies=200)
        companies = {c["company_id"]: c for c in selected}
    holdout = list({digest(_article_identity(r["article"])): _article_identity(r["article"])
                   for name in ("validation", "test") for r in splits[name]}.values())
    article_exclusions = []
    if articles is None:
        articles = {r["article"]["article_id"]: r["article"] for r in splits["train"]}
    else:
        supplied, articles = list(articles), {}
        for article in supplied:
            probe = validate_pair({"company": next(iter(companies.values())), "article": article,
                                   "as_of": pilot["as_of"], "role": "cfo"})
            try:
                _check_holdout([probe], holdout)
            except KnowledgeError:
                article_exclusions.append({"article_id": probe["article"]["article_id"], "reason": "frozen_holdout_overlap"})
                continue
            aid = probe["article"]["article_id"]
            if aid in articles and articles[aid] != probe["article"]:
                raise KnowledgeError("Conflicting versions of a supplied article")
            articles[aid] = probe["article"]
        if not articles:
            raise KnowledgeError("No supplied articles remain outside the frozen holdout")
    if len(companies) < 3:
        raise KnowledgeError("Rematching needs at least three company contexts")
    # Match activity for industry news. Explicit financial headlines may also
    # use confirmed products; generic profile boilerplate never ranks a pair.
    scoring_rows, originals = [], []
    for aid, article in sorted(articles.items()):
        for cid, company in sorted(companies.items()):
            row = validate_pair({"company": company, "article": article, "as_of": pilot["as_of"], "role": "cfo"})
            originals.append(row)
            topic = copy.deepcopy(row)
            topic.pop("pair_id")
            names = {"industry", "okved_code", "okved_name"}
            if re.search(r"ключев.{0,12}ставк|депозит|кредит|лизинг|факторинг", article["title"], re.I):
                names |= {"products", "loan", "deposit", "lease", "factoring"}
            topic["company"]["facts"] = [f for f in company["facts"] if f["name"] in names]
            scoring_rows.append(validate_pair(topic))
    scores, details, scoring = score_candidates(scoring_rows, encoder)
    rng, rows, selections = random.Random(seed), [], []
    width = len(companies)
    for start in range(0, len(originals), width):
        ranked = sorted(range(start, start + width), key=lambda i: (-scores[i], originals[i]["company"]["company_id"]))
        choices = [(i, "topic_match") for i in ranked[:2]]
        choices.append((rng.choice(ranked[2:]), "random_contrast"))
        for i, arm in choices:
            for role in ("cfo", "ceo"):
                row = {k: v for k, v in originals[i].items() if k != "pair_id"}
                row = validate_pair({**row, "role": role})
                if row["pair_id"] in excluded:
                    continue
                rows.append(row)
                selections.append({"pair_id": row["pair_id"], "arm": arm, "retrieval_similarity": scores[i]})
    _check_holdout(rows, holdout)
    with artifact_dir(output_dir) as stage:
        stage.chmod(0o700)
        write_dataset(stage / "train.jsonl", rows)
        manifest = {"schema_version": SUPPLEMENT_SCHEMA, "parent_plan_id": pilot["plan_id"],
            "parent_plan_sha256": file_hash(plan_path), "as_of": pilot["as_of"], "seed": seed,
            "train_sha256": file_hash(stage / "train.jsonl"), "frozen_holdout": holdout,
            "input_hashes": {r["pair_id"]: input_hash(r) for r in rows}, "scoring": scoring,
            "company_source": company_source,
            "article_source": article_source, "article_exclusions": article_exclusions,
            "selection": selections, "excluded_attempts": sorted(set(excluded)), "paid_calls": 0,
            "limitations": ["Separate train supplement; original pilot is unchanged",
                "New event diversity comes only from explicitly supplied new source articles",
                "All pairs are unlabeled; selected arms are not positive/negative truth",
                "Paraphrased reprints can evade exact content and publisher cluster checks"]}
        manifest["plan_id"] = "matched_" + digest(manifest)
        write_private_json(stage / "manifest.json", manifest)
    return {"pairs": len(rows), "companies": len(companies), "articles": len(articles), "paid_calls": 0,
            "manifest": str(Path(output_dir) / "manifest.json"), "labels_generated": 0}


def attempted_pairs(batch_dirs):
    """Exclude successes AND failures/unknown attempts, not unstarted reservations.

    Read only explicit batch roots. The caller still needs the shared ledger on
    the server: absence of local files cannot prove that a pair was never sent.
    """
    result = set()
    for root in batch_dirs:
        root = Path(root)
        batch = read_json(root / "batch.json")
        for path in root.glob("pair_*.json"):
            value = read_json(path)
            pid = value.get("pair_id")
            if not isinstance(pid, str):
                raise KnowledgeError("Invalid attempted-pair checkpoint")
            if not path.name.endswith(".error.json"):
                # Historical v4/v5 prompts need not be available to exclude an
                # attempt. Check its bound identity without importing its label.
                value = validate_pair(value)
                label = value.get("label", {})
                if (label.get("batch_id") != batch["batch_id"] or label.get("labeler") != batch["model_id"]
                        or label.get("prompt_hash") != batch["prompt_hash"]
                        or input_hash(value) not in batch["requests"]):
                    raise KnowledgeError("Attempt checkpoint does not match its batch")
            result.add(pid)
        for path in (root / "audit").glob("pair_*.json"):
            value = read_json(path)
            if value.get("batch_id") != batch["batch_id"] or not isinstance(value.get("pair_id"), str):
                raise KnowledgeError("Attempt audit belongs to another batch")
            result.add(value["pair_id"])
    return result


def select_candidates(rows, similarities, *, limit=24, excluded=(), seed=42,
                      max_per_cluster=2, max_per_company=4):
    """Balanced diagnostic selection with immutable offsets, without label access."""
    if type(limit) is not int or not 4 <= limit <= 100:
        raise KnowledgeError("Candidate limit must be 4..100")
    if any(type(n) is not int or n < 1 for n in (max_per_cluster, max_per_company)):
        raise KnowledgeError("Candidate diversity caps must be positive integers")
    rows = [validate_pair(row) for row in rows]
    if len(similarities) != len(rows) or any(type(v) not in {int, float} or not math.isfinite(v)
            or not -1 <= v <= 1 for v in similarities):
        raise KnowledgeError("Candidate similarities must be finite E5 cosines aligned with rows")
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise KnowledgeError("Candidate inputs contain duplicate pairs")
    excluded = set(excluded)
    eligible = [i for i, r in enumerate(rows) if r["pair_id"] not in excluded and not r.get("label")]
    ranked = sorted(eligible, key=lambda i: (-similarities[i], rows[i]["pair_id"]))
    rng = random.Random(seed)
    random_order = sorted(eligible, key=lambda i: rows[i]["pair_id"])
    rng.shuffle(random_order)
    middle = (len(ranked) - 1) / 2
    boundary = sorted(range(len(ranked)), key=lambda i: (abs(i - middle), rows[ranked[i]]["pair_id"]))
    orders = {
        "high_similarity": ranked,
        "boundary": [ranked[i] for i in boundary],
        "hard_negative_candidate": [i for i in ranked if HARD_NEGATIVE.search(rows[i]["article"]["text"])],
        "random_control": random_order,
    }
    # Reserve each diagnostic arm before filling unused slots with high recall.
    quotas = {"high_similarity": limit // 2, "boundary": limit // 4,
              "hard_negative_candidate": max(1, limit // 8), "random_control": max(1, limit // 8)}
    selected, chosen, company_story = [], set(), set()
    clusters, companies, roles = Counter(), Counter(), Counter()

    def take(order, count, arm):
        if count <= 0 or len(selected) >= limit:
            return
        order = list(order)
        added = 0
        while order:
            # Re-evaluate role balance after every choice; keep score order
            # within each role. Roles never change the underlying E5 score.
            position = min(range(len(order)), key=lambda j: (roles[rows[order[j]]["role"]], j))
            i = order.pop(position)
            row = rows[i]
            cid, cluster = row["company"]["company_id"], row["article"]["cluster_id"]
            story = (cid, cluster)
            if (i in chosen or story in company_story or clusters[cluster] >= max_per_cluster
                    or companies[cid] >= max_per_company):
                continue
            selected.append({"offset": i, "pair_id": row["pair_id"], "input_hash": input_hash(row),
                             "arm": arm, "similarity": similarities[i], "cluster_id": cluster,
                             "company_id": cid, "role": row["role"]})
            chosen.add(i); company_story.add(story)
            clusters[cluster] += 1; companies[cid] += 1; roles[row["role"]] += 1
            added += 1
            if added >= count or len(selected) >= limit:
                break

    # Random controls cannot disappear when the high-similarity arm fills a cap.
    for arm in ("random_control", "hard_negative_candidate", "boundary", "high_similarity"):
        take(orders[arm], quotas[arm], arm)
    if len(selected) < limit:
        take(ranked, limit - len(selected), "diversity_fill")
    return selected, {"eligible": len(eligible), "excluded": len(rows) - len(eligible),
        "selected": len(selected), "unique_clusters": len(clusters), "companies": len(companies),
        "role_counts": dict(roles), "arm_counts": dict(Counter(x["arm"] for x in selected)),
        "selection_is_not_a_label": True, "representative_production_sample": False}


def prepare_queue(input_path, plan_path, output_dir, encoder, *, limit=24, excluded=(), seed=42,
                  wire_model=None, prompt_variant="exact-evidence"):
    pilot = validate_labeling_input(plan_path, input_path)
    rows = read_dataset(input_path)
    similarities, details, scoring = score_candidates(rows, encoder)
    oversized, sizes = set(), {}
    if wire_model is not None:
        from app.intelligence.provider import input_bytes
        from .teacher import prepare_judgement
        for row in rows:
            plan = prepare_judgement(row, prompt_variant=prompt_variant)
            size = input_bytes(wire_model, plan["system"], plan["data"], plan["response_schema"])
            sizes[row["pair_id"]] = size
            if size > wire_model.input_limit:
                oversized.add(row["pair_id"])
    queue, summary = select_candidates(rows, similarities, limit=limit, excluded=set(excluded) | oversized, seed=seed)
    summary.update(known_attempt_ids_supplied=len(set(excluded)),
                   known_attempts_excluded=sum(r["pair_id"] in set(excluded) for r in rows), oversized_pairs=len(oversized),
                   wire_bound_checked=wire_model is not None)
    for item in queue:
        item.update(details[item["offset"]], request_bytes=sizes.get(item["pair_id"]))
    manifest = {"schema_version": QUEUE_SCHEMA, "source_sha256": file_hash(input_path),
                "pilot_plan_id": pilot["plan_id"], "seed": seed, "scoring": scoring,
                "prompt_variant": prompt_variant,
                "wire_model_id": wire_model.id if wire_model else None,
                "excluded_pair_ids": sorted(set(excluded) | oversized), "requests": queue,
                "summary": summary, "paid_calls": 0,
                "limitations": ["Candidate arms are hypotheses, never relevance grades",
                    "Only locally supplied attempt records are excluded; reconcile the shared ledger before live dispatch",
                    "All article text is embedded; cosine is not a probability or business materiality",
                    "One role per company/story in this diagnostic; roles are not assumed to have equal labels"]}
    manifest["queue_id"] = "queue_" + digest(manifest)
    with artifact_dir(output_dir) as stage:
        stage.chmod(0o700)
        write_private_json(stage / "queue.json", manifest)
        write_private_json(stage / "ranking.json", {"pairs": [
            {"offset": i, "pair_id": r["pair_id"], "similarity": similarities[i]}
            for i, r in enumerate(rows)]})
    return {**summary, "paid_calls": 0, "queue": str(Path(output_dir) / "queue.json")}


def queued_rows(input_path, queue_path, *, offset=0, limit=100):
    queue = read_json(queue_path)
    identity = {k: v for k, v in queue.items() if k != "queue_id"}
    if (queue.get("schema_version") != QUEUE_SCHEMA or queue.get("queue_id") != "queue_" + digest(identity)
            or queue.get("source_sha256") != file_hash(input_path)):
        raise KnowledgeError("Candidate queue or original input checksum changed")
    rows = read_dataset(input_path)
    requests = queue.get("requests")
    if not isinstance(requests, list) or not requests or len(requests) > 100:
        raise KnowledgeError("Invalid bounded candidate queue")
    seen, selected = set(), []
    for item in requests:
        position = item.get("offset")
        if type(position) is not int or not 0 <= position < len(rows):
            raise KnowledgeError("Candidate offset is outside the frozen input")
        row = rows[position]
        if (row["pair_id"] != item.get("pair_id") or input_hash(row) != item.get("input_hash")
                or row["pair_id"] in seen or row["pair_id"] in queue.get("excluded_pair_ids", [])):
            raise KnowledgeError("Candidate request does not match its original frozen row")
        seen.add(row["pair_id"]); selected.append(row)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise KnowledgeError("Invalid queue slice")
    return selected[offset:offset + limit]


def _checked_teacher(row):
    row = validate_pair(row)
    label = row.get("label", {})
    if (label.get("origin") != "teacher" or label.get("target") != "business_relevance"
            or label.get("input_hash") != input_hash(row) or not label.get("labeler")):
        raise KnowledgeError("Automatic QA requires source-bound teacher proposals")
    judgement = validate_judgement(label.get("judgement"), row)
    if judgement["grade"] != label.get("grade") or judgement["reason"] != label.get("reason"):
        raise KnowledgeError("Automatic QA label contradicts its judgement")
    return row


def machine_consensus(primary, verifier, *, policy="exact"):
    """Blind different-model agreement with abstention; no labels are averaged."""
    if policy not in {"exact", "binary", "business-binary"}:
        raise KnowledgeError("Machine agreement policy must be exact, binary or business-binary")
    primary, verifier = [_checked_teacher(r) for r in primary], [_checked_teacher(r) for r in verifier]
    first = {r["pair_id"]: r for r in primary}
    second = {r["pair_id"]: r for r in verifier}
    if len(first) != len(primary) or len(second) != len(verifier):
        raise KnowledgeError("Duplicate machine review pairs")
    accepted, abstained = [], []
    for pid in sorted(set(second) - set(first)):
        abstained.append({"pair_id": pid, "input_hash": input_hash(second[pid]), "reasons": ["missing_primary_proposal"]})
    comparisons = Counter()
    for pid, a in first.items():
        b = second.get(pid)
        errors = []
        if b is None:
            errors.append("missing_blind_review")
        else:
            if input_hash(a) != input_hash(b):
                raise KnowledgeError("Blind review changed the original company/article snapshot")
            la, lb = a["label"], b["label"]
            criteria = lambda label: label.get("criteria_version") or (
                "case-business-significance-v1" if label.get("label_version") == "relevance-judge-v9-case-calibrated"
                else "legacy-strict-business-fit")
            if criteria(la) != criteria(lb):
                errors.append("grading_criteria_disagreement")
            ja, jb = la["judgement"], lb["judgement"]
            ga, gb = ja["grade"], jb["grade"]
            comparisons[f"{ga}:{gb}"] += 1
            if la["labeler"] == lb["labeler"]:
                errors.append("same_model_not_cross_model_review")
            if policy == "exact":
                if ga != gb:
                    errors.append("grade_disagreement")
                if any(ja[k] != jb[k] for k in ("impact_type", "geography", "scale", "applicability", "event_status",
                                                "timeliness", "valid_from", "valid_until")):
                    errors.append("scope_or_temporal_disagreement")
            else:
                if (ga >= 4) != (gb >= 4) or (5 in {ga, gb} and ga != gb):
                    errors.append("binary_target_or_critical_grade_disagreement")
                if ga >= 4:
                    if policy == "binary":
                        if any(ja[k] != jb[k] for k in ("impact_type", "geography", "scale", "applicability", "event_status")):
                            errors.append("positive_mechanism_or_scope_disagreement")
                    else:
                        # This target is business significance for training,
                        # independently of permission to publish the news today.
                        if any(ja[k] != jb[k] for k in ("geography", "scale", "applicability")):
                            errors.append("positive_scope_disagreement")
                        impacts = {ja["impact_type"], jb["impact_type"]}
                        if "none" in impacts or (len(impacts) > 1 and "mixed" not in impacts):
                            errors.append("positive_mechanism_disagreement")
                        if 5 in {ga, gb} and any(ja[k] != jb[k] for k in (
                                "impact_type", "event_status", "timeliness", "valid_from", "valid_until")):
                            errors.append("critical_scope_or_temporal_disagreement")
            if ga >= 4 and not set(ja["company_fact_ids"]) & set(jb["company_fact_ids"]):
                errors.append("no_shared_company_basis")
        if errors:
            abstained.append({"pair_id": pid, "input_hash": input_hash(a), "reasons": errors})
            continue
        row = copy.deepcopy(a)
        row["label"]["machine_review"] = {"policy": "blind-cross-model-" + policy + "-v1",
            "origin": "teacher", "independently_reviewed": False, "reviewer": b["label"],
            "agreement_target": "grade>=4" if policy != "exact" else "exact_grade_and_scope",
            "grade_interval": sorted([a["label"]["grade"], b["label"]["grade"]]),
            "grade_is_primary_proposal": True,
            "temporal_consensus": all(a["label"]["judgement"][k] == b["label"]["judgement"][k]
                                      for k in ("timeliness", "valid_from", "valid_until")),
            "human_benchmark_eligible": False}
        if policy == "business-binary":
            row["label"]["machine_review"].update(
                mechanism_consensus=a["label"]["judgement"]["impact_type"] == b["label"]["judgement"]["impact_type"],
                event_status_consensus=a["label"]["judgement"]["event_status"] == b["label"]["judgement"]["event_status"],
                publishing_permission=False)
        row["label"]["independently_reviewed"] = False
        accepted.append(row)
    return accepted, abstained, {"primary": len(first), "reviewed": len(second), "candidate_pairs": len(set(first) | set(second)),
        "accepted": len(accepted),
        "abstained": len(abstained), "grade_confusion": dict(comparisons),
        "accepted_grades": dict(Counter(str(r["label"]["grade"]) for r in accepted)),
        "human_benchmark_eligible": False, "accuracy": None,
        "agreement_policy": policy,
        "graded_ranking_labels": "primary_teacher_proposals_not_independently_validated",
        "labels_are": "machine_agreement_silver_not_independent_gold"}


def export_consensus(primary_path, verifier_path, output_dir, *, policy="exact"):
    accepted, abstained, report = machine_consensus(read_dataset(primary_path), read_dataset(verifier_path), policy=policy)
    with artifact_dir(output_dir) as stage:
        stage.chmod(0o700)
        write_dataset(stage / "silver.jsonl", accepted)
        write_private_json(stage / "abstained.json", {"pairs": abstained})
        write_private_json(stage / "report.json", {**report, "primary_sha256": file_hash(primary_path),
                                                   "verifier_sha256": file_hash(verifier_path)})
    return {**report, "output": str(output_dir), "paid_calls": 0}
