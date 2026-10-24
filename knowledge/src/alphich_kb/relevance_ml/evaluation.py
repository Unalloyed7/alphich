"""Company-conditioned relevance metrics and leakage-resistant evaluation splits.

Teacher labels are useful training data, but never an independent benchmark.
All probabilities refer to ``grade >= 4``; they are not business grades.
"""

import hashlib
import math
import random
from collections import Counter

from ..common import KnowledgeError, digest


def label_grade(row):
    label = row.get("label")
    if not isinstance(label, dict):
        raise KnowledgeError("A labeled company/news pair is required")
    grade = label.get("grade")
    if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 5:
        raise KnowledgeError("Relevance label grade must be an integer from 0 to 5")
    if label.get("origin") not in {"human", "teacher"}:
        raise KnowledgeError("Relevance label origin must be human or teacher")
    return grade


def _independent_human(row):
    label = row.get("label", {})
    reviewers = label.get("reviewers", [])
    # A declared provenance check; this cannot establish real human independence.
    return (label.get("origin") == "human"
            and label.get("review_status") in {"agreed", "adjudicated"}
            and label.get("independently_reviewed") is not False
            and isinstance(reviewers, list)
            and len({r.strip() for r in reviewers if isinstance(r, str) and r.strip()}) >= 2)


def _probabilities(values, count):
    values = list(values)
    if len(values) != count:
        raise KnowledgeError("Prediction count does not match labeled pair count")
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("p_relevant")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # numpy scalar probabilities can expose item(), without importing numpy here.
            value = value.item() if hasattr(value, "item") else value
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise KnowledgeError("p_relevant must be a finite probability from 0 to 1")
        result.append(float(value))
    return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _wilson_lower(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    return ((p + z * z / (2 * total)
             - z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)))
            / (1 + z * z / total))


def _ranking_metrics(labels, probabilities, budget_k, *, grades=None):
    ranked = sorted(range(len(labels)), key=lambda index: (-probabilities[index], index))
    selected = ranked[:budget_k]
    hits = sum(labels[index] for index in selected)
    positives = sum(labels)
    # Average precision groups score ties. Input order cannot improve its value.
    ap, cumulative_hits, cumulative_count = 0.0, 0, 0
    cursor = 0
    while cursor < len(ranked):
        end = cursor + 1
        while end < len(ranked) and probabilities[ranked[end]] == probabilities[ranked[cursor]]:
            end += 1
        group_hits = sum(labels[index] for index in ranked[cursor:end])
        cumulative_hits += group_hits
        cumulative_count += end - cursor
        ap += group_hits * cumulative_hits / cumulative_count
        cursor = end
    top_five = ranked[:5]
    result = {
        "budget_k": budget_k,
        "selected_count": len(selected),
        "relevant_selected": hits,
        "precision_at_k": _ratio(hits, len(selected)),
        "recall_at_budget": _ratio(hits, positives),
        "average_precision": _ratio(ap, positives),
        "precision_at_5": _ratio(sum(labels[index] for index in top_five), len(top_five)),
        "selected_at_5": len(top_five),
        "ranking_tie_policy": "stable_input_order_at_budget_and_5; grouped_ties_for_average_precision",
    }
    if grades is not None:
        # Keep the full 0..5 utility scale rather than treating grade 4 and 5 as
        # interchangeable. The cutoff and gain function are fixed, not fit on test.
        def dcg(values):
            return sum((2 ** grade - 1) / math.log2(rank + 2)
                       for rank, grade in enumerate(values))
        ideal = dcg(sorted(grades, reverse=True)[:5])
        result["ndcg_at_5"] = _ratio(dcg([grades[index] for index in top_five]), ideal)
        result["ndcg_gain"] = "2^grade-1; log2(rank+1) discount; undefined when all gains are zero"
    return result


def _group_metrics(indices, labels, grades, probabilities, selected, budget_k):
    """Within one company/role, compare only candidates that actually compete."""
    positive_count = sum(labels[index] for index in indices)
    tp = sum(labels[index] == 1 and selected[index] for index in indices)
    fp = sum(labels[index] == 0 and selected[index] for index in indices)
    fn = positive_count - tp
    return {"count": len(indices), "positives": positive_count, "tp": tp, "fp": fp, "fn": fn,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, positive_count),
            "ranking": _ranking_metrics([labels[index] for index in indices],
                                        [probabilities[index] for index in indices], budget_k,
                                        grades=[grades[index] for index in indices])}


def evaluate_predictions(rows, probabilities, *, threshold=0.5, budget_k=10, calibration_bins=10):
    """Report observed selection quality, counts and explicitly qualified provenance.

    A caller must keep its final test set separate from threshold selection.
    No denominator is silently replaced by one, and no synthetic cost savings
    are claimed: actual paid calls/costs need the integration's billing ledger.
    """
    rows = list(rows)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise KnowledgeError("Classification threshold must be from 0 to 1")
    if isinstance(budget_k, bool) or not isinstance(budget_k, int) or budget_k < 1:
        raise KnowledgeError("budget_k must be a positive integer")
    if isinstance(calibration_bins, bool) or not isinstance(calibration_bins, int) or not 2 <= calibration_bins <= 100:
        raise KnowledgeError("Calibration bins must be 2..100")
    grades = [label_grade(row) for row in rows]
    labels = [int(grade >= 4) for grade in grades]
    probabilities = _probabilities(probabilities, len(rows))
    selected = [value >= threshold for value in probabilities]
    tp = sum(label == 1 and take for label, take in zip(labels, selected))
    fp = sum(label == 0 and take for label, take in zip(labels, selected))
    fn = sum(label == 1 and not take for label, take in zip(labels, selected))
    tn = len(rows) - tp - fp - fn
    eligible = sum(_independent_human(row) for row in rows)
    positives = sum(labels)
    bins = []
    ece = 0.0
    for index in range(calibration_bins):
        lower, upper = index / calibration_bins, (index + 1) / calibration_bins
        members = [i for i, value in enumerate(probabilities)
                   if lower <= value < upper or index == calibration_bins - 1 and value == 1]
        mean_probability = _ratio(sum(probabilities[i] for i in members), len(members))
        positive_rate = _ratio(sum(labels[i] for i in members), len(members))
        if members:
            ece += len(members) * abs(mean_probability - positive_rate)
        bins.append({"lower": lower, "upper": upper, "count": len(members),
                     "mean_probability": mean_probability, "positive_rate": positive_rate})
    warnings = []
    if not rows:
        warnings.append("empty_evaluation_set")
    if not positives:
        warnings.append("no_positive_pairs_recall_is_undefined")
    if positives == len(rows) and rows:
        warnings.append("no_negative_pairs_false_positive_performance_unknown")
    if positives < 30:
        warnings.append("few_positive_pairs_recall_estimate_is_uncertain")
    if eligible != len(rows):
        warnings.append("not_an_independent_human_benchmark")
    company_indices, role_indices, company_role_indices = {}, {}, {}
    for index, row in enumerate(rows):
        company_id = row.get("company", {}).get("company_id", "unknown")
        role = row.get("role", "unknown")
        company_indices.setdefault(company_id, []).append(index)
        role_indices.setdefault(role, []).append(index)
        company_role_indices.setdefault(company_id, {}).setdefault(role, []).append(index)
    def metrics(indices):
        return _group_metrics(indices, labels, grades, probabilities, selected, budget_k)
    by_company = {company: metrics(indices) for company, indices in company_indices.items()}
    by_role = {role: metrics(indices) for role, indices in role_indices.items()}
    by_company_role = {company: {role: metrics(indices) for role, indices in roles.items()}
                       for company, roles in company_role_indices.items()}
    group_rankings = [values["ranking"] for roles in by_company_role.values() for values in roles.values()]
    macro = {"group_count": len(group_rankings)}
    for key in ("precision_at_5", "ndcg_at_5"):
        known = [ranking[key] for ranking in group_rankings if ranking[key] is not None]
        macro[key] = _ratio(sum(known), len(known))
        macro[key + "_defined_groups"] = len(known)
    return {
        "schema_version": "alphich-relevance-evaluation-v1", "target": "grade>=4",
        "count": len(rows), "positive_count": positives, "negative_count": len(rows) - positives,
        "label_origins": dict(Counter(row["label"]["origin"] for row in rows)),
        "human_benchmark_eligible_count": eligible,
        "human_benchmark_eligible": bool(rows) and eligible == len(rows),
        "provenance_check_only": True, "threshold": threshold,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
        "recall_wilson_95_lower": _wilson_lower(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "critical_grade5_count": grades.count(5),
        "missed_grade5_pair_ids": [row.get("pair_id") for row, grade, take in zip(rows, grades, selected)
                                    if grade == 5 and not take],
        "ranking": _ranking_metrics(labels, probabilities, budget_k, grades=grades),
        "ranking_scope": "global diagnostic; use by_company_role for personalized feed ranking",
        "ranking_by_company_role_macro": macro,
        "calibration": {"brier_score": _ratio(sum((p - y) ** 2 for p, y in zip(probabilities, labels)), len(rows)),
                        "expected_calibration_error": _ratio(ece, len(rows)), "bins": bins,
                        "probabilities_are_calibrated": False},
        "by_company": by_company, "by_role": by_role, "by_company_role": by_company_role,
        "warnings": warnings,
    }


def _article_keys(row):
    article = row.get("article", {})
    keys = []
    for name in ("article_id", "cluster_id"):
        value = article.get(name)
        if isinstance(value, str) and value:
            keys.append((name, value))
    # Exact copies with different IDs/URLs are joined even if the source forgot cluster_id.
    text = article.get("text")
    if isinstance(text, str) and text.strip():
        normalized = " ".join(text.casefold().split())
        keys.append(("content", hashlib.sha256(normalized.encode()).hexdigest()))
    if not keys:
        raise KnowledgeError("Each split row needs an article_id, cluster_id or article text")
    return keys


def assert_disjoint_splits(train_rows, validation_rows=(), test_rows=(), *, company_holdout=False):
    """Reject repeated article IDs, supplied reprint clusters, exact copies or pair IDs."""
    seen = {}
    for split_name, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        for row in rows:
            keys = _article_keys(row)
            if row.get("pair_id"):
                keys.append(("pair_id", row["pair_id"]))
            if company_holdout:
                company_id = row.get("company", {}).get("company_id")
                if not company_id:
                    raise KnowledgeError("Company holdout needs company_id on every row")
                keys.append(("company", company_id))
            for key in keys:
                if key in seen and seen[key] != split_name:
                    raise KnowledgeError(f"Data leakage between {seen[key]} and {split_name}: {key[0]}")
                seen[key] = split_name


def split_rows(rows, *, mode="article", ratios=(0.6, 0.2, 0.2), seed=42, require_labels=True):
    """Group all company pairs for an article/reprint before splitting.

    Company holdout additionally joins articles shared by companies. If those
    dependencies connect the whole dataset, an honest disjoint split is impossible.
    This helper does not claim to detect paraphrased reprints without cluster IDs.
    ``require_labels=False`` freezes the same groups before API/human labeling;
    missing labels stay unknown and never become inferred negatives.
    """
    rows = list(rows)
    if mode not in {"article", "company"}:
        raise KnowledgeError("Split mode must be article or company")
    if type(require_labels) is not bool:
        raise KnowledgeError("require_labels must be an explicit boolean")
    if (len(ratios) != 3 or any(isinstance(r, bool) or not isinstance(r, (int, float))
                                or not math.isfinite(r) or r <= 0 for r in ratios)
            or not math.isclose(sum(ratios), 1.0, abs_tol=1e-8)):
        raise KnowledgeError("Three positive train/validation/test ratios must sum to one")
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        if require_labels or row.get("label") is not None:
            label_grade(row)
        keys = _article_keys(row)
        if row.get("pair_id"):
            keys.append(("pair_id", row["pair_id"]))
        if mode == "company":
            company_id = row.get("company", {}).get("company_id")
            if not company_id:
                raise KnowledgeError("Company holdout needs company_id on every row")
            keys.append(("company", company_id))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    components = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)
    if len(components) < 3:
        raise KnowledgeError("At least three independent article/company components are needed for three splits")
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    assigned = [[], [], []]
    targets = [ratio * len(rows) for ratio in ratios]
    for cursor, group in enumerate(groups):
        empty = [i for i, members in enumerate(assigned) if not members]
        if len(groups) - cursor == len(empty):
            chosen = max(empty, key=lambda i: targets[i])
        else:
            chosen = max(range(3), key=lambda i: (targets[i] - len(assigned[i])) / targets[i])
        assigned[chosen].extend(group)
    result = {name: [rows[i] for i in sorted(indices)]
              for name, indices in zip(("train", "validation", "test"), assigned)}
    assert_disjoint_splits(result["train"], result["validation"], result["test"], company_holdout=mode == "company")
    warnings = ["Paraphrased reprints require externally supplied cluster_id; only exact copies are detected automatically"]
    label_counts = {}
    for name in ("train", "validation", "test"):
        labeled = [row for row in result[name] if row.get("label") is not None]
        positives = sum(label_grade(row) >= 4 for row in labeled)
        unknown = len(result[name]) - len(labeled)
        label_counts[name] = {"labeled": len(labeled), "unlabeled": unknown,
                              "positive_4_5": positives if labeled else None,
                              "negative_0_3": len(labeled) - positives if labeled else None}
        if unknown:
            warnings.append(f"{name}_unlabeled_class_distribution_unknown")
        elif not positives or positives == len(labeled):
            warnings.append(f"{name}_contains_only_one_class")
    result.update({"mode": mode, "seed": seed, "component_count": len(components), "warnings": warnings,
                   "labels_required": require_labels, "label_counts": label_counts,
                   "split_fingerprint": digest({name: [row["pair_id"] for row in result[name]]
                                                for name in ("train", "validation", "test")})})
    return result


def calibrate_threshold(validation_rows, probabilities, *, training_rows, min_recall=0.95):
    """Choose the highest score cutoff meeting observed *validation* recall.

    This selects a decision threshold, not a probability calibration function and
    not a statistical promise about future recall. The test set must remain unused.
    """
    validation_rows, training_rows = list(validation_rows), list(training_rows)
    if isinstance(min_recall, bool) or not isinstance(min_recall, (int, float)) or not 0 < min_recall <= 1:
        raise KnowledgeError("min_recall must be greater than zero and at most one")
    if not training_rows or not validation_rows:
        raise KnowledgeError("Threshold selection requires nonempty training and validation sets")
    assert_disjoint_splits(training_rows, validation_rows)
    if not all(_independent_human(row) for row in validation_rows):
        raise KnowledgeError("Threshold selection requires agreed/adjudicated human validation labels with two reviewers")
    probabilities = _probabilities(probabilities, len(validation_rows))
    positive_scores = sorted((p for row, p in zip(validation_rows, probabilities) if label_grade(row) >= 4), reverse=True)
    if not positive_scores or len(positive_scores) == len(validation_rows):
        raise KnowledgeError("Threshold selection needs positive and negative validation pairs")
    required = math.ceil(min_recall * len(positive_scores))
    threshold = positive_scores[required - 1]
    report = evaluate_predictions(validation_rows, probabilities, threshold=threshold)
    return {"threshold": threshold, "selected_on": "validation", "min_recall_requested": min_recall,
            "observed_recall": report["recall"], "recall_wilson_95_lower": report["recall_wilson_95_lower"],
            "positive_count": len(positive_scores), "validation_count": len(validation_rows),
            "validation_pair_fingerprint": digest([row["pair_id"] for row in validation_rows]),
            "production_calibrated": False, "probabilities_are_calibrated": False,
            "warnings": report["warnings"] + ["Observed validation recall does not guarantee future recall"]}
