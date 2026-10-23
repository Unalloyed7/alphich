"""Small CPU baseline: train-only standardization and L2 logistic regression.

Artifacts are finite, content-addressed JSON data, never pickle or executable
model files. This module has no API access and cannot trigger paid processing.
"""

import json
import math
from collections import Counter

import numpy as np

from ..common import KnowledgeError, digest
from .evaluation import (assert_disjoint_splits, calibrate_threshold,
                         evaluate_predictions, label_grade)


SCHEMA_VERSION = "alphich-relevance-logreg-v1"


def _finite_json(value):
    try:
        json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise KnowledgeError("Model artifact and feature manifest must contain finite JSON data") from error


def _manifest(manifest):
    if not isinstance(manifest, dict):
        raise KnowledgeError("A feature manifest is required")
    _finite_json(manifest)
    dimension = manifest.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise KnowledgeError("Feature manifest dimension must be a positive integer")
    if not isinstance(manifest.get("feature_version"), str) or not manifest["feature_version"]:
        raise KnowledgeError("Feature manifest must declare feature_version")
    names = manifest.get("feature_names")
    if (not isinstance(names, list) or len(names) != dimension
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise KnowledgeError("Feature manifest needs distinct ordered feature_names matching dimension")
    if not isinstance(manifest.get("encoder_space_id"), str) or not manifest["encoder_space_id"]:
        raise KnowledgeError("Feature manifest must bind encoder_space_id")
    return dimension


def _matrix(vectors, dimension, *, count=None):
    try:
        source = np.asarray(vectors)
        if np.iscomplexobj(source) or source.dtype.kind in {"U", "S", "O", "b"}:
            raise KnowledgeError("Feature vectors must contain real numeric values")
        result = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise KnowledgeError("Feature vectors must be a finite numeric 2D matrix") from error
    if result.ndim != 2 or result.shape[1] != dimension or (count is not None and result.shape[0] != count):
        raise KnowledgeError("Feature matrix shape does not match row count and feature manifest")
    if not np.isfinite(result).all():
        raise KnowledgeError("Feature matrix contains non-finite values")
    return result


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1 + negative_exp)
    return result


def _scaled(values, mean, scale):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = (values - mean) / scale
    if not np.isfinite(result).all():
        raise KnowledgeError("Feature scaling overflowed; check input feature ranges")
    return result


def _fit(values, labels, *, l2, iterations, tolerance):
    with np.errstate(over="ignore", invalid="ignore"):
        mean, scale = values.mean(axis=0), values.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise KnowledgeError("Training feature statistics overflowed")
    scale = np.where(scale > 1e-12, scale, 1.0)
    values = _scaled(values, mean, scale)
    weights = np.zeros(values.shape[1], dtype=np.float64)
    intercept = math.log(float(labels.mean()) / (1 - float(labels.mean())))
    regularization = l2 / len(labels)

    def objective(new_weights, new_intercept):
        with np.errstate(over="ignore", invalid="ignore"):
            scores = values @ new_weights + new_intercept
            loss = np.mean(np.logaddexp(0, scores) - labels * scores)
            loss += regularization * np.dot(new_weights, new_weights) / 2
        return float(loss)

    loss, step, converged = objective(weights, intercept), 1.0, False
    completed = 0
    for completed in range(1, iterations + 1):
        residual = _sigmoid(values @ weights + intercept) - labels
        gradient = values.T @ residual / len(labels) + regularization * weights
        intercept_gradient = float(residual.mean())
        gradient_square = float(np.dot(gradient, gradient) + intercept_gradient ** 2)
        if not math.isfinite(gradient_square):
            raise KnowledgeError("Logistic-regression optimization overflowed")
        if gradient_square <= tolerance ** 2:
            converged = True
            break
        trial_step = min(step * 1.5, 10.0)
        accepted = False
        for _ in range(60):
            trial_weights = weights - trial_step * gradient
            trial_intercept = intercept - trial_step * intercept_gradient
            trial_loss = objective(trial_weights, trial_intercept)
            if math.isfinite(trial_loss) and trial_loss <= loss - 1e-4 * trial_step * gradient_square:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            if gradient_square <= tolerance:
                converged = True
                break
            raise KnowledgeError("Logistic-regression line search could not find a finite descent step")
        improvement = loss - trial_loss
        weights, intercept, loss, step = trial_weights, trial_intercept, trial_loss, trial_step
        if improvement <= tolerance * (1 + abs(loss)):
            converged = True
            break
    return mean, scale, weights, float(intercept), {
        "iterations": completed, "max_iterations": iterations,
        "converged": converged, "objective": loss, "l2": l2,
        "regularization_convention": "mean_logloss + l2/(2*n) * squared_weight_norm",
    }


def _probability(values, mean, scale, weights, intercept):
    with np.errstate(over="ignore", invalid="ignore"):
        scores = _scaled(values, mean, scale) @ weights + intercept
    if not np.isfinite(scores).all():
        raise KnowledgeError("Model scores overflowed; check input feature ranges")
    return _sigmoid(scores)


def train_model(rows, vectors, *, feature_manifest, threshold=0.5, uncertainty_margin=0.1,
                allow_teacher=False, validation_rows=None, validation_vectors=None,
                min_recall=None, l2=1.0, iterations=1000, tolerance=1e-8):
    """Train a deterministic baseline and return a portable JSON artifact.

    Validation features are transformed using *training* means/scales. A teacher
    training set needs explicit opt-in. Optional threshold selection requires
    disjoint, independently reviewed human validation labels; no test set is used.
    """
    rows = list(rows)
    if validation_rows is not None:
        validation_rows = list(validation_rows)
    if any(row.get("label", {}).get("label_version") == "relevance-judge-v1" for row in rows):
        raise KnowledgeError("Legacy freshness-capped teacher labels need relabeling for business relevance")
    if any(row.get("label", {}).get("label_version") == "relevance-judge-v1" for row in validation_rows or []):
        raise KnowledgeError("Legacy freshness-capped validation labels need relabeling")
    dimension = _manifest(feature_manifest)
    values = _matrix(vectors, dimension, count=len(rows))
    if not rows:
        raise KnowledgeError("Cannot train on an empty dataset")
    labels = np.asarray([int(label_grade(row) >= 4) for row in rows], dtype=np.float64)
    if len(set(labels.tolist())) != 2:
        raise KnowledgeError("Training requires both relevant (4..5) and non-relevant (0..3) pairs")
    origins = Counter(row["label"]["origin"] for row in rows)
    if origins.get("teacher") and not allow_teacher:
        raise KnowledgeError("Teacher-labeled training requires explicit allow_teacher=True")
    if not isinstance(allow_teacher, bool):
        raise KnowledgeError("allow_teacher must be an explicit boolean")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise KnowledgeError("Decision threshold must be a finite value from 0 to 1")
    if (isinstance(uncertainty_margin, bool) or not isinstance(uncertainty_margin, (int, float))
            or not math.isfinite(uncertainty_margin) or not 0 <= uncertainty_margin <= 0.5):
        raise KnowledgeError("uncertainty_margin must be a finite value from 0 to 0.5")
    if isinstance(l2, bool) or not isinstance(l2, (int, float)) or not math.isfinite(l2) or l2 <= 0:
        raise KnowledgeError("L2 regularization must be a positive finite value")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 10000:
        raise KnowledgeError("Training iterations must be 1..10000")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or not 0 < tolerance <= 0.01):
        raise KnowledgeError("Optimization tolerance must be greater than zero and at most 0.01")
    if (validation_rows is None) != (validation_vectors is None):
        raise KnowledgeError("Pass both validation_rows and validation_vectors")
    if min_recall is not None and validation_rows is None:
        raise KnowledgeError("Recall threshold selection requires a separate validation set")
    validation_values = None
    if validation_rows is not None:
        validation_rows = list(validation_rows)
        if not validation_rows:
            raise KnowledgeError("Validation set must not be empty")
        assert_disjoint_splits(rows, validation_rows)
        validation_values = _matrix(validation_vectors, dimension, count=len(validation_rows))
        for row in validation_rows:
            label_grade(row)
    mean, scale, weights, intercept, optimization = _fit(values, labels, l2=l2,
                                                       iterations=iterations, tolerance=tolerance)
    warnings = ["Prototype baseline; shadow mode only until independent held-out evaluation",
                "Scores are uncalibrated probabilities, not business grades",
                "Applicability, geography and materiality checks remain required downstream"]
    if origins.get("teacher"):
        warnings.append("Training includes provisional teacher labels and may reproduce teacher errors")
    if int(labels.sum()) < 30:
        warnings.append("Few positive training pairs; generalization is not established")
    if not optimization["converged"]:
        warnings.append("Optimization reached its iteration limit")
    threshold_selection = None
    validation_report = None
    if validation_values is not None:
        validation_probabilities = _probability(validation_values, mean, scale, weights, intercept)
        if min_recall is not None:
            threshold_selection = calibrate_threshold(validation_rows, validation_probabilities,
                                                      training_rows=rows, min_recall=min_recall)
            threshold = threshold_selection["threshold"]
        validation_report = evaluate_predictions(validation_rows, validation_probabilities, threshold=threshold)
        validation_report["estimate_scope"] = ("threshold_selection_validation_not_final_test" if min_recall is not None
                                               else "held_out_validation_not_final_test")
    train_report = evaluate_predictions(rows, _probability(values, mean, scale, weights, intercept), threshold=threshold)
    train_report["estimate_scope"] = "training_fit_not_generalization"
    artifact = {
        "schema_version": SCHEMA_VERSION, "algorithm": "numpy-logistic-regression-l2",
        "target": "grade>=4", "grade_meaning": "business_relevance_not_current_feed_eligibility",
        "feature_version": feature_manifest["feature_version"],
        "feature_manifest": json.loads(json.dumps(feature_manifest)), "feature_manifest_hash": digest(feature_manifest),
        "dimension": dimension, "mean": mean.tolist(), "scale": scale.tolist(),
        "weights": weights.tolist(), "intercept": intercept,
        "decision": {"threshold": float(threshold), "uncertainty_margin": float(uncertainty_margin),
                     "mode": "shadow", "probabilities_are_calibrated": False},
        "training": {"count": len(rows), "positives": int(labels.sum()),
                     "negatives": len(rows) - int(labels.sum()), "label_origins": dict(origins),
                     "teacher_opt_in": allow_teacher, "pair_fingerprint": digest([row["pair_id"] for row in rows]),
                     "optimization": optimization, "report": train_report},
        "threshold_selection": threshold_selection, "validation_report": validation_report,
        "limitations": warnings,
    }
    _finite_json(artifact)
    artifact["model_version"] = "relevance_logreg_" + digest(artifact)
    return artifact


def validate_artifact(artifact, *, feature_manifest):
    """Validate schema, numeric parameters, content hash and exact feature space."""
    if not isinstance(artifact, dict):
        raise KnowledgeError("Model artifact must be a JSON object")
    _finite_json(artifact)
    dimension = _manifest(feature_manifest)
    if (artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("algorithm") != "numpy-logistic-regression-l2"
            or artifact.get("target") != "grade>=4"):
        raise KnowledgeError("Unsupported relevance model artifact")
    if (artifact.get("feature_manifest_hash") != digest(feature_manifest)
            or artifact.get("feature_manifest") != feature_manifest
            or type(artifact.get("dimension")) is not int
            or artifact.get("dimension") != dimension
            or artifact.get("feature_version") != feature_manifest["feature_version"]):
        raise KnowledgeError("Model and feature manifest do not match")
    payload = {key: value for key, value in artifact.items() if key != "model_version"}
    if artifact.get("model_version") != "relevance_logreg_" + digest(payload):
        raise KnowledgeError("Model content hash mismatch")
    for name in ("mean", "scale", "weights"):
        if (not isinstance(artifact.get(name), list)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       for value in artifact[name])):
            raise KnowledgeError("Model parameters must be numeric arrays")
    try:
        mean, scale, weights = (np.asarray(artifact[name], dtype=np.float64) for name in ("mean", "scale", "weights"))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise KnowledgeError("Model has invalid numeric parameters") from error
    if any(value.shape != (dimension,) or not np.isfinite(value).all() for value in (mean, scale, weights)):
        raise KnowledgeError("Model parameter shape or finiteness is invalid")
    if np.any(scale <= 0):
        raise KnowledgeError("Model scaling values must be positive")
    intercept = artifact.get("intercept")
    if isinstance(intercept, bool) or not isinstance(intercept, (int, float)) or not math.isfinite(intercept):
        raise KnowledgeError("Model intercept must be finite")
    decision = artifact.get("decision", {})
    if not isinstance(decision, dict):
        raise KnowledgeError("Model decision policy must be an object")
    threshold, margin = decision.get("threshold"), decision.get("uncertainty_margin")
    if (decision.get("mode") != "shadow"
            or decision.get("probabilities_are_calibrated") is not False
            or isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1
            or isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= margin <= 0.5):
        raise KnowledgeError("Model decision policy is invalid")
    limitations = artifact.get("limitations")
    if not isinstance(limitations, list) or any(not isinstance(value, str) for value in limitations):
        raise KnowledgeError("Model limitations must be a list of strings")
    return {"dimension": dimension, "mean": mean, "scale": scale, "weights": weights,
            "intercept": float(intercept), "threshold": threshold, "uncertainty_margin": margin}


def predict(artifact, vectors, *, feature_manifest):
    """Return scores for vectors only; labels and teacher reasons cannot affect inference.

    Both candidate and uncertain decisions remain eligible for downstream review.
    A low_priority decision is a shadow observation, not permission to discard.
    """
    params = validate_artifact(artifact, feature_manifest=feature_manifest)
    values = _matrix(vectors, params["dimension"])
    probabilities = _probability(values, params["mean"], params["scale"], params["weights"], params["intercept"])
    result = []
    for probability in probabilities:
        p = float(probability)
        if p >= params["threshold"] + params["uncertainty_margin"]:
            decision = "candidate"
        elif p < params["threshold"] - params["uncertainty_margin"]:
            decision = "low_priority"
        else:
            decision = "uncertain"
        result.append({"p_relevant": p, "decision": decision, "model_version": artifact["model_version"],
                       "feature_version": artifact["feature_version"], "mode": "shadow",
                       "limitations": list(artifact["limitations"])})
    return result
