from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def binary_metrics(y_true: Any, probabilities: Any, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "brier_score": float(np.mean((probabilities - y_true) ** 2)),
        "expected_calibration_error": float(expected_calibration_error(y_true, probabilities)),
        "true_negative": float(tn), "false_positive": float(fp),
        "false_negative": float(fn), "true_positive": float(tp),
    }


def expected_calibration_error(y_true: Any, probabilities: Any, bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities <= upper if upper == 1 else probabilities < upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(y_true[mask].mean()))
    return error


def select_threshold(y_true: Any, probabilities: Any, objective: str = "f1") -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [binary_metrics(y_true, probabilities, t)[objective] for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def select_cost_threshold(y_true: Any, probabilities: Any, false_negative_cost: float = 5.0,
                          false_positive_cost: float = 1.0) -> float:
    """Select a threshold minimizing explicitly supplied operational error costs."""
    if false_negative_cost < 0 or false_positive_cost < 0 or false_negative_cost + false_positive_cost == 0:
        raise ValueError("Error costs must be non-negative and not both zero")
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        false_negatives = ((y_true == 1) & (predictions == 0)).sum()
        false_positives = ((y_true == 0) & (predictions == 1)).sum()
        costs.append(false_negative_cost * false_negatives + false_positive_cost * false_positives)
    return float(thresholds[int(np.argmin(costs))])


def cross_validated_metrics(estimator: Any, features: Any, labels: Any, folds: int = 5,
                            seed: int = 42) -> dict[str, float]:
    """Evaluate a probabilistic estimator without fitting on the evaluation fold."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        estimator, features, labels, cv=splitter, method="predict_proba", n_jobs=None,
    )[:, 1]
    threshold = select_threshold(labels, probabilities)
    return binary_metrics(labels, probabilities, threshold)
