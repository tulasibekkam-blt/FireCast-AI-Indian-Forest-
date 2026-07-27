import numpy as np

from firecast.evaluation import binary_metrics, select_threshold


def test_metrics_and_threshold_are_deterministic():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    threshold = select_threshold(y_true, probabilities)
    metrics = binary_metrics(y_true, probabilities, threshold)
    assert 0.01 <= threshold <= 0.99
    assert metrics["roc_auc"] == 1.0
    assert metrics["specificity"] >= 0.5
