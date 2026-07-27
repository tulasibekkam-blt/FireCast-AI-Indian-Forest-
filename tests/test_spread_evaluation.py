import numpy as np

from firecast.spread.evaluation import forecast_metrics


def test_forecast_metrics_perfect_prediction():
    values = np.array([[1.0, 2.0, 4.0]])
    metrics = forecast_metrics(values, values)
    assert metrics["mae"] == 0.0
    assert metrics["directional_accuracy"] == 1.0
