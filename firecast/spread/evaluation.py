from __future__ import annotations

import numpy as np


def forecast_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Evaluate continuous spread forecasts in original physical units."""
    actual = np.asarray(target, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: target={actual.shape}, prediction={predicted.shape}")
    error = predicted - actual
    scale = float(np.mean(np.abs(actual)))
    direction_actual = np.sign(np.diff(actual, axis=-1))
    direction_predicted = np.sign(np.diff(predicted, axis=-1))
    direction_accuracy = float(np.mean(direction_actual == direction_predicted)) if actual.shape[-1] > 1 else float("nan")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_bias": float(np.mean(error)),
        "normalized_mae": float(np.mean(np.abs(error)) / max(scale, 1e-8)),
        "directional_accuracy": direction_accuracy,
    }
