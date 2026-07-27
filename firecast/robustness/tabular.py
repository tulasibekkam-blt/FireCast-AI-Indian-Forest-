from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from firecast.evaluation import binary_metrics


def evaluate_sensor_noise(pipeline: Any, features: pd.DataFrame, labels: Any,
                          numeric_columns: list[str], noise_fraction: float = 0.05,
                          repetitions: int = 5, seed: int = 42) -> pd.DataFrame:
    """Measure risk-model degradation under bounded Gaussian sensor noise."""
    if not 0 <= noise_fraction or repetitions < 1:
        raise ValueError("noise_fraction must be non-negative and repetitions must be positive")
    missing = set(numeric_columns) - set(features.columns)
    if missing:
        raise ValueError(f"Unknown numeric columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    rows = []
    for repetition in range(repetitions):
        perturbed = features.copy()
        for column in numeric_columns:
            scale = float(perturbed[column].std())
            perturbed[column] += rng.normal(0, scale * noise_fraction, len(perturbed))
        probabilities = pipeline.predict_proba(perturbed)[:, 1]
        metrics = binary_metrics(labels, probabilities)
        rows.append({"repetition": repetition, "noise_fraction": noise_fraction, **metrics})
    return pd.DataFrame(rows)


def evaluate_sensor_dropout(pipeline: Any, features: pd.DataFrame, labels: Any,
                            columns: list[str], dropout_fraction: float = 0.2,
                            repetitions: int = 5, seed: int = 42) -> pd.DataFrame:
    """Measure degradation when telemetry values are missing at random."""
    if not 0 <= dropout_fraction <= 1 or repetitions < 1:
        raise ValueError("dropout_fraction must be in [0, 1] and repetitions must be positive")
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"Unknown columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    rows = []
    for repetition in range(repetitions):
        perturbed = features.copy()
        mask = rng.random((len(perturbed), len(columns))) < dropout_fraction
        perturbed.loc[:, columns] = perturbed[columns].mask(mask)
        probabilities = pipeline.predict_proba(perturbed)[:, 1]
        rows.append({"repetition": repetition, "dropout_fraction": dropout_fraction,
                     **binary_metrics(labels, probabilities)})
    return pd.DataFrame(rows)
