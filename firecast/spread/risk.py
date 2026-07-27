from __future__ import annotations

import numpy as np


def forecast_risk(forecast: np.ndarray, hazardous_level: float) -> float:
    """Convert predicted spread intensity into bounded risk using the hazard level."""
    if hazardous_level <= 0:
        raise ValueError("hazardous_level must be positive")
    values = np.asarray(forecast, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("forecast must contain finite values")
    return float(np.clip(np.max(values) / hazardous_level, 0.0, 1.0))
