from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def weighted_probability_ensemble(models: Mapping[str, Any], features: Any,
                                  weights: Mapping[str, float] | None = None) -> np.ndarray:
    """Combine fitted probabilistic estimators using normalized non-negative weights."""
    if not models:
        raise ValueError("At least one fitted model is required")
    weights = weights or {name: 1.0 for name in models}
    scores = []
    total = 0.0
    for name, model in models.items():
        weight = float(weights.get(name, 0.0))
        if weight < 0:
            raise ValueError("Ensemble weights must be non-negative")
        if weight:
            scores.append(weight * model.predict_proba(features)[:, 1])
            total += weight
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value")
    return np.sum(scores, axis=0) / total
