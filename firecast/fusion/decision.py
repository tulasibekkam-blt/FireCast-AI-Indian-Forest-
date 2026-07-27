from __future__ import annotations

from typing import Mapping


def aggregate_risk(probabilities: Mapping[str, float], availability: Mapping[str, bool],
                   weights: Mapping[str, float] | None = None) -> float:
    """Combine available calibrated modality probabilities into one bounded risk score."""
    weights = weights or {name: 1.0 for name in probabilities}
    usable = [(name, float(probabilities[name]), float(weights.get(name, 0)))
              for name in probabilities if availability.get(name, False)]
    if not usable:
        raise ValueError("At least one modality must be available")
    if any(not 0 <= probability <= 1 for _, probability, _ in usable):
        raise ValueError("All modality probabilities must be between 0 and 1")
    total_weight = sum(weight for _, _, weight in usable)
    if total_weight <= 0:
        raise ValueError("Available modality weights must sum to a positive value")
    return sum(probability * weight for _, probability, weight in usable) / total_weight
