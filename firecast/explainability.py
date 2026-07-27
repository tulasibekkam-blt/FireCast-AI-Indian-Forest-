from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_explanation(pipeline: Any, features: pd.DataFrame, labels: pd.Series,
                            repeats: int = 10, seed: int = 42) -> pd.DataFrame:
    """Estimate feature influence on raw input columns without leaking preprocessing."""
    result = permutation_importance(
        pipeline, features, labels, scoring="average_precision", n_repeats=repeats,
        random_state=seed, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": features.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)


def shap_explanation(pipeline: Any, features: pd.DataFrame, max_samples: int = 500) -> pd.DataFrame:
    """Return SHAP values for a fitted pipeline using its raw feature columns."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError("Install the tabular extra to generate SHAP explanations") from error
    sample = features.head(max_samples)
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    estimator = pipeline.named_steps["model"]
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    explainer = shap.Explainer(estimator, transformed)
    values = explainer(transformed).values
    if values.ndim == 3:
        values = values[:, :, 1]
    return pd.DataFrame(values, columns=[f"encoded_feature_{index}" for index in range(values.shape[1])])


def decision_explanation(probability: float, threshold: float, feature_importance: pd.DataFrame,
                         top_k: int = 5) -> dict[str, Any]:
    """Create a stable audit payload for an individual risk decision."""
    if not 0 <= probability <= 1 or not 0 <= threshold <= 1:
        raise ValueError("probability and threshold must be between 0 and 1")
    if not {"feature", "importance_mean"}.issubset(feature_importance.columns):
        raise ValueError("feature_importance requires feature and importance_mean columns")
    alert = probability >= threshold
    evidence = feature_importance.head(top_k).to_dict(orient="records")
    return {
        "risk_probability": float(probability), "alert_threshold": float(threshold),
        "decision": "alert" if alert else "monitor", "confidence_margin": float(abs(probability - threshold)),
        "top_evidence": evidence,
    }


def counterfactual_numeric(pipeline: Any, row: pd.DataFrame, numeric_ranges: dict[str, tuple[float, float]],
                           target_probability: float = 0.5, steps: int = 21) -> pd.DataFrame:
    """Search bounded one-feature changes that reduce predicted risk.

    This is intentionally conservative: it changes one numeric feature at a time and never
    proposes values outside ranges supplied from the real training population.
    """
    if len(row) != 1:
        raise ValueError("Counterfactual generation requires exactly one row")
    baseline = float(pipeline.predict_proba(row)[:, 1][0])
    suggestions = []
    for feature, (lower, upper) in numeric_ranges.items():
        if feature not in row.columns:
            continue
        values = np.linspace(lower, upper, steps)
        candidates = pd.concat([row] * len(values), ignore_index=True)
        candidates[feature] = values
        probabilities = pipeline.predict_proba(candidates)[:, 1]
        valid = np.where(probabilities <= target_probability)[0]
        if len(valid):
            index = valid[np.argmin(np.abs(values[valid] - float(row.iloc[0][feature])))]
            suggestions.append({
                "feature": feature, "original_value": row.iloc[0][feature],
                "suggested_value": values[index], "predicted_probability": probabilities[index],
                "baseline_probability": baseline,
            })
    return pd.DataFrame(suggestions).sort_values("predicted_probability", ignore_index=True)
