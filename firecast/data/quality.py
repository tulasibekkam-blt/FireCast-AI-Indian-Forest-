from __future__ import annotations

from typing import Any

import pandas as pd


def quality_report(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    """Produce a serializable data-quality report for experiment provenance."""
    numeric = frame.select_dtypes(include="number")
    outliers = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            outliers[column] = 0
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        outliers[column] = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
    return {
        "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items() if v},
        "class_counts": {str(k): int(v) for k, v in frame[target].value_counts().items()},
        "class_balance": {str(k): float(v) for k, v in frame[target].value_counts(normalize=True).items()},
        "numeric_outliers_iqr": outliers,
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
    }
