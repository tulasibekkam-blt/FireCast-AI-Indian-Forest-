from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(frame: pd.DataFrame, feature_columns: list[str], target_column: str,
                 lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Build chronological windows; callers must split by time before invoking this."""
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    missing = set(feature_columns + [target_column]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    target = frame[target_column].to_numpy(dtype=np.float32)
    sample_count = len(frame) - lookback - horizon + 1
    if sample_count < 1:
        raise ValueError("Insufficient rows for requested lookback and horizon")
    inputs = np.stack([features[index:index + lookback] for index in range(sample_count)])
    labels = np.stack([target[index + lookback:index + lookback + horizon] for index in range(sample_count)])
    return inputs, labels
