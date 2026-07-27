from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def chronological_split(frame: pd.DataFrame, time_column: str, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if time_column not in frame.columns:
        raise ValueError(f"Missing time column: {time_column}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    ordered = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    split = int(len(ordered) * (1 - validation_fraction))
    if split < 1 or split >= len(ordered):
        raise ValueError("Dataset is too small for requested validation fraction")
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def geographic_split(frame: pd.DataFrame, group_column: str, validation_fraction: float = 0.2,
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_column not in frame.columns:
        raise ValueError(f"Missing group column: {group_column}")
    if frame[group_column].nunique() < 2:
        raise ValueError("Geographic split requires at least two groups")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_indices, validation_indices = next(splitter.split(frame, groups=frame[group_column]))
    return frame.iloc[train_indices].copy(), frame.iloc[validation_indices].copy()
