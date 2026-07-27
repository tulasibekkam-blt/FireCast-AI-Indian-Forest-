from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def _load_dataframe(data):
    """
    Accept either:
      1. pandas DataFrame
      2. CSV file
      3. Parquet file
    """

    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)

    raise ValueError("Only CSV and Parquet files are supported")


def drift_report(
    reference,
    current,
    columns,
    threshold: float = 0.05,
):
    """
    Compute simple feature drift report.

    Parameters
    ----------
    reference : DataFrame or path
    current : DataFrame or path
    columns : list[str]
    threshold : float

    Returns
    -------
    pandas.DataFrame
    """

    reference = _load_dataframe(reference)
    current = _load_dataframe(current)

    results = []

    for column in columns:

        if column not in reference.columns:
            raise ValueError(f"Missing drift column: {column}")

        if column not in current.columns:
            raise ValueError(f"Missing drift column: {column}")

        ref = pd.to_numeric(reference[column], errors="coerce").dropna()
        cur = pd.to_numeric(current[column], errors="coerce").dropna()

        ref_mean = float(ref.mean())
        cur_mean = float(cur.mean())

        ref_std = float(ref.std())
        cur_std = float(cur.std())

        mean_shift = abs(cur_mean - ref_mean)

        std_shift = abs(cur_std - ref_std)

        drift_detected = (
            mean_shift > threshold
            or std_shift > threshold
        )

        results.append(
            {
                "feature": column,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "reference_std": ref_std,
                "current_std": cur_std,
                "mean_shift": mean_shift,
                "std_shift": std_shift,
                "drift_detected": drift_detected,
            }
        )

    return pd.DataFrame(results)