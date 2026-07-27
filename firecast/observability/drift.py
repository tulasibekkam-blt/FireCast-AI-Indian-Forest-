from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_dataframe(data):
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


def population_stability_index(reference, current, bins: int = 10):
    reference = pd.Series(reference).dropna().astype(float)
    current = pd.Series(current).dropna().astype(float)

    if reference.empty or current.empty:
        raise ValueError(
            "Reference and current distributions cannot be empty"
        )

    if bins < 1:
        raise ValueError("bins must be at least 1")

    if reference.equals(current):
        return 0.0

    min_value = min(reference.min(), current.min())
    max_value = max(reference.max(), current.max())

    if min_value == max_value:
        return 0.0

    bin_edges = np.linspace(
        min_value,
        max_value,
        bins + 1,
    )

    reference_counts, _ = np.histogram(
        reference,
        bins=bin_edges,
    )

    current_counts, _ = np.histogram(
        current,
        bins=bin_edges,
    )

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    epsilon = 1e-10

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None,
    )

    current_pct = np.clip(
        current_pct,
        epsilon,
        None,
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def drift_report(
    reference,
    current,
    columns,
    threshold: float = 0.05,
):
    reference = _load_dataframe(reference)
    current = _load_dataframe(current)

    results = []

    for column in columns:
        if column not in reference.columns:
            raise ValueError(
                f"Missing drift column: {column}"
            )

        if column not in current.columns:
            raise ValueError(
                f"Missing drift column: {column}"
            )

        ref = pd.to_numeric(
            reference[column],
            errors="coerce",
        ).dropna()

        cur = pd.to_numeric(
            current[column],
            errors="coerce",
        ).dropna()

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

        psi = population_stability_index(ref, cur)

        results.append(
            {
                "feature": column,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "reference_std": ref_std,
                "current_std": cur_std,
                "mean_shift": mean_shift,
                "std_shift": std_shift,
                "psi": psi,
                "drift_detected": drift_detected,
            }
        )

    return pd.DataFrame(results)