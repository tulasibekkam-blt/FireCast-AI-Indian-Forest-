from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class TabularDataset:
    frame: pd.DataFrame
    target: str
    time_column: str | None = None


def load_tabular(
    path: str | Path,
    target: str,
    time_column: str | None = None,
    deduplicate: bool = True,
) -> TabularDataset:
    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {source}"
        )

    if source.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(source)
        except Exception:
            frame = pd.read_csv(
                source,
                header=1,
            )

    elif source.suffix.lower() in {
        ".parquet",
        ".pq",
    }:
        frame = pd.read_parquet(source)

    else:
        raise ValueError(
            "Only CSV and Parquet datasets are supported"
        )

    if len(frame.columns) == 1:
        frame = pd.read_csv(
            source,
            header=1,
        )

    frame = frame.dropna(how="all")
    frame = frame.dropna(axis=1, how="all")

    frame.columns = (
        frame.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    target = (
        target.strip()
        .replace(" ", "_")
    )

    if target not in frame.columns:
        raise ValueError(
            f"Target column '{target}' not found.\n"
            f"Columns are:\n"
            f"{frame.columns.tolist()}"
        )

    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.strip()
            )

    frame = frame[
        frame[target].notna()
        & (frame[target] != "")
        & (frame[target] != "nan")
    ]

    frame[target] = (
        frame[target]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace(
            {
                "fire": 1,
                "not fire": 0,
                "not_fire": 0,
                "0": 0,
                "1": 1,
                "true": 1,
                "false": 0,
            }
        )
    )

    frame[target] = pd.to_numeric(
        frame[target],
        errors="coerce",
    )

    frame = frame.dropna(subset=[target])

    invalid_values = set(frame[target].unique()) - {0, 1}

    if invalid_values:
        raise ValueError(
            "Target must be binary and contain only 0 and 1. "
            f"Found invalid values: {sorted(invalid_values)}"
        )

    frame[target] = frame[target].astype(int)

    for column in frame.columns:
        if column == target:
            continue

        try:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            )
        except (ValueError, TypeError):
            pass

    if deduplicate:
        frame = (
            frame
            .drop_duplicates()
            .reset_index(drop=True)
        )

    if time_column:
        time_column = (
            time_column.strip()
            .replace(" ", "_")
        )

        if time_column not in frame.columns:
            raise ValueError(
                f"Time column '{time_column}' not found."
            )

        frame = (
            frame
            .sort_values(time_column)
            .reset_index(drop=True)
        )

    if frame.empty:
        raise ValueError(
            "Dataset is empty after cleaning."
        )

    if frame[target].nunique() != 2:
        raise ValueError(
            "Target must contain both binary classes "
            f"0 and 1. Found {frame[target].unique()}"
        )

    return TabularDataset(
        frame=frame,
        target=target,
        time_column=time_column,
    )


def feature_columns(
    dataset: TabularDataset,
    excluded: Iterable[str] = (),
) -> list[str]:
    excluded_set = {
        dataset.target,
        *(excluded or []),
    }

    return [
        column
        for column in dataset.frame.columns
        if column not in excluded_set
    ]