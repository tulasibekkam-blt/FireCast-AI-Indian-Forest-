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
    """
    Load wildfire tabular dataset.

    Automatically handles:
    - Algerian dataset title row
    - Blank rows
    - Extra spaces
    - fire/not fire labels
    - Numeric conversion
    """

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(f"Dataset does not exist: {source}")

    # -------------------------------------------------------
    # READ DATA
    # -------------------------------------------------------

    if source.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(source)
        except Exception:
            frame = pd.read_csv(source, header=1)
    elif source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise ValueError("Only CSV and Parquet datasets are supported")

    # -------------------------------------------------------
    # Algerian dataset fix
    # -------------------------------------------------------

    if len(frame.columns) == 1:
        frame = pd.read_csv(source, header=1)

    # -------------------------------------------------------
    # Remove empty rows/columns
    # -------------------------------------------------------

    frame = frame.dropna(how="all")
    frame = frame.dropna(axis=1, how="all")

    # -------------------------------------------------------
    # Clean column names
    # -------------------------------------------------------

    frame.columns = (
        frame.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    # target name after cleaning
    target = target.strip().replace(" ", "_")

    if target not in frame.columns:
        raise ValueError(
            f"Target column '{target}' not found.\nColumns are:\n{frame.columns.tolist()}"
        )

    # -------------------------------------------------------
    # Strip whitespace from strings
    # -------------------------------------------------------

    for c in frame.columns:
        if frame[c].dtype == object:
            frame[c] = frame[c].astype(str).str.strip()

    # -------------------------------------------------------
    # Remove rows without target
    # -------------------------------------------------------

    frame = frame[
        frame[target].notna()
        & (frame[target] != "")
        & (frame[target] != "nan")
    ]

    # -------------------------------------------------------
    # Convert fire/not fire -> 1/0
    # -------------------------------------------------------

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

    frame[target] = pd.to_numeric(frame[target], errors="coerce")

    frame = frame.dropna(subset=[target])

    frame[target] = frame[target].astype(int)

    # -------------------------------------------------------
    # Convert numeric columns
    # -------------------------------------------------------

    for c in frame.columns:
        if c == target:
            continue
        try:
            frame[c] = pd.to_numeric(frame[c], errors="raise")
        except (ValueError, TypeError):
            pass

    # -------------------------------------------------------
    # Remove duplicates
    # -------------------------------------------------------

    if deduplicate:
        frame = frame.drop_duplicates().reset_index(drop=True)

    # -------------------------------------------------------
    # Time column
    # -------------------------------------------------------

    if time_column:

        time_column = (
            time_column.strip()
            .replace(" ", "_")
        )

        if time_column not in frame.columns:
            raise ValueError(
                f"Time column '{time_column}' not found."
            )

        frame = frame.sort_values(time_column).reset_index(drop=True)

    # -------------------------------------------------------
    # Final checks
    # -------------------------------------------------------

    if frame.empty:
        raise ValueError("Dataset is empty after cleaning.")

    if frame[target].nunique() != 2:
        raise ValueError(
            f"Target must contain both classes. Found {frame[target].unique()}"
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

    excluded_set = {dataset.target, *(excluded or [])}

    return [
        c
        for c in dataset.frame.columns
        if c not in excluded_set
    ]