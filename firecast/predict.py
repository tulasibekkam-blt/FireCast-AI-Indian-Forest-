from __future__ import annotations

from pathlib import Path

import pandas as pd

from firecast.artifacts import load_model_artifact


def predict_file(artifact: str | Path, input_path: str | Path) -> pd.DataFrame:
    """Run inference using only the persisted training pipeline and feature contract."""
    pipeline, metadata = load_model_artifact(artifact)
    source = Path(input_path)
    frame = pd.read_csv(source) if source.suffix.lower() == ".csv" else pd.read_parquet(source)
    columns = metadata["feature_columns"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing trained feature columns: {missing}")
    probabilities = pipeline.predict_proba(frame[columns])[:, 1]
    threshold = float(metadata["threshold"])
    return pd.DataFrame({
        "risk_probability": probabilities,
        "predicted_ignition": (probabilities >= threshold).astype(int),
    })
