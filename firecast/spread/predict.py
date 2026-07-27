from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def forecast(model_dir: str | Path, frame: pd.DataFrame) -> np.ndarray:
    """Forecast the final window in a real feature frame and return physical units."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install PyTorch to run spread inference") from error
    from firecast.spread.models import SpreadModelConfig, build_spread_model

    directory = Path(model_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    features = metadata["features"]
    lookback = int(metadata["lookback"])
    horizon = int(metadata["horizon"])
    if len(frame) < lookback:
        raise ValueError("Input frame is shorter than the trained lookback")
    inputs = frame[features].tail(lookback).to_numpy(dtype=np.float32)[None, ...]
    scalers = joblib.load(directory / "scalers.joblib")
    feature_scaler = scalers["feature_scaler"]
    target_scaler = scalers["target_scaler"]
    scaled = feature_scaler.transform(inputs[-1].reshape(-1, len(features))).reshape(1, lookback, len(features))
    model = build_spread_model(metadata["model"], SpreadModelConfig(input_features=len(features), horizon=horizon))
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(scaled.astype(np.float32))).numpy().reshape(-1, 1)
    return target_scaler.inverse_transform(prediction).reshape(horizon)
