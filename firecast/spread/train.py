from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from firecast.spread.data import make_windows
from firecast.spread.losses import PhysicsRegularizedLoss
from firecast.spread.models import SpreadModelConfig, build_spread_model


def train_spread(data: str | Path, features: list[str], target: str, output: str | Path,
                 kind: str = "transformer", lookback: int = 24, horizon: int = 6,
                 epochs: int = 50, batch_size: int = 64, seed: int = 42) -> dict[str, float]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("Install PyTorch to train spread models") from error
    torch.manual_seed(seed)
    frame = pd.read_csv(data) if Path(data).suffix.lower() == ".csv" else pd.read_parquet(data)
    split = int(len(frame) * 0.8)
    train_x, train_y = make_windows(frame.iloc[:split], features, target, lookback, horizon)
    valid_x, valid_y = make_windows(frame.iloc[split - lookback:], features, target, lookback, horizon)
    feature_scaler = StandardScaler().fit(train_x.reshape(-1, train_x.shape[-1]))
    target_scaler = StandardScaler().fit(train_y.reshape(-1, 1))
    train_x = feature_scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape).astype("float32")
    valid_x = feature_scaler.transform(valid_x.reshape(-1, valid_x.shape[-1])).reshape(valid_x.shape).astype("float32")
    train_y = target_scaler.transform(train_y.reshape(-1, 1)).reshape(train_y.shape).astype("float32")
    valid_y = target_scaler.transform(valid_y.reshape(-1, 1)).reshape(valid_y.shape).astype("float32")
    config = SpreadModelConfig(input_features=len(features), horizon=horizon)
    model = build_spread_model(kind, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    loss_fn = PhysicsRegularizedLoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=batch_size, shuffle=True)
    valid_inputs = torch.from_numpy(valid_x)
    valid_targets = torch.from_numpy(valid_y)
    best_loss = float("inf")
    stale = 0
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(inputs), targets).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(valid_inputs), valid_targets))
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss, stale = validation_loss, 0
            torch.save(model.state_dict(), destination / "model.pt")
        else:
            stale += 1
            if stale >= 8:
                break
    metadata = {"model": kind, "features": features, "target": target, "lookback": lookback,
                "horizon": horizon, "seed": seed, "validation_loss": best_loss}
    joblib.dump({"feature_scaler": feature_scaler, "target_scaler": target_scaler}, destination / "scalers.joblib")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"validation_loss": best_loss}
