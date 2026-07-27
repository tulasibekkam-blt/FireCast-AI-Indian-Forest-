from __future__ import annotations

import json
from pathlib import Path


def export_spread_model(checkpoint: str | Path, output: str | Path, input_features: int,
                        lookback: int, horizon: int, kind: str, opset: int = 17) -> dict[str, str]:
    """Export a trained PyTorch spread model to TorchScript and ONNX."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install PyTorch to export spread models") from error
    from firecast.spread.models import SpreadModelConfig, build_spread_model

    source = Path(checkpoint)
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    model = build_spread_model(kind, SpreadModelConfig(input_features=input_features, horizon=horizon))
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
    model.eval()
    example = torch.zeros(1, lookback, input_features)
    scripted_path = destination / "model.torchscript"
    onnx_path = destination / "model.onnx"
    torch.jit.trace(model, example).save(str(scripted_path))
    torch.onnx.export(
        model, example, str(onnx_path), opset_version=opset,
        input_names=["features"], output_names=["spread_forecast"],
        dynamic_axes={"features": {0: "batch", 1: "time"}, "spread_forecast": {0: "batch"}},
    )
    metadata = {"model": kind, "input_features": input_features, "lookback": lookback,
                "horizon": horizon, "opset": opset}
    (destination / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"torchscript": str(scripted_path), "onnx": str(onnx_path)}
