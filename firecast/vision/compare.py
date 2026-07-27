from __future__ import annotations

from pathlib import Path

import pandas as pd

from firecast.vision.dataset import validate_yolo_manifest


def compare_checkpoints(data_yaml: str | Path, checkpoints: dict[str, str | Path], output: str | Path,
                        image_size: int = 640, batch: int = 16, device: str | None = None) -> pd.DataFrame:
    """Evaluate multiple existing detector checkpoints with the same validation manifest."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install the vision extra to compare detectors") from error
    validate_yolo_manifest(data_yaml)
    rows = []
    for name, checkpoint in checkpoints.items():
        source = Path(checkpoint)
        if not source.is_file():
            raise FileNotFoundError(f"Detector checkpoint does not exist: {source}")
        metrics = YOLO(str(source)).val(data=str(Path(data_yaml).resolve()), split="val",
                                        imgsz=image_size, batch=batch, device=device, plots=False)
        box = metrics.box
        rows.append({"model": name, "checkpoint": str(source), "map50": float(box.map50),
                     "map50_95": float(box.map), "precision": float(box.mp), "recall": float(box.mr),
                     "model_size_mb": source.stat().st_size / (1024 * 1024)})
    result = pd.DataFrame(rows).sort_values("map50_95", ascending=False, ignore_index=True)
    result.to_csv(output, index=False)
    return result
