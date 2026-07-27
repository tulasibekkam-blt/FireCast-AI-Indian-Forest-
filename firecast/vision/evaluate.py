from __future__ import annotations

from pathlib import Path
from typing import Any

from firecast.vision.dataset import validate_yolo_manifest


def evaluate_detector(data_yaml: str | Path, checkpoint: str | Path, output: str | Path,
                     image_size: int = 640, batch: int = 16, device: str | None = None) -> Any:
    """Evaluate a detector on the validated YOLO validation split."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install the vision extra to evaluate detectors") from error
    manifest = validate_yolo_manifest(data_yaml)
    weights = Path(checkpoint)
    if not weights.exists():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {weights}")
    return YOLO(str(weights)).val(
        data=str(Path(data_yaml).resolve()), split="val", imgsz=image_size, batch=batch,
        device=device, project=str(Path(output).resolve()), name="firecast-eval", plots=True,
    )
