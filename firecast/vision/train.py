from __future__ import annotations

from pathlib import Path
from typing import Any

from firecast.vision.dataset import validate_yolo_manifest


def train_detector(
    data_yaml: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    epochs: int = 100,
    image_size: int = 640,
    batch: int = 16,
    device: str | None = None,
    seed: int = 42,
    resume: bool = False,
) -> Any:
    """
    Train a YOLO detector or resume an interrupted training run.

    For new training:
        checkpoint = yolo11n.pt
        resume = False

    For resuming:
        checkpoint = path to last.pt
        resume = True

    IMPORTANT:
    `epochs` is the TOTAL target epoch count when resuming.

    Example:
        First run: 10 epochs
        Resume: --epochs 100

    YOLO will continue from epoch 10 and train until epoch 100.
    """

    from ultralytics import YOLO

    # ---------------------------------------------------------
    # Validate dataset
    # ---------------------------------------------------------

    manifest = validate_yolo_manifest(data_yaml)

    # ---------------------------------------------------------
    # Validate checkpoint
    # ---------------------------------------------------------

    weights = Path(checkpoint).resolve()

    if not weights.exists():
        raise FileNotFoundError(
            f"Model checkpoint does not exist: {weights}"
        )

    # ---------------------------------------------------------
    # Prepare output directory
    # ---------------------------------------------------------

    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Print training information
    # ---------------------------------------------------------

    print()
    print("=" * 70)

    if resume:
        print("RESUMING YOLO TRAINING")
    else:
        print("STARTING NEW YOLO TRAINING")

    print("=" * 70)

    print(f"Dataset YAML : {Path(data_yaml).resolve()}")
    print(f"Dataset root : {manifest['path']}")
    print(f"Classes      : {manifest['names']}")
    print(f"Checkpoint   : {weights}")
    print(f"Output       : {output_path}")
    print(f"Target epochs: {epochs}")
    print(f"Image size   : {image_size}")
    print(f"Batch size   : {batch}")
    print(f"Device       : {device}")
    print(f"Resume       : {resume}")

    print("=" * 70)
    print()

    # ---------------------------------------------------------
    # Load YOLO model
    # ---------------------------------------------------------

    model = YOLO(str(weights))

    # ---------------------------------------------------------
    # Training arguments
    # ---------------------------------------------------------

    train_args = {
        "data": str(Path(data_yaml).resolve()),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch,
        "device": device,
        "project": str(output_path),
        "name": "firecast",
        "seed": seed,
        "verbose": True,
    }

    # ---------------------------------------------------------
    # Resume interrupted training
    # ---------------------------------------------------------

    if resume:
        train_args["resume"] = True

    # ---------------------------------------------------------
    # Start training
    # ---------------------------------------------------------

    results = model.train(**train_args)

    return results