from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_model_artifact(output: str | Path, pipeline: Any, metadata: dict[str, Any]) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination / "model.joblib", compress=3)
    (destination / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8",
    )


def load_model_artifact(path: str | Path) -> tuple[Any, dict[str, Any]]:
    source = Path(path)
    pipeline = joblib.load(source / "model.joblib")
    metadata = json.loads((source / "model_metadata.json").read_text(encoding="utf-8"))
    return pipeline, metadata
