from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_experiment_manifest(output: str | Path, dataset: str | Path, config: dict[str, Any],
                             seed: int, results: dict[str, Any] | None = None) -> Path:
    """Write a reproducibility manifest next to experiment artifacts."""
    source = Path(dataset)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source}")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset": str(source.resolve()),
        "dataset_sha256": sha256_file(source), "config": config, "seed": seed,
        "python": sys.version, "platform": platform.platform(), "results": results or {},
    }
    path = destination / "experiment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
