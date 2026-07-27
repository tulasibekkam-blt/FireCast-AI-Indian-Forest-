from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


def readiness_report(artifact_dir: str | Path, required_modules: list[str] | None = None) -> dict:
    """Return a serializable readiness report without mutating the deployment."""
    directory = Path(artifact_dir)
    required_files = ["model.joblib", "model_metadata.json"]
    files = {name: (directory / name).is_file() for name in required_files}
    modules = {name: importlib.util.find_spec(name) is not None for name in (required_modules or [])}
    disk = shutil.disk_usage(directory if directory.exists() else directory.parent)
    return {
        "artifact_directory": str(directory), "artifacts_present": files,
        "modules_available": modules, "free_disk_bytes": int(disk.free),
        "ready": all(files.values()) and all(modules.values()),
    }
