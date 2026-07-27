from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment configuration."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Configuration does not exist: {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    required_sections = {"data", "model", "evaluation", "deployment"}
    missing = required_sections - set(document)
    if missing:
        raise ValueError(f"Configuration missing sections: {sorted(missing)}")
    return document
