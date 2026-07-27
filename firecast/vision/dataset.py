from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def validate_yolo_manifest(path: str | Path) -> dict[str, Any]:
    """Validate a YOLO dataset YAML and return its resolved manifest."""

    # Resolve the location of data.yaml itself.
    manifest_path = Path(path).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest does not exist: {manifest_path}"
        )

    # Load YAML.
    document = yaml.safe_load(
        manifest_path.read_text(encoding="utf-8")
    )

    if not isinstance(document, dict):
        raise ValueError(
            "YOLO manifest must be a YAML mapping"
        )

    # Read class names.
    names = document.get("names")

    if isinstance(names, list):
        class_names = names

    elif isinstance(names, dict):
        class_names = [
            names[index]
            for index in sorted(names)
        ]

    else:
        raise ValueError(
            "YOLO manifest requires names as a list or mapping"
        )

    # Validate class names.
    if (
        not class_names
        or any(
            not isinstance(name, str)
            or not name.strip()
            for name in class_names
        )
    ):
        raise ValueError(
            "YOLO class names must be non-empty strings"
        )

    # Determine dataset root.
    #
    # For your current data.yaml:
    #
    # train: ../train/images
    # val: ../valid/images
    # test: ../test/images
    #
    # There is no "path:" field, so the dataset root is
    # the directory containing data.yaml.
    dataset_root_value = document.get("path")

    if dataset_root_value is None:
        root = manifest_path.parent.resolve()

    else:
        root = Path(dataset_root_value)

        if not root.is_absolute():
            root = (
                manifest_path.parent / root
            ).resolve()
        else:
            root = root.resolve()

    # Resolve train / val / test directories.
    splits: dict[str, str] = {}

    for split in ("train", "val", "test"):
        value = document.get(split)

        if value is None:
            continue

        split_path = Path(value)

        if split_path.is_absolute():
            resolved = split_path.resolve()
        else:
            resolved = (
                root / split_path
            ).resolve()

        if not resolved.exists():
            raise FileNotFoundError(
                f"YOLO {split} split does not exist: {resolved}"
            )

        splits[split] = str(resolved)

    # YOLO training requires train and validation splits.
    if "train" not in splits:
        raise ValueError(
            "YOLO manifest requires an existing train split"
        )

    if "val" not in splits:
        raise ValueError(
            "YOLO manifest requires an existing val split"
        )

    # Return normalized manifest.
    return {
        "path": str(root),
        "names": class_names,
        "nc": len(class_names),
        "splits": splits,
    }