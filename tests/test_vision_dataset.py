import pytest

from firecast.vision.dataset import validate_yolo_manifest


def test_manifest_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_yolo_manifest(tmp_path / "missing.yaml")
