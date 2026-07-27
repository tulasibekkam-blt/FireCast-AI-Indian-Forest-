import pytest

torch = pytest.importorskip("torch")

from firecast.fusion.model import FusionConfig, SensorVisionFusion


def test_fusion_supports_missing_vision_modality():
    model = SensorVisionFusion(FusionConfig(3, 4))
    result = model(torch.ones(2, 3), torch.ones(2, 4), torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
    assert result.shape == (2, 1)
