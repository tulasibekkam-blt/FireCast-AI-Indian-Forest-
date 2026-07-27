from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class FusionConfig:
    sensor_features: int
    vision_features: int
    hidden_size: int = 128
    classes: int = 1


if torch is not None:
    class SensorVisionFusion(nn.Module):
        """Gated late-fusion network with explicit modality availability masks."""

        def __init__(self, config: FusionConfig):
            super().__init__()
            if min(config.sensor_features, config.vision_features, config.hidden_size) < 1:
                raise ValueError("Fusion dimensions must be positive")
            self.sensor = nn.Sequential(nn.LayerNorm(config.sensor_features), nn.Linear(config.sensor_features, config.hidden_size), nn.GELU())
            self.vision = nn.Sequential(nn.LayerNorm(config.vision_features), nn.Linear(config.vision_features, config.hidden_size), nn.GELU())
            self.gate = nn.Sequential(nn.Linear(config.hidden_size * 2 + 2, config.hidden_size), nn.Sigmoid())
            self.head = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, config.classes))

        def forward(self, sensors: Tensor, vision: Tensor, availability: Tensor) -> Tensor:
            if sensors.ndim != 2 or vision.ndim != 2 or availability.ndim != 2 or availability.shape[1] != 2:
                raise ValueError("Expected sensors [B,S], vision [B,V], availability [B,2]")
            sensor_embedding = self.sensor(sensors) * availability[:, 0:1]
            vision_embedding = self.vision(vision) * availability[:, 1:2]
            fused = torch.cat([sensor_embedding, vision_embedding, availability], dim=1)
            gate = self.gate(fused)
            return self.head(gate * sensor_embedding + (1 - gate) * vision_embedding)
