from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:
    class PhysicsRegularizedLoss(nn.Module):
        """Forecast loss with non-negativity and temporal smoothness penalties."""

        def __init__(self, physics_weight: float = 0.1, smoothness_weight: float = 0.05):
            super().__init__()
            if physics_weight < 0 or smoothness_weight < 0:
                raise ValueError("Loss weights must be non-negative")
            self.physics_weight = physics_weight
            self.smoothness_weight = smoothness_weight
            self.data_loss = nn.SmoothL1Loss()

        def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
            data = self.data_loss(prediction, target)
            nonnegative = torch.relu(-prediction).mean()
            smoothness = torch.diff(prediction, dim=-1).abs().mean() if prediction.shape[-1] > 1 else prediction.new_zeros(())
            return data + self.physics_weight * nonnegative + self.smoothness_weight * smoothness
