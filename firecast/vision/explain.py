from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:
    def integrated_gradients(model: nn.Module, image: torch.Tensor, score_fn, baseline: torch.Tensor | None = None,
                             steps: int = 32) -> torch.Tensor:
        """Compute pixel attributions along a straight path from baseline to image."""
        if image.ndim != 4 or steps < 1:
            raise ValueError("Expected batched image and positive integration steps")
        baseline = torch.zeros_like(image) if baseline is None else baseline
        if baseline.shape != image.shape:
            raise ValueError("Baseline and image must have identical shapes")
        delta = image - baseline
        total_gradient = torch.zeros_like(image)
        for index in range(1, steps + 1):
            point = (baseline + delta * (index / steps)).detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            score = score_fn(model(point))
            if score.ndim != 0:
                raise ValueError("score_fn must return a scalar tensor")
            score.backward()
            if point.grad is None:
                raise RuntimeError("Model produced no input gradient")
            total_gradient += point.grad.detach()
        return delta * total_gradient / steps


    class GradCAM:
        """Compute Grad-CAM activations for a selected convolutional layer."""

        def __init__(self, model: nn.Module, layer: nn.Module):
            self.model = model
            self.layer = layer
            self.activations = None
            self.gradients = None
            layer.register_forward_hook(self._capture_activation)
            layer.register_full_backward_hook(self._capture_gradient)

        def _capture_activation(self, _module, _inputs, output):
            self.activations = output

        def _capture_gradient(self, _module, _grad_inputs, grad_outputs):
            self.gradients = grad_outputs[0]

        def __call__(self, image: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
            if image.ndim != 4 or score.ndim != 0:
                raise ValueError("Expected image [batch, channels, height, width] and scalar score")
            self.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=True)
            if self.activations is None or self.gradients is None:
                raise RuntimeError("Selected layer did not produce gradients")
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            heatmap = (weights * self.activations).sum(dim=1, keepdim=True).relu()
            return heatmap / heatmap.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
