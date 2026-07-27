from __future__ import annotations

import torch
from torch import Tensor


def random_modality_dropout(availability: Tensor, probability: float = 0.2) -> Tensor:
    """Randomly remove available modalities during training, retaining at least one."""
    if availability.ndim != 2 or availability.shape[1] < 2:
        raise ValueError("availability must have shape [batch, modalities]")
    if not 0 <= probability < 1:
        raise ValueError("probability must be in [0, 1)")
    dropped = availability.clone()
    mask = (torch.rand_like(dropped) < probability) & (dropped > 0)
    dropped[mask] = 0
    missing = dropped.sum(dim=1) == 0
    if missing.any():
        available = availability[missing]
        choice = torch.multinomial(available.float(), 1).squeeze(1)
        dropped[missing, choice] = 1
    return dropped


def masked_binary_loss(logits: Tensor, labels: Tensor, valid: Tensor) -> Tensor:
    """Compute BCE only for samples with valid labels."""
    if logits.shape[0] != labels.shape[0] or valid.shape[0] != labels.shape[0]:
        raise ValueError("logits, labels, and valid must share batch dimension")
    mask = valid.bool().reshape(-1)
    if not mask.any():
        raise ValueError("No valid labels in batch")
    return torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1)[mask], labels.float().reshape(-1)[mask])
