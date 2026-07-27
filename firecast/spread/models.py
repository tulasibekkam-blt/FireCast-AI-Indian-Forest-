from __future__ import annotations

from dataclasses import dataclass


try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - dependency-gated module
    torch = None


@dataclass(frozen=True)
class SpreadModelConfig:
    input_features: int
    hidden_size: int = 128
    layers: int = 2
    horizon: int = 1
    dropout: float = 0.1
    heads: int = 4


if torch is not None:
    class SequenceSpreadModel(nn.Module):
        """Predict future spread intensity from [batch, time, features] inputs."""

        def __init__(self, config: SpreadModelConfig, recurrent: str):
            super().__init__()
            if config.input_features < 1 or config.horizon < 1:
                raise ValueError("input_features and horizon must be positive")
            if recurrent == "lstm":
                self.encoder = nn.LSTM(config.input_features, config.hidden_size, config.layers,
                                       batch_first=True, dropout=config.dropout if config.layers > 1 else 0)
            elif recurrent == "gru":
                self.encoder = nn.GRU(config.input_features, config.hidden_size, config.layers,
                                      batch_first=True, dropout=config.dropout if config.layers > 1 else 0)
            else:
                raise ValueError(f"Unsupported recurrent backbone: {recurrent}")
            self.head = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, config.horizon))

        def forward(self, values: Tensor) -> Tensor:
            if values.ndim != 3:
                raise ValueError("Expected input shape [batch, time, features]")
            encoded, _ = self.encoder(values)
            return self.head(encoded[:, -1])


    class TransformerSpreadModel(nn.Module):
        """Causal Transformer forecaster for weather/terrain feature sequences."""

        def __init__(self, config: SpreadModelConfig):
            super().__init__()
            if config.input_features < 1 or config.hidden_size % config.heads:
                raise ValueError("hidden_size must be divisible by heads")
            self.projection = nn.Linear(config.input_features, config.hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.heads, dropout=config.dropout,
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.head = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, config.horizon))

        def forward(self, values: Tensor) -> Tensor:
            if values.ndim != 3:
                raise ValueError("Expected input shape [batch, time, features]")
            sequence_length = values.shape[1]
            causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, device=values.device), diagonal=1).bool()
            encoded = self.encoder(self.projection(values), mask=causal_mask)
            return self.head(encoded[:, -1])


def build_spread_model(kind: str, config: SpreadModelConfig):
    if torch is None:
        raise RuntimeError("Install the vision extra or torch to use spread models")
    if kind in {"lstm", "gru"}:
        return SequenceSpreadModel(config, kind)
    if kind == "transformer":
        return TransformerSpreadModel(config)
    raise ValueError(f"Unknown spread model: {kind}")
