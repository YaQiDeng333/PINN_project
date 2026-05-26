"""PyTorch modules for COMSOL polygon inverse experiments."""

from __future__ import annotations

import torch
from torch import nn


class PolygonBzEncoder(nn.Module):
    """MLP encoder for flattened multi-height Bz signals."""

    def __init__(
        self,
        signal_len: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if signal_len <= 0:
            raise ValueError("signal_len must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        layers: list[nn.Module] = []
        in_dim = signal_len
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 2:
            raise ValueError(f"signals must have shape [B,L], got {tuple(signals.shape)}")
        return self.net(signals)


class PolygonInverseNet(nn.Module):
    """Predict fixed-slot polygon vertices from multi-height Bz signals."""

    def __init__(
        self,
        signal_len: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_components: int = 3,
        max_vertices: int = 4,
        num_types: int = 2,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if max_components != 3:
            raise ValueError("First polygon inverse route supports max_components=3 only.")
        if max_vertices != 4:
            raise ValueError("First polygon inverse route supports max_vertices=4 only.")
        if num_types <= 0:
            raise ValueError("num_types must be positive.")
        self.max_components = int(max_components)
        self.max_vertices = int(max_vertices)
        self.num_types = int(num_types)
        self.encoder = PolygonBzEncoder(
            signal_len=signal_len,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.presence_head = nn.Linear(latent_dim, self.max_components)
        self.type_head = nn.Linear(latent_dim, self.max_components * self.num_types)
        self.vertex_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_components * self.max_vertices * 2),
        )

    def forward(self, signals: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(signals)
        batch = signals.shape[0]
        presence_logits = self.presence_head(latent)
        type_logits = self.type_head(latent).view(batch, self.max_components, self.num_types)
        vertices_norm = self.vertex_head(latent).view(batch, self.max_components, self.max_vertices, 2)
        return {
            "presence_logits": presence_logits,
            "presence_prob": torch.sigmoid(presence_logits),
            "type_logits": type_logits,
            "vertices_norm": vertices_norm,
        }
