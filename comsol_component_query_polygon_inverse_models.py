"""Component-query models for COMSOL center-anchored polygon inverse experiments."""

from __future__ import annotations

import torch
from torch import nn

from comsol_polygon_inverse_models import PolygonBzEncoder


class ComponentQueryPolygonInverseNet(nn.Module):
    """Predict fixed-slot center and local polygon shape from shared component queries."""

    def __init__(
        self,
        signal_len: int,
        center_x_bins: int,
        center_y_bins: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_components: int = 3,
        max_vertices: int = 4,
        num_types: int = 2,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if max_components != 3:
            raise ValueError("Component-query polygon route supports max_components=3 only.")
        if max_vertices != 4:
            raise ValueError("Component-query polygon route supports max_vertices=4 only.")
        if center_x_bins <= 0 or center_y_bins <= 0:
            raise ValueError("center bin counts must be positive.")
        if num_types <= 0:
            raise ValueError("num_types must be positive.")
        self.max_components = int(max_components)
        self.max_vertices = int(max_vertices)
        self.num_types = int(num_types)
        self.center_x_bins = int(center_x_bins)
        self.center_y_bins = int(center_y_bins)
        self.encoder = PolygonBzEncoder(
            signal_len=signal_len,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.query_embedding = nn.Embedding(self.max_components, latent_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.presence_head = nn.Linear(latent_dim, 1)
        self.type_head = nn.Linear(latent_dim, self.num_types)
        self.center_x_bin_head = nn.Linear(latent_dim, self.center_x_bins)
        self.center_y_bin_head = nn.Linear(latent_dim, self.center_y_bins)
        self.center_offset_head = nn.Linear(latent_dim, 2)
        self.local_vertex_head = nn.Linear(latent_dim, self.max_vertices * 2)

    def forward(self, signals: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(signals)
        batch = signals.shape[0]
        slot_idx = torch.arange(self.max_components, device=signals.device)
        query = self.query_embedding(slot_idx)[None, :, :].expand(batch, self.max_components, latent.shape[-1])
        latent_slots = latent[:, None, :].expand(batch, self.max_components, latent.shape[-1])
        component_latent = self.query_mlp(torch.cat([latent_slots, query], dim=-1))
        presence_logits = self.presence_head(component_latent).squeeze(-1)
        type_logits = self.type_head(component_latent)
        center_x_logits = self.center_x_bin_head(component_latent)
        center_y_logits = self.center_y_bin_head(component_latent)
        center_offset = self.center_offset_head(component_latent)
        local_vertices = self.local_vertex_head(component_latent).view(batch, self.max_components, self.max_vertices, 2)
        return {
            "presence_logits": presence_logits,
            "presence_prob": torch.sigmoid(presence_logits),
            "type_logits": type_logits,
            "center_x_bin_logits": center_x_logits,
            "center_y_bin_logits": center_y_logits,
            "center_offset": center_offset,
            "local_vertices_grid": local_vertices,
        }
