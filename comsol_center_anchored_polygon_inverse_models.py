"""PyTorch modules for center-anchored COMSOL polygon inverse experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from comsol_polygon_inverse_models import PolygonBzEncoder


class CenterAnchoredPolygonInverseNet(nn.Module):
    """Predict component centers and local polygon vertices from multi-height Bz signals."""

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
        local_shape_conditioning_mode: str = "none",
        local_shape_conditioning_dim: int = 16,
        joint_center_shape_mode: str = "none",
    ) -> None:
        super().__init__()
        if max_components != 3:
            raise ValueError("Center-anchored polygon route supports max_components=3 only.")
        if max_vertices != 4:
            raise ValueError("Center-anchored polygon route supports max_vertices=4 only.")
        if center_x_bins <= 0 or center_y_bins <= 0:
            raise ValueError("center bin counts must be positive.")
        if num_types <= 0:
            raise ValueError("num_types must be positive.")
        if local_shape_conditioning_mode not in {"none", "center_bin", "center_bin_slot", "center_bin_slot_type"}:
            raise ValueError(f"Unsupported local_shape_conditioning_mode: {local_shape_conditioning_mode}")
        if local_shape_conditioning_dim <= 0:
            raise ValueError("local_shape_conditioning_dim must be positive.")
        if joint_center_shape_mode not in {"none", "soft_center_scheduled"}:
            raise ValueError(f"Unsupported joint_center_shape_mode: {joint_center_shape_mode}")
        if joint_center_shape_mode != "none" and local_shape_conditioning_mode != "none":
            raise ValueError("joint_center_shape_mode and local_shape_conditioning_mode cannot both be enabled.")
        self.max_components = int(max_components)
        self.max_vertices = int(max_vertices)
        self.num_types = int(num_types)
        self.center_x_bins = int(center_x_bins)
        self.center_y_bins = int(center_y_bins)
        self.local_shape_conditioning_mode = local_shape_conditioning_mode
        self.local_shape_conditioning_dim = int(local_shape_conditioning_dim)
        self.joint_center_shape_mode = joint_center_shape_mode
        self.encoder = PolygonBzEncoder(
            signal_len=signal_len,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.presence_head = nn.Linear(latent_dim, self.max_components)
        self.type_head = nn.Linear(latent_dim, self.max_components * self.num_types)
        self.center_x_bin_head = nn.Linear(latent_dim, self.max_components * self.center_x_bins)
        self.center_y_bin_head = nn.Linear(latent_dim, self.max_components * self.center_y_bins)
        self.center_offset_head = nn.Linear(latent_dim, self.max_components * 2)
        if self.local_shape_conditioning_mode == "none" and self.joint_center_shape_mode == "none":
            self.local_vertex_head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.max_components * self.max_vertices * 2),
            )
        elif self.joint_center_shape_mode == "soft_center_scheduled":
            self.joint_local_vertex_head = nn.Sequential(
                nn.Linear(latent_dim + 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.max_vertices * 2),
            )
        else:
            self.center_x_bin_embedding = nn.Embedding(self.center_x_bins, self.local_shape_conditioning_dim)
            self.center_y_bin_embedding = nn.Embedding(self.center_y_bins, self.local_shape_conditioning_dim)
            input_dim = latent_dim + 2 * self.local_shape_conditioning_dim + 2
            if self.local_shape_conditioning_mode in {"center_bin_slot", "center_bin_slot_type"}:
                self.slot_embedding = nn.Embedding(self.max_components, self.local_shape_conditioning_dim)
                input_dim += self.local_shape_conditioning_dim
            if self.local_shape_conditioning_mode == "center_bin_slot_type":
                self.type_embedding = nn.Embedding(self.num_types, self.local_shape_conditioning_dim)
                input_dim += self.local_shape_conditioning_dim
            self.conditioned_local_vertex_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.max_vertices * 2),
            )

    def _soft_embedding(self, logits: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        probs = F.softmax(logits.detach(), dim=-1)
        return probs @ embedding.weight

    def _conditioned_local_vertices(
        self,
        latent: torch.Tensor,
        center_x_logits: torch.Tensor,
        center_y_logits: torch.Tensor,
        center_offset: torch.Tensor,
        type_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch = latent.shape[0]
        slot_count = self.max_components
        latent_slots = latent[:, None, :].expand(batch, slot_count, latent.shape[-1])
        pieces = [
            latent_slots,
            self._soft_embedding(center_x_logits, self.center_x_bin_embedding),
            self._soft_embedding(center_y_logits, self.center_y_bin_embedding),
            center_offset.detach(),
        ]
        if self.local_shape_conditioning_mode in {"center_bin_slot", "center_bin_slot_type"}:
            slot_idx = torch.arange(slot_count, device=latent.device)
            slot_context = self.slot_embedding(slot_idx)[None, :, :].expand(batch, slot_count, self.local_shape_conditioning_dim)
            pieces.append(slot_context)
        if self.local_shape_conditioning_mode == "center_bin_slot_type":
            pieces.append(self._soft_embedding(type_logits, self.type_embedding))
        conditioned = torch.cat(pieces, dim=-1)
        return self.conditioned_local_vertex_head(conditioned).view(batch, self.max_components, self.max_vertices, 2)

    def _soft_center_context(
        self,
        center_x_logits: torch.Tensor,
        center_y_logits: torch.Tensor,
        center_offset: torch.Tensor,
        x_centers: torch.Tensor,
        y_centers: torch.Tensor,
        bin_width_x: torch.Tensor,
        bin_width_y: torch.Tensor,
        grid_dx: torch.Tensor,
        grid_dy: torch.Tensor,
    ) -> torch.Tensor:
        x_probs = F.softmax(center_x_logits, dim=-1)
        y_probs = F.softmax(center_y_logits, dim=-1)
        center_x = x_probs @ x_centers.to(center_x_logits.dtype) + center_offset[..., 0] * bin_width_x.to(center_x_logits.dtype)
        center_y = y_probs @ y_centers.to(center_y_logits.dtype) + center_offset[..., 1] * bin_width_y.to(center_y_logits.dtype)
        return torch.stack(
            [
                center_x / grid_dx.to(center_x_logits.dtype).clamp_min(1.0e-12),
                center_y / grid_dy.to(center_y_logits.dtype).clamp_min(1.0e-12),
            ],
            dim=-1,
        )

    def _joint_local_vertices(
        self,
        latent: torch.Tensor,
        center_context: torch.Tensor,
    ) -> torch.Tensor:
        batch = latent.shape[0]
        latent_slots = latent[:, None, :].expand(batch, self.max_components, latent.shape[-1])
        conditioned = torch.cat([latent_slots, center_context.to(latent.dtype)], dim=-1)
        return self.joint_local_vertex_head(conditioned).view(batch, self.max_components, self.max_vertices, 2)

    def forward(
        self,
        signals: torch.Tensor,
        *,
        x_centers: torch.Tensor | None = None,
        y_centers: torch.Tensor | None = None,
        bin_width_x: torch.Tensor | None = None,
        bin_width_y: torch.Tensor | None = None,
        grid_dx: torch.Tensor | None = None,
        grid_dy: torch.Tensor | None = None,
        teacher_center_context: torch.Tensor | None = None,
        teacher_forcing_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        latent = self.encoder(signals)
        batch = signals.shape[0]
        presence_logits = self.presence_head(latent)
        type_logits = self.type_head(latent).view(batch, self.max_components, self.num_types)
        center_x_logits = self.center_x_bin_head(latent).view(batch, self.max_components, self.center_x_bins)
        center_y_logits = self.center_y_bin_head(latent).view(batch, self.max_components, self.center_y_bins)
        center_offset = self.center_offset_head(latent).view(batch, self.max_components, 2)
        if self.joint_center_shape_mode == "soft_center_scheduled":
            required = [x_centers, y_centers, bin_width_x, bin_width_y, grid_dx, grid_dy]
            if any(item is None for item in required):
                raise ValueError("joint_center_shape_mode requires center/grid tensors in forward().")
            pred_context = self._soft_center_context(
                center_x_logits,
                center_y_logits,
                center_offset,
                x_centers,
                y_centers,
                bin_width_x,
                bin_width_y,
                grid_dx,
                grid_dy,
            )
            if teacher_center_context is not None and teacher_forcing_weight > 0.0:
                weight = float(max(0.0, min(1.0, teacher_forcing_weight)))
                context = weight * teacher_center_context.to(pred_context.dtype) + (1.0 - weight) * pred_context
            else:
                context = pred_context
            local_vertices = self._joint_local_vertices(latent, context)
        elif self.local_shape_conditioning_mode == "none":
            local_vertices = self.local_vertex_head(latent).view(batch, self.max_components, self.max_vertices, 2)
        else:
            local_vertices = self._conditioned_local_vertices(latent, center_x_logits, center_y_logits, center_offset, type_logits)
        return {
            "presence_logits": presence_logits,
            "presence_prob": torch.sigmoid(presence_logits),
            "type_logits": type_logits,
            "center_x_bin_logits": center_x_logits,
            "center_y_bin_logits": center_y_logits,
            "center_offset": center_offset,
            "local_vertices_grid": local_vertices,
        }
