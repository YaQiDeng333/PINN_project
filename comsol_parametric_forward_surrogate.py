"""Learned geometry-to-Bz surrogate utilities for COMSOL parametric experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def build_forward_geometry_vector(
    presence: torch.Tensor,
    type_targets_or_probs: torch.Tensor,
    continuous: torch.Tensor,
    num_types: int,
    target_schema: list[str] | tuple[str, ...],
) -> torch.Tensor:
    """Flatten fixed-order component geometry into a differentiable vector."""

    if presence.ndim == 3 and presence.shape[-1] == 1:
        presence = presence.squeeze(-1)
    if presence.ndim != 2:
        raise ValueError(f"presence must have shape [B,K] or [B,K,1], got {tuple(presence.shape)}")
    if continuous.ndim != 3:
        raise ValueError(f"continuous must have shape [B,K,P], got {tuple(continuous.shape)}")
    if presence.shape[:2] != continuous.shape[:2]:
        raise ValueError("presence and continuous must share [B,K].")
    if len(target_schema) != continuous.shape[2]:
        raise ValueError(
            f"target_schema length {len(target_schema)} does not match continuous P={continuous.shape[2]}."
        )
    if num_types <= 0:
        raise ValueError("num_types must be positive.")

    if type_targets_or_probs.ndim == 2:
        if type_targets_or_probs.shape != presence.shape:
            raise ValueError("integer type targets must have shape [B,K].")
        type_one_hot = torch.zeros(
            (*presence.shape, num_types),
            dtype=continuous.dtype,
            device=continuous.device,
        )
        type_targets = type_targets_or_probs.to(device=continuous.device, dtype=torch.long)
        valid = (type_targets >= 0) & (type_targets < num_types)
        if valid.any():
            safe_targets = type_targets.clamp(0, num_types - 1).unsqueeze(-1)
            type_one_hot.scatter_(2, safe_targets, 1.0)
            type_one_hot = type_one_hot * valid.unsqueeze(-1).to(type_one_hot.dtype)
    elif type_targets_or_probs.ndim == 3:
        if type_targets_or_probs.shape[:2] != presence.shape or type_targets_or_probs.shape[2] != num_types:
            raise ValueError(f"type probabilities must have shape [B,K,{num_types}].")
        type_one_hot = type_targets_or_probs.to(device=continuous.device, dtype=continuous.dtype)
    else:
        raise ValueError(
            "type_targets_or_probs must have shape [B,K] integer targets or [B,K,T] probabilities."
        )

    presence_f = presence.to(device=continuous.device, dtype=continuous.dtype).unsqueeze(-1)
    component_vector = torch.cat([presence_f, type_one_hot, continuous], dim=2)
    return component_vector.reshape(component_vector.shape[0], -1)


class ParametricForwardSurrogate(nn.Module):
    """Small MLP surrogate mapping fixed-order geometry vectors to Bz signals."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 600,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, geometry_vector: torch.Tensor) -> torch.Tensor:
        if geometry_vector.ndim != 2:
            raise ValueError(f"geometry_vector must have shape [B,D], got {tuple(geometry_vector.shape)}")
        if geometry_vector.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {geometry_vector.shape[1]}.")
        return self.net(geometry_vector)


@dataclass(frozen=True)
class SignalNormalizationStats:
    """Per-signal-dimension z-score statistics."""

    mean: np.ndarray
    std: np.ndarray


def compute_train_zscore_stats(signals_flat: np.ndarray, eps: float = 1e-8) -> SignalNormalizationStats:
    if signals_flat.ndim != 2:
        raise ValueError(f"signals_flat must have shape [N,D], got {signals_flat.shape}")
    mean = signals_flat.mean(axis=0).astype(np.float32)
    std = signals_flat.std(axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return SignalNormalizationStats(mean=mean, std=std)


def normalize_signals(signals_flat: np.ndarray, stats: SignalNormalizationStats) -> np.ndarray:
    if signals_flat.ndim != 2:
        raise ValueError(f"signals_flat must have shape [N,D], got {signals_flat.shape}")
    if signals_flat.shape[1] != stats.mean.shape[0]:
        raise ValueError("signals and normalization stats have different signal lengths.")
    return ((signals_flat - stats.mean.reshape(1, -1)) / stats.std.reshape(1, -1)).astype(np.float32)


def denormalize_signals(signals_norm: np.ndarray, stats: SignalNormalizationStats) -> np.ndarray:
    if signals_norm.ndim != 2:
        raise ValueError(f"signals_norm must have shape [N,D], got {signals_norm.shape}")
    if signals_norm.shape[1] != stats.mean.shape[0]:
        raise ValueError("signals and normalization stats have different signal lengths.")
    return (signals_norm * stats.std.reshape(1, -1) + stats.mean.reshape(1, -1)).astype(np.float32)


def _main() -> int:
    print(
        "COMSOL parametric forward surrogate utilities. "
        "Use train_comsol_parametric_forward_surrogate.py to run training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
