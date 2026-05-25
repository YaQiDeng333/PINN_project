#!/usr/bin/env python
"""RBC-style profile generator for Stage 20.83 profile-primary loss.

This module mirrors the existing NumPy RBC-style approximation in
``load_true_3d_rbc_pilot_dataset.py`` and adds a torch implementation for
differentiable training losses. It is not an exact Piao reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from load_true_3d_rbc_pilot_dataset import (
    GRID_U_COUNT,
    GRID_V_COUNT,
    MASK_HEIGHT,
    MASK_WIDTH,
    MASK_X_START_M,
    MASK_X_STOP_M,
    MASK_Y_START_M,
    MASK_Y_STOP_M,
    depth_grid_from_params,
    depth_map_from_params,
    projected_mask_from_params,
)


@dataclass(frozen=True)
class ProfileGeneratorConfig:
    grid_u_count: int = GRID_U_COUNT
    grid_v_count: int = GRID_V_COUNT
    mask_height: int = MASK_HEIGHT
    mask_width: int = MASK_WIDTH
    mask_x_start_m: float = MASK_X_START_M
    mask_x_stop_m: float = MASK_X_STOP_M
    mask_y_start_m: float = MASK_Y_START_M
    mask_y_stop_m: float = MASK_Y_STOP_M
    eps: float = 1.0e-9


def _as_batch_params(params: torch.Tensor) -> torch.Tensor:
    if params.ndim == 1:
        params = params.unsqueeze(0)
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError(f"expected params shape (N,6), got {tuple(params.shape)}")
    return params


def clip_params_torch(params: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    params = _as_batch_params(params)
    lower = lower.to(device=params.device, dtype=params.dtype).reshape(1, 6)
    upper = upper.to(device=params.device, dtype=params.dtype).reshape(1, 6)
    return torch.minimum(torch.maximum(params, lower), upper)


def soft_bound_penalty(params: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    params = _as_batch_params(params)
    lower = lower.to(device=params.device, dtype=params.dtype).reshape(1, 6)
    upper = upper.to(device=params.device, dtype=params.dtype).reshape(1, 6)
    below = torch.relu(lower - params)
    above = torch.relu(params - upper)
    span = torch.clamp(upper - lower, min=1.0e-12)
    return torch.mean(((below + above) / span) ** 2)


def rbc_weight_curve_torch(values: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    clipped = torch.clamp(values, 0.0, 1.0)
    safe_weight = torch.clamp(weight, min=eps)
    numerator = safe_weight * (1.0 - clipped * clipped)
    denominator = numerator + clipped * clipped + eps
    return torch.clamp(numerator / denominator, 0.0, 1.0)


def local_depth_torch(params: torch.Tensor, u: torch.Tensor, v: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    params = _as_batch_params(params)
    d_m = params[:, 2].reshape(-1, 1, 1)
    wld = params[:, 3].reshape(-1, 1, 1)
    wwd = params[:, 4].reshape(-1, 1, 1)
    wlw = params[:, 5].reshape(-1, 1, 1)
    u_abs = torch.abs(u).unsqueeze(0)
    v_abs = torch.abs(v).unsqueeze(0)
    length_profile = rbc_weight_curve_torch(u_abs, wld, eps)
    width_scale = torch.clamp(rbc_weight_curve_torch(u_abs, wlw, eps), min=eps)
    v_norm = v_abs / width_scale
    inside = (u_abs <= 1.0) & (v_norm <= 1.0)
    width_profile = rbc_weight_curve_torch(v_norm, wwd, eps)
    return torch.where(inside, torch.clamp(d_m, min=0.0) * length_profile * width_profile, torch.zeros_like(width_profile))


def grid_uv_torch(config: ProfileGeneratorConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.linspace(-1.0, 1.0, config.grid_u_count, device=device, dtype=dtype)
    v = torch.linspace(-1.0, 1.0, config.grid_v_count, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    return uu, vv


def depth_grid_torch(params: torch.Tensor, config: ProfileGeneratorConfig | None = None) -> torch.Tensor:
    config = config or ProfileGeneratorConfig()
    params = _as_batch_params(params)
    uu, vv = grid_uv_torch(config, params.device, params.dtype)
    return local_depth_torch(params, uu, vv, config.eps)


def xy_to_uv_torch(params: torch.Tensor, pose: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor, eps: float = 1.0e-9) -> tuple[torch.Tensor, torch.Tensor]:
    params = _as_batch_params(params)
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    pose = pose.to(device=params.device, dtype=params.dtype)
    l_m = torch.clamp(params[:, 0].reshape(-1, 1, 1), min=eps)
    w_m = torch.clamp(params[:, 1].reshape(-1, 1, 1), min=eps)
    center_x = pose[:, 0].reshape(-1, 1, 1)
    center_y = pose[:, 1].reshape(-1, 1, 1)
    angle_rad = pose[:, 2].reshape(-1, 1, 1)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    dx = xx.unsqueeze(0) - center_x
    dy = yy.unsqueeze(0) - center_y
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return 2.0 * local_x / l_m, 2.0 * local_y / w_m


def mask_xy_torch(config: ProfileGeneratorConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mask_x = torch.linspace(config.mask_x_start_m, config.mask_x_stop_m, config.mask_width, device=device, dtype=dtype)
    mask_y = torch.linspace(config.mask_y_start_m, config.mask_y_stop_m, config.mask_height, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(mask_y, mask_x, indexing="ij")
    return xx, yy


def depth_map_torch(params: torch.Tensor, pose: torch.Tensor, config: ProfileGeneratorConfig | None = None) -> torch.Tensor:
    config = config or ProfileGeneratorConfig()
    params = _as_batch_params(params)
    xx, yy = mask_xy_torch(config, params.device, params.dtype)
    u, v = xy_to_uv_torch(params, pose, xx, yy, config.eps)
    return local_depth_from_batched_uv(params, u, v, config.eps)


def local_depth_from_batched_uv(params: torch.Tensor, u: torch.Tensor, v: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    params = _as_batch_params(params)
    d_m = params[:, 2].reshape(-1, 1, 1)
    wld = params[:, 3].reshape(-1, 1, 1)
    wwd = params[:, 4].reshape(-1, 1, 1)
    wlw = params[:, 5].reshape(-1, 1, 1)
    u_abs = torch.abs(u)
    v_abs = torch.abs(v)
    length_profile = rbc_weight_curve_torch(u_abs, wld, eps)
    width_scale = torch.clamp(rbc_weight_curve_torch(u_abs, wlw, eps), min=eps)
    v_norm = v_abs / width_scale
    inside = (u_abs <= 1.0) & (v_norm <= 1.0)
    width_profile = rbc_weight_curve_torch(v_norm, wwd, eps)
    return torch.where(inside, torch.clamp(d_m, min=0.0) * length_profile * width_profile, torch.zeros_like(width_profile))


def soft_projected_mask_torch(params: torch.Tensor, pose: torch.Tensor, config: ProfileGeneratorConfig | None = None) -> torch.Tensor:
    config = config or ProfileGeneratorConfig()
    params = _as_batch_params(params)
    depth = depth_map_torch(params, pose, config)
    d_m = torch.clamp(params[:, 2].reshape(-1, 1, 1), min=config.eps)
    threshold = torch.clamp(0.01 * d_m, min=1.0e-6)
    temperature = torch.clamp(0.02 * d_m, min=2.0e-5)
    return torch.sigmoid((depth - threshold) / temperature)


def profile_depth_rmse(pred_grid: np.ndarray, true_grid: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(pred_grid) - np.asarray(true_grid)) ** 2, axis=(1, 2)))


def er_like_profile_error(pred_grid: np.ndarray, true_grid: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_grid, dtype=np.float64)
    true = np.asarray(true_grid, dtype=np.float64)
    numerator = np.sum((pred - true) ** 2, axis=(1, 2))
    denominator = np.maximum(np.sum(true * true, axis=(1, 2)), 1.0e-20)
    return np.sqrt(numerator / denominator)


def max_depth_error(pred_grid: np.ndarray, true_grid: np.ndarray) -> np.ndarray:
    return np.abs(np.max(pred_grid, axis=(1, 2)) - np.max(true_grid, axis=(1, 2)))


def generator_consistency_rows(dataset: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_grids = np.asarray([depth_grid_from_params(params) for params in dataset.rbc_params], dtype=np.float32)
    pred_maps = np.asarray([depth_map_from_params(params, pose) for params, pose in zip(dataset.rbc_params, dataset.profile_pose)], dtype=np.float32)
    pred_masks = np.asarray([projected_mask_from_params(params, pose) for params, pose in zip(dataset.rbc_params, dataset.profile_pose)], dtype=np.uint8)
    grid_abs = np.abs(pred_grids - dataset.profile_depth_grid_m)
    map_abs = np.abs(pred_maps - dataset.profile_depth_map_xy_m)
    mask_match = pred_masks == dataset.projected_mask_2d
    grid_rmse = profile_depth_rmse(pred_grids, dataset.profile_depth_grid_m)
    er_like = er_like_profile_error(pred_grids, dataset.profile_depth_grid_m)
    rows.extend(
        [
            {"check_name": "numpy_depth_grid_max_abs_error", "pass": float(grid_abs.max()) <= 1.0e-8, "observed": float(grid_abs.max()), "notes": "true params -> stored profile_depth_grid_m"},
            {"check_name": "numpy_depth_grid_mean_rmse", "pass": float(grid_rmse.mean()) <= 1.0e-9, "observed": float(grid_rmse.mean()), "notes": "RBC-style generator replay"},
            {"check_name": "numpy_er_like_mean", "pass": float(er_like.mean()) <= 1.0e-6, "observed": float(er_like.mean()), "notes": "self-consistency only"},
            {"check_name": "numpy_depth_map_max_abs_error", "pass": float(map_abs.max()) <= 1.0e-8, "observed": float(map_abs.max()), "notes": "true params -> stored profile_depth_map_xy_m"},
            {"check_name": "numpy_projected_mask_pixel_mismatch", "pass": int((~mask_match).sum()) == 0, "observed": int((~mask_match).sum()), "notes": f"total_pixels={mask_match.size}"},
        ]
    )
    with torch.no_grad():
        params = torch.as_tensor(dataset.rbc_params, dtype=torch.float32)
        pose = torch.as_tensor(dataset.profile_pose, dtype=torch.float32)
        torch_grid = depth_grid_torch(params).cpu().numpy()
        torch_map = depth_map_torch(params, pose).cpu().numpy()
    rows.extend(
        [
            {"check_name": "torch_depth_grid_max_abs_error_vs_numpy", "pass": float(np.max(np.abs(torch_grid - pred_grids))) <= 2.0e-8, "observed": float(np.max(np.abs(torch_grid - pred_grids))), "notes": "torch differentiable generator parity"},
            {"check_name": "torch_depth_map_max_abs_error_vs_numpy", "pass": float(np.max(np.abs(torch_map - pred_maps))) <= 2.0e-8, "observed": float(np.max(np.abs(torch_map - pred_maps))), "notes": "torch differentiable map parity"},
            {"check_name": "exact_piao_rbc_false", "pass": True, "observed": False, "notes": "RBC-style approximation, not exact Piao reproduction"},
        ]
    )
    return rows
