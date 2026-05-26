"""Train a center-anchored COMSOL polygon inverse model."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from comsol_center_anchored_polygon_inverse_models import CenterAnchoredPolygonInverseNet
from comsol_polygon_rasterizer import mask_iou_dice, rasterize_polygon_components


def _flatten_signals(signals: np.ndarray) -> np.ndarray:
    if signals.ndim == 3:
        return signals.reshape(signals.shape[0], -1).astype(np.float32)
    if signals.ndim == 2:
        return signals.astype(np.float32)
    raise ValueError(f"signals must have shape [B,C,L] or [B,L], got {signals.shape}")


def _zscore_per_sample(signals: np.ndarray) -> np.ndarray:
    mean = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((signals - mean) / std).astype(np.float32)


def _string_list(values: np.ndarray) -> list[str]:
    return [str(item) for item in values.tolist()]


def load_dataset(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        signals = _zscore_per_sample(_flatten_signals(data["signals"]))
        return {
            "signals": signals,
            "masks": data["masks"].astype(np.float32),
            "x": data["x"].astype(np.float32),
            "y": data["y"].astype(np.float32),
        }


def load_targets(path: str | Path, train_type_vocab: list[str] | None = None) -> dict:
    with np.load(path, allow_pickle=True) as data:
        vertices = data["polygon_vertices_norm"].astype(np.float32)
        vertex_mask = data["polygon_vertex_mask"].astype(np.float32)
        presence = data["presence_targets"].astype(np.float32)
        type_targets = data["type_targets"].astype(np.int64)
        type_vocab = _string_list(data["type_vocab"])
        sample_indices = data["sample_indices"].astype(np.int64) if "sample_indices" in data else np.arange(vertices.shape[0])
        component_counts = data["component_counts"].astype(np.int64)
        vertex_ordering = str(data["vertex_ordering"])
        center_x_bin = data["center_x_bin_targets"].astype(np.int64)
        center_y_bin = data["center_y_bin_targets"].astype(np.int64)
        center_offset = data["center_offset_targets"].astype(np.float32)
        center_targets = data["center_targets_norm"].astype(np.float32)
        local_vertices = data["local_vertices_grid"].astype(np.float32)
        x_centers = data["center_bin_x_centers"].astype(np.float32)
        y_centers = data["center_bin_y_centers"].astype(np.float32)
        bin_width_x = float(data["center_bin_width_x"])
        bin_width_y = float(data["center_bin_width_y"])
        grid_dx = float(data["grid_dx"])
        grid_dy = float(data["grid_dy"])
        center_bin_size_cells = int(data["center_bin_size_cells"])
    if train_type_vocab is not None and type_vocab != train_type_vocab:
        mapping = {name: idx for idx, name in enumerate(train_type_vocab)}
        remapped = np.full_like(type_targets, -1)
        for old_idx, name in enumerate(type_vocab):
            if name not in mapping:
                raise ValueError(f"Target type {name!r} is absent from train type_vocab.")
            remapped[type_targets == old_idx] = mapping[name]
        type_targets = remapped
        type_vocab = list(train_type_vocab)
    if vertices.ndim != 4 or vertices.shape[-1] != 2:
        raise ValueError(f"polygon_vertices_norm must have shape [N,K,V,2], got {vertices.shape}")
    if vertices.shape[1] != 3 or vertices.shape[2] != 4:
        raise ValueError("Center-anchored polygon route expects max_components=3 and max_vertices=4.")
    if vertex_ordering != "clockwise_top_left":
        raise ValueError(f"Unsupported vertex_ordering: {vertex_ordering}")
    if vertex_mask.shape != vertices.shape[:3]:
        raise ValueError("polygon_vertex_mask shape does not match polygon vertices.")
    if presence.shape != vertices.shape[:2]:
        raise ValueError("presence_targets shape does not match polygon vertices.")
    for name, arr in [
        ("center_x_bin_targets", center_x_bin),
        ("center_y_bin_targets", center_y_bin),
        ("center_offset_targets", center_offset),
        ("center_targets_norm", center_targets),
    ]:
        if arr.shape[:2] != presence.shape:
            raise ValueError(f"{name} shape does not match presence_targets.")
    if local_vertices.shape != vertices.shape:
        raise ValueError("local_vertices_grid shape does not match polygon_vertices_norm.")
    expected_counts = (presence > 0.5).sum(axis=1).astype(np.int64)
    if not np.array_equal(component_counts, expected_counts):
        raise ValueError("component_counts must match presence_targets per sample.")
    present = presence > 0.5
    if present.any():
        if np.max(np.abs(center_offset[present])) > 0.5001:
            raise ValueError("center_offset_targets must stay inside [-0.5,0.5] for present components.")
        if np.any(center_x_bin[present] < 0) or np.any(center_x_bin[present] >= len(x_centers)):
            raise ValueError("center_x_bin_targets out of range.")
        if np.any(center_y_bin[present] < 0) or np.any(center_y_bin[present] >= len(y_centers)):
            raise ValueError("center_y_bin_targets out of range.")
    return {
        "vertices": vertices,
        "vertex_mask": vertex_mask,
        "presence": presence,
        "type_targets": type_targets,
        "type_vocab": type_vocab,
        "sample_indices": sample_indices,
        "component_counts": component_counts,
        "center_x_bin": center_x_bin,
        "center_y_bin": center_y_bin,
        "center_offset": center_offset,
        "center_targets": center_targets,
        "local_vertices": local_vertices,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "bin_width_x": bin_width_x,
        "bin_width_y": bin_width_y,
        "grid_dx": grid_dx,
        "grid_dy": grid_dy,
        "center_bin_size_cells": center_bin_size_cells,
    }


def build_tensors(dataset: dict, targets: dict, device: torch.device) -> dict:
    if dataset["signals"].shape[0] != targets["presence"].shape[0]:
        raise ValueError("Dataset and center-anchored target sample counts do not match.")
    return {
        "signals": torch.from_numpy(dataset["signals"]).to(device),
        "masks": dataset["masks"],
        "x": dataset["x"],
        "y": dataset["y"],
        "vertices": torch.from_numpy(targets["vertices"]).to(device),
        "vertex_mask": torch.from_numpy(targets["vertex_mask"]).to(device),
        "presence": torch.from_numpy(targets["presence"]).to(device),
        "type_targets": torch.from_numpy(targets["type_targets"]).to(device),
        "center_x_bin": torch.from_numpy(targets["center_x_bin"]).to(device),
        "center_y_bin": torch.from_numpy(targets["center_y_bin"]).to(device),
        "center_offset": torch.from_numpy(targets["center_offset"]).to(device),
        "center_targets": torch.from_numpy(targets["center_targets"]).to(device),
        "local_vertices": torch.from_numpy(targets["local_vertices"]).to(device),
        "x_centers": torch.from_numpy(targets["x_centers"]).to(device),
        "y_centers": torch.from_numpy(targets["y_centers"]).to(device),
        "bin_width_x": torch.tensor(targets["bin_width_x"], dtype=torch.float32, device=device),
        "bin_width_y": torch.tensor(targets["bin_width_y"], dtype=torch.float32, device=device),
        "grid_dx": torch.tensor(targets["grid_dx"], dtype=torch.float32, device=device),
        "grid_dy": torch.tensor(targets["grid_dy"], dtype=torch.float32, device=device),
        "sample_indices": targets["sample_indices"],
    }


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def resolve_local_shape_bounds(train_targets: dict, args) -> None:
    """Resolve bounded local output limits from train targets only."""
    if args.local_shape_output_mode == "raw":
        args.local_shape_bound_x_grid = float(args.local_shape_fixed_bound_x_grid)
        args.local_shape_bound_y_grid = float(args.local_shape_fixed_bound_y_grid)
        return
    if args.local_shape_bound_mode == "fixed_grid":
        args.local_shape_bound_x_grid = float(args.local_shape_fixed_bound_x_grid)
        args.local_shape_bound_y_grid = float(args.local_shape_fixed_bound_y_grid)
        return
    if args.local_shape_bound_mode != "train_stats":
        raise ValueError(f"Unknown local shape bound mode: {args.local_shape_bound_mode}")
    valid = (train_targets["presence"][:, :, None] > 0.5) & (train_targets["vertex_mask"] > 0.5)
    if not valid.any():
        args.local_shape_bound_x_grid = float(args.local_shape_fixed_bound_x_grid)
        args.local_shape_bound_y_grid = float(args.local_shape_fixed_bound_y_grid)
        return
    local_vertices = train_targets["local_vertices"]
    max_abs_x = float(np.max(np.abs(local_vertices[..., 0][valid])))
    max_abs_y = float(np.max(np.abs(local_vertices[..., 1][valid])))
    args.local_shape_bound_x_grid = max(max_abs_x * args.local_shape_train_stats_margin, 1.0e-6)
    args.local_shape_bound_y_grid = max(max_abs_y * args.local_shape_train_stats_margin, 1.0e-6)


def _effective_local_vertices(out: dict, args) -> torch.Tensor:
    raw_local = out["local_vertices_grid"]
    if args.local_shape_output_mode == "raw":
        return raw_local
    if args.local_shape_output_mode != "bounded_tanh":
        raise ValueError(f"Unknown local shape output mode: {args.local_shape_output_mode}")
    bounds = torch.tensor(
        [args.local_shape_bound_x_grid, args.local_shape_bound_y_grid],
        dtype=raw_local.dtype,
        device=raw_local.device,
    )
    return torch.tanh(raw_local) * bounds


def soft_decoded_centers(out: dict, tensors: dict) -> torch.Tensor:
    """Differentiable center decode from bin distributions and offsets."""
    x_probs = F.softmax(out["center_x_bin_logits"], dim=-1)
    y_probs = F.softmax(out["center_y_bin_logits"], dim=-1)
    center_x = x_probs @ tensors["x_centers"].to(out["center_x_bin_logits"].dtype)
    center_y = y_probs @ tensors["y_centers"].to(out["center_y_bin_logits"].dtype)
    center_x = center_x + out["center_offset"][..., 0] * tensors["bin_width_x"].to(out["center_offset"].dtype)
    center_y = center_y + out["center_offset"][..., 1] * tensors["bin_width_y"].to(out["center_offset"].dtype)
    return torch.stack([center_x, center_y], dim=-1)


def soft_decoded_vertices(out: dict, tensors: dict, args) -> torch.Tensor:
    """Differentiable vertex decode using soft centers and effective local vertices."""
    centers = soft_decoded_centers(out, tensors)
    effective_local = _effective_local_vertices(out, args)
    vertices = torch.zeros_like(effective_local)
    vertices[..., 0] = centers[..., 0, None] + effective_local[..., 0] * tensors["grid_dx"]
    vertices[..., 1] = centers[..., 1, None] + effective_local[..., 1] * tensors["grid_dy"]
    return vertices


def _grid_scaled_delta(pred: torch.Tensor, target: torch.Tensor, tensors: dict) -> torch.Tensor:
    scale = torch.stack([tensors["grid_dx"], tensors["grid_dy"]]).to(pred.dtype)
    return (pred - target) / scale.clamp_min(1.0e-12)


def _teacher_center_context(tensors: dict) -> torch.Tensor:
    center = tensors["center_targets"]
    return torch.stack(
        [
            center[..., 0] / tensors["grid_dx"].clamp_min(1.0e-12),
            center[..., 1] / tensors["grid_dy"].clamp_min(1.0e-12),
        ],
        dim=-1,
    )


def _teacher_forcing_weight(step: int | None, args) -> float:
    if args.joint_center_shape_mode == "none" or step is None:
        return 0.0
    if args.joint_center_teacher_forcing_steps <= 1:
        return float(args.joint_center_teacher_forcing_end)
    progress = min(1.0, max(0.0, float(step - 1) / float(args.joint_center_teacher_forcing_steps - 1)))
    return float(
        args.joint_center_teacher_forcing_start
        + progress * (args.joint_center_teacher_forcing_end - args.joint_center_teacher_forcing_start)
    )


def forward_model(model: CenterAnchoredPolygonInverseNet, tensors: dict, args, step: int | None = None) -> dict:
    if args.joint_center_shape_mode == "none":
        return model(tensors["signals"])
    teacher_weight = _teacher_forcing_weight(step, args)
    teacher_context = _teacher_center_context(tensors) if teacher_weight > 0.0 else None
    return model(
        tensors["signals"],
        x_centers=tensors["x_centers"],
        y_centers=tensors["y_centers"],
        bin_width_x=tensors["bin_width_x"],
        bin_width_y=tensors["bin_width_y"],
        grid_dx=tensors["grid_dx"],
        grid_dy=tensors["grid_dy"],
        teacher_center_context=teacher_context,
        teacher_forcing_weight=teacher_weight,
    )


def _local_shape_stats(out: dict, effective_local: torch.Tensor, tensors: dict, args) -> dict[str, float]:
    raw_local = out["local_vertices_grid"]
    valid = (tensors["presence"].unsqueeze(-1) * tensors["vertex_mask"]).unsqueeze(-1).expand_as(raw_local) > 0.5
    if not valid.any():
        return {
            "local_shape_raw_abs_max": 0.0,
            "local_shape_effective_abs_max": 0.0,
            "local_shape_saturation_frac": 0.0,
        }
    raw_abs_max = float(raw_local.detach().abs()[valid].max().cpu())
    effective_abs_max = float(effective_local.detach().abs()[valid].max().cpu())
    saturation_frac = 0.0
    if args.local_shape_output_mode == "bounded_tanh":
        bounds = torch.tensor(
            [args.local_shape_bound_x_grid, args.local_shape_bound_y_grid],
            dtype=effective_local.dtype,
            device=effective_local.device,
        ).view(1, 1, 1, 2)
        ratio = effective_local.detach().abs() / bounds.clamp_min(1.0e-8)
        saturation_frac = float((ratio[valid] >= 0.98).to(torch.float32).mean().cpu())
    return {
        "local_shape_raw_abs_max": raw_abs_max,
        "local_shape_effective_abs_max": effective_abs_max,
        "local_shape_saturation_frac": saturation_frac,
    }


def decode_vertices(out: dict, tensors: dict, args=None) -> tuple[torch.Tensor, torch.Tensor]:
    x_bin = out["center_x_bin_logits"].argmax(dim=-1)
    y_bin = out["center_y_bin_logits"].argmax(dim=-1)
    center_x = tensors["x_centers"][x_bin] + out["center_offset"][..., 0] * tensors["bin_width_x"]
    center_y = tensors["y_centers"][y_bin] + out["center_offset"][..., 1] * tensors["bin_width_y"]
    effective_local = out["local_vertices_grid"] if args is None else _effective_local_vertices(out, args)
    vertices = torch.zeros_like(out["local_vertices_grid"])
    vertices[..., 0] = center_x[..., None] + effective_local[..., 0] * tensors["grid_dx"]
    vertices[..., 1] = center_y[..., None] + effective_local[..., 1] * tensors["grid_dy"]
    centers = torch.stack([center_x, center_y], dim=-1)
    return vertices, centers


def _center_consistency_loss(out: dict, tensors: dict, args, presence: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
    if args.lambda_center_consistency == 0.0 or args.center_consistency_mode == "none" or not presence.any():
        return out["center_x_bin_logits"].sum() * 0.0
    if args.center_consistency_mode == "soft_decoded_center":
        pred_delta = _grid_scaled_delta(soft_decoded_centers(out, tensors), tensors["center_targets"], tensors)
        target_delta = torch.zeros_like(pred_delta)
        return _masked_smooth_l1(
            pred_delta,
            target_delta,
            presence.unsqueeze(-1).expand_as(pred_delta),
            args.center_consistency_smoothl1_beta,
        )
    if args.center_consistency_mode == "soft_decoded_vertex":
        pred_delta = _grid_scaled_delta(soft_decoded_vertices(out, tensors, args), tensors["vertices"], tensors)
        target_delta = torch.zeros_like(pred_delta)
        valid_vertex = (presence.unsqueeze(-1) * vertex_mask).unsqueeze(-1).expand_as(pred_delta)
        return _masked_smooth_l1(pred_delta, target_delta, valid_vertex, args.center_consistency_smoothl1_beta)
    raise ValueError(f"Unknown center consistency mode: {args.center_consistency_mode}")


def _masked_polygon_centroids(vertices: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
    weights = vertex_mask.unsqueeze(-1)
    denom = weights.sum(dim=2).clamp_min(1.0)
    return (vertices * weights).sum(dim=2) / denom


def _derived_boxes(vertices: torch.Tensor, vertex_mask: torch.Tensor) -> torch.Tensor:
    valid = vertex_mask.unsqueeze(-1) > 0.5
    mins = torch.where(valid, vertices, torch.full_like(vertices, 1.0e6)).amin(dim=2)
    maxs = torch.where(valid, vertices, torch.full_like(vertices, -1.0e6)).amax(dim=2)
    return torch.cat([mins, maxs], dim=-1)


def _polygon_areas_grid(vertices: torch.Tensor, tensors: dict) -> torch.Tensor:
    scale = torch.tensor([1.0 / float(tensors["grid_dx"]), 1.0 / float(tensors["grid_dy"])], device=vertices.device, dtype=vertices.dtype)
    vertices_grid = vertices * scale
    x = vertices_grid[..., 0]
    y = vertices_grid[..., 1]
    return 0.5 * torch.abs((x * torch.roll(y, -1, dims=2) - y * torch.roll(x, -1, dims=2)).sum(dim=2))


def _edge_lengths_grid(vertices: torch.Tensor, tensors: dict) -> torch.Tensor:
    scale = torch.tensor([1.0 / float(tensors["grid_dx"]), 1.0 / float(tensors["grid_dy"])], device=vertices.device, dtype=vertices.dtype)
    edge_vectors = (torch.roll(vertices, -1, dims=2) - vertices) * scale
    return torch.linalg.norm(edge_vectors, dim=-1)


def _present_quad_mask(presence: torch.Tensor, vertex_mask: torch.Tensor, loss_name: str) -> torch.Tensor:
    quad_mask = presence * (vertex_mask.sum(dim=2) == vertex_mask.shape[2]).to(presence.dtype)
    if presence.sum() > quad_mask.sum():
        raise ValueError(f"{loss_name} currently requires four valid vertices for every present component.")
    return quad_mask


def _soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _neighbor_soft_targets(targets: torch.Tensor, num_bins: int, smoothing: float) -> torch.Tensor:
    if smoothing < 0.0 or smoothing >= 1.0:
        raise ValueError("--center-y-bin-neighbor-smoothing must be in [0, 1).")
    probs = torch.zeros((targets.numel(), num_bins), device=targets.device, dtype=torch.float32)
    if num_bins <= 1:
        probs[:, 0] = 1.0
        return probs
    probs.scatter_(1, targets[:, None], 1.0 - smoothing)
    if smoothing == 0.0:
        return probs
    left = targets - 1
    right = targets + 1
    left_valid = left >= 0
    right_valid = right < num_bins
    neighbor_count = left_valid.to(torch.float32) + right_valid.to(torch.float32)
    row_idx = torch.arange(targets.numel(), device=targets.device)
    if left_valid.any():
        probs[row_idx[left_valid], left[left_valid]] += smoothing / neighbor_count[left_valid]
    if right_valid.any():
        probs[row_idx[right_valid], right[right_valid]] += smoothing / neighbor_count[right_valid]
    return probs


def _distance_soft_targets(targets: torch.Tensor, num_bins: int, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("--center-y-bin-distance-sigma must be positive.")
    positions = torch.arange(num_bins, device=targets.device, dtype=torch.float32)
    dist = positions[None, :] - targets.to(torch.float32)[:, None]
    probs = torch.exp(-0.5 * (dist / sigma) ** 2)
    return probs / probs.sum(dim=1, keepdim=True)


def _center_y_bin_extra_loss(out: dict, tensors: dict, present: torch.Tensor, args) -> torch.Tensor:
    if not present.any() or args.lambda_center_y_bin_extra == 0.0 or args.center_y_bin_extra_loss_mode == "none":
        return out["center_y_bin_logits"].sum() * 0.0
    logits = out["center_y_bin_logits"][present]
    targets = tensors["center_y_bin"][present]
    num_bins = logits.shape[-1]
    if args.center_y_bin_extra_loss_mode == "neighbor_soft_ce":
        target_probs = _neighbor_soft_targets(targets, num_bins, args.center_y_bin_neighbor_smoothing)
    elif args.center_y_bin_extra_loss_mode == "distance_soft_ce":
        target_probs = _distance_soft_targets(targets, num_bins, args.center_y_bin_distance_sigma)
    else:
        raise ValueError(f"Unknown y-bin extra loss mode: {args.center_y_bin_extra_loss_mode}")
    return _soft_cross_entropy(logits, target_probs)


def compute_losses(out: dict, tensors: dict, args) -> tuple[torch.Tensor, dict[str, float]]:
    presence = tensors["presence"]
    present = presence > 0.5
    vertex_mask = tensors["vertex_mask"]
    presence_loss = F.binary_cross_entropy_with_logits(out["presence_logits"], presence)
    type_loss = (
        F.cross_entropy(out["type_logits"][present], tensors["type_targets"][present])
        if present.any()
        else out["type_logits"].sum() * 0.0
    )
    center_x_loss = (
        F.cross_entropy(out["center_x_bin_logits"][present], tensors["center_x_bin"][present])
        if present.any()
        else out["center_x_bin_logits"].sum() * 0.0
    )
    center_y_loss = (
        F.cross_entropy(out["center_y_bin_logits"][present], tensors["center_y_bin"][present])
        if present.any()
        else out["center_y_bin_logits"].sum() * 0.0
    )
    center_y_extra_loss = _center_y_bin_extra_loss(out, tensors, present, args)
    center_bin_loss = 0.5 * (center_x_loss + center_y_loss)
    center_offset_loss = (
        F.smooth_l1_loss(out["center_offset"][present], tensors["center_offset"][present], beta=args.center_offset_smoothl1_beta)
        if present.any()
        else out["center_offset"].sum() * 0.0
    )
    effective_local = _effective_local_vertices(out, args)
    valid_vertex = (presence.unsqueeze(-1) * vertex_mask).unsqueeze(-1).expand_as(tensors["local_vertices"])
    local_vertex_loss = _masked_smooth_l1(
        effective_local,
        tensors["local_vertices"],
        valid_vertex,
        args.local_vertex_smoothl1_beta,
    )
    decoded_vertices, decoded_centers = decode_vertices(out, tensors, args)
    center_aux_loss = decoded_vertices.sum() * 0.0
    if args.lambda_center_aux != 0.0 and present.any():
        center_aux_loss = _masked_smooth_l1(
            decoded_centers,
            tensors["center_targets"],
            presence.unsqueeze(-1).expand_as(decoded_centers),
            args.center_offset_smoothl1_beta,
        )
    decoded_center_aux_loss = decoded_vertices.sum() * 0.0
    if args.lambda_decoded_center_aux != 0.0 and present.any():
        pred_delta = _grid_scaled_delta(decoded_centers, tensors["center_targets"], tensors)
        decoded_center_aux_loss = _masked_smooth_l1(
            pred_delta,
            torch.zeros_like(pred_delta),
            presence.unsqueeze(-1).expand_as(pred_delta),
            args.center_centroid_aux_smoothl1_beta,
        )
    polygon_centroid_aux_loss = decoded_vertices.sum() * 0.0
    if args.lambda_polygon_centroid_aux != 0.0 and present.any():
        pred_centroid = _masked_polygon_centroids(decoded_vertices, vertex_mask)
        true_centroid = _masked_polygon_centroids(tensors["vertices"], vertex_mask)
        pred_delta = _grid_scaled_delta(pred_centroid, true_centroid, tensors)
        polygon_centroid_aux_loss = _masked_smooth_l1(
            pred_delta,
            torch.zeros_like(pred_delta),
            presence.unsqueeze(-1).expand_as(pred_delta),
            args.center_centroid_aux_smoothl1_beta,
        )
    box_loss = decoded_vertices.sum() * 0.0
    if args.lambda_box_aux != 0.0 and present.any():
        pred_box = _derived_boxes(decoded_vertices, vertex_mask)
        true_box = _derived_boxes(tensors["vertices"], vertex_mask)
        box_loss = _masked_smooth_l1(pred_box, true_box, presence.unsqueeze(-1).expand_as(pred_box), args.center_offset_smoothl1_beta)
    area_loss = decoded_vertices.sum() * 0.0
    if args.lambda_area_aux != 0.0 and present.any():
        pred_area = _polygon_areas_grid(decoded_vertices, tensors)
        true_area = _polygon_areas_grid(tensors["vertices"], tensors)
        area_loss = _masked_smooth_l1(
            pred_area,
            true_area,
            _present_quad_mask(presence, vertex_mask, "area auxiliary loss"),
            args.local_vertex_smoothl1_beta,
        )
    edge_loss = decoded_vertices.sum() * 0.0
    if args.lambda_edge_aux != 0.0 and present.any():
        pred_edges = _edge_lengths_grid(decoded_vertices, tensors)
        true_edges = _edge_lengths_grid(tensors["vertices"], tensors)
        edge_mask = _present_quad_mask(presence, vertex_mask, "edge auxiliary loss").unsqueeze(-1).expand_as(pred_edges)
        edge_loss = _masked_smooth_l1(pred_edges, true_edges, edge_mask, args.local_vertex_smoothl1_beta)
    center_consistency_loss = _center_consistency_loss(out, tensors, args, presence, vertex_mask)
    total = (
        args.lambda_presence * presence_loss
        + args.lambda_type * type_loss
        + args.lambda_center_bin * center_bin_loss
        + args.lambda_center_offset * center_offset_loss
        + args.lambda_local_vertex * local_vertex_loss
        + args.lambda_center_aux * center_aux_loss
        + args.lambda_box_aux * box_loss
        + args.lambda_area_aux * area_loss
        + args.lambda_edge_aux * edge_loss
        + args.lambda_center_y_bin_extra * center_y_extra_loss
        + args.lambda_center_consistency * center_consistency_loss
        + args.lambda_decoded_center_aux * decoded_center_aux_loss
        + args.lambda_polygon_centroid_aux * polygon_centroid_aux_loss
    )
    values = {
        "loss": float(total.detach().cpu()),
        "presence_loss": float(presence_loss.detach().cpu()),
        "type_loss": float(type_loss.detach().cpu()),
        "center_bin_loss": float(center_bin_loss.detach().cpu()),
        "center_x_bin_loss": float(center_x_loss.detach().cpu()),
        "center_y_bin_loss": float(center_y_loss.detach().cpu()),
        "center_y_bin_extra_loss": float(center_y_extra_loss.detach().cpu()),
        "center_offset_loss": float(center_offset_loss.detach().cpu()),
        "local_vertex_loss": float(local_vertex_loss.detach().cpu()),
        "center_aux_loss": float(center_aux_loss.detach().cpu()),
        "box_aux_loss": float(box_loss.detach().cpu()),
        "area_aux_loss": float(area_loss.detach().cpu()),
        "edge_aux_loss": float(edge_loss.detach().cpu()),
        "center_consistency_loss": float(center_consistency_loss.detach().cpu()),
        "decoded_center_aux_loss": float(decoded_center_aux_loss.detach().cpu()),
        "polygon_centroid_aux_loss": float(polygon_centroid_aux_loss.detach().cpu()),
        "weighted_presence_loss": float((args.lambda_presence * presence_loss).detach().cpu()),
        "weighted_type_loss": float((args.lambda_type * type_loss).detach().cpu()),
        "weighted_center_bin_loss": float((args.lambda_center_bin * center_bin_loss).detach().cpu()),
        "weighted_center_y_bin_extra_loss": float((args.lambda_center_y_bin_extra * center_y_extra_loss).detach().cpu()),
        "weighted_center_offset_loss": float((args.lambda_center_offset * center_offset_loss).detach().cpu()),
        "weighted_local_vertex_loss": float((args.lambda_local_vertex * local_vertex_loss).detach().cpu()),
        "weighted_center_consistency_loss": float((args.lambda_center_consistency * center_consistency_loss).detach().cpu()),
        "weighted_decoded_center_aux_loss": float((args.lambda_decoded_center_aux * decoded_center_aux_loss).detach().cpu()),
        "weighted_polygon_centroid_aux_loss": float((args.lambda_polygon_centroid_aux * polygon_centroid_aux_loss).detach().cpu()),
        **_local_shape_stats(out, effective_local, tensors, args),
    }
    return total, values


def _signed_area(vertices: np.ndarray) -> np.ndarray:
    x = vertices[..., 0]
    y = vertices[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=2) - np.roll(x, -1, axis=2) * y, axis=2)


def _polygon_metrics(out: dict, tensors: dict, args) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    presence_prob = torch.sigmoid(out["presence_logits"]).detach().cpu().numpy()
    pred_presence = (presence_prob >= args.presence_threshold).astype(np.float32)
    effective_local_t = _effective_local_vertices(out, args)
    pred_vertices_t, pred_centers_t = decode_vertices(out, tensors, args)
    soft_centers_t = soft_decoded_centers(out, tensors)
    pred_vertices = pred_vertices_t.detach().cpu().numpy().astype(np.float32)
    pred_centers = pred_centers_t.detach().cpu().numpy().astype(np.float32)
    soft_centers = soft_centers_t.detach().cpu().numpy().astype(np.float32)
    true_presence = tensors["presence"].detach().cpu().numpy().astype(np.float32)
    true_vertices = tensors["vertices"].detach().cpu().numpy().astype(np.float32)
    true_centers = tensors["center_targets"].detach().cpu().numpy().astype(np.float32)
    true_vertex_mask = tensors["vertex_mask"].detach().cpu().numpy().astype(np.float32)
    pred_vertex_mask = np.ones_like(true_vertex_mask, dtype=np.float32)
    pred_masks = rasterize_polygon_components(pred_vertices, pred_vertex_mask, pred_presence, tensors["x"], tensors["y"])
    ious, dices = mask_iou_dice(pred_masks, tensors["masks"])
    valid_coord = (true_presence[:, :, None, None] * true_vertex_mask[:, :, :, None]) > 0.5
    decoded_vertex_mae = float(np.abs(pred_vertices - true_vertices)[valid_coord.repeat(2, axis=-1)].mean()) if valid_coord.any() else 0.0
    local_mae = 0.0
    if valid_coord.any():
        pred_local = effective_local_t.detach().cpu().numpy().astype(np.float32)
        true_local = tensors["local_vertices"].detach().cpu().numpy().astype(np.float32)
        local_mae = float(np.abs(pred_local - true_local)[valid_coord.repeat(2, axis=-1)].mean())
    pred_types = out["type_logits"].detach().cpu().numpy().argmax(axis=-1)
    true_types = tensors["type_targets"].detach().cpu().numpy()
    present = true_presence > 0.5
    present_type_acc = float((pred_types[present] == true_types[present]).mean()) if present.any() else 1.0
    presence_acc = float((pred_presence == true_presence).mean())
    pred_x_bin = out["center_x_bin_logits"].detach().cpu().numpy().argmax(axis=-1)
    pred_y_bin = out["center_y_bin_logits"].detach().cpu().numpy().argmax(axis=-1)
    true_x_bin = tensors["center_x_bin"].detach().cpu().numpy()
    true_y_bin = tensors["center_y_bin"].detach().cpu().numpy()
    center_x_bin_acc = float((pred_x_bin[present] == true_x_bin[present]).mean()) if present.any() else 1.0
    center_y_bin_acc = float((pred_y_bin[present] == true_y_bin[present]).mean()) if present.any() else 1.0
    if present.any():
        x_bin_abs = np.abs(pred_x_bin[present] - true_x_bin[present])
        y_bin_abs = np.abs(pred_y_bin[present] - true_y_bin[present])
        center_x_bin_abs_error = float(x_bin_abs.mean())
        center_y_bin_abs_error = float(y_bin_abs.mean())
        center_x_bin_within1_acc = float((x_bin_abs <= 1).mean())
        center_y_bin_within1_acc = float((y_bin_abs <= 1).mean())
    else:
        center_x_bin_abs_error = 0.0
        center_y_bin_abs_error = 0.0
        center_x_bin_within1_acc = 1.0
        center_y_bin_within1_acc = 1.0
    pred_offset = out["center_offset"].detach().cpu().numpy()
    true_offset = tensors["center_offset"].detach().cpu().numpy()
    center_offset_mae = float(np.abs(pred_offset[present] - true_offset[present]).mean()) if present.any() else 0.0
    x_logits = out["center_x_bin_logits"].detach().cpu()
    y_logits = out["center_y_bin_logits"].detach().cpu()
    x_probs = torch.softmax(x_logits, dim=-1).numpy()
    y_probs = torch.softmax(y_logits, dim=-1).numpy()
    x_top2 = np.sort(x_probs, axis=-1)[..., -2:]
    y_top2 = np.sort(y_probs, axis=-1)[..., -2:]
    center_scale = np.array(
        [
            float(tensors["grid_dx"].detach().cpu()),
            float(tensors["grid_dy"].detach().cpu()),
        ],
        dtype=np.float32,
    )
    hard_center_delta = (pred_centers - true_centers) / center_scale
    soft_center_delta = (soft_centers - true_centers) / center_scale
    if present.any():
        hard_center_mae_grid = float(np.abs(hard_center_delta[present]).mean())
        soft_center_mae_grid = float(np.abs(soft_center_delta[present]).mean())
        hard_center_l2_grid = float(np.linalg.norm(hard_center_delta[present], axis=-1).mean())
        soft_center_l2_grid = float(np.linalg.norm(soft_center_delta[present], axis=-1).mean())
        center_x_bin_prob_margin = float((x_top2[..., 1] - x_top2[..., 0])[present].mean())
        center_y_bin_prob_margin = float((y_top2[..., 1] - y_top2[..., 0])[present].mean())
    else:
        hard_center_mae_grid = 0.0
        soft_center_mae_grid = 0.0
        hard_center_l2_grid = 0.0
        soft_center_l2_grid = 0.0
        center_x_bin_prob_margin = 1.0
        center_y_bin_prob_margin = 1.0
    pred_area_sign = _signed_area(pred_vertices)
    true_area_sign = _signed_area(true_vertices)
    signed_flip = present & (pred_area_sign * true_area_sign < 0.0)
    x_min, x_max = float(np.min(tensors["x"])), float(np.max(tensors["x"]))
    y_min, y_max = float(np.min(tensors["y"])), float(np.max(tensors["y"]))
    out_grid = (
        (pred_vertices[..., 0] < x_min)
        | (pred_vertices[..., 0] > x_max)
        | (pred_vertices[..., 1] < y_min)
        | (pred_vertices[..., 1] > y_max)
    ) & (true_presence[..., None] > 0.5) & (true_vertex_mask > 0.5)
    metrics = {
        "polygon_mask_iou": float(np.mean(ious)),
        "polygon_mask_iou_min": float(np.min(ious)),
        "polygon_dice": float(np.mean(dices)),
        "polygon_dice_min": float(np.min(dices)),
        "presence_acc": presence_acc,
        "present_type_acc": present_type_acc,
        "decoded_vertex_mae": decoded_vertex_mae,
        "local_vertex_mae_grid": local_mae,
        "center_x_bin_acc": center_x_bin_acc,
        "center_y_bin_acc": center_y_bin_acc,
        "center_x_bin_abs_error": center_x_bin_abs_error,
        "center_y_bin_abs_error": center_y_bin_abs_error,
        "center_x_bin_within1_acc": center_x_bin_within1_acc,
        "center_y_bin_within1_acc": center_y_bin_within1_acc,
        "center_offset_mae": center_offset_mae,
        "hard_decoded_center_mae_grid": hard_center_mae_grid,
        "soft_expected_center_mae_grid": soft_center_mae_grid,
        "hard_decoded_center_l2_grid": hard_center_l2_grid,
        "soft_expected_center_l2_grid": soft_center_l2_grid,
        "center_x_bin_prob_margin": center_x_bin_prob_margin,
        "center_y_bin_prob_margin": center_y_bin_prob_margin,
        "signed_area_flip_count": int(signed_flip.sum()),
        "out_of_grid_vertex_count": int(out_grid.sum()),
        "zero_iou_count": int((ious <= 0.0).sum()),
        "pred_area_mean": float(pred_masks.sum(axis=(1, 2)).mean()),
        "target_area_mean": float((tensors["masks"] > 0.5).sum(axis=(1, 2)).mean()),
        **_local_shape_stats(out, effective_local_t, tensors, args),
    }
    return metrics, pred_masks, pred_presence, pred_vertices, pred_area_sign


def evaluate(model: CenterAnchoredPolygonInverseNet, tensors: dict, split: str, args) -> dict:
    model.eval()
    with torch.no_grad():
        out = forward_model(model, tensors, args)
        _loss, loss_values = compute_losses(out, tensors, args)
        metric_values, _pred_masks, _pred_presence, _pred_vertices, _sign = _polygon_metrics(out, tensors, args)
    return {"split": split, **loss_values, **metric_values}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_predictions(
    output_dir: Path,
    split: str,
    model: CenterAnchoredPolygonInverseNet,
    tensors: dict,
    targets: dict,
    args,
) -> None:
    model.eval()
    with torch.no_grad():
        out = forward_model(model, tensors, args)
        _metrics, pred_masks, pred_presence, pred_vertices, pred_area_sign = _polygon_metrics(out, tensors, args)
    presence_prob = torch.sigmoid(out["presence_logits"]).detach().cpu().numpy()
    pred_types = out["type_logits"].detach().cpu().numpy().argmax(axis=-1)
    true_presence = tensors["presence"].detach().cpu().numpy()
    true_vertices = tensors["vertices"].detach().cpu().numpy()
    true_types = tensors["type_targets"].detach().cpu().numpy()
    vertex_mask = tensors["vertex_mask"].detach().cpu().numpy()
    pred_x_bin = out["center_x_bin_logits"].detach().cpu().numpy().argmax(axis=-1)
    pred_y_bin = out["center_y_bin_logits"].detach().cpu().numpy().argmax(axis=-1)
    true_x_bin = tensors["center_x_bin"].detach().cpu().numpy()
    true_y_bin = tensors["center_y_bin"].detach().cpu().numpy()
    pred_offset = out["center_offset"].detach().cpu().numpy()
    true_offset = tensors["center_offset"].detach().cpu().numpy()
    x_probs = torch.softmax(out["center_x_bin_logits"], dim=-1).detach().cpu().numpy()
    y_probs = torch.softmax(out["center_y_bin_logits"], dim=-1).detach().cpu().numpy()
    x_sorted = np.sort(x_probs, axis=-1)
    y_sorted = np.sort(y_probs, axis=-1)
    soft_centers = soft_decoded_centers(out, tensors).detach().cpu().numpy()
    _pred_vertices_t, hard_centers_t = decode_vertices(out, tensors, args)
    hard_centers = hard_centers_t.detach().cpu().numpy()
    true_centers = tensors["center_targets"].detach().cpu().numpy()
    grid_dx = float(tensors["grid_dx"].detach().cpu())
    grid_dy = float(tensors["grid_dy"].detach().cpu())
    pred_local_raw = out["local_vertices_grid"].detach().cpu().numpy()
    pred_local = _effective_local_vertices(out, args).detach().cpu().numpy()
    true_local = tensors["local_vertices"].detach().cpu().numpy()
    true_area_sign = _signed_area(true_vertices)
    component_rows = []
    for row_idx, sample_index in enumerate(targets["sample_indices"]):
        for slot in range(true_presence.shape[1]):
            valid = vertex_mask[row_idx, slot] > 0.5
            row = {
                "sample_index": int(sample_index),
                "component_slot": slot,
                "presence_true": float(true_presence[row_idx, slot]),
                "presence_prob": float(presence_prob[row_idx, slot]),
                "presence_pred": float(pred_presence[row_idx, slot]),
                "type_true": int(true_types[row_idx, slot]),
                "type_pred": int(pred_types[row_idx, slot]),
                "center_x_bin_true": int(true_x_bin[row_idx, slot]),
                "center_x_bin_pred": int(pred_x_bin[row_idx, slot]),
                "center_y_bin_true": int(true_y_bin[row_idx, slot]),
                "center_y_bin_pred": int(pred_y_bin[row_idx, slot]),
                "center_offset_mae": float(np.abs(pred_offset[row_idx, slot] - true_offset[row_idx, slot]).mean()),
                "center_offset_x_true": float(true_offset[row_idx, slot, 0]),
                "center_offset_x_pred": float(pred_offset[row_idx, slot, 0]),
                "center_offset_y_true": float(true_offset[row_idx, slot, 1]),
                "center_offset_y_pred": float(pred_offset[row_idx, slot, 1]),
                "center_x_bin_prob_top1": float(x_sorted[row_idx, slot, -1]),
                "center_x_bin_prob_top2": float(x_sorted[row_idx, slot, -2]),
                "center_x_bin_prob_margin": float(x_sorted[row_idx, slot, -1] - x_sorted[row_idx, slot, -2]),
                "center_y_bin_prob_top1": float(y_sorted[row_idx, slot, -1]),
                "center_y_bin_prob_top2": float(y_sorted[row_idx, slot, -2]),
                "center_y_bin_prob_margin": float(y_sorted[row_idx, slot, -1] - y_sorted[row_idx, slot, -2]),
                "center_x_true": float(true_centers[row_idx, slot, 0]),
                "center_y_true": float(true_centers[row_idx, slot, 1]),
                "hard_center_x_pred": float(hard_centers[row_idx, slot, 0]),
                "hard_center_y_pred": float(hard_centers[row_idx, slot, 1]),
                "soft_center_x_pred": float(soft_centers[row_idx, slot, 0]),
                "soft_center_y_pred": float(soft_centers[row_idx, slot, 1]),
                "hard_center_x_error_grid": float((hard_centers[row_idx, slot, 0] - true_centers[row_idx, slot, 0]) / grid_dx),
                "hard_center_y_error_grid": float((hard_centers[row_idx, slot, 1] - true_centers[row_idx, slot, 1]) / grid_dy),
                "soft_center_x_error_grid": float((soft_centers[row_idx, slot, 0] - true_centers[row_idx, slot, 0]) / grid_dx),
                "soft_center_y_error_grid": float((soft_centers[row_idx, slot, 1] - true_centers[row_idx, slot, 1]) / grid_dy),
                "signed_area_flip": int(pred_area_sign[row_idx, slot] * true_area_sign[row_idx, slot] < 0.0),
            }
            if valid.any():
                row["decoded_vertex_mae"] = float(np.abs(pred_vertices[row_idx, slot, valid] - true_vertices[row_idx, slot, valid]).mean())
                row["local_vertex_mae_grid"] = float(np.abs(pred_local[row_idx, slot, valid] - true_local[row_idx, slot, valid]).mean())
            else:
                row["decoded_vertex_mae"] = 0.0
                row["local_vertex_mae_grid"] = 0.0
            for vertex_idx in range(pred_vertices.shape[2]):
                row[f"vertex{vertex_idx}_valid"] = float(vertex_mask[row_idx, slot, vertex_idx])
                row[f"pred_x{vertex_idx}"] = float(pred_vertices[row_idx, slot, vertex_idx, 0])
                row[f"pred_y{vertex_idx}"] = float(pred_vertices[row_idx, slot, vertex_idx, 1])
                row[f"true_x{vertex_idx}"] = float(true_vertices[row_idx, slot, vertex_idx, 0])
                row[f"true_y{vertex_idx}"] = float(true_vertices[row_idx, slot, vertex_idx, 1])
                row[f"pred_local_x{vertex_idx}"] = float(pred_local[row_idx, slot, vertex_idx, 0])
                row[f"pred_local_y{vertex_idx}"] = float(pred_local[row_idx, slot, vertex_idx, 1])
                row[f"pred_local_raw_x{vertex_idx}"] = float(pred_local_raw[row_idx, slot, vertex_idx, 0])
                row[f"pred_local_raw_y{vertex_idx}"] = float(pred_local_raw[row_idx, slot, vertex_idx, 1])
                row[f"true_local_x{vertex_idx}"] = float(true_local[row_idx, slot, vertex_idx, 0])
                row[f"true_local_y{vertex_idx}"] = float(true_local[row_idx, slot, vertex_idx, 1])
            component_rows.append(row)
    write_csv(output_dir / f"{split}_center_anchored_polygon_predictions.csv", component_rows)
    ious, dices = mask_iou_dice(pred_masks, tensors["masks"])
    x_min, x_max = float(np.min(tensors["x"])), float(np.max(tensors["x"]))
    y_min, y_max = float(np.min(tensors["y"])), float(np.max(tensors["y"]))
    sample_rows = []
    for row_idx, sample_index in enumerate(targets["sample_indices"]):
        valid_pred_mask = (true_presence[row_idx, :, None] > 0.5) & (vertex_mask[row_idx] > 0.5)
        out_of_grid = 0
        if valid_pred_mask.any():
            points = pred_vertices[row_idx][valid_pred_mask]
            out_of_grid = int(((points[:, 0] < x_min) | (points[:, 0] > x_max) | (points[:, 1] < y_min) | (points[:, 1] > y_max)).sum())
        sample_rows.append(
            {
                "sample_index": int(sample_index),
                "polygon_mask_iou": float(ious[row_idx]),
                "polygon_dice": float(dices[row_idx]),
                "target_area": int((tensors["masks"][row_idx] > 0.5).sum()),
                "pred_area": int(pred_masks[row_idx].sum()),
                "true_component_count": int((true_presence[row_idx] > 0.5).sum()),
                "pred_component_count": int((pred_presence[row_idx] > 0.5).sum()),
                "out_of_grid_vertex_count": out_of_grid,
            }
        )
    write_csv(output_dir / f"{split}_center_anchored_polygon_mask_metrics.csv", sample_rows)


def _append_config(row: dict, args) -> dict:
    config = {
        "steps": args.steps,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "max_components": args.max_components,
        "max_vertices": args.max_vertices,
        "lambda_presence": args.lambda_presence,
        "lambda_type": args.lambda_type,
        "lambda_center_bin": args.lambda_center_bin,
        "lambda_center_offset": args.lambda_center_offset,
        "lambda_local_vertex": args.lambda_local_vertex,
        "lambda_center_aux": args.lambda_center_aux,
        "lambda_box_aux": args.lambda_box_aux,
        "lambda_area_aux": args.lambda_area_aux,
        "lambda_edge_aux": args.lambda_edge_aux,
        "center_consistency_mode": args.center_consistency_mode,
        "lambda_center_consistency": args.lambda_center_consistency,
        "center_consistency_smoothl1_beta": args.center_consistency_smoothl1_beta,
        "lambda_decoded_center_aux": args.lambda_decoded_center_aux,
        "lambda_polygon_centroid_aux": args.lambda_polygon_centroid_aux,
        "center_centroid_aux_smoothl1_beta": args.center_centroid_aux_smoothl1_beta,
        "center_y_bin_extra_loss_mode": args.center_y_bin_extra_loss_mode,
        "lambda_center_y_bin_extra": args.lambda_center_y_bin_extra,
        "center_y_bin_neighbor_smoothing": args.center_y_bin_neighbor_smoothing,
        "center_y_bin_distance_sigma": args.center_y_bin_distance_sigma,
        "local_shape_output_mode": args.local_shape_output_mode,
        "local_shape_bound_mode": args.local_shape_bound_mode,
        "local_shape_bound_x_grid": args.local_shape_bound_x_grid,
        "local_shape_bound_y_grid": args.local_shape_bound_y_grid,
        "local_shape_train_stats_margin": args.local_shape_train_stats_margin,
        "local_shape_conditioning_mode": args.local_shape_conditioning_mode,
        "local_shape_conditioning_dim": args.local_shape_conditioning_dim,
        "joint_center_shape_mode": args.joint_center_shape_mode,
        "joint_center_teacher_forcing_start": args.joint_center_teacher_forcing_start,
        "joint_center_teacher_forcing_end": args.joint_center_teacher_forcing_end,
        "joint_center_teacher_forcing_steps": args.joint_center_teacher_forcing_steps,
        "local_vertex_smoothl1_beta": args.local_vertex_smoothl1_beta,
        "center_offset_smoothl1_beta": args.center_offset_smoothl1_beta,
        "presence_threshold": args.presence_threshold,
        "seed": args.seed,
    }
    return {**row, **config}


def write_run_summary(output_dir: Path, args, train_row: dict, val_row: dict, test_row: dict, train_targets: dict) -> None:
    lines = [
        "# COMSOL center-anchored polygon inverse run summary",
        "",
        "## Config",
        "",
        f"- steps: `{args.steps}`",
        f"- lr: `{args.lr}`",
        f"- hidden_dim: `{args.hidden_dim}`",
        f"- latent_dim: `{args.latent_dim}`",
        f"- max_components: `{args.max_components}`",
        f"- max_vertices: `{args.max_vertices}`",
        f"- type_vocab: `{', '.join(train_targets['type_vocab'])}`",
        f"- center_bin_size_cells: `{train_targets['center_bin_size_cells']}`",
        f"- center_x_bins: `{len(train_targets['x_centers'])}`",
        f"- center_y_bins: `{len(train_targets['y_centers'])}`",
        f"- lambda_presence: `{args.lambda_presence}`",
        f"- lambda_type: `{args.lambda_type}`",
        f"- lambda_center_bin: `{args.lambda_center_bin}`",
        f"- lambda_center_offset: `{args.lambda_center_offset}`",
        f"- lambda_local_vertex: `{args.lambda_local_vertex}`",
        f"- lambda_center_aux: `{args.lambda_center_aux}`",
        f"- lambda_box_aux: `{args.lambda_box_aux}`",
        f"- lambda_area_aux: `{args.lambda_area_aux}`",
        f"- lambda_edge_aux: `{args.lambda_edge_aux}`",
        f"- center_consistency_mode: `{args.center_consistency_mode}`",
        f"- lambda_center_consistency: `{args.lambda_center_consistency}`",
        f"- center_consistency_smoothl1_beta: `{args.center_consistency_smoothl1_beta}`",
        f"- lambda_decoded_center_aux: `{args.lambda_decoded_center_aux}`",
        f"- lambda_polygon_centroid_aux: `{args.lambda_polygon_centroid_aux}`",
        f"- center_centroid_aux_smoothl1_beta: `{args.center_centroid_aux_smoothl1_beta}`",
        f"- center_y_bin_extra_loss_mode: `{args.center_y_bin_extra_loss_mode}`",
        f"- lambda_center_y_bin_extra: `{args.lambda_center_y_bin_extra}`",
        f"- center_y_bin_neighbor_smoothing: `{args.center_y_bin_neighbor_smoothing}`",
        f"- center_y_bin_distance_sigma: `{args.center_y_bin_distance_sigma}`",
        f"- local_shape_output_mode: `{args.local_shape_output_mode}`",
        f"- local_shape_bound_mode: `{args.local_shape_bound_mode}`",
        f"- local_shape_bound_x_grid: `{args.local_shape_bound_x_grid}`",
        f"- local_shape_bound_y_grid: `{args.local_shape_bound_y_grid}`",
        f"- local_shape_train_stats_margin: `{args.local_shape_train_stats_margin}`",
        f"- local_shape_conditioning_mode: `{args.local_shape_conditioning_mode}`",
        f"- local_shape_conditioning_dim: `{args.local_shape_conditioning_dim}`",
        f"- joint_center_shape_mode: `{args.joint_center_shape_mode}`",
        f"- joint_center_teacher_forcing_start: `{args.joint_center_teacher_forcing_start}`",
        f"- joint_center_teacher_forcing_end: `{args.joint_center_teacher_forcing_end}`",
        f"- joint_center_teacher_forcing_steps: `{args.joint_center_teacher_forcing_steps}`",
        f"- export_predictions: `{args.export_predictions}`",
        f"- seed: `{args.seed}`",
        "",
        "## Final Metrics",
        "",
        "| split | polygon_mask_iou | polygon_mask_iou_min | decoded_vertex_mae | local_vertex_mae_grid | hard_center_mae_grid | soft_center_mae_grid | presence_acc | type_acc | x_bin_acc | y_bin_acc | y_bin_abs_err | y_bin_within1 | out_of_grid | signed_flip | local_sat |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [train_row, val_row, test_row]:
        lines.append(
            f"| {row['split']} | `{row['polygon_mask_iou']:.6f}` | `{row['polygon_mask_iou_min']:.6f}` | "
            f"`{row['decoded_vertex_mae']:.6e}` | `{row['local_vertex_mae_grid']:.6e}` | "
            f"`{row['hard_decoded_center_mae_grid']:.6f}` | `{row['soft_expected_center_mae_grid']:.6f}` | "
            f"`{row['presence_acc']:.6f}` | `{row['present_type_acc']:.6f}` | "
            f"`{row['center_x_bin_acc']:.6f}` | `{row['center_y_bin_acc']:.6f}` | `{row['center_y_bin_abs_error']:.6f}` | "
            f"`{row['center_y_bin_within1_acc']:.6f}` | `{int(row['out_of_grid_vertex_count'])}` | `{int(row['signed_area_flip_count'])}` | `{row['local_shape_saturation_frac']:.6f}` |"
        )
    lines.extend(["", "No checkpoint or model weights are saved by this runner."])
    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz")
    parser.add_argument("--train-targets")
    parser.add_argument("--val-npz")
    parser.add_argument("--val-targets")
    parser.add_argument("--test-npz")
    parser.add_argument("--test-targets")
    parser.add_argument("--output-dir")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--max-vertices", type=int, default=4)
    parser.add_argument("--lambda-presence", type=float, default=1.0)
    parser.add_argument("--lambda-type", type=float, default=1.0)
    parser.add_argument("--lambda-center-bin", type=float, default=1.0)
    parser.add_argument("--lambda-center-offset", type=float, default=10.0)
    parser.add_argument("--lambda-local-vertex", type=float, default=1.0)
    parser.add_argument("--lambda-center-aux", type=float, default=0.0)
    parser.add_argument("--lambda-box-aux", type=float, default=0.0)
    parser.add_argument("--lambda-area-aux", type=float, default=0.0)
    parser.add_argument("--lambda-edge-aux", type=float, default=0.0)
    parser.add_argument(
        "--center-consistency-mode",
        choices=["none", "soft_decoded_center", "soft_decoded_vertex"],
        default="none",
    )
    parser.add_argument("--lambda-center-consistency", type=float, default=0.0)
    parser.add_argument("--center-consistency-smoothl1-beta", type=float, default=0.1)
    parser.add_argument("--lambda-decoded-center-aux", type=float, default=0.0)
    parser.add_argument("--lambda-polygon-centroid-aux", type=float, default=0.0)
    parser.add_argument("--center-centroid-aux-smoothl1-beta", type=float, default=0.01)
    parser.add_argument("--center-y-bin-extra-loss-mode", choices=["none", "neighbor_soft_ce", "distance_soft_ce"], default="none")
    parser.add_argument("--lambda-center-y-bin-extra", type=float, default=0.0)
    parser.add_argument("--center-y-bin-neighbor-smoothing", type=float, default=0.0)
    parser.add_argument("--center-y-bin-distance-sigma", type=float, default=0.75)
    parser.add_argument("--local-shape-output-mode", choices=["raw", "bounded_tanh"], default="raw")
    parser.add_argument("--local-shape-bound-mode", choices=["fixed_grid", "train_stats"], default="fixed_grid")
    parser.add_argument("--local-shape-fixed-bound-x-grid", type=float, default=24.0)
    parser.add_argument("--local-shape-fixed-bound-y-grid", type=float, default=8.0)
    parser.add_argument("--local-shape-train-stats-margin", type=float, default=1.25)
    parser.add_argument(
        "--local-shape-conditioning-mode",
        choices=["none", "center_bin", "center_bin_slot", "center_bin_slot_type"],
        default="none",
    )
    parser.add_argument("--local-shape-conditioning-dim", type=int, default=16)
    parser.add_argument("--joint-center-shape-mode", choices=["none", "soft_center_scheduled"], default="none")
    parser.add_argument("--joint-center-teacher-forcing-start", type=float, default=1.0)
    parser.add_argument("--joint-center-teacher-forcing-end", type=float, default=0.0)
    parser.add_argument("--joint-center-teacher-forcing-steps", type=int, default=20000)
    parser.add_argument("--local-vertex-smoothl1-beta", type=float, default=0.1)
    parser.add_argument("--center-offset-smoothl1-beta", type=float, default=0.01)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--history-interval", type=int, default=100)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    required = [args.train_npz, args.train_targets, args.val_npz, args.val_targets, args.test_npz, args.test_targets, args.output_dir]
    if not all(required):
        parser.print_help()
        return 0
    if args.max_components != 3 or args.max_vertices != 4:
        raise ValueError("Center-anchored polygon inverse runner supports max_components=3 and max_vertices=4 only.")
    if args.center_y_bin_neighbor_smoothing < 0.0 or args.center_y_bin_neighbor_smoothing >= 1.0:
        raise ValueError("--center-y-bin-neighbor-smoothing must be in [0, 1).")
    if args.center_y_bin_distance_sigma <= 0.0:
        raise ValueError("--center-y-bin-distance-sigma must be positive.")
    if args.center_y_bin_extra_loss_mode == "none" and args.lambda_center_y_bin_extra != 0.0:
        raise ValueError("--lambda-center-y-bin-extra requires a non-none --center-y-bin-extra-loss-mode.")
    if args.center_consistency_mode == "none" and args.lambda_center_consistency != 0.0:
        raise ValueError("--lambda-center-consistency requires a non-none --center-consistency-mode.")
    if args.center_consistency_mode != "none" and args.lambda_center_consistency < 0.0:
        raise ValueError("--lambda-center-consistency must be non-negative.")
    if args.center_consistency_smoothl1_beta <= 0.0:
        raise ValueError("--center-consistency-smoothl1-beta must be positive.")
    if args.lambda_decoded_center_aux < 0.0 or args.lambda_polygon_centroid_aux < 0.0:
        raise ValueError("--lambda-decoded-center-aux and --lambda-polygon-centroid-aux must be non-negative.")
    if args.center_centroid_aux_smoothl1_beta <= 0.0:
        raise ValueError("--center-centroid-aux-smoothl1-beta must be positive.")
    if args.local_shape_fixed_bound_x_grid <= 0.0 or args.local_shape_fixed_bound_y_grid <= 0.0:
        raise ValueError("--local-shape-fixed-bound-x-grid and --local-shape-fixed-bound-y-grid must be positive.")
    if args.local_shape_train_stats_margin <= 0.0:
        raise ValueError("--local-shape-train-stats-margin must be positive.")
    if args.local_shape_conditioning_dim <= 0:
        raise ValueError("--local-shape-conditioning-dim must be positive.")
    if args.joint_center_shape_mode != "none" and args.local_shape_conditioning_mode != "none":
        raise ValueError("--joint-center-shape-mode and --local-shape-conditioning-mode cannot both be enabled.")
    if not 0.0 <= args.joint_center_teacher_forcing_start <= 1.0:
        raise ValueError("--joint-center-teacher-forcing-start must be in [0,1].")
    if not 0.0 <= args.joint_center_teacher_forcing_end <= 1.0:
        raise ValueError("--joint-center-teacher-forcing-end must be in [0,1].")
    if args.joint_center_teacher_forcing_steps <= 0:
        raise ValueError("--joint-center-teacher-forcing-steps must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train_dataset = load_dataset(args.train_npz)
    val_dataset = load_dataset(args.val_npz)
    test_dataset = load_dataset(args.test_npz)
    train_targets = load_targets(args.train_targets)
    val_targets = load_targets(args.val_targets, train_type_vocab=train_targets["type_vocab"])
    test_targets = load_targets(args.test_targets, train_type_vocab=train_targets["type_vocab"])
    resolve_local_shape_bounds(train_targets, args)
    for split, targets in [("val", val_targets), ("test", test_targets)]:
        if len(targets["x_centers"]) != len(train_targets["x_centers"]) or len(targets["y_centers"]) != len(train_targets["y_centers"]):
            raise ValueError(f"{split} center bin counts do not match train targets.")
    train_tensors = build_tensors(train_dataset, train_targets, device)
    val_tensors = build_tensors(val_dataset, val_targets, device)
    test_tensors = build_tensors(test_dataset, test_targets, device)
    model = CenterAnchoredPolygonInverseNet(
        signal_len=train_dataset["signals"].shape[1],
        center_x_bins=len(train_targets["x_centers"]),
        center_y_bins=len(train_targets["y_centers"]),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_components=args.max_components,
        max_vertices=args.max_vertices,
        num_types=len(train_targets["type_vocab"]),
        num_layers=args.num_layers,
        local_shape_conditioning_mode=args.local_shape_conditioning_mode,
        local_shape_conditioning_dim=args.local_shape_conditioning_dim,
        joint_center_shape_mode=args.joint_center_shape_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        teacher_weight = _teacher_forcing_weight(step, args)
        out = forward_model(model, train_tensors, args, step=step)
        loss, values = compute_losses(out, train_tensors, args)
        values["joint_center_teacher_forcing_weight"] = teacher_weight
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        optimizer.step()
        if step == 1 or step == args.steps or (args.history_interval > 0 and step % args.history_interval == 0):
            history.append({"step": step, **values})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if history:
        write_csv(output_dir / "training_history.csv", history)
    train_row = _append_config(evaluate(model, train_tensors, "train", args), args)
    val_row = _append_config(evaluate(model, val_tensors, "val", args), args)
    test_row = _append_config(evaluate(model, test_tensors, "test", args), args)
    write_csv(output_dir / "metrics.csv", [train_row])
    write_csv(output_dir / "eval_metrics.csv", [val_row])
    write_csv(output_dir / "test_metrics.csv", [test_row])
    if args.export_predictions:
        export_predictions(output_dir, "train", model, train_tensors, train_targets, args)
        export_predictions(output_dir, "val", model, val_tensors, val_targets, args)
        export_predictions(output_dir, "test", model, test_tensors, test_targets, args)
    write_run_summary(output_dir, args, train_row, val_row, test_row, train_targets)
    print(f"Saved center-anchored polygon inverse metrics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
