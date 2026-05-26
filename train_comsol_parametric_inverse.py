"""Train a small COMSOL parametric inverse model from Bz signals to geometry parameters."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from comsol_parametric_inverse_models import ParametricInverseNet
from comsol_parametric_rasterizer import continuous_to_raw, mask_iou_dice, rasterize_components
from comsol_differentiable_parametric_rasterizer import (
    soft_bce_loss,
    soft_dice_loss,
    soft_dice_score,
    soft_iou_score,
    soft_rasterize_components,
)


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


def load_dataset(npz_path: str | Path) -> dict:
    path = Path(npz_path)
    with np.load(path, allow_pickle=True) as data:
        signals = _zscore_per_sample(_flatten_signals(data["signals"]))
        masks = data["masks"].astype(np.float32) if "masks" in data else None
        mu_maps = data["mu_maps"].astype(np.float32) if "mu_maps" in data else None
        x = data["x"].astype(np.float32) if "x" in data else None
        y = data["y"].astype(np.float32) if "y" in data else None
    return {"signals": signals, "masks": masks, "mu_maps": mu_maps, "x": x, "y": y}


def load_feature_matrix(path: str | Path, expected_count: int, expected_indices: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    feature_path = Path(path)
    with np.load(feature_path, allow_pickle=True) as data:
        if "features" not in data:
            raise ValueError(f"{feature_path} does not contain features.")
        features = data["features"].astype(np.float32)
        feature_names = [str(x) for x in data["feature_names"]] if "feature_names" in data else [f"feature_{i}" for i in range(features.shape[1])]
        sample_indices = data["sample_indices"].astype(np.int64) if "sample_indices" in data else None
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N,F], got {features.shape}")
    if features.shape[0] != expected_count:
        raise ValueError(f"Feature sample count {features.shape[0]} does not match expected {expected_count}.")
    if expected_indices is not None and sample_indices is not None and not np.array_equal(sample_indices, expected_indices):
        raise ValueError("Feature sample_indices do not match target sample_indices.")
    if not np.isfinite(features).all():
        raise ValueError(f"{feature_path} contains NaN or Inf features.")
    return features, feature_names


def compute_feature_norm(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)


def load_targets(path: str | Path, train_type_vocab: list[str] | None = None) -> dict:
    with np.load(path, allow_pickle=True) as data:
        continuous = data["continuous_targets"].astype(np.float32)
        continuous_raw = (
            data["continuous_targets_raw"].astype(np.float32)
            if "continuous_targets_raw" in data
            else continuous.copy()
        )
        continuous_unscaled = (
            data["continuous_targets_unscaled"].astype(np.float32)
            if "continuous_targets_unscaled" in data
            else continuous.copy()
        )
        presence = data["presence_targets"].astype(np.float32)
        type_targets = data["type_targets"].astype(np.int64)
        sample_indices = data["sample_indices"].astype(np.int64) if "sample_indices" in data else np.arange(continuous.shape[0])
        schema = [str(x) for x in data["target_schema"]]
        raw_schema = [str(x) for x in data["raw_target_schema"]] if "raw_target_schema" in data else list(schema)
        type_vocab = [str(x) for x in data["type_vocab"]]
        angle_encoding = str(data["angle_encoding"]) if "angle_encoding" in data else "raw"
        normalized = bool(data["continuous_targets_normalized"]) if "continuous_targets_normalized" in data else False
        stats_mean = data["continuous_targets_mean"].astype(np.float32) if "continuous_targets_mean" in data else None
        stats_std = data["continuous_targets_std"].astype(np.float32) if "continuous_targets_std" in data else None

    if train_type_vocab is not None and type_vocab != train_type_vocab:
        mapping = {name: i for i, name in enumerate(train_type_vocab)}
        remapped = np.full_like(type_targets, -1)
        for old_index, name in enumerate(type_vocab):
            if name not in mapping:
                raise ValueError(f"Target type {name!r} is absent from train type_vocab.")
            remapped[type_targets == old_index] = mapping[name]
        type_targets = remapped
        type_vocab = list(train_type_vocab)

    return {
        "continuous": continuous,
        "continuous_raw": continuous_raw,
        "continuous_unscaled": continuous_unscaled,
        "presence": presence,
        "type_targets": type_targets,
        "sample_indices": sample_indices,
        "target_schema": schema,
        "raw_target_schema": raw_schema,
        "type_vocab": type_vocab,
        "angle_encoding": angle_encoding,
        "continuous_targets_normalized": normalized,
        "continuous_targets_mean": stats_mean,
        "continuous_targets_std": stats_std,
    }


def compute_continuous_norm(targets: dict) -> tuple[np.ndarray, np.ndarray]:
    if targets.get("continuous_targets_normalized"):
        mean = targets.get("continuous_targets_mean")
        std = targets.get("continuous_targets_std")
        if mean is None or std is None:
            raise ValueError("Normalized targets must include continuous_targets_mean/std.")
        return mean.astype(np.float32), std.astype(np.float32)
    present = targets["presence"] > 0.5
    values = targets["continuous"][present]
    if values.size == 0:
        raise ValueError("No present components in train targets.")
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def normalize_continuous(continuous: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((continuous - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def build_center_bin_info(dataset: dict, args) -> dict | None:
    if getattr(args, "center_representation", "continuous") == "continuous":
        return None
    if args.center_representation != "bin_offset":
        raise ValueError(f"Unsupported center_representation: {args.center_representation}")
    if args.center_bin_size_cells <= 0:
        raise ValueError("center-bin-size-cells must be positive.")
    x = dataset["x"]
    y = dataset["y"]
    dx = _mean_grid_spacing(x, "x")
    dy = _mean_grid_spacing(y, "y")
    x_min = float(np.asarray(x, dtype=np.float64)[0])
    y_min = float(np.asarray(y, dtype=np.float64)[0])
    x_max = float(np.asarray(x, dtype=np.float64)[-1])
    y_max = float(np.asarray(y, dtype=np.float64)[-1])
    bin_width_x = float(args.center_bin_size_cells * dx)
    bin_width_y = float(args.center_bin_size_cells * dy)
    center_x_bins = int(np.ceil((x_max - x_min) / bin_width_x))
    center_y_bins = int(np.ceil((y_max - y_min) / bin_width_y))
    if center_x_bins <= 0 or center_y_bins <= 0:
        raise ValueError("center bin configuration produced no bins.")
    x_centers = (x_min + (np.arange(center_x_bins, dtype=np.float32) + 0.5) * bin_width_x).astype(np.float32)
    y_centers = (y_min + (np.arange(center_y_bins, dtype=np.float32) + 0.5) * bin_width_y).astype(np.float32)
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "dx": dx,
        "dy": dy,
        "bin_size_cells": int(args.center_bin_size_cells),
        "bin_width_x": bin_width_x,
        "bin_width_y": bin_width_y,
        "center_x_bins": center_x_bins,
        "center_y_bins": center_y_bins,
        "x_centers": x_centers,
        "y_centers": y_centers,
    }


def build_center_bin_targets(targets: dict, center_bin_info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    schema = targets["target_schema"]
    for name in ["center_x", "center_y"]:
        if name not in schema:
            raise ValueError(f"center bin targets require {name} in target_schema.")
    cx = schema.index("center_x")
    cy = schema.index("center_y")
    centers = targets["continuous_unscaled"]
    presence = targets["presence"] > 0.5
    center_x = centers[:, :, cx].astype(np.float64)
    center_y = centers[:, :, cy].astype(np.float64)
    if presence.any():
        if np.any(center_x[presence] < center_bin_info["x_min"]) or np.any(center_x[presence] > center_bin_info["x_max"]):
            raise ValueError("center_x target is outside the x grid range.")
        if np.any(center_y[presence] < center_bin_info["y_min"]) or np.any(center_y[presence] > center_bin_info["y_max"]):
            raise ValueError("center_y target is outside the y grid range.")
    x_bin = np.floor((center_x - center_bin_info["x_min"]) / center_bin_info["bin_width_x"]).astype(np.int64)
    y_bin = np.floor((center_y - center_bin_info["y_min"]) / center_bin_info["bin_width_y"]).astype(np.int64)
    x_bin = np.clip(x_bin, 0, center_bin_info["center_x_bins"] - 1)
    y_bin = np.clip(y_bin, 0, center_bin_info["center_y_bins"] - 1)
    x_center = center_bin_info["x_centers"][x_bin]
    y_center = center_bin_info["y_centers"][y_bin]
    x_offset = ((center_x - x_center) / center_bin_info["bin_width_x"]).astype(np.float32)
    y_offset = ((center_y - y_center) / center_bin_info["bin_width_y"]).astype(np.float32)
    if presence.any():
        max_abs_offset = float(max(np.max(np.abs(x_offset[presence])), np.max(np.abs(y_offset[presence]))))
        if max_abs_offset > 0.5001:
            raise ValueError(f"center bin offset target outside [-0.5, 0.5]: {max_abs_offset:.6f}")
    offsets = np.stack([x_offset, y_offset], axis=-1).astype(np.float32)
    return x_bin.astype(np.int64), y_bin.astype(np.int64), offsets


def build_tensors(
    dataset: dict,
    targets: dict,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    features: np.ndarray | None = None,
    center_bin_info: dict | None = None,
) -> dict:
    if dataset["signals"].shape[0] != targets["presence"].shape[0]:
        raise ValueError("signals and targets sample counts do not match.")
    if features is not None and features.shape[0] != dataset["signals"].shape[0]:
        raise ValueError("features and signals sample counts do not match.")
    continuous_norm = (
        targets["continuous"].astype(np.float32)
        if targets.get("continuous_targets_normalized")
        else normalize_continuous(targets["continuous"], mean, std)
    )
    tensors = {
        "signals": torch.from_numpy(dataset["signals"]).to(device),
        "features": None if features is None else torch.from_numpy(features.astype(np.float32)).to(device),
        "presence": torch.from_numpy(targets["presence"]).to(device),
        "type_targets": torch.from_numpy(targets["type_targets"]).to(device),
        "continuous_norm": torch.from_numpy(continuous_norm).to(device),
        "continuous_raw": targets["continuous_raw"],
        "continuous_unscaled": targets["continuous_unscaled"],
        "continuous_mean": torch.from_numpy(mean.astype(np.float32)).to(device),
        "continuous_std": torch.from_numpy(std.astype(np.float32)).to(device),
        "sample_indices": targets["sample_indices"],
        "target_schema": targets["target_schema"],
        "type_vocab": targets["type_vocab"],
        "masks": dataset["masks"],
        "mu_maps": dataset["mu_maps"],
        "x": dataset["x"],
        "y": dataset["y"],
    }
    if center_bin_info is not None:
        x_bin, y_bin, offsets = build_center_bin_targets(targets, center_bin_info)
        tensors.update(
            {
                "center_x_bin_targets": torch.from_numpy(x_bin).to(device),
                "center_y_bin_targets": torch.from_numpy(y_bin).to(device),
                "center_offset_targets": torch.from_numpy(offsets).to(device),
                "center_bin_x_centers": torch.from_numpy(center_bin_info["x_centers"]).to(device),
                "center_bin_y_centers": torch.from_numpy(center_bin_info["y_centers"]).to(device),
                "center_bin_width_x": torch.tensor(center_bin_info["bin_width_x"], dtype=torch.float32, device=device),
                "center_bin_width_y": torch.tensor(center_bin_info["bin_width_y"], dtype=torch.float32, device=device),
                "center_bin_info": center_bin_info,
            }
        )
    return tensors


def _group_indices(schema: list[str]) -> dict[str, list[int]]:
    groups = {"center": [], "axis": [], "depth": [], "rotation": [], "other": []}
    for index, name in enumerate(schema):
        if name in {"center_x", "center_y"}:
            groups["center"].append(index)
        elif name in {"axis_x", "axis_y", "width", "height"}:
            groups["axis"].append(index)
        elif name in {"depth_or_shape_param", "depth", "shape_param"}:
            groups["depth"].append(index)
        elif name in {"rotation_angle", "rotation_sin", "rotation_cos"}:
            groups["rotation"].append(index)
        else:
            groups["other"].append(index)
    return groups


def _continuous_loss(pred: torch.Tensor, target: torch.Tensor, schema: list[str], args) -> tuple[torch.Tensor, dict[str, float]]:
    skip_center = getattr(args, "center_representation", "continuous") == "bin_offset"
    base_loss = F.smooth_l1_loss(pred, target)
    group_weights = {
        "center": args.lambda_center,
        "axis": args.lambda_axis,
        "depth": args.lambda_depth,
        "rotation": args.lambda_rotation,
        "other": 1.0,
    }
    if not skip_center and all(abs(value - 1.0) < 1e-12 for value in group_weights.values()):
        return base_loss, {
            "continuous_loss": float(base_loss.detach().cpu()),
            "center_loss": float("nan"),
            "axis_loss": float("nan"),
            "depth_loss": float("nan"),
            "rotation_loss": float("nan"),
        }
    groups = _group_indices(schema)
    weighted_terms = []
    weight_total = 0.0
    group_losses: dict[str, float] = {}
    for group, indices in groups.items():
        if skip_center and group == "center":
            group_losses[f"{group}_loss"] = float("nan")
            continue
        if not indices:
            group_losses[f"{group}_loss"] = float("nan")
            continue
        group_loss = F.smooth_l1_loss(pred[:, indices], target[:, indices])
        weight = group_weights[group]
        weighted_terms.append(weight * group_loss)
        weight_total += weight
        group_losses[f"{group}_loss"] = float(group_loss.detach().cpu())
    if not weighted_terms or weight_total <= 0:
        return base_loss, {"continuous_loss": float(base_loss.detach().cpu()), **group_losses}
    loss = torch.stack(weighted_terms).sum() / weight_total
    return loss, {"continuous_loss": float(loss.detach().cpu()), **group_losses}


def _zero_continuous_parts(value: float = 0.0) -> dict[str, float]:
    return {
        "continuous_loss": value,
        "center_loss": float("nan"),
        "axis_loss": float("nan"),
        "depth_loss": float("nan"),
        "rotation_loss": float("nan"),
    }


def _rotation_extra_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    schema: list[str],
    args,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.sum() * 0.0
    if "rotation_angle" in schema:
        idx = schema.index("rotation_angle")
        mode = getattr(args, "rotation_loss_mode", "mse")
        if mode == "mse":
            return F.mse_loss(pred[:, idx], target[:, idx])
        if mode == "circular":
            if mean is None or std is None:
                raise ValueError("circular rotation loss requires continuous mean/std.")
            scale = std[idx].detach().clamp_min(1e-8)
            pred_angle = pred[:, idx] * scale + mean[idx].detach()
            true_angle = target[:, idx] * scale + mean[idx].detach()
            diff = torch.remainder(pred_angle - true_angle + 180.0, 360.0) - 180.0
            return torch.mean((diff / scale) ** 2)
        raise ValueError(f"Unsupported rotation_loss_mode: {mode}")
    sincos_indices = [schema.index(name) for name in ("rotation_sin", "rotation_cos") if name in schema]
    if sincos_indices:
        return F.mse_loss(pred[:, sincos_indices], target[:, sincos_indices])
    return pred.sum() * 0.0


def _mean_grid_spacing(values: np.ndarray | None, name: str) -> float:
    if values is None:
        raise ValueError(f"Center grid loss requires {name} coordinates in the NPZ.")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} coordinates must be a 1D grid with at least two points.")
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        raise ValueError(f"{name} coordinates must be strictly increasing.")
    spacing = float(np.mean(diffs))
    if spacing <= 0 or not np.isfinite(spacing):
        raise ValueError(f"{name} grid spacing is invalid.")
    if float(np.max(np.abs(diffs - spacing))) > max(abs(spacing) * 5e-2, 1e-9):
        raise ValueError(f"{name} grid spacing is not approximately uniform.")
    return spacing


def _decode_center_bin_offset(out: dict, tensors: dict, use_soft_bins: bool) -> torch.Tensor:
    required = ["center_x_bin_logits", "center_y_bin_logits", "center_offset"]
    missing = [name for name in required if name not in out]
    if missing:
        raise ValueError(f"bin_offset center representation requires output keys: {missing}")
    x_centers = tensors.get("center_bin_x_centers")
    y_centers = tensors.get("center_bin_y_centers")
    if x_centers is None or y_centers is None:
        raise ValueError("bin_offset center representation requires center bin tensors.")
    if use_soft_bins:
        x_center = torch.sum(torch.softmax(out["center_x_bin_logits"], dim=-1) * x_centers.view(1, 1, -1), dim=-1)
        y_center = torch.sum(torch.softmax(out["center_y_bin_logits"], dim=-1) * y_centers.view(1, 1, -1), dim=-1)
    else:
        x_index = out["center_x_bin_logits"].argmax(dim=-1)
        y_index = out["center_y_bin_logits"].argmax(dim=-1)
        x_center = x_centers[x_index]
        y_center = y_centers[y_index]
    offset = out["center_offset"]
    x = x_center + offset[:, :, 0] * tensors["center_bin_width_x"]
    y = y_center + offset[:, :, 1] * tensors["center_bin_width_y"]
    return torch.stack([x, y], dim=-1)


def _effective_continuous_unscaled(out: dict, tensors: dict, args, use_soft_bins: bool) -> torch.Tensor:
    continuous = out["continuous"] * tensors["continuous_std"].view(1, 1, -1) + tensors["continuous_mean"].view(1, 1, -1)
    if getattr(args, "center_representation", "continuous") == "continuous":
        return continuous
    if args.center_representation != "bin_offset":
        raise ValueError(f"Unsupported center_representation: {args.center_representation}")
    schema = tensors["target_schema"]
    if "center_x" not in schema or "center_y" not in schema:
        raise ValueError("bin_offset center representation requires center_x and center_y in target_schema.")
    decoded = _decode_center_bin_offset(out, tensors, use_soft_bins=use_soft_bins)
    effective = continuous.clone()
    effective[:, :, schema.index("center_x")] = decoded[:, :, 0]
    effective[:, :, schema.index("center_y")] = decoded[:, :, 1]
    return effective


def _center_extra_losses_unscaled(
    pred_unscaled: torch.Tensor,
    target_unscaled: torch.Tensor,
    schema: list[str],
    args,
    x: np.ndarray | None,
    y: np.ndarray | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = pred_unscaled.sum() * 0.0
    if pred_unscaled.numel() == 0:
        return zero, {
            "center_grid_loss": 0.0,
            "weighted_center_grid_loss": 0.0,
            "center_axis_relative_loss": 0.0,
            "weighted_center_axis_relative_loss": 0.0,
        }
    lambda_grid = getattr(args, "lambda_center_grid", 0.0)
    lambda_axis_relative = getattr(args, "lambda_center_axis_relative", 0.0)
    if lambda_grid == 0.0 and lambda_axis_relative == 0.0:
        return zero, {
            "center_grid_loss": 0.0,
            "weighted_center_grid_loss": 0.0,
            "center_axis_relative_loss": 0.0,
            "weighted_center_axis_relative_loss": 0.0,
        }
    required = ["center_x", "center_y", "axis_x", "axis_y"]
    missing = [name for name in required if name not in schema]
    if missing:
        raise ValueError(f"Center extra loss requires schema fields: {missing}")
    cx = schema.index("center_x")
    cy = schema.index("center_y")
    ax = schema.index("axis_x")
    ay = schema.index("axis_y")
    delta_x = pred_unscaled[:, cx] - target_unscaled[:, cx]
    delta_y = pred_unscaled[:, cy] - target_unscaled[:, cy]

    if lambda_grid != 0.0:
        dx = _mean_grid_spacing(x, "x")
        dy = _mean_grid_spacing(y, "y")
        grid_x = delta_x / dx
        grid_y = delta_y / dy
        center_grid_loss = torch.mean(grid_x.square() + grid_y.square())
    else:
        center_grid_loss = zero

    if lambda_axis_relative != 0.0:
        eps = float(getattr(args, "center_axis_relative_eps", 1e-6))
        axis_x = target_unscaled[:, ax].abs().clamp_min(eps)
        axis_y = target_unscaled[:, ay].abs().clamp_min(eps)
        rel_x = delta_x / axis_x
        rel_y = delta_y / axis_y
        center_axis_relative_loss = F.smooth_l1_loss(rel_x, torch.zeros_like(rel_x)) + F.smooth_l1_loss(
            rel_y, torch.zeros_like(rel_y)
        )
    else:
        center_axis_relative_loss = zero

    weighted_grid = lambda_grid * center_grid_loss
    weighted_axis = lambda_axis_relative * center_axis_relative_loss
    total = weighted_grid + weighted_axis
    return total, {
        "center_grid_loss": float(center_grid_loss.detach().cpu()),
        "weighted_center_grid_loss": float(weighted_grid.detach().cpu()),
        "center_axis_relative_loss": float(center_axis_relative_loss.detach().cpu()),
        "weighted_center_axis_relative_loss": float(weighted_axis.detach().cpu()),
    }


def _center_extra_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    schema: list[str],
    args,
    mean: torch.Tensor,
    std: torch.Tensor,
    x: np.ndarray | None,
    y: np.ndarray | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_unscaled = pred * std.view(1, -1) + mean.view(1, -1)
    target_unscaled = target * std.view(1, -1) + mean.view(1, -1)
    return _center_extra_losses_unscaled(pred_unscaled, target_unscaled, schema, args, x, y)


def _zero_center_bin_parts(value: float = 0.0) -> dict[str, float]:
    return {
        "center_bin_loss": value,
        "weighted_center_bin_loss": value,
        "center_x_bin_loss": value,
        "center_y_bin_loss": value,
        "weighted_center_x_bin_loss": value,
        "weighted_center_y_bin_loss": value,
        "center_bin_slot_weight_mean": value,
        "center_bin_slot_weight_max": value,
        "center_offset_loss": value,
        "weighted_center_offset_loss": value,
    }


def _zero_aux_center_parts(value: float = 0.0) -> dict[str, float]:
    return {
        "aux_center_bin_loss": value,
        "weighted_aux_center_bin_loss": value,
        "aux_center_offset_loss": value,
        "weighted_aux_center_offset_loss": value,
    }


def _parse_center_bin_slot_weights(value: str, max_components: int) -> list[float]:
    if not value:
        return [1.0] * max_components
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != max_components:
        raise ValueError(
            f"center-bin-slot-weights must contain {max_components} comma-separated values, got {len(parts)}."
        )
    weights = [float(part) for part in parts]
    if not all(np.isfinite(weight) and weight > 0 for weight in weights):
        raise ValueError("center-bin-slot-weights must be finite positive values.")
    return weights


def _center_bin_offset_loss(
    out: dict,
    tensors: dict,
    args,
    active: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = out["continuous"].sum() * 0.0
    if getattr(args, "center_representation", "continuous") == "continuous":
        return zero, _zero_center_bin_parts()
    if args.center_representation != "bin_offset":
        raise ValueError(f"Unsupported center_representation: {args.center_representation}")
    if not active.any():
        return zero, _zero_center_bin_parts()
    slot_weights = getattr(args, "_center_bin_slot_weights_resolved", None)
    if slot_weights is None:
        slot_weights = _parse_center_bin_slot_weights(getattr(args, "center_bin_slot_weights", ""), active.shape[1])
    slot_weights_t = torch.as_tensor(slot_weights, dtype=out["continuous"].dtype, device=out["continuous"].device)
    slot_weights_t = slot_weights_t.view(1, -1).expand_as(active)[active]
    weight_denom = slot_weights_t.sum().clamp_min(1e-12)
    x_loss_all = F.cross_entropy(
        out["center_x_bin_logits"].reshape(-1, out["center_x_bin_logits"].shape[-1]),
        tensors["center_x_bin_targets"].reshape(-1),
        reduction="none",
    ).view_as(tensors["presence"])
    y_loss_all = F.cross_entropy(
        out["center_y_bin_logits"].reshape(-1, out["center_y_bin_logits"].shape[-1]),
        tensors["center_y_bin_targets"].reshape(-1),
        reduction="none",
    ).view_as(tensors["presence"])
    x_loss = (x_loss_all[active] * slot_weights_t).sum() / weight_denom
    y_loss = (y_loss_all[active] * slot_weights_t).sum() / weight_denom
    axis_weight_sum = args.center_bin_x_weight + args.center_bin_y_weight
    if axis_weight_sum <= 0:
        raise ValueError("center-bin x/y weights must sum to a positive value.")
    bin_loss = (args.center_bin_x_weight * x_loss + args.center_bin_y_weight * y_loss) / axis_weight_sum
    offset_loss = F.smooth_l1_loss(out["center_offset"][active], tensors["center_offset_targets"][active])
    weighted_bin = args.lambda_center_bin * bin_loss
    weighted_offset = args.lambda_center_offset * offset_loss
    return weighted_bin + weighted_offset, {
        "center_bin_loss": float(bin_loss.detach().cpu()),
        "weighted_center_bin_loss": float(weighted_bin.detach().cpu()),
        "center_x_bin_loss": float(x_loss.detach().cpu()),
        "center_y_bin_loss": float(y_loss.detach().cpu()),
        "weighted_center_x_bin_loss": float((args.lambda_center_bin * args.center_bin_x_weight * x_loss / axis_weight_sum).detach().cpu()),
        "weighted_center_y_bin_loss": float((args.lambda_center_bin * args.center_bin_y_weight * y_loss / axis_weight_sum).detach().cpu()),
        "center_bin_slot_weight_mean": float(slot_weights_t.detach().mean().cpu()),
        "center_bin_slot_weight_max": float(slot_weights_t.detach().max().cpu()),
        "center_offset_loss": float(offset_loss.detach().cpu()),
        "weighted_center_offset_loss": float(weighted_offset.detach().cpu()),
    }


def _aux_center_loss(
    out: dict,
    tensors: dict,
    args,
    active: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = out["continuous"].sum() * 0.0
    if not getattr(args, "aux_center_head", False):
        return zero, _zero_aux_center_parts()
    if getattr(args, "center_representation", "continuous") != "bin_offset":
        raise ValueError("aux-center-head requires center-representation=bin_offset.")
    if not active.any():
        return zero, _zero_aux_center_parts()
    required = [
        "aux_center_x_bin_logits",
        "aux_center_y_bin_logits",
        "aux_center_offset",
    ]
    for key in required:
        if key not in out:
            raise ValueError(f"Missing auxiliary center output: {key}")
    weight_sum = args.aux_center_x_weight + args.aux_center_y_weight
    if weight_sum <= 0:
        raise ValueError("aux center x/y weights must sum to a positive value.")
    x_loss = F.cross_entropy(out["aux_center_x_bin_logits"][active], tensors["center_x_bin_targets"][active])
    y_loss = F.cross_entropy(out["aux_center_y_bin_logits"][active], tensors["center_y_bin_targets"][active])
    bin_loss = (args.aux_center_x_weight * x_loss + args.aux_center_y_weight * y_loss) / weight_sum
    offset_delta = F.smooth_l1_loss(
        out["aux_center_offset"][active],
        tensors["center_offset_targets"][active],
        reduction="none",
    )
    offset_loss = (
        args.aux_center_x_weight * offset_delta[:, 0].mean()
        + args.aux_center_y_weight * offset_delta[:, 1].mean()
    ) / weight_sum
    weighted_bin = args.lambda_aux_center_bin * bin_loss
    weighted_offset = args.lambda_aux_center_offset * offset_loss
    return weighted_bin + weighted_offset, {
        "aux_center_bin_loss": float(bin_loss.detach().cpu()),
        "weighted_aux_center_bin_loss": float(weighted_bin.detach().cpu()),
        "aux_center_offset_loss": float(offset_loss.detach().cpu()),
        "weighted_aux_center_offset_loss": float(weighted_offset.detach().cpu()),
    }


def _fixed_loss_from_output(out: dict, tensors: dict, args, type_weights=None) -> tuple[torch.Tensor, dict[str, float]]:
    presence_loss = F.binary_cross_entropy_with_logits(out["presence_logits"], tensors["presence"])
    active = tensors["presence"] > 0.5
    if active.any():
        type_loss = F.cross_entropy(out["type_logits"][active], tensors["type_targets"][active], weight=type_weights)
        continuous_loss, continuous_parts = _continuous_loss(
            out["continuous"][active],
            tensors["continuous_norm"][active],
            tensors["target_schema"],
            args,
        )
        rotation_extra_loss = _rotation_extra_loss(
            out["continuous"][active],
            tensors["continuous_norm"][active],
            tensors["target_schema"],
            args,
            mean=tensors["continuous_mean"],
            std=tensors["continuous_std"],
        )
        if getattr(args, "center_representation", "continuous") == "bin_offset":
            pred_unscaled = _effective_continuous_unscaled(out, tensors, args, use_soft_bins=True)[active]
            target_unscaled = (
                tensors["continuous_norm"] * tensors["continuous_std"].view(1, 1, -1)
                + tensors["continuous_mean"].view(1, 1, -1)
            )[active]
            center_extra_loss, center_extra_parts = _center_extra_losses_unscaled(
                pred_unscaled,
                target_unscaled,
                tensors["target_schema"],
                args,
                tensors["x"],
                tensors["y"],
            )
        else:
            center_extra_loss, center_extra_parts = _center_extra_losses(
                out["continuous"][active],
                tensors["continuous_norm"][active],
                tensors["target_schema"],
                args,
                tensors["continuous_mean"],
                tensors["continuous_std"],
                tensors["x"],
                tensors["y"],
            )
        center_bin_loss, center_bin_parts = _center_bin_offset_loss(out, tensors, args, active)
        aux_center_loss, aux_center_parts = _aux_center_loss(out, tensors, args, active)
    else:
        type_loss = out["type_logits"].sum() * 0.0
        continuous_loss = out["continuous"].sum() * 0.0
        rotation_extra_loss = out["continuous"].sum() * 0.0
        center_extra_loss = out["continuous"].sum() * 0.0
        center_bin_loss = out["continuous"].sum() * 0.0
        aux_center_loss = out["continuous"].sum() * 0.0
        continuous_parts = _zero_continuous_parts()
        center_extra_parts = {
            "center_grid_loss": 0.0,
            "weighted_center_grid_loss": 0.0,
            "center_axis_relative_loss": 0.0,
            "weighted_center_axis_relative_loss": 0.0,
        }
        center_bin_parts = _zero_center_bin_parts()
        aux_center_parts = _zero_aux_center_parts()
    total = (
        args.lambda_presence * presence_loss
        + args.lambda_type * type_loss
        + args.lambda_continuous * continuous_loss
        + getattr(args, "lambda_type_extra", 0.0) * type_loss
        + getattr(args, "lambda_rotation_extra", 0.0) * rotation_extra_loss
        + center_extra_loss
        + center_bin_loss
        + aux_center_loss
    )
    return total, {
        "total_loss": float(total.detach().cpu()),
        "presence_loss": float(presence_loss.detach().cpu()),
        "type_loss": float(type_loss.detach().cpu()),
        "type_extra_loss": float(type_loss.detach().cpu()),
        "rotation_extra_loss": float(rotation_extra_loss.detach().cpu()),
        "weighted_type_extra_loss": float((getattr(args, "lambda_type_extra", 0.0) * type_loss).detach().cpu()),
        "weighted_rotation_extra_loss": float((getattr(args, "lambda_rotation_extra", 0.0) * rotation_extra_loss).detach().cpu()),
        **continuous_parts,
        **center_extra_parts,
        **center_bin_parts,
        **aux_center_parts,
    }


def _component_loss_for_permutation(
    out: dict,
    tensors: dict,
    args,
    permutation: tuple[int, ...],
    type_weights=None,
) -> torch.Tensor:
    perm_index = torch.as_tensor(permutation, dtype=torch.long, device=tensors["presence"].device)
    presence_target = tensors["presence"][:, perm_index]
    type_target = tensors["type_targets"][:, perm_index]
    continuous_target = tensors["continuous_norm"][:, perm_index]
    presence_loss = F.binary_cross_entropy_with_logits(out["presence_logits"], presence_target, reduction="none").mean(dim=1)
    active = presence_target > 0.5
    type_per_sample = []
    continuous_per_sample = []
    rotation_extra_per_sample = []
    center_extra_per_sample = []
    for sample in range(presence_target.shape[0]):
        sample_active = active[sample]
        if sample_active.any():
            type_loss = F.cross_entropy(
                out["type_logits"][sample, sample_active],
                type_target[sample, sample_active],
                weight=type_weights,
                reduction="mean",
            )
            continuous_loss, _parts = _continuous_loss(
                out["continuous"][sample, sample_active],
                continuous_target[sample, sample_active],
                tensors["target_schema"],
                args,
            )
            rotation_extra_loss = _rotation_extra_loss(
                out["continuous"][sample, sample_active],
                continuous_target[sample, sample_active],
                tensors["target_schema"],
                args,
                mean=tensors["continuous_mean"],
                std=tensors["continuous_std"],
            )
            center_extra_loss, _center_parts = _center_extra_losses(
                out["continuous"][sample, sample_active],
                continuous_target[sample, sample_active],
                tensors["target_schema"],
                args,
                tensors["continuous_mean"],
                tensors["continuous_std"],
                tensors["x"],
                tensors["y"],
            )
        else:
            type_loss = out["type_logits"][sample].sum() * 0.0
            continuous_loss = out["continuous"][sample].sum() * 0.0
            rotation_extra_loss = out["continuous"][sample].sum() * 0.0
            center_extra_loss = out["continuous"][sample].sum() * 0.0
        type_per_sample.append(type_loss)
        continuous_per_sample.append(continuous_loss)
        rotation_extra_per_sample.append(rotation_extra_loss)
        center_extra_per_sample.append(center_extra_loss)
    type_loss_per_sample = torch.stack(type_per_sample)
    continuous_loss_per_sample = torch.stack(continuous_per_sample)
    rotation_extra_loss_per_sample = torch.stack(rotation_extra_per_sample)
    center_extra_loss_per_sample = torch.stack(center_extra_per_sample)
    return (
        args.lambda_presence * presence_loss
        + args.lambda_type * type_loss_per_sample
        + getattr(args, "lambda_type_extra", 0.0) * type_loss_per_sample
        + args.lambda_continuous * continuous_loss_per_sample
        + getattr(args, "lambda_rotation_extra", 0.0) * rotation_extra_loss_per_sample
        + center_extra_loss_per_sample
    )


def _permutation_min_loss_from_output(out: dict, tensors: dict, args, type_weights=None) -> tuple[torch.Tensor, dict[str, float]]:
    max_components = int(tensors["presence"].shape[1])
    if max_components > 6:
        raise ValueError("permutation_min is intended for small max_components.")
    per_perm = []
    for permutation in itertools.permutations(range(max_components)):
        per_perm.append(_component_loss_for_permutation(out, tensors, args, permutation, type_weights=type_weights))
    stacked = torch.stack(per_perm, dim=1)
    best_loss, _best_index = stacked.min(dim=1)
    total = best_loss.mean()
    # Report fixed-order decomposed parts for traceability while optimizing permutation-min loss.
    _fixed_total, fixed_parts = _fixed_loss_from_output(out, tensors, args, type_weights=type_weights)
    fixed_parts["total_loss"] = float(total.detach().cpu())
    fixed_parts["matching_loss"] = float(total.detach().cpu())
    return total, fixed_parts


def _raster_target_tensor(tensors: dict, args, device: torch.device) -> torch.Tensor | None:
    if args.raster_target_source == "masks":
        if tensors["masks"] is None:
            return None
        return torch.from_numpy((tensors["masks"] > 0.5).astype(np.float32)).to(device)
    if args.raster_target_source == "mu_threshold":
        if tensors["mu_maps"] is None:
            return None
        return torch.from_numpy((tensors["mu_maps"] < 500.0).astype(np.float32)).to(device)
    raise ValueError(f"Unsupported raster_target_source: {args.raster_target_source}")


def _has_raster_loss(args) -> bool:
    return args.lambda_raster_bce != 0.0 or args.lambda_raster_dice != 0.0


def _is_raster_loss_active(args, step: int | None) -> bool:
    if not _has_raster_loss(args):
        return False
    if args.raster_loss_start_step <= 0:
        return True
    if step is None:
        return True
    return step >= args.raster_loss_start_step


def _raster_loss_from_output(
    out: dict,
    tensors: dict,
    args,
    raster_loss_active: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = out["continuous"].sum() * 0.0
    if not raster_loss_active:
        return zero, {
            "raster_bce_loss": 0.0,
            "raster_dice_loss": 0.0,
            "raster_soft_iou": float("nan"),
            "raster_soft_dice": float("nan"),
            "raster_loss_active": 0.0,
        }
    if tensors["x"] is None or tensors["y"] is None:
        raise ValueError("Raster loss requires x and y coordinates in the train NPZ.")
    target = _raster_target_tensor(tensors, args, out["continuous"].device)
    if target is None:
        raise ValueError(f"Raster loss target source {args.raster_target_source!r} is absent from the train NPZ.")
    continuous_for_raster = _effective_continuous_unscaled(out, tensors, args, use_soft_bins=True)
    soft_mask = soft_rasterize_components(
        continuous_for_raster,
        out["presence_prob"],
        out["type_logits"],
        tensors["x"],
        tensors["y"],
        tensors["target_schema"],
        tensors["type_vocab"],
        softness_cells=args.raster_softness_cells,
    )
    bce = soft_bce_loss(soft_mask, target)
    dice = soft_dice_loss(soft_mask, target)
    total = args.lambda_raster_bce * bce + args.lambda_raster_dice * dice
    return total, {
        "raster_bce_loss": float(bce.detach().cpu()),
        "raster_dice_loss": float(dice.detach().cpu()),
        "raster_soft_iou": float(soft_iou_score(soft_mask.detach(), target.detach()).cpu()),
        "raster_soft_dice": float(soft_dice_score(soft_mask.detach(), target.detach()).cpu()),
        "raster_loss_active": 1.0,
    }


def compute_loss(model, tensors: dict, args, type_weights=None, step: int | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(tensors["signals"], tensors.get("features"))
    if args.component_matching_mode == "fixed":
        param_total, parts = _fixed_loss_from_output(out, tensors, args, type_weights=type_weights)
    elif args.component_matching_mode == "permutation_min":
        param_total, parts = _permutation_min_loss_from_output(out, tensors, args, type_weights=type_weights)
    else:
        raise ValueError(f"Unsupported component_matching_mode: {args.component_matching_mode}")
    raster_total, raster_parts = _raster_loss_from_output(
        out,
        tensors,
        args,
        raster_loss_active=_is_raster_loss_active(args, step),
    )
    total = param_total + raster_total
    parts["total_loss"] = float(total.detach().cpu())
    parts["parametric_total_loss"] = float(param_total.detach().cpu())
    parts["raster_total_loss"] = float(raster_total.detach().cpu())
    parts.update(raster_parts)
    return total, parts


def _angular_mae_deg(pred_raw: np.ndarray, true_raw: np.ndarray) -> float:
    pred = pred_raw[:, 5]
    true = true_raw[:, 5]
    diff = (pred - true + 180.0) % 360.0 - 180.0
    return float(np.abs(diff).mean())


def _angular_error_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs((pred - true + 180.0) % 360.0 - 180.0)


def _type_sequence(type_ids: np.ndarray, presence: np.ndarray, type_vocab: list[str]) -> str:
    names = []
    for type_id, present in zip(type_ids, presence):
        if present > 0.5 and int(type_id) >= 0:
            names.append(type_vocab[int(type_id)])
    return "|".join(names)


def _best_export_permutation(predicted: dict, sample_pos: int, matching_mode: str) -> tuple[int, ...]:
    components = predicted["presence_true"].shape[1]
    identity = tuple(range(components))
    if matching_mode == "fixed":
        return identity
    if matching_mode != "permutation_min":
        raise ValueError(f"Unsupported component_matching_mode for export: {matching_mode}")
    best_cost = float("inf")
    best_perm = identity
    for permutation in itertools.permutations(range(components)):
        presence_true = predicted["presence_true"][sample_pos, list(permutation)].astype(np.float32)
        presence_prob = predicted["presence_prob"][sample_pos].astype(np.float32)
        cost = float(np.abs(presence_prob - presence_true).mean())
        for pred_slot, true_slot in enumerate(permutation):
            if not predicted["presence_true"][sample_pos, true_slot]:
                continue
            type_mismatch = float(predicted["type_pred"][sample_pos, pred_slot] != predicted["type_true"][sample_pos, true_slot])
            pred_raw = predicted["continuous_pred_raw"][sample_pos, pred_slot]
            true_raw = predicted["continuous_true_raw"][sample_pos, true_slot]
            center_cost = float(np.linalg.norm(pred_raw[[0, 1]] - true_raw[[0, 1]]) / 0.01)
            axis_cost = float(np.linalg.norm(pred_raw[[2, 3]] - true_raw[[2, 3]]) / 0.01)
            depth_cost = float(abs(pred_raw[4] - true_raw[4]) / 0.01)
            rotation_cost = float(_angular_error_deg(np.asarray([pred_raw[5]]), np.asarray([true_raw[5]]))[0] / 30.0)
            cost += type_mismatch + center_cost + axis_cost + depth_cost + rotation_cost
        if cost < best_cost:
            best_cost = cost
            best_perm = tuple(permutation)
    return best_perm


def predict_arrays(model, tensors: dict, targets: dict, mean: np.ndarray, std: np.ndarray, args) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(tensors["signals"], tensors.get("features"))
    presence_prob = torch.sigmoid(out["presence_logits"]).cpu().numpy()
    presence_pred = presence_prob >= 0.5
    type_pred = out["type_logits"].argmax(dim=-1).cpu().numpy()
    continuous_pred_unscaled = _effective_continuous_unscaled(out, tensors, args, use_soft_bins=False).cpu().numpy()
    continuous_true_unscaled = targets["continuous_unscaled"]
    continuous_pred_raw = continuous_to_raw(
        continuous_pred_unscaled,
        targets["target_schema"],
        angle_encoding=targets.get("angle_encoding", "raw"),
    )
    continuous_true_raw = targets["continuous_raw"]
    presence_true = targets["presence"] > 0.5
    type_true = targets["type_targets"]
    predicted = {
        "presence_prob": presence_prob,
        "presence_pred": presence_pred,
        "presence_true": presence_true,
        "type_pred": type_pred,
        "type_true": type_true,
        "continuous_pred_raw": continuous_pred_raw,
        "continuous_true_raw": continuous_true_raw,
        "continuous_pred_unscaled": continuous_pred_unscaled,
        "continuous_true_unscaled": continuous_true_unscaled,
    }
    if getattr(args, "center_representation", "continuous") == "bin_offset":
        predicted.update(
            {
                "center_x_bin_pred": out["center_x_bin_logits"].argmax(dim=-1).cpu().numpy(),
                "center_y_bin_pred": out["center_y_bin_logits"].argmax(dim=-1).cpu().numpy(),
                "center_offset_pred": out["center_offset"].cpu().numpy(),
                "center_x_bin_true": tensors["center_x_bin_targets"].cpu().numpy(),
                "center_y_bin_true": tensors["center_y_bin_targets"].cpu().numpy(),
                "center_offset_true": tensors["center_offset_targets"].cpu().numpy(),
            }
        )
    if "aux_center_x_bin_logits" in out:
        predicted.update(
            {
                "aux_center_x_bin_pred": out["aux_center_x_bin_logits"].argmax(dim=-1).cpu().numpy(),
                "aux_center_y_bin_pred": out["aux_center_y_bin_logits"].argmax(dim=-1).cpu().numpy(),
                "aux_center_offset_pred": out["aux_center_offset"].cpu().numpy(),
            }
        )
    return predicted


def _center_eval_metrics(
    continuous_pred_raw: np.ndarray,
    continuous_true_raw: np.ndarray,
    presence_true: np.ndarray,
    x: np.ndarray | None,
    y: np.ndarray | None,
) -> tuple[float, float]:
    if not presence_true.any():
        return float("nan"), float("nan")
    pred = continuous_pred_raw[presence_true]
    true = continuous_true_raw[presence_true]
    delta_x = pred[:, 0] - true[:, 0]
    delta_y = pred[:, 1] - true[:, 1]
    if x is not None and y is not None:
        try:
            dx = _mean_grid_spacing(x, "x")
            dy = _mean_grid_spacing(y, "y")
            center_grid_mae = float(np.sqrt((delta_x / dx) ** 2 + (delta_y / dy) ** 2).mean())
        except ValueError:
            center_grid_mae = float("nan")
    else:
        center_grid_mae = float("nan")
    axis_x = np.maximum(np.abs(true[:, 2]), 1e-12)
    axis_y = np.maximum(np.abs(true[:, 3]), 1e-12)
    center_axis_relative_mae = float(np.sqrt((delta_x / axis_x) ** 2 + (delta_y / axis_y) ** 2).mean())
    return center_grid_mae, center_axis_relative_mae


def evaluate(model, tensors: dict, targets: dict, mean: np.ndarray, std: np.ndarray, split: str, args) -> dict:
    predicted = predict_arrays(model, tensors, targets, mean, std, args)
    presence_pred = predicted["presence_pred"]
    presence_true = predicted["presence_true"]
    type_pred = predicted["type_pred"]
    type_true = predicted["type_true"]
    continuous_pred_raw = predicted["continuous_pred_raw"]
    continuous_true_raw = predicted["continuous_true_raw"]
    continuous_pred_unscaled = predicted["continuous_pred_unscaled"]
    continuous_true_unscaled = predicted["continuous_true_unscaled"]

    presence_accuracy = float((presence_pred == presence_true).mean())
    if presence_true.any():
        type_accuracy = float((type_pred[presence_true] == type_true[presence_true]).mean())
        abs_err_raw = np.abs(continuous_pred_raw[presence_true] - continuous_true_raw[presence_true])
        abs_err_unscaled = np.abs(continuous_pred_unscaled[presence_true] - continuous_true_unscaled[presence_true])
        continuous_mae_mean = float(abs_err_unscaled.mean())
        center_mae = float(abs_err_raw[:, [0, 1]].mean())
        center_grid_mae, center_axis_relative_mae = _center_eval_metrics(
            continuous_pred_raw,
            continuous_true_raw,
            presence_true,
            tensors["x"],
            tensors["y"],
        )
        axis_mae = float(abs_err_raw[:, [2, 3]].mean())
        depth_mae = float(abs_err_raw[:, 4].mean())
        rotation_mae = _angular_mae_deg(continuous_pred_raw[presence_true], continuous_true_raw[presence_true])
        if "center_x_bin_pred" in predicted:
            center_x_bin_accuracy = float((predicted["center_x_bin_pred"][presence_true] == predicted["center_x_bin_true"][presence_true]).mean())
            center_y_bin_accuracy = float((predicted["center_y_bin_pred"][presence_true] == predicted["center_y_bin_true"][presence_true]).mean())
            center_offset_mae = float(np.abs(predicted["center_offset_pred"][presence_true] - predicted["center_offset_true"][presence_true]).mean())
        else:
            center_x_bin_accuracy = float("nan")
            center_y_bin_accuracy = float("nan")
            center_offset_mae = float("nan")
        if "aux_center_x_bin_pred" in predicted:
            aux_center_x_bin_accuracy = float((predicted["aux_center_x_bin_pred"][presence_true] == predicted["center_x_bin_true"][presence_true]).mean())
            aux_center_y_bin_accuracy = float((predicted["aux_center_y_bin_pred"][presence_true] == predicted["center_y_bin_true"][presence_true]).mean())
            aux_center_offset_mae = float(np.abs(predicted["aux_center_offset_pred"][presence_true] - predicted["center_offset_true"][presence_true]).mean())
        else:
            aux_center_x_bin_accuracy = float("nan")
            aux_center_y_bin_accuracy = float("nan")
            aux_center_offset_mae = float("nan")
    else:
        type_accuracy = float("nan")
        continuous_mae_mean = float("nan")
        center_mae = float("nan")
        center_grid_mae = float("nan")
        center_axis_relative_mae = float("nan")
        axis_mae = float("nan")
        depth_mae = float("nan")
        rotation_mae = float("nan")
        center_x_bin_accuracy = float("nan")
        center_y_bin_accuracy = float("nan")
        center_offset_mae = float("nan")
        aux_center_x_bin_accuracy = float("nan")
        aux_center_y_bin_accuracy = float("nan")
        aux_center_offset_mae = float("nan")

    if tensors["x"] is None or tensors["y"] is None or tensors["masks"] is None:
        mask_iou, mask_dice = float("nan"), float("nan")
    else:
        pred_masks = rasterize_components(
            continuous_pred_raw,
            type_pred,
            presence_pred.astype(np.float32),
            targets["raw_target_schema"],
            targets["type_vocab"],
            tensors["x"],
            tensors["y"],
        )
        mask_iou_arr, mask_dice_arr = mask_iou_dice(pred_masks, tensors["masks"])
        mask_iou, mask_dice = float(mask_iou_arr.mean()), float(mask_dice_arr.mean())
    return {
        "split": split,
        "presence_accuracy": presence_accuracy,
        "type_accuracy_present": type_accuracy,
        "continuous_mae_mean": continuous_mae_mean,
        "center_mae": center_mae,
        "center_grid_mae": center_grid_mae,
        "center_axis_relative_mae": center_axis_relative_mae,
        "center_x_bin_accuracy": center_x_bin_accuracy,
        "center_y_bin_accuracy": center_y_bin_accuracy,
        "center_offset_mae": center_offset_mae,
        "aux_center_x_bin_accuracy": aux_center_x_bin_accuracy,
        "aux_center_y_bin_accuracy": aux_center_y_bin_accuracy,
        "aux_center_offset_mae": aux_center_offset_mae,
        "axis_mae": axis_mae,
        "rotation_mae": rotation_mae,
        "depth_mae": depth_mae,
        "param_mask_iou": mask_iou,
        "param_mask_dice": mask_dice,
    }


def export_predictions(
    output_dir: Path,
    split: str,
    model,
    tensors: dict,
    targets: dict,
    mean: np.ndarray,
    std: np.ndarray,
    matching_mode: str,
    args,
) -> None:
    predicted = predict_arrays(model, tensors, targets, mean, std, args)
    type_vocab = targets["type_vocab"]
    component_rows = []
    for sample_pos, sample_index in enumerate(tensors["sample_indices"]):
        matched_perm = _best_export_permutation(predicted, sample_pos, matching_mode)
        for slot in range(predicted["presence_true"].shape[1]):
            true_slot = matched_perm[slot]
            true_type_id = int(predicted["type_true"][sample_pos, true_slot])
            pred_type_id = int(predicted["type_pred"][sample_pos, slot])
            target_present = bool(predicted["presence_true"][sample_pos, true_slot])
            true_name = type_vocab[true_type_id] if target_present and true_type_id >= 0 else ""
            pred_name = type_vocab[pred_type_id] if pred_type_id >= 0 else ""
            true_raw = predicted["continuous_true_raw"][sample_pos, true_slot]
            pred_raw = predicted["continuous_pred_raw"][sample_pos, slot]
            if target_present:
                center_error = float(np.linalg.norm(pred_raw[[0, 1]] - true_raw[[0, 1]]))
                axis_error = float(np.linalg.norm(pred_raw[[2, 3]] - true_raw[[2, 3]]))
                depth_error = float(abs(pred_raw[4] - true_raw[4]))
                rotation_error = float(_angular_error_deg(np.asarray([pred_raw[5]]), np.asarray([true_raw[5]]))[0])
                type_correct = int(true_type_id == pred_type_id)
            else:
                center_error = float("nan")
                axis_error = float("nan")
                depth_error = float("nan")
                rotation_error = float("nan")
                type_correct = ""
            component_rows.append(
                {
                    "split": split,
                    "sample_index": int(sample_index),
                    "component_slot": slot,
                    "matched_slot": int(true_slot),
                    "presence_true": int(target_present),
                    "presence_prob": float(predicted["presence_prob"][sample_pos, slot]),
                    "presence_pred": int(predicted["presence_pred"][sample_pos, slot]),
                    "type_true": true_name,
                    "type_pred": pred_name,
                    "type_correct": type_correct,
                    "center_x_true": float(true_raw[0]),
                    "center_x_pred": float(pred_raw[0]),
                    "center_y_true": float(true_raw[1]),
                    "center_y_pred": float(pred_raw[1]),
                    "axis_x_true": float(true_raw[2]),
                    "axis_x_pred": float(pred_raw[2]),
                    "axis_y_true": float(true_raw[3]),
                    "axis_y_pred": float(pred_raw[3]),
                    "depth_true": float(true_raw[4]),
                    "depth_pred": float(pred_raw[4]),
                    "rotation_true": float(true_raw[5]),
                    "rotation_pred": float(pred_raw[5]),
                    "center_error": center_error,
                    "axis_error": axis_error,
                    "depth_error": depth_error,
                    "rotation_error": rotation_error,
                    "target_schema": "|".join(targets["raw_target_schema"]),
                    "type_vocab": "|".join(type_vocab),
                }
            )
    write_csv(output_dir / f"{split}_predictions.csv", component_rows)

    if tensors["x"] is None or tensors["y"] is None or tensors["masks"] is None:
        return
    pred_masks = rasterize_components(
        predicted["continuous_pred_raw"],
        predicted["type_pred"],
        predicted["presence_pred"].astype(np.float32),
        targets["raw_target_schema"],
        targets["type_vocab"],
        tensors["x"],
        tensors["y"],
    )
    oracle_masks = rasterize_components(
        predicted["continuous_true_raw"],
        predicted["type_true"],
        predicted["presence_true"].astype(np.float32),
        targets["raw_target_schema"],
        targets["type_vocab"],
        tensors["x"],
        tensors["y"],
    )
    pred_iou, pred_dice = mask_iou_dice(pred_masks, tensors["masks"])
    oracle_iou, _oracle_dice = mask_iou_dice(oracle_masks, tensors["masks"])
    target_area = (tensors["masks"] > 0.5).sum(axis=(1, 2))
    pred_area = pred_masks.sum(axis=(1, 2))
    metric_rows = []
    for sample_pos, sample_index in enumerate(tensors["sample_indices"]):
        metric_rows.append(
            {
                "split": split,
                "sample_index": int(sample_index),
                "pred_mask_iou": float(pred_iou[sample_pos]),
                "pred_dice": float(pred_dice[sample_pos]),
                "oracle_mask_iou": float(oracle_iou[sample_pos]),
                "oracle_gap": float(oracle_iou[sample_pos] - pred_iou[sample_pos]),
                "target_area": int(target_area[sample_pos]),
                "pred_area": int(pred_area[sample_pos]),
                "area_diff": int(pred_area[sample_pos] - target_area[sample_pos]),
                "type_sequence_true": _type_sequence(
                    predicted["type_true"][sample_pos],
                    predicted["presence_true"][sample_pos],
                    targets["type_vocab"],
                ),
                "type_sequence_pred": _type_sequence(
                    predicted["type_pred"][sample_pos],
                    predicted["presence_pred"][sample_pos],
                    targets["type_vocab"],
                ),
            }
        )
    write_csv(output_dir / f"{split}_prediction_mask_metrics.csv", metric_rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    args,
    train_targets: dict,
    metrics: dict[str, dict],
    mean: np.ndarray,
    std: np.ndarray,
    selection_info: dict | None = None,
    feature_info: dict | None = None,
    center_bin_info: dict | None = None,
) -> None:
    selection_info = selection_info or {}
    feature_info = feature_info or {}
    center_bin_info = center_bin_info or {}
    lines = [
        "# COMSOL parametric inverse run summary",
        "",
        f"- train_npz: `{args.train_npz}`",
        f"- val_npz: `{args.val_npz}`",
        f"- test_npz: `{args.test_npz}`",
        f"- train_targets: `{args.train_targets}`",
        f"- seed: `{args.seed}`",
        f"- steps: `{args.steps}`",
        f"- lr: `{args.lr}`",
        f"- hidden_dim: `{args.hidden_dim}`",
        f"- latent_dim: `{args.latent_dim}`",
        f"- max_components: `{args.max_components}`",
        f"- encoder_type: `{args.encoder_type}`",
        f"- head_mode: `{args.head_mode}`",
        f"- feature_fusion_mode: `{args.feature_fusion_mode}`",
        f"- feature_dim: `{feature_info.get('feature_dim', 0)}`",
        f"- feature_npz: `{args.feature_npz}`",
        f"- val_feature_npz: `{args.val_feature_npz}`",
        f"- test_feature_npz: `{args.test_feature_npz}`",
        f"- target_schema: `{', '.join(train_targets['target_schema'])}`",
        f"- type_vocab: `{', '.join(train_targets['type_vocab'])}`",
        f"- angle_encoding: `{train_targets.get('angle_encoding', 'raw')}`",
        f"- continuous_targets_normalized: `{train_targets.get('continuous_targets_normalized', False)}`",
        f"- type_class_weighting: `{args.type_class_weighting}`",
        f"- component_matching_mode: `{args.component_matching_mode}`",
        f"- export_predictions: `{args.export_predictions}`",
        f"- lambda_raster_bce: `{args.lambda_raster_bce}`",
        f"- lambda_raster_dice: `{args.lambda_raster_dice}`",
        f"- raster_loss_start_step: `{args.raster_loss_start_step}`",
        f"- raster_softness_cells: `{args.raster_softness_cells}`",
        f"- raster_target_source: `{args.raster_target_source}`",
        f"- val_selection_metric: `{args.val_selection_metric}`",
        f"- val_selection_interval: `{args.val_selection_interval}`",
        f"- best_step: `{selection_info.get('best_step', '')}`",
        f"- best_val_mask_iou: `{selection_info.get('best_val_mask_iou', float('nan'))}`",
        f"- best_val_loss: `{selection_info.get('best_val_loss', float('nan'))}`",
        f"- lambda_center: `{args.lambda_center}`",
        f"- lambda_axis: `{args.lambda_axis}`",
        f"- lambda_depth: `{args.lambda_depth}`",
        f"- lambda_rotation: `{args.lambda_rotation}`",
        f"- lambda_center_grid: `{args.lambda_center_grid}`",
        f"- lambda_center_axis_relative: `{args.lambda_center_axis_relative}`",
        f"- center_axis_relative_eps: `{args.center_axis_relative_eps}`",
        f"- center_representation: `{args.center_representation}`",
        f"- center_bin_size_cells: `{args.center_bin_size_cells}`",
        f"- center_x_bins: `{center_bin_info.get('center_x_bins', 0)}`",
        f"- center_y_bins: `{center_bin_info.get('center_y_bins', 0)}`",
        f"- lambda_center_bin: `{args.lambda_center_bin}`",
        f"- lambda_center_offset: `{args.lambda_center_offset}`",
        f"- center_bin_x_weight: `{args.center_bin_x_weight}`",
        f"- center_bin_y_weight: `{args.center_bin_y_weight}`",
        f"- center_bin_slot_weights: `{args.center_bin_slot_weights}`",
        f"- center_bin_slot_weights_resolved: `{', '.join(str(v) for v in getattr(args, '_center_bin_slot_weights_resolved', []))}`",
        f"- aux_center_head: `{args.aux_center_head}`",
        f"- lambda_aux_center_bin: `{args.lambda_aux_center_bin}`",
        f"- lambda_aux_center_offset: `{args.lambda_aux_center_offset}`",
        f"- aux_center_x_weight: `{args.aux_center_x_weight}`",
        f"- aux_center_y_weight: `{args.aux_center_y_weight}`",
        f"- lambda_type_extra: `{args.lambda_type_extra}`",
        f"- lambda_rotation_extra: `{args.lambda_rotation_extra}`",
        f"- rotation_loss_mode: `{args.rotation_loss_mode}`",
        "- signal_normalization: `per_sample_zscore`",
        "",
        "## Continuous normalization",
        "",
    ]
    for name, m, s in zip(train_targets["target_schema"], mean, std):
        lines.append(f"- `{name}`: mean={m:.6g}, std={s:.6g}")
    if feature_info.get("feature_dim", 0):
        lines.extend(["", "## Feature normalization", ""])
        feature_names = feature_info.get("feature_names", [])
        feature_mean = feature_info.get("feature_mean", [])
        feature_std = feature_info.get("feature_std", [])
        for name, m, s in zip(feature_names[:20], feature_mean[:20], feature_std[:20]):
            lines.append(f"- `{name}`: mean={m:.6g}, std={s:.6g}")
        if len(feature_names) > 20:
            lines.append(f"- ... {len(feature_names) - 20} additional features omitted from summary")
    if center_bin_info:
        lines.extend(["", "## Center bin representation", ""])
        for name in [
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "dx",
            "dy",
            "bin_width_x",
            "bin_width_y",
            "center_x_bins",
            "center_y_bins",
        ]:
            lines.append(f"- `{name}`: {center_bin_info[name]}")
    lines.extend(["", "## Final metrics", ""])
    for split, row in metrics.items():
        lines.append(
            f"- `{split}`: presence_accuracy={row['presence_accuracy']:.6e}, "
            f"type_accuracy_present={row['type_accuracy_present']:.6e}, "
            f"continuous_mae_mean={row['continuous_mae_mean']:.6e}, "
            f"param_mask_iou={row['param_mask_iou']:.6e}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.val_selection_metric != "none" and args.val_selection_interval <= 0:
        raise ValueError("val-selection-interval must be > 0 when val-selection-metric is enabled.")
    if args.val_selection_metric == "val_loss" and _has_raster_loss(args) and args.raster_loss_start_step > 0:
        raise ValueError(
            "val_selection_metric='val_loss' is incompatible with delayed raster loss because "
            "the selected loss components change at raster_loss_start_step. Use val_mask_iou instead."
        )
    if args.center_representation == "bin_offset" and args.component_matching_mode != "fixed":
        raise ValueError("center-representation=bin_offset currently supports fixed component matching only.")
    if args.center_bin_x_weight <= 0 or args.center_bin_y_weight <= 0:
        raise ValueError("center-bin x/y weights must be positive.")
    args._center_bin_slot_weights_resolved = _parse_center_bin_slot_weights(args.center_bin_slot_weights, args.max_components)
    if args.aux_center_head and args.center_representation != "bin_offset":
        raise ValueError("aux-center-head requires center-representation=bin_offset.")
    if args.lambda_aux_center_bin < 0 or args.lambda_aux_center_offset < 0:
        raise ValueError("aux center loss weights must be non-negative.")
    if args.aux_center_x_weight <= 0 or args.aux_center_y_weight <= 0:
        raise ValueError("aux center x/y weights must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    train_dataset = load_dataset(args.train_npz)
    val_dataset = load_dataset(args.val_npz)
    test_dataset = load_dataset(args.test_npz)
    train_targets = load_targets(args.train_targets)
    val_targets = load_targets(args.val_targets, train_targets["type_vocab"])
    test_targets = load_targets(args.test_targets, train_targets["type_vocab"])
    center_bin_info = build_center_bin_info(train_dataset, args)
    train_features = val_features = test_features = None
    feature_names: list[str] = []
    feature_mean = feature_std = None
    if args.feature_fusion_mode != "none":
        if not args.feature_npz or not args.val_feature_npz or not args.test_feature_npz:
            raise ValueError("feature-npz, val-feature-npz, and test-feature-npz are required when feature_fusion_mode is enabled.")
        train_feature_raw, feature_names = load_feature_matrix(
            args.feature_npz,
            train_dataset["signals"].shape[0],
            train_targets["sample_indices"],
        )
        val_feature_raw, val_feature_names = load_feature_matrix(
            args.val_feature_npz,
            val_dataset["signals"].shape[0],
            val_targets["sample_indices"],
        )
        test_feature_raw, test_feature_names = load_feature_matrix(
            args.test_feature_npz,
            test_dataset["signals"].shape[0],
            test_targets["sample_indices"],
        )
        if val_feature_names != feature_names or test_feature_names != feature_names:
            raise ValueError("Feature names differ across train/val/test feature NPZ files.")
        feature_mean, feature_std = compute_feature_norm(train_feature_raw)
        train_features = normalize_features(train_feature_raw, feature_mean, feature_std)
        val_features = normalize_features(val_feature_raw, feature_mean, feature_std)
        test_features = normalize_features(test_feature_raw, feature_mean, feature_std)

    mean, std = compute_continuous_norm(train_targets)
    if train_targets.get("continuous_targets_normalized"):
        for split_name, target in [("val", val_targets), ("test", test_targets)]:
            if list(target["target_schema"]) != list(train_targets["target_schema"]):
                raise ValueError(f"{split_name} target_schema does not match train target_schema.")
            if not target.get("continuous_targets_normalized"):
                raise ValueError(f"{split_name} targets must be normalized when train targets are normalized.")
    train_tensors = build_tensors(train_dataset, train_targets, mean, std, device, features=train_features, center_bin_info=center_bin_info)
    val_tensors = build_tensors(val_dataset, val_targets, mean, std, device, features=val_features, center_bin_info=center_bin_info)
    test_tensors = build_tensors(test_dataset, test_targets, mean, std, device, features=test_features, center_bin_info=center_bin_info)

    signal_len = int(train_tensors["signals"].shape[1])
    feature_dim = 0 if train_features is None else int(train_features.shape[1])
    num_types = len(train_targets["type_vocab"])
    num_continuous = len(train_targets["target_schema"])
    model = ParametricInverseNet(
        signal_len=signal_len,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_components=args.max_components,
        num_types=num_types,
        num_continuous=num_continuous,
        num_layers=args.num_layers,
        encoder_type=args.encoder_type,
        head_mode=args.head_mode,
        feature_dim=feature_dim,
        feature_fusion_mode=args.feature_fusion_mode,
        center_representation=args.center_representation,
        center_x_bins=0 if center_bin_info is None else int(center_bin_info["center_x_bins"]),
        center_y_bins=0 if center_bin_info is None else int(center_bin_info["center_y_bins"]),
        aux_center_head=args.aux_center_head,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    type_weights = None
    if args.type_class_weighting == "inverse_freq":
        active_types = train_targets["type_targets"][train_targets["presence"] > 0.5]
        counts = np.bincount(active_types, minlength=num_types).astype(np.float32)
        if np.any(counts <= 0):
            raise ValueError("inverse_freq type weighting requires every train type to be present.")
        weights = counts.sum() / (counts * num_types)
        type_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    history = []
    best_state = None
    best_step: int | None = None
    best_val_mask_iou = float("-inf")
    best_val_loss = float("inf")
    raster_selection_reset_done = False
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        loss, losses = compute_loss(model, train_tensors, args, type_weights, step=step)
        loss.backward()
        optimizer.step()
        if (
            args.val_selection_metric != "none"
            and not raster_selection_reset_done
            and _has_raster_loss(args)
            and args.raster_loss_start_step > 0
            and step >= args.raster_loss_start_step
        ):
            best_state = None
            best_step = None
            best_val_mask_iou = float("-inf")
            best_val_loss = float("inf")
            raster_selection_reset_done = True
        val_mask_iou_at_step = float("nan")
        val_loss_at_step = float("nan")
        is_best_step = 0
        selection_due = (
            args.val_selection_metric != "none"
            and args.val_selection_interval > 0
            and (step % args.val_selection_interval == 0 or step == args.steps)
        )
        if selection_due:
            model.eval()
            if args.val_selection_metric == "val_mask_iou":
                val_row_at_step = evaluate(model, val_tensors, val_targets, mean, std, "val", args)
                val_mask_iou_at_step = float(val_row_at_step["param_mask_iou"])
                if val_mask_iou_at_step > best_val_mask_iou:
                    best_val_mask_iou = val_mask_iou_at_step
                    best_step = step
                    best_state = copy.deepcopy(model.state_dict())
                    is_best_step = 1
            elif args.val_selection_metric == "val_loss":
                with torch.no_grad():
                    val_loss_tensor, _val_parts = compute_loss(model, val_tensors, args, type_weights, step=step)
                val_loss_at_step = float(val_loss_tensor.detach().cpu())
                if val_loss_at_step < best_val_loss:
                    best_val_loss = val_loss_at_step
                    best_step = step
                    best_state = copy.deepcopy(model.state_dict())
                    is_best_step = 1
            else:
                raise ValueError(f"Unsupported val_selection_metric: {args.val_selection_metric}")
        if (
            step == 1
            or step == args.steps
            or (args.history_interval > 0 and step % args.history_interval == 0)
            or selection_due
        ):
            history.append(
                {
                    "step": step,
                    **losses,
                    "val_mask_iou_at_step": val_mask_iou_at_step,
                    "val_loss_at_step": val_loss_at_step,
                    "is_best_step": is_best_step,
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_step is not None:
        for row in history:
            row["is_best_step"] = int(row["step"] == best_step)

    train_row = evaluate(model, train_tensors, train_targets, mean, std, "train", args)
    val_row = evaluate(model, val_tensors, val_targets, mean, std, "val", args)
    test_row = evaluate(model, test_tensors, test_targets, mean, std, "test", args)
    for row in [train_row, val_row, test_row]:
        row["encoder_type"] = args.encoder_type
        row["head_mode"] = args.head_mode
        row["feature_fusion_mode"] = args.feature_fusion_mode
        row["feature_dim"] = feature_dim
        row["seed"] = args.seed
        row["component_matching_mode"] = args.component_matching_mode
        row["lambda_raster_bce"] = args.lambda_raster_bce
        row["lambda_raster_dice"] = args.lambda_raster_dice
        row["raster_loss_start_step"] = args.raster_loss_start_step
        row["raster_softness_cells"] = args.raster_softness_cells
        row["raster_target_source"] = args.raster_target_source
        row["val_selection_metric"] = args.val_selection_metric
        row["val_selection_interval"] = args.val_selection_interval
        row["lambda_type_extra"] = args.lambda_type_extra
        row["lambda_rotation_extra"] = args.lambda_rotation_extra
        row["rotation_loss_mode"] = args.rotation_loss_mode
        row["lambda_center_grid"] = args.lambda_center_grid
        row["lambda_center_axis_relative"] = args.lambda_center_axis_relative
        row["center_axis_relative_eps"] = args.center_axis_relative_eps
        row["center_representation"] = args.center_representation
        row["center_bin_size_cells"] = args.center_bin_size_cells
        row["center_x_bins"] = 0 if center_bin_info is None else int(center_bin_info["center_x_bins"])
        row["center_y_bins"] = 0 if center_bin_info is None else int(center_bin_info["center_y_bins"])
        row["lambda_center_bin"] = args.lambda_center_bin
        row["lambda_center_offset"] = args.lambda_center_offset
        row["center_bin_x_weight"] = args.center_bin_x_weight
        row["center_bin_y_weight"] = args.center_bin_y_weight
        row["center_bin_slot_weights"] = args.center_bin_slot_weights
        row["center_bin_slot_weights_resolved"] = ",".join(str(v) for v in args._center_bin_slot_weights_resolved)
        row["aux_center_head"] = args.aux_center_head
        row["lambda_aux_center_bin"] = args.lambda_aux_center_bin
        row["lambda_aux_center_offset"] = args.lambda_aux_center_offset
        row["aux_center_x_weight"] = args.aux_center_x_weight
        row["aux_center_y_weight"] = args.aux_center_y_weight
        row["best_step"] = "" if best_step is None else best_step
        row["best_val_mask_iou"] = float("nan") if best_val_mask_iou == float("-inf") else best_val_mask_iou
        row["best_val_loss"] = float("nan") if best_val_loss == float("inf") else best_val_loss
    write_csv(output_dir / "metrics.csv", [train_row])
    write_csv(output_dir / "eval_metrics.csv", [val_row])
    write_csv(output_dir / "test_metrics.csv", [test_row])
    write_csv(output_dir / "training_history.csv", history)
    if args.export_predictions:
        export_predictions(output_dir, "train", model, train_tensors, train_targets, mean, std, args.component_matching_mode, args)
        export_predictions(output_dir, "val", model, val_tensors, val_targets, mean, std, args.component_matching_mode, args)
        export_predictions(output_dir, "test", model, test_tensors, test_targets, mean, std, args.component_matching_mode, args)
    write_summary(
        output_dir / "run_summary.md",
        args,
        train_targets,
        {"train": train_row, "val": val_row, "test": test_row},
        mean,
        std,
        {
            "best_step": "" if best_step is None else best_step,
            "best_val_mask_iou": float("nan") if best_val_mask_iou == float("-inf") else best_val_mask_iou,
            "best_val_loss": float("nan") if best_val_loss == float("inf") else best_val_loss,
        },
        {
            "feature_dim": feature_dim,
            "feature_names": feature_names,
            "feature_mean": [] if feature_mean is None else feature_mean,
            "feature_std": [] if feature_std is None else feature_std,
        },
        center_bin_info,
    )
    print(f"Saved parametric inverse metrics to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", default="")
    parser.add_argument("--train-targets", default="")
    parser.add_argument("--val-npz", default="")
    parser.add_argument("--val-targets", default="")
    parser.add_argument("--test-npz", default="")
    parser.add_argument("--test-targets", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--encoder-type", choices=["mlp", "cnn1d", "cnn1d_attention"], default="mlp")
    parser.add_argument("--head-mode", choices=["shared", "component_specific"], default="shared")
    parser.add_argument("--feature-npz", default="")
    parser.add_argument("--val-feature-npz", default="")
    parser.add_argument("--test-feature-npz", default="")
    parser.add_argument("--feature-fusion-mode", choices=["none", "features_only", "concat_latent"], default="none")
    parser.add_argument("--lambda-presence", type=float, default=1.0)
    parser.add_argument("--lambda-type", type=float, default=1.0)
    parser.add_argument("--lambda-continuous", type=float, default=1.0)
    parser.add_argument("--lambda-center", type=float, default=1.0)
    parser.add_argument("--lambda-axis", type=float, default=1.0)
    parser.add_argument("--lambda-depth", type=float, default=1.0)
    parser.add_argument("--lambda-rotation", type=float, default=1.0)
    parser.add_argument("--lambda-center-grid", type=float, default=0.0)
    parser.add_argument("--lambda-center-axis-relative", type=float, default=0.0)
    parser.add_argument("--center-axis-relative-eps", type=float, default=1e-6)
    parser.add_argument("--center-representation", choices=["continuous", "bin_offset"], default="continuous")
    parser.add_argument("--center-bin-size-cells", type=int, default=8)
    parser.add_argument("--lambda-center-bin", type=float, default=1.0)
    parser.add_argument("--lambda-center-offset", type=float, default=1.0)
    parser.add_argument("--center-bin-x-weight", type=float, default=1.0)
    parser.add_argument("--center-bin-y-weight", type=float, default=1.0)
    parser.add_argument("--center-bin-slot-weights", default="")
    parser.add_argument("--aux-center-head", action="store_true")
    parser.add_argument("--lambda-aux-center-bin", type=float, default=0.0)
    parser.add_argument("--lambda-aux-center-offset", type=float, default=0.0)
    parser.add_argument("--aux-center-x-weight", type=float, default=1.0)
    parser.add_argument("--aux-center-y-weight", type=float, default=1.0)
    parser.add_argument("--lambda-type-extra", type=float, default=0.0)
    parser.add_argument("--lambda-rotation-extra", type=float, default=0.0)
    parser.add_argument("--rotation-loss-mode", choices=["mse", "circular"], default="mse")
    parser.add_argument("--history-interval", type=int, default=100)
    parser.add_argument("--type-class-weighting", choices=["none", "inverse_freq"], default="none")
    parser.add_argument("--component-matching-mode", choices=["fixed", "permutation_min"], default="fixed")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--lambda-raster-bce", type=float, default=0.0)
    parser.add_argument("--lambda-raster-dice", type=float, default=0.0)
    parser.add_argument("--raster-loss-start-step", type=int, default=0)
    parser.add_argument("--raster-softness-cells", type=float, default=1.0)
    parser.add_argument("--raster-target-source", choices=["masks", "mu_threshold"], default="masks")
    parser.add_argument("--val-selection-metric", choices=["none", "val_mask_iou", "val_loss"], default="none")
    parser.add_argument("--val-selection-interval", type=int, default=0)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    required = [
        args.train_npz,
        args.train_targets,
        args.val_npz,
        args.val_targets,
        args.test_npz,
        args.test_targets,
        args.output_dir,
    ]
    if not all(required):
        parser.print_help()
        print("\nExample: python train_comsol_parametric_inverse.py --train-npz train.npz --train-targets train_targets.npz --val-npz val.npz --val-targets val_targets.npz --test-npz test.npz --test-targets test_targets.npz --output-dir out")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
