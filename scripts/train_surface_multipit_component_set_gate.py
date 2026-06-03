#!/usr/bin/env python
"""Run the 25.10 surface multi-pit component-set training gate.

This is an explicit training-gate script, not a baseline transition script.
It trains a small fixed-K component-set model on the validated 25.9b pilot
pack and writes gate metrics, summary, and manifest records only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_surface_multipit_component_set_pilot_v1"
GATE_ID = "25_10_surface_multipit_component_set_training_gate"
DEFAULT_MANIFEST = ROOT / "results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json"
DEFAULT_METRICS = ROOT / "results/metrics/25_10_component_set_training_gate_metrics.json"
DEFAULT_SUMMARY = ROOT / "results/summaries/25_10_component_set_training_gate_summary.md"
DEFAULT_GATE_MANIFEST = ROOT / "results/manifests/25_10_component_set_training_gate_manifest.json"
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DEFAULT_COMPARISON_METRICS = ROOT / "results/metrics/25_10_component_set_training_gate_metrics.json"
DEFAULT_COMPARISON_METRICS_25_11 = ROOT / "results/metrics/25_11_mask_depth_loss_rebalance_training_metrics.json"
DEFAULT_COMPARISON_METRICS_25_12 = ROOT / "results/metrics/25_12_component_separation_rebalance_training_metrics.json"
DEFAULT_COMPARISON_METRICS_25_13 = ROOT / "results/metrics/25_13_target_v2_training_gate_metrics.json"
DEFAULT_COMPARISON_METRICS_25_15 = ROOT / "results/metrics/25_15_label_v3_training_gate_metrics.json"
DEFAULT_TARGET_REDESIGN_MANIFEST = ROOT / "results/manifests/25_12b_component_raster_depth_target_redesign_manifest.json"
DEFAULT_LABEL_V3_MANIFEST = ROOT / "results/manifests/25_14_label_v3_derivation_validator_manifest.json"
DEFAULT_LABEL_V3B_MANIFEST = ROOT / "results/manifests/25_16_label_v3b_derivation_validator_manifest.json"

TARGET_SPLIT = {"train": 72, "val": 20, "test": 20}
K_MAX = 3
LOW_H = 32
LOW_W = 64
PERMS = list(permutations(range(K_MAX)))
V3B_SOFT_OR_RAW_UNION_RATIO_TARGET = 1.35
V3B_HALO_CAP_FRACTION_OF_RAW_UNION = 0.25
V3B_SPARSE_LOWER_BOUND_PX = 20
MASK_DEPTH_TERMS = ("component_mask", "union_mask", "component_depth", "union_depth")
COMPONENT_SEPARATION_TERMS = ("separation_penalty", "merge_penalty")
FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]


@dataclass(frozen=True)
class LossConfig:
    name: str
    description: str
    weights: dict[str, float]
    mask_supervision: str
    depth_supervision: str
    union_warmup_epochs: int = 0
    union_ramp_epochs: int = 1


LOSS_CONFIGS = {
    "component_set_gate_v1": LossConfig(
        name="component_set_gate_v1",
        description="25.10 control loss: existence + active params/shape + component mask BCE/Dice + full-grid weighted depth proxy.",
        weights={
            "exist": 1.0,
            "param": 1.8,
            "shape": 0.25,
            "component_mask": 0.8,
            "union_mask": 0.0,
            "component_depth": 0.9,
            "union_depth": 0.0,
            "component_depth_background": 0.0,
            "union_depth_background": 0.0,
            "separation_penalty": 0.0,
            "merge_penalty": 0.0,
        },
        mask_supervision="component_mean_bce_plus_dice",
        depth_supervision="component_full_grid_target_weight_0p15_background",
    ),
    "mask_depth_rebalance_v1": LossConfig(
        name="mask_depth_rebalance_v1",
        description="25.11 rebalance: keep existence/geometry losses but emphasize active component and union mask/depth terms with foreground-normalized supervision.",
        weights={
            "exist": 0.75,
            "param": 1.05,
            "shape": 0.18,
            "component_mask": 2.00,
            "union_mask": 1.20,
            "component_depth": 1.00,
            "union_depth": 0.55,
            "component_depth_background": 0.0,
            "union_depth_background": 0.0,
            "separation_penalty": 0.0,
            "merge_penalty": 0.0,
        },
        mask_supervision="existing_slots_balanced_foreground_background_bce_plus_dice_with_union_loss",
        depth_supervision="existing_slots_and_union_smooth_l1_only_inside_valid_target_mask",
    ),
    "component_separation_rebalance_v1": LossConfig(
        name="component_separation_rebalance_v1",
        description="25.12 rebalance: keep architecture fixed, cap/delay union losses, emphasize matched component masks, add component separation and anti-merge penalties, and keep depth foreground-normalized.",
        weights={
            "exist": 1.10,
            "param": 1.40,
            "shape": 0.20,
            "component_mask": 1.70,
            "union_mask": 0.15,
            "component_depth": 0.35,
            "union_depth": 0.0,
            "component_depth_background": 0.0,
            "union_depth_background": 0.0,
            "separation_penalty": 1.00,
            "merge_penalty": 3.00,
        },
        mask_supervision="matched_existing_slots_balanced_bce_dice_plus_delayed_low_weight_union_loss",
        depth_supervision="component_normalized_foreground_depth_with_background_logged_not_weighted",
        union_warmup_epochs=60,
        union_ramp_epochs=120,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the 25.10 multi-pit component-set gate.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--gate-manifest", type=Path, default=DEFAULT_GATE_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--comparison-metrics", type=Path, default=DEFAULT_COMPARISON_METRICS)
    parser.add_argument("--comparison-metrics-25-11", type=Path, default=DEFAULT_COMPARISON_METRICS_25_11)
    parser.add_argument("--comparison-metrics-25-12", type=Path, default=DEFAULT_COMPARISON_METRICS_25_12)
    parser.add_argument("--comparison-metrics-25-13", type=Path, default=DEFAULT_COMPARISON_METRICS_25_13)
    parser.add_argument("--comparison-metrics-25-15", type=Path, default=DEFAULT_COMPARISON_METRICS_25_15)
    parser.add_argument("--target-redesign-manifest", type=Path, default=DEFAULT_TARGET_REDESIGN_MANIFEST)
    parser.add_argument("--label-v3-manifest", type=Path, default=DEFAULT_LABEL_V3_MANIFEST)
    parser.add_argument("--label-v3b-manifest", type=Path, default=DEFAULT_LABEL_V3B_MANIFEST)
    parser.add_argument("--stage", default="25.10")
    parser.add_argument("--gate-id", default=GATE_ID)
    parser.add_argument("--loss-config", choices=sorted(LOSS_CONFIGS), default="component_set_gate_v1")
    parser.add_argument("--target-version", choices=["v1", "v2", "v3", "v3b"], default="v1")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-registry-note", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    return value


def assert_manifest_and_registry(manifest_path: Path, registry_path: Path) -> tuple[dict[str, Any], Path]:
    if ROOT != Path(r"C:\Users\19166\Desktop\PINN_project"):
        raise RuntimeError(f"wrong project root: {ROOT}")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError(f"dataset_id mismatch: {manifest.get('dataset_id')}")
    if manifest.get("train_ready_candidate") is not True:
        raise ValueError("manifest train_ready_candidate must be true")
    if manifest.get("baseline_ready") is not False:
        raise ValueError("manifest baseline_ready must be false")
    if manifest.get("split_counts") != TARGET_SPLIT:
        raise ValueError(f"split mismatch: {manifest.get('split_counts')}")
    if int(manifest.get("K_max", -1)) != K_MAX:
        raise ValueError(f"K_max mismatch: {manifest.get('K_max')}")
    if manifest.get("auto_discovery_allowed") is not False or manifest.get("latest_newest_discovery_allowed") is not False:
        raise ValueError("manifest must disable auto/latest discovery")
    forbidden = set(manifest.get("forbidden_use", []))
    for required in {"baseline_update", "current_baseline_replacement", "latest_newest_auto_discovery"}:
        if required not in forbidden:
            raise ValueError(f"manifest missing forbidden_use={required}")
    registry = registry_path.read_text(encoding="utf-8")
    if f"## {DATASET_ID}" not in registry:
        raise ValueError(f"registry missing {DATASET_ID}")
    npz_path = Path(manifest["path"])
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    actual_hash = sha256_file(npz_path)
    if actual_hash != manifest.get("npz_sha256"):
        raise ValueError(f"NPZ sha256 mismatch: {actual_hash} != {manifest.get('npz_sha256')}")
    return manifest, npz_path


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as pack:
        return {name: pack[name].copy() for name in pack.files}


def downsample_np(array: np.ndarray, h: int = LOW_H, w: int = LOW_W) -> np.ndarray:
    tensor = torch.from_numpy(array.astype(np.float32))
    flat = tensor.reshape(-1, 1, tensor.shape[-2], tensor.shape[-1])
    out = F.interpolate(flat, size=(h, w), mode="area")
    return out.reshape(*tensor.shape[:-2], h, w).numpy().astype(np.float32)


def safe_std(values: np.ndarray, floor: float = 1.0e-6) -> np.ndarray:
    std = values.std(axis=0)
    return np.where(std < floor, 1.0, std).astype(np.float32)


def angle_diff_rad(pred: float, target: float) -> float:
    return float(abs(math.atan2(math.sin(pred - target), math.cos(pred - target))))


def mask_iou_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1.0e-8) -> tuple[float, float]:
    pred_bool = pred > 0.5
    target_bool = target > 0.5
    inter = float(np.logical_and(pred_bool, target_bool).sum())
    union = float(np.logical_or(pred_bool, target_bool).sum())
    denom = float(pred_bool.sum() + target_bool.sum())
    iou = inter / (union + eps)
    dice = (2.0 * inter) / (denom + eps)
    return iou, dice


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(-2, -1))
    denom = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps))


def soft_dice_loss_prob(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    inter = (prob * target).sum(dim=(-2, -1))
    denom = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps))


def balanced_bce_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pos = (target > 0.5).float()
    neg = 1.0 - pos
    pos_loss = (bce * pos).sum(dim=(-2, -1)) / pos.sum(dim=(-2, -1)).clamp_min(eps)
    neg_loss = (bce * neg).sum(dim=(-2, -1)) / neg.sum(dim=(-2, -1)).clamp_min(eps)
    return 0.5 * (pos_loss + neg_loss)


def balanced_bce_from_prob(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    prob = prob.clamp(eps, 1.0 - eps)
    bce = F.binary_cross_entropy(prob, target, reduction="none")
    pos = (target > 0.5).float()
    neg = 1.0 - pos
    pos_loss = (bce * pos).sum(dim=(-2, -1)) / pos.sum(dim=(-2, -1)).clamp_min(eps)
    neg_loss = (bce * neg).sum(dim=(-2, -1)) / neg.sum(dim=(-2, -1)).clamp_min(eps)
    return 0.5 * (pos_loss + neg_loss)


def valid_region_smooth_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    raw = F.smooth_l1_loss(pred, target, reduction="none")
    return (raw * valid).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(eps)


def scheduled_loss_weights(loss_config: LossConfig, epoch: int | None = None) -> dict[str, float]:
    weights = dict(loss_config.weights)
    if loss_config.name == "component_separation_rebalance_v1":
        if epoch is None:
            scale = 1.0
        elif epoch <= loss_config.union_warmup_epochs:
            scale = 0.0
        else:
            scale = min(1.0, (epoch - loss_config.union_warmup_epochs) / max(float(loss_config.union_ramp_epochs), 1.0))
        weights["union_mask"] *= scale
        weights["union_depth"] *= scale
    return weights


def pairwise_component_losses(
    pred_params: torch.Tensor,
    pred_mask_logits: torch.Tensor,
    target_params: torch.Tensor,
    target_mask: torch.Tensor,
    exists: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_prob = torch.sigmoid(pred_mask_logits)
    sep_terms = []
    merge_terms = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair_exists = exists[:, i] * exists[:, j]
        pred_i = mask_prob[:, i]
        pred_j = mask_prob[:, j]
        target_i = target_mask[:, i]
        target_j = target_mask[:, j]

        pred_overlap = (pred_i * pred_j).sum(dim=(-2, -1))
        pred_min_area = torch.minimum(pred_i.sum(dim=(-2, -1)), pred_j.sum(dim=(-2, -1))).clamp_min(eps)
        pred_overlap_fraction = pred_overlap / pred_min_area

        target_overlap = (target_i * target_j).sum(dim=(-2, -1))
        target_min_area = torch.minimum(target_i.sum(dim=(-2, -1)), target_j.sum(dim=(-2, -1))).clamp_min(eps)
        target_overlap_fraction = target_overlap / target_min_area
        allowed_overlap = (target_overlap_fraction + 0.04).clamp(max=0.85)
        merge_raw = F.relu(pred_overlap_fraction - allowed_overlap)

        pred_dist = torch.linalg.norm(pred_params[:, i, 0:2] - pred_params[:, j, 0:2], dim=-1)
        target_dist = torch.linalg.norm(target_params[:, i, 0:2] - target_params[:, j, 0:2], dim=-1)
        target_separated_weight = (1.0 - target_overlap_fraction.clamp(0.0, 1.0)).detach()
        sep_raw = F.smooth_l1_loss(pred_dist, target_dist, reduction="none") * target_separated_weight

        sep_terms.append(sep_raw * pair_exists)
        merge_terms.append(merge_raw * pair_exists * target_separated_weight)
    sep_stack = torch.stack(sep_terms, dim=1)
    merge_stack = torch.stack(merge_terms, dim=1)
    pair_counts = torch.stack([exists[:, 0] * exists[:, 1], exists[:, 0] * exists[:, 2], exists[:, 1] * exists[:, 2]], dim=1).sum(dim=1).clamp_min(1.0)
    return sep_stack.sum(dim=1) / pair_counts, merge_stack.sum(dim=1) / pair_counts


@dataclass
class Arrays:
    raw: dict[str, Any]
    signal: np.ndarray
    sensor_z_norm: np.ndarray
    exists: np.ndarray
    params_phys: np.ndarray
    params_norm: np.ndarray
    shape_target: np.ndarray
    mask_low: np.ndarray
    sdf_low: np.ndarray
    mask_valid_low: np.ndarray
    depth_low_norm: np.ndarray
    depth_valid_low: np.ndarray
    component_masks_full: np.ndarray
    component_depths_full: np.ndarray
    depth_scale: float
    param_mean: np.ndarray
    param_std: np.ndarray
    signal_mean: np.ndarray
    signal_std: np.ndarray
    shape_classes: list[str]
    split_indices: dict[str, np.ndarray]
    target_version: str
    target_transform_summary: dict[str, Any]


def grid_xy(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-0.04, 0.04, width)
    y = np.linspace(-0.01, 0.01, height)
    return np.meshgrid(x, y, indexing="xy")


def dilate_bool(mask: np.ndarray, radius: int = 1, mode: str = "square") -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if mode not in {"square", "cross"}:
        raise ValueError(f"unsupported dilation mode: {mode}")
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    if mode == "square":
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result)
        for dy, dx in offsets:
            y0 = 1 + dy
            x0 = 1 + dx
            expanded |= padded[y0 : y0 + result.shape[0], x0 : x0 + result.shape[1]]
        result = expanded
    return result


def normalized_signed_distance(mask: np.ndarray, clip_radius_px: float = 4.0) -> np.ndarray:
    fg = np.argwhere(mask)
    bg = np.argwhere(~mask)
    sdf = np.zeros(mask.shape, dtype=np.float32)
    if fg.size == 0:
        return sdf
    if bg.size == 0:
        return np.ones(mask.shape, dtype=np.float32)

    def nearest_distance(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
        out = np.empty(points.shape[0], dtype=np.float32)
        target = targets.astype(np.float32)
        chunk = 1024
        for start in range(0, points.shape[0], chunk):
            pts = points[start : start + chunk].astype(np.float32)
            diff = pts[:, None, :] - target[None, :, :]
            out[start : start + chunk] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        return out

    sdf[tuple(fg.T)] = nearest_distance(fg, bg)
    sdf[tuple(bg.T)] = -nearest_distance(bg, fg)
    return np.clip(sdf / float(clip_radius_px), -1.0, 1.0).astype(np.float32)


def build_target_v2(pack: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    exists = np.asarray(pack["component_exists"], dtype=bool)
    masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    centers = np.asarray(pack["component_center_xy_m"], dtype=np.float64)
    lwd = np.asarray(pack["component_lwd_m"], dtype=np.float64)
    n, k, height, width = masks.shape
    xx, yy = grid_xy(height, width)
    ownership = np.full((n, height, width), -1, dtype=np.int16)
    masks_v2 = np.zeros_like(masks, dtype=np.float32)
    depths_v2 = np.zeros_like(depths, dtype=np.float32)
    duplicate_before = 0
    duplicate_after = 0
    raw_overlap_samples = 0
    conflict_before = 0
    conflict_after = 0
    ownership_resolved_pixels = 0
    overlap_resolved_pixels = 0
    by_sample: list[dict[str, Any]] = []
    for i in range(n):
        active_slots = np.where(exists[i])[0]
        comp_sum = masks[i, active_slots].sum(axis=0) if active_slots.size else np.zeros((height, width), dtype=np.int64)
        raw_duplicate_pixels = int(np.maximum(comp_sum - 1, 0).sum())
        raw_overlap_pixel_count = int((comp_sum > 1).sum())
        duplicate_before += raw_duplicate_pixels
        raw_overlap_samples += int(raw_overlap_pixel_count > 0)
        sample_conflict_before = 0
        sample_overlap_resolved = 0
        for y, x in zip(*np.where(comp_sum > 0)):
            candidates = [int(slot) for slot in active_slots if masks[i, slot, y, x]]
            if not candidates:
                continue
            if len(candidates) == 1:
                owner = candidates[0]
            else:
                sample_overlap_resolved += 1
                depth_values = [float(depths[i, slot, y, x]) for slot in candidates if float(depths[i, slot, y, x]) > 0.0]
                if depth_values and (max(depth_values) - min(depth_values)) > 1.0e-12:
                    sample_conflict_before += 1
                scored = []
                px = float(xx[y, x])
                py = float(yy[y, x])
                for slot in candidates:
                    sx = max(float(lwd[i, slot, 0]), 1.0e-9)
                    sy = max(float(lwd[i, slot, 1]), 1.0e-9)
                    dist = math.hypot((px - float(centers[i, slot, 0])) / sx, (py - float(centers[i, slot, 1])) / sy)
                    scored.append((dist, -float(depths[i, slot, y, x]), int(slot)))
                owner = min(scored)[2]
            ownership[i, y, x] = owner
            masks_v2[i, owner, y, x] = 1.0
            depths_v2[i, owner, y, x] = depths[i, owner, y, x]
            ownership_resolved_pixels += 1
        after_sum = masks_v2[i].sum(axis=0)
        sample_duplicate_after = int(np.maximum(after_sum - 1.0, 0.0).sum())
        duplicate_after += sample_duplicate_after
        conflict_before += sample_conflict_before
        overlap_resolved_pixels += sample_overlap_resolved
        by_sample.append(
            {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": i,
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "raw_overlap_pixel_count": raw_overlap_pixel_count,
                "duplicate_ownership_before_v2": raw_duplicate_pixels,
                "duplicate_ownership_after_v2": sample_duplicate_after,
                "overlap_depth_conflict_before_v2": sample_conflict_before,
                "overlap_depth_conflict_after_v2": 0,
                "ownership_resolved_overlap_pixels": sample_overlap_resolved,
            }
        )
    union_v2 = masks_v2.max(axis=1)
    union_raw = np.asarray(pack["projected_mask_2d"], dtype=np.float32)
    union_mismatch = int(np.logical_xor(union_v2 > 0.5, union_raw > 0.5).sum())
    summary = {
        "target_version": "v2",
        "component_mask_target_v2": True,
        "component_depth_target_v2": True,
        "component_ownership_map": True,
        "target_loaded_count": int(n),
        "ownership_resolved_pixel_count": int(ownership_resolved_pixels),
        "ownership_resolved_overlap_pixel_count": int(overlap_resolved_pixels),
        "raw_overlap_sample_count": int(raw_overlap_samples),
        "duplicate_ownership_before_v2": int(duplicate_before),
        "duplicate_ownership_after_v2": int(duplicate_after),
        "overlap_depth_conflict_before_v2": int(conflict_before),
        "overlap_depth_conflict_after_v2": int(conflict_after),
        "union_mask_mismatch_after_v2_px": int(union_mismatch),
        "raw_overlap_diagnostics_retained": True,
        "component_count_3_overlap_samples": int(sum(row["component_count"] == 3 and row["raw_overlap_pixel_count"] > 0 for row in by_sample)),
        "partially_overlapping_overlap_samples": int(sum(row["separation_type"] == "partially_overlapping" and row["raw_overlap_pixel_count"] > 0 for row in by_sample)),
        "by_sample_overlap_diagnostics": by_sample,
        "rule": "ownership-resolved component targets; union mask/depth remain OR/max diagnostics",
    }
    return masks_v2.astype(np.float32), depths_v2.astype(np.float32), ownership, summary


def build_target_v3(pack: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    masks_v2, _depths_v2, ownership, v2_summary = build_target_v2(pack)
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    raw_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    n, k, height, width = raw_masks.shape
    soft = np.zeros((n, k, height, width), dtype=np.float32)
    sdf = np.zeros_like(soft, dtype=np.float32)
    valid = np.zeros_like(raw_masks, dtype=bool)
    depth_v3 = np.zeros_like(raw_depths, dtype=np.float32)
    depth_valid = np.zeros_like(raw_masks, dtype=bool)
    overlap_region = raw_masks.sum(axis=1) > 1
    contact_boundary = np.zeros((n, height, width), dtype=bool)
    rows: list[dict[str, Any]] = []

    for i in range(n):
        active_slots = [int(slot) for slot in np.where(exists[i])[0]]
        for slot in active_slots:
            raw = raw_masks[i, slot]
            owned = ownership[i] == slot
            band1 = dilate_bool(raw, radius=1)
            band2 = dilate_bool(raw, radius=2)
            component_soft = np.zeros((height, width), dtype=np.float32)
            component_soft[band2] = 0.25
            component_soft[band1] = 0.50
            component_soft[raw] = 0.80
            component_soft[owned] = 1.00
            soft[i, slot] = component_soft
            sdf[i, slot] = normalized_signed_distance(raw)
            valid[i, slot] = band2
            depth_v3[i, slot, raw] = raw_depths[i, slot, raw]
            depth_valid[i, slot] = raw
            hard_px = int(masks_v2[i, slot].sum())
            soft_px = int((component_soft > 0.0).sum())
            rows.append(
                {
                    "sample_id": str(pack["sample_ids"][i]),
                    "source_index": int(i),
                    "slot": int(slot),
                    "split": str(pack["split"][i]),
                    "component_count": int(pack["component_count"][i]),
                    "separation_type": str(pack["separation_type"][i]),
                    "topology_relation": str(pack["topology_relation"][i]),
                    "v2_hard_foreground_px": hard_px,
                    "v3_soft_positive_px": soft_px,
                    "v3_valid_region_px": int(valid[i, slot].sum()),
                    "v3_depth_valid_px": int(depth_valid[i, slot].sum()),
                    "v3_vs_v2_positive_ratio": float(soft_px / max(hard_px, 1)),
                    "existing_v3_empty_or_tiny": bool(soft_px < 20),
                }
            )
        for left_index, left in enumerate(active_slots):
            left_band = dilate_bool(raw_masks[i, left], radius=1)
            for right in active_slots[left_index + 1 :]:
                right_band = dilate_bool(raw_masks[i, right], radius=1)
                contact_boundary[i] |= left_band & right_band
        contact_boundary[i] &= ~overlap_region[i]

    inactive = ~exists
    inactive_violations = int(np.logical_or(soft[inactive] > 0.0, valid[inactive]).sum()) if inactive.any() else 0
    ratios = [float(row["v3_vs_v2_positive_ratio"]) for row in rows]
    soft_px_values = [int(row["v3_soft_positive_px"]) for row in rows]
    valid_px_values = [int(row["v3_valid_region_px"]) for row in rows]
    depth_valid_px_values = [int(row["v3_depth_valid_px"]) for row in rows]
    summary = {
        **v2_summary,
        "target_version": "v3",
        "component_mask_target_v2": False,
        "component_depth_target_v2": False,
        "component_mask_target_v3_soft": True,
        "component_sdf_target_v3": True,
        "component_valid_region_mask": True,
        "component_depth_target_v3": True,
        "component_depth_valid_region_mask": True,
        "overlap_region_mask": True,
        "contact_boundary_mask": True,
        "raw_component_mask_raw": True,
        "component_ownership_map": True,
        "v3_target_loaded_count": int(n),
        "v3_active_component_count": int(len(rows)),
        "v3_soft_support_pixel_mean": float(np.mean(soft_px_values)) if soft_px_values else math.nan,
        "v3_soft_support_pixel_min": int(min(soft_px_values)) if soft_px_values else 0,
        "v3_valid_region_pixel_mean": float(np.mean(valid_px_values)) if valid_px_values else math.nan,
        "v3_valid_region_pixel_min": int(min(valid_px_values)) if valid_px_values else 0,
        "v3_depth_valid_region_pixel_mean": float(np.mean(depth_valid_px_values)) if depth_valid_px_values else math.nan,
        "v3_depth_valid_region_pixel_min": int(min(depth_valid_px_values)) if depth_valid_px_values else 0,
        "v3_vs_v2_support_ratio_mean": float(np.mean(ratios)) if ratios else math.nan,
        "v3_vs_v2_support_ratio_min": float(np.min(ratios)) if ratios else math.nan,
        "empty_slot_violation_count": inactive_violations,
        "existing_v3_empty_or_tiny_count": int(sum(bool(row["existing_v3_empty_or_tiny"]) for row in rows)),
        "overlap_region_pixel_sum": int(overlap_region.sum()),
        "contact_boundary_pixel_sum": int(contact_boundary.sum()),
        "label_v3_training_rule": "25.10 loss weights; soft/SDF/valid-region component supervision; union mask/depth not weighted in training",
        "by_component_rows": rows,
    }
    return (
        {
            "soft_mask": soft,
            "sdf": sdf,
            "valid_region": valid.astype(np.float32),
            "depth": depth_v3,
            "depth_valid": depth_valid.astype(np.float32),
        },
        summary,
    )


def build_contact_boundary(raw_masks: np.ndarray, exists: np.ndarray, overlap_region: np.ndarray) -> np.ndarray:
    n, _k, height, width = raw_masks.shape
    contact = np.zeros((n, height, width), dtype=bool)
    for i in range(n):
        active_slots = [int(slot) for slot in np.where(exists[i])[0]]
        for left_index, left in enumerate(active_slots):
            left_band = dilate_bool(raw_masks[i, left], radius=1, mode="cross")
            for right in active_slots[left_index + 1 :]:
                right_band = dilate_bool(raw_masks[i, right], radius=1, mode="cross")
                contact[i] |= left_band & right_band
        contact[i] &= ~overlap_region[i]
    return contact


def fill_halo_depth_from_nearest_core(core: np.ndarray, halo: np.ndarray, raw_depth: np.ndarray) -> np.ndarray:
    depth = np.zeros(raw_depth.shape, dtype=np.float32)
    depth[core] = raw_depth[core]
    core_points = np.argwhere(core)
    if core_points.size == 0:
        return depth
    core_values = raw_depth[tuple(core_points.T)].astype(np.float32)
    for y, x in np.argwhere(halo):
        delta = core_points - np.array([y, x], dtype=np.int64)
        nearest = int(np.argmin(np.sum(delta * delta, axis=1)))
        depth[y, x] = core_values[nearest]
    return depth


def cap_halo_candidates(candidates: np.ndarray, raw_union_px: int) -> np.ndarray:
    capped = np.zeros_like(candidates, dtype=bool)
    total_candidate = int(candidates.sum())
    if total_candidate == 0:
        return capped
    allowed = int(math.floor(raw_union_px * V3B_HALO_CAP_FRACTION_OF_RAW_UNION))
    allowed = max(0, min(allowed, total_candidate))
    if allowed == 0:
        return capped
    counts = candidates.reshape(candidates.shape[0], -1).sum(axis=1)
    quotas = np.floor(allowed * counts / max(total_candidate, 1)).astype(int)
    active_candidate_count = int((counts > 0).sum())
    if allowed >= active_candidate_count:
        quotas = np.where((counts > 0) & (quotas == 0), 1, quotas)
    while int(quotas.sum()) > allowed:
        slot = int(np.argmax(quotas))
        quotas[slot] -= 1
    while int(quotas.sum()) < allowed:
        remainders = allowed * counts / max(total_candidate, 1) - quotas
        remainders[counts == 0] = -1.0
        slot = int(np.argmax(remainders))
        if remainders[slot] < 0.0:
            break
        quotas[slot] += 1
    for slot, quota in enumerate(quotas):
        if quota <= 0:
            continue
        coords = np.argwhere(candidates[slot])
        for y, x in coords[: int(quota)]:
            capped[slot, y, x] = True
    return capped


def build_target_v3b(pack: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    masks_v2, _depths_v2, ownership, v2_summary = build_target_v2(pack)
    exists = np.asarray(pack["component_exists"], dtype=bool)
    raw_masks = np.asarray(pack["component_projected_masks_2d"], dtype=bool)
    raw_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    n, k, height, width = raw_masks.shape
    hard_core = masks_v2.astype(bool)
    boundary_halo = np.zeros((n, k, height, width), dtype=bool)
    ignore_overlap = np.zeros((n, k, height, width), dtype=bool)
    soft = np.zeros((n, k, height, width), dtype=np.float32)
    sdf = np.zeros((n, k, height, width), dtype=np.float32)
    valid = np.zeros((n, k, height, width), dtype=bool)
    depth_v3b = np.zeros((n, k, height, width), dtype=np.float32)
    depth_valid = np.zeros((n, k, height, width), dtype=bool)
    overlap_region = raw_masks.sum(axis=1) > 1
    contact_boundary = build_contact_boundary(raw_masks, exists, overlap_region)
    identity_conflict = np.zeros((n, height, width), dtype=bool)
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for i in range(n):
        active_slots = [int(slot) for slot in np.where(exists[i])[0]]
        raw_union = raw_masks[i, active_slots].max(axis=0) if active_slots else np.zeros((height, width), dtype=bool)
        preliminary_halo = np.zeros((k, height, width), dtype=bool)
        for slot in active_slots:
            other_slots = [other for other in active_slots if other != slot]
            other_raw = raw_masks[i, other_slots].max(axis=0) if other_slots else np.zeros((height, width), dtype=bool)
            candidate = dilate_bool(hard_core[i, slot], radius=1, mode="cross") & ~hard_core[i, slot]
            candidate &= ~other_raw
            candidate &= ~overlap_region[i]
            preliminary_halo[slot] = candidate
        halo_claims = preliminary_halo.sum(axis=0)
        identity_conflict[i] = overlap_region[i] | contact_boundary[i] | (halo_claims > 1)
        exclusive_halo = preliminary_halo & (halo_claims[None, :, :] == 1)
        exclusive_halo &= ~identity_conflict[i][None, :, :]
        capped_halo = cap_halo_candidates(exclusive_halo, int(raw_union.sum()))
        boundary_halo[i] = capped_halo
        for slot in active_slots:
            ignored = (raw_masks[i, slot] | preliminary_halo[slot]) & identity_conflict[i] & ~hard_core[i, slot]
            ignore_overlap[i, slot] = ignored
            soft[i, slot, hard_core[i, slot]] = 1.0
            soft[i, slot, capped_halo[slot]] = 0.35
            valid[i, slot] = hard_core[i, slot] | capped_halo[slot]
            sdf[i, slot] = normalized_signed_distance(hard_core[i, slot])
            depth_valid[i, slot] = valid[i, slot]
            depth_v3b[i, slot] = fill_halo_depth_from_nearest_core(hard_core[i, slot], capped_halo[slot], raw_depths[i, slot])
            hard_px = int(hard_core[i, slot].sum())
            halo_px = int(capped_halo[slot].sum())
            soft_px = int((soft[i, slot] > 0.0).sum())
            rows.append(
                {
                    "sample_id": str(pack["sample_ids"][i]),
                    "source_index": int(i),
                    "slot": int(slot),
                    "split": str(pack["split"][i]),
                    "component_count": int(pack["component_count"][i]),
                    "separation_type": str(pack["separation_type"][i]),
                    "topology_relation": str(pack["topology_relation"][i]),
                    "v3b_hard_core_px": hard_px,
                    "v3b_boundary_halo_px": halo_px,
                    "v3b_soft_support_px": soft_px,
                    "v3b_valid_region_px": int(valid[i, slot].sum()),
                    "v3b_depth_valid_px": int(depth_valid[i, slot].sum()),
                    "v3b_soft_vs_hard_ratio": float(soft_px / max(hard_px, 1)),
                    "existing_v3b_empty_or_tiny": bool(soft_px < V3B_SPARSE_LOWER_BOUND_PX),
                    "existing_v3b_hard_core_empty": bool(hard_px == 0),
                    "existing_v3b_depth_valid_empty": bool(int(depth_valid[i, slot].sum()) == 0),
                }
            )
        soft_or = (soft[i, active_slots] > 0.0).max(axis=0) if active_slots else np.zeros((height, width), dtype=bool)
        hard_or = hard_core[i, active_slots].max(axis=0) if active_slots else np.zeros((height, width), dtype=bool)
        sample_rows.append(
            {
                "sample_id": str(pack["sample_ids"][i]),
                "source_index": int(i),
                "split": str(pack["split"][i]),
                "component_count": int(pack["component_count"][i]),
                "separation_type": str(pack["separation_type"][i]),
                "topology_relation": str(pack["topology_relation"][i]),
                "raw_union_px": int(raw_union.sum()),
                "v3b_hard_or_px": int(hard_or.sum()),
                "v3b_soft_or_px": int(soft_or.sum()),
                "v3b_soft_or_to_raw_union_ratio": float(soft_or.sum() / max(int(raw_union.sum()), 1)),
                "v3b_identity_conflict_px": int(identity_conflict[i].sum()),
                "v3b_ignore_overlap_px": int(ignore_overlap[i, active_slots].max(axis=0).sum()) if active_slots else 0,
                "v3b_hard_duplicate_px": int(np.maximum(hard_core[i, active_slots].sum(axis=0) - 1, 0).sum()) if active_slots else 0,
                "v3b_depth_valid_duplicate_px": int(np.maximum(depth_valid[i, active_slots].sum(axis=0) - 1, 0).sum()) if active_slots else 0,
                "raw_or_to_v3b_hard_or_mismatch_px": int(np.logical_xor(raw_union, hard_or).sum()),
            }
        )

    inactive = ~exists
    inactive_violations = int(np.logical_or.reduce((soft[inactive] > 0.0, valid[inactive], depth_valid[inactive], ignore_overlap[inactive])).sum()) if inactive.any() else 0
    hard_px_values = [int(row["v3b_hard_core_px"]) for row in rows]
    halo_px_values = [int(row["v3b_boundary_halo_px"]) for row in rows]
    soft_px_values = [int(row["v3b_soft_support_px"]) for row in rows]
    valid_px_values = [int(row["v3b_valid_region_px"]) for row in rows]
    depth_valid_px_values = [int(row["v3b_depth_valid_px"]) for row in rows]
    support_ratios = [float(row["v3b_soft_vs_hard_ratio"]) for row in rows]
    soft_or_ratios = [float(row["v3b_soft_or_to_raw_union_ratio"]) for row in sample_rows]
    summary = {
        **v2_summary,
        "target_version": "v3b",
        "component_mask_target_v2": False,
        "component_depth_target_v2": False,
        "component_mask_target_v3_soft": False,
        "component_sdf_target_v3": False,
        "component_valid_region_mask": False,
        "component_depth_target_v3": False,
        "component_hard_core_mask_v3b": True,
        "component_boundary_halo_mask_v3b": True,
        "component_ignore_overlap_mask_v3b": True,
        "component_mask_target_v3b_soft": True,
        "component_sdf_target_v3b": True,
        "component_valid_region_mask_v3b": True,
        "component_depth_target_v3b": True,
        "component_depth_valid_region_mask_v3b": True,
        "component_identity_conflict_mask_v3b": True,
        "raw_component_mask_raw": True,
        "component_ownership_map": True,
        "v3b_target_loaded_count": int(n),
        "v3b_active_component_count": int(len(rows)),
        "v3b_hard_core_pixel_mean": float(np.mean(hard_px_values)) if hard_px_values else math.nan,
        "v3b_hard_core_pixel_min": int(min(hard_px_values)) if hard_px_values else 0,
        "v3b_boundary_halo_pixel_mean": float(np.mean(halo_px_values)) if halo_px_values else math.nan,
        "v3b_boundary_halo_pixel_min": int(min(halo_px_values)) if halo_px_values else 0,
        "v3b_soft_support_pixel_mean": float(np.mean(soft_px_values)) if soft_px_values else math.nan,
        "v3b_soft_support_pixel_min": int(min(soft_px_values)) if soft_px_values else 0,
        "v3b_valid_region_pixel_mean": float(np.mean(valid_px_values)) if valid_px_values else math.nan,
        "v3b_valid_region_pixel_min": int(min(valid_px_values)) if valid_px_values else 0,
        "v3b_depth_valid_region_pixel_mean": float(np.mean(depth_valid_px_values)) if depth_valid_px_values else math.nan,
        "v3b_depth_valid_region_pixel_min": int(min(depth_valid_px_values)) if depth_valid_px_values else 0,
        "v3b_soft_vs_hard_support_ratio_mean": float(np.mean(support_ratios)) if support_ratios else math.nan,
        "v3b_soft_or_raw_union_ratio_mean": float(np.mean(soft_or_ratios)) if soft_or_ratios else math.nan,
        "v3b_soft_or_raw_union_ratio_max": float(np.max(soft_or_ratios)) if soft_or_ratios else math.nan,
        "v3b_empty_slot_violation_count": inactive_violations,
        "empty_slot_violation_count": inactive_violations,
        "existing_v3b_empty_or_tiny_count": int(sum(bool(row["existing_v3b_empty_or_tiny"]) for row in rows)),
        "existing_v3b_hard_core_empty_count": int(sum(bool(row["existing_v3b_hard_core_empty"]) for row in rows)),
        "existing_v3b_depth_valid_empty_count": int(sum(bool(row["existing_v3b_depth_valid_empty"]) for row in rows)),
        "v3b_identity_conflict_pixel_sum": int(identity_conflict.sum()),
        "v3b_ignore_overlap_pixel_sum": int(ignore_overlap.sum()),
        "v3b_hard_duplicate_pixel_sum": int(sum(row["v3b_hard_duplicate_px"] for row in sample_rows)),
        "v3b_depth_valid_duplicate_pixel_sum": int(sum(row["v3b_depth_valid_duplicate_px"] for row in sample_rows)),
        "v3b_soft_or_union_like_sample_count": int(sum(float(row["v3b_soft_or_to_raw_union_ratio"]) >= V3B_SOFT_OR_RAW_UNION_RATIO_TARGET for row in sample_rows)),
        "v3b_raw_union_mismatch_pixel_sum": int(sum(row["raw_or_to_v3b_hard_or_mismatch_px"] for row in sample_rows)),
        "label_v3b_training_rule": "25.10 loss weights; hard-core/halo/SDF valid-region component supervision; union mask/depth evaluation-only and unweighted",
        "by_component_rows": rows,
        "by_sample_rows": sample_rows,
    }
    return (
        {
            "soft_mask": soft.astype(np.float32),
            "hard_core": hard_core.astype(np.float32),
            "boundary_halo": boundary_halo.astype(np.float32),
            "ignore_overlap": ignore_overlap.astype(np.float32),
            "identity_conflict": identity_conflict.astype(np.float32),
            "sdf": sdf.astype(np.float32),
            "valid_region": valid.astype(np.float32),
            "depth": depth_v3b.astype(np.float32),
            "depth_valid": depth_valid.astype(np.float32),
        },
        summary,
    )


def build_arrays(pack: dict[str, Any], target_version: str = "v1") -> Arrays:
    n = int(pack["sample_ids"].shape[0])
    split = np.asarray(pack["split"]).astype(str)
    split_counts = {name: int(np.sum(split == name)) for name in ["train", "val", "test"]}
    if split_counts != TARGET_SPLIT:
        raise ValueError(f"NPZ split mismatch: {split_counts}")
    if int(np.asarray(pack["K_max"])[0]) != K_MAX:
        raise ValueError(f"NPZ K_max mismatch: {pack['K_max']}")
    if tuple(pack["delta_b"].shape) != (n, 3, 3, 201):
        raise ValueError(f"delta_b shape mismatch: {pack['delta_b'].shape}")
    for key in ["component_exists", "component_center_xy_m", "component_lwd_m", "component_rotation_angle", "component_projected_masks_2d", "component_depth_grids_m"]:
        if pack[key].shape[1] != K_MAX:
            raise ValueError(f"{key} does not have K={K_MAX}")
    if not np.isfinite(pack["delta_b"]).all():
        raise ValueError("delta_b contains non-finite values")

    train_idx = np.where(split == "train")[0].astype(np.int64)
    signal_raw = np.asarray(pack["delta_b"], dtype=np.float32).reshape(n, 9, 201)
    signal_mean = signal_raw[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    signal_std = safe_std(signal_raw[train_idx].reshape(train_idx.size, -1)).reshape(1, 9, 201)
    signal = ((signal_raw - signal_mean) / signal_std).astype(np.float32)

    exists = np.asarray(pack["component_exists"], dtype=bool)
    centers = np.asarray(pack["component_center_xy_m"], dtype=np.float32)
    lwd = np.asarray(pack["component_lwd_m"], dtype=np.float32)
    rotation = np.asarray(pack["component_rotation_angle"], dtype=np.float32)
    params_phys = np.concatenate(
        [
            centers,
            lwd,
            np.sin(rotation)[..., None],
            np.cos(rotation)[..., None],
        ],
        axis=-1,
    ).astype(np.float32)
    train_active = exists[train_idx]
    active_params = params_phys[train_idx][train_active]
    param_mean = active_params.mean(axis=0).astype(np.float32)
    param_std = safe_std(active_params, floor=1.0e-5)
    params_norm = ((params_phys - param_mean.reshape(1, 1, -1)) / param_std.reshape(1, 1, -1)).astype(np.float32)

    families = np.asarray(pack["component_shape_family"]).astype(str)
    shape_classes = sorted(name for name in set(families.reshape(-1).tolist()) if name != "none")
    shape_to_idx = {name: idx for idx, name in enumerate(shape_classes)}
    shape_target = np.zeros((n, K_MAX), dtype=np.int64)
    for i in range(n):
        for slot in range(K_MAX):
            if exists[i, slot]:
                shape_target[i, slot] = shape_to_idx[str(families[i, slot])]

    sdf_full = np.zeros_like(np.asarray(pack["component_projected_masks_2d"], dtype=np.float32))
    mask_valid_full = np.ones_like(sdf_full, dtype=np.float32)
    depth_valid_full = np.asarray(pack["component_projected_masks_2d"], dtype=np.float32)
    if target_version == "v2":
        component_masks_full, component_depths_full, _ownership, target_summary = build_target_v2(pack)
        training_masks_full = component_masks_full
        training_depths_full = component_depths_full
        mask_valid_full = (component_masks_full > 0.0).astype(np.float32)
        sdf_full = 2.0 * component_masks_full - 1.0
        eval_component_masks_full = component_masks_full
        eval_component_depths_full = component_depths_full
    elif target_version == "v3":
        v3_targets, target_summary = build_target_v3(pack)
        training_masks_full = v3_targets["soft_mask"].astype(np.float32)
        training_depths_full = v3_targets["depth"].astype(np.float32)
        sdf_full = v3_targets["sdf"].astype(np.float32)
        mask_valid_full = v3_targets["valid_region"].astype(np.float32)
        depth_valid_full = v3_targets["depth_valid"].astype(np.float32)
        eval_component_masks_full = np.asarray(pack["component_projected_masks_2d"], dtype=np.float32)
        eval_component_depths_full = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
    elif target_version == "v3b":
        v3b_targets, target_summary = build_target_v3b(pack)
        training_masks_full = v3b_targets["soft_mask"].astype(np.float32)
        training_depths_full = v3b_targets["depth"].astype(np.float32)
        sdf_full = v3b_targets["sdf"].astype(np.float32)
        mask_valid_full = v3b_targets["valid_region"].astype(np.float32)
        depth_valid_full = v3b_targets["depth_valid"].astype(np.float32)
        eval_component_masks_full = v3b_targets["hard_core"].astype(np.float32)
        eval_component_depths_full = v3b_targets["depth"].astype(np.float32)
    else:
        training_masks_full = np.asarray(pack["component_projected_masks_2d"], dtype=np.float32)
        training_depths_full = np.asarray(pack["component_depth_grids_m"], dtype=np.float32)
        eval_component_masks_full = training_masks_full
        eval_component_depths_full = training_depths_full
        sdf_full = 2.0 * training_masks_full - 1.0
        mask_valid_full = np.ones_like(training_masks_full, dtype=np.float32)
        depth_valid_full = training_masks_full
        target_summary = {
            "target_version": "v1",
            "component_mask_target_v2": False,
            "component_depth_target_v2": False,
            "component_ownership_map": False,
            "target_loaded_count": int(n),
        }
    mask_low = downsample_np(training_masks_full.astype(np.float32))
    sdf_low = downsample_np(sdf_full.astype(np.float32))
    mask_valid_low = (downsample_np(mask_valid_full.astype(np.float32)) > 0.0).astype(np.float32)
    depth_valid_low = (downsample_np(depth_valid_full.astype(np.float32)) > 0.0).astype(np.float32)
    depth_low = downsample_np(training_depths_full.astype(np.float32))
    active_depth = depth_low[exists]
    depth_scale = float(np.percentile(active_depth[active_depth > 0], 99)) if np.any(active_depth > 0) else 1.0
    depth_scale = max(depth_scale, 1.0e-6)
    depth_low_norm = (depth_low / depth_scale).astype(np.float32)
    sensor_z = np.full((n, 1), float(np.asarray(pack["sensor_z_m"])[0]), dtype=np.float32)
    sensor_z_norm = ((sensor_z - sensor_z[train_idx].mean(axis=0)) / max(float(sensor_z[train_idx].std()), 1.0)).astype(np.float32)

    split_indices = {name: np.where(split == name)[0].astype(np.int64) for name in ["train", "val", "test"]}
    return Arrays(
        raw=pack,
        signal=signal,
        sensor_z_norm=sensor_z_norm,
        exists=exists.astype(np.float32),
        params_phys=params_phys,
        params_norm=params_norm,
        shape_target=shape_target,
        mask_low=mask_low,
        sdf_low=sdf_low,
        mask_valid_low=mask_valid_low,
        depth_low_norm=depth_low_norm,
        depth_valid_low=depth_valid_low,
        component_masks_full=eval_component_masks_full.astype(np.float32),
        component_depths_full=eval_component_depths_full.astype(np.float32),
        depth_scale=depth_scale,
        param_mean=param_mean,
        param_std=param_std,
        signal_mean=signal_mean,
        signal_std=signal_std,
        shape_classes=shape_classes,
        split_indices=split_indices,
        target_version=target_version,
        target_transform_summary=target_summary,
    )


class ComponentSetDataset(Dataset):
    def __init__(self, arrays: Arrays, indices: np.ndarray):
        self.arrays = arrays
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        return {
            "index": torch.tensor(i, dtype=torch.long),
            "signal": torch.from_numpy(self.arrays.signal[i]).float(),
            "sensor_z": torch.from_numpy(self.arrays.sensor_z_norm[i]).float(),
            "exists": torch.from_numpy(self.arrays.exists[i]).float(),
            "params": torch.from_numpy(self.arrays.params_norm[i]).float(),
            "shape": torch.from_numpy(self.arrays.shape_target[i]).long(),
            "mask": torch.from_numpy(self.arrays.mask_low[i]).float(),
            "sdf": torch.from_numpy(self.arrays.sdf_low[i]).float(),
            "mask_valid": torch.from_numpy(self.arrays.mask_valid_low[i]).float(),
            "depth": torch.from_numpy(self.arrays.depth_low_norm[i]).float(),
            "depth_valid": torch.from_numpy(self.arrays.depth_valid_low[i]).float(),
            "label_v3": torch.tensor(1.0 if self.arrays.target_version in {"v3", "v3b"} else 0.0, dtype=torch.float32),
        }


class ComponentSetGateModel(nn.Module):
    def __init__(self, shape_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.sensor = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(64 * 16 + 16, 384),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(384, 256),
            nn.GELU(),
        )
        self.exist_head = nn.Linear(256, K_MAX)
        self.param_head = nn.Linear(256, K_MAX * 7)
        self.shape_head = nn.Linear(256, K_MAX * shape_classes)
        self.raster_head = nn.Linear(256, K_MAX * 2 * LOW_H * LOW_W)
        self.shape_classes = shape_classes

    def forward(self, signal: torch.Tensor, sensor_z: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encoder(signal).flatten(1)
        x = torch.cat([x, self.sensor(sensor_z)], dim=1)
        h = self.trunk(x)
        raster = self.raster_head(h).view(-1, K_MAX, 2, LOW_H, LOW_W)
        return {
            "exist_logits": self.exist_head(h),
            "params": self.param_head(h).view(-1, K_MAX, 7),
            "shape_logits": self.shape_head(h).view(-1, K_MAX, self.shape_classes),
            "mask_logits": raster[:, :, 0],
            "depth_raw": raster[:, :, 1],
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def loss_terms_for_perm(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], perm: tuple[int, ...], loss_config: LossConfig) -> dict[str, torch.Tensor]:
    exists = batch["exists"][:, perm]
    params = batch["params"][:, perm]
    shape = batch["shape"][:, perm]
    mask = batch["mask"][:, perm]
    depth = batch["depth"][:, perm]
    sdf = batch["sdf"][:, perm]
    mask_valid = batch["mask_valid"][:, perm]
    depth_valid = batch["depth_valid"][:, perm]
    label_v3 = bool(batch.get("label_v3", torch.zeros(1, device=exists.device))[0].detach().item() > 0.5)
    b = exists.shape[0]
    active_count = exists.sum(dim=1).clamp_min(1.0)

    exist_loss = F.binary_cross_entropy_with_logits(pred["exist_logits"], exists, reduction="none").mean(dim=1)

    param_raw = F.smooth_l1_loss(pred["params"], params, reduction="none").mean(dim=-1)
    param_loss = (param_raw * exists).sum(dim=1) / active_count

    shape_flat = F.cross_entropy(
        pred["shape_logits"].reshape(b * K_MAX, -1),
        shape.reshape(b * K_MAX),
        reduction="none",
    ).view(b, K_MAX)
    shape_loss = (shape_flat * exists).sum(dim=1) / active_count

    depth_pred = F.softplus(pred["depth_raw"])
    label_v3_sdf_loss = torch.zeros_like(exist_loss)
    label_v3_soft_bce_loss = torch.zeros_like(exist_loss)
    label_v3_soft_dice_loss = torch.zeros_like(exist_loss)
    label_v3_valid_depth_loss = torch.zeros_like(exist_loss)

    if loss_config.name in {"mask_depth_rebalance_v1", "component_separation_rebalance_v1"}:
        mask_bce = balanced_bce_from_logits(pred["mask_logits"], mask)
        mask_dice = soft_dice_loss(pred["mask_logits"], mask)
        component_mask_loss = ((mask_bce + mask_dice) * exists).sum(dim=1) / active_count

        mask_prob = torch.sigmoid(pred["mask_logits"])
        active_view = exists.unsqueeze(-1).unsqueeze(-1)
        target_union_mask = mask.max(dim=1).values
        pred_union_prob = (mask_prob * active_view).max(dim=1).values
        union_mask_loss = balanced_bce_from_prob(pred_union_prob, target_union_mask) + soft_dice_loss_prob(pred_union_prob, target_union_mask)

        valid = (mask > 0.5).float()
        background = 1.0 - valid
        component_depth_raw = valid_region_smooth_l1(depth_pred, depth, valid)
        component_depth_loss = (component_depth_raw * exists).sum(dim=1) / active_count
        component_depth_background_raw = valid_region_smooth_l1(depth_pred, depth, background)
        component_depth_background_loss = (component_depth_background_raw * exists).sum(dim=1) / active_count

        target_union_depth = depth.max(dim=1).values
        pred_union_depth = (depth_pred * mask_prob * active_view).max(dim=1).values
        valid_union = (target_union_mask > 0.5).float()
        background_union = 1.0 - valid_union
        union_depth_loss = valid_region_smooth_l1(pred_union_depth, target_union_depth, valid_union)
        union_depth_background_loss = valid_region_smooth_l1(pred_union_depth, target_union_depth, background_union)

        if loss_config.name == "component_separation_rebalance_v1":
            separation_loss, merge_loss = pairwise_component_losses(pred["params"], pred["mask_logits"], params, mask, exists)
        else:
            separation_loss = torch.zeros_like(component_mask_loss)
            merge_loss = torch.zeros_like(component_mask_loss)
    elif label_v3:
        valid_mask = (mask_valid > 0.0).float()
        valid_depth = (depth_valid > 0.0).float()
        mask_prob = torch.sigmoid(pred["mask_logits"])
        mask_bce_raw = F.binary_cross_entropy_with_logits(pred["mask_logits"], mask, reduction="none")
        mask_bce = (mask_bce_raw * valid_mask).sum(dim=(-2, -1)) / valid_mask.sum(dim=(-2, -1)).clamp_min(1.0e-6)
        mask_dice = soft_dice_loss_prob(mask_prob * valid_mask, mask * valid_mask)
        sdf_pred = torch.tanh(pred["mask_logits"] / 3.0)
        sdf_raw = F.smooth_l1_loss(sdf_pred, sdf, reduction="none")
        sdf_loss = (sdf_raw * valid_mask).sum(dim=(-2, -1)) / valid_mask.sum(dim=(-2, -1)).clamp_min(1.0e-6)
        component_mask_raw = mask_bce + mask_dice + 0.25 * sdf_loss
        component_mask_loss = (component_mask_raw * exists).sum(dim=1) / active_count
        union_mask_loss = torch.zeros_like(component_mask_loss)

        component_depth_raw = valid_region_smooth_l1(depth_pred, depth, valid_depth)
        component_depth_loss = (component_depth_raw * exists).sum(dim=1) / active_count
        union_depth_loss = torch.zeros_like(component_depth_loss)
        component_depth_background_loss = torch.zeros_like(component_depth_loss)
        union_depth_background_loss = torch.zeros_like(component_depth_loss)
        separation_loss = torch.zeros_like(component_depth_loss)
        merge_loss = torch.zeros_like(component_depth_loss)
        label_v3_sdf_loss = (sdf_loss * exists).sum(dim=1) / active_count
        label_v3_soft_bce_loss = (mask_bce * exists).sum(dim=1) / active_count
        label_v3_soft_dice_loss = (mask_dice * exists).sum(dim=1) / active_count
        label_v3_valid_depth_loss = component_depth_loss
    else:
        mask_bce = F.binary_cross_entropy_with_logits(pred["mask_logits"], mask, reduction="none").mean(dim=(-2, -1))
        mask_dice = soft_dice_loss(pred["mask_logits"], mask)
        component_mask_loss = ((mask_bce + mask_dice) * exists).sum(dim=1) / active_count
        union_mask_loss = torch.zeros_like(component_mask_loss)

        target_weight = 0.15 + 0.85 * mask
        depth_mse = (((depth_pred - depth) ** 2) * target_weight).mean(dim=(-2, -1))
        component_depth_loss = (depth_mse * exists).sum(dim=1) / active_count
        union_depth_loss = torch.zeros_like(component_depth_loss)
        component_depth_background_loss = torch.zeros_like(component_depth_loss)
        union_depth_background_loss = torch.zeros_like(component_depth_loss)
        separation_loss = torch.zeros_like(component_depth_loss)
        merge_loss = torch.zeros_like(component_depth_loss)

    return {
        "exist": exist_loss,
        "param": param_loss,
        "shape": shape_loss,
        "component_mask": component_mask_loss,
        "union_mask": union_mask_loss,
        "component_depth": component_depth_loss,
        "union_depth": union_depth_loss,
        "component_depth_background": component_depth_background_loss,
        "union_depth_background": union_depth_background_loss,
        "separation_penalty": separation_loss,
        "merge_penalty": merge_loss,
        "label_v3_sdf": label_v3_sdf_loss,
        "label_v3_soft_bce": label_v3_soft_bce_loss,
        "label_v3_soft_dice": label_v3_soft_dice_loss,
        "label_v3_valid_depth": label_v3_valid_depth_loss,
    }


def hungarian_loss_breakdown(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], loss_config: LossConfig, epoch: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    per_perm_terms = [loss_terms_for_perm(pred, batch, perm, loss_config) for perm in PERMS]
    weights = scheduled_loss_weights(loss_config, epoch)
    weighted_totals = []
    for terms in per_perm_terms:
        weighted_totals.append(sum(weights[name] * terms[name] for name in weights))
    weighted_stack = torch.stack(weighted_totals, dim=1)
    best_idx = weighted_stack.argmin(dim=1)
    selected_total = weighted_stack.gather(1, best_idx[:, None]).squeeze(1)
    selected_terms: dict[str, torch.Tensor] = {}
    for name in per_perm_terms[0]:
        stack = torch.stack([terms[name] for terms in per_perm_terms], dim=1)
        selected_terms[name] = stack.gather(1, best_idx[:, None]).squeeze(1)
    loss = selected_total.mean()
    unweighted = {name: float(value.detach().mean().cpu()) for name, value in selected_terms.items()}
    weighted = {name: float((selected_terms[name] * weights[name]).detach().mean().cpu()) for name in weights}
    total_weighted = float(sum(weighted.values()))
    mask_depth_weighted = float(sum(weighted[name] for name in MASK_DEPTH_TERMS))
    separation_weighted = float(sum(weighted[name] for name in COMPONENT_SEPARATION_TERMS))
    stats = {
        "loss": float(loss.detach().cpu()),
        "unweighted": unweighted,
        "weighted": weighted,
        "effective_weights": weights,
        "mask_depth_weighted_sum": mask_depth_weighted,
        "component_separation_weighted_sum": separation_weighted,
        "total_weighted_sum": total_weighted,
        "mask_depth_weighted_ratio": mask_depth_weighted / total_weighted if total_weighted > 0.0 else math.nan,
        "component_separation_weighted_ratio": separation_weighted / total_weighted if total_weighted > 0.0 else math.nan,
        "component_mask_to_union_mask_weighted_ratio": weighted["component_mask"] / max(weighted["union_mask"], 1.0e-12),
        "depth_foreground_to_background_unweighted_ratio": (
            (unweighted["component_depth"] + unweighted["union_depth"])
            / max(unweighted["component_depth_background"] + unweighted["union_depth_background"], 1.0e-12)
        ),
    }
    return loss, stats


def hungarian_training_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], loss_config: LossConfig | None = None, epoch: int | None = None) -> torch.Tensor:
    config = loss_config or LOSS_CONFIGS["component_set_gate_v1"]
    loss, _stats = hungarian_loss_breakdown(pred, batch, config, epoch)
    return loss


def predict(model: nn.Module, arrays: Arrays, indices: np.ndarray, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    ds = ComponentSetDataset(arrays, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["signal"].to(device), batch["sensor_z"].to(device))
            chunks["index"].append(batch["index"].numpy())
            chunks["exist_prob"].append(torch.sigmoid(out["exist_logits"]).cpu().numpy())
            chunks["params_norm"].append(out["params"].cpu().numpy())
            chunks["shape_class"].append(torch.argmax(out["shape_logits"], dim=-1).cpu().numpy())
            mask_prob_low = torch.sigmoid(out["mask_logits"])
            depth_low = F.softplus(out["depth_raw"])
            mask_prob = F.interpolate(mask_prob_low.reshape(-1, 1, LOW_H, LOW_W), size=(64, 128), mode="bilinear", align_corners=False)
            depth = F.interpolate(depth_low.reshape(-1, 1, LOW_H, LOW_W), size=(64, 128), mode="bilinear", align_corners=False)
            chunks["mask_prob"].append(mask_prob.reshape(-1, K_MAX, 64, 128).cpu().numpy())
            chunks["depth_norm"].append(depth.reshape(-1, K_MAX, 64, 128).cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def denormalize_params(params_norm: np.ndarray, arrays: Arrays) -> dict[str, np.ndarray]:
    params = params_norm * arrays.param_std.reshape(1, 1, -1) + arrays.param_mean.reshape(1, 1, -1)
    sin_v = params[:, :, 5]
    cos_v = params[:, :, 6]
    return {
        "center": params[:, :, 0:2],
        "lwd": np.maximum(params[:, :, 2:5], 1.0e-6),
        "rotation": np.arctan2(sin_v, cos_v),
    }


def match_components(
    true_centers: np.ndarray,
    true_masks: np.ndarray,
    pred_centers: np.ndarray,
    pred_masks: np.ndarray,
) -> list[tuple[int, int]]:
    if true_centers.shape[0] == 0 or pred_centers.shape[0] == 0:
        return []
    cost = np.zeros((true_centers.shape[0], pred_centers.shape[0]), dtype=np.float64)
    valid = np.zeros_like(cost, dtype=bool)
    for i in range(true_centers.shape[0]):
        for j in range(pred_centers.shape[0]):
            center_err = float(np.linalg.norm(true_centers[i] - pred_centers[j]))
            iou, _dice = mask_iou_dice(pred_masks[j], true_masks[i])
            cost[i, j] = center_err / 0.010 + (1.0 - iou)
            valid[i, j] = center_err <= 0.008 or iou >= 0.04
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if valid[r, c]]


def sample_metrics(arrays: Arrays, pred: dict[str, np.ndarray], threshold: float) -> list[dict[str, Any]]:
    params = denormalize_params(pred["params_norm"], arrays)
    rows: list[dict[str, Any]] = []
    raw = arrays.raw
    index_lookup = {int(src_idx): pos for pos, src_idx in enumerate(pred["index"].tolist())}
    for src_idx in pred["index"].tolist():
        pos = index_lookup[int(src_idx)]
        true_exists = np.asarray(raw["component_exists"][src_idx], dtype=bool)
        pred_exists = pred["exist_prob"][pos] >= threshold
        true_slots = np.where(true_exists)[0]
        pred_slots = np.where(pred_exists)[0]

        true_centers = np.asarray(raw["component_center_xy_m"][src_idx, true_slots], dtype=np.float64)
        true_lwd = np.asarray(raw["component_lwd_m"][src_idx, true_slots], dtype=np.float64)
        true_rot = np.asarray(raw["component_rotation_angle"][src_idx, true_slots], dtype=np.float64)
        true_masks = np.asarray(arrays.component_masks_full[src_idx, true_slots], dtype=np.float64)
        true_depths = np.asarray(arrays.component_depths_full[src_idx, true_slots], dtype=np.float64)

        pred_centers = params["center"][pos, pred_slots]
        pred_lwd = params["lwd"][pos, pred_slots]
        pred_rot = params["rotation"][pos, pred_slots]
        pred_masks_prob = pred["mask_prob"][pos, pred_slots]
        pred_masks = (pred_masks_prob >= 0.5).astype(np.float64)
        pred_depths = pred["depth_norm"][pos, pred_slots] * arrays.depth_scale

        matches = match_components(true_centers, true_masks, pred_centers, pred_masks)
        matched_true = {t for t, _p in matches}
        matched_pred = {p for _t, p in matches}
        missed = len(true_slots) - len(matched_true)
        extra = len(pred_slots) - len(matched_pred)

        center_errors: list[float] = []
        lwd_rel_errors: list[float] = []
        rot_errors: list[float] = []
        comp_ious: list[float] = []
        comp_dices: list[float] = []
        comp_depth_rmses: list[float] = []
        for t, p in matches:
            center_errors.append(float(np.linalg.norm(pred_centers[p] - true_centers[t])))
            lwd_rel_errors.append(float(np.mean(np.abs(pred_lwd[p] - true_lwd[t]) / np.maximum(true_lwd[t], 1.0e-9))))
            rot_errors.append(angle_diff_rad(float(pred_rot[p]), float(true_rot[t])))
            iou, dice = mask_iou_dice(pred_masks[p], true_masks[t])
            comp_ious.append(iou)
            comp_dices.append(dice)
            comp_depth_rmses.append(float(np.sqrt(np.mean((pred_depths[p] - true_depths[t]) ** 2))))

        pred_union_mask = np.max(pred_masks_prob, axis=0) if len(pred_slots) else np.zeros((64, 128), dtype=np.float64)
        pred_union_depth = np.max(pred_depths * pred_masks_prob, axis=0) if len(pred_slots) else np.zeros((64, 128), dtype=np.float64)
        true_union_mask = np.asarray(raw["projected_mask_2d"][src_idx], dtype=np.float64)
        true_union_depth = np.asarray(raw["depth_grid_m"][src_idx], dtype=np.float64)
        union_iou, union_dice = mask_iou_dice(pred_union_mask, true_union_mask)
        depth_rmse = float(np.sqrt(np.mean((pred_union_depth - true_union_depth) ** 2)))

        merged_component = False
        if len(pred_slots):
            for pred_mask in pred_masks:
                overlap_count = 0
                for true_mask in true_masks:
                    intersection = float(np.logical_and(pred_mask > 0.5, true_mask > 0.5).sum())
                    if intersection >= 8.0:
                        overlap_count += 1
                if overlap_count >= 2:
                    merged_component = True
                    break
        if len(pred_slots) < len(true_slots) and union_iou >= 0.08:
            merged_component = True

        rows.append(
            {
                "sample_id": str(raw["sample_ids"][src_idx]),
                "source_index": int(src_idx),
                "split": str(raw["split"][src_idx]),
                "component_count": int(raw["component_count"][src_idx]),
                "separation_type": str(raw["separation_type"][src_idx]),
                "topology_relation": str(raw["topology_relation"][src_idx]),
                "orientation_type": str(raw["orientation_type"][src_idx]),
                "true_component_count": int(len(true_slots)),
                "pred_component_count": int(len(pred_slots)),
                "matched_components": int(len(matches)),
                "missed_components": int(missed),
                "extra_components": int(extra),
                "merged_sample": bool(merged_component),
                "single_component_collapse_sample": bool(len(pred_slots) <= 1 and len(true_slots) >= 2),
                "component_recall": float(len(matches) / max(len(true_slots), 1)),
                "component_precision": float(len(matches) / max(len(pred_slots), 1)) if len(pred_slots) else 0.0,
                "center_error_m_mean": float(np.mean(center_errors)) if center_errors else math.nan,
                "lwd_relative_error_mean": float(np.mean(lwd_rel_errors)) if lwd_rel_errors else math.nan,
                "rotation_error_rad_mean": float(np.mean(rot_errors)) if rot_errors else math.nan,
                "component_mask_iou_mean": float(np.mean(comp_ious)) if comp_ious else 0.0,
                "component_mask_dice_mean": float(np.mean(comp_dices)) if comp_dices else 0.0,
                "component_depth_rmse_m_mean": float(np.mean(comp_depth_rmses)) if comp_depth_rmses else math.nan,
                "union_mask_iou": float(union_iou),
                "union_mask_dice": float(union_dice),
                "depth_grid_rmse_m": float(depth_rmse),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def vals(key: str) -> list[float]:
        out = []
        for row in rows:
            value = row.get(key, math.nan)
            if value is not None and np.isfinite(float(value)):
                out.append(float(value))
        return out

    true_components = sum(int(row["true_component_count"]) for row in rows)
    pred_components = sum(int(row["pred_component_count"]) for row in rows)
    matched = sum(int(row["matched_components"]) for row in rows)
    missed = sum(int(row["missed_components"]) for row in rows)
    extra = sum(int(row["extra_components"]) for row in rows)
    return {
        "sample_count": len(rows),
        "true_components": true_components,
        "pred_components": pred_components,
        "matched_components": matched,
        "component_recall": matched / max(true_components, 1),
        "component_precision": matched / max(pred_components, 1) if pred_components else 0.0,
        "missed_rate": missed / max(true_components, 1),
        "extra_rate": extra / max(pred_components, 1) if pred_components else 0.0,
        "merged_rate": sum(bool(row["merged_sample"]) for row in rows) / len(rows),
        "single_component_collapse_rate": sum(bool(row["single_component_collapse_sample"]) for row in rows) / len(rows),
        "pred_component_count_mean": float(np.mean(vals("pred_component_count"))),
        "center_error_m_mean": float(np.mean(vals("center_error_m_mean"))) if vals("center_error_m_mean") else math.nan,
        "lwd_relative_error_mean": float(np.mean(vals("lwd_relative_error_mean"))) if vals("lwd_relative_error_mean") else math.nan,
        "rotation_error_rad_mean": float(np.mean(vals("rotation_error_rad_mean"))) if vals("rotation_error_rad_mean") else math.nan,
        "component_mask_iou_mean": float(np.mean(vals("component_mask_iou_mean"))),
        "component_mask_dice_mean": float(np.mean(vals("component_mask_dice_mean"))),
        "component_depth_rmse_m_mean": float(np.mean(vals("component_depth_rmse_m_mean"))) if vals("component_depth_rmse_m_mean") else math.nan,
        "union_mask_iou_mean": float(np.mean(vals("union_mask_iou"))),
        "union_mask_dice_mean": float(np.mean(vals("union_mask_dice"))),
        "depth_grid_rmse_m_mean": float(np.mean(vals("depth_grid_rmse_m"))),
    }


def grouped(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for field in ["separation_type", "topology_relation", "component_count"]:
        values = sorted({str(row[field]) for row in rows})
        output[field] = {value: aggregate_rows([row for row in rows if str(row[field]) == value]) for value in values}
    return output


def evaluate_splits(arrays: Arrays, predictions: dict[str, dict[str, np.ndarray]], threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    by_split: dict[str, Any] = {}
    for split, pred in predictions.items():
        rows = sample_metrics(arrays, pred, threshold)
        all_rows.extend(rows)
        by_split[split] = aggregate_rows(rows)
        by_split[split]["groups"] = grouped(rows)
    return by_split, all_rows


def empty_baseline(arrays: Arrays, split: str) -> dict[str, Any]:
    rows = []
    raw = arrays.raw
    for idx in arrays.split_indices[split]:
        true_count = int(raw["component_count"][idx])
        true_union_depth = np.asarray(raw["depth_grid_m"][idx], dtype=np.float64)
        rows.append(
            {
                "true_component_count": true_count,
                "pred_component_count": 0,
                "matched_components": 0,
                "missed_components": true_count,
                "extra_components": 0,
                "merged_sample": False,
                "single_component_collapse_sample": True,
                "center_error_m_mean": math.nan,
                "lwd_relative_error_mean": math.nan,
                "rotation_error_rad_mean": math.nan,
                "component_mask_iou_mean": 0.0,
                "component_mask_dice_mean": 0.0,
                "component_depth_rmse_m_mean": math.nan,
                "union_mask_iou": 0.0,
                "union_mask_dice": 0.0,
                "depth_grid_rmse_m": float(np.sqrt(np.mean(true_union_depth**2))),
            }
        )
    return aggregate_rows(rows)


def one_slot_prior_baseline(arrays: Arrays, split: str) -> dict[str, Any]:
    train_idx = arrays.split_indices["train"]
    active = arrays.raw["component_exists"][train_idx].astype(bool)
    mean_center = arrays.raw["component_center_xy_m"][train_idx][active].mean(axis=0)
    mean_lwd = arrays.raw["component_lwd_m"][train_idx][active].mean(axis=0)
    mean_rot = float(arrays.raw["component_rotation_angle"][train_idx][active].mean())
    mean_mask = arrays.component_masks_full[train_idx][active].mean(axis=0)
    mean_depth = arrays.component_depths_full[train_idx][active].mean(axis=0)
    rows = []
    raw = arrays.raw
    for idx in arrays.split_indices[split]:
        true_slots = np.where(raw["component_exists"][idx].astype(bool))[0]
        true_centers = raw["component_center_xy_m"][idx, true_slots]
        true_lwd = raw["component_lwd_m"][idx, true_slots]
        true_rot = raw["component_rotation_angle"][idx, true_slots]
        true_masks = arrays.component_masks_full[idx, true_slots]
        true_depths = arrays.component_depths_full[idx, true_slots]
        matches = match_components(true_centers, true_masks, mean_center.reshape(1, 2), (mean_mask >= 0.5).reshape(1, 64, 128))
        center_errors = []
        lwd_errors = []
        rot_errors = []
        comp_ious = []
        comp_dices = []
        comp_depth = []
        for t, _p in matches:
            center_errors.append(float(np.linalg.norm(mean_center - true_centers[t])))
            lwd_errors.append(float(np.mean(np.abs(mean_lwd - true_lwd[t]) / np.maximum(true_lwd[t], 1.0e-9))))
            rot_errors.append(angle_diff_rad(mean_rot, float(true_rot[t])))
            iou, dice = mask_iou_dice(mean_mask, true_masks[t])
            comp_ious.append(iou)
            comp_dices.append(dice)
            comp_depth.append(float(np.sqrt(np.mean((mean_depth - true_depths[t]) ** 2))))
        union_iou, union_dice = mask_iou_dice(mean_mask, raw["projected_mask_2d"][idx])
        rows.append(
            {
                "true_component_count": int(len(true_slots)),
                "pred_component_count": 1,
                "matched_components": len(matches),
                "missed_components": int(len(true_slots) - len(matches)),
                "extra_components": int(1 - len(matches)),
                "merged_sample": bool(len(true_slots) > 1 and union_iou >= 0.08),
                "single_component_collapse_sample": True,
                "center_error_m_mean": float(np.mean(center_errors)) if center_errors else math.nan,
                "lwd_relative_error_mean": float(np.mean(lwd_errors)) if lwd_errors else math.nan,
                "rotation_error_rad_mean": float(np.mean(rot_errors)) if rot_errors else math.nan,
                "component_mask_iou_mean": float(np.mean(comp_ious)) if comp_ious else 0.0,
                "component_mask_dice_mean": float(np.mean(comp_dices)) if comp_dices else 0.0,
                "component_depth_rmse_m_mean": float(np.mean(comp_depth)) if comp_depth else math.nan,
                "union_mask_iou": union_iou,
                "union_mask_dice": union_dice,
                "depth_grid_rmse_m": float(np.sqrt(np.mean((mean_depth - raw["depth_grid_m"][idx]) ** 2))),
            }
        )
    return aggregate_rows(rows)


def select_threshold(arrays: Arrays, val_pred: dict[str, np.ndarray]) -> tuple[float, dict[str, Any]]:
    candidates = [0.25, 0.35, 0.45, 0.55, 0.65]
    rows = []
    best_threshold = candidates[0]
    best_score = -math.inf
    best_metrics: dict[str, Any] = {}
    for threshold in candidates:
        split_metrics, _sample_rows = evaluate_splits(arrays, {"val": val_pred}, threshold)
        metrics = split_metrics["val"]
        score = (
            metrics["component_recall"]
            + 0.5 * metrics["union_mask_dice_mean"]
            - 0.35 * metrics["extra_rate"]
            - 0.35 * metrics["single_component_collapse_rate"]
        )
        row = {"threshold": threshold, "selection_score": score, **metrics}
        rows.append(row)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, {"selected_threshold": best_threshold, "rows": rows, "selected_val_metrics": best_metrics}


def average_loss_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"loss": math.nan, "unweighted": {}, "weighted": {}, "mask_depth_weighted_ratio": math.nan}
    total_n = sum(int(record["batch_size"]) for record in records)
    unweighted_names = list(records[0]["unweighted"].keys())
    weighted_names = list(records[0]["weighted"].keys())
    unweighted = {
        name: float(sum(float(record["unweighted"][name]) * int(record["batch_size"]) for record in records) / total_n)
        for name in unweighted_names
    }
    weighted = {
        name: float(sum(float(record["weighted"][name]) * int(record["batch_size"]) for record in records) / total_n)
        for name in weighted_names
    }
    loss = float(sum(float(record["loss"]) * int(record["batch_size"]) for record in records) / total_n)
    total_weighted = float(sum(weighted.values()))
    mask_depth_weighted = float(sum(weighted[name] for name in MASK_DEPTH_TERMS))
    separation_weighted = float(sum(weighted[name] for name in COMPONENT_SEPARATION_TERMS))
    return {
        "loss": loss,
        "unweighted": unweighted,
        "weighted": weighted,
        "mask_depth_weighted_sum": mask_depth_weighted,
        "component_separation_weighted_sum": separation_weighted,
        "total_weighted_sum": total_weighted,
        "mask_depth_weighted_ratio": mask_depth_weighted / total_weighted if total_weighted > 0.0 else math.nan,
        "component_separation_weighted_ratio": separation_weighted / total_weighted if total_weighted > 0.0 else math.nan,
        "component_mask_to_union_mask_weighted_ratio": weighted.get("component_mask", 0.0) / max(weighted.get("union_mask", 0.0), 1.0e-12),
        "depth_foreground_to_background_unweighted_ratio": (
            (unweighted.get("component_depth", 0.0) + unweighted.get("union_depth", 0.0))
            / max(unweighted.get("component_depth_background", 0.0) + unweighted.get("union_depth_background", 0.0), 1.0e-12)
        ),
    }


def train_model(args: argparse.Namespace, arrays: Arrays, device: torch.device, loss_config: LossConfig) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    model = ComponentSetGateModel(shape_classes=len(arrays.shape_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = ComponentSetDataset(arrays, arrays.split_indices["train"])
    val_ds = ComponentSetDataset(arrays, arrays.split_indices["val"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_loss = math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_records = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["signal"], batch["sensor_z"])
            loss, stats = hungarian_loss_breakdown(out, batch, loss_config, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            stats["batch_size"] = int(batch["signal"].shape[0])
            train_records.append(stats)
        model.eval()
        val_records = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                out = model(batch["signal"], batch["sensor_z"])
                _loss, stats = hungarian_loss_breakdown(out, batch, loss_config, epoch)
                stats["batch_size"] = int(batch["signal"].shape[0])
                val_records.append(stats)
        train_terms = average_loss_records(train_records)
        val_terms = average_loss_records(val_records)
        train_loss = float(train_terms["loss"])
        val_loss = float(val_terms["loss"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_unweighted_terms": train_terms["unweighted"],
                "train_weighted_terms": train_terms["weighted"],
                "train_mask_depth_weighted_ratio": train_terms["mask_depth_weighted_ratio"],
                "train_component_separation_weighted_ratio": train_terms["component_separation_weighted_ratio"],
                "train_component_mask_to_union_mask_weighted_ratio": train_terms["component_mask_to_union_mask_weighted_ratio"],
                "train_depth_foreground_to_background_unweighted_ratio": train_terms["depth_foreground_to_background_unweighted_ratio"],
                "val_unweighted_terms": val_terms["unweighted"],
                "val_weighted_terms": val_terms["weighted"],
                "val_mask_depth_weighted_ratio": val_terms["mask_depth_weighted_ratio"],
                "val_component_separation_weighted_ratio": val_terms["component_separation_weighted_ratio"],
                "val_component_mask_to_union_mask_weighted_ratio": val_terms["component_mask_to_union_mask_weighted_ratio"],
                "val_depth_foreground_to_background_unweighted_ratio": val_terms["depth_foreground_to_background_unweighted_ratio"],
                "effective_loss_weights": train_records[-1]["effective_weights"] if train_records else scheduled_loss_weights(loss_config, epoch),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }
        )
    if best_state is None:
        raise RuntimeError("no best state recorded")
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


def decide_gate(history: list[dict[str, Any]], split_metrics: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    first_train = float(history[0]["train_loss"])
    final_train = float(history[-1]["train_loss"])
    best_val = min(float(row["val_loss"]) for row in history)
    val = split_metrics["val"]
    test = split_metrics["test"]
    empty_test = baselines["empty"]["test"]
    prior_test = baselines["one_slot_prior"]["test"]
    loss_descended = final_train < 0.75 * first_train and best_val < first_train
    recall_better_than_degenerate = test["component_recall"] > max(empty_test["component_recall"], prior_test["component_recall"]) + 0.15
    union_better_than_degenerate = test["union_mask_dice_mean"] > max(empty_test["union_mask_dice_mean"], prior_test["union_mask_dice_mean"]) + 0.10
    depth_better_than_empty = test["depth_grid_rmse_m_mean"] < 0.90 * empty_test["depth_grid_rmse_m_mean"]
    no_empty_collapse = test["pred_component_count_mean"] >= 1.5 and test["component_recall"] >= 0.35
    no_single_component_collapse = test["single_component_collapse_rate"] <= 0.55
    severe_extra = test["extra_rate"] > 0.65
    severe_missed = test["missed_rate"] > 0.65
    stable_val = val["component_recall"] >= 0.35 and val["union_mask_dice_mean"] >= 0.30

    if (
        loss_descended
        and recall_better_than_degenerate
        and union_better_than_degenerate
        and depth_better_than_empty
        and no_empty_collapse
        and no_single_component_collapse
        and stable_val
        and test["component_recall"] >= 0.60
        and test["union_mask_dice_mean"] >= 0.45
        and not severe_extra
        and not severe_missed
    ):
        decision = "PASS"
        next_route = "A. enter 25.11 component-set refinement / stronger training protocol"
    elif loss_descended and (recall_better_than_degenerate or union_better_than_degenerate or depth_better_than_empty) and no_empty_collapse:
        decision = "PARTIAL"
        next_route = "B. run 25.10b failure audit for merged/missed, overlap/touching, slot permutation, and three-component rows"
    else:
        decision = "FAIL"
        next_route = "C. return to representation/loss design before more training"
    return {
        "decision": decision,
        "next_route": next_route,
        "criteria": {
            "loss_descended": loss_descended,
            "recall_better_than_degenerate": recall_better_than_degenerate,
            "union_better_than_degenerate": union_better_than_degenerate,
            "depth_better_than_empty": depth_better_than_empty,
            "no_empty_collapse": no_empty_collapse,
            "no_single_component_collapse": no_single_component_collapse,
            "stable_val": stable_val,
            "severe_extra": severe_extra,
            "severe_missed": severe_missed,
        },
    }


def metric_delta(current: dict[str, Any], previous: dict[str, Any], key: str) -> dict[str, float | None]:
    cur = current.get(key)
    prev = previous.get(key)
    if cur is None or prev is None:
        return {"current": None, "previous": None, "delta": None}
    cur_f = float(cur)
    prev_f = float(prev)
    if not math.isfinite(cur_f) or not math.isfinite(prev_f):
        return {"current": None, "previous": None, "delta": None}
    return {"current": cur_f, "previous": prev_f, "delta": cur_f - prev_f}


def compare_to_previous(metrics_by_split: dict[str, Any], comparison_metrics_path: Path) -> dict[str, Any]:
    previous_payload = read_json(comparison_metrics_path)
    previous_test = previous_payload["metrics_by_split"]["test"]
    current_test = metrics_by_split["test"]
    keys = [
        "component_recall",
        "missed_rate",
        "merged_rate",
        "extra_rate",
        "center_error_m_mean",
        "lwd_relative_error_mean",
        "rotation_error_rad_mean",
        "component_mask_dice_mean",
        "component_mask_iou_mean",
        "union_mask_dice_mean",
        "union_mask_iou_mean",
        "depth_grid_rmse_m_mean",
    ]
    return {
        "comparison_metrics_path": str(comparison_metrics_path),
        "previous_gate_id": previous_payload.get("gate_id"),
        "previous_gate_decision": previous_payload.get("gate_decision"),
        "test_deltas": {key: metric_delta(current_test, previous_test, key) for key in keys},
        "previous_test_metrics": {key: previous_test.get(key) for key in keys},
    }


def decide_rebalance_gate(history: list[dict[str, Any]], split_metrics: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    first_train = float(history[0]["train_loss"])
    final_train = float(history[-1]["train_loss"])
    best_val = min(float(row["val_loss"]) for row in history)
    test = split_metrics["test"]
    deltas = comparison["test_deltas"]

    component_dice_delta = float(deltas["component_mask_dice_mean"]["delta"])
    union_dice_delta = float(deltas["union_mask_dice_mean"]["delta"])
    recall_delta = float(deltas["component_recall"]["delta"])
    center_delta = float(deltas["center_error_m_mean"]["delta"])
    lwd_delta = float(deltas["lwd_relative_error_mean"]["delta"])
    depth_delta = float(deltas["depth_grid_rmse_m_mean"]["delta"])

    loss_descended = final_train < 0.80 * first_train and best_val < first_train
    mask_clearly_improved = component_dice_delta >= 0.03 and union_dice_delta >= 0.03
    mask_somewhat_improved = component_dice_delta > 0.005 or union_dice_delta > 0.005
    recall_not_collapsed = recall_delta >= -0.08 and float(test["component_recall"]) >= 0.70
    geometry_not_degraded = center_delta <= 0.0025 and lwd_delta <= 0.05
    depth_not_worse = depth_delta <= 0.00005
    severe_merge_or_miss = float(test["merged_rate"]) >= 0.30 or float(test["missed_rate"]) >= 0.25
    three_component = test.get("groups", {}).get("component_count", {}).get("3", {})
    three_component_still_merged = bool(three_component and float(three_component.get("merged_rate", 0.0)) >= 0.75)

    if loss_descended and mask_clearly_improved and recall_not_collapsed and geometry_not_degraded and depth_not_worse:
        decision = "IMPROVED"
        next_route = "A. enter 25.12 topology-aware / three-component focused training, no baseline transition"
    elif loss_descended and (mask_somewhat_improved or depth_delta < 0.0) and recall_not_collapsed:
        decision = "PARTIAL"
        next_route = "B. run 25.11b targeted rebalance or topology-focused failure audit"
    else:
        decision = "FAIL"
        next_route = "C. return to loss formulation / mask-depth target design before more model capacity"

    return {
        "decision": decision,
        "next_route": next_route,
        "criteria": {
            "loss_descended": loss_descended,
            "mask_clearly_improved": mask_clearly_improved,
            "mask_somewhat_improved": mask_somewhat_improved,
            "recall_not_collapsed": recall_not_collapsed,
            "geometry_not_degraded": geometry_not_degraded,
            "depth_not_worse": depth_not_worse,
            "severe_merge_or_miss": severe_merge_or_miss,
            "three_component_still_merged": three_component_still_merged,
            "component_mask_dice_delta": component_dice_delta,
            "union_mask_dice_delta": union_dice_delta,
            "component_recall_delta": recall_delta,
            "center_error_delta_m": center_delta,
            "lwd_relative_error_delta": lwd_delta,
            "depth_grid_rmse_delta_m": depth_delta,
        },
    }


def compare_sample_rows(current_rows: list[dict[str, Any]], previous_path: Path, label: str) -> dict[str, Any]:
    previous = read_json(previous_path)
    prev_rows = {str(row["sample_id"]): row for row in previous["sample_metrics"]}
    rows = []
    for row in current_rows:
        sample_id = str(row["sample_id"])
        old = prev_rows[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "split": str(row["split"]),
                "component_count": int(row["component_count"]),
                "separation_type": str(row["separation_type"]),
                "topology_relation": str(row["topology_relation"]),
                "merged_previous": bool(old["merged_sample"]),
                "merged_current": bool(row["merged_sample"]),
                "newly_merged": bool((not old["merged_sample"]) and row["merged_sample"]),
                "recovered_from_merged": bool(old["merged_sample"] and not row["merged_sample"]),
                "component_mask_dice_delta": float(row["component_mask_dice_mean"]) - float(old["component_mask_dice_mean"]),
                "union_mask_dice_delta": float(row["union_mask_dice"]) - float(old["union_mask_dice"]),
                "depth_grid_rmse_delta_m": float(row["depth_grid_rmse_m"]) - float(old["depth_grid_rmse_m"]),
                "component_recall_delta": float(row["component_recall"]) - float(old["component_recall"]),
                "pred_component_count_delta": int(row["pred_component_count"]) - int(old["pred_component_count"]),
            }
        )

    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"sample_count": 0}
        return {
            "sample_count": len(subset),
            "newly_merged_rate": sum(bool(item["newly_merged"]) for item in subset) / len(subset),
            "recovered_from_merged_rate": sum(bool(item["recovered_from_merged"]) for item in subset) / len(subset),
            "component_mask_dice_delta_mean": float(np.mean([float(item["component_mask_dice_delta"]) for item in subset])),
            "union_mask_dice_delta_mean": float(np.mean([float(item["union_mask_dice_delta"]) for item in subset])),
            "depth_grid_rmse_delta_m_mean": float(np.mean([float(item["depth_grid_rmse_delta_m"]) for item in subset])),
        }

    test_rows = [row for row in rows if row["split"] == "test"]
    return {
        "label": label,
        "comparison_path": str(previous_path),
        "test": {
            "overall": aggregate(test_rows),
            "by_component_count": {str(key): aggregate([row for row in test_rows if int(row["component_count"]) == int(key)]) for key in sorted({row["component_count"] for row in test_rows})},
            "by_separation": {str(key): aggregate([row for row in test_rows if row["separation_type"] == key]) for key in sorted({row["separation_type"] for row in test_rows})},
            "by_topology": {str(key): aggregate([row for row in test_rows if row["topology_relation"] == key]) for key in sorted({row["topology_relation"] for row in test_rows})},
            "newly_merged_samples": [row for row in test_rows if row["newly_merged"]],
            "recovered_from_merged_samples": [row for row in test_rows if row["recovered_from_merged"]],
        },
    }


def decide_component_separation_gate(
    history: list[dict[str, Any]],
    split_metrics: dict[str, Any],
    comparison_25_10: dict[str, Any],
    comparison_25_11: dict[str, Any],
) -> dict[str, Any]:
    first_train = float(history[0]["train_loss"])
    final_train = float(history[-1]["train_loss"])
    best_val = min(float(row["val_loss"]) for row in history)
    test = split_metrics["test"]
    d10 = comparison_25_10["test_deltas"]
    d11 = comparison_25_11["test_deltas"]

    merged_delta_vs_25_11 = float(d11["merged_rate"]["delta"])
    recall_delta_vs_25_11 = float(d11["component_recall"]["delta"])
    missed_delta_vs_25_11 = float(d11["missed_rate"]["delta"])
    extra_delta_vs_25_11 = float(d11["extra_rate"]["delta"])
    component_dice_delta_vs_25_10 = float(d10["component_mask_dice_mean"]["delta"])
    component_dice_delta_vs_25_11 = float(d11["component_mask_dice_mean"]["delta"])
    union_dice_delta_vs_25_11 = float(d11["union_mask_dice_mean"]["delta"])
    depth_delta_vs_25_11 = float(d11["depth_grid_rmse_m_mean"]["delta"])
    depth_delta_vs_25_10 = float(d10["depth_grid_rmse_m_mean"]["delta"])
    three_component = test.get("groups", {}).get("component_count", {}).get("3", {})
    three_component_still_merged = bool(three_component and float(three_component.get("merged_rate", 0.0)) >= 0.75)

    loss_descended = final_train < 0.85 * first_train and best_val < first_train
    merged_clearly_improved = merged_delta_vs_25_11 <= -0.25 and float(test["merged_rate"]) <= 0.65
    merged_somewhat_improved = merged_delta_vs_25_11 <= -0.10
    component_or_depth_improved = component_dice_delta_vs_25_10 >= 0.0 or depth_delta_vs_25_11 <= -0.00010
    recall_not_collapsed = recall_delta_vs_25_11 >= -0.08 and float(test["component_recall"]) >= 0.75
    missed_extra_not_collapsed = missed_delta_vs_25_11 <= 0.08 and extra_delta_vs_25_11 <= 0.08
    depth_close_to_25_10 = depth_delta_vs_25_10 <= 0.00015
    severe_depth_worse = depth_delta_vs_25_11 > 0.00005

    if (
        loss_descended
        and merged_clearly_improved
        and component_or_depth_improved
        and recall_not_collapsed
        and missed_extra_not_collapsed
    ):
        decision = "IMPROVED"
        next_route = "A. enter 25.13 topology-aware / three-component focused training gate, no baseline transition"
    elif loss_descended and merged_somewhat_improved and recall_not_collapsed and missed_extra_not_collapsed:
        decision = "PARTIAL"
        next_route = "B. enter 25.12b targeted failure audit for separation penalty and depth foreground loss"
    else:
        decision = "FAIL"
        next_route = "C. rollback to 25.10 loss mainline and redesign component raster/depth targets before further training"

    return {
        "decision": decision,
        "next_route": next_route,
        "criteria": {
            "loss_descended": loss_descended,
            "merged_clearly_improved": merged_clearly_improved,
            "merged_somewhat_improved": merged_somewhat_improved,
            "component_or_depth_improved": component_or_depth_improved,
            "recall_not_collapsed": recall_not_collapsed,
            "missed_extra_not_collapsed": missed_extra_not_collapsed,
            "depth_close_to_25_10": depth_close_to_25_10,
            "severe_depth_worse": severe_depth_worse,
            "three_component_still_merged": three_component_still_merged,
            "merged_rate_delta_vs_25_11": merged_delta_vs_25_11,
            "component_recall_delta_vs_25_11": recall_delta_vs_25_11,
            "missed_rate_delta_vs_25_11": missed_delta_vs_25_11,
            "extra_rate_delta_vs_25_11": extra_delta_vs_25_11,
            "component_mask_dice_delta_vs_25_10": component_dice_delta_vs_25_10,
            "component_mask_dice_delta_vs_25_11": component_dice_delta_vs_25_11,
            "union_mask_dice_delta_vs_25_11": union_dice_delta_vs_25_11,
            "depth_grid_rmse_delta_vs_25_10_m": depth_delta_vs_25_10,
            "depth_grid_rmse_delta_vs_25_11_m": depth_delta_vs_25_11,
        },
    }


def decide_target_v2_gate(
    history: list[dict[str, Any]],
    split_metrics: dict[str, Any],
    comparison_25_10: dict[str, Any],
    comparison_25_11: dict[str, Any],
    comparison_25_12: dict[str, Any],
    target_summary: dict[str, Any],
) -> dict[str, Any]:
    first_train = float(history[0]["train_loss"])
    final_train = float(history[-1]["train_loss"])
    best_val = min(float(row["val_loss"]) for row in history)
    test = split_metrics["test"]
    d10 = comparison_25_10["test_deltas"]
    d11 = comparison_25_11["test_deltas"]
    d12 = comparison_25_12["test_deltas"]

    component_dice_delta_vs_25_10 = float(d10["component_mask_dice_mean"]["delta"])
    depth_delta_vs_25_10 = float(d10["depth_grid_rmse_m_mean"]["delta"])
    merged_delta_vs_25_10 = float(d10["merged_rate"]["delta"])
    recall_delta_vs_25_10 = float(d10["component_recall"]["delta"])
    missed_delta_vs_25_10 = float(d10["missed_rate"]["delta"])
    extra_delta_vs_25_10 = float(d10["extra_rate"]["delta"])
    merged_delta_vs_25_11 = float(d11["merged_rate"]["delta"])
    merged_delta_vs_25_12 = float(d12["merged_rate"]["delta"])
    component_dice_delta_vs_25_12 = float(d12["component_mask_dice_mean"]["delta"])
    depth_delta_vs_25_12 = float(d12["depth_grid_rmse_m_mean"]["delta"])
    three_component = test.get("groups", {}).get("component_count", {}).get("3", {})
    partially_overlapping = test.get("groups", {}).get("separation_type", {}).get("partially_overlapping", {})

    loss_descended = final_train < 0.85 * first_train and best_val < first_train
    target_v2_loaded = (
        target_summary.get("component_mask_target_v2") is True
        and target_summary.get("component_depth_target_v2") is True
        and target_summary.get("component_ownership_map") is True
        and int(target_summary.get("duplicate_ownership_after_v2", -1)) == 0
        and int(target_summary.get("overlap_depth_conflict_after_v2", -1)) == 0
    )
    merged_close_to_25_10 = float(test["merged_rate"]) <= 0.30 and merged_delta_vs_25_10 <= 0.10
    component_or_depth_better_than_25_10 = component_dice_delta_vs_25_10 >= 0.02 or depth_delta_vs_25_10 <= -0.00005
    recall_missed_extra_not_collapsed_vs_25_10 = recall_delta_vs_25_10 >= -0.06 and missed_delta_vs_25_10 <= 0.06 and extra_delta_vs_25_10 <= 0.08
    merged_alleviated_vs_rebalance = merged_delta_vs_25_11 <= -0.20 or merged_delta_vs_25_12 <= -0.20
    component_or_depth_alleviated_vs_25_12 = component_dice_delta_vs_25_12 >= 0.005 or depth_delta_vs_25_12 <= -0.00010
    no_severe_collapse = float(test["component_recall"]) >= 0.65 and float(test["missed_rate"]) <= 0.35 and float(test["extra_rate"]) <= 0.35
    no_mask_union_collapse = float(test["component_mask_dice_mean"]) >= 0.07 and float(test["union_mask_dice_mean"]) >= 0.05
    three_component_still_failed = bool(three_component and float(three_component.get("merged_rate", 0.0)) >= 0.75)
    partially_overlapping_still_failed = bool(partially_overlapping and float(partially_overlapping.get("merged_rate", 0.0)) >= 0.50)

    if (
        loss_descended
        and target_v2_loaded
        and merged_close_to_25_10
        and component_or_depth_better_than_25_10
        and recall_missed_extra_not_collapsed_vs_25_10
    ):
        decision = "IMPROVED"
        next_route = "A. enter 25.14 topology/three-component focused training gate, no baseline transition"
    elif (
        loss_descended
        and target_v2_loaded
        and no_severe_collapse
        and no_mask_union_collapse
        and (merged_alleviated_vs_rebalance or component_or_depth_alleviated_vs_25_12)
    ):
        decision = "PARTIAL"
        next_route = "B. enter 25.13b target-v2 failure audit focused on three-component and partially_overlapping subsets"
    else:
        decision = "FAIL"
        next_route = "C. return to generator/label schema; do not continue loss tuning"

    return {
        "decision": decision,
        "next_route": next_route,
        "criteria": {
            "loss_descended": loss_descended,
            "target_v2_loaded": target_v2_loaded,
            "merged_close_to_25_10": merged_close_to_25_10,
            "component_or_depth_better_than_25_10": component_or_depth_better_than_25_10,
            "recall_missed_extra_not_collapsed_vs_25_10": recall_missed_extra_not_collapsed_vs_25_10,
            "merged_alleviated_vs_rebalance": merged_alleviated_vs_rebalance,
            "component_or_depth_alleviated_vs_25_12": component_or_depth_alleviated_vs_25_12,
            "no_severe_collapse": no_severe_collapse,
            "no_mask_union_collapse": no_mask_union_collapse,
            "three_component_still_failed": three_component_still_failed,
            "partially_overlapping_still_failed": partially_overlapping_still_failed,
            "component_mask_dice_delta_vs_25_10": component_dice_delta_vs_25_10,
            "component_mask_dice_delta_vs_25_12": component_dice_delta_vs_25_12,
            "depth_grid_rmse_delta_vs_25_10_m": depth_delta_vs_25_10,
            "depth_grid_rmse_delta_vs_25_12_m": depth_delta_vs_25_12,
            "merged_rate_delta_vs_25_10": merged_delta_vs_25_10,
            "merged_rate_delta_vs_25_11": merged_delta_vs_25_11,
            "merged_rate_delta_vs_25_12": merged_delta_vs_25_12,
            "component_recall_delta_vs_25_10": recall_delta_vs_25_10,
            "missed_rate_delta_vs_25_10": missed_delta_vs_25_10,
            "extra_rate_delta_vs_25_10": extra_delta_vs_25_10,
        },
    }


def decide_label_v3_gate(
    history: list[dict[str, Any]],
    split_metrics: dict[str, Any],
    comparison_25_10: dict[str, Any],
    comparison_25_13: dict[str, Any],
    target_summary: dict[str, Any],
) -> dict[str, Any]:
    first_train = float(history[0]["train_loss"])
    final_train = float(history[-1]["train_loss"])
    best_val = min(float(row["val_loss"]) for row in history)
    test = split_metrics["test"]
    d10 = comparison_25_10["test_deltas"]
    d13 = comparison_25_13["test_deltas"]

    component_dice_delta_vs_25_13 = float(d13["component_mask_dice_mean"]["delta"])
    union_dice_delta_vs_25_13 = float(d13["union_mask_dice_mean"]["delta"])
    recall_delta_vs_25_13 = float(d13["component_recall"]["delta"])
    missed_delta_vs_25_13 = float(d13["missed_rate"]["delta"])
    extra_delta_vs_25_13 = float(d13["extra_rate"]["delta"])
    depth_delta_vs_25_13 = float(d13["depth_grid_rmse_m_mean"]["delta"])
    component_dice_delta_vs_25_10 = float(d10["component_mask_dice_mean"]["delta"])
    union_dice_delta_vs_25_10 = float(d10["union_mask_dice_mean"]["delta"])
    recall_delta_vs_25_10 = float(d10["component_recall"]["delta"])
    missed_delta_vs_25_10 = float(d10["missed_rate"]["delta"])
    extra_delta_vs_25_10 = float(d10["extra_rate"]["delta"])
    depth_delta_vs_25_10 = float(d10["depth_grid_rmse_m_mean"]["delta"])
    merged_delta_vs_25_10 = float(d10["merged_rate"]["delta"])
    three_component = test.get("groups", {}).get("component_count", {}).get("3", {})
    partially_overlapping = test.get("groups", {}).get("separation_type", {}).get("partially_overlapping", {})
    touching_boundary = test.get("groups", {}).get("topology_relation", {}).get("touching_boundary", {})

    label_v3_loaded = (
        target_summary.get("component_mask_target_v3_soft") is True
        and target_summary.get("component_sdf_target_v3") is True
        and target_summary.get("component_valid_region_mask") is True
        and target_summary.get("component_depth_target_v3") is True
        and int(target_summary.get("empty_slot_violation_count", -1)) == 0
        and int(target_summary.get("duplicate_ownership_after_v2", -1)) == 0
    )
    loss_descended = final_train < 0.85 * first_train and best_val < first_train
    near_empty_alleviated = (
        float(test["component_mask_dice_mean"]) >= 0.03
        and float(test["union_mask_dice_mean"]) >= 0.03
        and component_dice_delta_vs_25_13 >= 0.025
        and union_dice_delta_vs_25_13 >= 0.025
        and float(test["pred_component_count_mean"]) >= 1.5
    )
    merged_collapse_not_back = float(test["merged_rate"]) < 0.70
    recall_not_collapsed_vs_25_10 = recall_delta_vs_25_10 >= -0.12 and float(test["component_recall"]) >= 0.70
    missed_extra_not_collapsed_vs_25_10 = missed_delta_vs_25_10 <= 0.13 and extra_delta_vs_25_10 <= 0.18
    depth_not_clearly_worse_vs_25_10 = depth_delta_vs_25_10 <= 0.00015
    component_or_union_not_worse_than_25_10 = component_dice_delta_vs_25_10 >= -0.03 or union_dice_delta_vs_25_10 >= -0.03
    three_component_still_failed = bool(three_component and (float(three_component.get("merged_rate", 0.0)) >= 0.75 or float(three_component.get("component_mask_dice_mean", 0.0)) < 0.03))
    partially_overlapping_still_failed = bool(partially_overlapping and float(partially_overlapping.get("component_mask_dice_mean", 0.0)) < 0.03)
    touching_boundary_still_failed = bool(touching_boundary and float(touching_boundary.get("component_mask_dice_mean", 0.0)) < 0.03)

    if (
        label_v3_loaded
        and loss_descended
        and near_empty_alleviated
        and merged_collapse_not_back
        and recall_not_collapsed_vs_25_10
        and missed_extra_not_collapsed_vs_25_10
        and depth_not_clearly_worse_vs_25_10
        and component_or_union_not_worse_than_25_10
        and not three_component_still_failed
    ):
        decision = "IMPROVED"
        next_route = "A. enter 25.16 topology/three-component focused training gate, no baseline transition"
    elif label_v3_loaded and loss_descended and near_empty_alleviated and merged_collapse_not_back:
        decision = "PARTIAL"
        next_route = "B. enter 25.15b label-v3 failure audit focused on soft/SDF/valid-region loss usage"
    else:
        decision = "FAIL"
        next_route = "C. return to label-v3 derivation or generator/export schema; do not continue loss tuning"

    return {
        "decision": decision,
        "next_route": next_route,
        "criteria": {
            "label_v3_loaded": label_v3_loaded,
            "loss_descended": loss_descended,
            "near_empty_alleviated": near_empty_alleviated,
            "merged_collapse_not_back": merged_collapse_not_back,
            "recall_not_collapsed_vs_25_10": recall_not_collapsed_vs_25_10,
            "missed_extra_not_collapsed_vs_25_10": missed_extra_not_collapsed_vs_25_10,
            "depth_not_clearly_worse_vs_25_10": depth_not_clearly_worse_vs_25_10,
            "component_or_union_not_worse_than_25_10": component_or_union_not_worse_than_25_10,
            "three_component_still_failed": three_component_still_failed,
            "partially_overlapping_still_failed": partially_overlapping_still_failed,
            "touching_boundary_still_failed": touching_boundary_still_failed,
            "component_mask_dice_delta_vs_25_13": component_dice_delta_vs_25_13,
            "union_mask_dice_delta_vs_25_13": union_dice_delta_vs_25_13,
            "component_recall_delta_vs_25_13": recall_delta_vs_25_13,
            "missed_rate_delta_vs_25_13": missed_delta_vs_25_13,
            "extra_rate_delta_vs_25_13": extra_delta_vs_25_13,
            "depth_grid_rmse_delta_vs_25_13_m": depth_delta_vs_25_13,
            "component_mask_dice_delta_vs_25_10": component_dice_delta_vs_25_10,
            "union_mask_dice_delta_vs_25_10": union_dice_delta_vs_25_10,
            "component_recall_delta_vs_25_10": recall_delta_vs_25_10,
            "missed_rate_delta_vs_25_10": missed_delta_vs_25_10,
            "extra_rate_delta_vs_25_10": extra_delta_vs_25_10,
            "merged_rate_delta_vs_25_10": merged_delta_vs_25_10,
            "depth_grid_rmse_delta_vs_25_10_m": depth_delta_vs_25_10,
        },
    }


def decide_label_v3b_gate(
    history: list[dict[str, Any]],
    split_metrics: dict[str, Any],
    comparison_25_10: dict[str, Any],
    comparison_25_13: dict[str, Any],
    comparison_25_15: dict[str, Any],
    target_summary: dict[str, Any],
) -> dict[str, Any]:
    first_train = float(history[0]["train_loss"])
    final_train = float(history[-1]["train_loss"])
    best_val = min(float(row["val_loss"]) for row in history)
    test = split_metrics["test"]
    d10 = comparison_25_10["test_deltas"]
    d13 = comparison_25_13["test_deltas"]
    d15 = comparison_25_15["test_deltas"]

    component_dice_delta_vs_25_13 = float(d13["component_mask_dice_mean"]["delta"])
    union_dice_delta_vs_25_13 = float(d13["union_mask_dice_mean"]["delta"])
    depth_delta_vs_25_13 = float(d13["depth_grid_rmse_m_mean"]["delta"])
    component_dice_delta_vs_25_10 = float(d10["component_mask_dice_mean"]["delta"])
    union_dice_delta_vs_25_10 = float(d10["union_mask_dice_mean"]["delta"])
    recall_delta_vs_25_10 = float(d10["component_recall"]["delta"])
    missed_delta_vs_25_10 = float(d10["missed_rate"]["delta"])
    extra_delta_vs_25_10 = float(d10["extra_rate"]["delta"])
    depth_delta_vs_25_10 = float(d10["depth_grid_rmse_m_mean"]["delta"])
    merged_delta_vs_25_10 = float(d10["merged_rate"]["delta"])
    component_dice_delta_vs_25_15 = float(d15["component_mask_dice_mean"]["delta"])
    union_dice_delta_vs_25_15 = float(d15["union_mask_dice_mean"]["delta"])
    merged_delta_vs_25_15 = float(d15["merged_rate"]["delta"])
    depth_delta_vs_25_15 = float(d15["depth_grid_rmse_m_mean"]["delta"])

    three_component = test.get("groups", {}).get("component_count", {}).get("3", {})
    partially_overlapping = test.get("groups", {}).get("separation_type", {}).get("partially_overlapping", {})
    touching_boundary = test.get("groups", {}).get("topology_relation", {}).get("touching_boundary", {})

    label_v3b_loaded = (
        target_summary.get("component_hard_core_mask_v3b") is True
        and target_summary.get("component_boundary_halo_mask_v3b") is True
        and target_summary.get("component_ignore_overlap_mask_v3b") is True
        and target_summary.get("component_mask_target_v3b_soft") is True
        and target_summary.get("component_sdf_target_v3b") is True
        and target_summary.get("component_valid_region_mask_v3b") is True
        and target_summary.get("component_depth_target_v3b") is True
        and target_summary.get("component_identity_conflict_mask_v3b") is True
        and int(target_summary.get("v3b_empty_slot_violation_count", -1)) == 0
        and int(target_summary.get("v3b_hard_duplicate_pixel_sum", -1)) == 0
        and float(target_summary.get("v3b_soft_or_raw_union_ratio_max", 999.0)) <= V3B_SOFT_OR_RAW_UNION_RATIO_TARGET
    )
    loss_descended = final_train < 0.85 * first_train and best_val < first_train
    near_empty_alleviated_vs_25_13 = (
        float(test["component_mask_dice_mean"]) >= 0.03
        and float(test["union_mask_dice_mean"]) >= 0.03
        and component_dice_delta_vs_25_13 >= 0.025
        and union_dice_delta_vs_25_13 >= 0.025
        and float(test["pred_component_count_mean"]) >= 1.5
    )
    union_like_merged_alleviated_vs_25_15 = float(test["merged_rate"]) <= 0.70 and merged_delta_vs_25_15 <= -0.25
    recall_not_collapsed_vs_25_10 = recall_delta_vs_25_10 >= -0.15 and float(test["component_recall"]) >= 0.68
    missed_extra_not_collapsed_vs_25_10 = missed_delta_vs_25_10 <= 0.17 and extra_delta_vs_25_10 <= 0.22
    depth_not_severely_worse_vs_25_10 = depth_delta_vs_25_10 <= 0.00055
    three_component_still_failed = bool(
        three_component
        and (
            float(three_component.get("merged_rate", 0.0)) >= 0.75
            or float(three_component.get("component_mask_dice_mean", 0.0)) < 0.03
        )
    )
    partially_overlapping_still_failed = bool(
        partially_overlapping
        and (
            float(partially_overlapping.get("merged_rate", 0.0)) >= 0.75
            or float(partially_overlapping.get("component_mask_dice_mean", 0.0)) < 0.03
        )
    )
    touching_boundary_still_failed = bool(
        touching_boundary
        and (
            float(touching_boundary.get("merged_rate", 0.0)) >= 0.75
            or float(touching_boundary.get("component_mask_dice_mean", 0.0)) < 0.03
        )
    )

    if (
        label_v3b_loaded
        and loss_descended
        and near_empty_alleviated_vs_25_13
        and union_like_merged_alleviated_vs_25_15
        and recall_not_collapsed_vs_25_10
        and missed_extra_not_collapsed_vs_25_10
        and depth_not_severely_worse_vs_25_10
        and not three_component_still_failed
    ):
        decision = "IMPROVED"
        next_route = "A. enter 25.18 topology/three-component focused training gate, no baseline transition"
    elif label_v3b_loaded and loss_descended and (near_empty_alleviated_vs_25_13 or union_like_merged_alleviated_vs_25_15):
        decision = "PARTIAL"
        next_route = "B. enter 25.17b label-v3b failure audit focused on hard-core/halo/SDF/depth-valid-region usage"
    else:
        decision = "FAIL"
        next_route = "C. return to label-v3b derivation or generator/export schema; do not continue loss tuning"

    return {
        "decision": decision,
        "next_route": next_route,
        "criteria": {
            "label_v3b_loaded": label_v3b_loaded,
            "loss_descended": loss_descended,
            "near_empty_alleviated_vs_25_13": near_empty_alleviated_vs_25_13,
            "union_like_merged_alleviated_vs_25_15": union_like_merged_alleviated_vs_25_15,
            "recall_not_collapsed_vs_25_10": recall_not_collapsed_vs_25_10,
            "missed_extra_not_collapsed_vs_25_10": missed_extra_not_collapsed_vs_25_10,
            "depth_not_severely_worse_vs_25_10": depth_not_severely_worse_vs_25_10,
            "three_component_still_failed": three_component_still_failed,
            "partially_overlapping_still_failed": partially_overlapping_still_failed,
            "touching_boundary_still_failed": touching_boundary_still_failed,
            "component_mask_dice_delta_vs_25_13": component_dice_delta_vs_25_13,
            "union_mask_dice_delta_vs_25_13": union_dice_delta_vs_25_13,
            "depth_grid_rmse_delta_vs_25_13_m": depth_delta_vs_25_13,
            "component_mask_dice_delta_vs_25_10": component_dice_delta_vs_25_10,
            "union_mask_dice_delta_vs_25_10": union_dice_delta_vs_25_10,
            "component_recall_delta_vs_25_10": recall_delta_vs_25_10,
            "missed_rate_delta_vs_25_10": missed_delta_vs_25_10,
            "extra_rate_delta_vs_25_10": extra_delta_vs_25_10,
            "merged_rate_delta_vs_25_10": merged_delta_vs_25_10,
            "depth_grid_rmse_delta_vs_25_10_m": depth_delta_vs_25_10,
            "component_mask_dice_delta_vs_25_15": component_dice_delta_vs_25_15,
            "union_mask_dice_delta_vs_25_15": union_dice_delta_vs_25_15,
            "merged_rate_delta_vs_25_15": merged_delta_vs_25_15,
            "depth_grid_rmse_delta_vs_25_15_m": depth_delta_vs_25_15,
        },
    }


def allowed_use_for_stage(stage: str) -> list[str]:
    if stage == "25.17":
        return [
            "label_v3b_training_gate_evaluation",
            "label_v3b_failure_audit_input",
            "component_set_training_gate_comparison_only",
        ]
    return [
        "component_set_training_gate_evaluation",
        "target_v2_training_gate_evaluation",
        "label_v3_training_gate_evaluation",
        "label_v3b_training_gate_evaluation",
        "mask_depth_loss_rebalance_training",
        "failure_audit_input",
    ]


def update_registry_note(path: Path, gate_manifest: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    marker = f"## {DATASET_ID}"
    if marker not in text:
        raise ValueError(f"registry missing {DATASET_ID}")
    note = (
        f"- 25.10 training_gate_consumed: gate_id={GATE_ID}; "
        f"decision={gate_manifest['gate_decision']}; "
        f"metrics=`{gate_manifest['metrics_path']}`; "
        "baseline_ready=false; CURRENT_BASELINE.md unchanged."
    )
    if "25.10 training_gate_consumed: gate_id=25_10_surface_multipit_component_set_training_gate" in text:
        lines = text.splitlines()
        lines = [note if line.startswith("- 25.10 training_gate_consumed: gate_id=25_10_surface_multipit_component_set_training_gate") else line for line in lines]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    start = text.index(marker)
    next_heading = text.find("\n## ", start + 1)
    insert_at = len(text) if next_heading == -1 else next_heading
    text = text[:insert_at].rstrip() + "\n" + note + "\n" + text[insert_at:]
    path.write_text(text, encoding="utf-8")


def fmt_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    value_f = float(value)
    if not math.isfinite(value_f):
        return "null"
    return f"{value_f:.{digits}f}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    decision = payload["gate_decision"]
    test = payload["metrics_by_split"]["test"]
    val = payload["metrics_by_split"]["val"]
    test_groups = test.get("groups", {})
    three_component = test_groups.get("component_count", {}).get("3", {})
    partially_overlapping = test_groups.get("separation_type", {}).get("partially_overlapping", {})
    touching_boundary = test_groups.get("topology_relation", {}).get("touching_boundary", {})
    comparison = payload.get("comparison_to_25_10", {})
    comparison_25_11 = payload.get("comparison_to_25_11", {})
    comparison_25_12 = payload.get("comparison_to_25_12", {})
    comparison_25_15 = payload.get("comparison_to_25_15", {})
    deltas = comparison.get("test_deltas", {})
    deltas_25_11 = comparison_25_11.get("test_deltas", {})
    deltas_25_12 = comparison_25_12.get("test_deltas", {})
    deltas_25_15 = comparison_25_15.get("test_deltas", {})
    if payload["stage"] == "25.12":
        title = "25.12 Component-Separation-Aware Rebalance Training"
    elif payload["stage"] == "25.13":
        title = "25.13 Target-V2 Component-Set Training Gate"
    elif payload["stage"] == "25.15":
        title = "25.15 Label-V3 Component-Set Training Gate"
    elif payload["stage"] == "25.17":
        title = "25.17 Label-V3B Component-Set Training Gate"
    elif payload["stage"] == "25.11":
        title = "25.11 Mask/Depth Loss Rebalance Training"
    else:
        title = "25.10 Surface Multi-Pit Component-Set Training Gate"
    lines = [
        f"# {title}",
        "",
        f"- gate_decision: `{decision}`",
        f"- dataset_id: `{payload['dataset_id']}`",
        f"- model_route: `{payload['model']['route']}`",
        f"- loss_config: `{payload['model']['loss_config']}`",
        f"- target_version: `{payload['target_v2']['target_version']}`",
        f"- split: `{payload['data']['split_counts']}`",
        f"- selected_existence_threshold: `{payload['selection']['selected_threshold']}`",
        f"- best_epoch: `{payload['training']['best_epoch']}`",
        f"- first_train_loss: `{payload['training']['first_train_loss']:.6f}`",
        f"- final_train_loss: `{payload['training']['final_train_loss']:.6f}`",
        f"- best_val_loss: `{payload['training']['best_val_loss']:.6f}`",
        f"- final_train_mask_depth_weighted_ratio: `{fmt_metric(payload['training']['final_train_mask_depth_weighted_ratio'])}`",
        f"- final_val_mask_depth_weighted_ratio: `{fmt_metric(payload['training']['final_val_mask_depth_weighted_ratio'])}`",
        f"- final_train_component_separation_weighted_ratio: `{fmt_metric(payload['training'].get('final_train_component_separation_weighted_ratio'))}`",
        f"- final_val_component_separation_weighted_ratio: `{fmt_metric(payload['training'].get('final_val_component_separation_weighted_ratio'))}`",
        f"- final_train_component_mask_to_union_mask_weighted_ratio: `{fmt_metric(payload['training'].get('final_train_component_mask_to_union_mask_weighted_ratio'))}`",
        f"- final_train_label_v3_sdf_loss: `{fmt_metric(payload['training']['final_train_unweighted_terms'].get('label_v3_sdf'))}`",
        f"- final_val_label_v3_sdf_loss: `{fmt_metric(payload['training']['final_val_unweighted_terms'].get('label_v3_sdf'))}`",
        f"- final_train_label_v3_soft_bce_loss: `{fmt_metric(payload['training']['final_train_unweighted_terms'].get('label_v3_soft_bce'))}`",
        f"- final_val_label_v3_soft_bce_loss: `{fmt_metric(payload['training']['final_val_unweighted_terms'].get('label_v3_soft_bce'))}`",
        f"- final_train_label_v3_valid_depth_loss: `{fmt_metric(payload['training']['final_train_unweighted_terms'].get('label_v3_valid_depth'))}`",
        f"- final_val_label_v3_valid_depth_loss: `{fmt_metric(payload['training']['final_val_unweighted_terms'].get('label_v3_valid_depth'))}`",
        "",
        "## Target Transform Usage",
        "",
        f"- component_mask_target_v3_soft: `{payload['target_v2'].get('component_mask_target_v3_soft')}`",
        f"- component_sdf_target_v3: `{payload['target_v2'].get('component_sdf_target_v3')}`",
        f"- component_valid_region_mask: `{payload['target_v2'].get('component_valid_region_mask')}`",
        f"- component_depth_target_v3: `{payload['target_v2'].get('component_depth_target_v3')}`",
        f"- component_hard_core_mask_v3b: `{payload['target_v2'].get('component_hard_core_mask_v3b')}`",
        f"- component_boundary_halo_mask_v3b: `{payload['target_v2'].get('component_boundary_halo_mask_v3b')}`",
        f"- component_ignore_overlap_mask_v3b: `{payload['target_v2'].get('component_ignore_overlap_mask_v3b')}`",
        f"- component_mask_target_v3b_soft: `{payload['target_v2'].get('component_mask_target_v3b_soft')}`",
        f"- component_sdf_target_v3b: `{payload['target_v2'].get('component_sdf_target_v3b')}`",
        f"- component_valid_region_mask_v3b: `{payload['target_v2'].get('component_valid_region_mask_v3b')}`",
        f"- component_depth_target_v3b: `{payload['target_v2'].get('component_depth_target_v3b')}`",
        f"- component_identity_conflict_mask_v3b: `{payload['target_v2'].get('component_identity_conflict_mask_v3b')}`",
        f"- v3b_hard_core_pixel_mean/min: `{fmt_metric(payload['target_v2'].get('v3b_hard_core_pixel_mean'))}` / `{payload['target_v2'].get('v3b_hard_core_pixel_min')}`",
        f"- v3b_boundary_halo_pixel_mean/min: `{fmt_metric(payload['target_v2'].get('v3b_boundary_halo_pixel_mean'))}` / `{payload['target_v2'].get('v3b_boundary_halo_pixel_min')}`",
        f"- v3b_soft_support_pixel_mean/min: `{fmt_metric(payload['target_v2'].get('v3b_soft_support_pixel_mean'))}` / `{payload['target_v2'].get('v3b_soft_support_pixel_min')}`",
        f"- v3b_soft_or_raw_union_ratio_mean/max: `{fmt_metric(payload['target_v2'].get('v3b_soft_or_raw_union_ratio_mean'))}` / `{fmt_metric(payload['target_v2'].get('v3b_soft_or_raw_union_ratio_max'))}`",
        f"- v3b_identity_conflict_pixel_sum: `{payload['target_v2'].get('v3b_identity_conflict_pixel_sum')}`",
        f"- v3b_ignore_overlap_pixel_sum: `{payload['target_v2'].get('v3b_ignore_overlap_pixel_sum')}`",
        f"- v3_target_loaded_count: `{payload['target_v2'].get('v3_target_loaded_count')}`",
        f"- v3_soft_support_pixel_mean/min: `{fmt_metric(payload['target_v2'].get('v3_soft_support_pixel_mean'))}` / `{payload['target_v2'].get('v3_soft_support_pixel_min')}`",
        f"- v3_valid_region_pixel_mean/min: `{fmt_metric(payload['target_v2'].get('v3_valid_region_pixel_mean'))}` / `{payload['target_v2'].get('v3_valid_region_pixel_min')}`",
        f"- v3_depth_valid_region_pixel_mean/min: `{fmt_metric(payload['target_v2'].get('v3_depth_valid_region_pixel_mean'))}` / `{payload['target_v2'].get('v3_depth_valid_region_pixel_min')}`",
        f"- empty_slot_violation_count: `{payload['target_v2'].get('empty_slot_violation_count')}`",
        f"- component_mask_target_v2: `{payload['target_v2'].get('component_mask_target_v2')}`",
        f"- component_depth_target_v2: `{payload['target_v2'].get('component_depth_target_v2')}`",
        f"- component_ownership_map: `{payload['target_v2'].get('component_ownership_map')}`",
        f"- target_loaded_count: `{payload['target_v2'].get('target_loaded_count')}`",
        f"- ownership_resolved_pixel_count: `{payload['target_v2'].get('ownership_resolved_pixel_count')}`",
        f"- ownership_resolved_overlap_pixel_count: `{payload['target_v2'].get('ownership_resolved_overlap_pixel_count')}`",
        f"- duplicate_ownership_before_v2: `{payload['target_v2'].get('duplicate_ownership_before_v2')}`",
        f"- duplicate_ownership_after_v2: `{payload['target_v2'].get('duplicate_ownership_after_v2')}`",
        f"- overlap_depth_conflict_before_v2: `{payload['target_v2'].get('overlap_depth_conflict_before_v2')}`",
        f"- overlap_depth_conflict_after_v2: `{payload['target_v2'].get('overlap_depth_conflict_after_v2')}`",
        "",
        "## Validation Metrics",
        "",
        f"- component_recall: `{val['component_recall']:.6f}`",
        f"- missed_rate: `{val['missed_rate']:.6f}`",
        f"- merged_rate: `{val['merged_rate']:.6f}`",
        f"- extra_rate: `{val['extra_rate']:.6f}`",
        f"- union_mask_dice: `{val['union_mask_dice_mean']:.6f}`",
        f"- depth_grid_RMSE_m: `{val['depth_grid_rmse_m_mean']:.9f}`",
        "",
        "## Test Metrics",
        "",
        f"- component_recall: `{test['component_recall']:.6f}`",
        f"- missed_rate: `{test['missed_rate']:.6f}`",
        f"- merged_rate: `{test['merged_rate']:.6f}`",
        f"- extra_rate: `{test['extra_rate']:.6f}`",
        f"- center_error_m_mean: `{test['center_error_m_mean']:.9f}`",
        f"- lwd_relative_error_mean: `{test['lwd_relative_error_mean']:.6f}`",
        f"- rotation_error_rad_mean: `{test['rotation_error_rad_mean']:.6f}`",
        f"- component_mask_dice: `{test['component_mask_dice_mean']:.6f}`",
        f"- union_mask_dice: `{test['union_mask_dice_mean']:.6f}`",
        f"- depth_grid_RMSE_m: `{test['depth_grid_rmse_m_mean']:.9f}`",
        "",
        "## Required Test Subsets",
        "",
        f"- component_count=3: recall `{fmt_metric(three_component.get('component_recall'))}`, merged `{fmt_metric(three_component.get('merged_rate'))}`, component Dice `{fmt_metric(three_component.get('component_mask_dice_mean'))}`, union Dice `{fmt_metric(three_component.get('union_mask_dice_mean'))}`",
        f"- partially_overlapping: recall `{fmt_metric(partially_overlapping.get('component_recall'))}`, merged `{fmt_metric(partially_overlapping.get('merged_rate'))}`, component Dice `{fmt_metric(partially_overlapping.get('component_mask_dice_mean'))}`, union Dice `{fmt_metric(partially_overlapping.get('union_mask_dice_mean'))}`",
        f"- touching_boundary: recall `{fmt_metric(touching_boundary.get('component_recall'))}`, merged `{fmt_metric(touching_boundary.get('merged_rate'))}`, component Dice `{fmt_metric(touching_boundary.get('component_mask_dice_mean'))}`, union Dice `{fmt_metric(touching_boundary.get('union_mask_dice_mean'))}`",
        "",
        "## 25.10 Comparison",
        "",
        f"- component_recall_delta: `{fmt_metric(deltas.get('component_recall', {}).get('delta'))}`",
        f"- component_mask_dice_delta: `{fmt_metric(deltas.get('component_mask_dice_mean', {}).get('delta'))}`",
        f"- union_mask_dice_delta: `{fmt_metric(deltas.get('union_mask_dice_mean', {}).get('delta'))}`",
        f"- center_error_delta_m: `{fmt_metric(deltas.get('center_error_m_mean', {}).get('delta'), 9)}`",
        f"- lwd_relative_error_delta: `{fmt_metric(deltas.get('lwd_relative_error_mean', {}).get('delta'))}`",
        f"- depth_grid_RMSE_delta_m: `{fmt_metric(deltas.get('depth_grid_rmse_m_mean', {}).get('delta'), 9)}`",
        "",
        "## 25.11 Comparison",
        "",
        f"- component_recall_delta: `{fmt_metric(deltas_25_11.get('component_recall', {}).get('delta'))}`",
        f"- missed_rate_delta: `{fmt_metric(deltas_25_11.get('missed_rate', {}).get('delta'))}`",
        f"- extra_rate_delta: `{fmt_metric(deltas_25_11.get('extra_rate', {}).get('delta'))}`",
        f"- merged_rate_delta: `{fmt_metric(deltas_25_11.get('merged_rate', {}).get('delta'))}`",
        f"- component_mask_dice_delta: `{fmt_metric(deltas_25_11.get('component_mask_dice_mean', {}).get('delta'))}`",
        f"- union_mask_dice_delta: `{fmt_metric(deltas_25_11.get('union_mask_dice_mean', {}).get('delta'))}`",
        f"- depth_grid_RMSE_delta_m: `{fmt_metric(deltas_25_11.get('depth_grid_rmse_m_mean', {}).get('delta'), 9)}`",
        "",
        "## 25.12 Comparison",
        "",
        f"- component_recall_delta: `{fmt_metric(deltas_25_12.get('component_recall', {}).get('delta'))}`",
        f"- missed_rate_delta: `{fmt_metric(deltas_25_12.get('missed_rate', {}).get('delta'))}`",
        f"- extra_rate_delta: `{fmt_metric(deltas_25_12.get('extra_rate', {}).get('delta'))}`",
        f"- merged_rate_delta: `{fmt_metric(deltas_25_12.get('merged_rate', {}).get('delta'))}`",
        f"- component_mask_dice_delta: `{fmt_metric(deltas_25_12.get('component_mask_dice_mean', {}).get('delta'))}`",
        f"- union_mask_dice_delta: `{fmt_metric(deltas_25_12.get('union_mask_dice_mean', {}).get('delta'))}`",
        f"- depth_grid_RMSE_delta_m: `{fmt_metric(deltas_25_12.get('depth_grid_rmse_m_mean', {}).get('delta'), 9)}`",
        "",
        "## 25.13 Comparison",
        "",
        f"- component_recall_delta: `{fmt_metric(payload.get('comparison_to_25_13', {}).get('test_deltas', {}).get('component_recall', {}).get('delta'))}`",
        f"- missed_rate_delta: `{fmt_metric(payload.get('comparison_to_25_13', {}).get('test_deltas', {}).get('missed_rate', {}).get('delta'))}`",
        f"- extra_rate_delta: `{fmt_metric(payload.get('comparison_to_25_13', {}).get('test_deltas', {}).get('extra_rate', {}).get('delta'))}`",
        f"- merged_rate_delta: `{fmt_metric(payload.get('comparison_to_25_13', {}).get('test_deltas', {}).get('merged_rate', {}).get('delta'))}`",
        f"- component_mask_dice_delta: `{fmt_metric(payload.get('comparison_to_25_13', {}).get('test_deltas', {}).get('component_mask_dice_mean', {}).get('delta'))}`",
        f"- union_mask_dice_delta: `{fmt_metric(payload.get('comparison_to_25_13', {}).get('test_deltas', {}).get('union_mask_dice_mean', {}).get('delta'))}`",
        f"- depth_grid_RMSE_delta_m: `{fmt_metric(payload.get('comparison_to_25_13', {}).get('test_deltas', {}).get('depth_grid_rmse_m_mean', {}).get('delta'), 9)}`",
        "",
        "## 25.15 Comparison",
        "",
        f"- component_recall_delta: `{fmt_metric(deltas_25_15.get('component_recall', {}).get('delta'))}`",
        f"- missed_rate_delta: `{fmt_metric(deltas_25_15.get('missed_rate', {}).get('delta'))}`",
        f"- extra_rate_delta: `{fmt_metric(deltas_25_15.get('extra_rate', {}).get('delta'))}`",
        f"- merged_rate_delta: `{fmt_metric(deltas_25_15.get('merged_rate', {}).get('delta'))}`",
        f"- component_mask_dice_delta: `{fmt_metric(deltas_25_15.get('component_mask_dice_mean', {}).get('delta'))}`",
        f"- union_mask_dice_delta: `{fmt_metric(deltas_25_15.get('union_mask_dice_mean', {}).get('delta'))}`",
        f"- depth_grid_RMSE_delta_m: `{fmt_metric(deltas_25_15.get('depth_grid_rmse_m_mean', {}).get('delta'), 9)}`",
        "",
        "## Boundary",
        "",
        "- This is a training gate, not a baseline replacement.",
        "- No `CURRENT_BASELINE.md` transition is authorized.",
        "- Model architecture, K=3 component-set representation, fixed split, and Hungarian matching are unchanged.",
        f"- {payload['stage']} uses the 25.10 loss mainline with explicit label supervision, not the 25.11/25.12 rebalance stack.",
        "- No checkpoint or generated data artifact is committed by this gate.",
        f"- next_route: `{payload['route_decision']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    loss_config = LOSS_CONFIGS[args.loss_config]
    if args.stage == "25.13":
        redesign = read_json(args.target_redesign_manifest)
        if redesign.get("target_redesign_acceptance_decision") != "READY_FOR_25.13_TRAINING":
            raise ValueError("25.13 requires 25.12b READY_FOR_25.13_TRAINING target-redesign manifest")
        if args.loss_config != "component_set_gate_v1":
            raise ValueError("25.13 must use the 25.10 component_set_gate_v1 loss mainline")
        if args.target_version != "v2":
            raise ValueError("25.13 must use --target-version v2")
    if args.stage == "25.15":
        label_v3_manifest = read_json(args.label_v3_manifest)
        if label_v3_manifest.get("acceptance_decision") != "READY_FOR_25.15_TRAINING":
            raise ValueError("25.15 requires 25.14 READY_FOR_25.15_TRAINING label-v3 manifest")
        if args.loss_config != "component_set_gate_v1":
            raise ValueError("25.15 must use the 25.10 component_set_gate_v1 loss mainline")
        if args.target_version != "v3":
            raise ValueError("25.15 must use --target-version v3")
    if args.stage == "25.17":
        label_v3b_manifest = read_json(args.label_v3b_manifest)
        if label_v3b_manifest.get("acceptance_decision") != "READY_FOR_25_17_TRAINING":
            raise ValueError("25.17 requires 25.16 READY_FOR_25_17_TRAINING label-v3b manifest")
        if label_v3b_manifest.get("ready_for_training_v3b") is not True:
            raise ValueError("25.17 requires ready_for_training_v3b=true")
        if args.loss_config != "component_set_gate_v1":
            raise ValueError("25.17 must use the 25.10 component_set_gate_v1 loss mainline")
        if args.target_version != "v3b":
            raise ValueError("25.17 must use --target-version v3b")
    manifest, npz_path = assert_manifest_and_registry(args.manifest, args.registry)
    pack = load_npz(npz_path)
    arrays = build_arrays(pack, target_version=args.target_version)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, history, training_info = train_model(args, arrays, device, loss_config)
    val_pred = predict(model, arrays, arrays.split_indices["val"], device, args.batch_size)
    threshold, selection = select_threshold(arrays, val_pred)
    predictions = {
        "train": predict(model, arrays, arrays.split_indices["train"], device, args.batch_size),
        "val": val_pred,
        "test": predict(model, arrays, arrays.split_indices["test"], device, args.batch_size),
    }
    metrics_by_split, sample_rows = evaluate_splits(arrays, predictions, threshold)
    baselines = {
        "empty": {split: empty_baseline(arrays, split) for split in ["train", "val", "test"]},
        "one_slot_prior": {split: one_slot_prior_baseline(arrays, split) for split in ["train", "val", "test"]},
    }
    comparison = (
        compare_to_previous(metrics_by_split, args.comparison_metrics)
        if args.loss_config in {"mask_depth_rebalance_v1", "component_separation_rebalance_v1"} or args.stage in {"25.13", "25.15", "25.17"}
        else {}
    )
    comparison_25_11 = (
        compare_to_previous(metrics_by_split, args.comparison_metrics_25_11)
        if args.loss_config == "component_separation_rebalance_v1" or args.stage == "25.13"
        else {}
    )
    comparison_25_12 = compare_to_previous(metrics_by_split, args.comparison_metrics_25_12) if args.stage == "25.13" else {}
    comparison_25_13 = compare_to_previous(metrics_by_split, args.comparison_metrics_25_13) if args.stage in {"25.15", "25.17"} else {}
    comparison_25_15 = compare_to_previous(metrics_by_split, args.comparison_metrics_25_15) if args.stage == "25.17" else {}
    if args.stage == "25.13":
        gate = decide_target_v2_gate(history, metrics_by_split, comparison, comparison_25_11, comparison_25_12, arrays.target_transform_summary)
    elif args.stage == "25.15":
        gate = decide_label_v3_gate(history, metrics_by_split, comparison, comparison_25_13, arrays.target_transform_summary)
    elif args.stage == "25.17":
        gate = decide_label_v3b_gate(history, metrics_by_split, comparison, comparison_25_13, comparison_25_15, arrays.target_transform_summary)
    elif args.loss_config == "component_separation_rebalance_v1":
        gate = decide_component_separation_gate(history, metrics_by_split, comparison, comparison_25_11)
    elif args.loss_config == "mask_depth_rebalance_v1":
        gate = decide_rebalance_gate(history, metrics_by_split, comparison)
    else:
        gate = decide_gate(history, metrics_by_split, baselines)
    sample_comparisons = (
        {
            "vs_25_10": compare_sample_rows(sample_rows, args.comparison_metrics, "25.10"),
            "vs_25_11": compare_sample_rows(sample_rows, args.comparison_metrics_25_11, "25.11"),
            **({"vs_25_12": compare_sample_rows(sample_rows, args.comparison_metrics_25_12, "25.12")} if args.stage == "25.13" else {}),
            **({"vs_25_13": compare_sample_rows(sample_rows, args.comparison_metrics_25_13, "25.13")} if args.stage in {"25.15", "25.17"} else {}),
            **({"vs_25_15": compare_sample_rows(sample_rows, args.comparison_metrics_25_15, "25.15")} if args.stage == "25.17" else {}),
        }
        if args.loss_config == "component_separation_rebalance_v1" or args.stage in {"25.13", "25.15", "25.17"}
        else {}
    )
    split_counts = {split: int(indices.size) for split, indices in arrays.split_indices.items()}
    family_counts = dict(Counter(arrays.raw["component_shape_family"].astype(str).reshape(-1).tolist()))
    payload = {
        "stage": args.stage,
        "gate_id": args.gate_id,
        "dataset_id": DATASET_ID,
        "dataset_manifest": str(args.manifest),
        "dataset_npz_path": str(npz_path),
        "dataset_npz_sha256": manifest["npz_sha256"],
        "gate_decision": gate["decision"],
        "route_decision": gate["next_route"],
        "criteria": gate["criteria"],
        "comparison_to_25_10": comparison,
        "comparison_to_25_11": comparison_25_11,
        "comparison_to_25_12": comparison_25_12,
        "comparison_to_25_13": comparison_25_13,
        "comparison_to_25_15": comparison_25_15,
        "sample_comparisons": sample_comparisons,
        "target_v2": arrays.target_transform_summary,
        "data": {
            "n_samples": int(arrays.raw["sample_ids"].shape[0]),
            "split_counts": split_counts,
            "K_max": K_MAX,
            "component_count_counts": {str(k): int(v) for k, v in Counter(arrays.raw["component_count"].astype(int).tolist()).items()},
            "separation_counts": {str(k): int(v) for k, v in Counter(arrays.raw["separation_type"].astype(str).tolist()).items()},
            "topology_counts": {str(k): int(v) for k, v in Counter(arrays.raw["topology_relation"].astype(str).tolist()).items()},
            "shape_family_counts": family_counts,
            "sensor_z_m": float(np.asarray(arrays.raw["sensor_z_m"])[0]),
            "axis_names": [str(item) for item in arrays.raw["axis_names"].tolist()],
            "signal_shape": list(arrays.raw["delta_b"].shape),
        },
        "model": {
            "route": "C1_fixed_K_component_set_lightweight_gate",
            "input": "delta_b Bx/By/Bz plus constant sensor_z_m scalar",
            "outputs": "K=3 existence, center_xy, L/W/D, rotation, shape_family, component mask, component depth grid",
            "checkpoint_saved": False,
            "shape_classes": arrays.shape_classes,
            "loss": loss_config.description,
            "loss_config": loss_config.name,
            "loss_weights": loss_config.weights,
            "loss_effective_weights_final_epoch": scheduled_loss_weights(loss_config, args.epochs),
            "union_warmup_epochs": loss_config.union_warmup_epochs,
            "union_ramp_epochs": loss_config.union_ramp_epochs,
            "mask_supervision": loss_config.mask_supervision,
            "depth_supervision": loss_config.depth_supervision,
            "architecture_changed_from_25_10": False,
            "component_set_representation_changed": False,
            "hungarian_matching_changed": False,
            "label_v3b_manifest": str(args.label_v3b_manifest),
            "uses_25_10_loss_mainline": loss_config.name == "component_set_gate_v1",
            "uses_25_11_rebalance_stack": loss_config.name == "mask_depth_rebalance_v1",
            "uses_25_12_rebalance_stack": loss_config.name == "component_separation_rebalance_v1",
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "best_epoch": training_info["best_epoch"],
            "best_val_loss": training_info["best_val_loss"],
            "first_train_loss": float(history[0]["train_loss"]),
            "final_train_loss": float(history[-1]["train_loss"]),
            "first_train_unweighted_terms": history[0]["train_unweighted_terms"],
            "first_train_weighted_terms": history[0]["train_weighted_terms"],
            "final_train_unweighted_terms": history[-1]["train_unweighted_terms"],
            "final_train_weighted_terms": history[-1]["train_weighted_terms"],
            "final_train_mask_depth_weighted_ratio": history[-1]["train_mask_depth_weighted_ratio"],
            "final_train_component_separation_weighted_ratio": history[-1]["train_component_separation_weighted_ratio"],
            "final_train_component_mask_to_union_mask_weighted_ratio": history[-1]["train_component_mask_to_union_mask_weighted_ratio"],
            "final_train_depth_foreground_to_background_unweighted_ratio": history[-1]["train_depth_foreground_to_background_unweighted_ratio"],
            "final_val_unweighted_terms": history[-1]["val_unweighted_terms"],
            "final_val_weighted_terms": history[-1]["val_weighted_terms"],
            "final_val_mask_depth_weighted_ratio": history[-1]["val_mask_depth_weighted_ratio"],
            "final_val_component_separation_weighted_ratio": history[-1]["val_component_separation_weighted_ratio"],
            "final_val_component_mask_to_union_mask_weighted_ratio": history[-1]["val_component_mask_to_union_mask_weighted_ratio"],
            "final_val_depth_foreground_to_background_unweighted_ratio": history[-1]["val_depth_foreground_to_background_unweighted_ratio"],
            "history": history,
        },
        "selection": selection,
        "metrics_by_split": metrics_by_split,
        "degenerate_baselines": baselines,
        "sample_metrics": sample_rows,
        "boundary": {
            "baseline_replacement": False,
            "current_baseline_updated": False,
            "formal_inference_artifact_exported": False,
            "checkpoint_committed": False,
            "data_npz_committed": False,
            "model_capacity_expanded": False,
            "component_set_representation_changed": False,
            "uses_25_10_loss_mainline": loss_config.name == "component_set_gate_v1",
            "uses_25_11_rebalance_stack": loss_config.name == "mask_depth_rebalance_v1",
            "uses_25_12_rebalance_stack": loss_config.name == "component_separation_rebalance_v1",
        },
        "git": {
            "branch": git_value(["branch", "--show-current"]),
            "head_before_commit": git_value(["rev-parse", "HEAD"]),
            "protected_path_diff_before_write": git_value(["diff", "--name-only", "--", *FORBIDDEN_DIFF_PATHS]),
        },
    }
    write_json(args.metrics, payload)
    gate_manifest = {
        "stage": args.stage,
        "gate_id": args.gate_id,
        "dataset_id": DATASET_ID,
        "dataset_manifest": str(args.manifest),
        "metrics_path": str(args.metrics),
        "summary_path": str(args.summary),
        "script": "scripts/train_surface_multipit_component_set_gate.py",
        "gate_decision": gate["decision"],
        "route_decision": gate["next_route"],
        "loss_config": loss_config.name,
        "target_version": args.target_version,
        "target_redesign_manifest": str(args.target_redesign_manifest),
        "label_v3_manifest": str(args.label_v3_manifest),
        "label_v3b_manifest": str(args.label_v3b_manifest),
        "target_v2_loaded": arrays.target_transform_summary.get("component_mask_target_v2") is True,
        "target_v3_loaded": arrays.target_transform_summary.get("component_mask_target_v3_soft") is True,
        "target_v3b_loaded": arrays.target_transform_summary.get("component_mask_target_v3b_soft") is True,
        "model_capacity_expanded": False,
        "component_set_representation_changed": False,
        "train_ready_candidate_consumed": True,
        "baseline_ready": False,
        "current_baseline_updated": False,
        "checkpoint_saved": False,
        "inference_artifact_exported": False,
        "allowed_use": allowed_use_for_stage(args.stage),
        "forbidden_use": ["baseline_update", "current_baseline_replacement", "automatic_mainline_training", "formal_inference_artifact"],
    }
    write_json(args.gate_manifest, gate_manifest)
    write_summary(args.summary, payload)
    if not args.no_registry_note:
        update_registry_note(args.registry, gate_manifest)
    print(json.dumps(to_jsonable({"gate_decision": gate["decision"], "next_route": gate["next_route"], "metrics": str(args.metrics)}), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
