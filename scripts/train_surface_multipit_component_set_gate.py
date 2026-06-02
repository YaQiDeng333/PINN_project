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

TARGET_SPLIT = {"train": 72, "val": 20, "test": 20}
K_MAX = 3
LOW_H = 32
LOW_W = 64
PERMS = list(permutations(range(K_MAX)))
MASK_DEPTH_TERMS = ("component_mask", "union_mask", "component_depth", "union_depth")
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
        },
        mask_supervision="existing_slots_balanced_foreground_background_bce_plus_dice_with_union_loss",
        depth_supervision="existing_slots_and_union_smooth_l1_only_inside_valid_target_mask",
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
    parser.add_argument("--stage", default="25.10")
    parser.add_argument("--gate-id", default=GATE_ID)
    parser.add_argument("--loss-config", choices=sorted(LOSS_CONFIGS), default="component_set_gate_v1")
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
    depth_low_norm: np.ndarray
    depth_scale: float
    param_mean: np.ndarray
    param_std: np.ndarray
    signal_mean: np.ndarray
    signal_std: np.ndarray
    shape_classes: list[str]
    split_indices: dict[str, np.ndarray]


def build_arrays(pack: dict[str, Any]) -> Arrays:
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

    mask_low = downsample_np(np.asarray(pack["component_projected_masks_2d"], dtype=np.float32))
    depth_low = downsample_np(np.asarray(pack["component_depth_grids_m"], dtype=np.float32))
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
        depth_low_norm=depth_low_norm,
        depth_scale=depth_scale,
        param_mean=param_mean,
        param_std=param_std,
        signal_mean=signal_mean,
        signal_std=signal_std,
        shape_classes=shape_classes,
        split_indices=split_indices,
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
            "depth": torch.from_numpy(self.arrays.depth_low_norm[i]).float(),
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
    if loss_config.name == "mask_depth_rebalance_v1":
        mask_bce = balanced_bce_from_logits(pred["mask_logits"], mask)
        mask_dice = soft_dice_loss(pred["mask_logits"], mask)
        component_mask_loss = ((mask_bce + mask_dice) * exists).sum(dim=1) / active_count

        mask_prob = torch.sigmoid(pred["mask_logits"])
        active_view = exists.unsqueeze(-1).unsqueeze(-1)
        target_union_mask = mask.max(dim=1).values
        pred_union_prob = (mask_prob * active_view).max(dim=1).values
        union_mask_loss = balanced_bce_from_prob(pred_union_prob, target_union_mask) + soft_dice_loss_prob(pred_union_prob, target_union_mask)

        valid = (mask > 0.5).float()
        component_depth_raw = valid_region_smooth_l1(depth_pred, depth, valid)
        component_depth_loss = (component_depth_raw * exists).sum(dim=1) / active_count

        target_union_depth = depth.max(dim=1).values
        pred_union_depth = (depth_pred * mask_prob * active_view).max(dim=1).values
        valid_union = (target_union_mask > 0.5).float()
        union_depth_loss = valid_region_smooth_l1(pred_union_depth, target_union_depth, valid_union)
    else:
        mask_bce = F.binary_cross_entropy_with_logits(pred["mask_logits"], mask, reduction="none").mean(dim=(-2, -1))
        mask_dice = soft_dice_loss(pred["mask_logits"], mask)
        component_mask_loss = ((mask_bce + mask_dice) * exists).sum(dim=1) / active_count
        union_mask_loss = torch.zeros_like(component_mask_loss)

        target_weight = 0.15 + 0.85 * mask
        depth_mse = (((depth_pred - depth) ** 2) * target_weight).mean(dim=(-2, -1))
        component_depth_loss = (depth_mse * exists).sum(dim=1) / active_count
        union_depth_loss = torch.zeros_like(component_depth_loss)

    return {
        "exist": exist_loss,
        "param": param_loss,
        "shape": shape_loss,
        "component_mask": component_mask_loss,
        "union_mask": union_mask_loss,
        "component_depth": component_depth_loss,
        "union_depth": union_depth_loss,
    }


def hungarian_loss_breakdown(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], loss_config: LossConfig) -> tuple[torch.Tensor, dict[str, Any]]:
    per_perm_terms = [loss_terms_for_perm(pred, batch, perm, loss_config) for perm in PERMS]
    weighted_totals = []
    for terms in per_perm_terms:
        weighted_totals.append(sum(loss_config.weights[name] * terms[name] for name in loss_config.weights))
    weighted_stack = torch.stack(weighted_totals, dim=1)
    best_idx = weighted_stack.argmin(dim=1)
    selected_total = weighted_stack.gather(1, best_idx[:, None]).squeeze(1)
    selected_terms: dict[str, torch.Tensor] = {}
    for name in loss_config.weights:
        stack = torch.stack([terms[name] for terms in per_perm_terms], dim=1)
        selected_terms[name] = stack.gather(1, best_idx[:, None]).squeeze(1)
    loss = selected_total.mean()
    unweighted = {name: float(value.detach().mean().cpu()) for name, value in selected_terms.items()}
    weighted = {name: float((selected_terms[name] * loss_config.weights[name]).detach().mean().cpu()) for name in loss_config.weights}
    total_weighted = float(sum(weighted.values()))
    mask_depth_weighted = float(sum(weighted[name] for name in MASK_DEPTH_TERMS))
    stats = {
        "loss": float(loss.detach().cpu()),
        "unweighted": unweighted,
        "weighted": weighted,
        "mask_depth_weighted_sum": mask_depth_weighted,
        "total_weighted_sum": total_weighted,
        "mask_depth_weighted_ratio": mask_depth_weighted / total_weighted if total_weighted > 0.0 else math.nan,
    }
    return loss, stats


def hungarian_training_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], loss_config: LossConfig | None = None) -> torch.Tensor:
    config = loss_config or LOSS_CONFIGS["component_set_gate_v1"]
    loss, _stats = hungarian_loss_breakdown(pred, batch, config)
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
        true_masks = np.asarray(raw["component_projected_masks_2d"][src_idx, true_slots], dtype=np.float64)
        true_depths = np.asarray(raw["component_depth_grids_m"][src_idx, true_slots], dtype=np.float64)

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
    mean_mask = arrays.raw["component_projected_masks_2d"][train_idx][active].mean(axis=0)
    mean_depth = arrays.raw["component_depth_grids_m"][train_idx][active].mean(axis=0)
    rows = []
    raw = arrays.raw
    for idx in arrays.split_indices[split]:
        true_slots = np.where(raw["component_exists"][idx].astype(bool))[0]
        true_centers = raw["component_center_xy_m"][idx, true_slots]
        true_lwd = raw["component_lwd_m"][idx, true_slots]
        true_rot = raw["component_rotation_angle"][idx, true_slots]
        true_masks = raw["component_projected_masks_2d"][idx, true_slots]
        true_depths = raw["component_depth_grids_m"][idx, true_slots]
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
    term_names = list(records[0]["unweighted"].keys())
    unweighted = {
        name: float(sum(float(record["unweighted"][name]) * int(record["batch_size"]) for record in records) / total_n)
        for name in term_names
    }
    weighted = {
        name: float(sum(float(record["weighted"][name]) * int(record["batch_size"]) for record in records) / total_n)
        for name in term_names
    }
    loss = float(sum(float(record["loss"]) * int(record["batch_size"]) for record in records) / total_n)
    total_weighted = float(sum(weighted.values()))
    mask_depth_weighted = float(sum(weighted[name] for name in MASK_DEPTH_TERMS))
    return {
        "loss": loss,
        "unweighted": unweighted,
        "weighted": weighted,
        "mask_depth_weighted_sum": mask_depth_weighted,
        "total_weighted_sum": total_weighted,
        "mask_depth_weighted_ratio": mask_depth_weighted / total_weighted if total_weighted > 0.0 else math.nan,
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
            loss, stats = hungarian_loss_breakdown(out, batch, loss_config)
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
                _loss, stats = hungarian_loss_breakdown(out, batch, loss_config)
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
                "val_unweighted_terms": val_terms["unweighted"],
                "val_weighted_terms": val_terms["weighted"],
                "val_mask_depth_weighted_ratio": val_terms["mask_depth_weighted_ratio"],
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
    comparison = payload.get("comparison_to_25_10", {})
    deltas = comparison.get("test_deltas", {})
    title = "25.11 Mask/Depth Loss Rebalance Training" if payload["stage"] == "25.11" else "25.10 Surface Multi-Pit Component-Set Training Gate"
    lines = [
        f"# {title}",
        "",
        f"- gate_decision: `{decision}`",
        f"- dataset_id: `{payload['dataset_id']}`",
        f"- model_route: `{payload['model']['route']}`",
        f"- loss_config: `{payload['model']['loss_config']}`",
        f"- split: `{payload['data']['split_counts']}`",
        f"- selected_existence_threshold: `{payload['selection']['selected_threshold']}`",
        f"- best_epoch: `{payload['training']['best_epoch']}`",
        f"- first_train_loss: `{payload['training']['first_train_loss']:.6f}`",
        f"- final_train_loss: `{payload['training']['final_train_loss']:.6f}`",
        f"- best_val_loss: `{payload['training']['best_val_loss']:.6f}`",
        f"- final_train_mask_depth_weighted_ratio: `{fmt_metric(payload['training']['final_train_mask_depth_weighted_ratio'])}`",
        f"- final_val_mask_depth_weighted_ratio: `{fmt_metric(payload['training']['final_val_mask_depth_weighted_ratio'])}`",
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
        "## 25.10 Comparison",
        "",
        f"- component_recall_delta: `{fmt_metric(deltas.get('component_recall', {}).get('delta'))}`",
        f"- component_mask_dice_delta: `{fmt_metric(deltas.get('component_mask_dice_mean', {}).get('delta'))}`",
        f"- union_mask_dice_delta: `{fmt_metric(deltas.get('union_mask_dice_mean', {}).get('delta'))}`",
        f"- center_error_delta_m: `{fmt_metric(deltas.get('center_error_m_mean', {}).get('delta'), 9)}`",
        f"- lwd_relative_error_delta: `{fmt_metric(deltas.get('lwd_relative_error_mean', {}).get('delta'))}`",
        f"- depth_grid_RMSE_delta_m: `{fmt_metric(deltas.get('depth_grid_rmse_m_mean', {}).get('delta'), 9)}`",
        "",
        "## Boundary",
        "",
        "- This is a training gate, not a baseline replacement.",
        "- No `CURRENT_BASELINE.md` transition is authorized.",
        "- Model architecture, K=3 component-set representation, fixed split, and Hungarian matching are unchanged.",
        "- No checkpoint or generated data artifact is committed by this gate.",
        f"- next_route: `{payload['route_decision']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    loss_config = LOSS_CONFIGS[args.loss_config]
    manifest, npz_path = assert_manifest_and_registry(args.manifest, args.registry)
    pack = load_npz(npz_path)
    arrays = build_arrays(pack)
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
    comparison = compare_to_previous(metrics_by_split, args.comparison_metrics) if args.loss_config == "mask_depth_rebalance_v1" else {}
    gate = decide_rebalance_gate(history, metrics_by_split, comparison) if args.loss_config == "mask_depth_rebalance_v1" else decide_gate(history, metrics_by_split, baselines)
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
            "mask_supervision": loss_config.mask_supervision,
            "depth_supervision": loss_config.depth_supervision,
            "architecture_changed_from_25_10": False,
            "component_set_representation_changed": False,
            "hungarian_matching_changed": False,
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
            "final_val_unweighted_terms": history[-1]["val_unweighted_terms"],
            "final_val_weighted_terms": history[-1]["val_weighted_terms"],
            "final_val_mask_depth_weighted_ratio": history[-1]["val_mask_depth_weighted_ratio"],
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
        "train_ready_candidate_consumed": True,
        "baseline_ready": False,
        "current_baseline_updated": False,
        "checkpoint_saved": False,
        "inference_artifact_exported": False,
        "allowed_use": ["component_set_training_gate_evaluation", "mask_depth_loss_rebalance_training", "failure_audit_input"],
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
