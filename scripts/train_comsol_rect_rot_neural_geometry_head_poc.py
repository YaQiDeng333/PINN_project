from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
DEFAULT_LABELS = PROJECT_ROOT / "results/metrics/comsol_single_defect_geometry_labels.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_neural_geometry_head_poc_summary.txt"
DEFAULT_AUDIT = (
    PROJECT_ROOT / "results/summaries/comsol_rect_rot_neural_geometry_head_failure_audit_summary.txt"
)
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_neural_geometry_head_poc_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_neural_geometry_head_poc_epoch_log.csv"
DEFAULT_GROUP_SUMMARY = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_neural_geometry_head_poc_group_summary.csv"
)
DEFAULT_GEOMETRY_SUMMARY = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_neural_geometry_head_poc_geometry_summary.csv"
)
DEFAULT_FAILURE_CASES = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_neural_geometry_head_poc_failure_cases.csv"
)
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_rect_rot_neural_geometry_head_poc"

MAIN_TYPES = {"rectangular_notch": 0, "rotated_rect": 1}
TYPE_NAMES = {0: "rectangular_notch", 1: "rotated_rect"}
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SEED = 42
TEMPERATURE_M = 5.0e-4
DENSE_SINGLE_BASELINE_IOU = 0.6515
DENSE_SINGLE_BASELINE_DICE = 0.7861
PIAO_WEAK_IOU = 0.4467
PIAO_WEAK_DICE = 0.5499
PIAO_WEAK_TYPE_ACC = 0.4040
PIAO_WEAK_ANGLE_MAE = 18.29

METRIC_FIELDS = [
    "sample_id",
    "source_index",
    "split",
    "defect_type",
    "source_pack",
    "threshold",
    "iou",
    "dice",
    "area_error",
    "center_error_px",
    "pred_area",
    "true_area",
    "pred_area_zero",
    "type_prob_rectangular_notch",
    "type_prob_rotated_rect",
    "pred_defect_type",
    "type_correct",
    "true_center_x",
    "true_center_y",
    "pred_center_x",
    "pred_center_y",
    "center_mae_m",
    "true_width",
    "pred_width",
    "width_abs_error_m",
    "true_length",
    "pred_length",
    "length_abs_error_m",
    "true_depth",
    "pred_depth",
    "depth_abs_error_m",
    "true_angle_deg",
    "pred_angle_deg",
    "angle_abs_error_deg",
    "sin_cos_error",
    "notes",
]

EPOCH_FIELDS = [
    "epoch",
    "train_loss",
    "train_bce_loss",
    "train_dice_loss",
    "train_type_loss",
    "train_center_loss",
    "train_size_loss",
    "train_depth_loss",
    "train_angle_loss",
    "best_val_threshold",
    "val_iou",
    "val_dice",
    "val_area_error",
    "val_score",
]

GROUP_FIELDS = [
    "split",
    "group_name",
    "group_value",
    "sample_count",
    "iou_mean",
    "dice_mean",
    "area_error_mean",
    "center_error_px_mean",
    "pred_area_mean",
    "true_area_mean",
    "pred_area_zero_mean",
    "type_accuracy",
    "center_mae_m",
    "width_mae_m",
    "length_mae_m",
    "depth_mae_m",
    "angle_mae_deg",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [to_float(row.get(key, "")) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return float(np.mean(values)) if values else math.nan


def safe_std(rows: list[dict[str, Any]], key: str) -> float:
    values = [to_float(row.get(key, "")) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return float(np.std(values, ddof=0)) if values else math.nan


def circular_angle_error_deg(pred_deg: float, true_deg: float) -> float:
    diff = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return abs(float(diff))


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth = 1.0
    prob_flat = prob.reshape(prob.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    inter = (prob_flat * target_flat).sum(dim=1)
    denom = prob_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def soft_rect_mask(
    mask_x: torch.Tensor,
    mask_y: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    width: torch.Tensor,
    length: torch.Tensor,
    angle_rad: torch.Tensor,
    temperature: float = TEMPERATURE_M,
) -> torch.Tensor:
    x_grid, y_grid = torch.meshgrid(mask_x, mask_y, indexing="xy")
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)
    cx = center_x.view(-1, 1, 1)
    cy = center_y.view(-1, 1, 1)
    half_w = width.view(-1, 1, 1).clamp_min(1e-6) / 2.0
    half_l = length.view(-1, 1, 1).clamp_min(1e-6) / 2.0
    angle = angle_rad.view(-1, 1, 1)
    dx0 = x_grid - cx
    dy0 = y_grid - cy
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    x_rot = dx0 * cos_a + dy0 * sin_a
    y_rot = -dx0 * sin_a + dy0 * cos_a
    dx = torch.abs(x_rot) - half_w
    dy = torch.abs(y_rot) - half_l
    outside = torch.sqrt(torch.clamp(dx, min=0.0).square() + torch.clamp(dy, min=0.0).square() + 1e-18)
    inside = torch.clamp(torch.maximum(dx, dy), max=0.0)
    signed_distance = outside + inside
    return torch.sigmoid(-signed_distance / temperature)


class RectRotDataset(Dataset):
    def __init__(self, indices: np.ndarray, arrays: dict[str, Any]):
        self.indices = indices.astype(np.int64)
        self.signals = arrays["signals_norm"][self.indices]
        self.masks = arrays["masks"][self.indices]
        self.type_targets = arrays["type_targets"][self.indices]
        self.geom_targets = arrays["geom_targets"][self.indices]
        self.angle_targets = arrays["angle_targets"][self.indices]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        source_idx = int(self.indices[idx])
        return {
            "source_index": torch.tensor(source_idx, dtype=torch.long),
            "signal": torch.from_numpy(self.signals[idx]).float(),
            "mask": torch.from_numpy(self.masks[idx]).float(),
            "type_target": torch.tensor(int(self.type_targets[idx]), dtype=torch.long),
            "geom_target": torch.from_numpy(self.geom_targets[idx]).float(),
            "angle_target": torch.from_numpy(self.angle_targets[idx]).float(),
        }


class BzEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128, latent_dim), nn.GELU())

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(signal))


class NeuralGeometryHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = BzEncoder(latent_dim=latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 9),
        )

    def forward(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.head(self.encoder(signal))
        return {
            "type_logits": raw[:, :2],
            "geom_norm": raw[:, 2:7],
            "angle_sincos": raw[:, 7:9],
        }


def load_arrays(npz_path: Path, labels_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    labels = read_csv(labels_path)
    sample_ids = data["sample_ids"].astype(str)
    masks = data["masks"].astype(np.float32)
    delta_bz = data["delta_bz"].astype(np.float32)
    mask_x = data["mask_x"].astype(np.float32)
    mask_y = data["mask_y"].astype(np.float32)
    sensor_x = data["sensor_x"].astype(np.float32)

    label_by_id = {row["sample_id"]: row for row in labels}
    keep: list[int] = []
    label_rows: list[dict[str, str]] = []
    for idx, sample_id in enumerate(sample_ids):
        row = label_by_id.get(sample_id)
        if row is None:
            continue
        if row["defect_type"] not in MAIN_TYPES:
            continue
        keep.append(idx)
        label_rows.append(row)
    if not keep:
        raise RuntimeError("No rectangular_notch / rotated_rect samples found")
    keep_arr = np.array(keep, dtype=np.int64)
    splits = np.array([row["split"] for row in label_rows])
    type_targets = np.array([MAIN_TYPES[row["defect_type"]] for row in label_rows], dtype=np.int64)
    raw_geom = np.array(
        [
            [
                to_float(row["center_x"]),
                to_float(row["center_y"]),
                to_float(row["width"]),
                to_float(row["length"]),
                to_float(row["depth"]),
            ]
            for row in label_rows
        ],
        dtype=np.float32,
    )
    angle_targets = np.array(
        [[to_float(row["angle_sin"], 0.0), to_float(row["angle_cos"], 1.0)] for row in label_rows],
        dtype=np.float32,
    )

    train_local = np.where(splits == "train")[0]
    if train_local.size == 0:
        raise RuntimeError("No train samples in rect/rot subset")
    train_global = keep_arr[train_local]
    signal_mean = float(delta_bz[train_global].mean())
    signal_std = float(delta_bz[train_global].std())
    if signal_std <= 0:
        signal_std = 1.0
    signals_norm_all = ((delta_bz - signal_mean) / signal_std).astype(np.float32)

    geom_mean = raw_geom[train_local].mean(axis=0)
    geom_std = raw_geom[train_local].std(axis=0)
    geom_std = np.where(geom_std <= 0, 1.0, geom_std).astype(np.float32)
    geom_targets = ((raw_geom - geom_mean) / geom_std).astype(np.float32)

    source_packs = np.array([row.get("source_pack", "") for row in label_rows])
    arrays = {
        "signals_norm": signals_norm_all[keep_arr],
        "masks": masks[keep_arr],
        "splits": splits,
        "sample_ids": sample_ids[keep_arr],
        "source_indices": keep_arr,
        "source_packs": source_packs,
        "defect_types": np.array([row["defect_type"] for row in label_rows]),
        "type_targets": type_targets,
        "raw_geom": raw_geom,
        "geom_targets": geom_targets,
        "angle_targets": angle_targets,
        "mask_x": mask_x,
        "mask_y": mask_y,
        "sensor_x": sensor_x,
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "geom_mean": geom_mean.astype(np.float32),
        "geom_std": geom_std.astype(np.float32),
    }
    split_indices = {split: np.where(splits == split)[0].astype(np.int64) for split in ["train", "val", "test"]}
    diagnostics = {
        "n_rect_rot": int(keep_arr.shape[0]),
        "split_counts": {key: int(value.shape[0]) for key, value in split_indices.items()},
        "type_counts": {
            key: int(np.sum(arrays["defect_types"] == key)) for key in sorted(MAIN_TYPES)
        },
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "geom_mean": geom_mean.tolist(),
        "geom_std": geom_std.tolist(),
    }
    return {**arrays, "split_indices": split_indices}, diagnostics


def decode_geometry(
    outputs: dict[str, torch.Tensor],
    arrays: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    geom_mean = torch.tensor(arrays["geom_mean"], device=device).view(1, -1)
    geom_std = torch.tensor(arrays["geom_std"], device=device).view(1, -1)
    raw_geom = outputs["geom_norm"] * geom_std + geom_mean
    mask_x = torch.tensor(arrays["mask_x"], device=device)
    mask_y = torch.tensor(arrays["mask_y"], device=device)
    center_x = raw_geom[:, 0].clamp(float(mask_x.min()), float(mask_x.max()))
    center_y = raw_geom[:, 1].clamp(float(mask_y.min()), float(mask_y.max()))
    width = raw_geom[:, 2].clamp(0.001, 0.025)
    length = raw_geom[:, 3].clamp(0.001, 0.020)
    depth = raw_geom[:, 4].clamp(0.0001, 0.004)
    angle = torch.atan2(outputs["angle_sincos"][:, 0], outputs["angle_sincos"][:, 1])
    type_prob = torch.softmax(outputs["type_logits"], dim=1)
    rect_prob = soft_rect_mask(mask_x, mask_y, center_x, center_y, width, length, torch.zeros_like(angle))
    rot_prob = soft_rect_mask(mask_x, mask_y, center_x, center_y, width, length, angle)
    mask_prob = type_prob[:, 0].view(-1, 1, 1) * rect_prob + type_prob[:, 1].view(-1, 1, 1) * rot_prob
    mask_logits = torch.logit(mask_prob.clamp(1e-4, 1.0 - 1e-4))
    return {
        "raw_geom": torch.stack([center_x, center_y, width, length, depth], dim=1),
        "angle_rad": angle,
        "type_prob": type_prob,
        "mask_prob": mask_prob,
        "mask_logits": mask_logits,
    }


def batch_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    arrays: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    decoded = decode_geometry(outputs, arrays, device)
    target_mask = batch["mask"].to(device)
    bce = F.binary_cross_entropy_with_logits(decoded["mask_logits"], target_mask)
    dice = soft_dice_loss(decoded["mask_prob"], target_mask)
    type_loss = F.cross_entropy(outputs["type_logits"], batch["type_target"].to(device))
    geom_target = batch["geom_target"].to(device)
    center_loss = F.smooth_l1_loss(outputs["geom_norm"][:, :2], geom_target[:, :2])
    size_loss = F.smooth_l1_loss(outputs["geom_norm"][:, 2:4], geom_target[:, 2:4])
    depth_loss = F.smooth_l1_loss(outputs["geom_norm"][:, 4:5], geom_target[:, 4:5])
    angle_loss = F.smooth_l1_loss(outputs["angle_sincos"], batch["angle_target"].to(device))
    loss = (
        bce
        + dice
        + 0.1 * type_loss
        + 0.1 * center_loss
        + 0.1 * size_loss
        + 0.05 * depth_loss
        + 0.05 * angle_loss
    )
    return loss, {
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "type": float(type_loss.detach().cpu()),
        "center": float(center_loss.detach().cpu()),
        "size": float(size_loss.detach().cpu()),
        "depth": float(depth_loss.detach().cpu()),
        "angle": float(angle_loss.detach().cpu()),
    }


def mask_metric(pred_prob: np.ndarray, true_mask: np.ndarray, threshold: float) -> dict[str, float]:
    pred = pred_prob >= threshold
    true = true_mask > 0.5
    inter = int(np.logical_and(pred, true).sum())
    union = int(np.logical_or(pred, true).sum())
    pred_area = int(pred.sum())
    true_area = int(true.sum())
    iou = inter / union if union else 1.0
    dice = 2.0 * inter / (pred_area + true_area) if (pred_area + true_area) else 1.0
    area_error = abs(pred_area - true_area) / max(true_area, 1)
    center_error = center_error_px(pred, true)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "area_error": float(area_error),
        "center_error_px": float(center_error),
        "pred_area": float(pred_area),
        "true_area": float(true_area),
        "pred_area_zero": float(pred_area == 0),
    }


def center_error_px(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.sum() == 0 or true.sum() == 0:
        return float("nan")
    py, px = np.argwhere(pred).mean(axis=0)
    ty, tx = np.argwhere(true).mean(axis=0)
    return float(math.hypot(px - tx, py - ty))


def predict_split(
    model: nn.Module,
    dataset: RectRotDataset,
    arrays: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_indices: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    all_type_prob: list[np.ndarray] = []
    all_geom: list[np.ndarray] = []
    all_angle: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            outputs = model(signal)
            decoded = decode_geometry(outputs, arrays, device)
            all_indices.append(batch["source_index"].cpu().numpy())
            all_prob.append(decoded["mask_prob"].cpu().numpy())
            all_type_prob.append(decoded["type_prob"].cpu().numpy())
            all_geom.append(decoded["raw_geom"].cpu().numpy())
            all_angle.append(decoded["angle_rad"].cpu().numpy())
    return {
        "indices": np.concatenate(all_indices),
        "mask_prob": np.concatenate(all_prob),
        "type_prob": np.concatenate(all_type_prob),
        "raw_geom": np.concatenate(all_geom),
        "angle_rad": np.concatenate(all_angle),
    }


def score_predictions(
    pred: dict[str, np.ndarray],
    arrays: dict[str, Any],
    threshold: float,
) -> dict[str, float]:
    rows = []
    for order, local_idx in enumerate(pred["indices"]):
        local_idx = int(local_idx)
        metric = mask_metric(pred["mask_prob"][order], arrays["masks"][local_idx], threshold)
        rows.append(metric)
    return {
        "iou": safe_mean(rows, "iou"),
        "dice": safe_mean(rows, "dice"),
        "area_error": safe_mean(rows, "area_error"),
    }


def best_threshold_score(pred: dict[str, np.ndarray], arrays: dict[str, Any]) -> dict[str, float]:
    best = {"threshold": math.nan, "iou": -1.0, "dice": -1.0, "area_error": math.inf, "score": -math.inf}
    for threshold in THRESHOLDS:
        metrics = score_predictions(pred, arrays, threshold)
        score = metrics["iou"] + metrics["dice"] - metrics["area_error"]
        if score > best["score"]:
            best = {
                "threshold": threshold,
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "area_error": metrics["area_error"],
                "score": score,
            }
    return best


def prediction_rows(
    pred: dict[str, np.ndarray],
    arrays: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, local_idx_raw in enumerate(pred["indices"]):
        local_idx = int(local_idx_raw)
        true_geom = arrays["raw_geom"][local_idx]
        pred_geom = pred["raw_geom"][order]
        true_angle = math.atan2(float(arrays["angle_targets"][local_idx, 0]), float(arrays["angle_targets"][local_idx, 1]))
        pred_angle = float(pred["angle_rad"][order])
        true_angle_deg = math.degrees(true_angle)
        pred_angle_deg = math.degrees(pred_angle)
        type_prob = pred["type_prob"][order]
        pred_type_id = int(np.argmax(type_prob))
        pred_type = TYPE_NAMES[pred_type_id]
        true_type = str(arrays["defect_types"][local_idx])
        metric = mask_metric(pred["mask_prob"][order], arrays["masks"][local_idx], threshold)
        angle_error = (
            circular_angle_error_deg(pred_angle_deg, true_angle_deg)
            if true_type == "rotated_rect"
            else math.nan
        )
        sin_cos_error = float(np.linalg.norm(
            np.array([math.sin(pred_angle), math.cos(pred_angle)])
            - arrays["angle_targets"][local_idx]
        ))
        rows.append(
            {
                "sample_id": str(arrays["sample_ids"][local_idx]),
                "source_index": int(arrays["source_indices"][local_idx]),
                "split": str(arrays["splits"][local_idx]),
                "defect_type": true_type,
                "source_pack": str(arrays["source_packs"][local_idx]),
                "threshold": threshold,
                **metric,
                "type_prob_rectangular_notch": float(type_prob[0]),
                "type_prob_rotated_rect": float(type_prob[1]),
                "pred_defect_type": pred_type,
                "type_correct": float(pred_type == true_type),
                "true_center_x": float(true_geom[0]),
                "true_center_y": float(true_geom[1]),
                "pred_center_x": float(pred_geom[0]),
                "pred_center_y": float(pred_geom[1]),
                "center_mae_m": float(math.hypot(pred_geom[0] - true_geom[0], pred_geom[1] - true_geom[1])),
                "true_width": float(true_geom[2]),
                "pred_width": float(pred_geom[2]),
                "width_abs_error_m": float(abs(pred_geom[2] - true_geom[2])),
                "true_length": float(true_geom[3]),
                "pred_length": float(pred_geom[3]),
                "length_abs_error_m": float(abs(pred_geom[3] - true_geom[3])),
                "true_depth": float(true_geom[4]),
                "pred_depth": float(pred_geom[4]),
                "depth_abs_error_m": float(abs(pred_geom[4] - true_geom[4])),
                "true_angle_deg": true_angle_deg,
                "pred_angle_deg": pred_angle_deg,
                "angle_abs_error_deg": angle_error,
                "sin_cos_error": sin_cos_error,
                "notes": "",
            }
        )
    return rows


def summarize_group(rows: list[dict[str, Any]], split: str, group_name: str, group_value: str) -> dict[str, Any]:
    return {
        "split": split,
        "group_name": group_name,
        "group_value": group_value,
        "sample_count": len(rows),
        "iou_mean": safe_mean(rows, "iou"),
        "dice_mean": safe_mean(rows, "dice"),
        "area_error_mean": safe_mean(rows, "area_error"),
        "center_error_px_mean": safe_mean(rows, "center_error_px"),
        "pred_area_mean": safe_mean(rows, "pred_area"),
        "true_area_mean": safe_mean(rows, "true_area"),
        "pred_area_zero_mean": safe_mean(rows, "pred_area_zero"),
        "type_accuracy": safe_mean(rows, "type_correct"),
        "center_mae_m": safe_mean(rows, "center_mae_m"),
        "width_mae_m": safe_mean(rows, "width_abs_error_m"),
        "length_mae_m": safe_mean(rows, "length_abs_error_m"),
        "depth_mae_m": safe_mean(rows, "depth_abs_error_m"),
        "angle_mae_deg": safe_mean(rows, "angle_abs_error_deg"),
    }


def build_group_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        group_rows.append(summarize_group(split_rows, split, "overall", "rect_rot"))
        geometry_rows.append(summarize_group(split_rows, split, "overall", "rect_rot"))
        for group_name in ["defect_type", "source_pack"]:
            values = sorted({str(row[group_name]) for row in split_rows})
            for value in values:
                subset = [row for row in split_rows if str(row[group_name]) == value]
                group_rows.append(summarize_group(subset, split, group_name, value))
                geometry_rows.append(summarize_group(subset, split, group_name, value))
        area_values = np.array([to_float(row["true_area"]) for row in split_rows], dtype=float)
        if len(area_values) >= 3:
            q1, q2 = np.quantile(area_values, [1 / 3, 2 / 3])
            for bin_name, predicate in [
                ("small", lambda v: v <= q1),
                ("medium", lambda v: q1 < v <= q2),
                ("large", lambda v: v > q2),
            ]:
                subset = [row for row in split_rows if predicate(to_float(row["true_area"]))]
                group_rows.append(summarize_group(subset, split, "area_bin", bin_name))
    return group_rows, geometry_rows


def failure_category(row: dict[str, Any]) -> str:
    if not bool(row["type_correct"]):
        return "wrong_type"
    if row["defect_type"] == "rotated_rect" and to_float(row["angle_abs_error_deg"]) > 15.0:
        return "wrong_angle"
    if to_float(row["center_mae_m"]) > 0.003 or to_float(row["center_error_px"]) > 8:
        return "wrong_center"
    if max(to_float(row["width_abs_error_m"]), to_float(row["length_abs_error_m"])) > 0.004:
        return "wrong_size"
    if to_float(row["iou"]) < 0.45:
        return "rasterized_geometry_mismatch"
    return "acceptable_or_minor_error"


def build_failure_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    test_rows = [row for row in rows if row["split"] == "test"]
    for row in sorted(test_rows, key=lambda item: to_float(item["iou"]))[:40]:
        out = dict(row)
        out["failure_category"] = failure_category(row)
        cases.append(out)
    return cases


def preview(rows: list[dict[str, Any]], arrays: dict[str, Any], preview_dir: Path, max_count: int = 24) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        return 0
    worst = sorted(test_rows, key=lambda row: to_float(row["iou"]))[:8]
    best = sorted(test_rows, key=lambda row: to_float(row["iou"]))[-8:]
    angle_worst = sorted(
        [row for row in test_rows if row["defect_type"] == "rotated_rect"],
        key=lambda row: to_float(row["angle_abs_error_deg"]),
        reverse=True,
    )[:8]
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in worst + best + angle_worst:
        if row["sample_id"] in seen:
            continue
        seen.add(row["sample_id"])
        chosen.append(row)
        if len(chosen) >= max_count:
            break

    mask_x_t = torch.tensor(arrays["mask_x"], dtype=torch.float32)
    mask_y_t = torch.tensor(arrays["mask_y"], dtype=torch.float32)
    sample_to_local = {str(sample_id): idx for idx, sample_id in enumerate(arrays["sample_ids"])}
    count = 0
    for row in chosen:
        local_idx = sample_to_local[row["sample_id"]]
        pred_angle = 0.0 if row["pred_defect_type"] == "rectangular_notch" else math.radians(to_float(row["pred_angle_deg"], 0.0))
        with torch.no_grad():
            pred_mask = soft_rect_mask(
                mask_x_t,
                mask_y_t,
                torch.tensor([to_float(row["pred_center_x"])], dtype=torch.float32),
                torch.tensor([to_float(row["pred_center_y"])], dtype=torch.float32),
                torch.tensor([to_float(row["pred_width"])], dtype=torch.float32),
                torch.tensor([to_float(row["pred_length"])], dtype=torch.float32),
                torch.tensor([pred_angle], dtype=torch.float32),
            )[0].numpy()
        true_mask = arrays["masks"][local_idx]
        signal = arrays["signals_norm"][local_idx]
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes[0, 0].imshow(true_mask, origin="lower", cmap="gray")
        axes[0, 0].set_title("true mask")
        axes[0, 1].imshow(pred_mask >= to_float(row["threshold"]), origin="lower", cmap="gray")
        axes[0, 1].set_title("pred geometry")
        axes[0, 2].imshow(true_mask, origin="lower", cmap="gray", alpha=0.55)
        axes[0, 2].imshow(pred_mask >= to_float(row["threshold"]), origin="lower", cmap="Reds", alpha=0.45)
        axes[0, 2].set_title(f"IoU {to_float(row['iou']):.3f}")
        for line_idx in range(signal.shape[0]):
            axes[1, 0].plot(arrays["sensor_x"], signal[line_idx], label=f"line {line_idx}")
        axes[1, 0].set_title("delta_bz normalized")
        axes[1, 0].legend(fontsize=7)
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.0,
            0.95,
            "\n".join(
                [
                    f"true: {row['defect_type']}",
                    f"pred: {row['pred_defect_type']}",
                    f"true center=({to_float(row['true_center_x']):.4g},{to_float(row['true_center_y']):.4g})",
                    f"pred center=({to_float(row['pred_center_x']):.4g},{to_float(row['pred_center_y']):.4g})",
                    f"true w/l={to_float(row['true_width']):.4g}/{to_float(row['true_length']):.4g}",
                    f"pred w/l={to_float(row['pred_width']):.4g}/{to_float(row['pred_length']):.4g}",
                    f"angle err={to_float(row['angle_abs_error_deg']):.2f}",
                ]
            ),
            va="top",
            fontsize=8,
        )
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.0,
            0.95,
            f"Dice {to_float(row['dice']):.3f}\nArea err {to_float(row['area_error']):.3f}\nType ok {row['type_correct']}",
            va="top",
            fontsize=9,
        )
        for ax in axes.ravel()[:3]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(row["sample_id"])
        fig.tight_layout()
        out = preview_dir / f"{count:02d}_{row['sample_id']}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        count += 1
    return count


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    arrays, diagnostics = load_arrays(args.npz, args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = RectRotDataset(arrays["split_indices"]["train"], arrays)
    val_ds = RectRotDataset(arrays["split_indices"]["val"], arrays)
    test_ds = RectRotDataset(arrays["split_indices"]["test"], arrays)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = NeuralGeometryHead(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val = {"score": -math.inf, "threshold": 0.5, "iou": 0.0, "dice": 0.0, "area_error": math.inf}
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        parts = defaultdict(float)
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["signal"].to(device))
            loss, loss_parts = batch_losses(outputs, batch, arrays, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            for key, value in loss_parts.items():
                parts[key] += value
            n_batches += 1

        val_pred = predict_split(model, val_ds, arrays, device, args.batch_size)
        val_best = best_threshold_score(val_pred, arrays)
        if val_best["score"] > best_val["score"]:
            best_val = val_best
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": total / max(n_batches, 1),
                "train_bce_loss": parts["bce"] / max(n_batches, 1),
                "train_dice_loss": parts["dice"] / max(n_batches, 1),
                "train_type_loss": parts["type"] / max(n_batches, 1),
                "train_center_loss": parts["center"] / max(n_batches, 1),
                "train_size_loss": parts["size"] / max(n_batches, 1),
                "train_depth_loss": parts["depth"] / max(n_batches, 1),
                "train_angle_loss": parts["angle"] / max(n_batches, 1),
                "best_val_threshold": val_best["threshold"],
                "val_iou": val_best["iou"],
                "val_dice": val_best["dice"],
                "val_area_error": val_best["area_error"],
                "val_score": val_best["score"],
            }
        )
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={epoch_rows[-1]['train_loss']:.4f} "
                f"val_score={val_best['score']:.4f} thr={val_best['threshold']:.2f}"
            )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    selected_threshold = float(best_val["threshold"])
    all_rows: list[dict[str, Any]] = []
    for split, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        pred = predict_split(model, ds, arrays, device, args.batch_size)
        split_rows = prediction_rows(pred, arrays, selected_threshold)
        for row in split_rows:
            row["split"] = split
        all_rows.extend(split_rows)

    group_rows, geometry_rows = build_group_summaries(all_rows)
    failure_rows = build_failure_cases(all_rows)
    preview_count = preview(all_rows, arrays, args.preview_dir, max_count=24)

    write_csv(args.metrics, all_rows, METRIC_FIELDS)
    write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
    write_csv(args.group_summary, group_rows, GROUP_FIELDS)
    write_csv(args.geometry_summary, geometry_rows, GROUP_FIELDS)
    write_csv(args.failure_cases, failure_rows, list(failure_rows[0].keys()) if failure_rows else METRIC_FIELDS)

    summary = write_summary(
        args.summary,
        args.audit,
        args,
        diagnostics,
        all_rows,
        group_rows,
        failure_rows,
        best_epoch,
        best_val,
        selected_threshold,
        preview_count,
        device,
    )
    return summary


def split_summary(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split]
    return {
        "iou": safe_mean(subset, "iou"),
        "dice": safe_mean(subset, "dice"),
        "area_error": safe_mean(subset, "area_error"),
        "center_error_px": safe_mean(subset, "center_error_px"),
        "type_accuracy": safe_mean(subset, "type_correct"),
        "center_mae_m": safe_mean(subset, "center_mae_m"),
        "width_mae_m": safe_mean(subset, "width_abs_error_m"),
        "length_mae_m": safe_mean(subset, "length_abs_error_m"),
        "depth_mae_m": safe_mean(subset, "depth_abs_error_m"),
        "angle_mae_deg": safe_mean([row for row in subset if row["defect_type"] == "rotated_rect"], "angle_abs_error_deg"),
    }


def write_summary(
    summary_path: Path,
    audit_path: Path,
    args: argparse.Namespace,
    diagnostics: dict[str, Any],
    rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    best_epoch: int,
    best_val: dict[str, float],
    selected_threshold: float,
    preview_count: int,
    device: torch.device,
) -> dict[str, Any]:
    train_stats = split_summary(rows, "train")
    val_stats = split_summary(rows, "val")
    test_stats = split_summary(rows, "test")
    test_rect = summarize_group(
        [row for row in rows if row["split"] == "test" and row["defect_type"] == "rectangular_notch"],
        "test",
        "defect_type",
        "rectangular_notch",
    )
    test_rot = summarize_group(
        [row for row in rows if row["split"] == "test" and row["defect_type"] == "rotated_rect"],
        "test",
        "defect_type",
        "rotated_rect",
    )
    piao_iou_gain = test_stats["iou"] - PIAO_WEAK_IOU
    piao_dice_gain = test_stats["dice"] - PIAO_WEAK_DICE
    promising = (
        test_stats["type_accuracy"] >= 0.75
        and test_stats["angle_mae_deg"] <= 15.0
        and test_stats["iou"] > PIAO_WEAK_IOU
        and test_stats["dice"] > PIAO_WEAK_DICE
    )
    no_blocker = diagnostics["n_rect_rot"] == 400 and diagnostics["split_counts"]["train"] > 0

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "COMSOL rect/rot neural geometry head + differentiable rasterization POC summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Input labels: {args.labels}",
        "Scope: rectangular_notch + rotated_rect only. Polygon is parsed for reporting but excluded from train/val/test metrics.",
        "Model input: delta_bz only, shape (3, 201).",
        "Forbidden metadata is not used as input: defect_type, geometry_params, source_pack, angle, vertex_count.",
        "Geometry params and type labels are used only as supervision.",
        "This is a POC, not a baseline; no COMSOL run, no new data, no baseline doc update.",
        "",
        "Model:",
        "- Conv1d BzEncoder -> neural geometry head.",
        "- Outputs type logits, center_x/center_y/width/length/depth normalized params, and angle sin/cos.",
        "- Differentiable rasterization is in the forward path.",
        "- Final mask is predicted type probability mixture: p_rect * angle0 rectangle + p_rot * predicted-angle rectangle.",
        "- No true defect_type hard routing is used.",
        f"- Rasterizer temperature_m = {TEMPERATURE_M}",
        "",
        "Training:",
        f"- seed = {args.seed}",
        f"- epochs = {args.epochs}",
        f"- batch_size = {args.batch_size}",
        f"- device = {device}",
        "- train-only delta_bz normalization.",
        "- train-only geometry target normalization.",
        "- checkpoint/threshold selection uses validation only.",
        "- validation score = IoU + Dice - area_error.",
        f"- best epoch = {best_epoch}",
        f"- selected threshold = {selected_threshold}",
        f"- best validation score = {best_val['score']:.6f}",
        "",
        "Dataset:",
        f"- rect+rotated N = {diagnostics['n_rect_rot']}",
        f"- split counts = {diagnostics['split_counts']}",
        f"- type counts = {diagnostics['type_counts']}",
        "",
        "Mask metrics:",
        f"- train IoU/Dice/area_error = {train_stats['iou']:.4f} / {train_stats['dice']:.4f} / {train_stats['area_error']:.4f}",
        f"- val IoU/Dice/area_error = {val_stats['iou']:.4f} / {val_stats['dice']:.4f} / {val_stats['area_error']:.4f}",
        f"- test IoU/Dice/area_error = {test_stats['iou']:.4f} / {test_stats['dice']:.4f} / {test_stats['area_error']:.4f}",
        f"- test rectangular_notch IoU/Dice = {test_rect['iou_mean']:.4f} / {test_rect['dice_mean']:.4f}",
        f"- test rotated_rect IoU/Dice = {test_rot['iou_mean']:.4f} / {test_rot['dice_mean']:.4f}",
        "",
        "Geometry metrics:",
        f"- test type accuracy = {test_stats['type_accuracy']:.4f}",
        f"- test center MAE m = {test_stats['center_mae_m']:.6f}",
        f"- test width MAE m = {test_stats['width_mae_m']:.6f}",
        f"- test length MAE m = {test_stats['length_mae_m']:.6f}",
        f"- test depth MAE m = {test_stats['depth_mae_m']:.6f}",
        f"- test rotated angle MAE deg = {test_stats['angle_mae_deg']:.4f}",
        "",
        "Comparisons:",
        f"- Piao weak adaptation rect+rotated test IoU/Dice = {PIAO_WEAK_IOU:.4f} / {PIAO_WEAK_DICE:.4f}",
        f"- This POC test IoU/Dice gain vs Piao weak adaptation = {piao_iou_gain:.4f} / {piao_dice_gain:.4f}",
        f"- Piao weak adaptation all-3 type accuracy = {PIAO_WEAK_TYPE_ACC:.4f}; this rect/rot type accuracy = {test_stats['type_accuracy']:.4f}",
        f"- Piao weak adaptation angle MAE = {PIAO_WEAK_ANGLE_MAE:.2f} deg; this POC angle MAE = {test_stats['angle_mae_deg']:.2f} deg",
        f"- Dense COMSOL single-defect baseline test IoU/Dice = {DENSE_SINGLE_BASELINE_IOU:.4f} / {DENSE_SINGLE_BASELINE_DICE:.4f}",
        "- This POC does not need to beat the dense baseline; it tests explicit geometry inversion.",
        "",
        f"Preview PNG generated: {preview_count} (not for submission)",
        f"No label / rasterizer / geometry learning blocker found: {no_blocker}",
        f"POC promising by acceptance criteria: {promising}",
        "Next recommendation: "
        + ("A. Add lightweight forward consistency." if promising else "B. Fix neural geometry head before forward consistency."),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    categories = defaultdict(int)
    for row in rows:
        if row["split"] == "test":
            categories[failure_category(row)] += 1
    audit_lines = [
        "COMSOL rect/rot neural geometry head failure audit summary",
        "",
        f"Failure categories on test: {dict(sorted(categories.items()))}",
        "",
        "Audit focus:",
        "- Type confusion is evaluated from predicted type probabilities; true defect_type is not used for routing.",
        "- Predicted geometry is rasterized, so boundary shape is rectangle-like by construction rather than blob-like.",
        "- Failure cases should be attributed to wrong type, center, size, or angle errors.",
        "",
        "Worst test cases:",
    ]
    for row in failure_rows[:20]:
        audit_lines.append(
            f"- {row['sample_id']}: {row.get('failure_category', failure_category(row))}, "
            f"IoU={to_float(row['iou']):.3f}, Dice={to_float(row['dice']):.3f}, "
            f"type={row['defect_type']}->{row['pred_defect_type']}, "
            f"angle_error={to_float(row['angle_abs_error_deg']):.2f}"
        )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    return {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "promising": promising,
        "no_blocker": no_blocker,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY_SUMMARY)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE_CASES)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
