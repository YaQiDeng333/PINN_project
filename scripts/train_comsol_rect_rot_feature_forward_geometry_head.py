from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_rect_rot_geometry_forward_surrogate as forward_mod  # noqa: E402
import train_comsol_rect_rot_neural_geometry_head_v2_poc as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = base.DEFAULT_NPZ
DEFAULT_LABELS = base.DEFAULT_LABELS
DEFAULT_FEATURES = PROJECT_ROOT / "results/metrics/comsol_mfl_physics_features.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_feature_forward_geometry_head_summary.txt"
DEFAULT_AUDIT = (
    PROJECT_ROOT / "results/summaries/comsol_rect_rot_feature_forward_geometry_head_failure_audit_summary.txt"
)
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_feature_forward_geometry_head_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_feature_forward_geometry_head_epoch_log.csv"
DEFAULT_GROUP_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_feature_forward_geometry_head_group_summary.csv"
DEFAULT_GEOMETRY_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_feature_forward_geometry_head_geometry_summary.csv"
DEFAULT_FORWARD_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_feature_forward_geometry_head_forward_summary.csv"
DEFAULT_FAILURE_CASES = PROJECT_ROOT / "results/metrics/comsol_rect_rot_feature_forward_geometry_head_failure_cases.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_rect_rot_feature_forward_geometry_head"

SEED = 42
EPOCHS = 200
BATCH_SIZE = 8
MAX_ANGLE_RAD = math.radians(35.0)
THRESHOLDS = base.THRESHOLDS

REF_2048_IOU = 0.5908
REF_2048_DICE = 0.7385
REF_2048_TYPE_ACC = 0.6364
REF_2048_ANGLE_MAE = 20.14
REF_2050_IOU = 0.5877
REF_2050_DICE = 0.7351
REF_2050_TYPE_ACC = 0.6515
REF_2050_ANGLE_MAE = 20.7332
PIAO_WEAK_IOU = base.PIAO_WEAK_IOU
PIAO_WEAK_DICE = base.PIAO_WEAK_DICE
DENSE_SINGLE_BASELINE_IOU = base.DENSE_SINGLE_BASELINE_IOU
DENSE_SINGLE_BASELINE_DICE = base.DENSE_SINGLE_BASELINE_DICE

METRIC_FIELDS = base.METRIC_FIELDS[:-1] + [
    "type_entropy",
    "forward_mse",
    "forward_mae",
    "forward_rmse",
    "forward_nrmse",
    "forward_correlation",
    "forward_line0_mse",
    "forward_line1_mse",
    "forward_line2_mse",
    "forward_amplitude_abs_error",
    "forward_abs_peak_index_error_mean",
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
    "train_forward_loss",
    "best_val_threshold",
    "val_iou",
    "val_dice",
    "val_area_error",
    "val_type_accuracy",
    "val_angle_mae_deg",
    "val_forward_nrmse",
    "val_score",
]

FORWARD_GROUP_FIELDS = [
    "split",
    "group_name",
    "group_value",
    "sample_count",
    "forward_mse_mean",
    "forward_mae_mean",
    "forward_rmse_mean",
    "forward_nrmse_mean",
    "forward_correlation_mean",
    "forward_amplitude_abs_error_mean",
    "forward_abs_peak_index_error_mean",
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


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    return base.safe_mean(rows, key)


def load_feature_matrix(features_path: Path, arrays: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    rows = read_csv(features_path)
    by_id = {row["sample_id"]: row for row in rows}
    z_fields = [field for field in rows[0] if field.startswith("z_")]
    if not z_fields:
        raise RuntimeError(f"No train-scaled z_ feature columns found in {features_path}")
    matrix = []
    missing = []
    for sample_id in arrays["sample_ids"]:
        row = by_id.get(str(sample_id))
        if row is None:
            missing.append(str(sample_id))
            matrix.append([0.0] * len(z_fields))
            continue
        matrix.append([base.to_float(row[field], 0.0) for field in z_fields])
    feature_matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Feature matrix contains NaN/inf")
    diagnostics = {
        "feature_dim": int(feature_matrix.shape[1]),
        "missing_feature_rows": len(missing),
        "feature_policy": "z_ columns derived from delta_bz/sensor_x/scan_line_y with train-only scaler",
    }
    return feature_matrix, z_fields, diagnostics


class FeatureRectRotDataset(Dataset):
    def __init__(self, indices: np.ndarray, arrays: dict[str, Any]):
        self.indices = indices.astype(np.int64)
        self.signals = arrays["signals_norm"][self.indices]
        self.features = arrays["feature_matrix"][self.indices]
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
            "features": torch.from_numpy(self.features[idx]).float(),
            "mask": torch.from_numpy(self.masks[idx]).float(),
            "type_target": torch.tensor(int(self.type_targets[idx]), dtype=torch.long),
            "geom_target": torch.from_numpy(self.geom_targets[idx]).float(),
            "angle_target": torch.from_numpy(self.angle_targets[idx]).float(),
        }


class FeatureForwardGeometryHead(nn.Module):
    def __init__(self, feature_dim: int, raw_latent_dim: int = 128, feature_latent_dim: int = 64):
        super().__init__()
        self.raw_encoder = base.BzEncoder(latent_dim=raw_latent_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, feature_latent_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(raw_latent_dim + feature_latent_dim, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 96),
            nn.GELU(),
        )
        self.type_head = nn.Linear(96, 2)
        self.geom_head = nn.Linear(96, 6)

    def forward(self, signal: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = torch.cat([self.raw_encoder(signal), self.feature_encoder(features)], dim=1)
        hidden = self.fusion(latent)
        raw = self.geom_head(hidden)
        angle_rad = torch.tanh(raw[:, 5]) * MAX_ANGLE_RAD
        return {
            "type_logits": self.type_head(hidden),
            "geom_norm": raw[:, :5],
            "angle_rad": angle_rad,
            "angle_norm_target": (angle_rad / MAX_ANGLE_RAD).unsqueeze(1),
        }


def decode_outputs(
    outputs: dict[str, torch.Tensor],
    arrays: dict[str, Any],
    forward_model: nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    geom_mean = torch.tensor(arrays["geom_mean"], device=device).view(1, -1)
    geom_std = torch.tensor(arrays["geom_std"], device=device).view(1, -1)
    raw_geom = outputs["geom_norm"] * geom_std + geom_mean
    mask_x = torch.tensor(arrays["mask_x"], device=device)
    mask_y = torch.tensor(arrays["mask_y"], device=device)
    cx = raw_geom[:, 0].clamp(float(mask_x.min()), float(mask_x.max()))
    cy = raw_geom[:, 1].clamp(float(mask_y.min()), float(mask_y.max()))
    width = raw_geom[:, 2].clamp(0.001, 0.025)
    length = raw_geom[:, 3].clamp(0.001, 0.020)
    depth = raw_geom[:, 4].clamp(0.0001, 0.004)
    clamped_geom = torch.stack([cx, cy, width, length, depth], dim=1)
    clamped_geom_norm = (clamped_geom - geom_mean) / geom_std
    type_prob = torch.softmax(outputs["type_logits"], dim=1)
    angle_rad = outputs["angle_rad"]
    rect_prob = base.soft_rect_mask(mask_x, mask_y, cx, cy, width, length, torch.zeros_like(angle_rad))
    rot_prob = base.soft_rect_mask(mask_x, mask_y, cx, cy, width, length, angle_rad)
    mask_prob = type_prob[:, 0].view(-1, 1, 1) * rect_prob + type_prob[:, 1].view(-1, 1, 1) * rot_prob
    mask_logits = torch.logit(mask_prob.clamp(1e-4, 1.0 - 1e-4))
    forward_input = torch.cat(
        [
            type_prob,
            clamped_geom_norm,
            torch.sin(angle_rad).unsqueeze(1),
            torch.cos(angle_rad).unsqueeze(1),
        ],
        dim=1,
    )
    forward_pred = forward_model(forward_input).view(-1, 3, 201)
    return {
        "raw_geom": clamped_geom,
        "geom_norm": clamped_geom_norm,
        "angle_rad": angle_rad,
        "type_prob": type_prob,
        "mask_prob": mask_prob,
        "mask_logits": mask_logits,
        "forward_pred": forward_pred,
    }


def batch_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    arrays: dict[str, Any],
    forward_model: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    decoded = decode_outputs(outputs, arrays, forward_model, device)
    target_mask = batch["mask"].to(device)
    bce = F.binary_cross_entropy_with_logits(decoded["mask_logits"], target_mask)
    dice = base.soft_dice_loss(decoded["mask_prob"], target_mask)
    type_loss = F.cross_entropy(outputs["type_logits"], batch["type_target"].to(device))
    geom_target = batch["geom_target"].to(device)
    center_loss = F.smooth_l1_loss(outputs["geom_norm"][:, :2], geom_target[:, :2])
    size_loss = F.smooth_l1_loss(outputs["geom_norm"][:, 2:4], geom_target[:, 2:4])
    depth_loss = F.smooth_l1_loss(outputs["geom_norm"][:, 4:5], geom_target[:, 4:5])
    true_angle = torch.atan2(batch["angle_target"].to(device)[:, 0], batch["angle_target"].to(device)[:, 1])
    angle_loss = F.smooth_l1_loss(outputs["angle_norm_target"].squeeze(1), true_angle / MAX_ANGLE_RAD)
    forward_loss = F.mse_loss(decoded["forward_pred"], batch["signal"].to(device))
    loss = (
        bce
        + dice
        + 0.40 * type_loss
        + 0.15 * center_loss
        + 0.15 * size_loss
        + 0.05 * depth_loss
        + 0.20 * angle_loss
        + 0.10 * forward_loss
    )
    return loss, {
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "type": float(type_loss.detach().cpu()),
        "center": float(center_loss.detach().cpu()),
        "size": float(size_loss.detach().cpu()),
        "depth": float(depth_loss.detach().cpu()),
        "angle": float(angle_loss.detach().cpu()),
        "forward": float(forward_loss.detach().cpu()),
    }


def predict(
    model: nn.Module,
    dataset: FeatureRectRotDataset,
    arrays: dict[str, Any],
    forward_model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["signal"].to(device), batch["features"].to(device))
            decoded = decode_outputs(outputs, arrays, forward_model, device)
            chunks["indices"].append(batch["source_index"].cpu().numpy())
            chunks["mask_prob"].append(decoded["mask_prob"].cpu().numpy())
            chunks["type_prob"].append(decoded["type_prob"].cpu().numpy())
            chunks["raw_geom"].append(decoded["raw_geom"].cpu().numpy())
            chunks["angle_rad"].append(decoded["angle_rad"].cpu().numpy())
            chunks["forward_pred"].append(decoded["forward_pred"].cpu().numpy())
            chunks["signal"].append(batch["signal"].cpu().numpy())
    return {key: np.concatenate(value) for key, value in chunks.items()}


def prediction_rows(
    pred: dict[str, np.ndarray],
    arrays: dict[str, Any],
    split: str,
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
        true_type = str(arrays["defect_types"][local_idx])
        type_prob = pred["type_prob"][order]
        pred_type = base.TYPE_NAMES[int(np.argmax(type_prob))]
        mask_metric = base.mask_metric(pred["mask_prob"][order], arrays["masks"][local_idx], threshold)
        angle_error = (
            base.circular_angle_error_deg(pred_angle_deg, true_angle_deg)
            if true_type == "rotated_rect"
            else math.nan
        )
        sin_cos_error = float(
            np.linalg.norm(np.array([math.sin(pred_angle), math.cos(pred_angle)]) - arrays["angle_targets"][local_idx])
        )
        forward_stats = forward_mod.signal_metrics(pred["forward_pred"][order], pred["signal"][order])
        type_entropy = -float(np.sum(type_prob * np.log(type_prob + 1e-12)))
        rows.append(
            {
                "sample_id": str(arrays["sample_ids"][local_idx]),
                "source_index": int(arrays["source_indices"][local_idx]),
                "split": split,
                "defect_type": true_type,
                "source_pack": str(arrays["source_packs"][local_idx]),
                "threshold": threshold,
                **mask_metric,
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
                "type_entropy": type_entropy,
                "forward_mse": forward_stats["mse"],
                "forward_mae": forward_stats["mae"],
                "forward_rmse": forward_stats["rmse"],
                "forward_nrmse": forward_stats["nrmse"],
                "forward_correlation": forward_stats["correlation"],
                "forward_line0_mse": forward_stats["line0_mse"],
                "forward_line1_mse": forward_stats["line1_mse"],
                "forward_line2_mse": forward_stats["line2_mse"],
                "forward_amplitude_abs_error": forward_stats["amplitude_abs_error"],
                "forward_abs_peak_index_error_mean": forward_stats["abs_peak_index_error_mean"],
                "notes": "",
            }
        )
    return rows


def split_summary(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    stats = base.split_summary(rows, split)
    subset = [row for row in rows if row["split"] == split]
    stats.update(
        {
            "forward_mse": safe_mean(subset, "forward_mse"),
            "forward_mae": safe_mean(subset, "forward_mae"),
            "forward_rmse": safe_mean(subset, "forward_rmse"),
            "forward_nrmse": safe_mean(subset, "forward_nrmse"),
            "forward_correlation": safe_mean(subset, "forward_correlation"),
            "type_entropy": safe_mean(subset, "type_entropy"),
        }
    )
    return stats


def score_rows(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    stats = split_summary(rows, split)
    score = (
        stats["iou"]
        + stats["dice"]
        - stats["area_error"]
        + 0.10 * stats["type_accuracy"]
        - 0.003 * stats["angle_mae_deg"]
        - 0.05 * stats["forward_nrmse"]
    )
    if not math.isfinite(score):
        score = stats["iou"] + stats["dice"] - stats["area_error"] + 0.10 * stats["type_accuracy"]
    return {**stats, "score": score}


def best_threshold(pred: dict[str, np.ndarray], arrays: dict[str, Any]) -> dict[str, float]:
    best = {"score": -math.inf, "threshold": math.nan}
    for threshold in THRESHOLDS:
        rows = prediction_rows(pred, arrays, "val", threshold)
        current = score_rows(rows, "val")
        current["threshold"] = threshold
        if current["score"] > best["score"]:
            best = current
    return best


def build_forward_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        for group_name, values in [
            ("overall", ["rect_rot"]),
            ("defect_type", sorted({str(row["defect_type"]) for row in split_rows})),
            ("source_pack", sorted({str(row["source_pack"]) for row in split_rows})),
        ]:
            for value in values:
                subset = split_rows if group_name == "overall" else [row for row in split_rows if str(row[group_name]) == value]
                out.append(
                    {
                        "split": split,
                        "group_name": group_name,
                        "group_value": value,
                        "sample_count": len(subset),
                        "forward_mse_mean": safe_mean(subset, "forward_mse"),
                        "forward_mae_mean": safe_mean(subset, "forward_mae"),
                        "forward_rmse_mean": safe_mean(subset, "forward_rmse"),
                        "forward_nrmse_mean": safe_mean(subset, "forward_nrmse"),
                        "forward_correlation_mean": safe_mean(subset, "forward_correlation"),
                        "forward_amplitude_abs_error_mean": safe_mean(subset, "forward_amplitude_abs_error"),
                        "forward_abs_peak_index_error_mean": safe_mean(subset, "forward_abs_peak_index_error_mean"),
                    }
                )
    return out


def failure_category(row: dict[str, Any]) -> str:
    if base.to_float(row["forward_nrmse"]) > 1.0:
        return "forward_surrogate_mismatch"
    return base.failure_category(row)


def build_failure_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for row in sorted([r for r in rows if r["split"] == "test"], key=lambda item: base.to_float(item["iou"]))[:40]:
        out = dict(row)
        out["failure_category"] = failure_category(row)
        cases.append(out)
    return cases


def preview(
    rows: list[dict[str, Any]],
    arrays: dict[str, Any],
    pred_cache: dict[str, dict[str, np.ndarray]],
    preview_dir: Path,
    max_count: int = 24,
) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        return 0
    worst = sorted(test_rows, key=lambda row: base.to_float(row["iou"]))[:8]
    best = sorted(test_rows, key=lambda row: base.to_float(row["iou"]))[-8:]
    forward_worst = sorted(test_rows, key=lambda row: base.to_float(row["forward_nrmse"]), reverse=True)[:8]
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in worst + best + forward_worst:
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
        pred_angle = math.radians(base.to_float(row["pred_angle_deg"], 0.0))
        with torch.no_grad():
            pred_mask = base.soft_rect_mask(
                mask_x_t,
                mask_y_t,
                torch.tensor([base.to_float(row["pred_center_x"])], dtype=torch.float32),
                torch.tensor([base.to_float(row["pred_center_y"])], dtype=torch.float32),
                torch.tensor([base.to_float(row["pred_width"])], dtype=torch.float32),
                torch.tensor([base.to_float(row["pred_length"])], dtype=torch.float32),
                torch.tensor([pred_angle], dtype=torch.float32),
            )[0].numpy()
        pred_info = pred_cache[row["sample_id"]]
        observed = pred_info["signal"]
        forward_pred = pred_info["forward_pred"]
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        axes[0, 0].imshow(arrays["masks"][local_idx], origin="lower", cmap="gray")
        axes[0, 0].set_title("true mask")
        axes[0, 1].imshow(pred_mask >= base.to_float(row["threshold"]), origin="lower", cmap="gray")
        axes[0, 1].set_title("pred geometry")
        axes[0, 2].imshow(arrays["masks"][local_idx], origin="lower", cmap="gray", alpha=0.55)
        axes[0, 2].imshow(pred_mask >= base.to_float(row["threshold"]), origin="lower", cmap="Reds", alpha=0.45)
        axes[0, 2].set_title(f"IoU {base.to_float(row['iou']):.3f}")
        for line_idx in range(3):
            axes[1, 0].plot(arrays["sensor_x"], observed[line_idx], label=f"obs {line_idx}")
            axes[1, 1].plot(arrays["sensor_x"], forward_pred[line_idx], label=f"fwd {line_idx}")
        axes[1, 0].set_title("observed delta_bz norm")
        axes[1, 1].set_title(f"forward pred nrmse {base.to_float(row['forward_nrmse']):.3f}")
        axes[1, 0].legend(fontsize=6)
        axes[1, 1].legend(fontsize=6)
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.0,
            0.95,
            "\n".join(
                [
                    f"true: {row['defect_type']}",
                    f"pred: {row['pred_defect_type']}",
                    f"p_rect={base.to_float(row['type_prob_rectangular_notch']):.3f}",
                    f"p_rot={base.to_float(row['type_prob_rotated_rect']):.3f}",
                    f"Dice={base.to_float(row['dice']):.3f}",
                    f"angle err={base.to_float(row['angle_abs_error_deg']):.2f}",
                    f"fwd corr={base.to_float(row['forward_correlation']):.3f}",
                ]
            ),
            va="top",
            fontsize=8,
        )
        for ax in axes.ravel()[:3]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(row["sample_id"])
        fig.tight_layout()
        fig.savefig(preview_dir / f"{count:02d}_{row['sample_id']}.png", dpi=140)
        plt.close(fig)
        count += 1
    return count


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    surrogate_args = argparse.Namespace(
        npz=args.npz,
        labels=args.labels,
        features=args.features,
        input_check_summary=args.input_check_summary,
        input_check=args.input_check,
        summary=args.surrogate_summary,
        metrics=args.surrogate_metrics,
        epoch_log=args.surrogate_epoch_log,
        group_summary=args.surrogate_group_summary,
        seed=args.seed,
        epochs=args.forward_epochs,
        batch_size=args.forward_batch_size,
        lr=args.forward_lr,
        cpu=args.cpu,
    )
    surrogate_bundle = forward_mod.train_forward_surrogate(surrogate_args, write_outputs=True)
    arrays = surrogate_bundle.arrays
    feature_matrix, feature_cols, feature_diag = load_feature_matrix(args.features, arrays)
    arrays = dict(arrays)
    arrays["feature_matrix"] = feature_matrix
    arrays["feature_columns"] = feature_cols
    if feature_diag["missing_feature_rows"] != 0:
        raise RuntimeError(f"Missing feature rows for rect/rot samples: {feature_diag['missing_feature_rows']}")
    device = surrogate_bundle.device
    forward_model = surrogate_bundle.model.to(device)
    forward_model.eval()
    for param in forward_model.parameters():
        param.requires_grad_(False)

    train_ds = FeatureRectRotDataset(arrays["split_indices"]["train"], arrays)
    val_ds = FeatureRectRotDataset(arrays["split_indices"]["val"], arrays)
    test_ds = FeatureRectRotDataset(arrays["split_indices"]["test"], arrays)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = FeatureForwardGeometryHead(feature_dim=feature_matrix.shape[1], raw_latent_dim=args.raw_latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.inverse_lr, weight_decay=1e-4)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val = {"score": -math.inf, "threshold": math.nan}
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.inverse_epochs + 1):
        model.train()
        totals = defaultdict(float)
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["signal"].to(device), batch["features"].to(device))
            loss, parts = batch_loss(outputs, batch, arrays, forward_model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            for key, value in parts.items():
                totals[key] += value
            n_batches += 1

        val_pred = predict(model, val_ds, arrays, forward_model, device, args.batch_size)
        val_best = best_threshold(val_pred, arrays)
        if val_best["score"] > best_val["score"]:
            best_val = val_best
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                "train_bce_loss": totals["bce"] / max(n_batches, 1),
                "train_dice_loss": totals["dice"] / max(n_batches, 1),
                "train_type_loss": totals["type"] / max(n_batches, 1),
                "train_center_loss": totals["center"] / max(n_batches, 1),
                "train_size_loss": totals["size"] / max(n_batches, 1),
                "train_depth_loss": totals["depth"] / max(n_batches, 1),
                "train_angle_loss": totals["angle"] / max(n_batches, 1),
                "train_forward_loss": totals["forward"] / max(n_batches, 1),
                "best_val_threshold": val_best["threshold"],
                "val_iou": val_best["iou"],
                "val_dice": val_best["dice"],
                "val_area_error": val_best["area_error"],
                "val_type_accuracy": val_best["type_accuracy"],
                "val_angle_mae_deg": val_best["angle_mae_deg"],
                "val_forward_nrmse": val_best["forward_nrmse"],
                "val_score": val_best["score"],
            }
        )
        if epoch == 1 or epoch % 25 == 0 or epoch == args.inverse_epochs:
            print(
                f"inverse epoch={epoch:03d} loss={epoch_rows[-1]['train_loss']:.4f} "
                f"val_score={val_best['score']:.4f} thr={val_best['threshold']:.2f} "
                f"fwd={val_best['forward_nrmse']:.3f}"
            )

    if best_state is None:
        raise RuntimeError("No inverse validation checkpoint selected")
    model.load_state_dict(best_state)
    threshold = float(best_val["threshold"])
    all_rows: list[dict[str, Any]] = []
    pred_cache: dict[str, dict[str, np.ndarray]] = {}
    for split, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        pred = predict(model, ds, arrays, forward_model, device, args.batch_size)
        rows = prediction_rows(pred, arrays, split, threshold)
        all_rows.extend(rows)
        for order, local_idx_raw in enumerate(pred["indices"]):
            sample_id = str(arrays["sample_ids"][int(local_idx_raw)])
            pred_cache[sample_id] = {
                "forward_pred": pred["forward_pred"][order],
                "signal": pred["signal"][order],
            }

    group_rows, geometry_rows = base.build_group_summaries(all_rows)
    forward_rows = build_forward_summary(all_rows)
    failure_rows = build_failure_cases(all_rows)
    preview_count = preview(all_rows, arrays, pred_cache, args.preview_dir, max_count=24)

    write_csv(args.metrics, all_rows, METRIC_FIELDS)
    write_csv(args.epoch_log, epoch_rows, EPOCH_FIELDS)
    write_csv(args.group_summary, group_rows, base.GROUP_FIELDS)
    write_csv(args.geometry_summary, geometry_rows, base.GROUP_FIELDS)
    write_csv(args.forward_summary, forward_rows, FORWARD_GROUP_FIELDS)
    write_csv(args.failure_cases, failure_rows, list(failure_rows[0].keys()) if failure_rows else METRIC_FIELDS)
    summary = write_summary(
        args,
        all_rows,
        failure_rows,
        best_epoch,
        best_val,
        threshold,
        preview_count,
        surrogate_bundle,
        feature_diag,
        device,
    )
    return summary


def write_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    best_epoch: int,
    best_val: dict[str, float],
    threshold: float,
    preview_count: int,
    surrogate_bundle: forward_mod.ForwardSurrogateBundle,
    feature_diag: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    train_stats = split_summary(rows, "train")
    val_stats = split_summary(rows, "val")
    test_stats = split_summary(rows, "test")
    test_type = base.type_report(rows, "test")
    categories = defaultdict(int)
    for row in rows:
        if row["split"] == "test":
            categories[failure_category(row)] += 1
    promising = (
        (test_stats["iou"] >= REF_2048_IOU + 0.015 or test_stats["dice"] >= REF_2048_DICE + 0.012)
        and test_stats["type_accuracy"] >= 0.70
        and test_stats["angle_mae_deg"] <= 18.0
        and test_stats["area_error"] <= 0.28
        and test_stats["forward_nrmse"] <= 1.0
    )
    recommendation = (
        "A. Continue direct neural geometry head."
        if promising
        else "B. Priewald-style coarse-to-fine refinement."
    )
    lines = [
        "COMSOL rect/rot feature-assisted geometry head + lightweight forward consistency summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Geometry labels: {args.labels}",
        f"Physics features: {args.features}",
        "Scope: rectangular_notch + rotated_rect only; polygon parsed/reported but excluded.",
        "Inverse model input: raw delta_bz and delta_bz-derived physics/NLS-style features only.",
        "Forbidden metadata is not used as inverse input: defect_type, geometry_params, source_pack, true angle, true size, mask.",
        "Labels are used only for supervision and metrics.",
        "This is a candidate, not a baseline; no COMSOL run, no new data, no baseline doc update.",
        "",
        "Forward surrogate:",
        f"- best epoch = {surrogate_bundle.best_epoch}",
        f"- val MSE/NRMSE/corr = {surrogate_bundle.best_val['mse']:.6f} / {surrogate_bundle.best_val['nrmse']:.6f} / {surrogate_bundle.best_val['correlation']:.4f}",
        "- frozen during inverse training",
        "- consistency uses predicted geometry vector, not true geometry",
        "",
        "Feature encoder:",
        f"- feature_dim = {feature_diag['feature_dim']}",
        f"- missing feature rows = {feature_diag['missing_feature_rows']}",
        "- features are train-scaled z_ columns from the physics feature extractor.",
        "",
        "Inverse training:",
        f"- device = {device}",
        f"- seed = {args.seed}",
        f"- inverse epochs = {args.inverse_epochs}",
        f"- best epoch = {best_epoch}",
        f"- selected threshold = {threshold}",
        "- checkpoint and threshold selection use validation only.",
        "- validation score = IoU + Dice - area_error + 0.10*type_accuracy - 0.003*rotated_angle_mae_deg - 0.05*forward_nrmse.",
        "",
        "Mask metrics:",
        f"- train IoU/Dice/area_error = {train_stats['iou']:.4f} / {train_stats['dice']:.4f} / {train_stats['area_error']:.4f}",
        f"- val IoU/Dice/area_error = {val_stats['iou']:.4f} / {val_stats['dice']:.4f} / {val_stats['area_error']:.4f}",
        f"- test IoU/Dice/area_error = {test_stats['iou']:.4f} / {test_stats['dice']:.4f} / {test_stats['area_error']:.4f}",
        "",
        "Type / geometry metrics:",
        f"- test type accuracy = {test_stats['type_accuracy']:.4f}",
        f"- test rect precision / recall = {test_type['rect_precision']:.4f} / {test_type['rect_recall']:.4f}",
        f"- test rotated precision / recall = {test_type['rotated_precision']:.4f} / {test_type['rotated_recall']:.4f}",
        f"- test type confusion rect->rotated / rotated->rect = {int(test_type['rect_as_rotated'])} / {int(test_type['rotated_as_rect'])}",
        f"- test type entropy = {test_stats['type_entropy']:.4f}",
        f"- test center MAE m = {test_stats['center_mae_m']:.6f}",
        f"- test width MAE m = {test_stats['width_mae_m']:.6f}",
        f"- test length MAE m = {test_stats['length_mae_m']:.6f}",
        f"- test depth MAE m = {test_stats['depth_mae_m']:.6f}",
        f"- test rotated angle MAE deg = {test_stats['angle_mae_deg']:.4f}",
        "",
        "Forward residual metrics:",
        f"- train forward MSE/NRMSE/corr = {train_stats['forward_mse']:.6f} / {train_stats['forward_nrmse']:.6f} / {train_stats['forward_correlation']:.4f}",
        f"- val forward MSE/NRMSE/corr = {val_stats['forward_mse']:.6f} / {val_stats['forward_nrmse']:.6f} / {val_stats['forward_correlation']:.4f}",
        f"- test forward MSE/NRMSE/corr = {test_stats['forward_mse']:.6f} / {test_stats['forward_nrmse']:.6f} / {test_stats['forward_correlation']:.4f}",
        "",
        "Comparisons:",
        f"- 20.48 IoU/Dice/type/angle = {REF_2048_IOU:.4f} / {REF_2048_DICE:.4f} / {REF_2048_TYPE_ACC:.4f} / {REF_2048_ANGLE_MAE:.2f}",
        f"- 20.50 IoU/Dice/type/angle = {REF_2050_IOU:.4f} / {REF_2050_DICE:.4f} / {REF_2050_TYPE_ACC:.4f} / {REF_2050_ANGLE_MAE:.2f}",
        f"- gain vs 20.48 IoU/Dice/type/angle-improvement = {test_stats['iou'] - REF_2048_IOU:.4f} / {test_stats['dice'] - REF_2048_DICE:.4f} / {test_stats['type_accuracy'] - REF_2048_TYPE_ACC:.4f} / {REF_2048_ANGLE_MAE - test_stats['angle_mae_deg']:.2f}",
        f"- gain vs 20.50 IoU/Dice/type/angle-improvement = {test_stats['iou'] - REF_2050_IOU:.4f} / {test_stats['dice'] - REF_2050_DICE:.4f} / {test_stats['type_accuracy'] - REF_2050_TYPE_ACC:.4f} / {REF_2050_ANGLE_MAE - test_stats['angle_mae_deg']:.2f}",
        f"- Piao weak adaptation IoU/Dice = {PIAO_WEAK_IOU:.4f} / {PIAO_WEAK_DICE:.4f}",
        f"- Dense single-defect baseline IoU/Dice = {DENSE_SINGLE_BASELINE_IOU:.4f} / {DENSE_SINGLE_BASELINE_DICE:.4f}",
        "",
        f"Failure categories on test: {dict(sorted(categories.items()))}",
        f"Preview PNG generated: {preview_count} (not for submission)",
        f"Candidate promising by acceptance criteria: {promising}",
        f"Next recommendation: {recommendation}",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit_lines = [
        "COMSOL rect/rot feature-forward geometry head failure audit summary",
        "",
        f"Failure categories on test: {dict(sorted(categories.items()))}",
        f"Type report on test: {test_type}",
        f"Forward residual test NRMSE: {test_stats['forward_nrmse']:.6f}",
        "",
        "Conclusion:",
        (
            "- Candidate passed acceptance; direct neural geometry head may continue."
            if promising
            else "- Candidate did not pass acceptance; stop direct head tuning and prefer Priewald-style coarse-to-fine refinement."
        ),
        "",
        "Worst test cases:",
    ]
    for row in failure_rows[:20]:
        audit_lines.append(
            f"- {row['sample_id']}: {row.get('failure_category', failure_category(row))}, "
            f"IoU={base.to_float(row['iou']):.3f}, Dice={base.to_float(row['dice']):.3f}, "
            f"type={row['defect_type']}->{row['pred_defect_type']}, "
            f"angle_error={base.to_float(row['angle_abs_error_deg']):.2f}, "
            f"forward_nrmse={base.to_float(row['forward_nrmse']):.3f}"
        )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    return {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "promising": promising,
        "recommendation": recommendation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--input-check-summary", type=Path, default=forward_mod.DEFAULT_INPUT_CHECK_SUMMARY)
    parser.add_argument("--input-check", type=Path, default=forward_mod.DEFAULT_INPUT_CHECK)
    parser.add_argument("--surrogate-summary", type=Path, default=forward_mod.DEFAULT_SUMMARY)
    parser.add_argument("--surrogate-metrics", type=Path, default=forward_mod.DEFAULT_METRICS)
    parser.add_argument("--surrogate-epoch-log", type=Path, default=forward_mod.DEFAULT_EPOCH_LOG)
    parser.add_argument("--surrogate-group-summary", type=Path, default=forward_mod.DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY_SUMMARY)
    parser.add_argument("--forward-summary", type=Path, default=DEFAULT_FORWARD_SUMMARY)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE_CASES)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--forward-epochs", type=int, default=forward_mod.EPOCHS)
    parser.add_argument("--forward-batch-size", type=int, default=32)
    parser.add_argument("--forward-lr", type=float, default=1e-3)
    parser.add_argument("--inverse-epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--inverse-lr", type=float, default=2e-3)
    parser.add_argument("--raw-latent-dim", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
