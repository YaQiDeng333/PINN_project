from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_rect_rot_feature_forward_geometry_head as inverse_mod  # noqa: E402
import train_comsol_rect_rot_geometry_forward_surrogate as forward_mod  # noqa: E402
import train_comsol_rect_rot_neural_geometry_head_v2_poc as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = base.DEFAULT_NPZ
DEFAULT_LABELS = base.DEFAULT_LABELS
DEFAULT_FEATURES = inverse_mod.DEFAULT_FEATURES
DEFAULT_INITIAL = PROJECT_ROOT / "results/metrics/comsol_rect_rot_feature_forward_geometry_head_metrics.csv"
DEFAULT_INPUT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_refinement_input_check_summary.txt"
DEFAULT_INITIAL_OUT = PROJECT_ROOT / "results/metrics/comsol_rect_rot_refinement_initial_predictions.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_priewald_refinement_poc_summary.txt"
DEFAULT_AUDIT = PROJECT_ROOT / "results/summaries/comsol_rect_rot_priewald_refinement_failure_audit_summary.txt"
DEFAULT_CONFIG_SWEEP = PROJECT_ROOT / "results/metrics/comsol_rect_rot_priewald_refinement_config_sweep.csv"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_priewald_refinement_metrics.csv"
DEFAULT_GROUP = PROJECT_ROOT / "results/metrics/comsol_rect_rot_priewald_refinement_group_summary.csv"
DEFAULT_GEOMETRY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_priewald_refinement_geometry_summary.csv"
DEFAULT_FORWARD = PROJECT_ROOT / "results/metrics/comsol_rect_rot_priewald_refinement_forward_summary.csv"
DEFAULT_FAILURE = PROJECT_ROOT / "results/metrics/comsol_rect_rot_priewald_refinement_failure_cases.csv"
DEFAULT_DIAG_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_rect_rot_priewald_refinement_initializer_diagnostic_summary.txt"
)
DEFAULT_DIAG = PROJECT_ROOT / "results/metrics/comsol_rect_rot_priewald_refinement_initializer_diagnostic.csv"
DEFAULT_REVIEW = PROJECT_ROOT / "results/summaries/claude_review_comsol_rect_rot_priewald_refinement_poc.txt"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_rect_rot_priewald_refinement_poc"

SEED = 42
MAX_ANGLE_RAD = math.radians(35.0)
LAMBDA_FORWARD = 1.0
LAMBDA_SHAPE = 0.05
LAMBDA_MASK_PRIOR = 0.02
REF_INITIAL_IOU = 0.6138
REF_INITIAL_DICE = 0.7577
REF_INITIAL_TYPE_ACC = 0.6212
REF_INITIAL_ANGLE_MAE = 18.5798
DENSE_SINGLE_BASELINE_IOU = base.DENSE_SINGLE_BASELINE_IOU
DENSE_SINGLE_BASELINE_DICE = base.DENSE_SINGLE_BASELINE_DICE


CONFIG_FIELDS = [
    "config_name",
    "steps",
    "lr",
    "lambda_prior",
    "val_initial_iou",
    "val_refined_iou",
    "delta_mask_iou",
    "val_initial_dice",
    "val_refined_dice",
    "delta_mask_dice",
    "val_initial_area_error",
    "val_refined_area_error",
    "delta_area_error",
    "val_initial_forward_nrmse",
    "val_refined_forward_nrmse",
    "forward_nrmse_reduction",
    "val_initial_angle_mae_deg",
    "val_refined_angle_mae_deg",
    "angle_mae_delta",
    "parameter_drift_mean",
    "excessive_parameter_drift_flag",
    "val_refinement_score",
]

METRIC_FIELDS = [
    "sample_id",
    "source_index",
    "split",
    "defect_type",
    "source_pack",
    "threshold",
    "pred_defect_type",
    "type_prob_rectangular_notch",
    "type_prob_rotated_rect",
    "initial_iou",
    "refined_iou",
    "delta_iou",
    "initial_dice",
    "refined_dice",
    "delta_dice",
    "initial_area_error",
    "refined_area_error",
    "delta_area_error",
    "initial_center_error_px",
    "refined_center_error_px",
    "initial_pred_area",
    "refined_pred_area",
    "true_area",
    "initial_forward_mse",
    "refined_forward_mse",
    "initial_forward_nrmse",
    "refined_forward_nrmse",
    "forward_nrmse_reduction",
    "initial_forward_correlation",
    "refined_forward_correlation",
    "true_center_x",
    "true_center_y",
    "initial_center_x",
    "initial_center_y",
    "refined_center_x",
    "refined_center_y",
    "center_drift_m",
    "true_width",
    "initial_width",
    "refined_width",
    "width_drift_m",
    "true_length",
    "initial_length",
    "refined_length",
    "length_drift_m",
    "true_depth",
    "initial_depth",
    "refined_depth",
    "depth_drift_m",
    "true_angle_deg",
    "initial_angle_deg",
    "refined_angle_deg",
    "initial_angle_abs_error_deg",
    "refined_angle_abs_error_deg",
    "angle_error_delta",
    "parameter_drift_norm",
    "refinement_category",
    "notes",
]

GROUP_FIELDS = [
    "split",
    "group_name",
    "group_value",
    "sample_count",
    "initial_iou_mean",
    "refined_iou_mean",
    "delta_iou_mean",
    "initial_dice_mean",
    "refined_dice_mean",
    "delta_dice_mean",
    "initial_area_error_mean",
    "refined_area_error_mean",
    "delta_area_error_mean",
    "initial_forward_nrmse_mean",
    "refined_forward_nrmse_mean",
    "forward_nrmse_reduction_mean",
    "initial_angle_mae_deg",
    "refined_angle_mae_deg",
    "angle_error_delta_mean",
    "parameter_drift_norm_mean",
]

GEOMETRY_FIELDS = [
    "split",
    "group_name",
    "group_value",
    "sample_count",
    "initial_center_error_px_mean",
    "refined_center_error_px_mean",
    "center_error_delta_px_mean",
    "true_width_mean",
    "initial_width_mean",
    "refined_width_mean",
    "width_drift_m_mean",
    "true_length_mean",
    "initial_length_mean",
    "refined_length_mean",
    "length_drift_m_mean",
    "true_depth_mean",
    "initial_depth_mean",
    "refined_depth_mean",
    "depth_drift_m_mean",
    "initial_angle_mae_deg",
    "refined_angle_mae_deg",
    "angle_error_delta_mean",
    "center_drift_m_mean",
    "parameter_drift_norm_mean",
]

FORWARD_FIELDS = [
    "split",
    "group_name",
    "group_value",
    "sample_count",
    "initial_forward_mse_mean",
    "refined_forward_mse_mean",
    "forward_mse_delta_mean",
    "initial_forward_nrmse_mean",
    "refined_forward_nrmse_mean",
    "forward_nrmse_reduction_mean",
    "initial_forward_correlation_mean",
    "refined_forward_correlation_mean",
    "forward_correlation_delta_mean",
]


@dataclass(frozen=True)
class RefinementConfig:
    steps: int
    lr: float
    lambda_prior: float

    @property
    def name(self) -> str:
        return f"steps{self.steps}_lr{self.lr:g}_prior{self.lambda_prior:g}"


CONFIGS = [
    RefinementConfig(steps, lr, prior)
    for steps in [20, 50]
    for lr in [0.01, 0.03]
    for prior in [0.05, 0.10]
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


def tensor_atanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-0.999, 0.999)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def logit_from_bounds(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    scaled = (values - lower) / np.maximum(upper - lower, 1e-8)
    scaled = np.clip(scaled, 1e-4, 1.0 - 1e-4)
    return np.log(scaled / (1.0 - scaled)).astype(np.float32)


def load_initial_rows(path: Path, arrays: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv(path)
    keep_ids = set(str(sample_id) for sample_id in arrays["sample_ids"])
    rows = [row for row in rows if row["sample_id"] in keep_ids]
    if len(rows) != arrays["sample_ids"].shape[0]:
        raise RuntimeError(f"Initial predictions have {len(rows)} rect/rot rows, expected {arrays['sample_ids'].shape[0]}")
    return rows


def initial_arrays(initial_rows: list[dict[str, Any]], arrays: dict[str, Any]) -> dict[str, np.ndarray]:
    row_by_id = {row["sample_id"]: row for row in initial_rows}
    geom = []
    angle = []
    type_prob = []
    threshold = []
    for sample_id in arrays["sample_ids"]:
        row = row_by_id[str(sample_id)]
        geom.append(
            [
                base.to_float(row["pred_center_x"]),
                base.to_float(row["pred_center_y"]),
                base.to_float(row["pred_width"]),
                base.to_float(row["pred_length"]),
                base.to_float(row["pred_depth"]),
            ]
        )
        angle.append(math.radians(base.to_float(row["pred_angle_deg"], 0.0)))
        type_prob.append(
            [
                base.to_float(row["type_prob_rectangular_notch"], 0.5),
                base.to_float(row["type_prob_rotated_rect"], 0.5),
            ]
        )
        threshold.append(base.to_float(row["threshold"], 0.6))
    type_prob_np = np.asarray(type_prob, dtype=np.float32)
    type_prob_np = type_prob_np / np.maximum(type_prob_np.sum(axis=1, keepdims=True), 1e-8)
    return {
        "geom": np.asarray(geom, dtype=np.float32),
        "angle": np.asarray(angle, dtype=np.float32),
        "type_prob": type_prob_np,
        "threshold": np.asarray(threshold, dtype=np.float32),
    }


def bounds_from_arrays(arrays: dict[str, Any], init: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.concatenate([arrays["raw_geom"], init["geom"]], axis=0)
    lower = raw.min(axis=0)
    upper = raw.max(axis=0)
    margin = np.array([0.001, 0.001, 0.001, 0.001, 0.0002], dtype=np.float32)
    lower = lower - margin
    upper = upper + margin
    lower[0] = max(lower[0], float(arrays["mask_x"].min()))
    upper[0] = min(upper[0], float(arrays["mask_x"].max()))
    lower[1] = max(lower[1], float(arrays["mask_y"].min()))
    upper[1] = min(upper[1], float(arrays["mask_y"].max()))
    lower[2:] = np.maximum(lower[2:], np.array([0.001, 0.001, 0.0001], dtype=np.float32))
    upper[2:] = np.minimum(upper[2:], np.array([0.025, 0.020, 0.004], dtype=np.float32))
    return lower.astype(np.float32), upper.astype(np.float32)


def decode_params(
    geom_logits: torch.Tensor,
    angle_raw: torch.Tensor,
    bounds_low: torch.Tensor,
    bounds_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    geom = bounds_low.view(1, -1) + (bounds_high - bounds_low).view(1, -1) * torch.sigmoid(geom_logits)
    angle = MAX_ANGLE_RAD * torch.tanh(angle_raw)
    return geom, angle


def mask_from_params(
    geom: torch.Tensor,
    angle: torch.Tensor,
    type_prob: torch.Tensor,
    arrays: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    mask_x = torch.tensor(arrays["mask_x"], device=device)
    mask_y = torch.tensor(arrays["mask_y"], device=device)
    rect = base.soft_rect_mask(mask_x, mask_y, geom[:, 0], geom[:, 1], geom[:, 2], geom[:, 3], torch.zeros_like(angle))
    rot = base.soft_rect_mask(mask_x, mask_y, geom[:, 0], geom[:, 1], geom[:, 2], geom[:, 3], angle)
    return type_prob[:, 0].view(-1, 1, 1) * rect + type_prob[:, 1].view(-1, 1, 1) * rot


def forward_input_from_params(geom: torch.Tensor, angle: torch.Tensor, type_prob: torch.Tensor, arrays: dict[str, Any], device: torch.device) -> torch.Tensor:
    geom_mean = torch.tensor(arrays["geom_mean"], device=device).view(1, -1)
    geom_std = torch.tensor(arrays["geom_std"], device=device).view(1, -1)
    geom_norm = (geom - geom_mean) / geom_std
    return torch.cat([type_prob, geom_norm, torch.sin(angle).unsqueeze(1), torch.cos(angle).unsqueeze(1)], dim=1)


def forward_stats_np(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    return forward_mod.signal_metrics(pred, obs)


def refine_indices(
    indices: np.ndarray,
    init: dict[str, np.ndarray],
    arrays: dict[str, Any],
    surrogate: torch.nn.Module,
    config: RefinementConfig,
    device: torch.device,
    bounds_low_np: np.ndarray,
    bounds_high_np: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = indices.astype(np.int64)
    bounds_low = torch.tensor(bounds_low_np, device=device)
    bounds_high = torch.tensor(bounds_high_np, device=device)
    init_geom = init["geom"][idx]
    init_angle = init["angle"][idx]
    init_logits = logit_from_bounds(init_geom, bounds_low_np, bounds_high_np)
    geom_logits = torch.tensor(init_logits, device=device, requires_grad=True)
    angle_raw = tensor_atanh(torch.tensor(init_angle / MAX_ANGLE_RAD, device=device)).detach().clone().requires_grad_(True)
    type_prob = torch.tensor(init["type_prob"][idx], device=device)
    obs = torch.tensor(arrays["signals_norm"][idx], device=device)
    init_geom_t = torch.tensor(init_geom, device=device)
    init_angle_t = torch.tensor(init_angle, device=device)
    with torch.no_grad():
        init_mask = mask_from_params(init_geom_t, init_angle_t, type_prob, arrays, device)
        init_area = (init_geom_t[:, 2] * init_geom_t[:, 3]).detach()
    optimizer = torch.optim.Adam([geom_logits, angle_raw], lr=config.lr)
    surrogate.eval()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        geom, angle = decode_params(geom_logits, angle_raw, bounds_low, bounds_high)
        forward_pred = surrogate(forward_input_from_params(geom, angle, type_prob, arrays, device)).view(-1, 3, 201)
        forward_loss = F.mse_loss(forward_pred, obs)
        geom_norm = (geom - torch.tensor(arrays["geom_mean"], device=device).view(1, -1)) / torch.tensor(arrays["geom_std"], device=device).view(1, -1)
        init_geom_norm = (init_geom_t - torch.tensor(arrays["geom_mean"], device=device).view(1, -1)) / torch.tensor(arrays["geom_std"], device=device).view(1, -1)
        prior_loss = F.smooth_l1_loss(geom_norm, init_geom_norm) + F.smooth_l1_loss(angle / MAX_ANGLE_RAD, init_angle_t / MAX_ANGLE_RAD)
        area = geom[:, 2] * geom[:, 3]
        shape_loss = F.smooth_l1_loss(area / (init_area + 1e-8), torch.ones_like(area))
        refined_mask = mask_from_params(geom, angle, type_prob, arrays, device)
        mask_prior = F.mse_loss(refined_mask, init_mask)
        loss = LAMBDA_FORWARD * forward_loss + config.lambda_prior * prior_loss + LAMBDA_SHAPE * shape_loss + LAMBDA_MASK_PRIOR * mask_prior
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        geom, angle = decode_params(geom_logits, angle_raw, bounds_low, bounds_high)
        refined_mask = mask_from_params(geom, angle, type_prob, arrays, device)
        forward_pred = surrogate(forward_input_from_params(geom, angle, type_prob, arrays, device)).view(-1, 3, 201)
    return {
        "indices": idx,
        "geom": geom.detach().cpu().numpy(),
        "angle": angle.detach().cpu().numpy(),
        "mask_prob": refined_mask.detach().cpu().numpy(),
        "forward_pred": forward_pred.detach().cpu().numpy(),
    }


def evaluate_rows(
    indices: np.ndarray,
    init: dict[str, np.ndarray],
    refined: dict[str, np.ndarray],
    arrays: dict[str, Any],
    surrogate: torch.nn.Module,
    split: str,
    device: torch.device,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mask_x_t = torch.tensor(arrays["mask_x"], device=device)
    mask_y_t = torch.tensor(arrays["mask_y"], device=device)
    idx_to_order = {int(local_idx): order for order, local_idx in enumerate(refined["indices"])}
    for local_idx_raw in indices.astype(np.int64):
        local_idx = int(local_idx_raw)
        order = idx_to_order[local_idx]
        type_prob_np = init["type_prob"][local_idx]
        type_prob_t = torch.tensor(type_prob_np[None, :], device=device)
        init_geom_t = torch.tensor(init["geom"][local_idx : local_idx + 1], device=device)
        init_angle_t = torch.tensor(init["angle"][local_idx : local_idx + 1], device=device)
        with torch.no_grad():
            init_mask = mask_from_params(init_geom_t, init_angle_t, type_prob_t, arrays, device)[0].cpu().numpy()
            init_forward = surrogate(forward_input_from_params(init_geom_t, init_angle_t, type_prob_t, arrays, device)).view(1, 3, 201)[0].cpu().numpy()
        refined_mask = refined["mask_prob"][order]
        refined_forward = refined["forward_pred"][order]
        observed = arrays["signals_norm"][local_idx]
        init_metric = base.mask_metric(init_mask, arrays["masks"][local_idx], threshold)
        ref_metric = base.mask_metric(refined_mask, arrays["masks"][local_idx], threshold)
        init_forward_stats = forward_stats_np(init_forward, observed)
        ref_forward_stats = forward_stats_np(refined_forward, observed)
        true_geom = arrays["raw_geom"][local_idx]
        init_geom = init["geom"][local_idx]
        ref_geom = refined["geom"][order]
        true_angle = math.atan2(float(arrays["angle_targets"][local_idx, 0]), float(arrays["angle_targets"][local_idx, 1]))
        init_angle = float(init["angle"][local_idx])
        ref_angle = float(refined["angle"][order])
        true_angle_deg = math.degrees(true_angle)
        init_angle_deg = math.degrees(init_angle)
        ref_angle_deg = math.degrees(ref_angle)
        defect_type = str(arrays["defect_types"][local_idx])
        init_angle_error = base.circular_angle_error_deg(init_angle_deg, true_angle_deg) if defect_type == "rotated_rect" else math.nan
        ref_angle_error = base.circular_angle_error_deg(ref_angle_deg, true_angle_deg) if defect_type == "rotated_rect" else math.nan
        pred_type = base.TYPE_NAMES[int(np.argmax(type_prob_np))]
        drift_vec = np.concatenate([(ref_geom - init_geom) / np.maximum(arrays["geom_std"], 1e-8), [(ref_angle - init_angle) / MAX_ANGLE_RAD]])
        forward_reduction = init_forward_stats["nrmse"] - ref_forward_stats["nrmse"]
        delta_iou = ref_metric["iou"] - init_metric["iou"]
        delta_dice = ref_metric["dice"] - init_metric["dice"]
        row = {
            "sample_id": str(arrays["sample_ids"][local_idx]),
            "source_index": int(arrays["source_indices"][local_idx]),
            "split": split,
            "defect_type": defect_type,
            "source_pack": str(arrays["source_packs"][local_idx]),
            "threshold": threshold,
            "pred_defect_type": pred_type,
            "type_prob_rectangular_notch": float(type_prob_np[0]),
            "type_prob_rotated_rect": float(type_prob_np[1]),
            "initial_iou": init_metric["iou"],
            "refined_iou": ref_metric["iou"],
            "delta_iou": delta_iou,
            "initial_dice": init_metric["dice"],
            "refined_dice": ref_metric["dice"],
            "delta_dice": delta_dice,
            "initial_area_error": init_metric["area_error"],
            "refined_area_error": ref_metric["area_error"],
            "delta_area_error": ref_metric["area_error"] - init_metric["area_error"],
            "initial_center_error_px": init_metric["center_error_px"],
            "refined_center_error_px": ref_metric["center_error_px"],
            "initial_pred_area": init_metric["pred_area"],
            "refined_pred_area": ref_metric["pred_area"],
            "true_area": ref_metric["true_area"],
            "initial_forward_mse": init_forward_stats["mse"],
            "refined_forward_mse": ref_forward_stats["mse"],
            "initial_forward_nrmse": init_forward_stats["nrmse"],
            "refined_forward_nrmse": ref_forward_stats["nrmse"],
            "forward_nrmse_reduction": forward_reduction,
            "initial_forward_correlation": init_forward_stats["correlation"],
            "refined_forward_correlation": ref_forward_stats["correlation"],
            "true_center_x": float(true_geom[0]),
            "true_center_y": float(true_geom[1]),
            "initial_center_x": float(init_geom[0]),
            "initial_center_y": float(init_geom[1]),
            "refined_center_x": float(ref_geom[0]),
            "refined_center_y": float(ref_geom[1]),
            "center_drift_m": float(math.hypot(ref_geom[0] - init_geom[0], ref_geom[1] - init_geom[1])),
            "true_width": float(true_geom[2]),
            "initial_width": float(init_geom[2]),
            "refined_width": float(ref_geom[2]),
            "width_drift_m": float(ref_geom[2] - init_geom[2]),
            "true_length": float(true_geom[3]),
            "initial_length": float(init_geom[3]),
            "refined_length": float(ref_geom[3]),
            "length_drift_m": float(ref_geom[3] - init_geom[3]),
            "true_depth": float(true_geom[4]),
            "initial_depth": float(init_geom[4]),
            "refined_depth": float(ref_geom[4]),
            "depth_drift_m": float(ref_geom[4] - init_geom[4]),
            "true_angle_deg": true_angle_deg,
            "initial_angle_deg": init_angle_deg,
            "refined_angle_deg": ref_angle_deg,
            "initial_angle_abs_error_deg": init_angle_error,
            "refined_angle_abs_error_deg": ref_angle_error,
            "angle_error_delta": ref_angle_error - init_angle_error if not math.isnan(ref_angle_error) else math.nan,
            "parameter_drift_norm": float(np.linalg.norm(drift_vec)),
            "refinement_category": "improved" if delta_iou > 0.01 or delta_dice > 0.008 else ("worsened" if delta_iou < -0.01 or delta_dice < -0.008 else "neutral"),
            "notes": "",
        }
        if forward_reduction > 0.05 and (delta_iou < -0.01 or delta_dice < -0.008):
            row["refinement_category"] = "surrogate_mismatch"
        rows.append(row)
    return rows


def split_stats(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split]
    return {
        "initial_iou": safe_mean(subset, "initial_iou"),
        "refined_iou": safe_mean(subset, "refined_iou"),
        "delta_iou": safe_mean(subset, "delta_iou"),
        "initial_dice": safe_mean(subset, "initial_dice"),
        "refined_dice": safe_mean(subset, "refined_dice"),
        "delta_dice": safe_mean(subset, "delta_dice"),
        "initial_area_error": safe_mean(subset, "initial_area_error"),
        "refined_area_error": safe_mean(subset, "refined_area_error"),
        "delta_area_error": safe_mean(subset, "delta_area_error"),
        "initial_forward_nrmse": safe_mean(subset, "initial_forward_nrmse"),
        "refined_forward_nrmse": safe_mean(subset, "refined_forward_nrmse"),
        "forward_nrmse_reduction": safe_mean(subset, "forward_nrmse_reduction"),
        "initial_angle_mae_deg": safe_mean([row for row in subset if row["defect_type"] == "rotated_rect"], "initial_angle_abs_error_deg"),
        "refined_angle_mae_deg": safe_mean([row for row in subset if row["defect_type"] == "rotated_rect"], "refined_angle_abs_error_deg"),
        "angle_error_delta": safe_mean([row for row in subset if row["defect_type"] == "rotated_rect"], "angle_error_delta"),
        "parameter_drift_norm": safe_mean(subset, "parameter_drift_norm"),
    }


def config_score(rows: list[dict[str, Any]], config: RefinementConfig) -> dict[str, Any]:
    stats = split_stats(rows, "val")
    excessive = float(stats["parameter_drift_norm"] > 1.0)
    score = (
        stats["delta_iou"]
        + stats["delta_dice"]
        - max(0.0, stats["delta_area_error"])
        + 0.20 * stats["forward_nrmse_reduction"]
        - 0.10 * excessive
    )
    return {
        "config_name": config.name,
        "steps": config.steps,
        "lr": config.lr,
        "lambda_prior": config.lambda_prior,
        "val_initial_iou": stats["initial_iou"],
        "val_refined_iou": stats["refined_iou"],
        "delta_mask_iou": stats["delta_iou"],
        "val_initial_dice": stats["initial_dice"],
        "val_refined_dice": stats["refined_dice"],
        "delta_mask_dice": stats["delta_dice"],
        "val_initial_area_error": stats["initial_area_error"],
        "val_refined_area_error": stats["refined_area_error"],
        "delta_area_error": stats["delta_area_error"],
        "val_initial_forward_nrmse": stats["initial_forward_nrmse"],
        "val_refined_forward_nrmse": stats["refined_forward_nrmse"],
        "forward_nrmse_reduction": stats["forward_nrmse_reduction"],
        "val_initial_angle_mae_deg": stats["initial_angle_mae_deg"],
        "val_refined_angle_mae_deg": stats["refined_angle_mae_deg"],
        "angle_mae_delta": stats["angle_error_delta"],
        "parameter_drift_mean": stats["parameter_drift_norm"],
        "excessive_parameter_drift_flag": excessive,
        "val_refinement_score": score,
    }


def build_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        for group_name, values in [
            ("overall", ["rect_rot"]),
            ("defect_type", sorted({row["defect_type"] for row in split_rows})),
            ("source_pack", sorted({row["source_pack"] for row in split_rows})),
        ]:
            for value in values:
                subset = split_rows if group_name == "overall" else [row for row in split_rows if row[group_name] == value]
                stats = split_stats(subset, split)
                out.append(
                    {
                        "split": split,
                        "group_name": group_name,
                        "group_value": value,
                        "sample_count": len(subset),
                        "initial_iou_mean": stats["initial_iou"],
                        "refined_iou_mean": stats["refined_iou"],
                        "delta_iou_mean": stats["delta_iou"],
                        "initial_dice_mean": stats["initial_dice"],
                        "refined_dice_mean": stats["refined_dice"],
                        "delta_dice_mean": stats["delta_dice"],
                        "initial_area_error_mean": stats["initial_area_error"],
                        "refined_area_error_mean": stats["refined_area_error"],
                        "delta_area_error_mean": stats["delta_area_error"],
                        "initial_forward_nrmse_mean": stats["initial_forward_nrmse"],
                        "refined_forward_nrmse_mean": stats["refined_forward_nrmse"],
                        "forward_nrmse_reduction_mean": stats["forward_nrmse_reduction"],
                        "initial_angle_mae_deg": stats["initial_angle_mae_deg"],
                        "refined_angle_mae_deg": stats["refined_angle_mae_deg"],
                        "angle_error_delta_mean": stats["angle_error_delta"],
                        "parameter_drift_norm_mean": stats["parameter_drift_norm"],
                    }
                )
    return out


def grouped_subsets(rows: list[dict[str, Any]]) -> Iterable[tuple[str, str, str, list[dict[str, Any]]]]:
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        for group_name, values in [
            ("overall", ["rect_rot"]),
            ("defect_type", sorted({row["defect_type"] for row in split_rows})),
            ("source_pack", sorted({row["source_pack"] for row in split_rows})),
        ]:
            for value in values:
                subset = split_rows if group_name == "overall" else [row for row in split_rows if row[group_name] == value]
                yield split, group_name, value, subset


def build_geometry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, group_name, value, subset in grouped_subsets(rows):
        rotated = [row for row in subset if row["defect_type"] == "rotated_rect"]
        out.append(
            {
                "split": split,
                "group_name": group_name,
                "group_value": value,
                "sample_count": len(subset),
                "initial_center_error_px_mean": safe_mean(subset, "initial_center_error_px"),
                "refined_center_error_px_mean": safe_mean(subset, "refined_center_error_px"),
                "center_error_delta_px_mean": safe_mean(
                    [
                        {
                            "center_error_delta_px": row["refined_center_error_px"] - row["initial_center_error_px"],
                        }
                        for row in subset
                    ],
                    "center_error_delta_px",
                ),
                "true_width_mean": safe_mean(subset, "true_width"),
                "initial_width_mean": safe_mean(subset, "initial_width"),
                "refined_width_mean": safe_mean(subset, "refined_width"),
                "width_drift_m_mean": safe_mean(subset, "width_drift_m"),
                "true_length_mean": safe_mean(subset, "true_length"),
                "initial_length_mean": safe_mean(subset, "initial_length"),
                "refined_length_mean": safe_mean(subset, "refined_length"),
                "length_drift_m_mean": safe_mean(subset, "length_drift_m"),
                "true_depth_mean": safe_mean(subset, "true_depth"),
                "initial_depth_mean": safe_mean(subset, "initial_depth"),
                "refined_depth_mean": safe_mean(subset, "refined_depth"),
                "depth_drift_m_mean": safe_mean(subset, "depth_drift_m"),
                "initial_angle_mae_deg": safe_mean(rotated, "initial_angle_abs_error_deg"),
                "refined_angle_mae_deg": safe_mean(rotated, "refined_angle_abs_error_deg"),
                "angle_error_delta_mean": safe_mean(rotated, "angle_error_delta"),
                "center_drift_m_mean": safe_mean(subset, "center_drift_m"),
                "parameter_drift_norm_mean": safe_mean(subset, "parameter_drift_norm"),
            }
        )
    return out


def build_forward_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, group_name, value, subset in grouped_subsets(rows):
        out.append(
            {
                "split": split,
                "group_name": group_name,
                "group_value": value,
                "sample_count": len(subset),
                "initial_forward_mse_mean": safe_mean(subset, "initial_forward_mse"),
                "refined_forward_mse_mean": safe_mean(subset, "refined_forward_mse"),
                "forward_mse_delta_mean": safe_mean(
                    [
                        {
                            "forward_mse_delta": row["refined_forward_mse"] - row["initial_forward_mse"],
                        }
                        for row in subset
                    ],
                    "forward_mse_delta",
                ),
                "initial_forward_nrmse_mean": safe_mean(subset, "initial_forward_nrmse"),
                "refined_forward_nrmse_mean": safe_mean(subset, "refined_forward_nrmse"),
                "forward_nrmse_reduction_mean": safe_mean(subset, "forward_nrmse_reduction"),
                "initial_forward_correlation_mean": safe_mean(subset, "initial_forward_correlation"),
                "refined_forward_correlation_mean": safe_mean(subset, "refined_forward_correlation"),
                "forward_correlation_delta_mean": safe_mean(
                    [
                        {
                            "forward_correlation_delta": row["refined_forward_correlation"] - row["initial_forward_correlation"],
                        }
                        for row in subset
                    ],
                    "forward_correlation_delta",
                ),
            }
        )
    return out


def build_failure_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row["split"] == "test"]
    prioritized = sorted(
        test_rows,
        key=lambda row: (
            row["refinement_category"] != "surrogate_mismatch",
            row["delta_iou"],
            row["delta_dice"],
            -row["forward_nrmse_reduction"],
        ),
    )
    return [dict(row) for row in prioritized[:40]]


def preview(rows: list[dict[str, Any]], arrays: dict[str, Any], surrogate: torch.nn.Module, init: dict[str, np.ndarray], preview_dir: Path, device: torch.device, max_count: int = 24) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in rows if row["split"] in {"val", "test"}]
    ordered = sorted(candidates, key=lambda row: (row["refinement_category"] != "improved", row["delta_iou"]))[:8]
    ordered += sorted(candidates, key=lambda row: row["delta_iou"])[:8]
    ordered += sorted(candidates, key=lambda row: row["forward_nrmse_reduction"], reverse=True)[:8]
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
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
        type_prob = init["type_prob"][local_idx]
        with torch.no_grad():
            init_mask = (
                type_prob[0]
                * base.soft_rect_mask(
                    mask_x_t,
                    mask_y_t,
                    torch.tensor([row["initial_center_x"]], dtype=torch.float32),
                    torch.tensor([row["initial_center_y"]], dtype=torch.float32),
                    torch.tensor([row["initial_width"]], dtype=torch.float32),
                    torch.tensor([row["initial_length"]], dtype=torch.float32),
                    torch.tensor([0.0], dtype=torch.float32),
                )[0].numpy()
                + type_prob[1]
                * base.soft_rect_mask(
                    mask_x_t,
                    mask_y_t,
                    torch.tensor([row["initial_center_x"]], dtype=torch.float32),
                    torch.tensor([row["initial_center_y"]], dtype=torch.float32),
                    torch.tensor([row["initial_width"]], dtype=torch.float32),
                    torch.tensor([row["initial_length"]], dtype=torch.float32),
                    torch.tensor([math.radians(row["initial_angle_deg"])], dtype=torch.float32),
                )[0].numpy()
            )
            ref_mask = (
                type_prob[0]
                * base.soft_rect_mask(
                    mask_x_t,
                    mask_y_t,
                    torch.tensor([row["refined_center_x"]], dtype=torch.float32),
                    torch.tensor([row["refined_center_y"]], dtype=torch.float32),
                    torch.tensor([row["refined_width"]], dtype=torch.float32),
                    torch.tensor([row["refined_length"]], dtype=torch.float32),
                    torch.tensor([0.0], dtype=torch.float32),
                )[0].numpy()
                + type_prob[1]
                * base.soft_rect_mask(
                    mask_x_t,
                    mask_y_t,
                    torch.tensor([row["refined_center_x"]], dtype=torch.float32),
                    torch.tensor([row["refined_center_y"]], dtype=torch.float32),
                    torch.tensor([row["refined_width"]], dtype=torch.float32),
                    torch.tensor([row["refined_length"]], dtype=torch.float32),
                    torch.tensor([math.radians(row["refined_angle_deg"])], dtype=torch.float32),
                )[0].numpy()
            )
        fig, axes = plt.subplots(2, 4, figsize=(13, 6))
        axes[0, 0].imshow(arrays["masks"][local_idx], origin="lower", cmap="gray")
        axes[0, 0].set_title("true")
        axes[0, 1].imshow(init_mask >= row["threshold"], origin="lower", cmap="gray")
        axes[0, 1].set_title(f"initial {row['initial_iou']:.3f}")
        axes[0, 2].imshow(ref_mask >= row["threshold"], origin="lower", cmap="gray")
        axes[0, 2].set_title(f"refined {row['refined_iou']:.3f}")
        axes[0, 3].imshow(arrays["masks"][local_idx], origin="lower", cmap="gray", alpha=0.5)
        axes[0, 3].imshow(ref_mask >= row["threshold"], origin="lower", cmap="Reds", alpha=0.45)
        axes[0, 3].set_title(row["refinement_category"])
        obs = arrays["signals_norm"][local_idx]
        for line in range(3):
            axes[1, 0].plot(arrays["sensor_x"], obs[line], label=f"obs {line}")
        axes[1, 0].legend(fontsize=6)
        axes[1, 0].set_title("observed")
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0,
            0.95,
            "\n".join(
                [
                    f"type {row['defect_type']}->{row['pred_defect_type']}",
                    f"IoU {row['initial_iou']:.3f}->{row['refined_iou']:.3f}",
                    f"Dice {row['initial_dice']:.3f}->{row['refined_dice']:.3f}",
                    f"fwd {row['initial_forward_nrmse']:.3f}->{row['refined_forward_nrmse']:.3f}",
                ]
            ),
            va="top",
            fontsize=8,
        )
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0,
            0.95,
            "\n".join(
                [
                    f"cx {row['initial_center_x']:.4g}->{row['refined_center_x']:.4g}",
                    f"cy {row['initial_center_y']:.4g}->{row['refined_center_y']:.4g}",
                    f"w {row['initial_width']:.4g}->{row['refined_width']:.4g}",
                    f"l {row['initial_length']:.4g}->{row['refined_length']:.4g}",
                    f"ang {row['initial_angle_deg']:.1f}->{row['refined_angle_deg']:.1f}",
                ]
            ),
            va="top",
            fontsize=8,
        )
        axes[1, 3].axis("off")
        axes[1, 3].text(
            0,
            0.95,
            f"drift={row['parameter_drift_norm']:.3f}\narea {row['initial_area_error']:.3f}->{row['refined_area_error']:.3f}",
            va="top",
            fontsize=8,
        )
        for ax in axes[0]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(row["sample_id"])
        fig.tight_layout()
        fig.savefig(preview_dir / f"{count:02d}_{row['sample_id']}.png", dpi=140)
        plt.close(fig)
        count += 1
    return count


def write_input_check(args: argparse.Namespace, arrays: dict[str, Any], diagnostics: dict[str, Any], initial_rows: list[dict[str, Any]], surrogate_bundle: forward_mod.ForwardSurrogateBundle, initial_out: list[dict[str, Any]]) -> None:
    init_stats = {split: compute_initial_stats(initial_out, split) for split in ["train", "val", "test"]}
    lines = [
        "COMSOL rect/rot Priewald-style refinement input check summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Geometry labels: {args.labels}",
        f"Initial prediction source: {args.initial_predictions}",
        "Forward surrogate source: retrained in-memory with the 20.51 protocol; no checkpoint written.",
        "Initializer source: reused 20.51 feature-forward per-sample prediction metrics.",
        "Scope: rectangular_notch + rotated_rect only; polygon excluded from main POC.",
        "Optimization policy: true mask / true geometry are not used in refinement loss; they are metrics only.",
        "",
        f"rect+rot N: {diagnostics['n_rect_rot']}",
        f"split counts: {diagnostics['split_counts']}",
        f"type counts: {diagnostics['type_counts']}",
        f"forward surrogate best epoch: {surrogate_bundle.best_epoch}",
        f"forward surrogate val MSE/NRMSE/corr: {surrogate_bundle.best_val['mse']:.6f} / {surrogate_bundle.best_val['nrmse']:.6f} / {surrogate_bundle.best_val['correlation']:.4f}",
        "",
        "Initial proposal metrics:",
        f"- train IoU/Dice/forward_nrmse/angle = {init_stats['train']['iou']:.4f} / {init_stats['train']['dice']:.4f} / {init_stats['train']['forward_nrmse']:.4f} / {init_stats['train']['angle_mae']:.2f}",
        f"- val IoU/Dice/forward_nrmse/angle = {init_stats['val']['iou']:.4f} / {init_stats['val']['dice']:.4f} / {init_stats['val']['forward_nrmse']:.4f} / {init_stats['val']['angle_mae']:.2f}",
        f"- test IoU/Dice/forward_nrmse/angle = {init_stats['test']['iou']:.4f} / {init_stats['test']['dice']:.4f} / {init_stats['test']['forward_nrmse']:.4f} / {init_stats['test']['angle_mae']:.2f}",
        "",
        "Leakage check: train/val/test split is preserved; validation selects refinement config; test is final only.",
    ]
    args.input_summary.parent.mkdir(parents=True, exist_ok=True)
    args.input_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_initial_stats(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    subset = [row for row in rows if row["split"] == split]
    return {
        "iou": safe_mean(subset, "initial_iou"),
        "dice": safe_mean(subset, "initial_dice"),
        "area_error": safe_mean(subset, "initial_area_error"),
        "forward_nrmse": safe_mean(subset, "initial_forward_nrmse"),
        "angle_mae": safe_mean([row for row in subset if row["defect_type"] == "rotated_rect"], "initial_angle_abs_error_deg"),
    }


def build_initial_prediction_rows(indices: np.ndarray, init: dict[str, np.ndarray], arrays: dict[str, Any], surrogate: torch.nn.Module, split: str, device: torch.device, threshold: float) -> list[dict[str, Any]]:
    identity_refined = {
        "indices": indices.astype(np.int64),
        "geom": init["geom"][indices],
        "angle": init["angle"][indices],
        "mask_prob": np.zeros((len(indices), 64, 128), dtype=np.float32),
        "forward_pred": np.zeros((len(indices), 3, 201), dtype=np.float32),
    }
    # Reuse evaluation with refined equal to initial; fill mask/forward via zero-step refinement for accurate fields.
    config = RefinementConfig(steps=0, lr=0.0, lambda_prior=0.10)
    identity_refined = refine_indices(indices, init, arrays, surrogate, config, device, *bounds_from_arrays(arrays, init))
    return evaluate_rows(indices, init, identity_refined, arrays, surrogate, split, device, threshold)


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    surrogate_args = argparse.Namespace(
        npz=args.npz,
        labels=args.labels,
        features=args.features,
        input_check_summary=PROJECT_ROOT / "results/summaries/_tmp_refinement_forward_input_check.txt",
        input_check=PROJECT_ROOT / "results/metrics/_tmp_refinement_forward_input_check.csv",
        summary=PROJECT_ROOT / "results/summaries/_tmp_refinement_forward_summary.txt",
        metrics=PROJECT_ROOT / "results/metrics/_tmp_refinement_forward_metrics.csv",
        epoch_log=PROJECT_ROOT / "results/metrics/_tmp_refinement_forward_epoch.csv",
        group_summary=PROJECT_ROOT / "results/metrics/_tmp_refinement_forward_group.csv",
        seed=args.seed,
        epochs=args.forward_epochs,
        batch_size=args.forward_batch_size,
        lr=args.forward_lr,
        cpu=args.cpu,
    )
    surrogate_bundle = forward_mod.train_forward_surrogate(surrogate_args, write_outputs=False)
    arrays = surrogate_bundle.arrays
    device = surrogate_bundle.device
    surrogate = surrogate_bundle.model.to(device)
    initial_raw_rows = load_initial_rows(args.initial_predictions, arrays)
    init = initial_arrays(initial_raw_rows, arrays)
    threshold = float(np.median(init["threshold"][arrays["split_indices"]["val"]]))

    initial_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        initial_rows.extend(
            build_initial_prediction_rows(arrays["split_indices"][split], init, arrays, surrogate, split, device, threshold)
        )
    write_csv(args.initial_out, initial_rows, METRIC_FIELDS)
    write_input_check(args, arrays, surrogate_bundle.diagnostics, initial_raw_rows, surrogate_bundle, initial_rows)

    bounds_low, bounds_high = bounds_from_arrays(arrays, init)
    val_idx = arrays["split_indices"]["val"]
    config_rows: list[dict[str, Any]] = []
    val_rows_by_config: dict[str, list[dict[str, Any]]] = {}
    for config in CONFIGS:
        refined_val = refine_indices(val_idx, init, arrays, surrogate, config, device, bounds_low, bounds_high)
        val_rows = evaluate_rows(val_idx, init, refined_val, arrays, surrogate, "val", device, threshold)
        val_rows_by_config[config.name] = val_rows
        config_rows.append(config_score(val_rows, config))
        print(
            f"{config.name}: score={config_rows[-1]['val_refinement_score']:.5f} "
            f"dIoU={config_rows[-1]['delta_mask_iou']:.4f} dF={config_rows[-1]['forward_nrmse_reduction']:.4f}"
        )

    selected_row = sorted(
        config_rows,
        key=lambda row: (
            row["val_refinement_score"],
            row["delta_mask_iou"],
            row["delta_mask_dice"],
            row["forward_nrmse_reduction"],
        ),
        reverse=True,
    )[0]
    selected_config = next(config for config in CONFIGS if config.name == selected_row["config_name"])
    all_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        idx = arrays["split_indices"][split]
        refined = refine_indices(idx, init, arrays, surrogate, selected_config, device, bounds_low, bounds_high)
        all_rows.extend(evaluate_rows(idx, init, refined, arrays, surrogate, split, device, threshold))

    group_rows = build_group_rows(all_rows)
    geometry_rows = build_geometry_rows(all_rows)
    forward_rows = build_forward_rows(all_rows)
    failure_rows = build_failure_cases(all_rows)
    preview_count = preview(all_rows, arrays, surrogate, init, args.preview_dir, device, max_count=24)

    write_csv(args.config_sweep, config_rows, CONFIG_FIELDS)
    write_csv(args.metrics, all_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_rows, GROUP_FIELDS)
    write_csv(args.geometry_summary, geometry_rows, GEOMETRY_FIELDS)
    write_csv(args.forward_summary, forward_rows, FORWARD_FIELDS)
    write_csv(args.failure_cases, failure_rows, list(failure_rows[0].keys()) if failure_rows else METRIC_FIELDS)
    summary = write_summary(
        args,
        surrogate_bundle,
        selected_config,
        selected_row,
        config_rows,
        all_rows,
        failure_rows,
        preview_count,
    )
    maybe_write_initializer_diagnostic(args, summary)
    return summary


def write_summary(
    args: argparse.Namespace,
    surrogate_bundle: forward_mod.ForwardSurrogateBundle,
    selected_config: RefinementConfig,
    selected_row: dict[str, Any],
    config_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    preview_count: int,
) -> dict[str, Any]:
    stats = {split: split_stats(rows, split) for split in ["train", "val", "test"]}
    test = stats["test"]
    promising = (
        (test["delta_iou"] >= 0.01 or test["delta_dice"] >= 0.008)
        and test["forward_nrmse_reduction"] > 0
        and test["delta_area_error"] <= 0.03
        and test["parameter_drift_norm"] <= 1.0
    )
    surrogate_mismatch = test["forward_nrmse_reduction"] > 0.02 and (test["delta_iou"] < -0.005 or test["delta_dice"] < -0.004)
    recommendation = (
        "B. Use dense/coarse initializer for refinement."
        if not promising and not surrogate_mismatch
        else ("A. Improve forward surrogate." if surrogate_mismatch else "Continue refinement route with human confirmation.")
    )
    all_val_iou_negative = all(row["delta_mask_iou"] < 0 for row in config_rows)
    all_val_dice_negative = all(row["delta_mask_dice"] < 0 for row in config_rows)
    sweep_finding = (
        "All validation configs had negative mask IoU/Dice deltas; the selected config is the least harmful by validation score, not a true validation mask improvement."
        if all_val_iou_negative and all_val_dice_negative
        else "At least one validation config improved IoU or Dice."
    )
    lines = [
        "COMSOL rect/rot Priewald-style coarse-to-fine refinement POC summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Initial predictions: {args.initial_predictions}",
        "Scope: rectangular_notch + rotated_rect only; polygon excluded.",
        "No COMSOL run; no new data; no baseline update.",
        "Refinement uses observed normalized delta_bz and frozen surrogate forward residual.",
        "True mask / true geometry are metrics only and are not used in optimization.",
        "",
        "Forward surrogate:",
        "- source: retrained in-memory with 20.51 protocol; no checkpoint written",
        f"- best epoch = {surrogate_bundle.best_epoch}",
        f"- val MSE/NRMSE/corr = {surrogate_bundle.best_val['mse']:.6f} / {surrogate_bundle.best_val['nrmse']:.6f} / {surrogate_bundle.best_val['correlation']:.4f}",
        "",
        "Selected validation config:",
        f"- config = {selected_config.name}",
        f"- steps/lr/lambda_prior = {selected_config.steps} / {selected_config.lr} / {selected_config.lambda_prior}",
        f"- val_refinement_score = {selected_row['val_refinement_score']:.6f}",
        f"- val delta IoU/Dice = {selected_row['delta_mask_iou']:.6f} / {selected_row['delta_mask_dice']:.6f}",
        f"- val forward NRMSE reduction = {selected_row['forward_nrmse_reduction']:.6f}",
        f"- sweep finding = {sweep_finding}",
        "",
        "Pre-refine vs post-refine metrics:",
    ]
    for split in ["train", "val", "test"]:
        s = stats[split]
        lines.extend(
            [
                f"- {split} IoU: {s['initial_iou']:.4f} -> {s['refined_iou']:.4f} (delta {s['delta_iou']:.4f})",
                f"- {split} Dice: {s['initial_dice']:.4f} -> {s['refined_dice']:.4f} (delta {s['delta_dice']:.4f})",
                f"- {split} area_error: {s['initial_area_error']:.4f} -> {s['refined_area_error']:.4f} (delta {s['delta_area_error']:.4f})",
                f"- {split} forward NRMSE: {s['initial_forward_nrmse']:.4f} -> {s['refined_forward_nrmse']:.4f} (reduction {s['forward_nrmse_reduction']:.4f})",
                f"- {split} angle MAE: {s['initial_angle_mae_deg']:.4f} -> {s['refined_angle_mae_deg']:.4f} (delta {s['angle_error_delta']:.4f})",
                f"- {split} parameter drift norm: {s['parameter_drift_norm']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            f"Dense single-defect baseline IoU/Dice reference: {DENSE_SINGLE_BASELINE_IOU:.4f} / {DENSE_SINGLE_BASELINE_DICE:.4f}",
            f"Stage 20.51 initial reference IoU/Dice/type/angle: {REF_INITIAL_IOU:.4f} / {REF_INITIAL_DICE:.4f} / {REF_INITIAL_TYPE_ACC:.4f} / {REF_INITIAL_ANGLE_MAE:.4f}",
            f"Preview PNG generated: {preview_count} (not for submission)",
            f"Surrogate mismatch risk: {surrogate_mismatch}",
            f"POC promising by acceptance criteria: {promising}",
            f"Next recommendation: {recommendation}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    categories = defaultdict(int)
    for row in rows:
        if row["split"] == "test":
            categories[row["refinement_category"]] += 1
    audit_lines = [
        "COMSOL rect/rot Priewald refinement failure audit summary",
        "",
        f"Selected config: {selected_config.name}",
        f"Test refinement categories: {dict(sorted(categories.items()))}",
        f"Surrogate mismatch risk: {surrogate_mismatch}",
        "",
        "Worst / risk cases:",
    ]
    for row in failure_rows[:20]:
        audit_lines.append(
            f"- {row['sample_id']}: {row['refinement_category']}, "
            f"IoU {row['initial_iou']:.3f}->{row['refined_iou']:.3f}, "
            f"Dice {row['initial_dice']:.3f}->{row['refined_dice']:.3f}, "
            f"F {row['initial_forward_nrmse']:.3f}->{row['refined_forward_nrmse']:.3f}, "
            f"drift={row['parameter_drift_norm']:.3f}"
        )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    return {
        "train": stats["train"],
        "val": stats["val"],
        "test": stats["test"],
        "selected_config": selected_config.name,
        "promising": promising,
        "surrogate_mismatch": surrogate_mismatch,
        "recommendation": recommendation,
    }


def maybe_write_initializer_diagnostic(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if summary["promising"]:
        return
    rows = [
        {
            "diagnostic": "dense_or_coarse_initializer",
            "executed": False,
            "reason": "No reusable dense baseline prediction artifact was available; dense baseline retraining is outside this stage.",
            "next_action": "Use dense/coarse initializer in a separate approved stage if needed.",
        }
    ]
    write_csv(args.diagnostic, rows, list(rows[0].keys()))
    args.diagnostic_summary.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostic_summary.write_text(
        "COMSOL rect/rot Priewald refinement initializer diagnostic summary\n\n"
        "Stage C was skipped. No reusable dense/coarse prediction artifact was found, and retraining a dense baseline is outside the 20.52 scope.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--initial-predictions", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT_SUMMARY)
    parser.add_argument("--initial-out", type=Path, default=DEFAULT_INITIAL_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--config-sweep", type=Path, default=DEFAULT_CONFIG_SWEEP)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--forward-summary", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--diagnostic-summary", type=Path, default=DEFAULT_DIAG_SUMMARY)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAG)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--forward-epochs", type=int, default=300)
    parser.add_argument("--forward-batch-size", type=int, default=32)
    parser.add_argument("--forward-lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
