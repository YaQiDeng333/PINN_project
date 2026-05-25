#!/usr/bin/env python
"""Seed-42 R1 profile-primary candidate screen for true-3D RBC v3_240.

The model consumes only delta_b. RBC params, projected masks, and profile
depth grids are labels used for supervision, validation selection, and metrics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from load_true_3d_rbc_pilot_dataset import (
    V3_240_DATASET_ID,
    PARAM_NAMES,
    ROOT,
    check_no_overwrite,
    denormalize_y,
    depth_grid_from_params,
    depth_map_from_params,
    evaluate_param_predictions,
    load_dataset,
    mask_metrics,
    normalize_x,
    normalize_y,
    split_indices,
    train_normalization,
    write_csv,
)
from true_3d_rbc_profile_generator import (
    clip_params_torch,
    depth_grid_torch,
    er_like_profile_error,
    max_depth_error,
    profile_depth_rmse,
    soft_bound_penalty,
    soft_projected_mask_torch,
)


SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_profile_primary_candidate_screen_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_screen_metrics.csv"
GROUP = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_group_summary.csv"
PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_profile_metrics.csv"
REFERENCE_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_profile_metrics.csv"

REF_NEURAL = {
    "label": "20.77_neural",
    "total": 0.6780143536818333,
    "L_mm": 1.8918915996566796,
    "W_mm": 2.1857599088778863,
    "D_mm": 0.8002313476246901,
    "curvature": 0.20107580616306037,
    "wLD": 0.2094394713640213,
    "wWD": 0.20446909964084625,
    "wLW": 0.18931882083415985,
    "iou": 0.7506502455785019,
    "dice": 0.8477271366767738,
    "profile_rmse": 0.0003877372636895579,
}
REF_FEATURE = {
    "label": "20.80_feature_only",
    "total": 0.6957241464883853,
    "L_mm": 2.5945624946019588,
    "W_mm": 2.3609011820875683,
    "D_mm": 0.9663975906486695,
    "curvature": 0.1903038014872716,
    "wLD": 0.2096490095823239,
    "wWD": 0.1947971781094869,
    "wLW": 0.1664652182505681,
    "iou": 0.7145338928326161,
    "dice": 0.8262715758785393,
    "profile_rmse": 0.0004496398864713951,
}
REF_FUSION = {
    "label": "20.81_feature_fusion",
    "total": 0.6678883980,
    "L_mm": 2.030387,
    "W_mm": 1.806630,
    "D_mm": 0.956618,
    "curvature": 0.1944833390,
    "wLD": 0.217079,
    "wWD": 0.202304,
    "wLW": 0.164068,
    "iou": 0.774541,
    "dice": 0.866573,
    "profile_rmse": 0.000445297332,
}

PROFILE_FIELDS = [
    "variant",
    "seed",
    "selected_by_validation",
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "size_bin",
    "clip_applied",
    "clip_fraction",
    "normalized_param_mae_mean",
    "dimension_param_mae_norm",
    "curvature_param_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "curvature_mae_mean",
    "projected_mask_iou",
    "projected_mask_dice",
    "projected_mask_area_error",
    "projected_mask_center_error_px",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "max_depth_error_m",
    "volume_proxy_rel_error",
    "true_L_m",
    "true_W_m",
    "true_D_m",
    "true_wLD",
    "true_wWD",
    "true_wLW",
    "pred_L_m",
    "pred_W_m",
    "pred_D_m",
    "pred_wLD",
    "pred_wWD",
    "pred_wLW",
]
METRIC_FIELDS = [
    "variant",
    "seed",
    "selected_by_validation",
    "eligible_for_multiseed",
    "split",
    "sample_count",
    "selection_score",
    "test_final_only",
    "profile_weight",
    "dimension_weight",
    "curvature_aux_weight",
    "soft_mask_weight",
    "normalized_param_mae",
    "dimension_mae_norm",
    "curvature_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "curvature_mae",
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
    "profile_depth_rmse_norm",
    "er_like_profile_error",
    "max_depth_error_m",
    "volume_proxy_rel_error",
    "profile_rmse_delta_vs_20_77",
    "dice_delta_vs_20_77",
    "dimension_delta_vs_ref_val",
    "notes",
]
GROUP_FIELDS = [
    "variant",
    "seed",
    "selected_by_validation",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "normalized_param_mae",
    "dimension_mae_norm",
    "curvature_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
    "er_like_profile_error",
]


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    profile_weight: float
    dimension_weight: float
    curvature_aux_weight: float
    soft_mask_weight: float
    bound_weight: float = 0.01


class RBCConvRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 96),
            nn.GELU(),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def candidate_configs() -> list[CandidateConfig]:
    return [
        CandidateConfig("P1_profile_primary_w0p5", 0.5, 0.25, 0.05, 0.10),
        CandidateConfig("P2_profile_primary_dim_guard_w1p0", 1.0, 0.50, 0.05, 0.10),
        CandidateConfig("P3_profile_primary_w2p0", 2.0, 0.25, 0.05, 0.10),
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)


def selected_reference_profile(path: Path = REFERENCE_PROFILE) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f) if str(row.get("selected_seed", "")).lower() == "true"]
    for row in rows:
        row["variant"] = "P0_reference_20_77"
        row["seed"] = 42
        row["selected_by_validation"] = False
        row["er_like_profile_error"] = row.get("er_like_profile_error", "")
        row["max_depth_error_m"] = row.get("max_depth_error_m", "")
    return rows


def add_profile_extras(dataset: Any, pred_raw: np.ndarray, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pred_clipped = np.asarray([[float(row[f"pred_{name}"]) for name in PARAM_NAMES] for row in rows], dtype=np.float32) if rows and "pred_L_m" in rows[0] else None
    if pred_clipped is None:
        pred_clipped = np.asarray(pred_raw, dtype=np.float32)
    pred_grids = np.asarray([depth_grid_from_params(params) for params in pred_clipped], dtype=np.float32)
    er_like = er_like_profile_error(pred_grids, dataset.profile_depth_grid_m)
    max_depth = max_depth_error(pred_grids, dataset.profile_depth_grid_m)
    for idx, row in enumerate(rows):
        row["er_like_profile_error"] = float(er_like[idx])
        row["max_depth_error_m"] = float(max_depth[idx])
        for pidx, name in enumerate(PARAM_NAMES):
            row[f"true_{name}"] = float(dataset.rbc_params[idx, pidx])
            row[f"pred_{name}"] = float(pred_raw[idx, pidx])
    return rows


def evaluate_predictions(dataset: Any, pred_raw: np.ndarray, stats: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = evaluate_param_predictions(dataset, pred_raw, stats)
    return add_profile_extras(dataset, pred_raw, rows)


def aggregate_rows(rows: list[dict[str, Any]], variant: str, seed: int, config: CandidateConfig | None, selected: bool, eligible: bool, split: str, score: float | str, depth_scale: float, notes: str = "", test_final: bool | str = "") -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split]
    out: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "selected_by_validation": selected,
        "eligible_for_multiseed": eligible,
        "split": split,
        "sample_count": len(subset),
        "selection_score": score,
        "test_final_only": test_final,
        "profile_weight": "" if config is None else config.profile_weight,
        "dimension_weight": "" if config is None else config.dimension_weight,
        "curvature_aux_weight": "" if config is None else config.curvature_aux_weight,
        "soft_mask_weight": "" if config is None else config.soft_mask_weight,
        "notes": notes,
    }
    keys = [
        "normalized_param_mae_mean",
        "dimension_param_mae_norm",
        "curvature_param_mae_norm",
        "L_mae_mm",
        "W_mae_mm",
        "D_mae_mm",
        "wLD_abs_error",
        "wWD_abs_error",
        "wLW_abs_error",
        "curvature_mae_mean",
        "projected_mask_iou",
        "projected_mask_dice",
        "profile_depth_rmse_m",
        "er_like_profile_error",
        "max_depth_error_m",
        "volume_proxy_rel_error",
    ]
    aliases = {
        "normalized_param_mae_mean": "normalized_param_mae",
        "dimension_param_mae_norm": "dimension_mae_norm",
        "curvature_param_mae_norm": "curvature_mae_norm",
        "curvature_mae_mean": "curvature_mae",
    }
    if subset:
        for key in keys:
            vals = [float(row[key]) for row in subset if str(row.get(key, "")) != ""]
            out[aliases.get(key, key)] = float(np.mean(vals)) if vals else ""
    out["profile_depth_rmse_norm"] = float(out.get("profile_depth_rmse_m", math.nan)) / max(depth_scale, 1.0e-12) if out.get("profile_depth_rmse_m", "") != "" else ""
    out["profile_rmse_delta_vs_20_77"] = float(out["profile_depth_rmse_m"]) - REF_NEURAL["profile_rmse"] if out.get("profile_depth_rmse_m", "") != "" else ""
    out["dice_delta_vs_20_77"] = float(out["projected_mask_dice"]) - REF_NEURAL["dice"] if out.get("projected_mask_dice", "") != "" else ""
    return out


def group_rows(rows: list[dict[str, Any]], variant: str, seed: int, selected: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        base = [row for row in rows if row["split"] == split]
        for field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
            for value in sorted({str(row[field]) for row in base}):
                subset = [row for row in base if str(row[field]) == value]
                if not subset:
                    continue

                def avg(key: str) -> float:
                    return float(np.mean([float(row[key]) for row in subset]))

                out.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "selected_by_validation": selected,
                        "split": split,
                        "group_field": field,
                        "group_value": value,
                        "sample_count": len(subset),
                        "normalized_param_mae": avg("normalized_param_mae_mean"),
                        "dimension_mae_norm": avg("dimension_param_mae_norm"),
                        "curvature_mae": avg("curvature_mae_mean"),
                        "L_mae_mm": avg("L_mae_mm"),
                        "W_mae_mm": avg("W_mae_mm"),
                        "D_mae_mm": avg("D_mae_mm"),
                        "wLD_abs_error": avg("wLD_abs_error"),
                        "wWD_abs_error": avg("wWD_abs_error"),
                        "wLW_abs_error": avg("wLW_abs_error"),
                        "projected_mask_iou": avg("projected_mask_iou"),
                        "projected_mask_dice": avg("projected_mask_dice"),
                        "profile_depth_rmse_m": avg("profile_depth_rmse_m"),
                        "er_like_profile_error": avg("er_like_profile_error"),
                    }
                )
    return out


def reference_metrics(rows: list[dict[str, Any]], split: str, depth_scale: float) -> dict[str, float]:
    agg = aggregate_rows(rows, "P0_reference_20_77", 42, None, False, False, split, "", depth_scale, "reference")
    return {
        "profile_rmse": float(agg["profile_depth_rmse_m"]),
        "profile_rmse_norm": float(agg["profile_depth_rmse_norm"]),
        "dimension": float(agg["dimension_mae_norm"]),
        "curvature": float(agg["curvature_mae"]),
        "dice": float(agg["projected_mask_dice"]),
        "total": float(agg["normalized_param_mae"]),
    }


def candidate_selection_score(row: dict[str, Any], ref_val: dict[str, float]) -> float:
    return (
        float(row["profile_depth_rmse_norm"])
        + 0.50 * float(row["er_like_profile_error"])
        + 0.25 * max(0.0, ref_val["dice"] - float(row["projected_mask_dice"]))
        + 0.20 * float(row["dimension_mae_norm"])
        + 0.05 * float(row["curvature_mae_norm"])
    )


def fast_split_metrics(
    pred_norm_np: np.ndarray,
    y_norm: np.ndarray,
    dataset: Any,
    stats: dict[str, np.ndarray],
    indices: np.ndarray,
    depth_scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, Any]:
    pred_norm = torch.as_tensor(pred_norm_np, dtype=torch.float32)
    y_mean = torch.as_tensor(stats["y_mean"], dtype=torch.float32)
    y_std = torch.as_tensor(stats["y_std"], dtype=torch.float32)
    pred_raw = clip_params_torch(denormalize_torch(pred_norm, y_mean, y_std), lower, upper)
    true_grid = torch.as_tensor(dataset.profile_depth_grid_m[indices], dtype=torch.float32)
    pred_grid = depth_grid_torch(pred_raw)
    diff = pred_grid - true_grid
    profile_rmse = torch.sqrt(torch.mean(diff * diff, dim=(1, 2))).cpu().numpy()
    er_num = torch.sum(diff * diff, dim=(1, 2))
    er_den = torch.clamp(torch.sum(true_grid * true_grid, dim=(1, 2)), min=1.0e-20)
    er_like = torch.sqrt(er_num / er_den).cpu().numpy()
    pose = torch.as_tensor(dataset.profile_pose[indices], dtype=torch.float32)
    true_mask = torch.as_tensor(dataset.projected_mask_2d[indices].astype(np.float32), dtype=torch.float32)
    pred_mask = soft_projected_mask_torch(pred_raw, pose) >= 0.5
    target_mask = true_mask >= 0.5
    intersection = torch.logical_and(pred_mask, target_mask).sum(dim=(1, 2)).to(torch.float32)
    union = torch.logical_or(pred_mask, target_mask).sum(dim=(1, 2)).to(torch.float32)
    pred_area = pred_mask.sum(dim=(1, 2)).to(torch.float32)
    true_area = target_mask.sum(dim=(1, 2)).to(torch.float32)
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union)).cpu().numpy()
    dice = torch.where(pred_area + true_area > 0, 2.0 * intersection / (pred_area + true_area), torch.ones_like(pred_area)).cpu().numpy()
    norm_err = np.abs(pred_norm_np - y_norm[indices])
    return {
        "profile_depth_rmse_m": float(np.mean(profile_rmse)),
        "profile_depth_rmse_norm": float(np.mean(profile_rmse)) / max(depth_scale, 1.0e-12),
        "er_like_profile_error": float(np.mean(er_like)),
        "dimension_mae_norm": float(norm_err[:, :3].mean()),
        "curvature_mae_norm": float(norm_err[:, 3:].mean()),
        "projected_mask_iou": float(np.mean(iou)),
        "projected_mask_dice": float(np.mean(dice)),
        "normalized_param_mae": float(norm_err.mean()),
        "curvature_mae": float(np.mean(np.abs(pred_raw.cpu().numpy()[:, 3:] - dataset.rbc_params[indices, 3:]))),
    }


def soft_dice_loss(pred_soft: torch.Tensor, target_mask: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    target = target_mask.to(dtype=pred_soft.dtype)
    intersection = torch.sum(pred_soft * target, dim=(1, 2))
    denom = torch.sum(pred_soft, dim=(1, 2)) + torch.sum(target, dim=(1, 2)) + eps
    return torch.mean(1.0 - (2.0 * intersection + eps) / denom)


def denormalize_torch(y_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return y_norm * y_std.to(y_norm.device) + y_mean.to(y_norm.device)


def profile_primary_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    true_grid: torch.Tensor,
    true_mask: torch.Tensor,
    pose: torch.Tensor,
    config: CandidateConfig,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    depth_scale: float,
) -> torch.Tensor:
    pred_raw = denormalize_torch(pred_norm, y_mean, y_std)
    pred_clipped = clip_params_torch(pred_raw, lower, upper)
    pred_grid = depth_grid_torch(pred_clipped)
    profile_loss = torch.nn.functional.smooth_l1_loss(pred_grid / depth_scale, true_grid / depth_scale)
    dim_loss = torch.nn.functional.smooth_l1_loss(pred_norm[:, :3], target_norm[:, :3])
    curv_loss = torch.nn.functional.smooth_l1_loss(pred_norm[:, 3:], target_norm[:, 3:])
    mask_loss = torch.tensor(0.0, dtype=pred_norm.dtype, device=pred_norm.device)
    if config.soft_mask_weight > 0:
        pred_soft = soft_projected_mask_torch(pred_clipped, pose)
        mask_loss = soft_dice_loss(pred_soft, true_mask)
    bound = soft_bound_penalty(pred_raw, lower, upper)
    return (
        config.profile_weight * profile_loss
        + config.dimension_weight * dim_loss
        + config.curvature_aux_weight * curv_loss
        + config.soft_mask_weight * mask_loss
        + config.bound_weight * bound
    )


def predict_norm(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
    return pred.astype(np.float32)


def train_candidate(config: CandidateConfig, seed: int, x_norm: np.ndarray, y_norm: np.ndarray, dataset: Any, stats: dict[str, np.ndarray], splits: dict[str, np.ndarray], epochs: int, batch_size: int, lr: float, weight_decay: float, depth_scale: float) -> dict[str, Any]:
    set_seed(seed)
    model = RBCConvRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = splits["train"]
    train_params = dataset.rbc_params[train_idx]
    lower = torch.as_tensor(train_params.min(axis=0), dtype=torch.float32)
    upper = torch.as_tensor(train_params.max(axis=0), dtype=torch.float32)
    y_mean = torch.as_tensor(stats["y_mean"], dtype=torch.float32)
    y_std = torch.as_tensor(stats["y_std"], dtype=torch.float32)
    train_ds = TensorDataset(
        torch.as_tensor(x_norm[train_idx], dtype=torch.float32),
        torch.as_tensor(y_norm[train_idx], dtype=torch.float32),
        torch.as_tensor(dataset.profile_depth_grid_m[train_idx], dtype=torch.float32),
        torch.as_tensor(dataset.projected_mask_2d[train_idx], dtype=torch.float32),
        torch.as_tensor(dataset.profile_pose[train_idx], dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = math.inf
    min_train = math.inf
    epoch_rows: list[dict[str, Any]] = []
    ref_val_for_score = reference_metrics(selected_reference_profile(), "val", depth_scale)
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb, gb, mb, pb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = profile_primary_loss(pred, yb, gb, mb, pb, config, y_mean, y_std, lower, upper, depth_scale)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        pred_train = predict_norm(model, x_norm[train_idx])
        pred_val = predict_norm(model, x_norm[splits["val"]])
        train_agg = fast_split_metrics(pred_train, y_norm, dataset, stats, train_idx, depth_scale, lower, upper)
        val_agg = fast_split_metrics(pred_val, y_norm, dataset, stats, splits["val"], depth_scale, lower, upper)
        val_score = candidate_selection_score(val_agg, ref_val_for_score)
        min_train = min(min_train, float(train_agg["normalized_param_mae"]))
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "variant": config.name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_profile_depth_rmse_m": train_agg["profile_depth_rmse_m"],
                "val_profile_depth_rmse_m": val_agg["profile_depth_rmse_m"],
                "train_er_like_profile_error": train_agg["er_like_profile_error"],
                "val_er_like_profile_error": val_agg["er_like_profile_error"],
                "train_dimension_mae_norm": train_agg["dimension_mae_norm"],
                "val_dimension_mae_norm": val_agg["dimension_mae_norm"],
                "train_curvature_mae_norm": train_agg["curvature_mae_norm"],
                "val_curvature_mae_norm": val_agg["curvature_mae_norm"],
                "train_projected_mask_dice": train_agg["projected_mask_dice"],
                "val_projected_mask_dice": val_agg["projected_mask_dice"],
                "val_selection_score": val_score,
            }
        )
    if best_state is None:
        raise RuntimeError(f"no validation state selected for {config.name}")
    model.load_state_dict(best_state)
    pred_all = predict_norm(model, x_norm)
    return {"variant": config.name, "seed": seed, "best_epoch": best_epoch, "best_val_score": best_score, "min_train_normalized_param_mae": min_train, "pred_norm": pred_all, "epoch_rows": epoch_rows}


def evaluate_predictions_subset(dataset: Any, pred_raw_subset: np.ndarray, stats: dict[str, np.ndarray], indices: np.ndarray) -> list[dict[str, Any]]:
    full_pred = dataset.rbc_params.copy()
    full_pred[indices] = pred_raw_subset
    full_rows = evaluate_predictions(dataset, full_pred, stats)
    wanted = set(int(i) for i in indices)
    return [row for idx, row in enumerate(full_rows) if idx in wanted]


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics, args.group_summary, args.profile_metrics], args.overwrite)
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    stats = train_normalization(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    depth_scale = float(np.max(dataset.rbc_params[splits["train"], 2]) - np.min(dataset.rbc_params[splits["train"], 2]))
    if depth_scale <= 1.0e-12:
        depth_scale = float(np.mean(dataset.rbc_params[splits["train"], 2]))
    reference_rows = selected_reference_profile()
    ref_val = reference_metrics(reference_rows, "val", depth_scale)
    ref_test = reference_metrics(reference_rows, "test", depth_scale)

    metric_rows: list[dict[str, Any]] = []
    profile_rows_out: list[dict[str, Any]] = []
    group_rows_out: list[dict[str, Any]] = []
    val_rows_for_selection: list[dict[str, Any]] = []
    trained: list[tuple[CandidateConfig, dict[str, Any]]] = []
    for ref_split in ("train", "val", "test"):
        metric_rows.append(aggregate_rows(reference_rows, "P0_reference_20_77", 42, None, False, False, ref_split, "", depth_scale, "reference metrics only", False))

    for config in candidate_configs():
        out = train_candidate(config, args.seed, x_norm, y_norm, dataset, stats, splits, args.epochs, args.batch_size, args.lr, args.weight_decay, depth_scale)
        trained.append((config, out))
        pred_raw = denormalize_y(out["pred_norm"], stats)
        train_val_idx = np.concatenate([splits["train"], splits["val"]])
        rows = evaluate_predictions_subset(dataset, pred_raw[train_val_idx], stats, train_val_idx)
        for row in rows:
            row["variant"] = config.name
            row["seed"] = args.seed
            row["selected_by_validation"] = False
        for split in ("train", "val"):
            agg = aggregate_rows(rows, config.name, args.seed, config, False, False, split, "", depth_scale, "validation screen only")
            if split == "val":
                score = candidate_selection_score(agg, ref_val)
                agg["selection_score"] = score
                val_rows_for_selection.append(agg)
            metric_rows.append(agg)
        profile_rows_out.extend(rows)
        group_rows_out.extend(group_rows(rows, config.name, args.seed, False))

    selected_val = min(val_rows_for_selection, key=lambda row: float(row["selection_score"]))
    selected_name = str(selected_val["variant"])
    selected_config, selected_out = [item for item in trained if item[0].name == selected_name][0]
    eligible = (
        float(selected_val["profile_depth_rmse_m"]) < ref_val["profile_rmse"]
        and float(selected_val["projected_mask_dice"]) >= ref_val["dice"] - 0.02
        and float(selected_val["dimension_mae_norm"]) <= ref_val["dimension"] + 0.10
    )
    selected_pred_raw = denormalize_y(selected_out["pred_norm"], stats)
    selected_rows = evaluate_predictions(dataset, selected_pred_raw, stats)
    for row in selected_rows:
        row["variant"] = selected_config.name
        row["seed"] = args.seed
        row["selected_by_validation"] = True
    selected_test = aggregate_rows(selected_rows, selected_config.name, args.seed, selected_config, True, eligible, "test", "", depth_scale, "validation-selected candidate; test final only for screen", True)
    metric_rows.append(selected_test)
    profile_rows_out.extend(selected_rows)
    group_rows_out.extend(group_rows(selected_rows, selected_config.name, args.seed, True))
    for row in metric_rows:
        if row["variant"] == selected_name:
            row["selected_by_validation"] = True
            row["eligible_for_multiseed"] = eligible
        row["dimension_delta_vs_ref_val"] = float(row["dimension_mae_norm"]) - ref_val["dimension"] if row.get("dimension_mae_norm", "") != "" else ""
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_rows_out, GROUP_FIELDS)
    write_csv(args.profile_metrics, profile_rows_out, PROFILE_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 profile-primary candidate screen summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"seed: {args.seed}",
                "scope: R1 six-params profile-primary loss candidate screen; no COMSOL, no data generation, no NPZ modification, no baseline.",
                "model_input: delta_b only, reshaped to 9x201; no rbc_params/profile/mask/split/template/bin/sample_id as input.",
                "target_usage: rbc_params/profile_depth_grid/projected_mask are supervision or metrics only.",
                "candidate_selection: validation-only; test is emitted only for the validation-selected candidate.",
                f"candidates: {', '.join(config.name for config in candidate_configs())}",
                f"reference_20_77_val_profile_depth_rmse_m: {ref_val['profile_rmse']:.9f}",
                f"reference_20_77_val_dimension_mae_norm: {ref_val['dimension']:.6f}",
                f"reference_20_77_val_projected_mask_dice: {ref_val['dice']:.6f}",
                f"reference_20_77_test_profile_depth_rmse_m: {ref_test['profile_rmse']:.9f}",
                f"selected_candidate: {selected_name}",
                f"selected_val_selection_score: {float(selected_val['selection_score']):.6f}",
                f"selected_val_profile_depth_rmse_m: {float(selected_val['profile_depth_rmse_m']):.9f}",
                f"selected_val_er_like_profile_error: {float(selected_val['er_like_profile_error']):.6f}",
                f"selected_val_dimension_mae_norm: {float(selected_val['dimension_mae_norm']):.6f}",
                f"selected_val_projected_mask_dice: {float(selected_val['projected_mask_dice']):.6f}",
                f"eligible_for_multiseed: {eligible}",
                "",
                f"selected_test_total_mae: {float(selected_test['normalized_param_mae']):.6f}",
                f"selected_test_LWD_mae_mm: {float(selected_test['L_mae_mm']):.6f}/{float(selected_test['W_mae_mm']):.6f}/{float(selected_test['D_mae_mm']):.6f}",
                f"selected_test_wLD_wWD_wLW_aux: {float(selected_test['wLD_abs_error']):.6f}/{float(selected_test['wWD_abs_error']):.6f}/{float(selected_test['wLW_abs_error']):.6f}",
                f"selected_test_profile_depth_rmse_m: {float(selected_test['profile_depth_rmse_m']):.9f}",
                f"selected_test_er_like_profile_error: {float(selected_test['er_like_profile_error']):.6f}",
                f"selected_test_projected_mask_iou_dice: {float(selected_test['projected_mask_iou']):.6f}/{float(selected_test['projected_mask_dice']):.6f}",
                f"selected_test_profile_rmse_delta_vs_20_77: {float(selected_test['profile_rmse_delta_vs_20_77']):.9f}",
                "stage_C_gate: run selected multi-seed only if eligible_for_multiseed is True.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP)
    parser.add_argument("--profile-metrics", type=Path, default=PROFILE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
