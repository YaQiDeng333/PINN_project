#!/usr/bin/env python
"""Seed-42 screen for surface RBC delta_b + NLS-lite feature fusion.

Inputs are restricted to normalized delta_b/BxByBz channels and train-scaled
``nlslite_*`` columns. sample_id is used only for joining/reporting, split only
for train/val/test routing. Candidate selection is validation-only; test rows
are written only for the validation-selected fusion candidate plus fixed
reference comparators.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_surface_rbc_nls_feature_fusion_dataset import build_inputs
from load_true_3d_rbc_pilot_dataset import (
    ROOT,
    V3_240_DATASET_ID,
    aggregate_prediction_rows,
    check_no_overwrite,
    clip_params_to_train_bounds,
    denormalize_y,
    depth_grid_from_params,
    depth_map_from_params,
    mask_metrics,
    projected_mask_from_params,
    write_csv,
)
from true_3d_rbc_profile_generator import depth_grid_torch, soft_bound_penalty


SUMMARY = ROOT / "results/summaries/surface_rbc_nls_feature_fusion_candidate_screen_summary.txt"
METRICS = ROOT / "results/metrics/surface_rbc_nls_feature_fusion_candidate_screen_metrics.csv"

REF_20_85 = {
    "label": "20.85_formal_rerun_20.77_protocol",
    "total": 0.6780143536818333,
    "L_mm": 1.8918915996566796,
    "W_mm": 2.1857599088778863,
    "D_mm": 0.8002313476246901,
    "wMAE": 0.20107580616306037,
    "wLD": 0.2094394748027508,
    "wWD": 0.20446911683449379,
    "wLW": 0.18931882809369993,
    "profile_rmse": 0.0003877372636895579,
    "er_like": 0.3405436946031375,
    "iou": 0.7506502455785019,
    "dice": 0.8477271366767738,
}
REF_20_77 = dict(REF_20_85, label="20.77_original_candidate", er_like=math.nan)
REF_24_1 = {
    "label": "24.1_NLS_lite_feature_baseline",
    "total": 0.6540464762693796,
    "L_mm": 1.9136665956332133,
    "W_mm": 1.9917513088156016,
    "D_mm": 0.9674908621953084,
    "wMAE": 0.18572381000297192,
    "wLD": 0.1975561517935533,
    "wWD": 0.1913395753273597,
    "wLW": 0.16827570016567522,
    "profile_rmse": 0.00044518171694690886,
    "er_like": 0.431187,
    "iou": 0.769352875827035,
    "dice": 0.8629883531612551,
}
REF_20_81 = {
    "label": "20.81_feature_fusion_visual_comparator",
    "total": 0.667888397971789,
    "L_mm": 2.0303866893817215,
    "W_mm": 1.8066299970333393,
    "D_mm": 0.9566183099761988,
    "wMAE": 0.19448333899848735,
    "wLD": 0.21707869340211916,
    "wWD": 0.20230379012914804,
    "wLW": 0.1640675358283214,
    "profile_rmse": 0.00044529733223988354,
    "er_like": math.nan,
    "iou": 0.77454093657963,
    "dice": 0.8665725224434271,
}

METRIC_FIELDS = [
    "candidate",
    "seed",
    "candidate_role",
    "feature_count",
    "selected_by_validation",
    "eligible_for_multiseed",
    "split",
    "sample_count",
    "selection_score",
    "best_epoch",
    "test_final_only",
    "normalized_param_mae",
    "dimension_mae_norm",
    "curvature_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "wMAE",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "projected_mask_iou",
    "projected_mask_dice",
    "total_delta_vs_20_85",
    "wMAE_delta_vs_20_85",
    "profile_rmse_delta_vs_20_85",
    "dice_delta_vs_20_85",
    "total_delta_vs_24_1",
    "wMAE_delta_vs_24_1",
    "profile_rmse_delta_vs_24_1",
    "dice_delta_vs_24_1",
    "notes",
]


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    role: str
    model_kind: str
    w_loss_weight: float
    profile_loss_weight: float


class ConvEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.net(x), 1)


class FeatureMLP(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, latent_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class F1LateFusion(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.feature_mlp = FeatureMLP(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(64 * 8 + 96, 128),
            nn.GELU(),
            nn.Linear(128, 48),
            nn.GELU(),
            nn.Linear(48, 6),
        )

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(x), self.feature_mlp(features)], dim=1))


class F2FeatureGatedFusion(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.feature_mlp = FeatureMLP(feature_dim)
        self.base_head = nn.Sequential(nn.Linear(64 * 8, 96), nn.GELU(), nn.Linear(96, 6))
        self.feature_delta = nn.Sequential(nn.Linear(64 * 8 + 96, 96), nn.GELU(), nn.Linear(96, 6))
        self.gate = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 6), nn.Sigmoid())

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        feature_latent = self.feature_mlp(features)
        return self.base_head(latent) + self.gate(feature_latent) * self.feature_delta(torch.cat([latent, feature_latent], dim=1))


class F3WHeadAssisted(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.feature_mlp = FeatureMLP(feature_dim)
        self.dim_head = nn.Sequential(nn.Linear(64 * 8, 96), nn.GELU(), nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 3))
        self.w_head = nn.Sequential(nn.Linear(64 * 8 + 96, 96), nn.GELU(), nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 3))

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        feature_latent = self.feature_mlp(features)
        return torch.cat([self.dim_head(latent), self.w_head(torch.cat([latent, feature_latent], dim=1))], dim=1)


def candidate_configs() -> list[CandidateConfig]:
    return [
        CandidateConfig("F1_late_fusion", "Conv1D latent + feature MLP concat six-param head", "F1", 0.15, 0.08),
        CandidateConfig("F2_feature_gated_fusion", "feature gate controls six-param correction", "F2", 0.25, 0.08),
        CandidateConfig("F3_w_head_assisted", "Conv LWD head with NLS-assisted w head", "F3", 0.35, 0.08),
    ]


def make_model(config: CandidateConfig, feature_dim: int) -> nn.Module:
    if config.model_kind == "F1":
        return F1LateFusion(feature_dim)
    if config.model_kind == "F2":
        return F2FeatureGatedFusion(feature_dim)
    if config.model_kind == "F3":
        return F3WHeadAssisted(feature_dim)
    raise RuntimeError(f"unknown model kind: {config.model_kind}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)


def predict_norm(model: nn.Module, x: np.ndarray, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(features, dtype=torch.float32))
    return pred.cpu().numpy().astype(np.float32)


def normalized_components(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.abs(y_true - y_pred)
    return {
        "total": float(err.mean()),
        "dimension": float(err[:, :3].mean()),
        "curvature": float(err[:, 3:].mean()),
    }


def clip_subset_to_train_bounds(dataset: Any, pred_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_mask = dataset.split == "train"
    low = dataset.rbc_params[train_mask].min(axis=0)
    high = dataset.rbc_params[train_mask].max(axis=0)
    pred = np.asarray(pred_raw, dtype=np.float32)
    clipped = np.clip(pred, low[None, :], high[None, :])
    clipped_flag = np.any(np.abs(clipped - pred) > 1.0e-12, axis=1)
    return clipped.astype(np.float32), clipped_flag


def profile_rmse_for_indices(dataset: Any, pred_raw: np.ndarray, indices: np.ndarray) -> float:
    pred_params, _ = clip_subset_to_train_bounds(dataset, np.asarray(pred_raw, dtype=np.float32))
    values = []
    for row_idx, idx in enumerate(indices):
        pred_depth = depth_grid_from_params(pred_params[row_idx])
        values.append(float(np.sqrt(np.mean((pred_depth - dataset.profile_depth_grid_m[idx]) ** 2))))
    return float(np.mean(values)) if values else math.nan


def predict_raw_for_indices(model: nn.Module, inputs: Any, indices: np.ndarray) -> np.ndarray:
    pred_norm = predict_norm(model, inputs.x_norm[indices], inputs.feature_norm[indices])
    return denormalize_y(pred_norm, inputs.stats)


def epoch_selection_score(total: float, dimension: float, curvature: float, profile_rmse: float) -> float:
    return float(total + 0.10 * dimension + 0.20 * curvature + 180.0 * profile_rmse)


def train_candidate(
    config: CandidateConfig,
    seed: int,
    inputs: Any,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    set_seed(seed)
    model = make_model(config, inputs.feature_norm.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = inputs.splits["train"]
    train_params = inputs.dataset.rbc_params[train_idx]
    lower = torch.as_tensor(train_params.min(axis=0), dtype=torch.float32)
    upper = torch.as_tensor(train_params.max(axis=0), dtype=torch.float32)
    y_mean = torch.as_tensor(inputs.stats["y_mean"], dtype=torch.float32)
    y_std = torch.as_tensor(inputs.stats["y_std"], dtype=torch.float32)
    profile_scale = float(max(np.mean(inputs.dataset.rbc_params[train_idx, 2]), 1.0e-6))
    train_ds = TensorDataset(
        torch.as_tensor(inputs.x_norm[train_idx], dtype=torch.float32),
        torch.as_tensor(inputs.feature_norm[train_idx], dtype=torch.float32),
        torch.as_tensor(inputs.y_norm[train_idx], dtype=torch.float32),
        torch.as_tensor(inputs.dataset.profile_depth_grid_m[train_idx] / profile_scale, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = math.inf
    min_train = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, fb, yb, depth_b in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, fb)
            raw_pred = pred * y_std + y_mean
            profile_pred = depth_grid_torch(raw_pred) / profile_scale
            base = torch.nn.functional.smooth_l1_loss(pred, yb)
            w_aux = torch.nn.functional.smooth_l1_loss(pred[:, 3:], yb[:, 3:])
            profile = torch.nn.functional.mse_loss(profile_pred, depth_b)
            bounds = soft_bound_penalty(raw_pred, lower, upper)
            loss = base + config.w_loss_weight * w_aux + config.profile_loss_weight * profile + 0.03 * bounds
            loss.backward()
            optimizer.step()

        pred_train = predict_norm(model, inputs.x_norm[train_idx], inputs.feature_norm[train_idx])
        pred_val = predict_norm(model, inputs.x_norm[inputs.splits["val"]], inputs.feature_norm[inputs.splits["val"]])
        train_comp = normalized_components(inputs.y_norm[train_idx], pred_train)
        val_comp = normalized_components(inputs.y_norm[inputs.splits["val"]], pred_val)
        pred_val_raw = denormalize_y(pred_val, inputs.stats)
        val_profile = profile_rmse_for_indices(inputs.dataset, pred_val_raw, inputs.splits["val"])
        score = epoch_selection_score(val_comp["total"], val_comp["dimension"], val_comp["curvature"], val_profile)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        min_train = min(min_train, train_comp["total"])

    if best_state is None:
        raise RuntimeError(f"no best validation state for {config.name}")
    model.load_state_dict(best_state)
    return {
        "candidate": config.name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_epoch_score": best_score,
        "min_train_normalized_param_mae": min_train,
        "model": model,
    }


def evaluate_subset(dataset: Any, pred_raw: np.ndarray, stats: dict[str, np.ndarray], indices: np.ndarray) -> list[dict[str, Any]]:
    if len(pred_raw) != len(indices):
        raise RuntimeError(f"pred_raw length must match indices for test-final-only enforcement: {len(pred_raw)} != {len(indices)}")
    pred_params, clipped = clip_subset_to_train_bounds(dataset, np.asarray(pred_raw, dtype=np.float32))
    pred_norm = (pred_params - stats["y_mean"]) / stats["y_std"]
    true_norm = (dataset.rbc_params - stats["y_mean"]) / stats["y_std"]
    rows: list[dict[str, Any]] = []
    clip_fraction = float(np.mean(clipped)) if len(indices) else 0.0
    for row_idx, idx in enumerate(indices):
        pred_mask = projected_mask_from_params(pred_params[row_idx], dataset.profile_pose[idx])
        mask_row = mask_metrics(pred_mask, dataset.projected_mask_2d[idx])
        pred_depth = depth_grid_from_params(pred_params[row_idx])
        true_depth = dataset.profile_depth_grid_m[idx]
        depth_rmse = float(np.sqrt(np.mean((pred_depth - true_depth) ** 2)))
        denom = float(np.sum(true_depth**2))
        er_like = 0.0 if denom <= 1.0e-20 else float(np.sqrt(np.sum((pred_depth - true_depth) ** 2) / denom))
        true_volume = float(dataset.profile_depth_map_xy_m[idx].sum())
        pred_volume = float(depth_map_from_params(pred_params[row_idx], dataset.profile_pose[idx]).sum())
        volume_error = 0.0 if abs(true_volume) < 1.0e-12 else abs(pred_volume - true_volume) / abs(true_volume)
        param_abs = np.abs(pred_params[row_idx] - dataset.rbc_params[idx])
        param_norm_abs = np.abs(pred_norm[row_idx] - true_norm[idx])
        rows.append(
            {
                "sample_id": str(dataset.sample_ids[idx]),
                "split": str(dataset.split[idx]),
                "curvature_template": str(dataset.curvature_template[idx]),
                "depth_bin": str(dataset.depth_bin[idx]),
                "aspect_bin": str(dataset.aspect_bin[idx]),
                "size_bin": str(dataset.size_bin[idx]),
                "clip_applied": bool(clipped[row_idx]),
                "clip_fraction": clip_fraction,
                "normalized_param_mae_mean": float(np.mean(param_norm_abs)),
                "dimension_param_mae_norm": float(np.mean(param_norm_abs[:3])),
                "curvature_param_mae_norm": float(np.mean(param_norm_abs[3:])),
                "L_mae_m": float(param_abs[0]),
                "W_mae_m": float(param_abs[1]),
                "D_mae_m": float(param_abs[2]),
                "L_mae_mm": float(param_abs[0] * 1000.0),
                "W_mae_mm": float(param_abs[1] * 1000.0),
                "D_mae_mm": float(param_abs[2] * 1000.0),
                "wLD_abs_error": float(param_abs[3]),
                "wWD_abs_error": float(param_abs[4]),
                "wLW_abs_error": float(param_abs[5]),
                "curvature_mae_mean": float(np.mean(param_abs[3:])),
                "projected_mask_iou": mask_row["iou"],
                "projected_mask_dice": mask_row["dice"],
                "projected_mask_area_error": mask_row["area_error"],
                "projected_mask_center_error_px": mask_row["center_error"],
                "profile_depth_rmse_m": depth_rmse,
                "er_like_profile_error": er_like,
                "volume_proxy_rel_error": float(volume_error),
            }
        )
    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and str(row[key]) not in {"", "nan"}]
    return float(np.mean(values)) if values else math.nan


def aggregate_eval_rows(rows: list[dict[str, Any]], candidate: str, split: str) -> dict[str, Any]:
    agg = aggregate_prediction_rows(rows, candidate, split)
    subset = [row for row in rows if row["split"] == split]
    agg["er_like_profile_error_mean"] = mean(subset, "er_like_profile_error")
    return agg


def row_from_aggregate(
    candidate: str,
    seed: int,
    role: str,
    feature_count: int | str,
    split: str,
    aggregate: dict[str, Any],
    score: float | str,
    best_epoch: int | str,
    selected: bool,
    eligible: bool,
    test_final_only: bool,
    notes: str,
) -> dict[str, Any]:
    total = aggregate.get("normalized_param_mae_mean_mean", math.nan)
    wmae = aggregate.get("curvature_mae_mean_mean", math.nan)
    profile = aggregate.get("profile_depth_rmse_m_mean", math.nan)
    dice = aggregate.get("projected_mask_dice_mean", math.nan)
    return {
        "candidate": candidate,
        "seed": seed,
        "candidate_role": role,
        "feature_count": feature_count,
        "selected_by_validation": selected,
        "eligible_for_multiseed": eligible,
        "split": split,
        "sample_count": aggregate.get("sample_count", 0),
        "selection_score": score,
        "best_epoch": best_epoch,
        "test_final_only": test_final_only,
        "normalized_param_mae": total,
        "dimension_mae_norm": aggregate.get("dimension_param_mae_norm_mean", math.nan),
        "curvature_mae_norm": aggregate.get("curvature_param_mae_norm_mean", math.nan),
        "L_mae_mm": aggregate.get("L_mae_mm_mean", math.nan),
        "W_mae_mm": aggregate.get("W_mae_mm_mean", math.nan),
        "D_mae_mm": aggregate.get("D_mae_mm_mean", math.nan),
        "wLD_abs_error": aggregate.get("wLD_abs_error_mean", math.nan),
        "wWD_abs_error": aggregate.get("wWD_abs_error_mean", math.nan),
        "wLW_abs_error": aggregate.get("wLW_abs_error_mean", math.nan),
        "wMAE": wmae,
        "profile_depth_rmse_m": profile,
        "er_like_profile_error": aggregate.get("er_like_profile_error_mean", math.nan),
        "projected_mask_iou": aggregate.get("projected_mask_iou_mean", math.nan),
        "projected_mask_dice": dice,
        "total_delta_vs_20_85": total - REF_20_85["total"] if math.isfinite(float(total)) else math.nan,
        "wMAE_delta_vs_20_85": wmae - REF_20_85["wMAE"] if math.isfinite(float(wmae)) else math.nan,
        "profile_rmse_delta_vs_20_85": profile - REF_20_85["profile_rmse"] if math.isfinite(float(profile)) else math.nan,
        "dice_delta_vs_20_85": dice - REF_20_85["dice"] if math.isfinite(float(dice)) else math.nan,
        "total_delta_vs_24_1": total - REF_24_1["total"] if math.isfinite(float(total)) else math.nan,
        "wMAE_delta_vs_24_1": wmae - REF_24_1["wMAE"] if math.isfinite(float(wmae)) else math.nan,
        "profile_rmse_delta_vs_24_1": profile - REF_24_1["profile_rmse"] if math.isfinite(float(profile)) else math.nan,
        "dice_delta_vs_24_1": dice - REF_24_1["dice"] if math.isfinite(float(dice)) else math.nan,
        "notes": notes,
    }


def aggregate_selection_score(row: dict[str, Any]) -> float:
    return float(
        row["normalized_param_mae"]
        + 0.10 * row["dimension_mae_norm"]
        + 0.20 * row["curvature_mae_norm"]
        + 180.0 * row["profile_depth_rmse_m"]
        + 0.10 * max(0.0, REF_20_85["dice"] - row["projected_mask_dice"])
    )


def reference_row(reference: dict[str, float], seed: int = 42) -> dict[str, Any]:
    aggregate = {
        "sample_count": 39,
        "normalized_param_mae_mean_mean": reference["total"],
        "dimension_param_mae_norm_mean": math.nan,
        "curvature_param_mae_norm_mean": math.nan,
        "L_mae_mm_mean": reference["L_mm"],
        "W_mae_mm_mean": reference["W_mm"],
        "D_mae_mm_mean": reference["D_mm"],
        "wLD_abs_error_mean": reference["wLD"],
        "wWD_abs_error_mean": reference["wWD"],
        "wLW_abs_error_mean": reference["wLW"],
        "curvature_mae_mean_mean": reference["wMAE"],
        "profile_depth_rmse_m_mean": reference["profile_rmse"],
        "er_like_profile_error_mean": reference["er_like"],
        "projected_mask_iou_mean": reference["iou"],
        "projected_mask_dice_mean": reference["dice"],
    }
    return row_from_aggregate(
        str(reference["label"]),
        seed,
        "fixed_reference_comparator",
        "",
        "test",
        aggregate,
        "",
        "",
        False,
        False,
        False,
        "reference only; not trained or selected in Stage C",
    )


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics], args.overwrite)
    inputs = build_inputs(args.dataset_id)
    metric_rows: list[dict[str, Any]] = [
        reference_row(REF_20_85, args.seed),
        reference_row(REF_20_77, args.seed),
        reference_row(REF_24_1, args.seed),
        reference_row(REF_20_81, 2026),
    ]
    trained: list[tuple[CandidateConfig, dict[str, Any], dict[str, Any]]] = []

    for config in candidate_configs():
        out = train_candidate(config, args.seed, inputs, args.epochs, args.batch_size, args.lr, args.weight_decay)
        train_val_idx = np.concatenate([inputs.splits["train"], inputs.splits["val"]])
        train_val_pred = predict_raw_for_indices(out["model"], inputs, train_val_idx)
        rows = evaluate_subset(inputs.dataset, train_val_pred, inputs.stats, train_val_idx)
        split_aggs: dict[str, Any] = {}
        for split in ("train", "val"):
            agg = aggregate_eval_rows(rows, config.name, split)
            row = row_from_aggregate(
                config.name,
                args.seed,
                config.role,
                len(inputs.feature_names),
                split,
                agg,
                "",
                out["best_epoch"],
                False,
                False,
                False,
                "validation screen only",
            )
            if split == "val":
                row["selection_score"] = aggregate_selection_score(row)
                split_aggs["val_row"] = row
            metric_rows.append(row)
        trained.append((config, out, split_aggs["val_row"]))

    selected_config, selected_out, selected_val = min(trained, key=lambda item: float(item[2]["selection_score"]))
    eligible = (
        math.isfinite(float(selected_val["selection_score"]))
        and float(selected_val["normalized_param_mae"]) <= 1.05
        and float(selected_val["profile_depth_rmse_m"]) <= 0.0008
    )
    all_idx = np.arange(len(inputs.dataset.sample_ids))
    selected_pred = predict_raw_for_indices(selected_out["model"], inputs, all_idx)
    selected_rows = evaluate_subset(inputs.dataset, selected_pred, inputs.stats, all_idx)
    selected_test_agg = aggregate_eval_rows(selected_rows, selected_config.name, "test")
    selected_test = row_from_aggregate(
        selected_config.name,
        args.seed,
        selected_config.role,
        len(inputs.feature_names),
        "test",
        selected_test_agg,
        "",
        selected_out["best_epoch"],
        True,
        eligible,
        True,
        "validation-selected candidate; test final only",
    )
    metric_rows.append(selected_test)
    for row in metric_rows:
        if row["candidate"] == selected_config.name:
            row["selected_by_validation"] = True
            row["eligible_for_multiseed"] = eligible

    write_csv(args.metrics, metric_rows, METRIC_FIELDS)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface_rbc_nls_feature_fusion_candidate_screen_summary",
                "stage: 24.2 Stage C",
                "",
                f"dataset_id: {inputs.dataset.dataset_id}",
                f"seed: {args.seed}",
                f"epochs: {args.epochs}",
                f"batch_size: {args.batch_size}",
                "scope: candidate diagnostic only; no COMSOL, no data/NPZ write, no checkpoint, no baseline update.",
                "model_inputs: delta_b/BxByBz + train-scaled nlslite_* features",
                "sample_id_use: join/reporting only",
                "split_use: train/val/test routing only",
                "model_selection: validation-only composite of profile RMSE and total normalized MAE",
                f"candidates: {', '.join(config.name for config in candidate_configs())}",
                f"selected_candidate: {selected_config.name}",
                f"selected_candidate_role: {selected_config.role}",
                f"selected_feature_count: {len(inputs.feature_names)}",
                f"selected_val_selection_score: {float(selected_val['selection_score']):.6f}",
                f"selected_val_total_mae: {float(selected_val['normalized_param_mae']):.6f}",
                f"selected_val_profile_depth_rmse_m: {float(selected_val['profile_depth_rmse_m']):.9f}",
                f"selected_val_wMAE: {float(selected_val['wMAE']):.6f}",
                f"selected_val_projected_mask_iou_dice: {float(selected_val['projected_mask_iou']):.6f}/{float(selected_val['projected_mask_dice']):.6f}",
                f"selected_best_epoch: {selected_out['best_epoch']}",
                f"eligible_for_multiseed: {eligible}",
                "",
                f"selected_test_total_mae: {float(selected_test['normalized_param_mae']):.6f}",
                f"selected_test_LWD_mae_mm: {float(selected_test['L_mae_mm']):.6f}/{float(selected_test['W_mae_mm']):.6f}/{float(selected_test['D_mae_mm']):.6f}",
                f"selected_test_wMAE: {float(selected_test['wMAE']):.6f}",
                f"selected_test_wLD_wWD_wLW: {float(selected_test['wLD_abs_error']):.6f}/{float(selected_test['wWD_abs_error']):.6f}/{float(selected_test['wLW_abs_error']):.6f}",
                f"selected_test_profile_depth_rmse_m: {float(selected_test['profile_depth_rmse_m']):.9f}",
                f"selected_test_er_like_profile_error: {float(selected_test['er_like_profile_error']):.6f}",
                f"selected_test_projected_mask_iou_dice: {float(selected_test['projected_mask_iou']):.6f}/{float(selected_test['projected_mask_dice']):.6f}",
                "stage_D_gate: run multi-seed only if eligible_for_multiseed is true.",
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
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
