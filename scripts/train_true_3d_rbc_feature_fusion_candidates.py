#!/usr/bin/env python
"""Seed-42 feature-fusion candidate screen for true-3D RBC curvature.

The model consumes delta_b plus train-scaled, delta_b-derived feature columns.
sample_id and split are used only for joining and splitting, never as model
input. Candidate selection is validation-only; test metrics are emitted only
for the validation-selected candidate.
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
    ROOT,
    aggregate_prediction_rows,
    check_no_overwrite,
    clip_params_to_train_bounds,
    denormalize_y,
    depth_grid_from_params,
    depth_map_from_params,
    evaluate_param_predictions,
    load_dataset,
    mask_metrics,
    normalize_x,
    normalize_y,
    projected_mask_from_params,
    split_indices,
    train_normalization,
    write_csv,
)


FEATURES = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_features.csv"
REFERENCE_PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_profile_metrics.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_feature_fusion_candidate_screen_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_candidate_screen_metrics.csv"
GROUP = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_candidate_group_summary.csv"
PROFILE = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_candidate_profile_metrics.csv"

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
REF_REFINED = {
    "label": "20.79_failed_refinement",
    "total": 0.7533873075093979,
    "L_mm": 2.660231047715896,
    "W_mm": 2.135249491303395,
    "D_mm": 1.1115478296310475,
    "curvature": 0.21158406477517042,
    "wLD": 0.2320939057912582,
    "wWD": 0.21763882270226112,
    "wLW": 0.18501947017816398,
    "iou": 0.7282398476842469,
    "dice": 0.8345969696140306,
    "profile_rmse": 0.0005550889249067181,
}

METRIC_FIELDS = [
    "variant",
    "seed",
    "feature_set",
    "feature_count",
    "curvature_weight",
    "selected_by_validation",
    "eligible_for_multiseed",
    "split",
    "sample_count",
    "selection_score",
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
    "curvature_mae",
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
    "curvature_delta_vs_20_77",
    "total_delta_vs_20_77",
    "dice_delta_vs_20_77",
    "notes",
]
GROUP_FIELDS = [
    "variant",
    "seed",
    "feature_set",
    "selected_by_validation",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "normalized_param_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "curvature_mae",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
]
PROFILE_FIELDS = [
    "variant",
    "seed",
    "feature_set",
    "selected_by_validation",
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "size_bin",
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
]


@dataclass(frozen=True)
class FeatureTransform:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).copy()
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = np.take(self.median, np.where(bad)[1])
        return ((arr - self.mean) / self.std).astype(np.float32)


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    feature_set: str
    prefixes: tuple[str, ...]
    curvature_weight: float


class CurvatureFusionRegressor(nn.Module):
    def __init__(self, feature_dim: int, feature_latent_dim: int = 64) -> None:
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
        latent_dim = 64 * 8
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.GELU(),
            nn.Linear(96, feature_latent_dim),
            nn.GELU(),
        )
        self.dim_head = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.GELU(),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )
        self.curv_head = nn.Sequential(
            nn.Linear(latent_dim + feature_latent_dim, 96),
            nn.GELU(),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        latent = torch.flatten(self.encoder(x), 1)
        feature_latent = self.feature_mlp(features)
        dim = self.dim_head(latent)
        curv = self.curv_head(torch.cat([latent, feature_latent], dim=1))
        return torch.cat([dim, curv], dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)


def read_features(path: Path) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        feature_names = [name for name in fieldnames if name not in {"sample_id", "split"}]
        rows = list(reader)
    sample_ids = [row["sample_id"] for row in rows]
    split = [row["split"] for row in rows]
    matrix = np.asarray(
        [[float(row[name]) if row[name] not in {"", "nan", "NaN"} else math.nan for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    return feature_names, matrix, sample_ids, split


def feature_indices(feature_names: list[str], prefixes: tuple[str, ...]) -> list[int]:
    idx = [i for i, name in enumerate(feature_names) if any(name.startswith(prefix) for prefix in prefixes)]
    if not idx:
        raise RuntimeError(f"empty feature set for prefixes: {prefixes}")
    return idx


def fit_feature_transform(features: np.ndarray, train_idx: np.ndarray) -> FeatureTransform:
    train = np.asarray(features[train_idx], dtype=np.float64).copy()
    train[~np.isfinite(train)] = np.nan
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    bad = ~np.isfinite(train)
    if np.any(bad):
        train[bad] = np.take(median, np.where(bad)[1])
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1.0e-12, 1.0, std)
    return FeatureTransform(median.astype(np.float64), mean.astype(np.float64), std.astype(np.float64))


def feature_matrix(raw: np.ndarray, train_idx: np.ndarray, idx: list[int]) -> tuple[np.ndarray, FeatureTransform]:
    selected = raw[:, idx]
    transform = fit_feature_transform(selected, train_idx)
    scaled = transform.transform(selected)
    if not np.isfinite(scaled).all():
        raise RuntimeError("scaled feature matrix contains non-finite values")
    return scaled, transform


def predict_norm(model: nn.Module, x: np.ndarray, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(features, dtype=torch.float32)).cpu().numpy().astype(np.float32)


def loss_fn(pred: torch.Tensor, target: torch.Tensor, curvature_weight: float) -> torch.Tensor:
    dim = torch.nn.functional.smooth_l1_loss(pred[:, :3], target[:, :3])
    curv = torch.nn.functional.smooth_l1_loss(pred[:, 3:], target[:, 3:])
    return dim + float(curvature_weight) * curv


def norm_components(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.abs(y_true - y_pred)
    return {
        "total": float(err.mean()),
        "dimension": float(err[:, :3].mean()),
        "curvature": float(err[:, 3:].mean()),
    }


def epoch_score(comp: dict[str, float]) -> float:
    return comp["total"] + 0.75 * comp["curvature"] + 0.20 * comp["dimension"]


def train_candidate(
    config: CandidateConfig,
    seed: int,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    feature_norm: np.ndarray,
    splits: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    set_seed(seed)
    model = CurvatureFusionRegressor(feature_norm.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = splits["train"]
    train_ds = TensorDataset(
        torch.as_tensor(x_norm[train_idx], dtype=torch.float32),
        torch.as_tensor(feature_norm[train_idx], dtype=torch.float32),
        torch.as_tensor(y_norm[train_idx], dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = math.inf
    min_train = math.inf
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, fb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, fb)
            loss = loss_fn(pred, yb, config.curvature_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        pred_train = predict_norm(model, x_norm[train_idx], feature_norm[train_idx])
        pred_val = predict_norm(model, x_norm[splits["val"]], feature_norm[splits["val"]])
        train_comp = norm_components(y_norm[train_idx], pred_train)
        val_comp = norm_components(y_norm[splits["val"]], pred_val)
        val_score = epoch_score(val_comp)
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        min_train = min(min_train, train_comp["total"])
        epoch_rows.append(
            {
                "variant": config.name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_total_mae": train_comp["total"],
                "val_total_mae": val_comp["total"],
                "train_dimension_mae_norm": train_comp["dimension"],
                "val_dimension_mae_norm": val_comp["dimension"],
                "train_curvature_mae_norm": train_comp["curvature"],
                "val_curvature_mae_norm": val_comp["curvature"],
                "val_epoch_selection_score": val_score,
            }
        )
    if best_state is None:
        raise RuntimeError(f"no best validation state for {config.name}")
    model.load_state_dict(best_state)
    pred_all = predict_norm(model, x_norm, feature_norm)
    return {
        "variant": config.name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_epoch_score": best_score,
        "min_train_normalized_param_mae": min_train,
        "pred_norm": pred_all,
        "epoch_rows": epoch_rows,
    }


def evaluate_subset(dataset: Any, pred_raw: np.ndarray, stats: dict[str, np.ndarray], indices: np.ndarray) -> list[dict[str, Any]]:
    pred_params, clipped = clip_params_to_train_bounds(np.asarray(pred_raw, dtype=np.float32), dataset)
    pred_norm = (pred_params - stats["y_mean"]) / stats["y_std"]
    true_norm = (dataset.rbc_params - stats["y_mean"]) / stats["y_std"]
    rows: list[dict[str, Any]] = []
    clip_fraction = float(np.mean(clipped[indices])) if len(indices) else 0.0
    for idx in indices:
        pred_mask = projected_mask_from_params(pred_params[idx], dataset.profile_pose[idx])
        mask_row = mask_metrics(pred_mask, dataset.projected_mask_2d[idx])
        pred_depth = depth_grid_from_params(pred_params[idx])
        depth_rmse = float(np.sqrt(np.mean((pred_depth - dataset.profile_depth_grid_m[idx]) ** 2)))
        true_volume = float(dataset.profile_depth_map_xy_m[idx].sum())
        pred_volume = float(depth_map_from_params(pred_params[idx], dataset.profile_pose[idx]).sum())
        volume_error = 0.0 if abs(true_volume) < 1.0e-12 else abs(pred_volume - true_volume) / abs(true_volume)
        param_abs = np.abs(pred_params[idx] - dataset.rbc_params[idx])
        param_norm_abs = np.abs(pred_norm[idx] - true_norm[idx])
        rows.append(
            {
                "sample_id": str(dataset.sample_ids[idx]),
                "split": str(dataset.split[idx]),
                "curvature_template": str(dataset.curvature_template[idx]),
                "depth_bin": str(dataset.depth_bin[idx]),
                "aspect_bin": str(dataset.aspect_bin[idx]),
                "size_bin": str(dataset.size_bin[idx]),
                "clip_applied": bool(clipped[idx]),
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
                "volume_proxy_rel_error": float(volume_error),
            }
        )
    return rows


def read_reference_profile() -> list[dict[str, Any]]:
    with REFERENCE_PROFILE.open(newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f) if str(row.get("selected_seed", "")).lower() == "true"]
    for row in rows:
        row["variant"] = "C0_reference_20_77"
        row["seed"] = 42
        row["feature_set"] = "delta_b_only"
        row["selected_by_validation"] = False
        row["L_mae_m"] = float(row["L_mae_mm"]) / 1000.0
        row["W_mae_m"] = float(row["W_mae_mm"]) / 1000.0
        row["D_mae_m"] = float(row["D_mae_mm"]) / 1000.0
        row["projected_mask_area_error"] = row.get("projected_mask_area_error", 0.0)
        row["projected_mask_center_error_px"] = row.get("projected_mask_center_error_px", 0.0)
        row["volume_proxy_rel_error"] = row.get("volume_proxy_rel_error", 0.0)
        row["clip_applied"] = 1.0 if str(row.get("clip_applied", "")).lower() == "true" else 0.0
    return rows


def aggregate_rows(rows: list[dict[str, Any]], variant: str, seed: int, config: CandidateConfig | None, selected: bool, eligible: bool, split: str, score: float | str, notes: str = "", test_final: bool | str = "") -> dict[str, Any]:
    agg = aggregate_prediction_rows(rows, variant, split)
    return {
        "variant": variant,
        "seed": seed,
        "feature_set": "delta_b_only" if config is None else config.feature_set,
        "feature_count": "" if config is None else notes.split("feature_count=")[-1].split(";")[0] if "feature_count=" in notes else "",
        "curvature_weight": "" if config is None else config.curvature_weight,
        "selected_by_validation": selected,
        "eligible_for_multiseed": eligible,
        "split": split,
        "sample_count": agg.get("sample_count", 0),
        "selection_score": score,
        "test_final_only": test_final,
        "normalized_param_mae": agg.get("normalized_param_mae_mean_mean", math.nan),
        "dimension_mae_norm": agg.get("dimension_param_mae_norm_mean", math.nan),
        "curvature_mae_norm": agg.get("curvature_param_mae_norm_mean", math.nan),
        "L_mae_mm": agg.get("L_mae_mm_mean", math.nan),
        "W_mae_mm": agg.get("W_mae_mm_mean", math.nan),
        "D_mae_mm": agg.get("D_mae_mm_mean", math.nan),
        "wLD_abs_error": agg.get("wLD_abs_error_mean", math.nan),
        "wWD_abs_error": agg.get("wWD_abs_error_mean", math.nan),
        "wLW_abs_error": agg.get("wLW_abs_error_mean", math.nan),
        "curvature_mae": agg.get("curvature_mae_mean_mean", math.nan),
        "projected_mask_iou": agg.get("projected_mask_iou_mean", math.nan),
        "projected_mask_dice": agg.get("projected_mask_dice_mean", math.nan),
        "profile_depth_rmse_m": agg.get("profile_depth_rmse_m_mean", math.nan),
        "curvature_delta_vs_20_77": agg.get("curvature_mae_mean_mean", math.nan) - REF_NEURAL["curvature"],
        "total_delta_vs_20_77": agg.get("normalized_param_mae_mean_mean", math.nan) - REF_NEURAL["total"],
        "dice_delta_vs_20_77": agg.get("projected_mask_dice_mean", math.nan) - REF_NEURAL["dice"],
        "notes": notes,
    }


def group_rows(rows: list[dict[str, Any]], variant: str, seed: int, feature_set: str, selected: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        base = [row for row in rows if row["split"] == split]
        for field in ("curvature_template", "depth_bin", "aspect_bin"):
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
                        "feature_set": feature_set,
                        "selected_by_validation": selected,
                        "split": split,
                        "group_field": field,
                        "group_value": value,
                        "sample_count": len(subset),
                        "normalized_param_mae": avg("normalized_param_mae_mean"),
                        "L_mae_mm": avg("L_mae_mm"),
                        "W_mae_mm": avg("W_mae_mm"),
                        "D_mae_mm": avg("D_mae_mm"),
                        "curvature_mae": avg("curvature_mae_mean"),
                        "wLD_abs_error": avg("wLD_abs_error"),
                        "wWD_abs_error": avg("wWD_abs_error"),
                        "wLW_abs_error": avg("wLW_abs_error"),
                        "projected_mask_iou": avg("projected_mask_iou"),
                        "projected_mask_dice": avg("projected_mask_dice"),
                        "profile_depth_rmse_m": avg("profile_depth_rmse_m"),
                    }
                )
    return out


def candidate_selection_score(row: dict[str, Any]) -> float:
    return (
        float(row["normalized_param_mae"])
        + 0.75 * float(row["curvature_mae"])
        + 0.20 * float(row["dimension_mae_norm"])
        + 0.20 * max(0.0, REF_NEURAL["dice"] - float(row["projected_mask_dice"]))
    )


def reference_val_metrics() -> dict[str, float]:
    rows = read_reference_profile()
    val = aggregate_rows(rows, "C0_reference_20_77", 42, None, False, False, "val", "", "reference")
    return {
        "curvature": float(val["curvature_mae"]),
        "dimension": float(val["dimension_mae_norm"]),
        "dice": float(val["projected_mask_dice"]),
        "total": float(val["normalized_param_mae"]),
    }


def candidate_configs() -> list[CandidateConfig]:
    return [
        CandidateConfig("H1_curv_fusion_F1F2_w0p5", "FS_F1F2_curvature_only", ("F1__", "F2__"), 0.5),
        CandidateConfig("H2_curv_fusion_F0F1F2_w0p5", "FS_basic_physical", ("F0__", "F1__", "F2__"), 0.5),
        CandidateConfig("H3_curv_fusion_F0F1F2_w1p0", "FS_basic_physical", ("F0__", "F1__", "F2__"), 1.0),
    ]


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics, args.group_summary, args.profile_metrics], args.overwrite)
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    stats = train_normalization(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    feature_names, raw_features, sample_ids, feature_split = read_features(args.features)
    if sample_ids != [str(x) for x in dataset.sample_ids]:
        raise RuntimeError("feature CSV sample_id order does not match dataset")
    if feature_split != [str(x) for x in dataset.split]:
        raise RuntimeError("feature CSV split order does not match dataset")
    if not all(name.startswith(("F0__", "F1__", "F2__", "F3__", "F4__", "F5__")) for name in feature_names):
        raise RuntimeError("feature CSV contains non-allowlisted feature columns")

    ref_val = reference_val_metrics()
    trained: list[tuple[CandidateConfig, int, dict[str, Any], np.ndarray, int]] = []
    metric_rows: list[dict[str, Any]] = []
    profile_rows_out: list[dict[str, Any]] = []
    group_rows_out: list[dict[str, Any]] = []
    val_rows_for_selection: list[dict[str, Any]] = []
    for config in candidate_configs():
        idx = feature_indices(feature_names, config.prefixes)
        features_norm, _transform = feature_matrix(raw_features, splits["train"], idx)
        trained_out = train_candidate(config, args.seed, x_norm, y_norm, features_norm, splits, args.epochs, args.batch_size, args.lr, args.weight_decay)
        trained.append((config, args.seed, trained_out, features_norm, len(idx)))
        pred_raw = denormalize_y(trained_out["pred_norm"], stats)
        train_val_indices = np.concatenate([splits["train"], splits["val"]])
        rows = evaluate_subset(dataset, pred_raw, stats, train_val_indices)
        for row in rows:
            row["variant"] = config.name
            row["seed"] = args.seed
            row["feature_set"] = config.feature_set
            row["selected_by_validation"] = False
        for split in ("train", "val"):
            agg = aggregate_rows(rows, config.name, args.seed, config, False, False, split, "", f"feature_count={len(idx)}; validation-screen only")
            if split == "val":
                score = candidate_selection_score(agg)
                agg["selection_score"] = score
                val_rows_for_selection.append(agg)
            metric_rows.append(agg)
        profile_rows_out.extend(rows)
        group_rows_out.extend(group_rows(rows, config.name, args.seed, config.feature_set, False))

    selected_val = min(val_rows_for_selection, key=lambda row: float(row["selection_score"]))
    selected_name = str(selected_val["variant"])
    selected_tuple = [item for item in trained if item[0].name == selected_name][0]
    selected_config, selected_seed, selected_out, _features_norm, selected_feature_count = selected_tuple
    eligible = (
        float(selected_val["curvature_mae"]) < ref_val["curvature"]
        and float(selected_val["dimension_mae_norm"]) <= ref_val["dimension"] + 0.15
        and float(selected_val["projected_mask_dice"]) >= ref_val["dice"] - 0.02
    )

    selected_pred_raw = denormalize_y(selected_out["pred_norm"], stats)
    selected_rows = evaluate_param_predictions(dataset, selected_pred_raw, stats)
    for row in selected_rows:
        row["variant"] = selected_config.name
        row["seed"] = selected_seed
        row["feature_set"] = selected_config.feature_set
        row["selected_by_validation"] = True
    selected_test = aggregate_rows(
        selected_rows,
        selected_config.name,
        selected_seed,
        selected_config,
        True,
        eligible,
        "test",
        "",
        f"feature_count={selected_feature_count}; validation-selected candidate; test final only for screen",
        True,
    )
    metric_rows.append(selected_test)
    selected_group = group_rows(selected_rows, selected_config.name, selected_seed, selected_config.feature_set, True)
    group_rows_out.extend(selected_group)
    profile_rows_out.extend(selected_rows)

    for row in metric_rows:
        if row["variant"] == selected_name:
            row["selected_by_validation"] = True
            row["eligible_for_multiseed"] = eligible
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_rows_out, GROUP_FIELDS)
    write_csv(args.profile_metrics, profile_rows_out, PROFILE_FIELDS)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 feature-fusion candidate screen summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"seed: {args.seed}",
                "scope: bounded feature-fusion neural diagnostic; no COMSOL, no data generation, no NPZ modification, no baseline.",
                "feature_input_boundary: only F0__..F5__ columns from the 20.80 delta_b-derived feature CSV; sample_id/split used only for joining and split indices.",
                "candidate_selection: validation-only. Test metrics are written only for the validation-selected candidate.",
                f"candidates: {', '.join(config.name for config in candidate_configs())}",
                f"selected_candidate: {selected_name}",
                f"selected_feature_set: {selected_config.feature_set}",
                f"selected_feature_count: {selected_feature_count}",
                f"selected_val_selection_score: {float(selected_val['selection_score']):.6f}",
                f"selected_val_total_mae: {float(selected_val['normalized_param_mae']):.6f}",
                f"selected_val_curvature_mae: {float(selected_val['curvature_mae']):.6f}",
                f"selected_val_dimension_mae_norm: {float(selected_val['dimension_mae_norm']):.6f}",
                f"selected_val_projected_mask_dice: {float(selected_val['projected_mask_dice']):.6f}",
                f"reference_20_77_val_curvature_mae: {ref_val['curvature']:.6f}",
                f"reference_20_77_val_dimension_mae_norm: {ref_val['dimension']:.6f}",
                f"reference_20_77_val_projected_mask_dice: {ref_val['dice']:.6f}",
                f"eligible_for_multiseed: {eligible}",
                "",
                f"selected_test_total_mae: {float(selected_test['normalized_param_mae']):.6f}",
                f"selected_test_LWD_mae_mm: {float(selected_test['L_mae_mm']):.6f}/{float(selected_test['W_mae_mm']):.6f}/{float(selected_test['D_mae_mm']):.6f}",
                f"selected_test_curvature_mae: {float(selected_test['curvature_mae']):.6f}",
                f"selected_test_wLD_wWD_wLW: {float(selected_test['wLD_abs_error']):.6f}/{float(selected_test['wWD_abs_error']):.6f}/{float(selected_test['wLW_abs_error']):.6f}",
                f"selected_test_projected_mask_iou_dice: {float(selected_test['projected_mask_iou']):.6f}/{float(selected_test['projected_mask_dice']):.6f}",
                f"selected_test_profile_depth_rmse_m: {float(selected_test['profile_depth_rmse_m']):.9f}",
                f"test_curvature_delta_vs_20_77: {float(selected_test['curvature_delta_vs_20_77']):.6f}",
                f"test_total_delta_vs_20_77: {float(selected_test['total_delta_vs_20_77']):.6f}",
                f"test_dice_delta_vs_20_77: {float(selected_test['dice_delta_vs_20_77']):.6f}",
                "stage_C_gate: enter Stage D only if eligible_for_multiseed is True.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--features", type=Path, default=FEATURES)
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
