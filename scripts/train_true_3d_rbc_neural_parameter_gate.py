#!/usr/bin/env python
"""Neural Bx/By/Bz -> RBC parameter training gate.

The model consumes only delta_b channels. Labels and metadata are used only for
supervision, validation selection, and metrics. Dataset loading is gated by the
explicit COMSOL_DATA_REGISTRY + manifest loader.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from load_true_3d_rbc_pilot_dataset import (
    DATASET_ID,
    PARAM_NAMES,
    ROOT,
    aggregate_prediction_rows,
    check_no_overwrite,
    denormalize_y,
    evaluate_param_predictions,
    load_dataset,
    normalize_x,
    normalize_y,
    run_name_for_dataset,
    split_indices,
    train_normalization,
    write_csv,
)


SUMMARY_PATH = ROOT / "results/summaries/true_3d_rbc_neural_training_gate_summary.txt"
DECISION_SUMMARY_PATH = ROOT / "results/summaries/true_3d_rbc_training_gate_decision_summary.txt"
SEED_SUMMARY_PATH = ROOT / "results/metrics/true_3d_rbc_neural_training_gate_seed_summary.csv"
METRICS_PATH = ROOT / "results/metrics/true_3d_rbc_neural_training_gate_metrics.csv"
EPOCH_LOG_PATH = ROOT / "results/metrics/true_3d_rbc_neural_training_gate_epoch_log.csv"
GROUP_PATH = ROOT / "results/metrics/true_3d_rbc_neural_training_gate_group_summary.csv"
PROFILE_PATH = ROOT / "results/metrics/true_3d_rbc_neural_training_gate_profile_metrics.csv"
DECISION_MATRIX_PATH = ROOT / "results/metrics/true_3d_rbc_training_gate_decision_matrix.csv"
FEATURE_METRICS_PATH = ROOT / "results/metrics/true_3d_rbc_feature_baseline_metrics.csv"
V1_SEED_SUMMARY_PATH = ROOT / "results/metrics/true_3d_rbc_neural_training_gate_seed_summary.csv"
V1_FEATURE_METRICS_PATH = ROOT / "results/metrics/true_3d_rbc_feature_baseline_metrics.csv"
V2_SEED_SUMMARY_PATH = ROOT / "results/metrics/true_3d_rbc_v2_120_neural_training_gate_seed_summary.csv"
V2_FEATURE_METRICS_PATH = ROOT / "results/metrics/true_3d_rbc_v2_120_feature_baseline_metrics.csv"
COMPARISON_PATH = ROOT / "results/metrics/true_3d_rbc_v2_120_vs_v1_56_comparison.csv"
V3_V2_COMPARISON_PATH = ROOT / "results/metrics/true_3d_rbc_v3_240_vs_v2_120_comparison.csv"

PARAM_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=torch.float32)

SEED_FIELDS = [
    "seed",
    "selected_seed",
    "best_epoch",
    "best_val_selection_metric",
    "min_train_epoch",
    "min_train_normalized_param_mae",
    "val_normalized_at_min_train",
    "train_normalized_param_mae",
    "val_normalized_param_mae",
    "test_normalized_param_mae",
    "train_dimension_mae_norm",
    "val_dimension_mae_norm",
    "test_dimension_mae_norm",
    "train_curvature_mae_norm",
    "val_curvature_mae_norm",
    "test_curvature_mae_norm",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_curvature_mae",
    "test_projected_mask_iou",
    "test_projected_mask_dice",
    "test_profile_depth_rmse_m",
    "can_fit_train",
    "beats_mean_baseline_test",
    "beats_feature_baseline_test",
]

METRIC_FIELDS = [
    "model",
    "seed",
    "selected_seed",
    "split",
    "param",
    "sample_count",
    "normalized_mae",
    "physical_mae",
    "physical_mae_mm",
    "relative_mae",
    "r2",
]

EPOCH_FIELDS = [
    "seed",
    "epoch",
    "train_loss",
    "train_selection_metric",
    "val_selection_metric",
    "train_normalized_param_mae",
    "val_normalized_param_mae",
    "train_dimension_mae_norm",
    "val_dimension_mae_norm",
    "train_curvature_mae_norm",
    "val_curvature_mae_norm",
]

GROUP_FIELDS = [
    "seed",
    "selected_seed",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "normalized_param_mae_mean",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "curvature_mae_mean",
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
]

PROFILE_FIELDS = [
    "seed",
    "selected_seed",
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
    "volume_proxy_rel_error",
]

DECISION_FIELDS = ["question", "answer", "evidence", "decision"]
COMPARISON_FIELDS = ["metric", "reference_label", "current_label", "reference_value", "current_value", "delta", "improved", "notes"]


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)


def selection_components(y_true_norm: np.ndarray, y_pred_norm: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_true_norm - y_pred_norm)
    return {
        "normalized_param_mae": float(abs_err.mean()),
        "dimension_mae_norm": float(abs_err[:, :3].mean()),
        "curvature_mae_norm": float(abs_err[:, 3:].mean()),
    }


def selection_metric(components: dict[str, float]) -> float:
    return components["normalized_param_mae"] + 0.25 * components["dimension_mae_norm"] + 0.10 * components["curvature_mae_norm"]


def predict_norm(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
    return pred.astype(np.float32)


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = PARAM_WEIGHTS.to(pred.device)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights).mean()


def train_one_seed(seed: int, x_norm: np.ndarray, y_norm: np.ndarray, splits: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    set_seed(seed)
    model = RBCConvRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_idx = splits["train"]
    train_ds = TensorDataset(torch.as_tensor(x_norm[train_idx], dtype=torch.float32), torch.as_tensor(y_norm[train_idx], dtype=torch.float32))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_score = math.inf
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = weighted_smooth_l1(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        pred_train = predict_norm(model, x_norm[train_idx])
        pred_val = predict_norm(model, x_norm[splits["val"]])
        train_comp = selection_components(y_norm[train_idx], pred_train)
        val_comp = selection_components(y_norm[splits["val"]], pred_val)
        train_score = selection_metric(train_comp)
        val_score = selection_metric(val_comp)
        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        epoch_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_selection_metric": train_score,
                "val_selection_metric": val_score,
                "train_normalized_param_mae": train_comp["normalized_param_mae"],
                "val_normalized_param_mae": val_comp["normalized_param_mae"],
                "train_dimension_mae_norm": train_comp["dimension_mae_norm"],
                "val_dimension_mae_norm": val_comp["dimension_mae_norm"],
                "train_curvature_mae_norm": train_comp["curvature_mae_norm"],
                "val_curvature_mae_norm": val_comp["curvature_mae_norm"],
            }
        )
    if best_state is None:
        raise RuntimeError("no validation state selected")
    min_train_row = min(epoch_rows, key=lambda row: float(row["train_normalized_param_mae"]))
    model.load_state_dict(best_state)
    pred_norm_all = predict_norm(model, x_norm)
    return {
        "seed": seed,
        "model": model,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "min_train_epoch": int(min_train_row["epoch"]),
        "min_train_normalized_param_mae": float(min_train_row["train_normalized_param_mae"]),
        "val_normalized_at_min_train": float(min_train_row["val_normalized_param_mae"]),
        "pred_norm": pred_norm_all,
        "epoch_rows": epoch_rows,
    }


def compute_param_metric_rows(seed: int, selected_seed: bool, y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_norm: np.ndarray, y_pred_norm: np.ndarray, splits: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, idx in splits.items():
        true_raw = y_true_raw[idx]
        pred_raw = y_pred_raw[idx]
        true_norm = y_true_norm[idx]
        pred_norm = y_pred_norm[idx]
        for pidx, param_name in enumerate(PARAM_NAMES):
            abs_raw = np.abs(pred_raw[:, pidx] - true_raw[:, pidx])
            abs_norm = np.abs(pred_norm[:, pidx] - true_norm[:, pidx])
            denom = np.maximum(np.abs(true_raw[:, pidx]), 1.0e-12)
            ss_res = float(np.sum((true_raw[:, pidx] - pred_raw[:, pidx]) ** 2))
            ss_tot = float(np.sum((true_raw[:, pidx] - np.mean(true_raw[:, pidx])) ** 2))
            r2 = float("nan") if ss_tot <= 1.0e-20 else 1.0 - ss_res / ss_tot
            rows.append(
                {
                    "model": "conv1d_rbc_param_gate",
                    "seed": seed,
                    "selected_seed": selected_seed,
                    "split": split_name,
                    "param": param_name,
                    "sample_count": len(idx),
                    "normalized_mae": float(abs_norm.mean()),
                    "physical_mae": float(abs_raw.mean()),
                    "physical_mae_mm": float(abs_raw.mean() * 1000.0) if pidx < 3 else "",
                    "relative_mae": float(np.mean(abs_raw / denom)) if pidx < 3 else "",
                    "r2": r2,
                }
            )
        abs_norm_all = np.abs(pred_norm - true_norm)
        abs_raw_all = np.abs(pred_raw - true_raw)
        rows.append(
            {
                "model": "conv1d_rbc_param_gate",
                "seed": seed,
                "selected_seed": selected_seed,
                "split": split_name,
                "param": "ALL",
                "sample_count": len(idx),
                "normalized_mae": float(abs_norm_all.mean()),
                "physical_mae": float(abs_raw_all.mean()),
                "physical_mae_mm": "",
                "relative_mae": "",
                "r2": "",
            }
        )
        rows.append(
            {
                "model": "conv1d_rbc_param_gate",
                "seed": seed,
                "selected_seed": selected_seed,
                "split": split_name,
                "param": "DIMENSION_MEAN",
                "sample_count": len(idx),
                "normalized_mae": float(abs_norm_all[:, :3].mean()),
                "physical_mae": float(abs_raw_all[:, :3].mean()),
                "physical_mae_mm": float(abs_raw_all[:, :3].mean() * 1000.0),
                "relative_mae": "",
                "r2": "",
            }
        )
        rows.append(
            {
                "model": "conv1d_rbc_param_gate",
                "seed": seed,
                "selected_seed": selected_seed,
                "split": split_name,
                "param": "CURVATURE_MEAN",
                "sample_count": len(idx),
                "normalized_mae": float(abs_norm_all[:, 3:].mean()),
                "physical_mae": float(abs_raw_all[:, 3:].mean()),
                "physical_mae_mm": "",
                "relative_mae": "",
                "r2": "",
            }
        )
    return rows


def summarize_profile_rows(seed: int, selected_seed: bool, profile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in profile_rows if row["split"] == split_name]
        for group_field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
            for group_value in sorted({str(row[group_field]) for row in split_rows}):
                subset = [row for row in split_rows if str(row[group_field]) == group_value]
                if not subset:
                    continue
                out.append(
                    {
                        "seed": seed,
                        "selected_seed": selected_seed,
                        "split": split_name,
                        "group_field": group_field,
                        "group_value": group_value,
                        "sample_count": len(subset),
                        "normalized_param_mae_mean": float(np.mean([row["normalized_param_mae_mean"] for row in subset])),
                        "L_mae_mm": float(np.mean([row["L_mae_mm"] for row in subset])),
                        "W_mae_mm": float(np.mean([row["W_mae_mm"] for row in subset])),
                        "D_mae_mm": float(np.mean([row["D_mae_mm"] for row in subset])),
                        "curvature_mae_mean": float(np.mean([row["curvature_mae_mean"] for row in subset])),
                        "projected_mask_iou": float(np.mean([row["projected_mask_iou"] for row in subset])),
                        "projected_mask_dice": float(np.mean([row["projected_mask_dice"] for row in subset])),
                        "profile_depth_rmse_m": float(np.mean([row["profile_depth_rmse_m"] for row in subset])),
                    }
                )
    return out


def read_feature_baseline_scores(path: Path) -> dict[str, float]:
    scores = {"mean_test": math.nan, "feature_test": math.nan}
    if not path.exists():
        return scores
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "test":
                continue
            value = float(row.get("normalized_param_mae_mean_mean") or "nan")
            if row.get("model") == "mean_train_target":
                scores["mean_test"] = value
            if str(row.get("selected_by_val")).lower() == "true":
                scores["feature_test"] = value
    return scores


def read_selected_v1_neural_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("selected_seed")).lower() != "true":
                continue
            for key in (
                "test_normalized_param_mae",
                "test_L_mae_mm",
                "test_W_mae_mm",
                "test_D_mae_mm",
                "test_curvature_mae",
                "test_projected_mask_iou",
                "test_projected_mask_dice",
                "test_profile_depth_rmse_m",
            ):
                out[key] = float(row.get(key) or "nan")
            break
    return out


def comparison_row(metric: str, reference_label: str, reference_value: float, current_label: str, current_value: float, lower_is_better: bool, notes: str = "") -> dict[str, Any]:
    delta = current_value - reference_value if not (math.isnan(reference_value) or math.isnan(current_value)) else math.nan
    if math.isnan(delta):
        improved = ""
    elif lower_is_better:
        improved = delta < 0.0
    else:
        improved = delta > 0.0
    return {
        "metric": metric,
        "reference_label": reference_label,
        "current_label": current_label,
        "reference_value": reference_value,
        "current_value": current_value,
        "delta": delta,
        "improved": improved,
        "notes": notes,
    }


def get_seed_split_aggregate(profile_rows: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    return aggregate_prediction_rows(profile_rows, "conv1d_rbc_param_gate", split_name)


def run(args: argparse.Namespace) -> int:
    output_paths = [args.summary, args.seed_summary, args.metrics, args.epoch_log, args.group_summary, args.profile_metrics, args.decision_summary, args.decision_matrix]
    if args.comparison_output is not None:
        output_paths.append(args.comparison_output)
    if args.v2_comparison_output is not None:
        output_paths.append(args.v2_comparison_output)
    check_no_overwrite(output_paths, args.overwrite)
    dataset = load_dataset(args.dataset_id)
    stats = train_normalization(dataset)
    splits = split_indices(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    feature_scores = read_feature_baseline_scores(args.feature_metrics)

    seed_outputs = [train_one_seed(seed, x_norm, y_norm, splits, args) for seed in args.seeds]
    selected = min(seed_outputs, key=lambda item: item["best_val_score"])
    selected_seed = int(selected["seed"])

    all_epoch_rows: list[dict[str, Any]] = []
    all_seed_summary: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    all_profile_rows: list[dict[str, Any]] = []
    selected_profile_rows: list[dict[str, Any]] = []
    selected_raw_pred: np.ndarray | None = None
    selected_param_metric_rows: list[dict[str, Any]] = []

    for seed_out in seed_outputs:
        seed = int(seed_out["seed"])
        is_selected = seed == selected_seed
        pred_raw = denormalize_y(seed_out["pred_norm"], stats)
        profile_rows = evaluate_param_predictions(dataset, pred_raw, stats)
        for row in profile_rows:
            row["seed"] = seed
            row["selected_seed"] = is_selected
        all_profile_rows.extend(profile_rows)
        all_group_rows.extend(summarize_profile_rows(seed, is_selected, profile_rows))
        metric_rows = compute_param_metric_rows(seed, is_selected, dataset.rbc_params, pred_raw, y_norm, seed_out["pred_norm"], splits)
        all_metric_rows.extend(metric_rows)
        if is_selected:
            selected_profile_rows = profile_rows
            selected_raw_pred = pred_raw
            selected_param_metric_rows = metric_rows
        all_epoch_rows.extend(seed_out["epoch_rows"])
        train_agg = get_seed_split_aggregate(profile_rows, "train")
        val_agg = get_seed_split_aggregate(profile_rows, "val")
        test_agg = get_seed_split_aggregate(profile_rows, "test")
        all_seed_summary.append(
            {
                "seed": seed,
                "selected_seed": is_selected,
                "best_epoch": seed_out["best_epoch"],
                "best_val_selection_metric": seed_out["best_val_score"],
                "min_train_epoch": seed_out["min_train_epoch"],
                "min_train_normalized_param_mae": seed_out["min_train_normalized_param_mae"],
                "val_normalized_at_min_train": seed_out["val_normalized_at_min_train"],
                "train_normalized_param_mae": train_agg["normalized_param_mae_mean_mean"],
                "val_normalized_param_mae": val_agg["normalized_param_mae_mean_mean"],
                "test_normalized_param_mae": test_agg["normalized_param_mae_mean_mean"],
                "train_dimension_mae_norm": train_agg["dimension_param_mae_norm_mean"],
                "val_dimension_mae_norm": val_agg["dimension_param_mae_norm_mean"],
                "test_dimension_mae_norm": test_agg["dimension_param_mae_norm_mean"],
                "train_curvature_mae_norm": train_agg["curvature_param_mae_norm_mean"],
                "val_curvature_mae_norm": val_agg["curvature_param_mae_norm_mean"],
                "test_curvature_mae_norm": test_agg["curvature_param_mae_norm_mean"],
                "test_L_mae_mm": test_agg["L_mae_mm_mean"],
                "test_W_mae_mm": test_agg["W_mae_mm_mean"],
                "test_D_mae_mm": test_agg["D_mae_mm_mean"],
                "test_curvature_mae": test_agg["curvature_mae_mean_mean"],
                "test_projected_mask_iou": test_agg["projected_mask_iou_mean"],
                "test_projected_mask_dice": test_agg["projected_mask_dice_mean"],
                "test_profile_depth_rmse_m": test_agg["profile_depth_rmse_m_mean"],
                "can_fit_train": seed_out["min_train_normalized_param_mae"] < 0.20,
                "beats_mean_baseline_test": bool(test_agg["normalized_param_mae_mean_mean"] < feature_scores["mean_test"]) if not math.isnan(feature_scores["mean_test"]) else "",
                "beats_feature_baseline_test": bool(test_agg["normalized_param_mae_mean_mean"] < feature_scores["feature_test"]) if not math.isnan(feature_scores["feature_test"]) else "",
            }
        )

    write_csv(args.seed_summary, all_seed_summary, SEED_FIELDS)
    write_csv(args.metrics, all_metric_rows, METRIC_FIELDS)
    write_csv(args.epoch_log, all_epoch_rows, EPOCH_FIELDS)
    write_csv(args.group_summary, all_group_rows, GROUP_FIELDS)
    write_csv(args.profile_metrics, all_profile_rows, PROFILE_FIELDS)

    if selected_raw_pred is None:
        raise RuntimeError("selected prediction missing")
    selected_test_profile = get_seed_split_aggregate(selected_profile_rows, "test")
    selected_train_profile = get_seed_split_aggregate(selected_profile_rows, "train")
    selected_val_profile = get_seed_split_aggregate(selected_profile_rows, "val")
    selected_test_param = [row for row in selected_param_metric_rows if row["split"] == "test" and row["param"] in PARAM_NAMES]
    learnable = [row["param"] for row in selected_test_param if float(row["normalized_mae"]) < 0.75]
    not_learnable = [row["param"] for row in selected_test_param if float(row["normalized_mae"]) >= 0.75]
    beats_mean = bool(selected_test_profile["normalized_param_mae_mean_mean"] < feature_scores["mean_test"]) if not math.isnan(feature_scores["mean_test"]) else False
    beats_feature = bool(selected_test_profile["normalized_param_mae_mean_mean"] < feature_scores["feature_test"]) if not math.isnan(feature_scores["feature_test"]) else False
    min_train_fit = min(float(row["min_train_normalized_param_mae"]) for row in all_seed_summary)
    min_train_seed = [row for row in all_seed_summary if float(row["min_train_normalized_param_mae"]) == min_train_fit][0]["seed"]
    can_fit_train = bool(min_train_fit < 0.20)
    v1_neural = read_selected_v1_neural_summary(args.v1_neural_seed_summary)
    v2_neural = read_selected_v1_neural_summary(args.v2_neural_seed_summary)
    v1_feature_scores = read_feature_baseline_scores(args.v1_feature_metrics)
    v2_feature_scores = read_feature_baseline_scores(args.v2_feature_metrics)
    current_n = len(dataset.sample_ids)
    split_counts = {name: len(idx) for name, idx in splits.items()}
    current_label = "20.77_N240" if "v3_240" in args.run_name else ("20.75_N112" if "v2_120" in args.run_name else args.run_name)
    primary_reference_label = "20.75_N112" if "v3_240" in args.run_name else "20.73_N56"
    primary_neural = v2_neural if primary_reference_label == "20.75_N112" else v1_neural
    primary_features = v2_feature_scores if primary_reference_label == "20.75_N112" else v1_feature_scores
    ref_neural_test = float(primary_neural.get("test_normalized_param_mae", math.nan))
    ref_d_mae = float(primary_neural.get("test_D_mae_mm", math.nan))
    ref_curv_mae = float(primary_neural.get("test_curvature_mae", math.nan))
    ref_dice = float(primary_neural.get("test_projected_mask_dice", math.nan))
    improved_over_reference = (not math.isnan(ref_neural_test)) and selected_test_profile["normalized_param_mae_mean_mean"] < ref_neural_test
    d_improved = (not math.isnan(ref_d_mae)) and selected_test_profile["D_mae_mm_mean"] < ref_d_mae
    curvature_improved = (not math.isnan(ref_curv_mae)) and selected_test_profile["curvature_mae_mean_mean"] < ref_curv_mae
    dice_stable = math.isnan(ref_dice) or selected_test_profile["projected_mask_dice_mean"] >= (ref_dice - 0.02)
    v1_neural_test = float(v1_neural.get("test_normalized_param_mae", math.nan))
    improved_over_v1 = (not math.isnan(v1_neural_test)) and selected_test_profile["normalized_param_mae_mean_mean"] < v1_neural_test
    if "v3_240" in args.run_name and can_fit_train and beats_mean and beats_feature and improved_over_reference and (d_improved or curvature_improved) and dice_stable:
        next_step = "A_formal_true_3D_RBC_benchmark_candidate"
        overall = "v3_240_promising_benchmark_candidate"
        enough = f"N={current_n} is enough for a formal benchmark-candidate pass, but not for automatic baseline replacement."
    elif "v3_240" in args.run_name and can_fit_train and beats_mean and beats_feature and (improved_over_reference or d_improved or curvature_improved):
        next_step = "C_targeted_curvature_depth_topup"
        overall = "v3_240_partially_improved"
        enough = f"N={current_n} improves part of the N=112 signal, but curvature/depth residual weakness still needs targeted follow-up."
    elif can_fit_train and beats_mean and beats_feature and improved_over_reference and selected_test_profile["projected_mask_dice_mean"] >= 0.80:
        next_step = "A_expand_true_3d_RBC_dataset_to_240"
        overall = "v2_120_promising_but_not_baseline"
        enough = f"N={current_n} gives a positive training-gate signal, but it remains too small for baseline claims."
    elif can_fit_train and beats_mean and (improved_over_reference or d_improved or curvature_improved) and dice_stable:
        next_step = "A_expand_true_3d_RBC_dataset"
        overall = "v2_120_partially_improved"
        enough = f"N={current_n} improves at least part of the reference signal, but curvature/depth generalization should be checked before architecture changes."
    elif can_fit_train and not beats_feature:
        next_step = "C_targeted_curvature_depth_topup" if "v3_240" in args.run_name else "A_expand_true_3d_RBC_dataset"
        overall = "v3_240_generalization_limited" if "v3_240" in args.run_name else "v2_120_generalization_limited"
        enough = f"N={current_n} is still not enough to beat the feature comparator; expand or target weak bins before model changes."
    elif not can_fit_train:
        next_step = "D_improve_model_or_schema"
        overall = "train_fit_blocker"
        enough = f"N={current_n} does not yet show train fit with this small Conv1D."
    else:
        next_step = "C_targeted_curvature_depth_topup" if "v3_240" in args.run_name else "A_expand_true_3d_RBC_dataset"
        overall = "partial_promising"
        enough = f"N={current_n} is still only a pilot; expand before baseline claims."

    def make_comparison_rows(reference_label: str, reference_neural: dict[str, float], reference_features: dict[str, float]) -> list[dict[str, Any]]:
        return [
            comparison_row("neural_test_normalized_mae", reference_label, float(reference_neural.get("test_normalized_param_mae", math.nan)), current_label, selected_test_profile["normalized_param_mae_mean_mean"], True, "selected validation checkpoint; lower is better"),
            comparison_row("feature_test_normalized_mae", reference_label, reference_features["feature_test"], current_label, feature_scores["feature_test"], True, "feature baseline selected by validation; lower is better"),
            comparison_row("mean_test_normalized_mae", reference_label, reference_features["mean_test"], current_label, feature_scores["mean_test"], True, "train-target mean predictor; lower is better"),
            comparison_row("L_mae_mm", reference_label, float(reference_neural.get("test_L_mae_mm", math.nan)), current_label, selected_test_profile["L_mae_mm_mean"], True, "neural selected seed test split"),
            comparison_row("W_mae_mm", reference_label, float(reference_neural.get("test_W_mae_mm", math.nan)), current_label, selected_test_profile["W_mae_mm_mean"], True, "neural selected seed test split"),
            comparison_row("D_mae_mm", reference_label, float(reference_neural.get("test_D_mae_mm", math.nan)), current_label, selected_test_profile["D_mae_mm_mean"], True, "D_m boundary learnability proxy"),
            comparison_row("curvature_mae", reference_label, float(reference_neural.get("test_curvature_mae", math.nan)), current_label, selected_test_profile["curvature_mae_mean_mean"], True, "mean absolute error over wLD/wWD/wLW"),
            comparison_row("projected_mask_iou", reference_label, float(reference_neural.get("test_projected_mask_iou", math.nan)), current_label, selected_test_profile["projected_mask_iou_mean"], False, "higher is better"),
            comparison_row("projected_mask_dice", reference_label, float(reference_neural.get("test_projected_mask_dice", math.nan)), current_label, selected_test_profile["projected_mask_dice_mean"], False, "higher is better"),
            comparison_row("profile_depth_rmse_m", reference_label, float(reference_neural.get("test_profile_depth_rmse_m", math.nan)), current_label, selected_test_profile["profile_depth_rmse_m_mean"], True, "lower is better"),
        ]

    comparison_rows = make_comparison_rows("20.73_N56", v1_neural, v1_feature_scores)
    if args.comparison_output is not None:
        write_csv(args.comparison_output, comparison_rows, COMPARISON_FIELDS)
    if args.v2_comparison_output is not None:
        write_csv(args.v2_comparison_output, make_comparison_rows("20.75_N112", v2_neural, v2_feature_scores), COMPARISON_FIELDS)

    decision_rows = [
        {"question": "Can the model fit train samples?", "answer": str(can_fit_train), "evidence": f"min_train_normalized_mae={min_train_fit:.6f}; seed={min_train_seed}", "decision": overall},
        {"question": "Does validation/test beat mean baseline?", "answer": str(beats_mean), "evidence": f"neural_test={selected_test_profile['normalized_param_mae_mean_mean']:.6f}; mean_test={feature_scores['mean_test']:.6f}", "decision": overall},
        {"question": "Does neural beat feature baseline?", "answer": str(beats_feature), "evidence": f"neural_test={selected_test_profile['normalized_param_mae_mean_mean']:.6f}; feature_test={feature_scores['feature_test']:.6f}", "decision": overall},
        {"question": f"Did {current_label} improve over {primary_reference_label}?", "answer": str(improved_over_reference), "evidence": f"current_neural_test={selected_test_profile['normalized_param_mae_mean_mean']:.6f}; reference_neural_test={ref_neural_test:.6f}", "decision": overall},
        {"question": "Did N=240 improve over N=56?", "answer": str(improved_over_v1), "evidence": f"current_neural_test={selected_test_profile['normalized_param_mae_mean_mean']:.6f}; v1_neural_test={v1_neural_test:.6f}", "decision": overall},
        {"question": f"Did D_m improve over {primary_reference_label}?", "answer": str(d_improved), "evidence": f"current_D_mae_mm={selected_test_profile['D_mae_mm_mean']:.6f}; reference_D_mae_mm={ref_d_mae:.6f}", "decision": overall},
        {"question": f"Did curvature improve over {primary_reference_label}?", "answer": str(curvature_improved), "evidence": f"current_curvature_mae={selected_test_profile['curvature_mae_mean_mean']:.6f}; reference_curvature_mae={ref_curv_mae:.6f}", "decision": overall},
        {"question": "Which params are learnable?", "answer": ", ".join(learnable) if learnable else "none", "evidence": "learnable threshold: selected test normalized MAE < 0.75", "decision": overall},
        {"question": "Which params are not learnable?", "answer": ", ".join(not_learnable) if not_learnable else "none", "evidence": "threshold is provisional for training gate only", "decision": overall},
        {"question": f"Is N={current_n} enough?", "answer": enough, "evidence": f"split={split_counts}; selected_seed={selected_seed}", "decision": overall},
        {"question": "Next step", "answer": next_step, "evidence": "This is a training gate, not a baseline update.", "decision": overall},
    ]
    write_csv(args.decision_matrix, decision_rows, DECISION_FIELDS)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                f"{args.run_name} neural training gate summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"input: delta_b only, shape={list(dataset.delta_b.shape)} -> Conv1D channels={list(dataset.x_channels.shape)}",
                "model: small Conv1D encoder + MLP head, output 6 train-normalized RBC params",
                f"seeds: {list(args.seeds)}",
                f"selected_seed_by_validation: {selected_seed}",
                f"selected_best_epoch: {selected['best_epoch']}",
                f"selected_best_val_metric: {selected['best_val_score']:.6f}",
                f"best_train_fit_over_all_epochs: seed={min_train_seed}, normalized_mae={min_train_fit:.6f}",
                f"train_normalized_mae: {selected_train_profile['normalized_param_mae_mean_mean']:.6f}",
                f"val_normalized_mae: {selected_val_profile['normalized_param_mae_mean_mean']:.6f}",
                f"test_normalized_mae: {selected_test_profile['normalized_param_mae_mean_mean']:.6f}",
                f"test_LWD_mae_mm: L={selected_test_profile['L_mae_mm_mean']:.6f}, W={selected_test_profile['W_mae_mm_mean']:.6f}, D={selected_test_profile['D_mae_mm_mean']:.6f}",
                f"test_curvature_mae: {selected_test_profile['curvature_mae_mean_mean']:.6f}",
                f"test_projected_mask: IoU={selected_test_profile['projected_mask_iou_mean']:.6f}, Dice={selected_test_profile['projected_mask_dice_mean']:.6f}",
                f"test_profile_depth_rmse_m: {selected_test_profile['profile_depth_rmse_m_mean']:.9f}",
                f"can_fit_train: {can_fit_train}",
                f"beats_mean_baseline_test: {beats_mean}",
                f"beats_feature_baseline_test: {beats_feature}",
                f"primary_reference: {primary_reference_label}",
                f"improved_over_primary_reference_neural_test_mae: {improved_over_reference}",
                f"improved_over_20_73_neural_test_mae: {improved_over_v1}",
                f"D_m_improved_vs_primary_reference: {d_improved}",
                f"curvature_improved_vs_primary_reference: {curvature_improved}",
                f"projected_mask_dice_stable_vs_primary_reference: {dice_stable}",
                f"learnable_params_provisional: {', '.join(learnable) if learnable else 'none'}",
                f"not_learnable_params_provisional: {', '.join(not_learnable) if not_learnable else 'none'}",
                "Boundary: no COMSOL run, no data generation, no NPZ modification, no checkpoint committed, no baseline update.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.decision_summary.parent.mkdir(parents=True, exist_ok=True)
    args.decision_summary.write_text(
        "\n".join(
            [
                f"{args.run_name} decision summary",
                "",
                f"overall_decision: {overall}",
                f"can_fit_train: {can_fit_train}",
                f"beats_mean_baseline_test: {beats_mean}",
                f"beats_feature_baseline_test: {beats_feature}",
                f"primary_reference: {primary_reference_label}",
                f"improved_over_primary_reference_neural_test_mae: {improved_over_reference}",
                f"improved_over_20_73_neural_test_mae: {improved_over_v1}",
                f"D_m_improved_vs_primary_reference: {d_improved}",
                f"curvature_improved_vs_primary_reference: {curvature_improved}",
                f"learnable_params_provisional: {', '.join(learnable) if learnable else 'none'}",
                f"not_learnable_params_provisional: {', '.join(not_learnable) if not_learnable else 'none'}",
                f"n{current_n}_sufficiency: {enough}",
                f"next_step: {next_step}",
                "",
                "No baseline update is allowed from this gate. Any training/evaluation must continue through explicit dataset_id + manifest + registry loading.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--run-name")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--seed-summary", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--epoch-log", type=Path)
    parser.add_argument("--group-summary", type=Path)
    parser.add_argument("--profile-metrics", type=Path)
    parser.add_argument("--decision-summary", type=Path)
    parser.add_argument("--decision-matrix", type=Path)
    parser.add_argument("--feature-metrics", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--v2-comparison-output", type=Path)
    parser.add_argument("--v1-neural-seed-summary", type=Path, default=V1_SEED_SUMMARY_PATH)
    parser.add_argument("--v1-feature-metrics", type=Path, default=V1_FEATURE_METRICS_PATH)
    parser.add_argument("--v2-neural-seed-summary", type=Path, default=V2_SEED_SUMMARY_PATH)
    parser.add_argument("--v2-feature-metrics", type=Path, default=V2_FEATURE_METRICS_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_name = args.run_name or run_name_for_dataset(args.dataset_id)
    args.run_name = run_name
    legacy = run_name == "true_3d_rbc_training_gate"
    base_name = run_name.removesuffix("_training_gate")
    if args.summary is None:
        args.summary = SUMMARY_PATH if legacy else ROOT / f"results/summaries/{base_name}_neural_training_gate_summary.txt"
    if args.seed_summary is None:
        args.seed_summary = SEED_SUMMARY_PATH if legacy else ROOT / f"results/metrics/{base_name}_neural_training_gate_seed_summary.csv"
    if args.metrics is None:
        args.metrics = METRICS_PATH if legacy else ROOT / f"results/metrics/{base_name}_neural_training_gate_metrics.csv"
    if args.epoch_log is None:
        args.epoch_log = EPOCH_LOG_PATH if legacy else ROOT / f"results/metrics/{base_name}_neural_training_gate_epoch_log.csv"
    if args.group_summary is None:
        args.group_summary = GROUP_PATH if legacy else ROOT / f"results/metrics/{base_name}_neural_training_gate_group_summary.csv"
    if args.profile_metrics is None:
        args.profile_metrics = PROFILE_PATH if legacy else ROOT / f"results/metrics/{base_name}_neural_training_gate_profile_metrics.csv"
    if args.decision_summary is None:
        args.decision_summary = DECISION_SUMMARY_PATH if legacy else ROOT / f"results/summaries/{run_name}_decision_summary.txt"
    if args.decision_matrix is None:
        args.decision_matrix = DECISION_MATRIX_PATH if legacy else ROOT / f"results/metrics/{run_name}_decision_matrix.csv"
    if args.feature_metrics is None:
        args.feature_metrics = FEATURE_METRICS_PATH if legacy else ROOT / f"results/metrics/{base_name}_feature_baseline_metrics.csv"
    if args.comparison_output is None and not legacy:
        args.comparison_output = COMPARISON_PATH if base_name == "true_3d_rbc_v2_120" else ROOT / f"results/metrics/{base_name}_vs_v1_56_comparison.csv"
    if args.v2_comparison_output is None and base_name == "true_3d_rbc_v3_240":
        args.v2_comparison_output = V3_V2_COMPARISON_PATH
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
