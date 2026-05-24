#!/usr/bin/env python
"""20.73 Piao-inspired feature-regression sanity for the true-3D RBC pilot.

This is not a Piao 2019 reproduction. It uses simple Bx/By/Bz signal features
from delta_b only and keeps all dataset access behind the explicit
dataset_id + registry + manifest loader.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    DATASET_ID,
    PARAM_NAMES,
    ROOT,
    aggregate_prediction_rows,
    check_no_overwrite,
    evaluate_param_predictions,
    load_dataset,
    normalize_y,
    split_indices,
    train_normalization,
    write_csv,
)


SUMMARY_PATH = ROOT / "results/summaries/true_3d_rbc_feature_baseline_summary.txt"
METRICS_PATH = ROOT / "results/metrics/true_3d_rbc_feature_baseline_metrics.csv"
GROUP_PATH = ROOT / "results/metrics/true_3d_rbc_feature_baseline_group_summary.csv"

METRIC_FIELDS = [
    "model",
    "split",
    "sample_count",
    "normalized_param_mae_mean_mean",
    "dimension_param_mae_norm_mean",
    "curvature_param_mae_norm_mean",
    "L_mae_mm_mean",
    "W_mae_mm_mean",
    "D_mae_mm_mean",
    "wLD_abs_error_mean",
    "wWD_abs_error_mean",
    "wLW_abs_error_mean",
    "curvature_mae_mean_mean",
    "projected_mask_iou_mean",
    "projected_mask_dice_mean",
    "projected_mask_area_error_mean",
    "profile_depth_rmse_m_mean",
    "volume_proxy_rel_error_mean",
    "clip_applied_mean",
    "selection_metric",
    "selected_by_val",
]

GROUP_FIELDS = [
    "model",
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


def extract_signal_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract deterministic per-channel features from x=(N,9,201)."""
    grad = np.diff(x, axis=2)
    idx = np.linspace(-1.0, 1.0, x.shape[2], dtype=np.float32)
    grad_idx = np.linspace(-1.0, 1.0, grad.shape[2], dtype=np.float32)
    feature_blocks: list[np.ndarray] = []
    names: list[str] = []
    specs = [
        ("max", x.max(axis=2)),
        ("min", x.min(axis=2)),
        ("ptp", np.ptp(x, axis=2)),
        ("argmax_pos", idx[np.argmax(x, axis=2)]),
        ("argmin_pos", idx[np.argmin(x, axis=2)]),
        ("abs_peak", np.max(np.abs(x), axis=2)),
        ("energy", np.mean(x * x, axis=2)),
        ("mean", x.mean(axis=2)),
        ("std", x.std(axis=2)),
        ("integral", np.trapezoid(x, dx=1.0 / max(1, x.shape[2] - 1), axis=2)),
        ("grad_max", np.max(grad, axis=2)),
        ("grad_min", np.min(grad, axis=2)),
        ("grad_abs_peak", np.max(np.abs(grad), axis=2)),
        ("grad_energy", np.mean(grad * grad, axis=2)),
        ("grad_arg_abs_peak", grad_idx[np.argmax(np.abs(grad), axis=2)]),
    ]
    for feature_name, block in specs:
        feature_blocks.append(np.asarray(block, dtype=np.float32))
        names.extend([f"ch{channel}_{feature_name}" for channel in range(x.shape[1])])
    return np.concatenate(feature_blocks, axis=1), names


def fit_feature_scaler(features: np.ndarray, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-12, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def transform_features(features: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return ((features - scaler["mean"]) / scaler["std"]).astype(np.float32)


def normalized_mae(y_true_norm: np.ndarray, y_pred_norm: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true_norm - y_pred_norm)))


def fit_linear_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)
    return coef[:-1].astype(np.float32), coef[-1:].astype(np.float32)


def predict_linear(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weights + bias


def try_svr(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[str, Any, np.ndarray, float] | None:
    try:
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVR
    except Exception:
        return None

    candidates: list[tuple[str, Any, np.ndarray, float]] = []
    for c_value in (1.0, 10.0):
        model = MultiOutputRegressor(SVR(kernel="rbf", C=c_value, gamma="scale", epsilon=0.03))
        model.fit(x_train, y_train)
        pred_val = np.asarray(model.predict(x_val), dtype=np.float32)
        score = normalized_mae(y_val, pred_val)
        candidates.append((f"svr_rbf_C{c_value:g}", model, pred_val, score))
    return min(candidates, key=lambda item: item[3])


def prediction_group_rows(model_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        for group_field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
            for group_value in sorted({str(row[group_field]) for row in split_rows}):
                subset = [row for row in split_rows if str(row[group_field]) == group_value]
                if not subset:
                    continue
                out.append(
                    {
                        "model": model_name,
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


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics, args.group_summary], args.overwrite)
    dataset = load_dataset(args.dataset_id)
    stats = train_normalization(dataset)
    splits = split_indices(dataset)
    train_idx, val_idx = splits["train"], splits["val"]

    raw_features, feature_names = extract_signal_features(dataset.x_channels)
    feature_scaler = fit_feature_scaler(raw_features, train_idx)
    features = transform_features(raw_features, feature_scaler)
    y_norm = normalize_y(dataset, stats)

    predictions: dict[str, np.ndarray] = {}
    train_mean = dataset.rbc_params[train_idx].mean(axis=0, keepdims=True)
    predictions["mean_train_target"] = np.repeat(train_mean, len(dataset.sample_ids), axis=0)

    ridge_candidates: list[tuple[str, np.ndarray, float]] = []
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        weights, bias = fit_linear_ridge(features[train_idx], y_norm[train_idx], alpha)
        pred_val_norm = predict_linear(features[val_idx], weights, bias)
        score = normalized_mae(y_norm[val_idx], pred_val_norm)
        ridge_candidates.append((f"ridge_alpha_{alpha:g}", np.concatenate([weights, bias], axis=0), score))
    selected_ridge_name, selected_ridge_packed, selected_ridge_score = min(ridge_candidates, key=lambda item: item[2])
    weights, bias = selected_ridge_packed[:-1], selected_ridge_packed[-1:]
    predictions[selected_ridge_name] = stats["y_mean"] + predict_linear(features, weights, bias) * stats["y_std"]

    svr_selected = try_svr(features[train_idx], y_norm[train_idx], features[val_idx], y_norm[val_idx])
    if svr_selected is not None:
        svr_name, svr_model, _pred_val, svr_score = svr_selected
        predictions[svr_name] = stats["y_mean"] + np.asarray(svr_model.predict(features), dtype=np.float32) * stats["y_std"]
    else:
        svr_name, svr_score = "svr_rbf_unavailable", math.nan

    all_metric_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    aggregate_by_model: dict[str, dict[str, Any]] = {}
    val_scores: dict[str, float] = {}
    for model_name, pred in predictions.items():
        rows = evaluate_param_predictions(dataset, pred, stats)
        for split_name in ("train", "val", "test"):
            agg = aggregate_prediction_rows(rows, model_name, split_name)
            agg["selection_metric"] = float(agg.get("normalized_param_mae_mean_mean", math.nan))
            agg["selected_by_val"] = False
            all_metric_rows.append(agg)
            if split_name == "val":
                val_scores[model_name] = float(agg["selection_metric"])
        all_group_rows.extend(prediction_group_rows(model_name, rows))
    selected_model = min(val_scores, key=val_scores.get)
    for row in all_metric_rows:
        row["selected_by_val"] = row["model"] == selected_model
        if row["model"] == selected_model and row["split"] == "test":
            aggregate_by_model["selected_test"] = row
        if row["model"] == "mean_train_target" and row["split"] == "test":
            aggregate_by_model["mean_test"] = row

    write_csv(args.metrics, all_metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, all_group_rows, GROUP_FIELDS)

    selected_test = aggregate_by_model.get("selected_test", {})
    mean_test = aggregate_by_model.get("mean_test", {})
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "20.73 true 3D RBC feature baseline summary",
                "",
                "Scope: Piao-inspired feature regression sanity only; this is not full Piao 2019 NLS + LS-SVM reproduction.",
                f"dataset_id: {dataset.dataset_id}",
                f"input: delta_b only, shape={list(dataset.delta_b.shape)}, flattened_feature_channels=9",
                f"feature_count: {len(feature_names)}",
                f"models: {', '.join(predictions.keys())}",
                f"selected_model_by_validation: {selected_model}",
                f"selected_val_normalized_mae: {val_scores[selected_model]:.6f}",
                f"mean_test_normalized_mae: {float(mean_test.get('normalized_param_mae_mean_mean', math.nan)):.6f}",
                f"selected_test_normalized_mae: {float(selected_test.get('normalized_param_mae_mean_mean', math.nan)):.6f}",
                f"selected_test_LWD_mae_mm: L={float(selected_test.get('L_mae_mm_mean', math.nan)):.6f}, W={float(selected_test.get('W_mae_mm_mean', math.nan)):.6f}, D={float(selected_test.get('D_mae_mm_mean', math.nan)):.6f}",
                f"selected_test_curvature_mae: {float(selected_test.get('curvature_mae_mean_mean', math.nan)):.6f}",
                f"selected_test_projected_mask: IoU={float(selected_test.get('projected_mask_iou_mean', math.nan)):.6f}, Dice={float(selected_test.get('projected_mask_dice_mean', math.nan)):.6f}",
                f"ridge_selected: {selected_ridge_name}, val_normalized_mae={selected_ridge_score:.6f}",
                f"svr_status: {svr_name}, val_normalized_mae={svr_score if not math.isnan(svr_score) else 'nan'}",
                "Data boundary: no COMSOL run, no data generation, no NPZ modification, no baseline update.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--group-summary", type=Path, default=GROUP_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
