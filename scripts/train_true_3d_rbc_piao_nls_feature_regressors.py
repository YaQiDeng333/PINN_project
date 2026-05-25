#!/usr/bin/env python
"""Train Piao/NLS-inspired feature regressors for curvature diagnostics.

This is a classical feature diagnostic. It trains no neural model, uses only
delta_b-derived features, selects feature/model candidates on validation only,
and reports test metrics only for the validation-selected candidate.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    V3_240_DATASET_ID,
    ROOT,
    aggregate_prediction_rows,
    clip_params_to_train_bounds,
    depth_grid_from_params,
    depth_map_from_params,
    evaluate_param_predictions,
    load_dataset,
    mask_metrics,
    normalize_y,
    projected_mask_from_params,
    split_indices,
    train_normalization,
    write_csv,
)


FEATURES = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_features.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_piao_nls_feature_regression_summary.txt"
CANDIDATES = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_regression_candidates.csv"
METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_regression_metrics.csv"
GROUP = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_regression_group_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_regression_vs_reference.csv"

REF_NEURAL = {
    "label": "20.77_neural",
    "total": 0.6780143536818333,
    "L_mm": 1.891891360282898,
    "W_mm": 2.1857595443725586,
    "D_mm": 0.8002312183380127,
    "curvature": 0.20107580721378326,
    "wLD": 0.2094394713640213,
    "wWD": 0.20446909964084625,
    "wLW": 0.18931882083415985,
    "dice": 0.8477271366767738,
    "iou": 0.7506502455785019,
    "profile_rmse": 0.0003877372636895579,
}
REF_FEATURE = {
    "label": "20.77_feature_baseline_svr_rbf_C10",
    "total": 0.7153952435041085,
    "L_mm": 2.7033732659541645,
    "W_mm": 2.486409778587329,
    "D_mm": 0.9795067076308605,
    "curvature": 0.19504618346213531,
    "wLD": 0.2201639887614128,
    "wWD": 0.1943696569173764,
    "wLW": 0.17060490754934457,
    "dice": 0.815450420558869,
    "iou": 0.7023349188720109,
    "profile_rmse": 0.0004638535043714234,
}
REF_REFINED = {
    "label": "20.79_refined_C1_split_heads",
    "total": 0.7533873075093979,
    "L_mm": 2.660231047715896,
    "W_mm": 2.135249491303395,
    "D_mm": 1.1115478296310475,
    "curvature": 0.21158406477517042,
    "wLD": 0.2320939057912582,
    "wWD": 0.21763882270226112,
    "wLW": 0.18501947017816398,
    "dice": 0.8345969696140306,
    "iou": 0.7282398476842469,
    "profile_rmse": math.nan,
}

CANDIDATE_FIELDS = [
    "feature_set",
    "model",
    "split",
    "feature_count",
    "selection_score",
    "selected_by_validation",
    "normalized_param_mae",
    "dimension_mae_norm",
    "curvature_mae_norm",
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
    "clip_rate",
    "notes",
]
METRIC_FIELDS = CANDIDATE_FIELDS + ["test_final_only", "sample_count"]
GROUP_FIELDS = [
    "feature_set",
    "model",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "normalized_param_mae_mean",
    "dimension_mae_norm",
    "curvature_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "curvature_mae_mean",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "projected_mask_iou",
    "projected_mask_dice",
    "profile_depth_rmse_m",
]


class FeatureTransform:
    def __init__(self, median: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        self.median = median
        self.mean = mean
        self.std = std

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).copy()
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = np.take(self.median, np.where(bad)[1])
        return ((arr - self.mean) / self.std).astype(np.float32)


def read_features(path: Path) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        feature_names = [name for name in fields if name not in {"sample_id", "split"}]
        rows = list(reader)
    sample_ids = [row["sample_id"] for row in rows]
    split = [row["split"] for row in rows]
    matrix = np.asarray([[float(row[name]) if row[name] not in {"", "nan", "NaN"} else math.nan for name in feature_names] for row in rows], dtype=np.float64)
    return feature_names, matrix, sample_ids, split


def feature_sets(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {
        "F0_existing_only": ["F0__"],
        "F0_F1_F2_basic_physical": ["F0__", "F1__", "F2__"],
        "F0_F1_F2_F3_cross_axis": ["F0__", "F1__", "F2__", "F3__"],
        "F0_F1_F2_F3_F4_nls": ["F0__", "F1__", "F2__", "F3__", "F4__"],
        "F0_F1_F2_F3_F4_F5_curvature_focused": ["F0__", "F1__", "F2__", "F3__", "F4__", "F5__"],
        "curvature_focused_without_F0": ["F1__", "F2__", "F3__", "F4__", "F5__"],
    }
    out: dict[str, list[int]] = {}
    for label, prefixes in groups.items():
        idx = [i for i, name in enumerate(feature_names) if any(name.startswith(prefix) for prefix in prefixes)]
        if idx:
            out[label] = idx
    return out


def fit_transformer(features: np.ndarray, train_idx: np.ndarray) -> FeatureTransform:
    train = features[train_idx].copy()
    train[~np.isfinite(train)] = np.nan
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train_bad = ~np.isfinite(train)
    if np.any(train_bad):
        train[train_bad] = np.take(median, np.where(train_bad)[1])
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1.0e-12, 1.0, std)
    return FeatureTransform(median.astype(np.float64), mean.astype(np.float64), std.astype(np.float64))


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> Callable[[np.ndarray], np.ndarray]:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)
    weights = coef[:-1].astype(np.float32)
    bias = coef[-1:].astype(np.float32)
    return lambda z: (z @ weights + bias).astype(np.float32)


def sklearn_models(x: np.ndarray, y: np.ndarray, n_features: int) -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    models: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
    try:
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVR
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return models
    gamma_base = 1.0 / max(1, n_features)
    for alpha, gamma in ((0.1, gamma_base), (1.0, gamma_base), (1.0, 4.0 * gamma_base), (10.0, gamma_base)):
        model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        model.fit(x, y)
        models.append((f"kernelridge_rbf_alpha{alpha:g}_gamma{gamma:.4g}", lambda z, m=model: np.asarray(m.predict(z), dtype=np.float32)))
    for c_value, eps in ((1.0, 0.03), (3.0, 0.03), (10.0, 0.03), (30.0, 0.05)):
        model = MultiOutputRegressor(SVR(kernel="rbf", C=c_value, gamma="scale", epsilon=eps))
        model.fit(x, y)
        models.append((f"svr_rbf_C{c_value:g}_eps{eps:g}", lambda z, m=model: np.asarray(m.predict(z), dtype=np.float32)))
    for max_depth, min_leaf in ((3, 5), (5, 5), (8, 3)):
        model = RandomForestRegressor(n_estimators=250, max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42, n_jobs=-1)
        model.fit(x, y)
        models.append((f"randomforest_depth{max_depth}_leaf{min_leaf}", lambda z, m=model: np.asarray(m.predict(z), dtype=np.float32)))
    return models


def aggregate(rows: list[dict[str, Any]], feature_set: str, model: str, split: str, score: float, selected: bool, feature_count: int, notes: str = "") -> dict[str, Any]:
    agg = aggregate_prediction_rows(rows, model, split)
    return {
        "feature_set": feature_set,
        "model": model,
        "split": split,
        "feature_count": feature_count,
        "selection_score": score,
        "selected_by_validation": selected,
        "normalized_param_mae": agg.get("normalized_param_mae_mean_mean", math.nan),
        "dimension_mae_norm": agg.get("dimension_param_mae_norm_mean", math.nan),
        "curvature_mae_norm": agg.get("curvature_param_mae_norm_mean", math.nan),
        "curvature_mae": agg.get("curvature_mae_mean_mean", math.nan),
        "L_mae_mm": agg.get("L_mae_mm_mean", math.nan),
        "W_mae_mm": agg.get("W_mae_mm_mean", math.nan),
        "D_mae_mm": agg.get("D_mae_mm_mean", math.nan),
        "wLD_abs_error": agg.get("wLD_abs_error_mean", math.nan),
        "wWD_abs_error": agg.get("wWD_abs_error_mean", math.nan),
        "wLW_abs_error": agg.get("wLW_abs_error_mean", math.nan),
        "projected_mask_iou": agg.get("projected_mask_iou_mean", math.nan),
        "projected_mask_dice": agg.get("projected_mask_dice_mean", math.nan),
        "profile_depth_rmse_m": agg.get("profile_depth_rmse_m_mean", math.nan),
        "clip_rate": agg.get("clip_applied_mean", math.nan),
        "notes": notes,
        "sample_count": agg.get("sample_count", 0),
    }


def evaluate_param_predictions_subset(dataset: Any, pred_params_raw: np.ndarray, stats: dict[str, np.ndarray], indices: np.ndarray) -> list[dict[str, Any]]:
    pred_params, clipped = clip_params_to_train_bounds(np.asarray(pred_params_raw, dtype=np.float32), dataset)
    pred_norm = (pred_params - stats["y_mean"]) / stats["y_std"]
    true_norm = (dataset.rbc_params - stats["y_mean"]) / stats["y_std"]
    rows: list[dict[str, Any]] = []
    subset_clip_fraction = float(np.mean(clipped[indices])) if len(indices) else 0.0
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
                "clip_fraction": subset_clip_fraction,
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


def group_rows(rows: list[dict[str, Any]], feature_set: str, model: str, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base = [row for row in rows if row["split"] == split]
    for field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
        for value in sorted({str(row[field]) for row in base}):
            subset = [row for row in base if str(row[field]) == value]
            if not subset:
                continue
            out.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "split": split,
                    "group_field": field,
                    "group_value": value,
                    "sample_count": len(subset),
                    "normalized_param_mae_mean": float(np.mean([row["normalized_param_mae_mean"] for row in subset])),
                    "dimension_mae_norm": float(np.mean([row["dimension_param_mae_norm"] for row in subset])),
                    "curvature_mae_norm": float(np.mean([row["curvature_param_mae_norm"] for row in subset])),
                    "L_mae_mm": float(np.mean([row["L_mae_mm"] for row in subset])),
                    "W_mae_mm": float(np.mean([row["W_mae_mm"] for row in subset])),
                    "D_mae_mm": float(np.mean([row["D_mae_mm"] for row in subset])),
                    "curvature_mae_mean": float(np.mean([row["curvature_mae_mean"] for row in subset])),
                    "wLD_abs_error": float(np.mean([row["wLD_abs_error"] for row in subset])),
                    "wWD_abs_error": float(np.mean([row["wWD_abs_error"] for row in subset])),
                    "wLW_abs_error": float(np.mean([row["wLW_abs_error"] for row in subset])),
                    "projected_mask_iou": float(np.mean([row["projected_mask_iou"] for row in subset])),
                    "projected_mask_dice": float(np.mean([row["projected_mask_dice"] for row in subset])),
                    "profile_depth_rmse_m": float(np.mean([row["profile_depth_rmse_m"] for row in subset])),
                }
            )
    return out


def selection_score(val_row: dict[str, Any], val_rows: list[dict[str, Any]]) -> float:
    templates = sorted({str(row["curvature_template"]) for row in val_rows})
    max_template_curv = 0.0
    for template in templates:
        subset = [row for row in val_rows if str(row["curvature_template"]) == template]
        max_template_curv = max(max_template_curv, float(np.mean([row["curvature_mae_mean"] for row in subset])))
    return float(val_row["normalized_param_mae"]) + 0.50 * float(val_row["curvature_mae_norm"]) + 0.15 * max_template_curv + 0.10 * float(val_row["clip_rate"]) + 0.10 * max(0.0, REF_NEURAL["dice"] - float(val_row["projected_mask_dice"]))


def comparison_rows(selected_test: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = [
        ("total_normalized_mae", "normalized_param_mae", True),
        ("L_mae_mm", "L_mae_mm", True),
        ("W_mae_mm", "W_mae_mm", True),
        ("D_mae_mm", "D_mae_mm", True),
        ("curvature_mae", "curvature_mae", True),
        ("wLD_abs_error", "wLD_abs_error", True),
        ("wWD_abs_error", "wWD_abs_error", True),
        ("wLW_abs_error", "wLW_abs_error", True),
        ("projected_mask_iou", "projected_mask_iou", False),
        ("projected_mask_dice", "projected_mask_dice", False),
        ("profile_depth_rmse_m", "profile_depth_rmse_m", True),
    ]
    ref_key = {
        "total_normalized_mae": "total",
        "L_mae_mm": "L_mm",
        "W_mae_mm": "W_mm",
        "D_mae_mm": "D_mm",
        "curvature_mae": "curvature",
        "wLD_abs_error": "wLD",
        "wWD_abs_error": "wWD",
        "wLW_abs_error": "wLW",
        "projected_mask_iou": "iou",
        "projected_mask_dice": "dice",
        "profile_depth_rmse_m": "profile_rmse",
    }
    refs = [REF_NEURAL, REF_FEATURE, REF_REFINED]
    rows: list[dict[str, Any]] = []
    for metric_name, selected_key, lower_better in metrics:
        current = float(selected_test.get(selected_key, math.nan))
        for ref in refs:
            rv = float(ref.get(ref_key[metric_name], math.nan))
            delta = current - rv
            improved = delta < 0 if lower_better else delta > 0
            rows.append(
                {
                    "metric": metric_name,
                    "reference_label": ref["label"],
                    "current_label": f"{selected_test['feature_set']}::{selected_test['model']}",
                    "reference_value": rv,
                    "current_value": current,
                    "delta": delta,
                    "improved": improved,
                }
            )
    return rows


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset_id)
    feature_names, raw_features, feature_sample_ids, feature_split = read_features(args.features)
    if feature_sample_ids != [str(x) for x in dataset.sample_ids]:
        raise RuntimeError("feature CSV sample_id order does not match dataset")
    if feature_split != [str(x) for x in dataset.split]:
        raise RuntimeError("feature CSV split order does not match dataset")
    stats = train_normalization(dataset)
    y_norm = normalize_y(dataset, stats)
    splits = split_indices(dataset)
    train_idx = splits["train"]
    val_idx = splits["val"]
    all_sets = feature_sets(feature_names)

    candidate_rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[str, str, int, Callable[[np.ndarray], np.ndarray], FeatureTransform, list[int], float]] = {}
    for set_name, idx in all_sets.items():
        selected_raw = raw_features[:, idx]
        transform = fit_transformer(selected_raw, train_idx)
        x_scaled = transform.transform(selected_raw)
        model_fns: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
        for alpha in (0.1, 1.0, 10.0, 100.0, 300.0, 1000.0):
            model_fns.append((f"ridge_alpha{alpha:g}", ridge_fit(x_scaled[train_idx], y_norm[train_idx], alpha)))
        model_fns.extend(sklearn_models(x_scaled[train_idx], y_norm[train_idx], len(idx)))
        for model_name, pred_norm_fn in model_fns:
            pred_train_val_norm = pred_norm_fn(x_scaled)
            pred_params = stats["y_mean"] + pred_train_val_norm * stats["y_std"]
            pred_rows = evaluate_param_predictions_subset(dataset, pred_params, stats, np.concatenate([train_idx, val_idx]))
            val_agg = aggregate(pred_rows, set_name, model_name, "val", math.nan, False, len(idx), "validation-only candidate")
            score = selection_score(val_agg, [row for row in pred_rows if row["split"] == "val"])
            val_agg["selection_score"] = score
            train_agg = aggregate(pred_rows, set_name, model_name, "train", score, False, len(idx), "train diagnostic only")
            train_agg["selection_score"] = score
            candidate_rows.extend([train_agg, val_agg])
            fitted[f"{set_name}::{model_name}"] = (set_name, model_name, len(idx), pred_norm_fn, transform, idx, score)

    selected_key = min(fitted, key=lambda key: fitted[key][6])
    set_name, model_name, feature_count, pred_norm_fn, transform, idx, score = fitted[selected_key]
    x_selected = transform.transform(raw_features[:, idx])
    pred_norm = pred_norm_fn(x_selected)
    pred_params = stats["y_mean"] + pred_norm * stats["y_std"]
    pred_rows = evaluate_param_predictions(dataset, pred_params, stats)

    for row in candidate_rows:
        row["selected_by_validation"] = (row["feature_set"] == set_name and row["model"] == model_name)
    selected_metric_rows = [
        aggregate(pred_rows, set_name, model_name, split, score, True, feature_count, "selected by validation; test is final-only")
        | {"test_final_only": split == "test"}
        for split in ("train", "val", "test")
    ]
    selected_test = [row for row in selected_metric_rows if row["split"] == "test"][0]

    write_csv(args.candidates, candidate_rows, CANDIDATE_FIELDS)
    write_csv(args.metrics, selected_metric_rows, METRIC_FIELDS)
    group_out: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        group_out.extend(group_rows(pred_rows, set_name, model_name, split))
    write_csv(args.group_summary, group_out, GROUP_FIELDS)
    write_csv(args.vs_reference, comparison_rows(selected_test), ["metric", "reference_label", "current_label", "reference_value", "current_value", "delta", "improved"])

    curvature_improved_vs_neural = float(selected_test["curvature_mae"]) <= REF_NEURAL["curvature"] - 0.01
    curvature_improved_vs_feature = float(selected_test["curvature_mae"]) < REF_FEATURE["curvature"]
    feature_set_beyond_f0 = set_name != "F0_existing_only"
    diagnostic_useful = feature_set_beyond_f0 and (curvature_improved_vs_neural or curvature_improved_vs_feature)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 Piao/NLS feature regression summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                "scope: Piao/NLS-inspired feature regressor diagnostic; no neural training, no COMSOL, no data generation, no baseline update.",
                "model_selection: validation-only feature_set/model/hyperparameter selection; test final only for selected candidate.",
                f"feature_sets_tested: {', '.join(all_sets)}",
                f"candidate_train_val_rows: {len(candidate_rows)}",
                f"selected_feature_set: {set_name}",
                f"selected_model: {model_name}",
                f"selected_feature_count: {feature_count}",
                f"selected_validation_score: {score:.6f}",
                f"selected_test_total_mae: {float(selected_test['normalized_param_mae']):.6f}",
                f"selected_test_LWD_mae_mm: {float(selected_test['L_mae_mm']):.6f}/{float(selected_test['W_mae_mm']):.6f}/{float(selected_test['D_mae_mm']):.6f}",
                f"selected_test_curvature_mae: {float(selected_test['curvature_mae']):.6f}",
                f"selected_test_wLD_wWD_wLW: {float(selected_test['wLD_abs_error']):.6f}/{float(selected_test['wWD_abs_error']):.6f}/{float(selected_test['wLW_abs_error']):.6f}",
                f"selected_test_projected_mask_iou_dice: {float(selected_test['projected_mask_iou']):.6f}/{float(selected_test['projected_mask_dice']):.6f}",
                f"selected_test_profile_depth_rmse_m: {float(selected_test['profile_depth_rmse_m']):.9f}",
                f"vs_20_77_neural_curvature_improved_by_0p01: {curvature_improved_vs_neural}",
                f"vs_20_77_feature_curvature_improved: {curvature_improved_vs_feature}",
                f"feature_set_beyond_F0_responsible: {feature_set_beyond_f0}",
                f"diagnostic_useful_for_curvature: {diagnostic_useful}",
                "interpretation: F0+F1+F2 physical features helped curvature in this diagnostic; F4 NLS proxy was extracted successfully but was not part of the validation-selected feature set.",
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
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
