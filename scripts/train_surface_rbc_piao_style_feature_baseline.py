#!/usr/bin/env python
"""Train a Piao-style NLS-lite feature baseline for the surface RBC v3_240 pack.

The model input is restricted to the existing 24.0A ``nlslite_*`` feature CSV.
No feature extraction, COMSOL run, NPZ write, checkpoint write, or baseline
promotion happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    PARAM_NAMES,
    ROOT,
    V3_240_DATASET_ID,
    aggregate_prediction_rows,
    check_no_overwrite,
    evaluate_param_predictions,
    load_dataset,
    normalize_y,
    split_indices,
    train_normalization,
    write_csv,
)


FEATURE_MANIFEST = ROOT / "results/manifests/surface_rbc_nls_lite_feature_manifest.json"
FEATURE_CSV = ROOT / "results/metrics/surface_rbc_nls_lite_features.csv"
QUALITY_CSV = ROOT / "results/metrics/surface_rbc_nls_lite_feature_quality.csv"

SUMMARY = ROOT / "results/summaries/surface_rbc_piao_style_feature_baseline_summary.txt"
METRICS = ROOT / "results/metrics/surface_rbc_piao_style_feature_baseline_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/surface_rbc_piao_style_feature_baseline_group_summary.csv"

METRIC_FIELDS = [
    "candidate_order",
    "model",
    "family",
    "alpha",
    "gamma",
    "C",
    "epsilon",
    "sklearn_available",
    "selected_by_validation",
    "validation_selection_metric",
    "test_final_only",
    "evaluation_role",
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
    "projected_mask_center_error_px_mean",
    "profile_depth_rmse_m_mean",
    "volume_proxy_rel_error_mean",
    "clip_applied_mean",
    "feature_count",
    "notes",
]

GROUP_FIELDS = [
    "model",
    "family",
    "selected_by_validation",
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
    "volume_proxy_rel_error",
]

FORBIDDEN_INPUT_COLUMNS = set(PARAM_NAMES) | {
    "profile",
    "profile_depth_grid_m",
    "profile_depth_map_xy_m",
    "mask",
    "projected_mask_2d",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "size_bin",
}


@dataclass(frozen=True)
class CandidateResult:
    candidate_order: int
    model: str
    family: str
    pred_raw: np.ndarray
    val_score: float
    alpha: str = ""
    gamma: str = ""
    c_value: str = ""
    epsilon: str = ""
    sklearn_available: bool | str = ""
    notes: str = ""


@dataclass(frozen=True)
class TrainingResult:
    dataset: Any
    stats: dict[str, np.ndarray]
    feature_names: list[str]
    candidates: list[CandidateResult]
    selected: CandidateResult
    feature_manifest: dict[str, Any]
    quality: dict[str, str]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def validate_feature_manifest(path: Path, dataset_id: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "dataset_id": manifest.get("dataset_id") == dataset_id,
        "piao_nls_lite": manifest.get("piao_nls_lite") is True,
        "exact_piao_nls_false": manifest.get("exact_piao_nls") is False,
        "target_labels_absent": manifest.get("target_labels_in_feature_csv") is False,
        "training_run_false": manifest.get("training_run") is False,
        "COMSOL_run_false": manifest.get("COMSOL_run") is False,
        "data_or_NPZ_modified_false": manifest.get("data_or_NPZ_modified") is False,
        "CURRENT_BASELINE_update_false": manifest.get("CURRENT_BASELINE_update") is False,
        "feature_prefix": manifest.get("formal_feature_prefix") == "nlslite_",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"NLS-lite feature manifest gate failed: {failed}")
    for required_path in (FEATURE_CSV, QUALITY_CSV):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    return manifest


def read_quality(path: Path) -> dict[str, str]:
    for row in read_csv_rows(path):
        if row.get("quality_scope") == "overall":
            return row
    raise RuntimeError(f"overall quality row missing: {path}")


def load_nlslite_feature_matrix(dataset: Any, feature_csv: Path) -> tuple[np.ndarray, list[str]]:
    rows = read_csv_rows(feature_csv)
    if not rows:
        raise RuntimeError(f"empty feature CSV: {feature_csv}")
    fieldnames = list(rows[0].keys())
    feature_names = [name for name in fieldnames if name.startswith("nlslite_")]
    metadata_columns = [name for name in fieldnames if not name.startswith("nlslite_")]
    if metadata_columns != ["sample_id", "split"]:
        raise RuntimeError(f"unexpected non-feature columns in feature CSV: {metadata_columns}")
    if len(feature_names) == 0:
        raise RuntimeError("no nlslite_* feature columns found")
    leaked = sorted(FORBIDDEN_INPUT_COLUMNS.intersection(feature_names))
    if leaked:
        raise RuntimeError(f"forbidden columns present as model inputs: {leaked}")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            raise RuntimeError("feature row missing sample_id")
        if sample_id in by_id:
            raise RuntimeError(f"duplicate feature sample_id: {sample_id}")
        by_id[sample_id] = row
    x = np.empty((len(dataset.sample_ids), len(feature_names)), dtype=np.float64)
    for idx, sample_id in enumerate(dataset.sample_ids.astype(str)):
        if sample_id not in by_id:
            raise RuntimeError(f"feature row missing for dataset sample_id: {sample_id}")
        row = by_id[sample_id]
        if row.get("split") != str(dataset.split[idx]):
            raise RuntimeError(f"split mismatch for {sample_id}: feature={row.get('split')} dataset={dataset.split[idx]}")
        for col_idx, name in enumerate(feature_names):
            x[idx, col_idx] = float(row[name])
    if not np.isfinite(x).all():
        raise RuntimeError("NLS-lite feature matrix contains non-finite values")
    return x.astype(np.float32), feature_names


def fit_feature_scaler(x: np.ndarray, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    x_train = x[train_idx]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-12, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def transform_features(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)


def normalized_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def denorm_y(y_norm: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(y_norm, dtype=np.float32) * stats["y_std"] + stats["y_mean"]).astype(np.float32)


def fit_linear_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + penalty
    rhs = x_aug.T @ y_train
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return coef[:-1].astype(np.float32), coef[-1:].astype(np.float32)


def predict_linear(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(x @ weights + bias, dtype=np.float32)


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    a2 = np.sum(a64 * a64, axis=1, keepdims=True)
    b2 = np.sum(b64 * b64, axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * (a64 @ b64.T), 0.0)


def fit_predict_lssvm_rbf(x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    k_train = np.exp(-float(gamma) * pairwise_sq_dists(x_train, x_train))
    lhs = k_train + float(alpha) * np.eye(k_train.shape[0], dtype=np.float64)
    try:
        coef = np.linalg.solve(lhs, np.asarray(y_train, dtype=np.float64))
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, np.asarray(y_train, dtype=np.float64), rcond=None)[0]
    k_all = np.exp(-float(gamma) * pairwise_sq_dists(x_all, x_train))
    return np.asarray(k_all @ coef, dtype=np.float32)


def append_mean_candidate(
    out: list[CandidateResult],
    order: int,
    dataset: Any,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_norm: np.ndarray,
    stats: dict[str, np.ndarray],
) -> int:
    train_mean = dataset.rbc_params[train_idx].mean(axis=0, keepdims=True)
    pred_raw = np.repeat(train_mean, len(dataset.sample_ids), axis=0).astype(np.float32)
    pred_norm = (pred_raw - stats["y_mean"]) / stats["y_std"]
    out.append(
        CandidateResult(
            candidate_order=order,
            model="mean_train_target",
            family="mean",
            pred_raw=pred_raw,
            val_score=normalized_mae(y_norm[val_idx], pred_norm[val_idx]),
            notes="train-target mean comparator; no labels used as input",
        )
    )
    return order + 1


def append_ridge_candidates(
    out: list[CandidateResult],
    order: int,
    x: np.ndarray,
    y_norm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    stats: dict[str, np.ndarray],
) -> int:
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        weights, bias = fit_linear_ridge(x[train_idx], y_norm[train_idx], alpha)
        pred_norm = predict_linear(x, weights, bias)
        out.append(
            CandidateResult(
                candidate_order=order,
                model=f"ridge_alpha_{token(alpha)}",
                family="Ridge",
                pred_raw=denorm_y(pred_norm, stats),
                val_score=normalized_mae(y_norm[val_idx], pred_norm[val_idx]),
                alpha=f"{alpha:g}",
            )
        )
        order += 1
    return order


def append_lssvm_candidates(
    out: list[CandidateResult],
    order: int,
    x: np.ndarray,
    y_norm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    stats: dict[str, np.ndarray],
) -> int:
    n_features = max(1, x.shape[1])
    gamma_values = [0.5 / n_features, 1.0 / n_features, 2.0 / n_features, 5.0 / n_features]
    for alpha in (0.001, 0.01, 0.1, 1.0):
        for gamma in gamma_values:
            pred_norm = fit_predict_lssvm_rbf(x[train_idx], y_norm[train_idx], x, alpha, gamma)
            out.append(
                CandidateResult(
                    candidate_order=order,
                    model=f"lssvm_rbf_alpha_{token(alpha)}_gamma_{token(gamma)}",
                    family="LS-SVM-like-RBF",
                    pred_raw=denorm_y(pred_norm, stats),
                    val_score=normalized_mae(y_norm[val_idx], pred_norm[val_idx]),
                    alpha=f"{alpha:g}",
                    gamma=f"{gamma:.9g}",
                    notes="closed-form kernel ridge used as deterministic LS-SVM-like RBF",
                )
            )
            order += 1
    return order


def append_kernel_ridge_candidates(
    out: list[CandidateResult],
    order: int,
    x: np.ndarray,
    y_norm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    stats: dict[str, np.ndarray],
) -> int:
    try:
        from sklearn.kernel_ridge import KernelRidge
    except Exception as exc:
        out.append(
            CandidateResult(
                candidate_order=order,
                model="kernel_ridge_rbf_unavailable",
                family="KernelRidge_RBF",
                pred_raw=np.repeat(stats["y_mean"], len(x), axis=0).astype(np.float32),
                val_score=math.inf,
                sklearn_available=False,
                notes=f"sklearn import failed: {type(exc).__name__}",
            )
        )
        return order + 1

    n_features = max(1, x.shape[1])
    gamma_values = [0.5 / n_features, 1.0 / n_features, 2.0 / n_features, 5.0 / n_features]
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0):
        for gamma in gamma_values:
            model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            model.fit(x[train_idx], y_norm[train_idx])
            pred_norm = np.asarray(model.predict(x), dtype=np.float32)
            out.append(
                CandidateResult(
                    candidate_order=order,
                    model=f"kernel_ridge_rbf_alpha_{token(alpha)}_gamma_{token(gamma)}",
                    family="KernelRidge_RBF",
                    pred_raw=denorm_y(pred_norm, stats),
                    val_score=normalized_mae(y_norm[val_idx], pred_norm[val_idx]),
                    alpha=f"{alpha:g}",
                    gamma=f"{gamma:.9g}",
                    sklearn_available=True,
                )
            )
            order += 1
    return order


def append_svr_candidates(
    out: list[CandidateResult],
    order: int,
    x: np.ndarray,
    y_norm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    stats: dict[str, np.ndarray],
) -> int:
    try:
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVR
    except Exception as exc:
        out.append(
            CandidateResult(
                candidate_order=order,
                model="svr_rbf_unavailable",
                family="SVR_RBF",
                pred_raw=np.repeat(stats["y_mean"], len(x), axis=0).astype(np.float32),
                val_score=math.inf,
                sklearn_available=False,
                notes=f"sklearn import failed: {type(exc).__name__}",
            )
        )
        return order + 1

    n_features = max(1, x.shape[1])
    gamma_values: list[str | float] = ["scale", 1.0 / n_features]
    for c_value in (1.0, 10.0, 100.0):
        for epsilon in (0.03, 0.08):
            for gamma in gamma_values:
                model = MultiOutputRegressor(SVR(kernel="rbf", C=c_value, gamma=gamma, epsilon=epsilon))
                model.fit(x[train_idx], y_norm[train_idx])
                pred_norm = np.asarray(model.predict(x), dtype=np.float32)
                gamma_label = gamma if isinstance(gamma, str) else f"{gamma:.9g}"
                out.append(
                    CandidateResult(
                        candidate_order=order,
                        model=f"svr_rbf_C_{token(c_value)}_eps_{token(epsilon)}_gamma_{str(gamma_label).replace('.', 'p')}",
                        family="SVR_RBF",
                        pred_raw=denorm_y(pred_norm, stats),
                        val_score=normalized_mae(y_norm[val_idx], pred_norm[val_idx]),
                        gamma=str(gamma_label),
                        c_value=f"{c_value:g}",
                        epsilon=f"{epsilon:g}",
                        sklearn_available=True,
                    )
                )
                order += 1
    return order


def run_model_selection(dataset_id: str = V3_240_DATASET_ID) -> TrainingResult:
    manifest = validate_feature_manifest(FEATURE_MANIFEST, dataset_id)
    quality = read_quality(QUALITY_CSV)
    dataset = load_dataset(dataset_id)
    splits = split_indices(dataset)
    stats = train_normalization(dataset)
    raw_x, feature_names = load_nlslite_feature_matrix(dataset, FEATURE_CSV)
    if int(manifest.get("feature_count", -1)) != len(feature_names):
        raise RuntimeError(f"feature count mismatch: manifest={manifest.get('feature_count')} csv={len(feature_names)}")
    x = transform_features(raw_x, fit_feature_scaler(raw_x, splits["train"]))
    y_norm = normalize_y(dataset, stats)

    candidates: list[CandidateResult] = []
    order = 0
    order = append_mean_candidate(candidates, order, dataset, splits["train"], splits["val"], y_norm, stats)
    order = append_ridge_candidates(candidates, order, x, y_norm, splits["train"], splits["val"], stats)
    order = append_lssvm_candidates(candidates, order, x, y_norm, splits["train"], splits["val"], stats)
    order = append_kernel_ridge_candidates(candidates, order, x, y_norm, splits["train"], splits["val"], stats)
    order = append_svr_candidates(candidates, order, x, y_norm, splits["train"], splits["val"], stats)
    finite_candidates = [candidate for candidate in candidates if math.isfinite(candidate.val_score)]
    if not finite_candidates:
        raise RuntimeError("no finite candidate validation scores")
    selected = min(finite_candidates, key=lambda item: (item.val_score, item.candidate_order))
    return TrainingResult(dataset, stats, feature_names, candidates, selected, manifest, quality)


def prediction_group_rows(candidate: CandidateResult, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                        "model": candidate.model,
                        "family": candidate.family,
                        "selected_by_validation": True,
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
                        "volume_proxy_rel_error": float(np.mean([row["volume_proxy_rel_error"] for row in subset])),
                    }
                )
    return out


def metric_rows_for_result(result: TrainingResult) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_model_split: dict[str, dict[str, Any]] = {}
    for candidate in result.candidates:
        if not math.isfinite(candidate.val_score):
            continue
        pred_rows = evaluate_param_predictions(result.dataset, candidate.pred_raw, result.stats)
        if candidate.model == result.selected.model:
            split_names = ("train", "val", "test")
            role = "validation_selected_final"
        elif candidate.model == "mean_train_target":
            split_names = ("train", "val", "test")
            role = "fixed_mean_comparator"
        else:
            split_names = ("train", "val")
            role = "validation_screen_only"
        for split_name in split_names:
            agg = aggregate_prediction_rows(pred_rows, candidate.model, split_name)
            agg.update(
                {
                    "candidate_order": candidate.candidate_order,
                    "family": candidate.family,
                    "alpha": candidate.alpha,
                    "gamma": candidate.gamma,
                    "C": candidate.c_value,
                    "epsilon": candidate.epsilon,
                    "sklearn_available": candidate.sklearn_available,
                    "selected_by_validation": candidate.model == result.selected.model,
                    "validation_selection_metric": candidate.val_score,
                    "test_final_only": split_name == "test" and candidate.model == result.selected.model,
                    "evaluation_role": role,
                    "feature_count": len(result.feature_names),
                    "notes": candidate.notes,
                }
            )
            rows.append(agg)
            by_model_split[f"{candidate.model}::{split_name}"] = agg
    return rows, by_model_split


def write_summary(result: TrainingResult, by_model_split: dict[str, dict[str, Any]], path: Path) -> None:
    selected = result.selected
    selected_test = by_model_split[f"{selected.model}::test"]
    selected_train = by_model_split[f"{selected.model}::train"]
    selected_val = by_model_split[f"{selected.model}::val"]
    mean_test = by_model_split.get("mean_train_target::test", {})
    fit_success = f(result.quality.get("fit_success_rate"))
    fallback = f(result.quality.get("fallback_rate"))
    lines = [
        "surface_rbc_piao_style_feature_baseline_summary",
        "stage: 24.1",
        "",
        "scope: Piao-style NLS-lite feature-to-geometry comparator; not exact Piao 18-feature reproduction.",
        f"dataset_id: {result.dataset.dataset_id}",
        f"feature_manifest: {FEATURE_MANIFEST}",
        f"feature_csv: {FEATURE_CSV}",
        f"quality_csv: {QUALITY_CSV}",
        f"feature_count: {len(result.feature_names)}",
        "formal_model_input: nlslite_* columns only",
        "sample_id_use: join/reporting only",
        "split_use: split selection only",
        "target_labels_in_feature_matrix: false",
        "COMSOL_run: false",
        "data_or_NPZ_modified: false",
        "CURRENT_BASELINE_update: false",
        "checkpoint_written: false",
        "selection_protocol: train-only feature/target scaling, validation-only model selection, test final only",
        f"candidate_count: {len([c for c in result.candidates if math.isfinite(c.val_score)])}",
        f"selected_model: {selected.model}",
        f"selected_family: {selected.family}",
        f"selected_val_normalized_mae: {selected_val['normalized_param_mae_mean_mean']:.6f}",
        f"selected_train_normalized_mae: {selected_train['normalized_param_mae_mean_mean']:.6f}",
        f"selected_test_normalized_mae: {selected_test['normalized_param_mae_mean_mean']:.6f}",
        f"mean_test_normalized_mae: {float(mean_test.get('normalized_param_mae_mean_mean', math.nan)):.6f}",
        f"selected_test_dimension_mae_norm: {selected_test['dimension_param_mae_norm_mean']:.6f}",
        f"selected_test_curvature_mae_norm: {selected_test['curvature_param_mae_norm_mean']:.6f}",
        f"selected_test_LWD_mae_mm: L={selected_test['L_mae_mm_mean']:.6f}, W={selected_test['W_mae_mm_mean']:.6f}, D={selected_test['D_mae_mm_mean']:.6f}",
        f"selected_test_w_errors: wLD={selected_test['wLD_abs_error_mean']:.6f}, wWD={selected_test['wWD_abs_error_mean']:.6f}, wLW={selected_test['wLW_abs_error_mean']:.6f}",
        f"selected_test_wMAE_auxiliary: {selected_test['curvature_mae_mean_mean']:.6f}",
        f"selected_test_projected_mask_iou_dice: {selected_test['projected_mask_iou_mean']:.6f}, {selected_test['projected_mask_dice_mean']:.6f}",
        f"selected_test_profile_depth_rmse_m: {selected_test['profile_depth_rmse_m_mean']:.9f}",
        f"selected_test_volume_proxy_rel_error: {selected_test['volume_proxy_rel_error_mean']:.6f}",
        f"selected_test_clip_applied_fraction: {selected_test['clip_applied_mean']:.6f}",
        f"nls_lite_fit_success_rate_24_0A: {fit_success:.6f}",
        f"nls_lite_fallback_rate_24_0A: {fallback:.6f}",
        "docs_sync_skipped_due_to_unrelated_24_0B_changes: true",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics, args.group_summary], args.overwrite)
    result = run_model_selection(args.dataset_id)
    metric_rows, by_model_split = metric_rows_for_result(result)
    selected_rows = evaluate_param_predictions(result.dataset, result.selected.pred_raw, result.stats)
    group_rows = prediction_group_rows(result.selected, selected_rows)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_rows, GROUP_FIELDS)
    write_summary(result, by_model_split, args.summary)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
