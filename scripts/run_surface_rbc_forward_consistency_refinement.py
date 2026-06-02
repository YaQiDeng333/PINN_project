#!/usr/bin/env python
"""Run the 25.5 six-parameter feature-space refinement diagnostic.

The optimization objective uses only observed delta_b-derived features and the
frozen 20.85 predicted six params. Labels are used after refinement for
validation hyperparameter selection and final metrics only.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - fallback used only if scipy is missing
    minimize = None

from audit_surface_shape_extension_rbc_oracle_fit import (
    DATASET_ID,
    ROOT,
    connected_component_count,
    er_like_profile_error,
    load_surface_dataset,
    mask_metrics,
    pose_for_sample,
    profile_rmse,
)
from build_surface_forward_refinement_target_set import PARAM_NAMES, REGISTRY, as_bool, as_float, write_csv
from fit_surface_feature_space_forward_surrogate import (
    PARAM_BOUNDS,
    fit_selected_surrogate,
    observed_feature_matrix,
    params_from_rows,
    residual_mse_norm,
)
from load_true_3d_rbc_pilot_dataset import depth_map_from_params, projected_mask_from_params


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_run_summary.txt"
METRICS = ROOT / "results/metrics/surface_forward_refinement_metrics.csv"
GROUP = ROOT / "results/metrics/surface_forward_refinement_by_group.csv"
FAILURES = ROOT / "results/metrics/surface_forward_refinement_failure_cases.csv"

LAMBDA_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

METRIC_FIELDS = [
    "sample_index",
    "sample_id",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "target_role",
    "diagnosis",
    "rbc_representable",
    "include_in_success_gate",
    "include_in_rbc_control_gate",
    "include_as_negative_control",
    "refinement_applied",
    "lambda_param",
    "selected_surrogate",
    "optimizer_success",
    "optimizer_iterations",
    "optimizer_message",
    "feature_residual_mse_before",
    "feature_residual_mse_after",
    "feature_residual_delta",
    "feature_residual_improved",
    "baseline_profile_depth_rmse_m",
    "refined_profile_depth_rmse_m",
    "profile_rmse_delta_m",
    "profile_rmse_improved",
    "baseline_Er_like_error",
    "refined_Er_like_error",
    "Er_like_delta",
    "Er_like_improved",
    "baseline_projected_mask_IoU",
    "refined_projected_mask_IoU",
    "IoU_delta",
    "IoU_improved",
    "baseline_projected_mask_Dice",
    "refined_projected_mask_Dice",
    "Dice_delta",
    "Dice_improved",
    "baseline_area_error",
    "refined_area_error",
    "area_error_delta",
    "area_error_improved",
    "baseline_pred_component_count",
    "refined_component_count",
    "params_in_bounds",
    "profile_nonnegative",
    "test_time_label_used_for_refinement",
    "labels_used_for_metric_only",
    *[f"initial_{name}" for name in PARAM_NAMES],
    *[f"refined_{name}" for name in PARAM_NAMES],
]

GROUP_FIELDS = [
    "group_field",
    "group_value",
    "split",
    "sample_count",
    "refinement_applied_count",
    "baseline_profile_rmse_mean_m",
    "refined_profile_rmse_mean_m",
    "profile_rmse_delta_mean_m",
    "baseline_Er_like_mean",
    "refined_Er_like_mean",
    "Er_like_delta_mean",
    "baseline_IoU_mean",
    "refined_IoU_mean",
    "IoU_delta_mean",
    "baseline_Dice_mean",
    "refined_Dice_mean",
    "Dice_delta_mean",
    "baseline_area_error_mean",
    "refined_area_error_mean",
    "area_error_delta_mean",
    "feature_residual_before_mean",
    "feature_residual_after_mean",
    "feature_residual_delta_mean",
    "profile_rmse_improved_rate",
    "forward_residual_improved_rate",
]

FAILURE_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "target_role",
    "failure_mode",
    "feature_residual_mse_before",
    "feature_residual_mse_after",
    "baseline_profile_depth_rmse_m",
    "refined_profile_depth_rmse_m",
    "baseline_projected_mask_Dice",
    "refined_projected_mask_Dice",
    "notes",
]


def clip_params(params: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(params, dtype=np.float64), PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


def params_in_bounds(params: np.ndarray) -> bool:
    p = np.asarray(params, dtype=np.float64)
    return bool(np.all(p >= PARAM_BOUNDS[:, 0] - 1.0e-12) and np.all(p <= PARAM_BOUNDS[:, 1] + 1.0e-12))


def metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in {"", None} and math.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def param_array_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([as_float(row[f"{prefix}_{name}"]) for name in PARAM_NAMES], dtype=np.float64)


def feature_residual(surrogate: Any, params: np.ndarray, observed_features: np.ndarray) -> float:
    return float(residual_mse_norm(surrogate, np.asarray(params, dtype=np.float64).reshape(1, 6), observed_features.reshape(1, -1))[0])


def evaluate_params(dataset: Any, sample_index: int, params: np.ndarray) -> dict[str, Any]:
    pose = pose_for_sample(dataset, sample_index)
    true_depth = np.asarray(dataset.depth_grid_m[sample_index], dtype=np.float64)
    true_mask = np.asarray(dataset.projected_mask_2d[sample_index], dtype=np.uint8)
    pred_depth = depth_map_from_params(params, pose)
    pred_mask = projected_mask_from_params(params, pose)
    mm = mask_metrics(pred_mask, true_mask)
    return {
        "profile_depth_rmse_m": profile_rmse(pred_depth, true_depth),
        "Er_like_error": er_like_profile_error(pred_depth, true_depth),
        "projected_mask_IoU": mm["iou"],
        "projected_mask_Dice": mm["dice"],
        "area_error": mm["area_error"],
        "component_count": connected_component_count(pred_mask),
        "profile_nonnegative": bool(np.min(pred_depth) >= -1.0e-12),
    }


def objective_factory(surrogate: Any, observed_features: np.ndarray, init_params: np.ndarray, lambda_param: float):
    init = clip_params(init_params)
    scale = np.maximum(PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0], 1.0e-12)
    target = surrogate.observed_norm(observed_features.reshape(1, -1))

    def objective(candidate: np.ndarray) -> float:
        params = clip_params(candidate)
        pred = surrogate.predict_norm(params.reshape(1, 6))
        feature_loss = float(np.mean((pred - target) ** 2))
        proximity = float(np.mean(((params - init) / scale) ** 2))
        curvature_reg = 1.0e-5 * float(np.mean(np.square(np.log(np.maximum(params[3:6], 1.0e-12)))))
        return feature_loss + float(lambda_param) * proximity + curvature_reg

    return objective


def fallback_optimize(objective: Any, init_params: np.ndarray) -> tuple[np.ndarray, bool, int, str]:
    current = clip_params(init_params)
    current_score = objective(current)
    steps = (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]) * 0.08
    iterations = 0
    for _ in range(60):
        improved = False
        for dim in range(6):
            for sign in (-1.0, 1.0):
                candidate = clip_params(current + sign * steps[dim] * np.eye(6)[dim])
                score = objective(candidate)
                iterations += 1
                if score + 1.0e-12 < current_score:
                    current, current_score, improved = candidate, score, True
        if not improved:
            steps *= 0.5
            if float(np.max(steps)) < 1.0e-6:
                break
    return current, True, iterations, "fallback_coordinate_search"


def optimize_params(surrogate: Any, observed_features: np.ndarray, init_params: np.ndarray, lambda_param: float) -> tuple[np.ndarray, bool, int, str]:
    init = clip_params(init_params)
    objective = objective_factory(surrogate, observed_features, init, lambda_param)
    if minimize is None:
        return fallback_optimize(objective, init)
    result = minimize(
        objective,
        init,
        method="L-BFGS-B",
        bounds=[tuple(row) for row in PARAM_BOUNDS],
        options={"maxiter": 80, "ftol": 1.0e-10, "gtol": 1.0e-8},
    )
    return clip_params(result.x), bool(result.success), int(getattr(result, "nit", 0)), str(result.message)


def run_rows_for_lambda(
    rows: list[dict[str, str]],
    dataset: Any,
    observed_features: np.ndarray,
    surrogate: Any,
    indices: np.ndarray,
    lambda_param: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx in indices:
        row = rows[int(idx)]
        init_params = param_array_from_row(row, "pred")
        refined, success, iters, message = optimize_params(surrogate, observed_features[int(idx)], init_params, lambda_param)
        metrics = evaluate_params(dataset, int(row["sample_index"]), refined)
        before = feature_residual(surrogate, init_params, observed_features[int(idx)])
        after = feature_residual(surrogate, refined, observed_features[int(idx)])
        out.append(
            {
                "row_index": int(idx),
                "params": refined,
                "optimizer_success": success,
                "optimizer_iterations": iters,
                "optimizer_message": message,
                "feature_residual_mse_before": before,
                "feature_residual_mse_after": after,
                **metrics,
            }
        )
    return out


def validation_target_indices(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [i for i, row in enumerate(rows) if row["split"] == "val" and row["target_role"] == "refinement_target"],
        dtype=np.int64,
    )


def select_lambda(rows: list[dict[str, str]], dataset: Any, observed_features: np.ndarray, surrogate: Any) -> tuple[float, list[dict[str, Any]]]:
    val_idx = validation_target_indices(rows)
    if val_idx.size == 0:
        return 10.0, []
    selection_rows: list[dict[str, Any]] = []
    best: tuple[float, float] | None = None
    for lambda_param in LAMBDA_CANDIDATES:
        results = run_rows_for_lambda(rows, dataset, observed_features, surrogate, val_idx, lambda_param)
        baseline_rmse = np.asarray([as_float(rows[item["row_index"]]["baseline_profile_depth_rmse_m"]) for item in results], dtype=np.float64)
        baseline_dice = np.asarray([as_float(rows[item["row_index"]]["baseline_projected_mask_Dice"]) for item in results], dtype=np.float64)
        baseline_residual = np.asarray([item["feature_residual_mse_before"] for item in results], dtype=np.float64)
        refined_rmse = np.asarray([item["profile_depth_rmse_m"] for item in results], dtype=np.float64)
        refined_dice = np.asarray([item["projected_mask_Dice"] for item in results], dtype=np.float64)
        refined_residual = np.asarray([item["feature_residual_mse_after"] for item in results], dtype=np.float64)
        rmse_ratio = float(np.mean(refined_rmse) / max(float(np.mean(baseline_rmse)), 1.0e-30))
        dice_delta = float(np.mean(refined_dice) - np.mean(baseline_dice))
        residual_ratio = float(np.mean(refined_residual) / max(float(np.mean(baseline_residual)), 1.0e-30))
        score = rmse_ratio - 0.10 * dice_delta + 0.02 * residual_ratio
        row = {
            "lambda_param": lambda_param,
            "val_sample_count": len(results),
            "val_baseline_profile_rmse_mean_m": float(np.mean(baseline_rmse)),
            "val_refined_profile_rmse_mean_m": float(np.mean(refined_rmse)),
            "val_profile_rmse_delta_mean_m": float(np.mean(refined_rmse) - np.mean(baseline_rmse)),
            "val_baseline_dice_mean": float(np.mean(baseline_dice)),
            "val_refined_dice_mean": float(np.mean(refined_dice)),
            "val_dice_delta_mean": dice_delta,
            "val_feature_residual_before_mean": float(np.mean(baseline_residual)),
            "val_feature_residual_after_mean": float(np.mean(refined_residual)),
            "val_feature_residual_delta_mean": float(np.mean(refined_residual) - np.mean(baseline_residual)),
            "selection_score": score,
            "selection_policy": "validation_only_profile_rmse_dice_forward_residual_score",
            "selected": False,
        }
        selection_rows.append(row)
        if best is None or score < best[0]:
            best = (score, lambda_param)
    assert best is not None
    for row in selection_rows:
        row["selected"] = float(row["lambda_param"]) == float(best[1])
    return float(best[1]), selection_rows


def build_metric_row(
    row: dict[str, str],
    refined_params: np.ndarray,
    refine_meta: dict[str, Any],
    refined_metrics: dict[str, Any],
    lambda_param: float,
    selected_surrogate: str,
    refinement_applied: bool,
) -> dict[str, Any]:
    init_params = param_array_from_row(row, "pred")
    before_residual = float(refine_meta["feature_residual_mse_before"])
    after_residual = float(refine_meta["feature_residual_mse_after"])
    baseline_rmse = as_float(row["baseline_profile_depth_rmse_m"])
    baseline_er = as_float(row["baseline_Er_like_error"])
    baseline_iou = as_float(row["baseline_projected_mask_IoU"])
    baseline_dice = as_float(row["baseline_projected_mask_Dice"])
    baseline_area = as_float(row["baseline_area_error"])
    refined_rmse = float(refined_metrics["profile_depth_rmse_m"])
    refined_er = float(refined_metrics["Er_like_error"])
    refined_iou = float(refined_metrics["projected_mask_IoU"])
    refined_dice = float(refined_metrics["projected_mask_Dice"])
    refined_area = float(refined_metrics["area_error"])
    out: dict[str, Any] = {
        "sample_index": row["sample_index"],
        "sample_id": row["sample_id"],
        "split": row["split"],
        "shape_type": row["shape_type"],
        "topology_type": row["topology_type"],
        "representation_target": row["representation_target"],
        "target_role": row["target_role"],
        "diagnosis": row["diagnosis"],
        "rbc_representable": as_bool(row["rbc_representable"]),
        "include_in_success_gate": as_bool(row["include_in_success_gate"]),
        "include_in_rbc_control_gate": as_bool(row["include_in_rbc_control_gate"]),
        "include_as_negative_control": as_bool(row["include_as_negative_control"]),
        "refinement_applied": refinement_applied,
        "lambda_param": lambda_param if refinement_applied else "",
        "selected_surrogate": selected_surrogate,
        "optimizer_success": bool(refine_meta["optimizer_success"]) if refinement_applied else "",
        "optimizer_iterations": int(refine_meta["optimizer_iterations"]) if refinement_applied else "",
        "optimizer_message": refine_meta["optimizer_message"] if refinement_applied else "not_refined_excluded_negative_control",
        "feature_residual_mse_before": before_residual,
        "feature_residual_mse_after": after_residual,
        "feature_residual_delta": after_residual - before_residual,
        "feature_residual_improved": after_residual < before_residual,
        "baseline_profile_depth_rmse_m": baseline_rmse,
        "refined_profile_depth_rmse_m": refined_rmse,
        "profile_rmse_delta_m": refined_rmse - baseline_rmse,
        "profile_rmse_improved": refined_rmse < baseline_rmse,
        "baseline_Er_like_error": baseline_er,
        "refined_Er_like_error": refined_er,
        "Er_like_delta": refined_er - baseline_er,
        "Er_like_improved": refined_er < baseline_er,
        "baseline_projected_mask_IoU": baseline_iou,
        "refined_projected_mask_IoU": refined_iou,
        "IoU_delta": refined_iou - baseline_iou,
        "IoU_improved": refined_iou > baseline_iou,
        "baseline_projected_mask_Dice": baseline_dice,
        "refined_projected_mask_Dice": refined_dice,
        "Dice_delta": refined_dice - baseline_dice,
        "Dice_improved": refined_dice > baseline_dice,
        "baseline_area_error": baseline_area,
        "refined_area_error": refined_area,
        "area_error_delta": refined_area - baseline_area,
        "area_error_improved": refined_area < baseline_area,
        "baseline_pred_component_count": row["baseline_pred_component_count"],
        "refined_component_count": int(refined_metrics["component_count"]),
        "params_in_bounds": params_in_bounds(refined_params),
        "profile_nonnegative": bool(refined_metrics["profile_nonnegative"]),
        "test_time_label_used_for_refinement": False,
        "labels_used_for_metric_only": True,
    }
    for i, name in enumerate(PARAM_NAMES):
        out[f"initial_{name}"] = float(init_params[i])
        out[f"refined_{name}"] = float(refined_params[i])
    return out


def run_all_refinements() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    surrogate, rows, observed_features, _feature_names, _validation_rows = fit_selected_surrogate()
    dataset = load_surface_dataset(DATASET_ID, REGISTRY)
    observed_features_check, _ = observed_feature_matrix(dataset.delta_b)
    if not np.allclose(observed_features, observed_features_check):
        raise RuntimeError("observed feature extraction mismatch")
    selected_lambda, lambda_selection_rows = select_lambda(rows, dataset, observed_features, surrogate)
    metric_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        init_params = param_array_from_row(row, "pred")
        refinement_applied = as_bool(row["suitable_for_six_param_refinement"]) and not as_bool(row["include_as_negative_control"])
        if refinement_applied:
            refined_params, success, iters, message = optimize_params(surrogate, observed_features[idx], init_params, selected_lambda)
        else:
            refined_params, success, iters, message = clip_params(init_params), False, 0, "not_refined_excluded_negative_control"
        refined_metrics = evaluate_params(dataset, int(row["sample_index"]), refined_params)
        before = feature_residual(surrogate, init_params, observed_features[idx])
        after = feature_residual(surrogate, refined_params, observed_features[idx])
        meta = {
            "optimizer_success": success,
            "optimizer_iterations": iters,
            "optimizer_message": message,
            "feature_residual_mse_before": before,
            "feature_residual_mse_after": after,
        }
        metric_rows.append(build_metric_row(row, refined_params, meta, refined_metrics, selected_lambda, surrogate.candidate_id, refinement_applied))
    return metric_rows, lambda_selection_rows, surrogate


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    group_fields = ["target_role", "shape_type", "representation_target", "topology_type"]
    for group_field in group_fields:
        values = sorted({str(row[group_field]) for row in rows})
        for split in ["all", "train", "val", "test"]:
            for value in values:
                subset = [row for row in rows if str(row[group_field]) == value and (split == "all" or row["split"] == split)]
                if not subset:
                    continue
                out.append(
                    {
                        "group_field": group_field,
                        "group_value": value,
                        "split": split,
                        "sample_count": len(subset),
                        "refinement_applied_count": sum(bool(row["refinement_applied"]) for row in subset),
                        "baseline_profile_rmse_mean_m": metric_mean(subset, "baseline_profile_depth_rmse_m"),
                        "refined_profile_rmse_mean_m": metric_mean(subset, "refined_profile_depth_rmse_m"),
                        "profile_rmse_delta_mean_m": metric_mean(subset, "profile_rmse_delta_m"),
                        "baseline_Er_like_mean": metric_mean(subset, "baseline_Er_like_error"),
                        "refined_Er_like_mean": metric_mean(subset, "refined_Er_like_error"),
                        "Er_like_delta_mean": metric_mean(subset, "Er_like_delta"),
                        "baseline_IoU_mean": metric_mean(subset, "baseline_projected_mask_IoU"),
                        "refined_IoU_mean": metric_mean(subset, "refined_projected_mask_IoU"),
                        "IoU_delta_mean": metric_mean(subset, "IoU_delta"),
                        "baseline_Dice_mean": metric_mean(subset, "baseline_projected_mask_Dice"),
                        "refined_Dice_mean": metric_mean(subset, "refined_projected_mask_Dice"),
                        "Dice_delta_mean": metric_mean(subset, "Dice_delta"),
                        "baseline_area_error_mean": metric_mean(subset, "baseline_area_error"),
                        "refined_area_error_mean": metric_mean(subset, "refined_area_error"),
                        "area_error_delta_mean": metric_mean(subset, "area_error_delta"),
                        "feature_residual_before_mean": metric_mean(subset, "feature_residual_mse_before"),
                        "feature_residual_after_mean": metric_mean(subset, "feature_residual_mse_after"),
                        "feature_residual_delta_mean": metric_mean(subset, "feature_residual_delta"),
                        "profile_rmse_improved_rate": float(np.mean([bool(row["profile_rmse_improved"]) for row in subset])),
                        "forward_residual_improved_rate": float(np.mean([bool(row["feature_residual_improved"]) for row in subset])),
                    }
                )
    return out


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["target_role"] != "refinement_target":
            continue
        modes: list[str] = []
        if not bool(row["profile_rmse_improved"]):
            modes.append("profile_rmse_not_improved")
        if not bool(row["Dice_improved"]):
            modes.append("dice_not_improved")
        if bool(row["feature_residual_improved"]) and not bool(row["profile_rmse_improved"]):
            modes.append("forward_residual_improved_without_profile_improvement")
        if not bool(row["params_in_bounds"]) or not bool(row["profile_nonnegative"]):
            modes.append("nonphysical_refined_params")
        if modes:
            out.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "shape_type": row["shape_type"],
                    "target_role": row["target_role"],
                    "failure_mode": "|".join(modes),
                    "feature_residual_mse_before": row["feature_residual_mse_before"],
                    "feature_residual_mse_after": row["feature_residual_mse_after"],
                    "baseline_profile_depth_rmse_m": row["baseline_profile_depth_rmse_m"],
                    "refined_profile_depth_rmse_m": row["refined_profile_depth_rmse_m"],
                    "baseline_projected_mask_Dice": row["baseline_projected_mask_Dice"],
                    "refined_projected_mask_Dice": row["refined_projected_mask_Dice"],
                    "notes": "failure case counted only inside the 25.5 diagnostic; no baseline transition",
                }
            )
    out.sort(key=lambda item: float(item["refined_profile_depth_rmse_m"]) - float(item["baseline_profile_depth_rmse_m"]), reverse=True)
    return out[:40]


def write_summary(rows: list[dict[str, Any]], lambda_rows: list[dict[str, Any]], surrogate: Any) -> None:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    refs = [row for row in rows if row["target_role"] == "already_pass_reference"]
    negative = [row for row in rows if row["target_role"] == "excluded_negative_control"]
    selected_lambda = ""
    if lambda_rows:
        selected = [row for row in lambda_rows if row.get("selected")]
        selected_lambda = selected[0]["lambda_param"] if selected else ""
    lines = [
        "25.5 surface RBC forward-consistency refinement run",
        "",
        f"selected_surrogate: {surrogate.candidate_id}",
        f"selected_lambda_param: {selected_lambda}",
        "optimizer: L-BFGS-B when scipy is available; deterministic coordinate-search fallback otherwise",
        "objective_inputs: observed delta_b-derived features plus frozen 20.85 predicted six params",
        "objective_excludes: true depth, true mask, shape labels, oracle params, test labels",
        f"lambda_selection_rows_recorded_in_summary_only: {len(lambda_rows)}",
        "",
        f"refinement_target_count: {len(targets)}",
        f"already_pass_reference_count: {len(refs)}",
        f"excluded_negative_control_count: {len(negative)}",
        f"target_baseline_profile_rmse_mean_m: {metric_mean(targets, 'baseline_profile_depth_rmse_m'):.12g}",
        f"target_refined_profile_rmse_mean_m: {metric_mean(targets, 'refined_profile_depth_rmse_m'):.12g}",
        f"target_profile_rmse_delta_mean_m: {metric_mean(targets, 'profile_rmse_delta_m'):.12g}",
        f"target_baseline_Er_like_mean: {metric_mean(targets, 'baseline_Er_like_error'):.12g}",
        f"target_refined_Er_like_mean: {metric_mean(targets, 'refined_Er_like_error'):.12g}",
        f"target_baseline_IoU_mean: {metric_mean(targets, 'baseline_projected_mask_IoU'):.12g}",
        f"target_refined_IoU_mean: {metric_mean(targets, 'refined_projected_mask_IoU'):.12g}",
        f"target_baseline_Dice_mean: {metric_mean(targets, 'baseline_projected_mask_Dice'):.12g}",
        f"target_refined_Dice_mean: {metric_mean(targets, 'refined_projected_mask_Dice'):.12g}",
        f"target_feature_residual_before_mean: {metric_mean(targets, 'feature_residual_mse_before'):.12g}",
        f"target_feature_residual_after_mean: {metric_mean(targets, 'feature_residual_mse_after'):.12g}",
        "",
        f"rbc_like_control_baseline_rmse_mean_m: {metric_mean([row for row in rows if row['shape_type'] == 'rbc_like_smooth_pit'], 'baseline_profile_depth_rmse_m'):.12g}",
        f"rbc_like_control_refined_rmse_mean_m: {metric_mean([row for row in rows if row['shape_type'] == 'rbc_like_smooth_pit'], 'refined_profile_depth_rmse_m'):.12g}",
        "multi_pit_handling: excluded_negative_control rows are not refined and are not success-credit eligible.",
        f"metrics_csv: {METRICS}",
        f"group_csv: {GROUP}",
        f"failure_cases_csv: {FAILURES}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows, lambda_rows, surrogate = run_all_refinements()
    group = group_rows(rows)
    failures = failure_rows(rows)
    write_csv(METRICS, rows, METRIC_FIELDS)
    write_csv(GROUP, group, GROUP_FIELDS)
    write_csv(FAILURES, failures, FAILURE_FIELDS)
    write_summary(rows, lambda_rows, surrogate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
