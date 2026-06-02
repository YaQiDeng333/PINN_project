#!/usr/bin/env python
"""Run the 25.6 fixed-protocol formal benchmark for surface refinement.

This script replays the 25.5 F0/R1 candidate with fixed configuration:
ridge_param_only_linear_alpha_10 and lambda_param=1.0. It does not search
hyperparameters, train the main neural model, run COMSOL, or write data/NPZ.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_surface_shape_extension_rbc_oracle_fit import DATASET_ID, ROOT, load_surface_dataset
from build_surface_forward_refinement_target_set import (
    PARAM_NAMES,
    REGISTRY,
    TARGET_MATERIALIZED,
    as_bool,
    as_float,
    read_csv,
    write_csv,
)
from fit_surface_feature_space_forward_surrogate import (
    fit_candidate,
    observed_feature_matrix,
    residual_mse_norm,
    row_indices,
)
from run_surface_rbc_forward_consistency_refinement import (
    METRIC_FIELDS as REFINEMENT_BASE_FIELDS,
    build_metric_row,
    clip_params,
    evaluate_params,
    failure_rows as diagnostic_failure_rows,
    feature_residual,
    metric_mean,
    optimize_params,
    params_in_bounds,
)


PREFLIGHT_SUMMARY = ROOT / "results/summaries/surface_forward_refinement_formal_benchmark_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/surface_forward_refinement_formal_benchmark_summary.txt"
METRICS = ROOT / "results/metrics/surface_forward_refinement_formal_benchmark_metrics.csv"
BY_SPLIT = ROOT / "results/metrics/surface_forward_refinement_formal_benchmark_by_split.csv"
BY_SHAPE = ROOT / "results/metrics/surface_forward_refinement_formal_benchmark_by_shape.csv"
FAILURES = ROOT / "results/metrics/surface_forward_refinement_formal_benchmark_failure_cases.csv"

PILOT_MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
BASELINE_ARTIFACT = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
ORACLE_METRICS = ROOT / "results/metrics/surface_shape_extension_rbc_oracle_fit_metrics.csv"
BASELINE_METRICS = ROOT / "results/metrics/surface_shape_extension_current_baseline_metrics.csv"
DIAGNOSIS_MATRIX = ROOT / "results/metrics/surface_shape_extension_oracle_vs_baseline_matrix.csv"
REFINEMENT_25_5_METRICS = ROOT / "results/metrics/surface_forward_refinement_metrics.csv"
REFINEMENT_25_5_GATES = ROOT / "results/metrics/surface_forward_refinement_acceptance_gate_results.csv"
REFINEMENT_25_5_SURROGATE = ROOT / "results/metrics/surface_feature_space_forward_surrogate_validation.csv"
PROFILE_GENERATOR = ROOT / "scripts/load_true_3d_rbc_pilot_dataset.py"
SCRIPT_25_5_SURROGATE = ROOT / "scripts/fit_surface_feature_space_forward_surrogate.py"
SCRIPT_25_5_REFINEMENT = ROOT / "scripts/run_surface_rbc_forward_consistency_refinement.py"

FIXED_SURROGATE_ID = "ridge_param_only_linear_alpha_10"
FIXED_DESCRIPTOR_KIND = "param_only"
FIXED_FEATURE_MODE = "linear"
FIXED_ALPHA = 10.0
FIXED_LAMBDA = 1.0

FORBIDDEN_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]

FORMAL_FIELDS = [
    *REFINEMENT_BASE_FIELDS,
    "formal_benchmark_protocol",
    "fixed_surrogate_protocol",
    "fixed_lambda_protocol",
    "hyperparameter_search_performed",
    "main_neural_training_performed",
    "comsol_run_performed",
    "data_npz_mutation_performed",
    "oracle_profile_depth_rmse_m",
    "oracle_Er_like_error",
    "oracle_projected_mask_IoU",
    "oracle_projected_mask_Dice",
    "oracle_area_error",
    "refined_minus_oracle_profile_rmse_m",
    "refined_minus_oracle_Er_like",
    "refined_minus_oracle_IoU",
    "refined_minus_oracle_Dice",
    *[f"oracle_{name}" for name in PARAM_NAMES],
]

GROUP_FIELDS = [
    "split",
    "shape_type",
    "target_role",
    "sample_count",
    "refinement_applied_count",
    "baseline_profile_rmse_mean_m",
    "refined_profile_rmse_mean_m",
    "oracle_profile_rmse_mean_m",
    "baseline_Er_like_mean",
    "refined_Er_like_mean",
    "oracle_Er_like_mean",
    "baseline_IoU_mean",
    "refined_IoU_mean",
    "oracle_IoU_mean",
    "baseline_Dice_mean",
    "refined_Dice_mean",
    "oracle_Dice_mean",
    "feature_residual_before_mean",
    "feature_residual_after_mean",
    "profile_rmse_improved_rate",
    "forward_residual_improved_rate",
    "success_credit_allowed",
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
    "oracle_profile_depth_rmse_m",
    "baseline_projected_mask_Dice",
    "refined_projected_mask_Dice",
    "oracle_projected_mask_Dice",
    "notes",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception as exc:
        return f"git_error:{exc}"


def required_inputs() -> list[Path]:
    return [
        REGISTRY,
        PILOT_MANIFEST,
        BASELINE_ARTIFACT,
        ORACLE_METRICS,
        BASELINE_METRICS,
        DIAGNOSIS_MATRIX,
        TARGET_MATERIALIZED,
        REFINEMENT_25_5_METRICS,
        REFINEMENT_25_5_GATES,
        REFINEMENT_25_5_SURROGATE,
        PROFILE_GENERATOR,
        SCRIPT_25_5_SURROGATE,
        SCRIPT_25_5_REFINEMENT,
    ]


def validate_fixed_protocol_materials() -> dict[str, Any]:
    missing = [path for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing 25.6 fixed-protocol input(s): " + ", ".join(str(path) for path in missing))
    selected = [row for row in read_csv(REFINEMENT_25_5_SURROGATE) if row["candidate_id"] == FIXED_SURROGATE_ID]
    if len(selected) != 1 or not as_bool(selected[0]["selected"]):
        raise RuntimeError(f"25.5 selected surrogate material is not fixed to {FIXED_SURROGATE_ID}")
    gate_rows = read_csv(REFINEMENT_25_5_GATES)
    if not gate_rows or not all(as_bool(row["pass"]) for row in gate_rows):
        raise RuntimeError("25.5 acceptance gates are missing or not all PASS")
    run_rows = read_csv(REFINEMENT_25_5_METRICS)
    lambdas = {row["lambda_param"] for row in run_rows if row.get("lambda_param")}
    if lambdas != {"1.0"}:
        raise RuntimeError(f"25.5 lambda protocol mismatch: {lambdas}")
    target_rows = [row for row in read_csv(TARGET_MATERIALIZED) if row["target_role"] == "refinement_target"]
    if len(target_rows) != 82:
        raise RuntimeError(f"expected 82 refinement targets, observed {len(target_rows)}")
    return {
        "selected_surrogate_row": selected[0],
        "gate_count": len(gate_rows),
        "target_count": len(target_rows),
        "lambda_values": sorted(lambdas),
    }


def write_preflight_summary(protocol: dict[str, Any]) -> None:
    pilot = read_json(PILOT_MANIFEST)
    baseline = read_json(BASELINE_ARTIFACT)
    dataset = load_surface_dataset(DATASET_ID, REGISTRY)
    forbidden_diff = git_value(["diff", "--name-only", "--", *FORBIDDEN_PATHS])
    staged_forbidden = git_value(["diff", "--cached", "--name-only", "--", *FORBIDDEN_PATHS])
    target_roles = Counter(row["target_role"] for row in read_csv(TARGET_MATERIALIZED))
    lines = [
        "25.6 surface forward-refinement formal benchmark preflight",
        "",
        "scope: fixed-protocol formal benchmark; no baseline transition.",
        f"dataset_id: {pilot.get('dataset_id')}",
        f"dataset_manifest: {PILOT_MANIFEST}",
        f"explicit_registry_path: {REGISTRY}",
        f"npz_sha256_verified_by_loader: true",
        f"dataset_sample_count: {len(dataset.sample_ids)}",
        f"split_counts: {dict(Counter(str(x) for x in dataset.split))}",
        f"baseline_artifact_id: {baseline.get('artifact_id')}",
        f"baseline_model_family: {baseline.get('model_family')}",
        "",
        "fixed_protocol:",
        f"- selected_surrogate: {FIXED_SURROGATE_ID}",
        f"- descriptor_kind: {FIXED_DESCRIPTOR_KIND}",
        f"- feature_mode: {FIXED_FEATURE_MODE}",
        f"- alpha: {FIXED_ALPHA:g}",
        f"- lambda_param: {FIXED_LAMBDA:g}",
        "- train-only surrogate/scaler fit; validation-only sanity check; test final only.",
        "- no surrogate family, alpha, loss weight, or optimizer hyperparameter search.",
        "",
        f"target_role_counts: {dict(target_roles)}",
        f"25.5_acceptance_gate_count: {protocol['gate_count']}",
        f"25.5_selected_surrogate_val_mse_norm: {protocol['selected_surrogate_row'].get('val_feature_mse_norm')}",
        "",
        f"forbidden_diff_present: {bool(forbidden_diff)}",
        "forbidden_diff_paths: " + (forbidden_diff if forbidden_diff else "none"),
        f"staged_forbidden_present: {bool(staged_forbidden)}",
        "staged_forbidden_paths: " + (staged_forbidden if staged_forbidden else "none"),
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_fixed_surrogate(rows: list[dict[str, str]], observed_features: np.ndarray, observed_feature_names: list[str]) -> Any:
    train_idx = row_indices(rows, split="train", representable_only=True)
    if len(train_idx) == 0:
        raise RuntimeError("train-split representable rows required for fixed surrogate fit")
    return fit_candidate(
        rows=rows,
        observed_features=observed_features,
        train_idx=train_idx,
        descriptor_kind=FIXED_DESCRIPTOR_KIND,
        feature_mode=FIXED_FEATURE_MODE,
        alpha=FIXED_ALPHA,
        observed_feature_names=observed_feature_names,
    )


def param_array_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([as_float(row[f"{prefix}_{name}"]) for name in PARAM_NAMES], dtype=np.float64)


def enrich_with_oracle(base_row: dict[str, Any], materialized_row: dict[str, str]) -> dict[str, Any]:
    out = dict(base_row)
    oracle_rmse = as_float(materialized_row["oracle_profile_depth_rmse_m"])
    oracle_er = as_float(materialized_row["oracle_Er_like_error"])
    oracle_iou = as_float(materialized_row["oracle_projected_mask_IoU"])
    oracle_dice = as_float(materialized_row["oracle_projected_mask_Dice"])
    oracle_area = as_float(materialized_row["oracle_area_error"])
    out.update(
        {
            "formal_benchmark_protocol": "25.6_fixed_replay_of_25.5_F0_R1",
            "fixed_surrogate_protocol": FIXED_SURROGATE_ID,
            "fixed_lambda_protocol": FIXED_LAMBDA if out["refinement_applied"] else "",
            "hyperparameter_search_performed": False,
            "main_neural_training_performed": False,
            "comsol_run_performed": False,
            "data_npz_mutation_performed": False,
            "oracle_profile_depth_rmse_m": oracle_rmse,
            "oracle_Er_like_error": oracle_er,
            "oracle_projected_mask_IoU": oracle_iou,
            "oracle_projected_mask_Dice": oracle_dice,
            "oracle_area_error": oracle_area,
            "refined_minus_oracle_profile_rmse_m": float(out["refined_profile_depth_rmse_m"]) - oracle_rmse,
            "refined_minus_oracle_Er_like": float(out["refined_Er_like_error"]) - oracle_er,
            "refined_minus_oracle_IoU": float(out["refined_projected_mask_IoU"]) - oracle_iou,
            "refined_minus_oracle_Dice": float(out["refined_projected_mask_Dice"]) - oracle_dice,
        }
    )
    for name in PARAM_NAMES:
        out[f"oracle_{name}"] = as_float(materialized_row[f"oracle_{name}"])
    return out


def run_formal_benchmark() -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    protocol = validate_fixed_protocol_materials()
    write_preflight_summary(protocol)
    rows = read_csv(TARGET_MATERIALIZED)
    dataset = load_surface_dataset(DATASET_ID, REGISTRY)
    observed_features, observed_feature_names = observed_feature_matrix(dataset.delta_b)
    surrogate = fit_fixed_surrogate(rows, observed_features, observed_feature_names)
    if surrogate.candidate_id != FIXED_SURROGATE_ID:
        raise RuntimeError(f"fixed surrogate id mismatch: {surrogate.candidate_id}")
    metric_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        init_params = param_array_from_row(row, "pred")
        refinement_applied = as_bool(row["suitable_for_six_param_refinement"]) and not as_bool(row["include_as_negative_control"])
        if refinement_applied:
            refined_params, success, iters, message = optimize_params(surrogate, observed_features[idx], init_params, FIXED_LAMBDA)
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
        base = build_metric_row(row, refined_params, meta, refined_metrics, FIXED_LAMBDA, surrogate.candidate_id, refinement_applied)
        metric_rows.append(enrich_with_oracle(base, row))
    return metric_rows, surrogate, protocol


def group_summary(rows: list[dict[str, Any]], split: str, shape_type: str, target_role: str) -> dict[str, Any]:
    subset = [
        row
        for row in rows
        if (split == "all" or row["split"] == split)
        and (shape_type == "all" or row["shape_type"] == shape_type)
        and (target_role == "all" or row["target_role"] == target_role)
    ]
    return {
        "split": split,
        "shape_type": shape_type,
        "target_role": target_role,
        "sample_count": len(subset),
        "refinement_applied_count": sum(bool(row["refinement_applied"]) for row in subset),
        "baseline_profile_rmse_mean_m": metric_mean(subset, "baseline_profile_depth_rmse_m"),
        "refined_profile_rmse_mean_m": metric_mean(subset, "refined_profile_depth_rmse_m"),
        "oracle_profile_rmse_mean_m": metric_mean(subset, "oracle_profile_depth_rmse_m"),
        "baseline_Er_like_mean": metric_mean(subset, "baseline_Er_like_error"),
        "refined_Er_like_mean": metric_mean(subset, "refined_Er_like_error"),
        "oracle_Er_like_mean": metric_mean(subset, "oracle_Er_like_error"),
        "baseline_IoU_mean": metric_mean(subset, "baseline_projected_mask_IoU"),
        "refined_IoU_mean": metric_mean(subset, "refined_projected_mask_IoU"),
        "oracle_IoU_mean": metric_mean(subset, "oracle_projected_mask_IoU"),
        "baseline_Dice_mean": metric_mean(subset, "baseline_projected_mask_Dice"),
        "refined_Dice_mean": metric_mean(subset, "refined_projected_mask_Dice"),
        "oracle_Dice_mean": metric_mean(subset, "oracle_projected_mask_Dice"),
        "feature_residual_before_mean": metric_mean(subset, "feature_residual_mse_before"),
        "feature_residual_after_mean": metric_mean(subset, "feature_residual_mse_after"),
        "profile_rmse_improved_rate": float(np.mean([bool(row["profile_rmse_improved"]) for row in subset])) if subset else float("nan"),
        "forward_residual_improved_rate": float(np.mean([bool(row["feature_residual_improved"]) for row in subset])) if subset else float("nan"),
        "success_credit_allowed": bool(subset) and all(bool(row["include_in_success_gate"]) for row in subset),
    }


def build_by_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["all", "train", "val", "test"]:
        for role in ["all", "refinement_target", "already_pass_reference", "excluded_negative_control"]:
            summary = group_summary(rows, split, "all", role)
            if summary["sample_count"]:
                out.append(summary)
    return out


def build_by_shape(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shape in sorted({str(row["shape_type"]) for row in rows}):
        for role in ["all", "refinement_target", "already_pass_reference", "excluded_negative_control"]:
            summary = group_summary(rows, "all", shape, role)
            if summary["sample_count"]:
                out.append(summary)
    return out


def formal_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    "oracle_profile_depth_rmse_m": row["oracle_profile_depth_rmse_m"],
                    "baseline_projected_mask_Dice": row["baseline_projected_mask_Dice"],
                    "refined_projected_mask_Dice": row["refined_projected_mask_Dice"],
                    "oracle_projected_mask_Dice": row["oracle_projected_mask_Dice"],
                    "notes": "formal 25.6 failure case; no baseline transition",
                }
            )
    out.sort(key=lambda item: float(item["refined_profile_depth_rmse_m"]) - float(item["baseline_profile_depth_rmse_m"]), reverse=True)
    return out[:40]


def write_summary_file(rows: list[dict[str, Any]], surrogate: Any, protocol: dict[str, Any]) -> None:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in rows if row["shape_type"] == "rbc_like_smooth_pit"]
    negative = [row for row in rows if row["target_role"] == "excluded_negative_control"]
    lines = [
        "25.6 surface forward-refinement formal benchmark",
        "",
        f"fixed_protocol_replayed: true",
        f"selected_surrogate: {surrogate.candidate_id}",
        f"surrogate_family: F0_feature_space_consistency",
        f"refinement_strategy: R1_low_dim_param_refinement",
        f"lambda_param: {FIXED_LAMBDA:g}",
        "hyperparameter_search_performed: false",
        "main_neural_training_performed: false",
        "COMSOL_run_performed: false",
        "data_npz_mutation_performed: false",
        "",
        f"target_subset_count: {len(targets)}",
        f"negative_control_count: {len(negative)}",
        f"target_baseline_profile_rmse_mean_m: {metric_mean(targets, 'baseline_profile_depth_rmse_m'):.12g}",
        f"target_refined_profile_rmse_mean_m: {metric_mean(targets, 'refined_profile_depth_rmse_m'):.12g}",
        f"target_oracle_profile_rmse_mean_m: {metric_mean(targets, 'oracle_profile_depth_rmse_m'):.12g}",
        f"target_baseline_Er_like_mean: {metric_mean(targets, 'baseline_Er_like_error'):.12g}",
        f"target_refined_Er_like_mean: {metric_mean(targets, 'refined_Er_like_error'):.12g}",
        f"target_oracle_Er_like_mean: {metric_mean(targets, 'oracle_Er_like_error'):.12g}",
        f"target_baseline_IoU_mean: {metric_mean(targets, 'baseline_projected_mask_IoU'):.12g}",
        f"target_refined_IoU_mean: {metric_mean(targets, 'refined_projected_mask_IoU'):.12g}",
        f"target_oracle_IoU_mean: {metric_mean(targets, 'oracle_projected_mask_IoU'):.12g}",
        f"target_baseline_Dice_mean: {metric_mean(targets, 'baseline_projected_mask_Dice'):.12g}",
        f"target_refined_Dice_mean: {metric_mean(targets, 'refined_projected_mask_Dice'):.12g}",
        f"target_oracle_Dice_mean: {metric_mean(targets, 'oracle_projected_mask_Dice'):.12g}",
        f"target_feature_residual_before_mean: {metric_mean(targets, 'feature_residual_mse_before'):.12g}",
        f"target_feature_residual_after_mean: {metric_mean(targets, 'feature_residual_mse_after'):.12g}",
        "",
        f"rbc_like_control_baseline_rmse_mean_m: {metric_mean(rbc_like, 'baseline_profile_depth_rmse_m'):.12g}",
        f"rbc_like_control_refined_rmse_mean_m: {metric_mean(rbc_like, 'refined_profile_depth_rmse_m'):.12g}",
        f"rbc_like_control_baseline_Dice_mean: {metric_mean(rbc_like, 'baseline_projected_mask_Dice'):.12g}",
        f"rbc_like_control_refined_Dice_mean: {metric_mean(rbc_like, 'refined_projected_mask_Dice'):.12g}",
        "multi_pit_handling: excluded_negative_control; no refinement applied; no RBC success credit.",
        "",
        "interpretation: formal benchmark confirms a surface forward-refinement candidate for rbc_representable_but_model_fail rows only; it is not a baseline replacement.",
        f"metrics_csv: {METRICS}",
        f"by_split_csv: {BY_SPLIT}",
        f"by_shape_csv: {BY_SHAPE}",
        f"failure_cases_csv: {FAILURES}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows, surrogate, protocol = run_formal_benchmark()
    by_split = build_by_split(rows)
    by_shape = build_by_shape(rows)
    failures = formal_failure_rows(rows)
    write_csv(METRICS, rows, FORMAL_FIELDS)
    write_csv(BY_SPLIT, by_split, GROUP_FIELDS)
    write_csv(BY_SHAPE, by_shape, GROUP_FIELDS)
    write_csv(FAILURES, failures, FAILURE_FIELDS)
    write_summary_file(rows, surrogate, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
