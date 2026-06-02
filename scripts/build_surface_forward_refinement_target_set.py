#!/usr/bin/env python
"""Materialize the 25.5 surface forward-refinement target set.

This script reads only existing 25.2/25.3/25.4 artifacts. It does not run
COMSOL, train a neural model, write data/NPZ files, or update baseline docs.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_surface_shape_extension_rbc_oracle_fit import (
    DATASET_ID,
    ROOT,
    load_surface_dataset,
)


REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
PILOT_MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
BASELINE_ARTIFACT = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
TARGET_SET_25_4 = ROOT / "results/metrics/surface_forward_refinement_target_set.csv"
DIAGNOSIS_MATRIX = ROOT / "results/metrics/surface_shape_extension_oracle_vs_baseline_matrix.csv"
BASELINE_METRICS = ROOT / "results/metrics/surface_shape_extension_current_baseline_metrics.csv"
ORACLE_METRICS = ROOT / "results/metrics/surface_shape_extension_rbc_oracle_fit_metrics.csv"
SURROGATE_PLAN = ROOT / "results/metrics/surface_forward_consistency_surrogate_matrix.csv"
REFINEMENT_PLAN = ROOT / "results/metrics/surface_rbc_parameter_refinement_strategy_matrix.csv"
ACCEPTANCE_PLAN = ROOT / "results/metrics/surface_forward_refinement_acceptance_gate_matrix.csv"

PREFLIGHT_SUMMARY = ROOT / "results/summaries/surface_forward_refinement_diagnostic_preflight_summary.txt"
TARGET_SUMMARY = ROOT / "results/summaries/surface_forward_refinement_target_set_materialized_summary.txt"
TARGET_MATERIALIZED = ROOT / "results/metrics/surface_forward_refinement_target_set_materialized.csv"

PARAM_NAMES = ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW"]
FORBIDDEN_DIFF_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]

TARGET_FIELDS = [
    "sample_index",
    "sample_id",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "rbc_compatible",
    "component_count",
    "diagnosis",
    "target_role",
    "suitable_for_six_param_refinement",
    "include_in_success_gate",
    "include_in_rbc_control_gate",
    "include_as_negative_control",
    "exclude_reason",
    "failure_reason",
    "model_pass",
    "rbc_representable",
    "baseline_profile_depth_rmse_m",
    "baseline_Er_like_error",
    "baseline_projected_mask_IoU",
    "baseline_projected_mask_Dice",
    "baseline_area_error",
    "baseline_mask_center_error_px",
    "baseline_component_recall_proxy",
    "baseline_merge_component_proxy",
    "baseline_true_component_count",
    "baseline_pred_component_count",
    "oracle_profile_depth_rmse_m",
    "oracle_Er_like_error",
    "oracle_projected_mask_IoU",
    "oracle_projected_mask_Dice",
    "oracle_area_error",
    "oracle_mask_center_error_px",
    "oracle_component_count",
    "oracle_minus_baseline_rmse_m",
    "baseline_minus_oracle_rmse_m",
    "label_L_m",
    "label_W_m",
    "label_D_m",
    "label_aspect_ratio",
    "label_rotation_angle",
    "label_asymmetry_score",
    "label_edge_steepness",
    "feature_input_policy",
    "label_use_policy",
    "test_time_label_use_allowed",
    *[f"pred_{name}" for name in PARAM_NAMES],
    *[f"oracle_{name}" for name in PARAM_NAMES],
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def as_float(value: Any) -> float:
    if value in {"", None}:
        return float("nan")
    return float(value)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            val = float(row[key])
        except Exception:
            continue
        if np.isfinite(val):
            values.append(val)
    return float(np.mean(values)) if values else float("nan")


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
        TARGET_SET_25_4,
        DIAGNOSIS_MATRIX,
        BASELINE_METRICS,
        ORACLE_METRICS,
        SURROGATE_PLAN,
        REFINEMENT_PLAN,
        ACCEPTANCE_PLAN,
    ]


def write_preflight_summary() -> None:
    missing = [path for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing 25.5 preflight input(s): " + ", ".join(str(path) for path in missing))
    pilot = read_json(PILOT_MANIFEST)
    baseline = read_json(BASELINE_ARTIFACT)
    dataset = load_surface_dataset(DATASET_ID, REGISTRY)
    forbidden_diff = git_value(["diff", "--name-only", "--", *FORBIDDEN_DIFF_PATHS])
    staged = git_value(["diff", "--cached", "--name-only"])
    lines = [
        "25.5 surface feature-space forward-refinement diagnostic preflight",
        "",
        "scope: execute lightweight feature-space diagnostic only; no COMSOL, no main neural training, no data/NPZ mutation, no CURRENT_BASELINE.md update.",
        f"dataset_id: {pilot.get('dataset_id')}",
        f"dataset_manifest: {PILOT_MANIFEST}",
        f"registry_path: {REGISTRY}",
        f"explicit_manifest_loaded: {dataset.manifest.get('dataset_id') == DATASET_ID}",
        f"npz_sha256_verified_by_loader: true",
        f"dataset_allowed_use: {pilot.get('allowed_use')}",
        f"dataset_forbidden_use: {pilot.get('forbidden_use')}",
        f"sample_count: {len(dataset.sample_ids)}",
        f"split_counts: {dict(Counter(str(x) for x in dataset.split))}",
        f"axes: {pilot.get('axes')}",
        f"sensor_z_m: {pilot.get('sensor_z_m')}",
        f"baseline_artifact_id: {baseline.get('artifact_id')}",
        f"baseline_model_family: {baseline.get('model_family')}",
        "frozen_baseline_source: results/metrics/surface_shape_extension_current_baseline_metrics.csv",
        "target_role_source: results/metrics/surface_forward_refinement_target_set.csv",
        "oracle_source: results/metrics/surface_shape_extension_rbc_oracle_fit_metrics.csv",
        "",
        "split_policy:",
        "- Surrogate/scaler fit uses train split only.",
        "- Hyperparameter selection uses validation split only.",
        "- Test split is final evaluation only.",
        "",
        "feature_input_policy:",
        "- Test-time refinement objective may use observed delta_b-derived features plus frozen 20.85 predicted six params only.",
        "- Labels and oracle params are allowed only for oracle fit, surrogate training targets, validation selection metrics, and final reporting metrics.",
        "",
        f"forbidden_diff_present: {bool(forbidden_diff)}",
        "forbidden_diff_paths: " + (forbidden_diff if forbidden_diff else "none"),
        "currently_staged_paths: " + (staged if staged else "none"),
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rows() -> list[dict[str, Any]]:
    dataset = load_surface_dataset(DATASET_ID, REGISTRY)
    sample_to_index = {str(sample_id): idx for idx, sample_id in enumerate(dataset.sample_ids)}
    target_by_id = {row["sample_id"]: row for row in read_csv(TARGET_SET_25_4)}
    baseline_by_id = {row["sample_id"]: row for row in read_csv(BASELINE_METRICS)}
    oracle_by_id = {row["sample_id"]: row for row in read_csv(ORACLE_METRICS)}
    diagnosis_by_id = {row["sample_id"]: row for row in read_csv(DIAGNOSIS_MATRIX)}
    rows: list[dict[str, Any]] = []
    for sample_id, idx in sample_to_index.items():
        if sample_id not in target_by_id or sample_id not in baseline_by_id or sample_id not in oracle_by_id:
            raise RuntimeError(f"missing joined row for sample_id={sample_id}")
        target = target_by_id[sample_id]
        baseline = baseline_by_id[sample_id]
        oracle = oracle_by_id[sample_id]
        diagnosis = diagnosis_by_id.get(sample_id, {})
        baseline_rmse = as_float(baseline["baseline_profile_depth_rmse_m"])
        oracle_rmse = as_float(oracle["oracle_profile_depth_rmse_m"])
        row: dict[str, Any] = {
            "sample_index": idx,
            "sample_id": sample_id,
            "split": str(dataset.split[idx]),
            "shape_type": str(dataset.shape_type[idx]),
            "topology_type": str(dataset.topology_type[idx]),
            "representation_target": str(dataset.representation_target[idx]),
            "rbc_compatible": bool(dataset.rbc_compatible[idx]),
            "component_count": int(dataset.component_count[idx]),
            "diagnosis": target["diagnosis"],
            "target_role": target["target_role"],
            "suitable_for_six_param_refinement": as_bool(target["suitable_for_six_param_refinement"]),
            "include_in_success_gate": as_bool(target["include_in_success_gate"]),
            "include_in_rbc_control_gate": as_bool(target["include_in_rbc_control_gate"]),
            "include_as_negative_control": as_bool(target["include_as_negative_control"]),
            "exclude_reason": target.get("exclude_reason", ""),
            "failure_reason": target.get("failure_reason", diagnosis.get("primary_reason", "")),
            "model_pass": as_bool(baseline["model_pass"]),
            "rbc_representable": as_bool(baseline["rbc_representable"]),
            "baseline_profile_depth_rmse_m": baseline_rmse,
            "baseline_Er_like_error": as_float(baseline["baseline_Er_like_error"]),
            "baseline_projected_mask_IoU": as_float(baseline["baseline_projected_mask_IoU"]),
            "baseline_projected_mask_Dice": as_float(baseline["baseline_projected_mask_Dice"]),
            "baseline_area_error": as_float(baseline["baseline_area_error"]),
            "baseline_mask_center_error_px": as_float(baseline["baseline_mask_center_error_px"]),
            "baseline_component_recall_proxy": baseline.get("component_recall_proxy", ""),
            "baseline_merge_component_proxy": baseline.get("merge_component_proxy", ""),
            "baseline_true_component_count": baseline.get("true_component_count", ""),
            "baseline_pred_component_count": baseline.get("pred_component_count", ""),
            "oracle_profile_depth_rmse_m": oracle_rmse,
            "oracle_Er_like_error": as_float(oracle["oracle_Er_like_error"]),
            "oracle_projected_mask_IoU": as_float(oracle["oracle_projected_mask_IoU"]),
            "oracle_projected_mask_Dice": as_float(oracle["oracle_projected_mask_Dice"]),
            "oracle_area_error": as_float(oracle["oracle_area_error"]),
            "oracle_mask_center_error_px": as_float(oracle["oracle_mask_center_error_px"]),
            "oracle_component_count": oracle.get("oracle_component_count", ""),
            "oracle_minus_baseline_rmse_m": oracle_rmse - baseline_rmse,
            "baseline_minus_oracle_rmse_m": baseline_rmse - oracle_rmse,
            "label_L_m": float(dataset.L_m[idx]),
            "label_W_m": float(dataset.W_m[idx]),
            "label_D_m": float(dataset.D_m[idx]),
            "label_aspect_ratio": float(dataset.aspect_ratio[idx]),
            "label_rotation_angle": float(dataset.rotation_angle[idx]),
            "label_asymmetry_score": float(dataset.asymmetry_score[idx]),
            "label_edge_steepness": float(dataset.edge_steepness[idx]),
            "feature_input_policy": "observed_delta_b_features_plus_frozen_20_85_predicted_params",
            "label_use_policy": "labels_allowed_only_for_oracle_fit_surrogate_train_targets_validation_selection_and_final_metrics",
            "test_time_label_use_allowed": False,
        }
        for name in PARAM_NAMES:
            row[f"pred_{name}"] = as_float(baseline[f"pred_{name}"])
            row[f"oracle_{name}"] = as_float(oracle[f"oracle_{name}"])
        rows.append(row)
    return rows


def write_target_summary(rows: list[dict[str, Any]]) -> None:
    role_counts = Counter(str(row["target_role"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    target_rows = [row for row in rows if row["target_role"] == "refinement_target"]
    pass_rows = [row for row in rows if row["target_role"] == "already_pass_reference"]
    negative_rows = [row for row in rows if row["target_role"] == "excluded_negative_control"]
    target_by_split = Counter(str(row["split"]) for row in target_rows)
    target_by_shape = Counter(str(row["shape_type"]) for row in target_rows)
    lines = [
        "25.5 materialized surface forward-refinement target set",
        "",
        f"source_25_4_target_set: {TARGET_SET_25_4}",
        f"source_25_3_baseline_metrics: {BASELINE_METRICS}",
        f"source_25_3_oracle_metrics: {ORACLE_METRICS}",
        f"sample_count: {len(rows)}",
        f"split_counts: {dict(split_counts)}",
        f"target_role_counts: {dict(role_counts)}",
        f"refinement_target_count: {len(target_rows)}",
        f"already_pass_reference_count: {len(pass_rows)}",
        f"excluded_negative_control_count: {len(negative_rows)}",
        f"refinement_targets_by_split: {dict(target_by_split)}",
        f"refinement_targets_by_shape: {dict(target_by_shape)}",
        "",
        f"target_baseline_profile_rmse_mean_m: {mean(target_rows, 'baseline_profile_depth_rmse_m'):.12g}",
        f"target_oracle_profile_rmse_mean_m: {mean(target_rows, 'oracle_profile_depth_rmse_m'):.12g}",
        f"target_baseline_Er_like_mean: {mean(target_rows, 'baseline_Er_like_error'):.12g}",
        f"target_oracle_Er_like_mean: {mean(target_rows, 'oracle_Er_like_error'):.12g}",
        f"target_baseline_IoU_mean: {mean(target_rows, 'baseline_projected_mask_IoU'):.12g}",
        f"target_baseline_Dice_mean: {mean(target_rows, 'baseline_projected_mask_Dice'):.12g}",
        f"target_oracle_IoU_mean: {mean(target_rows, 'oracle_projected_mask_IoU'):.12g}",
        f"target_oracle_Dice_mean: {mean(target_rows, 'oracle_projected_mask_Dice'):.12g}",
        "",
        "multi_pit_policy: excluded_negative_control only; no six-param RBC refinement success credit.",
        "test_split_policy: materialized here for final reporting only; no test labels may drive refinement or hyperparameter selection.",
        f"target_materialized_csv: {TARGET_MATERIALIZED}",
    ]
    TARGET_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    TARGET_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    write_preflight_summary()
    rows = build_rows()
    write_csv(TARGET_MATERIALIZED, rows, TARGET_FIELDS)
    write_target_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
