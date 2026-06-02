#!/usr/bin/env python
"""Export the fixed 25.6 surface forward-refinement inference artifact.

This script serializes the fixed F0/R1 refinement candidate into an ignored
runtime artifact and writes only a commit-safe manifest under results/.
It does not run COMSOL, train the main neural model, or mutate data/NPZ files.
"""

from __future__ import annotations

import json
import hashlib
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
)
from fit_surface_feature_space_forward_surrogate import PARAM_BOUNDS, fit_candidate, observed_feature_matrix, row_indices
from run_surface_forward_refinement_formal_benchmark import (
    BASELINE_ARTIFACT,
    DIAGNOSIS_MATRIX,
    FIXED_ALPHA,
    FIXED_DESCRIPTOR_KIND,
    FIXED_FEATURE_MODE,
    FIXED_LAMBDA,
    FIXED_SURROGATE_ID,
    FORMAL_FIELDS,
    METRICS as FORMAL_METRICS,
    ORACLE_METRICS,
    PILOT_MANIFEST,
    PROFILE_GENERATOR,
)
from evaluate_surface_forward_refinement_formal_acceptance_gates import MATRIX as FORMAL_GATES


PREFLIGHT_SUMMARY = ROOT / "results/summaries/surface_forward_refinement_inference_preflight_summary.txt"
ARTIFACT_DIR = ROOT / "checkpoints/surface_forward_refinement_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "surface_forward_refinement_inference_artifact_v1.json"
MANIFEST_PATH = ROOT / "results/manifests/surface_forward_refinement_inference_artifact_manifest.json"

SCRIPT_25_5_SURROGATE = ROOT / "scripts/fit_surface_feature_space_forward_surrogate.py"
SCRIPT_25_5_REFINEMENT = ROOT / "scripts/run_surface_rbc_forward_consistency_refinement.py"
SCRIPT_25_6_FORMAL = ROOT / "scripts/run_surface_forward_refinement_formal_benchmark.py"
SCRIPT_25_6_GATES = ROOT / "scripts/evaluate_surface_forward_refinement_formal_acceptance_gates.py"

FORBIDDEN_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        TARGET_MATERIALIZED,
        FORMAL_METRICS,
        FORMAL_GATES,
        ORACLE_METRICS,
        DIAGNOSIS_MATRIX,
        PROFILE_GENERATOR,
        SCRIPT_25_5_SURROGATE,
        SCRIPT_25_5_REFINEMENT,
        SCRIPT_25_6_FORMAL,
        SCRIPT_25_6_GATES,
    ]


def metric_mean(rows: list[dict[str, str]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = as_float(row[key])
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def formal_reference_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in rows if row["shape_type"] == "rbc_like_smooth_pit"]
    negative = [row for row in rows if row["target_role"] == "excluded_negative_control"]
    pass_refs = [row for row in rows if row["target_role"] == "already_pass_reference"]
    return {
        "formal_metrics_path": str(FORMAL_METRICS),
        "target_subset_count": len(targets),
        "already_pass_reference_count": len(pass_refs),
        "excluded_negative_control_count": len(negative),
        "target_baseline_profile_rmse_mean_m": metric_mean(targets, "baseline_profile_depth_rmse_m"),
        "target_refined_profile_rmse_mean_m": metric_mean(targets, "refined_profile_depth_rmse_m"),
        "target_oracle_profile_rmse_mean_m": metric_mean(targets, "oracle_profile_depth_rmse_m"),
        "target_baseline_Er_like_mean": metric_mean(targets, "baseline_Er_like_error"),
        "target_refined_Er_like_mean": metric_mean(targets, "refined_Er_like_error"),
        "target_oracle_Er_like_mean": metric_mean(targets, "oracle_Er_like_error"),
        "target_baseline_IoU_mean": metric_mean(targets, "baseline_projected_mask_IoU"),
        "target_refined_IoU_mean": metric_mean(targets, "refined_projected_mask_IoU"),
        "target_oracle_IoU_mean": metric_mean(targets, "oracle_projected_mask_IoU"),
        "target_baseline_Dice_mean": metric_mean(targets, "baseline_projected_mask_Dice"),
        "target_refined_Dice_mean": metric_mean(targets, "refined_projected_mask_Dice"),
        "target_oracle_Dice_mean": metric_mean(targets, "oracle_projected_mask_Dice"),
        "target_feature_residual_before_mean": metric_mean(targets, "feature_residual_mse_before"),
        "target_feature_residual_after_mean": metric_mean(targets, "feature_residual_mse_after"),
        "rbc_like_control_baseline_rmse_mean_m": metric_mean(rbc_like, "baseline_profile_depth_rmse_m"),
        "rbc_like_control_refined_rmse_mean_m": metric_mean(rbc_like, "refined_profile_depth_rmse_m"),
        "rbc_like_control_baseline_Dice_mean": metric_mean(rbc_like, "baseline_projected_mask_Dice"),
        "rbc_like_control_refined_Dice_mean": metric_mean(rbc_like, "refined_projected_mask_Dice"),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_preflight_summary(dataset: Any, formal_rows: list[dict[str, str]], gate_rows: list[dict[str, str]]) -> None:
    pilot = read_json(PILOT_MANIFEST)
    baseline = read_json(BASELINE_ARTIFACT)
    target_roles = Counter(row["target_role"] for row in formal_rows)
    all_gates_pass = bool(gate_rows) and all(as_bool(row["pass"]) for row in gate_rows)
    forbidden_diff = git_value(["diff", "--name-only", "--", *FORBIDDEN_PATHS])
    staged_forbidden = git_value(["diff", "--cached", "--name-only", "--", *FORBIDDEN_PATHS])
    lines = [
        "25.7 surface forward-refinement inference preflight",
        "",
        "scope: export fixed runtime artifact and runner; no COMSOL, no main neural training, no data/NPZ mutation, no CURRENT_BASELINE.md update.",
        f"dataset_id: {DATASET_ID}",
        f"dataset_manifest: {PILOT_MANIFEST}",
        f"registry_path: {REGISTRY}",
        f"explicit_registry_manifest_loaded: {dataset.manifest.get('dataset_id') == DATASET_ID}",
        f"npz_sha256_verified_by_loader: true",
        f"sample_count: {len(dataset.sample_ids)}",
        f"split_counts: {dict(Counter(str(x) for x in dataset.split))}",
        f"baseline_artifact_manifest: {BASELINE_ARTIFACT}",
        f"baseline_artifact_id: {baseline.get('artifact_id')}",
        f"pilot_allowed_use: {pilot.get('allowed_use')}",
        f"pilot_forbidden_use: {pilot.get('forbidden_use')}",
        "",
        "fixed_protocol:",
        f"- selected_surrogate: {FIXED_SURROGATE_ID}",
        f"- descriptor_kind: {FIXED_DESCRIPTOR_KIND}",
        f"- feature_mode: {FIXED_FEATURE_MODE}",
        f"- alpha: {FIXED_ALPHA:g}",
        f"- lambda_profile: 1",
        f"- lambda_param: {FIXED_LAMBDA:g}",
        "- refinement_strategy: R1_low_dim_param_refinement",
        "- initialization: frozen 20.85 predicted six params",
        "- optimized_params: L_m/W_m/D_m/wLD/wWD/wLW",
        "",
        f"formal_metric_rows: {len(formal_rows)}",
        f"target_role_counts: {dict(target_roles)}",
        f"formal_acceptance_gates_pass: {all_gates_pass}",
        f"formal_acceptance_gate_count: {len(gate_rows)}",
        f"profile_generator: {PROFILE_GENERATOR}",
        "",
        f"artifact_body_path_ignored: {ARTIFACT_PATH}",
        f"commit_manifest_path: {MANIFEST_PATH}",
        "allowed_use: explicit_surface_forward_refinement_inference",
        "forbidden_use: current_baseline_replacement, automatic_baseline_update",
        "",
        f"forbidden_diff_present: {bool(forbidden_diff)}",
        "forbidden_diff_paths: " + (forbidden_diff if forbidden_diff else "none"),
        f"staged_forbidden_present: {bool(staged_forbidden)}",
        "staged_forbidden_paths: " + (staged_forbidden if staged_forbidden else "none"),
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_artifact() -> tuple[dict[str, Any], dict[str, Any]]:
    missing = [path for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing 25.7 export input(s): " + ", ".join(str(path) for path in missing))

    dataset = load_surface_dataset(DATASET_ID, REGISTRY)
    materialized_rows = read_csv(TARGET_MATERIALIZED)
    formal_rows = read_csv(FORMAL_METRICS)
    gate_rows = read_csv(FORMAL_GATES)
    write_preflight_summary(dataset, formal_rows, gate_rows)

    if len(materialized_rows) != len(dataset.sample_ids):
        raise RuntimeError(f"target materialized row count mismatch: {len(materialized_rows)} != {len(dataset.sample_ids)}")
    if len(formal_rows) != len(dataset.sample_ids):
        raise RuntimeError(f"formal metrics row count mismatch: {len(formal_rows)} != {len(dataset.sample_ids)}")
    if not gate_rows or not all(as_bool(row["pass"]) for row in gate_rows):
        raise RuntimeError("25.6 formal gates are missing or not all PASS")

    observed_features, observed_feature_names = observed_feature_matrix(dataset.delta_b)
    train_idx = row_indices(materialized_rows, split="train", representable_only=True)
    surrogate = fit_candidate(
        rows=materialized_rows,
        observed_features=observed_features,
        train_idx=train_idx,
        descriptor_kind=FIXED_DESCRIPTOR_KIND,
        feature_mode=FIXED_FEATURE_MODE,
        alpha=FIXED_ALPHA,
        observed_feature_names=observed_feature_names,
    )
    if surrogate.candidate_id != FIXED_SURROGATE_ID:
        raise RuntimeError(f"fixed surrogate id mismatch: {surrogate.candidate_id}")

    baseline = read_json(BASELINE_ARTIFACT)
    benchmark = formal_reference_metrics(formal_rows)
    artifact = {
        "artifact_id": "surface_forward_refinement_inference_artifact_v1",
        "stage": "25.7",
        "dataset_id": DATASET_ID,
        "created_by": "scripts/export_surface_forward_refinement_artifact.py",
        "artifact_format": "json_serialized_ridge_surrogate_and_refinement_contract",
        "allowed_use": ["explicit_surface_forward_refinement_inference"],
        "forbidden_use": ["current_baseline_replacement", "automatic_baseline_update"],
        "baseline_artifact_manifest": str(BASELINE_ARTIFACT),
        "baseline_artifact_id": baseline.get("artifact_id"),
        "baseline_dataset_id": baseline.get("dataset_id"),
        "surface_dataset_manifest": str(PILOT_MANIFEST),
        "protocol": {
            "selected_surrogate": FIXED_SURROGATE_ID,
            "surrogate_family": "F0_feature_space_consistency",
            "refinement_strategy": "R1_low_dim_param_refinement",
            "descriptor_kind": FIXED_DESCRIPTOR_KIND,
            "feature_mode": FIXED_FEATURE_MODE,
            "alpha": FIXED_ALPHA,
            "lambda_profile": 1.0,
            "lambda_param": FIXED_LAMBDA,
            "initialization": "frozen_20_85_predicted_six_params",
            "optimized_params": PARAM_NAMES,
            "main_neural_weight_update": False,
            "COMSOL_call": False,
            "hyperparameter_search": False,
            "surrogate_fit_scope": "train_split_rbc_representable_rows_from_25_6_protocol",
            "label_use_at_runtime": "labels_forbidden_as_refinement_inputs",
            "labels_allowed_for": "offline_metrics_when_available",
        },
        "parameter_bounds": {name: {"min": float(PARAM_BOUNDS[i, 0]), "max": float(PARAM_BOUNDS[i, 1])} for i, name in enumerate(PARAM_NAMES)},
        "feature_columns": {
            "observed_feature_names": surrogate.observed_feature_names,
            "descriptor_names": surrogate.descriptor_names,
            "model_feature_names": surrogate.model_feature_names,
        },
        "surrogate": {
            "candidate_id": surrogate.candidate_id,
            "descriptor_kind": surrogate.descriptor_kind,
            "feature_mode": surrogate.feature_mode,
            "alpha": surrogate.alpha,
            "x_mean": surrogate.x_mean,
            "x_std": surrogate.x_std,
            "y_mean": surrogate.y_mean,
            "y_std": surrogate.y_std,
            "coef": surrogate.coef,
        },
        "reference_benchmark_metrics": benchmark,
    }
    manifest = {
        "artifact_id": artifact["artifact_id"],
        "stage": artifact["stage"],
        "dataset_id": DATASET_ID,
        "baseline_artifact_manifest": str(BASELINE_ARTIFACT),
        "baseline_artifact_id": baseline.get("artifact_id"),
        "selected_surrogate": FIXED_SURROGATE_ID,
        "alpha": FIXED_ALPHA,
        "lambda_profile": 1.0,
        "lambda_param": FIXED_LAMBDA,
        "refinement_strategy": "R1_low_dim_param_refinement",
        "initialization": "frozen_20_85_predicted_six_params",
        "optimized_params": PARAM_NAMES,
        "feature_columns": artifact["feature_columns"],
        "artifact_path": str(ARTIFACT_PATH),
        "artifact_sha256": "",
        "artifact_committed": False,
        "artifact_paths_are_ignored": True,
        "benchmark_metrics": benchmark,
        "allowed_use": ["explicit_surface_forward_refinement_inference"],
        "forbidden_use": ["current_baseline_replacement", "automatic_baseline_update"],
    }
    return to_jsonable(artifact), to_jsonable(manifest)


def write_outputs(artifact: dict[str, Any], manifest: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["artifact_sha256"] = sha256_file(ARTIFACT_PATH)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    artifact, manifest = build_artifact()
    write_outputs(artifact, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
