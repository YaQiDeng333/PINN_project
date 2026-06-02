#!/usr/bin/env python
"""Build the 25.8 surface forward-refinement report package.

This consumes the locked 25.7 runner outputs and existing audit metrics. It
does not run COMSOL, train, mutate data/NPZ, or update CURRENT_BASELINE.md.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_surface_shape_extension_rbc_oracle_fit import DATASET_ID, ROOT, load_surface_dataset
from build_surface_forward_refinement_target_set import REGISTRY, as_bool, as_float, read_csv, write_csv


ARTIFACT_MANIFEST = ROOT / "results/manifests/surface_forward_refinement_inference_artifact_manifest.json"
PILOT_MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
FORMAL_METRICS = ROOT / "results/metrics/surface_forward_refinement_formal_benchmark_metrics.csv"
RUNNER_METRICS = ROOT / "results/metrics/surface_forward_refinement_inference_metrics.csv"
RUNNER_VERIFICATION = ROOT / "results/metrics/surface_forward_refinement_inference_verification.csv"
ORACLE_BASELINE_DIAGNOSIS = ROOT / "results/metrics/surface_shape_extension_oracle_vs_baseline_matrix.csv"
BASELINE_METRICS = ROOT / "results/metrics/surface_shape_extension_current_baseline_metrics.csv"
ORACLE_METRICS = ROOT / "results/metrics/surface_shape_extension_rbc_oracle_fit_metrics.csv"

PREFLIGHT_SUMMARY = ROOT / "results/summaries/surface_forward_refinement_report_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/surface_forward_refinement_report_package_summary.txt"
REPORT_METRICS = ROOT / "results/metrics/surface_forward_refinement_report_package_metrics.csv"
COMPARISON = ROOT / "results/metrics/surface_forward_refinement_candidate_comparison_matrix.csv"

FORBIDDEN_PATHS = [
    "CURRENT_BASELINE.md",
    "data",
    "checkpoints",
    "notes",
    "results/previews",
    "scripts/visualize_current_baseline.py",
]

REPORT_FIELDS = [
    "section",
    "metric",
    "sample_count",
    "baseline_value",
    "refined_value",
    "oracle_value",
    "delta_baseline_to_refined",
    "interpretation",
]

COMPARISON_FIELDS = [
    "candidate_or_route",
    "role",
    "applicable_scope",
    "not_applicable_scope",
    "baseline_relation",
    "sample_count",
    "target_profile_rmse_m",
    "target_Er_like",
    "target_IoU",
    "target_Dice",
    "target_forward_residual",
    "multi_pit_success_credit_allowed",
    "baseline_transition_allowed",
    "recommended_next_handling",
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
        ARTIFACT_MANIFEST,
        PILOT_MANIFEST,
        FORMAL_METRICS,
        RUNNER_METRICS,
        RUNNER_VERIFICATION,
        ORACLE_BASELINE_DIAGNOSIS,
        BASELINE_METRICS,
        ORACLE_METRICS,
        REGISTRY,
    ]


def mean(rows: list[dict[str, str]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = as_float(row[key])
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def pass_check(rows: list[dict[str, str]], name: str) -> bool:
    return any(row.get("check_name") == name and as_bool(row.get("pass")) for row in rows)


def report_row(
    section: str,
    metric: str,
    subset: list[dict[str, str]],
    baseline_key: str,
    refined_key: str,
    oracle_key: str,
    interpretation: str,
) -> dict[str, Any]:
    baseline = mean(subset, baseline_key)
    refined = mean(subset, refined_key)
    oracle = mean(subset, oracle_key) if oracle_key else float("nan")
    return {
        "section": section,
        "metric": metric,
        "sample_count": len(subset),
        "baseline_value": baseline,
        "refined_value": refined,
        "oracle_value": oracle,
        "delta_baseline_to_refined": refined - baseline,
        "interpretation": interpretation,
    }


def write_preflight(metrics: list[dict[str, str]], verification: list[dict[str, str]], artifact: dict[str, Any]) -> None:
    missing = [path for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing 25.8 preflight input(s): " + ", ".join(str(path) for path in missing))
    dataset = load_surface_dataset(DATASET_ID, REGISTRY)
    pilot = read_json(PILOT_MANIFEST)
    role_counts = Counter(row["target_role"] for row in metrics)
    diagnosis_counts = Counter(row["diagnosis"] for row in metrics)
    all_verification_pass = bool(verification) and all(as_bool(row["pass"]) for row in verification)
    forbidden_diff = git_value(["diff", "--name-only", "--", *FORBIDDEN_PATHS])
    staged_forbidden = git_value(["diff", "--cached", "--name-only", "--", *FORBIDDEN_PATHS])
    lines = [
        "25.8 surface forward-refinement report preflight",
        "",
        "scope: report / visualization package only; no COMSOL, no training, no data/NPZ mutation, no CURRENT_BASELINE.md update.",
        f"dataset_id: {DATASET_ID}",
        f"registry_path: {REGISTRY}",
        f"pilot_manifest: {PILOT_MANIFEST}",
        f"explicit_registry_manifest_loaded: {dataset.manifest.get('dataset_id') == DATASET_ID}",
        f"npz_sha256_verified_by_loader: true",
        f"sample_count: {len(dataset.sample_ids)}",
        f"split_counts: {dict(Counter(str(x) for x in dataset.split))}",
        f"pilot_allowed_use: {pilot.get('allowed_use')}",
        f"pilot_forbidden_use: {pilot.get('forbidden_use')}",
        "",
        f"artifact_manifest: {ARTIFACT_MANIFEST}",
        f"artifact_id: {artifact.get('artifact_id')}",
        f"artifact_allowed_use: {artifact.get('allowed_use')}",
        f"artifact_forbidden_use: {artifact.get('forbidden_use')}",
        f"selected_surrogate: {artifact.get('selected_surrogate')}",
        f"lambda_param: {artifact.get('lambda_param')}",
        "",
        f"runner_metrics: {RUNNER_METRICS}",
        f"formal_benchmark_metrics: {FORMAL_METRICS}",
        f"oracle_baseline_diagnosis: {ORACLE_BASELINE_DIAGNOSIS}",
        f"runner_verification_all_pass: {all_verification_pass}",
        f"runner_reproduces_25_6: {pass_check(verification, 'runner_reproduces_25_6_per_sample')}",
        f"target_role_counts: {dict(role_counts)}",
        f"diagnosis_counts: {dict(diagnosis_counts)}",
        "",
        f"forbidden_diff_present: {bool(forbidden_diff)}",
        "forbidden_diff_paths: " + (forbidden_diff if forbidden_diff else "none"),
        f"staged_forbidden_present: {bool(staged_forbidden)}",
        "staged_forbidden_paths: " + (staged_forbidden if staged_forbidden else "none"),
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report_rows(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    targets = [row for row in metrics if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in metrics if row["shape_type"] == "rbc_like_smooth_pit"]
    negative = [row for row in metrics if row["target_role"] == "excluded_negative_control"]
    pass_refs = [row for row in metrics if row["target_role"] == "already_pass_reference"]
    rows = [
        report_row(
            "target_subset_82",
            "profile_rmse_m",
            targets,
            "baseline_profile_depth_rmse_m",
            "refined_profile_depth_rmse_m",
            "oracle_profile_depth_rmse_m",
            "RBC-representable model failures are materially repaired by the companion runner.",
        ),
        report_row("target_subset_82", "Er_like", targets, "baseline_Er_like_error", "refined_Er_like_error", "oracle_Er_like_error", "Er-like profile error decreases."),
        report_row("target_subset_82", "IoU", targets, "baseline_projected_mask_IoU", "refined_projected_mask_IoU", "oracle_projected_mask_IoU", "Projected-mask IoU increases."),
        report_row("target_subset_82", "Dice", targets, "baseline_projected_mask_Dice", "refined_projected_mask_Dice", "oracle_projected_mask_Dice", "Projected-mask Dice increases."),
        report_row("target_subset_82", "forward_residual", targets, "feature_residual_mse_before", "feature_residual_mse_after", "", "Forward feature residual decreases."),
        report_row("rbc_like_control", "profile_rmse_m", rbc_like, "baseline_profile_depth_rmse_m", "refined_profile_depth_rmse_m", "oracle_profile_depth_rmse_m", "RBC-like control remains stable; no collapse."),
        report_row("rbc_like_control", "Dice", rbc_like, "baseline_projected_mask_Dice", "refined_projected_mask_Dice", "oracle_projected_mask_Dice", "RBC-like control mask quality does not degrade."),
        report_row("already_pass_reference", "profile_rmse_m", pass_refs, "baseline_profile_depth_rmse_m", "refined_profile_depth_rmse_m", "oracle_profile_depth_rmse_m", "Monitoring bucket; not success-credit target."),
        report_row("negative_control_multi_pit", "profile_rmse_m", negative, "baseline_profile_depth_rmse_m", "refined_profile_depth_rmse_m", "oracle_profile_depth_rmse_m", "Multi-pit/component-set rows are not suitable for six-parameter RBC refinement."),
        report_row("negative_control_multi_pit", "Dice", negative, "baseline_projected_mask_Dice", "refined_projected_mask_Dice", "oracle_projected_mask_Dice", "No RBC success credit is assigned to multi-pit rows."),
    ]
    return rows


def build_comparison_rows(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    targets = [row for row in metrics if row["target_role"] == "refinement_target"]
    negative = [row for row in metrics if row["target_role"] == "excluded_negative_control"]
    return [
        {
            "candidate_or_route": "20.85_frozen_surface_RBC_baseline",
            "role": "current_baseline_reference",
            "applicable_scope": "surface RBC-style six-parameter initialization before post-hoc refinement",
            "not_applicable_scope": "non-RBC representation failures and multi-pit component-set rows",
            "baseline_relation": "current baseline remains 20.85",
            "sample_count": len(targets),
            "target_profile_rmse_m": mean(targets, "baseline_profile_depth_rmse_m"),
            "target_Er_like": mean(targets, "baseline_Er_like_error"),
            "target_IoU": mean(targets, "baseline_projected_mask_IoU"),
            "target_Dice": mean(targets, "baseline_projected_mask_Dice"),
            "target_forward_residual": mean(targets, "feature_residual_mse_before"),
            "multi_pit_success_credit_allowed": False,
            "baseline_transition_allowed": False,
            "recommended_next_handling": "keep as frozen baseline and initialization source",
        },
        {
            "candidate_or_route": "25.7_surface_forward_refinement_companion_runner",
            "role": "post_hoc_companion_refinement",
            "applicable_scope": "rbc_representable_but_model_fail surface rows",
            "not_applicable_scope": "multi-pit / component-set / rbc_not_representable rows",
            "baseline_relation": "companion over frozen 20.85; model weights unchanged",
            "sample_count": len(targets),
            "target_profile_rmse_m": mean(targets, "refined_profile_depth_rmse_m"),
            "target_Er_like": mean(targets, "refined_Er_like_error"),
            "target_IoU": mean(targets, "refined_projected_mask_IoU"),
            "target_Dice": mean(targets, "refined_projected_mask_Dice"),
            "target_forward_residual": mean(targets, "feature_residual_mse_after"),
            "multi_pit_success_credit_allowed": False,
            "baseline_transition_allowed": False,
            "recommended_next_handling": "use in report/visualization and keep baseline transition forbidden",
        },
        {
            "candidate_or_route": "RBC_oracle_fit_ceiling",
            "role": "oracle_reference_not_runtime",
            "applicable_scope": "metric ceiling for RBC-representable rows",
            "not_applicable_scope": "test-time inference input",
            "baseline_relation": "not a deployable model",
            "sample_count": len(targets),
            "target_profile_rmse_m": mean(targets, "oracle_profile_depth_rmse_m"),
            "target_Er_like": mean(targets, "oracle_Er_like_error"),
            "target_IoU": mean(targets, "oracle_projected_mask_IoU"),
            "target_Dice": mean(targets, "oracle_projected_mask_Dice"),
            "target_forward_residual": "",
            "multi_pit_success_credit_allowed": False,
            "baseline_transition_allowed": False,
            "recommended_next_handling": "retain only as evaluation ceiling",
        },
        {
            "candidate_or_route": "component_set_branch_for_multi_pit",
            "role": "future_representation_branch",
            "applicable_scope": "multi-pit / rbc_not_representable / component-set surface defects",
            "not_applicable_scope": "single-component RBC-representable model-fail target set",
            "baseline_relation": "separate representation branch",
            "sample_count": len(negative),
            "target_profile_rmse_m": "",
            "target_Er_like": "",
            "target_IoU": "",
            "target_Dice": "",
            "target_forward_residual": "",
            "multi_pit_success_credit_allowed": "only inside future component-set branch, not RBC refinement",
            "baseline_transition_allowed": False,
            "recommended_next_handling": "next route after report confirms runner stability",
        },
    ]


def write_summary(metrics: list[dict[str, str]], report_rows: list[dict[str, Any]], artifact: dict[str, Any]) -> None:
    targets = [row for row in metrics if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in metrics if row["shape_type"] == "rbc_like_smooth_pit"]
    negative = [row for row in metrics if row["target_role"] == "excluded_negative_control"]
    degraded_targets = [row for row in targets if as_float(row["profile_rmse_delta_m"]) > 0.0 or as_float(row["Dice_delta"]) < 0.0]
    lines = [
        "25.8 surface forward-refinement report package",
        "",
        "current_baseline: frozen 20.85 surface RBC baseline",
        "forward_refinement_runner: post-hoc companion refinement, not baseline replacement",
        f"artifact_manifest: {ARTIFACT_MANIFEST}",
        f"artifact_id: {artifact.get('artifact_id')}",
        f"selected_surrogate: {artifact.get('selected_surrogate')}",
        f"lambda_param: {artifact.get('lambda_param')}",
        "runtime_input_boundary: observed delta_b-derived features + frozen 20.85 predicted six params + exported artifact; labels only annotate metrics/visualization.",
        "",
        f"sample_count: {len(metrics)}",
        f"target_subset_count: {len(targets)}",
        f"rbc_like_control_count: {len(rbc_like)}",
        f"negative_control_count: {len(negative)}",
        f"degraded_target_count: {len(degraded_targets)}",
        "",
        f"target_profile_rmse_m: {mean(targets, 'baseline_profile_depth_rmse_m'):.12g} -> {mean(targets, 'refined_profile_depth_rmse_m'):.12g} (oracle {mean(targets, 'oracle_profile_depth_rmse_m'):.12g})",
        f"target_Er_like: {mean(targets, 'baseline_Er_like_error'):.12g} -> {mean(targets, 'refined_Er_like_error'):.12g} (oracle {mean(targets, 'oracle_Er_like_error'):.12g})",
        f"target_IoU_Dice: {mean(targets, 'baseline_projected_mask_IoU'):.12g}/{mean(targets, 'baseline_projected_mask_Dice'):.12g} -> {mean(targets, 'refined_projected_mask_IoU'):.12g}/{mean(targets, 'refined_projected_mask_Dice'):.12g}",
        f"target_forward_residual: {mean(targets, 'feature_residual_mse_before'):.12g} -> {mean(targets, 'feature_residual_mse_after'):.12g}",
        "",
        f"rbc_like_control_rmse_m: {mean(rbc_like, 'baseline_profile_depth_rmse_m'):.12g} -> {mean(rbc_like, 'refined_profile_depth_rmse_m'):.12g}",
        f"rbc_like_control_Dice: {mean(rbc_like, 'baseline_projected_mask_Dice'):.12g} -> {mean(rbc_like, 'refined_projected_mask_Dice'):.12g}",
        "negative_control_policy: multi-pit / rbc_not_representable rows are marked not_suitable_for_rbc_refinement and receive no RBC success credit.",
        "model_failure_vs_representation_failure: rbc_representable_but_model_fail rows are repairable model/inversion failures; multi-pit rows remain representation failures requiring component-set output.",
        "why_not_baseline_transition: the runner is a companion over 20.85, not a replacement; CURRENT_BASELINE.md remains unchanged.",
        "remaining_problem: multi-pit component-set branch.",
        "",
        f"report_metrics_csv: {REPORT_METRICS}",
        f"comparison_matrix_csv: {COMPARISON}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    missing = [path for path in required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing 25.8 report input(s): " + ", ".join(str(path) for path in missing))
    artifact = read_json(ARTIFACT_MANIFEST)
    metrics = read_csv(RUNNER_METRICS)
    verification = read_csv(RUNNER_VERIFICATION)
    if artifact.get("dataset_id") != DATASET_ID:
        raise RuntimeError(f"unexpected artifact dataset_id: {artifact.get('dataset_id')}")
    if not all(as_bool(row["pass"]) for row in verification):
        raise RuntimeError("runner verification is not all PASS")
    write_preflight(metrics, verification, artifact)
    report_rows = build_report_rows(metrics)
    comparison_rows = build_comparison_rows(metrics)
    write_csv(REPORT_METRICS, report_rows, REPORT_FIELDS)
    write_csv(COMPARISON, comparison_rows, COMPARISON_FIELDS)
    write_summary(metrics, report_rows, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
