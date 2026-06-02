#!/usr/bin/env python
"""Verify the 25.7 surface forward-refinement inference runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from build_surface_forward_refinement_target_set import ROOT, as_bool, as_float, read_csv, write_csv
from run_surface_forward_refinement_formal_benchmark import METRICS as FORMAL_METRICS
from run_surface_forward_refinement_inference import DEFAULT_MANIFEST, METRICS as RUNNER_METRICS


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_inference_verification_summary.txt"
MATRIX = ROOT / "results/metrics/surface_forward_refinement_inference_verification.csv"

FIELDS = ["check_name", "pass", "observed", "reference", "threshold_or_condition", "notes"]

REPRO_KEYS = [
    "feature_residual_mse_before",
    "feature_residual_mse_after",
    "baseline_profile_depth_rmse_m",
    "refined_profile_depth_rmse_m",
    "baseline_Er_like_error",
    "refined_Er_like_error",
    "baseline_projected_mask_IoU",
    "refined_projected_mask_IoU",
    "baseline_projected_mask_Dice",
    "refined_projected_mask_Dice",
    "baseline_area_error",
    "refined_area_error",
    "refined_L_m",
    "refined_W_m",
    "refined_D_m",
    "refined_wLD",
    "refined_wWD",
    "refined_wLW",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception as exc:
        return f"git_error:{exc}"


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            val = as_float(row[key])
        except Exception:
            continue
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def add_row(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, reference: Any, condition: str, notes: str) -> None:
    rows.append(
        {
            "check_name": name,
            "pass": bool(passed),
            "observed": observed,
            "reference": reference,
            "threshold_or_condition": condition,
            "notes": notes,
        }
    )


def max_abs_reproduction_diff(runner_rows: list[dict[str, str]], formal_rows: list[dict[str, str]]) -> tuple[float, str]:
    formal_by_id = {row["sample_id"]: row for row in formal_rows}
    max_diff = 0.0
    max_field = ""
    for row in runner_rows:
        sample_id = row["sample_id"]
        if sample_id not in formal_by_id:
            return float("inf"), f"missing_formal_row:{sample_id}"
        formal = formal_by_id[sample_id]
        for key in REPRO_KEYS:
            diff = abs(as_float(row[key]) - as_float(formal[key]))
            if diff > max_diff:
                max_diff = float(diff)
                max_field = f"{sample_id}:{key}"
    return max_diff, max_field


def build_checks(runner_rows: list[dict[str, str]], formal_rows: list[dict[str, str]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    runner_targets = [row for row in runner_rows if row["target_role"] == "refinement_target"]
    formal_targets = [row for row in formal_rows if row["target_role"] == "refinement_target"]
    runner_rbc_like = [row for row in runner_rows if row["shape_type"] == "rbc_like_smooth_pit"]
    runner_multipit = [row for row in runner_rows if row["shape_type"] == "multi_pit_two_component_surface_defect"]

    add_row(
        checks,
        "artifact_manifest_present",
        DEFAULT_MANIFEST.exists(),
        DEFAULT_MANIFEST,
        "manifest exists",
        "exists",
        "Commit-safe manifest is required; artifact body remains in ignored checkpoints/.",
    )
    add_row(
        checks,
        "artifact_allowed_use",
        manifest.get("allowed_use") == ["explicit_surface_forward_refinement_inference"],
        manifest.get("allowed_use"),
        ["explicit_surface_forward_refinement_inference"],
        "exact allowed_use",
        "Manifest must not authorize baseline replacement.",
    )
    add_row(
        checks,
        "artifact_forbidden_use",
        set(manifest.get("forbidden_use", [])) == {"current_baseline_replacement", "automatic_baseline_update"},
        manifest.get("forbidden_use"),
        ["current_baseline_replacement", "automatic_baseline_update"],
        "exact forbidden_use set",
        "Baseline transition remains forbidden.",
    )
    add_row(
        checks,
        "runner_row_count_matches_25_6",
        len(runner_rows) == len(formal_rows) == 120,
        len(runner_rows),
        len(formal_rows),
        "runner rows == formal rows == 120",
        "Full pilot replay is required for verification.",
    )
    max_diff, max_field = max_abs_reproduction_diff(runner_rows, formal_rows)
    add_row(
        checks,
        "runner_reproduces_25_6_per_sample",
        max_diff <= 1.0e-8,
        f"max_abs_diff={max_diff:.12g} at {max_field}",
        "25.6 formal benchmark CSV",
        "max_abs_diff <= 1e-8 on key per-sample fields",
        "Runner must reproduce the fixed 25.6 protocol within deterministic floating-point tolerance.",
    )
    add_row(
        checks,
        "target_subset_count",
        len(runner_targets) == len(formal_targets) == 82,
        len(runner_targets),
        len(formal_targets),
        "target count == 82",
        "Target subset remains rbc_representable_but_model_fail.",
    )
    target_rmse_ok = mean(runner_targets, "refined_profile_depth_rmse_m") < mean(runner_targets, "baseline_profile_depth_rmse_m")
    add_row(
        checks,
        "target_profile_rmse_improves",
        target_rmse_ok,
        f"{mean(runner_targets, 'baseline_profile_depth_rmse_m'):.12g}->{mean(runner_targets, 'refined_profile_depth_rmse_m'):.12g}",
        f"{mean(formal_targets, 'baseline_profile_depth_rmse_m'):.12g}->{mean(formal_targets, 'refined_profile_depth_rmse_m'):.12g}",
        "refined < baseline",
        "Primary target subset must improve.",
    )
    target_mask_ok = (
        mean(runner_targets, "refined_projected_mask_IoU") > mean(runner_targets, "baseline_projected_mask_IoU")
        and mean(runner_targets, "refined_projected_mask_Dice") > mean(runner_targets, "baseline_projected_mask_Dice")
    )
    add_row(
        checks,
        "target_iou_dice_improve",
        target_mask_ok,
        (
            f"IoU {mean(runner_targets, 'baseline_projected_mask_IoU'):.12g}->{mean(runner_targets, 'refined_projected_mask_IoU'):.12g}; "
            f"Dice {mean(runner_targets, 'baseline_projected_mask_Dice'):.12g}->{mean(runner_targets, 'refined_projected_mask_Dice'):.12g}"
        ),
        "25.6 target IoU/Dice improvement",
        "IoU and Dice both increase",
        "Projected-mask metrics must improve on the target subset.",
    )
    residual_ok = mean(runner_targets, "feature_residual_mse_after") < mean(runner_targets, "feature_residual_mse_before")
    add_row(
        checks,
        "target_forward_residual_improves",
        residual_ok,
        f"{mean(runner_targets, 'feature_residual_mse_before'):.12g}->{mean(runner_targets, 'feature_residual_mse_after'):.12g}",
        "25.6 target forward residual improvement",
        "after < before",
        "Forward-feature residual must decrease.",
    )
    rbc_control_ok = (
        mean(runner_rbc_like, "refined_profile_depth_rmse_m") <= mean(runner_rbc_like, "baseline_profile_depth_rmse_m")
        and mean(runner_rbc_like, "refined_projected_mask_Dice") >= mean(runner_rbc_like, "baseline_projected_mask_Dice")
    )
    add_row(
        checks,
        "rbc_like_control_not_degraded",
        rbc_control_ok,
        (
            f"RMSE {mean(runner_rbc_like, 'baseline_profile_depth_rmse_m'):.12g}->{mean(runner_rbc_like, 'refined_profile_depth_rmse_m'):.12g}; "
            f"Dice {mean(runner_rbc_like, 'baseline_projected_mask_Dice'):.12g}->{mean(runner_rbc_like, 'refined_projected_mask_Dice'):.12g}"
        ),
        "RMSE non-increase and Dice non-decrease",
        "control guard",
        "RBC-like control must not collapse.",
    )
    multipit_ok = bool(runner_multipit) and all(
        row["eligibility_status"] == "not_suitable_for_rbc_refinement"
        and not as_bool(row["refinement_applied"])
        and not as_bool(row["include_in_success_gate"])
        for row in runner_multipit
    )
    add_row(
        checks,
        "multi_pit_not_success_credit",
        multipit_ok,
        f"multi_pit_rows={len(runner_multipit)}",
        "all multi-pit rows excluded",
        "not_suitable_for_rbc_refinement and no success credit",
        "Component-set representation failures must not be counted as RBC refinement wins.",
    )
    current_baseline_diff = git_value(["diff", "--name-only", "--", "CURRENT_BASELINE.md"])
    add_row(
        checks,
        "current_baseline_unchanged",
        not bool(current_baseline_diff),
        current_baseline_diff if current_baseline_diff else "unchanged",
        "unchanged",
        "no CURRENT_BASELINE.md diff",
        "25.7 is not a baseline transition.",
    )
    checkpoint_staged = git_value(["diff", "--cached", "--name-only", "--", "checkpoints"])
    add_row(
        checks,
        "artifact_body_not_staged",
        not bool(checkpoint_staged),
        checkpoint_staged if checkpoint_staged else "none",
        "none",
        "no staged checkpoints path",
        "Only the manifest is commit-safe.",
    )
    committed_artifacts = git_value(["ls-files", "checkpoints/surface_forward_refinement_artifacts"])
    add_row(
        checks,
        "artifact_body_not_tracked",
        not bool(committed_artifacts),
        committed_artifacts if committed_artifacts else "none",
        "none",
        "git ls-files empty for artifact body path",
        "Ignored artifact body must remain uncommitted.",
    )
    return checks


def write_summary(checks: list[dict[str, Any]], runner_rows: list[dict[str, str]]) -> None:
    targets = [row for row in runner_rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in runner_rows if row["shape_type"] == "rbc_like_smooth_pit"]
    all_pass = all(bool(row["pass"]) for row in checks)
    lines = [
        "25.7 surface forward-refinement inference runner verification",
        "",
        f"all_checks_pass: {all_pass}",
        f"check_count: {len(checks)}",
        f"target_subset_count: {len(targets)}",
        f"target_baseline_profile_rmse_mean_m: {mean(targets, 'baseline_profile_depth_rmse_m'):.12g}",
        f"target_refined_profile_rmse_mean_m: {mean(targets, 'refined_profile_depth_rmse_m'):.12g}",
        f"target_baseline_Er_like_mean: {mean(targets, 'baseline_Er_like_error'):.12g}",
        f"target_refined_Er_like_mean: {mean(targets, 'refined_Er_like_error'):.12g}",
        f"target_baseline_IoU_mean: {mean(targets, 'baseline_projected_mask_IoU'):.12g}",
        f"target_refined_IoU_mean: {mean(targets, 'refined_projected_mask_IoU'):.12g}",
        f"target_baseline_Dice_mean: {mean(targets, 'baseline_projected_mask_Dice'):.12g}",
        f"target_refined_Dice_mean: {mean(targets, 'refined_projected_mask_Dice'):.12g}",
        f"target_forward_residual_before_mean: {mean(targets, 'feature_residual_mse_before'):.12g}",
        f"target_forward_residual_after_mean: {mean(targets, 'feature_residual_mse_after'):.12g}",
        "",
        f"rbc_like_control_baseline_rmse_mean_m: {mean(rbc_like, 'baseline_profile_depth_rmse_m'):.12g}",
        f"rbc_like_control_refined_rmse_mean_m: {mean(rbc_like, 'refined_profile_depth_rmse_m'):.12g}",
        f"rbc_like_control_baseline_Dice_mean: {mean(rbc_like, 'baseline_projected_mask_Dice'):.12g}",
        f"rbc_like_control_refined_Dice_mean: {mean(rbc_like, 'refined_projected_mask_Dice'):.12g}",
        "multi_pit_policy: not_suitable_for_rbc_refinement; no RBC success credit.",
        "CURRENT_BASELINE_updated: false",
        f"verification_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in [RUNNER_METRICS, FORMAL_METRICS, DEFAULT_MANIFEST]:
        if not path.exists():
            raise FileNotFoundError(path)
    runner_rows = read_csv(RUNNER_METRICS)
    formal_rows = read_csv(FORMAL_METRICS)
    manifest = read_json(DEFAULT_MANIFEST)
    checks = build_checks(runner_rows, formal_rows, manifest)
    write_csv(MATRIX, checks, FIELDS)
    write_summary(checks, runner_rows)
    if not all(bool(row["pass"]) for row in checks):
        failed = [row["check_name"] for row in checks if not bool(row["pass"])]
        raise RuntimeError(f"25.7 runner verification failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
