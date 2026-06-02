#!/usr/bin/env python
"""Evaluate 25.5 forward-refinement results against oracle/profile gates."""

from __future__ import annotations

import csv
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from build_surface_forward_refinement_target_set import (
    ROOT,
    TARGET_MATERIALIZED,
    as_bool,
    as_float,
    read_csv,
    write_csv,
)
from run_surface_rbc_forward_consistency_refinement import METRICS as RUN_METRICS


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_vs_oracle_summary.txt"
VS_ORACLE_METRICS = ROOT / "results/metrics/surface_forward_refinement_vs_oracle_metrics.csv"
ACCEPTANCE_GATES = ROOT / "results/metrics/surface_forward_refinement_acceptance_gate_results.csv"

VS_FIELDS = [
    "group_field",
    "group_value",
    "split",
    "sample_count",
    "baseline_profile_rmse_mean_m",
    "refined_profile_rmse_mean_m",
    "oracle_profile_rmse_mean_m",
    "refined_minus_baseline_rmse_mean_m",
    "refined_minus_oracle_rmse_mean_m",
    "baseline_Er_like_mean",
    "refined_Er_like_mean",
    "oracle_Er_like_mean",
    "baseline_IoU_mean",
    "refined_IoU_mean",
    "oracle_IoU_mean",
    "baseline_Dice_mean",
    "refined_Dice_mean",
    "oracle_Dice_mean",
    "baseline_area_error_mean",
    "refined_area_error_mean",
    "oracle_area_error_mean",
    "feature_residual_before_mean",
    "feature_residual_after_mean",
    "feature_residual_delta_mean",
    "refinement_applied_rate",
    "success_credit_allowed",
]

GATE_FIELDS = [
    "gate_group",
    "gate_id",
    "subset",
    "metric",
    "baseline_value",
    "refined_value",
    "oracle_value",
    "threshold_or_condition",
    "pass",
    "interpretation",
]


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


def joined_oracle(rows: list[dict[str, str]], materialized_by_id: dict[str, dict[str, str]], oracle_key: str) -> float:
    vals: list[float] = []
    for row in rows:
        source = materialized_by_id[row["sample_id"]]
        val = as_float(source[oracle_key])
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception as exc:
        return f"git_error:{exc}"


def add_vs_row(
    out: list[dict[str, Any]],
    rows: list[dict[str, str]],
    materialized_by_id: dict[str, dict[str, str]],
    group_field: str,
    group_value: str,
    split: str,
) -> None:
    if not rows:
        return
    success_credit = all(as_bool(row["include_in_success_gate"]) for row in rows)
    out.append(
        {
            "group_field": group_field,
            "group_value": group_value,
            "split": split,
            "sample_count": len(rows),
            "baseline_profile_rmse_mean_m": mean(rows, "baseline_profile_depth_rmse_m"),
            "refined_profile_rmse_mean_m": mean(rows, "refined_profile_depth_rmse_m"),
            "oracle_profile_rmse_mean_m": joined_oracle(rows, materialized_by_id, "oracle_profile_depth_rmse_m"),
            "refined_minus_baseline_rmse_mean_m": mean(rows, "profile_rmse_delta_m"),
            "refined_minus_oracle_rmse_mean_m": mean(rows, "refined_profile_depth_rmse_m")
            - joined_oracle(rows, materialized_by_id, "oracle_profile_depth_rmse_m"),
            "baseline_Er_like_mean": mean(rows, "baseline_Er_like_error"),
            "refined_Er_like_mean": mean(rows, "refined_Er_like_error"),
            "oracle_Er_like_mean": joined_oracle(rows, materialized_by_id, "oracle_Er_like_error"),
            "baseline_IoU_mean": mean(rows, "baseline_projected_mask_IoU"),
            "refined_IoU_mean": mean(rows, "refined_projected_mask_IoU"),
            "oracle_IoU_mean": joined_oracle(rows, materialized_by_id, "oracle_projected_mask_IoU"),
            "baseline_Dice_mean": mean(rows, "baseline_projected_mask_Dice"),
            "refined_Dice_mean": mean(rows, "refined_projected_mask_Dice"),
            "oracle_Dice_mean": joined_oracle(rows, materialized_by_id, "oracle_projected_mask_Dice"),
            "baseline_area_error_mean": mean(rows, "baseline_area_error"),
            "refined_area_error_mean": mean(rows, "refined_area_error"),
            "oracle_area_error_mean": joined_oracle(rows, materialized_by_id, "oracle_area_error"),
            "feature_residual_before_mean": mean(rows, "feature_residual_mse_before"),
            "feature_residual_after_mean": mean(rows, "feature_residual_mse_after"),
            "feature_residual_delta_mean": mean(rows, "feature_residual_delta"),
            "refinement_applied_rate": float(np.mean([as_bool(row["refinement_applied"]) for row in rows])),
            "success_credit_allowed": success_credit,
        }
    )


def build_vs_metrics(rows: list[dict[str, str]], materialized_by_id: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["all", "train", "val", "test"]:
        split_rows = rows if split == "all" else [row for row in rows if row["split"] == split]
        for role in sorted({row["target_role"] for row in rows}):
            add_vs_row(out, [row for row in split_rows if row["target_role"] == role], materialized_by_id, "target_role", role, split)
        add_vs_row(out, [row for row in split_rows if row["shape_type"] == "rbc_like_smooth_pit"], materialized_by_id, "control", "rbc_like_smooth_pit", split)
        add_vs_row(out, [row for row in split_rows if row["target_role"] == "refinement_target" and row["shape_type"] != "rbc_like_smooth_pit"], materialized_by_id, "control", "non_rbc_refinement_target", split)
    for shape in sorted({row["shape_type"] for row in rows}):
        add_vs_row(out, [row for row in rows if row["shape_type"] == shape], materialized_by_id, "shape_type", shape, "all")
    return out


def gate_row(
    group: str,
    gate_id: str,
    subset: str,
    metric: str,
    baseline: float | str,
    refined: float | str,
    oracle: float | str,
    condition: str,
    passed: bool,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "gate_group": group,
        "gate_id": gate_id,
        "subset": subset,
        "metric": metric,
        "baseline_value": baseline,
        "refined_value": refined,
        "oracle_value": oracle,
        "threshold_or_condition": condition,
        "pass": bool(passed),
        "interpretation": interpretation,
    }


def build_acceptance_gates(rows: list[dict[str, str]], materialized_by_id: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in rows if row["shape_type"] == "rbc_like_smooth_pit"]
    multi_pit = [row for row in rows if row["shape_type"] == "multi_pit_two_component_surface_defect"]
    refined_rows = [row for row in rows if as_bool(row["refinement_applied"])]
    gates: list[dict[str, Any]] = []

    target_base_rmse = mean(targets, "baseline_profile_depth_rmse_m")
    target_ref_rmse = mean(targets, "refined_profile_depth_rmse_m")
    target_oracle_rmse = joined_oracle(targets, materialized_by_id, "oracle_profile_depth_rmse_m")
    gates.append(
        gate_row(
            "primary",
            "P1_profile_rmse",
            "refinement_target",
            "profile_depth_rmse_m_mean",
            target_base_rmse,
            target_ref_rmse,
            target_oracle_rmse,
            "refined <= 0.98 * baseline",
            bool(target_ref_rmse <= 0.98 * target_base_rmse),
            "Profile RMSE must improve on the RBC-representable model-fail subset.",
        )
    )

    target_base_er = mean(targets, "baseline_Er_like_error")
    target_ref_er = mean(targets, "refined_Er_like_error")
    gates.append(
        gate_row(
            "primary",
            "P2_er_like",
            "refinement_target",
            "Er_like_error_mean",
            target_base_er,
            target_ref_er,
            joined_oracle(targets, materialized_by_id, "oracle_Er_like_error"),
            "refined <= 0.98 * baseline",
            bool(target_ref_er <= 0.98 * target_base_er),
            "Er-like error must improve with the same target subset.",
        )
    )

    target_base_iou = mean(targets, "baseline_projected_mask_IoU")
    target_ref_iou = mean(targets, "refined_projected_mask_IoU")
    target_base_dice = mean(targets, "baseline_projected_mask_Dice")
    target_ref_dice = mean(targets, "refined_projected_mask_Dice")
    gates.append(
        gate_row(
            "primary",
            "P3_projected_mask",
            "refinement_target",
            "IoU_mean_and_Dice_mean",
            f"IoU={target_base_iou}; Dice={target_base_dice}",
            f"IoU={target_ref_iou}; Dice={target_ref_dice}",
            f"IoU={joined_oracle(targets, materialized_by_id, 'oracle_projected_mask_IoU')}; Dice={joined_oracle(targets, materialized_by_id, 'oracle_projected_mask_Dice')}",
            "IoU >= baseline + 0.01 and Dice >= baseline + 0.01",
            bool(target_ref_iou >= target_base_iou + 0.01 and target_ref_dice >= target_base_dice + 0.01),
            "Projected mask metrics must both improve; no single metric substitution.",
        )
    )

    target_base_area = mean(targets, "baseline_area_error")
    target_ref_area = mean(targets, "refined_area_error")
    gates.append(
        gate_row(
            "primary",
            "P4_area_error",
            "refinement_target",
            "area_error_mean",
            target_base_area,
            target_ref_area,
            joined_oracle(targets, materialized_by_id, "oracle_area_error"),
            "refined <= max(1.05 * baseline, baseline + 0.02)",
            bool(target_ref_area <= max(1.05 * target_base_area, target_base_area + 0.02)),
            "Area error cannot worsen beyond the tolerance used in the 25.4 gate.",
        )
    )

    rbc_base_rmse = mean(rbc_like, "baseline_profile_depth_rmse_m")
    rbc_ref_rmse = mean(rbc_like, "refined_profile_depth_rmse_m")
    rbc_base_dice = mean(rbc_like, "baseline_projected_mask_Dice")
    rbc_ref_dice = mean(rbc_like, "refined_projected_mask_Dice")
    gates.append(
        gate_row(
            "secondary",
            "S1_rbc_like_control",
            "rbc_like_smooth_pit",
            "profile_rmse_and_Dice",
            f"rmse={rbc_base_rmse}; Dice={rbc_base_dice}",
            f"rmse={rbc_ref_rmse}; Dice={rbc_ref_dice}",
            "not_applicable",
            "rmse <= 1.05 * baseline and Dice >= baseline - 0.02",
            bool(rbc_ref_rmse <= 1.05 * rbc_base_rmse and rbc_ref_dice >= rbc_base_dice - 0.02),
            "RBC-like control cannot collapse while non-RBC rows improve.",
        )
    )

    shape_families = [
        "flat_bottom_pit",
        "sharp_wall_boxy_corrosion",
        "asymmetric_corrosion",
        "elongated_crack_like_surface_defect",
        "irregular_corrosion_non_rbc",
    ]
    improved_shapes = 0
    for shape in shape_families:
        shape_rows = [row for row in targets if row["shape_type"] == shape]
        if shape_rows and mean(shape_rows, "refined_profile_depth_rmse_m") < mean(shape_rows, "baseline_profile_depth_rmse_m"):
            improved_shapes += 1
    gates.append(
        gate_row(
            "secondary",
            "S2_shape_family_coverage",
            "non_multi_non_rbc_refinement_target",
            "per_shape_profile_rmse_mean",
            "5 families",
            improved_shapes,
            "not_applicable",
            "at least 4 of 5 families improve",
            bool(improved_shapes >= 4),
            "Aggregate improvement must cover most non-multi shape families.",
        )
    )

    base_residual = mean(targets, "feature_residual_mse_before")
    ref_residual = mean(targets, "feature_residual_mse_after")
    gates.append(
        gate_row(
            "secondary",
            "S3_forward_residual_alignment",
            "refinement_target",
            "forward_feature_residual_and_profile_rmse",
            f"residual={base_residual}; rmse={target_base_rmse}",
            f"residual={ref_residual}; rmse={target_ref_rmse}",
            "not_applicable",
            "forward residual decreases and profile RMSE decreases",
            bool(ref_residual < base_residual and target_ref_rmse < target_base_rmse),
            "Forward residual gains must align with profile improvement.",
        )
    )

    nonphysical = [
        row
        for row in refined_rows
        if not as_bool(row["params_in_bounds"]) or not as_bool(row["profile_nonnegative"])
    ]
    gates.append(
        gate_row(
            "failure",
            "F1_nonphysical_params",
            "all_refined_six_param_rows",
            "parameter_bounds_and_profile_physics",
            0,
            len(nonphysical),
            "not_applicable",
            "zero nonphysical refined rows",
            len(nonphysical) == 0,
            "All refined params must remain in declared R1 bounds with nonnegative profiles.",
        )
    )

    multi_pit_ok = all(
        as_bool(row["include_as_negative_control"]) and not as_bool(row["refinement_applied"]) and not as_bool(row["include_in_success_gate"])
        for row in multi_pit
    )
    gates.append(
        gate_row(
            "failure",
            "F2_multi_pit_credit",
            "multi_pit_two_component_surface_defect",
            "success_accounting",
            len(multi_pit),
            "negative_control_no_refinement",
            "not_applicable",
            "multi-pit excluded from RBC refinement success credit",
            bool(multi_pit_ok),
            "Multi-pit remains a future component-set branch.",
        )
    )

    baseline_diff = git_value(["diff", "--name-only", "--", "CURRENT_BASELINE.md"])
    gates.append(
        gate_row(
            "failure",
            "F3_baseline_transition",
            "repository_docs",
            "CURRENT_BASELINE.md",
            "unchanged_required",
            baseline_diff if baseline_diff else "unchanged",
            "not_applicable",
            "CURRENT_BASELINE.md unchanged",
            not bool(baseline_diff),
            "25.5 is diagnostic only; no baseline transition is allowed.",
        )
    )
    return gates


def write_summary(vs_rows: list[dict[str, Any]], gates: list[dict[str, Any]], run_rows: list[dict[str, str]]) -> None:
    target_summary = [row for row in vs_rows if row["group_field"] == "target_role" and row["group_value"] == "refinement_target" and row["split"] == "all"][0]
    rbc_summary = [row for row in vs_rows if row["group_field"] == "control" and row["group_value"] == "rbc_like_smooth_pit" and row["split"] == "all"][0]
    gate_counts = Counter("PASS" if as_bool(row["pass"]) else "FAIL" for row in gates)
    primary_pass = all(as_bool(row["pass"]) for row in gates if row["gate_group"] == "primary")
    failure_pass = all(as_bool(row["pass"]) for row in gates if row["gate_group"] == "failure")
    secondary_pass = all(as_bool(row["pass"]) for row in gates if row["gate_group"] == "secondary")
    candidate_formed = primary_pass and secondary_pass and failure_pass
    lines = [
        "25.5 surface forward-refinement vs oracle evaluation",
        "",
        f"target_subset_count: {target_summary['sample_count']}",
        f"target_baseline_profile_rmse_mean_m: {float(target_summary['baseline_profile_rmse_mean_m']):.12g}",
        f"target_refined_profile_rmse_mean_m: {float(target_summary['refined_profile_rmse_mean_m']):.12g}",
        f"target_oracle_profile_rmse_mean_m: {float(target_summary['oracle_profile_rmse_mean_m']):.12g}",
        f"target_baseline_Er_like_mean: {float(target_summary['baseline_Er_like_mean']):.12g}",
        f"target_refined_Er_like_mean: {float(target_summary['refined_Er_like_mean']):.12g}",
        f"target_baseline_IoU_mean: {float(target_summary['baseline_IoU_mean']):.12g}",
        f"target_refined_IoU_mean: {float(target_summary['refined_IoU_mean']):.12g}",
        f"target_baseline_Dice_mean: {float(target_summary['baseline_Dice_mean']):.12g}",
        f"target_refined_Dice_mean: {float(target_summary['refined_Dice_mean']):.12g}",
        f"target_forward_residual_before_mean: {float(target_summary['feature_residual_before_mean']):.12g}",
        f"target_forward_residual_after_mean: {float(target_summary['feature_residual_after_mean']):.12g}",
        "",
        f"rbc_like_control_baseline_rmse_mean_m: {float(rbc_summary['baseline_profile_rmse_mean_m']):.12g}",
        f"rbc_like_control_refined_rmse_mean_m: {float(rbc_summary['refined_profile_rmse_mean_m']):.12g}",
        f"rbc_like_control_baseline_Dice_mean: {float(rbc_summary['baseline_Dice_mean']):.12g}",
        f"rbc_like_control_refined_Dice_mean: {float(rbc_summary['refined_Dice_mean']):.12g}",
        "",
        f"gate_counts: {dict(gate_counts)}",
        f"primary_gates_pass: {primary_pass}",
        f"secondary_gates_pass: {secondary_pass}",
        f"failure_gates_pass: {failure_pass}",
        f"refinement_candidate_formed: {candidate_formed}",
        "multi_pit_handling: excluded_negative_control, no refinement applied, no success credit.",
        f"run_rows: {len(run_rows)}",
        f"vs_oracle_metrics_csv: {VS_ORACLE_METRICS}",
        f"acceptance_gate_results_csv: {ACCEPTANCE_GATES}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not RUN_METRICS.exists():
        raise FileNotFoundError(f"run metrics missing; run scripts/run_surface_rbc_forward_consistency_refinement.py first: {RUN_METRICS}")
    if not TARGET_MATERIALIZED.exists():
        raise FileNotFoundError(TARGET_MATERIALIZED)
    run_rows = read_csv(RUN_METRICS)
    materialized_rows = read_csv(TARGET_MATERIALIZED)
    materialized_by_id = {row["sample_id"]: row for row in materialized_rows}
    vs_rows = build_vs_metrics(run_rows, materialized_by_id)
    gates = build_acceptance_gates(run_rows, materialized_by_id)
    write_csv(VS_ORACLE_METRICS, vs_rows, VS_FIELDS)
    write_csv(ACCEPTANCE_GATES, gates, GATE_FIELDS)
    write_summary(vs_rows, gates, run_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
