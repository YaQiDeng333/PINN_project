#!/usr/bin/env python
"""Rerun formal 25.6 acceptance gates for the fixed refinement candidate."""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_surface_forward_refinement_target_set import ROOT, as_bool, as_float, read_csv, write_csv
from run_surface_forward_refinement_formal_benchmark import METRICS as FORMAL_METRICS


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_formal_acceptance_gate_summary.txt"
MATRIX = ROOT / "results/metrics/surface_forward_refinement_formal_acceptance_gate_matrix.csv"

FIELDS = [
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


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception as exc:
        return f"git_error:{exc}"


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


def build_gates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in rows if row["shape_type"] == "rbc_like_smooth_pit"]
    multi_pit = [row for row in rows if row["shape_type"] == "multi_pit_two_component_surface_defect"]
    refined_rows = [row for row in rows if as_bool(row["refinement_applied"])]
    gates: list[dict[str, Any]] = []

    base_rmse = mean(targets, "baseline_profile_depth_rmse_m")
    refined_rmse = mean(targets, "refined_profile_depth_rmse_m")
    oracle_rmse = mean(targets, "oracle_profile_depth_rmse_m")
    gates.append(
        gate_row(
            "primary",
            "P1_target_profile_rmse",
            "refinement_target",
            "profile_depth_rmse_m_mean",
            base_rmse,
            refined_rmse,
            oracle_rmse,
            "refined < baseline",
            refined_rmse < base_rmse,
            "Target subset profile RMSE must decrease.",
        )
    )

    base_er = mean(targets, "baseline_Er_like_error")
    refined_er = mean(targets, "refined_Er_like_error")
    gates.append(
        gate_row(
            "primary",
            "P2_target_er_like",
            "refinement_target",
            "Er_like_error_mean",
            base_er,
            refined_er,
            mean(targets, "oracle_Er_like_error"),
            "refined < baseline",
            refined_er < base_er,
            "Target subset Er-like error must decrease.",
        )
    )

    base_iou = mean(targets, "baseline_projected_mask_IoU")
    refined_iou = mean(targets, "refined_projected_mask_IoU")
    base_dice = mean(targets, "baseline_projected_mask_Dice")
    refined_dice = mean(targets, "refined_projected_mask_Dice")
    gates.append(
        gate_row(
            "primary",
            "P3_target_iou_dice",
            "refinement_target",
            "IoU_mean_and_Dice_mean",
            f"IoU={base_iou}; Dice={base_dice}",
            f"IoU={refined_iou}; Dice={refined_dice}",
            f"IoU={mean(targets, 'oracle_projected_mask_IoU')}; Dice={mean(targets, 'oracle_projected_mask_Dice')}",
            "IoU and Dice both increase",
            bool(refined_iou > base_iou and refined_dice > base_dice),
            "Target subset projected-mask IoU and Dice must improve together.",
        )
    )

    base_residual = mean(targets, "feature_residual_mse_before")
    refined_residual = mean(targets, "feature_residual_mse_after")
    gates.append(
        gate_row(
            "primary",
            "P4_forward_residual",
            "refinement_target",
            "feature_residual_mse_mean",
            base_residual,
            refined_residual,
            "not_applicable",
            "refined residual < baseline residual",
            refined_residual < base_residual,
            "Forward feature residual must decrease.",
        )
    )

    gates.append(
        gate_row(
            "primary",
            "P5_residual_profile_alignment",
            "refinement_target",
            "forward_residual_and_profile_mask",
            f"residual={base_residual}; rmse={base_rmse}; iou={base_iou}; dice={base_dice}",
            f"residual={refined_residual}; rmse={refined_rmse}; iou={refined_iou}; dice={refined_dice}",
            "not_applicable",
            "residual decreases while RMSE decreases and IoU/Dice increase",
            bool(refined_residual < base_residual and refined_rmse < base_rmse and refined_iou > base_iou and refined_dice > base_dice),
            "Forward residual and profile/mask metrics must improve in the same direction.",
        )
    )

    rbc_base_rmse = mean(rbc_like, "baseline_profile_depth_rmse_m")
    rbc_refined_rmse = mean(rbc_like, "refined_profile_depth_rmse_m")
    rbc_base_dice = mean(rbc_like, "baseline_projected_mask_Dice")
    rbc_refined_dice = mean(rbc_like, "refined_projected_mask_Dice")
    gates.append(
        gate_row(
            "secondary",
            "S1_rbc_like_control",
            "rbc_like_smooth_pit",
            "profile_rmse_and_Dice",
            f"rmse={rbc_base_rmse}; dice={rbc_base_dice}",
            f"rmse={rbc_refined_rmse}; dice={rbc_refined_dice}",
            "not_applicable",
            "RMSE does not increase and Dice does not decrease",
            bool(rbc_refined_rmse <= rbc_base_rmse and rbc_refined_dice >= rbc_base_dice),
            "RBC-like control must not degrade.",
        )
    )

    multi_pit_ok = all(
        as_bool(row["include_as_negative_control"]) and not as_bool(row["refinement_applied"]) and not as_bool(row["include_in_success_gate"])
        for row in multi_pit
    )
    gates.append(
        gate_row(
            "failure",
            "F1_multi_pit_success_credit",
            "rbc_not_representable_or_multi_pit",
            "success_accounting",
            len(multi_pit),
            "negative_control_no_refinement",
            "not_applicable",
            "multi-pit/rbc_not_representable rows are not RBC refinement success",
            bool(multi_pit_ok),
            "Representation failures must remain negative controls.",
        )
    )

    lwd_bounds_ok = all(
        0.0015 <= as_float(row["refined_L_m"]) <= 0.035
        and 0.00075 <= as_float(row["refined_W_m"]) <= 0.018
        and 0.00005 <= as_float(row["refined_D_m"]) <= 0.0045
        for row in refined_rows
    )
    gates.append(
        gate_row(
            "failure",
            "F2_lwd_physical_bounds",
            "all_refined_rows",
            "L_W_D_bounds",
            "declared_R1_bounds",
            "checked",
            "not_applicable",
            "L/W/D remain within R1 bounds",
            bool(lwd_bounds_ok),
            "Physical size/depth parameters must remain bounded.",
        )
    )

    w_bounds_ok = all(
        0.03 <= as_float(row["refined_wLD"]) <= 10.0
        and 0.03 <= as_float(row["refined_wWD"]) <= 10.0
        and 0.03 <= as_float(row["refined_wLW"]) <= 10.0
        for row in refined_rows
    )
    gates.append(
        gate_row(
            "failure",
            "F3_w_parameter_bounds",
            "all_refined_rows",
            "wLD_wWD_wLW_bounds",
            "declared_R1_bounds",
            "checked",
            "not_applicable",
            "w parameters remain within [0.03, 10.0]",
            bool(w_bounds_ok),
            "Curvature weights must remain bounded.",
        )
    )

    baseline_diff = git_value(["diff", "--name-only", "--", "CURRENT_BASELINE.md"])
    gates.append(
        gate_row(
            "failure",
            "F4_current_baseline_unchanged",
            "repository_docs",
            "CURRENT_BASELINE.md",
            "unchanged_required",
            baseline_diff if baseline_diff else "unchanged",
            "not_applicable",
            "CURRENT_BASELINE.md unchanged",
            not bool(baseline_diff),
            "Formal benchmark is not a baseline transition.",
        )
    )
    return gates


def write_summary(gates: list[dict[str, Any]], rows: list[dict[str, str]]) -> None:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in rows if row["shape_type"] == "rbc_like_smooth_pit"]
    gate_counts = Counter("PASS" if as_bool(row["pass"]) else "FAIL" for row in gates)
    all_pass = all(as_bool(row["pass"]) for row in gates)
    lines = [
        "25.6 surface forward-refinement formal acceptance gates",
        "",
        f"gate_counts: {dict(gate_counts)}",
        f"all_gates_pass: {all_pass}",
        f"target_subset_count: {len(targets)}",
        f"target_baseline_profile_rmse_mean_m: {mean(targets, 'baseline_profile_depth_rmse_m'):.12g}",
        f"target_refined_profile_rmse_mean_m: {mean(targets, 'refined_profile_depth_rmse_m'):.12g}",
        f"target_oracle_profile_rmse_mean_m: {mean(targets, 'oracle_profile_depth_rmse_m'):.12g}",
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
        "multi_pit_policy: negative control only; no RBC refinement success credit.",
        f"acceptance_gate_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not FORMAL_METRICS.exists():
        raise FileNotFoundError(f"formal metrics missing; run scripts/run_surface_forward_refinement_formal_benchmark.py first: {FORMAL_METRICS}")
    rows = read_csv(FORMAL_METRICS)
    gates = build_gates(rows)
    write_csv(MATRIX, gates, FIELDS)
    write_summary(gates, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
