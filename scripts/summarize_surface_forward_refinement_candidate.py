#!/usr/bin/env python
"""Summarize the 25.6 surface forward-refinement candidate."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from build_surface_forward_refinement_target_set import ROOT, as_bool, as_float, read_csv, write_csv
from evaluate_surface_forward_refinement_formal_acceptance_gates import MATRIX as GATE_MATRIX
from run_surface_forward_refinement_formal_benchmark import METRICS as FORMAL_METRICS


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_candidate_report.txt"
COMPARISON = ROOT / "results/metrics/surface_forward_refinement_candidate_comparison_matrix.csv"

FIELDS = [
    "candidate_or_route",
    "role",
    "applicable_scope",
    "not_applicable_scope",
    "baseline_relation",
    "target_profile_rmse_m",
    "target_Er_like",
    "target_IoU",
    "target_Dice",
    "target_forward_residual",
    "multi_pit_success_credit_allowed",
    "baseline_transition_allowed",
    "recommended_next_handling",
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


def gate_pass(gates: list[dict[str, str]]) -> bool:
    return bool(gates) and all(as_bool(row["pass"]) for row in gates)


def build_rows(metrics: list[dict[str, str]], gates: list[dict[str, str]]) -> list[dict[str, Any]]:
    targets = [row for row in metrics if row["target_role"] == "refinement_target"]
    return [
        {
            "candidate_or_route": "20.85_frozen_surface_RBC_baseline",
            "role": "frozen_reference",
            "applicable_scope": "surface RBC-style six-parameter prediction before post-hoc refinement",
            "not_applicable_scope": "non-RBC representation failures and multi-pit component-set rows",
            "baseline_relation": "current baseline reference only",
            "target_profile_rmse_m": mean(targets, "baseline_profile_depth_rmse_m"),
            "target_Er_like": mean(targets, "baseline_Er_like_error"),
            "target_IoU": mean(targets, "baseline_projected_mask_IoU"),
            "target_Dice": mean(targets, "baseline_projected_mask_Dice"),
            "target_forward_residual": mean(targets, "feature_residual_mse_before"),
            "multi_pit_success_credit_allowed": False,
            "baseline_transition_allowed": False,
            "recommended_next_handling": "keep as frozen initialization source for the refinement candidate",
        },
        {
            "candidate_or_route": "25.6_F0_R1_surface_forward_refinement_candidate",
            "role": "formal_refinement_candidate",
            "applicable_scope": "rbc_representable_but_model_fail rows with six-param RBC oracle fit",
            "not_applicable_scope": "multi-pit / rbc_not_representable / component_set representation failures",
            "baseline_relation": "frozen 20.85 baseline plus post-hoc parameter refinement; no weight update",
            "target_profile_rmse_m": mean(targets, "refined_profile_depth_rmse_m"),
            "target_Er_like": mean(targets, "refined_Er_like_error"),
            "target_IoU": mean(targets, "refined_projected_mask_IoU"),
            "target_Dice": mean(targets, "refined_projected_mask_Dice"),
            "target_forward_residual": mean(targets, "feature_residual_mse_after"),
            "multi_pit_success_credit_allowed": False,
            "baseline_transition_allowed": False,
            "recommended_next_handling": "export a dedicated inference artifact/runner if formal route gate selects A",
        },
        {
            "candidate_or_route": "RBC_oracle_fit_ceiling",
            "role": "oracle_reference_not_runtime",
            "applicable_scope": "metric ceiling for RBC-representable rows",
            "not_applicable_scope": "test-time inference; oracle labels are not runtime inputs",
            "baseline_relation": "not a deployable model",
            "target_profile_rmse_m": mean(targets, "oracle_profile_depth_rmse_m"),
            "target_Er_like": mean(targets, "oracle_Er_like_error"),
            "target_IoU": mean(targets, "oracle_projected_mask_IoU"),
            "target_Dice": mean(targets, "oracle_projected_mask_Dice"),
            "target_forward_residual": "",
            "multi_pit_success_credit_allowed": False,
            "baseline_transition_allowed": False,
            "recommended_next_handling": "retain as evaluation ceiling only",
        },
        {
            "candidate_or_route": "component_set_branch_for_multi_pit",
            "role": "future_representation_branch",
            "applicable_scope": "multi-pit / component-set surface defects",
            "not_applicable_scope": "single-component RBC-representable model-fail target set",
            "baseline_relation": "separate branch, not a 20.85 post-hoc refinement",
            "target_profile_rmse_m": "",
            "target_Er_like": "",
            "target_IoU": "",
            "target_Dice": "",
            "target_forward_residual": "",
            "multi_pit_success_credit_allowed": "only inside future component-set branch, not RBC refinement",
            "baseline_transition_allowed": False,
            "recommended_next_handling": "record as future branch after F0/R1 runner export decision",
        },
    ]


def write_summary(metrics: list[dict[str, str]], gates: list[dict[str, str]]) -> None:
    targets = [row for row in metrics if row["target_role"] == "refinement_target"]
    negative = [row for row in metrics if row["target_role"] == "excluded_negative_control"]
    pass_refs = [row for row in metrics if row["target_role"] == "already_pass_reference"]
    role_counts = Counter(row["target_role"] for row in metrics)
    shape_counts = Counter(row["shape_type"] for row in negative)
    all_gates_pass = gate_pass(gates)
    lines = [
        "25.6 surface forward-refinement candidate report",
        "",
        f"refinement_candidate_formed: {all_gates_pass}",
        "candidate_name: 25.6_F0_R1_surface_forward_refinement_candidate",
        "candidate_scope: rbc_representable_but_model_fail rows",
        "not_applicable_scope: multi-pit / rbc_not_representable / component_set representation failures",
        "baseline_relation: frozen 20.85 baseline plus post-hoc six-parameter refinement; model weights unchanged.",
        "why_not_baseline_transition: no CURRENT_BASELINE update was requested or allowed; this is a runtime refinement candidate needing a dedicated artifact/runner and later approval before any baseline discussion.",
        "",
        f"target_role_counts: {dict(role_counts)}",
        f"refinement_target_count: {len(targets)}",
        f"already_pass_reference_count: {len(pass_refs)}",
        f"excluded_negative_control_count: {len(negative)}",
        f"negative_control_shapes: {dict(shape_counts)}",
        "",
        f"target_baseline_profile_rmse_mean_m: {mean(targets, 'baseline_profile_depth_rmse_m'):.12g}",
        f"target_refined_profile_rmse_mean_m: {mean(targets, 'refined_profile_depth_rmse_m'):.12g}",
        f"target_oracle_profile_rmse_mean_m: {mean(targets, 'oracle_profile_depth_rmse_m'):.12g}",
        f"target_baseline_Er_like_mean: {mean(targets, 'baseline_Er_like_error'):.12g}",
        f"target_refined_Er_like_mean: {mean(targets, 'refined_Er_like_error'):.12g}",
        f"target_baseline_IoU_Dice_mean: {mean(targets, 'baseline_projected_mask_IoU'):.12g}/{mean(targets, 'baseline_projected_mask_Dice'):.12g}",
        f"target_refined_IoU_Dice_mean: {mean(targets, 'refined_projected_mask_IoU'):.12g}/{mean(targets, 'refined_projected_mask_Dice'):.12g}",
        f"target_forward_residual_before_after: {mean(targets, 'feature_residual_mse_before'):.12g} -> {mean(targets, 'feature_residual_mse_after'):.12g}",
        "",
        "inference_artifact_runner_value: yes; the candidate is worth a dedicated runner/export route, still without baseline transition.",
        "component_set_branch_needed_for_multi_pit: yes; multi-pit remains a representation branch, not a six-param refinement success.",
        f"comparison_matrix: {COMPARISON}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not FORMAL_METRICS.exists():
        raise FileNotFoundError(FORMAL_METRICS)
    if not GATE_MATRIX.exists():
        raise FileNotFoundError(GATE_MATRIX)
    metrics = read_csv(FORMAL_METRICS)
    gates = read_csv(GATE_MATRIX)
    rows = build_rows(metrics, gates)
    write_csv(COMPARISON, rows, FIELDS)
    write_summary(metrics, gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
