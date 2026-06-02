#!/usr/bin/env python
"""Decide the next route after the 25.5 forward-refinement diagnostic."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from build_surface_forward_refinement_target_set import ROOT, as_bool, as_float, read_csv, write_csv
from evaluate_surface_forward_refinement_vs_oracle import ACCEPTANCE_GATES, VS_ORACLE_METRICS
from fit_surface_feature_space_forward_surrogate import SURROGATE_VALIDATION
from run_surface_rbc_forward_consistency_refinement import METRICS as RUN_METRICS


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_diagnostic_route_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/surface_forward_refinement_diagnostic_decision_matrix.csv"

FIELDS = [
    "option",
    "selected",
    "decision",
    "evidence",
    "blocked_by",
]


def require_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    for path in [ACCEPTANCE_GATES, VS_ORACLE_METRICS, SURROGATE_VALIDATION, RUN_METRICS]:
        if not path.exists():
            raise FileNotFoundError(path)
    return read_csv(ACCEPTANCE_GATES), read_csv(VS_ORACLE_METRICS), read_csv(SURROGATE_VALIDATION), read_csv(RUN_METRICS)


def target_summary(vs_rows: list[dict[str, str]]) -> dict[str, str]:
    for row in vs_rows:
        if row["group_field"] == "target_role" and row["group_value"] == "refinement_target" and row["split"] == "all":
            return row
    raise RuntimeError("target_role/refinement_target/all row missing from vs-oracle metrics")


def selected_surrogate(validation_rows: list[dict[str, str]]) -> str:
    selected = [row for row in validation_rows if as_bool(row["selected"])]
    if len(selected) != 1:
        raise RuntimeError("expected exactly one selected surrogate candidate")
    return selected[0]["candidate_id"]


def decide(
    gates: list[dict[str, str]],
    vs_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    run_rows: list[dict[str, str]],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    gate_pass = {row["gate_id"]: as_bool(row["pass"]) for row in gates}
    primary_pass = all(as_bool(row["pass"]) for row in gates if row["gate_group"] == "primary")
    secondary_pass = all(as_bool(row["pass"]) for row in gates if row["gate_group"] == "secondary")
    failure_pass = all(as_bool(row["pass"]) for row in gates if row["gate_group"] == "failure")
    candidate_formed = primary_pass and secondary_pass and failure_pass
    target = target_summary(vs_rows)
    residual_improved = as_float(target["feature_residual_after_mean"]) < as_float(target["feature_residual_before_mean"])
    profile_improved = as_float(target["refined_profile_rmse_mean_m"]) < as_float(target["baseline_profile_rmse_mean_m"])
    mask_improved = (
        as_float(target["refined_IoU_mean"]) > as_float(target["baseline_IoU_mean"])
        and as_float(target["refined_Dice_mean"]) > as_float(target["baseline_Dice_mean"])
    )
    multi_pit_ok = gate_pass.get("F2_multi_pit_credit", False)
    current_baseline_ok = gate_pass.get("F3_baseline_transition", False)
    surrogate = selected_surrogate(validation_rows)
    target_count = int(target["sample_count"])

    if candidate_formed:
        selected_option = "A. lock 25.5 F0/R1 candidate for a formal no-baseline-transition benchmark"
    elif residual_improved and not (profile_improved and mask_improved):
        selected_option = "B. diagnose surrogate mismatch before any further six-param refinement"
    elif profile_improved and not mask_improved:
        selected_option = "C. switch next diagnostic to profile-basis or contour-aware representation"
    elif not residual_improved:
        selected_option = "F. stop the F0/R1 forward-refinement route and return to model-route design"
    else:
        selected_option = "E. rerun feature extraction design with stricter validation gates"

    options = [
        {
            "option": "A. lock 25.5 F0/R1 candidate for a formal no-baseline-transition benchmark",
            "selected": selected_option.startswith("A."),
            "decision": "selected" if selected_option.startswith("A.") else "not_selected",
            "evidence": (
                f"candidate_formed={candidate_formed}; primary_pass={primary_pass}; secondary_pass={secondary_pass}; "
                f"failure_pass={failure_pass}; target_count={target_count}; surrogate={surrogate}"
            ),
            "blocked_by": "" if selected_option.startswith("A.") else "one or more acceptance gates did not pass",
        },
        {
            "option": "B. diagnose surrogate mismatch before any further six-param refinement",
            "selected": selected_option.startswith("B."),
            "decision": "selected" if selected_option.startswith("B.") else "not_selected",
            "evidence": (
                f"residual_improved={residual_improved}; profile_improved={profile_improved}; "
                f"mask_improved={mask_improved}; P1={gate_pass.get('P1_profile_rmse')}; "
                f"P3={gate_pass.get('P3_projected_mask')}"
            ),
            "blocked_by": "" if selected_option.startswith("B.") else "surrogate residual/profile relation did not match this branch",
        },
        {
            "option": "C. switch next diagnostic to profile-basis or contour-aware representation",
            "selected": selected_option.startswith("C."),
            "decision": "selected" if selected_option.startswith("C.") else "not_selected",
            "evidence": (
                f"profile_improved={profile_improved}; mask_improved={mask_improved}; "
                "six-param RBC footprint may be the limiting representation for several shape families"
            ),
            "blocked_by": "" if selected_option.startswith("C.") else "profile/mask pattern did not require this as the unique next step",
        },
        {
            "option": "D. move multi-pit/component-set branch forward",
            "selected": False,
            "decision": "not_selected",
            "evidence": f"multi_pit_accounting_ok={multi_pit_ok}; multi-pit is excluded_negative_control in 25.5",
            "blocked_by": "does not address the 82 RBC-representable model-fail rows",
        },
        {
            "option": "E. rerun feature extraction design with stricter validation gates",
            "selected": selected_option.startswith("E."),
            "decision": "selected" if selected_option.startswith("E.") else "not_selected",
            "evidence": f"residual_improved={residual_improved}; profile_improved={profile_improved}; mask_improved={mask_improved}",
            "blocked_by": "" if selected_option.startswith("E.") else "another branch better matches the measured failure",
        },
        {
            "option": "F. stop the F0/R1 forward-refinement route and return to model-route design",
            "selected": selected_option.startswith("F."),
            "decision": "selected" if selected_option.startswith("F.") else "not_selected",
            "evidence": f"residual_improved={residual_improved}; current_baseline_unchanged={current_baseline_ok}",
            "blocked_by": "" if selected_option.startswith("F.") else "some forward-refinement signal remains worth diagnosing",
        },
    ]
    context = {
        "selected_option": selected_option,
        "selected_surrogate": surrogate,
        "target_count": target_count,
        "primary_pass": primary_pass,
        "secondary_pass": secondary_pass,
        "failure_pass": failure_pass,
        "candidate_formed": candidate_formed,
        "residual_improved": residual_improved,
        "profile_improved": profile_improved,
        "mask_improved": mask_improved,
        "multi_pit_ok": multi_pit_ok,
        "current_baseline_ok": current_baseline_ok,
        "run_row_count": len(run_rows),
        "target_baseline_rmse": as_float(target["baseline_profile_rmse_mean_m"]),
        "target_refined_rmse": as_float(target["refined_profile_rmse_mean_m"]),
        "target_baseline_dice": as_float(target["baseline_Dice_mean"]),
        "target_refined_dice": as_float(target["refined_Dice_mean"]),
        "target_residual_before": as_float(target["feature_residual_before_mean"]),
        "target_residual_after": as_float(target["feature_residual_after_mean"]),
    }
    return selected_option, options, context


def write_summary(context: dict[str, Any]) -> None:
    lines = [
        "25.5 surface forward-refinement diagnostic route decision",
        "",
        f"decision: {context['selected_option']}",
        f"selected_surrogate: {context['selected_surrogate']}",
        f"target_subset_count: {context['target_count']}",
        f"refinement_candidate_formed: {context['candidate_formed']}",
        f"primary_gates_pass: {context['primary_pass']}",
        f"secondary_gates_pass: {context['secondary_pass']}",
        f"failure_gates_pass: {context['failure_pass']}",
        f"forward_residual_improved: {context['residual_improved']}",
        f"profile_improved: {context['profile_improved']}",
        f"mask_improved: {context['mask_improved']}",
        f"multi_pit_accounting_ok: {context['multi_pit_ok']}",
        f"CURRENT_BASELINE_unchanged: {context['current_baseline_ok']}",
        "",
        f"target_baseline_profile_rmse_mean_m: {context['target_baseline_rmse']:.12g}",
        f"target_refined_profile_rmse_mean_m: {context['target_refined_rmse']:.12g}",
        f"target_baseline_Dice_mean: {context['target_baseline_dice']:.12g}",
        f"target_refined_Dice_mean: {context['target_refined_dice']:.12g}",
        f"target_feature_residual_before_mean: {context['target_residual_before']:.12g}",
        f"target_feature_residual_after_mean: {context['target_residual_after']:.12g}",
        "",
        "baseline_transition_allowed: false",
        "COMSOL_allowed_next_without_new_plan: false",
        "main_neural_training_allowed_next_without_new_plan: false",
        f"decision_matrix: {DECISION_MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    gates, vs_rows, validation_rows, run_rows = require_inputs()
    _selected_option, options, context = decide(gates, vs_rows, validation_rows, run_rows)
    write_csv(DECISION_MATRIX, options, FIELDS)
    write_summary(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
