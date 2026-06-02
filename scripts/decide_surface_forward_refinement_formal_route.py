#!/usr/bin/env python
"""Decide the route after the 25.6 formal refinement benchmark."""

from __future__ import annotations

from typing import Any

from build_surface_forward_refinement_target_set import ROOT, as_bool, as_float, read_csv, write_csv
from evaluate_surface_forward_refinement_formal_acceptance_gates import MATRIX as GATE_MATRIX
from run_surface_forward_refinement_formal_benchmark import METRICS as FORMAL_METRICS
from summarize_surface_forward_refinement_candidate import COMPARISON


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_formal_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/surface_forward_refinement_formal_decision_matrix.csv"

FIELDS = [
    "option",
    "selected",
    "decision",
    "evidence",
    "blocked_by",
]


def target_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["target_role"] == "refinement_target"]


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [as_float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return sum(vals) / len(vals) if vals else float("nan")


def decide(gates: list[dict[str, str]], metrics: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    all_gates_pass = bool(gates) and all(as_bool(row["pass"]) for row in gates)
    current_baseline_unchanged = any(row["gate_id"] == "F4_current_baseline_unchanged" and as_bool(row["pass"]) for row in gates)
    multi_pit_ok = any(row["gate_id"] == "F1_multi_pit_success_credit" and as_bool(row["pass"]) for row in gates)
    targets = target_rows(metrics)
    target_count = len(targets)
    profile_improved = mean(targets, "refined_profile_depth_rmse_m") < mean(targets, "baseline_profile_depth_rmse_m")
    dice_improved = mean(targets, "refined_projected_mask_Dice") > mean(targets, "baseline_projected_mask_Dice")
    residual_improved = mean(targets, "feature_residual_mse_after") < mean(targets, "feature_residual_mse_before")
    candidate_formed = all_gates_pass and current_baseline_unchanged

    if candidate_formed:
        selected = "A. export surface forward-refinement inference artifact / runner"
    elif not all_gates_pass:
        selected = "E. stop refinement and report only"
    else:
        selected = "D. profile-basis decoder plan"

    options = [
        {
            "option": "A. export surface forward-refinement inference artifact / runner",
            "selected": selected.startswith("A."),
            "decision": "selected" if selected.startswith("A.") else "not_selected",
            "evidence": (
                f"formal_gates_pass={all_gates_pass}; CURRENT_BASELINE_unchanged={current_baseline_unchanged}; "
                f"target_count={target_count}; profile_improved={profile_improved}; dice_improved={dice_improved}; "
                f"forward_residual_improved={residual_improved}"
            ),
            "blocked_by": "" if selected.startswith("A.") else "formal gates did not establish an exportable no-baseline-transition candidate",
        },
        {
            "option": "B. build component-set branch for multi-pit",
            "selected": False,
            "decision": "future_branch",
            "evidence": f"multi_pit_accounting_ok={multi_pit_ok}; multi-pit remains excluded negative control",
            "blocked_by": "does not block A because target failures are RBC-representable model-fail rows",
        },
        {
            "option": "C. train neural forward surrogate",
            "selected": False,
            "decision": "defer",
            "evidence": "fixed F0/R1 candidate passed formal gates without neural surrogate training",
            "blocked_by": "main neural training is outside 25.6 and not needed before runner export",
        },
        {
            "option": "D. profile-basis decoder plan",
            "selected": selected.startswith("D."),
            "decision": "selected" if selected.startswith("D.") else "not_selected",
            "evidence": f"profile_improved={profile_improved}; dice_improved={dice_improved}",
            "blocked_by": "" if selected.startswith("D.") else "25.6 candidate addresses the intended RBC-representable subset",
        },
        {
            "option": "E. stop refinement and report only",
            "selected": selected.startswith("E."),
            "decision": "selected" if selected.startswith("E.") else "not_selected",
            "evidence": f"formal_gates_pass={all_gates_pass}",
            "blocked_by": "" if selected.startswith("E.") else "formal benchmark supports a next export/runner step",
        },
        {
            "option": "F. baseline transition request",
            "selected": False,
            "decision": "forbidden_this_stage",
            "evidence": "CURRENT_BASELINE.md remains unchanged and user did not authorize a baseline transition",
            "blocked_by": "requires explicit user confirmation and separate baseline-transition review",
        },
    ]
    context = {
        "selected": selected,
        "all_gates_pass": all_gates_pass,
        "current_baseline_unchanged": current_baseline_unchanged,
        "multi_pit_ok": multi_pit_ok,
        "target_count": target_count,
        "profile_improved": profile_improved,
        "dice_improved": dice_improved,
        "residual_improved": residual_improved,
        "target_baseline_rmse": mean(targets, "baseline_profile_depth_rmse_m"),
        "target_refined_rmse": mean(targets, "refined_profile_depth_rmse_m"),
        "target_oracle_rmse": mean(targets, "oracle_profile_depth_rmse_m"),
        "target_baseline_dice": mean(targets, "baseline_projected_mask_Dice"),
        "target_refined_dice": mean(targets, "refined_projected_mask_Dice"),
        "target_residual_before": mean(targets, "feature_residual_mse_before"),
        "target_residual_after": mean(targets, "feature_residual_mse_after"),
    }
    return selected, options, context


def write_summary(context: dict[str, Any]) -> None:
    lines = [
        "25.6 surface forward-refinement formal route decision",
        "",
        f"decision: {context['selected']}",
        f"formal_gates_pass: {context['all_gates_pass']}",
        f"CURRENT_BASELINE_unchanged: {context['current_baseline_unchanged']}",
        f"multi_pit_accounting_ok: {context['multi_pit_ok']}",
        f"target_subset_count: {context['target_count']}",
        f"profile_improved: {context['profile_improved']}",
        f"dice_improved: {context['dice_improved']}",
        f"forward_residual_improved: {context['residual_improved']}",
        "",
        f"target_baseline_profile_rmse_mean_m: {context['target_baseline_rmse']:.12g}",
        f"target_refined_profile_rmse_mean_m: {context['target_refined_rmse']:.12g}",
        f"target_oracle_profile_rmse_mean_m: {context['target_oracle_rmse']:.12g}",
        f"target_baseline_Dice_mean: {context['target_baseline_dice']:.12g}",
        f"target_refined_Dice_mean: {context['target_refined_dice']:.12g}",
        f"target_forward_residual_before_mean: {context['target_residual_before']:.12g}",
        f"target_forward_residual_after_mean: {context['target_residual_after']:.12g}",
        "",
        "baseline_transition_allowed: false",
        "component_set_branch_status: future branch for multi-pit; does not block runner export.",
        f"decision_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in [GATE_MATRIX, FORMAL_METRICS, COMPARISON]:
        if not path.exists():
            raise FileNotFoundError(path)
    gates = read_csv(GATE_MATRIX)
    metrics = read_csv(FORMAL_METRICS)
    _selected, rows, context = decide(gates, metrics)
    write_csv(MATRIX, rows, FIELDS)
    write_summary(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
