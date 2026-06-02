#!/usr/bin/env python
"""Decide the route after the 25.8 report / visualization package."""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np

from build_surface_forward_refinement_target_set import ROOT, as_bool, as_float, read_csv, write_csv
from run_surface_forward_refinement_inference import METRICS as RUNNER_METRICS
from verify_surface_forward_refinement_inference_runner import MATRIX as RUNNER_VERIFICATION

from build_surface_forward_refinement_report_package import REPORT_METRICS
from audit_surface_forward_refinement_improvement_cases import DEGRADED_CASES, GROUP_AUDIT, IMPROVEMENT_CASES
from export_surface_forward_refinement_gallery import INDEX as GALLERY_INDEX


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_report_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/surface_forward_refinement_report_decision_matrix.csv"

FIELDS = ["option", "selected", "decision", "evidence", "blocked_by"]


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


def check_pass(rows: list[dict[str, str]], name: str) -> bool:
    return any(row.get("check_name") == name and as_bool(row.get("pass")) for row in rows)


def decide(metrics: list[dict[str, str]], verification: list[dict[str, str]], gallery: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    targets = [row for row in metrics if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in metrics if row["shape_type"] == "rbc_like_smooth_pit"]
    negative = [row for row in metrics if row["target_role"] == "excluded_negative_control"]
    multipit = [row for row in metrics if row["shape_type"] == "multi_pit_two_component_surface_defect"]
    verification_ok = bool(verification) and all(as_bool(row["pass"]) for row in verification)
    runner_stable = (
        verification_ok
        and check_pass(verification, "runner_reproduces_25_6_per_sample")
        and mean(targets, "refined_profile_depth_rmse_m") < mean(targets, "baseline_profile_depth_rmse_m")
        and mean(targets, "refined_projected_mask_Dice") > mean(targets, "baseline_projected_mask_Dice")
        and mean(rbc_like, "refined_profile_depth_rmse_m") <= mean(rbc_like, "baseline_profile_depth_rmse_m")
        and mean(rbc_like, "refined_projected_mask_Dice") >= mean(rbc_like, "baseline_projected_mask_Dice")
    )
    multipit_remaining = bool(multipit) and all(row.get("eligibility_status") == "not_suitable_for_rbc_refinement" for row in multipit)
    multipit_no_success_credit = bool(negative) and all(not as_bool(row["include_in_success_gate"]) for row in negative)
    current_baseline_diff = git_value(["diff", "--name-only", "--", "CURRENT_BASELINE.md"])
    baseline_unchanged = not bool(current_baseline_diff)
    report_complete = REPORT_METRICS.exists() and IMPROVEMENT_CASES.exists() and GROUP_AUDIT.exists() and DEGRADED_CASES.exists() and GALLERY_INDEX.exists() and bool(gallery)
    selected = "A. component-set branch for multi-pit" if runner_stable and multipit_remaining and multipit_no_success_credit and baseline_unchanged and report_complete else "F. stop surface refinement"

    options = [
        {
            "option": "A. component-set branch for multi-pit",
            "selected": selected.startswith("A."),
            "decision": "selected" if selected.startswith("A.") else "not_selected",
            "evidence": (
                f"runner_stable={runner_stable}; multipit_remaining={multipit_remaining}; "
                f"multipit_no_success_credit={multipit_no_success_credit}; report_complete={report_complete}; "
                f"multi_pit_rows={len(multipit)}"
            ),
            "blocked_by": "" if selected.startswith("A.") else "runner/report stability or multi-pit accounting was not established",
        },
        {
            "option": "B. formal surface refinement artifact lock",
            "selected": False,
            "decision": "already_satisfied_by_25_7",
            "evidence": "25.7 artifact manifest records sha256/protocol and verification reproduces 25.6",
            "blocked_by": "not the immediate next route after this report package",
        },
        {
            "option": "C. real-sample metadata alignment",
            "selected": False,
            "decision": "defer",
            "evidence": "unknown real samples still need metadata/oracle/human representability confirmation before success claims",
            "blocked_by": "multi-pit representation branch is the clearer remaining surface modeling gap",
        },
        {
            "option": "D. final benchmark/report package",
            "selected": False,
            "decision": "current_stage_completed",
            "evidence": f"report_complete={report_complete}; gallery_index_rows={len(gallery)}",
            "blocked_by": "25.8 already prepared the report package; next work should address the residual branch",
        },
        {
            "option": "E. baseline transition request",
            "selected": False,
            "decision": "forbidden_this_stage",
            "evidence": f"CURRENT_BASELINE_unchanged={baseline_unchanged}; runner is companion only",
            "blocked_by": "requires separate explicit user request and baseline-transition review",
        },
        {
            "option": "F. stop surface refinement",
            "selected": selected.startswith("F."),
            "decision": "selected" if selected.startswith("F.") else "not_selected",
            "evidence": f"runner_stable={runner_stable}; report_complete={report_complete}",
            "blocked_by": "" if selected.startswith("F.") else "report confirms a concrete remaining component-set branch",
        },
    ]
    context = {
        "selected": selected,
        "runner_stable": runner_stable,
        "report_complete": report_complete,
        "multipit_remaining": multipit_remaining,
        "multipit_no_success_credit": multipit_no_success_credit,
        "baseline_unchanged": baseline_unchanged,
        "target_count": len(targets),
        "multi_pit_count": len(multipit),
        "gallery_index_rows": len(gallery),
        "target_baseline_rmse": mean(targets, "baseline_profile_depth_rmse_m"),
        "target_refined_rmse": mean(targets, "refined_profile_depth_rmse_m"),
        "target_baseline_dice": mean(targets, "baseline_projected_mask_Dice"),
        "target_refined_dice": mean(targets, "refined_projected_mask_Dice"),
        "rbc_like_baseline_rmse": mean(rbc_like, "baseline_profile_depth_rmse_m"),
        "rbc_like_refined_rmse": mean(rbc_like, "refined_profile_depth_rmse_m"),
    }
    return selected, options, context


def write_summary(context: dict[str, Any]) -> None:
    lines = [
        "25.8 surface forward-refinement report route decision",
        "",
        f"decision: {context['selected']}",
        f"runner_stable: {context['runner_stable']}",
        f"report_complete: {context['report_complete']}",
        f"CURRENT_BASELINE_unchanged: {context['baseline_unchanged']}",
        f"multi_pit_remaining_representation_gap: {context['multipit_remaining']}",
        f"multi_pit_no_rbc_success_credit: {context['multipit_no_success_credit']}",
        "baseline_transition_allowed: false",
        "",
        f"target_subset_count: {context['target_count']}",
        f"multi_pit_count: {context['multi_pit_count']}",
        f"gallery_index_rows: {context['gallery_index_rows']}",
        f"target_profile_rmse_m: {context['target_baseline_rmse']:.12g} -> {context['target_refined_rmse']:.12g}",
        f"target_Dice: {context['target_baseline_dice']:.12g} -> {context['target_refined_dice']:.12g}",
        f"rbc_like_control_rmse_m: {context['rbc_like_baseline_rmse']:.12g} -> {context['rbc_like_refined_rmse']:.12g}",
        "",
        "interpretation: the companion runner is stable for RBC-representable surface model failures; multi-pit remains a representation problem requiring component-set output.",
        f"decision_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in [RUNNER_METRICS, RUNNER_VERIFICATION, REPORT_METRICS, IMPROVEMENT_CASES, GROUP_AUDIT, DEGRADED_CASES, GALLERY_INDEX]:
        if not path.exists():
            raise FileNotFoundError(path)
    metrics = read_csv(RUNNER_METRICS)
    verification = read_csv(RUNNER_VERIFICATION)
    gallery = read_csv(GALLERY_INDEX)
    _selected, rows, context = decide(metrics, verification, gallery)
    write_csv(MATRIX, rows, FIELDS)
    write_summary(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
