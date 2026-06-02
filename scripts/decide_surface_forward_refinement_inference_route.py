#!/usr/bin/env python
"""Decide the route after the 25.7 inference runner export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from build_surface_forward_refinement_target_set import ROOT, as_bool, as_float, read_csv, write_csv
from run_surface_forward_refinement_inference import DEFAULT_MANIFEST, METRICS as RUNNER_METRICS
from verify_surface_forward_refinement_inference_runner import MATRIX as VERIFICATION_MATRIX


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_inference_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/surface_forward_refinement_inference_decision_matrix.csv"

FIELDS = ["option", "selected", "decision", "evidence", "blocked_by"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def check_pass(checks: list[dict[str, str]], check_name: str) -> bool:
    return any(row["check_name"] == check_name and as_bool(row["pass"]) for row in checks)


def decide(checks: list[dict[str, str]], metrics: list[dict[str, str]], manifest: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    all_checks_pass = bool(checks) and all(as_bool(row["pass"]) for row in checks)
    reproduces_25_6 = check_pass(checks, "runner_reproduces_25_6_per_sample")
    target_improves = check_pass(checks, "target_profile_rmse_improves") and check_pass(checks, "target_iou_dice_improve")
    rbc_control_ok = check_pass(checks, "rbc_like_control_not_degraded")
    multi_pit_ok = check_pass(checks, "multi_pit_not_success_credit")
    baseline_unchanged = check_pass(checks, "current_baseline_unchanged")
    artifact_lock_complete = (
        DEFAULT_MANIFEST.exists()
        and bool(manifest.get("artifact_sha256"))
        and manifest.get("artifact_committed") is False
        and manifest.get("artifact_paths_are_ignored") is True
    )
    runner_usable = all_checks_pass and reproduces_25_6 and target_improves and rbc_control_ok and multi_pit_ok and baseline_unchanged
    companion_ready = runner_usable and artifact_lock_complete
    selected = "A. surface refinement visualization/report package" if companion_ready else "E. stop and report"

    targets = [row for row in metrics if row["target_role"] == "refinement_target"]
    options = [
        {
            "option": "A. surface refinement visualization/report package",
            "selected": selected.startswith("A."),
            "decision": "selected" if selected.startswith("A.") else "not_selected",
            "evidence": (
                f"runner_usable={runner_usable}; reproduces_25_6={reproduces_25_6}; "
                f"artifact_lock_complete={artifact_lock_complete}; companion_ready={companion_ready}; "
                f"target_rmse={mean(targets, 'baseline_profile_depth_rmse_m'):.12g}->{mean(targets, 'refined_profile_depth_rmse_m'):.12g}"
            ),
            "blocked_by": "" if selected.startswith("A.") else "runner verification or artifact lock failed",
        },
        {
            "option": "B. component-set branch for multi-pit",
            "selected": False,
            "decision": "future_branch",
            "evidence": f"multi_pit_accounting_ok={multi_pit_ok}; multi-pit rows are not_suitable_for_rbc_refinement",
            "blocked_by": "not the immediate next step after successful runner export; remains required later for multi-pit",
        },
        {
            "option": "C. real-sample metadata alignment",
            "selected": False,
            "decision": "defer",
            "evidence": "unknown real samples can only claim refinement_applied until metadata/oracle/human representability confirmation exists",
            "blocked_by": "visualization/report package should first lock how the companion output is presented",
        },
        {
            "option": "D. baseline transition request",
            "selected": False,
            "decision": "forbidden_this_stage",
            "evidence": f"CURRENT_BASELINE_unchanged={baseline_unchanged}; forbidden_use={manifest.get('forbidden_use')}",
            "blocked_by": "requires separate explicit user request and formal baseline-transition gate",
        },
        {
            "option": "E. stop and report",
            "selected": selected.startswith("E."),
            "decision": "selected" if selected.startswith("E.") else "not_selected",
            "evidence": f"all_checks_pass={all_checks_pass}; companion_ready={companion_ready}",
            "blocked_by": "" if selected.startswith("E.") else "runner export is verified and can move to reporting/visualization",
        },
    ]
    context = {
        "selected": selected,
        "runner_usable": runner_usable,
        "reproduces_25_6": reproduces_25_6,
        "target_improves": target_improves,
        "rbc_control_ok": rbc_control_ok,
        "multi_pit_ok": multi_pit_ok,
        "baseline_unchanged": baseline_unchanged,
        "artifact_lock_complete": artifact_lock_complete,
        "companion_ready": companion_ready,
        "target_count": len(targets),
        "target_baseline_rmse": mean(targets, "baseline_profile_depth_rmse_m"),
        "target_refined_rmse": mean(targets, "refined_profile_depth_rmse_m"),
        "target_baseline_dice": mean(targets, "baseline_projected_mask_Dice"),
        "target_refined_dice": mean(targets, "refined_projected_mask_Dice"),
        "target_residual_before": mean(targets, "feature_residual_mse_before"),
        "target_residual_after": mean(targets, "feature_residual_mse_after"),
        "artifact_id": manifest.get("artifact_id"),
    }
    return selected, options, context


def write_summary(context: dict[str, Any]) -> None:
    lines = [
        "25.7 surface forward-refinement inference route decision",
        "",
        f"decision: {context['selected']}",
        f"runner_usable: {context['runner_usable']}",
        f"reproduces_25_6: {context['reproduces_25_6']}",
        f"surface_refinement_companion_ready: {context['companion_ready']}",
        f"formal_artifact_lock_complete: {context['artifact_lock_complete']}",
        f"CURRENT_BASELINE_unchanged: {context['baseline_unchanged']}",
        f"multi_pit_accounting_ok: {context['multi_pit_ok']}",
        "baseline_transition_allowed: false",
        "",
        f"artifact_id: {context['artifact_id']}",
        f"target_subset_count: {context['target_count']}",
        f"target_baseline_profile_rmse_mean_m: {context['target_baseline_rmse']:.12g}",
        f"target_refined_profile_rmse_mean_m: {context['target_refined_rmse']:.12g}",
        f"target_baseline_Dice_mean: {context['target_baseline_dice']:.12g}",
        f"target_refined_Dice_mean: {context['target_refined_dice']:.12g}",
        f"target_forward_residual_before_mean: {context['target_residual_before']:.12g}",
        f"target_forward_residual_after_mean: {context['target_residual_after']:.12g}",
        "",
        "component_set_branch_status: future route B for multi-pit; not success credit inside RBC refinement.",
        "real_sample_boundary: unknown real samples may report refinement_applied, not representable success, without ground truth or human confirmation.",
        f"decision_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in [VERIFICATION_MATRIX, RUNNER_METRICS, DEFAULT_MANIFEST]:
        if not path.exists():
            raise FileNotFoundError(path)
    checks = read_csv(VERIFICATION_MATRIX)
    metrics = read_csv(RUNNER_METRICS)
    manifest = read_json(DEFAULT_MANIFEST)
    _selected, rows, context = decide(checks, metrics, manifest)
    write_csv(MATRIX, rows, FIELDS)
    write_summary(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
