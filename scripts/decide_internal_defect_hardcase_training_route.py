#!/usr/bin/env python
"""22.3 route decision for hard-case augmented internal training gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from internal_defect_hardcase_utils import read_csv, safe_float
from load_internal_defect_pilot_dataset import ROOT, write_csv


SUMMARY = ROOT / "results/summaries/internal_defect_hardcase_training_route_decision_summary.txt"
DECISION = ROOT / "results/metrics/internal_defect_hardcase_training_decision_matrix.csv"
VS_B2 = ROOT / "results/metrics/internal_defect_hardcase_vs_b2_reference.csv"
SEEDS = ROOT / "results/metrics/internal_defect_hardcase_seed_summary.csv"
FAILURE_SUMMARY = ROOT / "results/summaries/internal_defect_hardcase_failure_audit_summary.txt"

FIELDS = ["decision_item", "status", "evidence", "threshold_or_rule", "decision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide 22.3 hard-case route.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--decision", type=Path, default=DECISION)
    parser.add_argument("--vs-b2", type=Path, default=VS_B2)
    parser.add_argument("--seed-summary", type=Path, default=SEEDS)
    parser.add_argument("--failure-summary", type=Path, default=FAILURE_SUMMARY)
    return parser.parse_args()


def selected_seed_row(path: Path) -> dict[str, str]:
    for row in read_csv(path):
        if str(row.get("selected_model", "")).lower() == "true":
            return row
    raise RuntimeError("missing selected seed row")


def vs(path: Path) -> dict[str, dict[str, str]]:
    return {row["metric"]: row for row in read_csv(path)}


def main() -> int:
    args = parse_args()
    seed = selected_seed_row(args.seed_summary)
    compare = vs(args.vs_b2)
    catastrophic_rate = safe_float(seed["test_catastrophic_failure_rate"])
    geometry_count = safe_float(seed["test_geometry_branch_failure_count"])
    center_p95 = safe_float(seed["test_center_p95_mm"])
    center_max = safe_float(seed["test_center_max_mm"])
    burial_p95 = safe_float(seed["test_burial_p95_mm"])
    burial_max = safe_float(seed["test_burial_max_mm"])
    total = safe_float(seed["test_total_normalized_mae"])
    shape_f1 = safe_float(seed["test_shape_macro_f1"])

    hardcase_effective = (
        str(compare["catastrophic_failure_rate"]["passes_gate"]).lower() == "true"
        and str(compare["geometry_branch_failure_count"]["passes_gate"]).lower() == "true"
        and str(compare["center_xyz_error_p95_mm"]["passes_gate"]).lower() == "true"
        and str(compare["burial_depth_error_p95_mm"]["passes_gate"]).lower() == "true"
    )
    stable_candidate = hardcase_effective and catastrophic_rate <= 0.05 and geometry_count == 0 and shape_f1 >= 0.95
    second_topup = not stable_candidate and (catastrophic_rate > 0.05 or geometry_count > 0)
    label_revision = False
    inference_smoke = stable_candidate

    rows: list[dict[str, Any]] = [
        {
            "decision_item": "hard_case_topup_effective",
            "status": hardcase_effective,
            "evidence": f"cat_rate={catastrophic_rate:.6f}, geometry={geometry_count:.0f}, center_p95={center_p95:.3f}, burial_p95={burial_p95:.3f}",
            "threshold_or_rule": "must improve catastrophic, geometry branch, center p95 and burial p95 vs old B2",
            "decision": "effective" if hardcase_effective else "not_enough",
        },
        {
            "decision_item": "stable_internal_inference_candidate",
            "status": stable_candidate,
            "evidence": f"cat_rate={catastrophic_rate:.6f}, geometry={geometry_count:.0f}, shape_f1={shape_f1:.6f}",
            "threshold_or_rule": "catastrophic <=5%, geometry_branch=0, shape_f1>=0.95",
            "decision": "stable_candidate" if stable_candidate else "benchmark_candidate_only",
        },
        {
            "decision_item": "second_hard_case_topup_needed",
            "status": second_topup,
            "evidence": f"cat_rate={catastrophic_rate:.6f}, geometry={geometry_count:.0f}",
            "threshold_or_rule": "needed if tail remains above stable gate",
            "decision": "second_topup" if second_topup else "not_needed_now",
        },
        {
            "decision_item": "revise_output_labels_needed",
            "status": label_revision,
            "evidence": "22.3 uses existing L/W/D + burial + center + shape labels consistently; no label inconsistency detected by this gate.",
            "threshold_or_rule": "revise only if failure evidence points to label/schema mismatch",
            "decision": "not_needed",
        },
        {
            "decision_item": "internal_inference_smoke_or_real_alignment",
            "status": inference_smoke,
            "evidence": f"stable_candidate={stable_candidate}",
            "threshold_or_rule": "only after stable candidate gate passes",
            "decision": "enter_internal_inference_smoke" if inference_smoke else "defer_real_sample_alignment",
        },
    ]
    write_csv(args.decision, rows, FIELDS)
    recommendation = "internal inference smoke / real sample metadata alignment" if inference_smoke else "second hard-case top-up or tail-specific refinement before real internal smoke"
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "22.3 hard-case augmented internal training route decision",
                f"selected_model: {seed['model']}",
                f"selected_seed: {seed['seed']}",
                f"test_total_normalized_mae: {total:.6f}",
                f"test_shape_macro_f1: {shape_f1:.6f}",
                f"test_catastrophic_failure_count_rate: {seed['test_catastrophic_failure_count']} / {catastrophic_rate:.6f}",
                f"test_geometry_branch_failure_count_rate: {seed['test_geometry_branch_failure_count']} / {seed['test_geometry_branch_failure_rate']}",
                f"test_center_p95_max_mm: {center_p95:.3f} / {center_max:.3f}",
                f"test_burial_p95_max_mm: {burial_p95:.3f} / {burial_max:.3f}",
                f"hard_case_topup_effective: {hardcase_effective}",
                f"stable_internal_inference_candidate: {stable_candidate}",
                "baseline_status: internal branch only; CURRENT_BASELINE unchanged",
                f"unique_next_step: {recommendation}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
