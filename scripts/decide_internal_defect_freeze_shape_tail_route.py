#!/usr/bin/env python
"""22.5 route decision for freeze-shape internal tail-regression model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from internal_defect_hardcase_utils import safe_float
from load_internal_defect_pilot_dataset import ROOT, write_csv


SUMMARY = ROOT / "results/summaries/internal_defect_freeze_shape_tail_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_freeze_shape_tail_decision_matrix.csv"
SEEDS = ROOT / "results/metrics/internal_defect_freeze_shape_tail_seed_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/internal_defect_freeze_shape_tail_vs_reference.csv"

FIELDS = ["decision_item", "status", "evidence", "threshold_or_rule", "decision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide 22.5 freeze-shape route.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--seed-summary", type=Path, default=SEEDS)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def selected_row(path: Path) -> dict[str, str]:
    for row in read_csv(path):
        if row.get("selected_seed") == "True":
            return row
    return {}


def lookup(path: Path) -> dict[str, dict[str, str]]:
    return {row["metric"]: row for row in read_csv(path)}


def passed(row: dict[str, str] | None) -> bool:
    return str((row or {}).get("passes_gate", "")).lower() == "true"


def main() -> int:
    args = parse_args()
    selected = selected_row(args.seed_summary)
    vs = lookup(args.vs_reference)
    if not selected:
        rows = [
            {
                "decision_item": "freeze_shape_candidate_available",
                "status": False,
                "evidence": "no selected_seed row",
                "threshold_or_rule": "candidate screen must select a formal freeze-shape model",
                "decision": "blocked",
            }
        ]
        write_csv(args.matrix, rows, FIELDS)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text("22.5 freeze-shape route decision\nstatus: blocked\nreason: no selected formal candidate\ncurrent_baseline_update: false\n", encoding="utf-8")
        return 0

    shape_preserved = passed(vs.get("shape_macro_f1"))
    center_tail_down = passed(vs.get("center_p95_mm")) and passed(vs.get("center_max_mm"))
    burial_tail_ok = passed(vs.get("burial_p95_mm")) and passed(vs.get("burial_max_mm"))
    catastrophic_down = passed(vs.get("catastrophic_failure_count"))
    geometry_down = passed(vs.get("geometry_branch_failure_count"))
    total_ok = passed(vs.get("total_normalized_mae"))
    stable = shape_preserved and center_tail_down and burial_tail_ok and catastrophic_down and geometry_down and safe_float(selected.get("test_catastrophic_failure_rate")) <= 0.05 and safe_float(selected.get("test_geometry_branch_failure_count")) == 0
    benchmark_candidate = shape_preserved and total_ok and (center_tail_down or catastrophic_down or geometry_down)
    next_decision = "internal inference artifact/gateway" if stable else ("tail-specific refinement plus uncertainty/output gate" if shape_preserved else "revise labels/output or pause branch")

    rows = [
        {
            "decision_item": "shape_branch_preserved",
            "status": shape_preserved,
            "evidence": f"shape_acc/F1={selected.get('test_shape_accuracy')} / {selected.get('test_shape_macro_f1')}",
            "threshold_or_rule": "shape F1 close to B2 and not H2-style collapse",
            "decision": "preserved" if shape_preserved else "not_preserved",
        },
        {
            "decision_item": "center_tail_reduced",
            "status": center_tail_down,
            "evidence": f"center_p95/max={selected.get('test_center_p95_mm')} / {selected.get('test_center_max_mm')}",
            "threshold_or_rule": "center p95 and max below B2 reference; no H2-style regression",
            "decision": "reduced" if center_tail_down else "not_enough",
        },
        {
            "decision_item": "burial_tail_not_regressed",
            "status": burial_tail_ok,
            "evidence": f"burial_p95/max={selected.get('test_burial_p95_mm')} / {selected.get('test_burial_max_mm')}",
            "threshold_or_rule": "burial p95 and max should not regress vs B2/H2",
            "decision": "ok" if burial_tail_ok else "regressed",
        },
        {
            "decision_item": "catastrophic_failure_reduced",
            "status": catastrophic_down,
            "evidence": f"catastrophic={selected.get('test_catastrophic_failure_count')} / rate={selected.get('test_catastrophic_failure_rate')}",
            "threshold_or_rule": "below both B2 and H2; stable target <=5%",
            "decision": "reduced" if catastrophic_down else "not_enough",
        },
        {
            "decision_item": "geometry_branch_reduced",
            "status": geometry_down,
            "evidence": f"geometry_branch={selected.get('test_geometry_branch_failure_count')} / rate={selected.get('test_geometry_branch_failure_rate')}",
            "threshold_or_rule": "below both B2 and H2; stable target 0",
            "decision": "reduced" if geometry_down else "not_enough",
        },
        {
            "decision_item": "stable_internal_inference_candidate",
            "status": stable,
            "evidence": f"stable={stable}; shape={shape_preserved}; center={center_tail_down}; burial={burial_tail_ok}; cat={catastrophic_down}; geometry={geometry_down}",
            "threshold_or_rule": "shape preserved, center/burial tails improved, catastrophic <=5%, geometry_branch=0",
            "decision": "stable_candidate" if stable else "not_stable",
        },
        {
            "decision_item": "internal_benchmark_candidate",
            "status": benchmark_candidate,
            "evidence": f"benchmark_candidate={benchmark_candidate}; total_ok={total_ok}",
            "threshold_or_rule": "can remain benchmark candidate if shape/mean are safe but stable tail gate fails",
            "decision": "candidate" if benchmark_candidate else "weak_candidate",
        },
        {
            "decision_item": "next_step",
            "status": True,
            "evidence": "derived from freeze-shape gate",
            "threshold_or_rule": "choose one route only",
            "decision": next_decision,
        },
    ]
    write_csv(args.matrix, rows, FIELDS)
    next_step = rows[-1]["decision"]
    lines = [
        "22.5 freeze-shape internal tail-regression route decision",
        f"selected_candidate: {selected.get('candidate')}",
        f"selected_seed: {selected.get('seed')}",
        f"shape_preserved: {shape_preserved}",
        f"center_tail_reduced: {center_tail_down}",
        f"burial_tail_not_regressed: {burial_tail_ok}",
        f"catastrophic_reduced: {catastrophic_down}",
        f"geometry_branch_reduced: {geometry_down}",
        f"stable_internal_inference_candidate: {stable}",
        f"internal_benchmark_candidate: {benchmark_candidate}",
        "baseline_status: internal branch only; CURRENT_BASELINE unchanged",
        f"unique_next_step: {next_step}",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
