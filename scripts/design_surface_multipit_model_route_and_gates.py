#!/usr/bin/env python
"""Design 25.9 surface multi-pit model route and acceptance gates."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATION_MATRIX = ROOT / "results/metrics/surface_multipit_component_set_representation_matrix.csv"
TOPUP_PLAN = ROOT / "results/metrics/surface_multipit_dataset_topup_plan.csv"
COMSOL_FEASIBILITY = ROOT / "results/metrics/surface_multipit_comsol_generation_feasibility.csv"

SUMMARY = ROOT / "results/summaries/surface_multipit_model_route_summary.txt"
GATES = ROOT / "results/metrics/surface_multipit_acceptance_gate_matrix.csv"

FIELDS = [
    "gate_id",
    "gate_name",
    "applies_to",
    "threshold_or_rule",
    "success_credit_policy",
    "failure_action",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gate_rows() -> list[dict[str, Any]]:
    return [
        {
            "gate_id": "M1",
            "gate_name": "component_recall",
            "applies_to": "assembled multi-pit test split",
            "threshold_or_rule": "mean recall >= 0.85 and close/touching subset recall >= 0.75",
            "success_credit_policy": "component-set branch success only; no six-param RBC credit",
            "failure_action": "inspect missed secondary pits and close/touching strata before more training",
        },
        {
            "gate_id": "M2",
            "gate_name": "missed_merged_extra_rates",
            "applies_to": "all multi-pit topology strata",
            "threshold_or_rule": "missed <= 0.15; merged <= 0.20; extra fragments reported and must not dominate any split",
            "success_credit_policy": "multi-pit metric only",
            "failure_action": "route to C2/C3 or revise top-up coverage",
        },
        {
            "gate_id": "M3",
            "gate_name": "matched_component_geometry_error",
            "applies_to": "Hungarian-matched active components",
            "threshold_or_rule": "center error mean <= 2.0 mm; L/W/D relative MAE <= 25%",
            "success_credit_policy": "component geometry success only",
            "failure_action": "audit scale/coordinate labels and per-component rotation",
        },
        {
            "gate_id": "M4",
            "gate_name": "union_mask_depth_quality",
            "applies_to": "union projected_mask_2d and depth_grid_m",
            "threshold_or_rule": "projected mask Dice >= 0.70 or IoU >= 0.55; depth-grid RMSE better than single-RBC comparator",
            "success_credit_policy": "component-set branch success only",
            "failure_action": "add C2 raster auxiliary loss or C4 fallback comparator",
        },
        {
            "gate_id": "M5",
            "gate_name": "single_component_control_noncollapse",
            "applies_to": "RBC-like and single-component non-RBC controls from shape-extension pilot",
            "threshold_or_rule": "component-set work must not degrade established single-component control metrics",
            "success_credit_policy": "control gate only",
            "failure_action": "do not promote component branch; isolate branch-specific head/training",
        },
        {
            "gate_id": "M6",
            "gate_name": "negative_control_boundary",
            "applies_to": "20.85 baseline and 25.5/25.8 forward-refinement runner",
            "threshold_or_rule": "single-RBC/refinement comparators cannot claim multi-pit success even if union mask improves",
            "success_credit_policy": "no RBC success credit for multi-pit",
            "failure_action": "fix report accounting before any route transition",
        },
        {
            "gate_id": "M7",
            "gate_name": "baseline_transition_block",
            "applies_to": "CURRENT_BASELINE.md and roadmap docs",
            "threshold_or_rule": "no CURRENT_BASELINE.md update without later formal benchmark plus explicit user confirmation",
            "success_credit_policy": "none in 25.9",
            "failure_action": "revert baseline doc change and rerun review",
        },
    ]


def write_summary() -> None:
    lines = [
        "25.9 surface multi-pit model route and acceptance gates",
        "",
        "route_1: component-set training gate on future top-up assembled dataset; first trainable route after data generation, not in 25.9.",
        "route_2: current 20.85 baseline plus surface forward-refinement runner remains negative/control comparator only.",
        "route_3: depth-grid fallback if component labels remain unstable after top-up audit.",
        "",
        "recommended_first_model_route_after_topup: C1 fixed-K component-set with K=3 and Hungarian matching.",
        "training_in_this_stage: false",
        "CURRENT_BASELINE_transition_allowed: false",
        "multi_pit_rbc_success_credit_allowed: false",
        "",
        "required_metrics: component recall, missed rate, merged rate, extra fragment rate, matched center error, L/W/D relative MAE, projected mask Dice/IoU, depth-grid RMSE.",
        f"acceptance_gate_matrix: {GATES}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in [REPRESENTATION_MATRIX, TOPUP_PLAN, COMSOL_FEASIBILITY]:
        if not path.exists():
            raise FileNotFoundError(f"missing prerequisite for model route design: {path}")
    write_csv(GATES, gate_rows(), FIELDS)
    write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
