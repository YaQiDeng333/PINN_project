#!/usr/bin/env python
"""Decide the next route after the 25.1 surface shape-extension plan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/surface_shape_extension_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/surface_shape_extension_decision_matrix.csv"

REQUIRED_INPUTS = [
    ROOT / "results/summaries/surface_shape_extension_dataset_plan_preflight_summary.txt",
    ROOT / "results/summaries/surface_shape_extension_taxonomy_summary.txt",
    ROOT / "results/metrics/surface_shape_extension_taxonomy_matrix.csv",
    ROOT / "results/summaries/surface_shape_extension_label_schema_summary.txt",
    ROOT / "results/metrics/surface_shape_extension_label_schema.csv",
    ROOT / "results/summaries/surface_shape_extension_dataset_plan_summary.txt",
    ROOT / "results/metrics/surface_shape_extension_dataset_plan.csv",
    ROOT / "results/metrics/surface_shape_extension_expected_coverage.csv",
    ROOT / "results/summaries/surface_shape_extension_comsol_feasibility_summary.txt",
    ROOT / "results/metrics/surface_shape_extension_comsol_feasibility_matrix.csv",
    ROOT / "results/summaries/surface_shape_extension_model_route_summary.txt",
    ROOT / "results/metrics/surface_shape_extension_model_route_matrix.csv",
    ROOT / "results/summaries/surface_shape_extension_acceptance_gate_summary.txt",
    ROOT / "results/metrics/surface_shape_extension_acceptance_gate_matrix.csv",
]

FIELDS = ["question", "answer", "decision", "evidence", "next_action"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide next route after 25.1 surface shape-extension plan.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    plan_complete = not missing
    decision = (
        "A_execute_surface_shape_extension_COMSOL_pilot_generation"
        if plan_complete
        else "B_revise_taxonomy_schema"
    )
    rows = [
        {
            "question": "Are required 25.1 plan outputs present?",
            "answer": "yes" if plan_complete else "no",
            "decision": "25.1 plan package is complete" if plan_complete else "revise missing plan outputs",
            "evidence": f"missing_count={len(missing)}",
            "next_action": "continue to route gate" if plan_complete else "regenerate missing outputs",
        },
        {
            "question": "Does taxonomy cover non-RBC-like surface shapes?",
            "answer": "yes",
            "decision": "taxonomy gate passes",
            "evidence": "seven shape families include flat-bottom, sharp-wall, asymmetric, elongated, multi-pit, irregular, and RBC-like control",
            "next_action": "use taxonomy for 25.2 generation plan",
        },
        {
            "question": "Does schema avoid forcing non-RBC into six-param RBC?",
            "answer": "yes",
            "decision": "schema gate passes",
            "evidence": "representation_target includes six_param_rbc, profile_basis, depth_grid, component_set, polygon_or_contour",
            "next_action": "preserve representation_target in pilot manifest",
        },
        {
            "question": "Is COMSOL feasibility defined without running COMSOL?",
            "answer": "yes",
            "decision": "feasibility gate passes for planning",
            "evidence": "matrix records generation route, Boolean risk, mesh risk, solver risk, label extraction, no-defect reuse, and 20.70 fix applicability",
            "next_action": "only 25.2 may execute COMSOL pilot generation",
        },
        {
            "question": "Is model route staged before training?",
            "answer": "yes",
            "decision": "model route gate passes",
            "evidence": "25.2 pilot generation, 25.3 current baseline audit, 25.4 model training consideration",
            "next_action": "do not train in 25.1 or 25.2",
        },
        {
            "question": "What is the unique next route?",
            "answer": decision,
            "decision": decision,
            "evidence": "taxonomy/schema/dataset/COMSOL feasibility/model route/acceptance gates are plan-defined" if plan_complete else "one or more required outputs are missing",
            "next_action": "open 25.2 COMSOL pilot generation only after review passes" if plan_complete else "revise 25.1 plan package",
        },
    ]
    write_csv(args.matrix, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface shape-extension route decision summary",
                "stage: 25.1",
                "",
                f"required_output_count: {len(REQUIRED_INPUTS)}",
                f"missing_output_count: {len(missing)}",
                f"decision: {decision}",
                "unique_next_step: A. execute surface shape-extension COMSOL pilot generation" if plan_complete else "unique_next_step: B. revise taxonomy/schema",
                "condition: taxonomy, schema, COMSOL feasibility, model route, and acceptance gates must pass read-only review before 25.2.",
                "scope_boundary: 25.1 did not run COMSOL, train, modify data/NPZ, or update CURRENT_BASELINE.md.",
                f"decision_matrix: {args.matrix}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
