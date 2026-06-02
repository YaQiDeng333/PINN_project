#!/usr/bin/env python
"""Decide the 20.99 internal / buried defect feasibility route.

This script reads the schema and smoke-pack plan outputs, then writes a route
decision summary and matrix. It does not run COMSOL, train models, generate
data, write NPZ files, or update the current baseline.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "INTERNAL_DEFECT_SCHEMA.md"
PLAN_CSV = ROOT / "results/metrics/internal_defect_comsol_smoke_pack_plan.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_feasibility_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_feasibility_decision_matrix.csv"

MATRIX_FIELDS = ["question", "answer", "decision", "evidence", "gate_status"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide the 20.99 internal defect feasibility route.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    return parser.parse_args()


def git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assert_schema_fields(text: str) -> None:
    required_terms = [
        "burial_depth_m",
        "depth_to_surface_m",
        "defect_center_xyz_m",
        "shape_type",
        "ground_truth_method",
        "Bz-only",
        "CURRENT_BASELINE.md",
    ]
    missing = [term for term in required_terms if term not in text]
    if missing:
        raise RuntimeError("schema missing required terms: " + ", ".join(missing))


def main() -> int:
    args = parse_args()
    if git_root() != ROOT.resolve():
        raise SystemExit(f"wrong repository root: {ROOT}")
    schema_text = SCHEMA.read_text(encoding="utf-8")
    assert_schema_fields(schema_text)
    rows = read_csv(args.plan_csv)
    if not rows:
        raise RuntimeError(f"empty plan CSV: {args.plan_csv}")

    shape_types = sorted({row["shape_type"] for row in rows})
    burial_levels = sorted({row["burial_depth_level"] for row in rows})
    row_count = len(rows)

    decisions = [
        {
            "question": "can_migrate_from_surface_rbc_baseline",
            "answer": "no",
            "decision": "Do not migrate directly from the current surface RBC baseline.",
            "evidence": "surface RBC has no burial_depth_m target and represents surface profile/depth, not buried cavity geometry",
            "gate_status": "blocked",
        },
        {
            "question": "need_independent_comsol_generator",
            "answer": "yes",
            "decision": "Use an independent internal-cavity COMSOL generator in a later stage.",
            "evidence": "internal cavity geometry, no-defect reference semantics, and labels differ from surface RBC",
            "gate_status": "required",
        },
        {
            "question": "first_model_route_if_smoke_passes",
            "answer": "shape_type_conditioned_sizing",
            "decision": "If a later smoke pack passes, first evaluate shape_type + L/W/D + burial_depth + center_xyz, not free-form occupancy.",
            "evidence": f"planned shape_types={shape_types}; planned_rows={row_count}",
            "gate_status": "future_only_no_training_this_stage",
        },
        {
            "question": "need_Bx_By_Bz",
            "answer": "yes",
            "decision": "Keep tri-axis Bx/By/Bz as the mainline input requirement.",
            "evidence": "20.98 dry-run blockers and current true 3D MFL tooling both require tri-axis input for mainline claims",
            "gate_status": "required",
        },
        {
            "question": "Bz_only_feasible",
            "answer": "limited_diagnostic_only",
            "decision": "Bz-only may be recorded as a low-capability diagnostic branch, not as the internal mainline.",
            "evidence": "Bz-only loses vector and lateral response information and was a blocker for the current route",
            "gate_status": "degraded_branch_only",
        },
        {
            "question": "enter_internal_COMSOL_smoke_pack_next",
            "answer": "yes_after_metadata_confirmation",
            "decision": "Unique next step: A. execute internal COMSOL smoke pack after confirming required metadata and labels.",
            "evidence": f"schema exists; plan rows={row_count}; burial_levels={burial_levels}; this stage did not run COMSOL",
            "gate_status": "recommended_next",
        },
    ]

    write_csv(args.matrix, decisions)
    lines = [
        "20.99 internal / buried defect feasibility route decision",
        "",
        "surface_rbc_direct_use: false",
        "need_independent_schema: true",
        "need_independent_comsol_generator: true",
        "first_model_route_if_smoke_passes: shape_type_conditioned_sizing",
        "recommended_output_representation: shape_type + L/W/D + burial_depth + center_xyz",
        f"recommended_smoke_pack_size: {row_count} samples",
        "smoke_pack_size_range: 6-12 samples",
        "need_Bx_By_Bz: true",
        "Bz_only_status: limited diagnostic branch only",
        "training_allowed_this_stage: false",
        "COMSOL_allowed_this_stage: false",
        "data_npz_generation_allowed_this_stage: false",
        "CURRENT_BASELINE_update: false",
        "baseline_transition_allowed: false",
        "",
        "rationale:",
        "- Internal defects require burial_depth_m / depth_to_surface_m, which the surface RBC six-parameter output does not encode.",
        "- Direct surface RBC inference would confuse burial-depth changes with surface profile, footprint, or curvature changes.",
        "- Tri-axis Bx/By/Bz remains the mainline because Bz-only is underdetermined for true 3D internal geometry claims.",
        "- The first later smoke pack should prove geometry/reference/label feasibility before any training gate.",
        "",
        "next_unique_step: A. execute internal COMSOL smoke pack after metadata confirmation",
        f"decision_matrix: {args.matrix}",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
