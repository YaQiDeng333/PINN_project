#!/usr/bin/env python
"""Design acceptance gates for the 25.1 surface shape-extension plan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/surface_shape_extension_acceptance_gate_summary.txt"
MATRIX = ROOT / "results/metrics/surface_shape_extension_acceptance_gate_matrix.csv"

FIELDS = [
    "gate_group",
    "gate_id",
    "criterion",
    "required_evidence",
    "pass_rule",
    "blocks_next_route",
    "notes",
]

ROWS: list[dict[str, Any]] = [
    {
        "gate_group": "generation",
        "gate_id": "G1_closed_body",
        "criterion": "closed defect body",
        "required_evidence": "COMSOL inventory row or mesh validation row with closed_body_success=true",
        "pass_rule": "every generated defect body is closed before Boolean subtract",
        "blocks_next_route": True,
        "notes": "not evaluated in 25.1; required for 25.2",
    },
    {
        "gate_group": "generation",
        "gate_id": "G2_boolean_subtract",
        "criterion": "Boolean subtract",
        "required_evidence": "boolean_subtract_success=true and nonempty steel_notched domain",
        "pass_rule": "no empty-domain or sliver-domain blocker",
        "blocks_next_route": True,
        "notes": "apply 20.70 dynamic material/domain pattern",
    },
    {
        "gate_group": "generation",
        "gate_id": "G3_mesh_solver",
        "criterion": "mesh and solver complete",
        "required_evidence": "mesh_build_success=true and stationary solve complete",
        "pass_rule": "solver produces finite field export for each accepted sample",
        "blocks_next_route": True,
        "notes": "COMSOL is not run in 25.1",
    },
    {
        "gate_group": "generation",
        "gate_id": "G4_field_finite",
        "criterion": "Bx/By/Bz finite",
        "required_evidence": "all exported Bx, By, Bz arrays finite over scan grid",
        "pass_rule": "no NaN/Inf and axis order explicitly recorded",
        "blocks_next_route": True,
        "notes": "input shape should remain compatible with (N,3,3,201) pilot unless route changes explicitly",
    },
    {
        "gate_group": "generation",
        "gate_id": "G5_delta_b",
        "criterion": "delta_b check",
        "required_evidence": "delta_b equals b_defect - b_no_defect within tolerance",
        "pass_rule": "max absolute reconstruction error is recorded and acceptable",
        "blocks_next_route": True,
        "notes": "no-defect reference reuse must be explicit",
    },
    {
        "gate_group": "generation",
        "gate_id": "G6_profile_labels",
        "criterion": "profile/depth labels valid",
        "required_evidence": "finite depth_grid_m, nonempty projected_mask_2d, valid profile_descriptor",
        "pass_rule": "all accepted samples pass label validation",
        "blocks_next_route": True,
        "notes": "non-RBC labels must keep their richer target representation",
    },
    {
        "gate_group": "dataset",
        "gate_id": "D1_shape_coverage",
        "criterion": "shape coverage",
        "required_evidence": "dataset coverage CSV",
        "pass_rule": "all seven shape families appear in train/val/test and meet full coverage minima",
        "blocks_next_route": True,
        "notes": "N=84 cannot pass the full coverage rule",
    },
    {
        "gate_group": "dataset",
        "gate_id": "D2_split_coverage",
        "criterion": "split coverage",
        "required_evidence": "manifest split counts and stratification table",
        "pass_rule": "train/val/test are fixed before training and test is final-only",
        "blocks_next_route": True,
        "notes": "default pilot split is 72/24/24",
    },
    {
        "gate_group": "dataset",
        "gate_id": "D3_topology_coverage",
        "criterion": "topology coverage",
        "required_evidence": "topology_type counts by split",
        "pass_rule": "single_component, multi_component, elongated, irregular are covered",
        "blocks_next_route": True,
        "notes": "topology errors become audit metrics",
    },
    {
        "gate_group": "dataset",
        "gate_id": "D4_no_label_leakage",
        "criterion": "no label leakage",
        "required_evidence": "loader/input audit",
        "pass_rule": "target labels do not enter model inputs unless route explicitly permits conditioning",
        "blocks_next_route": True,
        "notes": "shape_type and component labels default to target/audit metadata",
    },
    {
        "gate_group": "dataset",
        "gate_id": "D5_registry_manifest",
        "criterion": "registry/manifest explicit read",
        "required_evidence": "COMSOL_DATA_REGISTRY and manifest path are explicitly read",
        "pass_rule": "no latest/newest auto-discovery",
        "blocks_next_route": True,
        "notes": "inherits current project data safety rule",
    },
    {
        "gate_group": "baseline_audit",
        "gate_id": "B1_current_20_85_failure_modes",
        "criterion": "current 20.85 failure modes",
        "required_evidence": "R0 audit summary",
        "pass_rule": "documents non-RBC-like failures before any training route",
        "blocks_next_route": True,
        "notes": "25.3 before 25.4 training",
    },
    {
        "gate_group": "baseline_audit",
        "gate_id": "B2_profile_rmse",
        "criterion": "profile RMSE",
        "required_evidence": "profile/depth metric table",
        "pass_rule": "profile RMSE remains primary over Dice-only gains",
        "blocks_next_route": True,
        "notes": "compares by shape family",
    },
    {
        "gate_group": "baseline_audit",
        "gate_id": "B3_component_recall",
        "criterion": "component recall",
        "required_evidence": "multi-pit component audit",
        "pass_rule": "component recall and merge/split errors are measured",
        "blocks_next_route": True,
        "notes": "required for component_set route",
    },
    {
        "gate_group": "baseline_audit",
        "gate_id": "B4_edge_corner_metrics",
        "criterion": "edge/corner metrics",
        "required_evidence": "flat/sharp-wall/elongated audit metrics",
        "pass_rule": "edge and corner errors are reported, not hidden by mask Dice",
        "blocks_next_route": True,
        "notes": "required for contour route",
    },
    {
        "gate_group": "baseline_audit",
        "gate_id": "B5_multi_pit_merge_rate",
        "criterion": "multi-pit merge rate",
        "required_evidence": "connected component prediction audit",
        "pass_rule": "merged-component cases are counted",
        "blocks_next_route": True,
        "notes": "separate from component recall",
    },
    {
        "gate_group": "baseline_audit",
        "gate_id": "B6_crack_like_miss_rate",
        "criterion": "crack-like miss rate",
        "required_evidence": "elongated footprint audit",
        "pass_rule": "missed narrow defects and aspect collapse are counted",
        "blocks_next_route": True,
        "notes": "required for elongated shape route",
    },
    {
        "gate_group": "model",
        "gate_id": "M1_non_rbc_improvement",
        "criterion": "must beat current baseline on non-RBC-like",
        "required_evidence": "formal benchmark by shape family",
        "pass_rule": "non-RBC profile/depth metrics improve over R0",
        "blocks_next_route": True,
        "notes": "no benchmark, no baseline transition",
    },
    {
        "gate_group": "model",
        "gate_id": "M2_rbc_control_not_collapse",
        "criterion": "must not collapse RBC-like",
        "required_evidence": "RBC-like control metrics",
        "pass_rule": "RBC-like profile metrics do not materially regress",
        "blocks_next_route": True,
        "notes": "protects current baseline scope",
    },
    {
        "gate_group": "model",
        "gate_id": "M3_forward_residual_non_worsening",
        "criterion": "forward consistency residual must not worsen",
        "required_evidence": "validated forward/surrogate residual table",
        "pass_rule": "forward residual is not worse on validation/test audits",
        "blocks_next_route": True,
        "notes": "R5 is a later gate, not a 25.1 implementation",
    },
    {
        "gate_group": "model",
        "gate_id": "M4_no_baseline_transition_without_benchmark",
        "criterion": "no baseline transition unless formal benchmark passes",
        "required_evidence": "formal route decision and approved baseline-transition prompt",
        "pass_rule": "CURRENT_BASELINE remains unchanged until later explicit approval",
        "blocks_next_route": True,
        "notes": "25.1 never updates CURRENT_BASELINE.md",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 25.1 surface shape-extension acceptance gates.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    write_csv(args.matrix, ROWS, FIELDS)
    groups: dict[str, int] = {}
    for row in ROWS:
        groups[row["gate_group"]] = groups.get(row["gate_group"], 0) + 1
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface shape-extension acceptance gate summary",
                "stage: 25.1",
                "",
                f"gate_count: {len(ROWS)}",
                *[f"{group}_gate_count: {count}" for group, count in sorted(groups.items())],
                "",
                "generation_gate: closed body, Boolean subtract, mesh/solver, finite Bx/By/Bz, delta_b check, profile/depth labels valid.",
                "dataset_gate: shape/split/topology coverage, no label leakage, explicit registry/manifest read.",
                "baseline_audit_gate: audit 20.85 failures before any training.",
                "model_gate: beat current baseline on non-RBC-like, keep RBC-like control stable, forward residual non-worsening, no baseline transition without formal benchmark.",
                "scope_boundary: these are acceptance definitions only; no COMSOL, no training, no CURRENT_BASELINE update.",
                f"acceptance_gate_matrix: {args.matrix}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
