#!/usr/bin/env python
"""Design model routes for the surface shape-extension branch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/surface_shape_extension_model_route_summary.txt"
MATRIX = ROOT / "results/metrics/surface_shape_extension_model_route_matrix.csv"

FIELDS = [
    "route_id",
    "route_name",
    "stage",
    "allowed_before_training",
    "target_shape_types",
    "output_representation",
    "input_policy",
    "primary_metric",
    "success_gate",
    "failure_modes_to_audit",
    "baseline_policy",
]

ROWS: list[dict[str, Any]] = [
    {
        "route_id": "R0",
        "route_name": "current_baseline_audit",
        "stage": "25.3",
        "allowed_before_training": True,
        "target_shape_types": "all shape-extension pilot families",
        "output_representation": "existing six_param_rbc prediction replayed as diagnostic",
        "input_policy": "load frozen 20.85/20.77 artifact explicitly by manifest; no retraining",
        "primary_metric": "profile RMSE / Er-like / shape-specific failure metrics",
        "success_gate": "document where current RBC baseline fails on non-RBC-like shapes",
        "failure_modes_to_audit": "RBC collapse, component merge, crack miss, edge/corner blur, asymmetric recentering",
        "baseline_policy": "audit only; does not update CURRENT_BASELINE",
    },
    {
        "route_id": "R1",
        "route_name": "six_param_rbc_control",
        "stage": "25.3_or_later",
        "allowed_before_training": False,
        "target_shape_types": "rbc_like_smooth_pit only",
        "output_representation": "L_m,W_m,D_m,wLD,wWD,wLW",
        "input_policy": "same Bx/By/Bz delta_b input contract as current baseline",
        "primary_metric": "RBC-like profile RMSE and Er-like error",
        "success_gate": "must not regress RBC-like controls relative to current baseline audit",
        "failure_modes_to_audit": "nominal RBC profile degradation",
        "baseline_policy": "control branch only; not valid for non-RBC-like claims",
    },
    {
        "route_id": "R2",
        "route_name": "profile_basis_decoder",
        "stage": "25.4_candidate",
        "allowed_before_training": False,
        "target_shape_types": "asymmetric_corrosion; irregular_non_rbc_corrosion",
        "output_representation": "profile_basis or low-dimensional depth/profile coefficients",
        "input_policy": "delta_b Bx/By/Bz with train-only normalization and validation-only selection",
        "primary_metric": "profile RMSE, Er-like error, asymmetry_score error",
        "success_gate": "beats R0 on non-RBC-like profile metrics without RBC-like collapse",
        "failure_modes_to_audit": "over-smoothing and local-depth extrema loss",
        "baseline_policy": "candidate only until formal benchmark passes",
    },
    {
        "route_id": "R3",
        "route_name": "component_set_decoder",
        "stage": "25.4_candidate",
        "allowed_before_training": False,
        "target_shape_types": "multi_pit_two_component_surface_defect",
        "output_representation": "K component set with component_params_json-compatible fields",
        "input_policy": "delta_b Bx/By/Bz; component labels are targets, not inputs",
        "primary_metric": "component recall, merge rate, component depth/pose error",
        "success_gate": "component recall improves over R0 and merge rate is bounded",
        "failure_modes_to_audit": "merged pits, missing secondary component, swapped component order",
        "baseline_policy": "candidate only; no baseline transition without benchmark",
    },
    {
        "route_id": "R4",
        "route_name": "geometry_aware_contour_decoder",
        "stage": "25.4_candidate",
        "allowed_before_training": False,
        "target_shape_types": "flat_bottom_pit; sharp_wall_boxy_corrosion; elongated_crack_like_surface_defect",
        "output_representation": "polygon_or_contour or rotated primitives",
        "input_policy": "delta_b Bx/By/Bz; shape/topology labels used for stratified audit",
        "primary_metric": "edge/corner metrics, aspect/rotation error, profile RMSE",
        "success_gate": "preserves sharp edges and elongated footprints better than R0",
        "failure_modes_to_audit": "rounded corners, width collapse, rotation error",
        "baseline_policy": "candidate only; no Dice-only upgrade",
    },
    {
        "route_id": "R5",
        "route_name": "forward_consistency_refinement",
        "stage": "25.4_or_later_after_representation_selection",
        "allowed_before_training": False,
        "target_shape_types": "all representation-compatible shape families",
        "output_representation": "predicted profile -> forward surrogate -> Bx/By/Bz residual",
        "input_policy": "forward residual is validation-gated; test set is final only",
        "primary_metric": "profile metric plus forward residual non-worsening",
        "success_gate": "forward residual does not worsen and profile metrics improve on non-RBC-like cases",
        "failure_modes_to_audit": "physics residual rewards wrong geometry or overfits validation",
        "baseline_policy": "refinement gate only; no standalone baseline transition",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 25.1 surface shape-extension model routes.")
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
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface shape-extension model route summary",
                "stage: 25.1",
                "",
                "stage_policy:",
                "- 25.2: COMSOL pilot generation only; no training.",
                "- 25.3: audit current 20.85 baseline on the shape-extension pilot.",
                "- 25.4: consider model training only after pilot and audit gates pass.",
                "",
                "routes:",
                *[f"- {row['route_id']}_{row['route_name']}: stage={row['stage']}; target={row['target_shape_types']}; output={row['output_representation']}" for row in ROWS],
                "",
                "baseline_policy: CURRENT_BASELINE remains 20.85 unless a later formal benchmark explicitly passes and a baseline-transition prompt approves it.",
                f"model_route_matrix: {args.matrix}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
