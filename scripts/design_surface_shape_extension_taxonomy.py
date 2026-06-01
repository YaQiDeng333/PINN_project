#!/usr/bin/env python
"""Design the 25.1 surface shape-extension taxonomy.

This is a planning script only. It does not run COMSOL, train a model, write
data/NPZ artifacts, or update CURRENT_BASELINE.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_SUMMARY = ROOT / "results/summaries/surface_shape_extension_dataset_plan_preflight_summary.txt"
TAXONOMY_SUMMARY = ROOT / "results/summaries/surface_shape_extension_taxonomy_summary.txt"
TAXONOMY_MATRIX = ROOT / "results/metrics/surface_shape_extension_taxonomy_matrix.csv"

TAXONOMY_FIELDS = [
    "shape_type",
    "profile_family",
    "geometry_parameters",
    "depth_map_generation_rule",
    "projected_mask_generation_rule",
    "topology_label",
    "expected_difficulty",
    "rbc_six_param_representation_valid",
    "new_output_representation_required",
    "recommended_representation_target",
    "primary_failure_mode_to_audit",
]

TAXONOMY_ROWS: list[dict[str, Any]] = [
    {
        "shape_type": "rbc_like_smooth_pit",
        "profile_family": "smooth radial RBC-style / Piao-inspired pit",
        "geometry_parameters": "L_m,W_m,D_m,wLD,wWD,wLW,center_xyz_m,rotation_angle",
        "depth_map_generation_rule": "evaluate existing RBC-style six-parameter depth profile on the profile grid",
        "projected_mask_generation_rule": "threshold depth_grid_m > 0 and rasterize projected footprint",
        "topology_label": "single_component",
        "expected_difficulty": "control_anchor",
        "rbc_six_param_representation_valid": True,
        "new_output_representation_required": False,
        "recommended_representation_target": "six_param_rbc",
        "primary_failure_mode_to_audit": "do not regress current 20.85 RBC-like profile/depth behavior",
    },
    {
        "shape_type": "flat_bottom_pit",
        "profile_family": "plateau bottom with steep sidewall transition",
        "geometry_parameters": "L_m,W_m,D_m,wall_slope,corner_radius_m,center_xyz_m,rotation_angle",
        "depth_map_generation_rule": "constant central depth with finite edge ramp controlled by wall_slope/corner_radius_m",
        "projected_mask_generation_rule": "threshold nonzero plateau plus edge ramp footprint",
        "topology_label": "single_component",
        "expected_difficulty": "medium_high_edge_sharpness",
        "rbc_six_param_representation_valid": False,
        "new_output_representation_required": True,
        "recommended_representation_target": "polygon_or_contour",
        "primary_failure_mode_to_audit": "smooth RBC output blurs the flat bottom and underestimates edge steepness",
    },
    {
        "shape_type": "sharp_wall_boxy_corrosion",
        "profile_family": "boxy / cuboid-like surface material loss",
        "geometry_parameters": "L_m,W_m,D_m,corner_radius_m,edge_angle_deg,center_xyz_m,rotation_angle",
        "depth_map_generation_rule": "rectangular or polygonal footprint with near-vertical walls and uniform depth",
        "projected_mask_generation_rule": "polygon/rotated-rectangle rasterization of the top-view footprint",
        "topology_label": "single_component",
        "expected_difficulty": "high_corner_boundary_fidelity",
        "rbc_six_param_representation_valid": False,
        "new_output_representation_required": True,
        "recommended_representation_target": "polygon_or_contour",
        "primary_failure_mode_to_audit": "rounded predicted corners, edge leakage, and wrong straight-boundary length",
    },
    {
        "shape_type": "asymmetric_corrosion",
        "profile_family": "single pit with skewed depth distribution",
        "geometry_parameters": "L_m,W_m,D_m,skew_x,skew_y,edge_steepness_left,edge_steepness_right,center_xyz_m",
        "depth_map_generation_rule": "depth surface with one steep side and one shallow side; max depth offset from center",
        "projected_mask_generation_rule": "threshold asymmetric depth grid and keep largest connected footprint",
        "topology_label": "single_component",
        "expected_difficulty": "high_profile_asymmetry",
        "rbc_six_param_representation_valid": False,
        "new_output_representation_required": True,
        "recommended_representation_target": "profile_basis",
        "primary_failure_mode_to_audit": "six-parameter output recenters the pit and loses skewed profile shape",
    },
    {
        "shape_type": "elongated_crack_like_surface_defect",
        "profile_family": "long narrow connected slot / crack-like pit",
        "geometry_parameters": "L_m,W_m,D_m,aspect_ratio,rotation_angle,tip_radius_m,center_xyz_m",
        "depth_map_generation_rule": "rotated high-aspect slot with narrow width and optional tapered tips",
        "projected_mask_generation_rule": "rotated elongated footprint rasterization with width-preserving threshold",
        "topology_label": "elongated",
        "expected_difficulty": "high_aspect_rotation_width_recall",
        "rbc_six_param_representation_valid": False,
        "new_output_representation_required": True,
        "recommended_representation_target": "polygon_or_contour",
        "primary_failure_mode_to_audit": "missed narrow component, aspect collapse, and rotation error",
    },
    {
        "shape_type": "multi_pit_two_component_surface_defect",
        "profile_family": "two neighboring or separated surface pits",
        "geometry_parameters": "component_count,component_params_json,component_distance_m,depth_ratio,center_xyz_m",
        "depth_map_generation_rule": "union of two component depth maps with independent size/depth/profile parameters",
        "projected_mask_generation_rule": "union of component masks while preserving connected_component_count",
        "topology_label": "multi_component",
        "expected_difficulty": "high_component_recall_merge_split",
        "rbc_six_param_representation_valid": False,
        "new_output_representation_required": True,
        "recommended_representation_target": "component_set",
        "primary_failure_mode_to_audit": "merged components, missed secondary pit, and wrong component depths",
    },
    {
        "shape_type": "irregular_non_rbc_corrosion",
        "profile_family": "free-form irregular corrosion with nonuniform depth",
        "geometry_parameters": "contour_vertices_json,depth_grid_m,local_depth_modes,roughness_score,center_xyz_m",
        "depth_map_generation_rule": "generate a bounded irregular contour and nonuniform depth grid with finite max depth",
        "projected_mask_generation_rule": "rasterize contour/depth threshold and store full projected mask",
        "topology_label": "irregular",
        "expected_difficulty": "very_high_free_form_profile",
        "rbc_six_param_representation_valid": False,
        "new_output_representation_required": True,
        "recommended_representation_target": "depth_grid",
        "primary_failure_mode_to_audit": "over-smoothing, boundary hallucination, and loss of local depth extrema",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 25.1 surface shape-extension taxonomy.")
    parser.add_argument("--preflight-summary", type=Path, default=PREFLIGHT_SUMMARY)
    parser.add_argument("--summary", type=Path, default=TAXONOMY_SUMMARY)
    parser.add_argument("--matrix", type=Path, default=TAXONOMY_MATRIX)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def preflight_lines() -> list[str]:
    required_files = [
        ROOT / "CURRENT_BASELINE.md",
        ROOT / "COMSOL_DATA_REGISTRY.md",
        ROOT / "NLS_FULL_COMPATIBLE_FEATURE_SCHEMA.md",
        ROOT / "results/summaries/surface_piao_nls_branch_closeout_summary.txt",
        ROOT / "results/summaries/surface_shape_extension_forward_consistency_plan_summary.txt",
        ROOT / "results/metrics/surface_piao_nls_branch_closeout_matrix.csv",
        ROOT / "results/metrics/surface_shape_extension_forward_consistency_plan.csv",
    ]
    reference_metrics = [
        ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_metrics.csv",
        ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_metrics.csv",
        ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_screen_metrics.csv",
        ROOT / "results/metrics/surface_rbc_nls_lite_feature_decision_matrix.csv",
        ROOT / "results/metrics/surface_rbc_nls_full_compatible_decision_matrix.csv",
        ROOT / "results/metrics/surface_rbc_piao_style_feature_baseline_metrics.csv",
        ROOT / "results/metrics/surface_rbc_nls_feature_fusion_metrics.csv",
    ]
    existing_design_scripts = sorted((ROOT / "scripts").glob("design_true_3d_rbc*.py"))
    existing_surface_scripts = sorted((ROOT / "scripts").glob("*surface_rbc*.py"))
    forbidden_roots = [
        ROOT / "data",
        ROOT / "checkpoints",
        ROOT / "notes",
        ROOT / "results/previews",
    ]
    lines = [
        "surface shape-extension dataset plan preflight summary",
        "stage: 25.1",
        "",
        "scope: read-only preflight for taxonomy/schema/dataset planning; no COMSOL, no training, no data/NPZ generation, no CURRENT_BASELINE update.",
        "",
        "required_context_files:",
    ]
    for path in required_files:
        lines.append(f"- {path.relative_to(ROOT)}: exists={bool_text(path.exists())}")
    lines.extend(["", "reference_metric_files:"])
    for path in reference_metrics:
        lines.append(f"- {path.relative_to(ROOT)}: exists={bool_text(path.exists())}")
    lines.extend(
        [
            "",
            f"existing_true_3d_design_script_count: {len(existing_design_scripts)}",
            f"existing_surface_rbc_script_count: {len(existing_surface_scripts)}",
            "forbidden_artifact_roots_present_but_not_touched:",
        ]
    )
    for path in forbidden_roots:
        lines.append(f"- {path.relative_to(ROOT)}: exists={bool_text(path.exists())}; action=do_not_stage_or_modify")
    lines.extend(
        [
            "",
            "anchor_baseline: 20.85 / 20.86 CURRENT_BASELINE remains true 3D RBC-style profile-depth baseline.",
            "anchor_dataset_id: comsol_true_3d_rbc_imported_watertight_pilot_v3_240",
            "current_gap: current six-parameter RBC representation cannot claim arbitrary surface-defect coverage.",
            "preflight_decision: proceed with plan-only taxonomy, schema, split coverage, feasibility, model route, and gates.",
        ]
    )
    return lines


def summary_lines() -> list[str]:
    rbc_count = sum(1 for row in TAXONOMY_ROWS if row["rbc_six_param_representation_valid"])
    non_rbc_count = len(TAXONOMY_ROWS) - rbc_count
    return [
        "surface shape-extension taxonomy summary",
        "stage: 25.1",
        "",
        f"shape_family_count: {len(TAXONOMY_ROWS)}",
        f"rbc_compatible_count: {rbc_count}",
        f"non_rbc_like_count: {non_rbc_count}",
        "rbc_anchor: rbc_like_smooth_pit remains a control / anchor family only.",
        "non_rbc_boundary: non-RBC-like surface defects must not be forced into L_m/W_m/D_m/wLD/wWD/wLW.",
        "required_new_targets: profile_basis, depth_grid, component_set, polygon_or_contour.",
        "",
        "shape_types:",
        *[
            "- {shape_type}: topology={topology_label}; target={recommended_representation_target}; rbc_valid={rbc}".format(
                rbc=bool_text(bool(row["rbc_six_param_representation_valid"])),
                **row,
            )
            for row in TAXONOMY_ROWS
        ],
        "",
        f"taxonomy_matrix: {TAXONOMY_MATRIX}",
    ]


def run(args: argparse.Namespace) -> int:
    args.preflight_summary.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_summary.write_text("\n".join(preflight_lines()) + "\n", encoding="utf-8")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary_lines()) + "\n", encoding="utf-8")
    write_csv(args.matrix, TAXONOMY_ROWS, TAXONOMY_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
