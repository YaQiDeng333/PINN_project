#!/usr/bin/env python
"""Design COMSOL generation feasibility for the surface shape-extension pilot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/surface_shape_extension_comsol_feasibility_summary.txt"
MATRIX = ROOT / "results/metrics/surface_shape_extension_comsol_feasibility_matrix.csv"

FIELDS = [
    "shape_type",
    "preferred_generation_route",
    "alternative_generation_routes",
    "boolean_feasibility",
    "mesh_risk",
    "solver_risk",
    "label_extraction_method",
    "expected_failure_modes",
    "no_defect_reference_reusable",
    "material_domain_solver_fix_20_70_applies",
    "pilot_generation_gate",
    "notes",
]

ROWS: list[dict[str, Any]] = [
    {
        "shape_type": "rbc_like_smooth_pit",
        "preferred_generation_route": "imported_watertight_mesh_solid",
        "alternative_generation_routes": "stacked_layer_control; high_layer_depth_control",
        "boolean_feasibility": "known_feasible_for_current_RBC_route",
        "mesh_risk": "medium",
        "solver_risk": "medium",
        "label_extraction_method": "existing RBC params -> depth_grid_m -> projected_mask_2d",
        "expected_failure_modes": "import repair/domain selection drift; smooth route not exact Piao RBC",
        "no_defect_reference_reusable": True,
        "material_domain_solver_fix_20_70_applies": True,
        "pilot_generation_gate": "must reproduce finite Bx/By/Bz and current label schema",
        "notes": "anchor/control family only",
    },
    {
        "shape_type": "flat_bottom_pit",
        "preferred_generation_route": "COMSOL-native geometry",
        "alternative_generation_routes": "stacked_layer_control; imported_watertight_mesh_solid",
        "boolean_feasibility": "expected_feasible",
        "mesh_risk": "medium_at_sharp_edges",
        "solver_risk": "medium",
        "label_extraction_method": "analytic plateau depth map plus contour/mask rasterization",
        "expected_failure_modes": "edge singularity, mesh refinement near wall, rounded edge after repair",
        "no_defect_reference_reusable": True,
        "material_domain_solver_fix_20_70_applies": True,
        "pilot_generation_gate": "closed pit body, nonempty plateau, edge_steepness label finite",
        "notes": "start with finite wall slope before near-vertical wall",
    },
    {
        "shape_type": "sharp_wall_boxy_corrosion",
        "preferred_generation_route": "COMSOL-native geometry",
        "alternative_generation_routes": "multi-component Boolean subtract; imported_watertight_mesh_solid",
        "boolean_feasibility": "expected_feasible",
        "mesh_risk": "medium_high_at_corners",
        "solver_risk": "medium",
        "label_extraction_method": "rotated rectangle / polygon contour plus constant depth grid",
        "expected_failure_modes": "corner mesh concentration, Boolean sliver domains, straight-edge loss",
        "no_defect_reference_reusable": True,
        "material_domain_solver_fix_20_70_applies": True,
        "pilot_generation_gate": "corner/edge metrics valid and projected mask matches polygon",
        "notes": "do not treat rounded RBC reconstruction as acceptable",
    },
    {
        "shape_type": "asymmetric_corrosion",
        "preferred_generation_route": "height_depth_map_solid",
        "alternative_generation_routes": "imported_watertight_mesh_solid; stacked_layer_control",
        "boolean_feasibility": "needs_pilot_validation",
        "mesh_risk": "high_if_depth_gradient_large",
        "solver_risk": "medium_high",
        "label_extraction_method": "depth_grid_m with asymmetry_score and edge_steepness proxies",
        "expected_failure_modes": "open/invalid imported body, excessive ramp slope, max-depth offset lost",
        "no_defect_reference_reusable": True,
        "material_domain_solver_fix_20_70_applies": True,
        "pilot_generation_gate": "closed body and finite depth grid with asymmetric max-depth offset",
        "notes": "first use bounded skew values",
    },
    {
        "shape_type": "elongated_crack_like_surface_defect",
        "preferred_generation_route": "COMSOL-native geometry",
        "alternative_generation_routes": "imported_watertight_mesh_solid; polygon_or_contour primitive",
        "boolean_feasibility": "expected_feasible_for_rotated_slot",
        "mesh_risk": "high_for_narrow_width",
        "solver_risk": "medium_high",
        "label_extraction_method": "rotated slot contour, aspect_ratio, rotation_angle, depth grid",
        "expected_failure_modes": "thin feature disappears in mesh, width over-smoothing, rotation mismatch",
        "no_defect_reference_reusable": True,
        "material_domain_solver_fix_20_70_applies": True,
        "pilot_generation_gate": "minimum width survives mesh and projected_mask_2d is nonempty",
        "notes": "bound aspect ratio before crack-like extreme cases",
    },
    {
        "shape_type": "multi_pit_two_component_surface_defect",
        "preferred_generation_route": "multi-component Boolean subtract",
        "alternative_generation_routes": "COMSOL-native geometry; imported watertight mesh per component",
        "boolean_feasibility": "known_feasible_for_two_rect_rotated_components_Bz_branch",
        "mesh_risk": "medium_high_when_components_close",
        "solver_risk": "medium_high",
        "label_extraction_method": "component_params_json plus union depth/mask and component_count",
        "expected_failure_modes": "merged domains, component overlap, missed secondary component, label/union mismatch",
        "no_defect_reference_reusable": True,
        "material_domain_solver_fix_20_70_applies": True,
        "pilot_generation_gate": "component_count=2 preserved in labels and mask topology",
        "notes": "surface Bx/By/Bz pilot must validate beyond older Bz-oriented multi-defect work",
    },
    {
        "shape_type": "irregular_non_rbc_corrosion",
        "preferred_generation_route": "imported_watertight_mesh_solid",
        "alternative_generation_routes": "height_depth_map_solid; stacked_layer_control",
        "boolean_feasibility": "highest_risk_needs_smoke",
        "mesh_risk": "high",
        "solver_risk": "high",
        "label_extraction_method": "contour_vertices_json and full depth_grid_m",
        "expected_failure_modes": "self-intersecting contour, non-watertight mesh, Boolean failure, local extrema lost",
        "no_defect_reference_reusable": True,
        "material_domain_solver_fix_20_70_applies": True,
        "pilot_generation_gate": "watertight closed body, finite grid, nonempty mask, no self-intersection",
        "notes": "pilot should keep irregularity bounded before benchmark expansion",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design COMSOL feasibility plan for 25.1 surface shape extension.")
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
    high_risk = [row["shape_type"] for row in ROWS if "high" in str(row["mesh_risk"]) or "high" in str(row["solver_risk"])]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface shape-extension COMSOL feasibility summary",
                "stage: 25.1",
                "",
                "scope: feasibility plan only; no COMSOL execution.",
                "no_defect_reference_policy: reuse no-defect reference is allowed only when source/material/domain/sensor setup remains identical and manifest records it explicitly.",
                "material_domain_solver_policy: 20.70 dynamic material/domain/solver fix applies as a required safety pattern for imported or Boolean-subtracted defect solids.",
                f"shape_count: {len(ROWS)}",
                f"high_risk_shapes: {', '.join(high_risk)}",
                "",
                "preferred_routes:",
                *[f"- {row['shape_type']}: {row['preferred_generation_route']}" for row in ROWS],
                "",
                "pilot_boundary: 25.2 may generate pilot only after taxonomy/schema/feasibility gates pass; 25.1 generates no data.",
                f"feasibility_matrix: {args.matrix}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
