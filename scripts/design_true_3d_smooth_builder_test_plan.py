#!/usr/bin/env python
"""Design the 20.68 smooth / near-smooth true-3D builder test plan.

This is a planning and accounting script. It does not run COMSOL, does not
generate data packs, and does not train models. The RBC profile remains an
RBC-style engineering approximation unless a later stage explicitly verifies
the exact Piao formula.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

csv.field_size_limit(2**31 - 1)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import design_true_3d_rbc_smoke_plan as smoke  # noqa: E402


DEFAULT_PLAN_CSV = ROOT / "results/metrics/true_3d_smooth_builder_test_plan.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_smooth_builder_test_plan_summary.txt"
DEFAULT_PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_smooth_builder_preflight_summary.txt"
DEFAULT_ROUTE_SUMMARY = ROOT / "results/summaries/true_3d_smooth_variable_depth_builder_route_decision_summary.txt"
DEFAULT_ROUTE_MATRIX = ROOT / "results/metrics/true_3d_smooth_variable_depth_builder_route_decision_matrix.csv"
DEFAULT_BUILDER_INVENTORY = Path(
    r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\true_3d_smooth_variable_depth_builder_test_inventory.csv"
)

SELECTED_SAMPLE_IDS = ("medium_round", "deep_round", "medium_boxy")
PRIMARY_SAMPLE_ID = "medium_round"
LEVEL_FRACTIONS_24 = tuple(float(v) for v in np.linspace(0.04, 0.98, 24))
LEVEL_FRACTIONS_32 = tuple(float(v) for v in np.linspace(0.03, 0.985, 32))

PLAN_FIELDS = [
    "sample_id",
    "selection_priority",
    "forward_eligible",
    "split_tag",
    "profile_type",
    "exact_piao_rbc",
    "rbc_style_approximation",
    "rbc_formula_status",
    "L_m",
    "W_m",
    "D_m",
    "wLD",
    "wWD",
    "wLW",
    "center_x_m",
    "center_y_m",
    "angle_rad",
    "angle_deg",
    "sensor_z_m",
    "scan_line_y_json",
    "axis_names_json",
    "axis_expressions_json",
    "candidate_geometry_methods_json",
    "bounded_attempt_policy",
    "stage_c_forward_gate_json",
    "profile_pose_json",
    "rbc_params_json",
    "profile_depth_grid_shape_json",
    "profile_depth_grid_m_json",
    "profile_depth_map_xy_shape_json",
    "profile_depth_map_xy_m_json",
    "projected_mask_2d_shape_json",
    "projected_mask_2d_json",
    "projection_threshold_m",
    "expected_depth_max_m",
    "depth_max_abs_error_m",
    "footprint_area_m2",
    "footprint_area_px",
    "depth_levels_24_m_json",
    "depth_level_polygons_24_json",
    "depth_levels_32_m_json",
    "depth_level_polygons_32_json",
    "geometry_params_json",
    "geometry_validation_metadata_json",
    "stage_a_validation_pass",
    "notes",
]

ROUTE_FIELDS = [
    "decision_option",
    "selected",
    "condition",
    "observed",
    "next_step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.68 smooth builder plan.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT_SUMMARY)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--builder-inventory", type=Path, default=DEFAULT_BUILDER_INVENTORY)
    parser.add_argument("--route-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing files:\n{joined}")


def rounded_list(array: np.ndarray, decimals: int = 9) -> list[Any]:
    return np.round(array.astype(float), decimals=decimals).tolist()


def polygons_for_fractions(sample: smoke.SmokeSample, fractions: tuple[float, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, fraction in enumerate(fractions, start=1):
        depth_level = float(sample.D_m * fraction)
        vertices = smoke.contour_polygon_for_depth(sample, depth_level)
        area = smoke.polygon_area(vertices)
        if len(vertices) < 3 or area <= 0.0:
            continue
        rows.append(
            {
                "level_index": index,
                "depth_m": depth_level,
                "fraction_of_D": float(fraction),
                "vertex_count": int(len(vertices)),
                "area_m2": float(area),
                "vertices": rounded_list(vertices, decimals=9),
            }
        )
    return rows


def build_rows() -> list[dict[str, Any]]:
    samples = {sample.sample_id: sample for sample in smoke.sample_table()}
    missing = [sample_id for sample_id in SELECTED_SAMPLE_IDS if sample_id not in samples]
    if missing:
        raise RuntimeError(f"missing required 20.66 smoke samples: {missing}")

    mask_x = np.linspace(smoke.MASK_X_START_M, smoke.MASK_X_STOP_M, smoke.MASK_WIDTH)
    mask_y = np.linspace(smoke.MASK_Y_START_M, smoke.MASK_Y_STOP_M, smoke.MASK_HEIGHT)
    pixel_area = float((mask_x[1] - mask_x[0]) * (mask_y[1] - mask_y[0]))
    rows: list[dict[str, Any]] = []

    for priority, sample_id in enumerate(SELECTED_SAMPLE_IDS, start=1):
        sample = samples[sample_id]
        _, _, depth_grid = smoke.build_profile_depth_grid(sample)
        depth_map = smoke.build_depth_map(sample, mask_x, mask_y)
        projection_threshold = max(1.0e-6, 0.01 * sample.D_m)
        projected_mask = (depth_map >= projection_threshold).astype(np.uint8)
        polygons_24 = polygons_for_fractions(sample, LEVEL_FRACTIONS_24)
        polygons_32 = polygons_for_fractions(sample, LEVEL_FRACTIONS_32)
        max_depth = float(depth_grid.max())
        depth_error = abs(max_depth - sample.D_m)
        cc_count = smoke.connected_component_count(projected_mask)
        profile_pose = {
            "center_x_m": sample.center_x_m,
            "center_y_m": sample.center_y_m,
            "angle_rad": math.radians(sample.angle_deg),
            "angle_deg": sample.angle_deg,
            "L_m": sample.L_m,
            "W_m": sample.W_m,
            "D_m": sample.D_m,
        }
        rbc_params = {
            "L_m": sample.L_m,
            "W_m": sample.W_m,
            "D_m": sample.D_m,
            "wLD": sample.wLD,
            "wWD": sample.wWD,
            "wLW": sample.wLW,
        }
        stage_c_gate = {
            "closed_body_success": True,
            "boolean_subtract_success": True,
            "mesh_precheck_success": True,
            "spatial_depth_variation": True,
            "is_constant_depth": False,
            "status_allowed": ["variable_depth_pass", "near_smooth_pass"],
            "high_layer_control_pass_enters_forward": False,
        }
        geometry_params = {
            "sample_id": sample.sample_id,
            "task": "20.68 smooth / near-smooth true 3D variable-depth builder completion",
            "profile_type": "rbc_style_symmetric_pit",
            "exact_piao_rbc": False,
            "rbc_style_approximation": True,
            "rbc_formula_status": smoke.RBC_FORMULA_STATUS,
            "rbc_params": rbc_params,
            "profile_pose": profile_pose,
            "projection_threshold_m": projection_threshold,
            "candidate_geometry_methods": [
                "lofted_contour_solid",
                "stacked_workplane_contour_loft",
                "interpolated_surface_solid",
                "imported_closed_mesh_solid",
                "high_layer_control_24_or_32",
            ],
            "bounded_attempt_policy": "one bounded attempt per smooth/near-smooth candidate; no unbounded COMSOL debugging",
            "stage_c_forward_gate": stage_c_gate,
            "depth_levels_24_m": [row["depth_m"] for row in polygons_24],
            "depth_level_polygons_24": polygons_24,
            "depth_levels_32_m": [row["depth_m"] for row in polygons_32],
            "depth_level_polygons_32": polygons_32,
            "mesh_source_policy": {
                "imported_closed_mesh_solid": "must distinguish smooth_depth_surface / triangulated_depth_grid / stepped_layers",
                "stepped_layers_never_near_smooth": True,
            },
            "smooth_variable_depth_solid_verified": False,
            "near_smooth_pass": False,
            "high_layer_control_pass": False,
            "constant_depth_extrusion_used_as_success": False,
            "units": "coordinates=m, field=T",
        }
        validation_metadata = {
            "depth_grid_finite": bool(np.isfinite(depth_grid).all()),
            "depth_map_finite": bool(np.isfinite(depth_map).all()),
            "depth_nonnegative": bool((depth_grid >= -1.0e-12).all() and (depth_map >= -1.0e-12).all()),
            "projected_mask_nonempty": int(projected_mask.sum()) > 0,
            "projected_mask_connected_components": int(cc_count),
            "max_depth_rel_error": float(depth_error / max(sample.D_m, 1.0e-12)),
            "polygons_24_count": len(polygons_24),
            "polygons_24_min_area_m2": min((float(row["area_m2"]) for row in polygons_24), default=0.0),
            "polygons_32_count": len(polygons_32),
            "polygons_32_min_area_m2": min((float(row["area_m2"]) for row in polygons_32), default=0.0),
        }
        stage_a_pass = (
            validation_metadata["depth_grid_finite"]
            and validation_metadata["depth_map_finite"]
            and validation_metadata["depth_nonnegative"]
            and validation_metadata["projected_mask_nonempty"]
            and validation_metadata["projected_mask_connected_components"] == 1
            and validation_metadata["max_depth_rel_error"] <= 0.03
            and len(polygons_24) > 5
            and len(polygons_32) > 5
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "selection_priority": priority,
                "forward_eligible": sample.sample_id == PRIMARY_SAMPLE_ID,
                "split_tag": sample.split_tag,
                "profile_type": "rbc_style_symmetric_pit",
                "exact_piao_rbc": False,
                "rbc_style_approximation": True,
                "rbc_formula_status": smoke.RBC_FORMULA_STATUS,
                "L_m": sample.L_m,
                "W_m": sample.W_m,
                "D_m": sample.D_m,
                "wLD": sample.wLD,
                "wWD": sample.wWD,
                "wLW": sample.wLW,
                "center_x_m": sample.center_x_m,
                "center_y_m": sample.center_y_m,
                "angle_rad": math.radians(sample.angle_deg),
                "angle_deg": sample.angle_deg,
                "sensor_z_m": 0.008,
                "scan_line_y_json": json_dumps([-0.001, 0.0, 0.001]),
                "axis_names_json": json_dumps(["Bx", "By", "Bz"]),
                "axis_expressions_json": json_dumps(["mf.Bx", "mf.By", "mf.Bz"]),
                "candidate_geometry_methods_json": json_dumps(geometry_params["candidate_geometry_methods"]),
                "bounded_attempt_policy": geometry_params["bounded_attempt_policy"],
                "stage_c_forward_gate_json": json_dumps(stage_c_gate),
                "profile_pose_json": json_dumps(profile_pose),
                "rbc_params_json": json_dumps(rbc_params),
                "profile_depth_grid_shape_json": json_dumps(list(depth_grid.shape)),
                "profile_depth_grid_m_json": json_dumps(rounded_list(depth_grid, decimals=9)),
                "profile_depth_map_xy_shape_json": json_dumps(list(depth_map.shape)),
                "profile_depth_map_xy_m_json": json_dumps(rounded_list(depth_map, decimals=9)),
                "projected_mask_2d_shape_json": json_dumps(list(projected_mask.shape)),
                "projected_mask_2d_json": json_dumps(projected_mask.astype(int).tolist()),
                "projection_threshold_m": projection_threshold,
                "expected_depth_max_m": sample.D_m,
                "depth_max_abs_error_m": depth_error,
                "footprint_area_m2": int(projected_mask.sum()) * pixel_area,
                "footprint_area_px": int(projected_mask.sum()),
                "depth_levels_24_m_json": json_dumps([row["depth_m"] for row in polygons_24]),
                "depth_level_polygons_24_json": json_dumps(polygons_24),
                "depth_levels_32_m_json": json_dumps([row["depth_m"] for row in polygons_32]),
                "depth_level_polygons_32_json": json_dumps(polygons_32),
                "geometry_params_json": json_dumps(geometry_params),
                "geometry_validation_metadata_json": json_dumps(validation_metadata),
                "stage_a_validation_pass": stage_a_pass,
                "notes": "primary forward sample" if sample.sample_id == PRIMARY_SAMPLE_ID else "optional geometry-only sample after primary passes",
            }
        )
    return rows


def write_preflight(path: Path) -> None:
    lines = [
        "20.68 smooth / near-smooth true 3D builder preflight summary",
        "",
        "Subagent preflight status: completed in the planning pass; existing six read-only subagents were reused because new subagent spawn hit a thread limit.",
        "",
        "Agent A method/literature conclusion:",
        "- GO. Smooth / near-smooth builder hardening is the right next step after 20.67 high_layer_approx_12.",
        "- This stage is not a full Piao 2019 reproduction; exact_piao_rbc remains False unless the exact formula and geometry are separately verified.",
        "",
        "Agent B COMSOL capability conclusion:",
        "- Boolean Difference, WorkPlane/Polygon/Extrude, mesh precheck, 3D volume solve, and Bx/By/Bz export are already usable.",
        "- Loft / ParametricSurface / Import existed only as unverified probe paths; 20.67 did not produce a verified smooth closed defect body.",
        "- The proven route remains layered nested contours, but it must be labeled high_layer control rather than smooth or near-smooth.",
        "",
        "Agent C geometry builder design conclusion:",
        "- Candidate order: lofted contour solid, stacked workplane contour loft, interpolated surface solid, imported closed mesh solid, high-layer 24/32 control.",
        "- Imported mesh must record mesh_source and cannot treat stepped-layer triangulation as near-smooth.",
        "",
        "Agent D experiment design conclusion:",
        "- Primary sample is medium_round. deep_round and medium_boxy are geometry-only optional checks after primary success.",
        "- Stage C forward is allowed only after variable_depth_pass or near_smooth_pass.",
        "",
        "Agent E safety conclusion:",
        "- Do not submit data, NPZ, checkpoint, preview PNG, .mph, raw CSV, notes, baseline docs, MODEL_STRUCTURE_PLAN.md deletion, or unrelated dirty items.",
        "",
        "Agent F implementation conclusion:",
        "- Geometry-only builder probe must precede any forward solve.",
        "- If only high_layer_control_pass is reached, do not run Stage C and do not write pilot-ready.",
        "",
        "Stop conditions:",
        "- only constant-depth geometry is available",
        "- no candidate has closed_body_success, boolean_subtract_success, mesh_precheck_success, spatial_depth_variation, and is_constant_depth=False",
        "- Bx/By/Bz export cannot be kept available for a later forward smoke",
        "- data/NPZ/.mph/raw CSV boundary cannot be preserved",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    passes = sum(1 for row in rows if bool(row["stage_a_validation_pass"]))
    lines = [
        "20.68 smooth builder test plan summary",
        "",
        f"sample_count: {len(rows)}",
        f"primary_sample: {PRIMARY_SAMPLE_ID}",
        f"stage_a_validation_pass_count: {passes}",
        "candidate_order: lofted_contour_solid -> stacked_workplane_contour_loft -> interpolated_surface_solid -> imported_closed_mesh_solid -> high_layer_control_24_or_32",
        "stage_c_gate: variable_depth_pass or near_smooth_pass only",
        "high_layer_control_policy: no Stage C forward; human confirmation required before any pilot expansion",
        "exact_piao_rbc: False",
        f"rbc_formula_status: {smoke.RBC_FORMULA_STATUS}",
        "",
        "Selected samples:",
    ]
    for row in rows:
        lines.append(
            f"- {row['sample_id']}: L={row['L_m']}, W={row['W_m']}, D={row['D_m']}, "
            f"forward_eligible={row['forward_eligible']}, validation_pass={row['stage_a_validation_pass']}"
        )
    lines.extend(
        [
            "",
            "Important boundary:",
            "- projected_mask_2d is only a 2D comparator label.",
            "- The 3D label remains reconstructable from rbc_params, profile_pose, profile_depth_grid_m/profile_depth_map_xy_m, and geometry_params_json.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def determine_builder_status(rows: list[dict[str, str]]) -> str:
    statuses = {row.get("geometry_status") or row.get("status") for row in rows}
    if "variable_depth_pass" in statuses:
        return "variable_depth_pass"
    if "near_smooth_pass" in statuses:
        return "near_smooth_pass"
    if "high_layer_control_pass" in statuses:
        return "high_layer_control_pass"
    return "failed"


def route_rows(status: str) -> list[dict[str, Any]]:
    return [
        {
            "decision_option": "A_variable_depth_pass",
            "selected": status == "variable_depth_pass",
            "condition": "smooth continuous variable-depth closed solid passes geometry gate and forward/schema validation",
            "observed": status,
            "next_step": "generate true 3D RBC pilot",
        },
        {
            "decision_option": "B_near_smooth_pass",
            "selected": status == "near_smooth_pass",
            "condition": "near-smooth loft/import/interpolated closed solid passes; approximation is explicit",
            "observed": status,
            "next_step": "consider near-smooth approximation pilot or continue smooth builder improvement",
        },
        {
            "decision_option": "C_high_layer_control_pass",
            "selected": status == "high_layer_control_pass",
            "condition": "only high-layer control passes; no smooth/near-smooth candidate qualifies",
            "observed": status,
            "next_step": "human confirmation required before accepting high-layer approximation as pilot definition",
        },
        {
            "decision_option": "D_failed",
            "selected": status == "failed",
            "condition": "no variable-depth/non-constant-depth candidate passes the geometry-only gate",
            "observed": status,
            "next_step": "continue COMSOL geometry builder work; do not expand samples or train",
        },
    ]


def write_route(args: argparse.Namespace) -> None:
    rows = read_csv(args.builder_inventory)
    status = determine_builder_status(rows)
    selected_rows = [row for row in rows if (row.get("geometry_status") or row.get("status")) == status]
    method = selected_rows[0].get("geometry_method", selected_rows[0].get("geometry_method_used", "none")) if selected_rows else "none"
    check_no_overwrite([args.route_summary, args.route_matrix], args.overwrite)
    write_csv(args.route_matrix, route_rows(status), ROUTE_FIELDS)
    lines = [
        "20.68 smooth / near-smooth variable-depth builder route decision summary",
        "",
        f"selected_status: {status}",
        f"selected_geometry_method: {method}",
        f"smooth_variable_depth_pass: {status == 'variable_depth_pass'}",
        f"near_smooth_pass: {status == 'near_smooth_pass'}",
        f"high_layer_control_pass: {status == 'high_layer_control_pass'}",
        f"failed: {status == 'failed'}",
        "constant_depth_success: False",
        "",
        "Decision:",
    ]
    if status == "variable_depth_pass":
        lines.append("- Smooth variable-depth builder is technically ready for true 3D RBC pilot generation.")
    elif status == "near_smooth_pass":
        lines.append("- Near-smooth builder is technically usable only with explicit approximation labeling.")
    elif status == "high_layer_control_pass":
        lines.append("- Smooth builder remains incomplete. The only passing result is high-layer control; human confirmation is required before accepting high-layer approximation as pilot definition.")
    else:
        lines.append("- Builder failed. Continue COMSOL geometry builder work; do not expand samples and do not train.")
    lines.extend(
        [
            "",
            "Boundary:",
            "- high_layer_control_pass is not smooth_variable_depth_pass.",
            "- high_layer_control_pass is not near_smooth_pass.",
            "- high_layer_control_pass is not exact Piao RBC geometry.",
            "- No 60-sample pilot is authorized by this stage unless variable_depth_pass or near_smooth_pass is achieved.",
        ]
    )
    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    args.route_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.route_only:
        write_route(args)
        return 0
    check_no_overwrite([args.plan_csv, args.summary, args.preflight_summary], args.overwrite)
    rows = build_rows()
    if not any(row["sample_id"] == PRIMARY_SAMPLE_ID and bool(row["stage_a_validation_pass"]) for row in rows):
        raise RuntimeError("primary medium_round plan row is missing or invalid")
    write_csv(args.plan_csv, rows, PLAN_FIELDS)
    write_summary(args.summary, rows)
    write_preflight(args.preflight_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
