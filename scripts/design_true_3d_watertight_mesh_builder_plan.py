#!/usr/bin/env python
"""Design the 20.69 watertight imported solid builder plan.

This script is planning/accounting only. It does not run COMSOL, does not
generate a mesh, and does not create any training data. The selected sample is
fixed to medium_round so the imported-solid route can be debugged without
expanding into a pilot.
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


DEFAULT_PLAN_CSV = ROOT / "results/metrics/true_3d_watertight_mesh_builder_plan.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_watertight_mesh_builder_plan_summary.txt"
DEFAULT_PREFLIGHT = ROOT / "results/summaries/true_3d_watertight_imported_solid_preflight_summary.txt"
TEMP_MESH_DIR = ROOT / "data/comsol_mfl/generated/temp_true_3d_mesh_import"
SELECTED_SAMPLE_ID = "medium_round"
MESH_RESOLUTION = [smoke.GRID_U_COUNT, smoke.GRID_V_COUNT]
MESH_FILENAME = f"{SELECTED_SAMPLE_ID}_watertight_depth_surface.stl"

PLAN_FIELDS = [
    "sample_id",
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
    "profile_pose_json",
    "rbc_params_json",
    "profile_depth_grid_shape_json",
    "profile_depth_grid_m_json",
    "profile_depth_map_xy_shape_json",
    "profile_depth_map_xy_m_json",
    "projected_mask_2d_shape_json",
    "projected_mask_2d_json",
    "projection_threshold_m",
    "mesh_units",
    "mesh_resolution_json",
    "mesh_source",
    "surface_continuity_assumption",
    "top_cap_plane",
    "depth_sign_convention",
    "profile_pose_to_comsol_json",
    "steel_surface_z_m",
    "steel_z_min_m",
    "steel_z_max_m",
    "steel_x_min_m",
    "steel_x_max_m",
    "steel_y_min_m",
    "steel_y_max_m",
    "defect_void_position_note",
    "temp_mesh_output_path",
    "temp_mesh_git_policy",
    "target_max_depth_m",
    "target_volume_proxy_m3",
    "target_footprint_area_m2",
    "target_footprint_area_px",
    "expected_watertight_checks_json",
    "stage_a_validation_pass",
    "geometry_params_json",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.69 watertight imported solid plan.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rounded_list(value: np.ndarray, decimals: int = 9) -> list[Any]:
    return np.round(np.asarray(value, dtype=np.float64), decimals=decimals).tolist()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def selected_sample() -> smoke.SmokeSample:
    for sample in smoke.sample_table():
        if sample.sample_id == SELECTED_SAMPLE_ID:
            return sample
    raise RuntimeError(f"required sample_id not found in 20.66 smoke table: {SELECTED_SAMPLE_ID}")


def build_plan_row() -> dict[str, Any]:
    sample = selected_sample()
    mask_x = np.linspace(smoke.MASK_X_START_M, smoke.MASK_X_STOP_M, smoke.MASK_WIDTH)
    mask_y = np.linspace(smoke.MASK_Y_START_M, smoke.MASK_Y_STOP_M, smoke.MASK_HEIGHT)
    pixel_area = float((mask_x[1] - mask_x[0]) * (mask_y[1] - mask_y[0]))
    _, _, depth_grid = smoke.build_profile_depth_grid(sample)
    depth_map = smoke.build_depth_map(sample, mask_x, mask_y)
    projection_threshold = max(1.0e-6, 0.01 * sample.D_m)
    projected_mask = (depth_map >= projection_threshold).astype(np.uint8)
    footprint_area = int(projected_mask.sum()) * pixel_area
    target_volume = float(np.sum(depth_map * projected_mask) * pixel_area)
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
    steel = {
        "x_min_m": -0.04,
        "x_max_m": 0.04,
        "y_min_m": -0.01,
        "y_max_m": 0.01,
        "z_min_m": -0.006,
        "z_max_m": 0.0,
        "surface_z_m": 0.0,
    }
    mesh_path = TEMP_MESH_DIR / MESH_FILENAME
    expected_checks = {
        "finite_vertices": True,
        "watertight": True,
        "edge_incidence_all_two": True,
        "nonmanifold_edges_count": 0,
        "zero_area_triangles_count": 0,
        "volume_positive": True,
        "max_depth_close_to_D": True,
        "bbox_inside_steel": True,
        "defect_intersects_top_surface": True,
        "defect_embedded_in_steel": True,
    }
    pose_to_comsol = {
        "mesh_units": "m",
        "x_comsol_m": "center_x_m + rotated local profile x",
        "y_comsol_m": "center_y_m + rotated local profile y",
        "z_comsol_m": "top cap at steel surface z=0; bottom surface z=-depth",
        "angle_rad": math.radians(sample.angle_deg),
        "surface_alignment": "top cap coincides with steel surface z=0 and defect void extends into steel negative z",
    }
    max_depth = float(depth_grid.max())
    stage_a_pass = (
        np.isfinite(depth_grid).all()
        and np.isfinite(depth_map).all()
        and int(projected_mask.sum()) > 0
        and abs(max_depth - sample.D_m) <= 0.03 * sample.D_m
        and str(mesh_path).lower().find(r"data\comsol_mfl\generated\temp_true_3d_mesh_import") >= 0
    )
    geometry_params = {
        "task": "20.69 watertight imported solid builder hardening",
        "sample_id": sample.sample_id,
        "profile_type": "rbc_style_symmetric_pit",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "rbc_formula_status": smoke.RBC_FORMULA_STATUS,
        "rbc_params": rbc_params,
        "profile_pose": profile_pose,
        "mesh_source": "triangulated_depth_grid",
        "mesh_units": "m",
        "top_cap_plane": "z=0",
        "depth_sign_convention": "bottom surface z=-depth",
        "profile_pose_to_comsol": pose_to_comsol,
        "steel_surface_z_m": 0.0,
        "steel_z_extent_m": [steel["z_min_m"], steel["z_max_m"]],
        "temp_mesh_output_path": str(mesh_path),
        "projected_mask_role": "2D comparator only; does not replace 3D depth/profile label",
        "known_sanity_probe_policy": "known cube/prism import status is reported separately and does not count as RBC imported solid success",
    }
    return {
        "sample_id": sample.sample_id,
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
        "profile_pose_json": json_dumps(profile_pose),
        "rbc_params_json": json_dumps(rbc_params),
        "profile_depth_grid_shape_json": json_dumps(list(depth_grid.shape)),
        "profile_depth_grid_m_json": json_dumps(rounded_list(depth_grid, decimals=9)),
        "profile_depth_map_xy_shape_json": json_dumps(list(depth_map.shape)),
        "profile_depth_map_xy_m_json": json_dumps(rounded_list(depth_map, decimals=9)),
        "projected_mask_2d_shape_json": json_dumps(list(projected_mask.shape)),
        "projected_mask_2d_json": json_dumps(projected_mask.astype(int).tolist()),
        "projection_threshold_m": projection_threshold,
        "mesh_units": "m",
        "mesh_resolution_json": json_dumps(MESH_RESOLUTION),
        "mesh_source": "triangulated_depth_grid",
        "surface_continuity_assumption": "piecewise-linear depth surface from positive-depth RBC-style grid cells; not stepped layers",
        "top_cap_plane": "z=0",
        "depth_sign_convention": "bottom surface z=-depth",
        "profile_pose_to_comsol_json": json_dumps(pose_to_comsol),
        "steel_surface_z_m": steel["surface_z_m"],
        "steel_z_min_m": steel["z_min_m"],
        "steel_z_max_m": steel["z_max_m"],
        "steel_x_min_m": steel["x_min_m"],
        "steel_x_max_m": steel["x_max_m"],
        "steel_y_min_m": steel["y_min_m"],
        "steel_y_max_m": steel["y_max_m"],
        "defect_void_position_note": "top cap intersects steel surface z=0; bottom lies inside steel at negative z",
        "temp_mesh_output_path": str(mesh_path),
        "temp_mesh_git_policy": "generated temp STL under data/ is forbidden from commit",
        "target_max_depth_m": sample.D_m,
        "target_volume_proxy_m3": target_volume,
        "target_footprint_area_m2": footprint_area,
        "target_footprint_area_px": int(projected_mask.sum()),
        "expected_watertight_checks_json": json_dumps(expected_checks),
        "stage_a_validation_pass": bool(stage_a_pass),
        "geometry_params_json": json_dumps(geometry_params),
        "notes": "20.69 uses only medium_round; high_layer_control_24 is historical reference only",
    }


def write_preflight(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "20.69 watertight imported solid builder preflight summary",
        "",
        "Decision: GO for a bounded imported watertight mesh solid hardening pass.",
        "",
        "Agent A Method/Literature:",
        "- Imported watertight solid hardening is consistent with the true 3D / Piao-style route after 20.66-20.68.",
        "- This is geometry-builder work only, not a full Piao 2019 reproduction and not a training stage.",
        "",
        "Agent B Mesh/Geometry Construction:",
        "- Use pure NumPy because the available environments do not reliably provide trimesh, meshio, numpy-stl, pyvista, or open3d.",
        "- The mesh must be a closed positive-depth volume with a top cap at z=0, bottom surface z=-depth(x,y), side walls, consistent edge incidence, positive volume, and no zero-area triangles.",
        "",
        "Agent C COMSOL Import Capability:",
        "- COMSOL Import exists, but 20.68 showed import alone is not enough: imported_closed_mesh_solid failed because Boolean subtract produced an empty steel domain selection.",
        "- 20.69 must separate import_success, repair_success, form_solid_success, imported_domain_count, boolean_subtract_success, steel_notched_domain_count, and mesh_precheck_success.",
        "",
        "Agent D Experiment Design:",
        "- Use one primary RBC sample: medium_round.",
        "- Run a known cube/prism import sanity probe separately, then the RBC mesh probe. The known probe can validate the import pipeline only; it never counts as RBC success.",
        "- Run one-sample Bx/By/Bz forward only if the RBC imported solid gate passes.",
        "",
        "Agent E Safety/Git/Data Boundary:",
        "- Correct target repositories are C:\\Users\\19166\\Desktop\\PINN_project and C:\\Users\\19166\\Desktop\\COMSOL_Multiphysics_MCP.",
        "- Current known PINN dirty items to exclude: MODEL_STRUCTURE_PLAN.md deletion and scripts/visualize_current_baseline.py.",
        "- Current known COMSOL dirty items to exclude: src/tools/physics.py, src/tools/results.py, src/tools/study.py, configs/, knowledge_base/chroma.sqlite3, scripts/generate_mfl_rectangular_sweep.py.",
        "- Do not commit data/, NPZ, .mph, raw CSV, checkpoint, preview PNG, notes, temp STL/temp mesh, baseline docs, MODEL_STRUCTURE_PLAN.md deletion, or unrelated dirty items.",
        "",
        "Agent F Implementation Feasibility:",
        "- Execute as pure-Python mesh validation -> COMSOL import/Boolean/mesh precheck -> optional forward -> optional NPZ validation -> route decision.",
        "- If RBC mesh is watertight but not placed inside/intersecting the steel surface correctly, stop before COMSOL Boolean.",
        "",
        "Hard blockers:",
        "- watertight closed mesh cannot be generated or validated;",
        "- mesh is not embedded in steel volume or does not intersect z=0 steel surface;",
        "- COMSOL cannot form/import a solid domain from the STL;",
        "- Boolean subtract or mesh precheck fails for the RBC mesh;",
        "- only high-layer or constant-depth routes remain;",
        "- forbidden generated artifacts cannot be kept out of git.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "20.69 watertight mesh builder plan summary",
        "",
        f"sample_count: 1",
        f"primary_sample: {row['sample_id']}",
        f"L_m: {row['L_m']}",
        f"W_m: {row['W_m']}",
        f"D_m: {row['D_m']}",
        f"exact_piao_rbc: {row['exact_piao_rbc']}",
        f"rbc_formula_status: {row['rbc_formula_status']}",
        f"mesh_units: {row['mesh_units']}",
        f"top_cap_plane: {row['top_cap_plane']}",
        f"depth_sign_convention: {row['depth_sign_convention']}",
        f"temp_mesh_output_path: {row['temp_mesh_output_path']}",
        f"target_footprint_area_px: {row['target_footprint_area_px']}",
        f"target_footprint_area_m2: {row['target_footprint_area_m2']}",
        f"target_volume_proxy_m3: {row['target_volume_proxy_m3']}",
        f"stage_a_validation_pass: {row['stage_a_validation_pass']}",
        "",
        "Coordinate convention:",
        "- mesh_units=m",
        "- steel surface is z=0; steel spans z=[-0.006, 0.0] m",
        "- defect void top cap lies on z=0 and bottom surface is z=-depth, so the void intersects the steel surface and extends into steel.",
        "- profile_pose maps RBC local x/y into COMSOL x/y using center_x_m, center_y_m, and angle_rad.",
        "",
        "Gate:",
        "- This plan only authorizes pure-Python watertight mesh generation for medium_round.",
        "- Known cube/prism sanity probe and RBC imported solid probe must be reported separately.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    check_no_overwrite([args.plan_csv, args.summary, args.preflight_summary], args.overwrite)
    row = build_plan_row()
    if not row["stage_a_validation_pass"]:
        raise RuntimeError("20.69 watertight mesh plan validation failed")
    write_preflight(args.preflight_summary)
    write_csv(args.plan_csv, [row], PLAN_FIELDS)
    write_summary(args.summary, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
