#!/usr/bin/env python
"""Design the 20.71 true-3D RBC-style imported-mesh pilot plan.

This is a planning/label-generation script only. It does not call COMSOL and
does not write generated data packs. The profile formula is the same
RBC-style/Piao-inspired engineering approximation used by the true-3D smoke
steps; it is not an exact Piao 2019 RBC reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import design_true_3d_rbc_smoke_plan as smoke_plan  # noqa: E402


DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc"
ROUTE = "true_3d_piao_style"
GEOMETRY_METHOD = "imported_watertight_mesh_solid"
EXACT_PIAO_RBC = False
RBC_STYLE_APPROXIMATION = True
RBC_FORMULA_STATUS = "RBC-style / Piao-inspired engineering approximation; exact Piao RBC formula not implemented"

DEFAULT_PLAN_CSV = ROOT / "results/metrics/true_3d_rbc_pilot_pack_plan.csv"
DEFAULT_PROFILE_VALIDATION_CSV = ROOT / "results/metrics/true_3d_rbc_pilot_pack_profile_validation.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_pack_plan_summary.txt"
DEFAULT_PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_pack_preflight_summary.txt"
DEFAULT_TEMP_MESH_DIR = ROOT / "data/comsol_mfl/generated/temp_true_3d_rbc_pilot_meshes_v1"

MASK_WIDTH = smoke_plan.MASK_WIDTH
MASK_HEIGHT = smoke_plan.MASK_HEIGHT
MASK_X_START_M = smoke_plan.MASK_X_START_M
MASK_X_STOP_M = smoke_plan.MASK_X_STOP_M
MASK_Y_START_M = smoke_plan.MASK_Y_START_M
MASK_Y_STOP_M = smoke_plan.MASK_Y_STOP_M

PLAN_FIELDS = [
    "dataset_id",
    "schema_version",
    "route",
    "sample_id",
    "split",
    "split_tag",
    "depth_bin",
    "size_bin",
    "aspect_bin",
    "curvature_bin",
    "curvature_template",
    "profile_type",
    "exact_piao_rbc",
    "rbc_style_approximation",
    "rbc_formula_status",
    "geometry_method",
    "generated_geometry_type",
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
    "footprint_connected_components",
    "target_footprint_area_m2",
    "target_volume_proxy_m3",
    "mesh_resolution",
    "mesh_units",
    "mesh_source",
    "surface_continuity_assumption",
    "top_cap_plane",
    "depth_sign_convention",
    "profile_pose_to_comsol_json",
    "steel_surface_z_m",
    "steel_x_min_m",
    "steel_x_max_m",
    "steel_y_min_m",
    "steel_y_max_m",
    "steel_z_min_m",
    "steel_z_max_m",
    "temp_mesh_output_path",
    "geometry_params_json",
    "allowed_use",
    "forbidden_use",
    "notes",
]

VALIDATION_FIELDS = [
    "sample_id",
    "split",
    "depth_bin",
    "size_bin",
    "aspect_bin",
    "curvature_template",
    "depth_grid_finite",
    "depth_map_finite",
    "depth_nonnegative",
    "max_depth_m",
    "target_D_m",
    "max_depth_abs_error_m",
    "max_depth_rel_error",
    "projected_mask_area_px",
    "projected_mask_area_m2",
    "footprint_nonempty",
    "footprint_connected_components",
    "profile_depth_grid_serializable",
    "profile_depth_map_serializable",
    "projected_mask_serializable",
    "geometry_params_serializable",
    "validation_pass",
    "notes",
]


@dataclass(frozen=True)
class PilotSpec:
    depth_bin: str
    size_bin: str
    aspect_bin: str
    curvature_template: str
    split: str
    L_m: float
    W_m: float
    D_m: float
    wLD: float
    wWD: float
    wLW: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.71 true-3D RBC pilot plan.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--profile-validation-csv", type=Path, default=DEFAULT_PROFILE_VALIDATION_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT_SUMMARY)
    parser.add_argument("--temp-mesh-dir", type=Path, default=DEFAULT_TEMP_MESH_DIR)
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


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def rounded_list(array: np.ndarray, decimals: int = 9) -> list[Any]:
    return np.round(np.asarray(array, dtype=np.float64), decimals=decimals).tolist()


def sample_from_spec(spec: PilotSpec, index: int) -> smoke_plan.SmokeSample:
    sample_id = f"rbc_pilot_{index:03d}_{spec.depth_bin}_{spec.size_bin}_{spec.curvature_template}"
    return smoke_plan.SmokeSample(
        sample_id=sample_id,
        split_tag=spec.split,
        L_m=spec.L_m,
        W_m=spec.W_m,
        D_m=spec.D_m,
        wLD=spec.wLD,
        wWD=spec.wWD,
        wLW=spec.wLW,
        center_x_m=0.0,
        center_y_m=0.0,
        angle_deg=0.0,
        notes=f"{spec.depth_bin}/{spec.size_bin}/{spec.curvature_template}; imported watertight pilot row",
    )


def split_for(template_index: int, depth_index: int, size_index: int) -> str:
    # Each curvature template contributes 8 train / 2 val / 2 test while the
    # global depth distribution stays near-balanced.
    val_pairs = {
        0: {(0, 0), (1, 0)},
        1: {(1, 1), (2, 1)},
        2: {(2, 2), (0, 3)},
    }
    test_pairs = {
        0: {(2, 0), (0, 1)},
        1: {(0, 2), (1, 2)},
        2: {(1, 3), (2, 3)},
    }
    pattern = template_index % 3
    key = (depth_index, size_index)
    if key in val_pairs[pattern]:
        return "val"
    if key in test_pairs[pattern]:
        return "test"
    return "train"


def pilot_specs() -> list[PilotSpec]:
    depth_bins = [
        ("shallow", [0.0010, 0.0013, 0.0016, 0.0018]),
        ("medium", [0.0022, 0.0026, 0.0031, 0.0035]),
        ("deep", [0.0045, 0.0050, 0.0055, 0.0060]),
    ]
    size_cells = [
        ("small_compact", "compact", [0.010, 0.012, 0.014], [0.006, 0.007, 0.008]),
        ("medium_balanced", "balanced", [0.016, 0.019, 0.022], [0.009, 0.0105, 0.012]),
        ("large_wide", "wide", [0.024, 0.027, 0.030], [0.015, 0.0175, 0.020]),
        ("elongated", "narrow", [0.018, 0.024, 0.030], [0.006, 0.008, 0.010]),
    ]
    curvatures = [
        ("sharp", 0.55, 0.60, 0.55),
        ("round", 0.707, 0.707, 0.707),
        ("boxy", 1.10, 1.10, 1.10),
        ("LD_dominant", 1.20, 0.85, 1.00),
        ("WD_dominant", 0.85, 1.20, 1.00),
    ]
    specs: list[PilotSpec] = []
    for template_index, (curv_name, wld, wwd, wlw) in enumerate(curvatures):
        for depth_index, (depth_name, depths_by_size) in enumerate(depth_bins):
            for size_index, (size_name, aspect_name, lengths_by_depth, widths_by_depth) in enumerate(size_cells):
                specs.append(
                    PilotSpec(
                        depth_bin=depth_name,
                        size_bin=size_name,
                        aspect_bin=aspect_name,
                        curvature_template=curv_name,
                        split=split_for(template_index, depth_index, size_index),
                        L_m=lengths_by_depth[depth_index],
                        W_m=widths_by_depth[depth_index],
                        D_m=depths_by_size[size_index],
                        wLD=wld,
                        wWD=wwd,
                        wLW=wlw,
                    )
                )
    return specs


def build_rows(temp_mesh_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mask_x = np.linspace(MASK_X_START_M, MASK_X_STOP_M, MASK_WIDTH)
    mask_y = np.linspace(MASK_Y_START_M, MASK_Y_STOP_M, MASK_HEIGHT)
    pixel_area = float((mask_x[1] - mask_x[0]) * (mask_y[1] - mask_y[0]))
    plan_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for index, spec in enumerate(pilot_specs(), start=1):
        sample = sample_from_spec(spec, index)
        _, _, depth_grid = smoke_plan.build_profile_depth_grid(sample)
        depth_map = smoke_plan.build_depth_map(sample, mask_x, mask_y)
        projection_threshold = max(1.0e-6, 0.01 * sample.D_m)
        mask = (depth_map >= projection_threshold).astype(np.uint8)
        connected_components = smoke_plan.connected_component_count(mask)
        max_depth = float(depth_grid.max())
        rel_error = abs(max_depth - sample.D_m) / max(sample.D_m, 1.0e-12)
        footprint_area = int(mask.sum()) * pixel_area
        volume_proxy = float(np.sum(depth_map) * pixel_area)
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
        profile_pose_to_comsol = {
            "mesh_units": "m",
            "comsol_coordinates": "x/y on steel surface plane; top cap z=0; bottom surface z=-depth",
            "center_x_m": sample.center_x_m,
            "center_y_m": sample.center_y_m,
            "angle_rad": 0.0,
            "steel_surface_z_m": 0.0,
        }
        geometry_params = {
            "dataset_id": DATASET_ID,
            "sample_id": sample.sample_id,
            "profile_type": "rbc_style_symmetric_pit",
            "geometry_method": GEOMETRY_METHOD,
            "exact_piao_rbc": EXACT_PIAO_RBC,
            "rbc_style_approximation": RBC_STYLE_APPROXIMATION,
            "rbc_formula_status": RBC_FORMULA_STATUS,
            "rbc_params": rbc_params,
            "profile_pose": profile_pose,
            "depth_bin": spec.depth_bin,
            "size_bin": spec.size_bin,
            "aspect_bin": spec.aspect_bin,
            "curvature_template": spec.curvature_template,
            "projection_threshold_m": projection_threshold,
            "target_footprint_area_m2": footprint_area,
            "target_volume_proxy_m3": volume_proxy,
            "mesh_source": "triangulated_depth_grid",
            "projected_mask_role": "2D comparator only; 3D label uses rbc_params and depth grid/map",
        }
        validation_pass = (
            bool(np.isfinite(depth_grid).all())
            and bool(np.isfinite(depth_map).all())
            and bool((depth_grid >= -1.0e-12).all())
            and bool((depth_map >= -1.0e-12).all())
            and rel_error <= 0.03
            and int(mask.sum()) > 0
            and connected_components == 1
        )
        temp_mesh_output_path = temp_mesh_dir / f"{sample.sample_id}.stl"
        row = {
            "dataset_id": DATASET_ID,
            "schema_version": SCHEMA_VERSION,
            "route": ROUTE,
            "sample_id": sample.sample_id,
            "split": spec.split,
            "split_tag": spec.split,
            "depth_bin": spec.depth_bin,
            "size_bin": spec.size_bin,
            "aspect_bin": spec.aspect_bin,
            "curvature_bin": spec.curvature_template,
            "curvature_template": spec.curvature_template,
            "profile_type": "rbc_style_symmetric_pit",
            "exact_piao_rbc": str(EXACT_PIAO_RBC),
            "rbc_style_approximation": str(RBC_STYLE_APPROXIMATION),
            "rbc_formula_status": RBC_FORMULA_STATUS,
            "geometry_method": GEOMETRY_METHOD,
            "generated_geometry_type": "rbc_style_imported_watertight_mesh_candidate",
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
            "projected_mask_2d_shape_json": json_dumps(list(mask.shape)),
            "projected_mask_2d_json": json_dumps(mask.astype(int).tolist()),
            "projection_threshold_m": projection_threshold,
            "footprint_connected_components": connected_components,
            "target_footprint_area_m2": footprint_area,
            "target_volume_proxy_m3": volume_proxy,
            "mesh_resolution": "profile_depth_grid_33x17",
            "mesh_units": "m",
            "mesh_source": "triangulated_depth_grid",
            "surface_continuity_assumption": "piecewise-linear continuous depth surface from RBC-style depth grid; not stepped_layers",
            "top_cap_plane": "z=0",
            "depth_sign_convention": "bottom surface z=-depth",
            "profile_pose_to_comsol_json": json_dumps(profile_pose_to_comsol),
            "steel_surface_z_m": 0.0,
            "steel_x_min_m": -0.05,
            "steel_x_max_m": 0.05,
            "steel_y_min_m": -0.02,
            "steel_y_max_m": 0.02,
            "steel_z_min_m": -0.01,
            "steel_z_max_m": 0.0,
            "temp_mesh_output_path": str(temp_mesh_output_path),
            "geometry_params_json": json_dumps(geometry_params),
            "allowed_use": "schema_validation, explicit_pilot_training_gate",
            "forbidden_use": "automatic_mainline_training, baseline_update, current_baseline_replacement",
            "notes": sample.notes,
        }
        plan_rows.append(row)
        validation_rows.append(
            {
                "sample_id": sample.sample_id,
                "split": spec.split,
                "depth_bin": spec.depth_bin,
                "size_bin": spec.size_bin,
                "aspect_bin": spec.aspect_bin,
                "curvature_template": spec.curvature_template,
                "depth_grid_finite": bool(np.isfinite(depth_grid).all()),
                "depth_map_finite": bool(np.isfinite(depth_map).all()),
                "depth_nonnegative": bool((depth_grid >= -1.0e-12).all() and (depth_map >= -1.0e-12).all()),
                "max_depth_m": max_depth,
                "target_D_m": sample.D_m,
                "max_depth_abs_error_m": abs(max_depth - sample.D_m),
                "max_depth_rel_error": rel_error,
                "projected_mask_area_px": int(mask.sum()),
                "projected_mask_area_m2": footprint_area,
                "footprint_nonempty": int(mask.sum()) > 0,
                "footprint_connected_components": connected_components,
                "profile_depth_grid_serializable": True,
                "profile_depth_map_serializable": True,
                "projected_mask_serializable": True,
                "geometry_params_serializable": True,
                "validation_pass": validation_pass,
                "notes": "profile validation for imported watertight pilot candidate",
            }
        )
    return plan_rows, validation_rows


def write_preflight_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "20.71 true 3D RBC pilot pack preflight summary",
        "",
        "Decision: GO for bounded smooth/mesh-based RBC-style imported watertight pilot generation.",
        "",
        "Subagent core conclusions:",
        "- Method/literature: GO only as RBC-style / Piao-inspired engineering pilot; exact_piao_rbc=False.",
        "- COMSOL generation: conditional GO; 20.70 one-sample chain is reusable but existing scripts are one-row only.",
        "- Data design: GO for N=60 with train/val/test=40/10/10, angle_rad=0, Bx/By/Bz @ sensor_z_m=0.008.",
        "- Registry/manifest: GO; create COMSOL_DATA_REGISTRY.md and tracked manifest, forbid latest/newest auto-discovery.",
        "- Safety/git: conditional GO with exact whitelist staging; do not submit data, NPZ, temp STL, .mph, raw CSV, notes, previews, checkpoints, or baseline docs.",
        "- Implementation feasibility: GO after adding batch plan, mesh, COMSOL generation, validation, registry, and partial-pack gates.",
        "",
        "20.70 support evidence:",
        "- imported watertight mesh solid full-source Jscale=1.0 forward smoke passed for medium_round.",
        "- selected_solver_protocol=default, mesh_auto_size=5, material_fix_applied=True.",
        "- Bx/By/Bz export, delta_b check, and one-sample schema validation passed.",
        "",
        "Hard blockers:",
        "- cannot batch-generate watertight meshes;",
        "- cannot reuse imported watertight solver protocol without high-layer fallback;",
        "- Bx/By/Bz export or delta_b check broadly fails;",
        "- registry/manifest cannot be created;",
        "- data/NPZ/temp STL/.mph/raw CSV cannot be kept out of git;",
        "- pilot/baseline distinction cannot be recorded.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, plan_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = Counter(row["split"] for row in plan_rows)
    depth_counts = Counter(row["depth_bin"] for row in plan_rows)
    curvature_counts = Counter(row["curvature_template"] for row in plan_rows)
    pass_count = sum(1 for row in validation_rows if row["validation_pass"])
    lengths = [float(row["L_m"]) for row in plan_rows]
    widths = [float(row["W_m"]) for row in plan_rows]
    depths = [float(row["D_m"]) for row in plan_rows]
    lines = [
        "20.71 true 3D RBC pilot pack plan summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"route: {ROUTE}",
        f"geometry_method: {GEOMETRY_METHOD}",
        f"sample_count: {len(plan_rows)}",
        f"profile_validation_pass_count: {pass_count}",
        f"split_counts: {dict(split_counts)}",
        f"depth_bin_counts: {dict(depth_counts)}",
        f"curvature_template_counts: {dict(curvature_counts)}",
        f"L_m_range: {min(lengths):.6f} to {max(lengths):.6f}",
        f"W_m_range: {min(widths):.6f} to {max(widths):.6f}",
        f"D_m_range: {min(depths):.6f} to {max(depths):.6f}",
        "angle_rad: 0.0 for all rows",
        "sensor_z_m: 0.008",
        "scan_line_y_m: [-0.001, 0.0, 0.001]",
        "axis_names: [Bx, By, Bz]",
        "axis_expressions: [mf.Bx, mf.By, mf.Bz]",
        "exact_piao_rbc: False",
        "rbc_style_approximation: True",
        "allowed_use: schema_validation, explicit_pilot_training_gate",
        "forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement",
        "",
        "Gate result:",
        "PASS" if len(plan_rows) == 60 and pass_count == 60 and dict(split_counts) == {"train": 40, "val": 10, "test": 10} else "FAIL",
        "",
        "Boundary: this is a pilot pack plan, not a baseline and not an exact Piao 2019 reproduction.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    check_no_overwrite([args.plan_csv, args.profile_validation_csv, args.summary, args.preflight_summary], args.overwrite)
    plan_rows, validation_rows = build_rows(args.temp_mesh_dir)
    write_preflight_summary(args.preflight_summary)
    write_csv(args.plan_csv, plan_rows, PLAN_FIELDS)
    write_csv(args.profile_validation_csv, validation_rows, VALIDATION_FIELDS)
    write_summary(args.summary, plan_rows, validation_rows)
    split_counts = Counter(row["split"] for row in plan_rows)
    failed = [row["sample_id"] for row in validation_rows if not row["validation_pass"]]
    if len(plan_rows) != 60 or dict(split_counts) != {"train": 40, "val": 10, "test": 10}:
        raise RuntimeError(f"pilot plan gate failed: rows={len(plan_rows)}, split_counts={dict(split_counts)}")
    if failed:
        raise RuntimeError(f"profile validation failed for {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
