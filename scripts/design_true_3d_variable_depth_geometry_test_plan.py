#!/usr/bin/env python
"""Design the 20.67 variable-depth true-3D geometry feasibility plan.

This script is pure Python. It reuses the 20.66 RBC-style engineering
approximation and prepares only a small geometry feasibility plan. It does not
call COMSOL, does not generate data packs, and does not claim exact Piao RBC.
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import design_true_3d_rbc_smoke_plan as smoke  # noqa: E402


DEFAULT_PLAN_CSV = ROOT / "results/metrics/true_3d_variable_depth_geometry_test_plan.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_variable_depth_geometry_test_plan_summary.txt"
DEFAULT_PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_variable_depth_geometry_preflight_summary.txt"

SELECTED_SAMPLE_IDS = ("medium_round", "deep_round", "medium_boxy")
MINIMUM_SAMPLE_ID = "medium_round"
HIGH_LAYER_12_FRACTIONS = tuple(float(v) for v in np.linspace(0.08, 0.96, 12))
HIGH_LAYER_16_FRACTIONS = tuple(float(v) for v in np.linspace(0.06, 0.98, 16))

PLAN_FIELDS = [
    "sample_id",
    "selection_priority",
    "split_tag",
    "profile_type",
    "exact_piao_rbc",
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
    "recommended_method_sequence_json",
    "smooth_probe_policy",
    "high_layer_fallback_depth_levels",
    "baseline_20_66_depth_levels",
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
    "depth_levels_12_m_json",
    "depth_level_polygons_12_json",
    "depth_levels_16_m_json",
    "depth_level_polygons_16_json",
    "geometry_params_json",
    "geometry_validation_metadata_json",
    "stage_a_validation_pass",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.67 variable-depth geometry test plan.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT_SUMMARY)
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
        raise RuntimeError(f"20.66 sample ids missing from smoke plan generator: {missing}")

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
        polygons_12 = polygons_for_fractions(sample, HIGH_LAYER_12_FRACTIONS)
        polygons_16 = polygons_for_fractions(sample, HIGH_LAYER_16_FRACTIONS)
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
        geometry_params = {
            "sample_id": sample.sample_id,
            "task": "20.67 smooth / near-smooth variable-depth geometry feasibility",
            "profile_type": "rbc_style_symmetric_pit",
            "exact_piao_rbc": False,
            "rbc_formula_status": smoke.RBC_FORMULA_STATUS,
            "rbc_params": rbc_params,
            "profile_pose": profile_pose,
            "projection_threshold_m": projection_threshold,
            "candidate_geometry_methods": [
                "smooth_surface_probe",
                "lofted_contours_probe",
                "imported_closed_surface_probe",
                "high_layer_approx_12",
                "high_layer_approx_16",
            ],
            "recommended_method_sequence": [
                "limited_smooth_closed_surface_probe",
                "high_layer_approx_12",
                "high_layer_approx_16_if_needed",
            ],
            "baseline_20_66_depth_levels": 5,
            "high_layer_depth_levels": [12, 16],
            "depth_levels_12_m": [row["depth_m"] for row in polygons_12],
            "depth_level_polygons_12": polygons_12,
            "depth_levels_16_m": [row["depth_m"] for row in polygons_16],
            "depth_level_polygons_16": polygons_16,
            "smooth_variable_depth_solid_verified": False,
            "near_smooth_pass": False,
            "high_layer_pass": False,
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
            "polygons_12_count": len(polygons_12),
            "polygons_16_count": len(polygons_16),
            "polygons_12_min_area_m2": min((float(row["area_m2"]) for row in polygons_12), default=0.0),
            "polygons_16_min_area_m2": min((float(row["area_m2"]) for row in polygons_16), default=0.0),
        }
        stage_a_pass = (
            validation_metadata["depth_grid_finite"]
            and validation_metadata["depth_map_finite"]
            and validation_metadata["depth_nonnegative"]
            and validation_metadata["projected_mask_nonempty"]
            and validation_metadata["projected_mask_connected_components"] == 1
            and validation_metadata["max_depth_rel_error"] <= 0.03
            and len(polygons_12) > 5
            and len(polygons_16) > 5
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "selection_priority": priority,
                "split_tag": sample.split_tag,
                "profile_type": "rbc_style_symmetric_pit",
                "exact_piao_rbc": False,
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
                "recommended_method_sequence_json": json_dumps(geometry_params["recommended_method_sequence"]),
                "smooth_probe_policy": "limited_probe_then_record_failure_and_fallback; no unbounded smooth-route debugging",
                "high_layer_fallback_depth_levels": "12_then_16_if_needed",
                "baseline_20_66_depth_levels": 5,
                "profile_pose_json": json_dumps(profile_pose),
                "rbc_params_json": json_dumps(rbc_params),
                "profile_depth_grid_shape_json": json_dumps(list(depth_grid.shape)),
                "profile_depth_grid_m_json": json_dumps(rounded_list(depth_grid)),
                "profile_depth_map_xy_shape_json": json_dumps(list(depth_map.shape)),
                "profile_depth_map_xy_m_json": json_dumps(rounded_list(depth_map)),
                "projected_mask_2d_shape_json": json_dumps(list(projected_mask.shape)),
                "projected_mask_2d_json": json_dumps(projected_mask.astype(int).tolist()),
                "projection_threshold_m": projection_threshold,
                "expected_depth_max_m": max_depth,
                "depth_max_abs_error_m": depth_error,
                "footprint_area_m2": int(projected_mask.sum()) * pixel_area,
                "footprint_area_px": int(projected_mask.sum()),
                "depth_levels_12_m_json": json_dumps([row["depth_m"] for row in polygons_12]),
                "depth_level_polygons_12_json": json_dumps(polygons_12),
                "depth_levels_16_m_json": json_dumps([row["depth_m"] for row in polygons_16]),
                "depth_level_polygons_16_json": json_dumps(polygons_16),
                "geometry_params_json": json_dumps(geometry_params),
                "geometry_validation_metadata_json": json_dumps(validation_metadata),
                "stage_a_validation_pass": stage_a_pass,
                "notes": (
                    "Minimum forward sample" if sample.sample_id == MINIMUM_SAMPLE_ID else "Optional only after medium_round passes"
                ),
            }
        )
    return rows


def write_preflight_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "20.67 smooth / near-smooth variable-depth true 3D geometry preflight summary",
        "",
        "Decision: proceed with 20.67 as bounded geometry feasibility only.",
        "",
        "Subagent core conclusions:",
        "- Method/literature: 20.67 is the necessary next check after 20.66 because 20.66 passed only at stepped-depth smoke level. This is not a full Piao 2019 reproduction.",
        "- COMSOL geometry capability: current MFL scripts support real 3D volume solve, Boolean Difference, WorkPlane Polygon Extrude, no-defect/defect pair solves, and mf.Bx/mf.By/mf.Bz export. No verified smooth RBC variable-depth solid builder exists yet.",
        "- Geometry implementation design: run only a limited smooth/loft/imported closed-surface probe. If it cannot form a closed defect body, Boolean subtract, or mesh precheck, record the reason and fall back to high-layer nested contours.",
        "- Experiment design: target 3 samples from 20.66 (medium_round, deep_round, medium_boxy), minimum 1 sample (medium_round). Geometry-only must pass before any forward solve.",
        "- Safety/git: do not submit data, NPZ, .mph, raw CSV, checkpoints, preview PNG, notes, baseline docs, MODEL_STRUCTURE_PLAN.md deletion, scripts/visualize_current_baseline.py, or existing COMSOL dirty items.",
        "- Implementation feasibility: reuse 20.66 RBC depth labels and Bx/By/Bz forward infrastructure; distinguish 12/16-layer high-layer fallback from the 20.66 5-layer stepped approximation.",
        "",
        "Required status vocabulary:",
        "- variable_depth_pass: smooth continuous variable-depth closed solid succeeds and one-sample forward/schema validation passes.",
        "- near_smooth_pass: loft/import/interpolated near-smooth approximation succeeds but is not exact smooth.",
        "- high_layer_pass: 12/16-layer high-layer nested contour approximation succeeds; this is not smooth and not exact Piao RBC.",
        "- failed: only constant-depth, no variable-depth evidence, no Bx/By/Bz, delta check failure, or schema failure.",
        "",
        "Hard blockers:",
        "- no method can construct variable-depth, near-smooth, or high-layer 3D defect solid;",
        "- only constant-depth extrusion builds;",
        "- geometry depth variation cannot be verified;",
        "- Bx/By/Bz export fails;",
        "- no-defect/defect pair or delta_b check fails;",
        "- forbidden artifacts would need to be staged.",
        "",
        "Proceed rule: Stage B geometry-only must pass before Stage C forward. Stage C starts with medium_round only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = [row["sample_id"] for row in rows]
    passes = sum(1 for row in rows if str(row["stage_a_validation_pass"]) == "True")
    lines = [
        "20.67 true 3D variable-depth geometry test plan summary",
        "",
        f"target_samples: {len(rows)}",
        f"minimum_sample: {MINIMUM_SAMPLE_ID}",
        f"sample_ids: {sample_ids}",
        f"stage_a_validation_pass_count: {passes}",
        "sensor_z_m: 0.008",
        "axis_names: [Bx, By, Bz]",
        "baseline_20_66_depth_levels: 5",
        "high_layer_fallback_depth_levels: [12, 16]",
        "smooth_probe_policy: limited; no unbounded smooth-route debugging",
        "exact_piao_rbc: False",
        "",
        "Gate result: PASS" if passes >= 1 and rows[0]["sample_id"] == MINIMUM_SAMPLE_ID else "Gate result: FAIL",
        "",
        "Boundary:",
        "- This plan does not claim smooth variable-depth success.",
        "- 12/16-layer fallback must be reported as high_layer_pass, not variable_depth_pass.",
        "- projected_mask_2d remains a 2D comparator; 3D labels remain RBC params plus depth grid/map metadata.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    check_no_overwrite([args.plan_csv, args.summary, args.preflight_summary], args.overwrite)
    rows = build_rows()
    if not rows or rows[0]["sample_id"] != MINIMUM_SAMPLE_ID:
        raise RuntimeError("medium_round must be the first and minimum forward sample")
    if not bool(rows[0]["stage_a_validation_pass"]):
        raise RuntimeError("medium_round Stage A validation failed")
    write_preflight_summary(args.preflight_summary)
    write_csv(args.plan_csv, rows, PLAN_FIELDS)
    write_summary(args.summary, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
