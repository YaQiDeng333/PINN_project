#!/usr/bin/env python
"""Design the 20.66 true-3D RBC-style smoke sample plan.

This script is intentionally pure Python. It does not call COMSOL and does not
write data packs. It builds a small, deterministic RBC-style depth/profile
plan, validates that the derived 3D labels are reconstructable, and writes the
CSV/summary artifacts used by the COMSOL smoke generator.

The surface formula is an engineering approximation inspired by Piao-style RBC
six-parameter labels. It is not a verified exact reproduction of Piao 2019.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_CSV = ROOT / "results/metrics/true_3d_rbc_smoke_plan.csv"
DEFAULT_VALIDATION_CSV = ROOT / "results/metrics/true_3d_rbc_smoke_profile_validation.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_smoke_plan_summary.txt"
DEFAULT_PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_smoke_preflight_summary.txt"

MASK_WIDTH = 128
MASK_HEIGHT = 64
MASK_X_START_M = -0.04
MASK_X_STOP_M = 0.04
MASK_Y_START_M = -0.01
MASK_Y_STOP_M = 0.01
GRID_U_COUNT = 33
GRID_V_COUNT = 17
CONTOUR_U_COUNT = 96
DEPTH_LEVEL_FRACTIONS = (0.20, 0.40, 0.60, 0.80, 0.95)
EXACT_PIAO_RBC = False
RBC_FORMULA_STATUS = "RBC-style / RBC-inspired engineering approximation; exact Piao RBC formula not implemented"


PLAN_FIELDS = [
    "sample_id",
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
    "profile_pose_json",
    "rbc_params_json",
    "profile_depth_grid_shape_json",
    "profile_depth_grid_m_json",
    "profile_depth_map_xy_shape_json",
    "profile_depth_map_xy_m_json",
    "projected_mask_2d_shape_json",
    "projected_mask_2d_json",
    "projection_threshold_m",
    "depth_levels_m_json",
    "depth_level_polygons_json",
    "geometry_params_json",
    "generated_geometry_type",
    "stepped_depth_approximation_planned",
    "notes",
]

VALIDATION_FIELDS = [
    "sample_id",
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
    "depth_level_count",
    "all_depth_level_polygons_valid",
    "min_depth_level_polygon_area_m2",
    "profile_depth_grid_serializable",
    "profile_depth_map_serializable",
    "projected_mask_serializable",
    "geometry_params_serializable",
    "validation_pass",
    "notes",
]


@dataclass(frozen=True)
class SmokeSample:
    sample_id: str
    split_tag: str
    L_m: float
    W_m: float
    D_m: float
    wLD: float
    wWD: float
    wLW: float
    center_x_m: float
    center_y_m: float
    angle_deg: float
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design true-3D RBC-style smoke plan.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rounded_list(array: np.ndarray, decimals: int = 9) -> list[Any]:
    return np.round(np.asarray(array, dtype=np.float64), decimals=decimals).tolist()


def rbc_weight_curve(t: np.ndarray | float, weight: float) -> np.ndarray:
    values = np.asarray(t, dtype=np.float64)
    clipped = np.clip(values, 0.0, 1.0)
    numerator = weight * (1.0 - clipped * clipped)
    denominator = numerator + clipped * clipped + 1.0e-12
    return np.clip(numerator / denominator, 0.0, 1.0)


def inverse_weight_curve(value: float, weight: float) -> float:
    target = float(np.clip(value, 0.0, 1.0))
    if target <= 0.0:
        return 1.0
    if target >= 1.0:
        return 0.0
    denom = weight * (1.0 - target) + target
    return float(math.sqrt(max(0.0, weight * (1.0 - target) / denom)))


def sample_table() -> list[SmokeSample]:
    return [
        SmokeSample(
            "shallow_smooth_small",
            "smoke_train_like",
            0.010,
            0.006,
            0.0010,
            0.707,
            0.707,
            0.707,
            -0.004,
            0.000,
            0.0,
            "shallow/small round-like profile",
        ),
        SmokeSample(
            "medium_round",
            "smoke_train_like",
            0.018,
            0.010,
            0.0025,
            0.707,
            0.707,
            0.707,
            0.004,
            0.000,
            0.0,
            "medium round-like reference",
        ),
        SmokeSample(
            "deep_round",
            "smoke_train_like",
            0.020,
            0.012,
            0.0055,
            0.707,
            0.707,
            0.707,
            -0.002,
            0.0015,
            0.0,
            "deep round-like strong signal",
        ),
        SmokeSample(
            "medium_boxy",
            "smoke_train_like",
            0.022,
            0.014,
            0.0030,
            1.200,
            1.200,
            1.100,
            0.002,
            -0.0015,
            0.0,
            "medium boxier/flatter transition",
        ),
        SmokeSample(
            "wide_shallow",
            "smoke_val_like",
            0.030,
            0.020,
            0.0015,
            1.000,
            0.850,
            1.200,
            -0.003,
            0.0010,
            0.0,
            "wide shallow footprint stress case",
        ),
        SmokeSample(
            "narrow_deep",
            "smoke_test_like",
            0.014,
            0.006,
            0.0060,
            0.550,
            0.600,
            0.550,
            0.003,
            -0.0010,
            0.0,
            "narrow deep sharper profile",
        ),
    ]


def local_depth(sample: SmokeSample, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u_abs = np.abs(u)
    v_abs = np.abs(v)
    length_profile = rbc_weight_curve(u_abs, sample.wLD)
    width_scale = rbc_weight_curve(u_abs, sample.wLW)
    inside_u = u_abs <= 1.0
    safe_width = np.maximum(width_scale, 1.0e-9)
    v_norm = np.divide(v_abs, safe_width, out=np.ones_like(v_abs), where=safe_width > 0.0)
    inside = inside_u & (v_norm <= 1.0)
    width_profile = rbc_weight_curve(v_norm, sample.wWD)
    depth = sample.D_m * length_profile * width_profile
    return np.where(inside, depth, 0.0).astype(np.float64)


def xy_to_uv(sample: SmokeSample, xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(sample.angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = xx - sample.center_x_m
    dy = yy - sample.center_y_m
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return 2.0 * local_x / sample.L_m, 2.0 * local_y / sample.W_m


def local_to_xy(sample: SmokeSample, local_points: np.ndarray) -> np.ndarray:
    angle = math.radians(sample.angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x_local = local_points[:, 0]
    y_local = local_points[:, 1]
    x = cos_a * x_local - sin_a * y_local + sample.center_x_m
    y = sin_a * x_local + cos_a * y_local + sample.center_y_m
    return np.stack([x, y], axis=1)


def build_profile_depth_grid(sample: SmokeSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(-1.0, 1.0, GRID_U_COUNT)
    v = np.linspace(-1.0, 1.0, GRID_V_COUNT)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return u, v, local_depth(sample, uu, vv)


def build_depth_map(sample: SmokeSample, mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    yy, xx = np.meshgrid(mask_y, mask_x, indexing="ij")
    u, v = xy_to_uv(sample, xx, yy)
    return local_depth(sample, u, v)


def polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def contour_polygon_for_depth(sample: SmokeSample, depth_level_m: float) -> np.ndarray:
    u_values = np.linspace(-1.0, 1.0, CONTOUR_U_COUNT)
    top: list[tuple[float, float]] = []
    for u_value in u_values:
        lp = float(rbc_weight_curve(abs(u_value), sample.wLD))
        width_scale = float(rbc_weight_curve(abs(u_value), sample.wLW))
        if lp <= 0.0 or width_scale <= 1.0e-6:
            continue
        required_width_profile = depth_level_m / max(sample.D_m * lp, 1.0e-12)
        if required_width_profile > 1.0:
            continue
        v_norm = inverse_weight_curve(required_width_profile, sample.wWD)
        v_max = max(0.0, min(1.0, width_scale * v_norm))
        if v_max <= 1.0e-5:
            continue
        top.append((0.5 * sample.L_m * u_value, 0.5 * sample.W_m * v_max))
    if len(top) < 3:
        return np.empty((0, 2), dtype=np.float64)
    bottom = [(x, -y) for x, y in reversed(top)]
    local = np.asarray(top + bottom, dtype=np.float64)
    return local_to_xy(sample, local)


def build_depth_level_polygons(sample: SmokeSample) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, fraction in enumerate(DEPTH_LEVEL_FRACTIONS, start=1):
        depth_level = float(sample.D_m * fraction)
        vertices = contour_polygon_for_depth(sample, depth_level)
        if len(vertices) < 3 or polygon_area(vertices) <= 0.0:
            continue
        rows.append(
            {
                "level_index": index,
                "depth_m": depth_level,
                "fraction_of_D": float(fraction),
                "vertex_count": int(len(vertices)),
                "area_m2": polygon_area(vertices),
                "vertices": rounded_list(vertices, decimals=9),
            }
        )
    return rows


def connected_component_count(mask: np.ndarray) -> int:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    count = 0
    for row in range(height):
        for col in range(width):
            if mask[row, col] == 0 or visited[row, col]:
                continue
            count += 1
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                r, c = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
    return count


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mask_x = np.linspace(MASK_X_START_M, MASK_X_STOP_M, MASK_WIDTH)
    mask_y = np.linspace(MASK_Y_START_M, MASK_Y_STOP_M, MASK_HEIGHT)
    pixel_area = float((mask_x[1] - mask_x[0]) * (mask_y[1] - mask_y[0]))
    plan_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for sample in sample_table():
        _, _, depth_grid = build_profile_depth_grid(sample)
        depth_map = build_depth_map(sample, mask_x, mask_y)
        projection_threshold = max(1.0e-6, 0.01 * sample.D_m)
        mask = (depth_map >= projection_threshold).astype(np.uint8)
        polygons = build_depth_level_polygons(sample)
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
            "profile_type": "rbc_style_symmetric_pit",
            "exact_piao_rbc": EXACT_PIAO_RBC,
            "rbc_formula_status": RBC_FORMULA_STATUS,
            "rbc_params": rbc_params,
            "profile_pose": profile_pose,
            "projection_threshold_m": projection_threshold,
            "depth_levels_m": [row["depth_m"] for row in polygons],
            "depth_level_polygons": polygons,
            "stepped_depth_approximation": True,
            "smooth_variable_depth_solid_verified": False,
            "generated_geometry_type": "rbc_style_stepped_depth_candidate",
            "units": "coordinates=m, field=T",
        }
        max_depth = float(depth_grid.max())
        rel_error = abs(max_depth - sample.D_m) / max(sample.D_m, 1.0e-12)
        min_polygon_area = min((float(row["area_m2"]) for row in polygons), default=0.0)
        cc_count = connected_component_count(mask)
        validation_pass = (
            bool(np.isfinite(depth_grid).all())
            and bool(np.isfinite(depth_map).all())
            and bool((depth_grid >= -1.0e-12).all())
            and rel_error <= 0.03
            and int(mask.sum()) > 0
            and cc_count == 1
            and len(polygons) >= 4
            and min_polygon_area > 0.0
        )
        plan_rows.append(
            {
                "sample_id": sample.sample_id,
                "split_tag": sample.split_tag,
                "profile_type": "rbc_style_symmetric_pit",
                "exact_piao_rbc": str(EXACT_PIAO_RBC),
                "rbc_formula_status": RBC_FORMULA_STATUS,
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
                "depth_levels_m_json": json_dumps([row["depth_m"] for row in polygons]),
                "depth_level_polygons_json": json_dumps(polygons),
                "geometry_params_json": json_dumps(geometry_params),
                "generated_geometry_type": "rbc_style_stepped_depth_candidate",
                "stepped_depth_approximation_planned": "True",
                "notes": sample.notes,
            }
        )
        validation_rows.append(
            {
                "sample_id": sample.sample_id,
                "depth_grid_finite": bool(np.isfinite(depth_grid).all()),
                "depth_map_finite": bool(np.isfinite(depth_map).all()),
                "depth_nonnegative": bool((depth_grid >= -1.0e-12).all() and (depth_map >= -1.0e-12).all()),
                "max_depth_m": max_depth,
                "target_D_m": sample.D_m,
                "max_depth_abs_error_m": abs(max_depth - sample.D_m),
                "max_depth_rel_error": rel_error,
                "projected_mask_area_px": int(mask.sum()),
                "projected_mask_area_m2": int(mask.sum()) * pixel_area,
                "footprint_nonempty": int(mask.sum()) > 0,
                "depth_level_count": len(polygons),
                "all_depth_level_polygons_valid": len(polygons) >= 4 and min_polygon_area > 0.0,
                "min_depth_level_polygon_area_m2": min_polygon_area,
                "profile_depth_grid_serializable": True,
                "profile_depth_map_serializable": True,
                "projected_mask_serializable": True,
                "geometry_params_serializable": True,
                "validation_pass": validation_pass,
                "notes": f"connected_components={cc_count}; {sample.notes}",
            }
        )
    return plan_rows, validation_rows


def write_preflight_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "20.66 true 3D RBC-style smoke preflight summary",
                "",
                "Decision: proceed with a small true-3D smoke if execution remains bounded to RBC-style depth labels, COMSOL variable-depth or stepped-depth geometry, Bx/By/Bz @ sensor_z_m=0.008, and schema validation.",
                "",
                "Subagent core conclusions:",
                "- Method/literature: 20.66 follows the true 3D / Piao-style route, but it is not a full Piao 2019 reproduction. Piao-style migration is limited to three-axis MFL observation and RBC six-parameter 3D profile labels.",
                "- COMSOL capability: current scripts support real 3D volume solves, Boolean Difference, polygon prism extrusion, no-defect/defect pairs, and mf.Bx/mf.By/mf.Bz export. Smooth RBC variable-depth solid generation is not verified.",
                "- RBC/depth-grid design: use a deterministic RBC-style engineering approximation unless an exact Piao RBC formula is explicitly implemented and marked exact_piao_rbc=True.",
                "- Experiment design: target 6 samples, minimum 3, single defect only, covering shallow/medium/deep, narrow/wide, and round/boxy/sharper curvature cases.",
                "- Safety/git: do not submit data, NPZ, .mph, raw CSV, checkpoints, preview PNG, notes, baseline docs, MODEL_STRUCTURE_PLAN.md deletion, scripts/visualize_current_baseline.py, or existing COMSOL dirty items.",
                "- Implementation feasibility: execute in layers: pure-Python RBC validation, COMSOL geometry/one-sample smoke, 6-sample pack, PINN schema validation.",
                "",
                "Hard blockers:",
                "- only constant-depth top-view extrusion is possible;",
                "- fewer than 3 samples can be generated;",
                "- Bx/By/Bz export fails;",
                "- no-defect/defect pair or delta_b = b_defect - b_no_defect check fails;",
                "- depth/profile label cannot be reconstructed from saved rbc_params/profile_depth data.",
                "",
                "Allowed submission files are limited to the 20.66 scripts, summaries, metrics, and route Markdown listed in the user prompt. No generated data artifacts are allowed in git.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_summary(path: Path, plan_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depths = [float(row["D_m"]) for row in plan_rows]
    lengths = [float(row["L_m"]) for row in plan_rows]
    widths = [float(row["W_m"]) for row in plan_rows]
    passes = sum(1 for row in validation_rows if str(row["validation_pass"]) == "True")
    split_counts = Counter(row["split_tag"] for row in plan_rows)
    path.write_text(
        "\n".join(
            [
                "20.66 true 3D RBC-style smoke plan summary",
                "",
                f"sample_count: {len(plan_rows)}",
                f"validation_pass_count: {passes}",
                f"split_tag_distribution: {dict(split_counts)}",
                f"L_m_range: {min(lengths):.6f} to {max(lengths):.6f}",
                f"W_m_range: {min(widths):.6f} to {max(widths):.6f}",
                f"D_m_range: {min(depths):.6f} to {max(depths):.6f}",
                f"exact_piao_rbc: {EXACT_PIAO_RBC}",
                f"rbc_formula_status: {RBC_FORMULA_STATUS}",
                "sensor_z_m: 0.008",
                "scan_line_y_m: [-0.001, 0.0, 0.001]",
                "axis_names: [Bx, By, Bz]",
                "axis_expressions: [mf.Bx, mf.By, mf.Bz]",
                "profile_depth_grid_shape: [33, 17]",
                "profile_depth_map_xy_shape: [64, 128]",
                "projection_rule: depth > max(1e-6 m, 0.01 * D_m)",
                "planned_geometry: stepped-depth layered approximation unless smooth true variable-depth solid is explicitly implemented in COMSOL.",
                "",
                "Gate result: PASS for Stage A pure-Python profile validation." if passes == len(plan_rows) else "Gate result: FAIL for Stage A pure-Python profile validation.",
                "",
                "Important boundary: projected_mask_2d is only a 2D comparator label. The 3D label is reconstructable from rbc_params, profile_pose, profile_depth_grid_m/profile_depth_map_xy_m, depth_levels_m, and geometry_params_json.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    paths = [args.plan_csv, args.validation_csv, args.summary, args.preflight_summary]
    check_no_overwrite(paths, args.overwrite)
    plan_rows, validation_rows = build_rows()
    write_preflight_summary(args.preflight_summary)
    write_csv(args.plan_csv, plan_rows, PLAN_FIELDS)
    write_csv(args.validation_csv, validation_rows, VALIDATION_FIELDS)
    write_summary(args.summary, plan_rows, validation_rows)
    failures = [row for row in validation_rows if str(row["validation_pass"]) != "True"]
    if failures:
        raise RuntimeError(f"RBC smoke profile validation failed for {[row['sample_id'] for row in failures]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
