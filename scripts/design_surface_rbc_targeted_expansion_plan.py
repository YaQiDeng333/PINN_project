#!/usr/bin/env python
"""Design the surface RBC targeted +120 top-up plan.

This is plan/label generation only. It does not call COMSOL, does not write
NPZ data, and does not create an assembled training dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import design_true_3d_rbc_pilot_pack_plan as base_plan  # noqa: E402


DATASET_ID = "comsol_true_3d_rbc_surface_targeted_topup_v1_120"
SOURCE_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc"
NPZ_ROUTE = "true_3d_piao_style"
REGISTRY_ROUTE = "true_3d_rbc_surface_targeted_expansion"

DEFAULT_PREFLIGHT = ROOT / "results/summaries/surface_rbc_targeted_expansion_preflight_summary.txt"
DEFAULT_SUMMARY = ROOT / "results/summaries/surface_rbc_targeted_expansion_plan_summary.txt"
DEFAULT_PLAN = ROOT / "results/metrics/surface_rbc_targeted_expansion_plan.csv"
DEFAULT_COVERAGE = ROOT / "results/metrics/surface_rbc_targeted_expansion_expected_coverage.csv"
DEFAULT_TEMP_MESH_DIR = ROOT / "data/comsol_mfl/generated/temp_surface_rbc_targeted_expansion_meshes"
SOURCE_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
CURRENT_BASELINE = ROOT / "CURRENT_BASELINE.md"
COMSOL_ROOT = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")

TARGET_ROLE_COUNTS = {
    "balanced_interior": 60,
    "hard_depth_aspect": 24,
    "edge_position": 24,
    "old_distribution_anchor": 12,
}
TARGET_SPLIT_BY_ROLE = {
    "balanced_interior": {"train": 40, "val": 10, "test": 10},
    "hard_depth_aspect": {"train": 16, "val": 4, "test": 4},
    "edge_position": {"train": 16, "val": 4, "test": 4},
    "old_distribution_anchor": {"train": 8, "val": 2, "test": 2},
}
EDGE_CENTERS = {
    "interior": (0.0, 0.0),
    "near_left_roi": (-0.024, 0.0),
    "near_right_roi": (0.024, 0.0),
    "near_lower_scan_window": (0.0, -0.006),
    "near_upper_scan_window": (0.0, 0.006),
}
PLAN_FIELDS = base_plan.PLAN_FIELDS + [
    "targeted_role",
    "edge_position_bin",
    "axis_order",
    "scan_line_y_m",
    "sensor_x_count",
    "coverage_signature",
    "replacement_of_sample_id",
]
COVERAGE_FIELDS = ["coverage_type", "key", "planned_count", "target_count", "target_met"]

COMSOL_SAFE_REPLACEMENTS = {
    "surface_rbc_targeted_008_balanced_interior_sharp_medium_narrow": {
        "sample_id": "surface_rbc_targeted_008_balanced_interior_sharp_medium_narrow_repl03",
        "rbc_params": {"L_m": 0.024, "W_m": 0.0095, "D_m": 0.0031, "wLD": 0.55, "wWD": 0.6105, "wLW": 0.55},
        "reason": "COMSOL boolean blocker for narrow sharp medium row; relax only depth/width interaction within the same signature",
    },
    "surface_rbc_targeted_022_balanced_interior_round_deep_balanced": {
        "sample_id": "surface_rbc_targeted_022_balanced_interior_round_deep_balanced_repl01",
        "rbc_params": {"wLD": 0.722, "wWD": 0.6965, "wLW": 0.713},
        "reason": "COMSOL free-tet blocker for symmetric deep round row; add minimal round-template anisotropy used by neighboring passing rows",
    },
    "surface_rbc_targeted_033_balanced_interior_boxy_deep_compact": {
        "sample_id": "surface_rbc_targeted_033_balanced_interior_boxy_deep_compact_repl01",
        "rbc_params": {"D_m": 0.0044, "wLD": 1.085, "wWD": 1.1105, "wLW": 1.094},
        "reason": "COMSOL mesh self-intersection for deep boxy compact row; relax boxy curvature and depth within signature",
    },
    "surface_rbc_targeted_034_balanced_interior_boxy_deep_balanced": {
        "sample_id": "surface_rbc_targeted_034_balanced_interior_boxy_deep_balanced_repl01",
        "rbc_params": {"D_m": 0.0044, "wLD": 1.1, "wWD": 1.1, "wLW": 1.1},
        "reason": "COMSOL imported-domain blocker for deep boxy balanced row; use symmetric boxy curvature and lower deep-bin depth",
    },
    "surface_rbc_targeted_036_balanced_interior_boxy_deep_narrow": {
        "sample_id": "surface_rbc_targeted_036_balanced_interior_boxy_deep_narrow_repl01",
        "rbc_params": {"D_m": 0.0044, "W_m": 0.010, "wLD": 1.1, "wWD": 1.1, "wLW": 1.1},
        "reason": "COMSOL mesh self-intersection for deep boxy narrow row; relax width-depth and boxy curvature within signature",
    },
    "surface_rbc_targeted_042_balanced_interior_LD_dominant_medium_balanced": {
        "sample_id": "surface_rbc_targeted_042_balanced_interior_LD_dominant_medium_balanced_repl03",
        "rbc_params": {"L_m": 0.01957, "W_m": 0.01032675, "D_m": 0.0025272, "wLD": 1.17, "wWD": 0.871, "wLW": 0.988},
        "reason": "COMSOL free-tet boundary conformity blocker for LD-dominant medium balanced row; relax size/depth interaction within signature",
    },
    "surface_rbc_targeted_056_balanced_interior_WD_dominant_medium_narrow": {
        "sample_id": "surface_rbc_targeted_056_balanced_interior_WD_dominant_medium_narrow_repl01",
        "rbc_params": {"W_m": 0.0095, "D_m": 0.0031, "wLD": 0.895, "wWD": 1.1685, "wLW": 1.018},
        "reason": "Full run long-solve blocker for WD-dominant medium narrow row; relax width-depth and WD extremity within signature",
    },
    "surface_rbc_targeted_074_hard_depth_aspect_round_medium_narrow": {
        "sample_id": "surface_rbc_targeted_074_hard_depth_aspect_round_medium_narrow_repl01",
        "rbc_params": {"W_m": 0.0085, "D_m": 0.0031, "wLD": 0.677, "wWD": 0.728, "wLW": 0.695},
        "reason": "COMSOL imported-domain blocker for hard round medium narrow row; relax width-depth interaction within signature",
    },
    "surface_rbc_targeted_080_hard_depth_aspect_boxy_medium_narrow": {
        "sample_id": "surface_rbc_targeted_080_hard_depth_aspect_boxy_medium_narrow_repl01",
        "rbc_params": {"L_m": 0.02424, "W_m": 0.0085, "D_m": 0.0031, "wLD": 1.085, "wWD": 1.1105, "wLW": 1.094},
        "reason": "COMSOL imported-domain blocker for hard boxy medium narrow row; relax hard aspect and boxy curvature within signature",
    },
    "surface_rbc_targeted_087_edge_position_sharp_medium_narrow": {
        "sample_id": "surface_rbc_targeted_087_edge_position_sharp_medium_narrow_repl01",
        "rbc_params": {"W_m": 0.0095, "D_m": 0.0031, "wLD": 0.55, "wWD": 0.6105, "wLW": 0.55, "center_y_m": -0.0045},
        "reason": "COMSOL geometry union blocker for edge sharp medium narrow row; relax edge margin and width-depth interaction within signature",
    },
    "surface_rbc_targeted_093_edge_position_round_medium_narrow": {
        "sample_id": "surface_rbc_targeted_093_edge_position_round_medium_narrow_repl01",
        "rbc_params": {"W_m": 0.0085, "D_m": 0.0031, "wLD": 0.677, "wWD": 0.728, "wLW": 0.695, "center_x_m": -0.021},
        "reason": "COMSOL mesh boundary blocker for edge round medium narrow row; relax left edge margin and width-depth interaction within signature",
    },
    "surface_rbc_targeted_095_edge_position_round_deep_balanced": {
        "sample_id": "surface_rbc_targeted_095_edge_position_round_deep_balanced_repl01",
        "rbc_params": {"D_m": 0.0044, "wLD": 0.722, "wWD": 0.6965, "wLW": 0.713, "center_y_m": -0.0045},
        "reason": "COMSOL geometry union blocker for edge round deep balanced row; relax lower edge margin and depth within signature",
    },
    "surface_rbc_targeted_096_edge_position_round_deep_narrow": {
        "sample_id": "surface_rbc_targeted_096_edge_position_round_deep_narrow_repl01",
        "rbc_params": {"D_m": 0.0044, "W_m": 0.010, "wLD": 0.737, "wWD": 0.686, "wLW": 0.719, "center_y_m": 0.0045},
        "reason": "COMSOL geometry union blocker for edge round deep narrow row; relax upper edge margin and width-depth interaction within signature",
    },
    "surface_rbc_targeted_099_edge_position_boxy_medium_narrow": {
        "sample_id": "surface_rbc_targeted_099_edge_position_boxy_medium_narrow_repl01",
        "rbc_params": {"W_m": 0.0095, "D_m": 0.0031, "wLD": 1.085, "wWD": 1.1105, "wLW": 1.094, "center_y_m": -0.0045},
        "reason": "COMSOL mesh boundary blocker for edge boxy medium narrow row; relax lower edge margin, width-depth, and boxy curvature within signature",
    },
    "surface_rbc_targeted_100_edge_position_boxy_deep_compact": {
        "sample_id": "surface_rbc_targeted_100_edge_position_boxy_deep_compact_repl01",
        "rbc_params": {"D_m": 0.0044, "wLD": 1.085, "wWD": 1.1105, "wLW": 1.094, "center_y_m": 0.0045},
        "reason": "COMSOL geometry union blocker for edge boxy deep compact row; relax upper edge margin, depth, and boxy curvature within signature",
    },
    "surface_rbc_targeted_102_edge_position_boxy_deep_narrow": {
        "sample_id": "surface_rbc_targeted_102_edge_position_boxy_deep_narrow_repl01",
        "rbc_params": {"D_m": 0.0044, "W_m": 0.010, "wLD": 1.1, "wWD": 1.1, "wLW": 1.1, "center_x_m": 0.021},
        "reason": "COMSOL mesh boundary blocker for edge boxy deep narrow row; relax right edge margin, width-depth, and boxy curvature within signature",
    },
    "surface_rbc_targeted_107_edge_position_LD_dominant_deep_balanced": {
        "sample_id": "surface_rbc_targeted_107_edge_position_LD_dominant_deep_balanced_repl01",
        "rbc_params": {"D_m": 0.0044, "wLD": 1.155, "wWD": 0.8815, "wLW": 0.982, "center_y_m": -0.0045},
        "reason": "COMSOL geometry union blocker for edge LD-dominant deep balanced row; relax lower edge margin, depth, and LD extremity within signature",
    },
    "surface_rbc_targeted_108_edge_position_LD_dominant_deep_narrow": {
        "sample_id": "surface_rbc_targeted_108_edge_position_LD_dominant_deep_narrow_repl01",
        "rbc_params": {"D_m": 0.0044, "W_m": 0.010, "wLD": 1.17, "wWD": 0.871, "wLW": 0.988, "center_y_m": 0.0045},
        "reason": "COMSOL geometry union blocker for edge LD-dominant deep narrow row; relax upper edge margin, width-depth, and LD extremity within signature",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design surface RBC targeted expansion top-up plan.")
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--temp-mesh-dir", type=Path, default=DEFAULT_TEMP_MESH_DIR)
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
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def git_value(cwd: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def split_sequence(role: str) -> list[str]:
    counts = TARGET_SPLIT_BY_ROLE[role]
    return ["train"] * counts["train"] + ["val"] * counts["val"] + ["test"] * counts["test"]


def perturbed(spec: Any, variant: int, role: str) -> tuple[float, float, float, float, float, float]:
    scale = ((variant % 7) - 3) * 0.010
    depth_scale = ((variant % 5) - 2) * 0.014
    curv_scale = ((variant % 9) - 4) * 0.015
    L = float(spec.L_m) * (1.0 + scale)
    W = float(spec.W_m) * (1.0 - 0.55 * scale)
    D = float(spec.D_m) * (1.0 + depth_scale)
    wLD = float(spec.wLD) + curv_scale
    wWD = float(spec.wWD) - 0.7 * curv_scale
    wLW = float(spec.wLW) + 0.4 * curv_scale
    if role == "hard_depth_aspect":
        if spec.depth_bin == "deep":
            D = min(max(D * 1.08, 0.0044), 0.0060)
        if spec.aspect_bin == "narrow":
            L = min(max(L * 1.08, 0.010), 0.030)
            W = max(W * 0.92, 0.006)
        if spec.depth_bin == "shallow":
            D = max(min(D * 0.90, 0.0018), 0.0010)
    if spec.curvature_template == "LD_dominant":
        wLD = max(wLD, 1.02)
    if spec.curvature_template == "WD_dominant":
        wWD = max(wWD, 1.02)
    return (
        min(max(L, 0.010), 0.030),
        min(max(W, 0.006), 0.020),
        min(max(D, 0.001), 0.006),
        min(max(wLD, 0.55), 1.20),
        min(max(wWD, 0.55), 1.20),
        min(max(wLW, 0.55), 1.20),
    )


def coverage_signature(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["targeted_role"]),
        str(row["depth_bin"]),
        str(row["aspect_bin"]),
        str(row["curvature_template"]),
        str(row["edge_position_bin"]),
    )


def make_row(
    *,
    sample_id: str,
    split: str,
    spec: Any,
    role: str,
    edge_position_bin: str,
    variant: int,
    temp_mesh_dir: Path,
    replacement_of_sample_id: str = "",
    rbc_param_override: dict[str, float] | None = None,
    replacement_reason: str = "",
) -> dict[str, Any]:
    L, W, D, wLD, wWD, wLW = perturbed(spec, variant, role)
    if rbc_param_override:
        L = float(rbc_param_override.get("L_m", L))
        W = float(rbc_param_override.get("W_m", W))
        D = float(rbc_param_override.get("D_m", D))
        wLD = float(rbc_param_override.get("wLD", wLD))
        wWD = float(rbc_param_override.get("wWD", wWD))
        wLW = float(rbc_param_override.get("wLW", wLW))
    center_x, center_y = EDGE_CENTERS[edge_position_bin]
    if rbc_param_override:
        center_x = float(rbc_param_override.get("center_x_m", center_x))
        center_y = float(rbc_param_override.get("center_y_m", center_y))
    sample = base_plan.smoke_plan.SmokeSample(
        sample_id=sample_id,
        split_tag=split,
        L_m=L,
        W_m=W,
        D_m=D,
        wLD=wLD,
        wWD=wWD,
        wLW=wLW,
        center_x_m=center_x,
        center_y_m=center_y,
        angle_deg=0.0,
        notes=f"surface RBC targeted top-up; role={role}; edge={edge_position_bin}",
    )
    mask_x = np.linspace(base_plan.MASK_X_START_M, base_plan.MASK_X_STOP_M, base_plan.MASK_WIDTH)
    mask_y = np.linspace(base_plan.MASK_Y_START_M, base_plan.MASK_Y_STOP_M, base_plan.MASK_HEIGHT)
    pixel_area = float((mask_x[1] - mask_x[0]) * (mask_y[1] - mask_y[0]))
    _, _, depth_grid = base_plan.smoke_plan.build_profile_depth_grid(sample)
    depth_map = base_plan.smoke_plan.build_depth_map(sample, mask_x, mask_y)
    projection_threshold = max(1.0e-6, 0.01 * D)
    mask = (depth_map >= projection_threshold).astype(np.uint8)
    connected_components = base_plan.smoke_plan.connected_component_count(mask)
    footprint_area = int(mask.sum()) * pixel_area
    volume_proxy = float(np.sum(depth_map) * pixel_area)
    profile_pose = {
        "center_x_m": center_x,
        "center_y_m": center_y,
        "angle_rad": 0.0,
        "angle_deg": 0.0,
        "L_m": L,
        "W_m": W,
        "D_m": D,
    }
    rbc_params = {"L_m": L, "W_m": W, "D_m": D, "wLD": wLD, "wWD": wWD, "wLW": wLW}
    profile_pose_to_comsol = {
        "mesh_units": "m",
        "comsol_coordinates": "x/y on steel surface plane; top cap z=0; bottom surface z=-depth",
        "center_x_m": center_x,
        "center_y_m": center_y,
        "angle_rad": 0.0,
        "steel_surface_z_m": 0.0,
    }
    geometry_params = {
        "dataset_id": DATASET_ID,
        "sample_id": sample_id,
        "profile_type": "rbc_style_symmetric_pit",
        "geometry_method": "imported_watertight_mesh_solid",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "rbc_formula_status": base_plan.RBC_FORMULA_STATUS,
        "rbc_params": rbc_params,
        "profile_pose": profile_pose,
        "targeted_role": role,
        "edge_position_bin": edge_position_bin,
        "depth_bin": spec.depth_bin,
        "size_bin": spec.size_bin,
        "aspect_bin": spec.aspect_bin,
        "curvature_template": spec.curvature_template,
        "replacement_of_sample_id": replacement_of_sample_id,
        "replacement_reason": replacement_reason,
    }
    row: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "schema_version": SCHEMA_VERSION,
        "route": NPZ_ROUTE,
        "sample_id": sample_id,
        "split": split,
        "split_tag": split,
        "depth_bin": spec.depth_bin,
        "size_bin": spec.size_bin,
        "aspect_bin": spec.aspect_bin,
        "curvature_bin": spec.curvature_template,
        "curvature_template": spec.curvature_template,
        "profile_type": "rbc_style_symmetric_pit",
        "exact_piao_rbc": "False",
        "rbc_style_approximation": "True",
        "rbc_formula_status": base_plan.RBC_FORMULA_STATUS,
        "geometry_method": "imported_watertight_mesh_solid",
        "generated_geometry_type": "rbc_style_imported_watertight_mesh_candidate",
        "L_m": L,
        "W_m": W,
        "D_m": D,
        "wLD": wLD,
        "wWD": wWD,
        "wLW": wLW,
        "center_x_m": center_x,
        "center_y_m": center_y,
        "angle_rad": 0.0,
        "angle_deg": 0.0,
        "sensor_z_m": 0.008,
        "scan_line_y_json": json_dumps([-0.001, 0.0, 0.001]),
        "axis_names_json": json_dumps(["Bx", "By", "Bz"]),
        "axis_expressions_json": json_dumps(["mf.Bx", "mf.By", "mf.Bz"]),
        "profile_pose_json": json_dumps(profile_pose),
        "rbc_params_json": json_dumps(rbc_params),
        "profile_depth_grid_shape_json": json_dumps(list(depth_grid.shape)),
        "profile_depth_grid_m_json": json_dumps(base_plan.rounded_list(depth_grid, decimals=9)),
        "profile_depth_map_xy_shape_json": json_dumps(list(depth_map.shape)),
        "profile_depth_map_xy_m_json": json_dumps(base_plan.rounded_list(depth_map, decimals=9)),
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
        "temp_mesh_output_path": str(temp_mesh_dir / f"{sample_id}.stl"),
        "geometry_params_json": json_dumps(geometry_params),
        "allowed_use": "schema_validation, explicit_surface_rbc_expansion_training_gate",
        "forbidden_use": "automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate",
        "notes": f"surface RBC targeted top-up source only; role={role}; not assembled dataset"
        + (f"; deterministic COMSOL-safe replacement: {replacement_reason}" if replacement_reason else ""),
        "targeted_role": role,
        "edge_position_bin": edge_position_bin,
        "axis_order": "Bx,By,Bz",
        "scan_line_y_m": "-0.001,0.0,0.001",
        "sensor_x_count": "201",
        "replacement_of_sample_id": replacement_of_sample_id,
    }
    row["coverage_signature"] = "|".join(coverage_signature(row))
    return row


def selected_specs(role: str) -> list[Any]:
    specs = base_plan.pilot_specs()
    if role == "balanced_interior":
        return specs
    if role == "hard_depth_aspect":
        hard = [
            spec
            for spec in specs
            if spec.depth_bin == "deep"
            or spec.aspect_bin == "narrow"
            or spec.curvature_template in {"sharp", "LD_dominant", "WD_dominant"}
        ]
        return (hard * 3)[:24]
    if role == "edge_position":
        edge_specs = [spec for spec in specs if spec.depth_bin in {"medium", "deep"} and spec.aspect_bin != "wide"]
        return (edge_specs * 2)[:24]
    anchors = [spec for spec in specs if spec.depth_bin != "deep" and spec.aspect_bin in {"compact", "balanced", "wide"}]
    return anchors[:12]


def build_plan(temp_mesh_dir: Path = DEFAULT_TEMP_MESH_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role_variant: Counter[str] = Counter()
    for role in ("balanced_interior", "hard_depth_aspect", "edge_position", "old_distribution_anchor"):
        splits = split_sequence(role)
        specs = selected_specs(role)
        if len(specs) != TARGET_ROLE_COUNTS[role]:
            raise RuntimeError(f"role {role} expected {TARGET_ROLE_COUNTS[role]} specs, got {len(specs)}")
        for index, (split, spec) in enumerate(zip(splits, specs, strict=True), start=1):
            role_variant[role] += 1
            edge_bin = "interior"
            if role == "edge_position":
                edge_cycle = ["near_left_roi", "near_right_roi", "near_lower_scan_window", "near_upper_scan_window"]
                edge_bin = edge_cycle[(index - 1) % len(edge_cycle)]
            sample_id = f"surface_rbc_targeted_{len(rows) + 1:03d}_{role}_{spec.curvature_template}_{spec.depth_bin}_{spec.aspect_bin}"
            replacement = COMSOL_SAFE_REPLACEMENTS.get(sample_id)
            output_sample_id = replacement["sample_id"] if replacement else sample_id
            rows.append(
                make_row(
                    sample_id=output_sample_id,
                    split=split,
                    spec=spec,
                    role=role,
                    edge_position_bin=edge_bin,
                    variant=role_variant[role],
                    temp_mesh_dir=temp_mesh_dir,
                    replacement_of_sample_id=sample_id if replacement else "",
                    rbc_param_override=replacement.get("rbc_params") if replacement else None,
                    replacement_reason=replacement.get("reason", "") if replacement else "",
                )
            )
    validate_plan_contract(rows)
    return rows


def make_replacement_row(row: dict[str, Any], replacement_index: int, temp_mesh_dir: Path = DEFAULT_TEMP_MESH_DIR) -> dict[str, Any]:
    replacement = dict(row)
    old_id = str(row["sample_id"])
    replacement["sample_id"] = f"{old_id}_repl{replacement_index:02d}"
    replacement["replacement_of_sample_id"] = old_id
    replacement["temp_mesh_output_path"] = str(temp_mesh_dir / f"{replacement['sample_id']}.stl")
    geom = json.loads(str(row["geometry_params_json"]))
    geom["sample_id"] = replacement["sample_id"]
    geom["replacement_of_sample_id"] = old_id
    replacement["geometry_params_json"] = json_dumps(geom)
    replacement["notes"] = str(row.get("notes", "")) + "; deterministic equivalent replacement"
    replacement["coverage_signature"] = "|".join(coverage_signature(replacement))
    return replacement


def validate_plan_contract(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 120:
        raise RuntimeError(f"expected 120 rows, got {len(rows)}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample_id in plan")
    if Counter(row["targeted_role"] for row in rows) != TARGET_ROLE_COUNTS:
        raise RuntimeError(f"targeted role mismatch: {Counter(row['targeted_role'] for row in rows)}")
    if Counter(row["split"] for row in rows) != {"train": 80, "val": 20, "test": 20}:
        raise RuntimeError(f"split mismatch: {Counter(row['split'] for row in rows)}")
    for row in rows:
        if row["exact_piao_rbc"] != "False" or row["rbc_style_approximation"] != "True":
            raise RuntimeError(f"RBC-style boundary failed for {row['sample_id']}")
        if float(row["sensor_z_m"]) != 0.008 or row["axis_order"] != "Bx,By,Bz":
            raise RuntimeError(f"sensor/axis boundary failed for {row['sample_id']}")
        if int(row["projected_mask_2d_json"].count("1")) <= 0:
            raise RuntimeError(f"projected mask empty for {row['sample_id']}")
        if "multi" in row["sample_id"].lower() or "internal" in row["sample_id"].lower():
            raise RuntimeError(f"forbidden branch token in sample_id: {row['sample_id']}")


def expected_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target_maps = {
        "targeted_role": TARGET_ROLE_COUNTS,
        "split": {"train": 80, "val": 20, "test": 20},
        "edge_position_bin": {"interior": 96, "near_left_roi": 6, "near_right_roi": 6, "near_lower_scan_window": 6, "near_upper_scan_window": 6},
    }
    for coverage_type, targets in target_maps.items():
        counts = Counter(str(row[coverage_type]) for row in rows)
        for key, target in targets.items():
            planned = counts.get(key, 0)
            out.append(
                {
                    "coverage_type": coverage_type,
                    "key": key,
                    "planned_count": planned,
                    "target_count": target,
                    "target_met": planned == target,
                }
            )
    for coverage_type in ("depth_bin", "aspect_bin", "curvature_template"):
        counts = Counter(str(row[coverage_type]) for row in rows)
        for key in sorted(counts):
            out.append(
                {
                    "coverage_type": coverage_type,
                    "key": key,
                    "planned_count": counts[key],
                    "target_count": counts[key],
                    "target_met": True,
                }
            )
    return out


def write_preflight(path: Path) -> None:
    baseline_text = CURRENT_BASELINE.read_text(encoding="utf-8", errors="replace") if CURRENT_BASELINE.exists() else ""
    registry_text = REGISTRY.read_text(encoding="utf-8", errors="replace") if REGISTRY.exists() else ""
    staged_pinn = git_value(ROOT, ["diff", "--cached", "--name-only"])
    staged_comsol = git_value(COMSOL_ROOT, ["diff", "--cached", "--name-only"])
    checks = {
        "pinn_root": git_value(ROOT, ["rev-parse", "--show-toplevel"]).replace("\\", "/").endswith("/PINN_project"),
        "comsol_root": git_value(COMSOL_ROOT, ["rev-parse", "--show-toplevel"]).replace("\\", "/").endswith("/COMSOL_Multiphysics_MCP"),
        "current_baseline_20_85": "20.85" in baseline_text and SOURCE_DATASET_ID in baseline_text,
        "source_registry_entry": SOURCE_DATASET_ID in registry_text,
        "source_manifest_exists": SOURCE_MANIFEST.exists(),
        "forbidden_staged_clear": not any(token in (staged_pinn + "\n" + staged_comsol).lower() for token in ["data/", ".npz", ".mph", "checkpoints/", "results/previews", "notes/"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "surface RBC targeted expansion preflight summary",
                "",
                *[f"{key}: {value}" for key, value in checks.items()],
                "target_dataset_id: " + DATASET_ID,
                "source_dataset_id: " + SOURCE_DATASET_ID,
                "boundary: top-up source only; no training; CURRENT_BASELINE.md unchanged by this script",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()):
        raise RuntimeError(f"preflight failed: {checks}")


def write_summary(path: Path, rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "surface RBC targeted expansion plan summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"source_dataset_id: {SOURCE_DATASET_ID}",
        "dataset_role: topup_source",
        f"planned_N: {len(rows)}",
        f"split_counts: {dict(Counter(row['split'] for row in rows))}",
        f"targeted_role_counts: {dict(Counter(row['targeted_role'] for row in rows))}",
        f"depth_counts: {dict(Counter(row['depth_bin'] for row in rows))}",
        f"aspect_counts: {dict(Counter(row['aspect_bin'] for row in rows))}",
        f"curvature_counts: {dict(Counter(row['curvature_template'] for row in rows))}",
        f"edge_position_counts: {dict(Counter(row['edge_position_bin'] for row in rows))}",
        "sensor_z_m: 0.008",
        "axis_order: [Bx, By, Bz]",
        "scan_line_y_m: [-0.001, 0.0, 0.001]",
        "sensor_x_count: 201",
        "",
        "Boundary:",
        "- exact_piao_rbc=False; RBC-style / Piao-inspired approximation only.",
        "- This plan excludes multi-pit, internal/buried, and free-form polygon branches.",
        "- This plan creates only a top-up source pack; assembly with v3_240 is a later training gate.",
        "",
        "Coverage gates:",
    ]
    lines.extend(f"- {row['coverage_type']} {row['key']}: {row['planned_count']}/{row['target_count']} met={row['target_met']}" for row in coverage_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit(f"Run from PINN_project root: {ROOT}")
    check_no_overwrite([args.preflight_summary, args.summary, args.plan_csv, args.coverage_csv], args.overwrite)
    write_preflight(args.preflight_summary)
    rows = build_plan(args.temp_mesh_dir)
    coverage_rows = expected_coverage(rows)
    write_csv(args.plan_csv, rows, PLAN_FIELDS)
    write_csv(args.coverage_csv, coverage_rows, COVERAGE_FIELDS)
    write_summary(args.summary, rows, coverage_rows)
    print(f"wrote {args.preflight_summary}")
    print(f"wrote {args.plan_csv}")
    print(f"wrote {args.coverage_csv}")
    print(f"wrote {args.summary}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
