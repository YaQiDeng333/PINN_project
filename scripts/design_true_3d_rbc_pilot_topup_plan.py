#!/usr/bin/env python
"""Design the 20.72 top-up plan for the true-3D RBC pilot pack."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

DEFAULT_PLAN = ROOT / "results/metrics/true_3d_rbc_pilot_pack_plan.csv"
DEFAULT_AUDIT = ROOT / "results/metrics/true_3d_rbc_pilot_partial_audit.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_topup_plan_summary.txt"
DEFAULT_TOPUP_PLAN = ROOT / "results/metrics/true_3d_rbc_pilot_topup_plan.csv"
DEFAULT_COVERAGE = ROOT / "results/metrics/true_3d_rbc_pilot_topup_expected_coverage.csv"

TOPUP_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_topup_20_72"
PARTIAL_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_partial_20_71"
ASSEMBLED_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled"
MESH_TEMP_DIR = ROOT / "data/comsol_mfl/generated/temp_true_3d_rbc_pilot_topup_meshes_v1"

EXTRA_FIELDS = [
    "source_stage",
    "intended_split",
    "replacement_for_sample_id",
    "replacement_reason",
    "source_sample_id",
]

COVERAGE_FIELDS = ["group_key", "group_value", "existing_success", "topup_planned", "assembled_expected", "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.72 true-3D RBC pilot top-up plan.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--topup-plan", type=Path, default=DEFAULT_TOPUP_PLAN)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def adjusted_replacement(row: dict[str, str], suffix: str) -> dict[str, str]:
    new_row = dict(row)
    new_row["sample_id"] = f"rbc_topup_repl_{suffix}_{row['curvature_template']}"
    new_row["replacement_for_sample_id"] = row["sample_id"]
    new_row["replacement_reason"] = "bounded retry replacement id: preserves original profile/depth label; adjust only if this row times out again"
    return new_row


def custom_replacement_row(
    sample_id: str,
    split: str,
    depth_bin: str,
    size_bin: str,
    aspect_bin: str,
    curvature_template: str,
    l_m: float,
    w_m: float,
    d_m: float,
    wld: float,
    wwd: float,
    wlw: float,
    replacement_reason: str,
) -> dict[str, Any]:
    spec = base_plan.PilotSpec(
        depth_bin=depth_bin,
        size_bin=size_bin,
        aspect_bin=aspect_bin,
        curvature_template=curvature_template,
        split=split,
        L_m=l_m,
        W_m=w_m,
        D_m=d_m,
        wLD=wld,
        wWD=wwd,
        wLW=wlw,
    )
    sample = base_plan.smoke_plan.SmokeSample(
        sample_id=sample_id,
        split_tag=split,
        L_m=l_m,
        W_m=w_m,
        D_m=d_m,
        wLD=wld,
        wWD=wwd,
        wLW=wlw,
        center_x_m=0.0,
        center_y_m=0.0,
        angle_deg=0.0,
        notes=f"20.72 train safety replacement; {replacement_reason}",
    )
    mask_x = np.linspace(base_plan.MASK_X_START_M, base_plan.MASK_X_STOP_M, base_plan.MASK_WIDTH)
    mask_y = np.linspace(base_plan.MASK_Y_START_M, base_plan.MASK_Y_STOP_M, base_plan.MASK_HEIGHT)
    pixel_area = float((mask_x[1] - mask_x[0]) * (mask_y[1] - mask_y[0]))
    _, _, depth_grid = base_plan.smoke_plan.build_profile_depth_grid(sample)
    depth_map = base_plan.smoke_plan.build_depth_map(sample, mask_x, mask_y)
    projection_threshold = max(1.0e-6, 0.01 * sample.D_m)
    mask = (depth_map >= projection_threshold).astype(np.uint8)
    connected_components = base_plan.smoke_plan.connected_component_count(mask)
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
    rbc_params = {"L_m": l_m, "W_m": w_m, "D_m": d_m, "wLD": wld, "wWD": wwd, "wLW": wlw}
    profile_pose_to_comsol = {
        "mesh_units": "m",
        "comsol_coordinates": "x/y on steel surface plane; top cap z=0; bottom surface z=-depth",
        "center_x_m": sample.center_x_m,
        "center_y_m": sample.center_y_m,
        "angle_rad": 0.0,
        "steel_surface_z_m": 0.0,
    }
    geometry_params = {
        "dataset_id": TOPUP_DATASET_ID,
        "sample_id": sample_id,
        "profile_type": "rbc_style_symmetric_pit",
        "geometry_method": "imported_watertight_mesh_solid",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "rbc_formula_status": base_plan.RBC_FORMULA_STATUS,
        "rbc_params": rbc_params,
        "profile_pose": profile_pose,
        "depth_bin": depth_bin,
        "size_bin": size_bin,
        "aspect_bin": aspect_bin,
        "curvature_template": curvature_template,
        "projection_threshold_m": projection_threshold,
        "target_footprint_area_m2": footprint_area,
        "target_volume_proxy_m3": volume_proxy,
        "mesh_source": "triangulated_depth_grid",
        "projected_mask_role": "2D comparator only; 3D label uses rbc_params and depth grid/map",
        "replacement_reason": replacement_reason,
    }
    return {
        "dataset_id": TOPUP_DATASET_ID,
        "schema_version": base_plan.SCHEMA_VERSION,
        "route": base_plan.ROUTE,
        "sample_id": sample_id,
        "split": split,
        "split_tag": split,
        "depth_bin": depth_bin,
        "size_bin": size_bin,
        "aspect_bin": aspect_bin,
        "curvature_bin": curvature_template,
        "curvature_template": curvature_template,
        "profile_type": "rbc_style_symmetric_pit",
        "exact_piao_rbc": "False",
        "rbc_style_approximation": "True",
        "rbc_formula_status": base_plan.RBC_FORMULA_STATUS,
        "geometry_method": "imported_watertight_mesh_solid",
        "generated_geometry_type": "rbc_style_imported_watertight_mesh_candidate",
        "L_m": l_m,
        "W_m": w_m,
        "D_m": d_m,
        "wLD": wld,
        "wWD": wwd,
        "wLW": wlw,
        "center_x_m": 0.0,
        "center_y_m": 0.0,
        "angle_rad": 0.0,
        "angle_deg": 0.0,
        "sensor_z_m": 0.008,
        "scan_line_y_json": json.dumps([-0.001, 0.0, 0.001], separators=(",", ":")),
        "axis_names_json": json.dumps(["Bx", "By", "Bz"], separators=(",", ":")),
        "axis_expressions_json": json.dumps(["mf.Bx", "mf.By", "mf.Bz"], separators=(",", ":")),
        "profile_pose_json": json.dumps(profile_pose, sort_keys=True, separators=(",", ":")),
        "rbc_params_json": json.dumps(rbc_params, sort_keys=True, separators=(",", ":")),
        "profile_depth_grid_shape_json": json.dumps(list(depth_grid.shape), separators=(",", ":")),
        "profile_depth_grid_m_json": json.dumps(base_plan.rounded_list(depth_grid, decimals=9), separators=(",", ":")),
        "profile_depth_map_xy_shape_json": json.dumps(list(depth_map.shape), separators=(",", ":")),
        "profile_depth_map_xy_m_json": json.dumps(base_plan.rounded_list(depth_map, decimals=9), separators=(",", ":")),
        "projected_mask_2d_shape_json": json.dumps(list(mask.shape), separators=(",", ":")),
        "projected_mask_2d_json": json.dumps(mask.astype(int).tolist(), separators=(",", ":")),
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
        "profile_pose_to_comsol_json": json.dumps(profile_pose_to_comsol, sort_keys=True, separators=(",", ":")),
        "steel_surface_z_m": 0.0,
        "steel_x_min_m": -0.05,
        "steel_x_max_m": 0.05,
        "steel_y_min_m": -0.02,
        "steel_y_max_m": 0.02,
        "steel_z_min_m": -0.01,
        "steel_z_max_m": 0.0,
        "temp_mesh_output_path": str(MESH_TEMP_DIR / f"{sample_id}.stl"),
        "geometry_params_json": json.dumps(geometry_params, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "allowed_use": "schema_validation, assembly_input",
        "forbidden_use": "automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate",
        "notes": f"20.72 train safety replacement; {replacement_reason}",
        "source_stage": "20.72_topup",
        "intended_split": split,
        "replacement_for_sample_id": "",
        "replacement_reason": replacement_reason,
        "source_sample_id": sample_id,
    }


def normalize_topup_row(row: dict[str, str], source_sample_id: str) -> dict[str, Any]:
    out = dict(row)
    out["source_stage"] = "20.72_topup"
    out["dataset_id"] = TOPUP_DATASET_ID
    out["intended_split"] = out.get("split", out.get("intended_split", ""))
    out["exact_piao_rbc"] = "False"
    out["rbc_style_approximation"] = "True"
    out["geometry_method"] = "imported_watertight_mesh_solid"
    out["source_sample_id"] = source_sample_id
    if "replacement_for_sample_id" not in out:
        out["replacement_for_sample_id"] = ""
    if "replacement_reason" not in out:
        out["replacement_reason"] = ""
    out["temp_mesh_output_path"] = str(MESH_TEMP_DIR / f"{out['sample_id']}.stl")
    params = json.loads(out["geometry_params_json"])
    params.update(
        {
            "dataset_id": TOPUP_DATASET_ID,
            "partial_source_dataset_id": PARTIAL_DATASET_ID,
            "assembled_dataset_id": ASSEMBLED_DATASET_ID,
            "source_stage": "20.72_topup",
            "source_sample_id": source_sample_id,
            "replacement_for_sample_id": out["replacement_for_sample_id"],
            "replacement_reason": out["replacement_reason"],
            "exact_piao_rbc": False,
            "rbc_style_approximation": True,
        }
    )
    out["geometry_params_json"] = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return out


def select_topup_rows(plan_rows: list[dict[str, str]], audit_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    plan_by_id = {row["sample_id"]: row for row in plan_rows}
    audit_by_id = {row["sample_id"]: row for row in audit_rows}
    selected_ids: list[str] = []
    for row in plan_rows:
        audit = audit_by_id[row["sample_id"]]
        if audit["classification"] == "success":
            continue
        if row["curvature_template"] in {"LD_dominant", "WD_dominant"}:
            selected_ids.append(row["sample_id"])
    for row in plan_rows:
        audit = audit_by_id[row["sample_id"]]
        if audit["classification"] == "success":
            continue
        if row["curvature_template"] == "boxy" and row["depth_bin"] == "deep":
            selected_ids.append(row["sample_id"])
    topup: list[dict[str, Any]] = []
    for sample_id in selected_ids:
        topup.append(normalize_topup_row(plan_by_id[sample_id], sample_id))
    timeout_rows = [row for row in audit_rows if row["classification"] == "timeout_or_bounded_skip"]
    for row in timeout_rows:
        source = plan_by_id[row["sample_id"]]
        replacement = adjusted_replacement(source, row["sample_id"].split("_")[2])
        topup.append(normalize_topup_row(replacement, source["sample_id"]))
    topup.extend(
        [
            custom_replacement_row(
                "rbc_topup_safety_train_wd_medium_balanced",
                "train",
                "medium",
                "medium_balanced",
                "balanced",
                "WD_dominant",
                0.0205,
                0.0115,
                0.0028,
                0.83,
                1.16,
                1.02,
                "extra train replacement to satisfy assembled train split after WD failures",
            ),
            custom_replacement_row(
                "rbc_topup_safety_train_ld_medium_balanced",
                "train",
                "medium",
                "medium_balanced",
                "balanced",
                "LD_dominant",
                0.0200,
                0.0110,
                0.0029,
                1.16,
                0.82,
                0.98,
                "extra train replacement to satisfy assembled train split after LD failure",
            ),
            custom_replacement_row(
                "rbc_topup_safety_train_boxy_medium_balanced",
                "train",
                "medium",
                "medium_balanced",
                "balanced",
                "boxy",
                0.0208,
                0.0112,
                0.0030,
                1.05,
                1.05,
                1.12,
                "extra train replacement hedge for top-up solver failures",
            ),
        ]
    )
    return topup


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.topup_plan, args.coverage], args.overwrite)
    plan_rows = read_csv(args.plan_csv)
    audit_rows = read_csv(args.audit_csv)
    topup_rows = select_topup_rows(plan_rows, audit_rows)
    if len(topup_rows) < 30:
        raise RuntimeError(f"expected at least 30 top-up rows, got {len(topup_rows)}")
    existing_success = [row for row in audit_rows if row["classification"] == "success"]
    existing_keys = {
        (row["L_m"], row["W_m"], row["D_m"], row["wLD"], row["wWD"], row["wLW"])
        for row in existing_success
    }
    topup_keys = {
        (row["L_m"], row["W_m"], row["D_m"], row["wLD"], row["wWD"], row["wLW"])
        for row in topup_rows
    }
    if len(topup_keys) != len(topup_rows):
        raise RuntimeError("duplicate full six-parameter key within top-up plan")
    if existing_keys & topup_keys:
        raise RuntimeError("top-up plan duplicates a successful existing full six-parameter key")
    if len({row["sample_id"] for row in topup_rows}) != len(topup_rows):
        raise RuntimeError("duplicate top-up sample_id")
    split_counts = Counter(row["split"] for row in topup_rows)
    curvature_counts = Counter(row["curvature_template"] for row in topup_rows)
    depth_counts = Counter(row["depth_bin"] for row in topup_rows)
    size_counts = Counter(row["size_bin"] for row in topup_rows)
    if split_counts.get("train", 0) < 20 or split_counts.get("val", 0) != 5 or split_counts.get("test", 0) != 5:
        raise RuntimeError(f"unexpected top-up split counts: {dict(split_counts)}")
    if curvature_counts.get("LD_dominant", 0) < 12 or curvature_counts.get("WD_dominant", 0) < 12:
        raise RuntimeError(f"LD/WD top-up coverage incomplete: {dict(curvature_counts)}")
    source_fields = list(plan_rows[0].keys())
    output_fields = list(dict.fromkeys(source_fields + EXTRA_FIELDS))
    write_csv(args.topup_plan, topup_rows, output_fields)
    coverage_rows: list[dict[str, Any]] = []
    targets = {
        "split": {"train": 40, "val": 10, "test": 10},
        "curvature_template": {"sharp": 12, "round": 12, "boxy": 12, "LD_dominant": 12, "WD_dominant": 12},
        "depth_bin": {"shallow": 20, "medium": 20, "deep": 20},
        "size_bin": {"small_compact": 15, "medium_balanced": 15, "large_wide": 15, "elongated": 15},
    }
    for key, target_map in targets.items():
        existing_counter = Counter(row["plan_split" if key == "split" else key] for row in existing_success)
        topup_counter = Counter(row["split" if key == "split" else key] for row in topup_rows)
        for value, target in target_map.items():
            coverage_rows.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "existing_success": existing_counter.get(value, 0),
                    "topup_planned": topup_counter.get(value, 0),
                    "assembled_expected": existing_counter.get(value, 0) + topup_counter.get(value, 0),
                    "target": target,
                }
            )
    write_csv(args.coverage, coverage_rows, COVERAGE_FIELDS)
    lines = [
        "20.72 true 3D RBC pilot top-up plan summary",
        "",
        f"topup_planned_rows: {len(topup_rows)}",
        f"topup_split_counts: {dict(split_counts)}",
        f"topup_curvature_counts: {dict(curvature_counts)}",
        f"topup_depth_counts: {dict(depth_counts)}",
        f"topup_size_counts: {dict(size_counts)}",
        "replacement_rows: "
        + ", ".join(row["sample_id"] for row in topup_rows if row["replacement_for_sample_id"]),
        "",
        "Strategy:",
        "- 24 rows cover all missing LD_dominant and WD_dominant original plan rows.",
        "- 4 rows cover the not-attempted deep boxy cells.",
        "- 2 rows are replacement-id bounded retries for the bounded-timeout sharp/round rows; profile/depth labels are preserved.",
        "- 3 train safety replacements use newly generated profile/depth labels to protect the assembled minimum split gate.",
        "- exact_piao_rbc remains False; rbc_style_approximation remains True.",
        "- No high-layer fallback is introduced.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
