#!/usr/bin/env python
"""Design the 21.1 internal / buried defect COMSOL pilot pack.

The script writes the pilot plan, preflight summary, and plan summary only.
It does not run COMSOL, generate data, train models, or update the baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SMOKE_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_smoke_pack_v1.manifest.json"
SCHEMA = ROOT / "INTERNAL_DEFECT_SCHEMA.md"
PREFLIGHT_SUMMARY = ROOT / "results/summaries/internal_defect_pilot_pack_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/internal_defect_pilot_pack_plan_summary.txt"
PLAN_CSV = ROOT / "results/metrics/internal_defect_pilot_pack_plan.csv"

DATASET_ID = "comsol_internal_defect_pilot_pack_v1"
SOURCE_DATASET_ID = "comsol_internal_defect_smoke_pack_v1"
AXIS_ORDER = ["Bx", "By", "Bz"]
SCAN_LINE_Y_M = [-0.001, 0.0, 0.001]
SENSOR_X_START_M = -0.04
SENSOR_X_STOP_M = 0.04
SENSOR_X_COUNT = 201
SENSOR_Z_M = 0.008
EXPECTED_LABEL_FIELDS = [
    "L_m",
    "W_m",
    "D_m",
    "burial_depth_m",
    "depth_to_surface_m",
    "defect_center_xyz_m",
    "shape_type",
    "aspect_bin",
    "ground_truth_method",
    "cavity_internal",
]

CSV_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "burial_depth_level",
    "size_level",
    "aspect_bin",
    "center_xyz_m",
    "center_x_m",
    "center_y_m",
    "center_z_m",
    "L_m",
    "W_m",
    "D_m",
    "D_m_or_cavity_size_m",
    "burial_depth_m",
    "depth_to_surface_m",
    "scan_surface",
    "sensor_z_m",
    "axis_order",
    "scan_line_y_m",
    "sensor_x_count",
    "sensor_x_start_m",
    "sensor_x_stop_m",
    "expected_label_fields",
    "geometry_params_json",
    "ground_truth_method",
    "cavity_internal",
    "source_dataset_id",
    "notes",
]

DEPTH_TO_SURFACE = {
    "shallow": 0.0008,
    "medium": 0.0020,
    "deep": 0.0032,
    "deep_plus": 0.0042,
}
VERTICAL_SIZE = {
    "small": 0.0010,
    "medium": 0.0012,
    "large": 0.0014,
}
HORIZONTAL_BASE = {
    "small": 0.0020,
    "medium": 0.0035,
    "large": 0.0050,
}
ASPECT_MULTIPLIERS = {
    "compact": (1.0, 1.0),
    "elongated_x": (1.6, 0.75),
    "elongated_y": (0.75, 1.6),
}
SHAPE_CENTER_X = {
    "internal_sphere": -0.012,
    "internal_ellipsoid": 0.0,
    "internal_cuboid": 0.012,
}
SPHERE_LATERAL_VARIANTS = [-0.003, 0.003]
Y_BY_BURIAL = {
    "shallow": -0.003,
    "medium": -0.001,
    "deep": 0.001,
    "deep_plus": 0.003,
}
Y_BY_ASPECT = {
    "compact": -0.003,
    "elongated_x": 0.0,
    "elongated_y": 0.003,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 21.1 internal defect pilot pack.")
    parser.add_argument("--preflight-summary", type=Path, default=PREFLIGHT_SUMMARY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def git_root() -> str:
    proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, capture_output=True, check=True)
    return proc.stdout.strip().replace("\\", "/")


def git_status(path: Path) -> list[str]:
    proc = subprocess.run(["git", "status", "--short"], cwd=path, text=True, capture_output=True, check=True)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def split_for_index(index: int) -> str:
    if index <= 64:
        return "train"
    if index <= 80:
        return "val"
    return "test"


def dims_for(shape_type: str, size_level: str, aspect_bin: str) -> tuple[float, float, float]:
    d_m = VERTICAL_SIZE[size_level]
    if shape_type == "internal_sphere":
        return d_m, d_m, d_m
    h = HORIZONTAL_BASE[size_level]
    mx, my = ASPECT_MULTIPLIERS[aspect_bin]
    return h * mx, h * my, d_m


def add_row(rows: list[dict[str, Any]], shape_type: str, burial_level: str, size_level: str, aspect_bin: str, center_y_m: float) -> None:
    index = len(rows) + 1
    l_m, w_m, d_m = dims_for(shape_type, size_level, aspect_bin)
    depth_to_surface_m = DEPTH_TO_SURFACE[burial_level]
    if depth_to_surface_m <= 0 or depth_to_surface_m + d_m > 0.0056:
        raise ValueError(f"invalid internal cavity depth: {shape_type}/{burial_level}/{size_level}/{aspect_bin}")
    center_x_m = SHAPE_CENTER_X[shape_type]
    center_z_m = -(depth_to_surface_m + d_m / 2.0)
    center_xyz = [center_x_m, center_y_m, center_z_m]
    geometry_params = {
        "steel_block_size_m": [0.08, 0.02, 0.006],
        "steel_block_center_m": [0.0, 0.0, -0.003],
        "top_scan_surface_z_m": 0.0,
        "aspect_bin": aspect_bin,
        "size_level": size_level,
        "burial_depth_level": burial_level,
        "cavity_internal_rule": "depth_to_surface_m > 0 and depth_to_surface_m + D_m <= 0.0056",
    }
    rows.append(
        {
            "sample_id": f"internal_pilot_{index:03d}",
            "split": split_for_index(index),
            "shape_type": shape_type,
            "burial_depth_level": burial_level,
            "size_level": size_level,
            "aspect_bin": aspect_bin,
            "center_xyz_m": json_compact(center_xyz),
            "center_x_m": f"{center_x_m:.9g}",
            "center_y_m": f"{center_y_m:.9g}",
            "center_z_m": f"{center_z_m:.9g}",
            "L_m": f"{l_m:.9g}",
            "W_m": f"{w_m:.9g}",
            "D_m": f"{d_m:.9g}",
            "D_m_or_cavity_size_m": f"{d_m:.9g}",
            "burial_depth_m": f"{depth_to_surface_m:.9g}",
            "depth_to_surface_m": f"{depth_to_surface_m:.9g}",
            "scan_surface": "top_z_0",
            "sensor_z_m": f"{SENSOR_Z_M:.9g}",
            "axis_order": json_compact(AXIS_ORDER),
            "scan_line_y_m": json_compact(SCAN_LINE_Y_M),
            "sensor_x_count": SENSOR_X_COUNT,
            "sensor_x_start_m": SENSOR_X_START_M,
            "sensor_x_stop_m": SENSOR_X_STOP_M,
            "expected_label_fields": json_compact(EXPECTED_LABEL_FIELDS),
            "geometry_params_json": json_compact(geometry_params),
            "ground_truth_method": "COMSOL_parametric_internal_cavity",
            "cavity_internal": True,
            "source_dataset_id": SOURCE_DATASET_ID,
            "notes": "21.1 internal pilot row; feasibility/training-gate candidate only, not baseline",
        }
    )


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for burial_level in DEPTH_TO_SURFACE:
        for size_level in VERTICAL_SIZE:
            for center_y_m in SPHERE_LATERAL_VARIANTS:
                add_row(rows, "internal_sphere", burial_level, size_level, "compact", center_y_m)
    for shape_type in ["internal_ellipsoid", "internal_cuboid"]:
        for burial_level in DEPTH_TO_SURFACE:
            for size_level in VERTICAL_SIZE:
                for aspect_bin in ASPECT_MULTIPLIERS:
                    add_row(rows, shape_type, burial_level, size_level, aspect_bin, Y_BY_ASPECT[aspect_bin])
    if len(rows) != 96:
        raise RuntimeError(f"expected 96 pilot rows, got {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_preflight(path: Path) -> None:
    smoke = json.loads(SMOKE_MANIFEST.read_text(encoding="utf-8"))
    comsol_root = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")
    comsol_status = git_status(comsol_root)
    lines = [
        "21.1 内部/埋藏缺陷 pilot pack preflight",
        "",
        f"pinn_root: {ROOT}",
        f"pinn_git_root: {git_root()}",
        f"comsol_root: {comsol_root}",
        "method_agent: 21.1 仍是 internal/buried 独立分支，不混入 surface RBC baseline。",
        "comsol_agent: 21.0 已证明 sphere/ellipsoid/cuboid 的 Boolean、mesh、solve、Bx/By/Bz 导出可跑通；21.1 复用同一建模链路。",
        "pinn_schema_agent: INTERNAL_DEFECT_SCHEMA.md 存在；registry/manifest 必须显式 dataset_id，不允许 latest/newest discovery。",
        "experiment_agent: target N=96，fallback N>=72；split 固定 train/val/test=64/16/16。",
        "safety_agent: 禁止提交 data/、NPZ、.mph、raw CSV、checkpoint、preview PNG、notes、temp STL；CURRENT_BASELINE.md 不修改。",
        "",
        f"source_dataset_id: {smoke.get('dataset_id')}",
        f"source_status: {smoke.get('status')}",
        f"source_n_samples: {smoke.get('n_samples')}",
        f"source_validation_script: {smoke.get('validation_script')}",
        "",
        "comsol_dirty_items_kept_unmodified:",
        *(f"- {line}" for line in comsol_status),
        "",
        "stop_conditions:",
        "- PINN_project root mismatch.",
        "- COMSOL_Multiphysics_MCP root mismatch.",
        "- 21.0 smoke manifest is missing or not smoke_generated.",
        "- Plan geometry cannot keep every cavity fully internal.",
        "- Generated data appears in git staging.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]], plan_csv: Path) -> None:
    counters = {
        "shape_type": Counter(row["shape_type"] for row in rows),
        "burial_depth_level": Counter(row["burial_depth_level"] for row in rows),
        "size_level": Counter(row["size_level"] for row in rows),
        "aspect_bin": Counter(row["aspect_bin"] for row in rows),
        "split": Counter(row["split"] for row in rows),
    }
    lines = [
        "21.1 内部/埋藏缺陷 COMSOL pilot pack 计划",
        "",
        f"dataset_id: {DATASET_ID}",
        f"source_dataset_id: {SOURCE_DATASET_ID}",
        "scope: 只生成 internal/buried defect pilot pack；不训练，不更新 CURRENT_BASELINE，不混入 surface RBC baseline。",
        f"target_samples: {len(rows)}",
        "fallback_minimum_samples: 72",
        "fallback_split_minimum: train/val/test = 48/12/12",
        f"sensor_z_m: {SENSOR_Z_M}",
        f"scan_line_y_m: {SCAN_LINE_Y_M}",
        f"axis_order: {AXIS_ORDER}",
        f"sensor_x_count: {SENSOR_X_COUNT}",
        "",
        f"shape_type_counts: {dict(counters['shape_type'])}",
        f"burial_depth_counts: {dict(counters['burial_depth_level'])}",
        f"size_counts: {dict(counters['size_level'])}",
        f"aspect_counts: {dict(counters['aspect_bin'])}",
        f"split_counts: {dict(counters['split'])}",
        "",
        "geometry_rule: top scan surface z=0，钢块内部 z<0；center_z_m = -(depth_to_surface_m + D_m/2)。",
        "internal_gate: 每个样本必须满足 depth_to_surface_m > 0 且 depth_to_surface_m + D_m <= 0.0056。",
        "train_ready_candidate_gate: 成功样本 >=72、schema validation 无 blocker、split 达到 fallback、registry/manifest 显式 gate 完整。",
        f"plan_csv: {plan_csv}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if git_root() != str(ROOT).replace("\\", "/"):
        raise SystemExit(f"错误：当前仓库不是 PINN_project: {ROOT}")
    if not SCHEMA.exists():
        raise FileNotFoundError(SCHEMA)
    if not SMOKE_MANIFEST.exists():
        raise FileNotFoundError(SMOKE_MANIFEST)
    smoke = json.loads(SMOKE_MANIFEST.read_text(encoding="utf-8"))
    if smoke.get("status") != "smoke_generated" or int(smoke.get("n_samples", 0)) != 12:
        raise RuntimeError(f"21.0 smoke manifest is not a full pass: {smoke}")
    check_no_overwrite([args.preflight_summary, args.summary, args.plan_csv], args.overwrite)
    rows = build_rows()
    write_csv(args.plan_csv, rows)
    write_preflight(args.preflight_summary)
    write_summary(args.summary, rows, args.plan_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
