#!/usr/bin/env python
"""Design the 21.0 internal / buried defect COMSOL smoke pack.

This script writes only a plan CSV and summary. It does not run COMSOL,
generate data, train models, or update the current baseline.
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
SUMMARY = ROOT / "results/summaries/internal_defect_comsol_smoke_plan_summary.txt"
PLAN_CSV = ROOT / "results/metrics/internal_defect_comsol_smoke_plan.csv"
PREFLIGHT_SUMMARY = ROOT / "results/summaries/internal_defect_comsol_smoke_preflight_summary.txt"

SHAPE_TYPES = ["internal_sphere", "internal_ellipsoid", "internal_cuboid"]
ROW_PATTERN = [
    ("shallow", "small", 0.0010, (0.0030, 0.0020, 0.0010)),
    ("shallow", "large", 0.0015, (0.0060, 0.0040, 0.0018)),
    ("medium", "medium", 0.0030, (0.0050, 0.0030, 0.0014)),
    ("deep", "medium", 0.0040, (0.0050, 0.0030, 0.0014)),
]
EXPECTED_LABEL_FIELDS = [
    "L_m",
    "W_m",
    "D_m",
    "burial_depth_m",
    "depth_to_surface_m",
    "defect_center_xyz_m",
    "shape_type",
    "ground_truth_method",
    "cavity_internal",
]
CSV_FIELDS = [
    "sample_id",
    "shape_type",
    "burial_depth_level",
    "size_level",
    "center_xyz_m",
    "center_x_m",
    "center_y_m",
    "center_z_m",
    "L_m",
    "W_m",
    "D_m",
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
    "ground_truth_method",
    "cavity_internal",
    "split_tag",
    "geometry_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 21.0 internal defect COMSOL smoke pack.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--preflight-summary", type=Path, default=PREFLIGHT_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_root() -> str:
    proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, capture_output=True, check=True)
    return proc.stdout.strip().replace("\\", "/")


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 1
    center_xy = {
        "internal_sphere": (0.0, 0.0),
        "internal_ellipsoid": (-0.004, 0.0),
        "internal_cuboid": (0.004, 0.0),
    }
    for shape_type in SHAPE_TYPES:
        center_x_m, center_y_m = center_xy[shape_type]
        for burial_level, size_level, depth_to_surface_m, dims in ROW_PATTERN:
            l_m, w_m, d_m = dims
            if shape_type == "internal_sphere":
                diameter_by_size = {"small": 0.0010, "medium": 0.0014, "large": 0.0018}
                l_m = w_m = d_m = diameter_by_size[size_level]
            center_z_m = -(float(depth_to_surface_m) + float(d_m) / 2.0)
            rows.append(
                {
                    "sample_id": f"internal_smoke_{index:03d}",
                    "shape_type": shape_type,
                    "burial_depth_level": burial_level,
                    "size_level": size_level,
                    "center_xyz_m": json_compact([center_x_m, center_y_m, center_z_m]),
                    "center_x_m": center_x_m,
                    "center_y_m": center_y_m,
                    "center_z_m": center_z_m,
                    "L_m": l_m,
                    "W_m": w_m,
                    "D_m": d_m,
                    "burial_depth_m": depth_to_surface_m,
                    "depth_to_surface_m": depth_to_surface_m,
                    "scan_surface": "top_z_0",
                    "sensor_z_m": 0.008,
                    "axis_order": json_compact(["Bx", "By", "Bz"]),
                    "scan_line_y_m": json_compact([-0.001, 0.0, 0.001]),
                    "sensor_x_count": 201,
                    "sensor_x_start_m": -0.04,
                    "sensor_x_stop_m": 0.04,
                    "expected_label_fields": json_compact(EXPECTED_LABEL_FIELDS),
                    "ground_truth_method": "COMSOL_parametric_internal_cavity",
                    "cavity_internal": True,
                    "split_tag": "smoke_only_no_training_split",
                    "geometry_notes": (
                        "Internal cavity must remain fully buried inside the steel block; "
                        "center_z_m assumes top scan surface z=0 and steel interior z<0."
                    ),
                }
            )
            index += 1
    return rows


def write_preflight(path: Path) -> None:
    lines = [
        "21.0 内部/埋藏缺陷 COMSOL smoke pack preflight",
        "",
        "subagent_support: available",
        "method_agent: internal defect 是独立 feasibility 分支，必须显式记录 burial_depth_m 和 defect_center_xyz_m；本轮不能称为 baseline。",
        "comsol_agent: steel block、material/domain/solver、sensor grid 和 Bx/By/Bz export 可复用；ellipsoid cavity 是主要几何风险。",
        "pinn_schema_agent: registry/manifest 必须使用显式 dataset_id，并禁止 latest/newest 自动发现。",
        "experiment_agent: target N=12，partial smoke 最低 N>=6，不建立训练 split。",
        "safety_agent: 禁止 staging data、NPZ、.mph、raw CSV、checkpoint、preview、notes、temp STL、CURRENT_BASELINE.md 和 baseline docs。",
        "",
        "stop_conditions:",
        "- PINN_project root mismatch.",
        "- COMSOL_Multiphysics_MCP root mismatch.",
        "- INTERNAL_DEFECT_SCHEMA.md missing.",
        "- COMSOL geometry cannot establish a fully internal cavity.",
        "- Generated data appears in git staging.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]], plan_csv: Path) -> None:
    shape_counts = Counter(row["shape_type"] for row in rows)
    burial_counts = Counter(row["burial_depth_level"] for row in rows)
    size_counts = Counter(row["size_level"] for row in rows)
    lines = [
        "21.0 内部/埋藏缺陷 COMSOL smoke pack 计划",
        "",
        "scope: 仅做 internal defect feasibility smoke；不是训练集，不是 baseline，也不是 surface RBC top-up。",
        f"target_samples: {len(rows)}",
        "minimum_partial_samples: 6",
        "sensor_z_m: 0.008",
        "scan_line_y_m: [-0.001, 0.0, 0.001]",
        "axis_order: [Bx, By, Bz]",
        "sensor_x_count: 201",
        f"shape_type_coverage: {dict(shape_counts)}",
        f"burial_depth_coverage: {dict(burial_counts)}",
        f"size_level_coverage: {dict(size_counts)}",
        "coordinate_convention: 扫描上表面 z=0，钢块内部 z<0，center_z_m = -(depth_to_surface_m + D_m/2)。",
        "",
        "required_label_fields: " + ", ".join(EXPECTED_LABEL_FIELDS),
        "",
        "pass_gates:",
        "- smoke_generated: 12/12 行成功，三种 shape 和三种 burial depth 均通过 schema validation。",
        "- partial_smoke_generated: 至少 6 行成功且成功行通过 validation，但覆盖不完整。",
        "- blocked: 成功少于 6 行、Boolean/mesh/solve 系统性失败、缺 Bx/By/Bz、delta check 失败，或 generated data 被 staged。",
        "",
        "ellipsoid_note: 优先尝试 native COMSOL ellipsoid；不可用时记录失败，不 silent skip。",
        f"plan_csv: {plan_csv}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if git_root() != str(ROOT).replace("\\", "/"):
        raise SystemExit(f"错误：当前仓库不是 PINN_project: {ROOT}")
    if not (ROOT / "INTERNAL_DEFECT_SCHEMA.md").exists():
        raise FileNotFoundError(ROOT / "INTERNAL_DEFECT_SCHEMA.md")
    check_no_overwrite([args.summary, args.plan_csv, args.preflight_summary], args.overwrite)
    rows = build_rows()
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 smoke plan rows, got {len(rows)}")
    write_csv(args.plan_csv, rows)
    write_preflight(args.preflight_summary)
    write_summary(args.summary, rows, args.plan_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
