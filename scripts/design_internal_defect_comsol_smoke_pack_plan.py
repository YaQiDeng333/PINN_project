#!/usr/bin/env python
"""20.99 internal / buried defect COMSOL smoke pack 设计脚本。

本脚本只写 plan CSV 和 summary，不运行 COMSOL，不读取或生成 NPZ。
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/internal_defect_comsol_smoke_pack_plan_summary.txt"
PLAN_CSV = ROOT / "results/metrics/internal_defect_comsol_smoke_pack_plan.csv"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_root() -> str:
    proc = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, capture_output=True, check=True)
    return proc.stdout.strip().replace("\\", "/")


def smoke_rows() -> list[dict[str, Any]]:
    shapes = ["internal_ellipsoid", "internal_cuboid", "sphere_like"]
    depths = [
        ("shallow", 0.0015),
        ("medium", 0.0030),
        ("deep", 0.0050),
        ("deep_plus", 0.0070),
    ]
    rows: list[dict[str, Any]] = []
    sample_idx = 1
    for shape in shapes:
        for depth_bin, burial_depth_m in depths:
            rows.append(
                {
                    "sample_slot": f"internal_smoke_{sample_idx:02d}",
                    "shape_type": shape,
                    "burial_depth_bin": depth_bin,
                    "burial_depth_m": burial_depth_m,
                    "center_x_m": 0.0,
                    "center_y_m": 0.0,
                    "center_z_definition": "measured_from_scan_surface_to_defect_center",
                    "L_m": 0.006 if shape != "sphere_like" else 0.004,
                    "W_m": 0.0035 if shape != "sphere_like" else 0.004,
                    "D_m_or_cavity_size_m": 0.002,
                    "sensor_z_m": 0.008,
                    "scan_line_y_m": "[-0.001,0.0,0.001]",
                    "sensor_x_count": 201,
                    "axis_order": "[Bx,By,Bz]",
                    "no_defect_reference_required": True,
                    "output_required": "b_defect,b_no_defect,delta_b,Bx,By,Bz",
                    "label_required": "L_m,W_m,D_m_or_cavity_size_m,burial_depth_m,defect_center_xyz_m,shape_type,ground_truth_method",
                    "split_recommendation": "smoke_only_no_training_split",
                }
            )
            sample_idx += 1
    return rows


def main() -> int:
    if git_root() != str(ROOT).replace("\\", "/"):
        raise SystemExit(f"错误：当前仓库不是 PINN_project: {ROOT}")
    rows = smoke_rows()
    write_csv(PLAN_CSV, rows)
    shape_counts = {shape: sum(1 for row in rows if row["shape_type"] == shape) for shape in sorted({row["shape_type"] for row in rows})}
    depth_counts = {depth: sum(1 for row in rows if row["burial_depth_bin"] == depth) for depth in sorted({row["burial_depth_bin"] for row in rows})}
    lines = [
        "20.99 internal / buried defect COMSOL smoke pack 设计",
        "",
        "scope: 只做设计，不运行 COMSOL，不生成 data/NPZ，不训练，不更新 CURRENT_BASELINE.md。",
        "recommended_size: 12 samples",
        "minimum_size: 6 samples",
        "selected_plan_size: 12 samples",
        "shape_type_coverage: " + ", ".join(f"{key}={value}" for key, value in shape_counts.items()),
        "burial_depth_coverage: " + ", ".join(f"{key}={value}" for key, value in depth_counts.items()),
        "sensor_z_m: 0.008 nominal smoke first; later liftoff variation should be a separate pack.",
        "axis_order: [Bx, By, Bz]",
        "scan_line_y_m: [-0.001, 0.0, 0.001]",
        "sensor_x_count: 201",
        "",
        "required_outputs:",
        "- `b_defect`, `b_no_defect`, `delta_b=b_defect-b_no_defect`",
        "- three-axis `Bx/By/Bz`",
        "- labels: `L_m`, `W_m`, `D_m_or_cavity_size_m`, `burial_depth_m`, `defect_center_xyz_m`, `shape_type`, `ground_truth_method`",
        "",
        "route_boundary:",
        "- 该 pack 是 internal defect feasibility smoke，不是 surface RBC baseline top-up。",
        "- 如果 smoke 通过，下一步才设计 internal-specific training gate。",
        "- Bz-only 只能作为低能力诊断分支，不是默认主线。",
        "",
        f"plan_csv: {PLAN_CSV.relative_to(ROOT)}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
