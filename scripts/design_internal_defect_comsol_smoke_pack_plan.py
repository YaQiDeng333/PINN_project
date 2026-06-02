#!/usr/bin/env python
"""Design the 20.99 internal / buried defect COMSOL smoke pack.

This is a design-only script. It writes a summary and CSV plan, but it does not
run COMSOL, train a model, generate data, write NPZ files, or update the current
baseline.
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
SUMMARY = ROOT / "results/summaries/internal_defect_comsol_smoke_pack_plan_summary.txt"
PLAN_CSV = ROOT / "results/metrics/internal_defect_comsol_smoke_pack_plan.csv"
SCHEMA = ROOT / "INTERNAL_DEFECT_SCHEMA.md"

CSV_FIELDS = [
    "sample_slot",
    "shape_type",
    "size_level",
    "burial_depth_level",
    "burial_depth_m",
    "depth_to_surface_m",
    "center_x_m",
    "center_y_m",
    "center_z_m",
    "L_m",
    "W_m",
    "D_m_or_cavity_size_m",
    "sensor_z_m",
    "scan_line_y_m",
    "sensor_x_count",
    "axis_order",
    "requires_no_defect_reference",
    "expected_arrays",
    "required_labels",
    "ground_truth_method",
    "generator_route",
    "comsol_action",
    "training_allowed",
    "baseline_update_allowed",
    "notes",
]

REQUIRED_LABELS = [
    "L_m",
    "W_m",
    "D_m_or_cavity_size_m",
    "burial_depth_m",
    "depth_to_surface_m",
    "defect_center_xyz_m",
    "shape_type",
    "profile_descriptor_or_cavity_mask",
    "ground_truth_method",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design the 20.99 internal defect COMSOL smoke pack.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    return parser.parse_args()


def git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def build_rows() -> list[dict[str, Any]]:
    shape_dims = {
        "internal_ellipsoid": {
            "small": (0.0040, 0.0025, 0.0012),
            "medium": (0.0060, 0.0035, 0.0016),
            "large": (0.0080, 0.0045, 0.0020),
        },
        "internal_cuboid": {
            "small": (0.0035, 0.0035, 0.0012),
            "medium": (0.0050, 0.0050, 0.0016),
            "large": (0.0065, 0.0065, 0.0020),
        },
        "sphere_like": {
            "small": (0.0025, 0.0025, 0.0025),
            "medium": (0.0035, 0.0035, 0.0035),
            "large": (0.0045, 0.0045, 0.0045),
        },
    }
    burial_depths = {
        "shallow": 0.0015,
        "medium": 0.0030,
        "deep": 0.0050,
    }
    pattern = [
        ("shallow", "small", -0.0040, 0.0),
        ("medium", "medium", 0.0, 0.0),
        ("deep", "large", 0.0040, 0.0),
        ("shallow", "medium", 0.0, 0.0010),
    ]

    rows: list[dict[str, Any]] = []
    slot = 1
    for shape_type, dims_by_size in shape_dims.items():
        for burial_level, size_level, center_x_m, center_y_m in pattern:
            l_m, w_m, d_m = dims_by_size[size_level]
            burial_depth_m = burial_depths[burial_level]
            center_z_m = -(burial_depth_m + d_m / 2.0)
            rows.append(
                {
                    "sample_slot": f"internal_smoke_{slot:02d}",
                    "shape_type": shape_type,
                    "size_level": size_level,
                    "burial_depth_level": burial_level,
                    "burial_depth_m": burial_depth_m,
                    "depth_to_surface_m": burial_depth_m,
                    "center_x_m": center_x_m,
                    "center_y_m": center_y_m,
                    "center_z_m": center_z_m,
                    "L_m": l_m,
                    "W_m": w_m,
                    "D_m_or_cavity_size_m": d_m,
                    "sensor_z_m": 0.008,
                    "scan_line_y_m": compact_json([-0.001, 0.0, 0.001]),
                    "sensor_x_count": 201,
                    "axis_order": compact_json(["Bx", "By", "Bz"]),
                    "requires_no_defect_reference": True,
                    "expected_arrays": "b_defect,b_no_defect,delta_b,Bx,By,Bz",
                    "required_labels": ",".join(REQUIRED_LABELS),
                    "ground_truth_method": "COMSOL_parametric_internal_cavity",
                    "generator_route": "internal_cavity_comsol_solid_design_only",
                    "comsol_action": "not_run_in_20_99",
                    "training_allowed": False,
                    "baseline_update_allowed": False,
                    "notes": "Design row only; cavity must remain fully buried below the scan surface.",
                }
            )
            slot += 1
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]], plan_csv: Path) -> None:
    shape_counts = Counter(row["shape_type"] for row in rows)
    burial_counts = Counter(row["burial_depth_level"] for row in rows)
    size_counts = Counter(row["size_level"] for row in rows)
    lines = [
        "20.99 internal / buried defect COMSOL smoke pack plan",
        "",
        "scope: design only; no COMSOL, no training, no data/NPZ generation, no CURRENT_BASELINE update.",
        "recommended_size: 6-12 samples",
        f"selected_plan_size: {len(rows)} samples",
        "shape_types: internal_ellipsoid, internal_cuboid, sphere_like",
        "burial_depth_levels: shallow, medium, deep",
        "required_outputs: Bx/By/Bz, b_defect, b_no_defect, delta_b=b_defect-b_no_defect",
        "required_reference: matched no-defect reference for every defect row",
        "required_labels: " + ", ".join(REQUIRED_LABELS),
        "sensor_z_m: 0.008 nominal first smoke; liftoff sweep belongs to a later pack",
        "axis_order: [Bx, By, Bz]",
        "scan_line_y_m: [-0.001, 0.0, 0.001]",
        "sensor_x_count: 201",
        f"shape_type_coverage: {dict(shape_counts)}",
        f"burial_depth_coverage: {dict(burial_counts)}",
        f"size_level_coverage: {dict(size_counts)}",
        "",
        "generation_gates_for_later_stage:",
        "- closed internal cavity fully inside the specimen",
        "- Boolean subtract succeeds",
        "- mesh and solver succeed",
        "- finite Bx/By/Bz exports are present",
        "- no-defect reference is paired and reusable",
        "- delta_b equality check passes",
        "- internal labels include L/W/D, burial depth, center, shape_type, and ground_truth_method",
        "",
        "route_boundary:",
        "- This pack is not a surface RBC top-up.",
        "- This pack is not train-ready by design.",
        "- Bz-only is a low-capability diagnostic branch only, not the mainline.",
        f"plan_csv: {plan_csv}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if git_root() != ROOT.resolve():
        raise SystemExit(f"wrong repository root: {ROOT}")
    if not SCHEMA.exists():
        raise FileNotFoundError(SCHEMA)
    rows = build_rows()
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 smoke rows, got {len(rows)}")
    write_csv(args.plan_csv, rows)
    write_summary(args.summary, rows, args.plan_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
