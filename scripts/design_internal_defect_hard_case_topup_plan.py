#!/usr/bin/env python
"""Design the 22.2 targeted internal hard-case top-up plan.

The output is a COMSOL-ready design matrix for a future 22.2b generation run.
This script does not run COMSOL, train, or create/modify data/NPZ files.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, write_csv


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
FUTURE_TOPUP_DATASET_ID = "comsol_internal_defect_hard_case_topup_pack_v1"
TARGET_ROWS = 120
MINIMUM_ROWS = 72
SENSOR_Z_M = 0.008
SCAN_LINE_Y_M = [-0.001, 0.0, 0.001]
AXIS_ORDER = ["Bx", "By", "Bz"]
SENSOR_X_COUNT = 201
SENSOR_X_START_M = -0.04
SENSOR_X_STOP_M = 0.04

TARGETS = ROOT / "results/metrics/internal_defect_hard_case_topup_targets.csv"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_hard_case_topup_plan_summary.txt"
PLAN_OUT = ROOT / "results/metrics/internal_defect_hard_case_topup_plan.csv"
EXPECTED_OUT = ROOT / "results/metrics/internal_defect_hard_case_expected_coverage.csv"

SHAPES = ["internal_cuboid", "internal_ellipsoid", "internal_sphere"]
BURIALS = ["shallow", "medium", "deep", "deep_plus"]
SIZES = ["small", "medium", "large"]
ASPECTS = ["compact", "elongated_x", "elongated_y"]
DEPTH_TO_SURFACE = {"shallow": 0.0008, "medium": 0.0020, "deep": 0.0032, "deep_plus": 0.0042}
VERTICAL_SIZE = {"small": 0.0010, "medium": 0.0012, "large": 0.0014}
HORIZONTAL_BASE = {"small": 0.0020, "medium": 0.0035, "large": 0.0050}
ASPECT_MULTIPLIERS = {"compact": (1.0, 1.0), "elongated_x": (1.6, 0.75), "elongated_y": (0.75, 1.6)}
CENTER_REGIONS = {
    "central": (0.0, 0.0),
    "x_pos_y_pos": (0.012, 0.004),
    "x_pos_y_neg": (0.012, -0.004),
    "x_neg_y_pos": (-0.012, 0.004),
    "x_neg_y_neg": (-0.012, -0.004),
}
EXPECTED_LABEL_FIELDS = [
    "L_m",
    "W_m",
    "D_m_or_cavity_size_m",
    "burial_depth_m",
    "depth_to_surface_m",
    "defect_center_xyz_m",
    "shape_type",
    "burial_depth_level",
    "size_level",
    "aspect_bin",
    "ground_truth_method",
    "cavity_internal",
]

PLAN_FIELDS = [
    "planned_sample_id",
    "future_dataset_id",
    "source_dataset_id",
    "source_failure_sample_id",
    "target_id",
    "target_reason",
    "neighbor_strategy",
    "split_hint",
    "selection_priority",
    "shape_type",
    "burial_depth_level",
    "size_level",
    "aspect_bin",
    "center_region",
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
    "surface_rbc_mix",
    "comsol_required",
    "plan_only",
    "notes",
]

EXPECTED_FIELDS = ["scope", "group_field", "group_value", "split_hint", "count", "minimum_expected", "pass", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 22.2 internal hard-case top-up plan.")
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--summary", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_OUT)
    parser.add_argument("--expected-coverage", type=Path, default=EXPECTED_OUT)
    return parser.parse_args()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dims_for(shape_type: str, size_level: str, aspect_bin: str) -> tuple[float, float, float]:
    d_m = VERTICAL_SIZE[size_level]
    if shape_type == "internal_sphere":
        return d_m, d_m, d_m
    base = HORIZONTAL_BASE[size_level]
    mx, my = ASPECT_MULTIPLIERS[aspect_bin]
    return base * mx, base * my, d_m


def clip_center(x_m: float, y_m: float, l_m: float, w_m: float) -> tuple[float, float]:
    x_margin = 0.04 - l_m / 2.0 - 0.001
    y_margin = 0.01 - w_m / 2.0 - 0.001
    return max(-x_margin, min(x_margin, x_m)), max(-y_margin, min(y_margin, y_m))


def split_hint(index: int) -> str:
    if index <= 80:
        return "train"
    if index <= 100:
        return "val"
    return "test"


def source_id_for(targets: dict[str, dict[str, str]], target_id: str) -> str:
    row = targets.get(target_id, {})
    source_ids = [item for item in row.get("source_sample_ids", "").split("|") if item]
    return source_ids[0] if source_ids else ""


def target_reason_for(targets: dict[str, dict[str, str]], target_id: str) -> str:
    return targets.get(target_id, {}).get("target_reason", target_id)


def add_row(
    rows: list[dict[str, Any]],
    targets: dict[str, dict[str, str]],
    target_id: str,
    neighbor_strategy: str,
    shape_type: str,
    burial_depth_level: str,
    size_level: str,
    aspect_bin: str,
    center_region: str,
    offset_index: int,
    priority: str,
) -> None:
    if shape_type == "internal_sphere":
        aspect_bin = "compact"
    index = len(rows) + 1
    l_m, w_m, d_m = dims_for(shape_type, size_level, aspect_bin)
    depth_to_surface_m = DEPTH_TO_SURFACE[burial_depth_level]
    if depth_to_surface_m <= 0 or depth_to_surface_m + d_m > 0.0056:
        raise ValueError(f"invalid internal depth: {shape_type}/{burial_depth_level}/{size_level}/{aspect_bin}")
    base_x, base_y = CENTER_REGIONS[center_region]
    jitter = [(-0.0015, 0.0), (0.0015, 0.0), (0.0, -0.001), (0.0, 0.001), (0.001, 0.001), (-0.001, -0.001)]
    dx, dy = jitter[offset_index % len(jitter)]
    center_x_m, center_y_m = clip_center(base_x + dx, base_y + dy, l_m, w_m)
    center_z_m = -(depth_to_surface_m + d_m / 2.0)
    geometry_params = {
        "steel_block_size_m": [0.08, 0.02, 0.006],
        "steel_block_center_m": [0.0, 0.0, -0.003],
        "top_scan_surface_z_m": 0.0,
        "target_hard_case_reason": target_reason_for(targets, target_id),
        "neighbor_strategy": neighbor_strategy,
        "center_region": center_region,
        "cavity_internal_rule": "depth_to_surface_m > 0 and depth_to_surface_m + D_m <= 0.0056",
    }
    rows.append(
        {
            "planned_sample_id": f"internal_hard_topup_{index:03d}",
            "future_dataset_id": FUTURE_TOPUP_DATASET_ID,
            "source_dataset_id": DATASET_ID,
            "source_failure_sample_id": source_id_for(targets, target_id),
            "target_id": target_id,
            "target_reason": target_reason_for(targets, target_id),
            "neighbor_strategy": neighbor_strategy,
            "split_hint": split_hint(index),
            "selection_priority": priority,
            "shape_type": shape_type,
            "burial_depth_level": burial_depth_level,
            "size_level": size_level,
            "aspect_bin": aspect_bin,
            "center_region": center_region,
            "center_xyz_m": compact_json([center_x_m, center_y_m, center_z_m]),
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
            "axis_order": compact_json(AXIS_ORDER),
            "scan_line_y_m": compact_json(SCAN_LINE_Y_M),
            "sensor_x_count": SENSOR_X_COUNT,
            "sensor_x_start_m": SENSOR_X_START_M,
            "sensor_x_stop_m": SENSOR_X_STOP_M,
            "expected_label_fields": compact_json(EXPECTED_LABEL_FIELDS),
            "geometry_params_json": compact_json(geometry_params),
            "ground_truth_method": "COMSOL_parametric_internal_cavity",
            "cavity_internal": True,
            "surface_rbc_mix": False,
            "comsol_required": True,
            "plan_only": True,
            "notes": "22.2 hard-case top-up design row; COMSOL generation deferred to 22.2b; not baseline",
        }
    )


def build_rows(targets: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    target_expected = {target_id: int(row["recommended_rows"]) for target_id, row in targets.items()}

    # target_01: 24 rows, paired cuboid/ellipsoid variants around branch confusion.
    burials = ["deep_plus", "shallow", "deep_plus", "shallow", "medium", "deep"]
    sizes = ["medium", "large", "large", "medium", "large", "medium"]
    aspects = ["compact", "compact", "elongated_y", "elongated_x", "compact", "elongated_y"]
    centers = ["x_pos_y_neg", "x_pos_y_pos", "x_neg_y_pos", "x_pos_y_pos", "x_neg_y_pos", "central"]
    for i in range(12):
        for shape_type in ["internal_cuboid", "internal_ellipsoid"]:
            add_row(
                rows,
                targets,
                "target_01_geometry_branch_cuboid_ellipsoid",
                "cuboid_ellipsoid_pair",
                shape_type,
                burials[i % len(burials)],
                sizes[i % len(sizes)],
                aspects[i % len(aspects)],
                centers[i % len(centers)],
                i,
                "P0",
            )

    # target_02: 20 rows, direct center/burial tradeoff probes around full-shift failures.
    for i in range(20):
        add_row(
            rows,
            targets,
            "target_02_full_shift_catastrophic",
            "center_burial_tradeoff",
            ["internal_cuboid", "internal_ellipsoid", "internal_sphere", "internal_cuboid", "internal_ellipsoid"][i % 5],
            ["shallow", "deep_plus", "shallow", "deep_plus", "medium", "deep"][i % 6],
            "large" if i % 3 != 0 else "medium",
            ["compact", "elongated_y", "compact", "elongated_x"][i % 4],
            ["x_pos_y_pos", "x_pos_y_neg", "x_neg_y_pos", "x_neg_y_neg", "central"][i % 5],
            i,
            "P0" if i % 6 in {0, 1, 2, 3} else "P1",
        )

    # target_03: 18 rows, same size/aspect with varied center regions.
    center_configs = [
        ("internal_cuboid", "large", "compact", "deep_plus"),
        ("internal_ellipsoid", "large", "elongated_y", "shallow"),
        ("internal_cuboid", "medium", "compact", "shallow"),
        ("internal_ellipsoid", "medium", "elongated_x", "deep_plus"),
        ("internal_cuboid", "large", "elongated_y", "deep"),
        ("internal_ellipsoid", "large", "compact", "medium"),
    ]
    for i in range(18):
        shape_type, size_level, aspect_bin, burial_depth_level = center_configs[i % len(center_configs)]
        add_row(
            rows,
            targets,
            "target_03_worst_center_regions",
            "same_size_aspect_varied_center",
            shape_type,
            burial_depth_level,
            size_level,
            aspect_bin,
            ["x_pos_y_pos", "x_pos_y_neg", "x_neg_y_pos", "x_neg_y_neg", "central", "x_pos_y_pos"][i % 6],
            i,
            "P0" if i % 6 in {0, 1, 5} else "P1",
        )

    # target_04: 16 rows, burial ladder to separate burial signal from geometry.
    ladder_configs = [
        ("internal_cuboid", "medium", "compact", "x_pos_y_pos"),
        ("internal_ellipsoid", "large", "compact", "x_pos_y_neg"),
        ("internal_sphere", "large", "compact", "x_neg_y_pos"),
        ("internal_cuboid", "large", "elongated_y", "x_pos_y_pos"),
    ]
    for cfg_index, (shape_type, size_level, aspect_bin, center_region) in enumerate(ladder_configs):
        for burial_depth_level in BURIALS:
            add_row(
                rows,
                targets,
                "target_04_worst_burial_depth",
                "same_shape_varied_burial",
                shape_type,
                burial_depth_level,
                size_level,
                aspect_bin,
                center_region,
                cfg_index,
                "P0" if burial_depth_level in {"shallow", "deep_plus"} else "P1",
            )

    # target_05: 14 rows, compact medium/large hard cases.
    for i in range(14):
        add_row(
            rows,
            targets,
            "target_05_compact_medium_large",
            "compact_medium_large_pair",
            "internal_cuboid" if i % 2 == 0 else "internal_ellipsoid",
            ["shallow", "deep_plus", "medium", "deep"][i % 4],
            "large" if i % 3 != 0 else "medium",
            "compact",
            ["x_pos_y_pos", "x_pos_y_neg", "x_neg_y_pos", "central"][i % 4],
            i,
            "P1",
        )

    # target_06: 10 rows, shallow edge ambiguity.
    for i in range(10):
        add_row(
            rows,
            targets,
            "target_06_shallow_edge",
            "same_burial_varied_shape",
            SHAPES[i % len(SHAPES)],
            "shallow",
            "large" if i % 2 == 0 else "medium",
            ["compact", "elongated_y", "compact", "elongated_x"][i % 4],
            ["x_neg_y_pos", "x_pos_y_pos", "x_pos_y_neg", "central"][i % 4],
            i,
            "P1",
        )

    # target_07: 10 rows, deep_plus edge ambiguity.
    for i in range(10):
        add_row(
            rows,
            targets,
            "target_07_deep_plus_edge",
            "same_burial_varied_shape",
            SHAPES[(i + 1) % len(SHAPES)],
            "deep_plus",
            "large" if i % 2 == 0 else "medium",
            ["compact", "elongated_y", "compact", "elongated_x"][i % 4],
            ["x_pos_y_pos", "x_pos_y_neg", "x_neg_y_pos", "central"][i % 4],
            i,
            "P1",
        )

    # target_08 and target_09: 4 rows each, explicit center-region sentinels.
    for i in range(4):
        add_row(
            rows,
            targets,
            "target_08_x_pos_y_pos_region",
            "same_size_aspect_varied_center",
            "internal_cuboid" if i % 2 == 0 else "internal_ellipsoid",
            "shallow" if i % 2 == 0 else "deep_plus",
            "large",
            "elongated_y" if i % 2 else "compact",
            "x_pos_y_pos",
            i,
            "P2",
        )
    for i in range(4):
        add_row(
            rows,
            targets,
            "target_09_x_neg_y_pos_region",
            "same_size_aspect_varied_center",
            ["internal_sphere", "internal_cuboid", "internal_ellipsoid", "internal_cuboid"][i],
            "shallow" if i % 2 == 0 else "deep",
            "medium" if i % 2 == 0 else "large",
            "compact",
            "x_neg_y_pos",
            i,
            "P2",
        )

    if len(rows) != TARGET_ROWS:
        raise RuntimeError(f"expected {TARGET_ROWS} plan rows, got {len(rows)}")
    actual_by_target = Counter(str(row["target_id"]) for row in rows)
    if actual_by_target != target_expected:
        raise RuntimeError(f"target quota mismatch: actual={dict(actual_by_target)} expected={target_expected}")
    sample_ids = [row["planned_sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("duplicate planned_sample_id")
    return rows


def coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    group_fields = ["shape_type", "burial_depth_level", "size_level", "aspect_bin", "center_region", "neighbor_strategy", "split_hint"]
    for field in group_fields:
        counts = Counter(str(row[field]) for row in rows)
        for value, count in sorted(counts.items()):
            minimum = 1
            if field == "shape_type" and value in {"internal_cuboid", "internal_ellipsoid"}:
                minimum = 36
            if field == "burial_depth_level" and value in {"shallow", "deep_plus"}:
                minimum = 24
            if field == "size_level" and value in {"medium", "large"}:
                minimum = 36
            if field == "aspect_bin" and value == "compact":
                minimum = 50
            if field == "split_hint":
                minimum = {"train": 72, "val": 18, "test": 18}.get(value, 1)
            out.append(
                {
                    "scope": "hard_case_topup_plan",
                    "group_field": field,
                    "group_value": value,
                    "split_hint": "all",
                    "count": count,
                    "minimum_expected": minimum,
                    "pass": count >= minimum,
                    "notes": "重点 hard-case coverage" if count >= minimum else "低于建议 minimum，22.2b 执行后需复核",
                }
            )
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split_hint"] == split]
        for field in ["shape_type", "burial_depth_level", "size_level"]:
            counts = Counter(str(row[field]) for row in split_rows)
            for value in sorted(counts):
                out.append(
                    {
                        "scope": "split_hint_coverage",
                        "group_field": field,
                        "group_value": value,
                        "split_hint": split,
                        "count": counts[value],
                        "minimum_expected": 1,
                        "pass": counts[value] >= 1,
                        "notes": "每个 split_hint 均保留 shape/burial/size coverage",
                    }
                )
    return out


def main() -> int:
    args = parse_args()
    if not args.targets.exists():
        raise FileNotFoundError(args.targets)
    target_rows = read_csv(args.targets)
    targets = {row["target_id"]: row for row in target_rows}
    rows = build_rows(targets)
    expected = coverage_rows(rows)
    write_csv(args.plan_csv, rows, PLAN_FIELDS)
    write_csv(args.expected_coverage, expected, EXPECTED_FIELDS)

    counters = {field: Counter(str(row[field]) for row in rows) for field in ["shape_type", "burial_depth_level", "size_level", "aspect_bin", "center_region", "neighbor_strategy", "split_hint"]}
    key_focus_pass = {
        "cuboid_ellipsoid": counters["shape_type"]["internal_cuboid"] >= 36 and counters["shape_type"]["internal_ellipsoid"] >= 36,
        "compact": counters["aspect_bin"]["compact"] >= 50,
        "medium_large": counters["size_level"]["medium"] + counters["size_level"]["large"] >= 96,
        "shallow_deep_plus": counters["burial_depth_level"]["shallow"] + counters["burial_depth_level"]["deep_plus"] >= 60,
        "center_neighbors": len(counters["center_region"]) >= 5,
    }
    lines = [
        "22.2 内部/埋藏缺陷 hard-case top-up plan",
        f"source_dataset_id: {DATASET_ID}",
        f"future_topup_dataset_id: {FUTURE_TOPUP_DATASET_ID}",
        "stage_scope: plan_only; no_COMSOL=true; no_training=true; no_data_or_npz_mutation=true; current_baseline_update=false",
        f"target_topup_rows: {TARGET_ROWS}",
        f"minimum_usable_rows_for_22_2b: {MINIMUM_ROWS}",
        f"sensor_z_m: {SENSOR_Z_M}",
        "liftoff_variation: false",
        "surface_rbc_mix: false",
        f"shape_counts: {dict(counters['shape_type'])}",
        f"burial_depth_counts: {dict(counters['burial_depth_level'])}",
        f"size_counts: {dict(counters['size_level'])}",
        f"aspect_counts: {dict(counters['aspect_bin'])}",
        f"center_region_counts: {dict(counters['center_region'])}",
        f"neighbor_strategy_counts: {dict(counters['neighbor_strategy'])}",
        f"split_hint_counts: {dict(counters['split_hint'])}",
        f"key_focus_gate: {key_focus_pass}",
        "planned_strategy: matched neighbor samples around 22.0/22.1 failures: same shape varied burial, same burial varied shape, same size/aspect varied center, and cuboid/ellipsoid paired variants.",
        f"plan_csv: {args.plan_csv}",
        f"expected_coverage_csv: {args.expected_coverage}",
    ]
    if not all(key_focus_pass.values()):
        lines.append("plan_gate: warning; some focus minimums need review before 22.2b")
    else:
        lines.append("plan_gate: pass")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
