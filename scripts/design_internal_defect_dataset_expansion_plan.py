#!/usr/bin/env python
"""Design the 21.3 internal defect dataset expansion plan.

This creates the planned top-up rows and expected v2_240 coverage only. It
does not run COMSOL, create NPZ/data, train, or update CURRENT_BASELINE.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, write_csv


SUMMARY = ROOT / "results/summaries/internal_defect_dataset_expansion_plan_summary.txt"
PLAN = ROOT / "results/metrics/internal_defect_dataset_expansion_plan.csv"
EXPECTED = ROOT / "results/metrics/internal_defect_dataset_expected_coverage.csv"

V2_DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SOURCE_DATASET_ID = "comsol_internal_defect_pilot_pack_v1"
TOPUP_DATASET_ID = "comsol_internal_defect_dataset_topup_pack_v1"
AXIS_ORDER = ["Bx", "By", "Bz"]
SCAN_LINE_Y_M = [-0.001, 0.0, 0.001]
SENSOR_Z_M = 0.008
SENSOR_X_COUNT = 201
SENSOR_X_START_M = -0.04
SENSOR_X_STOP_M = 0.04

SHAPES = ["internal_sphere", "internal_ellipsoid", "internal_cuboid"]
BURIALS = ["shallow", "medium", "deep", "deep_plus"]
SIZES = ["small", "medium", "large"]
ASPECTS = ["compact", "elongated_x", "elongated_y"]
SPLITS = ["train", "val", "test"]

SOURCE_COUNTS = {
    "shape_type": {"internal_sphere": 24, "internal_ellipsoid": 36, "internal_cuboid": 36},
    "burial_depth_level": {"shallow": 24, "medium": 24, "deep": 24, "deep_plus": 24},
    "size_level": {"small": 32, "medium": 32, "large": 32},
    "aspect_bin": {"compact": 48, "elongated_x": 24, "elongated_y": 24},
}
SELECTED_TOPUP_COUNTS = {
    "shape_type": {"internal_sphere": 56, "internal_ellipsoid": 44, "internal_cuboid": 44},
    "burial_depth_level": {"shallow": 36, "medium": 36, "deep": 36, "deep_plus": 36},
    "size_level": {"small": 48, "medium": 48, "large": 48},
}
PLANNED_TOPUP_COUNTS = {
    "shape_type": {"internal_sphere": 64, "internal_ellipsoid": 52, "internal_cuboid": 52},
    "burial_depth_level": {"shallow": 42, "medium": 42, "deep": 42, "deep_plus": 42},
    "size_level": {"small": 56, "medium": 56, "large": 56},
}
SELECTED_ASPECT_BY_SHAPE = {
    "internal_sphere": {"compact": 56},
    "internal_ellipsoid": {"compact": 15, "elongated_x": 15, "elongated_y": 14},
    "internal_cuboid": {"compact": 15, "elongated_x": 15, "elongated_y": 14},
}
BUFFER_COUNTS = {
    "shape_type": {"internal_sphere": 8, "internal_ellipsoid": 8, "internal_cuboid": 8},
    "burial_depth_level": {"shallow": 6, "medium": 6, "deep": 6, "deep_plus": 6},
    "size_level": {"small": 8, "medium": 8, "large": 8},
}
BUFFER_ASPECT_BY_SHAPE = {
    "internal_sphere": {"compact": 8},
    "internal_ellipsoid": {"compact": 3, "elongated_x": 2, "elongated_y": 3},
    "internal_cuboid": {"compact": 3, "elongated_x": 2, "elongated_y": 3},
}
SELECTED_SPLIT_QUOTAS = {
    "train": {
        "shape_type": {"internal_sphere": 38, "internal_ellipsoid": 29, "internal_cuboid": 29},
        "burial_depth_level": {"shallow": 24, "medium": 24, "deep": 24, "deep_plus": 24},
        "size_level": {"small": 32, "medium": 32, "large": 32},
        "aspect_by_shape": {
            "internal_sphere": {"compact": 38},
            "internal_ellipsoid": {"compact": 10, "elongated_x": 10, "elongated_y": 9},
            "internal_cuboid": {"compact": 10, "elongated_x": 10, "elongated_y": 9},
        },
    },
    "val": {
        "shape_type": {"internal_sphere": 9, "internal_ellipsoid": 8, "internal_cuboid": 7},
        "burial_depth_level": {"shallow": 6, "medium": 6, "deep": 6, "deep_plus": 6},
        "size_level": {"small": 8, "medium": 8, "large": 8},
        "aspect_by_shape": {
            "internal_sphere": {"compact": 9},
            "internal_ellipsoid": {"compact": 3, "elongated_x": 3, "elongated_y": 2},
            "internal_cuboid": {"compact": 2, "elongated_x": 2, "elongated_y": 3},
        },
    },
    "test": {
        "shape_type": {"internal_sphere": 9, "internal_ellipsoid": 7, "internal_cuboid": 8},
        "burial_depth_level": {"shallow": 6, "medium": 6, "deep": 6, "deep_plus": 6},
        "size_level": {"small": 8, "medium": 8, "large": 8},
        "aspect_by_shape": {
            "internal_sphere": {"compact": 9},
            "internal_ellipsoid": {"compact": 2, "elongated_x": 2, "elongated_y": 3},
            "internal_cuboid": {"compact": 3, "elongated_x": 3, "elongated_y": 2},
        },
    },
}
SOURCE_SPLIT_STRATEGY = {
    "shape_type": {
        "internal_sphere": {"train": 16, "val": 4, "test": 4},
        "internal_ellipsoid": {"train": 24, "val": 6, "test": 6},
        "internal_cuboid": {"train": 24, "val": 6, "test": 6},
    },
    "burial_depth_level": {value: {"train": 16, "val": 4, "test": 4} for value in BURIALS},
    "size_level": {
        "small": {"train": 22, "val": 5, "test": 5},
        "medium": {"train": 21, "val": 6, "test": 5},
        "large": {"train": 21, "val": 5, "test": 6},
    },
    "aspect_bin": {
        "compact": {"train": 32, "val": 8, "test": 8},
        "elongated_x": {"train": 16, "val": 4, "test": 4},
        "elongated_y": {"train": 16, "val": 4, "test": 4},
    },
}
FINAL_SHAPE_SPLIT_TARGET = {
    "internal_sphere": {"train": 54, "val": 13, "test": 13},
    "internal_ellipsoid": {"train": 53, "val": 14, "test": 13},
    "internal_cuboid": {"train": 53, "val": 13, "test": 14},
}
FINAL_BURIAL_SPLIT_TARGET = {value: {"train": 40, "val": 10, "test": 10} for value in BURIALS}
FINAL_SIZE_SPLIT_TARGET = {
    "small": {"train": 54, "val": 13, "test": 13},
    "medium": {"train": 53, "val": 14, "test": 13},
    "large": {"train": 53, "val": 13, "test": 14},
}

PLAN_FIELDS = [
    "topup_sample_id",
    "target_dataset_id",
    "source_dataset_id",
    "topup_dataset_id",
    "topup_role",
    "v2_split_hint",
    "selection_priority",
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
    "notes",
]
EXPECTED_FIELDS = [
    "scope",
    "group_field",
    "group_value",
    "split",
    "source_count",
    "selected_topup_count",
    "assembled_expected_count",
    "target_count",
    "pass",
    "notes",
]

DEPTH_TO_SURFACE = {"shallow": 0.0008, "medium": 0.0020, "deep": 0.0032, "deep_plus": 0.0042}
VERTICAL_SIZE = {"small": 0.0010, "medium": 0.0012, "large": 0.0014}
HORIZONTAL_BASE = {"small": 0.0020, "medium": 0.0035, "large": 0.0050}
ASPECT_MULTIPLIERS = {"compact": (1.0, 1.0), "elongated_x": (1.6, 0.75), "elongated_y": (0.75, 1.6)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design internal defect v2_240 expansion plan.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--expected-coverage", type=Path, default=EXPECTED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def dims_for(shape_type: str, size_level: str, aspect_bin: str) -> tuple[float, float, float]:
    d_m = VERTICAL_SIZE[size_level]
    if shape_type == "internal_sphere":
        return d_m, d_m, d_m
    base = HORIZONTAL_BASE[size_level]
    mx, my = ASPECT_MULTIPLIERS[aspect_bin]
    return base * mx, base * my, d_m


def center_for(row_index: int, shape_type: str, aspect_bin: str, depth_to_surface_m: float, d_m: float) -> tuple[float, float, float]:
    lanes = {
        "internal_sphere": [-0.026, -0.020, -0.014, -0.008],
        "internal_ellipsoid": [-0.006, 0.000, 0.006, 0.012],
        "internal_cuboid": [0.018, 0.024, 0.030, 0.034],
    }
    y_by_aspect = {"compact": [-0.006, -0.002, 0.002, 0.006], "elongated_x": [-0.004, 0.000, 0.004], "elongated_y": [-0.005, 0.000, 0.005]}
    x_values = lanes[shape_type]
    y_values = y_by_aspect[aspect_bin]
    x = x_values[row_index % len(x_values)]
    y = y_values[(row_index // len(x_values)) % len(y_values)]
    z = -(depth_to_surface_m + d_m / 2.0)
    return x, y, z


def decrement(counter: dict[str, int], key: str) -> None:
    if counter.get(key, 0) <= 0:
        raise RuntimeError(f"quota exhausted for {key}: {counter}")
    counter[key] -= 1


def choose(counter: dict[str, int], order: list[str]) -> str:
    available = [(counter.get(value, 0), -order.index(value), value) for value in order if counter.get(value, 0) > 0]
    if not available:
        raise RuntimeError(f"no quota available: {counter}")
    return max(available)[2]


def allocate_rows(role: str, split_hint: str, shape_quota: dict[str, int], burial_quota: dict[str, int], size_quota: dict[str, int], aspect_by_shape: dict[str, dict[str, int]], start_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shape_remaining = dict(shape_quota)
    burial_remaining = dict(burial_quota)
    size_remaining = dict(size_quota)
    aspect_remaining = {shape: dict(values) for shape, values in aspect_by_shape.items()}
    total = sum(shape_remaining.values())
    for _ in range(total):
        shape_type = choose(shape_remaining, SHAPES)
        burial = choose(burial_remaining, BURIALS)
        size = choose(size_remaining, SIZES)
        aspect = choose(aspect_remaining[shape_type], ASPECTS)
        decrement(shape_remaining, shape_type)
        decrement(burial_remaining, burial)
        decrement(size_remaining, size)
        decrement(aspect_remaining[shape_type], aspect)
        row_index = start_index + len(rows)
        l_m, w_m, d_m = dims_for(shape_type, size, aspect)
        depth = DEPTH_TO_SURFACE[burial]
        if depth + d_m > 0.0056:
            raise RuntimeError(f"invalid internal cavity depth: {shape_type}/{burial}/{size}/{aspect}")
        cx, cy, cz = center_for(row_index, shape_type, aspect, depth, d_m)
        geometry_params = {
            "steel_block_size_m": [0.08, 0.02, 0.006],
            "steel_block_center_m": [0.0, 0.0, -0.003],
            "top_scan_surface_z_m": 0.0,
            "planned_stage": "21.3b",
            "role": role,
        }
        rows.append(
            {
                "topup_sample_id": f"internal_topup_{row_index:03d}",
                "target_dataset_id": V2_DATASET_ID,
                "source_dataset_id": SOURCE_DATASET_ID,
                "topup_dataset_id": TOPUP_DATASET_ID,
                "topup_role": role,
                "v2_split_hint": split_hint,
                "selection_priority": row_index,
                "shape_type": shape_type,
                "burial_depth_level": burial,
                "size_level": size,
                "aspect_bin": aspect,
                "center_xyz_m": compact_json([cx, cy, cz]),
                "center_x_m": f"{cx:.9g}",
                "center_y_m": f"{cy:.9g}",
                "center_z_m": f"{cz:.9g}",
                "L_m": f"{l_m:.9g}",
                "W_m": f"{w_m:.9g}",
                "D_m": f"{d_m:.9g}",
                "D_m_or_cavity_size_m": f"{d_m:.9g}",
                "burial_depth_m": f"{depth:.9g}",
                "depth_to_surface_m": f"{depth:.9g}",
                "scan_surface": "top_z_0",
                "sensor_z_m": f"{SENSOR_Z_M:.9g}",
                "axis_order": compact_json(AXIS_ORDER),
                "scan_line_y_m": compact_json(SCAN_LINE_Y_M),
                "sensor_x_count": SENSOR_X_COUNT,
                "sensor_x_start_m": SENSOR_X_START_M,
                "sensor_x_stop_m": SENSOR_X_STOP_M,
                "expected_label_fields": compact_json(["L_m", "W_m", "D_m", "burial_depth_m", "depth_to_surface_m", "defect_center_xyz_m", "shape_type", "aspect_bin", "ground_truth_method", "cavity_internal"]),
                "geometry_params_json": compact_json(geometry_params),
                "ground_truth_method": "COMSOL_parametric_internal_cavity",
                "cavity_internal": True,
                "notes": "21.3 plan row only; COMSOL generation happens in 21.3b; not baseline",
            }
        )
    if any(shape_remaining.values()) or any(burial_remaining.values()) or any(size_remaining.values()) or any(v for quota in aspect_remaining.values() for v in quota.values()):
        raise RuntimeError("quota allocation did not exhaust all counters")
    return rows


def build_plan_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_index = 1
    for split_name in SPLITS:
        q = SELECTED_SPLIT_QUOTAS[split_name]
        split_rows = allocate_rows(
            "selected_quota",
            split_name,
            q["shape_type"],
            q["burial_depth_level"],
            q["size_level"],
            q["aspect_by_shape"],
            next_index,
        )
        rows.extend(split_rows)
        next_index += len(split_rows)
    buffer_rows = allocate_rows(
        "buffer",
        "buffer_only",
        BUFFER_COUNTS["shape_type"],
        BUFFER_COUNTS["burial_depth_level"],
        BUFFER_COUNTS["size_level"],
        BUFFER_ASPECT_BY_SHAPE,
        next_index,
    )
    rows.extend(buffer_rows)
    return rows


def verify_counts(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    planned = rows
    selected = [row for row in rows if row["topup_role"] == "selected_quota"]
    if len(planned) != 168:
        errors.append(f"planned rows expected 168, got {len(planned)}")
    if len(selected) != 144:
        errors.append(f"selected rows expected 144, got {len(selected)}")
    for field, expected in PLANNED_TOPUP_COUNTS.items():
        counts = Counter(row[field] for row in planned)
        if dict(counts) != expected:
            errors.append(f"planned {field} mismatch: {dict(counts)} != {expected}")
    for field, expected in SELECTED_TOPUP_COUNTS.items():
        counts = Counter(row[field] for row in selected)
        if dict(counts) != expected:
            errors.append(f"selected {field} mismatch: {dict(counts)} != {expected}")
    split_counts = Counter(row["v2_split_hint"] for row in selected)
    if dict(split_counts) != {"train": 96, "val": 24, "test": 24}:
        errors.append(f"selected split mismatch: {dict(split_counts)}")
    return errors


def expected_rows(plan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in plan_rows if row["topup_role"] == "selected_quota"]
    rows: list[dict[str, Any]] = []

    def add(scope: str, field: str, value: str, split: str, source: int, topup: int, target: int, notes: str) -> None:
        assembled = source + topup
        rows.append(
            {
                "scope": scope,
                "group_field": field,
                "group_value": value,
                "split": split,
                "source_count": source,
                "selected_topup_count": topup,
                "assembled_expected_count": assembled,
                "target_count": target,
                "pass": assembled == target,
                "notes": notes,
            }
        )

    add("total", "row_count", "all", "all", 96, 144, 240, "assembled v2 target")
    for split, source in {"train": 64, "val": 16, "test": 16}.items():
        topup = sum(1 for row in selected if row["v2_split_hint"] == split)
        add("split", "split", split, split, source, topup, {"train": 160, "val": 40, "test": 40}[split], "v2 split quota")
    for field, target_total in [("shape_type", {"internal_sphere": 80, "internal_ellipsoid": 80, "internal_cuboid": 80}), ("burial_depth_level", {value: 60 for value in BURIALS}), ("size_level", {value: 80 for value in SIZES})]:
        for value, target in target_total.items():
            source = SOURCE_COUNTS[field][value]
            topup = sum(1 for row in selected if row[field] == value)
            add("total", field, value, "all", source, topup, target, f"v2 total {field} balance")
    hard_targets = {
        "shape_type": FINAL_SHAPE_SPLIT_TARGET,
        "burial_depth_level": FINAL_BURIAL_SPLIT_TARGET,
        "size_level": FINAL_SIZE_SPLIT_TARGET,
    }
    for field in ["shape_type", "burial_depth_level", "size_level", "aspect_bin"]:
        for value in sorted(SOURCE_SPLIT_STRATEGY[field]):
            for split in SPLITS:
                source = SOURCE_SPLIT_STRATEGY[field][value][split]
                topup = sum(1 for row in selected if row.get(field) == value and row["v2_split_hint"] == split)
                if field == "aspect_bin":
                    target = source + topup
                    notes = "v2 split must retain aspect coverage; exact aspect balance is not global because sphere is compact only"
                else:
                    target = hard_targets[field][value][split]
                    notes = "v2 split hard quota"
                add("split", field, value, split, source, topup, target, notes)
    for shape in ["internal_ellipsoid", "internal_cuboid"]:
        for aspect in ASPECTS:
            for split in SPLITS:
                topup = sum(1 for row in selected if row["shape_type"] == shape and row["aspect_bin"] == aspect and row["v2_split_hint"] == split)
                rows.append(
                    {
                        "scope": "split_coverage_gate",
                        "group_field": "shape_type_x_aspect_bin",
                        "group_value": f"{shape}|{aspect}",
                        "split": split,
                        "source_count": "",
                        "selected_topup_count": topup,
                        "assembled_expected_count": topup,
                        "target_count": ">=1",
                        "pass": topup >= 1,
                        "notes": "ellipsoid/cuboid 每个 split 必须覆盖 compact、elongated_x、elongated_y；sphere 固定 compact，不参与该 gate",
                    }
                )
    return rows


def write_plan_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    plan_rows = build_plan_rows()
    errors = verify_counts(plan_rows)
    if errors:
        raise RuntimeError("; ".join(errors))
    expected = expected_rows(plan_rows)
    write_plan_csv(args.plan, plan_rows)
    write_csv(args.expected_coverage, expected, EXPECTED_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    selected = [row for row in plan_rows if row["topup_role"] == "selected_quota"]
    summary = [
        "21.3 internal defect dataset expansion plan summary",
        "",
        f"target_dataset_id: {V2_DATASET_ID}",
        f"source_dataset_id: {SOURCE_DATASET_ID}",
        f"source_rows_reused: 96",
        f"selected_topup_target_rows: {len(selected)}",
        f"planned_topup_rows: {len(plan_rows)}",
        "assembled_target_rows: 240",
        "assembled_split_target: train/val/test = 160/40/40",
        f"selected_topup_shape_counts: {dict(Counter(row['shape_type'] for row in selected))}",
        f"selected_topup_burial_counts: {dict(Counter(row['burial_depth_level'] for row in selected))}",
        f"selected_topup_size_counts: {dict(Counter(row['size_level'] for row in selected))}",
        f"planned_topup_shape_counts: {dict(Counter(row['shape_type'] for row in plan_rows))}",
        "old_v1_split_reused: false",
        "v2_split_policy: source rows are re-split together with selected top-up rows by deterministic stratified quotas.",
        "comsol_run: false",
        "training_run: false",
        "data_npz_mutation: false",
        "current_baseline_update: false",
        "next_stage: 21.3b generate top-up COMSOL pack, then assemble and validate v2_240.",
    ]
    args.summary.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
