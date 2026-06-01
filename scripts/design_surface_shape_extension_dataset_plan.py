#!/usr/bin/env python
"""Design the 25.1 surface shape-extension pilot dataset plan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/surface_shape_extension_dataset_plan_summary.txt"
PLAN_CSV = ROOT / "results/metrics/surface_shape_extension_dataset_plan.csv"
COVERAGE_CSV = ROOT / "results/metrics/surface_shape_extension_expected_coverage.csv"

TARGET_N = 120
FULL_COVERAGE_MIN_FALLBACK_N = 96
REDUCED_FEASIBILITY_ONLY_N = 84
SPLIT_TOTALS = {"train": 72, "val": 24, "test": 24}

SHAPE_QUOTAS = [
    ("rbc_like_smooth_pit", "single_component", "six_param_rbc", {"train": 14, "val": 5, "test": 5}),
    ("flat_bottom_pit", "single_component", "polygon_or_contour", {"train": 10, "val": 3, "test": 3}),
    ("sharp_wall_boxy_corrosion", "single_component", "polygon_or_contour", {"train": 10, "val": 3, "test": 3}),
    ("asymmetric_corrosion", "single_component", "profile_basis", {"train": 10, "val": 3, "test": 3}),
    ("elongated_crack_like_surface_defect", "elongated", "polygon_or_contour", {"train": 10, "val": 3, "test": 3}),
    ("multi_pit_two_component_surface_defect", "multi_component", "component_set", {"train": 9, "val": 3, "test": 4}),
    ("irregular_non_rbc_corrosion", "irregular", "depth_grid", {"train": 9, "val": 4, "test": 3}),
]

PLAN_FIELDS = [
    "plan_id",
    "target_N",
    "fallback_N",
    "reduced_feasibility_only_N",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "planned_count",
    "depth_bins",
    "size_bins",
    "aspect_bins",
    "minimum_per_shape_total",
    "coverage_gate",
    "notes",
]

COVERAGE_FIELDS = [
    "coverage_scope",
    "group_field",
    "group_value",
    "train_min",
    "val_min",
    "test_min",
    "planned_train",
    "planned_val",
    "planned_test",
    "planned_total",
    "pass",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 25.1 surface shape-extension pilot dataset plan.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--plan", type=Path, default=PLAN_CSV)
    parser.add_argument("--expected-coverage", type=Path, default=COVERAGE_CSV)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plan_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape_type, topology_type, representation_target, split_counts in SHAPE_QUOTAS:
        for split, count in split_counts.items():
            rows.append(
                {
                    "plan_id": "surface_shape_extension_pilot_v1",
                    "target_N": TARGET_N,
                    "fallback_N": FULL_COVERAGE_MIN_FALLBACK_N,
                    "reduced_feasibility_only_N": REDUCED_FEASIBILITY_ONLY_N,
                    "split": split,
                    "shape_type": shape_type,
                    "topology_type": topology_type,
                    "representation_target": representation_target,
                    "planned_count": count,
                    "depth_bins": "shallow|medium|deep",
                    "size_bins": "small|medium|large",
                    "aspect_bins": "compact|elongated",
                    "minimum_per_shape_total": 12,
                    "coverage_gate": "each split covers all seven shape families and all governing depth/size/topology groups",
                    "notes": "plan row only; COMSOL generation is deferred to 25.2",
                }
            )
    return rows


def coverage_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape_type, topology_type, representation_target, split_counts in SHAPE_QUOTAS:
        total = sum(split_counts.values())
        rows.append(
            {
                "coverage_scope": "shape_family",
                "group_field": "shape_type",
                "group_value": shape_type,
                "train_min": 1,
                "val_min": 1,
                "test_min": 1,
                "planned_train": split_counts["train"],
                "planned_val": split_counts["val"],
                "planned_test": split_counts["test"],
                "planned_total": total,
                "pass": total >= 12 and all(value > 0 for value in split_counts.values()),
                "notes": f"topology={topology_type}; target={representation_target}",
            }
        )
    topology_totals: dict[str, dict[str, int]] = {}
    for _, topology_type, _, split_counts in SHAPE_QUOTAS:
        bucket = topology_totals.setdefault(topology_type, {"train": 0, "val": 0, "test": 0})
        for split, value in split_counts.items():
            bucket[split] += value
    for topology_type, split_counts in topology_totals.items():
        rows.append(
            {
                "coverage_scope": "topology",
                "group_field": "topology_type",
                "group_value": topology_type,
                "train_min": 1,
                "val_min": 1,
                "test_min": 1,
                "planned_train": split_counts["train"],
                "planned_val": split_counts["val"],
                "planned_test": split_counts["test"],
                "planned_total": sum(split_counts.values()),
                "pass": all(value > 0 for value in split_counts.values()),
                "notes": "topology present in every split",
            }
        )
    for field, values in {
        "depth_bin": ["shallow", "medium", "deep"],
        "size_bin": ["small", "medium", "large"],
        "aspect_bin": ["compact", "elongated"],
    }.items():
        for value in values:
            rows.append(
                {
                    "coverage_scope": "stratification",
                    "group_field": field,
                    "group_value": value,
                    "train_min": 1,
                    "val_min": 1,
                    "test_min": 1,
                    "planned_train": "balanced_within_each_shape_quota",
                    "planned_val": "balanced_within_each_shape_quota",
                    "planned_test": "balanced_within_each_shape_quota",
                    "planned_total": "covered_by_generation_grid",
                    "pass": True,
                    "notes": "generation stage must instantiate bins inside each shape family without test-set tuning",
                }
            )
    rows.append(
        {
            "coverage_scope": "split_total",
            "group_field": "split",
            "group_value": "train/val/test",
            "train_min": 72,
            "val_min": 24,
            "test_min": 24,
            "planned_train": SPLIT_TOTALS["train"],
            "planned_val": SPLIT_TOTALS["val"],
            "planned_test": SPLIT_TOTALS["test"],
            "planned_total": TARGET_N,
            "pass": True,
            "notes": "fixed 72/24/24 split; test remains final evaluation only",
        }
    )
    return rows


def run(args: argparse.Namespace) -> int:
    rows = plan_rows()
    coverage = coverage_rows()
    write_csv(args.plan, rows, PLAN_FIELDS)
    write_csv(args.expected_coverage, coverage, COVERAGE_FIELDS)
    per_shape = {shape: sum(counts.values()) for shape, _, _, counts in SHAPE_QUOTAS}
    non_rbc_total = sum(total for shape, total in per_shape.items() if shape != "rbc_like_smooth_pit")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface shape-extension dataset plan summary",
                "stage: 25.1",
                "",
                f"target_N: {TARGET_N}",
                "split: train/val/test = 72/24/24",
                f"rbc_like_control_count: {per_shape['rbc_like_smooth_pit']}",
                f"non_rbc_like_count: {non_rbc_total}",
                f"minimum_full_coverage_fallback_N: {FULL_COVERAGE_MIN_FALLBACK_N}",
                f"reduced_feasibility_only_N: {REDUCED_FEASIBILITY_ONLY_N}",
                "fallback_note: N=84 conflicts with RBC-like >=24 plus seven shape families >=12 each; it is reduced_feasibility_only, not full coverage.",
                "v2_benchmark_target: N=300-480 after pilot generation/label validation passes.",
                "",
                "per_shape_totals:",
                *[f"- {shape}: {total}" for shape, total in per_shape.items()],
                "",
                "coverage_gate: every split covers RBC-like, flat/sharp-wall, asymmetric, elongated, multi-pit, irregular, shallow/medium/deep, small/medium/large, compact/elongated.",
                "scope_boundary: plan only; no COMSOL, no training, no data/NPZ writing.",
                f"dataset_plan_csv: {args.plan}",
                f"expected_coverage_csv: {args.expected_coverage}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
