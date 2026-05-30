#!/usr/bin/env python
"""22.3 failure re-audit for the selected hard-case augmented internal model."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from internal_defect_hardcase_utils import read_csv, safe_float
from load_internal_defect_pilot_dataset import ROOT, write_csv


PREDICTIONS = ROOT / "results/metrics/internal_defect_hardcase_selected_predictions.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_hardcase_failure_audit_summary.txt"
FAILURE_CASES = ROOT / "results/metrics/internal_defect_hardcase_failure_cases.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_hardcase_failure_group_summary.csv"

FAILURE_FIELDS = [
    "sample_id",
    "split",
    "subset",
    "row_origin",
    "true_shape_type",
    "pred_shape_type",
    "shape_correct",
    "burial_depth_level",
    "size_level",
    "aspect_bin",
    "hardcase_target_id",
    "center_xyz_error_mm",
    "burial_depth_error_mm",
    "total_abs_normalized_error",
    "failure_tags",
    "rank_reason",
]

GROUP_FIELDS = [
    "group_field",
    "group_value",
    "split",
    "subset",
    "sample_count",
    "catastrophic_failure_count",
    "catastrophic_failure_rate",
    "geometry_branch_failure_count",
    "shape_error_count",
    "center_error_mean_mm",
    "center_error_p95_mm",
    "center_error_max_mm",
    "burial_error_mean_mm",
    "burial_error_p95_mm",
    "burial_error_max_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit selected hard-case model failures.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--failure-cases", type=Path, default=FAILURE_CASES)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    return parser.parse_args()


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def group_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in ["true_shape_type", "burial_depth_level", "size_level", "aspect_bin", "subset", "hardcase_target_id"]:
        values = sorted(set(row.get(field, "") for row in rows))
        for split in ["train", "val", "test"]:
            split_rows = [row for row in rows if row.get("split") == split]
            for subset in ["all", "source_v2", "hardcase_topup"]:
                base = split_rows if subset == "all" else [row for row in split_rows if row.get("subset") == subset]
                for value in values:
                    selected = [row for row in base if row.get(field, "") == value]
                    if not selected:
                        continue
                    center = [safe_float(row["center_xyz_error_mm"]) for row in selected]
                    burial = [safe_float(row["burial_depth_error_mm"]) for row in selected]
                    catastrophic = sum("full_shift_failure" in row.get("failure_tags", "") for row in selected)
                    geometry = sum("geometry_branch_failure" in row.get("failure_tags", "") for row in selected)
                    shape_error = sum(not bool_value(row.get("shape_correct", "")) for row in selected)
                    out.append(
                        {
                            "group_field": field,
                            "group_value": value,
                            "split": split,
                            "subset": subset,
                            "sample_count": len(selected),
                            "catastrophic_failure_count": catastrophic,
                            "catastrophic_failure_rate": catastrophic / len(selected),
                            "geometry_branch_failure_count": geometry,
                            "shape_error_count": shape_error,
                            "center_error_mean_mm": float(np.mean(center)),
                            "center_error_p95_mm": quantile(center, 95),
                            "center_error_max_mm": max(center),
                            "burial_error_mean_mm": float(np.mean(burial)),
                            "burial_error_p95_mm": quantile(burial, 95),
                            "burial_error_max_mm": max(burial),
                        }
                    )
    return out


def selected_failure_cases(test_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    picked: dict[str, dict[str, Any]] = {}

    def add_many(rows: list[dict[str, str]], reason: str, limit: int) -> None:
        for row in rows[:limit]:
            item = {key: row.get(key, "") for key in FAILURE_FIELDS}
            item["rank_reason"] = reason
            picked[f"{row['sample_id']}::{reason}"] = item

    catastrophic = [row for row in test_rows if "full_shift_failure" in row.get("failure_tags", "")]
    geometry = [row for row in test_rows if "geometry_branch_failure" in row.get("failure_tags", "")]
    shape_mis = [row for row in test_rows if not bool_value(row.get("shape_correct", ""))]
    add_many(sorted(test_rows, key=lambda r: safe_float(r["center_xyz_error_mm"]), reverse=True), "worst_center", 8)
    add_many(sorted(test_rows, key=lambda r: safe_float(r["burial_depth_error_mm"]), reverse=True), "worst_burial", 8)
    add_many(sorted(test_rows, key=lambda r: safe_float(r["total_abs_normalized_error"]), reverse=True), "worst_total", 8)
    add_many(catastrophic, "catastrophic_failure", len(catastrophic))
    add_many(geometry, "geometry_branch_failure", len(geometry))
    add_many(shape_mis, "shape_misclassified", len(shape_mis))
    return list(picked.values())


def main() -> int:
    args = parse_args()
    rows = read_csv(args.predictions)
    if not rows:
        raise RuntimeError(f"missing selected prediction rows: {args.predictions}")
    test_rows = [row for row in rows if row.get("split") == "test"]
    center = [safe_float(row["center_xyz_error_mm"]) for row in test_rows]
    burial = [safe_float(row["burial_depth_error_mm"]) for row in test_rows]
    catastrophic = [row for row in test_rows if "full_shift_failure" in row.get("failure_tags", "")]
    geometry = [row for row in test_rows if "geometry_branch_failure" in row.get("failure_tags", "")]
    cuboid_ellipsoid = [
        row
        for row in test_rows
        if {row.get("true_shape_type"), row.get("pred_shape_type")} == {"internal_cuboid", "internal_ellipsoid"}
        and row.get("true_shape_type") != row.get("pred_shape_type")
    ]
    failures = selected_failure_cases(test_rows)
    group = group_rows(rows)
    write_csv(args.failure_cases, failures, FAILURE_FIELDS)
    write_csv(args.group_summary, group, GROUP_FIELDS)

    def count_by(field: str, selected: list[dict[str, str]]) -> dict[str, int]:
        return dict(Counter(row.get(field, "") for row in selected))

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "22.3 hard-case augmented internal defect failure re-audit",
        f"prediction_source: {args.predictions}",
        f"test_count: {len(test_rows)}",
        f"catastrophic_failure_count: {len(catastrophic)}",
        f"geometry_branch_failure_count: {len(geometry)}",
        f"cuboid_ellipsoid_confusion_count: {len(cuboid_ellipsoid)}",
        f"center_mean_p95_max_mm: {float(np.mean(center)):.3f} / {quantile(center, 95):.3f} / {max(center):.3f}",
        f"burial_mean_p95_max_mm: {float(np.mean(burial)):.3f} / {quantile(burial, 95):.3f} / {max(burial):.3f}",
        f"catastrophic_by_shape: {count_by('true_shape_type', catastrophic)}",
        f"catastrophic_by_burial: {count_by('burial_depth_level', catastrophic)}",
        f"catastrophic_by_size: {count_by('size_level', catastrophic)}",
        f"catastrophic_by_aspect: {count_by('aspect_bin', catastrophic)}",
        f"geometry_branch_cases: {[row['sample_id'] for row in geometry]}",
        "hardcase_topup_effect: see internal_defect_hardcase_vs_b2_reference.csv and group summary.",
        "current_baseline_update: false",
    ]
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
