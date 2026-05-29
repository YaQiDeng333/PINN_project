#!/usr/bin/env python
"""Failure-driven audit for internal defect B2 predictions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import ROOT, write_csv


REPLAY = ROOT / "results/metrics/internal_defect_b2_inference_replay_metrics.csv"
FEATURE_METRICS = ROOT / "results/metrics/internal_defect_v2_feature_baseline_metrics.csv"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_b2_failure_audit_summary.txt"
CASES_OUT = ROOT / "results/metrics/internal_defect_b2_failure_cases.csv"
GROUP_OUT = ROOT / "results/metrics/internal_defect_b2_failure_group_summary.csv"
TAIL_OUT = ROOT / "results/metrics/internal_defect_b2_tail_error_summary.csv"
BRANCH_OUT = ROOT / "results/metrics/internal_defect_b2_geometry_branch_failure_summary.csv"


CASE_FIELDS = [
    "sample_id",
    "split",
    "true_shape_type",
    "pred_shape_type",
    "shape_correct",
    "burial_depth_level",
    "size_level",
    "aspect_bin",
    "center_region",
    "true_L_mm",
    "pred_L_mm",
    "true_W_mm",
    "pred_W_mm",
    "true_D_mm",
    "pred_D_mm",
    "true_burial_depth_mm",
    "pred_burial_depth_mm",
    "true_center_x_mm",
    "pred_center_x_mm",
    "true_center_y_mm",
    "pred_center_y_mm",
    "true_center_z_mm",
    "pred_center_z_mm",
    "total_abs_normalized_error",
    "L_error_mm",
    "W_error_mm",
    "D_error_mm",
    "burial_depth_error_mm",
    "center_xyz_error_mm",
    "dimension_relative_max",
    "failure_tags",
    "is_catastrophic_failure",
    "is_geometry_branch_failure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit internal defect B2 failure cases.")
    parser.add_argument("--replay-metrics", type=Path, default=REPLAY)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def center_region(row: dict[str, Any]) -> str:
    x = safe_float(row["true_center_x_mm"])
    y = safe_float(row["true_center_y_mm"])
    if abs(x) < 1.0 and abs(y) < 0.5:
        return "central"
    x_side = "x_pos" if x >= 0 else "x_neg"
    y_side = "y_pos" if y >= 0 else "y_neg"
    return f"{x_side}_{y_side}"


def add_failure_tags(row: dict[str, Any]) -> dict[str, Any]:
    l_err = safe_float(row["L_error_mm"])
    w_err = safe_float(row["W_error_mm"])
    d_err = safe_float(row["D_error_mm"])
    true_l = max(abs(safe_float(row["true_L_mm"])), 1e-9)
    true_w = max(abs(safe_float(row["true_W_mm"])), 1e-9)
    true_d = max(abs(safe_float(row["true_D_mm"])), 1e-9)
    rel_max = max(l_err / true_l, w_err / true_w, d_err / true_d)
    center_outlier = safe_float(row["center_xyz_error_mm"]) > 3.0
    burial_outlier = safe_float(row["burial_depth_error_mm"]) > 1.0
    dimension_outlier = max(l_err, w_err, d_err) > 2.0 or rel_max > 0.30
    shape_misclassified = not bool_value(row["shape_correct"])
    full_shift_failure = center_outlier and burial_outlier
    geometry_branch_failure = shape_misclassified and full_shift_failure
    tags = []
    if center_outlier:
        tags.append("center_outlier")
    if burial_outlier:
        tags.append("burial_outlier")
    if dimension_outlier:
        tags.append("dimension_outlier")
    if shape_misclassified:
        tags.append("shape_misclassified")
    if full_shift_failure:
        tags.append("full_shift_failure")
    if geometry_branch_failure:
        tags.append("geometry_branch_failure")
    if full_shift_failure or geometry_branch_failure:
        tags.append("visual_suspect")
    result = dict(row)
    result["center_region"] = center_region(row)
    result["dimension_relative_max"] = rel_max
    result["failure_tags"] = "|".join(tags) if tags else "none"
    result["is_catastrophic_failure"] = full_shift_failure or geometry_branch_failure
    result["is_geometry_branch_failure"] = geometry_branch_failure
    return result


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def tail_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        ("total_abs_normalized_error", "normalized"),
        ("L_error_mm", "mm"),
        ("W_error_mm", "mm"),
        ("D_error_mm", "mm"),
        ("burial_depth_error_mm", "mm"),
        ("center_xyz_error_mm", "mm"),
        ("dimension_relative_max", "ratio"),
    ]
    out = []
    for metric, unit in metrics:
        values = [safe_float(row[metric]) for row in rows]
        out.append(
            {
                "split": "test",
                "metric": metric,
                "unit": unit,
                "mean": float(np.mean(values)),
                "median": percentile(values, 50),
                "p75": percentile(values, 75),
                "p90": percentile(values, 90),
                "p95": percentile(values, 95),
                "max": max(values),
            }
        )
    return out


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_fields = [
        "true_shape_type",
        "burial_depth_level",
        "size_level",
        "aspect_bin",
        "center_region",
        "true_shape_type|burial_depth_level",
        "true_shape_type|aspect_bin",
        "true_shape_type|size_level",
    ]
    out = []
    for group_field in group_fields:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        parts = group_field.split("|")
        for row in rows:
            key = "|".join(str(row[p]) for p in parts)
            buckets[key].append(row)
        for key, bucket in sorted(buckets.items()):
            out.append(
                {
                    "split": "test",
                    "group_field": group_field,
                    "group_value": key,
                    "sample_count": len(bucket),
                    "mean_total_abs_normalized_error": float(np.mean([safe_float(r["total_abs_normalized_error"]) for r in bucket])),
                    "mean_burial_depth_error_mm": float(np.mean([safe_float(r["burial_depth_error_mm"]) for r in bucket])),
                    "mean_center_xyz_error_mm": float(np.mean([safe_float(r["center_xyz_error_mm"]) for r in bucket])),
                    "shape_accuracy": float(np.mean([bool_value(r["shape_correct"]) for r in bucket])),
                    "catastrophic_failure_count": sum(bool(r["is_catastrophic_failure"]) for r in bucket),
                    "geometry_branch_failure_count": sum(bool(r["is_geometry_branch_failure"]) for r in bucket),
                    "center_outlier_count": sum("center_outlier" in r["failure_tags"] for r in bucket),
                    "burial_outlier_count": sum("burial_outlier" in r["failure_tags"] for r in bucket),
                    "dimension_outlier_count": sum("dimension_outlier" in r["failure_tags"] for r in bucket),
                }
            )
    return out


def branch_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [row for row in rows if row["is_geometry_branch_failure"]]
    pair_counts = Counter(f"{row['true_shape_type']}->{row['pred_shape_type']}" for row in rows if not bool_value(row["shape_correct"]))
    failure_pair_counts = Counter(f"{row['true_shape_type']}->{row['pred_shape_type']}" for row in failures)
    out = [
        {
            "summary_key": "geometry_branch_failure_count",
            "value": len(failures),
            "notes": "shape_misclassified AND center_xyz_error_mm>3 AND burial_depth_error_mm>1",
        },
        {
            "summary_key": "shape_misclassification_count",
            "value": sum(not bool_value(row["shape_correct"]) for row in rows),
            "notes": "; ".join(f"{k}:{v}" for k, v in sorted(pair_counts.items())) or "none",
        },
        {
            "summary_key": "geometry_branch_failure_pairs",
            "value": len(failure_pair_counts),
            "notes": "; ".join(f"{k}:{v}" for k, v in sorted(failure_pair_counts.items())) or "none",
        },
    ]
    for field in ["true_shape_type", "burial_depth_level", "size_level", "aspect_bin", "center_region"]:
        counts = Counter(row[field] for row in failures)
        out.append(
            {
                "summary_key": f"geometry_branch_failure_by_{field}",
                "value": sum(counts.values()),
                "notes": "; ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none",
            }
        )
    return out


def main() -> int:
    args = parse_args()
    rows = [add_failure_tags(row) for row in read_csv(args.replay_metrics) if row.get("split") == "test"]
    if not rows:
        raise RuntimeError("no test rows in replay metrics")
    write_csv(CASES_OUT, rows, CASE_FIELDS)
    groups = group_summary(rows)
    write_csv(
        GROUP_OUT,
        groups,
        [
            "split",
            "group_field",
            "group_value",
            "sample_count",
            "mean_total_abs_normalized_error",
            "mean_burial_depth_error_mm",
            "mean_center_xyz_error_mm",
            "shape_accuracy",
            "catastrophic_failure_count",
            "geometry_branch_failure_count",
            "center_outlier_count",
            "burial_outlier_count",
            "dimension_outlier_count",
        ],
    )
    tails = tail_rows(rows)
    write_csv(TAIL_OUT, tails, ["split", "metric", "unit", "mean", "median", "p75", "p90", "p95", "max"])
    branches = branch_summary(rows)
    write_csv(BRANCH_OUT, branches, ["summary_key", "value", "notes"])

    catastrophic = [row for row in rows if row["is_catastrophic_failure"]]
    branch_failures = [row for row in rows if row["is_geometry_branch_failure"]]
    worst_center = max(rows, key=lambda row: safe_float(row["center_xyz_error_mm"]))
    worst_burial = max(rows, key=lambda row: safe_float(row["burial_depth_error_mm"]))
    total_tail = next(row for row in tails if row["metric"] == "total_abs_normalized_error")
    burial_tail = next(row for row in tails if row["metric"] == "burial_depth_error_mm")
    center_tail = next(row for row in tails if row["metric"] == "center_xyz_error_mm")
    cuboid_to_ellipsoid = sum(row["true_shape_type"] == "internal_cuboid" and row["pred_shape_type"] == "internal_ellipsoid" for row in rows)
    cuboid_to_ellipsoid_branch = sum(row["true_shape_type"] == "internal_cuboid" and row["pred_shape_type"] == "internal_ellipsoid" and row["is_geometry_branch_failure"] for row in rows)
    summary = [
        "22.0 内部缺陷 B2 failure-driven audit",
        f"test_sample_count: {len(rows)}",
        f"total_abs_normalized_error 的 mean/median/p95/max: {float(total_tail['mean']):.3f} / {float(total_tail['median']):.3f} / {float(total_tail['p95']):.3f} / {float(total_tail['max']):.3f}",
        f"burial_depth_error_mm 的 mean/median/p95/max: {float(burial_tail['mean']):.3f} / {float(burial_tail['median']):.3f} / {float(burial_tail['p95']):.3f} / {float(burial_tail['max']):.3f}",
        f"center_xyz_error_mm 的 mean/median/p95/max: {float(center_tail['mean']):.3f} / {float(center_tail['median']):.3f} / {float(center_tail['p95']):.3f} / {float(center_tail['max']):.3f}",
        f"catastrophic_failure_count: {len(catastrophic)}",
        f"geometry_branch_failure_count: {len(branch_failures)}",
        f"cuboid_to_ellipsoid_misclassification_count: {cuboid_to_ellipsoid}",
        f"cuboid_to_ellipsoid_geometry_branch_failure_count: {cuboid_to_ellipsoid_branch}",
        f"worst_center: {worst_center['sample_id']}, true={worst_center['true_shape_type']}, pred={worst_center['pred_shape_type']}, center={safe_float(worst_center['center_xyz_error_mm']):.3f}mm, burial={safe_float(worst_center['burial_depth_error_mm']):.3f}mm, tags={worst_center['failure_tags']}",
        f"worst_burial: {worst_burial['sample_id']}, true={worst_burial['true_shape_type']}, pred={worst_burial['pred_shape_type']}, center={safe_float(worst_burial['center_xyz_error_mm']):.3f}mm, burial={safe_float(worst_burial['burial_depth_error_mm']):.3f}mm, tags={worst_burial['failure_tags']}",
        "feature_baseline_per_sample_comparison: 不可用；当前可追踪的 feature baseline 只有 aggregate/group metrics。",
        "benchmark_status: B2 仍是 internal benchmark candidate，但不是 stable inference model。",
    ]
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
