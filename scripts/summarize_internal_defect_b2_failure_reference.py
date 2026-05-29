#!/usr/bin/env python
"""22.1 固定 B2 failure reference。

只读取 21.9/22.0 已有 artifact 与 replay/failure metrics，不训练、不运行
COMSOL、不写 data/NPZ。输出供 shape-conditioned 候选 screen 使用的固定 B2
tail reference。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import ROOT


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
ARTIFACT_MANIFEST = ROOT / "results/manifests/internal_defect_b2_inference_artifact_manifest.json"
REPLAY_METRICS = ROOT / "results/metrics/internal_defect_b2_inference_replay_metrics.csv"
FAILURE_CASES = ROOT / "results/metrics/internal_defect_b2_failure_cases.csv"
TAIL_SUMMARY = ROOT / "results/metrics/internal_defect_b2_tail_error_summary.csv"
BRANCH_SUMMARY = ROOT / "results/metrics/internal_defect_b2_geometry_branch_failure_summary.csv"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_shape_conditioned_reference_summary.txt"
METRICS_OUT = ROOT / "results/metrics/internal_defect_shape_conditioned_reference_metrics.csv"


FIELDS = [
    "reference",
    "split",
    "sample_count",
    "total_error_mean",
    "total_error_median",
    "total_error_p90",
    "total_error_p95",
    "total_error_max",
    "burial_depth_error_mean_mm",
    "burial_depth_error_median_mm",
    "burial_depth_error_p90_mm",
    "burial_depth_error_p95_mm",
    "burial_depth_error_max_mm",
    "center_xyz_error_mean_mm",
    "center_xyz_error_median_mm",
    "center_xyz_error_p90_mm",
    "center_xyz_error_p95_mm",
    "center_xyz_error_max_mm",
    "shape_accuracy",
    "shape_error_rate",
    "catastrophic_failure_count",
    "catastrophic_failure_rate",
    "geometry_branch_failure_count",
    "geometry_branch_failure_rate",
    "worst_center_sample_id",
    "worst_burial_sample_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fixed B2 failure reference for 22.1.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--artifact-manifest", type=Path, default=ARTIFACT_MANIFEST)
    parser.add_argument("--replay", type=Path, default=REPLAY_METRICS)
    parser.add_argument("--failure-cases", type=Path, default=FAILURE_CASES)
    parser.add_argument("--summary", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--metrics", type=Path, default=METRICS_OUT)
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


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def tag_has(row: dict[str, str], tag: str) -> bool:
    return tag in str(row.get("failure_tags", "")).split("|")


def split_row(split: str, rows: list[dict[str, str]], case_rows: list[dict[str, str]]) -> dict[str, Any]:
    replay = [row for row in rows if row.get("split") == split]
    cases = [row for row in case_rows if row.get("split") == split]
    if not replay:
        raise RuntimeError(f"missing replay rows for split={split}")
    total = quantiles([safe_float(row["total_abs_normalized_error"]) for row in replay])
    burial = quantiles([safe_float(row["burial_depth_error_mm"]) for row in replay])
    center = quantiles([safe_float(row["center_xyz_error_mm"]) for row in replay])
    shape_correct = [bool_value(row["shape_correct"]) for row in replay]
    catastrophic = [row for row in cases if tag_has(row, "full_shift_failure")]
    branch = [row for row in cases if tag_has(row, "geometry_branch_failure")]
    worst_center = max(replay, key=lambda row: safe_float(row["center_xyz_error_mm"]))
    worst_burial = max(replay, key=lambda row: safe_float(row["burial_depth_error_mm"]))
    n = len(replay)
    return {
        "reference": "B2_feature_fusion_burial_head",
        "split": split,
        "sample_count": n,
        "total_error_mean": total["mean"],
        "total_error_median": total["median"],
        "total_error_p90": total["p90"],
        "total_error_p95": total["p95"],
        "total_error_max": total["max"],
        "burial_depth_error_mean_mm": burial["mean"],
        "burial_depth_error_median_mm": burial["median"],
        "burial_depth_error_p90_mm": burial["p90"],
        "burial_depth_error_p95_mm": burial["p95"],
        "burial_depth_error_max_mm": burial["max"],
        "center_xyz_error_mean_mm": center["mean"],
        "center_xyz_error_median_mm": center["median"],
        "center_xyz_error_p90_mm": center["p90"],
        "center_xyz_error_p95_mm": center["p95"],
        "center_xyz_error_max_mm": center["max"],
        "shape_accuracy": float(np.mean(shape_correct)),
        "shape_error_rate": 1.0 - float(np.mean(shape_correct)),
        "catastrophic_failure_count": len(catastrophic),
        "catastrophic_failure_rate": len(catastrophic) / n,
        "geometry_branch_failure_count": len(branch),
        "geometry_branch_failure_rate": len(branch) / n,
        "worst_center_sample_id": worst_center["sample_id"],
        "worst_burial_sample_id": worst_burial["sample_id"],
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.artifact_manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != args.dataset_id:
        raise RuntimeError("B2 artifact manifest dataset_id mismatch")
    rows = read_csv(args.replay)
    cases = read_csv(args.failure_cases)
    out = [split_row(split, rows, cases) for split in ["train", "val", "test"]]
    write_csv(args.metrics, out, FIELDS)
    test = next(row for row in out if row["split"] == "test")
    lines = [
        "22.1 shape-conditioned internal model 的 B2 failure reference",
        f"dataset_id: {args.dataset_id}",
        f"artifact_manifest: {args.artifact_manifest}",
        "reference_model: B2_feature_fusion_burial_head",
        f"test_total_error_mean_median_p95_max: {test['total_error_mean']:.3f} / {test['total_error_median']:.3f} / {test['total_error_p95']:.3f} / {test['total_error_max']:.3f}",
        f"test_burial_depth_error_mean_median_p95_max_mm: {test['burial_depth_error_mean_mm']:.3f} / {test['burial_depth_error_median_mm']:.3f} / {test['burial_depth_error_p95_mm']:.3f} / {test['burial_depth_error_max_mm']:.3f}",
        f"test_center_xyz_error_mean_median_p95_max_mm: {test['center_xyz_error_mean_mm']:.3f} / {test['center_xyz_error_median_mm']:.3f} / {test['center_xyz_error_p95_mm']:.3f} / {test['center_xyz_error_max_mm']:.3f}",
        f"test_catastrophic_failure_count: {test['catastrophic_failure_count']}",
        f"test_geometry_branch_failure_count: {test['geometry_branch_failure_count']}",
        f"worst_center_sample_id: {test['worst_center_sample_id']}",
        f"worst_burial_sample_id: {test['worst_burial_sample_id']}",
        "current_baseline_update: false",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
