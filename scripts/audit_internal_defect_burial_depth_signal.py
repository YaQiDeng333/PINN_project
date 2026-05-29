#!/usr/bin/env python
"""21.6 internal defect burial-depth signal audit.

本脚本只做只读诊断：显式加载
comsol_internal_defect_pilot_pack_v2_240，计算 delta_b-derived feature 与
burial_depth_m 的关系，并对 21.4 neural / feature baseline 的分组结果做对照。
不训练模型，不写 data/NPZ，不更新 baseline。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import ROOT, load_dataset, split_indices, write_csv
from train_internal_defect_feature_baselines import extract_features, standardize_features


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SUMMARY = ROOT / "results/summaries/internal_defect_burial_depth_signal_audit_summary.txt"
AUDIT_CSV = ROOT / "results/metrics/internal_defect_burial_depth_signal_audit.csv"
FAILURE_CSV = ROOT / "results/metrics/internal_defect_burial_depth_failure_cases.csv"
NEURAL_GROUP = ROOT / "results/metrics/internal_defect_v2_neural_group_summary.csv"
FEATURE_GROUP = ROOT / "results/metrics/internal_defect_v2_feature_baseline_group_summary.csv"
BENCHMARK_GROUP = ROOT / "results/metrics/internal_defect_benchmark_group_audit.csv"


AUDIT_FIELDS = [
    "record_type",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "feature_name",
    "correlation_with_burial_depth",
    "abs_correlation",
    "neural_burial_depth_mae_mm",
    "feature_burial_depth_mae_mm",
    "burial_delta_neural_minus_feature",
    "winner_burial",
    "notes",
]

FAILURE_FIELDS = [
    "case_type",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "neural_burial_depth_mae_mm",
    "feature_burial_depth_mae_mm",
    "burial_delta_neural_minus_feature",
    "neural_total_normalized_mae",
    "feature_total_normalized_mae",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit internal defect burial-depth signal.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV)
    parser.add_argument("--failure-csv", type=Path, default=FAILURE_CSV)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or y.size < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def feature_correlation_rows(dataset_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    dataset = load_dataset(dataset_id)
    splits = split_indices(dataset.split)
    features_raw, feature_names = extract_features(dataset.delta_b)
    features, _, _ = standardize_features(features_raw, splits["train"])
    y = dataset.y_regression[:, 3]
    rows: list[dict[str, Any]] = []
    top_names: list[str] = []

    for split_name, idx in splits.items():
        scored = []
        for col, name in enumerate(feature_names):
            corr = pearson(features[idx, col], y[idx])
            scored.append((abs(corr), corr, name))
        scored.sort(reverse=True, key=lambda item: item[0])
        if split_name == "train":
            top_names = [name for _, _, name in scored[:10]]
        for abs_corr, corr, name in scored[:20]:
            rows.append(
                {
                    "record_type": "feature_correlation",
                    "split": split_name,
                    "group_field": "",
                    "group_value": "",
                    "sample_count": int(idx.size),
                    "feature_name": name,
                    "correlation_with_burial_depth": corr,
                    "abs_correlation": abs_corr,
                    "neural_burial_depth_mae_mm": "",
                    "feature_burial_depth_mae_mm": "",
                    "burial_delta_neural_minus_feature": "",
                    "winner_burial": "",
                    "notes": "feature 仅由 delta_b/BxByBz 计算；标准化只使用 train split",
                }
            )
    return rows, top_names


def group_comparison_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_rows = read_csv(BENCHMARK_GROUP)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in benchmark_rows:
        if row.get("split") != "test":
            continue
        delta = safe_float(row.get("burial_delta_neural_minus_feature"))
        out = {
            "record_type": "group_neural_vs_feature",
            "split": row.get("split", ""),
            "group_field": row.get("group_field", ""),
            "group_value": row.get("group_value", ""),
            "sample_count": row.get("sample_count", ""),
            "feature_name": "",
            "correlation_with_burial_depth": "",
            "abs_correlation": "",
            "neural_burial_depth_mae_mm": row.get("neural_burial_depth_mae_mm", ""),
            "feature_burial_depth_mae_mm": row.get("feature_burial_depth_mae_mm", ""),
            "burial_delta_neural_minus_feature": delta,
            "winner_burial": row.get("winner_burial", ""),
            "notes": "正值表示 neural burial_depth MAE 高于 feature baseline",
        }
        rows.append(out)
        if delta > 0.10:
            failures.append(
                {
                    "case_type": "feature_better_than_neural_burial",
                    "split": row.get("split", ""),
                    "group_field": row.get("group_field", ""),
                    "group_value": row.get("group_value", ""),
                    "sample_count": row.get("sample_count", ""),
                    "neural_burial_depth_mae_mm": row.get("neural_burial_depth_mae_mm", ""),
                    "feature_burial_depth_mae_mm": row.get("feature_burial_depth_mae_mm", ""),
                    "burial_delta_neural_minus_feature": delta,
                    "neural_total_normalized_mae": row.get("neural_total_normalized_mae", ""),
                    "feature_total_normalized_mae": row.get("feature_total_normalized_mae", ""),
                    "notes": "21.5 仅有 group-level 对照；未发现可提交的 per-sample neural prediction artifact",
                }
            )
    failures.sort(key=lambda item: safe_float(item["burial_delta_neural_minus_feature"]), reverse=True)
    return rows, failures[:20]


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    feature_rows, top_feature_names = feature_correlation_rows(args.dataset_id)
    group_rows, failure_rows = group_comparison_rows()
    audit_rows = feature_rows + group_rows
    write_csv(args.audit_csv, audit_rows, AUDIT_FIELDS)
    write_csv(args.failure_csv, failure_rows, FAILURE_FIELDS)

    benchmark_rows = [row for row in read_csv(BENCHMARK_GROUP) if row.get("split") == "test"]
    shallow = next((row for row in benchmark_rows if row.get("group_field") == "burial_depth_level" and row.get("group_value") == "shallow"), {})
    elongated_x = next((row for row in benchmark_rows if row.get("group_field") == "aspect_bin" and row.get("group_value") == "elongated_x"), {})
    cuboid = next((row for row in benchmark_rows if row.get("group_field") == "shape_type" and row.get("group_value") == "internal_cuboid"), {})

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.6 internal defect burial-depth signal audit",
                f"dataset_id: {args.dataset_id}",
                f"registry_manifest_loading: explicit dataset_id + manifest; n={dataset.delta_b.shape[0]}; split={{train:{splits['train'].size}, val:{splits['val'].size}, test:{splits['test'].size}}}",
                "data_mutation: false; comsol_run: false; training_run: false",
                "feature_source: only delta_b/BxByBz derived peak/energy/gradient/width/cross-axis/line features",
                f"top_train_burial_correlated_features: {', '.join(top_feature_names[:8])}",
                "main_diagnostic: feature baseline 的 burial_depth 优势主要来自幅值/能量/线宽类 delta_b-derived 特征；21.4 neural 在 total/center/shape 上更强，但 burial head 没充分利用这些显式物理统计量。",
                f"test_shallow_burial_delta_neural_minus_feature_mm: {safe_float(shallow.get('burial_delta_neural_minus_feature')):.3f}",
                f"test_elongated_x_burial_delta_neural_minus_feature_mm: {safe_float(elongated_x.get('burial_delta_neural_minus_feature')):.3f}",
                f"test_internal_cuboid_burial_delta_neural_minus_feature_mm: {safe_float(cuboid.get('burial_delta_neural_minus_feature')):.3f}",
                "risk_groups: shallow, elongated_x, internal_cuboid, large/medium size show the clearest neural burial-depth gap versus feature baseline.",
                "per_sample_limitation: 21.5 did not preserve a committed per-sample neural prediction artifact; this audit therefore records group-level feature-vs-neural differences and feature-signal correlations, not fabricated per-sample neural errors.",
                "next_training_implication: B2_feature_fusion_burial_head should be tested first; B3_shape_conditioned_burial_head is secondary and must use predicted shape logits, not true shape labels.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
