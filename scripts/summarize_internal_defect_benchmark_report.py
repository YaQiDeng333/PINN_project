#!/usr/bin/env python
"""生成 21.5 internal defect benchmark report package。

只读取 21.4 已有 summaries/metrics 和 v2_240 manifest；不训练、
不运行 COMSOL、不读写 data/NPZ、不更新 CURRENT_BASELINE.md。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json"
FEATURE_METRICS = ROOT / "results/metrics/internal_defect_v2_feature_baseline_metrics.csv"
NEURAL_METRICS = ROOT / "results/metrics/internal_defect_v2_neural_metrics.csv"
VS_FEATURE = ROOT / "results/metrics/internal_defect_v2_vs_feature_baseline.csv"
DECISION = ROOT / "results/metrics/internal_defect_v2_training_gate_decision_matrix.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_benchmark_report_summary.txt"
REPORT_METRICS = ROOT / "results/metrics/internal_defect_benchmark_report_metrics.csv"
COMPARISON = ROOT / "results/metrics/internal_defect_benchmark_candidate_comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="总结 internal defect v2_240 benchmark candidate。")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--report-metrics", type=Path, default=REPORT_METRICS)
    parser.add_argument("--comparison", type=Path, default=COMPARISON)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def selected_feature_test() -> dict[str, str]:
    rows = read_csv(FEATURE_METRICS)
    for row in rows:
        if row.get("split") == "test" and row.get("selected_model"):
            return row
    raise RuntimeError("selected feature test row not found")


def mean_test() -> dict[str, str]:
    rows = read_csv(VS_FEATURE)
    for row in rows:
        if row.get("model") == "mean_baseline" and row.get("split") == "test":
            return row
    raise RuntimeError("mean baseline test row not found")


def selected_neural(split: str = "test") -> dict[str, str]:
    rows = read_csv(NEURAL_METRICS)
    for row in rows:
        if row.get("split") == split and row.get("selected_seed", "").lower() == "true":
            return row
    raise RuntimeError(f"selected neural {split} row not found")


def run(args: argparse.Namespace) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError(f"manifest dataset_id mismatch: {manifest.get('dataset_id')}")
    mean = mean_test()
    feature = selected_feature_test()
    neural_train = selected_neural("train")
    neural_val = selected_neural("val")
    neural = selected_neural("test")
    decision_rows = read_csv(DECISION)
    learned = {row["question"]: row["answer"] for row in decision_rows}

    comparison_rows = []
    for role, row in [("mean_baseline", mean), ("feature_selected_svr_rbf_C10", feature), ("neural_selected_seed42", neural)]:
        comparison_rows.append(
            {
                "role": role,
                "model": row.get("model") or row.get("candidate"),
                "split": row.get("split"),
                "total_normalized_mae": row.get("total_normalized_mae"),
                "L_mae_mm": row.get("L_mae_mm"),
                "W_mae_mm": row.get("W_mae_mm"),
                "D_mae_mm": row.get("D_mae_mm"),
                "burial_depth_mae_mm": row.get("burial_depth_mae_mm"),
                "center_xyz_mae_mm": row.get("center_xyz_mae_mm"),
                "shape_accuracy": row.get("shape_accuracy"),
                "shape_macro_f1": row.get("shape_macro_f1"),
                "notes": "test-final metrics; no baseline transition",
            }
        )
    write_csv(args.comparison, comparison_rows, ["role", "model", "split", "total_normalized_mae", "L_mae_mm", "W_mae_mm", "D_mae_mm", "burial_depth_mae_mm", "center_xyz_mae_mm", "shape_accuracy", "shape_macro_f1", "notes"])

    report_rows = [
        {"metric": "dataset_id", "value": DATASET_ID, "notes": "explicit registry/manifest dataset"},
        {"metric": "n_samples", "value": manifest.get("n_samples"), "notes": "internal/buried defect v2_240"},
        {"metric": "split_counts", "value": manifest.get("split_counts"), "notes": "train/val/test"},
        {"metric": "shape_counts", "value": manifest.get("shape_counts"), "notes": "balanced 80/80/80"},
        {"metric": "burial_depth_counts", "value": manifest.get("burial_depth_counts"), "notes": "balanced 60 each"},
        {"metric": "feature_selected_model", "value": feature.get("model"), "notes": "validation selected"},
        {"metric": "feature_test_total_normalized_mae", "value": feature.get("total_normalized_mae"), "notes": "test final only"},
        {"metric": "feature_test_LWD_mae_mm", "value": f"{float(feature['L_mae_mm']):.3f}/{float(feature['W_mae_mm']):.3f}/{float(feature['D_mae_mm']):.3f}", "notes": "actual 21.4 CSV value"},
        {"metric": "feature_test_burial_depth_mae_mm", "value": feature.get("burial_depth_mae_mm"), "notes": "feature remains better on burial"},
        {"metric": "feature_test_center_xyz_mae_mm", "value": feature.get("center_xyz_mae_mm"), "notes": ""},
        {"metric": "feature_test_shape_accuracy_f1", "value": f"{float(feature['shape_accuracy']):.6f}/{float(feature['shape_macro_f1']):.6f}", "notes": ""},
        {"metric": "neural_selected_seed", "value": neural.get("seed"), "notes": "validation selected"},
        {"metric": "neural_train_val_test_total_normalized_mae", "value": f"{float(neural_train['total_normalized_mae']):.6f}/{float(neural_val['total_normalized_mae']):.6f}/{float(neural['total_normalized_mae']):.6f}", "notes": ""},
        {"metric": "neural_test_LWD_mae_mm", "value": f"{float(neural['L_mae_mm']):.3f}/{float(neural['W_mae_mm']):.3f}/{float(neural['D_mae_mm']):.3f}", "notes": ""},
        {"metric": "neural_test_burial_depth_mae_mm", "value": neural.get("burial_depth_mae_mm"), "notes": "risk: worse than feature"},
        {"metric": "neural_test_center_xyz_mae_mm", "value": neural.get("center_xyz_mae_mm"), "notes": ""},
        {"metric": "neural_test_shape_accuracy_f1", "value": f"{float(neural['shape_accuracy']):.6f}/{float(neural['shape_macro_f1']):.6f}", "notes": ""},
        {"metric": "beat_mean_baseline", "value": learned.get("neural 是否 beat mean baseline", ""), "notes": ""},
        {"metric": "beat_feature_baseline", "value": learned.get("neural 是否 beat feature baseline", ""), "notes": ""},
        {"metric": "baseline_ready", "value": manifest.get("baseline_ready"), "notes": "must stay false"},
    ]
    write_csv(args.report_metrics, report_rows, ["metric", "value", "notes"])

    feature_burial = f(feature, "burial_depth_mae_mm")
    neural_burial = f(neural, "burial_depth_mae_mm")
    lines = [
        "21.5 internal / buried defect benchmark candidate report",
        f"dataset_id: {DATASET_ID}",
        "task_definition: Bx/By/Bz delta_b -> internal cavity labels: L/W/D, burial_depth, center_xyz, shape_type.",
        f"dataset_identity: N={manifest.get('n_samples')}; split={manifest.get('split_counts')}; shape={manifest.get('shape_counts')}; burial={manifest.get('burial_depth_counts')}; size={manifest.get('size_counts')}; aspect={manifest.get('aspect_counts')}",
        "relation_to_CURRENT_BASELINE: independent internal branch; does not replace surface/near-surface true 3D RBC CURRENT_BASELINE.",
        f"feature_baseline: {feature.get('model')} test total MAE={float(feature['total_normalized_mae']):.6f}; L/W/D={float(feature['L_mae_mm']):.3f}/{float(feature['W_mae_mm']):.3f}/{float(feature['D_mae_mm']):.3f} mm; burial={feature_burial:.3f} mm; center={float(feature['center_xyz_mae_mm']):.3f} mm; shape acc/F1={float(feature['shape_accuracy']):.6f}/{float(feature['shape_macro_f1']):.6f}.",
        f"neural_candidate: seed={neural.get('seed')} test total MAE={float(neural['total_normalized_mae']):.6f}; L/W/D={float(neural['L_mae_mm']):.3f}/{float(neural['W_mae_mm']):.3f}/{float(neural['D_mae_mm']):.3f} mm; burial={neural_burial:.3f} mm; center={float(neural['center_xyz_mae_mm']):.3f} mm; shape acc/F1={float(neural['shape_accuracy']):.6f}/{float(neural['shape_macro_f1']):.6f}.",
        "learnable_params: L/W/D、center_xyz、shape_type 明确可学习；burial_depth 有信号但仍是风险项。",
        f"burial_depth_risk: feature baseline better than neural ({feature_burial:.3f} mm vs {neural_burial:.3f} mm).",
        "why_not_baseline: v2_240 是 training-gate / benchmark-candidate evidence，不是 formal baseline；仍缺 formal benchmark rerun、真实实验验证和 burial-depth-focused follow-up。",
        "route_note: neural beats mean baseline and composite feature baseline, but feature baseline remains an important burial-depth comparator.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
