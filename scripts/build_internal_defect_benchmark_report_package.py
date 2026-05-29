#!/usr/bin/env python
"""21.8 internal defect benchmark report package.

只读取 registry / manifest / existing metrics，不读取或写入 NPZ。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json"
RERUN_SUMMARY = ROOT / "results/summaries/internal_defect_benchmark_rerun_b2_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_seed_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_vs_reference.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_group_summary.csv"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_benchmark_report_package_summary.txt"
METRICS_OUT = ROOT / "results/metrics/internal_defect_benchmark_report_package_metrics.csv"
COMPARISON_OUT = ROOT / "results/metrics/internal_defect_benchmark_candidate_comparison_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def selected_seed_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if str(row.get("selected_seed", "")).lower() == "true":
            return row
    raise RuntimeError("missing selected seed row")


def reference_row(rows: list[dict[str, str]], source: str) -> dict[str, str]:
    for row in rows:
        if row.get("source") == source:
            return row
    raise RuntimeError(f"missing reference source: {source}")


def main() -> int:
    if DATASET_ID not in REGISTRY.read_text(encoding="utf-8"):
        raise RuntimeError(f"{DATASET_ID} missing from COMSOL_DATA_REGISTRY.md")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError("manifest dataset_id mismatch")
    if manifest.get("baseline_ready") is not False:
        raise RuntimeError("internal v2_240 must not be baseline_ready")

    seed_rows = read_csv(SEED_SUMMARY)
    selected = selected_seed_row(seed_rows)
    vs_rows = read_csv(VS_REFERENCE)
    b0 = reference_row(vs_rows, "21.4_neural_reference")
    feature = reference_row(vs_rows, "21.4_feature_baseline")
    b2 = reference_row(vs_rows, "21.7_B2_formal_rerun")

    comparison_rows = []
    for role, row in [
        ("21.4 neural reference", b0),
        ("21.4 feature baseline", feature),
        ("21.7 B2 benchmark candidate", b2),
    ]:
        comparison_rows.append(
            {
                "role": role,
                "model": row["model"],
                "source": row["source"],
                "selected": row["selected"],
                "test_total_normalized_mae": row["total_normalized_mae"],
                "test_L_mae_mm": row["L_mae_mm"],
                "test_W_mae_mm": row["W_mae_mm"],
                "test_D_mae_mm": row["D_mae_mm"],
                "test_burial_depth_mae_mm": row["burial_depth_mae_mm"],
                "test_center_xyz_mae_mm": row["center_xyz_mae_mm"],
                "test_shape_accuracy": row["shape_accuracy"],
                "test_shape_macro_f1": row["shape_macro_f1"],
                "baseline_status": "not_current_baseline",
            }
        )

    metric_rows = [
        {
            "metric": "selected_seed",
            "value": selected["seed"],
            "unit": "seed",
            "interpretation": "validation-only selected B2 rerun seed",
        },
        {
            "metric": "test_total_normalized_mae",
            "value": selected["test_total_normalized_mae"],
            "unit": "normalized",
            "interpretation": "B2 beats 21.4 neural and feature baseline on composite score",
        },
        {
            "metric": "test_burial_depth_mae_mm",
            "value": selected["test_burial_depth_mae_mm"],
            "unit": "mm",
            "interpretation": "B2 beats 21.4 neural 0.595 mm and feature baseline 0.472 mm",
        },
        {
            "metric": "test_center_xyz_mae_mm",
            "value": selected["test_center_xyz_mae_mm"],
            "unit": "mm",
            "interpretation": "minor trade-off vs 21.4 neural, still better than feature baseline",
        },
        {
            "metric": "test_shape_macro_f1",
            "value": selected["test_shape_macro_f1"],
            "unit": "f1",
            "interpretation": "minor trade-off vs 21.4 neural, still above feature baseline",
        },
    ]

    write_csv(
        COMPARISON_OUT,
        comparison_rows,
        [
            "role",
            "model",
            "source",
            "selected",
            "test_total_normalized_mae",
            "test_L_mae_mm",
            "test_W_mae_mm",
            "test_D_mae_mm",
            "test_burial_depth_mae_mm",
            "test_center_xyz_mae_mm",
            "test_shape_accuracy",
            "test_shape_macro_f1",
            "baseline_status",
        ],
    )
    write_csv(METRICS_OUT, metric_rows, ["metric", "value", "unit", "interpretation"])

    group_rows = [row for row in read_csv(GROUP_SUMMARY) if row.get("split") == "test"]
    weakest_groups = sorted(group_rows, key=lambda row: safe_float(row.get("total_normalized_mae")), reverse=True)[:5]
    summary_lines = [
        "21.8 internal defect benchmark report package",
        "",
        "任务定义：internal / buried defect branch 预测内部缺陷的 L/W/D、burial_depth、center_xyz 和 shape_type；它不是 surface RBC profile baseline。",
        f"dataset_id: {DATASET_ID}",
        f"coverage: N={manifest['n_samples']}; split={manifest['split_counts']}; shape={manifest['shape_counts']}; burial_depth={manifest['burial_depth_counts']}; size={manifest['size_counts']}; aspect={manifest['aspect_counts']}",
        "model_chain: Bx/By/Bz delta_b -> Conv1D encoder + delta_b-derived feature MLP -> L/W/D, burial_depth, center_xyz, shape_type heads.",
        "B2 input policy: delta_b/BxByBz and delta_b-derived features only; no true shape_type, burial bin, size/aspect, split, or sample_id as model input.",
        f"selected_seed: {selected['seed']}; best_epoch: {selected['best_epoch']}",
        f"train/val/test total MAE: {selected['train_total_normalized_mae']} / {selected['val_total_normalized_mae']} / {selected['test_total_normalized_mae']}",
        f"test L/W/D MAE: {selected['test_L_mae_mm']} / {selected['test_W_mae_mm']} / {selected['test_D_mae_mm']} mm",
        f"test burial_depth MAE: {selected['test_burial_depth_mae_mm']} mm",
        f"test center_xyz MAE: {selected['test_center_xyz_mae_mm']} mm",
        f"test shape accuracy/F1: {selected['test_shape_accuracy']} / {selected['test_shape_macro_f1']}",
        f"comparison: B2 total={b2['total_normalized_mae']}, 21.4 neural total={b0['total_normalized_mae']}, feature baseline total={feature['total_normalized_mae']}",
        f"burial_depth comparison: B2={b2['burial_depth_mae_mm']} mm, 21.4 neural={b0['burial_depth_mae_mm']} mm, feature baseline={feature['burial_depth_mae_mm']} mm",
        "learnable_outputs: L/W/D, burial_depth, center_xyz, shape_type all show learnable signal in v2_240.",
        "limitations: COMSOL simulation only; shapes limited to internal_sphere/internal_ellipsoid/internal_cuboid; no real data validation; not CURRENT_BASELINE.",
        "branch_boundary: CURRENT_BASELINE remains the surface / near-surface true 3D RBC baseline; internal defect is an independent benchmark candidate branch.",
        "",
        "弱势分组（按 test total_normalized_mae 排序）：",
    ]
    for row in weakest_groups:
        summary_lines.append(
            f"- {row['group_field']}={row['group_value']}: total={safe_float(row['total_normalized_mae']):.3f}, burial={safe_float(row['burial_depth_mae_mm']):.3f} mm, center={safe_float(row['center_xyz_mae_mm']):.3f} mm, shape_acc={safe_float(row['shape_accuracy']):.3f}"
        )

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
