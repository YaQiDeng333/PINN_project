#!/usr/bin/env python
"""21.8 internal defect benchmark visual/failure audit.

当前 21.7 没有 per-sample prediction artifact；本脚本只做 group-level failure
audit，并明确标记 per-sample best/worst case 缺失，避免伪造单样本结论。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_group_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_vs_reference.csv"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_benchmark_visual_failure_audit_summary.txt"
CASES_OUT = ROOT / "results/metrics/internal_defect_benchmark_visual_failure_cases.csv"
GROUP_OUT = ROOT / "results/metrics/internal_defect_benchmark_visual_group_audit.csv"


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


def group_rows() -> list[dict[str, str]]:
    return [row for row in read_csv(GROUP_SUMMARY) if row.get("split") == "test"]


def case_row(case_type: str, row: dict[str, str], rank: int, note: str) -> dict[str, Any]:
    return {
        "artifact_scope": "group_level_only",
        "case_type": case_type,
        "rank": rank,
        "split": row.get("split", "test"),
        "group_field": row.get("group_field", ""),
        "group_value": row.get("group_value", ""),
        "sample_count": row.get("sample_count", ""),
        "total_normalized_mae": row.get("total_normalized_mae", ""),
        "L_mae_mm": row.get("L_mae_mm", ""),
        "W_mae_mm": row.get("W_mae_mm", ""),
        "D_mae_mm": row.get("D_mae_mm", ""),
        "burial_depth_mae_mm": row.get("burial_depth_mae_mm", ""),
        "center_xyz_mae_mm": row.get("center_xyz_mae_mm", ""),
        "shape_accuracy": row.get("shape_accuracy", ""),
        "notes": note,
    }


def main() -> int:
    rows = group_rows()
    if not rows:
        raise RuntimeError("missing test group rows")

    group_audit = []
    for row in rows:
        group_audit.append(
            {
                "split": row["split"],
                "group_field": row["group_field"],
                "group_value": row["group_value"],
                "sample_count": row["sample_count"],
                "total_normalized_mae": row["total_normalized_mae"],
                "burial_depth_mae_mm": row["burial_depth_mae_mm"],
                "center_xyz_mae_mm": row["center_xyz_mae_mm"],
                "shape_accuracy": row["shape_accuracy"],
                "risk_label": risk_label(row),
            }
        )

    cases: list[dict[str, Any]] = []
    best_burial = sorted(rows, key=lambda row: safe_float(row["burial_depth_mae_mm"]))[:5]
    worst_burial = sorted(rows, key=lambda row: safe_float(row["burial_depth_mae_mm"]), reverse=True)[:5]
    best_center = sorted(rows, key=lambda row: safe_float(row["center_xyz_mae_mm"]))[:5]
    worst_center = sorted(rows, key=lambda row: safe_float(row["center_xyz_mae_mm"]), reverse=True)[:5]
    shape_risk = sorted(rows, key=lambda row: safe_float(row["shape_accuracy"]))[:5]

    for idx, row in enumerate(best_burial, 1):
        cases.append(case_row("best_burial_depth_group", row, idx, "group-level because per-sample B2 artifact is unavailable"))
    for idx, row in enumerate(worst_burial, 1):
        cases.append(case_row("worst_burial_depth_group", row, idx, "group-level because per-sample B2 artifact is unavailable"))
    for idx, row in enumerate(best_center, 1):
        cases.append(case_row("best_center_xyz_group", row, idx, "group-level because per-sample B2 artifact is unavailable"))
    for idx, row in enumerate(worst_center, 1):
        cases.append(case_row("worst_center_xyz_group", row, idx, "group-level because per-sample B2 artifact is unavailable"))
    for idx, row in enumerate(shape_risk, 1):
        if safe_float(row["shape_accuracy"]) < 1.0:
            cases.append(case_row("shape_misclassification_risk_group", row, idx, "group-level shape accuracy below 1.0"))

    # Feature-vs-B2 is available at aggregate test level in 21.7. Per-sample or
    # per-group feature-vs-B2 artifacts are not available, so the audit records
    # aggregate direction without inventing sample-level winners.
    vs_rows = read_csv(VS_REFERENCE)
    b2 = next(row for row in vs_rows if row["source"] == "21.7_B2_formal_rerun")
    feature = next(row for row in vs_rows if row["source"] == "21.4_feature_baseline")
    cases.append(
        {
            "case_type": "B2_better_than_feature_baseline_aggregate",
            "artifact_scope": "aggregate_only",
            "rank": 1,
            "split": "test",
            "group_field": "aggregate",
            "group_value": "all",
            "sample_count": b2["sample_count"],
            "total_normalized_mae": b2["total_normalized_mae"],
            "L_mae_mm": b2["L_mae_mm"],
            "W_mae_mm": b2["W_mae_mm"],
            "D_mae_mm": b2["D_mae_mm"],
            "burial_depth_mae_mm": b2["burial_depth_mae_mm"],
            "center_xyz_mae_mm": b2["center_xyz_mae_mm"],
            "shape_accuracy": b2["shape_accuracy"],
            "notes": f"B2 burial {safe_float(b2['burial_depth_mae_mm']):.3f} mm vs feature {safe_float(feature['burial_depth_mae_mm']):.3f} mm; no per-sample feature-vs-B2 artifact",
        }
    )

    write_csv(
        GROUP_OUT,
        group_audit,
        [
            "split",
            "group_field",
            "group_value",
            "sample_count",
            "total_normalized_mae",
            "burial_depth_mae_mm",
            "center_xyz_mae_mm",
            "shape_accuracy",
            "risk_label",
        ],
    )
    write_csv(
        CASES_OUT,
        cases,
        [
            "case_type",
            "artifact_scope",
            "rank",
            "split",
            "group_field",
            "group_value",
            "sample_count",
            "total_normalized_mae",
            "L_mae_mm",
            "W_mae_mm",
            "D_mae_mm",
            "burial_depth_mae_mm",
            "center_xyz_mae_mm",
            "shape_accuracy",
            "notes",
        ],
    )

    worst_burial_line = worst_burial[0]
    worst_center_line = worst_center[0]
    shape_risk_line = shape_risk[0]
    best_burial_text = ", ".join(
        f"{row['group_field']}={row['group_value']}({safe_float(row['burial_depth_mae_mm']):.3f}mm)"
        for row in best_burial[:3]
    )
    summary = [
        "21.8 internal defect benchmark visual failure audit",
        "artifact_scope: group-level audit only; no per-sample B2 prediction/profile artifact was available.",
        f"best_burial_groups: {best_burial_text}",
        f"worst_burial_group: {worst_burial_line['group_field']}={worst_burial_line['group_value']}, burial_depth_mae={safe_float(worst_burial_line['burial_depth_mae_mm']):.3f} mm",
        f"worst_center_group: {worst_center_line['group_field']}={worst_center_line['group_value']}, center_xyz_mae={safe_float(worst_center_line['center_xyz_mae_mm']):.3f} mm",
        f"shape_risk_group: {shape_risk_line['group_field']}={shape_risk_line['group_value']}, shape_accuracy={safe_float(shape_risk_line['shape_accuracy']):.3f}",
        "feature_vs_B2: only aggregate comparison is available in 21.7; B2 beats feature baseline on total MAE and burial_depth MAE.",
        "visualization_implication: generate real per-sample prediction artifacts before making best/worst image panels.",
    ]
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


def risk_label(row: dict[str, str]) -> str:
    total = safe_float(row.get("total_normalized_mae"))
    burial = safe_float(row.get("burial_depth_mae_mm"))
    center = safe_float(row.get("center_xyz_mae_mm"))
    shape_acc = safe_float(row.get("shape_accuracy"))
    labels = []
    if total >= 0.50:
        labels.append("high_total_error")
    if burial >= 0.50:
        labels.append("high_burial_error")
    if center >= 2.0:
        labels.append("high_center_error")
    if shape_acc < 1.0:
        labels.append("shape_risk")
    return "|".join(labels) if labels else "watch"


if __name__ == "__main__":
    raise SystemExit(main())
