#!/usr/bin/env python
"""审计 21.5 internal defect benchmark 的 group-level failures / risks。

21.4 未保存 per-sample prediction artifact，因此本脚本只做 group-level
failure/risk audit，不伪造逐样本 failure ranking。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FEATURE_GROUP = ROOT / "results/metrics/internal_defect_v2_feature_baseline_group_summary.csv"
NEURAL_GROUP = ROOT / "results/metrics/internal_defect_v2_neural_group_summary.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_benchmark_failure_audit_summary.txt"
GROUP_AUDIT = ROOT / "results/metrics/internal_defect_benchmark_group_audit.csv"
FAILURE_CASES = ROOT / "results/metrics/internal_defect_benchmark_failure_cases.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 internal defect benchmark failures。")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--group-audit", type=Path, default=GROUP_AUDIT)
    parser.add_argument("--failure-cases", type=Path, default=FAILURE_CASES)
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
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("split", ""), row.get("group_field", ""), row.get("group_value", ""))


def test_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("split") == "test"]


def run(args: argparse.Namespace) -> int:
    feature_rows = test_rows(read_csv(FEATURE_GROUP))
    neural_rows = test_rows(read_csv(NEURAL_GROUP))
    feature_by_key = {key(row): row for row in feature_rows}
    audit_rows: list[dict[str, Any]] = []
    for nrow in neural_rows:
        frow = feature_by_key.get(key(nrow))
        if not frow:
            continue
        total_delta = f(nrow, "total_normalized_mae") - f(frow, "total_normalized_mae")
        burial_delta = f(nrow, "burial_depth_mae_mm") - f(frow, "burial_depth_mae_mm")
        center_delta = f(nrow, "center_xyz_mae_mm") - f(frow, "center_xyz_mae_mm")
        shape_delta = f(nrow, "shape_accuracy") - f(frow, "shape_accuracy")
        audit_rows.append(
            {
                "split": nrow.get("split"),
                "group_field": nrow.get("group_field"),
                "group_value": nrow.get("group_value"),
                "sample_count": nrow.get("sample_count"),
                "neural_total_normalized_mae": nrow.get("total_normalized_mae"),
                "feature_total_normalized_mae": frow.get("total_normalized_mae"),
                "total_delta_neural_minus_feature": total_delta,
                "neural_burial_depth_mae_mm": nrow.get("burial_depth_mae_mm"),
                "feature_burial_depth_mae_mm": frow.get("burial_depth_mae_mm"),
                "burial_delta_neural_minus_feature": burial_delta,
                "neural_center_xyz_mae_mm": nrow.get("center_xyz_mae_mm"),
                "feature_center_xyz_mae_mm": frow.get("center_xyz_mae_mm"),
                "center_delta_neural_minus_feature": center_delta,
                "neural_shape_accuracy": nrow.get("shape_accuracy"),
                "feature_shape_accuracy": frow.get("shape_accuracy"),
                "shape_accuracy_delta_neural_minus_feature": shape_delta,
                "winner_total": "neural" if total_delta < 0 else "feature",
                "winner_burial": "neural" if burial_delta < 0 else "feature",
                "winner_center": "neural" if center_delta < 0 else "feature",
            }
        )
    write_csv(
        args.group_audit,
        audit_rows,
        [
            "split",
            "group_field",
            "group_value",
            "sample_count",
            "neural_total_normalized_mae",
            "feature_total_normalized_mae",
            "total_delta_neural_minus_feature",
            "neural_burial_depth_mae_mm",
            "feature_burial_depth_mae_mm",
            "burial_delta_neural_minus_feature",
            "neural_center_xyz_mae_mm",
            "feature_center_xyz_mae_mm",
            "center_delta_neural_minus_feature",
            "neural_shape_accuracy",
            "feature_shape_accuracy",
            "shape_accuracy_delta_neural_minus_feature",
            "winner_total",
            "winner_burial",
            "winner_center",
        ],
    )

    cases: list[dict[str, Any]] = []
    for row in sorted(audit_rows, key=lambda r: float(r["neural_burial_depth_mae_mm"]), reverse=True)[:8]:
        cases.append({**row, "case_type": "high_neural_burial_depth_error", "notes": "group-level case; no per-sample prediction artifact"})
    for row in sorted(audit_rows, key=lambda r: float(r["neural_center_xyz_mae_mm"]), reverse=True)[:8]:
        cases.append({**row, "case_type": "high_neural_center_xyz_error", "notes": "group-level case; no per-sample prediction artifact"})
    for row in sorted(audit_rows, key=lambda r: float(r["burial_delta_neural_minus_feature"]), reverse=True)[:8]:
        cases.append({**row, "case_type": "feature_better_than_neural_burial", "notes": "positive delta means feature has lower burial MAE"})
    for row in sorted(audit_rows, key=lambda r: float(r["total_delta_neural_minus_feature"]))[:8]:
        cases.append({**row, "case_type": "neural_better_than_feature_total", "notes": "negative delta means neural has lower total MAE"})
    write_csv(args.failure_cases, cases, list(cases[0].keys()) if cases else ["case_type"])

    total_feature_wins = sum(1 for row in audit_rows if row["winner_total"] == "feature")
    burial_feature_wins = sum(1 for row in audit_rows if row["winner_burial"] == "feature")
    center_neural_wins = sum(1 for row in audit_rows if row["winner_center"] == "neural")
    worst_burial = max(audit_rows, key=lambda row: float(row["neural_burial_depth_mae_mm"]))
    worst_center = max(audit_rows, key=lambda row: float(row["neural_center_xyz_mae_mm"]))
    lines = [
        "21.5 internal defect benchmark failure / group audit",
        "audit_scope: group-level only; 21.4 did not persist per-sample predictions/checkpoints.",
        f"group_rows_compared: {len(audit_rows)}",
        f"total_metric_feature_wins: {total_feature_wins}",
        f"burial_metric_feature_wins: {burial_feature_wins}",
        f"center_metric_neural_wins: {center_neural_wins}",
        f"worst_neural_burial_group: {worst_burial['group_field']}={worst_burial['group_value']}; burial_mae={float(worst_burial['neural_burial_depth_mae_mm']):.3f} mm; feature={float(worst_burial['feature_burial_depth_mae_mm']):.3f} mm",
        f"worst_neural_center_group: {worst_center['group_field']}={worst_center['group_value']}; center_mae={float(worst_center['neural_center_xyz_mae_mm']):.3f} mm; feature={float(worst_center['feature_center_xyz_mae_mm']):.3f} mm",
        "risk: burial_depth remains the clearest feature-baseline advantage and should drive the next model diagnostic.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
