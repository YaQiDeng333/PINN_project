#!/usr/bin/env python
"""根据 21.5 benchmark report 决定 internal defect 下一步路线。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "results/metrics/internal_defect_benchmark_candidate_comparison.csv"
GROUP_AUDIT = ROOT / "results/metrics/internal_defect_benchmark_group_audit.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_benchmark_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_benchmark_decision_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="判定 internal defect benchmark route。")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
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


def find_role(rows: list[dict[str, str]], role: str) -> dict[str, str]:
    for row in rows:
        if row.get("role") == role:
            return row
    raise RuntimeError(f"role not found: {role}")


def run(args: argparse.Namespace) -> int:
    comparison = read_csv(COMPARISON)
    audit = read_csv(GROUP_AUDIT)
    feature = find_role(comparison, "feature_selected_svr_rbf_C10")
    neural = find_role(comparison, "neural_selected_seed42")
    neural_beats_total = f(neural, "total_normalized_mae") < f(feature, "total_normalized_mae")
    neural_beats_center = f(neural, "center_xyz_mae_mm") < f(feature, "center_xyz_mae_mm")
    neural_beats_shape = f(neural, "shape_accuracy") >= f(feature, "shape_accuracy")
    feature_beats_burial = f(feature, "burial_depth_mae_mm") < f(neural, "burial_depth_mae_mm")
    burial_feature_wins = sum(1 for row in audit if row.get("winner_burial") == "feature")
    next_step = "B_improve_burial_depth_head_model"
    rows = [
        {
            "question": "是否完成 internal benchmark candidate consolidation",
            "answer": "是",
            "evidence": "21.4 feature/neural/mean metrics consolidated in report matrix",
            "decision": "A 已完成，不作为下一步",
        },
        {
            "question": "是否应改进 burial-depth head/model",
            "answer": "是" if feature_beats_burial else "否",
            "evidence": f"feature_burial={float(feature['burial_depth_mae_mm']):.3f} mm; neural_burial={float(neural['burial_depth_mae_mm']):.3f} mm; group_feature_burial_wins={burial_feature_wins}",
            "decision": "B",
        },
        {
            "question": "是否应 add shape-conditioned model",
            "answer": "作为次优 ablation",
            "evidence": f"neural_shape_accuracy={float(neural['shape_accuracy']):.6f}; feature_shape_accuracy={float(feature['shape_accuracy']):.6f}",
            "decision": "C secondary",
        },
        {
            "question": "是否需要继续 expand internal dataset",
            "answer": "暂不优先",
            "evidence": "v2_240 已解决 split/coverage blocker，当前 blocker 是 burial-depth modeling trade-off",
            "decision": "D later",
        },
        {
            "question": "是否进入 internal real-data schema/smoke",
            "answer": "暂缓",
            "evidence": "simulation benchmark candidate 尚未 formalized，burial-depth 风险未收口",
            "decision": "E later",
        },
        {
            "question": "是否 pause internal branch",
            "answer": "否",
            "evidence": f"neural_total<{feature['total_normalized_mae']}? {neural_beats_total}; center_better={neural_beats_center}; shape_better={neural_beats_shape}",
            "decision": "continue",
        },
        {
            "question": "下一步唯一建议",
            "answer": next_step,
            "evidence": "feature baseline still beats neural on burial_depth while neural is stronger on total/shape/center.",
            "decision": "route",
        },
    ]
    write_csv(args.matrix, rows, ["question", "answer", "evidence", "decision"])
    lines = [
        "21.5 internal defect benchmark route decision",
        f"next_step: {next_step}",
        "reason: neural candidate is positive for total/shape/center, but burial_depth remains feature-baseline stronger.",
        "baseline_decision: no baseline transition; CURRENT_BASELINE remains surface/near-surface true 3D RBC baseline.",
        "recommended_scope: burial-depth-focused model / feature-fusion diagnostic, with shape-conditioned model as secondary ablation.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
