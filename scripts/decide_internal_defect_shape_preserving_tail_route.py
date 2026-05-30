#!/usr/bin/env python
"""22.4 route decision for shape-preserving internal tail strategy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, write_csv


SUMMARY = ROOT / "results/summaries/internal_defect_shape_preserving_tail_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_shape_preserving_tail_decision_matrix.csv"
STRATEGY = ROOT / "results/metrics/internal_defect_shape_preserving_tail_strategy_matrix.csv"
TRADEOFF = ROOT / "results/metrics/internal_defect_shape_tail_tradeoff_matrix.csv"

FIELDS = [
    "option",
    "decision",
    "evidence",
    "requires_training",
    "requires_comsol",
    "updates_current_baseline",
    "next_stage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide 22.4 shape-preserving tail route.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--strategy", type=Path, default=STRATEGY)
    parser.add_argument("--tradeoff", type=Path, default=TRADEOFF)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_audit(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    for row in rows:
        if row.get("audit_item") == key:
            return row
    return {}


def main() -> int:
    args = parse_args()
    tradeoff = read_csv(args.tradeoff)
    shape_regression = get_audit(tradeoff, "shape_branch_regression")
    center_improvement = get_audit(tradeoff, "center_tail_improvement")
    burial_regression = get_audit(tradeoff, "burial_tail_regression")
    failure_concentration = get_audit(tradeoff, "failure_concentration")

    rows: list[dict[str, Any]] = [
        {
            "option": "A_train_freeze_shape_then_tail_regression_model",
            "decision": "recommended",
            "evidence": "H2 降低 center tail 但 shape F1 从旧 B2 0.841143 降到 0.778163；下一步必须保护 shape branch 后再优化 tail。",
            "requires_training": True,
            "requires_comsol": False,
            "updates_current_baseline": False,
            "next_stage": "22.5 freeze-shape then tail-regression internal model",
        },
        {
            "option": "B_train_shape_confidence_router",
            "decision": "secondary_after_A",
            "evidence": "当前 prediction artifact 没有 shape logits/probability；router 需要下一训练阶段导出 calibrated shape confidence。",
            "requires_training": True,
            "requires_comsol": False,
            "updates_current_baseline": False,
            "next_stage": "after freeze-shape candidate or as inference safety ablation",
        },
        {
            "option": "C_second_targeted_hard_case_top_up",
            "decision": "defer",
            "evidence": failure_concentration.get("interpretation", "failure 不应按盲目 top-up 处理。"),
            "requires_training": False,
            "requires_comsol": True,
            "updates_current_baseline": False,
            "next_stage": "only if A still leaves concentrated strata failure",
        },
        {
            "option": "D_add_uncertainty_abstention_output",
            "decision": "future_safety_layer",
            "evidence": "可降低误用风险，但不能替代模型修复；需要 shape confidence 或 tail risk score。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
            "next_stage": "after stable candidate metrics are available",
        },
        {
            "option": "E_revise_internal_labels_output",
            "decision": "not_recommended",
            "evidence": "22.4 证据指向训练策略破坏 shape branch，不是 label/schema 字段缺失。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
            "next_stage": "none",
        },
        {
            "option": "F_pause_internal_branch",
            "decision": "not_recommended",
            "evidence": "hard-case 数据和 B2/H2 对比已有明确下一步，不需要暂停。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
            "next_stage": "none",
        },
    ]
    write_csv(args.matrix, rows, FIELDS)
    lines = [
        "22.4 shape-preserving internal defect tail route decision",
        "scope: analysis artifact generator；只读取 22.4 audit/strategy CSV 并写 route decision summary/CSV，不训练，不运行 COMSOL，不生成或修改 data/NPZ。",
        "unique_next_step: A_train_freeze_shape_then_tail_regression_model",
        "reason: H2 的 hard-case tail weighting 改善 center tail，但 shape branch 明显退化，burial max 也退化；直接继续 H2 tail weighting 不可取。",
        f"shape_evidence: {shape_regression.get('evidence', '')}",
        f"center_evidence: {center_improvement.get('evidence', '')}",
        f"burial_evidence: {burial_regression.get('evidence', '')}",
        "requires_training: true",
        "requires_new_comsol: false",
        "current_baseline_update: false",
        "stable_inference_status: false; internal branch 仍是 benchmark branch，不是 stable inference model。",
        "deferred: real internal inference smoke, baseline transition, direct H2 tail weighting, blind second top-up。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
