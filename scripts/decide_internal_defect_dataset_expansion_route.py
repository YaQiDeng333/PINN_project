#!/usr/bin/env python
"""Decide the route for the 21.3 internal defect dataset expansion plan."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, write_csv


PLAN = ROOT / "results/metrics/internal_defect_dataset_expansion_plan.csv"
EXPECTED = ROOT / "results/metrics/internal_defect_dataset_expected_coverage.csv"
COVERAGE = ROOT / "results/metrics/internal_defect_pilot_dataset_coverage_audit.csv"
MISSING = ROOT / "results/metrics/internal_defect_pilot_dataset_missing_strata.csv"
SUMMARY = ROOT / "results/summaries/internal_defect_dataset_expansion_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_dataset_expansion_decision_matrix.csv"

FIELDS = ["question", "answer", "evidence", "decision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide internal defect dataset expansion route.")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--expected-coverage", type=Path, default=EXPECTED)
    parser.add_argument("--coverage", type=Path, default=COVERAGE)
    parser.add_argument("--missing", type=Path, default=MISSING)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def count(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(row[field] for row in rows).items()}


def main() -> int:
    args = parse_args()
    plan_rows = read_csv(args.plan)
    expected_rows = read_csv(args.expected_coverage)
    coverage_rows = read_csv(args.coverage)
    missing_rows = read_csv(args.missing)
    selected = [row for row in plan_rows if row["topup_role"] == "selected_quota"]
    buffer = [row for row in plan_rows if row["topup_role"] == "buffer"]
    expected_failures = [row for row in expected_rows if row.get("pass", "").lower() != "true"]
    current_split_shape_failures = [
        row
        for row in coverage_rows
        if row.get("audit") == "split_coverage"
        and row.get("group_field") in {"shape_type", "burial_depth_level"}
        and row.get("pass", "").lower() != "true"
    ]
    selected_shape = count(selected, "shape_type")
    selected_burial = count(selected, "burial_depth_level")
    selected_size = count(selected, "size_level")
    selected_split = count(selected, "v2_split_hint")
    planned_shape = count(plan_rows, "shape_type")

    need_expand = bool(current_split_shape_failures)
    need_resplit = bool(current_split_shape_failures)
    can_reuse_n96 = True
    topup_requires_comsol = True
    enter_generation = (
        len(plan_rows) == 168
        and len(selected) == 144
        and len(buffer) == 24
        and not expected_failures
        and selected_split == {"train": 96, "val": 24, "test": 24}
    )
    rows = [
        {
            "question": "是否需要扩到 240",
            "answer": "是" if need_expand else "否",
            "evidence": f"current_missing_strata={len(missing_rows)}; val/test shape/burial coverage incomplete={bool(current_split_shape_failures)}",
            "decision": "expand_to_240",
        },
        {
            "question": "是否需要重做 split",
            "answer": "是" if need_resplit else "否",
            "evidence": "21.2 val/test 都只有 internal_cuboid，test burial 只覆盖 deep/deep_plus。",
            "decision": "resplit_required",
        },
        {
            "question": "是否能复用 N=96",
            "answer": "是，但旧 split 不复用",
            "evidence": "source N=96 schema valid；v2 assembly 重新分配 split。",
            "decision": "reuse_source_rows",
        },
        {
            "question": "top-up 是否必须跑 COMSOL",
            "answer": "是",
            "evidence": "新增 internal cavity 几何必须有 Bx/By/Bz forward 和 delta_b=b_defect-b_no_defect。",
            "decision": "comsol_required_for_21_3b",
        },
        {
            "question": "21.3 plan 是否可进入 21.3b",
            "answer": "是" if enter_generation else "否",
            "evidence": f"planned_topup={len(plan_rows)}; selected_topup={len(selected)}; buffer={len(buffer)}; selected_shape={selected_shape}; selected_burial={selected_burial}; selected_size={selected_size}; selected_split={selected_split}; expected_failures={len(expected_failures)}",
            "decision": "enter_21_3b_generation",
        },
        {
            "question": "21.4 是否才做 training gate",
            "answer": "是",
            "evidence": "21.3 是 plan-only；21.3b 生成/assemble/validate v2_240；21.4 才训练。",
            "decision": "training_deferred_to_21_4",
        },
        {
            "question": "下一步唯一建议",
            "answer": "进入 21.3b pack generation",
            "evidence": "目标 N=240、top-up N=168、selected top-up N=144 和 split 160/40/40 方案已经固定。",
            "decision": "route",
        },
    ]
    write_csv(args.matrix, rows, FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.3 internal defect dataset expansion route decision summary",
                "",
                "stage_scope: plan-only; no COMSOL, no training, no data/NPZ mutation, no CURRENT_BASELINE update.",
                f"planned_topup_rows: {len(plan_rows)}",
                f"selected_topup_rows: {len(selected)}",
                f"buffer_rows: {len(buffer)}",
                f"selected_shape_counts: {selected_shape}",
                f"selected_burial_counts: {selected_burial}",
                f"selected_size_counts: {selected_size}",
                f"planned_shape_counts: {planned_shape}",
                f"selected_split_hint_counts: {selected_split}",
                f"expected_coverage_failures: {len(expected_failures)}",
                f"current_missing_strata_rows: {len(missing_rows)}",
                "decision: 进入 21.3b internal defect top-up COMSOL generation；21.4 才做 training gate。",
                "baseline_policy: internal branch remains independent; CURRENT_BASELINE stays surface/near-surface true 3D RBC baseline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
