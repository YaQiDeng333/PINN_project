#!/usr/bin/env python
"""Route decision for the 22.2 internal hard-case top-up plan."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, write_csv


PLAN = ROOT / "results/metrics/internal_defect_hard_case_topup_plan.csv"
TARGETS = ROOT / "results/metrics/internal_defect_hard_case_topup_targets.csv"
EXPECTED = ROOT / "results/metrics/internal_defect_hard_case_expected_coverage.csv"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_hard_case_topup_route_decision_summary.txt"
MATRIX_OUT = ROOT / "results/metrics/internal_defect_hard_case_topup_decision_matrix.csv"


FIELDS = [
    "option",
    "decision",
    "requires_comsol",
    "requires_training",
    "requires_schema_change",
    "updates_current_baseline",
    "reason",
    "next_step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide route after internal hard-case top-up plan.")
    parser.add_argument("--plan-csv", type=Path, default=PLAN)
    parser.add_argument("--targets-csv", type=Path, default=TARGETS)
    parser.add_argument("--expected-coverage", type=Path, default=EXPECTED)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def main() -> int:
    args = parse_args()
    for path in [args.plan_csv, args.targets_csv, args.expected_coverage]:
        if not path.exists():
            raise FileNotFoundError(path)
    plan_rows = read_csv(args.plan_csv)
    target_rows = read_csv(args.targets_csv)
    expected_rows = read_csv(args.expected_coverage)
    focus_pass = all(bool_value(row.get("pass")) for row in expected_rows if row.get("group_field") in {"shape_type", "burial_depth_level", "size_level", "aspect_bin", "center_region"})
    row_count = len(plan_rows)
    target_count = sum(safe_int(row.get("recommended_rows")) for row in target_rows)
    expected_by_target = {row["target_id"]: safe_int(row.get("recommended_rows")) for row in target_rows}
    actual_by_target = Counter(row.get("target_id", "") for row in plan_rows)
    target_quota_pass = all(actual_by_target.get(target_id, 0) == count for target_id, count in expected_by_target.items()) and not (set(actual_by_target) - set(expected_by_target))
    minimum_by_target = {row["target_id"]: safe_int(row.get("minimum_rows")) for row in target_rows}
    target_minimum_pass = all(actual_by_target.get(target_id, 0) >= count for target_id, count in minimum_by_target.items())
    plan_valid = row_count == 120 and target_count == 120 and focus_pass and target_quota_pass and target_minimum_pass
    unique_next_step = "22.2b_targeted_internal_hard_case_topup_generation"

    matrix = [
        {
            "option": "A_targeted_internal_hard_case_topup",
            "decision": "recommended" if plan_valid else "recommended_with_plan_warning",
            "requires_comsol": True,
            "requires_training": False,
            "requires_schema_change": False,
            "updates_current_baseline": False,
            "reason": "22.0/22.1 的 catastrophic failure 仍为 5/40，geometry_branch_failure 仍为 1/40；模型继续微调没有解决 tail，需要 targeted COMSOL hard-case pack。",
            "next_step": unique_next_step,
        },
        {
            "option": "B_further_model_refinement_now",
            "decision": "defer",
            "requires_comsol": False,
            "requires_training": True,
            "requires_schema_change": False,
            "updates_current_baseline": False,
            "reason": "22.1 shape-conditioned/T3 改善均值但 tail 未改善，继续不补数据地调模型收益有限。",
            "next_step": "defer_until_hard_case_pack_available",
        },
        {
            "option": "C_real_internal_inference_smoke_now",
            "decision": "defer",
            "requires_comsol": False,
            "requires_training": False,
            "requires_schema_change": False,
            "updates_current_baseline": False,
            "reason": "B2 与 T3 都不是 stable inference model，center/burial full-shift 风险仍高。",
            "next_step": "wait_for_22_2b_and_22_3",
        },
        {
            "option": "D_revise_internal_schema_now",
            "decision": "reject",
            "requires_comsol": False,
            "requires_training": False,
            "requires_schema_change": True,
            "updates_current_baseline": False,
            "reason": "当前 failure 证据指向 hard-case coverage 不足，而不是 burial_depth/center label 定义缺失或不一致。",
            "next_step": "keep_schema_for_22_2b",
        },
        {
            "option": "E_pause_internal_branch",
            "decision": "reject",
            "requires_comsol": False,
            "requires_training": False,
            "requires_schema_change": False,
            "updates_current_baseline": False,
            "reason": "internal branch 已有 benchmark candidate 和明确 failure strata，适合继续 targeted data route。",
            "next_step": unique_next_step,
        },
    ]
    write_csv(MATRIX_OUT, matrix, FIELDS)

    summary = [
        "22.2 内部/埋藏缺陷 hard-case top-up route decision",
        f"plan_rows: {row_count}",
        "target_topup_N: 120",
        "minimum_usable_N: 72",
        f"target_rows_from_strata: {target_count}",
        f"target_quota_pass: {str(target_quota_pass).lower()}",
        f"target_minimum_pass: {str(target_minimum_pass).lower()}",
        f"actual_rows_by_target: {dict(sorted(actual_by_target.items()))}",
        f"coverage_plan_gate: {'pass' if focus_pass else 'warning'}",
        f"plan_valid: {str(plan_valid).lower()}",
        "unique_next_step: 22.2b targeted COMSOL hard-case top-up pack generation",
        "requires_COMSOL_next: true",
        "requires_training_now: false",
        "requires_schema_change_now: false",
        "further_model_refinement: deferred",
        "real_internal_inference_smoke: deferred",
        "current_baseline_update: false",
        "baseline_status: internal branch 仍不是 CURRENT_BASELINE；surface/near-surface RBC baseline 不变。",
        "reason: B2 与 T3 均未解决 catastrophic tail；hard-case targets 来自 22.0 failure cases 和 22.1 tail metrics。",
        f"decision_matrix: {MATRIX_OUT}",
    ]
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
