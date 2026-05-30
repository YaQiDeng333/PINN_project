#!/usr/bin/env python
"""22.7 internal defect inference abstention route decision."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from internal_defect_hardcase_utils import safe_float
from load_internal_defect_pilot_dataset import ROOT, write_csv


METRICS_CSV = ROOT / "results/metrics/internal_defect_inference_abstention_metrics.csv"
SUMMARY_PATH = ROOT / "results/summaries/internal_defect_inference_abstention_route_decision_summary.txt"
DECISION_CSV = ROOT / "results/metrics/internal_defect_inference_abstention_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def by_scope(rows: list[dict[str, str]], scope: str) -> dict[str, str]:
    for row in rows:
        if row["metric_scope"] == scope:
            return row
    raise RuntimeError(f"missing metric scope: {scope}")


def main() -> int:
    rows = read_csv(METRICS_CSV)
    gate = by_scope(rows, "abstention_gate_test")
    accepted = by_scope(rows, "abstention_b2_accepted_test")
    full = by_scope(rows, "no_abstention_b2_full_test")

    coverage = safe_float(gate["coverage_retained"])
    cat_recall = safe_float(gate["catastrophic_failure_recall"])
    geo_recall = safe_float(gate["geometry_branch_failure_recall"])
    false_alarm = safe_float(gate["false_alarm_rate"])
    accepted_center_p95 = safe_float(accepted["center_error_p95_mm"])
    accepted_burial_p95 = safe_float(accepted["burial_error_p95_mm"])
    full_center_p95 = safe_float(full["center_error_p95_mm"])
    full_burial_p95 = safe_float(full["burial_error_p95_mm"])

    runner_available = True
    catches_tail = cat_recall >= 0.80 and geo_recall >= 0.80
    accepted_stable_enough_for_smoke = accepted_center_p95 < full_center_p95 and accepted_burial_p95 < full_burial_p95
    coverage_low = coverage < 0.30

    if runner_available and catches_tail and accepted_stable_enough_for_smoke:
        unique_next = "A_internal_real_sample_metadata_alignment_with_abstention"
        reason = "runner 能捕获 catastrophic/geometry tail，accepted subset 明显降低 center/burial tail；下一步可做真实样品 metadata alignment，但仍必须带 abstention。"
    elif coverage_low:
        unique_next = "B_richer_observation_internal_COMSOL_plan"
        reason = "coverage 过低，说明现有三条 scan-line 的可接受样本太少，应先规划更丰富观测。"
    else:
        unique_next = "E_pause_internal_branch"
        reason = "risk gate 未形成足够安全的 accept/abstain 边界。"

    matrix = [
        {
            "decision_item": "runner_available",
            "result": runner_available,
            "evidence": "run_internal_defect_inference_with_abstention.py completed",
        },
        {
            "decision_item": "tail_failures_captured",
            "result": catches_tail,
            "evidence": f"cat_recall={cat_recall:.3f}; geo_recall={geo_recall:.3f}",
        },
        {
            "decision_item": "accepted_subset_more_stable",
            "result": accepted_stable_enough_for_smoke,
            "evidence": f"center_p95 {full_center_p95:.3f}->{accepted_center_p95:.3f}; burial_p95 {full_burial_p95:.3f}->{accepted_burial_p95:.3f}",
        },
        {
            "decision_item": "coverage_low",
            "result": coverage_low,
            "evidence": f"coverage={coverage:.3f}; false_alarm={false_alarm:.3f}",
        },
        {
            "decision_item": "direct_real_sample_inference_paused",
            "result": True,
            "evidence": "只能进入 metadata alignment；真实推理仍必须先做 schema validation 和 abstention。",
        },
        {
            "decision_item": "unique_next_step",
            "result": unique_next,
            "evidence": reason,
        },
    ]
    write_csv(DECISION_CSV, matrix, ["decision_item", "result", "evidence"])

    lines = [
        "22.7 internal defect inference abstention route decision",
        "",
        f"- runner available：{runner_available}。",
        f"- catastrophic recall={cat_recall:.3f}，geometry_branch recall={geo_recall:.3f}，false alarm={false_alarm:.3f}，coverage={coverage:.3f}。",
        f"- accepted center p95={accepted_center_p95:.3f} mm，full-set center p95={full_center_p95:.3f} mm。",
        f"- accepted burial p95={accepted_burial_p95:.3f} mm，full-set burial p95={full_burial_p95:.3f} mm。",
        f"- coverage 是否过低：{coverage_low}。",
        f"- 唯一下一步：{unique_next}。",
        f"- 理由：{reason}",
        "",
        "口径：可以做带 abstention 的 real-sample metadata alignment，但仍暂缓直接真实样品推理；internal branch 不是 baseline。",
    ]
    write_text(SUMMARY_PATH, "\n".join(lines) + "\n")
    print(unique_next)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
