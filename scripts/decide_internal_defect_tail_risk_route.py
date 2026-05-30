#!/usr/bin/env python
"""基于 22.6 tail-risk gate 结果给出路线决策。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from internal_defect_hardcase_utils import safe_float
from load_internal_defect_pilot_dataset import ROOT, write_csv


METRICS_PATH = ROOT / "results/metrics/internal_defect_tail_risk_gate_metrics.csv"
SUMMARY_PATH = ROOT / "results/summaries/internal_defect_tail_risk_route_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/internal_defect_tail_risk_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def selected_test_row(rows: list[dict[str, str]]) -> dict[str, str]:
    selected = [row for row in rows if row.get("selected", "").lower() == "true" and row.get("split") == "test"]
    if not selected:
        raise RuntimeError("未找到 selected test risk gate metrics")
    return selected[0]


def main() -> int:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(METRICS_PATH)
    rows = read_csv(METRICS_PATH)
    test = selected_test_row(rows)

    cat_recall = safe_float(test["catastrophic_failure_recall"])
    geo_recall = safe_float(test["geometry_branch_failure_recall"])
    false_alarm = safe_float(test["false_alarm_rate"])
    coverage = safe_float(test["coverage_retained"])
    center_reduction = safe_float(test["center_p95_reduction_mm"])
    burial_reduction = safe_float(test["burial_p95_reduction_mm"])
    high_risk_count = int(safe_float(test["high_risk_count"]))
    cat_count = int(safe_float(test["catastrophic_failure_count"]))
    geo_count = int(safe_float(test["geometry_branch_failure_count"]))

    catches_tail = cat_recall >= 0.80 and geo_recall >= 0.75 and center_reduction > 0.0
    acceptable_false_alarm = false_alarm <= 0.50 and coverage >= 0.25
    inference_smoke_ready = catches_tail and acceptable_false_alarm

    if inference_smoke_ready:
        unique_next = "A_internal_inference_smoke_with_abstention"
        next_reason = "risk gate 能捕获主要 catastrophic/geometry tail，且 accepted subset 的 center p95 有下降；下一步可以只做带 abstention 的 internal inference smoke。"
    elif cat_recall < 0.80 or geo_recall < 0.75:
        unique_next = "B_targeted_hard_case_top_up_v2"
        next_reason = "risk gate 对 tail 捕获不足，说明现有观测信号不够稳定，应继续补 hard-case 或更丰富扫描观测。"
    elif false_alarm > 0.50:
        unique_next = "D_collect_richer_observations_more_scan_lines"
        next_reason = "tail 可被捕获但误报过高，现有三条 scan-line 的风险信号区分度不足。"
    else:
        unique_next = "C_revise_output_labels"
        next_reason = "tail 风险可见但无法形成稳定 accept/abstain 边界，应检查 center/burial 输出语义。"

    matrix = [
        {
            "decision_item": "tail_failures_detectable",
            "result": catches_tail,
            "evidence": f"cat_recall={cat_recall:.3f}; geo_recall={geo_recall:.3f}; center_p95_reduction={center_reduction:.3f}mm",
        },
        {
            "decision_item": "false_alarm_acceptable",
            "result": acceptable_false_alarm,
            "evidence": f"false_alarm={false_alarm:.3f}; coverage={coverage:.3f}; high_risk_count={high_risk_count}",
        },
        {
            "decision_item": "enter_internal_inference_smoke_with_abstention",
            "result": inference_smoke_ready,
            "evidence": unique_next,
        },
        {
            "decision_item": "pause_stable_inference_claim",
            "result": True,
            "evidence": "internal branch 仍不是 baseline；abstention 只是风险门控。",
        },
        {
            "decision_item": "unique_next_step",
            "result": unique_next,
            "evidence": next_reason,
        },
    ]
    write_csv(DECISION_MATRIX, matrix, ["decision_item", "result", "evidence"])

    lines = [
        "22.6 internal defect tail-risk route decision",
        "",
        f"- selected gate test：catastrophic recall={cat_recall:.3f} ({cat_count} positives)，geometry recall={geo_recall:.3f} ({geo_count} positives)。",
        f"- false alarm={false_alarm:.3f}，coverage retained={coverage:.3f}。",
        f"- accepted subset center p95 reduction={center_reduction:.3f} mm，burial p95 reduction={burial_reduction:.3f} mm。",
        f"- 是否可进入带 abstention 的 internal inference smoke：{inference_smoke_ready}。",
        f"- 唯一下一步：{unique_next}。",
        f"- 理由：{next_reason}",
        "",
        "口径：risk gate 不能把 internal branch 写成 CURRENT_BASELINE，也不能把高风险样本当作稳定 center/burial 预测。",
    ]
    write_text(SUMMARY_PATH, "\n".join(lines) + "\n")
    print(unique_next)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
