#!/usr/bin/env python
"""22.1 shape-conditioned internal model route decision."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, write_csv


SUMMARY = ROOT / "results/summaries/internal_defect_shape_conditioned_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_shape_conditioned_decision_matrix.csv"
TRAINING = ROOT / "results/metrics/internal_defect_shape_conditioned_metrics.csv"
TAIL = ROOT / "results/metrics/internal_defect_shape_conditioned_tail_metrics.csv"
VS_B2 = ROOT / "results/metrics/internal_defect_shape_conditioned_vs_b2.csv"


FIELDS = [
    "option",
    "decision",
    "reason",
    "requires_training",
    "requires_comsol",
    "updates_current_baseline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide 22.1 shape-conditioned internal route.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--tail", type=Path, default=TAIL)
    parser.add_argument("--vs-b2", type=Path, default=VS_B2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def selected_test(rows: list[dict[str, str]]) -> dict[str, str] | None:
    selected = [row for row in rows if row.get("selected_candidate") == "True" and row.get("split") == "test"]
    return selected[0] if selected else None


def metric_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["metric"]: row for row in rows}


def main() -> int:
    args = parse_args()
    metric = selected_test(read_csv(args.training))
    tail = selected_test(read_csv(args.tail))
    vs = metric_lookup(read_csv(args.vs_b2))
    if metric is None or tail is None:
        recommendation = "B_targeted_internal_hard_case_top_up"
        stable = False
        reason = "candidate screen 未产生 validation-eligible 正式候选，不能用 test 反选进入 multi-seed；下一步应针对 hard-case 或更强两阶段结构重新设计。"
        candidate = "none"
        matrix = [
            {
                "option": "A_shape_conditioned_two_stage_model",
                "decision": "blocked",
                "reason": reason,
                "requires_training": True,
                "requires_comsol": False,
                "updates_current_baseline": False,
            },
            {
                "option": "B_targeted_internal_hard_case_top_up",
                "decision": "candidate",
                "reason": "若现有 split 无法通过建模降低 tail failure，后续需要针对 hard cases 补数据。",
                "requires_training": False,
                "requires_comsol": True,
                "updates_current_baseline": False,
            },
        ]
    else:
        candidate = metric["candidate"]
        catastrophic = safe_float(tail["catastrophic_failure_count"])
        geometry = safe_float(tail["geometry_branch_failure_count"])
        center_p95_pass = str(vs.get("center_xyz_error_p95_mm", {}).get("passes_gate", "")).lower() == "true"
        center_max_pass = str(vs.get("center_xyz_error_max_mm", {}).get("passes_gate", "")).lower() == "true"
        burial_p95_pass = str(vs.get("burial_depth_error_p95_mm", {}).get("passes_gate", "")).lower() == "true"
        total_pass = str(vs.get("total_normalized_mae", {}).get("passes_gate", "")).lower() == "true"
        shape_pass = str(vs.get("shape_accuracy", {}).get("passes_gate", "")).lower() == "true"
        stable = catastrophic < 5 and geometry == 0 and center_p95_pass and burial_p95_pass and total_pass and shape_pass
        if stable:
            recommendation = "E_internal_inference_smoke_after_contract"
            reason = "selected shape-conditioned model 降低 catastrophic / geometry branch tail，同时保持 mean 和 shape gate。"
        elif geometry == 0 and catastrophic < 5:
            recommendation = "C_center_burial_focused_refinement"
            reason = "geometry branch 已压到 0，但 center/burial tail 或 mean gate 仍未完全稳定。"
        else:
            recommendation = "B_targeted_internal_hard_case_top_up"
            reason = "shape-conditioned/two-stage 仍未充分降低 catastrophic tail，后续需要 hard-case 数据或更强分支模型。"
        matrix = [
            {
                "option": "A_shape_conditioned_two_stage_model",
                "decision": "completed" if geometry == 0 else "partial",
                "reason": f"selected={candidate}; catastrophic={catastrophic:.0f}; geometry_branch={geometry:.0f}",
                "requires_training": True,
                "requires_comsol": False,
                "updates_current_baseline": False,
            },
            {
                "option": "B_targeted_internal_hard_case_top_up",
                "decision": "defer" if stable else "candidate",
                "reason": "如果 shape-conditioned model 仍保留 hard tail，则针对 compact/large/deep_plus/cuboid-like cases 补样。",
                "requires_training": False,
                "requires_comsol": True,
                "updates_current_baseline": False,
            },
            {
                "option": "C_center_burial_focused_refinement",
                "decision": "candidate" if not stable else "defer",
                "reason": "center/burial tail 仍是稳定推理的直接风险。",
                "requires_training": True,
                "requires_comsol": False,
                "updates_current_baseline": False,
            },
            {
                "option": "D_revise_internal_output_labels",
                "decision": "defer",
                "reason": "当前失败更像 model routing/tail 问题，而不是 schema 全局错误。",
                "requires_training": False,
                "requires_comsol": False,
                "updates_current_baseline": False,
            },
            {
                "option": "E_internal_inference_smoke_after_contract",
                "decision": "recommended" if stable else "defer",
                "reason": "只有 stable tail gate 通过后，才进入真实 internal inference smoke。",
                "requires_training": False,
                "requires_comsol": False,
                "updates_current_baseline": False,
            },
        ]
    write_csv(args.matrix, matrix, FIELDS)
    lines = [
        "22.1 shape-conditioned / two-stage internal defect route decision",
        f"selected_candidate: {candidate}",
        f"stable_inference_candidate: {str(stable).lower()}",
        "current_baseline_update: false",
        "baseline_status: internal branch 仍不是 CURRENT_BASELINE。",
        f"unique_next_step: {recommendation}",
        f"reason: {reason}",
    ]
    if metric and tail:
        lines.extend(
            [
                f"test_total_normalized_mae: {safe_float(metric['total_normalized_mae']):.6f}",
                f"test_burial_depth_mae_mm: {safe_float(metric['burial_depth_mae_mm']):.3f}",
                f"test_center_component_mae_mm: {safe_float(metric['center_xyz_component_mae_mm']):.3f}",
                f"test_shape_accuracy_f1: {safe_float(metric['shape_accuracy']):.6f} / {safe_float(metric['shape_macro_f1']):.6f}",
                f"test_center_p95_max_mm: {safe_float(tail['center_xyz_error_p95_mm']):.3f} / {safe_float(tail['center_xyz_error_max_mm']):.3f}",
                f"test_burial_p95_max_mm: {safe_float(tail['burial_depth_error_p95_mm']):.3f} / {safe_float(tail['burial_depth_error_max_mm']):.3f}",
                f"test_catastrophic_failure_count: {safe_float(tail['catastrophic_failure_count']):.0f}",
                f"test_geometry_branch_failure_count: {safe_float(tail['geometry_branch_failure_count']):.0f}",
            ]
        )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
