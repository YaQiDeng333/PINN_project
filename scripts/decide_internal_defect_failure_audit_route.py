#!/usr/bin/env python
"""Route decision for 22.0 internal B2 failure-driven audit."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, write_csv


CASES = ROOT / "results/metrics/internal_defect_b2_failure_cases.csv"
BRANCH = ROOT / "results/metrics/internal_defect_b2_geometry_branch_failure_summary.csv"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_b2_failure_audit_route_decision_summary.txt"
MATRIX_OUT = ROOT / "results/metrics/internal_defect_b2_failure_audit_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def main() -> int:
    rows = [row for row in read_csv(CASES) if row.get("split") == "test"]
    catastrophic = [row for row in rows if bool_value(row.get("is_catastrophic_failure"))]
    branch = [row for row in rows if bool_value(row.get("is_geometry_branch_failure"))]
    branch_cuboid = [row for row in branch if row.get("true_shape_type") == "internal_cuboid"]
    branch_deepish = [row for row in branch if row.get("burial_depth_level") in {"deep", "deep_plus"}]
    branch_rate = len(branch) / max(len(rows), 1)
    catastrophic_rate = len(catastrophic) / max(len(rows), 1)

    stable_inference = catastrophic_rate <= 0.05 and branch_rate <= 0.02
    if branch and len(branch_cuboid) == len(branch):
        recommendation = "B_shape_conditioned_two_stage_internal_model"
        reason = "geometry_branch_failure 集中在 internal_cuboid 且出现 cuboid->ellipsoid，说明需要先分 shape branch 或 two-stage routing。"
    elif catastrophic_rate > 0.10:
        recommendation = "C_center_burial_focused_model_refinement"
        reason = "full-shift catastrophic failures 达到 5/40，主要是 center/burial 同时偏移，真实样本 inference smoke 应暂缓。"
    elif stable_inference:
        recommendation = "E_internal_inference_smoke_acceptable"
        reason = "catastrophic 和 geometry branch failure rate 都低。"
    else:
        recommendation = "C_center_burial_focused_model_refinement"
        reason = "failure 不只是一张图的问题，center/burial tail error 仍偏高。"

    matrix = [
        {
            "option": "A_targeted_internal_hard_case_top_up",
            "decision": "secondary",
            "reason": "如果后续确认 cuboid/deep_plus/medium/compact 样本不足，可补 hard cases；但当前更直接的是 shape/two-stage 建模。",
            "requires_training": False,
            "requires_comsol": True,
            "updates_current_baseline": False,
        },
        {
            "option": "B_shape_conditioned_two_stage_internal_model",
            "decision": "recommended" if recommendation.startswith("B_") else "candidate",
            "reason": "当前唯一 geometry_branch_failure 是 internal_cuboid -> internal_ellipsoid，属于 geometry branch confusion。",
            "requires_training": True,
            "requires_comsol": False,
            "updates_current_baseline": False,
        },
        {
            "option": "C_center_burial_focused_model_refinement",
            "decision": "recommended" if recommendation.startswith("C_") else "candidate",
            "reason": "5/40 full-shift failures 表明 center/burial tail risk 仍需处理。",
            "requires_training": True,
            "requires_comsol": False,
            "updates_current_baseline": False,
        },
        {
            "option": "D_revise_internal_output_labels",
            "decision": "defer",
            "reason": "当前 failure 更像 branch/center-burial coupling，而非标签 schema 全局错误。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
        },
        {
            "option": "E_internal_inference_smoke_acceptable",
            "decision": "reject" if not stable_inference else "acceptable",
            "reason": "存在 full-shift failures 和 geometry branch failure，真实 internal inference smoke 不应直接推进。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
        },
        {
            "option": "F_pause_internal_branch",
            "decision": "reject",
            "reason": "B2 仍是 benchmark candidate，问题可诊断，不需要暂停。",
            "requires_training": False,
            "requires_comsol": False,
            "updates_current_baseline": False,
        },
    ]
    write_csv(
        MATRIX_OUT,
        matrix,
        ["option", "decision", "reason", "requires_training", "requires_comsol", "updates_current_baseline"],
    )

    worst = max(rows, key=lambda row: safe_float(row["center_xyz_error_mm"]))
    summary = [
        "22.0 内部缺陷 B2 failure audit 路线决策",
        f"test_sample_count: {len(rows)}",
        f"catastrophic_failure_count: {len(catastrophic)}",
        f"catastrophic_failure_rate: {catastrophic_rate:.3f}",
        f"geometry_branch_failure_count: {len(branch)}",
        f"geometry_branch_failure_rate: {branch_rate:.3f}",
        f"stable_inference_model: {str(stable_inference).lower()}",
        "benchmark_candidate_status: 保留 B2 作为 internal benchmark candidate，但不是 baseline。",
        "current_baseline_update: false",
        f"worst_case_for_route: {worst['sample_id']} true={worst['true_shape_type']} pred={worst['pred_shape_type']} center={safe_float(worst['center_xyz_error_mm']):.3f}mm burial={safe_float(worst['burial_depth_error_mm']):.3f}mm",
        f"unique_next_step: {recommendation}",
        f"reason: {reason}",
        "real_internal_inference_smoke: 暂缓，先处理 geometry branch 和 center/burial tail risk。",
    ]
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
