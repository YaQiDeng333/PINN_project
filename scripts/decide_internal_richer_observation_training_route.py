#!/usr/bin/env python
"""23.1 richer-observation training route decision。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from load_internal_richer_observation_dataset import ROOT, read_csv, write_csv


SUMMARY = ROOT / "results/summaries/internal_richer_observation_training_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_richer_observation_training_decision_matrix.csv"
TRAINING = ROOT / "results/metrics/internal_richer_observation_metrics.csv"
VS_REFERENCE = ROOT / "results/metrics/internal_richer_observation_vs_reference.csv"
EVAL_DECISION = ROOT / "results/metrics/internal_richer_observation_evaluation_decision_matrix.csv"

FIELDS = ["decision_item", "status", "value", "threshold_or_reference", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide 23.1 richer-observation training route.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    return parser.parse_args()


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def main() -> int:
    args = parse_args()
    rows = read_csv(TRAINING)
    selected_test = [row for row in rows if row.get("selected_model", "").lower() == "true" and row.get("split") == "test"]
    if not selected_test:
        raise RuntimeError("missing selected test metrics")
    sel = selected_test[0]
    eval_rows = read_csv(EVAL_DECISION)
    selected_eval = [row for row in eval_rows if row.get("selected_for_23_1_training", "").lower() == "true"]
    eval_sel = selected_eval[0] if selected_eval else {}
    vs_rows = read_csv(VS_REFERENCE)
    same_ref = [row for row in vs_rows if row.get("reference") == "23.0_ridge_feature_probe_same_config"]
    old_b2_ref = [row for row in vs_rows if row.get("reference") == "old_B2_v3_hardcase_scope_mismatch"]
    catastrophic = f(sel, "catastrophic_failure_count")
    geometry = f(sel, "geometry_branch_failure_count")
    shape_f1 = f(sel, "shape_macro_f1")
    total = f(sel, "total_normalized_mae")
    center_p95 = f(sel, "center_xyz_error_p95_mm")
    burial_p95 = f(sel, "burial_depth_error_p95_mm")
    stable = catastrophic == 0 and geometry == 0 and shape_f1 >= 0.80 and center_p95 <= 5.0 and burial_p95 <= 1.0
    benchmark_candidate = shape_f1 >= 0.60 and catastrophic <= 2 and geometry <= 1
    need_multiscan = geometry > 0 or shape_f1 < 0.80
    proceed_smoke = stable
    next_step = "23.2_internal_multi_scan_direction_plan" if need_multiscan else ("internal_inference_smoke_with_richer_observation" if proceed_smoke else "richer_observation_model_refinement")
    matrix = [
        {
            "decision_item": "23.0_selected_config",
            "status": "pass" if eval_sel else "fail",
            "value": sel.get("observation_config", ""),
            "threshold_or_reference": "must be R1/R2/R1+R2",
            "notes": "23.1 只按 23.0 选中的配置训练",
        },
        {
            "decision_item": "richer_observation_improves_tail",
            "status": "mixed" if not stable else "pass",
            "value": f"center_p95={center_p95:.3f}; burial_p95={burial_p95:.3f}",
            "threshold_or_reference": "center p95 <=5mm and burial p95 <=1mm preferred",
            "notes": "test final only；由于 30-base diagnostic pack 很小，不能过度泛化",
        },
        {
            "decision_item": "stable_internal_inference_candidate",
            "status": "pass" if stable else "fail",
            "value": stable,
            "threshold_or_reference": "catastrophic=0, geometry=0, shape F1>=0.80, center/burial p95 gate",
            "notes": "未通过时不能称 stable inference model",
        },
        {
            "decision_item": "benchmark_candidate",
            "status": "pass" if benchmark_candidate else "fail",
            "value": benchmark_candidate,
            "threshold_or_reference": "shape F1>=0.60 and catastrophic<=2 and geometry<=1",
            "notes": "通过也仍不是 baseline",
        },
        {
            "decision_item": "need_multi_scan_direction_pack",
            "status": "yes" if need_multiscan else "no",
            "value": need_multiscan,
            "threshold_or_reference": "geometry failure remains or shape F1 weak",
            "notes": "R3 只在 R1/R2 后仍有 shape/geometry branch risk 时进入",
        },
        {
            "decision_item": "next_step",
            "status": "selected",
            "value": next_step,
            "threshold_or_reference": "unique next step",
            "notes": "不更新 CURRENT_BASELINE",
        },
    ]
    args.matrix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.matrix, matrix, FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "# 23.1 internal richer-observation training route decision",
                "",
                f"selected_observation_config: {sel.get('observation_config', '')}",
                f"selected_model: {sel.get('model', '')}",
                f"test_total_normalized_mae: {total:.6f}",
                f"test_center_p95_mm: {center_p95:.3f}",
                f"test_burial_p95_mm: {burial_p95:.3f}",
                f"test_catastrophic_failure_count: {int(catastrophic)}",
                f"test_geometry_branch_failure_count: {int(geometry)}",
                f"test_shape_macro_f1: {shape_f1:.6f}",
                f"stable_internal_inference_candidate: {str(stable).lower()}",
                f"internal_benchmark_candidate: {str(benchmark_candidate).lower()}",
                f"need_multi_scan_direction_pack: {str(need_multiscan).lower()}",
                f"next_step: {next_step}",
                "",
                "结论：23.1 只能形成 internal richer-observation training gate 结论，不更新 CURRENT_BASELINE，也不把 internal branch 写成 baseline。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
