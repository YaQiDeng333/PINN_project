#!/usr/bin/env python
"""21.6 burial-depth refinement route decision."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, load_dataset, split_indices, write_csv


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SUMMARY = ROOT / "results/summaries/internal_defect_burial_depth_refinement_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_burial_depth_refinement_decision_matrix.csv"
REFINED = ROOT / "results/metrics/internal_defect_burial_depth_refined_vs_reference.csv"
SCREEN = ROOT / "results/metrics/internal_defect_burial_depth_candidate_screen_metrics.csv"

FIELDS = ["decision_item", "status", "evidence", "decision", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide internal defect burial-depth refinement route.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--refined", type=Path, default=REFINED)
    parser.add_argument("--screen", type=Path, default=SCREEN)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def selected_refined(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("source") == "21.6_refined_model" and row.get("selected") == "True":
            return row
    return {}


def best_screen_candidate(rows: list[dict[str, str]]) -> str:
    selected = sorted({row.get("candidate", "") for row in rows if row.get("selected_candidate") == "True" and row.get("candidate") != "B0_reference_neural"})
    return selected[0] if selected else ""


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    refined_rows = read_csv(args.refined)
    screen_rows = read_csv(args.screen)
    selected = selected_refined(refined_rows)
    selected_name = selected.get("model", best_screen_candidate(screen_rows) or "none")

    b0 = next((row for row in refined_rows if row.get("source") == "21.4_neural_reference"), {})
    feature = next((row for row in refined_rows if row.get("source") == "21.4_feature_baseline"), {})
    refined_burial = safe_float(selected.get("burial_depth_mae_mm"))
    b0_burial = safe_float(b0.get("burial_depth_mae_mm"), 0.5954996347427368)
    feature_burial = safe_float(feature.get("burial_depth_mae_mm"), 0.47151774168014526)
    refined_total = safe_float(selected.get("total_normalized_mae"))
    b0_total = safe_float(b0.get("total_normalized_mae"), 0.40636563301086426)
    refined_center = safe_float(selected.get("center_xyz_mae_mm"))
    b0_center = safe_float(b0.get("center_xyz_mae_mm"), 1.3799970149993896)
    refined_f1 = safe_float(selected.get("shape_macro_f1"))
    b0_f1 = safe_float(b0.get("shape_macro_f1"), 1.0)

    if not selected:
        next_step = "F_pause_internal_refinement"
        route_reason = "candidate screen did not produce a validation-selected model for multi-seed."
        forms_candidate = False
    else:
        burial_improved = refined_burial < b0_burial
        beat_feature = refined_burial <= feature_burial
        no_secondary_collapse = refined_total <= b0_total * 1.10 and refined_center <= b0_center * 1.15 and refined_f1 >= max(0.95, b0_f1 - 0.05)
        forms_candidate = burial_improved and no_secondary_collapse
        if beat_feature and no_secondary_collapse:
            next_step = "A_internal_benchmark_rerun_candidate_upgrade"
            route_reason = "burial_depth beats the feature baseline without material secondary collapse."
        elif burial_improved and no_secondary_collapse:
            next_step = "B_feature_fusion_internal_model"
            route_reason = "burial_depth improves versus 21.4 neural, but feature baseline remains stronger."
        elif not no_secondary_collapse:
            next_step = "C_shape_conditioned_internal_model"
            route_reason = "burial-depth attempt caused secondary metric risk; revise model structure rather than upgrade."
        else:
            next_step = "D_expand_internal_dataset"
            route_reason = "controlled candidates did not validate a stable burial-depth gain."

    matrix = [
        {
            "decision_item": "registry_manifest_gate",
            "status": "pass",
            "evidence": f"dataset_id={dataset.dataset_id}; n={dataset.delta_b.shape[0]}; split={splits['train'].size}/{splits['val'].size}/{splits['test'].size}",
            "decision": "使用显式 registry/manifest 加载",
            "notes": "未使用 latest/newest scan",
        },
        {
            "decision_item": "burial_depth_improved_vs_neural",
            "status": "pass" if selected and refined_burial < b0_burial else "fail",
            "evidence": f"refined={refined_burial:.3f}mm; 21.4_neural={b0_burial:.3f}mm",
            "decision": "improved" if selected and refined_burial < b0_burial else "not_improved",
            "notes": "主验收项",
        },
        {
            "decision_item": "beat_feature_burial_baseline",
            "status": "pass" if selected and refined_burial <= feature_burial else "fail",
            "evidence": f"refined={refined_burial:.3f}mm; feature={feature_burial:.3f}mm",
            "decision": "beat_feature" if selected and refined_burial <= feature_burial else "feature_remains_better",
            "notes": "feature baseline 仍是 burial_depth 关键 comparator",
        },
        {
            "decision_item": "secondary_metrics_guard",
            "status": "pass" if selected and refined_total <= b0_total * 1.10 and refined_center <= b0_center * 1.15 and refined_f1 >= 0.95 else "fail",
            "evidence": f"total={refined_total:.6f}; center={refined_center:.3f}mm; shape_f1={refined_f1:.6f}",
            "decision": "no_material_collapse" if selected else "not_applicable",
            "notes": "防止只改善 burial_depth 却牺牲 total/center/shape",
        },
        {
            "decision_item": "feature_fusion_value",
            "status": "pass" if selected_name == "B2_feature_fusion_burial_head" and forms_candidate else "warn",
            "evidence": f"selected_candidate={selected_name}",
            "decision": "feature_fusion_supported" if selected_name == "B2_feature_fusion_burial_head" and forms_candidate else "not_primary_evidence",
            "notes": "B2 只使用 delta_b-derived features",
        },
        {
            "decision_item": "shape_conditioned_value",
            "status": "warn" if selected_name != "B3_shape_conditioned_burial_head" else "pass",
            "evidence": f"selected_candidate={selected_name}",
            "decision": "secondary_ablation" if selected_name != "B3_shape_conditioned_burial_head" else "selected",
            "notes": "B3 使用 predicted shape logits，未使用 true shape_type 输入",
        },
        {
            "decision_item": "next_step",
            "status": "pass",
            "evidence": route_reason,
            "decision": next_step,
            "notes": "internal branch only; no CURRENT_BASELINE update",
        },
    ]
    write_csv(args.matrix, matrix, FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.6 internal defect burial-depth refinement route decision",
                f"dataset_id: {args.dataset_id}",
                f"selected_candidate: {selected_name}",
                f"burial_depth_mae_mm: refined={refined_burial:.3f}; 21.4_neural={b0_burial:.3f}; feature_baseline={feature_burial:.3f}",
                f"secondary_metrics: total={refined_total:.6f}; center={refined_center:.3f}mm; shape_f1={refined_f1:.6f}",
                f"forms_internal_refinement_candidate: {str(forms_candidate).lower()}",
                f"next_step: {next_step}",
                f"reason: {route_reason}",
                "baseline_decision: no CURRENT_BASELINE update; internal defect remains independent branch.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
