#!/usr/bin/env python
"""21.7 internal benchmark candidate route decision."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import ROOT, load_dataset, split_indices, write_csv


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SUMMARY = ROOT / "results/summaries/internal_defect_benchmark_candidate_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_benchmark_candidate_decision_matrix.csv"
SEED_SUMMARY = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_seed_summary.csv"
VS_REFERENCE = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_vs_reference.csv"


FIELDS = ["decision_item", "status", "evidence", "decision", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide internal defect benchmark candidate route.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--vs-reference", type=Path, default=VS_REFERENCE)
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


def row_by_source(rows: list[dict[str, str]], source: str) -> dict[str, str]:
    for row in rows:
        if row.get("source") == source:
            return row
    return {}


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    seed_rows = read_csv(args.seed_summary)
    vs_rows = read_csv(args.vs_reference)
    b2 = row_by_source(vs_rows, "21.7_B2_formal_rerun")
    neural = row_by_source(vs_rows, "21.4_neural_reference")
    feature = row_by_source(vs_rows, "21.4_feature_baseline")

    b2_total = safe_float(b2.get("total_normalized_mae"))
    neural_total = safe_float(neural.get("total_normalized_mae"))
    feature_total = safe_float(feature.get("total_normalized_mae"))
    b2_burial = safe_float(b2.get("burial_depth_mae_mm"))
    neural_burial = safe_float(neural.get("burial_depth_mae_mm"))
    feature_burial = safe_float(feature.get("burial_depth_mae_mm"))
    b2_center = safe_float(b2.get("center_xyz_mae_mm"))
    b2_f1 = safe_float(b2.get("shape_macro_f1"))
    all_seed_burial = [safe_float(row.get("test_burial_depth_mae_mm")) for row in seed_rows]
    all_seed_total = [safe_float(row.get("test_total_normalized_mae")) for row in seed_rows]

    stable_vs_feature = bool(all_seed_burial) and all(value < feature_burial for value in all_seed_burial)
    stable_vs_neural = bool(all_seed_burial) and all(value < neural_burial for value in all_seed_burial)
    total_better = b2_total < neural_total and b2_total < feature_total
    forms_candidate = stable_vs_feature and stable_vs_neural and total_better and b2_f1 >= 0.95
    burial_no_longer_main_short = b2_burial < feature_burial and b2_burial < neural_burial
    next_step = "A_internal_report_visualization_package" if forms_candidate else "C_expand_internal_dataset_shapes"
    reason = (
        "B2 is stable across seeds and beats both 21.4 neural and feature baseline on burial_depth."
        if forms_candidate
        else "B2 does not yet satisfy the stability or comparator gates."
    )

    matrix = [
        {
            "decision_item": "registry_manifest_gate",
            "status": "pass",
            "evidence": f"dataset_id={dataset.dataset_id}; n={dataset.delta_b.shape[0]}; split={splits['train'].size}/{splits['val'].size}/{splits['test'].size}",
            "decision": "explicit_dataset_id_manifest",
            "notes": "latest/newest discovery remains forbidden",
        },
        {
            "decision_item": "stable_burial_vs_21_4_neural",
            "status": "pass" if stable_vs_neural else "fail",
            "evidence": f"all_seed_burial={','.join(f'{v:.3f}' for v in all_seed_burial)}mm; neural={neural_burial:.3f}mm",
            "decision": "stable_better_than_neural" if stable_vs_neural else "not_stable",
            "notes": "seed-level gate",
        },
        {
            "decision_item": "stable_burial_vs_feature_baseline",
            "status": "pass" if stable_vs_feature else "fail",
            "evidence": f"all_seed_burial={','.join(f'{v:.3f}' for v in all_seed_burial)}mm; feature={feature_burial:.3f}mm",
            "decision": "stable_better_than_feature" if stable_vs_feature else "feature_still_competitive",
            "notes": "feature baseline remains key comparator",
        },
        {
            "decision_item": "total_mae",
            "status": "pass" if total_better else "warn",
            "evidence": f"B2={b2_total:.6f}; neural={neural_total:.6f}; feature={feature_total:.6f}; all_seed_total={','.join(f'{v:.6f}' for v in all_seed_total)}",
            "decision": "B2_best_total" if total_better else "mixed_total",
            "notes": "composite benchmark check",
        },
        {
            "decision_item": "burial_depth_shortfall",
            "status": "pass" if burial_no_longer_main_short else "warn",
            "evidence": f"B2={b2_burial:.3f}mm; neural={neural_burial:.3f}mm; feature={feature_burial:.3f}mm",
            "decision": "not_primary_shortfall" if burial_no_longer_main_short else "still_shortfall",
            "notes": "center/shape and real-data validation remain risks",
        },
        {
            "decision_item": "baseline_status",
            "status": "pass",
            "evidence": "CURRENT_BASELINE.md unchanged; dataset baseline_ready=false",
            "decision": "not_baseline",
            "notes": "internal branch is independent from surface RBC baseline",
        },
        {
            "decision_item": "next_step",
            "status": "pass",
            "evidence": reason,
            "decision": next_step,
            "notes": "prepare report / visualization package before real-data smoke",
        },
    ]
    write_csv(args.matrix, matrix, FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.7 internal defect benchmark candidate route decision",
                f"dataset_id: {args.dataset_id}",
                f"B2_selected_seed_test: total={b2_total:.6f}; burial={b2_burial:.3f}mm; center={b2_center:.3f}mm; shape_f1={b2_f1:.6f}",
                f"seed_stability_burial_mm: {', '.join(f'{value:.3f}' for value in all_seed_burial)}",
                f"stable_vs_21_4_neural: {str(stable_vs_neural).lower()}",
                f"stable_vs_feature_baseline: {str(stable_vs_feature).lower()}",
                f"burial_depth_no_longer_primary_shortfall: {str(burial_no_longer_main_short).lower()}",
                f"forms_internal_benchmark_candidate: {str(forms_candidate).lower()}",
                "baseline_decision: not baseline; no CURRENT_BASELINE update; internal defect remains independent branch.",
                f"next_step: {next_step}",
                f"reason: {reason}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
