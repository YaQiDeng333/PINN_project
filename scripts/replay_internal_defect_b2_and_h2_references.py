#!/usr/bin/env python
"""22.5 preflight and fixed B2/H2 reference replay for freeze-shape training."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from internal_defect_hardcase_utils import (
    B2_MANIFEST,
    DATASET_ID,
    METRIC_FIELDS,
    TAIL_FIELDS,
    load_old_b2_on_dataset,
    metric_rows_for_model,
    prepare_dataset,
    read_csv,
)
from load_internal_defect_pilot_dataset import ROOT, write_csv


PREFLIGHT = ROOT / "results/summaries/internal_defect_freeze_shape_tail_regression_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/internal_defect_freeze_shape_reference_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_freeze_shape_reference_metrics.csv"
TAIL = ROOT / "results/metrics/internal_defect_freeze_shape_reference_tail_metrics.csv"

V3_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
CURRENT_BASELINE = ROOT / "CURRENT_BASELINE.md"
H2_METRICS = ROOT / "results/metrics/internal_defect_hardcase_metrics.csv"
H2_TAIL = ROOT / "results/metrics/internal_defect_hardcase_tail_metrics.csv"
H2_SEEDS = ROOT / "results/metrics/internal_defect_hardcase_seed_summary.csv"
H2_FAILURES = ROOT / "results/metrics/internal_defect_hardcase_failure_cases.csv"
SHAPE_STRATEGY = ROOT / "results/metrics/internal_defect_shape_preserving_tail_decision_matrix.csv"
B2_21_7 = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_metrics.csv"
B2_22_0 = ROOT / "results/metrics/internal_defect_b2_inference_replay_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay B2/H2 references for 22.5.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--tail", type=Path, default=TAIL)
    return parser.parse_args()


def git_status(paths: list[str]) -> str:
    completed = subprocess.run(["git", "status", "--short", "--", *paths], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip()


def selected_h2_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("selected_model") == "True":
            new = dict(row)
            new["model"] = "H2_B2_hardcase_tail_weighted_reference"
            new["selected_model"] = False
            out.append(new)
    return out


def selected_h2_seed(path: Path) -> dict[str, str]:
    for row in read_csv(path):
        if row.get("selected_model") == "True":
            return row
    return {}


def preflight(args: argparse.Namespace) -> None:
    required = {
        "registry": REGISTRY,
        "v3_hardcase_manifest": V3_MANIFEST,
        "B2_artifact_manifest": B2_MANIFEST,
        "B2_checkpoint": Path(json.loads(B2_MANIFEST.read_text(encoding="utf-8"))["checkpoint_path"]) if B2_MANIFEST.exists() else Path("__missing__"),
        "22.3_H2_metrics": H2_METRICS,
        "22.3_H2_tail": H2_TAIL,
        "22.3_H2_seed_summary": H2_SEEDS,
        "22.3_H2_failure_cases": H2_FAILURES,
        "22.4_route_decision": SHAPE_STRATEGY,
    }
    optional = {"21.7_B2_metrics": B2_21_7, "22.0_B2_replay": B2_22_0}
    missing = [name for name, path in required.items() if not path.exists()]
    missing_optional = [name for name, path in optional.items() if not path.exists()]
    forbidden = git_status(["data", "checkpoints", "results\\previews", "notes", "CURRENT_BASELINE.md", "scripts\\visualize_current_baseline.py"])
    lines = [
        "22.5 freeze-shape tail-regression preflight",
        "scope: no COMSOL, no data/NPZ generation or mutation, no CURRENT_BASELINE.md update, no checkpoint/preview/notes commit.",
        f"dataset_id: {args.dataset_id}",
        f"critical_files_present: {not missing}",
        f"missing_critical: {missing}",
        f"missing_optional: {missing_optional}",
        f"forbidden_artifact_status_empty: {forbidden == ''}",
        f"forbidden_artifact_status: {forbidden if forbidden else 'clean'}",
        "input_policy: formal models may use only delta_b/BxByBz, delta_b-derived features, and frozen B2 predictions/logits/latent.",
        "label_policy: labels are supervision/metrics only; true shape is oracle diagnostic only.",
    ]
    args.preflight.parent.mkdir(parents=True, exist_ok=True)
    args.preflight.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if missing:
        raise RuntimeError(f"preflight blocker: {missing}")


def main() -> int:
    args = parse_args()
    preflight(args)
    prepared = prepare_dataset(args.dataset_id)
    dataset = prepared["dataset"]
    y_std = prepared["y_std"].reshape(-1)
    b2_pred, b2_shape, manifest = load_old_b2_on_dataset(prepared)
    b2_metrics, b2_tail = metric_rows_for_model("B2_feature_fusion_burial_head_reference", False, manifest.get("seed", 2026), dataset, prepared["splits"], b2_pred, b2_shape, y_std)
    h2_metrics = selected_h2_rows(read_csv(H2_METRICS))
    h2_tail = selected_h2_rows(read_csv(H2_TAIL))
    write_csv(args.metrics, b2_metrics + h2_metrics, METRIC_FIELDS)
    write_csv(args.tail, b2_tail + h2_tail, TAIL_FIELDS)
    b2_test = next(row for row in b2_metrics if row["split"] == "test" and row["subset"] == "all")
    b2_test_tail = next(row for row in b2_tail if row["split"] == "test" and row["subset"] == "all")
    h2_seed = selected_h2_seed(H2_SEEDS)
    lines = [
        "22.5 B2/H2 fixed reference summary",
        f"dataset_id: {args.dataset_id}",
        "B2_reference: 21.9 B2 artifact replayed on v3_hardcase; no retraining.",
        "H2_reference: 22.3 selected H2 metrics reused; no checkpoint required.",
        f"B2_test_total_mae: {float(b2_test['total_normalized_mae']):.6f}",
        f"B2_test_shape_acc_f1: {float(b2_test['shape_accuracy']):.6f} / {float(b2_test['shape_macro_f1']):.6f}",
        f"B2_test_catastrophic_geometry: {b2_test_tail['catastrophic_failure_count']} / {b2_test_tail['geometry_branch_failure_count']}",
        f"B2_test_center_p95_max_mm: {float(b2_test_tail['center_xyz_error_p95_mm']):.3f} / {float(b2_test_tail['center_xyz_error_max_mm']):.3f}",
        f"B2_test_burial_p95_max_mm: {float(b2_test_tail['burial_depth_error_p95_mm']):.3f} / {float(b2_test_tail['burial_depth_error_max_mm']):.3f}",
        f"H2_test_total_mae: {float(h2_seed['test_total_normalized_mae']):.6f}",
        f"H2_test_shape_acc_f1: {float(h2_seed['test_shape_accuracy']):.6f} / {float(h2_seed['test_shape_macro_f1']):.6f}",
        f"H2_test_catastrophic_geometry: {h2_seed['test_catastrophic_failure_count']} / {h2_seed['test_geometry_branch_failure_count']}",
        f"H2_test_center_p95_max_mm: {float(h2_seed['test_center_p95_mm']):.3f} / {float(h2_seed['test_center_max_mm']):.3f}",
        f"H2_test_burial_p95_max_mm: {float(h2_seed['test_burial_p95_mm']):.3f} / {float(h2_seed['test_burial_max_mm']):.3f}",
        "current_baseline_update: false",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
