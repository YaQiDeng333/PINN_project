#!/usr/bin/env python
"""Audit the 21.1 internal defect pilot pack coverage and split blocker.

This is a planning/audit script for 21.3. It reads the existing registered
pilot pack and metrics only; it does not run COMSOL, train, create data, or
modify NPZ files.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from load_internal_defect_pilot_dataset import DATASET_ID, ROOT, load_dataset, split_indices, write_csv


SUMMARY = ROOT / "results/summaries/internal_defect_pilot_dataset_coverage_audit_summary.txt"
PREFLIGHT = ROOT / "results/summaries/internal_defect_dataset_expansion_preflight_summary.txt"
COVERAGE = ROOT / "results/metrics/internal_defect_pilot_dataset_coverage_audit.csv"
MISSING = ROOT / "results/metrics/internal_defect_pilot_dataset_missing_strata.csv"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v1.manifest.json"
VALIDATION_SUMMARY = ROOT / "results/summaries/internal_defect_pilot_pack_validation_summary.txt"
DECISION_SUMMARY = ROOT / "results/summaries/internal_defect_training_gate_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/internal_defect_training_gate_decision_matrix.csv"
SCHEMA = ROOT / "INTERNAL_DEFECT_SCHEMA.md"
COMSOL_GENERATOR = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\scripts\generate_mfl_internal_defect_pilot_pack.py")

COVERAGE_FIELDS = ["audit", "group_field", "group_value", "split", "count", "expected", "pass", "notes"]
MISSING_FIELDS = ["split", "group_field", "group_value", "missing_reason", "impact"]
SPLITS = ["train", "val", "test"]
SHAPES = ["internal_sphere", "internal_ellipsoid", "internal_cuboid"]
BURIALS = ["shallow", "medium", "deep", "deep_plus"]
SIZES = ["small", "medium", "large"]
ASPECTS = ["compact", "elongated_x", "elongated_y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit internal defect pilot coverage.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--coverage", type=Path, default=COVERAGE)
    parser.add_argument("--missing", type=Path, default=MISSING)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_status(path: Path) -> list[str]:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=str(path), text=True, stderr=subprocess.DEVNULL)
        return [line for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"git_status_error: {exc}"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def count_by(values: list[str]) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(values).items()}


def add_coverage(rows: list[dict[str, Any]], audit: str, field: str, value: str, split: str, count: int, expected: str, passed: bool, notes: str) -> None:
    rows.append(
        {
            "audit": audit,
            "group_field": field,
            "group_value": value,
            "split": split,
            "count": count,
            "expected": expected,
            "pass": bool(passed),
            "notes": notes,
        }
    )


def write_preflight(path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pinn_status = git_status(ROOT)
    comsol_root = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")
    comsol_status = git_status(comsol_root)
    lines = [
        "21.3 internal defect dataset expansion preflight",
        "",
        f"dataset_id: {manifest.get('dataset_id')}",
        f"manifest_exists: {MANIFEST.exists()}",
        f"manifest_status: {manifest.get('status')}",
        f"n_samples: {manifest.get('n_samples')}",
        f"train_ready_candidate: {manifest.get('train_ready_candidate')}",
        f"baseline_ready: {manifest.get('baseline_ready')}",
        f"validation_summary_exists: {VALIDATION_SUMMARY.exists()}",
        f"21_2_decision_summary_exists: {DECISION_SUMMARY.exists()}",
        f"21_2_decision_matrix_exists: {DECISION_MATRIX.exists()}",
        f"internal_schema_exists: {SCHEMA.exists()}",
        f"comsol_pilot_generator_exists: {COMSOL_GENERATOR.exists()}",
        f"pinn_git_status: {pinn_status if pinn_status else 'clean'}",
        f"comsol_git_status: {comsol_status if comsol_status else 'clean'}",
        "comsol_run: false",
        "training_run: false",
        "data_npz_mutation: false",
        "current_baseline_update: false",
        "forbidden_artifacts_created: false",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    write_preflight(args.preflight)
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    coverage_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    fields = {
        "shape_type": (dataset.shape_type, SHAPES, "每个 split 至少覆盖三类 shape"),
        "burial_depth_level": (dataset.burial_depth_level, BURIALS, "每个 split 至少覆盖四档 burial depth"),
        "size_level": (dataset.size_level, SIZES, "每个 split 至少覆盖三档 size"),
        "aspect_bin": (dataset.aspect_bin, ASPECTS, "每个 split 至少覆盖 compact/elongated_x/elongated_y"),
    }
    split_counts: dict[str, dict[str, dict[str, int]]] = {}
    for field, (values, expected_values, note) in fields.items():
        split_counts[field] = {}
        for split_name in SPLITS:
            idx = splits[split_name]
            counts = count_by(values[idx].tolist())
            split_counts[field][split_name] = counts
            for value in expected_values:
                count = counts.get(value, 0)
                passed = count > 0
                add_coverage(coverage_rows, "split_coverage", field, value, split_name, count, ">0", passed, note)
                if not passed:
                    missing_rows.append(
                        {
                            "split": split_name,
                            "group_field": field,
                            "group_value": value,
                            "missing_reason": f"{split_name} split lacks {value}",
                            "impact": "21.2 不能证明该 strata 的泛化能力。",
                        }
                    )

    for split_name in SPLITS:
        idx = splits[split_name]
        pair_counts = Counter(zip(dataset.shape_type[idx].tolist(), dataset.burial_depth_level[idx].tolist()))
        for shape in SHAPES:
            for burial in BURIALS:
                value = f"{shape}|{burial}"
                count = int(pair_counts.get((shape, burial), 0))
                add_coverage(
                    coverage_rows,
                    "shape_by_burial_split",
                    "shape_type_x_burial_depth_level",
                    value,
                    split_name,
                    count,
                    ">0 preferred",
                    count > 0,
                    "shape 和 burial 的交叉覆盖决定 21.4 是否能判断内部缺陷泛化。",
                )
                if count == 0:
                    missing_rows.append(
                        {
                            "split": split_name,
                            "group_field": "shape_type_x_burial_depth_level",
                            "group_value": value,
                            "missing_reason": f"{split_name} split lacks {value}",
                            "impact": "shape/burial 交叉泛化不可判定。",
                        }
                    )

    decision_rows = read_csv_rows(DECISION_MATRIX)
    metric_credibility = [
        ("center_xyz", "可信但需要更大 val/test 复核", "center_xyz MAE 明显优于 mean baseline。"),
        ("cuboid_only_shape_signal", "局部可信", "val/test 只有 internal_cuboid，shape accuracy 不能代表三类 shape 泛化。"),
        ("three_shape_generalization", "不可信", "val/test 缺 internal_sphere 和 internal_ellipsoid。"),
        ("burial_depth_generalization", "不可信", "val/test burial coverage 不完整，test 只含 deep/deep_plus。"),
        ("baseline_or_candidate_upgrade", "不可信", "21.2 是 training gate，不是 baseline gate。"),
    ]
    for metric, verdict, notes in metric_credibility:
        add_coverage(coverage_rows, "metric_credibility", "21_2_metric", metric, "test", 0, verdict, verdict.startswith("可信") or verdict.startswith("局部"), notes)

    write_csv(args.coverage, coverage_rows, COVERAGE_FIELDS)
    write_csv(args.missing, missing_rows, MISSING_FIELDS)
    summary_lines = [
        "21.3 internal defect pilot dataset coverage audit",
        "",
        f"dataset_id: {args.dataset_id}",
        f"N: {dataset.delta_b.shape[0]}",
        f"split_counts: {{'train': {len(splits['train'])}, 'val': {len(splits['val'])}, 'test': {len(splits['test'])}}}",
        f"shape_by_split: {split_counts['shape_type']}",
        f"burial_by_split: {split_counts['burial_depth_level']}",
        f"size_by_split: {split_counts['size_level']}",
        f"aspect_by_split: {split_counts['aspect_bin']}",
        "main_blocker: val/test 都只有 internal_cuboid；test 只覆盖 deep/deep_plus burial depth。",
        "trusted_metrics: center_xyz 可学习信号、cuboid-only shape signal 可参考。",
        "untrusted_metrics: 三类 shape 泛化、完整 burial depth 泛化、baseline 级结论不可信。",
        f"missing_strata_rows: {len(missing_rows)}",
        f"21_2_decision_rows_available: {len(decision_rows)}",
        "recommended_action: 复用 N=96 source rows，但在 v2 assembly 中重做 stratified split，并用 COMSOL top-up 扩展到 N=240。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
