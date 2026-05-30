#!/usr/bin/env python
"""判定 22.2b hard-case top-up pack 是否可进入 22.3。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_pilot_pack_v3_hardcase"
TOPUP_DATASET_ID = "comsol_internal_defect_hard_case_topup_pack_v1"
TOPUP_VALIDATION = ROOT / "results/metrics/internal_defect_hard_case_topup_pack_validation_metrics.csv"
ASSEMBLY_METRICS = ROOT / "results/metrics/internal_defect_hard_case_augmented_pack_assembly_metrics.csv"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json"
SUMMARY = ROOT / "results/summaries/internal_defect_hard_case_topup_pack_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/internal_defect_hard_case_topup_pack_decision_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="判定 internal hard-case top-up route。")
    parser.add_argument("--topup-validation", type=Path, default=TOPUP_VALIDATION)
    parser.add_argument("--assembly-metrics", type=Path, default=ASSEMBLY_METRICS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_value(rows: list[dict[str, str]], name: str) -> str:
    for row in rows:
        if row.get("metric") == name or row.get("check_name") == name:
            return str(row.get("value") or row.get("observed") or "")
    return ""


def metric_pass(rows: list[dict[str, str]], name: str) -> bool:
    for row in rows:
        if row.get("metric") == name or row.get("check_name") == name:
            return str(row.get("pass", "")).lower() == "true"
    return False


def run(args: argparse.Namespace) -> int:
    validation = read_csv(args.topup_validation)
    assembly = read_csv(args.assembly_metrics)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {}

    topup_success = int(float(metric_value(validation, "inventory_success_minimum") or 0))
    validation_passed = bool(validation) and all(str(row.get("pass", "")).lower() == "true" for row in validation)
    assembly_train_ready = metric_pass(assembly, "train_ready_candidate")
    train_ready = bool(manifest.get("train_ready_candidate")) and assembly_train_ready
    status = str(manifest.get("status", "missing"))

    rows = [
        {
            "decision_item": "topup_success_target",
            "observed": topup_success,
            "decision": "pass" if topup_success >= 72 else "fail",
            "rationale": "22.2b minimum usable hard-case top-up is 72 rows; target is 120.",
        },
        {
            "decision_item": "target_strata_coverage",
            "observed": "validation metrics",
            "decision": "pass" if validation_passed else "fail",
            "rationale": "Validation checks cuboid/ellipsoid confusion, compact, medium/large, shallow/deep_plus, and center neighbor targets.",
        },
        {
            "decision_item": "assembled_dataset",
            "observed": manifest.get("n_samples", ""),
            "decision": "pass" if manifest.get("dataset_id") == DATASET_ID else "fail",
            "rationale": "v3_hardcase must be registered through explicit manifest, not latest/newest discovery.",
        },
        {
            "decision_item": "train_ready_candidate",
            "observed": str(train_ready).lower(),
            "decision": "pass" if train_ready else "fail",
            "rationale": "Train-ready means schema validation and assembly pass; it still is not a baseline.",
        },
        {
            "decision_item": "next_step",
            "observed": "22.3 hard-case augmented internal training gate" if train_ready else "second top-up or generator fix",
            "decision": "enter_22_3" if train_ready else "blocked",
            "rationale": "Do not continue model refinement or real internal smoke until hard-case data is available.",
        },
    ]
    write_csv(args.matrix, rows, ["decision_item", "observed", "decision", "rationale"])

    recommended = "进入 22.3 hard-case augmented internal training gate" if train_ready else "先修复或补跑 hard-case top-up"
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "22.2b internal defect hard-case top-up route decision summary",
                "",
                f"topup_dataset_id: {TOPUP_DATASET_ID}",
                f"assembled_dataset_id: {DATASET_ID}",
                f"manifest_status: {status}",
                f"topup_success_rows_observed: {topup_success}",
                f"validation_passed: {str(validation_passed).lower()}",
                f"train_ready_candidate: {str(train_ready).lower()}",
                f"next_step: {recommended}",
                "",
                "结论：internal branch 仍不是 baseline；CURRENT_BASELINE 保持 surface/near-surface true 3D RBC baseline。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if train_ready else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
