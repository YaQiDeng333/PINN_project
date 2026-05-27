#!/usr/bin/env python
"""Route decision for the 20.91b true-3D RBC liftoff augmentation pack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from load_true_3d_rbc_pilot_dataset import ROOT, write_csv


DATASET_ID = "comsol_true_3d_rbc_liftoff_aug_pack_v1"
MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json"
VALIDATION_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_aug_pack_validation_metrics.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_aug_pack_route_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/true_3d_rbc_liftoff_aug_pack_route_decision_matrix.csv"

FIELDS = ["decision_item", "status", "evidence", "decision", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide 20.91b true-3D RBC liftoff augmentation pack route.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--validation-metrics", type=Path, default=VALIDATION_METRICS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--decision-matrix", type=Path, default=DECISION_MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def add(rows: list[dict[str, Any]], item: str, status: str, evidence: str, decision: str, notes: str) -> None:
    rows.append({"decision_item": item, "status": status, "evidence": evidence, "decision": decision, "notes": notes})


def run(args: argparse.Namespace) -> int:
    check_overwrite([args.summary, args.decision_matrix], args.overwrite)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation = read_csv(args.validation_metrics)
    failed = [row for row in validation if str(row.get("pass", "")).lower() != "true"]
    failed_names = {row["check_name"] for row in failed}
    validation_hard_failures = failed_names - {"success_row_count", "paired_liftoff_success_complete"}
    planned_rows = int(manifest.get("planned_rows", 0))
    n_samples = int(manifest.get("n_samples", 0))
    base_count = int(manifest.get("base_count", 0))
    complete_base_count = int(manifest.get("complete_paired_base_count", 0))
    paired_complete = bool(manifest.get("paired_liftoff_complete"))
    status = str(manifest.get("status", ""))
    decisions: list[dict[str, Any]] = []

    full_pack = planned_rows == 192 and n_samples == 192 and base_count == 48 and paired_complete
    add(
        decisions,
        "pack_fullness",
        "pass" if full_pack else "partial",
        f"planned_rows={planned_rows}; n_samples={n_samples}; base_count={base_count}; complete_paired_base_count={complete_base_count}; status={status}",
        "full 192-row pack" if full_pack else "partial pack; top-up before 20.92 full training gate",
        "A full pack requires 48 bases and all four liftoff levels for every base.",
    )
    add(
        decisions,
        "paired_liftoff_completeness",
        "pass" if paired_complete else "fail",
        f"paired_liftoff_complete={paired_complete}; complete_paired_base_count={complete_base_count}",
        "paired comparison is valid" if paired_complete else "paired analysis is incomplete; generate top-up rows",
        "Each base must have sensor_z_m 0.006/0.008/0.010/0.012.",
    )
    add(
        decisions,
        "schema_validation",
        "pass" if not validation_hard_failures else "fail",
        f"failed_hard_checks={sorted(validation_hard_failures)}",
        "schema usable" if not validation_hard_failures else "fix validation blockers before training gate",
        "Success count checks are handled separately so partial packs can still be registered.",
    )
    can_enter_20_92 = full_pack and not validation_hard_failures
    add(
        decisions,
        "enter_20_92_liftoff_training_gate",
        "go" if can_enter_20_92 else "hold",
        f"full_pack={full_pack}; validation_hard_failures={sorted(validation_hard_failures)}",
        "enter 20.92 liftoff-aware training gate" if can_enter_20_92 else "do liftoff pack top-up or validation fix first",
        "20.92 should compare unconditioned vs scalar sensor_z_m conditioned model.",
    )
    add(
        decisions,
        "sensor_z_conditioning",
        "recommended",
        "20.90 showed liftoff is the main blocker; 20.91b pack is paired by geometry and liftoff.",
        "run sensor_z_m-conditioned ablation in 20.92",
        "sensor_z_m is a known acquisition condition, not a defect label leak.",
    )
    add(
        decisions,
        "internal_defect_feasibility",
        "defer",
        "liftoff robustness remains the active surface-defect blocker.",
        "keep internal/buried defect feasibility deferred",
        "Internal defects need a separate label schema and must not mix into the current surface RBC baseline.",
    )
    add(
        decisions,
        "baseline_status",
        "unchanged",
        "CURRENT_BASELINE is not part of this pack generation stage.",
        "do not update baseline",
        "This pack is diagnostic/training-gate input only.",
    )
    write_csv(args.decision_matrix, decisions, FIELDS)
    recommendation = (
        "enter 20.92 liftoff-aware training gate with unconditioned vs sensor_z_m-conditioned ablation"
        if can_enter_20_92
        else "top up or fix the liftoff pack before 20.92"
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "20.91b true 3D RBC liftoff augmentation pack route decision",
                "",
                f"dataset_id: {DATASET_ID}",
                f"pack_status: {status}",
                f"planned_rows: {planned_rows}",
                f"successful_rows: {n_samples}",
                f"base_count: {base_count}",
                f"complete_paired_base_count: {complete_base_count}",
                f"paired_liftoff_complete: {paired_complete}",
                f"validation_hard_failures: {sorted(validation_hard_failures)}",
                f"can_enter_20_92: {can_enter_20_92}",
                f"top_up_needed: {not full_pack}",
                "CURRENT_BASELINE_update: false",
                "internal_defect_feasibility: deferred",
                f"unique_next_step: {recommendation}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
