#!/usr/bin/env python
"""Decision summary for 20.97 real-data intake route."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_MATRIX = ROOT / "results/metrics/true_3d_rbc_real_data_schema_validation_matrix.csv"
PREPROCESSING_PLAN = ROOT / "results/summaries/true_3d_rbc_real_data_preprocessing_plan.md"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_real_data_intake_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_real_data_intake_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not VALIDATION_MATRIX.exists():
        raise FileNotFoundError(VALIDATION_MATRIX)
    if not PREPROCESSING_PLAN.exists():
        raise FileNotFoundError(PREPROCESSING_PLAN)
    rows = read_csv(VALIDATION_MATRIX)
    blockers = [row for row in rows if row["severity"] == "blocker" and row["pass"] == "False"]
    placeholder_warnings = [row for row in rows if row["severity"] == "warning" and row["pass"] == "False" and "placeholder" in row["message"].lower()]
    decisions = [
        {
            "question": "what_is_missing_before_real_data",
            "answer": "actual real-data manifest and sample table with measured sensor_z_m, tri-axis Bx/By/Bz, matched no-defect references, units, coordinate system, alignment status, and gain status",
            "decision": "prepare a manifest dry run before any inference claim",
        },
        {
            "question": "can_enter_real_data_manifest_dry_run",
            "answer": True,
            "decision": "yes, validator and templates are ready; dry run can be manifest-only first",
        },
        {
            "question": "need_no_defect_reference",
            "answer": True,
            "decision": "yes, either prepared delta_b with reference provenance or raw b_defect plus b_no_defect is required",
        },
        {
            "question": "if_only_bz_available",
            "answer": "block_current_route",
            "decision": "stop this true 3D RBC route or open a separate Bz-only branch; do not feed Bz-only data into the tri-axis baseline",
        },
        {
            "question": "internal_defect_schema",
            "answer": "separate_branch",
            "decision": "internal/buried defects need a separate label schema and must not be mixed into the surface RBC baseline intake",
        },
        {
            "question": "current_template_ready",
            "answer": len(blockers) == 0 and len(placeholder_warnings) == 0,
            "decision": "template is runnable but contains placeholders; replace placeholders for a real manifest dry run",
        },
    ]
    write_csv(MATRIX, decisions)
    lines = [
        "20.97 true 3D RBC real-data intake route decision",
        "",
        f"template_blocker_count: {len(blockers)}",
        f"template_placeholder_warning_count: {len(placeholder_warnings)}",
        "real_data_ready_for_inference: false until an actual manifest and sample table pass validation",
        "manifest_dry_run_ready: true",
        "",
        "next unique step: ask for or prepare a real-data manifest dry run with no data file required at first. It must include sensor_z_m, no-defect reference provenance, tri-axis Bx/By/Bz availability, units, coordinate system, sensor alignment, and gain status.",
        "",
        "Bz-only condition: blocker for this route; create a separate Bz-only branch only if tri-axis data cannot be provided.",
        "Internal defect condition: separate schema; do not mix with current surface-breaking RBC baseline.",
        "CURRENT_BASELINE_update: false",
        "COMSOL_run: false",
        "training_run: false",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
