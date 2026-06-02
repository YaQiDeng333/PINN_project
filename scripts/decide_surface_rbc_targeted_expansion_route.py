#!/usr/bin/env python
"""Route decision for surface RBC targeted +120 top-up."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_SUMMARY = ROOT / "results/summaries/surface_rbc_targeted_expansion_validation_summary.txt"
DEFAULT_CALIBRATION_SUMMARY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\surface_rbc_targeted_expansion_calibration_summary.txt")
DEFAULT_TOPUP_SUMMARY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\surface_rbc_targeted_expansion_topup_summary.txt")
DEFAULT_SUMMARY = ROOT / "results/summaries/surface_rbc_targeted_expansion_route_decision_summary.txt"
DEFAULT_MATRIX = ROOT / "results/metrics/surface_rbc_targeted_expansion_decision_matrix.csv"
FIELDS = ["decision_option", "selected", "condition", "observed", "next_step"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide surface RBC targeted expansion route.")
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--calibration-summary", type=Path, default=DEFAULT_CALIBRATION_SUMMARY)
    parser.add_argument("--topup-summary", type=Path, default=DEFAULT_TOPUP_SUMMARY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def text_has_true(text: str, key: str) -> bool:
    lowered = text.lower()
    return f"{key.lower()}: true" in lowered or f"{key.lower()}=true" in lowered


def text_value_contains(text: str, key: str, value: str) -> bool:
    return any(line.lower().startswith(key.lower() + ":") and value.lower() in line.lower() for line in text.splitlines())


def decide_route(
    *,
    calibration_pass: bool,
    full_success: bool,
    validation_pass: bool,
    n_success: int,
    systemic_blocker: bool,
) -> dict[str, Any]:
    can_enter = bool(calibration_pass and full_success and validation_pass and n_success == 120 and not systemic_blocker)
    if can_enter:
        next_step = "enter separate +120 training gate; assemble v3_240 + topup_v1_120 there"
    elif systemic_blocker:
        next_step = "stop; fix COMSOL/license/memory/systemic geometry blocker before any full generation"
    elif not calibration_pass:
        next_step = "stop; write blocker and replacement proposal before any full generation"
    elif not full_success:
        next_step = "retry or replace only rows with the same role/depth/aspect/curvature/edge signature"
    else:
        next_step = "fix generator/labels/validation; do not train"
    return {
        "can_enter_training_gate": can_enter,
        "recommend_continue_240_480": can_enter,
        "current_baseline_unchanged": True,
        "creates_assembled_dataset": False,
        "next_step": next_step,
    }


def run(args: argparse.Namespace) -> int:
    if args.summary.exists() and not args.overwrite:
        raise FileExistsError(args.summary)
    if args.matrix.exists() and not args.overwrite:
        raise FileExistsError(args.matrix)
    cal_text = read_text(args.calibration_summary)
    topup_text = read_text(args.topup_summary)
    val_text = read_text(args.validation_summary)
    systemic = "systemic_blocker: true" in cal_text.lower() or "license_conflict" in cal_text.lower() or "comsol_crash" in cal_text.lower()
    calibration_pass = text_has_true(cal_text, "calibration_pass") and not systemic
    full_success = "successful_rows: 120" in topup_text or "success_count: 120" in topup_text
    validation_pass = text_has_true(val_text, "validation_pass")
    n_success = 120 if "n_success: 120" in val_text else 0
    decision = decide_route(
        calibration_pass=calibration_pass,
        full_success=full_success,
        validation_pass=validation_pass,
        n_success=n_success,
        systemic_blocker=systemic,
    )
    rows = [
        {
            "decision_option": "A_enter_separate_plus120_training_gate",
            "selected": decision["can_enter_training_gate"],
            "condition": "calibration pass, full 120 success, validation pass, no systemic blocker",
            "observed": f"calibration={calibration_pass}; full={full_success}; validation={validation_pass}; n={n_success}; systemic={systemic}",
            "next_step": decision["next_step"],
        },
        {
            "decision_option": "B_retry_or_replace_missing_topup_rows",
            "selected": calibration_pass and not full_success,
            "condition": "calibration pass but full top-up incomplete",
            "observed": f"full_success={full_success}",
            "next_step": "same-signature replacement only",
        },
        {
            "decision_option": "C_stop_for_blocker",
            "selected": (not calibration_pass) or systemic,
            "condition": "calibration failure or systemic blocker",
            "observed": f"calibration={calibration_pass}; systemic={systemic}",
            "next_step": "do not run full generation",
        },
    ]
    write_csv(args.matrix, rows, FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface RBC targeted expansion route decision summary",
                "",
                f"calibration_pass: {calibration_pass}",
                f"full_120_success: {full_success}",
                f"validation_pass: {validation_pass}",
                f"can_enter_plus120_training_gate: {decision['can_enter_training_gate']}",
                f"recommend_continue_plus240_plus480: {decision['recommend_continue_240_480']}",
                "creates_assembled_dataset: false",
                "CURRENT_BASELINE_update: false",
                f"next_step: {decision['next_step']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"can_enter_training_gate={decision['can_enter_training_gate']}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
