#!/usr/bin/env python
"""Decide the next step after the 20.93 liftoff trade-off audit.

The decision is based on existing 20.90/20.91/20.92/20.93 outputs only. It does
not train, run COMSOL, or mutate datasets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_tradeoff_audit_summary.txt"
STRATEGY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_nominal_preserving_liftoff_strategy_summary.txt"
STRATEGY_MATRIX = ROOT / "results/metrics/true_3d_rbc_nominal_preserving_liftoff_strategy_matrix.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_tradeoff_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_liftoff_tradeoff_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def must_exist() -> None:
    missing = [str(path) for path in [AUDIT_SUMMARY, STRATEGY_SUMMARY, STRATEGY_MATRIX] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required 20.93 strategy artifact(s): " + ", ".join(missing))


def decision_rows(strategies: list[dict[str, str]]) -> list[dict[str, Any]]:
    strategy_by_id = {row["strategy_id"]: row for row in strategies}
    return [
        {
            "option": "A_train_nominal_preserving_sensor_z_conditioned_model",
            "recommendation": "secondary",
            "mapped_strategy": "S2_sensor_z_conditioned_revised_selection",
            "requires_training": True,
            "requires_new_comsol": False,
            "updates_current_baseline": False,
            "evidence": "20.92 C2 was not validation-selected, but conditioning directly addresses liftoff ambiguity if selection is revised with nominal/non-nominal guards.",
            "decision": "Run as ablation beside the adapter, not as the only next step.",
            "acceptance_gate": strategy_by_id["S2_sensor_z_conditioned_revised_selection"]["acceptance_criteria"],
        },
        {
            "option": "B_train_baseline_plus_adapter_model",
            "recommendation": "primary_next_step",
            "mapped_strategy": "S3_baseline_plus_liftoff_adapter",
            "requires_training": True,
            "requires_new_comsol": False,
            "updates_current_baseline": False,
            "evidence": "C1 improved non-nominal rows but severely regressed nominal; an adapter can preserve 20.85 nominal behavior while learning sensor_z-dependent corrections.",
            "decision": "Recommended unique next step.",
            "acceptance_gate": strategy_by_id["S3_baseline_plus_liftoff_adapter"]["acceptance_criteria"],
        },
        {
            "option": "C_add_paired_liftoff_consistency_loss",
            "recommendation": "supporting_regularizer",
            "mapped_strategy": "S4_paired_consistency_loss",
            "requires_training": True,
            "requires_new_comsol": False,
            "updates_current_baseline": False,
            "evidence": "20.91b pack has complete four-liftoff pairs per base; paired consistency can reduce liftoff-induced geometry drift.",
            "decision": "Use inside B or A if the base objective is stable.",
            "acceptance_gate": strategy_by_id["S4_paired_consistency_loss"]["acceptance_criteria"],
        },
        {
            "option": "D_collect_more_liftoff_COMSOL_data",
            "recommendation": "not_now",
            "mapped_strategy": "none",
            "requires_training": False,
            "requires_new_comsol": True,
            "updates_current_baseline": False,
            "evidence": "The existing full 48-base/192-row liftoff pack is enough for the next training gate; the blocker is protocol/model design, not pack absence.",
            "decision": "Defer top-up until B/A fails on validation with clear undercoverage.",
            "acceptance_gate": "Only collect more after adapter/conditioned gate proves data coverage is the limiting factor.",
        },
        {
            "option": "E_pause_liftoff_robustness_and_proceed_real_data_alignment",
            "recommendation": "reject",
            "mapped_strategy": "none",
            "requires_training": False,
            "requires_new_comsol": False,
            "updates_current_baseline": False,
            "evidence": "20.90 showed liftoff remains a major blocker even after calibration, so real-data alignment would inherit an unresolved acquisition-height risk.",
            "decision": "Do not proceed to real-data claims before nominal-preserving liftoff robustness is tested.",
            "acceptance_gate": "Resume real-data alignment only after liftoff robustness gate passes or is explicitly bounded.",
        },
        {
            "option": "F_keep_current_baseline_only",
            "recommendation": "insufficient",
            "mapped_strategy": "none",
            "requires_training": False,
            "requires_new_comsol": False,
            "updates_current_baseline": False,
            "evidence": "CURRENT_BASELINE should remain unchanged, but keeping it alone does not address the demonstrated liftoff blocker.",
            "decision": "Keep 20.85 as baseline while creating a separate robustness candidate in the next stage.",
            "acceptance_gate": "No baseline replacement unless a later formal gate preserves nominal and improves non-nominal behavior.",
        },
    ]


def write_summary(rows: list[dict[str, Any]]) -> None:
    primary = next(row for row in rows if row["recommendation"] == "primary_next_step")
    lines = [
        "20.93 true 3D RBC liftoff trade-off route decision",
        "",
        "Decision: B. train baseline+liftoff adapter model.",
        "",
        "Reason:",
        "- C1_unconditioned_liftoff_aug is not acceptable as a robustness candidate because it improves non-nominal liftoff but forgets nominal 0.008m behavior.",
        "- C2 sensor_z conditioning was not selected under the predeclared 20.92 validation protocol; test-only signs cannot be used for selection.",
        "- The next training gate should preserve the 20.85 nominal baseline path and learn a sensor_z-conditioned correction for non-nominal liftoff.",
        "",
        "Boundary:",
        "- This route requires a new training stage, but no new COMSOL data is required before that stage.",
        "- CURRENT_BASELINE remains unchanged.",
        "- Internal defect feasibility and real-data alignment remain deferred until liftoff robustness is bounded.",
        "",
        f"Primary acceptance gate: {primary['acceptance_gate']}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> int:
    _args = parse_args()
    must_exist()
    strategies = read_csv(STRATEGY_MATRIX)
    rows = decision_rows(strategies)
    write_csv(MATRIX, rows)
    write_summary(rows)
    print(f"wrote {SUMMARY}")
    print(f"wrote {MATRIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
