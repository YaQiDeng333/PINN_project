#!/usr/bin/env python
"""Design nominal-preserving liftoff robustness strategies for 20.93.

This is a planning/audit script. It reads the 20.93 trade-off audit outputs and
writes a strategy matrix. It does not train, run COMSOL, or touch data/NPZ.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_tradeoff_audit_summary.txt"
FAILURE_CASES = ROOT / "results/metrics/true_3d_rbc_liftoff_tradeoff_failure_cases.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_nominal_preserving_liftoff_strategy_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_nominal_preserving_liftoff_strategy_matrix.csv"


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
    missing = [str(path) for path in [AUDIT_SUMMARY, FAILURE_CASES] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required trade-off audit artifact(s): " + ", ".join(missing))


def nominal_forgetting_pct(failures: list[dict[str, str]]) -> float:
    for row in failures:
        if row.get("failure_type") == "nominal_forgetting":
            return float(row["relative_change_pct"])
    return float("nan")


def strategy_rows(nominal_regression_pct: float) -> list[dict[str, Any]]:
    return [
        {
            "strategy_id": "S3_baseline_plus_liftoff_adapter",
            "priority": 1,
            "recommended_role": "primary_next_training_design",
            "description": "Freeze or strongly anchor the 20.85 nominal baseline and train a small sensor_z-conditioned correction/adapter for liftoff deviations.",
            "needs_training": True,
            "needs_new_comsol": False,
            "uses_sensor_z_m_input": True,
            "uses_existing_liftoff_pack_v1": True,
            "preserves_nominal_expected": "high",
            "improves_non_nominal_expected": "high",
            "addresses_c1_failure": "Targets nominal forgetting directly by keeping the 0.008m path near the 20.85 baseline while learning non-nominal corrections.",
            "risk": "Adapter may under-correct severe 0.012m liftoff unless paired consistency or non-nominal loss is strong enough.",
            "acceptance_criteria": "Nominal profile RMSE degradation <=10% vs 20.85; non-nominal RMSE improves >=20% vs C0; Dice does not materially regress; no test selection.",
            "why_now": f"C1 nominal profile RMSE regressed by {nominal_regression_pct:.3f}%, so the next model must explicitly protect nominal behavior.",
        },
        {
            "strategy_id": "S2_sensor_z_conditioned_revised_selection",
            "priority": 2,
            "recommended_role": "secondary_ablation",
            "description": "Retrain a sensor_z-conditioned model with a validation score that separately gates nominal and non-nominal liftoff behavior.",
            "needs_training": True,
            "needs_new_comsol": False,
            "uses_sensor_z_m_input": True,
            "uses_existing_liftoff_pack_v1": True,
            "preserves_nominal_expected": "medium_high",
            "improves_non_nominal_expected": "medium_high",
            "addresses_c1_failure": "Removes the unconditioned inverse-map ambiguity and prevents aggregate validation from hiding nominal regression.",
            "risk": "20.92 C2 was not selected by the old validation metric; revised selection must be predeclared and cannot use test results.",
            "acceptance_criteria": "C2-style candidate must beat C1 on validation composite with explicit nominal guard; test final only after selection.",
            "why_now": "C2 had useful post-hoc test signals but failed validation selection, so protocol rather than test re-selection needs redesign.",
        },
        {
            "strategy_id": "S4_paired_consistency_loss",
            "priority": 3,
            "recommended_role": "regularizer_for_S3_or_S2",
            "description": "For each base geometry, require predictions from 0.006/0.008/0.010/0.012m liftoff rows to reconstruct consistent RBC geometry/profile.",
            "needs_training": True,
            "needs_new_comsol": False,
            "uses_sensor_z_m_input": "optional_but_recommended",
            "uses_existing_liftoff_pack_v1": True,
            "preserves_nominal_expected": "medium",
            "improves_non_nominal_expected": "medium",
            "addresses_c1_failure": "Uses the paired pack structure to prevent liftoff-specific geometry drift for the same defect.",
            "risk": "Too much consistency may suppress legitimate liftoff-dependent observability differences or overfit base pairs.",
            "acceptance_criteria": "Pairwise profile/parameter variance across liftoff decreases without nominal RMSE or Dice regression.",
            "why_now": "20.91b generated four complete liftoff rows per base, which makes paired consistency feasible without more COMSOL.",
        },
        {
            "strategy_id": "S1_nominal_weighted_loss",
            "priority": 4,
            "recommended_role": "simple_control",
            "description": "Increase 0.008m nominal loss weight and keep non-nominal losses in the objective.",
            "needs_training": True,
            "needs_new_comsol": False,
            "uses_sensor_z_m_input": False,
            "uses_existing_liftoff_pack_v1": True,
            "preserves_nominal_expected": "medium",
            "improves_non_nominal_expected": "low_medium",
            "addresses_c1_failure": "Counteracts nominal underweighting in aggregate row selection, but does not solve unconditioned liftoff ambiguity.",
            "risk": "May simply trade back toward the nominal baseline and lose the non-nominal gain.",
            "acceptance_criteria": "Nominal RMSE protected and non-nominal RMSE still improves materially; otherwise reject as weight tuning.",
            "why_now": "Useful as a low-complexity diagnostic, not the preferred route.",
        },
        {
            "strategy_id": "S5_two_model_gate",
            "priority": 5,
            "recommended_role": "engineering_comparator_only",
            "description": "Use 20.85 at nominal liftoff and a liftoff-augmented model for non-nominal rows.",
            "needs_training": "possibly_reuses_existing_C1",
            "needs_new_comsol": False,
            "uses_sensor_z_m_input": True,
            "uses_existing_liftoff_pack_v1": True,
            "preserves_nominal_expected": "high",
            "improves_non_nominal_expected": "medium",
            "addresses_c1_failure": "Avoids nominal forgetting by routing nominal rows to the fixed baseline.",
            "risk": "Creates a discontinuous engineering gate and depends on reliable sensor_z_m detection; not a clean modeling solution.",
            "acceptance_criteria": "Only use as comparator; not preferred as the research path unless adapter/conditioned models fail.",
            "why_now": "It clarifies the achievable upper bound for nominal preservation without changing CURRENT_BASELINE.",
        },
    ]


def write_summary(rows: list[dict[str, Any]], nominal_regression_pct: float) -> None:
    primary = rows[0]
    secondary = rows[1]
    lines = [
        "20.93 nominal-preserving liftoff robustness strategy design",
        "",
        "Scope: design only. No COMSOL, no training, no data/NPZ mutation, and no CURRENT_BASELINE update.",
        "",
        "Diagnosis carried forward from trade-off audit:",
        f"- C1_unconditioned_liftoff_aug improved non-nominal rows but regressed nominal 0.008m profile RMSE by {nominal_regression_pct:.3f}%.",
        "- The likely mechanism is liftoff ambiguity plus validation that did not explicitly preserve the fixed nominal baseline.",
        "",
        "Recommended strategy:",
        f"- Primary: {primary['strategy_id']}. {primary['description']}",
        f"- Secondary ablation: {secondary['strategy_id']}. {secondary['description']}",
        "- Add S4 paired consistency as a regularizer only after the primary adapter/conditioned setup is stable.",
        "",
        "Boundary:",
        "- The next stage requires training, but no additional COMSOL data is needed before that training gate.",
        "- CURRENT_BASELINE remains the 20.85 nominal true 3D RBC profile-depth baseline.",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> int:
    _args = parse_args()
    must_exist()
    failures = read_csv(FAILURE_CASES)
    nominal_pct = nominal_forgetting_pct(failures)
    rows = strategy_rows(nominal_pct)
    write_csv(MATRIX, rows)
    write_summary(rows, nominal_pct)
    print(f"wrote {SUMMARY}")
    print(f"wrote {MATRIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
