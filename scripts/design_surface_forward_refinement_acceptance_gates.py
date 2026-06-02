#!/usr/bin/env python
"""Design 25.4 acceptance gates for later forward-refinement execution."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_CSV = ROOT / "results/metrics/surface_forward_refinement_target_set.csv"
SUMMARY = ROOT / "results/summaries/surface_forward_refinement_acceptance_gate_summary.txt"
MATRIX = ROOT / "results/metrics/surface_forward_refinement_acceptance_gate_matrix.csv"

FIELDS = [
    "gate_group",
    "gate_id",
    "subset",
    "metric",
    "pass_condition",
    "fail_condition",
    "baseline_source",
    "multi_pit_success_credit_allowed",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not TARGET_CSV.exists():
        raise FileNotFoundError(TARGET_CSV)
    rows_in = read_csv(TARGET_CSV)
    target_count = sum(row["target_role"] == "refinement_target" for row in rows_in)
    multi_pit_count = sum(row["shape_type"] == "multi_pit_two_component_surface_defect" for row in rows_in)
    rows = [
        {
            "gate_group": "primary",
            "gate_id": "P1_profile_rmse",
            "subset": "refinement_target",
            "metric": "profile_depth_rmse_m_mean",
            "pass_condition": "25.5 <= 0.98 * 25.3 frozen baseline on same target subset",
            "fail_condition": "25.5 >= 25.3 frozen baseline or improvement only appears outside target subset",
            "baseline_source": "surface_forward_refinement_target_set.csv baseline_profile_depth_rmse_m",
            "multi_pit_success_credit_allowed": False,
        },
        {
            "gate_group": "primary",
            "gate_id": "P2_er_like",
            "subset": "refinement_target",
            "metric": "Er_like_error_mean",
            "pass_condition": "25.5 <= 0.98 * 25.3 frozen baseline on same target subset",
            "fail_condition": "Er-like worsens or is not reported",
            "baseline_source": "25.3 current baseline metrics joined by sample_id",
            "multi_pit_success_credit_allowed": False,
        },
        {
            "gate_group": "primary",
            "gate_id": "P3_projected_mask",
            "subset": "refinement_target",
            "metric": "IoU_mean_and_Dice_mean",
            "pass_condition": "IoU_mean >= baseline + 0.01 and Dice_mean >= baseline + 0.01",
            "fail_condition": "either IoU or Dice decreases on target subset",
            "baseline_source": "surface_forward_refinement_target_set.csv baseline_projected_mask_Dice plus 25.3 IoU",
            "multi_pit_success_credit_allowed": False,
        },
        {
            "gate_group": "primary",
            "gate_id": "P4_area_error",
            "subset": "refinement_target",
            "metric": "area_error_mean",
            "pass_condition": "area_error_mean <= max(1.05 * baseline_area_error_mean, baseline_area_error_mean + 0.02)",
            "fail_condition": "area_error improves no metric or worsens beyond tolerance",
            "baseline_source": "25.3 current baseline metrics joined by sample_id",
            "multi_pit_success_credit_allowed": False,
        },
        {
            "gate_group": "secondary",
            "gate_id": "S1_rbc_like_control",
            "subset": "rbc_like_smooth_pit",
            "metric": "profile_rmse_and_Dice",
            "pass_condition": "profile_rmse_mean <= 1.05 * baseline and Dice_mean >= baseline - 0.02",
            "fail_condition": "RBC-like control collapses while non-RBC target improves",
            "baseline_source": "25.3 frozen baseline on RBC-like rows",
            "multi_pit_success_credit_allowed": False,
        },
        {
            "gate_group": "secondary",
            "gate_id": "S2_shape_family_coverage",
            "subset": "flat/sharp/asymmetric/crack/irregular oracle-representable rows",
            "metric": "per_shape_profile_rmse_mean",
            "pass_condition": "at least 4 of 5 non-multi families improve profile_rmse_mean",
            "fail_condition": "aggregate improves but fewer than 3 families improve",
            "baseline_source": "25.3 failure_mode_by_shape and target set",
            "multi_pit_success_credit_allowed": False,
        },
        {
            "gate_group": "secondary",
            "gate_id": "S3_forward_residual_alignment",
            "subset": "refinement_target",
            "metric": "forward_feature_residual_and_profile_rmse",
            "pass_condition": "forward residual decreases and profile_rmse_mean also decreases",
            "fail_condition": "forward residual decreases but profile/mask metrics do not improve",
            "baseline_source": "25.5 F0 diagnostic must report residual before/after",
            "multi_pit_success_credit_allowed": False,
        },
        {
            "gate_group": "failure",
            "gate_id": "F1_nonphysical_params",
            "subset": "all refined six-param rows",
            "metric": "parameter_bounds_and_profile_physics",
            "pass_condition": "all refined params remain in declared R1 bounds and generated profiles are nonnegative",
            "fail_condition": "L/W/D/w out of bounds, negative profile, or D_m violates surface profile cap",
            "baseline_source": "surface_rbc_parameter_refinement_strategy_matrix.csv",
            "multi_pit_success_credit_allowed": False,
        },
        {
            "gate_group": "failure",
            "gate_id": "F2_multi_pit_credit",
            "subset": "multi_pit_two_component_surface_defect",
            "metric": "success_accounting",
            "pass_condition": "multi-pit reported only as excluded_negative_control or future component-set branch",
            "fail_condition": "multi-pit counted as RBC-refinement success",
            "baseline_source": "surface_forward_refinement_target_set.csv",
            "multi_pit_success_credit_allowed": False,
        },
        {
            "gate_group": "failure",
            "gate_id": "F3_baseline_transition",
            "subset": "repository_docs",
            "metric": "CURRENT_BASELINE.md",
            "pass_condition": "CURRENT_BASELINE.md unchanged",
            "fail_condition": "CURRENT_BASELINE.md modified or 25.5 described as baseline transition",
            "baseline_source": "git diff",
            "multi_pit_success_credit_allowed": False,
        },
    ]
    write_csv(MATRIX, rows)
    lines = [
        "25.4 surface forward-refinement acceptance gates",
        "",
        f"refinement_target_count: {target_count}",
        f"multi_pit_negative_control_count: {multi_pit_count}",
        "primary_gate: target subset profile RMSE, Er-like error, and projected mask metrics must improve versus 25.3 frozen baseline.",
        "secondary_gate: RBC-like control must not collapse; non-multi shape families must improve broadly; forward residual and profile metrics must move together.",
        "failure_gate: nonphysical params, multi-pit success credit, or CURRENT_BASELINE.md transition fails the route.",
        "",
        "25.5 reporting requirement:",
        "- Report metrics by target_role, shape_type, split, and representation_target.",
        "- Report multi-pit separately as excluded_negative_control.",
        "- Report before/after forward residual and before/after profile/mask metrics on the same sample set.",
        f"acceptance_gate_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
