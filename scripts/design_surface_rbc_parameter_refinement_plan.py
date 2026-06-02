#!/usr/bin/env python
"""Design 25.4 RBC six-parameter refinement strategies.

Plan-only: no training, no COMSOL, no data/NPZ writes, and no baseline update.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_CSV = ROOT / "results/metrics/surface_forward_refinement_target_set.csv"
SUMMARY = ROOT / "results/summaries/surface_rbc_parameter_refinement_plan_summary.txt"
MATRIX = ROOT / "results/metrics/surface_rbc_parameter_refinement_strategy_matrix.csv"

FIELDS = [
    "strategy_id",
    "strategy_name",
    "role",
    "recommended_for_25_5",
    "training_required",
    "model_weight_change",
    "multi_pit_success_credit_allowed",
    "initialization",
    "optimized_variables",
    "loss_terms",
    "parameter_bounds",
    "profile_constraints",
    "stop_criteria",
    "primary_risk",
]

PARAMETER_BOUNDS = {
    "L_m": [0.0015, 0.035],
    "W_m": [0.00075, 0.018],
    "D_m": [0.00005, 0.0045],
    "wLD": [0.03, 10.0],
    "wWD": [0.03, 10.0],
    "wLW": [0.03, 10.0],
}


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
    target_rows = [row for row in read_csv(TARGET_CSV) if row["target_role"] == "refinement_target"]
    bounds = "; ".join(f"{key}=[{lo},{hi}]" for key, (lo, hi) in PARAMETER_BOUNDS.items())
    profile_constraints = "profile_depth_m>=0; max_depth<=D_m; D_m<=0.0045; generated surface profile remains anchored to scan surface"
    stop_criteria = (
        "stop if forward residual decreases but profile RMSE/Dice do not improve; "
        "stop on parameter bound hit with nonphysical profile; "
        "do not trade RBC-like control degradation for non-RBC gains; "
        "do not count multi-pit as RBC refinement success"
    )
    rows = [
        {
            "strategy_id": "R0",
            "strategy_name": "current_20_85",
            "role": "frozen_reference",
            "recommended_for_25_5": False,
            "training_required": False,
            "model_weight_change": False,
            "multi_pit_success_credit_allowed": False,
            "initialization": "existing frozen 20.85 prediction",
            "optimized_variables": "none",
            "loss_terms": "none; reference metrics only",
            "parameter_bounds": "baseline train-bound clipping as in 25.3",
            "profile_constraints": "existing profile generator only",
            "stop_criteria": "not applicable",
            "primary_risk": "already shown to fail broadly on target subset",
        },
        {
            "strategy_id": "R1",
            "strategy_name": "low_dim_param_refinement",
            "role": "first_25_5_strategy",
            "recommended_for_25_5": True,
            "training_required": False,
            "model_weight_change": False,
            "multi_pit_success_credit_allowed": False,
            "initialization": "frozen 20.85 predicted six params",
            "optimized_variables": "L_m,W_m,D_m,wLD,wWD,wLW",
            "loss_terms": "forward_feature_residual + profile_regularity + parameter_bounds",
            "parameter_bounds": bounds,
            "profile_constraints": profile_constraints,
            "stop_criteria": stop_criteria,
            "primary_risk": "feature residual can overfit Bx/By/Bz features without profile improvement",
        },
        {
            "strategy_id": "R2",
            "strategy_name": "residual_correction_head",
            "role": "future_trainable_option",
            "recommended_for_25_5": False,
            "training_required": True,
            "model_weight_change": True,
            "multi_pit_success_credit_allowed": False,
            "initialization": "20.85 predicted params + delta_b features",
            "optimized_variables": "learned six-param residual",
            "loss_terms": "supervised residual loss + optional forward feature consistency",
            "parameter_bounds": bounds,
            "profile_constraints": profile_constraints,
            "stop_criteria": "requires separate training/validation/test protocol",
            "primary_risk": "would become a training stage, not 25.4/25.5 diagnostic-only work",
        },
        {
            "strategy_id": "R3",
            "strategy_name": "unrolled_refinement",
            "role": "future_iterative_option",
            "recommended_for_25_5": False,
            "training_required": True,
            "model_weight_change": True,
            "multi_pit_success_credit_allowed": False,
            "initialization": "20.85 predicted params",
            "optimized_variables": "learned iterative corrections to six params",
            "loss_terms": "unrolled correction loss + forward consistency",
            "parameter_bounds": bounds,
            "profile_constraints": profile_constraints,
            "stop_criteria": "requires separate validation of unrolled stability",
            "primary_risk": "higher complexity before proving R1 diagnostic value",
        },
        {
            "strategy_id": "R4",
            "strategy_name": "component_set_branch",
            "role": "future_multi_pit_branch",
            "recommended_for_25_5": False,
            "training_required": "future",
            "model_weight_change": "future",
            "multi_pit_success_credit_allowed": "only_in_component_set_branch_not_RBC_refinement",
            "initialization": "not derived from six-param RBC refinement",
            "optimized_variables": "component count and component geometry",
            "loss_terms": "future component-set loss",
            "parameter_bounds": "not six-param RBC",
            "profile_constraints": "component-level topology constraints",
            "stop_criteria": "separate from 25.5 RBC refinement gates",
            "primary_risk": "mixing this branch into R1 would hide representation failure",
        },
    ]
    write_csv(MATRIX, rows)
    lines = [
        "25.4 surface RBC parameter refinement plan",
        "",
        f"refinement_target_count: {len(target_rows)}",
        "recommended_first_strategy: R1_low_dim_param_refinement",
        "reference_strategy: R0_current_20_85",
        "future_component_strategy: R4_component_set_branch for multi-pit only",
        "",
        "R1 fixed design:",
        "- Initial value: frozen 20.85 predicted six params.",
        "- Optimized variables: L_m, W_m, D_m, wLD, wWD, wLW.",
        "- Loss: forward-feature residual + profile regularity + parameter bounds.",
        "- Model weights: unchanged.",
        "- Training: none.",
        f"- Parameter bounds: {bounds}.",
        f"- Profile constraints: {profile_constraints}.",
        "",
        "Stop criteria:",
        f"- {stop_criteria}.",
        f"strategy_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
