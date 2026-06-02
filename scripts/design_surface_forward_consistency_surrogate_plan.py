#!/usr/bin/env python
"""Design 25.4 forward-consistency surrogate options.

Plan-only: writes summary/CSV artifacts and does not train, run COMSOL, or
touch data/NPZ/CURRENT_BASELINE.md.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_CSV = ROOT / "results/metrics/surface_forward_refinement_target_set.csv"
SUMMARY = ROOT / "results/summaries/surface_forward_consistency_surrogate_plan_summary.txt"
MATRIX = ROOT / "results/metrics/surface_forward_consistency_surrogate_matrix.csv"

FIELDS = [
    "route_id",
    "route_name",
    "recommendation",
    "implementation_difficulty",
    "new_data_required",
    "training_required_for_25_5",
    "comsol_required_for_25_5",
    "suitable_for_test_time_optimization",
    "gradient_based_refinement_possible",
    "expected_profile_rmse_help",
    "expected_iou_dice_help",
    "primary_risk",
    "selected_for_25_5",
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
    targets = [row for row in read_csv(TARGET_CSV) if row["target_role"] == "refinement_target"]
    rows = [
        {
            "route_id": "F0",
            "route_name": "feature_space_consistency",
            "recommendation": "first_25_5_route",
            "implementation_difficulty": "low",
            "new_data_required": "no",
            "training_required_for_25_5": "no",
            "comsol_required_for_25_5": "no",
            "suitable_for_test_time_optimization": "yes",
            "gradient_based_refinement_possible": "yes_with_differentiable_feature_proxy_or_finite_difference",
            "expected_profile_rmse_help": "medium; constrains six params toward observed delta_b-derived features",
            "expected_iou_dice_help": "medium; footprint features can regularize area/width/center errors",
            "primary_risk": "feature residual can improve while profile metrics do not; acceptance gate must catch this",
            "selected_for_25_5": True,
        },
        {
            "route_id": "F1",
            "route_name": "neural_forward_surrogate",
            "recommendation": "future_after_F0",
            "implementation_difficulty": "medium",
            "new_data_required": "no_new_comsol_but_requires_training_on_existing_pilot",
            "training_required_for_25_5": "yes_if_used",
            "comsol_required_for_25_5": "no",
            "suitable_for_test_time_optimization": "yes",
            "gradient_based_refinement_possible": "yes",
            "expected_profile_rmse_help": "potentially high after surrogate validation",
            "expected_iou_dice_help": "potentially high if trained on compact MFL feature vector or delta_b",
            "primary_risk": "surrogate mismatch can create false forward-consistency gains",
            "selected_for_25_5": False,
        },
        {
            "route_id": "F2",
            "route_name": "cached_COMSOL_local_refinement",
            "recommendation": "future_lookup_option",
            "implementation_difficulty": "medium",
            "new_data_required": "no_for_lookup; limited by existing sampled coverage",
            "training_required_for_25_5": "no",
            "comsol_required_for_25_5": "no",
            "suitable_for_test_time_optimization": "limited",
            "gradient_based_refinement_possible": "no_or_weak_interpolated_gradient",
            "expected_profile_rmse_help": "low_to_medium inside covered parameter regions",
            "expected_iou_dice_help": "low_to_medium",
            "primary_risk": "coverage gaps and interpolation artifacts",
            "selected_for_25_5": False,
        },
        {
            "route_id": "F3",
            "route_name": "direct_COMSOL_refinement",
            "recommendation": "not_recommended_for_25_5",
            "implementation_difficulty": "high",
            "new_data_required": "yes_runtime_COMSOL",
            "training_required_for_25_5": "no",
            "comsol_required_for_25_5": "yes",
            "suitable_for_test_time_optimization": "theoretical_only",
            "gradient_based_refinement_possible": "no_practical_gradient",
            "expected_profile_rmse_help": "unknown",
            "expected_iou_dice_help": "unknown",
            "primary_risk": "too slow and violates the no-COMSOL boundary for 25.4/25.5 diagnostic planning",
            "selected_for_25_5": False,
        },
    ]
    write_csv(MATRIX, rows)
    lines = [
        "25.4 surface forward-consistency surrogate plan",
        "",
        f"refinement_target_count: {len(targets)}",
        "recommended_first_route: F0_feature_space_consistency",
        "",
        "F0 definition:",
        "- Predict compact delta_b-derived features from RBC six params/profile.",
        "- Compare predicted features with observed features from delta_b/BxByBz.",
        "- Use residual as a test-time refinement diagnostic; no training, no COMSOL, no data generation.",
        "",
        "why F0 first:",
        "- It can be implemented from existing 25.3 artifacts and existing feature-extraction conventions.",
        "- It is low cost and compatible with finite-difference or differentiable proxy refinement.",
        "- Acceptance gates can detect the main risk: lower feature residual without better profile/mask metrics.",
        "",
        "future options:",
        "- F1 neural forward surrogate requires training and surrogate validation, so it is not first.",
        "- F2 cached COMSOL lookup is limited by sampled coverage.",
        "- F3 direct COMSOL refinement is future-only and not recommended.",
        f"surrogate_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
