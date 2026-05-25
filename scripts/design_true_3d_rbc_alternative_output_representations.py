#!/usr/bin/env python
"""Design alternative true-3D RBC output representations for the next stage."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/true_3d_rbc_alternative_output_representation_design.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_alternative_output_representation_matrix.csv"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = [
        {
            "representation_id": "R0_current_raw_six_params",
            "output": "L_m,W_m,D_m,wLD,wWD,wLW",
            "primary_loss_or_metric": "normalized six-parameter loss",
            "required_labels": "rbc_params",
            "v3_240_supported": True,
            "needs_new_COMSOL": False,
            "needs_new_data": False,
            "benefit": "direct continuity with 20.77/20.81 and Piao-style six-param representation",
            "risk": "keeps unstable wLD/wWD/wLW as headline target",
            "first_minimal_experiment": "none; keep as reference/control",
            "acceptance_criteria": "only used as reference, not the next improvement target",
            "priority": 0,
            "recommendation": "keep reference",
        },
        {
            "representation_id": "R1_six_params_profile_primary_loss",
            "output": "L_m,W_m,D_m,wLD,wWD,wLW",
            "primary_loss_or_metric": "L/W/D param loss plus profile_depth_rmse_m / Er-like profile loss; wMAE auxiliary",
            "required_labels": "rbc_params, profile_pose, profile_depth_grid_m, profile_depth_map_xy_m, projected_mask_2d",
            "v3_240_supported": True,
            "needs_new_COMSOL": False,
            "needs_new_data": False,
            "benefit": "keeps six-param profile generator while aligning optimization with 3-D profile reconstruction",
            "risk": "still constrained by the same RBC parameterization and differentiable profile surrogate quality",
            "first_minimal_experiment": "20.83 train same model family with validation score dominated by profile RMSE and L/W/D, wMAE diagnostic only",
            "acceptance_criteria": "profile_depth_rmse_m and Dice improve or hold while L/W/D does not regress materially",
            "priority": 1,
            "recommendation": "recommended next",
        },
        {
            "representation_id": "R2_template_plus_residual",
            "output": "curvature_template class plus residual curvature/profile parameters",
            "primary_loss_or_metric": "template CE plus residual/profile loss",
            "required_labels": "curvature_template, rbc_params, profile_depth_grid_m",
            "v3_240_supported": True,
            "needs_new_COMSOL": False,
            "needs_new_data": False,
            "benefit": "matches observed template-dependent curvature failures and may stabilize boxy/sharp cases",
            "risk": "template labels are synthetic-generator categories; class error can dominate residual quality",
            "first_minimal_experiment": "train-only template prototypes and residual head; evaluate boxy/sharp profile RMSE",
            "acceptance_criteria": "reduces boxy/sharp profile error without template leakage or test selection",
            "priority": 2,
            "recommendation": "second priority",
        },
        {
            "representation_id": "R3_depth_profile_basis_coefficients",
            "output": "L/W/D plus PCA/DCT coefficients for normalized depth grid",
            "primary_loss_or_metric": "basis coefficient loss plus reconstructed depth-grid RMSE",
            "required_labels": "profile_depth_grid_m or profile_depth_map_xy_m, L_m,W_m,D_m",
            "v3_240_supported": True,
            "needs_new_COMSOL": False,
            "needs_new_data": False,
            "benefit": "profile-native target avoids over-weighting non-identifiable raw curvature scalars",
            "risk": "less directly Piao six-param aligned and may smooth sharp features with N=240",
            "first_minimal_experiment": "train-only PCA K=4/8/12, validation-selected K, test profile RMSE/Dice/volume",
            "acceptance_criteria": "basis reconstruction target is lower-variance and improves profile RMSE over six-param reference",
            "priority": 3,
            "recommendation": "profile-native ablation",
        },
        {
            "representation_id": "R4_surface_descriptors",
            "output": "L/W/D plus volume, mean depth, max depth, edge steepness, center curvature descriptors",
            "primary_loss_or_metric": "descriptor regression and profile QA",
            "required_labels": "profile_depth_grid_m, profile_depth_map_xy_m, projected_mask_2d",
            "v3_240_supported": True,
            "needs_new_COMSOL": False,
            "needs_new_data": False,
            "benefit": "tests whether observable profile descriptors are more stable than wLD/wWD/wLW",
            "risk": "descriptors do not reconstruct a full 3-D profile without another decoder",
            "first_minimal_experiment": "descriptor learnability audit before using as output head",
            "acceptance_criteria": "descriptor errors/correlations are more stable than raw wMAE in boxy/sharp cases",
            "priority": 4,
            "recommendation": "auxiliary diagnostic first",
        },
        {
            "representation_id": "R5_hybrid_multitask",
            "output": "L/W/D, profile/basis, optional template, descriptors, auxiliary wLD/wWD/wLW",
            "primary_loss_or_metric": "profile-depth primary, L/W/D secondary, wMAE auxiliary",
            "required_labels": "all v3_240 profile labels and rbc_params",
            "v3_240_supported": True,
            "needs_new_COMSOL": False,
            "needs_new_data": False,
            "benefit": "captures both Piao-style parameters and profile-native supervision",
            "risk": "loss balancing is complex; 20.81 already shows naive fusion can help total but not curvature",
            "first_minimal_experiment": "only after R1/R2/R3; use validation-only score with profile RMSE primary",
            "acceptance_criteria": "improves profile RMSE and preserves L/W/D/Dice without hidden test selection",
            "priority": 5,
            "recommendation": "later if simpler outputs pass",
        },
    ]
    fields = [
        "representation_id",
        "output",
        "primary_loss_or_metric",
        "required_labels",
        "v3_240_supported",
        "needs_new_COMSOL",
        "needs_new_data",
        "benefit",
        "risk",
        "first_minimal_experiment",
        "acceptance_criteria",
        "priority",
        "recommendation",
    ]
    write_csv(MATRIX, rows, fields)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.82 alternative output representation design",
                "",
                "scope: design only; no training, no COMSOL, no new data, no NPZ modification, no baseline update.",
                "recommended_next_representation: R1_six_params_profile_primary_loss",
                "profile_metric_role: primary true-3D branch metric",
                "wMAE_role: auxiliary diagnostic",
                "projected_mask_role: 2-D footprint QA, not sufficient for 3-D profile quality",
                "",
                "priority_order:",
                "1. R1 six params + profile-primary loss",
                "2. R2 template + residual curvature output",
                "3. R3 depth/profile basis coefficients",
                "4. R4 surface descriptors as diagnostic",
                "5. R5 hybrid multitask after simpler experiments",
                "",
                "current_v3_240_support: all listed representations can be designed from existing labels; none require new COMSOL data for the first experiment.",
                "baseline_boundary: none of these representations is a baseline replacement until a future training/benchmark stage passes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
