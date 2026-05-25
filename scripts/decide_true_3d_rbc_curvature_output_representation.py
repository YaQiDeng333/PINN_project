#!/usr/bin/env python
"""Decide the next true-3D RBC curvature/output representation direction."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_METRICS = ROOT / "results/metrics/true_3d_rbc_profile_vs_parameter_error_metrics.csv"
REP_MATRIX = ROOT / "results/metrics/true_3d_rbc_alternative_output_representation_matrix.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_curvature_output_representation_decision_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_curvature_output_representation_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return math.nan


def main() -> int:
    profile_rows = read_csv(PROFILE_METRICS)
    rep_rows = read_csv(REP_MATRIX)
    neural_test = next(row for row in profile_rows if row["method"] == "20.77_neural_selected_seed" and row["split"] == "test")
    fusion_test = next(row for row in profile_rows if row["method"] == "20.81_feature_fusion_selected" and row["split"] == "test")
    corr = abs(f(neural_test, "corr_curvature_vs_profile_rmse"))
    contradiction_count = int(float(neural_test.get("high_w_low_profile_count", 0))) + int(float(neural_test.get("high_dice_high_curvature_count", 0)))
    fusion_dice_better = f(fusion_test, "projected_mask_dice_mean") > f(neural_test, "projected_mask_dice_mean")
    fusion_profile_worse = f(fusion_test, "profile_depth_rmse_m_mean") > f(neural_test, "profile_depth_rmse_m_mean")

    profile_primary = True
    continue_raw_w_primary = False
    keep_w_aux = True
    needs_new_data = False
    needs_training_next = True
    baseline_ready = False
    decision = "A_six_params_profile_primary_loss"
    if corr < 0.35 and contradiction_count >= 3:
        secondary = "R3_depth_profile_basis_coefficients"
    else:
        secondary = "R2_template_plus_residual"

    rows = [
        {
            "question": "Continue optimizing isolated wLD/wWD/wLW as primary metric?",
            "answer": continue_raw_w_primary,
            "evidence": f"abs corr curvature-vs-profile RMSE on 20.77 test={corr:.6f}; contradiction_count={contradiction_count}",
            "decision": "No; keep wMAE as diagnostic only.",
        },
        {
            "question": "Promote profile-level metrics to primary?",
            "answer": profile_primary,
            "evidence": "Piao-style target is reconstructed 3-D profile; projected mask and raw wMAE can disagree.",
            "decision": "Yes; use profile_depth_rmse_m / Er-like profile error as main branch metric.",
        },
        {
            "question": "Keep wMAE auxiliary?",
            "answer": keep_w_aux,
            "evidence": "wLD remains a useful failure signal even when profile/Dice improves.",
            "decision": "Yes; report wLD/wWD/wLW in all future profile experiments.",
        },
        {
            "question": "Is projected mask enough?",
            "answer": False,
            "evidence": f"20.81 Dice better than 20.77={fusion_dice_better}; profile RMSE worse than 20.77={fusion_profile_worse}",
            "decision": "No; keep it as 2-D footprint QA.",
        },
        {
            "question": "Recommended output representation?",
            "answer": decision,
            "evidence": "R1 preserves six-param RBC output while making profile reconstruction the primary loss/metric.",
            "decision": "Next training stage should use six params + profile-primary loss.",
        },
        {
            "question": "Need new data?",
            "answer": needs_new_data,
            "evidence": "v3_240 already has rbc_params, profile_depth_grid_m, profile_depth_map_xy_m, projected_mask_2d.",
            "decision": "No new COMSOL/data before the next representation experiment.",
        },
        {
            "question": "Need new training stage?",
            "answer": needs_training_next,
            "evidence": "20.82 is only an audit; changing loss/output requires future training.",
            "decision": "Yes, in a later stage such as 20.83.",
        },
        {
            "question": "Can this be called baseline?",
            "answer": baseline_ready,
            "evidence": "No training/benchmark rerun in 20.82 and baseline_ready remains false.",
            "decision": "No baseline replacement; do not update CURRENT_BASELINE.",
        },
    ]
    write_csv(MATRIX, rows, ["question", "answer", "evidence", "decision"])

    known_reps = ", ".join(row["representation_id"] for row in rep_rows)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.82 curvature output representation decision summary",
                "",
                f"decision: {decision}",
                f"secondary_candidate: {secondary}",
                "next_step: A_six_params_plus_profile_primary_loss",
                "baseline_ready: false",
                "CURRENT_BASELINE_update: false",
                "COMSOL_run: false",
                "training_run: false",
                "data_or_NPZ_modified: false",
                "",
                f"available_representations: {known_reps}",
                f"20.77_test_abs_corr_curvature_vs_profile_rmse: {corr:.6f}",
                f"20.77_test_contradiction_count_high_w_or_high_dice: {contradiction_count}",
                f"20.81_dice_better_than_20.77: {fusion_dice_better}",
                f"20.81_profile_rmse_worse_than_20.77: {fusion_profile_worse}",
                "20.80_artifact_boundary: aggregate_only; it is useful as a method-level reference but is not equivalent to 20.77/20.81 per-sample profile artifacts.",
                "",
                "answers:",
                "1. Continue isolated wLD/wWD/wLW optimization as primary: no.",
                "2. Set profile-level metrics as primary: yes.",
                "3. Keep wMAE as auxiliary diagnostic: yes.",
                "4. Unique next recommendation: A_six_params_plus_profile_primary_loss.",
                "5. Next step needs training: yes, but not in 20.82.",
                "6. Next step needs new data: no.",
                "7. Baseline status: not baseline; CURRENT_BASELINE unchanged.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
