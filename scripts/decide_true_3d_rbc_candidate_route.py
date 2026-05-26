#!/usr/bin/env python
"""Route decision for Stage 20.84 true-3D RBC candidate consolidation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONSOLIDATION = ROOT / "results/metrics/true_3d_rbc_candidate_consolidation_matrix.csv"
GALLERY_AUDIT = ROOT / "results/metrics/true_3d_rbc_prediction_gallery_failure_audit.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_candidate_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_candidate_route_decision_matrix.csv"

FIELDS = ["decision_option", "selected", "answer", "evidence", "status"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_by_id(rows: list[dict[str, str]], candidate_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("candidate_id") == candidate_id:
            return row
    raise RuntimeError(f"missing candidate row: {candidate_id}")


def main() -> int:
    candidates = read_csv(CONSOLIDATION)
    gallery = read_csv(GALLERY_AUDIT)
    n77 = row_by_id(candidates, "20.77_neural_reference")
    n81 = row_by_id(candidates, "20.81_feature_fusion")
    n83 = row_by_id(candidates, "20.83_profile_primary_loss")
    high_dice_profile_cases = [row for row in gallery if row.get("audit_bucket") == "high_dice_high_profile_error"]
    curvature_risk_cases = [row for row in gallery if row.get("audit_bucket") == "curvature_risk"]

    selected = "A_keep_20_77_as_profile_depth_benchmark_candidate"
    rows = [
        {
            "decision_option": "A_keep_20_77_as_profile_depth_benchmark_candidate",
            "selected": "true",
            "answer": "recommended",
            "evidence": f"20.77 profile_depth_rmse_m={n77['test_profile_depth_rmse_m']} is lower than 20.81={n81['test_profile_depth_rmse_m']} and 20.83={n83['test_profile_depth_rmse_m']}.",
            "status": "primary_route",
        },
        {
            "decision_option": "B_use_20_81_as_projected_mask_visual_candidate",
            "selected": "false",
            "answer": "keep as reference role",
            "evidence": f"20.81 Dice={n81['test_projected_mask_dice']} is visual/mask strong, but profile RMSE remains worse than 20.77.",
            "status": "secondary_reference",
        },
        {
            "decision_option": "C_rerun_formal_benchmark",
            "selected": "false",
            "answer": "next procedural step after consolidation",
            "evidence": "A formal rerun can use 20.77 as the profile/depth candidate and 20.81 as visual comparator, but 20.84 itself is not a rerun.",
            "status": "future_step",
        },
        {
            "decision_option": "D_redefine_output_as_depth_profile_basis",
            "selected": "false",
            "answer": "research follow-up if improving representation",
            "evidence": "20.83 profile-primary loss is negative; a profile-native basis is more plausible than more small profile-primary loss tweaks.",
            "status": "research_followup",
        },
        {
            "decision_option": "E_collect_targeted_data",
            "selected": "false",
            "answer": "not needed for this consolidation",
            "evidence": "No new data is required to close candidate roles.",
            "status": "not_current",
        },
        {
            "decision_option": "F_pause_true_3d_route",
            "selected": "false",
            "answer": "not recommended",
            "evidence": "20.77 remains a valid profile/depth benchmark candidate and 20.81 remains useful for visual/mask reference.",
            "status": "not_current",
        },
    ]
    write_csv(MATRIX, rows, FIELDS)

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.84 true 3D RBC candidate route decision summary",
                "",
                f"unique_recommendation: {selected}",
                "baseline_ready: false",
                "CURRENT_BASELINE_update: false",
                "COMSOL_run: false",
                "training_run: false",
                "new_data_generated: false",
                "",
                "role_closure:",
                f"- profile/depth main candidate: 20.77 neural reference, profile_depth_rmse_m={n77['test_profile_depth_rmse_m']}.",
                f"- projected mask / visual reference among non-negative candidates: 20.81 feature-fusion, Dice={n81['test_projected_mask_dice']}.",
                f"- negative gate: 20.83 profile-primary loss, Dice={n83['test_projected_mask_dice']} but profile_depth_rmse_m={n83['test_profile_depth_rmse_m']}.",
                "",
                "gallery_decision_support:",
                f"- high-Dice/high-profile-error audited cases: {len(high_dice_profile_cases)}.",
                f"- curvature-risk audited cases: {len(curvature_risk_cases)}.",
                "The gallery supports the conclusion that projected mask quality can improve while 3-D profile quality does not.",
                "",
                "next_step:",
                "Use 20.77 as the profile/depth benchmark candidate for a formal benchmark rerun; keep 20.81 as the non-negative projected-mask visual comparator. Do not continue small tweaks to the current 20.83 profile-primary loss path.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
