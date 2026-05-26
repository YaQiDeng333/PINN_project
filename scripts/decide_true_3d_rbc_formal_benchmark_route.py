#!/usr/bin/env python
"""Route decision for the Stage 20.85 formal benchmark rerun."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/true_3d_rbc_formal_benchmark_decision_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_decision_matrix.csv"
COMPARISON = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_comparison_matrix.csv"

FIELDS = ["question", "answer", "evidence", "decision"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def by_id(rows: list[dict[str, str]], candidate_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("candidate_id") == candidate_id:
            return row
    raise RuntimeError(f"candidate_id missing: {candidate_id}")


def run(_: argparse.Namespace) -> int:
    rows = read_csv(COMPARISON)
    formal = by_id(rows, "20.85_formal_rerun_20.77_protocol")
    original = by_id(rows, "20.77_original_candidate")
    fusion = by_id(rows, "20.81_feature_fusion")
    profile_primary = by_id(rows, "20.83_profile_primary_negative_gate")

    profile_stable = f(formal["test_profile_depth_rmse_m"]) <= f(original["test_profile_depth_rmse_m"]) * 1.20
    beats_visual_profile = f(formal["test_profile_depth_rmse_m"]) < f(fusion["test_profile_depth_rmse_m"])
    beats_negative_profile = f(formal["test_profile_depth_rmse_m"]) < f(profile_primary["test_profile_depth_rmse_m"])
    dice_acceptable = f(formal["test_projected_mask_dice"]) >= 0.84
    dimensions_reasonable = max(f(formal["test_L_mae_mm"]), f(formal["test_W_mae_mm"]), f(formal["test_D_mae_mm"])) < 3.0
    benchmark_candidate = profile_stable and beats_visual_profile and beats_negative_profile and dice_acceptable and dimensions_reasonable
    decision = "A_prepare_report_or_formal_presentation" if benchmark_candidate else "B_revert_to_20_77_original_candidate_and_recheck"
    next_step = "prepare paper/report display around the formal 20.77-profile candidate" if benchmark_candidate else "model refinement or depth/profile basis output before presentation"

    decision_rows = [
        {
            "question": "Did formal rerun stably reproduce the 20.77 profile/depth advantage?",
            "answer": str(profile_stable),
            "evidence": f"formal_profile_rmse={formal['test_profile_depth_rmse_m']}; original_20.77={original['test_profile_depth_rmse_m']}; tolerance=20%",
            "decision": decision,
        },
        {
            "question": "Does formal rerun beat the 20.81 visual comparator on profile RMSE?",
            "answer": str(beats_visual_profile),
            "evidence": f"formal={formal['test_profile_depth_rmse_m']}; 20.81={fusion['test_profile_depth_rmse_m']}",
            "decision": decision,
        },
        {
            "question": "Does formal rerun beat the 20.83 negative gate on profile RMSE?",
            "answer": str(beats_negative_profile),
            "evidence": f"formal={formal['test_profile_depth_rmse_m']}; 20.83={profile_primary['test_profile_depth_rmse_m']}",
            "decision": decision,
        },
        {
            "question": "Is projected mask quality acceptable?",
            "answer": str(dice_acceptable),
            "evidence": f"formal_dice={formal['test_projected_mask_dice']}; threshold=0.84",
            "decision": decision,
        },
        {
            "question": "Are L/W/D dimensions still usable?",
            "answer": str(dimensions_reasonable),
            "evidence": f"L/W/D={formal['test_L_mae_mm']}/{formal['test_W_mae_mm']}/{formal['test_D_mae_mm']} mm",
            "decision": decision,
        },
        {
            "question": "Can this be called a benchmark candidate?",
            "answer": str(benchmark_candidate),
            "evidence": "This is still a fixed-dataset candidate, not a production baseline.",
            "decision": decision,
        },
        {
            "question": "Can this be called a baseline?",
            "answer": "False",
            "evidence": "baseline_ready=false; CURRENT_BASELINE.md remains the v3_complex mask baseline.",
            "decision": decision,
        },
        {
            "question": "Next step",
            "answer": next_step,
            "evidence": "No data generation, no COMSOL, no NPZ modification, no baseline update occurred in 20.85.",
            "decision": decision,
        },
    ]
    write_csv(MATRIX, decision_rows, FIELDS)

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.85 true 3D RBC formal benchmark route decision",
                "",
                f"decision: {decision}",
                f"benchmark_candidate: {benchmark_candidate}",
                "baseline_ready: false",
                f"profile_stable_vs_original_20_77: {profile_stable}",
                f"beats_20_81_profile_rmse: {beats_visual_profile}",
                f"beats_20_83_profile_rmse: {beats_negative_profile}",
                f"projected_mask_dice_acceptable: {dice_acceptable}",
                f"dimensions_reasonable: {dimensions_reasonable}",
                f"next_step: {next_step}",
                "",
                "20.77 remains the profile/depth candidate family. 20.81 remains the projected-mask visual comparator. 20.83 remains negative evidence for the tested profile-primary loss path.",
                "Do not update CURRENT_BASELINE.md from this result.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
