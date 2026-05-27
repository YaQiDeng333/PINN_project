#!/usr/bin/env python
"""Route decision for 20.92 liftoff-aware true-3D RBC training gate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

import load_true_3d_rbc_liftoff_aug_dataset as liftoff


ROOT = liftoff.ROOT
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_training_route_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/true_3d_rbc_liftoff_training_route_decision_matrix.csv"
TRAINING_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_training_metrics.csv"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_liftoff_training_seed_summary.csv"
VS_BASELINE = ROOT / "results/metrics/true_3d_rbc_liftoff_training_vs_baseline.csv"


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


def selected_seed_row(seed_rows: list[dict[str, str]]) -> dict[str, str]:
    selected = [row for row in seed_rows if str(row.get("selected_robustness_candidate", "")).lower() == "true"]
    if not selected:
        raise RuntimeError("no selected robustness candidate in seed summary")
    return selected[0]


def metric_delta(vs_rows: list[dict[str, str]], subset: str, metric: str) -> dict[str, str] | None:
    return next((row for row in vs_rows if row["liftoff_subset"] == subset and row["metric"] == metric), None)


def run() -> tuple[list[dict[str, Any]], str]:
    metrics = read_csv(TRAINING_METRICS)
    seeds = read_csv(SEED_SUMMARY)
    vs_rows = read_csv(VS_BASELINE)
    selected = selected_seed_row(seeds)
    candidate = selected["candidate"]
    seed = selected["seed"]
    non_rmse = metric_delta(vs_rows, "non_nominal", "profile_depth_rmse_m")
    nom_rmse = metric_delta(vs_rows, "nominal_0p008", "profile_depth_rmse_m")
    non_dice = metric_delta(vs_rows, "non_nominal", "projected_mask_dice")
    lwd_deltas = [
        metric_delta(vs_rows, "non_nominal", metric)
        for metric in ("L_mae_mm", "W_mae_mm", "D_mae_mm")
    ]
    c2_rows = [row for row in metrics if row["candidate"] == "C2_sensor_z_conditioned" and row["split"] == "test" and row["liftoff_subset"] == "non_nominal"]
    c1_rows = [row for row in metrics if row["candidate"] == "C1_unconditioned_liftoff_aug" and row["split"] == "test" and row["liftoff_subset"] == "non_nominal"]
    best_c2 = min(c2_rows, key=lambda row: float(row["profile_depth_rmse_m"])) if c2_rows else None
    best_c1 = min(c1_rows, key=lambda row: float(row["profile_depth_rmse_m"])) if c1_rows else None
    c2_better_than_c1 = bool(best_c1 and best_c2 and float(best_c2["profile_depth_rmse_m"]) < float(best_c1["profile_depth_rmse_m"]))
    profile_improved = bool(non_rmse and float(non_rmse["relative_change_pct"]) < -10.0)
    nominal_stable = bool(nom_rmse and float(nom_rmse["relative_change_pct"]) <= 10.0)
    dice_stable = bool(non_dice and float(non_dice["delta"]) >= -0.03)
    lwd_stable = all(row is not None and float(row["relative_change_pct"]) <= 25.0 for row in lwd_deltas)
    robustness_candidate = profile_improved and nominal_stable and dice_stable and lwd_stable
    next_step = (
        "formal_liftoff_benchmark_rerun_for_robustness_candidate"
        if robustness_candidate
        else "inspect_liftoff_pack_failure_cases_before_more_COMSOL_or_real_data"
    )
    matrix = [
        {
            "question": "Did liftoff augmentation improve non-nominal profile RMSE vs fixed 20.85 baseline?",
            "answer": profile_improved,
            "evidence": f"{non_rmse['relative_change_pct']}% relative change" if non_rmse else "missing",
            "decision": "go" if profile_improved else "risk",
        },
        {
            "question": "Did nominal 0.008m performance remain stable?",
            "answer": nominal_stable,
            "evidence": f"{nom_rmse['relative_change_pct']}% relative change" if nom_rmse else "missing",
            "decision": "go" if nominal_stable else "risk",
        },
        {
            "question": "Did projected mask Dice remain stable on non-nominal liftoff?",
            "answer": dice_stable,
            "evidence": f"Dice delta {non_dice['delta']}" if non_dice else "missing",
            "decision": "go" if dice_stable else "risk",
        },
        {
            "question": "Did L/W/D remain stable?",
            "answer": lwd_stable,
            "evidence": "; ".join(f"{row['metric']} {row['relative_change_pct']}%" for row in lwd_deltas if row),
            "decision": "go" if lwd_stable else "risk",
        },
        {
            "question": "Is sensor_z conditioning necessary/useful?",
            "answer": c2_better_than_c1 or candidate == "C2_sensor_z_conditioned",
            "evidence": (
                f"post-hoc test diagnostic only, not selection: best C2 non-nominal RMSE={best_c2['profile_depth_rmse_m']}; best C1={best_c1['profile_depth_rmse_m']}"
                if best_c1 and best_c2
                else "missing C1/C2 comparison"
            ),
            "decision": "prefer_C2" if c2_better_than_c1 or candidate == "C2_sensor_z_conditioned" else "C1_sufficient_for_now",
        },
        {
            "question": "Does 20.92 form a robustness candidate?",
            "answer": robustness_candidate,
            "evidence": f"selected={candidate}, seed={seed}; next_step={next_step}",
            "decision": next_step,
        },
        {
            "question": "Should CURRENT_BASELINE change?",
            "answer": False,
            "evidence": "20.92 is a liftoff robustness gate; 20.85 nominal profile-depth baseline remains current.",
            "decision": "do_not_update_current_baseline",
        },
        {
            "question": "Can internal defect feasibility proceed now?",
            "answer": False,
            "evidence": "Resolve/formalize liftoff robustness candidate first.",
            "decision": "defer_internal_defect_feasibility",
        },
    ]
    lines = [
        "20.92 true 3D RBC liftoff-aware training route decision",
        "",
        f"selected_candidate: {candidate}",
        f"selected_seed: {seed}",
        f"robustness_candidate: {robustness_candidate}",
        f"sensor_z_conditioning_useful: {c2_better_than_c1 or candidate == 'C2_sensor_z_conditioned'}",
        f"CURRENT_BASELINE_update: false",
        f"next_step: {next_step}",
        "",
        "Decision evidence:",
    ]
    for row in matrix:
        lines.append(f"- {row['question']} {row['answer']} ({row['evidence']})")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(DECISION_MATRIX, matrix)
    return matrix, next_step


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> int:
    _args = parse_args()
    _matrix, next_step = run()
    print(f"wrote {SUMMARY}")
    print(f"next_step={next_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
