#!/usr/bin/env python
"""Route decision for the 20.94 liftoff adapter training gate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import load_true_3d_rbc_liftoff_aug_dataset as liftoff


ROOT = liftoff.ROOT
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_adapter_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_decision_matrix.csv"
METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_metrics.csv"
VS_BASELINE = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_vs_baseline.csv"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_seed_summary.csv"
SCREEN_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_candidate_screen_metrics.csv"
TRAINING_20_92 = ROOT / "results/metrics/true_3d_rbc_liftoff_training_metrics.csv"


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


def selected_seed_row(rows: list[dict[str, str]]) -> dict[str, str]:
    selected = [row for row in rows if str(row.get("selected_robustness_candidate", "")).lower() == "true"]
    if not selected:
        raise RuntimeError("no selected robustness candidate in seed summary")
    return selected[0]


def metric_row(rows: list[dict[str, str]], candidate: str, seed: str, subset: str) -> dict[str, str]:
    return next(row for row in rows if row["candidate"] == candidate and str(row["seed"]) == str(seed) and row["split"] == "test" and row["liftoff_subset"] == subset)


def c0_row(rows: list[dict[str, str]], subset: str) -> dict[str, str]:
    return next(row for row in rows if row["candidate"] == "A0_baseline_replay" and row["split"] == "test" and row["liftoff_subset"] == subset)


def pct(cur: float, ref: float) -> float:
    return 0.0 if abs(ref) < 1.0e-20 else 100.0 * (cur - ref) / ref


def main() -> int:
    _args = argparse.ArgumentParser(description=__doc__).parse_args()
    metrics = read_csv(METRICS)
    seeds = read_csv(SEED_SUMMARY)
    selected = selected_seed_row(seeds)
    candidate = selected["candidate"]
    seed = selected["seed"]
    nom = metric_row(metrics, candidate, seed, "nominal_0p008")
    non = metric_row(metrics, candidate, seed, "non_nominal")
    c0_nom = c0_row(metrics, "nominal_0p008")
    c0_non = c0_row(metrics, "non_nominal")
    nom_change = pct(float(nom["profile_depth_rmse_m"]), float(c0_nom["profile_depth_rmse_m"]))
    non_change = pct(float(non["profile_depth_rmse_m"]), float(c0_non["profile_depth_rmse_m"]))
    dice_delta = float(non["projected_mask_dice"]) - float(c0_non["projected_mask_dice"])
    candidate_is_adapter = candidate in {"A1_output_residual_adapter", "A2_latent_residual_adapter"}
    nominal_pass = nom_change <= 10.0
    non_nominal_pass = non_change <= -20.0
    dice_pass = dice_delta >= -0.02
    robustness_candidate = nominal_pass and non_nominal_pass and dice_pass
    rows = [
        {
            "decision_item": "adapter_forms_robustness_candidate",
            "result": robustness_candidate,
            "evidence": f"nominal_change_pct={nom_change:.3f}; non_nominal_change_pct={non_change:.3f}; non_nominal_dice_delta={dice_delta:.6f}",
            "decision": "pass_candidate" if robustness_candidate else "not_yet_candidate",
        },
        {
            "decision_item": "sensor_z_conditioning_necessary",
            "result": True,
            "evidence": "20.93 diagnosed unconditioned liftoff ambiguity; 20.94 candidates all use sensor_z or a sensor_z ablation.",
            "decision": "keep_sensor_z_conditioning_for_liftoff_robustness",
        },
        {
            "decision_item": "adapter_preferred_over_full_model",
            "result": candidate_is_adapter,
            "evidence": f"validation-selected candidate={candidate}",
            "decision": "adapter_preferred" if candidate_is_adapter else "full_model_selected_by_validation",
        },
        {
            "decision_item": "current_baseline_update",
            "result": False,
            "evidence": "20.94 is a robustness candidate gate, not a baseline replacement.",
            "decision": "keep_CURRENT_BASELINE_20_85_unchanged",
        },
        {
            "decision_item": "next_step",
            "result": "formal_liftoff_benchmark" if robustness_candidate else "refine_adapter_or_pair_consistency",
            "evidence": "Proceed only if nominal is preserved and non-nominal improves without Dice collapse.",
            "decision": "formal_liftoff_benchmark" if robustness_candidate else "refine_nominal_preserving_adapter",
        },
    ]
    write_csv(MATRIX, rows)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.94 liftoff adapter route decision",
                "",
                f"selected_candidate: {candidate}",
                f"selected_seed: {seed}",
                f"nominal_profile_rmse_change_vs_C0_pct: {nom_change:.3f}",
                f"non_nominal_profile_rmse_change_vs_C0_pct: {non_change:.3f}",
                f"non_nominal_projected_mask_dice_delta_vs_C0: {dice_delta:.6f}",
                f"robustness_candidate: {robustness_candidate}",
                "CURRENT_BASELINE_update: false",
                "COMSOL_or_data_generation: false",
                "",
                "Decision:",
                "- If robustness_candidate is true, proceed to a formal liftoff benchmark; otherwise refine adapter/paired consistency before real-data alignment or internal defect feasibility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {SUMMARY}")
    print(f"wrote {MATRIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
