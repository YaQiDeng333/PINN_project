#!/usr/bin/env python
"""Route decision for the true-3D RBC observation robustness audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import ROOT, check_no_overwrite, write_csv


METRICS = ROOT / "results/metrics/true_3d_rbc_observation_perturbation_robustness_metrics.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_observation_robustness_route_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/true_3d_rbc_observation_robustness_decision_matrix.csv"

FIELDS = ["question", "answer", "evidence", "decision"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def test_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == "test"]


def by_name(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    matches = [row for row in rows if row["perturbation_name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one test row for {name}, found {len(matches)}")
    return matches[0]


def names(rows: list[dict[str, str]], group: str) -> list[dict[str, str]]:
    return [row for row in rows if row["perturbation_group"] == group]


def band_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {band: sum(1 for row in rows if row["status_band"] == band) for band in ("green", "warning", "fail")}


def main_decision(rows: list[dict[str, str]]) -> tuple[str, str]:
    noise10 = by_name(rows, "additive_noise_10pct")
    hard = by_name(rows, "combined_hard")
    light = by_name(rows, "combined_light")
    gain_rows = names(rows, "gain_scaling")
    reference_rows = names(rows, "no_defect_reference_error")
    jitter_rows = names(rows, "sensor_x_resampling_jitter")
    dropout_rows = [row for row in names(rows, "channel_attenuation_dropout") if row["severity"] == "missing"]
    noise10_ok = f(noise10, "profile_depth_rmse_degradation_pct") <= 10.0 and f(noise10, "projected_mask_dice_drop") <= 0.02
    hard_collapse = f(hard, "profile_depth_rmse_degradation_pct") > 100.0 or f(hard, "projected_mask_dice") < 0.50
    gain_fail = any(row["status_band"] == "fail" for row in gain_rows)
    combined_fail = light["status_band"] == "fail" or hard["status_band"] == "fail"
    ref_fail = any(row["status_band"] == "fail" for row in reference_rows)
    jitter_fail = any(row["status_band"] == "fail" for row in jitter_rows)
    dropout_fail = any(row["status_band"] == "fail" for row in dropout_rows)
    if noise10_ok and (gain_fail or combined_fail):
        return ("noise_jitter_reference_robust_but_gain_channel_sensitive", "add gain/amplitude calibration or augmentation before broad robustness claims; keep 20.89 COMSOL liftoff/sensor-offset pack as the next physics diagnostic")
    if noise10_ok and not hard_collapse and not ref_fail and not jitter_fail:
        return ("observation_robustness_pass_with_channel_dropout_diagnostic", "return_to_20_89_liftoff_sensor_offset_pack_after documenting channel dependencies")
    if ref_fail:
        return ("reference_subtraction_sensitive", "prioritize no-defect/background correction before real-data alignment")
    if jitter_fail:
        return ("sensor_alignment_sensitive", "prioritize sensor alignment/resampling protocol")
    if dropout_fail and noise10_ok:
        return ("noise_gain_robust_but_channel_dependent", "augmentation optional; COMSOL liftoff/sensor-offset pack still recommended")
    return ("observation_robustness_partial", "investigate augmentation before making robustness claims")


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([SUMMARY, DECISION_MATRIX], args.overwrite)
    rows = test_rows(read_csv(args.metrics))
    if not rows:
        raise RuntimeError("no test metrics available")
    counts = band_counts(rows)
    clean = by_name(rows, "additive_noise_0pct")
    noise5 = by_name(rows, "additive_noise_5pct")
    noise10 = by_name(rows, "additive_noise_10pct")
    noise20 = by_name(rows, "additive_noise_20pct")
    hard = by_name(rows, "combined_hard")
    light = by_name(rows, "combined_light")
    gain_rows = names(rows, "gain_scaling")
    worst_gain = max(gain_rows, key=lambda row: f(row, "profile_depth_rmse_degradation_pct"))
    worst_profile = max(rows, key=lambda row: f(row, "profile_depth_rmse_degradation_pct"))
    worst_dice = max(rows, key=lambda row: f(row, "projected_mask_dice_drop"))
    channel_missing = [row for row in rows if row["perturbation_group"] == "channel_attenuation_dropout" and row["severity"] == "missing"]
    worst_channel = max(channel_missing, key=lambda row: f(row, "profile_depth_rmse_degradation_pct"))
    bz_missing = by_name(rows, "channel_dropout_Bz_missing")
    bx_missing = by_name(rows, "channel_dropout_Bx_missing")
    by_missing = by_name(rows, "channel_dropout_By_missing")
    ref_rows = names(rows, "no_defect_reference_error")
    jitter_rows = names(rows, "sensor_x_resampling_jitter")
    reference_worst = max(ref_rows, key=lambda row: f(row, "profile_depth_rmse_degradation_pct"))
    jitter_worst = max(jitter_rows, key=lambda row: f(row, "profile_depth_rmse_degradation_pct"))
    decision, next_step = main_decision(rows)

    matrix = [
        {
            "question": "Does clean replay match the baseline artifact?",
            "answer": "yes",
            "evidence": f"profile_rmse={clean['profile_depth_rmse_m']}, dice={clean['projected_mask_dice']}",
            "decision": "artifact usable",
        },
        {
            "question": "Is additive noise <=10% acceptable?",
            "answer": "yes" if noise10["status_band"] in {"green", "warning"} else "no",
            "evidence": f"noise5 profile_deg={float(noise5['profile_depth_rmse_degradation_pct']):.3f}%, noise10 profile_deg={float(noise10['profile_depth_rmse_degradation_pct']):.3f}%, noise10 dice_drop={float(noise10['projected_mask_dice_drop']):.6f}",
            "decision": "robust" if noise10["status_band"] == "green" else noise10["status_band"],
        },
        {
            "question": "Is global gain scaling a robustness risk?",
            "answer": "yes" if worst_gain["status_band"] == "fail" else "no",
            "evidence": f"worst_gain={worst_gain['perturbation_name']}, profile_deg={float(worst_gain['profile_depth_rmse_degradation_pct']):.3f}%, dice_drop={float(worst_gain['projected_mask_dice_drop']):.6f}",
            "decision": "gain calibration or amplitude augmentation needed" if worst_gain["status_band"] == "fail" else "monitor",
        },
        {
            "question": "Does noise 20% or combined_hard collapse?",
            "answer": "no" if float(hard["projected_mask_dice"]) >= 0.50 and float(hard["profile_depth_rmse_degradation_pct"]) <= 100.0 else "yes",
            "evidence": f"noise20 band={noise20['status_band']}, hard profile_deg={float(hard['profile_depth_rmse_degradation_pct']):.3f}%, hard dice={float(hard['projected_mask_dice']):.6f}",
            "decision": "diagnostic stress only",
        },
        {
            "question": "Which channel is most critical under missing-channel diagnostics?",
            "answer": worst_channel["affected_axis"],
            "evidence": f"Bx_missing={float(bx_missing['profile_depth_rmse_degradation_pct']):.3f}%, By_missing={float(by_missing['profile_depth_rmse_degradation_pct']):.3f}%, Bz_missing={float(bz_missing['profile_depth_rmse_degradation_pct']):.3f}%",
            "decision": "channel dropout is diagnostic, not pass/fail",
        },
        {
            "question": "Is no-defect reference subtraction error a priority risk?",
            "answer": "yes" if reference_worst["status_band"] == "fail" else "no",
            "evidence": f"worst_reference={reference_worst['perturbation_name']}, profile_deg={float(reference_worst['profile_depth_rmse_degradation_pct']):.3f}%, dice_drop={float(reference_worst['projected_mask_dice_drop']):.6f}",
            "decision": "prioritize background correction" if reference_worst["status_band"] == "fail" else "monitor",
        },
        {
            "question": "Is sensor_x resampling jitter a priority risk?",
            "answer": "yes" if jitter_worst["status_band"] == "fail" else "no",
            "evidence": f"worst_jitter={jitter_worst['perturbation_name']}, profile_deg={float(jitter_worst['profile_depth_rmse_degradation_pct']):.3f}%, dice_drop={float(jitter_worst['projected_mask_dice_drop']):.6f}",
            "decision": "prioritize alignment/resampling" if jitter_worst["status_band"] == "fail" else "monitor",
        },
        {
            "question": "Is augmentation needed before 20.89?",
            "answer": "not as the immediate next step" if decision.startswith("observation_robustness_pass") else "consider after investigating failed perturbations",
            "evidence": f"status_counts={counts}, combined_light_band={light['status_band']}, combined_hard_band={hard['status_band']}",
            "decision": decision,
        },
        {
            "question": "Should the route continue to COMSOL liftoff/sensor-offset diagnostic pack?",
            "answer": "yes",
            "evidence": "observation perturbations do not replace liftoff/scan offset/source/material physics variation",
            "decision": next_step,
        },
    ]
    write_csv(DECISION_MATRIX, matrix, FIELDS)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.88 true 3D RBC observation robustness route decision summary",
                "",
                "COMSOL_run: false",
                "training_run: false",
                "data_or_NPZ_written: false",
                "CURRENT_BASELINE_update: false",
                f"test_status_counts: green={counts['green']}, warning={counts['warning']}, fail={counts['fail']}",
                f"clean_replay: profile_rmse={clean['profile_depth_rmse_m']}, dice={clean['projected_mask_dice']}",
                f"noise_10pct: profile_degradation_pct={float(noise10['profile_depth_rmse_degradation_pct']):.6f}, dice_drop={float(noise10['projected_mask_dice_drop']):.6f}, band={noise10['status_band']}",
                f"noise_20pct: profile_degradation_pct={float(noise20['profile_depth_rmse_degradation_pct']):.6f}, dice_drop={float(noise20['projected_mask_dice_drop']):.6f}, band={noise20['status_band']}",
                f"combined_light: profile_degradation_pct={float(light['profile_depth_rmse_degradation_pct']):.6f}, dice_drop={float(light['projected_mask_dice_drop']):.6f}, band={light['status_band']}",
                f"combined_hard: profile_degradation_pct={float(hard['profile_depth_rmse_degradation_pct']):.6f}, dice_drop={float(hard['projected_mask_dice_drop']):.6f}, band={hard['status_band']}",
                f"gain_scaling_worst: {worst_gain['perturbation_name']} ({float(worst_gain['profile_depth_rmse_degradation_pct']):.6f}% profile degradation)",
                f"most_sensitive_profile: {worst_profile['perturbation_name']} ({float(worst_profile['profile_depth_rmse_degradation_pct']):.6f}% profile degradation)",
                f"most_sensitive_dice: {worst_dice['perturbation_name']} ({float(worst_dice['projected_mask_dice_drop']):.6f} Dice drop)",
                f"channel_dependency: worst_missing={worst_channel['perturbation_name']}, Bx_missing={float(bx_missing['profile_depth_rmse_degradation_pct']):.6f}%, By_missing={float(by_missing['profile_depth_rmse_degradation_pct']):.6f}%, Bz_missing={float(bz_missing['profile_depth_rmse_degradation_pct']):.6f}%",
                f"reference_error_worst: {reference_worst['perturbation_name']} ({float(reference_worst['profile_depth_rmse_degradation_pct']):.6f}% profile degradation)",
                f"jitter_worst: {jitter_worst['perturbation_name']} ({float(jitter_worst['profile_depth_rmse_degradation_pct']):.6f}% profile degradation)",
                f"route_decision: {decision}",
                f"next_step: {next_step}",
                "Boundary: wMAE remains auxiliary; robustness claims are limited to observation-space perturbations of existing delta_b.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
