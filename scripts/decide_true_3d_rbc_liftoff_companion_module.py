#!/usr/bin/env python
"""Decide whether A2 is a companion liftoff robustness module."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import load_true_3d_rbc_liftoff_aug_dataset as liftoff


ROOT = liftoff.ROOT
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_companion_module_decision_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_liftoff_companion_module_decision_matrix.csv"
FORMAL = ROOT / "results/metrics/true_3d_rbc_formal_liftoff_benchmark_metrics.csv"
BY_LIFTOFF = ROOT / "results/metrics/true_3d_rbc_formal_liftoff_benchmark_by_liftoff.csv"


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


def metric(rows: list[dict[str, str]], candidate: str, subset: str, name: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["candidate"] == candidate and row["liftoff_subset"] == subset and row["metric"] == name
    ]
    if not matches:
        raise RuntimeError(f"missing metric {candidate} {subset} {name}")
    return matches[0]


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    rows = read_csv(FORMAL)
    levels = read_csv(BY_LIFTOFF)
    a2 = "A2_latent_residual_adapter_20_94_selected"

    nom_rmse = metric(rows, a2, "nominal_0p008", "profile_depth_rmse_m")
    non_rmse = metric(rows, a2, "non_nominal", "profile_depth_rmse_m")
    all_rmse = metric(rows, a2, "all_liftoff", "profile_depth_rmse_m")
    nom_dice = metric(rows, a2, "nominal_0p008", "projected_mask_dice")
    non_dice = metric(rows, a2, "non_nominal", "projected_mask_dice")
    non_l = metric(rows, a2, "non_nominal", "L_mae_mm")
    non_w = metric(rows, a2, "non_nominal", "W_mae_mm")
    non_d = metric(rows, a2, "non_nominal", "D_mae_mm")
    non_wmae = metric(rows, a2, "non_nominal", "wMAE_auxiliary")
    nom_change = f(nom_rmse, "relative_change_pct_vs_C0")
    non_change = f(non_rmse, "relative_change_pct_vs_C0")
    all_change = f(all_rmse, "relative_change_pct_vs_C0")
    non_dice_delta = f(non_dice, "delta_vs_C0")
    nominal_preserved = nom_change <= 10.0 and f(nom_dice, "delta_vs_C0") >= -0.02
    non_nominal_improved = non_change <= -20.0 and non_dice_delta >= 0.10
    companion = nominal_preserved and non_nominal_improved

    a2_level_rmse = [row for row in levels if row["candidate"] == a2 and row["metric"] == "profile_depth_rmse_m"]
    best_level = min(a2_level_rmse, key=lambda row: float(row["value"]))
    worst_level = max(a2_level_rmse, key=lambda row: float(row["value"]))

    decision_rows = [
        {
            "decision_item": "a2_preserves_nominal_operating_point",
            "result": nominal_preserved,
            "evidence": f"nominal_profile_rmse_change_pct={nom_change:.3f}; nominal_dice_delta={f(nom_dice, 'delta_vs_C0'):.6f}",
            "decision": "pass_nominal_guard" if nominal_preserved else "fail_nominal_guard",
        },
        {
            "decision_item": "a2_improves_non_nominal_liftoff",
            "result": non_nominal_improved,
            "evidence": f"non_nominal_profile_rmse_change_pct={non_change:.3f}; non_nominal_dice_delta={non_dice_delta:.6f}",
            "decision": "pass_non_nominal_guard" if non_nominal_improved else "fail_non_nominal_guard",
        },
        {
            "decision_item": "sensor_z_m_required_metadata",
            "result": True,
            "evidence": "A2 is sensor_z-conditioned and 20.93 diagnosed unconditioned liftoff ambiguity.",
            "decision": "require_sensor_z_m_for_multi_liftoff_or_real_experimental_inference",
        },
        {
            "decision_item": "current_baseline_companion_module",
            "result": companion,
            "evidence": "A2 preserves nominal C0 behavior while correcting non-nominal liftoff.",
            "decision": "accept_as_companion_module" if companion else "do_not_accept_as_companion_module",
        },
        {
            "decision_item": "current_baseline_update",
            "result": False,
            "evidence": "CURRENT_BASELINE remains the 20.85 nominal true 3D RBC profile-depth baseline.",
            "decision": "keep_CURRENT_BASELINE_unchanged",
        },
        {
            "decision_item": "internal_defect_feasibility",
            "result": "defer",
            "evidence": "Surface RBC liftoff robustness should first pass an inference smoke path with sensor_z_m metadata.",
            "decision": "defer_internal_defect_feasibility",
        },
        {
            "decision_item": "next_unique_step",
            "result": "liftoff_conditioned_inference_smoke",
            "evidence": "Formal benchmark supports A2 as companion; next verify operational loading and sensor_z_m contract before real-data alignment.",
            "decision": "run_liftoff_conditioned_inference_smoke",
        },
    ]
    write_csv(MATRIX, decision_rows)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.95 true 3D RBC liftoff companion module decision",
                "",
                "Decision: accept A2_latent_residual_adapter as the CURRENT_BASELINE companion robustness module.",
                "CURRENT_BASELINE_update: false",
                "CURRENT_BASELINE_scope: unchanged 20.85 nominal true 3D RBC profile-depth baseline",
                "A2_scope: liftoff robustness companion module for multi-liftoff inference, not baseline replacement",
                "",
                "Evidence:",
                f"- nominal profile RMSE change vs C0: {nom_change:.3f}% ({nom_rmse['reference_value']} -> {nom_rmse['value']} m)",
                f"- non-nominal profile RMSE change vs C0: {non_change:.3f}% ({non_rmse['reference_value']} -> {non_rmse['value']} m)",
                f"- all-liftoff profile RMSE change vs C0: {all_change:.3f}%",
                f"- non-nominal Dice change vs C0: {non_dice_delta:.6f} ({non_dice['reference_value']} -> {non_dice['value']})",
                f"- non-nominal L/W/D MAE: {float(non_l['value']):.3f} / {float(non_w['value']):.3f} / {float(non_d['value']):.3f} mm",
                f"- non-nominal wMAE auxiliary: {float(non_wmae['value']):.6f}",
                f"- best A2 liftoff RMSE level: {best_level['liftoff_subset']} = {float(best_level['value']):.12f} m",
                f"- worst A2 liftoff RMSE level: {worst_level['liftoff_subset']} = {float(worst_level['value']):.12f} m",
                "",
                "Operational boundary:",
                "- sensor_z_m is required metadata for multi-liftoff or real-experimental inference.",
                "- A2 should be loaded as a companion correction module after the frozen 20.85 baseline path.",
                "- wMAE remains auxiliary; profile RMSE, Er-like profile error, and Dice carry the benchmark decision.",
                "- Internal/buried defect feasibility remains deferred.",
                "",
                "Next unique step: liftoff-conditioned inference smoke, verifying frozen baseline + A2 adapter loading and the sensor_z_m metadata contract before real-data alignment.",
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
