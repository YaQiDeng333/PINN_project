#!/usr/bin/env python
"""Route decision for 20.96 liftoff-conditioned inference smoke."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

import load_true_3d_rbc_pilot_dataset as pilot


ROOT = pilot.ROOT
SMOKE_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_conditioned_inference_smoke_metrics.csv"
BY_LIFTOFF = ROOT / "results/metrics/true_3d_rbc_liftoff_conditioned_inference_by_liftoff.csv"
FAILURES = ROOT / "results/metrics/true_3d_rbc_liftoff_conditioned_inference_failure_cases.csv"
CONTRACT = ROOT / "results/summaries/true_3d_rbc_liftoff_inference_metadata_contract.md"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_inference_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/true_3d_rbc_liftoff_inference_route_decision_matrix.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_for(rows: list[dict[str, str]], route: str, split: str, subset: str) -> dict[str, str]:
    return next(row for row in rows if row["route_mode"] == route and row["split"] == split and row["liftoff_subset"] == subset)


def pct_improvement(candidate: float, reference: float) -> float:
    return 100.0 * (reference - candidate) / reference if reference else float("nan")


def main() -> int:
    missing = [path for path in (SMOKE_METRICS, BY_LIFTOFF, FAILURES, CONTRACT) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing 20.96 decision inputs: " + ", ".join(str(path) for path in missing))
    rows = read_csv(SMOKE_METRICS)
    auto_all = row_for(rows, "auto", "test", "all")
    auto_nom = row_for(rows, "auto", "test", "nominal_0p008")
    auto_non = row_for(rows, "auto", "test", "non_nominal")
    base_nom = row_for(rows, "force_baseline", "test", "nominal_0p008")
    base_non = row_for(rows, "force_baseline", "test", "non_nominal")
    adapter_non = row_for(rows, "force_adapter", "test", "non_nominal")
    adapter_nom = row_for(rows, "force_adapter", "test", "nominal_0p008")

    auto_nom_delta = 100.0 * (float(auto_nom["profile_depth_rmse_m"]) - float(base_nom["profile_depth_rmse_m"])) / float(base_nom["profile_depth_rmse_m"])
    non_improvement = pct_improvement(float(auto_non["profile_depth_rmse_m"]), float(base_non["profile_depth_rmse_m"]))
    auto_replays_adapter_non = np.isclose(float(auto_non["profile_depth_rmse_m"]), float(adapter_non["profile_depth_rmse_m"]), rtol=0, atol=1.0e-12)
    auto_route_ok = float(auto_all["route_used_accuracy"]) >= 0.999
    nominal_preserved = abs(auto_nom_delta) <= 1.0
    non_nominal_improved = non_improvement >= 40.0

    decisions = [
        {
            "question": "inference_runner_available",
            "answer": True,
            "evidence": "20.96 smoke metrics generated",
            "decision": "runner usable for dataset or batch evaluation",
        },
        {
            "question": "auto_route_correct",
            "answer": auto_route_ok,
            "evidence": f"auto test route_used_accuracy={auto_all['route_used_accuracy']}",
            "decision": "sensor_z_m routing is deterministic and recorded",
        },
        {
            "question": "nominal_preserved_by_auto",
            "answer": nominal_preserved,
            "evidence": f"auto nominal RMSE={auto_nom['profile_depth_rmse_m']}; force_baseline nominal RMSE={base_nom['profile_depth_rmse_m']}; delta_pct={auto_nom_delta:.3f}",
            "decision": "auto route keeps 20.85 baseline at nominal liftoff",
        },
        {
            "question": "non_nominal_replays_a2",
            "answer": bool(auto_replays_adapter_non),
            "evidence": f"auto non-nominal RMSE={auto_non['profile_depth_rmse_m']}; force_adapter non-nominal RMSE={adapter_non['profile_depth_rmse_m']}",
            "decision": "auto route reproduces A2 companion behavior on non-nominal liftoff",
        },
        {
            "question": "non_nominal_improved_vs_baseline",
            "answer": non_nominal_improved,
            "evidence": f"force_baseline non-nominal RMSE={base_non['profile_depth_rmse_m']}; auto non-nominal RMSE={auto_non['profile_depth_rmse_m']}; improvement_pct={non_improvement:.3f}",
            "decision": "A2 companion remains useful for non-nominal liftoff",
        },
        {
            "question": "sensor_z_contract_sufficient_for_real_data_schema_intake",
            "answer": True,
            "evidence": str(CONTRACT.relative_to(ROOT)),
            "decision": "next stage should define real-data schema intake with mandatory sensor_z_m and no-defect reference metadata",
        },
        {
            "question": "internal_defect_next",
            "answer": False,
            "evidence": "current contract is surface-breaking RBC-style only",
            "decision": "internal defect remains deferred",
        },
    ]
    write_csv(MATRIX, decisions)
    summary = [
        "20.96 liftoff-conditioned inference route decision",
        "",
        f"auto_test_all_profile_depth_rmse_m: {float(auto_all['profile_depth_rmse_m']):.9f}",
        f"auto_test_all_projected_mask_dice: {float(auto_all['projected_mask_dice']):.6f}",
        f"auto_nominal_profile_depth_rmse_m: {float(auto_nom['profile_depth_rmse_m']):.9f}",
        f"auto_non_nominal_profile_depth_rmse_m: {float(auto_non['profile_depth_rmse_m']):.9f}",
        f"force_baseline_non_nominal_profile_depth_rmse_m: {float(base_non['profile_depth_rmse_m']):.9f}",
        f"force_adapter_nominal_profile_depth_rmse_m: {float(adapter_nom['profile_depth_rmse_m']):.9f}",
        f"force_adapter_non_nominal_profile_depth_rmse_m: {float(adapter_non['profile_depth_rmse_m']):.9f}",
        f"auto_nominal_delta_vs_baseline_pct: {auto_nom_delta:.3f}",
        f"auto_non_nominal_improvement_vs_baseline_pct: {non_improvement:.3f}",
        f"auto_route_replays_20_95_non_nominal_a2: {bool(auto_replays_adapter_non)}",
        "",
        "decision: liftoff-conditioned inference runner is usable.",
        "CURRENT_BASELINE: unchanged 20.85 nominal true 3D RBC baseline.",
        "companion_module: A2 latent residual adapter for non-nominal liftoff.",
        "sensor_z_m: mandatory metadata for multi-liftoff or real-data use.",
        "next_step: real-data schema intake / acquisition metadata contract; internal defect remains deferred.",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
