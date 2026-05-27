#!/usr/bin/env python
"""Replay the frozen 20.85 baseline on the 20.91b liftoff pack for 20.94."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any

import evaluate_true_3d_rbc_liftoff_baseline as baseline_eval
import load_true_3d_rbc_liftoff_aug_dataset as liftoff


ROOT = liftoff.ROOT
PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_adapter_preflight_summary.txt"
INPUT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_adapter_input_summary.txt"
BASELINE_REPLAY_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_baseline_replay_metrics.csv"
ARTIFACT_MANIFEST = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
REFERENCE_FILES = [
    ROOT / "results/metrics/true_3d_rbc_liftoff_training_metrics.csv",
    ROOT / "results/metrics/true_3d_rbc_liftoff_training_by_liftoff.csv",
    ROOT / "results/metrics/true_3d_rbc_liftoff_tradeoff_matrix.csv",
    ROOT / "results/metrics/true_3d_rbc_nominal_preserving_liftoff_strategy_matrix.csv",
]


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


def forbidden_staged_files() -> list[str]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    forbidden: list[str] = []
    for line in result.stdout.splitlines():
        lower = line.lower()
        if (
            lower.startswith("data/")
            or lower.startswith("notes/")
            or lower.endswith(".npz")
            or lower.endswith(".pt")
            or lower.endswith(".pth")
            or lower.endswith(".png")
            or lower.endswith(".mph")
            or "checkpoint" in lower
            or "__pycache__" in lower
            or line == "CURRENT_BASELINE.md"
            or line == "scripts/visualize_current_baseline.py"
        ):
            forbidden.append(line)
    return forbidden


def write_preflight(dataset: liftoff.True3DRBCLiftoffDataset) -> None:
    splits = liftoff.split_indices(dataset)
    split_bases = liftoff.split_base_ids(dataset)
    leakage = liftoff.base_split_leakage(dataset)
    missing_refs = [str(path.relative_to(ROOT)) for path in REFERENCE_FILES if not path.exists()]
    artifact_exists = ARTIFACT_MANIFEST.exists()
    forbidden = forbidden_staged_files()
    checks = [
        "20.94 true 3D RBC liftoff adapter preflight",
        "",
        f"dataset_id: {dataset.dataset_id}",
        f"rows: {len(dataset.sample_ids)}",
        f"base_count: {len(set(dataset.base_sample_ids.astype(str)))}",
        f"liftoff_levels_m: {sorted({round(float(x), 3) for x in dataset.sensor_z_m})}",
        f"split_rows: {{train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}}}",
        f"split_bases: {{train: {len(split_bases['train'])}, val: {len(split_bases['val'])}, test: {len(split_bases['test'])}}}",
        f"paired_liftoff_complete: {liftoff.paired_liftoff_complete(dataset)}",
        f"base_split_leakage: {leakage}",
        f"baseline_artifact_manifest_exists: {artifact_exists}",
        f"missing_reference_files: {missing_refs}",
        f"forbidden_staged_files: {forbidden}",
        "COMSOL_run: false",
        "training_run_in_preflight: false",
        "data_or_npz_write: false",
        "CURRENT_BASELINE_update: false",
    ]
    if leakage or missing_refs or forbidden or not artifact_exists:
        checks.append("preflight_status: BLOCKED")
    else:
        checks.append("preflight_status: PASS")
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(checks) + "\n", encoding="utf-8")
    if leakage or missing_refs or forbidden or not artifact_exists:
        raise RuntimeError("20.94 preflight blocked; see " + str(PREFLIGHT_SUMMARY))


def write_input_summary(dataset: liftoff.True3DRBCLiftoffDataset, baseline_rows: list[dict[str, Any]]) -> None:
    test_nom = next(row for row in baseline_rows if row["split"] == "test" and row["liftoff_subset"] == "nominal_0p008")
    test_non = next(row for row in baseline_rows if row["split"] == "test" and row["liftoff_subset"] == "non_nominal")
    lines = [
        "20.94 liftoff adapter input and baseline replay summary",
        "",
        f"dataset_id: {dataset.dataset_id}",
        "load_path: COMSOL_DATA_REGISTRY.md + manifest only; no latest/newest discovery",
        f"npz_path_ignored: {dataset.npz_path}",
        "input_allowed: delta_b/BxByBz, sensor_z_m scalar, frozen baseline prediction/latent",
        "input_forbidden: rbc_params, mask, profile, template, depth_bin, aspect_bin, sample_id",
        "split_policy: grouped by base_sample_id, train/val/test bases=32/8/8",
        "baseline_model: frozen 20.85/20.77 small Conv1D + MLP six-parameter head",
        f"baseline_test_nominal_profile_depth_rmse_m: {float(test_nom['profile_depth_rmse_m']):.9f}",
        f"baseline_test_non_nominal_profile_depth_rmse_m: {float(test_non['profile_depth_rmse_m']):.9f}",
        f"baseline_test_non_nominal_projected_mask_dice: {float(test_non['projected_mask_dice']):.6f}",
    ]
    INPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    INPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(dataset_id: str = liftoff.DATASET_ID) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = liftoff.load_liftoff_dataset(dataset_id)
    write_preflight(dataset)
    baseline_rows, by_liftoff = baseline_eval.run(dataset_id)
    write_csv(BASELINE_REPLAY_METRICS, baseline_rows)
    write_input_summary(dataset, baseline_rows)
    return baseline_rows, by_liftoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=liftoff.DATASET_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args.dataset_id)
    print(f"wrote {PREFLIGHT_SUMMARY}")
    print(f"wrote {INPUT_SUMMARY}")
    print(f"wrote {BASELINE_REPLAY_METRICS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
