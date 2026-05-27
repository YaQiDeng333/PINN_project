#!/usr/bin/env python
"""20.92 liftoff-aware training input and split gate.

This is a read-only gate: it resolves comsol_true_3d_rbc_liftoff_aug_pack_v1
through COMSOL_DATA_REGISTRY.md + manifest, validates grouped base_id splits,
and writes summaries/metrics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import audit_true_3d_rbc_observation_perturbation_robustness as obs
import load_true_3d_rbc_liftoff_aug_dataset as liftoff
import load_true_3d_rbc_pilot_dataset as pilot


ROOT = liftoff.ROOT
PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_training_gate_preflight_summary.txt"
INPUT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_training_input_summary.txt"
INPUT_CHECK = ROOT / "results/metrics/true_3d_rbc_liftoff_training_input_check.csv"
ARTIFACT_MANIFEST = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
REFERENCES = [
    ROOT / "results/summaries/true_3d_rbc_benchmark_report_summary.txt",
    ROOT / "results/summaries/true_3d_rbc_liftoff_sensor_offset_route_decision_summary.txt",
    ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_group_summary.csv",
    ROOT / "results/summaries/true_3d_rbc_liftoff_aug_pack_route_decision_summary.txt",
]


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


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def git_staged_names() -> list[str]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def validate(dataset_id: str) -> tuple[Any, list[dict[str, Any]], dict[str, np.ndarray]]:
    entry, manifest, npz_path = liftoff.resolve_liftoff_dataset(dataset_id)
    rows = liftoff.gate_liftoff_manifest(entry, manifest, npz_path, dataset_id)
    dataset = liftoff.load_liftoff_dataset(dataset_id)
    stats = liftoff.train_normalization(dataset)
    splits = liftoff.split_indices(dataset)
    base_splits = liftoff.split_base_ids(dataset)
    leakage = liftoff.base_split_leakage(dataset)
    levels = sorted({round(float(x), 3) for x in dataset.sensor_z_m})
    split_rows = {name: len(idx) for name, idx in splits.items()}
    split_bases = {name: len(base_splits[name]) for name in ("train", "val", "test")}
    delta_error = float(np.max(np.abs(dataset.delta_b - (dataset.b_defect - dataset.b_no_defect))))
    staged = git_staged_names()
    forbidden_staged = [
        path
        for path in staged
        if path.startswith("data/")
        or path.lower().endswith((".npz", ".pt", ".pth", ".png", ".mph"))
        or path.startswith("notes/")
        or "CURRENT_BASELINE.md" == path
        or "scripts/visualize_current_baseline.py" == path
    ]

    add(rows, "row_count", len(dataset.sample_ids) == 192, len(dataset.sample_ids), 192)
    add(rows, "base_count", len(set(dataset.base_sample_ids.astype(str))) == 48, len(set(dataset.base_sample_ids.astype(str))), 48)
    add(rows, "liftoff_levels", levels == [0.006, 0.008, 0.01, 0.012], levels, [0.006, 0.008, 0.01, 0.012])
    add(rows, "paired_liftoff_complete", liftoff.paired_liftoff_complete(dataset), liftoff.paired_liftoff_complete(dataset), True)
    add(rows, "split_rows_by_base", split_rows == {"train": 128, "val": 32, "test": 32}, split_rows, {"train": 128, "val": 32, "test": 32})
    add(rows, "split_base_counts", split_bases == {"train": 32, "val": 8, "test": 8}, split_bases, {"train": 32, "val": 8, "test": 8})
    add(rows, "base_id_no_split_leakage", not leakage, json.dumps(leakage, ensure_ascii=False), "[]")
    add(rows, "delta_b_shape", dataset.delta_b.shape == (192, 3, 3, 201), list(dataset.delta_b.shape), [192, 3, 3, 201])
    add(rows, "conv1d_shape", dataset.x_channels.shape == (192, 9, 201), list(dataset.x_channels.shape), [192, 9, 201])
    add(rows, "axis_names", dataset.axis_names == ["Bx", "By", "Bz"], dataset.axis_names, ["Bx", "By", "Bz"])
    add(rows, "delta_b_finite", bool(np.isfinite(dataset.delta_b).all()), "finite", "finite")
    add(rows, "rbc_params_finite", bool(np.isfinite(dataset.rbc_params).all()), "finite", "finite")
    add(rows, "profile_labels_present", dataset.profile_depth_grid_m.shape == (192, 33, 17), list(dataset.profile_depth_grid_m.shape), [192, 33, 17])
    add(rows, "projected_mask_present", dataset.projected_mask_2d.shape == (192, 64, 128), list(dataset.projected_mask_2d.shape), [192, 64, 128])
    add(rows, "delta_recompute_error", delta_error <= 1.0e-7, f"{delta_error:.6e}", "<=1e-7")
    add(rows, "train_only_x_normalization", stats["x_mean"].shape == (1, 9, 1) and stats["x_std"].shape == (1, 9, 1), {"x_mean": list(stats["x_mean"].shape), "x_std": list(stats["x_std"].shape)}, "(1,9,1)")
    add(rows, "train_only_y_normalization", stats["y_mean"].shape == (1, 6) and stats["y_std"].shape == (1, 6), {"y_mean": list(stats["y_mean"].shape), "y_std": list(stats["y_std"].shape)}, "(1,6)")
    add(rows, "train_only_sensor_z_normalization", stats["sensor_z_mean"].shape == (1, 1) and stats["sensor_z_std"].shape == (1, 1), {"z_mean": float(stats["sensor_z_mean"][0, 0]), "z_std": float(stats["sensor_z_std"][0, 0])}, "(1,1)")
    add(rows, "artifact_manifest_exists", ARTIFACT_MANIFEST.exists(), str(ARTIFACT_MANIFEST), "20.88a artifact manifest")
    if ARTIFACT_MANIFEST.exists():
        try:
            artifact, checkpoint, _model = obs.load_artifact(ARTIFACT_MANIFEST)
            add(rows, "baseline_artifact_sha_verified", True, artifact.get("artifact_id"), "sha256 verified")
            add(rows, "baseline_artifact_seed42", checkpoint.get("seed") == 42, checkpoint.get("seed"), 42)
        except Exception as exc:  # noqa: BLE001
            add(rows, "baseline_artifact_loadable", False, repr(exc), "loadable")
    for ref in REFERENCES:
        add(rows, f"reference_exists:{ref.name}", ref.exists(), str(ref), "existing 20.85/20.90/20.91b reference")
    add(rows, "no_forbidden_artifacts_staged", not forbidden_staged, forbidden_staged, "none")
    return dataset, rows, stats


def write_summaries(dataset: Any, checks: list[dict[str, Any]], stats: dict[str, np.ndarray]) -> None:
    failed = [row for row in checks if not row["pass"]]
    splits = liftoff.split_indices(dataset)
    base_splits = liftoff.split_base_ids(dataset)
    lines = [
        "20.92 true 3D RBC liftoff-aware training preflight summary",
        "",
        f"dataset_id: {dataset.dataset_id}",
        f"registry_manifest_gate_pass: {not failed}",
        f"npz_path: {dataset.npz_path}",
        f"rows: {len(dataset.sample_ids)}",
        f"base_count: {len(set(dataset.base_sample_ids.astype(str)))}",
        f"liftoff_levels_m: {sorted({round(float(x), 3) for x in dataset.sensor_z_m})}",
        f"split_rows: {{'train': {len(splits['train'])}, 'val': {len(splits['val'])}, 'test': {len(splits['test'])}}}",
        f"split_bases: {{'train': {len(base_splits['train'])}, 'val': {len(base_splits['val'])}, 'test': {len(base_splits['test'])}}}",
        f"base_id_split_leakage: {liftoff.base_split_leakage(dataset)}",
        f"train_only_sensor_z_mean_std: {float(stats['sensor_z_mean'][0, 0]):.6f} / {float(stats['sensor_z_std'][0, 0]):.6f}",
        "model_inputs_allowed: delta_b/BxByBz; C2 may additionally use train-normalized sensor_z_m scalar",
        "forbidden_model_inputs: rbc_params, mask/profile labels, split, template/depth/aspect bins, sample_id/base_id",
        "labels_scope: supervision and metrics only",
        "COMSOL_run: false",
        "data_or_npz_write: false",
        "CURRENT_BASELINE_update: false",
        "",
        "Failed checks:",
    ]
    lines.extend([f"- {row['check_name']}: {row['observed']}" for row in failed] or ["- none"])
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    INPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    INPUT_SUMMARY.write_text(
        "\n".join(
            [
                "20.92 liftoff training input summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"delta_b_shape: {list(dataset.delta_b.shape)}",
                f"conv1d_shape: {list(dataset.x_channels.shape)}",
                f"rbc_params_shape: {list(dataset.rbc_params.shape)}",
                f"profile_depth_grid_shape: {list(dataset.profile_depth_grid_m.shape)}",
                f"projected_mask_shape: {list(dataset.projected_mask_2d.shape)}",
                f"base_id_paired_liftoff_complete: {liftoff.paired_liftoff_complete(dataset)}",
                "split_policy: existing 20.91b split is by base_sample_id; no geometry leakage",
                "normalization: input, target, and sensor_z scalers fit on train rows only",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=liftoff.DATASET_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset, checks, stats = validate(args.dataset_id)
    write_csv(INPUT_CHECK, checks)
    write_summaries(dataset, checks, stats)
    failed = [row for row in checks if not row["pass"]]
    if failed:
        raise RuntimeError(f"20.92 liftoff input gate failed: {[row['check_name'] for row in failed]}")
    print(f"wrote {PREFLIGHT_SUMMARY}")
    print(f"wrote {INPUT_CHECK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
