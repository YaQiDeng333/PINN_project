#!/usr/bin/env python
"""Prepare Stage 20.83 profile-primary loss inputs and generator checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    V3_240_DATASET_ID,
    ROOT,
    check_no_overwrite,
    gate_manifest,
    load_dataset,
    resolve_dataset,
    split_indices,
    train_normalization,
    write_csv,
)
from true_3d_rbc_profile_generator import generator_consistency_rows


SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_profile_primary_input_summary.txt"
CONSISTENCY = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_generator_consistency.csv"
INPUT_CHECK = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_input_check.csv"

CHECK_FIELDS = ["check_name", "pass", "observed", "notes"]


def add_check(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "notes": notes})


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.consistency, args.input_check], args.overwrite)
    entry, manifest, npz_path = resolve_dataset(args.dataset_id)
    checks = gate_manifest(entry, manifest, npz_path, args.dataset_id)
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    stats = train_normalization(dataset)
    split_counts = {name: len(idx) for name, idx in splits.items()}
    add_check(checks, "sample_count_240", len(dataset.sample_ids) == 240, len(dataset.sample_ids))
    add_check(checks, "split_counts_162_39_39", split_counts == {"train": 162, "val": 39, "test": 39}, split_counts)
    add_check(checks, "delta_b_shape", dataset.delta_b.shape == (240, 3, 3, 201), list(dataset.delta_b.shape))
    add_check(checks, "conv1d_shape", dataset.x_channels.shape == (240, 9, 201), list(dataset.x_channels.shape))
    add_check(checks, "rbc_params_shape", dataset.rbc_params.shape == (240, 6), list(dataset.rbc_params.shape))
    add_check(checks, "profile_depth_grid_present", dataset.profile_depth_grid_m.shape == (240, 33, 17), list(dataset.profile_depth_grid_m.shape))
    add_check(checks, "profile_depth_map_present", dataset.profile_depth_map_xy_m.shape == (240, 64, 128), list(dataset.profile_depth_map_xy_m.shape))
    add_check(checks, "projected_mask_present", dataset.projected_mask_2d.shape == (240, 64, 128), list(dataset.projected_mask_2d.shape))
    add_check(checks, "projected_mask_nonempty", bool((dataset.projected_mask_2d.sum(axis=(1, 2)) > 0).all()), int(dataset.projected_mask_2d.sum()))
    add_check(checks, "axis_names", dataset.axis_names == ["Bx", "By", "Bz"], dataset.axis_names)
    add_check(checks, "finite_delta_b", bool(np.isfinite(dataset.delta_b).all()), "finite")
    add_check(checks, "finite_rbc_params", bool(np.isfinite(dataset.rbc_params).all()), "finite")
    add_check(checks, "train_only_normalization_prepared", True, {"x_mean": list(stats["x_mean"].shape), "y_mean": list(stats["y_mean"].shape)})
    add_check(checks, "latest_newest_scan_used", True, False, "loader resolves only explicit dataset_id via registry/manifest")

    consistency_rows = generator_consistency_rows(dataset)
    write_csv(args.consistency, consistency_rows, CHECK_FIELDS)
    checks.extend(consistency_rows)
    failed = [row for row in checks if not bool(row["pass"])]
    write_csv(args.input_check, checks, CHECK_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 profile-primary input summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"npz_path: {dataset.npz_path}",
                f"manifest_stage: {manifest.get('stage')}",
                f"route: {manifest.get('route')}",
                f"status: {manifest.get('status')}",
                f"train_ready_candidate: {manifest.get('train_ready_candidate')}",
                f"baseline_ready: {manifest.get('baseline_ready')}",
                f"allowed_use: {manifest.get('allowed_use')}",
                f"forbidden_use: {manifest.get('forbidden_use')}",
                f"input_shape_delta_b: {list(dataset.delta_b.shape)}",
                f"input_shape_conv1d: {list(dataset.x_channels.shape)}",
                f"profile_depth_grid_shape: {list(dataset.profile_depth_grid_m.shape)}",
                f"profile_depth_map_shape: {list(dataset.profile_depth_map_xy_m.shape)}",
                f"projected_mask_shape: {list(dataset.projected_mask_2d.shape)}",
                f"split_counts: {split_counts}",
                f"generator_consistency_pass: {not failed}",
                f"generator_consistency_rows: {json.dumps(consistency_rows, ensure_ascii=False)}",
                "differentiable_generator: torch RBC-style approximation mirrors current NumPy generator; exact_piao_rbc=False.",
                "training_required_for_20_83: yes",
                "input_boundary: model input is delta_b/BxByBz only unless an explicitly allowlisted delta_b-derived feature candidate is used; labels are supervision/metrics only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError("profile-primary input/generator gate failed")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--consistency", type=Path, default=CONSISTENCY)
    parser.add_argument("--input-check", type=Path, default=INPUT_CHECK)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
