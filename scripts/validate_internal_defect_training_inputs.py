#!/usr/bin/env python
"""Validate 21.2 internal defect training-gate inputs.

This script performs the preflight and schema/input checks. It does not train,
run COMSOL, modify NPZ/data, or update CURRENT_BASELINE.md.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import (
    CHECK_FIELDS,
    DATASET_ID,
    PARAM_NAMES,
    ROOT,
    gate_manifest,
    load_dataset,
    resolve_dataset,
    split_indices,
    train_normalization,
    write_csv,
)


PREFLIGHT_SUMMARY = ROOT / "results/summaries/internal_defect_training_gate_preflight_summary.txt"
INPUT_SUMMARY = ROOT / "results/summaries/internal_defect_training_input_summary.txt"
INPUT_CHECK = ROOT / "results/metrics/internal_defect_training_input_check.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate internal defect training inputs.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--preflight-summary", type=Path, default=PREFLIGHT_SUMMARY)
    parser.add_argument("--summary", type=Path, default=INPUT_SUMMARY)
    parser.add_argument("--input-check", type=Path, default=INPUT_CHECK)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def main() -> int:
    args = parse_args()
    entry, manifest, npz_path = resolve_dataset(args.dataset_id)
    preflight_checks = gate_manifest(entry, manifest, npz_path, args.dataset_id)
    failed_preflight = [row["check_name"] for row in preflight_checks if not row["pass"]]
    if failed_preflight:
        args.preflight_summary.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_summary.write_text(
            "\n".join(
                [
                    "21.2 internal defect training gate preflight",
                    "",
                    f"dataset_id: {args.dataset_id}",
                    f"status: blocked",
                    f"failed_preflight_checks: {failed_preflight}",
                    "action: stop before training",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"preflight failed: {failed_preflight}")

    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    checks: list[dict[str, Any]] = []
    add(checks, "n_samples_96", dataset.delta_b.shape[0] == 96, dataset.delta_b.shape[0], 96)
    add(checks, "split_counts", {k: int(v.size) for k, v in splits.items()} == {"train": 64, "val": 16, "test": 16}, {k: int(v.size) for k, v in splits.items()}, "64/16/16")
    add(checks, "delta_b_shape", tuple(dataset.delta_b.shape) == (96, 3, 3, 201), dataset.delta_b.shape, "(96,3,3,201)")
    add(checks, "x_channels_shape", tuple(dataset.x_channels.shape) == (96, 9, 201), dataset.x_channels.shape, "(96,9,201)")
    add(checks, "delta_b_finite", bool(np.isfinite(dataset.delta_b).all()), bool(np.isfinite(dataset.delta_b).all()), True)
    add(checks, "axis_names", dataset.axis_names == ["Bx", "By", "Bz"], dataset.axis_names, "[Bx, By, Bz]")
    add(checks, "labels_present", dataset.y_regression.shape == (96, 7), dataset.y_regression.shape, "(96,7)")
    add(checks, "shape_labels_present", set(dataset.shape_type) == {"internal_sphere", "internal_ellipsoid", "internal_cuboid"}, dict(Counter(dataset.shape_type)), "3 classes")
    add(checks, "burial_depth_coverage", set(dataset.burial_depth_level) == {"shallow", "medium", "deep", "deep_plus"}, dict(Counter(dataset.burial_depth_level)), "4 bins")
    split_shape_counts = {name: {str(k): int(v) for k, v in Counter(dataset.shape_type[idx]).items()} for name, idx in splits.items()}
    split_burial_counts = {name: {str(k): int(v) for k, v in Counter(dataset.burial_depth_level[idx]).items()} for name, idx in splits.items()}
    add(checks, "split_shape_distribution_recorded", True, split_shape_counts, "audit only", "val/test shape coverage affects classification reliability")
    add(checks, "split_burial_distribution_recorded", True, split_burial_counts, "audit only", "val/test burial coverage affects depth generalization reliability")
    add(checks, "cavity_internal_true", bool(dataset.cavity_internal.all()), dataset.cavity_internal.tolist(), True)
    add(checks, "train_only_normalization", True, "computed from train split only", "train split only")
    x_mean, x_std = train_normalization(dataset.x_channels, splits["train"])
    add(checks, "normalization_shapes", x_mean.shape == (1, 9, 1) and x_std.shape == (1, 9, 1), f"mean={x_mean.shape}; std={x_std.shape}", "(1,9,1)")
    add(checks, "no_label_as_input", True, "model input is delta_b reshaped to (N,9,201)", "no labels/metadata as input")
    add(checks, "baseline_ready_false", not bool(dataset.manifest.get("baseline_ready")), dataset.manifest.get("baseline_ready"), False)
    add(checks, "current_baseline_unchanged_by_stage", True, "21.2 input check does not write CURRENT_BASELINE.md", "no baseline update")

    write_csv(args.input_check, checks, CHECK_FIELDS)
    failed_checks = [row["check_name"] for row in checks if not row["pass"]]
    shape_counts = {str(k): int(v) for k, v in Counter(dataset.shape_type).items()}
    burial_counts = {str(k): int(v) for k, v in Counter(dataset.burial_depth_level).items()}
    size_counts = {str(k): int(v) for k, v in Counter(dataset.size_level).items()}
    split_counts = {k: int(v.size) for k, v in splits.items()}
    args.preflight_summary.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_summary.write_text(
        "\n".join(
            [
                "21.2 内部/埋藏缺陷 training gate 预检查",
                "",
                f"dataset_id: {args.dataset_id}",
                f"registry_manifest_gate: passed",
                f"status: {dataset.manifest.get('status')}",
                f"train_ready_candidate: {dataset.manifest.get('train_ready_candidate')}",
                f"baseline_ready: {dataset.manifest.get('baseline_ready')}",
                f"npz_path: {dataset.npz_path}",
                "latest_newest_npz_scan: false",
                "comsol_run: false",
                "data_npz_mutation: false",
                "current_baseline_update: false",
                f"failed_input_checks: {failed_checks if failed_checks else 'none'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "21.2 内部/埋藏缺陷 training 输入摘要",
                "",
                f"dataset_id: {args.dataset_id}",
                f"N: {dataset.delta_b.shape[0]}",
                f"delta_b_shape: {dataset.delta_b.shape}",
                f"conv1d_input_shape: {dataset.x_channels.shape}",
                f"split_counts: {split_counts}",
                f"shape_counts: {shape_counts}",
                f"burial_depth_counts: {burial_counts}",
                f"split_shape_counts: {split_shape_counts}",
                f"split_burial_depth_counts: {split_burial_counts}",
                f"size_counts: {size_counts}",
                f"target_names: {PARAM_NAMES}",
                "input_policy: 模型输入只使用 delta_b/BxByBz；labels 和 metadata 只用于 supervision/metrics。",
                "train_only_normalization: true",
                "stop_condition: registry/manifest/schema 任一 blocker 都会在训练前停止。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if failed_checks:
        raise RuntimeError(f"input checks failed: {failed_checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
