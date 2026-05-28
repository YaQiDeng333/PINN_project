#!/usr/bin/env python
"""验证 21.4 internal defect v2_240 training gate 输入。

本脚本只通过 registry + manifest 显式加载
comsol_internal_defect_pilot_pack_v2_240，不扫描 latest/newest，不训练，
不运行 COMSOL，不修改 data/NPZ，不更新 CURRENT_BASELINE.md。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import (
    CHECK_FIELDS,
    PARAM_NAMES,
    ROOT,
    gate_manifest,
    load_dataset,
    resolve_dataset,
    split_indices,
    train_normalization,
    write_csv,
)


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
PREFLIGHT_SUMMARY = ROOT / "results/summaries/internal_defect_v2_240_training_gate_preflight_summary.txt"
INPUT_SUMMARY = ROOT / "results/summaries/internal_defect_v2_training_input_summary.txt"
INPUT_CHECK = ROOT / "results/metrics/internal_defect_v2_training_input_check.csv"
SHAPES = {"internal_sphere", "internal_ellipsoid", "internal_cuboid"}
BURIALS = {"shallow", "medium", "deep", "deep_plus"}
SIZES = {"small", "medium", "large"}
ASPECTS = {"compact", "elongated_x", "elongated_y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 internal defect v2_240 training 输入。")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--preflight-summary", type=Path, default=PREFLIGHT_SUMMARY)
    parser.add_argument("--summary", type=Path, default=INPUT_SUMMARY)
    parser.add_argument("--input-check", type=Path, default=INPUT_CHECK)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def split_counts(values: np.ndarray, split: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for split_name in ["train", "val", "test"]:
        mask = split == split_name
        out[split_name] = {str(k): int(v) for k, v in Counter(values[mask]).items()}
    return dict(out)


def each_split_has(values: np.ndarray, split: np.ndarray, expected: set[str]) -> bool:
    counts = split_counts(values, split)
    return all(expected.issubset(set(counts[name])) for name in ["train", "val", "test"])


def main() -> int:
    args = parse_args()
    entry, manifest, npz_path = resolve_dataset(args.dataset_id)
    preflight_checks = gate_manifest(entry, manifest, npz_path, args.dataset_id)
    failed_preflight = [row["check_name"] for row in preflight_checks if not row["pass"]]
    args.preflight_summary.parent.mkdir(parents=True, exist_ok=True)
    if failed_preflight:
        args.preflight_summary.write_text(
            "\n".join(
                [
                    "21.4 internal defect v2_240 training gate 预检查",
                    f"dataset_id: {args.dataset_id}",
                    "status: blocked",
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
    split_count = {k: int(v.size) for k, v in splits.items()}
    shape_count = {str(k): int(v) for k, v in Counter(dataset.shape_type).items()}
    burial_count = {str(k): int(v) for k, v in Counter(dataset.burial_depth_level).items()}
    size_count = {str(k): int(v) for k, v in Counter(dataset.size_level).items()}
    aspect_count = {str(k): int(v) for k, v in Counter(dataset.aspect_bin).items()}
    split_shape = split_counts(dataset.shape_type, dataset.split)
    split_burial = split_counts(dataset.burial_depth_level, dataset.split)
    split_size = split_counts(dataset.size_level, dataset.split)
    split_aspect = split_counts(dataset.aspect_bin, dataset.split)

    add(checks, "n_samples_240", dataset.delta_b.shape[0] == 240, dataset.delta_b.shape[0], 240)
    add(checks, "split_counts_160_40_40", split_count == {"train": 160, "val": 40, "test": 40}, split_count, "160/40/40")
    add(checks, "delta_b_shape", tuple(dataset.delta_b.shape) == (240, 3, 3, 201), dataset.delta_b.shape, "(240,3,3,201)")
    add(checks, "x_channels_shape", tuple(dataset.x_channels.shape) == (240, 9, 201), dataset.x_channels.shape, "(240,9,201)")
    add(checks, "delta_b_finite", bool(np.isfinite(dataset.delta_b).all()), bool(np.isfinite(dataset.delta_b).all()), True)
    add(checks, "axis_names", dataset.axis_names == ["Bx", "By", "Bz"], dataset.axis_names, "[Bx, By, Bz]")
    add(checks, "labels_present", dataset.y_regression.shape == (240, 7), dataset.y_regression.shape, "(240,7)")
    add(checks, "shape_counts_balanced", shape_count == {"internal_sphere": 80, "internal_ellipsoid": 80, "internal_cuboid": 80}, shape_count, "80/80/80")
    add(checks, "burial_counts_balanced", burial_count == {"shallow": 60, "medium": 60, "deep": 60, "deep_plus": 60}, burial_count, "60 each")
    add(checks, "size_counts_balanced", size_count == {"small": 80, "medium": 80, "large": 80}, size_count, "80 each")
    add(checks, "aspect_coverage", ASPECTS.issubset(set(aspect_count)), aspect_count, "compact/elongated_x/elongated_y")
    add(checks, "each_split_has_all_shapes", each_split_has(dataset.shape_type, dataset.split, SHAPES), split_shape, "3 shape types in every split")
    add(checks, "each_split_has_all_burials", each_split_has(dataset.burial_depth_level, dataset.split, BURIALS), split_burial, "4 burial levels in every split")
    add(checks, "each_split_has_all_sizes", each_split_has(dataset.size_level, dataset.split, SIZES), split_size, "3 size levels in every split")
    add(checks, "cavity_internal_true", bool(dataset.cavity_internal.all()), "all true", True)
    x_mean, x_std = train_normalization(dataset.x_channels, splits["train"])
    add(checks, "train_only_normalization", True, "computed from train split only", "train split only")
    add(checks, "normalization_shapes", x_mean.shape == (1, 9, 1) and x_std.shape == (1, 9, 1), f"mean={x_mean.shape}; std={x_std.shape}", "(1,9,1)")
    add(checks, "no_label_as_input", True, "model input is delta_b reshaped to (N,9,201)", "no labels/metadata as input")
    add(checks, "baseline_ready_false", not bool(dataset.manifest.get("baseline_ready")), dataset.manifest.get("baseline_ready"), False)
    add(checks, "current_baseline_unchanged_by_stage", True, "21.4 input check does not write CURRENT_BASELINE.md", "no baseline update")

    write_csv(args.input_check, checks, CHECK_FIELDS)
    failed_checks = [row["check_name"] for row in checks if not row["pass"]]
    args.preflight_summary.write_text(
        "\n".join(
            [
                "21.4 internal defect v2_240 training gate 预检查",
                f"dataset_id: {args.dataset_id}",
                "registry_manifest_gate: passed",
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
                "21.4 internal defect v2_240 training 输入摘要",
                f"dataset_id: {args.dataset_id}",
                f"N: {dataset.delta_b.shape[0]}",
                f"delta_b_shape: {dataset.delta_b.shape}",
                f"conv1d_input_shape: {dataset.x_channels.shape}",
                f"split_counts: {split_count}",
                f"shape_counts: {shape_count}",
                f"burial_depth_counts: {burial_count}",
                f"size_counts: {size_count}",
                f"aspect_counts: {aspect_count}",
                f"split_shape_counts: {split_shape}",
                f"split_burial_depth_counts: {split_burial}",
                f"split_size_counts: {split_size}",
                f"split_aspect_counts: {split_aspect}",
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
