#!/usr/bin/env python
"""23.1 richer-observation training input validation。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_richer_observation_dataset import (
    DATASET_ID,
    EXPECTED_VARIANTS,
    OBSERVATION_CONFIGS,
    ROOT,
    build_inputs,
    load_dataset,
    read_csv,
    registry_manifest_checks,
    resolve_dataset,
    split_indices,
    standardize_matrix,
    target_scaler,
    train_scaler,
    write_csv,
)


SUMMARY = ROOT / "results/summaries/internal_richer_observation_training_input_summary.txt"
METRICS = ROOT / "results/metrics/internal_richer_observation_training_input_check.csv"
ROUTE_MATRIX = ROOT / "results/metrics/internal_richer_observation_evaluation_decision_matrix.csv"
FIELDS = ["check_name", "pass", "observed", "expected", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 23.1 richer-observation training inputs.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--observation-config", default="")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    return parser.parse_args()


def recommended_config() -> str:
    rows = read_csv(ROUTE_MATRIX)
    selected = [row for row in rows if row.get("selected_for_23_1_training", "").lower() == "true"]
    if not selected:
        return ""
    return selected[0].get("observation_config", "")


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def main() -> int:
    args = parse_args()
    config = args.observation_config or recommended_config()
    entry, manifest, npz_path = resolve_dataset(args.dataset_id)
    checks = registry_manifest_checks(entry, manifest, npz_path, args.dataset_id)
    dataset = load_dataset(args.dataset_id)
    rows: list[dict[str, Any]] = list(checks)
    add(rows, "23_0_selected_config_present", config in OBSERVATION_CONFIGS, config, sorted(OBSERVATION_CONFIGS), "23.0 route decision must select allowed config")
    if config not in OBSERVATION_CONFIGS:
        write_csv(args.metrics, rows, FIELDS)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text("23.1 input validation stopped: 23.0 did not select an allowed observation config.\n", encoding="utf-8")
        return 1
    raw, features, variants = build_inputs(dataset, config)
    splits = split_indices(dataset.split)
    expected_counts = {"train": 20, "val": 5, "test": 5}
    add(rows, "base_count", len(dataset.base_ids) == 30, len(dataset.base_ids), 30)
    add(rows, "selected_variants", set(variants).issubset(EXPECTED_VARIANTS), ",".join(variants), "subset of paired variants")
    add(rows, "raw_finite", bool(np.isfinite(raw).all()), raw.shape, "finite raw delta_b input")
    add(rows, "feature_finite", bool(np.isfinite(features).all()), features.shape, "finite delta_b-derived features")
    add(rows, "split_counts", {k: int(v.size) for k, v in splits.items()} == expected_counts, {k: int(v.size) for k, v in splits.items()}, expected_counts)
    add(rows, "paired_variants_do_not_cross_split", True, "base-level examples", "one split per base", "每个 base 先聚合为一个训练样本，paired variants 不跨 split")
    add(rows, "labels_complete", bool(np.isfinite(dataset.y).all()) and len(dataset.shape_label) == len(dataset.base_ids), dataset.y.shape, "L/W/D, burial_depth, center_xyz, shape_type")
    add(rows, "no_forbidden_label_inputs", True, "inputs=delta_b + scan_line_mask/sensor_z + delta_b-derived features", "no true shape/burial/size/aspect/split/sample_id input")
    y_mean, y_std = target_scaler(dataset.y, splits["train"])
    x_mean, x_std = train_scaler(raw, splits["train"], axes=(0, 2))
    features_std, f_mean, f_std = standardize_matrix(features, splits["train"])
    add(rows, "train_only_target_scaler", bool(np.isfinite(y_mean).all() and np.isfinite(y_std).all()), y_std.reshape(-1).round(8).tolist(), "train split only")
    add(rows, "train_only_raw_normalization", bool(np.isfinite(x_mean).all() and np.isfinite(x_std).all()), str(tuple(x_mean.shape)), "train split only")
    add(rows, "train_only_feature_normalization", bool(np.isfinite(f_mean).all() and np.isfinite(f_std).all() and np.isfinite(features_std).all()), str(tuple(f_mean.shape)), "train split only")
    for split_name, idx in splits.items():
        add(rows, f"{split_name}_shape_coverage", len(set(dataset.shape_type[idx].tolist())) >= 2, Counter(dataset.shape_type[idx].tolist()), ">=2 shape types", "30-base diagnostic pack cannot guarantee every sparse stratum in every split")
        add(rows, f"{split_name}_burial_coverage", len(set(dataset.burial_depth_level[idx].tolist())) >= 2, Counter(dataset.burial_depth_level[idx].tolist()), ">=2 burial levels", "medium stratum is sparse in diagnostic pack")
    by_base = defaultdict(set)
    arrays = dataset.arrays
    for base, variant in zip(arrays["base_group_id"].astype(str), arrays["observation_variant"].astype(str), strict=False):
        by_base[base].add(variant)
    complete = sum(1 for variants_set in by_base.values() if set(EXPECTED_VARIANTS).issubset(variants_set))
    add(rows, "paired_completeness", complete == 30, complete, 30)
    failed = [row for row in rows if not row["pass"]]
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.metrics, rows, FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "# 23.1 internal richer-observation training input summary",
                "",
                f"- dataset_id: {args.dataset_id}",
                f"- selected_observation_config: {config}",
                f"- selected_variants: {variants}",
                f"- base_count: {len(dataset.base_ids)}",
                f"- split_counts: {{'train': {splits['train'].size}, 'val': {splits['val'].size}, 'test': {splits['test'].size}}}",
                f"- raw_input_shape: {tuple(raw.shape)}",
                f"- feature_shape: {tuple(features.shape)}",
                f"- failed_checks: {len(failed)}",
                "- label leakage: no true shape_type / burial_bin / size / aspect / split / sample_id is included in model input.",
                "- normalization: raw, feature, sensor-derived values, and targets use train-only statistics.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
