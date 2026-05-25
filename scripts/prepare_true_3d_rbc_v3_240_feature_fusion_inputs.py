#!/usr/bin/env python
"""Validate train-safe feature matrices for Stage 20.81 feature fusion."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import V3_240_DATASET_ID, ROOT, load_dataset, split_indices, write_csv


FEATURES = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_features.csv"
FEATURE_QUALITY = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_quality.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_feature_fusion_input_summary.txt"
CHECKS = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_input_check.csv"
FEATURE_SETS = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_feature_sets.csv"

CHECK_FIELDS = ["check_name", "pass", "observed", "notes"]
FEATURE_SET_FIELDS = [
    "feature_set",
    "prefixes",
    "feature_count",
    "train_finite_after_scaling",
    "val_finite_after_scaling",
    "test_finite_after_scaling",
    "train_nan_before_impute",
    "all_nan_before_impute",
    "usable",
    "notes",
]


@dataclass(frozen=True)
class FeatureTransform:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).copy()
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = np.take(self.median, np.where(bad)[1])
        return ((arr - self.mean) / self.std).astype(np.float32)


def read_features(path: Path) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        feature_names = [name for name in fieldnames if name not in {"sample_id", "split"}]
        rows = list(reader)
    sample_ids = [row["sample_id"] for row in rows]
    splits = [row["split"] for row in rows]
    values = np.asarray(
        [[float(row[name]) if row[name] not in {"", "nan", "NaN"} else math.nan for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    return feature_names, values, sample_ids, splits


def feature_set_defs(feature_names: list[str]) -> dict[str, list[str]]:
    defs = {
        "FS_basic_physical": ["F0__", "F1__", "F2__"],
        "FS_basic_cross_axis": ["F0__", "F1__", "F2__", "F3__"],
        "FS_nls_optional": ["F0__", "F1__", "F2__", "F3__", "F4__"],
        "FS_curvature_focused": ["F0__", "F1__", "F2__", "F3__", "F5__"],
        "FS_F1F2_curvature_only": ["F1__", "F2__"],
    }
    return {name: prefixes for name, prefixes in defs.items() if any(any(f.startswith(prefix) for prefix in prefixes) for f in feature_names)}


def indices_for(feature_names: list[str], prefixes: list[str]) -> list[int]:
    return [idx for idx, name in enumerate(feature_names) if any(name.startswith(prefix) for prefix in prefixes)]


def fit_transformer(features: np.ndarray, train_idx: np.ndarray) -> FeatureTransform:
    train = np.asarray(features[train_idx], dtype=np.float64).copy()
    train[~np.isfinite(train)] = np.nan
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    bad = ~np.isfinite(train)
    if np.any(bad):
        train[bad] = np.take(median, np.where(bad)[1])
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1.0e-12, 1.0, std)
    return FeatureTransform(median=median.astype(np.float64), mean=mean.astype(np.float64), std=std.astype(np.float64))


def quality_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["feature_group"]: row for row in csv.DictReader(f)}


def add_check(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "notes": notes})


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset)
    feature_names, raw_features, feature_sample_ids, feature_split = read_features(args.features)
    checks: list[dict[str, Any]] = []
    expected_ids = [str(x) for x in dataset.sample_ids]
    expected_split = [str(x) for x in dataset.split]
    add_check(checks, "sample_count", raw_features.shape[0] == len(dataset.sample_ids) == 240, raw_features.shape[0])
    add_check(checks, "split_counts", {key: len(value) for key, value in splits.items()} == {"train": 162, "val": 39, "test": 39}, {key: len(value) for key, value in splits.items()})
    add_check(checks, "sample_id_order", feature_sample_ids == expected_ids, "matched" if feature_sample_ids == expected_ids else "mismatch")
    add_check(checks, "split_order", feature_split == expected_split, "matched" if feature_split == expected_split else "mismatch")
    add_check(checks, "feature_allowlist", all(name.startswith(("F0__", "F1__", "F2__", "F3__", "F4__", "F5__")) for name in feature_names), len(feature_names))
    add_check(checks, "no_label_columns", not any(name in feature_names for name in ("L_m", "W_m", "D_m", "wLD", "wWD", "wLW", "projected_mask_2d")), "checked")
    add_check(checks, "no_metadata_columns", not any(name in feature_names for name in ("curvature_template", "depth_bin", "aspect_bin", "size_bin", "dataset_id", "source_index")), "checked")

    qmap = quality_map(args.feature_quality)
    add_check(checks, "F4_nls_quality_recorded", "F4_nls_curve_fit" in qmap, qmap.get("F4_nls_curve_fit", {}).get("fit_success_rate", "missing"))
    add_check(checks, "F4_nls_success_rate", float(qmap.get("F4_nls_curve_fit", {}).get("fit_success_rate", "0")) >= 0.95, qmap.get("F4_nls_curve_fit", {}).get("fit_success_rate", "missing"))

    set_rows: list[dict[str, Any]] = []
    for set_name, prefixes in feature_set_defs(feature_names).items():
        idx = indices_for(feature_names, prefixes)
        selected = raw_features[:, idx]
        transform = fit_transformer(selected, splits["train"])
        scaled = transform.transform(selected)
        row = {
            "feature_set": set_name,
            "prefixes": ",".join(prefixes),
            "feature_count": len(idx),
            "train_finite_after_scaling": bool(np.isfinite(scaled[splits["train"]]).all()),
            "val_finite_after_scaling": bool(np.isfinite(scaled[splits["val"]]).all()),
            "test_finite_after_scaling": bool(np.isfinite(scaled[splits["test"]]).all()),
            "train_nan_before_impute": int(np.isnan(selected[splits["train"]]).sum()),
            "all_nan_before_impute": int(np.isnan(selected).sum()),
            "usable": bool(len(idx) > 0 and np.isfinite(scaled).all()),
            "notes": "imputer/scaler fit on train split only",
        }
        set_rows.append(row)
    add_check(checks, "FS_basic_physical_usable", any(r["feature_set"] == "FS_basic_physical" and r["usable"] for r in set_rows), "checked")
    add_check(checks, "feature_matrix_finite_after_train_scaling", all(bool(r["usable"]) for r in set_rows), "checked")

    write_csv(args.input_check, checks, CHECK_FIELDS)
    write_csv(args.feature_sets, set_rows, FEATURE_SET_FIELDS)
    failed = [row for row in checks if not bool(row["pass"])]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 feature-fusion input summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"feature_csv: {args.features}",
                f"feature_rows: {raw_features.shape[0]}",
                f"feature_columns: {raw_features.shape[1]}",
                "join_columns: sample_id and split are used only to verify ordering and split membership.",
                "model_input_allowlist: columns starting with F0__, F1__, F2__, F3__, F4__, or F5__ only.",
                "forbidden_model_inputs: sample_id, split, curvature_template, depth_bin, aspect_bin, size_bin, rbc_params, projected_mask_2d, profile_depth_grid_m, profile_depth_map_xy_m, profile_pose.",
                "train_only_preprocessing: feature median imputer, feature mean, and feature std are fit on train rows only.",
                f"feature_sets: {[(row['feature_set'], row['feature_count'], row['usable']) for row in set_rows]}",
                f"F4_nls_fit_success_rate: {qmap.get('F4_nls_curve_fit', {}).get('fit_success_rate', 'missing')}",
                f"input_gate_pass: {not failed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(f"feature fusion input checks failed: {failed}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--feature-quality", type=Path, default=FEATURE_QUALITY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--input-check", type=Path, default=CHECKS)
    parser.add_argument("--feature-sets", type=Path, default=FEATURE_SETS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
