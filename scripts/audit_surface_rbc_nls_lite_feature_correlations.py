#!/usr/bin/env python
"""Correlation audit for surface RBC NLS-lite features.

This script is diagnostic only. Labels are loaded from the explicit v3_240
dataset solely as correlation targets and are not written into the formal
feature matrix.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import PARAM_NAMES, ROOT, V3_240_DATASET_ID, load_dataset, write_csv


FEATURES = ROOT / "results/metrics/surface_rbc_nls_lite_features.csv"
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_lite_feature_correlation_summary.txt"
CORRELATIONS = ROOT / "results/metrics/surface_rbc_nls_lite_feature_correlations.csv"

EPS = 1.0e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 3:
        return 0.0
    aa = a[finite]
    bb = b[finite]
    if float(np.std(aa)) <= EPS or float(np.std(bb)) <= EPS:
        return 0.0
    return safe_float(np.corrcoef(aa, bb)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 3:
        return 0.0
    return pearson(rankdata(a[finite]), rankdata(b[finite]))


def compute_correlations(features: dict[str, np.ndarray], targets: dict[str, np.ndarray], scope: str = "all") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_name in sorted(features):
        feature_values = np.asarray(features[feature_name], dtype=np.float64)
        for target_name in sorted(targets):
            target_values = np.asarray(targets[target_name], dtype=np.float64)
            finite = np.isfinite(feature_values) & np.isfinite(target_values)
            pearson_r = pearson(feature_values, target_values)
            spearman_r = spearman(feature_values, target_values)
            rows.append(
                {
                    "scope": scope,
                    "feature": feature_name,
                    "target": target_name,
                    "finite_count": int(np.sum(finite)),
                    "pearson_r": pearson_r,
                    "spearman_r": spearman_r,
                    "abs_pearson_r": abs(pearson_r),
                    "abs_spearman_r": abs(spearman_r),
                    "label_usage": "diagnostic_target_only",
                }
            )
    return rows


def profile_targets(dataset: Any) -> dict[str, np.ndarray]:
    depth_map = np.asarray(dataset.profile_depth_map_xy_m, dtype=np.float64)
    depth_grid = np.asarray(dataset.profile_depth_grid_m, dtype=np.float64)
    mask = np.asarray(dataset.projected_mask_2d, dtype=np.float64)
    depth_abs = np.abs(depth_map.reshape(depth_map.shape[0], -1))
    grid_abs = np.abs(depth_grid.reshape(depth_grid.shape[0], -1))
    return {
        "profile_max_abs_depth_m": np.max(depth_abs, axis=1),
        "profile_mean_abs_depth_m": np.mean(depth_abs, axis=1),
        "profile_depth_energy": np.mean(depth_abs * depth_abs, axis=1),
        "profile_grid_max_abs_depth_m": np.max(grid_abs, axis=1),
        "profile_grid_mean_abs_depth_m": np.mean(grid_abs, axis=1),
        "projected_mask_area_px": np.sum(mask.reshape(mask.shape[0], -1) > 0.5, axis=1).astype(np.float64),
    }


def target_arrays(dataset: Any) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for i, name in enumerate(PARAM_NAMES):
        targets[name] = np.asarray(dataset.rbc_params[:, i], dtype=np.float64)
    targets.update(profile_targets(dataset))
    return targets


def feature_arrays(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    feature_names = [name for name in rows[0] if name.startswith("nlslite_")]
    features = {
        name: np.asarray([safe_float(row.get(name, "0")) for row in rows], dtype=np.float64)
        for name in feature_names
    }
    sample_ids = np.asarray([row["sample_id"] for row in rows]).astype(str)
    splits = np.asarray([row["split"] for row in rows]).astype(str)
    return feature_names, features, sample_ids, splits


def top_rows(rows: list[dict[str, Any]], scope: str, limit: int = 3) -> dict[str, list[dict[str, Any]]]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    targets = sorted({str(row["target"]) for row in rows if row["scope"] == scope})
    for target in targets:
        subset = [row for row in rows if row["scope"] == scope and row["target"] == target]
        ordered = sorted(subset, key=lambda row: max(float(row["abs_pearson_r"]), float(row["abs_spearman_r"])), reverse=True)
        by_target[target] = ordered[:limit]
    return by_target


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset_id)
    rows = read_csv(args.features)
    if not rows:
        raise RuntimeError(f"empty feature CSV: {args.features}")
    feature_names, features, sample_ids, splits = feature_arrays(rows)
    dataset_order = {str(sample_id): i for i, sample_id in enumerate(dataset.sample_ids)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in dataset_order]
    if missing:
        raise RuntimeError(f"feature sample_id not found in dataset: {missing[:5]}")
    order = np.asarray([dataset_order[str(sample_id)] for sample_id in sample_ids], dtype=np.int64)
    all_targets_raw = target_arrays(dataset)
    all_targets = {name: values[order] for name, values in all_targets_raw.items()}

    corr_rows: list[dict[str, Any]] = []
    corr_rows.extend(compute_correlations(features, all_targets, "all"))
    for split_name in ("train", "val", "test"):
        mask = splits == split_name
        split_features = {name: values[mask] for name, values in features.items()}
        split_targets = {name: values[mask] for name, values in all_targets.items()}
        corr_rows.extend(compute_correlations(split_features, split_targets, split_name))

    fields = [
        "scope",
        "feature",
        "target",
        "finite_count",
        "pearson_r",
        "spearman_r",
        "abs_pearson_r",
        "abs_spearman_r",
        "label_usage",
    ]
    write_csv(args.correlations, corr_rows, fields)

    top_all = top_rows(corr_rows, "all")
    key_targets = ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW", "profile_max_abs_depth_m", "projected_mask_area_px"]
    lines = [
        "24.0A surface RBC NLS-lite feature correlation audit summary",
        "",
        f"dataset_id: {dataset.dataset_id}",
        "scope: diagnostic correlation audit only; no training and no model selection.",
        "label_usage: labels/profile metrics are used only as correlation targets and are not written into the formal feature matrix.",
        f"feature_csv: {args.features}",
        f"feature_count: {len(feature_names)}",
        f"sample_count: {len(rows)}",
        f"correlation_rows: {len(corr_rows)}",
        "",
        "top_all_scope_correlations:",
    ]
    for target in key_targets:
        if target not in top_all:
            continue
        best = top_all[target][0]
        lines.append(
            f"- {target}: {best['feature']} pearson={float(best['pearson_r']):.6f} spearman={float(best['spearman_r']):.6f}"
        )
    lines.extend(
        [
            "",
            "boundary: formal feature CSV contains only sample_id, split metadata, and nlslite_* delta_b-derived features; target labels are loaded separately for this audit.",
            f"correlations_csv: {args.correlations}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--correlations", type=Path, default=CORRELATIONS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
