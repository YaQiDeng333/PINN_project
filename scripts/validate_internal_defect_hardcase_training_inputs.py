#!/usr/bin/env python
"""22.3 hard-case augmented internal defect training input gate."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from internal_defect_hardcase_utils import DATASET_ID, prepare_dataset
from load_internal_defect_pilot_dataset import ROOT, normalize_x, train_normalization, write_csv
from train_internal_defect_feature_baselines import extract_features


SUMMARY = ROOT / "results/summaries/internal_defect_hardcase_training_input_summary.txt"
CHECKS = ROOT / "results/metrics/internal_defect_hardcase_training_input_check.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 22.3 hard-case training inputs.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--checks", type=Path, default=CHECKS)
    return parser.parse_args()


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def git_lines(args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def forbidden_staged() -> list[str]:
    staged = git_lines(["diff", "--cached", "--name-only"])
    forbidden: list[str] = []
    for path in staged:
        p = path.replace("\\", "/")
        if p.startswith("data/") or p.startswith("checkpoints/") or p.startswith("results/previews/") or p.startswith("notes/"):
            forbidden.append(path)
        elif p.endswith((".npz", ".pt", ".pth", ".mph", ".png")):
            forbidden.append(path)
        elif p in {"CURRENT_BASELINE.md", "scripts/visualize_current_baseline.py"}:
            forbidden.append(path)
    return forbidden


def main() -> int:
    args = parse_args()
    prepared = prepare_dataset(args.dataset_id)
    dataset = prepared["dataset"]
    splits = prepared["splits"]
    checks: list[dict[str, Any]] = []
    add(checks, "dataset_id", dataset.dataset_id == DATASET_ID, dataset.dataset_id, DATASET_ID)
    add(checks, "n_samples", dataset.delta_b.shape[0] == 360, dataset.delta_b.shape[0], 360)
    add(checks, "split_counts", {k: int(v.size) for k, v in splits.items()} == {"train": 240, "val": 60, "test": 60}, {k: int(v.size) for k, v in splits.items()}, "train/val/test=240/60/60")
    add(checks, "axis_names", dataset.axis_names == ["Bx", "By", "Bz"], dataset.axis_names, ["Bx", "By", "Bz"])
    add(checks, "delta_b_shape", tuple(dataset.delta_b.shape[1:]) == (3, 3, 201), tuple(dataset.delta_b.shape), "(N,3,3,201)")
    add(checks, "delta_b_finite", bool(np.isfinite(dataset.delta_b).all()), bool(np.isfinite(dataset.delta_b).all()), True)
    delta_err = float(np.max(np.abs(dataset.delta_b - (dataset.b_defect - dataset.b_no_defect))))
    add(checks, "delta_check", delta_err <= 1e-7, delta_err, "<=1e-7")
    add(checks, "labels_complete", dataset.y_regression.shape == (360, 7), dataset.y_regression.shape, "(360,7)")
    add(checks, "shape_labels_complete", set(dataset.shape_type.tolist()) == {"internal_sphere", "internal_ellipsoid", "internal_cuboid"}, sorted(set(dataset.shape_type.tolist())), "3 internal shape classes")
    add(checks, "row_origin_available", set(dataset.row_origin.tolist()) == {"source_v2_240", "hardcase_topup_v1"}, sorted(set(dataset.row_origin.tolist())), "source/top-up flags")
    add(checks, "model_x_input_shape", dataset.x_channels.shape == (360, 9, 201), dataset.x_channels.shape, "(360,9,201)")
    add(checks, "model_x_derived_from_delta_b", bool(np.array_equal(dataset.x_channels, dataset.delta_b.reshape(360, 9, 201))), "delta_b.reshape(360,9,201)", "x_channels")
    feature_raw, feature_names = extract_features(dataset.delta_b)
    add(checks, "feature_input_derived_from_delta_b", feature_raw.shape == (360, 110) and len(feature_names) == 110, f"shape={feature_raw.shape}; names={len(feature_names)}", "delta_b-derived 110 features")
    add(
        checks,
        "hardcase_flags_not_model_inputs",
        dataset.x_channels.shape[1:] == (9, 201) and feature_raw.shape[1] == 110,
        "hardcase/source flags are separate metadata arrays; H2 may use train split row_origin only as sample weight",
        "not concatenated into x/features",
    )
    for split, idx in splits.items():
        add(checks, f"{split}_shape_coverage", {"internal_sphere", "internal_ellipsoid", "internal_cuboid"}.issubset(set(dataset.shape_type[idx].tolist())), sorted(set(dataset.shape_type[idx].tolist())), "all shapes")
        add(checks, f"{split}_burial_coverage", {"shallow", "medium", "deep", "deep_plus"}.issubset(set(dataset.burial_depth_level[idx].tolist())), sorted(set(dataset.burial_depth_level[idx].tolist())), "all burial levels")
        add(checks, f"{split}_size_coverage", {"small", "medium", "large"}.issubset(set(dataset.size_level[idx].tolist())), sorted(set(dataset.size_level[idx].tolist())), "all sizes")
        add(checks, f"{split}_aspect_coverage", {"compact", "elongated_x", "elongated_y"}.issubset(set(dataset.aspect_bin[idx].tolist())), sorted(set(dataset.aspect_bin[idx].tolist())), "all aspects")
    x_mean, x_std = train_normalization(dataset.x_channels, splits["train"])
    x_norm = normalize_x(dataset.x_channels, x_mean, x_std)
    add(checks, "train_only_normalization_finite", bool(np.isfinite(x_norm).all()), bool(np.isfinite(x_norm).all()), True)
    forbidden = forbidden_staged()
    add(checks, "no_forbidden_staged_artifacts", not forbidden, ",".join(forbidden) or "none", "none")

    write_csv(args.checks, checks, ["check_name", "pass", "observed", "expected", "notes"])
    failed = [row for row in checks if not row["pass"]]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "22.3 hard-case augmented internal defect training input summary",
        f"dataset_id: {dataset.dataset_id}",
        f"status: {'pass' if not failed else 'blocked'}",
        f"N: {dataset.delta_b.shape[0]}",
        f"split: { {k: int(v.size) for k, v in splits.items()} }",
        "model_input_policy: delta_b/BxByBz plus delta_b-derived features only",
        "grouping_only_metadata: row_origin, hardcase_target_id, shape/burial/size/aspect are not model inputs",
        "current_baseline_update: false",
        "comsol_run: false",
        "data_npz_mutation: false",
    ]
    if failed:
        lines.append("failed_checks:")
        lines.extend(f"- {row['check_name']}: observed={row['observed']} expected={row['expected']}" for row in failed)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
