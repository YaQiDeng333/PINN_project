#!/usr/bin/env python
"""Build in-memory inputs for surface RBC Conv1D + NLS-lite fusion.

This loader reads the v3_240 RBC pack only through COMSOL_DATA_REGISTRY.md and
its manifest, then joins the existing 24.0A nlslite_* feature CSV by sample_id.
It writes only input-check summaries; no NPZ or data artifact is generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_true_3d_rbc_pilot_dataset import (
    ROOT,
    V3_240_DATASET_ID,
    check_no_overwrite,
    load_dataset,
    normalize_x,
    normalize_y,
    split_indices,
    train_normalization,
    write_csv,
)
from train_surface_rbc_piao_style_feature_baseline import (
    FEATURE_CSV,
    FEATURE_MANIFEST,
    QUALITY_CSV,
    fit_feature_scaler,
    load_nlslite_feature_matrix,
    read_quality,
    transform_features,
    validate_feature_manifest,
)


PREFLIGHT_SUMMARY = ROOT / "results/summaries/surface_rbc_nls_feature_fusion_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/surface_rbc_nls_feature_fusion_input_summary.txt"
INPUT_CHECK = ROOT / "results/metrics/surface_rbc_nls_feature_fusion_input_check.csv"
BASELINE_METRICS = ROOT / "results/metrics/surface_rbc_piao_style_feature_baseline_metrics.csv"
PROFILE_METRICS = ROOT / "results/metrics/surface_rbc_piao_style_feature_profile_metrics.csv"
FORMAL_REFERENCE = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_comparison_matrix.csv"

CHECK_FIELDS = ["check_name", "pass", "observed", "expected", "notes"]


@dataclass(frozen=True)
class FusionInputs:
    dataset: Any
    splits: dict[str, np.ndarray]
    stats: dict[str, np.ndarray]
    x_norm: np.ndarray
    y_norm: np.ndarray
    feature_raw: np.ndarray
    feature_norm: np.ndarray
    feature_names: list[str]
    feature_scaler: dict[str, np.ndarray]
    feature_manifest: dict[str, Any]
    quality: dict[str, str]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def git_status_lines() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return [f"git_status_error: {type(exc).__name__}: {exc}"]
    return [line for line in completed.stdout.splitlines() if line.strip()]


def feature_csv_columns(path: Path) -> tuple[list[str], list[str]]:
    rows = read_csv_rows(path)
    if not rows:
        return [], []
    fields = list(rows[0].keys())
    metadata = [name for name in fields if not name.startswith("nlslite_")]
    features = [name for name in fields if name.startswith("nlslite_")]
    return metadata, features


def preflight_rows(dataset_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any, notes: str = "") -> None:
        rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})

    required_paths = [FEATURE_MANIFEST, FEATURE_CSV, QUALITY_CSV, BASELINE_METRICS, PROFILE_METRICS, FORMAL_REFERENCE]
    for path in required_paths:
        add(f"required_path::{path.name}", path.exists(), str(path), "exists", "")

    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8")) if FEATURE_MANIFEST.exists() else {}
    add("dataset_id", manifest.get("dataset_id") == dataset_id, manifest.get("dataset_id"), dataset_id, "24.0A manifest gate")
    add("feature_manifest_piao_nls_lite", manifest.get("piao_nls_lite") is True, manifest.get("piao_nls_lite"), True, "")
    add("feature_manifest_exact_piao_false", manifest.get("exact_piao_nls") is False, manifest.get("exact_piao_nls"), False, "")
    add("feature_manifest_no_training", manifest.get("training_run") is False, manifest.get("training_run"), False, "")
    add("feature_manifest_no_comsol", manifest.get("COMSOL_run") is False, manifest.get("COMSOL_run"), False, "")
    add("feature_manifest_no_npz_write", manifest.get("data_or_NPZ_modified") is False, manifest.get("data_or_NPZ_modified"), False, "")
    add("feature_manifest_no_baseline_update", manifest.get("CURRENT_BASELINE_update") is False, manifest.get("CURRENT_BASELINE_update"), False, "")

    if FEATURE_CSV.exists():
        metadata, features = feature_csv_columns(FEATURE_CSV)
        add("feature_csv_metadata_columns", metadata == ["sample_id", "split"], metadata, ["sample_id", "split"], "metadata not model input")
        add("feature_csv_nlslite_prefix_only", bool(features) and all(name.startswith("nlslite_") for name in features), len(features), "all feature columns start nlslite_", "")
        forbidden = {"L_m", "W_m", "D_m", "wLD", "wWD", "wLW", "profile", "mask", "projected_mask_2d", "curvature_template", "depth_bin", "aspect_bin", "size_bin"}
        leaked = sorted(forbidden.intersection(features))
        add("feature_csv_no_label_leakage", not leaked, leaked, [], "target/profile/group labels are not feature inputs")

    if FORMAL_REFERENCE.exists():
        reference_rows = read_csv_rows(FORMAL_REFERENCE)
        candidate_ids = {row.get("candidate_id", "") for row in reference_rows}
        add("reference_20_85_present", "20.85_formal_rerun_20.77_protocol" in candidate_ids, sorted(candidate_ids), "20.85_formal_rerun_20.77_protocol", "")
        add("reference_20_77_present", "20.77_original_candidate" in candidate_ids, sorted(candidate_ids), "20.77_original_candidate", "")
        add("reference_20_81_present", "20.81_feature_fusion" in candidate_ids, sorted(candidate_ids), "20.81_feature_fusion", "")

    status = git_status_lines()
    forbidden_terms = (
        "data/",
        "data\\",
        ".npz",
        ".png",
        ".pt",
        ".pth",
        ".ckpt",
        "CURRENT_BASELINE.md",
        "scripts/visualize_current_baseline.py",
        "scripts\\visualize_current_baseline.py",
    )
    forbidden_status = [line for line in status if any(term in line for term in forbidden_terms)]
    add("forbidden_artifacts_in_git_status", not forbidden_status, forbidden_status, [], "preflight status scan; allowed 24.2 files may be untracked")
    return rows


def write_preflight_summary(path: Path, rows: list[dict[str, Any]], dataset_id: str) -> None:
    failed = [row for row in rows if not row["pass"]]
    lines = [
        "surface_rbc_nls_feature_fusion_preflight_summary",
        "stage: 24.2",
        "",
        f"dataset_id: {dataset_id}",
        "scope: surface RBC NLS-lite feature-fusion diagnostic; not a baseline update.",
        "COMSOL_run: false",
        "data_or_NPZ_modified: false",
        "CURRENT_BASELINE_update: false",
        "latest_newest_npz_scan: false",
        f"required_feature_manifest: {FEATURE_MANIFEST}",
        f"required_feature_csv: {FEATURE_CSV}",
        f"required_24_1_metrics: {BASELINE_METRICS}",
        f"required_24_1_profile_metrics: {PROFILE_METRICS}",
        f"required_reference_matrix: {FORMAL_REFERENCE}",
        f"checks_total: {len(rows)}",
        f"checks_failed: {len(failed)}",
        "",
        "failed_checks:",
    ]
    lines.extend([f"- {row['check_name']}: observed={row['observed']} expected={row['expected']}" for row in failed] or ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_inputs(dataset_id: str = V3_240_DATASET_ID) -> FusionInputs:
    feature_manifest = validate_feature_manifest(FEATURE_MANIFEST, dataset_id)
    quality = read_quality(QUALITY_CSV)
    dataset = load_dataset(dataset_id)
    splits = split_indices(dataset)
    stats = train_normalization(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    feature_raw, feature_names = load_nlslite_feature_matrix(dataset, FEATURE_CSV)
    if int(feature_manifest.get("feature_count", -1)) != len(feature_names):
        raise RuntimeError(
            f"NLS-lite feature count mismatch: manifest={feature_manifest.get('feature_count')} csv={len(feature_names)}"
        )
    feature_scaler = fit_feature_scaler(feature_raw, splits["train"])
    feature_norm = transform_features(feature_raw, feature_scaler)
    if not np.isfinite(x_norm).all():
        raise RuntimeError("delta_b normalized channels contain non-finite values")
    if not np.isfinite(feature_norm).all():
        raise RuntimeError("NLS-lite normalized feature matrix contains non-finite values")
    if not np.isfinite(y_norm).all():
        raise RuntimeError("normalized RBC targets contain non-finite values")
    return FusionInputs(
        dataset=dataset,
        splits=splits,
        stats=stats,
        x_norm=x_norm,
        y_norm=y_norm,
        feature_raw=feature_raw,
        feature_norm=feature_norm,
        feature_names=feature_names,
        feature_scaler=feature_scaler,
        feature_manifest=feature_manifest,
        quality=quality,
    )


def input_check_rows(inputs: FusionInputs) -> list[dict[str, Any]]:
    dataset = inputs.dataset
    split_counts = {name: int(len(idx)) for name, idx in inputs.splits.items()}
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any, notes: str = "") -> None:
        rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})

    add("dataset_id", dataset.dataset_id == V3_240_DATASET_ID, dataset.dataset_id, V3_240_DATASET_ID, "explicit dataset_id")
    add("delta_b_shape", tuple(dataset.delta_b.shape) == (240, 3, 3, 201), tuple(dataset.delta_b.shape), "(240, 3, 3, 201)", "")
    add("axis_order", dataset.axis_names == ["Bx", "By", "Bz"], dataset.axis_names, "[Bx, By, Bz]", "")
    add("split_counts", split_counts == {"train": 162, "val": 39, "test": 39}, split_counts, "{train:162,val:39,test:39}", "")
    add("nlslite_feature_count", len(inputs.feature_names) == 291, len(inputs.feature_names), 291, "")
    add("nlslite_prefix_only", all(name.startswith("nlslite_") for name in inputs.feature_names), "all nlslite_*", "all nlslite_*", "")
    add("sample_id_join", len(inputs.feature_raw) == len(dataset.sample_ids), len(inputs.feature_raw), len(dataset.sample_ids), "sample_id used only for join")
    add("x_norm_finite", bool(np.isfinite(inputs.x_norm).all()), float(np.isfinite(inputs.x_norm).mean()), 1.0, "train-only delta_b normalization")
    add("feature_norm_finite", bool(np.isfinite(inputs.feature_norm).all()), float(np.isfinite(inputs.feature_norm).mean()), 1.0, "train-only NLS feature scaler")
    add("target_norm_finite", bool(np.isfinite(inputs.y_norm).all()), float(np.isfinite(inputs.y_norm).mean()), 1.0, "targets for supervised training only")
    add("feature_train_mean_abs_max", True, float(np.max(np.abs(inputs.feature_norm[inputs.splits["train"]].mean(axis=0)))), "near 0", "diagnostic only")
    add("feature_train_std_mean", True, float(np.mean(inputs.feature_norm[inputs.splits["train"]].std(axis=0))), "near 1", "diagnostic only")
    add("fit_success_rate_24_0A", float(inputs.quality.get("fit_success_rate", "nan")) >= 0.95, inputs.quality.get("fit_success_rate"), ">=0.95", "")
    add("fallback_rate_24_0A", float(inputs.quality.get("fallback_rate", "nan")) <= 0.05, inputs.quality.get("fallback_rate"), "<=0.05", "")
    add("COMSOL_run", True, False, False, "loader never calls COMSOL")
    add("NPZ_write", True, False, False, "loader writes no data/NPZ artifact")
    return rows


def write_summary(path: Path, inputs: FusionInputs) -> None:
    dataset = inputs.dataset
    split_counts = {name: int(len(idx)) for name, idx in inputs.splits.items()}
    lines = [
        "surface_rbc_nls_feature_fusion_input_summary",
        "stage: 24.2",
        "",
        f"dataset_id: {dataset.dataset_id}",
        f"source_npz_from_manifest: {dataset.npz_path}",
        "latest_newest_npz_scan: false",
        "COMSOL_run: false",
        "data_or_NPZ_modified: false",
        "CURRENT_BASELINE_update: false",
        f"delta_b_shape: {tuple(dataset.delta_b.shape)}",
        f"axis_order: {dataset.axis_names}",
        f"split_counts: {split_counts}",
        f"nlslite_feature_manifest: {FEATURE_MANIFEST}",
        f"nlslite_feature_csv: {FEATURE_CSV}",
        f"nlslite_quality_csv: {QUALITY_CSV}",
        f"nlslite_feature_count: {len(inputs.feature_names)}",
        "model_input_delta_b: normalized delta_b/BxByBz channels only",
        "model_input_features: train-scaled nlslite_* only",
        "sample_id_use: join/reporting only",
        "split_use: train/val/test partition only",
        "label_use: supervised target and diagnostics only",
        f"x_norm_shape: {tuple(inputs.x_norm.shape)}",
        f"feature_norm_shape: {tuple(inputs.feature_norm.shape)}",
        f"y_norm_shape: {tuple(inputs.y_norm.shape)}",
        f"feature_norm_finite_fraction: {float(np.isfinite(inputs.feature_norm).mean()):.6f}",
        f"x_norm_finite_fraction: {float(np.isfinite(inputs.x_norm).mean()):.6f}",
        f"fit_success_rate_24_0A: {inputs.quality.get('fit_success_rate')}",
        f"fallback_rate_24_0A: {inputs.quality.get('fallback_rate')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.preflight_summary, args.summary, args.input_check], args.overwrite)
    preflight = preflight_rows(args.dataset_id)
    write_preflight_summary(args.preflight_summary, preflight, args.dataset_id)
    failed = [row for row in preflight if not row["pass"]]
    if failed:
        raise RuntimeError("surface RBC NLS feature-fusion preflight failed: " + json.dumps(failed, ensure_ascii=False))
    inputs = build_inputs(args.dataset_id)
    write_csv(args.input_check, input_check_rows(inputs), CHECK_FIELDS)
    write_summary(args.summary, inputs)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--preflight-summary", type=Path, default=PREFLIGHT_SUMMARY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--input-check", type=Path, default=INPUT_CHECK)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
