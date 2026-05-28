#!/usr/bin/env python
"""Recover/export the 20.94 A2 liftoff adapter inference artifact.

This is an artifact recovery stage: it replays the fixed 20.94 A2 protocol,
saves ignored checkpoint/prediction artifacts, and writes a tracked manifest.
It does not run COMSOL, generate data, modify NPZ, tune hyperparameters, or
update CURRENT_BASELINE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import load_true_3d_rbc_liftoff_aug_dataset as liftoff
import train_true_3d_rbc_liftoff_adapter_candidates as adapter


ROOT = liftoff.ROOT
DATASET_ID = liftoff.DATASET_ID
ADAPTER_NAME = "A2_latent_residual_adapter"
SEED = 2026
CHECKPOINT_DIR = ROOT / "checkpoints/true_3d_rbc_liftoff_adapter_artifacts"
CHECKPOINT_PATH = CHECKPOINT_DIR / "true_3d_rbc_liftoff_a2_adapter_seed2026.pt"
PREDICTION_PATH = CHECKPOINT_DIR / "true_3d_rbc_liftoff_a2_adapter_seed2026_predictions.npz"

PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_a2_adapter_artifact_recovery_preflight_summary.txt"
RECOVERY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_a2_adapter_artifact_recovery_summary.txt"
VERIFY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_a2_adapter_artifact_verification_summary.txt"
RECOVERY_METRICS = ROOT / "results/metrics/true_3d_rbc_a2_adapter_artifact_recovery_metrics.csv"
VERIFY_METRICS = ROOT / "results/metrics/true_3d_rbc_a2_adapter_artifact_verification.csv"
MANIFEST_PATH = ROOT / "results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json"

BASELINE_ARTIFACT_MANIFEST = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
LIFTOFF_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json"
REF_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_metrics.csv"
REF_FORMAL = ROOT / "results/metrics/true_3d_rbc_formal_liftoff_benchmark_metrics.csv"
REF_SELECTED = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_model_selected.csv"
BASELINE_REPLAY = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_baseline_replay_metrics.csv"

REFERENCE_FILES = [
    BASELINE_ARTIFACT_MANIFEST,
    LIFTOFF_MANIFEST,
    REF_METRICS,
    REF_FORMAL,
    REF_SELECTED,
    BASELINE_REPLAY,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def forbidden_staged_files() -> list[str]:
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=True)
    forbidden: list[str] = []
    for line in result.stdout.splitlines():
        lower = line.lower()
        if (
            lower.startswith("data/")
            or lower.startswith("notes/")
            or lower.endswith((".npz", ".pt", ".pth", ".png", ".mph"))
            or "__pycache__" in lower
            or line == "CURRENT_BASELINE.md"
            or line == "scripts/visualize_current_baseline.py"
        ):
            forbidden.append(line)
    return forbidden


def write_preflight(dataset: liftoff.True3DRBCLiftoffDataset) -> None:
    missing = [path for path in REFERENCE_FILES if not path.exists()]
    forbidden = forbidden_staged_files()
    baseline_dirty = subprocess.run(
        ["git", "diff", "--name-only", "--", "CURRENT_BASELINE.md"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    selected_rows = read_csv(REF_SELECTED) if REF_SELECTED.exists() else []
    selected_ok = bool(
        selected_rows
        and selected_rows[-1].get("selected_candidate") == ADAPTER_NAME
        and str(selected_rows[-1].get("selected_seed")) == str(SEED)
        and str(selected_rows[-1].get("eligible", "")).lower() == "true"
    )
    cfg_ok = any(cfg.name == ADAPTER_NAME and cfg.kind == "latent_residual" for cfg in adapter.candidate_configs(include_full_model=False))
    lines = [
        "20.96a true 3D RBC A2 liftoff adapter artifact recovery preflight",
        "",
        f"dataset_id: {dataset.dataset_id}",
        f"manifest_path: {LIFTOFF_MANIFEST.relative_to(ROOT)}",
        f"baseline_artifact_manifest: {BASELINE_ARTIFACT_MANIFEST.relative_to(ROOT)}",
        f"rows: {dataset.delta_b.shape[0]}",
        f"base_count: {len(set(dataset.base_sample_ids.tolist()))}",
        f"paired_liftoff_complete: {liftoff.paired_liftoff_complete(dataset)}",
        f"base_split_leakage: {liftoff.base_split_leakage(dataset)}",
        f"missing_reference_files: {[str(path.relative_to(ROOT)) for path in missing]}",
        f"selected_a2_reference_ok: {selected_ok}",
        f"a2_training_class_available: {cfg_ok}",
        f"forbidden_staged_files: {forbidden}",
        f"CURRENT_BASELINE_dirty: {bool(baseline_dirty)}",
        "COMSOL_run: false",
        "data_or_npz_write: false",
        "CURRENT_BASELINE_update: false",
        "protocol: fixed 20.94 A2 seed=2026, validation-only selection, test-final verification",
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    blockers = missing or forbidden or baseline_dirty or not selected_ok or not cfg_ok or liftoff.base_split_leakage(dataset)
    if blockers:
        raise RuntimeError("20.96a preflight blocked; see " + str(PREFLIGHT_SUMMARY))


def reference_value(rows: list[dict[str, str]], subset: str, metric: str) -> float:
    matches = [
        row
        for row in rows
        if row.get("candidate") == ADAPTER_NAME
        and str(row.get("seed")) == str(SEED)
        and row.get("split") == "test"
        and row.get("liftoff_subset") == subset
    ]
    if not matches:
        raise RuntimeError(f"missing reference {subset} {metric}")
    return float(matches[0][metric])


def train_fixed_a2(args: argparse.Namespace) -> tuple[
    liftoff.True3DRBCLiftoffDataset,
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, np.ndarray],
    adapter.CandidateConfig,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    dataset = liftoff.load_liftoff_dataset(args.dataset_id)
    stats = liftoff.train_normalization(dataset)
    baseline_rows = read_csv(BASELINE_REPLAY)
    _baseline_artifact, baseline_checkpoint, baseline_model = adapter.load_baseline_model()
    baseline = adapter.baseline_arrays(dataset, stats, baseline_checkpoint, baseline_model)
    cfg = next(cfg for cfg in adapter.candidate_configs(include_full_model=False) if cfg.name == ADAPTER_NAME)
    result = adapter.train_one(cfg, SEED, dataset, stats, baseline, baseline_rows, args)
    metric_rows, by_liftoff = adapter.evaluate_result(dataset, stats, result, selected=True)
    return dataset, stats, baseline_checkpoint, baseline, cfg, result, metric_rows, by_liftoff


def save_artifacts(
    dataset: liftoff.True3DRBCLiftoffDataset,
    stats: dict[str, np.ndarray],
    baseline_checkpoint: dict[str, Any],
    baseline: dict[str, np.ndarray],
    cfg: adapter.CandidateConfig,
    result: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    features = adapter.features_for(cfg, baseline, liftoff.normalize_sensor_z(dataset, stats))
    input_dim = int(features.shape[1]) if features is not None else None
    checkpoint = {
        "artifact_id": "true_3d_rbc_a2_liftoff_adapter_seed2026",
        "stage": "20.96a_a2_liftoff_adapter_artifact_recovery",
        "dataset_id": dataset.dataset_id,
        "adapter_type": ADAPTER_NAME,
        "seed": SEED,
        "best_epoch": int(result["best_epoch"]),
        "best_val_selection_metric": float(result["best_val_score"]),
        "candidate_config": asdict(cfg),
        "model_class": "ResidualAdapter",
        "input_dim": input_dim,
        "model_state_dict": result["model"].state_dict(),
        "normalization": {
            "x_mean": stats["x_mean"],
            "x_std": stats["x_std"],
            "y_mean": stats["y_mean"],
            "y_std": stats["y_std"],
            "sensor_z_mean": stats["sensor_z_mean"],
            "sensor_z_std": stats["sensor_z_std"],
            "fit_scope": "train_split_only",
        },
        "baseline_manifest": str(BASELINE_ARTIFACT_MANIFEST),
        "baseline_model_family": "20.85/20.77 frozen small Conv1D + MLP six-parameter head",
        "baseline_normalization": baseline_checkpoint.get("normalization", {}),
        "input_contract": {
            "allowed_model_input_fields": ["delta_b", "sensor_z_m", "frozen_baseline_latent", "frozen_baseline_prediction"],
            "forbidden_model_input_fields": [
                "rbc_params",
                "projected_mask_2d",
                "profile_depth_grid_m",
                "curvature_template",
                "depth_bin",
                "aspect_bin",
                "sample_id",
            ],
        },
        "training_protocol": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "split_policy": "base_sample_id_grouped_32_8_8",
            "validation_only_selection": True,
            "test_final_only": True,
            "hyperparameter_tuning": False,
        },
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    np.savez_compressed(
        PREDICTION_PATH,
        dataset_id=np.asarray([dataset.dataset_id]),
        adapter_type=np.asarray([ADAPTER_NAME]),
        seed=np.asarray([SEED], dtype=np.int64),
        best_epoch=np.asarray([int(result["best_epoch"])], dtype=np.int64),
        best_val_selection_metric=np.asarray([float(result["best_val_score"])], dtype=np.float64),
        pred_norm=result["pred_norm"].astype(np.float32),
        pred_params_m=result["pred_raw"].astype(np.float32),
        residual_norm=result["residual_norm"].astype(np.float32),
        baseline_pred_params_m=baseline["pred_raw"].astype(np.float32),
        sample_ids=dataset.sample_ids.astype(str),
        base_sample_ids=dataset.base_sample_ids.astype(str),
        split=dataset.split.astype(str),
        sensor_z_m=dataset.sensor_z_m.astype(np.float32),
        curvature_template=dataset.curvature_template.astype(str),
        depth_bin=dataset.depth_bin.astype(str),
        aspect_bin=dataset.aspect_bin.astype(str),
        size_bin=dataset.size_bin.astype(str),
    )
    refs = read_csv(REF_METRICS)
    selected_metrics = {
        "nominal_profile_depth_rmse_m": next(
            row for row in metric_rows if row["split"] == "test" and row["liftoff_subset"] == "nominal_0p008"
        )["profile_depth_rmse_m"],
        "non_nominal_profile_depth_rmse_m": next(
            row for row in metric_rows if row["split"] == "test" and row["liftoff_subset"] == "non_nominal"
        )["profile_depth_rmse_m"],
        "non_nominal_projected_mask_dice": next(
            row for row in metric_rows if row["split"] == "test" and row["liftoff_subset"] == "non_nominal"
        )["projected_mask_dice"],
        "reference_nominal_profile_depth_rmse_m": reference_value(refs, "nominal_0p008", "profile_depth_rmse_m"),
        "reference_non_nominal_profile_depth_rmse_m": reference_value(refs, "non_nominal", "profile_depth_rmse_m"),
        "reference_non_nominal_projected_mask_dice": reference_value(refs, "non_nominal", "projected_mask_dice"),
    }
    manifest = {
        "artifact_id": "true_3d_rbc_a2_liftoff_adapter_seed2026_inference_artifact",
        "stage": "20.96a_a2_liftoff_adapter_artifact_recovery",
        "dataset_id": dataset.dataset_id,
        "route": "true_3d_piao_style_liftoff_robustness",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "purpose": "Enable 20.96 liftoff-conditioned inference smoke without retraining.",
        "baseline_manifest": str(BASELINE_ARTIFACT_MANIFEST),
        "adapter_type": ADAPTER_NAME,
        "seed": SEED,
        "model_class": "ResidualAdapter",
        "model_config": asdict(cfg) | {"input_dim": input_dim},
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "prediction_artifact_path": str(PREDICTION_PATH),
        "prediction_artifact_sha256": sha256_file(PREDICTION_PATH),
        "checkpoint_committed": False,
        "prediction_artifact_committed": False,
        "artifact_paths_are_ignored": True,
        "normalization": {
            "sensor_z_mean": json_ready(stats["sensor_z_mean"]),
            "sensor_z_std": json_ready(stats["sensor_z_std"]),
            "y_mean_shape": list(stats["y_mean"].shape),
            "y_std_shape": list(stats["y_std"].shape),
            "fit_scope": "train_split_only",
        },
        "split_policy": {
            "split_by": "base_sample_id",
            "train_val_test_base_count": [32, 8, 8],
            "train_val_test_row_count": [128, 32, 32],
        },
        "selected_metrics": json_ready(selected_metrics),
        "input_contract": checkpoint["input_contract"],
        "inference_routing": {
            "nominal_sensor_z_m": 0.008,
            "nominal_tolerance_m": 0.0005,
            "default_nominal_route": "baseline",
            "default_non_nominal_route": "baseline_plus_adapter",
            "requires_sensor_z_m": True,
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def reload_and_verify(
    dataset: liftoff.True3DRBCLiftoffDataset,
    stats: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    cfg: adapter.CandidateConfig,
    original_result: dict[str, Any],
    metric_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    loaded = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = adapter.model_for(cfg, int(loaded["input_dim"]))
    model.load_state_dict(loaded["model_state_dict"])
    z_norm = liftoff.normalize_sensor_z(dataset, stats)
    features = adapter.features_for(cfg, baseline, z_norm)
    pred_norm, residual_norm = adapter.predict_candidate(model, cfg, liftoff.normalize_x(dataset, stats), z_norm, features, baseline["pred_norm"])
    max_pred_diff = float(np.max(np.abs(pred_norm - original_result["pred_norm"])))
    max_residual_diff = float(np.max(np.abs(residual_norm - original_result["residual_norm"])))
    reloaded_result = {
        "candidate": cfg.name,
        "seed": SEED,
        "description": cfg.description,
        "model": model,
        "best_epoch": loaded["best_epoch"],
        "best_val_score": loaded["best_val_selection_metric"],
        "pred_norm": pred_norm,
        "pred_raw": liftoff.denormalize_y(pred_norm, stats),
        "residual_norm": residual_norm,
        "epoch_rows": [],
    }
    reloaded_metrics, _by = adapter.evaluate_result(dataset, stats, reloaded_result, selected=True)
    refs = read_csv(REF_METRICS)
    checks: list[dict[str, Any]] = [
        {
            "check_name": "checkpoint_reloaded",
            "pass": True,
            "observed": str(CHECKPOINT_PATH),
            "reference": "",
            "abs_delta": "",
            "tolerance": "",
        },
        {
            "check_name": "prediction_norm_max_abs_diff_after_reload",
            "pass": max_pred_diff <= 1.0e-8,
            "observed": max_pred_diff,
            "reference": 0.0,
            "abs_delta": max_pred_diff,
            "tolerance": 1.0e-8,
        },
        {
            "check_name": "residual_norm_max_abs_diff_after_reload",
            "pass": max_residual_diff <= 1.0e-8,
            "observed": max_residual_diff,
            "reference": 0.0,
            "abs_delta": max_residual_diff,
            "tolerance": 1.0e-8,
        },
    ]
    for subset, metric, tolerance in [
        ("nominal_0p008", "profile_depth_rmse_m", 5.0e-6),
        ("non_nominal", "profile_depth_rmse_m", 5.0e-6),
        ("non_nominal", "projected_mask_dice", 1.0e-3),
    ]:
        cur = float(next(row for row in reloaded_metrics if row["split"] == "test" and row["liftoff_subset"] == subset)[metric])
        ref = reference_value(refs, subset, metric)
        checks.append(
            {
                "check_name": f"{subset}_{metric}_matches_20_94",
                "pass": abs(cur - ref) <= tolerance,
                "observed": cur,
                "reference": ref,
                "abs_delta": abs(cur - ref),
                "tolerance": tolerance,
            }
        )
    return checks, all(bool(row["pass"]) for row in checks)


def write_summaries(manifest: dict[str, Any], metric_rows: list[dict[str, Any]], verification: list[dict[str, Any]], verification_pass: bool) -> None:
    nom = next(row for row in metric_rows if row["split"] == "test" and row["liftoff_subset"] == "nominal_0p008")
    non = next(row for row in metric_rows if row["split"] == "test" and row["liftoff_subset"] == "non_nominal")
    RECOVERY_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    RECOVERY_SUMMARY.write_text(
        "\n".join(
            [
                "20.96a true 3D RBC A2 liftoff adapter artifact recovery",
                "",
                f"dataset_id: {DATASET_ID}",
                f"adapter: {ADAPTER_NAME}",
                f"seed: {SEED}",
                "COMSOL_run: false",
                "data_or_npz_write: false",
                "CURRENT_BASELINE_update: false",
                "protocol: fixed 20.94 A2 architecture/loss/validation selection; no hyperparameter tuning",
                f"checkpoint_path_ignored: {manifest['checkpoint_path']}",
                f"prediction_artifact_path_ignored: {manifest['prediction_artifact_path']}",
                f"manifest_path: {MANIFEST_PATH.relative_to(ROOT)}",
                f"test_nominal_profile_depth_rmse_m: {float(nom['profile_depth_rmse_m']):.9f}",
                f"test_non_nominal_profile_depth_rmse_m: {float(non['profile_depth_rmse_m']):.9f}",
                f"test_non_nominal_projected_mask_dice: {float(non['projected_mask_dice']):.6f}",
                f"test_non_nominal_LWD_MAE_mm: {float(non['L_mae_mm']):.3f}/{float(non['W_mae_mm']):.3f}/{float(non['D_mae_mm']):.3f}",
                f"test_non_nominal_wMAE_auxiliary: {float(non['wMAE_auxiliary']):.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    VERIFY_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    failed = [row for row in verification if not bool(row["pass"])]
    VERIFY_SUMMARY.write_text(
        "\n".join(
            [
                "20.96a true 3D RBC A2 adapter artifact verification",
                "",
                f"checkpoint_reloaded: {next(row for row in verification if row['check_name'] == 'checkpoint_reloaded')['pass']}",
                f"verification_pass: {verification_pass}",
                f"failed_checks: {failed}",
                "reference: 20.94/20.95 persisted A2 metrics",
                "mismatch_policy: stop if nominal/non-nominal RMSE or Dice exceeds tolerance",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = liftoff.load_liftoff_dataset(args.dataset_id)
    write_preflight(dataset)
    dataset, stats, baseline_checkpoint, baseline, cfg, result, metric_rows, _by = train_fixed_a2(args)
    manifest = save_artifacts(dataset, stats, baseline_checkpoint, baseline, cfg, result, metric_rows, args)
    verification, verification_pass = reload_and_verify(dataset, stats, baseline, cfg, result, metric_rows)
    write_csv(RECOVERY_METRICS, metric_rows)
    write_csv(VERIFY_METRICS, verification)
    write_summaries(manifest, metric_rows, verification, verification_pass)
    if not verification_pass:
        raise RuntimeError("A2 artifact verification failed; see " + str(VERIFY_METRICS))
    print(f"wrote {PREFLIGHT_SUMMARY}")
    print(f"wrote {RECOVERY_SUMMARY}")
    print(f"wrote {VERIFY_SUMMARY}")
    print(f"wrote {RECOVERY_METRICS}")
    print(f"wrote {VERIFY_METRICS}")
    print(f"wrote {MANIFEST_PATH}")
    print(f"checkpoint {CHECKPOINT_PATH}")
    print(f"prediction_artifact {PREDICTION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
