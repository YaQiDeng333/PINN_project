#!/usr/bin/env python
"""Recover/export 21.7 internal defect B2 inference artifact.

This script follows the fixed 21.7 protocol:
dataset_id -> registry/manifest gate -> train-only normalization ->
B2_feature_fusion_burial_head seed=2026 -> validation-only checkpoint
selection -> test-final verification. Checkpoint and prediction artifacts are
written under ignored checkpoints/ and are not intended for git staging.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from load_internal_defect_pilot_dataset import (
    ROOT,
    SHAPE_CLASSES,
    classification_metrics,
    denormalize_y,
    load_dataset,
    normalize_x,
    normalize_y,
    regression_metrics,
    split_indices,
    train_normalization,
    train_target_scaler,
    write_csv,
)
from train_internal_defect_burial_depth_candidates import (
    BurialDepthNet,
    CANDIDATE_CONFIGS,
    candidate_selection_score,
    predict,
    train_one_candidate,
)
from train_internal_defect_feature_baselines import extract_features, standardize_features


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
CANDIDATE = "B2_feature_fusion_burial_head"
SEED = 2026
EPOCHS = 300
BATCH_SIZE = 8
ARTIFACT_DIR = ROOT / "checkpoints/internal_defect_b2_artifacts"
CHECKPOINT_PATH = ARTIFACT_DIR / "internal_defect_b2_feature_fusion_seed2026.pt"
PREDICTION_PATH = ARTIFACT_DIR / "internal_defect_b2_feature_fusion_seed2026_predictions.npz"
MANIFEST_PATH = ROOT / "results/manifests/internal_defect_b2_inference_artifact_manifest.json"
SUMMARY_PATH = ROOT / "results/summaries/internal_defect_b2_artifact_recovery_summary.txt"
VERIFY_SUMMARY_PATH = ROOT / "results/summaries/internal_defect_b2_artifact_verification_summary.txt"
RECOVERY_METRICS = ROOT / "results/metrics/internal_defect_b2_artifact_recovery_metrics.csv"
VERIFY_METRICS = ROOT / "results/metrics/internal_defect_b2_artifact_verification.csv"
REF_SEED_METRICS = ROOT / "results/metrics/internal_defect_benchmark_rerun_b2_seed_summary.csv"


METRIC_FIELDS = [
    "candidate",
    "seed",
    "split",
    "sample_count",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_mae_mm",
    "shape_accuracy",
    "shape_macro_f1",
    "selection_score",
    "best_epoch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export internal defect B2 inference artifact.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifact files.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing artifact files: " + "; ".join(existing))


def split_metric_row(
    split_name: str,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    best_epoch: int,
) -> dict[str, Any]:
    reg = regression_metrics(y_true[idx], y_pred[idx], y_std.reshape(-1))
    cls = classification_metrics(shape_true[idx], shape_pred[idx])
    score = candidate_selection_score(reg, cls) if split_name == "val" else ""
    return {
        "candidate": CANDIDATE,
        "seed": SEED,
        "split": split_name,
        "sample_count": int(idx.size),
        **reg,
        **cls,
        "selection_score": score,
        "best_epoch": best_epoch,
    }


def reference_selected_row() -> dict[str, str]:
    for row in read_csv(REF_SEED_METRICS):
        if row.get("candidate") == CANDIDATE and row.get("selected_seed") == "True":
            return row
    raise RuntimeError("missing 21.7 selected B2 reference metrics")


def metric_diff_rows(metrics: list[dict[str, Any]], ref: dict[str, str], reload_max_pred_diff: float, reload_shape_diff_count: int) -> list[dict[str, Any]]:
    test = next(row for row in metrics if row["split"] == "test")
    checks = [
        ("test_total_normalized_mae", test["total_normalized_mae"], float(ref["test_total_normalized_mae"]), 1e-7),
        ("test_L_mae_mm", test["L_mae_mm"], float(ref["test_L_mae_mm"]), 1e-6),
        ("test_W_mae_mm", test["W_mae_mm"], float(ref["test_W_mae_mm"]), 1e-6),
        ("test_D_mae_mm", test["D_mae_mm"], float(ref["test_D_mae_mm"]), 1e-6),
        ("test_burial_depth_mae_mm", test["burial_depth_mae_mm"], float(ref["test_burial_depth_mae_mm"]), 1e-6),
        ("test_center_xyz_mae_mm", test["center_xyz_mae_mm"], float(ref["test_center_xyz_mae_mm"]), 1e-6),
        ("test_shape_accuracy", test["shape_accuracy"], float(ref["test_shape_accuracy"]), 1e-9),
        ("test_shape_macro_f1", test["shape_macro_f1"], float(ref["test_shape_macro_f1"]), 1e-9),
    ]
    rows: list[dict[str, Any]] = []
    for name, observed, expected, tolerance in checks:
        diff = abs(float(observed) - float(expected))
        rows.append(
            {
                "check_name": name,
                "pass": diff <= tolerance,
                "observed": observed,
                "expected": expected,
                "abs_diff": diff,
                "tolerance": tolerance,
                "notes": "21.7 selected B2 reference",
            }
        )
    rows.extend(
        [
            {
                "check_name": "checkpoint_reload_max_prediction_diff",
                "pass": reload_max_pred_diff <= 1e-7,
                "observed": reload_max_pred_diff,
                "expected": 0.0,
                "abs_diff": reload_max_pred_diff,
                "tolerance": 1e-7,
                "notes": "same model state reloaded from ignored checkpoint",
            },
            {
                "check_name": "checkpoint_reload_shape_diff_count",
                "pass": reload_shape_diff_count == 0,
                "observed": reload_shape_diff_count,
                "expected": 0,
                "abs_diff": reload_shape_diff_count,
                "tolerance": 0,
                "notes": "shape predictions should reload exactly",
            },
        ]
    )
    return rows


def main() -> int:
    args = parse_args()
    checkpoint_path = args.artifact_dir / CHECKPOINT_PATH.name
    prediction_path = args.artifact_dir / PREDICTION_PATH.name
    refuse_overwrite([checkpoint_path, prediction_path, args.manifest], args.overwrite)

    dataset = load_dataset(args.dataset_id)
    splits = split_indices(dataset.split)
    x_mean, x_std = train_normalization(dataset.x_channels, splits["train"])
    x = normalize_x(dataset.x_channels, x_mean, x_std)
    y = dataset.y_regression
    y_mean, y_std = train_target_scaler(y, splits["train"])
    y_norm = normalize_y(y, y_mean, y_std)
    feature_raw, feature_names = extract_features(dataset.delta_b)
    features, feature_mean, feature_std = standardize_features(feature_raw, splits["train"])

    result = train_one_candidate(CANDIDATE, args.seed, args.epochs, args.batch_size, x, y_norm, y, y_mean, y_std, dataset.shape_label, splits, features)
    pred = result["pred"]
    shape_pred = result["shape_pred"]
    if int(result["best_epoch"]) != 277:
        raise RuntimeError(f"best_epoch mismatch: {result['best_epoch']} != 277")

    metrics = [
        split_metric_row(split_name, idx, y, pred, y_std, dataset.shape_label, shape_pred, int(result["best_epoch"]))
        for split_name, idx in splits.items()
    ]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset_id": args.dataset_id,
            "candidate": CANDIDATE,
            "seed": args.seed,
            "best_epoch": int(result["best_epoch"]),
            "model_class": "BurialDepthNet",
            "model_config": {
                "feature_dim": int(features.shape[1]),
                "feature_fusion": True,
                "shape_conditioned": False,
                "candidate_config": CANDIDATE_CONFIGS[CANDIDATE],
            },
            "state_dict": result["model"].state_dict(),
            "normalization": {
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "feature_names": feature_names,
            },
            "shape_classes": SHAPE_CLASSES,
            "split_counts": {name: int(idx.size) for name, idx in splits.items()},
            "metrics": metrics,
        },
        checkpoint_path,
    )
    np.savez_compressed(
        prediction_path,
        dataset_id=np.asarray(args.dataset_id),
        candidate=np.asarray(CANDIDATE),
        seed=np.asarray(args.seed),
        sample_ids=dataset.sample_ids,
        split=dataset.split,
        y_true=y.astype(np.float32),
        y_pred=pred.astype(np.float32),
        param_names=np.asarray(["L_m", "W_m", "D_m", "burial_depth_m", "center_x_m", "center_y_m", "center_z_m"]),
        shape_true=dataset.shape_label.astype(np.int64),
        shape_pred=shape_pred.astype(np.int64),
        shape_classes=np.asarray(SHAPE_CLASSES),
        shape_type=dataset.shape_type,
        burial_depth_level=dataset.burial_depth_level,
        size_level=dataset.size_level,
        aspect_bin=dataset.aspect_bin,
        sensor_z_m=dataset.sensor_z_m.astype(np.float32),
        best_epoch=np.asarray(int(result["best_epoch"])),
    )

    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BurialDepthNet(feature_dim=int(features.shape[1]), feature_fusion=True, shape_conditioned=False)
    model.load_state_dict(reloaded["state_dict"])
    pred_norm_reload, shape_reload = predict(model, x, features)
    pred_reload = denormalize_y(pred_norm_reload, y_mean, y_std)
    reload_max_pred_diff = float(np.max(np.abs(pred_reload - pred)))
    reload_shape_diff_count = int(np.sum(shape_reload != shape_pred))
    verify_rows = metric_diff_rows(metrics, reference_selected_row(), reload_max_pred_diff, reload_shape_diff_count)
    failed = [row for row in verify_rows if not bool(row["pass"])]
    if failed:
        raise RuntimeError("artifact verification failed: " + json.dumps(failed, ensure_ascii=False))

    write_csv(RECOVERY_METRICS, metrics, METRIC_FIELDS)
    write_csv(VERIFY_METRICS, verify_rows, ["check_name", "pass", "observed", "expected", "abs_diff", "tolerance", "notes"])

    test = next(row for row in metrics if row["split"] == "test")
    manifest = {
        "artifact_id": "internal_defect_b2_feature_fusion_seed2026",
        "dataset_id": args.dataset_id,
        "dataset_manifest": str(ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json"),
        "model_type": CANDIDATE,
        "seed": args.seed,
        "best_epoch": int(result["best_epoch"]),
        "split": {name: int(idx.size) for name, idx in splits.items()},
        "input_policy": "delta_b/BxByBz plus delta_b-derived features only",
        "forbidden_inputs": ["labels", "shape_type", "burial_bin", "size_bin", "aspect_bin", "split", "sample_id"],
        "model_config": {
            "class": "BurialDepthNet",
            "feature_dim": int(features.shape[1]),
            "feature_fusion": True,
            "shape_conditioned": False,
            "candidate_config": CANDIDATE_CONFIGS[CANDIDATE],
        },
        "feature_config": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "standardization": "train split mean/std stored in checkpoint",
        },
        "normalization": {
            "x": "train split channel/time mean/std stored in checkpoint",
            "y": "train split target mean/std stored in checkpoint",
            "features": "train split feature mean/std stored in checkpoint",
        },
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "selected_metrics": {
            "test_total_normalized_mae": float(test["total_normalized_mae"]),
            "test_L_mae_mm": float(test["L_mae_mm"]),
            "test_W_mae_mm": float(test["W_mae_mm"]),
            "test_D_mae_mm": float(test["D_mae_mm"]),
            "test_burial_depth_mae_mm": float(test["burial_depth_mae_mm"]),
            "test_center_xyz_mae_mm": float(test["center_xyz_mae_mm"]),
            "test_shape_accuracy": float(test["shape_accuracy"]),
            "test_shape_macro_f1": float(test["shape_macro_f1"]),
        },
        "verification": {
            "checkpoint_reload_max_prediction_diff": reload_max_pred_diff,
            "checkpoint_reload_shape_diff_count": reload_shape_diff_count,
            "reference": "21.7/21.8 B2 benchmark candidate metrics",
            "status": "verified",
        },
        "git_policy": {
            "checkpoint_committed": False,
            "prediction_artifact_committed": False,
            "manifest_committable": True,
            "current_baseline_update": False,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        "\n".join(
            [
                "21.9 internal defect B2 inference artifact recovery",
                f"dataset_id: {args.dataset_id}",
                f"model: {CANDIDATE}",
                f"seed: {args.seed}",
                f"best_epoch: {result['best_epoch']}",
                f"checkpoint_path: {checkpoint_path}",
                f"prediction_path: {prediction_path}",
                f"manifest_path: {args.manifest}",
                "checkpoint_committed: false",
                "prediction_artifact_committed: false",
                "current_baseline_update: false",
                f"test_total_normalized_mae: {float(test['total_normalized_mae']):.6f}",
                f"test_LWD_mae_mm: {float(test['L_mae_mm']):.3f} / {float(test['W_mae_mm']):.3f} / {float(test['D_mae_mm']):.3f}",
                f"test_burial_depth_mae_mm: {float(test['burial_depth_mae_mm']):.3f}",
                f"test_center_xyz_mae_mm: {float(test['center_xyz_mae_mm']):.3f}",
                f"test_shape_accuracy_f1: {float(test['shape_accuracy']):.6f} / {float(test['shape_macro_f1']):.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    VERIFY_SUMMARY_PATH.write_text(
        "\n".join(
            [
                "21.9 internal defect B2 artifact verification",
                "checkpoint_reload: pass",
                f"reload_max_prediction_diff: {reload_max_pred_diff:.12g}",
                f"reload_shape_diff_count: {reload_shape_diff_count}",
                "reference_metric_match: pass",
                "matches_21_7_21_8: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
