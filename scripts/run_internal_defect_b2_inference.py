#!/usr/bin/env python
"""Replay internal B2 inference from the 21.9 artifact manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from load_internal_defect_pilot_dataset import ROOT, SHAPE_CLASSES, load_dataset, write_csv


DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
ARTIFACT_MANIFEST = ROOT / "results/manifests/internal_defect_b2_inference_artifact_manifest.json"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_b2_inference_replay_summary.txt"
METRICS_OUT = ROOT / "results/metrics/internal_defect_b2_inference_replay_metrics.csv"


FIELDS = [
    "sample_id",
    "split",
    "true_shape_type",
    "pred_shape_type",
    "shape_correct",
    "true_L_mm",
    "pred_L_mm",
    "L_error_mm",
    "true_W_mm",
    "pred_W_mm",
    "W_error_mm",
    "true_D_mm",
    "pred_D_mm",
    "D_error_mm",
    "true_burial_depth_mm",
    "pred_burial_depth_mm",
    "burial_depth_error_mm",
    "true_center_x_mm",
    "pred_center_x_mm",
    "center_x_error_mm",
    "true_center_y_mm",
    "pred_center_y_mm",
    "center_y_error_mm",
    "true_center_z_mm",
    "pred_center_z_mm",
    "center_z_error_mm",
    "center_xyz_error_mm",
    "total_abs_normalized_error",
    "burial_depth_level",
    "size_level",
    "aspect_bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay B2 internal defect inference artifact.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--artifact-manifest", type=Path, default=ARTIFACT_MANIFEST)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mm(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * 1000.0


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.artifact_manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != args.dataset_id:
        raise RuntimeError("artifact manifest dataset_id mismatch")
    if manifest.get("model_type") != "B2_feature_fusion_burial_head":
        raise RuntimeError("artifact manifest model_type mismatch")
    prediction_path = Path(manifest["prediction_path"])
    checkpoint_path = Path(manifest["checkpoint_path"])
    if not prediction_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("B2 checkpoint or prediction artifact missing")
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise RuntimeError("prediction artifact sha256 mismatch")
    if sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise RuntimeError("checkpoint sha256 mismatch")

    dataset = load_dataset(args.dataset_id)
    with np.load(prediction_path, allow_pickle=True) as pred_npz:
        y_true = np.asarray(pred_npz["y_true"], dtype=np.float32)
        y_pred = np.asarray(pred_npz["y_pred"], dtype=np.float32)
        shape_true = np.asarray(pred_npz["shape_true"], dtype=np.int64)
        shape_pred = np.asarray(pred_npz["shape_pred"], dtype=np.int64)
        sample_ids = np.asarray(pred_npz["sample_ids"]).astype(str)
    if not np.array_equal(sample_ids, dataset.sample_ids):
        raise RuntimeError("prediction artifact sample_ids do not match dataset")
    if y_true.shape != y_pred.shape or y_true.shape[1] != 7:
        raise RuntimeError("prediction artifact shape mismatch")

    train_idx = np.where(dataset.split == "train")[0]
    y_std = dataset.y_regression[train_idx].std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    rows: list[dict[str, Any]] = []
    for i, sample_id in enumerate(sample_ids):
        err = np.abs(y_pred[i] - y_true[i])
        center_err = float(np.linalg.norm(err[4:7]) * 1000.0)
        total_norm = float(np.mean(err / y_std))
        rows.append(
            {
                "sample_id": sample_id,
                "split": dataset.split[i],
                "true_shape_type": SHAPE_CLASSES[int(shape_true[i])],
                "pred_shape_type": SHAPE_CLASSES[int(shape_pred[i])],
                "shape_correct": bool(shape_true[i] == shape_pred[i]),
                "true_L_mm": float(y_true[i, 0] * 1000.0),
                "pred_L_mm": float(y_pred[i, 0] * 1000.0),
                "L_error_mm": float(err[0] * 1000.0),
                "true_W_mm": float(y_true[i, 1] * 1000.0),
                "pred_W_mm": float(y_pred[i, 1] * 1000.0),
                "W_error_mm": float(err[1] * 1000.0),
                "true_D_mm": float(y_true[i, 2] * 1000.0),
                "pred_D_mm": float(y_pred[i, 2] * 1000.0),
                "D_error_mm": float(err[2] * 1000.0),
                "true_burial_depth_mm": float(y_true[i, 3] * 1000.0),
                "pred_burial_depth_mm": float(y_pred[i, 3] * 1000.0),
                "burial_depth_error_mm": float(err[3] * 1000.0),
                "true_center_x_mm": float(y_true[i, 4] * 1000.0),
                "pred_center_x_mm": float(y_pred[i, 4] * 1000.0),
                "center_x_error_mm": float(err[4] * 1000.0),
                "true_center_y_mm": float(y_true[i, 5] * 1000.0),
                "pred_center_y_mm": float(y_pred[i, 5] * 1000.0),
                "center_y_error_mm": float(err[5] * 1000.0),
                "true_center_z_mm": float(y_true[i, 6] * 1000.0),
                "pred_center_z_mm": float(y_pred[i, 6] * 1000.0),
                "center_z_error_mm": float(err[6] * 1000.0),
                "center_xyz_error_mm": center_err,
                "total_abs_normalized_error": total_norm,
                "burial_depth_level": dataset.burial_depth_level[i],
                "size_level": dataset.size_level[i],
                "aspect_bin": dataset.aspect_bin[i],
            }
        )
    write_csv(METRICS_OUT, rows, FIELDS)

    test_rows = [row for row in rows if row["split"] == "test"]
    mean_total = float(np.mean([row["total_abs_normalized_error"] for row in test_rows]))
    mean_burial = float(np.mean([row["burial_depth_error_mm"] for row in test_rows]))
    mean_center = float(np.mean([row["center_xyz_error_mm"] for row in test_rows]))
    shape_acc = float(np.mean([bool(row["shape_correct"]) for row in test_rows]))
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(
        "\n".join(
            [
                "22.0 内部缺陷 B2 推理 replay",
                f"dataset_id: {args.dataset_id}",
                f"artifact_manifest: {args.artifact_manifest}",
                f"prediction_path: {prediction_path}",
                "输入规则: 只复用 21.9 由 delta_b/BxByBz 和 delta_b 派生特征生成的预测，不把标签作为输入。",
                f"test_mean_total_abs_normalized_error: {mean_total:.6f}",
                f"test_mean_burial_depth_error_mm: {mean_burial:.3f}",
                f"test_mean_center_xyz_error_mm: {mean_center:.3f}",
                f"test_shape_accuracy: {shape_acc:.6f}",
                "checkpoint_written: false",
                "npz_written: false",
                "current_baseline_update: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
