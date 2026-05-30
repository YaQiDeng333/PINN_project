#!/usr/bin/env python
"""22.3 hard-case internal defect training/evaluation shared helpers."""

from __future__ import annotations

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
from train_internal_defect_burial_depth_candidates import BurialDepthNet, CANDIDATE_CONFIGS, predict as predict_b2_net
from train_internal_defect_feature_baselines import extract_features, standardize_features


DATASET_ID = "comsol_internal_defect_pilot_pack_v3_hardcase"
B2_MANIFEST = ROOT / "results/manifests/internal_defect_b2_inference_artifact_manifest.json"

METRIC_FIELDS = [
    "model",
    "selected_model",
    "seed",
    "split",
    "subset",
    "sample_count",
    "selection_score",
    "best_epoch",
    "total_normalized_mae",
    "dimension_mae_mm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_component_mae_mm",
    "center_x_mae_mm",
    "center_y_mae_mm",
    "center_z_mae_mm",
    "shape_accuracy",
    "shape_macro_f1",
]

TAIL_FIELDS = [
    "model",
    "selected_model",
    "seed",
    "split",
    "subset",
    "sample_count",
    "total_error_mean",
    "total_error_median",
    "total_error_p90",
    "total_error_p95",
    "total_error_max",
    "burial_depth_error_mean_mm",
    "burial_depth_error_median_mm",
    "burial_depth_error_p90_mm",
    "burial_depth_error_p95_mm",
    "burial_depth_error_max_mm",
    "center_xyz_error_mean_mm",
    "center_xyz_error_median_mm",
    "center_xyz_error_p90_mm",
    "center_xyz_error_p95_mm",
    "center_xyz_error_max_mm",
    "dimension_outlier_count",
    "catastrophic_failure_count",
    "catastrophic_failure_rate",
    "geometry_branch_failure_count",
    "geometry_branch_failure_rate",
    "shape_error_count",
    "shape_error_rate",
]

PREDICTION_FIELDS = [
    "model",
    "seed",
    "sample_id",
    "split",
    "subset",
    "row_origin",
    "hardcase_target_id",
    "hardcase_target_reason",
    "hardcase_neighbor_strategy",
    "true_shape_type",
    "pred_shape_type",
    "shape_correct",
    "burial_depth_level",
    "size_level",
    "aspect_bin",
    "center_region",
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
    "dimension_relative_max",
    "failure_tags",
    "is_catastrophic_failure",
    "is_geometry_branch_failure",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def prepare_dataset(dataset_id: str = DATASET_ID) -> dict[str, Any]:
    dataset = load_dataset(dataset_id)
    splits = split_indices(dataset.split)
    x_mean, x_std = train_normalization(dataset.x_channels, splits["train"])
    x = normalize_x(dataset.x_channels, x_mean, x_std)
    y = dataset.y_regression
    y_mean, y_std = train_target_scaler(y, splits["train"])
    y_norm = normalize_y(y, y_mean, y_std)
    features_raw, feature_names = extract_features(dataset.delta_b)
    features, feature_mean, feature_std = standardize_features(features_raw, splits["train"])
    return {
        "dataset": dataset,
        "splits": splits,
        "x": x,
        "x_mean": x_mean,
        "x_std": x_std,
        "y": y,
        "y_mean": y_mean,
        "y_std": y_std,
        "y_norm": y_norm,
        "features_raw": features_raw,
        "features": features,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_names": feature_names,
    }


def subset_indices(dataset: Any, split: str | None = None, subset: str = "all") -> np.ndarray:
    mask = np.ones(dataset.sample_ids.shape[0], dtype=bool)
    if split:
        mask &= dataset.split == split
    if subset == "source_v2":
        mask &= dataset.row_origin == "source_v2_240"
    elif subset == "hardcase_topup":
        mask &= dataset.row_origin == "hardcase_topup_v1"
    elif subset != "all":
        raise ValueError(subset)
    return np.where(mask)[0]


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: 0.0 for key in ["mean", "median", "p90", "p95", "max"]}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def per_sample_error_arrays(y_true: np.ndarray, y_pred: np.ndarray, shape_true: np.ndarray, shape_pred: np.ndarray, y_std: np.ndarray) -> dict[str, np.ndarray]:
    err = np.abs(y_true - y_pred)
    total = np.mean(err / y_std.reshape(1, -1), axis=1)
    center = np.linalg.norm(err[:, 4:7], axis=1) * 1000.0
    burial = err[:, 3] * 1000.0
    dim_err = err[:, :3] * 1000.0
    dim_rel = dim_err / np.maximum(np.abs(y_true[:, :3]) * 1000.0, 1e-6)
    shape_mis = shape_true != shape_pred
    center_out = center > 3.0
    burial_out = burial > 1.0
    dim_out = (np.max(dim_err, axis=1) > 2.0) | (np.max(dim_rel, axis=1) > 0.30)
    return {
        "err": err,
        "total": total,
        "center": center,
        "burial": burial,
        "dim_err": dim_err,
        "dim_rel_max": np.max(dim_rel, axis=1),
        "dimension_outlier": dim_out,
        "center_outlier": center_out,
        "burial_outlier": burial_out,
        "shape_error": shape_mis,
        "catastrophic": center_out & burial_out,
        "geometry_branch": shape_mis & center_out & burial_out,
    }


def metric_row(model: str, selected: bool, seed: int | str, split: str, subset: str, idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, shape_true: np.ndarray, shape_pred: np.ndarray, y_std: np.ndarray, score: float | str = "", best_epoch: int | str = "") -> dict[str, Any]:
    reg = regression_metrics(y_true[idx], y_pred[idx], y_std)
    cls = classification_metrics(shape_true[idx], shape_pred[idx])
    return {
        "model": model,
        "selected_model": selected,
        "seed": seed,
        "split": split,
        "subset": subset,
        "sample_count": int(idx.size),
        "selection_score": score,
        "best_epoch": best_epoch,
        "total_normalized_mae": reg["total_normalized_mae"],
        "dimension_mae_mm": reg["dimension_mae_mm"],
        "L_mae_mm": reg["L_mae_mm"],
        "W_mae_mm": reg["W_mae_mm"],
        "D_mae_mm": reg["D_mae_mm"],
        "burial_depth_mae_mm": reg["burial_depth_mae_mm"],
        "center_xyz_component_mae_mm": reg["center_xyz_mae_mm"],
        "center_x_mae_mm": reg["center_x_mae_mm"],
        "center_y_mae_mm": reg["center_y_mae_mm"],
        "center_z_mae_mm": reg["center_z_mae_mm"],
        "shape_accuracy": cls["shape_accuracy"],
        "shape_macro_f1": cls["shape_macro_f1"],
    }


def tail_row(model: str, selected: bool, seed: int | str, split: str, subset: str, idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, shape_true: np.ndarray, shape_pred: np.ndarray, y_std: np.ndarray) -> dict[str, Any]:
    errors = per_sample_error_arrays(y_true[idx], y_pred[idx], shape_true[idx], shape_pred[idx], y_std)
    total = quantiles(errors["total"])
    burial = quantiles(errors["burial"])
    center = quantiles(errors["center"])
    n = int(idx.size)
    catastrophic = int(np.sum(errors["catastrophic"]))
    branch = int(np.sum(errors["geometry_branch"]))
    shape_error = int(np.sum(errors["shape_error"]))
    return {
        "model": model,
        "selected_model": selected,
        "seed": seed,
        "split": split,
        "subset": subset,
        "sample_count": n,
        "total_error_mean": total["mean"],
        "total_error_median": total["median"],
        "total_error_p90": total["p90"],
        "total_error_p95": total["p95"],
        "total_error_max": total["max"],
        "burial_depth_error_mean_mm": burial["mean"],
        "burial_depth_error_median_mm": burial["median"],
        "burial_depth_error_p90_mm": burial["p90"],
        "burial_depth_error_p95_mm": burial["p95"],
        "burial_depth_error_max_mm": burial["max"],
        "center_xyz_error_mean_mm": center["mean"],
        "center_xyz_error_median_mm": center["median"],
        "center_xyz_error_p90_mm": center["p90"],
        "center_xyz_error_p95_mm": center["p95"],
        "center_xyz_error_max_mm": center["max"],
        "dimension_outlier_count": int(np.sum(errors["dimension_outlier"])),
        "catastrophic_failure_count": catastrophic,
        "catastrophic_failure_rate": catastrophic / n if n else 0.0,
        "geometry_branch_failure_count": branch,
        "geometry_branch_failure_rate": branch / n if n else 0.0,
        "shape_error_count": shape_error,
        "shape_error_rate": shape_error / n if n else 0.0,
    }


def selection_score(metric: dict[str, Any], tail: dict[str, Any]) -> float:
    return float(
        safe_float(metric["total_normalized_mae"])
        + 0.35 * safe_float(metric["burial_depth_mae_mm"])
        + 0.10 * safe_float(metric["center_xyz_component_mae_mm"])
        + 0.35 * safe_float(tail["catastrophic_failure_rate"])
        + 0.25 * safe_float(tail["geometry_branch_failure_rate"])
        + 0.05 * (1.0 - safe_float(metric["shape_macro_f1"]))
    )


def load_old_b2_on_dataset(prepared: dict[str, Any], artifact_manifest: Path = B2_MANIFEST) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    checkpoint_path = Path(manifest["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise RuntimeError("B2 checkpoint sha256 mismatch")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm = ckpt["normalization"]
    x = normalize_x(prepared["dataset"].x_channels, norm["x_mean"], norm["x_std"])
    features_raw, _ = extract_features(prepared["dataset"].delta_b)
    features = ((features_raw - norm["feature_mean"]) / norm["feature_std"]).astype(np.float32)
    model = BurialDepthNet(feature_dim=int(features.shape[1]), feature_fusion=True, shape_conditioned=False)
    model.load_state_dict(ckpt["state_dict"])
    pred_norm, shape_pred = predict_b2_net(model, x, features)
    pred = denormalize_y(pred_norm, norm["y_mean"], norm["y_std"])
    return pred, shape_pred, manifest


def prediction_rows(model: str, seed: int | str, dataset: Any, y_pred: np.ndarray, shape_pred: np.ndarray, y_std: np.ndarray) -> list[dict[str, Any]]:
    y_true = dataset.y_regression
    shape_true = dataset.shape_label
    errors = per_sample_error_arrays(y_true, y_pred, shape_true, shape_pred, y_std)
    rows: list[dict[str, Any]] = []
    for i, sample_id in enumerate(dataset.sample_ids):
        tags: list[str] = []
        if errors["center_outlier"][i]:
            tags.append("center_outlier")
        if errors["burial_outlier"][i]:
            tags.append("burial_outlier")
        if errors["dimension_outlier"][i]:
            tags.append("dimension_outlier")
        if errors["shape_error"][i]:
            tags.append("shape_misclassified")
        if errors["catastrophic"][i]:
            tags.append("full_shift_failure")
        if errors["geometry_branch"][i]:
            tags.append("geometry_branch_failure")
        if errors["center_outlier"][i] or errors["burial_outlier"][i] or errors["shape_error"][i]:
            tags.append("visual_suspect")
        err = errors["err"][i]
        rows.append(
            {
                "model": model,
                "seed": seed,
                "sample_id": str(sample_id),
                "split": str(dataset.split[i]),
                "subset": "hardcase_topup" if str(dataset.row_origin[i]) == "hardcase_topup_v1" else "source_v2",
                "row_origin": str(dataset.row_origin[i]),
                "hardcase_target_id": str(dataset.hardcase_target_id[i]),
                "hardcase_target_reason": str(dataset.hardcase_target_reason[i]),
                "hardcase_neighbor_strategy": str(dataset.hardcase_neighbor_strategy[i]),
                "true_shape_type": SHAPE_CLASSES[int(shape_true[i])],
                "pred_shape_type": SHAPE_CLASSES[int(shape_pred[i])],
                "shape_correct": bool(shape_true[i] == shape_pred[i]),
                "burial_depth_level": str(dataset.burial_depth_level[i]),
                "size_level": str(dataset.size_level[i]),
                "aspect_bin": str(dataset.aspect_bin[i]),
                "center_region": str(dataset.hardcase_center_region[i]),
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
                "center_xyz_error_mm": float(errors["center"][i]),
                "total_abs_normalized_error": float(errors["total"][i]),
                "dimension_relative_max": float(errors["dim_rel_max"][i]),
                "failure_tags": "|".join(tags),
                "is_catastrophic_failure": bool(errors["catastrophic"][i]),
                "is_geometry_branch_failure": bool(errors["geometry_branch"][i]),
            }
        )
    return rows


def metric_rows_for_model(model: str, selected: bool, seed: int | str, dataset: Any, splits: dict[str, np.ndarray], y_pred: np.ndarray, shape_pred: np.ndarray, y_std: np.ndarray, score: float | str = "", best_epoch: int | str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        for subset in ["all", "source_v2", "hardcase_topup"]:
            idx = subset_indices(dataset, split, subset)
            if idx.size == 0:
                continue
            metrics.append(metric_row(model, selected, seed, split, subset, idx, dataset.y_regression, y_pred, dataset.shape_label, shape_pred, y_std, score if split == "val" and subset == "all" else "", best_epoch))
            tails.append(tail_row(model, selected, seed, split, subset, idx, dataset.y_regression, y_pred, dataset.shape_label, shape_pred, y_std))
    return metrics, tails
