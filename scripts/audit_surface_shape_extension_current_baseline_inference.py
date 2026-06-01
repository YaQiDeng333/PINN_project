#!/usr/bin/env python
"""Run frozen 20.85 surface RBC baseline inference on the shape-extension pilot.

The model input is delta_b/BxByBz only. Labels are used only after inference for
metrics, grouping, and oracle-vs-model diagnosis. This script does not train,
does not run COMSOL, does not save checkpoints, and does not modify NPZ/data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import train_true_3d_rbc_neural_parameter_gate as gate
from load_true_3d_rbc_pilot_dataset import (
    V3_240_DATASET_ID,
    depth_map_from_params,
    projected_mask_from_params,
    load_dataset as load_rbc_dataset,
    split_indices as rbc_split_indices,
    sha256_file,
)
from audit_surface_shape_extension_rbc_oracle_fit import (
    DATASET_ID,
    BASELINE_ARTIFACT,
    METRICS as ORACLE_METRICS,
    ROOT,
    SurfaceShapeDataset,
    connected_component_count,
    er_like_profile_error,
    load_surface_dataset,
    mask_metrics,
    pose_for_sample,
    profile_rmse,
    write_csv,
)


SUMMARY = ROOT / "results/summaries/surface_shape_extension_current_baseline_inference_summary.txt"
METRICS = ROOT / "results/metrics/surface_shape_extension_current_baseline_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/surface_shape_extension_current_baseline_group_summary.csv"
FAILURE_CASES = ROOT / "results/metrics/surface_shape_extension_current_baseline_failure_cases.csv"

BASELINE_PROFILE_RMSE_PASS_M = 6.0e-4
BASELINE_DICE_PASS = 0.70

METRIC_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "rbc_compatible",
    "component_count",
    "model_pass",
    "rbc_representable",
    "baseline_profile_depth_rmse_m",
    "baseline_Er_like_error",
    "baseline_projected_mask_IoU",
    "baseline_projected_mask_Dice",
    "baseline_area_error",
    "baseline_mask_center_error_px",
    "total_normalized_MAE_vs_oracle",
    "L_mae_mm_vs_label",
    "W_mae_mm_vs_label",
    "D_mae_mm_vs_label",
    "L_mae_mm_vs_oracle",
    "W_mae_mm_vs_oracle",
    "D_mae_mm_vs_oracle",
    "wLD_abs_error_vs_oracle",
    "wWD_abs_error_vs_oracle",
    "wLW_abs_error_vs_oracle",
    "component_recall_proxy",
    "merge_component_proxy",
    "missed_component_proxy",
    "true_component_count",
    "pred_component_count",
    "crack_pred_aspect_ratio",
    "crack_aspect_abs_error",
    "rotation_proxy_available",
    "clip_applied_baseline_train_bounds",
    "pred_L_m",
    "pred_W_m",
    "pred_D_m",
    "pred_wLD",
    "pred_wWD",
    "pred_wLW",
    "oracle_L_m",
    "oracle_W_m",
    "oracle_D_m",
    "oracle_wLD",
    "oracle_wWD",
    "oracle_wLW",
    "fit_oracle_profile_depth_rmse_m",
    "notes",
]

GROUP_FIELDS = [
    "group_field",
    "group_value",
    "split",
    "sample_count",
    "model_pass_rate",
    "rbc_representable_rate",
    "baseline_profile_depth_rmse_mean_m",
    "baseline_profile_depth_rmse_p95_m",
    "baseline_Er_like_mean",
    "baseline_projected_mask_Dice_mean",
    "baseline_projected_mask_IoU_mean",
    "baseline_area_error_mean",
    "total_normalized_MAE_vs_oracle_mean",
    "L_mae_mm_vs_label_mean",
    "W_mae_mm_vs_label_mean",
    "D_mae_mm_vs_label_mean",
    "component_recall_proxy_mean",
    "merge_component_proxy_rate",
]

FAILURE_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "rbc_representable",
    "model_pass",
    "baseline_profile_depth_rmse_m",
    "baseline_projected_mask_Dice",
    "baseline_Er_like_error",
    "total_normalized_MAE_vs_oracle",
    "true_component_count",
    "pred_component_count",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen 20.85 baseline inference on the shape-extension pilot.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifact() -> tuple[dict[str, Any], dict[str, Any], gate.RBCConvRegressor]:
    manifest = read_json(BASELINE_ARTIFACT)
    if manifest.get("dataset_id") != V3_240_DATASET_ID:
        raise RuntimeError(f"unexpected baseline artifact dataset_id: {manifest.get('dataset_id')}")
    checkpoint_path = Path(manifest["checkpoint_path"])
    prediction_path = Path(manifest["prediction_artifact_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    if sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("baseline checkpoint sha256 mismatch")
    if sha256_file(prediction_path) != manifest["prediction_artifact_sha256"]:
        raise RuntimeError("baseline prediction artifact sha256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = gate.RBCConvRegressor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return manifest, checkpoint, model


def load_oracle_rows() -> dict[str, dict[str, str]]:
    if not ORACLE_METRICS.exists():
        raise FileNotFoundError(f"oracle metrics missing; run audit_surface_shape_extension_rbc_oracle_fit.py first: {ORACLE_METRICS}")
    rows = read_csv(ORACLE_METRICS)
    return {row["sample_id"]: row for row in rows}


def baseline_train_bounds() -> tuple[np.ndarray, np.ndarray]:
    dataset = load_rbc_dataset(V3_240_DATASET_ID)
    train = rbc_split_indices(dataset)["train"]
    low = dataset.rbc_params[train].min(axis=0).astype(np.float32)
    high = dataset.rbc_params[train].max(axis=0).astype(np.float32)
    return low, high


def predict_params(dataset: SurfaceShapeDataset, checkpoint: dict[str, Any], model: gate.RBCConvRegressor) -> np.ndarray:
    x_raw = dataset.delta_b.reshape(dataset.delta_b.shape[0], 9, dataset.delta_b.shape[-1]).astype(np.float32)
    norm = checkpoint["normalization"]
    x_norm = ((x_raw - norm["x_mean"]) / norm["x_std"]).astype(np.float32)
    pred_norm = gate.predict_norm(model, x_norm)
    return (pred_norm * norm["y_std"] + norm["y_mean"]).astype(np.float32)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in {"", None, ""} and math.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else math.nan


def p95(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in {"", None, ""} and math.isfinite(float(row[key]))]
    return float(np.percentile(vals, 95)) if vals else math.nan


def oracle_target(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[f"oracle_{name}"]) for name in ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW"]], dtype=np.float32)


def evaluate_rows(dataset: SurfaceShapeDataset, pred_raw: np.ndarray, checkpoint: dict[str, Any], oracle_by_id: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    low, high = baseline_train_bounds()
    pred = np.clip(pred_raw, low[None, :], high[None, :]).astype(np.float32)
    clipped = np.any(np.abs(pred - pred_raw) > 1.0e-12, axis=1)
    y_std = np.asarray(checkpoint["normalization"]["y_std"], dtype=np.float32).reshape(1, 6)
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(dataset.sample_ids):
        sid = str(sample_id)
        if sid not in oracle_by_id:
            raise RuntimeError(f"oracle row missing for sample_id={sid}")
        oracle = oracle_by_id[sid]
        oracle_params = oracle_target(oracle)
        pose = pose_for_sample(dataset, idx)
        true_depth = np.asarray(dataset.depth_grid_m[idx], dtype=np.float32)
        true_mask = np.asarray(dataset.projected_mask_2d[idx], dtype=np.uint8)
        pred_depth = depth_map_from_params(pred[idx], pose)
        pred_mask = projected_mask_from_params(pred[idx], pose)
        mm = mask_metrics(pred_mask, true_mask)
        rmse = profile_rmse(pred_depth, true_depth)
        er = er_like_profile_error(pred_depth, true_depth)
        normalized_mae = float(np.mean(np.abs((pred[idx] - oracle_params) / y_std.reshape(6))))
        true_components = int(dataset.component_count[idx])
        pred_components = connected_component_count(pred_mask)
        component_recall = float(min(pred_components, true_components) / max(true_components, 1))
        merge_proxy = bool(true_components > 1 and pred_components < true_components)
        missed_proxy = bool(component_recall < 1.0)
        pred_aspect = float(pred[idx, 0] / max(float(pred[idx, 1]), 1.0e-12))
        aspect_error = abs(pred_aspect - float(dataset.aspect_ratio[idx]))
        model_pass = bool(rmse <= BASELINE_PROFILE_RMSE_PASS_M and mm["dice"] >= BASELINE_DICE_PASS)
        rows.append(
            {
                "sample_id": sid,
                "split": str(dataset.split[idx]),
                "shape_type": str(dataset.shape_type[idx]),
                "topology_type": str(dataset.topology_type[idx]),
                "representation_target": str(dataset.representation_target[idx]),
                "rbc_compatible": bool(dataset.rbc_compatible[idx]),
                "component_count": true_components,
                "model_pass": model_pass,
                "rbc_representable": str(oracle["rbc_representable"]).lower() == "true",
                "baseline_profile_depth_rmse_m": rmse,
                "baseline_Er_like_error": er,
                "baseline_projected_mask_IoU": mm["iou"],
                "baseline_projected_mask_Dice": mm["dice"],
                "baseline_area_error": mm["area_error"],
                "baseline_mask_center_error_px": mm["center_error_px"],
                "total_normalized_MAE_vs_oracle": normalized_mae,
                "L_mae_mm_vs_label": abs(float(pred[idx, 0]) - float(dataset.L_m[idx])) * 1000.0,
                "W_mae_mm_vs_label": abs(float(pred[idx, 1]) - float(dataset.W_m[idx])) * 1000.0,
                "D_mae_mm_vs_label": abs(float(pred[idx, 2]) - float(dataset.D_m[idx])) * 1000.0,
                "L_mae_mm_vs_oracle": abs(float(pred[idx, 0]) - float(oracle_params[0])) * 1000.0,
                "W_mae_mm_vs_oracle": abs(float(pred[idx, 1]) - float(oracle_params[1])) * 1000.0,
                "D_mae_mm_vs_oracle": abs(float(pred[idx, 2]) - float(oracle_params[2])) * 1000.0,
                "wLD_abs_error_vs_oracle": abs(float(pred[idx, 3]) - float(oracle_params[3])),
                "wWD_abs_error_vs_oracle": abs(float(pred[idx, 4]) - float(oracle_params[4])),
                "wLW_abs_error_vs_oracle": abs(float(pred[idx, 5]) - float(oracle_params[5])),
                "component_recall_proxy": component_recall,
                "merge_component_proxy": merge_proxy,
                "missed_component_proxy": missed_proxy,
                "true_component_count": true_components,
                "pred_component_count": pred_components,
                "crack_pred_aspect_ratio": pred_aspect if str(dataset.shape_type[idx]) == "elongated_crack_like_surface_defect" else "",
                "crack_aspect_abs_error": aspect_error if str(dataset.shape_type[idx]) == "elongated_crack_like_surface_defect" else "",
                "rotation_proxy_available": False,
                "clip_applied_baseline_train_bounds": bool(clipped[idx]),
                "pred_L_m": float(pred[idx, 0]),
                "pred_W_m": float(pred[idx, 1]),
                "pred_D_m": float(pred[idx, 2]),
                "pred_wLD": float(pred[idx, 3]),
                "pred_wWD": float(pred[idx, 4]),
                "pred_wLW": float(pred[idx, 5]),
                "oracle_L_m": float(oracle_params[0]),
                "oracle_W_m": float(oracle_params[1]),
                "oracle_D_m": float(oracle_params[2]),
                "oracle_wLD": float(oracle_params[3]),
                "oracle_wWD": float(oracle_params[4]),
                "oracle_wLW": float(oracle_params[5]),
                "fit_oracle_profile_depth_rmse_m": float(oracle["oracle_profile_depth_rmse_m"]),
                "notes": "center/rotation labels used only for metric reconstruction; model input is delta_b only",
            }
        )
    return rows


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_field in ("shape_type", "topology_type", "representation_target"):
        values = sorted({str(row[group_field]) for row in rows})
        for split_name in ("all", "train", "val", "test"):
            for value in values:
                subset = [row for row in rows if str(row[group_field]) == value and (split_name == "all" or row["split"] == split_name)]
                if not subset:
                    continue
                out.append(
                    {
                        "group_field": group_field,
                        "group_value": value,
                        "split": split_name,
                        "sample_count": len(subset),
                        "model_pass_rate": float(np.mean([bool(row["model_pass"]) for row in subset])),
                        "rbc_representable_rate": float(np.mean([bool(row["rbc_representable"]) for row in subset])),
                        "baseline_profile_depth_rmse_mean_m": mean(subset, "baseline_profile_depth_rmse_m"),
                        "baseline_profile_depth_rmse_p95_m": p95(subset, "baseline_profile_depth_rmse_m"),
                        "baseline_Er_like_mean": mean(subset, "baseline_Er_like_error"),
                        "baseline_projected_mask_Dice_mean": mean(subset, "baseline_projected_mask_Dice"),
                        "baseline_projected_mask_IoU_mean": mean(subset, "baseline_projected_mask_IoU"),
                        "baseline_area_error_mean": mean(subset, "baseline_area_error"),
                        "total_normalized_MAE_vs_oracle_mean": mean(subset, "total_normalized_MAE_vs_oracle"),
                        "L_mae_mm_vs_label_mean": mean(subset, "L_mae_mm_vs_label"),
                        "W_mae_mm_vs_label_mean": mean(subset, "W_mae_mm_vs_label"),
                        "D_mae_mm_vs_label_mean": mean(subset, "D_mae_mm_vs_label"),
                        "component_recall_proxy_mean": mean(subset, "component_recall_proxy"),
                        "merge_component_proxy_rate": float(np.mean([bool(row["merge_component_proxy"]) for row in subset])),
                    }
                )
    return out


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = [row for row in rows if not bool(row["model_pass"])]
    out: list[dict[str, Any]] = []
    for row in sorted(failed, key=lambda item: (float(item["baseline_profile_depth_rmse_m"]), -float(item["baseline_projected_mask_Dice"])), reverse=True):
        reason = []
        if float(row["baseline_profile_depth_rmse_m"]) > BASELINE_PROFILE_RMSE_PASS_M:
            reason.append("profile_rmse_above_threshold")
        if float(row["baseline_projected_mask_Dice"]) < BASELINE_DICE_PASS:
            reason.append("mask_dice_below_threshold")
        if bool(row["merge_component_proxy"]):
            reason.append("component_merge_proxy")
        out.append({**{field: row.get(field, "") for field in FAILURE_FIELDS if field != "failure_reason"}, "failure_reason": "|".join(reason) or "unknown"})
    return out


def summary_lines(rows: list[dict[str, Any]], groups: list[dict[str, Any]], artifact: dict[str, Any]) -> list[str]:
    total = len(rows)
    pass_count = sum(1 for row in rows if bool(row["model_pass"]))
    non_rbc = [row for row in rows if not bool(row["rbc_compatible"])]
    non_rbc_pass = sum(1 for row in non_rbc if bool(row["model_pass"]))
    by_shape = {row["group_value"]: row for row in groups if row["group_field"] == "shape_type" and row["split"] == "all"}
    lines = [
        "surface shape-extension current baseline inference summary",
        "stage: 25.3",
        "",
        f"dataset_id: {DATASET_ID}",
        f"baseline_artifact_id: {artifact.get('artifact_id')}",
        f"model_family: {artifact.get('model_family')}",
        f"sample_count: {total}",
        f"model_pass_count: {pass_count}",
        f"model_pass_rate: {pass_count / max(total, 1):.6f}",
        f"non_rbc_model_pass_count: {non_rbc_pass}",
        f"non_rbc_model_pass_rate: {non_rbc_pass / max(len(non_rbc), 1):.6f}",
        f"model_pass_thresholds: profile_rmse<={BASELINE_PROFILE_RMSE_PASS_M}, dice>={BASELINE_DICE_PASS}",
        "",
        "by_shape:",
    ]
    for shape in sorted(by_shape):
        row = by_shape[shape]
        lines.append(
            f"- {shape}: n={row['sample_count']} model_pass_rate={float(row['model_pass_rate']):.6f} "
            f"rmse_mean={float(row['baseline_profile_depth_rmse_mean_m']):.9f} dice_mean={float(row['baseline_projected_mask_Dice_mean']):.6f} "
            f"norm_mae_vs_oracle={float(row['total_normalized_MAE_vs_oracle_mean']):.6f}"
        )
    lines.extend(
        [
            "",
            "input_boundary:",
            "- model input: delta_b reshaped to 9x201 Bx/By/Bz channels only",
            "- labels are used only for profile/mask/parameter metrics after inference",
            "- no training, no COMSOL, no checkpoint save, no data/NPZ mutation",
            "",
            f"metrics: {METRICS}",
            f"group_summary: {GROUP_SUMMARY}",
            f"failure_cases: {FAILURE_CASES}",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> int:
    dataset = load_surface_dataset(args.dataset_id)
    oracle_by_id = load_oracle_rows()
    artifact, checkpoint, model = load_artifact()
    pred_raw = predict_params(dataset, checkpoint, model)
    rows = evaluate_rows(dataset, pred_raw, checkpoint, oracle_by_id)
    groups = group_rows(rows)
    failures = failure_rows(rows)
    write_csv(METRICS, rows, METRIC_FIELDS)
    write_csv(GROUP_SUMMARY, groups, GROUP_FIELDS)
    write_csv(FAILURE_CASES, failures, FAILURE_FIELDS)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(summary_lines(rows, groups, artifact)) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
