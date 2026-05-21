from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_comsol_rect_rot_dense_coarse_initializer as dense_mod  # noqa: E402
import refine_comsol_rect_rot_geometry_with_forward_surrogate as ref_mod  # noqa: E402
import train_comsol_rect_rot_geometry_forward_surrogate as forward_mod  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = dense_mod.DEFAULT_NPZ
DEFAULT_LABELS = dense_mod.DEFAULT_LABELS
DEFAULT_FEATURES = forward_mod.DEFAULT_FEATURES
DEFAULT_DENSE_METRICS = dense_mod.DEFAULT_METRICS
DEFAULT_INITIAL_GEOMETRY = dense_mod.DEFAULT_GEOMETRY

DEFAULT_INPUT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_dense_refinement_input_check_summary.txt"
DEFAULT_INPUT_CHECK = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_refinement_input_check.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_dense_priewald_refinement_poc_summary.txt"
DEFAULT_AUDIT = PROJECT_ROOT / "results/summaries/comsol_rect_rot_dense_priewald_refinement_failure_audit_summary.txt"
DEFAULT_CONFIG_SWEEP = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_priewald_refinement_config_sweep.csv"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_priewald_refinement_metrics.csv"
DEFAULT_GROUP = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_priewald_refinement_group_summary.csv"
DEFAULT_GEOMETRY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_priewald_refinement_geometry_summary.csv"
DEFAULT_FORWARD = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_priewald_refinement_forward_summary.csv"
DEFAULT_FAILURE = PROJECT_ROOT / "results/metrics/comsol_rect_rot_dense_priewald_refinement_failure_cases.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_rect_rot_dense_priewald_refinement_poc"

REF_2051_INITIAL_IOU = 0.6138
REF_2051_INITIAL_DICE = 0.7577
REF_2052_REFINED_IOU = 0.6194
REF_2052_REFINED_DICE = 0.7619
DENSE_SINGLE_BASELINE_IOU = dense_mod.base.DENSE_SINGLE_BASELINE_IOU
DENSE_SINGLE_BASELINE_DICE = dense_mod.base.DENSE_SINGLE_BASELINE_DICE

METRIC_FIELDS = [
    "sample_id",
    "source_index",
    "split",
    "defect_type",
    "source_pack",
    "dense_threshold",
    "geometry_threshold",
    "pred_defect_type",
    "type_prob_rectangular_notch",
    "type_prob_rotated_rect",
    "dense_iou",
    "dense_dice",
    "dense_area_error",
    "dense_center_error_px",
    "dense_pred_area",
    "initial_iou",
    "refined_iou",
    "delta_iou",
    "initial_dice",
    "refined_dice",
    "delta_dice",
    "initial_area_error",
    "refined_area_error",
    "delta_area_error",
    "initial_center_error_px",
    "refined_center_error_px",
    "initial_pred_area",
    "refined_pred_area",
    "true_area",
    "initial_forward_mse",
    "refined_forward_mse",
    "initial_forward_nrmse",
    "refined_forward_nrmse",
    "forward_nrmse_reduction",
    "initial_forward_correlation",
    "refined_forward_correlation",
    "true_center_x",
    "true_center_y",
    "initial_center_x",
    "initial_center_y",
    "refined_center_x",
    "refined_center_y",
    "center_drift_m",
    "true_width",
    "initial_width",
    "refined_width",
    "width_drift_m",
    "true_length",
    "initial_length",
    "refined_length",
    "length_drift_m",
    "true_depth",
    "initial_depth",
    "refined_depth",
    "depth_drift_m",
    "true_angle_deg",
    "initial_angle_deg",
    "refined_angle_deg",
    "initial_angle_abs_error_deg",
    "refined_angle_abs_error_deg",
    "angle_error_delta",
    "parameter_drift_norm",
    "refinement_category",
    "notes",
]

CONFIG_FIELDS = ref_mod.CONFIG_FIELDS
GROUP_FIELDS = ref_mod.GROUP_FIELDS
GEOMETRY_FIELDS = ref_mod.GEOMETRY_FIELDS
FORWARD_FIELDS = ref_mod.FORWARD_FIELDS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dense_lookup(metrics_path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(metrics_path)
    return {row["sample_id"]: row for row in rows}


def load_initial_rows(path: Path, arrays: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv(path)
    keep = set(str(sample_id) for sample_id in arrays["sample_ids"])
    rows = [row for row in rows if row["sample_id"] in keep]
    if len(rows) != int(arrays["sample_ids"].shape[0]):
        raise RuntimeError(f"Dense initializer geometry has {len(rows)} rect/rot rows, expected {arrays['sample_ids'].shape[0]}")
    return rows


def initial_arrays(initial_rows: list[dict[str, Any]], arrays: dict[str, Any]) -> dict[str, np.ndarray]:
    row_by_id = {row["sample_id"]: row for row in initial_rows}
    geom = []
    angle = []
    type_prob = []
    threshold = []
    dense_threshold = []
    for sample_id in arrays["sample_ids"]:
        row = row_by_id[str(sample_id)]
        geom.append(
            [
                ref_mod.base.to_float(row["pred_center_x"]),
                ref_mod.base.to_float(row["pred_center_y"]),
                ref_mod.base.to_float(row["pred_width"]),
                ref_mod.base.to_float(row["pred_length"]),
                ref_mod.base.to_float(row["pred_depth"]),
            ]
        )
        angle.append(ref_mod.base.to_float(row["pred_angle_rad"], 0.0))
        type_prob.append(
            [
                ref_mod.base.to_float(row["type_prob_rectangular_notch"], 0.5),
                ref_mod.base.to_float(row["type_prob_rotated_rect"], 0.5),
            ]
        )
        threshold.append(ref_mod.base.to_float(row.get("geometry_threshold", row.get("threshold", 0.5)), 0.5))
        dense_threshold.append(ref_mod.base.to_float(row.get("dense_threshold", row.get("threshold", 0.5)), 0.5))
    type_prob_np = np.asarray(type_prob, dtype=np.float32)
    type_prob_np = type_prob_np / np.maximum(type_prob_np.sum(axis=1, keepdims=True), 1e-8)
    return {
        "geom": np.asarray(geom, dtype=np.float32),
        "angle": np.asarray(angle, dtype=np.float32),
        "type_prob": type_prob_np,
        "threshold": np.asarray(threshold, dtype=np.float32),
        "dense_threshold": np.asarray(dense_threshold, dtype=np.float32),
    }


def build_initial_rows(indices: np.ndarray, init: dict[str, np.ndarray], arrays: dict[str, Any], surrogate, split: str, device, threshold: float, dense_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base_rows = ref_mod.build_initial_prediction_rows(indices, init, arrays, surrogate, split, device, threshold)
    return augment_with_dense(base_rows, dense_by_id, init, arrays)


def augment_with_dense(rows: list[dict[str, Any]], dense_by_id: dict[str, dict[str, Any]], init: dict[str, np.ndarray], arrays: dict[str, Any]) -> list[dict[str, Any]]:
    sample_to_local = {str(sample_id): idx for idx, sample_id in enumerate(arrays["sample_ids"])}
    out: list[dict[str, Any]] = []
    for row in rows:
        dense_row = dense_by_id[row["sample_id"]]
        local_idx = sample_to_local[row["sample_id"]]
        merged = {
            **row,
            "dense_threshold": ref_mod.base.to_float(dense_row.get("dense_threshold", dense_row.get("threshold", "")), math.nan),
            "geometry_threshold": ref_mod.base.to_float(dense_row.get("geometry_threshold", row["threshold"]), row["threshold"]),
            "dense_iou": ref_mod.base.to_float(dense_row["dense_iou"]),
            "dense_dice": ref_mod.base.to_float(dense_row["dense_dice"]),
            "dense_area_error": ref_mod.base.to_float(dense_row["dense_area_error"]),
            "dense_center_error_px": ref_mod.base.to_float(dense_row["dense_center_error_px"]),
            "dense_pred_area": ref_mod.base.to_float(dense_row["dense_pred_area"]),
            "threshold": float(init["threshold"][local_idx]),
        }
        out.append(merged)
    return out


def split_stats(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    stats = ref_mod.split_stats(rows, split)
    subset = [row for row in rows if row["split"] == split]
    stats.update(
        {
            "dense_iou": ref_mod.safe_mean(subset, "dense_iou"),
            "dense_dice": ref_mod.safe_mean(subset, "dense_dice"),
            "dense_area_error": ref_mod.safe_mean(subset, "dense_area_error"),
            "dense_center_error_px": ref_mod.safe_mean(subset, "dense_center_error_px"),
        }
    )
    return stats


def build_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = ref_mod.build_group_rows(rows)
    dense_stats = []
    for split, group_name, value, subset in ref_mod.grouped_subsets(rows):
        dense_stats.append(
            {
                "split": split,
                "group_name": f"dense_{group_name}",
                "group_value": value,
                "sample_count": len(subset),
                "initial_iou_mean": ref_mod.safe_mean(subset, "dense_iou"),
                "refined_iou_mean": ref_mod.safe_mean(subset, "refined_iou"),
                "delta_iou_mean": ref_mod.safe_mean(
                    [{"delta_dense_to_refined_iou": row["refined_iou"] - row["dense_iou"]} for row in subset],
                    "delta_dense_to_refined_iou",
                ),
                "initial_dice_mean": ref_mod.safe_mean(subset, "dense_dice"),
                "refined_dice_mean": ref_mod.safe_mean(subset, "refined_dice"),
                "delta_dice_mean": ref_mod.safe_mean(
                    [{"delta_dense_to_refined_dice": row["refined_dice"] - row["dense_dice"]} for row in subset],
                    "delta_dense_to_refined_dice",
                ),
                "initial_area_error_mean": ref_mod.safe_mean(subset, "dense_area_error"),
                "refined_area_error_mean": ref_mod.safe_mean(subset, "refined_area_error"),
                "delta_area_error_mean": ref_mod.safe_mean(
                    [{"delta_dense_to_refined_area": row["refined_area_error"] - row["dense_area_error"]} for row in subset],
                    "delta_dense_to_refined_area",
                ),
                "initial_forward_nrmse_mean": math.nan,
                "refined_forward_nrmse_mean": ref_mod.safe_mean(subset, "refined_forward_nrmse"),
                "forward_nrmse_reduction_mean": math.nan,
                "initial_angle_mae_deg": math.nan,
                "refined_angle_mae_deg": ref_mod.safe_mean(
                    [row for row in subset if row["defect_type"] == "rotated_rect"],
                    "refined_angle_abs_error_deg",
                ),
                "angle_error_delta_mean": math.nan,
                "parameter_drift_norm_mean": ref_mod.safe_mean(subset, "parameter_drift_norm"),
            }
        )
    return out + dense_stats


def config_score(rows: list[dict[str, Any]], config: ref_mod.RefinementConfig) -> dict[str, Any]:
    return ref_mod.config_score(rows, config)


def forward_surrogate_split_stats(
    surrogate_bundle: forward_mod.ForwardSurrogateBundle,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        ds = forward_mod.ForwardDataset(surrogate_bundle.arrays["split_indices"][split], surrogate_bundle.arrays)
        pred = forward_mod.predict(surrogate_bundle.model, ds, surrogate_bundle.device, batch_size)
        rows.extend(forward_mod.metric_rows(pred, surrogate_bundle.arrays, split))
    return {split: forward_mod.summarize_split(rows, split) for split in ["train", "val", "test"]}


def write_input_check(
    args: argparse.Namespace,
    arrays: dict[str, Any],
    diagnostics: dict[str, Any],
    surrogate_bundle: forward_mod.ForwardSurrogateBundle,
    forward_stats: dict[str, dict[str, float]],
    initial_rows: list[dict[str, Any]],
    initial_eval_rows: list[dict[str, Any]],
) -> None:
    dense = {split: split_stats(initial_eval_rows, split) for split in ["train", "val", "test"]}
    geometry_rows = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "defect_type": row["defect_type"],
            "dense_iou": row["dense_iou"],
            "dense_dice": row["dense_dice"],
            "dense_area_error": row["dense_area_error"],
            "geometry_iou": row["initial_iou"],
            "geometry_dice": row["initial_dice"],
            "geometry_area_error": row["initial_area_error"],
        }
        for row in initial_eval_rows
    ]
    check_row = {
        "npz_path": str(args.npz),
        "labels_path": str(args.labels),
        "initial_geometry_path": str(args.initial_geometry),
        "dense_metrics_path": str(args.dense_metrics),
        "rect_rot_n": diagnostics["n_rect_rot"],
        "split_train": diagnostics["split_counts"]["train"],
        "split_val": diagnostics["split_counts"]["val"],
        "split_test": diagnostics["split_counts"]["test"],
        "forward_surrogate_source": "retrained_in_memory_no_checkpoint",
        "forward_best_epoch": surrogate_bundle.best_epoch,
        "forward_val_mse": surrogate_bundle.best_val["mse"],
        "forward_val_nrmse": surrogate_bundle.best_val["nrmse"],
        "forward_val_correlation": surrogate_bundle.best_val["correlation"],
        "forward_test_mse": forward_stats["test"]["mse"],
        "forward_test_nrmse": forward_stats["test"]["nrmse"],
        "forward_test_correlation": forward_stats["test"]["correlation"],
        "initial_rows": len(initial_rows),
        "test_dense_iou": dense["test"]["dense_iou"],
        "test_dense_dice": dense["test"]["dense_dice"],
        "test_geometry_iou": dense["test"]["initial_iou"],
        "test_geometry_dice": dense["test"]["initial_dice"],
        "passed": True,
    }
    write_csv(args.input_check, [check_row], list(check_row.keys()))
    lines = [
        "COMSOL rect/rot dense-initialized Priewald refinement input check summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Geometry labels: {args.labels}",
        f"Dense/coarse initializer geometry: {args.initial_geometry}",
        f"Dense initializer metrics: {args.dense_metrics}",
        "Scope: rectangular_notch + rotated_rect only; polygon excluded.",
        "Forward surrogate source: retrained in-memory with the 20.51/20.52 protocol; no checkpoint written.",
        "Refinement policy: true mask / true geometry are metrics only and are not used in optimization.",
        "",
        f"rect+rot N: {diagnostics['n_rect_rot']}",
        f"split counts: {diagnostics['split_counts']}",
        f"forward surrogate best epoch: {surrogate_bundle.best_epoch}",
        f"forward surrogate val MSE/NRMSE/corr: {surrogate_bundle.best_val['mse']:.6f} / {surrogate_bundle.best_val['nrmse']:.6f} / {surrogate_bundle.best_val['correlation']:.4f}",
        f"forward surrogate test MSE/NRMSE/corr: {forward_stats['test']['mse']:.6f} / {forward_stats['test']['nrmse']:.6f} / {forward_stats['test']['correlation']:.4f}",
        "",
        "Initial dense mask metrics:",
        f"- train IoU/Dice = {dense['train']['dense_iou']:.4f} / {dense['train']['dense_dice']:.4f}",
        f"- val IoU/Dice = {dense['val']['dense_iou']:.4f} / {dense['val']['dense_dice']:.4f}",
        f"- test IoU/Dice = {dense['test']['dense_iou']:.4f} / {dense['test']['dense_dice']:.4f}",
        "",
        "Initial extracted geometry raster metrics:",
        f"- train IoU/Dice = {dense['train']['initial_iou']:.4f} / {dense['train']['initial_dice']:.4f}",
        f"- val IoU/Dice = {dense['val']['initial_iou']:.4f} / {dense['val']['initial_dice']:.4f}",
        f"- test IoU/Dice = {dense['test']['initial_iou']:.4f} / {dense['test']['initial_dice']:.4f}",
        "",
        "Leakage check: dense model input was delta_bz only; geometry proposal came from predicted mask; refinement selection uses val only.",
    ]
    args.input_summary.parent.mkdir(parents=True, exist_ok=True)
    args.input_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_failure_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row["split"] == "test"]
    prioritized = sorted(
        test_rows,
        key=lambda row: (
            row["refinement_category"] != "surrogate_mismatch",
            row["delta_iou"],
            row["refined_iou"] - row["dense_iou"],
            -row["forward_nrmse_reduction"],
        ),
    )
    return [dict(row) for row in prioritized[:40]]


def write_summary(
    args: argparse.Namespace,
    surrogate_bundle: forward_mod.ForwardSurrogateBundle,
    forward_stats: dict[str, dict[str, float]],
    selected_config: ref_mod.RefinementConfig,
    selected_row: dict[str, Any],
    config_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    preview_count: int,
) -> dict[str, Any]:
    stats = {split: split_stats(rows, split) for split in ["train", "val", "test"]}
    test = stats["test"]
    proposal_meets_reference = (
        test["initial_iou"] >= REF_2051_INITIAL_IOU
        or test["initial_dice"] >= REF_2051_INITIAL_DICE
    )
    promising = (
        proposal_meets_reference
        and (test["delta_iou"] >= 0.01 or test["delta_dice"] >= 0.008)
        and test["forward_nrmse_reduction"] > 0
        and test["delta_area_error"] <= 0.03
        and test["parameter_drift_norm"] <= 1.0
    )
    surrogate_mismatch = test["forward_nrmse_reduction"] > 0.02 and (test["delta_iou"] < -0.005 or test["delta_dice"] < -0.004)
    dense_beats_geometry = test["dense_iou"] > test["initial_iou"] + 0.03
    recommendation = (
        "B. Improve dense-to-geometry proposal extraction."
        if not proposal_meets_reference
        else (
        "C. Do mask/profile basis refinement."
        if dense_beats_geometry and not promising
        else ("A. Improve forward surrogate." if surrogate_mismatch else "Continue refinement route with human confirmation.")
        )
    )
    all_val_negative = all(row["delta_mask_iou"] < 0 and row["delta_mask_dice"] < 0 for row in config_rows)
    lines = [
        "COMSOL rect/rot dense/coarse initializer + Priewald-style refinement POC summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Dense/coarse initializer geometry: {args.initial_geometry}",
        "No COMSOL run; no new data; no baseline update.",
        "Dense initializer is only a coarse proposal source, not a new baseline.",
        "Refinement uses observed normalized delta_bz and frozen surrogate forward residual.",
        "True mask / true geometry are metrics only and are not used in optimization.",
        "",
        "Forward surrogate:",
        "- source: retrained in-memory with 20.51/20.52 protocol; no checkpoint written",
        f"- best epoch = {surrogate_bundle.best_epoch}",
        f"- val MSE/NRMSE/corr = {surrogate_bundle.best_val['mse']:.6f} / {surrogate_bundle.best_val['nrmse']:.6f} / {surrogate_bundle.best_val['correlation']:.4f}",
        f"- test MSE/NRMSE/corr = {forward_stats['test']['mse']:.6f} / {forward_stats['test']['nrmse']:.6f} / {forward_stats['test']['correlation']:.4f}",
        "",
        "Selected validation config:",
        f"- config = {selected_config.name}",
        f"- steps/lr/lambda_prior = {selected_config.steps} / {selected_config.lr} / {selected_config.lambda_prior}",
        f"- val_refinement_score = {selected_row['val_refinement_score']:.6f}",
        f"- val delta geometry-raster IoU/Dice = {selected_row['delta_mask_iou']:.6f} / {selected_row['delta_mask_dice']:.6f}",
        f"- val forward NRMSE reduction = {selected_row['forward_nrmse_reduction']:.6f}",
        f"- all validation configs hurt geometry-raster IoU/Dice: {all_val_negative}",
        "",
        "Dense mask vs extracted geometry vs refined geometry:",
    ]
    for split in ["train", "val", "test"]:
        s = stats[split]
        lines.extend(
            [
                f"- {split} dense mask IoU/Dice/area = {s['dense_iou']:.4f} / {s['dense_dice']:.4f} / {s['dense_area_error']:.4f}",
                f"- {split} extracted geometry IoU/Dice/area = {s['initial_iou']:.4f} / {s['initial_dice']:.4f} / {s['initial_area_error']:.4f}",
                f"- {split} refined geometry IoU/Dice/area = {s['refined_iou']:.4f} / {s['refined_dice']:.4f} / {s['refined_area_error']:.4f}",
                f"- {split} forward NRMSE = {s['initial_forward_nrmse']:.4f} -> {s['refined_forward_nrmse']:.4f} (reduction {s['forward_nrmse_reduction']:.4f})",
                f"- {split} angle MAE = {s['initial_angle_mae_deg']:.4f} -> {s['refined_angle_mae_deg']:.4f} (delta {s['angle_error_delta']:.4f})",
                f"- {split} parameter drift norm = {s['parameter_drift_norm']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            f"20.51 geometry-head initial IoU/Dice reference: {REF_2051_INITIAL_IOU:.4f} / {REF_2051_INITIAL_DICE:.4f}",
            f"20.52 refined IoU/Dice reference: {REF_2052_REFINED_IOU:.4f} / {REF_2052_REFINED_DICE:.4f}",
            f"Dense single-defect baseline IoU/Dice reference: {DENSE_SINGLE_BASELINE_IOU:.4f} / {DENSE_SINGLE_BASELINE_DICE:.4f}",
            f"Preview PNG generated: {preview_count} (not for submission)",
            f"Extracted geometry proposal meets 20.51 initializer reference: {proposal_meets_reference}",
            f"Surrogate mismatch risk: {surrogate_mismatch}",
            f"Dense mask stronger than extracted geometry: {dense_beats_geometry}",
            f"POC promising by acceptance criteria: {promising}",
            f"Next recommendation: {recommendation}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    categories = defaultdict(int)
    for row in rows:
        if row["split"] == "test":
            categories[row["refinement_category"]] += 1
    audit_lines = [
        "COMSOL rect/rot dense-initialized Priewald refinement failure audit summary",
        "",
        f"Selected config: {selected_config.name}",
        f"Test refinement categories: {dict(sorted(categories.items()))}",
        f"Extracted geometry proposal meets 20.51 initializer reference: {proposal_meets_reference}",
        f"Surrogate mismatch risk: {surrogate_mismatch}",
        f"Dense mask stronger than extracted geometry: {dense_beats_geometry}",
        "",
        "Worst / risk cases:",
    ]
    for row in failure_rows[:20]:
        audit_lines.append(
            f"- {row['sample_id']}: {row['refinement_category']}, "
            f"dense IoU={row['dense_iou']:.3f}, geom {row['initial_iou']:.3f}->{row['refined_iou']:.3f}, "
            f"F {row['initial_forward_nrmse']:.3f}->{row['refined_forward_nrmse']:.3f}, "
            f"drift={row['parameter_drift_norm']:.3f}"
        )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    return {
        "train": stats["train"],
        "val": stats["val"],
        "test": stats["test"],
        "selected_config": selected_config.name,
        "promising": promising,
        "proposal_meets_reference": proposal_meets_reference,
        "surrogate_mismatch": surrogate_mismatch,
        "recommendation": recommendation,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    ref_mod.set_seed(args.seed)
    surrogate_args = argparse.Namespace(
        npz=args.npz,
        labels=args.labels,
        features=args.features,
        input_check_summary=PROJECT_ROOT / "results/summaries/_tmp_dense_refinement_forward_input_check.txt",
        input_check=PROJECT_ROOT / "results/metrics/_tmp_dense_refinement_forward_input_check.csv",
        summary=PROJECT_ROOT / "results/summaries/_tmp_dense_refinement_forward_summary.txt",
        metrics=PROJECT_ROOT / "results/metrics/_tmp_dense_refinement_forward_metrics.csv",
        epoch_log=PROJECT_ROOT / "results/metrics/_tmp_dense_refinement_forward_epoch.csv",
        group_summary=PROJECT_ROOT / "results/metrics/_tmp_dense_refinement_forward_group.csv",
        seed=args.seed,
        epochs=args.forward_epochs,
        batch_size=args.forward_batch_size,
        lr=args.forward_lr,
        cpu=args.cpu,
    )
    surrogate_bundle = forward_mod.train_forward_surrogate(surrogate_args, write_outputs=False)
    arrays = surrogate_bundle.arrays
    device = surrogate_bundle.device
    surrogate = surrogate_bundle.model.to(device)
    forward_stats = forward_surrogate_split_stats(surrogate_bundle, args.forward_batch_size)
    dense_by_id = dense_lookup(args.dense_metrics)
    initial_raw_rows = load_initial_rows(args.initial_geometry, arrays)
    init = initial_arrays(initial_raw_rows, arrays)
    threshold = float(np.median(init["threshold"][arrays["split_indices"]["val"]]))
    initial_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        initial_rows.extend(
            build_initial_rows(arrays["split_indices"][split], init, arrays, surrogate, split, device, threshold, dense_by_id)
        )
    write_input_check(args, arrays, surrogate_bundle.diagnostics, surrogate_bundle, forward_stats, initial_raw_rows, initial_rows)

    bounds_low, bounds_high = ref_mod.bounds_from_arrays(arrays, init)
    val_idx = arrays["split_indices"]["val"]
    config_rows: list[dict[str, Any]] = []
    for config in ref_mod.CONFIGS:
        refined_val = ref_mod.refine_indices(val_idx, init, arrays, surrogate, config, device, bounds_low, bounds_high)
        val_rows = ref_mod.evaluate_rows(val_idx, init, refined_val, arrays, surrogate, "val", device, threshold)
        val_rows = augment_with_dense(val_rows, dense_by_id, init, arrays)
        config_rows.append(config_score(val_rows, config))
        print(
            f"{config.name}: score={config_rows[-1]['val_refinement_score']:.5f} "
            f"dIoU={config_rows[-1]['delta_mask_iou']:.4f} dF={config_rows[-1]['forward_nrmse_reduction']:.4f}"
        )
    selected_row = sorted(
        config_rows,
        key=lambda row: (
            row["val_refinement_score"],
            row["delta_mask_iou"],
            row["delta_mask_dice"],
            row["forward_nrmse_reduction"],
        ),
        reverse=True,
    )[0]
    selected_config = next(config for config in ref_mod.CONFIGS if config.name == selected_row["config_name"])
    all_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        idx = arrays["split_indices"][split]
        refined = ref_mod.refine_indices(idx, init, arrays, surrogate, selected_config, device, bounds_low, bounds_high)
        rows = ref_mod.evaluate_rows(idx, init, refined, arrays, surrogate, split, device, threshold)
        all_rows.extend(augment_with_dense(rows, dense_by_id, init, arrays))

    group_rows = build_group_rows(all_rows)
    geometry_rows = ref_mod.build_geometry_rows(all_rows)
    forward_rows = ref_mod.build_forward_rows(all_rows)
    failure_rows = build_failure_cases(all_rows)
    preview_count = ref_mod.preview(all_rows, arrays, surrogate, init, args.preview_dir, device, max_count=24)

    write_csv(args.config_sweep, config_rows, CONFIG_FIELDS)
    write_csv(args.metrics, all_rows, METRIC_FIELDS)
    write_csv(args.group_summary, group_rows, GROUP_FIELDS)
    write_csv(args.geometry_summary, geometry_rows, GEOMETRY_FIELDS)
    write_csv(args.forward_summary, forward_rows, FORWARD_FIELDS)
    write_csv(args.failure_cases, failure_rows, list(failure_rows[0].keys()) if failure_rows else METRIC_FIELDS)
    return write_summary(
        args,
        surrogate_bundle,
        forward_stats,
        selected_config,
        selected_row,
        config_rows,
        all_rows,
        failure_rows,
        preview_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--dense-metrics", type=Path, default=DEFAULT_DENSE_METRICS)
    parser.add_argument("--initial-geometry", type=Path, default=DEFAULT_INITIAL_GEOMETRY)
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT_SUMMARY)
    parser.add_argument("--input-check", type=Path, default=DEFAULT_INPUT_CHECK)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--config-sweep", type=Path, default=DEFAULT_CONFIG_SWEEP)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--forward-summary", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward-epochs", type=int, default=300)
    parser.add_argument("--forward-batch-size", type=int, default=32)
    parser.add_argument("--forward-lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
