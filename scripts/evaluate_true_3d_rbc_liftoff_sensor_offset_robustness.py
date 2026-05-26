#!/usr/bin/env python
"""Evaluate 20.90 liftoff/sensor-offset diagnostic rows with raw/calibrated input.

The model is the recovered 20.77/20.85 baseline artifact. Diagnostic rows come
from the explicit 20.90 COMSOL pack plus nominal-row postprocess axis shifts.
Calibration is fixed to the 20.89 validation-selected per_axis_rms_train_stats
protocol and is reported only as a diagnostic caveat.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import audit_true_3d_rbc_gain_calibration_strategies as cal
import audit_true_3d_rbc_observation_perturbation_robustness as obs
from load_true_3d_rbc_pilot_dataset import (
    ROOT,
    V3_240_DATASET_ID,
    depth_grid_from_params,
    depth_map_from_params,
    gate_manifest,
    load_dataset,
    mask_metrics,
    projected_mask_from_params,
    resolve_dataset,
    split_indices,
    write_csv,
)


PLAN_CSV = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_plan.csv"
DIAGNOSTIC_NPZ = (
    ROOT
    / "data/comsol_mfl/generated/true_3d_rbc_liftoff_sensor_offset_diagnostic_pack/true_3d_rbc_liftoff_sensor_offset_diagnostic_pack.npz"
)
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_sensor_offset_robustness_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_robustness_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_group_summary.csv"

BASELINE_PROFILE_RMSE = 0.000387737
BASELINE_DICE = 0.847727

POSTPROCESS_SHIFTS = {
    "axis_misalignment_x_light": {"Bx": 1, "By": -1, "Bz": 0},
    "axis_misalignment_x_hard": {"Bx": 2, "By": -2, "Bz": 1},
    "axis_misalignment_x_reverse": {"Bx": -1, "By": 1, "Bz": 0},
}

METRIC_FIELDS = [
    "diagnostic_row_id",
    "base_sample_id",
    "variant_name",
    "factor_group",
    "input_mode",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "sensor_z_m",
    "scan_bundle_y_offset_m",
    "jscale",
    "misalignment_shifts_json",
    "normalized_param_mae_mean",
    "dimension_param_mae_norm",
    "curvature_param_mae_norm",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wMAE_auxiliary",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "profile_depth_rmse_m",
    "profile_depth_rmse_degradation_pct_vs_nominal",
    "profile_depth_rmse_degradation_pct_vs_20_85",
    "er_like_profile_error",
    "er_like_degradation_pct_vs_nominal",
    "max_depth_error_m",
    "volume_proxy_rel_error",
    "projected_mask_iou",
    "projected_mask_dice",
    "projected_mask_dice_drop_vs_nominal",
    "projected_mask_dice_drop_vs_20_85",
    "raw_vs_calibrated_profile_rmse_improvement_pct",
    "status_band",
]

GROUP_FIELDS = [
    "input_mode",
    "factor_group",
    "variant_name",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "profile_depth_rmse_m",
    "profile_depth_rmse_degradation_pct_vs_nominal",
    "er_like_profile_error",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wMAE_auxiliary",
    "projected_mask_iou",
    "projected_mask_dice",
    "projected_mask_dice_drop_vs_nominal",
    "raw_vs_calibrated_profile_rmse_improvement_pct",
    "status_band",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 20.90 true-3D RBC liftoff/sensor-offset robustness.")
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--diagnostic-npz", type=Path, default=DIAGNOSTIC_NPZ)
    parser.add_argument("--artifact-manifest", type=Path, default=obs.ARTIFACT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None, ""} and not math.isnan(float(row[key]))]
    return float(np.mean(values)) if values else math.nan


def pct(value: float, baseline: float) -> float:
    return 0.0 if abs(baseline) < 1.0e-20 else 100.0 * (value - baseline) / baseline


def status_band(profile_degradation_pct: float, dice_drop: float) -> str:
    if profile_degradation_pct <= 10.0 and dice_drop <= 0.02:
        return "green"
    if profile_degradation_pct <= 25.0 and dice_drop <= 0.05:
        return "warning"
    return "fail"


def axis_index(axis: str) -> int:
    return {"Bx": 0, "By": 1, "Bz": 2}[axis]


def shift_axes(delta: np.ndarray, shifts: dict[str, int]) -> np.ndarray:
    shaped = np.asarray(delta, dtype=np.float32).reshape(1, 3, 3, delta.shape[-1]).copy()
    for axis, shift in shifts.items():
        shaped[:, axis_index(axis), :, :] = obs.shift_signal(shaped[:, axis_index(axis), :, :], float(shift))
    return shaped.reshape(3, 3, delta.shape[-1]).astype(np.float32)


def load_comsol_rows(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def append_postprocess_rows(arrays: dict[str, np.ndarray], plan_rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    row_count = int(arrays["delta_b"].shape[0])
    nominal_by_base: dict[str, int] = {}
    for idx, (base_id, variant) in enumerate(zip(arrays["base_sample_ids"].astype(str), arrays["variant_name"].astype(str))):
        if variant == "nominal":
            nominal_by_base[str(base_id)] = idx
    post_rows = [row for row in plan_rows if row.get("requires_comsol", "").lower() == "false"]
    row_keys = [key for key, value in arrays.items() if np.asarray(value).shape[:1] == (row_count,)]
    extras: dict[str, list[Any]] = {key: [] for key in row_keys}
    for row in post_rows:
        base_id = row["base_sample_id"]
        if base_id not in nominal_by_base:
            continue
        src = nominal_by_base[base_id]
        shifts = json.loads(row.get("misalignment_shifts_json") or "{}")
        for key in row_keys:
            value = arrays[key]
            if key == "delta_b":
                extras[key].append(shift_axes(value[src], shifts))
            elif key == "b_defect":
                extras[key].append(value[src])  # raw fields remain nominal; x input uses shifted delta_b.
            elif key == "sample_ids":
                extras[key].append(row["diagnostic_row_id"])
            elif key == "variant_name":
                extras[key].append(row["variant_name"])
            elif key == "factor_group":
                extras[key].append(row["factor_group"])
            elif key == "sensor_z_m":
                extras[key].append(float(row["sensor_z_m"]))
            elif key == "scan_line_y":
                extras[key].append(np.asarray(json.loads(row["scan_line_y_json"]), dtype=np.float32))
            elif key == "jscale":
                extras[key].append(float(row["jscale"]))
            elif key == "scan_bundle_y_offset_m":
                extras[key].append(float(row["scan_bundle_y_offset_m"]))
            elif key == "misalignment_shifts_json":
                extras[key].append(row.get("misalignment_shifts_json", "{}"))
            else:
                extras[key].append(value[src])
    merged: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key in extras and extras[key]:
            merged[key] = np.concatenate([value, np.asarray(extras[key], dtype=value.dtype)], axis=0)
        else:
            merged[key] = value
    return merged


def diagnostic_dataset(arrays: dict[str, np.ndarray], source_dataset: Any) -> Any:
    delta_b = np.asarray(arrays["delta_b"], dtype=np.float32)
    return SimpleNamespace(
        dataset_id="true_3d_rbc_liftoff_sensor_offset_diagnostic_pack",
        delta_b=delta_b,
        x_channels=delta_b.reshape(delta_b.shape[0], 9, delta_b.shape[-1]).astype(np.float32),
        b_defect=np.asarray(arrays.get("b_defect", delta_b), dtype=np.float32),
        b_no_defect=np.asarray(arrays.get("b_no_defect", np.zeros_like(delta_b)), dtype=np.float32),
        rbc_params=np.asarray(arrays["rbc_params"], dtype=np.float32).reshape(delta_b.shape[0], 6),
        profile_pose=np.asarray(arrays["profile_pose"], dtype=np.float32).reshape(delta_b.shape[0], 6),
        projected_mask_2d=np.asarray(arrays["projected_mask_2d"], dtype=np.uint8),
        profile_depth_grid_m=np.asarray(arrays["profile_depth_grid_m"], dtype=np.float32),
        profile_depth_map_xy_m=np.asarray(arrays["profile_depth_map_xy_m"], dtype=np.float32),
        sample_ids=np.asarray(arrays["sample_ids"]).astype(str),
        split=np.asarray(arrays["split"]).astype(str),
        axis_names=list(source_dataset.axis_names),
        sensor_x=np.asarray(arrays.get("sensor_x", source_dataset.sensor_x), dtype=np.float32),
        scan_line_y=np.asarray(arrays.get("scan_line_y", source_dataset.scan_line_y), dtype=np.float32),
        sensor_z_m=0.008,
        curvature_template=np.asarray(arrays["curvature_template"]).astype(str),
        depth_bin=np.asarray(arrays["depth_bin"]).astype(str),
        aspect_bin=np.asarray(arrays["aspect_bin"]).astype(str),
        size_bin=np.asarray(arrays["size_bin"]).astype(str),
    )


def evaluate_rows(source_dataset: Any, dataset: Any, pred_raw: np.ndarray, stats: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    train = split_indices(source_dataset)["train"]
    low = source_dataset.rbc_params[train].min(axis=0)
    high = source_dataset.rbc_params[train].max(axis=0)
    pred = np.clip(np.asarray(pred_raw, dtype=np.float32), low[None, :], high[None, :]).astype(np.float32)
    pred_norm = (pred - stats["y_mean"]) / stats["y_std"]
    true_norm = (dataset.rbc_params - stats["y_mean"]) / stats["y_std"]
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(dataset.sample_ids):
        pred_depth = depth_grid_from_params(pred[idx])
        true_depth = dataset.profile_depth_grid_m[idx]
        denom = float(np.sum(true_depth**2))
        er_like = 0.0 if denom <= 1.0e-20 else float(np.sqrt(np.sum((pred_depth - true_depth) ** 2) / denom))
        pred_map = depth_map_from_params(pred[idx], dataset.profile_pose[idx])
        true_volume = float(dataset.profile_depth_map_xy_m[idx].sum())
        pred_volume = float(pred_map.sum())
        mask = mask_metrics(projected_mask_from_params(pred[idx], dataset.profile_pose[idx]), dataset.projected_mask_2d[idx])
        abs_param = np.abs(pred[idx] - dataset.rbc_params[idx])
        norm_abs = np.abs(pred_norm[idx] - true_norm[idx])
        rows.append(
            {
                "sample_id": str(sample_id),
                "split": str(dataset.split[idx]),
                "curvature_template": str(dataset.curvature_template[idx]),
                "depth_bin": str(dataset.depth_bin[idx]),
                "aspect_bin": str(dataset.aspect_bin[idx]),
                "size_bin": str(dataset.size_bin[idx]),
                "normalized_param_mae_mean": float(np.mean(norm_abs)),
                "dimension_param_mae_norm": float(np.mean(norm_abs[:3])),
                "curvature_param_mae_norm": float(np.mean(norm_abs[3:])),
                "L_mae_mm": float(abs_param[0] * 1000.0),
                "W_mae_mm": float(abs_param[1] * 1000.0),
                "D_mae_mm": float(abs_param[2] * 1000.0),
                "wLD_abs_error": float(abs_param[3]),
                "wWD_abs_error": float(abs_param[4]),
                "wLW_abs_error": float(abs_param[5]),
                "wMAE_auxiliary": float(np.mean(abs_param[3:])),
                "profile_depth_rmse_m": float(np.sqrt(np.mean((pred_depth - true_depth) ** 2))),
                "er_like_profile_error": er_like,
                "max_depth_error_m": float(abs_param[2]),
                "volume_proxy_rel_error": 0.0 if abs(true_volume) < 1.0e-12 else abs(pred_volume - true_volume) / abs(true_volume),
                "projected_mask_iou": mask["iou"],
                "projected_mask_dice": mask["dice"],
            }
        )
    return rows


def enrich_metric_rows(rows: list[dict[str, Any]], arrays: dict[str, np.ndarray], input_mode: str) -> list[dict[str, Any]]:
    nominal: dict[tuple[str, str], dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        base_id = str(arrays["base_sample_ids"][idx])
        if str(arrays["variant_name"][idx]) == "nominal":
            nominal[(base_id, input_mode)] = row
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        base_id = str(arrays["base_sample_ids"][idx])
        nom = nominal.get((base_id, input_mode), row)
        enriched = dict(row)
        enriched.update(
            {
                "diagnostic_row_id": str(arrays["sample_ids"][idx]),
                "base_sample_id": base_id,
                "variant_name": str(arrays["variant_name"][idx]),
                "factor_group": str(arrays["factor_group"][idx]),
                "input_mode": input_mode,
                "sensor_z_m": float(np.asarray(arrays["sensor_z_m"]).reshape(-1)[idx]) if np.asarray(arrays["sensor_z_m"]).ndim else float(arrays["sensor_z_m"]),
                "scan_bundle_y_offset_m": float(np.asarray(arrays.get("scan_bundle_y_offset_m", np.zeros(len(rows)))).reshape(-1)[idx]),
                "jscale": float(np.asarray(arrays.get("jscale", np.ones(len(rows)))).reshape(-1)[idx]),
                "misalignment_shifts_json": str(np.asarray(arrays.get("misalignment_shifts_json", np.asarray(["{}"] * len(rows)))).astype(str)[idx]),
            }
        )
        enriched["profile_depth_rmse_degradation_pct_vs_nominal"] = pct(
            float(enriched["profile_depth_rmse_m"]), float(nom["profile_depth_rmse_m"])
        )
        enriched["profile_depth_rmse_degradation_pct_vs_20_85"] = pct(float(enriched["profile_depth_rmse_m"]), BASELINE_PROFILE_RMSE)
        enriched["er_like_degradation_pct_vs_nominal"] = pct(float(enriched["er_like_profile_error"]), float(nom["er_like_profile_error"]))
        enriched["projected_mask_dice_drop_vs_nominal"] = float(nom["projected_mask_dice"]) - float(enriched["projected_mask_dice"])
        enriched["projected_mask_dice_drop_vs_20_85"] = BASELINE_DICE - float(enriched["projected_mask_dice"])
        enriched["raw_vs_calibrated_profile_rmse_improvement_pct"] = ""
        enriched["status_band"] = status_band(
            float(enriched["profile_depth_rmse_degradation_pct_vs_nominal"]),
            float(enriched["projected_mask_dice_drop_vs_nominal"]),
        )
        out.append(enriched)
    return out


def add_raw_calibrated_improvement(all_rows: list[dict[str, Any]]) -> None:
    raw_by_id = {row["diagnostic_row_id"]: row for row in all_rows if row["input_mode"] == "raw"}
    for row in all_rows:
        if row["input_mode"] != "calibrated":
            continue
        raw = raw_by_id.get(row["diagnostic_row_id"])
        if raw is None:
            continue
        raw_rmse = float(raw["profile_depth_rmse_m"])
        row["raw_vs_calibrated_profile_rmse_improvement_pct"] = 0.0 if raw_rmse <= 1.0e-20 else 100.0 * (raw_rmse - float(row["profile_depth_rmse_m"])) / raw_rmse


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for input_mode in sorted({row["input_mode"] for row in rows}):
        for factor in sorted({row["factor_group"] for row in rows}):
            for variant in sorted({row["variant_name"] for row in rows if row["factor_group"] == factor}):
                subset_base = [row for row in rows if row["input_mode"] == input_mode and row["factor_group"] == factor and row["variant_name"] == variant]
                for split_name in sorted({row["split"] for row in subset_base}):
                    split_rows = [row for row in subset_base if row["split"] == split_name]
                    for group_field, values in [
                        ("all", ["all"]),
                        ("curvature_template", sorted({row["curvature_template"] for row in split_rows})),
                        ("depth_bin", sorted({row["depth_bin"] for row in split_rows})),
                    ]:
                        for value in values:
                            subset = split_rows if group_field == "all" else [row for row in split_rows if row[group_field] == value]
                            if not subset:
                                continue
                            degradation = mean(subset, "profile_depth_rmse_degradation_pct_vs_nominal")
                            dice_drop = mean(subset, "projected_mask_dice_drop_vs_nominal")
                            out.append(
                                {
                                    "input_mode": input_mode,
                                    "factor_group": factor,
                                    "variant_name": variant,
                                    "split": split_name,
                                    "group_field": group_field,
                                    "group_value": value,
                                    "sample_count": len(subset),
                                    "profile_depth_rmse_m": mean(subset, "profile_depth_rmse_m"),
                                    "profile_depth_rmse_degradation_pct_vs_nominal": degradation,
                                    "er_like_profile_error": mean(subset, "er_like_profile_error"),
                                    "L_mae_mm": mean(subset, "L_mae_mm"),
                                    "W_mae_mm": mean(subset, "W_mae_mm"),
                                    "D_mae_mm": mean(subset, "D_mae_mm"),
                                    "wMAE_auxiliary": mean(subset, "wMAE_auxiliary"),
                                    "projected_mask_iou": mean(subset, "projected_mask_iou"),
                                    "projected_mask_dice": mean(subset, "projected_mask_dice"),
                                    "projected_mask_dice_drop_vs_nominal": dice_drop,
                                    "raw_vs_calibrated_profile_rmse_improvement_pct": mean(subset, "raw_vs_calibrated_profile_rmse_improvement_pct"),
                                    "status_band": status_band(degradation, dice_drop),
                                }
                            )
    return out


def write_summary(path: Path, rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> None:
    raw_groups = [row for row in groups if row["input_mode"] == "raw" and row["split"] == "test" and row["group_field"] == "all"]
    calibrated_groups = [row for row in groups if row["input_mode"] == "calibrated" and row["split"] == "test" and row["group_field"] == "all"]
    worst_raw = max(raw_groups, key=lambda r: float(r["profile_depth_rmse_degradation_pct_vs_nominal"])) if raw_groups else None
    source_raw = [row for row in raw_groups if row["factor_group"] == "source_amplitude"]
    source_cal = [row for row in calibrated_groups if row["factor_group"] == "source_amplitude"]
    lines = [
        "20.90 true 3D RBC liftoff / sensor-offset robustness summary",
        "",
        f"metric_rows: {len(rows)}",
        f"input_modes: {sorted({row['input_mode'] for row in rows})}",
        "calibration_protocol: per_axis_rms_train_stats from 20.89; diagnostic caveat only, not baseline replacement",
        "baseline_artifact_manifest: results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json",
        "latest_newest_npz_scan: false",
        "training_run: false",
        "baseline_update: false",
        "",
        f"worst_raw_test_factor: {worst_raw['factor_group']} / {worst_raw['variant_name']} degradation={float(worst_raw['profile_depth_rmse_degradation_pct_vs_nominal']):.3f}% dice_drop={float(worst_raw['projected_mask_dice_drop_vs_nominal']):.6f}" if worst_raw else "worst_raw_test_factor: unavailable",
        f"source_raw_mean_degradation_pct: {float(np.mean([float(row['profile_depth_rmse_degradation_pct_vs_nominal']) for row in source_raw])):.3f}" if source_raw else "source_raw_mean_degradation_pct: unavailable",
        f"source_calibrated_mean_degradation_pct: {float(np.mean([float(row['profile_depth_rmse_degradation_pct_vs_nominal']) for row in source_cal])):.3f}" if source_cal else "source_calibrated_mean_degradation_pct: unavailable",
        "",
        "Boundary: spatial misalignment rows are nominal COMSOL rows shifted in postprocess; they are sensor-model diagnostics, not extra COMSOL solves.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    check_overwrite([args.summary, args.metrics, args.group_summary], args.overwrite)
    if args.artifact_manifest.resolve() != obs.ARTIFACT_MANIFEST.resolve():
        raise ValueError(f"20.90 must use 20.88a artifact manifest at {obs.ARTIFACT_MANIFEST}")
    entry, manifest, npz_path = resolve_dataset(args.dataset_id)
    gate_checks = gate_manifest(entry, manifest, npz_path, args.dataset_id)
    failed = [row for row in gate_checks if not row["pass"]]
    if failed:
        raise RuntimeError(f"registry/manifest gate failed: {failed}")
    source_dataset = load_dataset(args.dataset_id)
    plan_rows = read_csv(args.plan_csv)
    arrays = append_postprocess_rows(load_comsol_rows(args.diagnostic_npz), plan_rows)
    diag = diagnostic_dataset(arrays, source_dataset)

    artifact, checkpoint, model = obs.load_artifact(args.artifact_manifest)
    eval_ctx = obs.make_context(source_dataset, checkpoint)
    cal_ctx = cal.build_calibration_context(source_dataset)
    stats = {"y_mean": np.asarray(eval_ctx["y_mean"]), "y_std": np.asarray(eval_ctx["y_std"])}

    raw_pred = obs.predict(model, diag.x_channels, eval_ctx)
    raw_rows = enrich_metric_rows(evaluate_rows(source_dataset, diag, raw_pred, stats), arrays, "raw")
    calibrated_x = cal.per_axis_rms_train_stats(diag.x_channels, cal_ctx)
    calibrated_pred = obs.predict(model, calibrated_x, eval_ctx)
    calibrated_rows = enrich_metric_rows(evaluate_rows(source_dataset, diag, calibrated_pred, stats), arrays, "calibrated")
    all_rows = raw_rows + calibrated_rows
    add_raw_calibrated_improvement(all_rows)
    groups = group_summary(all_rows)
    write_csv(args.metrics, all_rows, METRIC_FIELDS)
    write_csv(args.group_summary, groups, GROUP_FIELDS)
    write_summary(args.summary, all_rows, groups)
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
