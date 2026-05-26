#!/usr/bin/env python
"""Observation perturbation robustness audit for the true-3D RBC baseline.

This script evaluates the recovered 20.77/20.85 baseline inference artifact on
in-memory perturbations of v3_240 delta_b. It does not run COMSOL, train, write
NPZ data, or modify CURRENT_BASELINE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import train_true_3d_rbc_neural_parameter_gate as gate
from load_true_3d_rbc_pilot_dataset import (
    ROOT,
    V3_240_DATASET_ID,
    aggregate_prediction_rows,
    check_no_overwrite,
    denormalize_y,
    evaluate_param_predictions,
    gate_manifest,
    load_dataset,
    resolve_dataset,
    sha256_file,
    split_indices,
    write_csv,
)
from run_true_3d_rbc_formal_benchmark_20_77_candidate import add_profile_error_rows


ARTIFACT_MANIFEST = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_observation_robustness_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_observation_perturbation_robustness_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_observation_perturbation_robustness_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/true_3d_rbc_observation_perturbation_robustness_group_summary.csv"
FAILURE_CASES = ROOT / "results/metrics/true_3d_rbc_observation_perturbation_failure_cases.csv"

clean: dict[str, float] = {}

METRIC_FIELDS = [
    "perturbation_name",
    "perturbation_group",
    "severity",
    "affected_axis",
    "split",
    "sample_count",
    "profile_depth_rmse_m",
    "profile_depth_rmse_degradation_pct",
    "er_like_profile_error",
    "er_like_degradation_pct",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "max_lwd_degradation_pct",
    "wMAE_auxiliary",
    "projected_mask_iou",
    "projected_mask_dice",
    "projected_mask_dice_drop",
    "max_depth_error_m",
    "volume_proxy_rel_error",
    "status_band",
    "notes",
]

GROUP_FIELDS = [
    "perturbation_name",
    "perturbation_group",
    "severity",
    "affected_axis",
    "split",
    "group_field",
    "group_value",
    "sample_count",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "projected_mask_dice",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wMAE_auxiliary",
]

FAILURE_FIELDS = [
    "perturbation_name",
    "perturbation_group",
    "severity",
    "affected_axis",
    "failure_type",
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "projected_mask_dice",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wMAE_auxiliary",
    "notes",
]


class Perturbation:
    def __init__(
        self,
        name: str,
        group: str,
        severity: str,
        affected_axis: str,
        apply: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
        notes: str = "",
    ) -> None:
        self.name = name
        self.group = group
        self.severity = severity
        self.affected_axis = affected_axis
        self.apply = apply
        self.notes = notes


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stable_seed(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
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


def as_axes(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(x.shape[0], 3, 3, x.shape[-1]).copy()


def flatten_axes(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(x.shape[0], 9, x.shape[-1])


def shift_signal(x: np.ndarray, shift: float) -> np.ndarray:
    xp = np.arange(x.shape[-1], dtype=np.float32)
    query = xp - float(shift)
    flat = x.reshape(-1, x.shape[-1])
    shifted = np.empty_like(flat)
    for idx, row in enumerate(flat):
        shifted[idx] = np.interp(query, xp, row, left=row[0], right=row[-1]).astype(np.float32)
    return shifted.reshape(x.shape).astype(np.float32)


def load_artifact(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], gate.RBCConvRegressor]:
    manifest = read_json(manifest_path)
    if manifest.get("dataset_id") != V3_240_DATASET_ID:
        raise RuntimeError(f"unexpected artifact dataset_id: {manifest.get('dataset_id')}")
    checkpoint_path = Path(manifest["checkpoint_path"])
    prediction_path = Path(manifest["prediction_artifact_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    if sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("checkpoint sha256 mismatch")
    if sha256_file(prediction_path) != manifest["prediction_artifact_sha256"]:
        raise RuntimeError("prediction artifact sha256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = gate.RBCConvRegressor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return manifest, checkpoint, model


def make_context(dataset: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    train_idx = split_indices(dataset)["train"]
    train_x = dataset.x_channels[train_idx]
    return {
        "x_mean": checkpoint["normalization"]["x_mean"],
        "x_std": checkpoint["normalization"]["x_std"],
        "y_mean": checkpoint["normalization"]["y_mean"],
        "y_std": checkpoint["normalization"]["y_std"],
        "train_rms": float(np.sqrt(np.mean(train_x**2))),
        "train_abs_peak": float(np.max(np.abs(train_x))),
    }


def perturbations() -> list[Perturbation]:
    items: list[Perturbation] = []
    for level in [0, 5, 10, 15, 20]:
        def apply_noise(x: np.ndarray, ctx: dict[str, Any], level: int = level) -> np.ndarray:
            if level == 0:
                return x.copy()
            rng = np.random.default_rng(stable_seed(f"additive_noise_{level}"))
            return (x + rng.normal(0.0, (level / 100.0) * ctx["train_rms"], size=x.shape)).astype(np.float32)
        items.append(Perturbation(f"additive_noise_{level}pct", "additive_noise", f"{level}pct", "all", apply_noise))

    for gain in [0.8, 0.9, 1.1, 1.2]:
        items.append(Perturbation(f"gain_scaling_{gain:.1f}x", "gain_scaling", f"{gain:.1f}x", "all", lambda x, _ctx, gain=gain: (x * gain).astype(np.float32)))

    for axis in ["Bx", "By", "Bz"]:
        for gain in [0.9, 1.1]:
            def apply_axis_gain(x: np.ndarray, _ctx: dict[str, Any], axis: str = axis, gain: float = gain) -> np.ndarray:
                shaped = as_axes(x)
                shaped[:, axis_index(axis), :, :] *= gain
                return flatten_axes(shaped)
            items.append(Perturbation(f"per_axis_gain_{axis}_{gain:.1f}x", "per_axis_gain", f"{gain:.1f}x", axis, apply_axis_gain))

    for level in [1, 2]:
        def apply_drift(x: np.ndarray, ctx: dict[str, Any], level: int = level) -> np.ndarray:
            return (x + (level / 100.0) * ctx["train_abs_peak"]).astype(np.float32)
        items.append(Perturbation(f"zero_drift_{level}pct_peak", "zero_drift", f"{level}pct_train_abs_peak", "all", apply_drift))

    for level in [1, 2]:
        def apply_reference_error(x: np.ndarray, ctx: dict[str, Any], level: int = level) -> np.ndarray:
            shaped = as_axes(x)
            t = np.linspace(-1.0, 1.0, x.shape[-1], dtype=np.float32)
            base = (level / 100.0) * ctx["train_abs_peak"]
            for a in range(3):
                for line in range(3):
                    offset = (0.35 + 0.1 * a + 0.05 * line) * base
                    trend = (0.65 + 0.05 * a - 0.03 * line) * base * t
                    shaped[:, a, line, :] += offset + trend
            return flatten_axes(shaped)
        items.append(Perturbation(f"no_defect_reference_error_{level}pct_peak", "no_defect_reference_error", f"{level}pct_train_abs_peak", "all", apply_reference_error, "low-frequency offset/trend residual"))

    for shift in [-2, -1, 1, 2]:
        items.append(Perturbation(f"sensor_x_resampling_jitter_{shift:+d}sample", "sensor_x_resampling_jitter", f"{shift:+d}sample", "all", lambda x, _ctx, shift=shift: shift_signal(x, shift), "integer-sample interpolation shift"))

    for axis in ["Bx", "By", "Bz"]:
        def apply_missing(x: np.ndarray, _ctx: dict[str, Any], axis: str = axis) -> np.ndarray:
            shaped = as_axes(x)
            shaped[:, axis_index(axis), :, :] = 0.0
            return flatten_axes(shaped)
        items.append(Perturbation(f"channel_dropout_{axis}_missing", "channel_attenuation_dropout", "missing", axis, apply_missing))
        def apply_attenuation(x: np.ndarray, _ctx: dict[str, Any], axis: str = axis) -> np.ndarray:
            shaped = as_axes(x)
            shaped[:, axis_index(axis), :, :] *= 0.5
            return flatten_axes(shaped)
        items.append(Perturbation(f"channel_attenuation_{axis}_50pct", "channel_attenuation_dropout", "50pct", axis, apply_attenuation))

    def combined_light(x: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        rng = np.random.default_rng(stable_seed("combined_light"))
        out = (x * 1.1 + rng.normal(0.0, 0.05 * ctx["train_rms"], size=x.shape)).astype(np.float32)
        return shift_signal(out, 1)
    items.append(Perturbation("combined_light", "combined", "noise5_gain10_jitter1", "all", combined_light))

    def combined_hard(x: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
        rng = np.random.default_rng(stable_seed("combined_hard"))
        out = (x * 1.2 + rng.normal(0.0, 0.15 * ctx["train_rms"], size=x.shape)).astype(np.float32)
        out = shift_signal(out, 2)
        return (out + 0.02 * ctx["train_abs_peak"]).astype(np.float32)
    items.append(Perturbation("combined_hard", "combined", "noise15_gain20_jitter2_drift2", "all", combined_hard))
    return items


def predict(model: gate.RBCConvRegressor, x_raw: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    x_norm = ((x_raw - ctx["x_mean"]) / ctx["x_std"]).astype(np.float32)
    pred_norm = gate.predict_norm(model, x_norm)
    return denormalize_y(pred_norm, {"y_mean": ctx["y_mean"], "y_std": ctx["y_std"]})


def split_metrics(rows: list[dict[str, Any]], perturb: Perturbation, split_name: str) -> dict[str, Any]:
    agg = aggregate_prediction_rows(rows, perturb.name, split_name)
    subset = [row for row in rows if row["split"] == split_name]
    profile_rmse = float(agg["profile_depth_rmse_m_mean"])
    er_like = mean(subset, "er_like_profile_error")
    dice = float(agg["projected_mask_dice_mean"])
    l_degradation = pct(float(agg["L_mae_mm_mean"]), clean["L_mae_mm"])
    w_degradation = pct(float(agg["W_mae_mm_mean"]), clean["W_mae_mm"])
    d_degradation = pct(float(agg["D_mae_mm_mean"]), clean["D_mae_mm"])
    profile_degradation = pct(profile_rmse, clean["profile_depth_rmse_m"])
    dice_drop = clean["projected_mask_dice"] - dice
    return {
        "perturbation_name": perturb.name,
        "perturbation_group": perturb.group,
        "severity": perturb.severity,
        "affected_axis": perturb.affected_axis,
        "split": split_name,
        "sample_count": int(agg["sample_count"]),
        "profile_depth_rmse_m": profile_rmse,
        "profile_depth_rmse_degradation_pct": profile_degradation,
        "er_like_profile_error": er_like,
        "er_like_degradation_pct": pct(er_like, clean["er_like_profile_error"]),
        "L_mae_mm": float(agg["L_mae_mm_mean"]),
        "W_mae_mm": float(agg["W_mae_mm_mean"]),
        "D_mae_mm": float(agg["D_mae_mm_mean"]),
        "max_lwd_degradation_pct": max(l_degradation, w_degradation, d_degradation),
        "wMAE_auxiliary": float(agg["curvature_mae_mean_mean"]),
        "projected_mask_iou": float(agg["projected_mask_iou_mean"]),
        "projected_mask_dice": dice,
        "projected_mask_dice_drop": dice_drop,
        "max_depth_error_m": mean(subset, "max_depth_error_m"),
        "volume_proxy_rel_error": float(agg["volume_proxy_rel_error_mean"]),
        "status_band": status_band(profile_degradation, dice_drop) if split_name == "test" else "",
        "notes": perturb.notes,
    }


def group_metrics(rows: list[dict[str, Any]], perturb: Perturbation) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        for field in ("curvature_template", "depth_bin", "aspect_bin", "size_bin"):
            for value in sorted({str(row[field]) for row in split_rows}):
                subset = [row for row in split_rows if str(row[field]) == value]
                out.append(
                    {
                        "perturbation_name": perturb.name,
                        "perturbation_group": perturb.group,
                        "severity": perturb.severity,
                        "affected_axis": perturb.affected_axis,
                        "split": split_name,
                        "group_field": field,
                        "group_value": value,
                        "sample_count": len(subset),
                        "profile_depth_rmse_m": mean(subset, "profile_depth_rmse_m"),
                        "er_like_profile_error": mean(subset, "er_like_profile_error"),
                        "projected_mask_dice": mean(subset, "projected_mask_dice"),
                        "L_mae_mm": mean(subset, "L_mae_mm"),
                        "W_mae_mm": mean(subset, "W_mae_mm"),
                        "D_mae_mm": mean(subset, "D_mae_mm"),
                        "wMAE_auxiliary": mean(subset, "curvature_mae_mean"),
                    }
                )
    return out


def failure_rows(rows: list[dict[str, Any]], perturb: Perturbation) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row["split"] == "test"]
    tagged: list[tuple[str, dict[str, Any]]] = []
    tagged.extend(("worst_profile_rmse", row) for row in sorted(test_rows, key=lambda r: float(r["profile_depth_rmse_m"]), reverse=True)[:3])
    tagged.extend(("worst_projected_dice", row) for row in sorted(test_rows, key=lambda r: float(r["projected_mask_dice"]))[:3])
    tagged.extend(("worst_lwd_error", row) for row in sorted(test_rows, key=lambda r: float(r["L_mae_mm"]) + float(r["W_mae_mm"]) + float(r["D_mae_mm"]), reverse=True)[:3])
    tagged.extend(("worst_w_auxiliary", row) for row in sorted(test_rows, key=lambda r: float(r["curvature_mae_mean"]), reverse=True)[:3])
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for failure_type, row in tagged:
        key = (failure_type, str(row["sample_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "perturbation_name": perturb.name,
                "perturbation_group": perturb.group,
                "severity": perturb.severity,
                "affected_axis": perturb.affected_axis,
                "failure_type": failure_type,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "curvature_template": row["curvature_template"],
                "depth_bin": row["depth_bin"],
                "aspect_bin": row["aspect_bin"],
                "profile_depth_rmse_m": row["profile_depth_rmse_m"],
                "er_like_profile_error": row["er_like_profile_error"],
                "projected_mask_dice": row["projected_mask_dice"],
                "L_mae_mm": row["L_mae_mm"],
                "W_mae_mm": row["W_mae_mm"],
                "D_mae_mm": row["D_mae_mm"],
                "wMAE_auxiliary": row["curvature_mae_mean"],
                "notes": perturb.notes,
            }
        )
    return out


def write_preflight(dataset: Any, manifest: dict[str, Any], artifact: dict[str, Any]) -> None:
    splits = split_indices(dataset)
    lines = [
        "20.88 true 3D RBC observation perturbation robustness preflight summary",
        "",
        f"dataset_id: {dataset.dataset_id}",
        "registry_manifest_gate_pass: true",
        f"npz_path_resolved_from_manifest: {dataset.npz_path}",
        f"artifact_manifest: {ARTIFACT_MANIFEST}",
        f"checkpoint_path: {artifact['checkpoint_path']}",
        f"prediction_artifact_path: {artifact['prediction_artifact_path']}",
        "baseline_artifact_available: true",
        "artifact_sha256_verified: true",
        f"input_shape: delta_b={list(dataset.delta_b.shape)}, conv1d={list(dataset.x_channels.shape)}",
        f"split_counts: {{'train': {len(splits['train'])}, 'val': {len(splits['val'])}, 'test': {len(splits['test'])}}}",
        f"artifact_seed: {artifact['seed']}",
        "latest_newest_npz_scan: false",
        "COMSOL_run: false",
        "training_run: false",
        "data_or_NPZ_written: false",
        "CURRENT_BASELINE_update: false",
        "stage_gate: pass",
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if args.artifact_manifest.resolve() != ARTIFACT_MANIFEST.resolve():
        raise ValueError(f"20.88 must use the recovered artifact manifest at {ARTIFACT_MANIFEST}; got {args.artifact_manifest}")
    check_no_overwrite([PREFLIGHT_SUMMARY, SUMMARY, METRICS, GROUP_SUMMARY, FAILURE_CASES], args.overwrite)
    entry, data_manifest, npz_path = resolve_dataset(args.dataset_id)
    gate_checks = gate_manifest(entry, data_manifest, npz_path, args.dataset_id)
    failed = [row for row in gate_checks if not row["pass"]]
    if failed:
        raise RuntimeError("registry/manifest gate failed")
    dataset = load_dataset(args.dataset_id)
    artifact, checkpoint, model = load_artifact(args.artifact_manifest)
    if checkpoint.get("seed") != 42:
        raise RuntimeError("artifact checkpoint is not fixed seed=42")
    ctx = make_context(dataset, checkpoint)
    clean.clear()
    clean.update({key: float(value) for key, value in artifact["test_metrics"].items() if isinstance(value, (int, float))})
    write_preflight(dataset, data_manifest, artifact)

    all_metric_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    all_failure_rows: list[dict[str, Any]] = []
    perts = perturbations()
    for perturb in perts:
        x_perturbed = perturb.apply(dataset.x_channels, ctx)
        pred_raw = predict(model, x_perturbed, ctx)
        rows = add_profile_error_rows(dataset, pred_raw, evaluate_param_predictions(dataset, pred_raw, {"y_mean": ctx["y_mean"], "y_std": ctx["y_std"]}))
        for split_name in ("train", "val", "test"):
            all_metric_rows.append(split_metrics(rows, perturb, split_name))
        all_group_rows.extend(group_metrics(rows, perturb))
        all_failure_rows.extend(failure_rows(rows, perturb))

    write_csv(METRICS, all_metric_rows, METRIC_FIELDS)
    write_csv(GROUP_SUMMARY, all_group_rows, GROUP_FIELDS)
    write_csv(FAILURE_CASES, all_failure_rows, FAILURE_FIELDS)

    test_rows = [row for row in all_metric_rows if row["split"] == "test"]
    worst_profile = max(test_rows, key=lambda r: float(r["profile_depth_rmse_degradation_pct"]))
    worst_dice = max(test_rows, key=lambda r: float(r["projected_mask_dice_drop"]))
    noise10 = next(row for row in test_rows if row["perturbation_name"] == "additive_noise_10pct")
    combined_hard = next(row for row in test_rows if row["perturbation_name"] == "combined_hard")
    channel_rows = [row for row in test_rows if row["perturbation_group"] == "channel_attenuation_dropout"]
    worst_channel = max(channel_rows, key=lambda r: float(r["profile_depth_rmse_degradation_pct"]))
    green_count = sum(1 for row in test_rows if row["status_band"] == "green")
    warning_count = sum(1 for row in test_rows if row["status_band"] == "warning")
    fail_count = sum(1 for row in test_rows if row["status_band"] == "fail")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.88 true 3D RBC observation perturbation robustness summary",
                "",
                f"dataset_id: {args.dataset_id}",
                f"artifact_manifest: {args.artifact_manifest}",
                "COMSOL_run: false",
                "training_run: false",
                "data_or_NPZ_written: false",
                "CURRENT_BASELINE_update: false",
                f"perturbation_count: {len(perts)}",
                f"test_status_counts: green={green_count}, warning={warning_count}, fail={fail_count}",
                f"clean_replay_profile_rmse_m: {next(row for row in test_rows if row['perturbation_name'] == 'additive_noise_0pct')['profile_depth_rmse_m']}",
                f"noise_10pct_profile_degradation_pct: {noise10['profile_depth_rmse_degradation_pct']:.6f}",
                f"noise_10pct_dice_drop: {noise10['projected_mask_dice_drop']:.6f}",
                f"combined_hard_profile_degradation_pct: {combined_hard['profile_depth_rmse_degradation_pct']:.6f}",
                f"combined_hard_dice_drop: {combined_hard['projected_mask_dice_drop']:.6f}",
                f"most_sensitive_profile: {worst_profile['perturbation_name']} ({worst_profile['profile_depth_rmse_degradation_pct']:.6f}% profile RMSE degradation)",
                f"most_sensitive_dice: {worst_dice['perturbation_name']} ({worst_dice['projected_mask_dice_drop']:.6f} Dice drop)",
                f"most_sensitive_channel_diagnostic: {worst_channel['perturbation_name']} ({worst_channel['profile_depth_rmse_degradation_pct']:.6f}% profile RMSE degradation)",
                "wMAE_usage: auxiliary diagnostic only",
                "Boundary: perturbations are observation-space diagnostics on delta_b, not claims about new COMSOL physics or real sensor validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--artifact-manifest", type=Path, default=ARTIFACT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
