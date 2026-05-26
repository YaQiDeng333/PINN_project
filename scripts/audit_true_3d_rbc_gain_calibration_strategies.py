"""Audit gain/amplitude calibration strategies for the true 3D RBC baseline.

This is a calibration-only evaluation. It loads the fixed 20.77/20.85
baseline checkpoint through the 20.88a artifact manifest, perturbs v3_240
delta_b in memory, applies candidate input calibrations in memory, and writes
summary/metric CSV files. It does not write data/NPZ/checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_true_3d_rbc_observation_perturbation_robustness as obs  # noqa: E402
import load_true_3d_rbc_pilot_dataset as loader  # noqa: E402


DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
ARTIFACT_MANIFEST = ROOT / "results" / "manifests" / "true_3d_rbc_baseline_inference_artifact_manifest.json"
OBS_METRICS = ROOT / "results" / "metrics" / "true_3d_rbc_observation_perturbation_robustness_metrics.csv"

PREFLIGHT_SUMMARY = ROOT / "results" / "summaries" / "true_3d_rbc_gain_calibration_preflight_summary.txt"
SUMMARY = ROOT / "results" / "summaries" / "true_3d_rbc_gain_calibration_strategy_summary.txt"
METRICS = ROOT / "results" / "metrics" / "true_3d_rbc_gain_calibration_strategy_metrics.csv"

PROFILE_BASELINE_RMSE = 0.000387737
BASELINE_DICE = 0.847727


@dataclass(frozen=True)
class CalibrationStrategy:
    name: str
    description: str
    transform: Callable[[np.ndarray, "CalibrationContext"], np.ndarray]


@dataclass
class CalibrationContext:
    dataset: loader.True3DRBCDataset
    train_indices: np.ndarray
    train_sample_rms_median: float
    train_axis_rms_median: np.ndarray
    train_peak_median: float
    train_robust_median: float
    train_robust_iqr: float
    train_global_abs_peak: float


def ensure_dirs() -> None:
    for path in [PREFLIGHT_SUMMARY, SUMMARY, METRICS]:
        path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def finite_rms(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=axis, keepdims=keepdims))


def clip_scale(scale: np.ndarray | float, lo: float = 0.5, hi: float = 2.0) -> np.ndarray | float:
    return np.clip(scale, lo, hi)


def build_calibration_context(dataset: loader.True3DRBCDataset) -> CalibrationContext:
    train_idx = loader.split_indices(dataset)["train"]
    x_train = dataset.x_channels[train_idx].astype(np.float64)
    sample_rms = finite_rms(x_train, axis=(1, 2))
    axis_rms = finite_rms(x_train.reshape(x_train.shape[0], 3, 3, x_train.shape[-1]), axis=(2, 3))
    sample_peak = np.max(np.abs(x_train), axis=(1, 2))
    q25, q75 = np.percentile(x_train, [25, 75])
    iqr = float(q75 - q25)
    return CalibrationContext(
        dataset=dataset,
        train_indices=train_idx,
        train_sample_rms_median=float(np.median(sample_rms)),
        train_axis_rms_median=np.median(axis_rms, axis=0),
        train_peak_median=float(np.median(sample_peak)),
        train_robust_median=float(np.median(x_train)),
        train_robust_iqr=max(iqr, 1e-12),
        train_global_abs_peak=float(np.max(np.abs(x_train))),
    )


def no_calibration(x: np.ndarray, ctx: CalibrationContext) -> np.ndarray:
    return x.astype(np.float32, copy=True)


def per_sample_rms_normalization(x: np.ndarray, ctx: CalibrationContext) -> np.ndarray:
    y = x.astype(np.float64, copy=True)
    rms = finite_rms(y, axis=(1, 2), keepdims=True)
    scale = clip_scale(ctx.train_sample_rms_median / np.maximum(rms, 1e-12))
    return (y * scale).astype(np.float32)


def per_axis_rms_train_stats(x: np.ndarray, ctx: CalibrationContext) -> np.ndarray:
    y = x.astype(np.float64, copy=True).reshape(x.shape[0], 3, 3, x.shape[-1])
    rms = finite_rms(y, axis=(2, 3), keepdims=True)
    target = ctx.train_axis_rms_median.reshape(1, 3, 1, 1)
    scale = clip_scale(target / np.maximum(rms, 1e-12))
    return (y * scale).reshape(x.shape).astype(np.float32)


def peak_amplitude_normalization(x: np.ndarray, ctx: CalibrationContext) -> np.ndarray:
    y = x.astype(np.float64, copy=True)
    peak = np.max(np.abs(y), axis=(1, 2), keepdims=True)
    scale = clip_scale(ctx.train_peak_median / np.maximum(peak, 1e-12))
    return (y * scale).astype(np.float32)


def robust_median_iqr_scale(x: np.ndarray, ctx: CalibrationContext) -> np.ndarray:
    y = x.astype(np.float64, copy=True)
    sample_median = np.median(y, axis=(1, 2), keepdims=True)
    q25 = np.percentile(y, 25, axis=(1, 2), keepdims=True)
    q75 = np.percentile(y, 75, axis=(1, 2), keepdims=True)
    sample_iqr = np.maximum(q75 - q25, 1e-12)
    scaled = (y - sample_median) / sample_iqr
    restored = scaled * ctx.train_robust_iqr + ctx.train_robust_median
    return restored.astype(np.float32)


def reference_proxy_global_gain_correction(x: np.ndarray, ctx: CalibrationContext) -> np.ndarray:
    # No independent no-defect amplitude channel is available in the artifact.
    # Use a delta_b RMS proxy and report it honestly as a proxy correction.
    return per_sample_rms_normalization(x, ctx)


def calibration_strategies() -> List[CalibrationStrategy]:
    return [
        CalibrationStrategy("no_calibration", "raw perturbed delta_b", no_calibration),
        CalibrationStrategy("per_sample_rms_normalization", "scale each sample to train median RMS", per_sample_rms_normalization),
        CalibrationStrategy("per_axis_rms_train_stats", "scale Bx/By/Bz separately to train median axis RMS", per_axis_rms_train_stats),
        CalibrationStrategy("peak_amplitude_normalization", "scale each sample to train median absolute peak", peak_amplitude_normalization),
        CalibrationStrategy("robust_median_iqr_scale", "match each sample robust median/IQR to train robust scale", robust_median_iqr_scale),
        CalibrationStrategy(
            "reference_proxy_global_gain_correction",
            "delta_b RMS proxy because independent no-defect scale is not available",
            reference_proxy_global_gain_correction,
        ),
    ]


def selected_perturbations() -> List[obs.Perturbation]:
    wanted = {
        "additive_noise_0pct": "clean",
        "gain_scaling_0.8x": "gain_scaling_0.8x",
        "gain_scaling_0.9x": "gain_scaling_0.9x",
        "gain_scaling_1.1x": "gain_scaling_1.1x",
        "gain_scaling_1.2x": "gain_scaling_1.2x",
        "per_axis_gain_Bx_minus10pct": "per_axis_gain_Bx_minus10pct",
        "per_axis_gain_Bx_plus10pct": "per_axis_gain_Bx_plus10pct",
        "per_axis_gain_By_minus10pct": "per_axis_gain_By_minus10pct",
        "per_axis_gain_By_plus10pct": "per_axis_gain_By_plus10pct",
        "per_axis_gain_Bz_minus10pct": "per_axis_gain_Bz_minus10pct",
        "per_axis_gain_Bz_plus10pct": "per_axis_gain_Bz_plus10pct",
        "channel_attenuation_Bx_50pct": "channel_attenuation_Bx_50pct",
        "combined_light": "combined_light",
        "combined_hard": "combined_hard",
    }
    found: List[obs.Perturbation] = []
    for perturbation in obs.perturbations():
        if perturbation.name in wanted:
            found.append(
                obs.Perturbation(
                    wanted[perturbation.name],
                    perturbation.group,
                    perturbation.severity,
                    perturbation.affected_axis,
                    perturbation.apply,
                )
            )
    return found


def split_metric_rows(
    predictions: np.ndarray,
    dataset: loader.True3DRBCDataset,
    ctx: dict[str, object],
    strategy: str,
    perturbation: obs.Perturbation,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    row_metrics = loader.evaluate_param_predictions(
        dataset,
        predictions,
        stats={"y_mean": np.asarray(ctx["y_mean"]), "y_std": np.asarray(ctx["y_std"])},
    )
    profile_rows = obs.add_profile_error_rows(dataset, predictions, row_metrics)
    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in profile_rows if row["split"] == split_name]
        if not split_rows:
            continue
        agg = loader.aggregate_prediction_rows(split_rows, f"{strategy}_{perturbation.name}", split_name)
        out = {
            "normalized_param_mae_mean": agg["normalized_param_mae_mean_mean"],
            "dimension_param_mae_norm": agg["dimension_param_mae_norm_mean"],
            "curvature_param_mae_norm": agg["curvature_param_mae_norm_mean"],
            "L_mae_mm": agg["L_mae_mm_mean"],
            "W_mae_mm": agg["W_mae_mm_mean"],
            "D_mae_mm": agg["D_mae_mm_mean"],
            "wLD_abs_error": agg["wLD_abs_error_mean"],
            "wWD_abs_error": agg["wWD_abs_error_mean"],
            "wLW_abs_error": agg["wLW_abs_error_mean"],
            "curvature_mae_mean": agg["curvature_mae_mean_mean"],
            "projected_mask_iou": agg["projected_mask_iou_mean"],
            "projected_mask_dice": agg["projected_mask_dice_mean"],
            "projected_mask_area_error": agg["projected_mask_area_error_mean"],
            "projected_mask_center_error_px": agg["projected_mask_center_error_px_mean"],
            "profile_depth_rmse_m": agg["profile_depth_rmse_m_mean"],
            "volume_proxy_rel_error": agg["volume_proxy_rel_error_mean"],
            "sample_count": agg["sample_count"],
            "er_like_profile_error": obs.mean(split_rows, "er_like_profile_error"),
            "max_depth_error_m": obs.mean(split_rows, "max_depth_error_m"),
        }
        out["strategy"] = strategy
        out["perturbation_name"] = perturbation.name
        out["perturbation_category"] = perturbation.group
        out["severity"] = perturbation.severity
        out["affected_axis"] = perturbation.affected_axis
        out["split"] = split_name
        out["profile_rmse_degradation_pct_vs_20_85"] = obs.pct(
            out["profile_depth_rmse_m"], PROFILE_BASELINE_RMSE
        )
        out["dice_drop_vs_20_85"] = BASELINE_DICE - float(out["projected_mask_dice"])
        out["clean_performance_drop_vs_20_85_pct"] = (
            obs.pct(out["profile_depth_rmse_m"], PROFILE_BASELINE_RMSE)
            if perturbation.name == "clean"
            else ""
        )
        rows.append(out)
    return rows


def evaluate_strategy(
    strategy: CalibrationStrategy,
    perturbation: obs.Perturbation,
    dataset: loader.True3DRBCDataset,
    model: object,
    eval_ctx: dict[str, object],
    cal_ctx: CalibrationContext,
) -> List[Dict[str, object]]:
    raw_x = perturbation.apply(dataset.x_channels.copy(), eval_ctx)
    calibrated_x = strategy.transform(raw_x, cal_ctx)
    predictions = obs.predict(model, calibrated_x, eval_ctx)
    return split_metric_rows(predictions, dataset, eval_ctx, strategy.name, perturbation)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_row(rows: Iterable[Dict[str, object]], strategy: str, perturbation: str) -> Dict[str, object] | None:
    for row in rows:
        if row.get("strategy") == strategy and row.get("perturbation_name") == perturbation and row.get("split") == "test":
            return row
    return None


def strategy_candidate(rows: List[Dict[str, object]], strategy: str, split_name: str) -> Dict[str, object]:
    reference_08 = test_row(rows, "no_calibration", "gain_scaling_0.8x")
    reference_12 = test_row(rows, "no_calibration", "gain_scaling_1.2x")
    if split_name != "test":
        reference_08 = next(
            row
            for row in rows
            if row.get("strategy") == "no_calibration"
            and row.get("perturbation_name") == "gain_scaling_0.8x"
            and row.get("split") == split_name
        )
        reference_12 = next(
            row
            for row in rows
            if row.get("strategy") == "no_calibration"
            and row.get("perturbation_name") == "gain_scaling_1.2x"
            and row.get("split") == split_name
        )
    def row_for(perturbation: str) -> Dict[str, object] | None:
        return next(
            (
                row
                for row in rows
                if row.get("strategy") == strategy
                and row.get("perturbation_name") == perturbation
                and row.get("split") == split_name
            ),
            None,
        )
    clean = row_for("clean")
    gain08 = row_for("gain_scaling_0.8x")
    gain12 = row_for("gain_scaling_1.2x")
    bx50 = row_for("channel_attenuation_Bx_50pct")
    if not (clean and gain08 and gain12):
        return {}
    clean_drop = float(clean["profile_rmse_degradation_pct_vs_20_85"])
    gain08_drop = float(gain08["profile_rmse_degradation_pct_vs_20_85"])
    gain12_drop = float(gain12["profile_rmse_degradation_pct_vs_20_85"])
    ref08_drop = float(reference_08["profile_rmse_degradation_pct_vs_20_85"]) if reference_08 else math.nan
    ref12_drop = float(reference_12["profile_rmse_degradation_pct_vs_20_85"]) if reference_12 else math.nan
    reduction08 = 100.0 * (ref08_drop - gain08_drop) / max(abs(ref08_drop), 1e-9)
    reduction12 = 100.0 * (ref12_drop - gain12_drop) / max(abs(ref12_drop), 1e-9)
    clean_dice = float(clean["projected_mask_dice"])
    score = clean_drop + max(gain08_drop, gain12_drop) - 0.25 * min(reduction08, reduction12)
    return {
        "strategy": strategy,
        "selection_split": split_name,
        "clean_drop_pct": clean_drop,
        "gain08_drop_pct": gain08_drop,
        "gain12_drop_pct": gain12_drop,
        "gain08_reduction_pct": reduction08,
        "gain12_reduction_pct": reduction12,
        "clean_dice": clean_dice,
        "bx50_drop_pct": float(bx50["profile_rmse_degradation_pct_vs_20_85"]) if bx50 else math.nan,
        "score": score,
    }


def calibration_decision(rows: List[Dict[str, object]]) -> Tuple[bool, str, Dict[str, object], Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for strategy in sorted({str(row["strategy"]) for row in rows}):
        if strategy == "no_calibration":
            continue
        candidate = strategy_candidate(rows, strategy, "val")
        if candidate:
            candidates.append(candidate)
    if not candidates:
        return False, "no_calibration_candidate", {}, {}
    best_val = sorted(candidates, key=lambda item: item["score"])[0]
    best_test = strategy_candidate(rows, str(best_val["strategy"]), "test")
    enough = (
        best_test["clean_drop_pct"] <= 10.0
        and best_test["gain08_reduction_pct"] >= 50.0
        and best_test["gain12_reduction_pct"] >= 50.0
        and best_test["clean_dice"] >= BASELINE_DICE - 0.02
    )
    reason = (
        "calibration_only_pass"
        if enough
        else "calibration_only_not_enough_gain_sensitivity_remains_or_clean_drop_too_large"
    )
    return enough, reason, best_val, best_test


def write_preflight(
    dataset: loader.True3DRBCDataset,
    artifact_manifest: Dict[str, object],
    obs_rows: List[Dict[str, str]],
) -> None:
    split_counts = {name: len(idx) for name, idx in loader.split_indices(dataset).items()}
    lines = [
        "20.89 gain/amplitude calibration preflight",
        "",
        f"repo_root: {ROOT}",
        f"dataset_id: {DATASET_ID}",
        "registry_manifest_gate: pass",
        f"artifact_manifest: {ARTIFACT_MANIFEST}",
        f"checkpoint_path: {artifact_manifest.get('checkpoint_path')}",
        f"prediction_artifact_path: {artifact_manifest.get('prediction_artifact_path')}",
        f"artifact_seed: {artifact_manifest.get('seed')}",
        f"n_samples: {len(dataset.sample_ids)}",
        f"split_counts: {split_counts}",
        f"20_88_metrics_available: {bool(obs_rows)}",
        f"20_88_metrics_path: {OBS_METRICS}",
        "",
        "Subagent preflight summary:",
        "- Registry/Data: v3_240 is loaded through COMSOL_DATA_REGISTRY.md and manifest; no latest/newest NPZ scan is used.",
        "- Artifact: 20.88a checkpoint and prediction artifact manifest are present; checkpoint is loaded from ignored path.",
        "- Calibration design: only in-memory delta_b transforms are evaluated; labels are used only for metrics.",
        "- Safety/Git: no COMSOL, no data/NPZ writes, no checkpoint staging, CURRENT_BASELINE remains unchanged.",
        "",
        "Stop conditions:",
        "- Missing registry/manifest, missing checkpoint manifest, missing 20.88 metrics, or failed checkpoint load.",
    ]
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    rows: List[Dict[str, object]],
    enough: bool,
    reason: str,
    selected_val: Dict[str, object],
    selected_test: Dict[str, object],
) -> None:
    reference_clean = test_row(rows, "no_calibration", "clean")
    reference_08 = test_row(rows, "no_calibration", "gain_scaling_0.8x")
    reference_12 = test_row(rows, "no_calibration", "gain_scaling_1.2x")
    reference_bx = test_row(rows, "no_calibration", "channel_attenuation_Bx_50pct")
    lines = [
        "20.89 gain/amplitude calibration-only strategy audit",
        "",
        "Scope:",
        "- Dataset: comsol_true_3d_rbc_imported_watertight_pilot_v3_240",
        "- Model: fixed 20.77/20.85 seed=42 baseline checkpoint from 20.88a artifact manifest",
        "- Operation: perturb delta_b in memory, then apply input calibration in memory",
        "- No COMSOL, no NPZ/data write, no training, no CURRENT_BASELINE update",
        "",
        "Calibration strategies:",
    ]
    for strategy in calibration_strategies():
        lines.append(f"- {strategy.name}: {strategy.description}")
    lines.extend(
        [
            "",
            "No-calibration reference on test split:",
            f"- clean profile_depth_rmse_m: {float(reference_clean['profile_depth_rmse_m']):.9f}" if reference_clean else "- clean: missing",
            f"- gain 0.8 degradation vs 20.85: {float(reference_08['profile_rmse_degradation_pct_vs_20_85']):.3f}%" if reference_08 else "- gain 0.8: missing",
            f"- gain 1.2 degradation vs 20.85: {float(reference_12['profile_rmse_degradation_pct_vs_20_85']):.3f}%" if reference_12 else "- gain 1.2: missing",
            f"- Bx 50pct attenuation degradation vs 20.85: {float(reference_bx['profile_rmse_degradation_pct_vs_20_85']):.3f}%" if reference_bx else "- Bx 50pct: missing",
            "",
            "Validation-selected calibration candidate:",
        ]
    )
    if selected_val and selected_test:
        lines.extend(
            [
                f"- strategy: {selected_val['strategy']}",
                f"- validation selection score: {selected_val['score']:.6f}",
                f"- test clean profile RMSE drop vs 20.85: {selected_test['clean_drop_pct']:.3f}%",
                f"- test gain 0.8 profile degradation: {selected_test['gain08_drop_pct']:.3f}%",
                f"- test gain 0.8 degradation reduction vs no calibration: {selected_test['gain08_reduction_pct']:.3f}%",
                f"- test gain 1.2 profile degradation: {selected_test['gain12_drop_pct']:.3f}%",
                f"- test gain 1.2 degradation reduction vs no calibration: {selected_test['gain12_reduction_pct']:.3f}%",
                f"- test clean Dice: {selected_test['clean_dice']:.6f}",
                f"- test Bx 50pct attenuation degradation: {selected_test['bx50_drop_pct']:.3f}%",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            f"Calibration-only sufficient: {enough}",
            f"Decision reason: {reason}",
            "",
            "Interpretation:",
            "- Calibration strategy selection uses validation split only; test split is final reporting.",
            "- Calibration-only is considered sufficient only if test clean profile RMSE remains within +10%, gain 0.8 and 1.2 degradation are reduced by at least 50%, and clean Dice stays within 0.02 of baseline.",
            "- If this fails, Stage C augmentation training gate should run because the model remains gain/amplitude sensitive.",
        ]
    )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not ARTIFACT_MANIFEST.exists():
        raise FileNotFoundError(f"missing artifact manifest: {ARTIFACT_MANIFEST}")
    if not OBS_METRICS.exists():
        raise FileNotFoundError(f"missing 20.88 metrics: {OBS_METRICS}")

    entry, manifest, npz_path = loader.resolve_dataset(DATASET_ID)
    checks = loader.gate_manifest(entry, manifest, npz_path, DATASET_ID)
    failed = [row for row in checks if not row["pass"]]
    if failed:
        raise RuntimeError(f"registry/manifest gate failed: {failed}")
    dataset = loader.load_dataset(DATASET_ID)
    if dataset.dataset_id != DATASET_ID:
        raise RuntimeError(f"unexpected dataset_id: {dataset.dataset_id}")
    artifact_manifest, checkpoint, model = obs.load_artifact(ARTIFACT_MANIFEST)
    eval_ctx = obs.make_context(dataset, checkpoint)
    obs_rows = read_csv_rows(OBS_METRICS)
    write_preflight(dataset, artifact_manifest, obs_rows)

    cal_ctx = build_calibration_context(dataset)
    metric_rows: List[Dict[str, object]] = []
    for strategy in calibration_strategies():
        for perturbation in selected_perturbations():
            metric_rows.extend(evaluate_strategy(strategy, perturbation, dataset, model, eval_ctx, cal_ctx))
    write_csv(METRICS, metric_rows)

    enough, reason, selected_val, selected_test = calibration_decision(metric_rows)
    write_summary(metric_rows, enough, reason, selected_val, selected_test)
    print(f"wrote {SUMMARY}")
    print(f"wrote {METRICS}")
    print(f"calibration_only_sufficient={enough} reason={reason}")


if __name__ == "__main__":
    main()
