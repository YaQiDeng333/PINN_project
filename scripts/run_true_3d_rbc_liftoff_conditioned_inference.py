#!/usr/bin/env python
"""20.96 liftoff-conditioned inference smoke for true-3D RBC.

This script loads the fixed 20.85 baseline artifact and the 20.94/20.96a A2
adapter artifact. It does not train, write checkpoints, write NPZ files, or
perform latest/newest discovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

import audit_true_3d_rbc_observation_perturbation_robustness as obs
import load_true_3d_rbc_liftoff_aug_dataset as liftoff
import load_true_3d_rbc_pilot_dataset as pilot
import train_true_3d_rbc_liftoff_adapter_candidates as adapter_train
from run_true_3d_rbc_formal_benchmark_20_77_candidate import add_profile_error_rows


ROOT = pilot.ROOT
DATASET_ID = "comsol_true_3d_rbc_liftoff_aug_pack_v1"
BASELINE_MANIFEST = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
A2_MANIFEST = ROOT / "results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json"
LIFTOFF_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json"
FORMAL_LIFTOFF_METRICS = ROOT / "results/metrics/true_3d_rbc_formal_liftoff_benchmark_metrics.csv"
PRE_FLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_inference_smoke_preflight_summary.txt"
SMOKE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_conditioned_inference_smoke_summary.txt"
SMOKE_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_conditioned_inference_smoke_metrics.csv"
BY_LIFTOFF = ROOT / "results/metrics/true_3d_rbc_liftoff_conditioned_inference_by_liftoff.csv"
FAILURE_CASES = ROOT / "results/metrics/true_3d_rbc_liftoff_conditioned_inference_failure_cases.csv"

ROUTE_MODES = ("auto", "force_baseline", "force_adapter")
NOMINAL_SENSOR_Z_M = 0.008
DEFAULT_NOMINAL_TOLERANCE_M = 0.0005
MIN_SENSOR_Z_M = 0.006
MAX_SENSOR_Z_M = 0.012
SENSOR_RANGE_TOLERANCE_M = 1.0e-9
FORBIDDEN_STAGE_PREFIXES = (
    "data/",
    "notes/",
    "checkpoints/",
    "results/previews/",
)
FORBIDDEN_STAGE_SUFFIXES = (
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".png",
    ".mph",
    ".csv.raw",
)
FORBIDDEN_STAGE_EXACT = {
    "CURRENT_BASELINE.md",
    "scripts/visualize_current_baseline.py",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    if field_order:
        keys.extend(field_order)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and row[key] not in ("", None)]
    return float(np.mean(vals)) if vals else float("nan")


def pct_delta(value: float, reference: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference) or abs(reference) < 1.0e-20:
        return float("nan")
    return 100.0 * (value - reference) / abs(reference)


def staged_files() -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def forbidden_staged(paths: list[str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        if path in FORBIDDEN_STAGE_EXACT:
            out.append(path)
            continue
        if any(path.startswith(prefix) for prefix in FORBIDDEN_STAGE_PREFIXES):
            out.append(path)
            continue
        if any(path.lower().endswith(suffix) for suffix in FORBIDDEN_STAGE_SUFFIXES):
            out.append(path)
    return out


def load_adapter_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError(f"unexpected A2 dataset_id: {manifest.get('dataset_id')}")
    if manifest.get("adapter_type") != "A2_latent_residual_adapter":
        raise RuntimeError(f"unexpected adapter type: {manifest.get('adapter_type')}")
    ckpt = Path(manifest["checkpoint_path"])
    pred = Path(manifest["prediction_artifact_path"])
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if not pred.exists():
        raise FileNotFoundError(pred)
    if obs.sha256_file(ckpt) != manifest["checkpoint_sha256"]:
        raise RuntimeError("A2 checkpoint sha256 mismatch")
    if obs.sha256_file(pred) != manifest["prediction_artifact_sha256"]:
        raise RuntimeError("A2 prediction artifact sha256 mismatch")
    return manifest


def candidate_config_from_manifest(manifest: dict[str, Any], checkpoint: dict[str, Any]) -> adapter_train.CandidateConfig:
    raw = dict(checkpoint.get("candidate_config") or manifest.get("model_config") or {})
    allowed = {field.name for field in fields(adapter_train.CandidateConfig)}
    return adapter_train.CandidateConfig(**{key: raw[key] for key in raw if key in allowed})


def load_a2_adapter(manifest: dict[str, Any]) -> tuple[dict[str, Any], adapter_train.CandidateConfig, torch.nn.Module]:
    checkpoint_path = Path(manifest["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = candidate_config_from_manifest(manifest, checkpoint)
    input_dim = int(checkpoint.get("input_dim", manifest["model_config"]["input_dim"]))
    model = adapter_train.model_for(config, input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, config, model


def choose_route(
    sensor_z_m: float | None,
    route_mode: str = "auto",
    nominal_sensor_z_m: float = NOMINAL_SENSOR_Z_M,
    tolerance_m: float = DEFAULT_NOMINAL_TOLERANCE_M,
) -> tuple[str, bool]:
    if sensor_z_m is None:
        raise ValueError("sensor_z_m is required; inference cannot guess liftoff.")
    z = float(sensor_z_m)
    if not math.isfinite(z):
        raise ValueError("sensor_z_m must be finite.")
    out_of_range = z < (MIN_SENSOR_Z_M - SENSOR_RANGE_TOLERANCE_M) or z > (MAX_SENSOR_Z_M + SENSOR_RANGE_TOLERANCE_M)
    if route_mode == "force_baseline":
        return "baseline", out_of_range
    if route_mode == "force_adapter":
        return "baseline_plus_adapter", out_of_range
    if route_mode != "auto":
        raise ValueError(f"unknown route_mode={route_mode}")
    if abs(z - nominal_sensor_z_m) < tolerance_m or abs(z - nominal_sensor_z_m) < 1.0e-12:
        return "baseline", out_of_range
    return "baseline_plus_adapter", out_of_range


def route_expected(sensor_z_m: float, tolerance_m: float) -> str:
    return "baseline" if abs(float(sensor_z_m) - NOMINAL_SENSOR_Z_M) < tolerance_m else "baseline_plus_adapter"


def preflight(args: argparse.Namespace) -> None:
    summary_lines: list[str] = ["20.96 liftoff-conditioned inference smoke preflight", ""]
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, notes: str = "") -> None:
        checks.append({"check": name, "pass": bool(passed), "observed": observed, "notes": notes})

    add("repo_root", ROOT == Path.cwd(), str(Path.cwd()))
    add("liftoff_manifest_exists", LIFTOFF_MANIFEST.exists(), str(LIFTOFF_MANIFEST))
    add("baseline_artifact_manifest_exists", BASELINE_MANIFEST.exists(), str(BASELINE_MANIFEST))
    add("a2_artifact_manifest_exists", A2_MANIFEST.exists(), str(A2_MANIFEST))
    add("formal_liftoff_metrics_exists", FORMAL_LIFTOFF_METRICS.exists(), str(FORMAL_LIFTOFF_METRICS))
    dataset_loaded = False
    try:
        dataset = liftoff.load_liftoff_dataset(args.dataset_id)
        dataset_loaded = True
        add("dataset_id", dataset.dataset_id == DATASET_ID, dataset.dataset_id)
        add("rows", len(dataset.sample_ids) == 192, len(dataset.sample_ids))
        add("base_count", len(set(dataset.base_sample_ids.astype(str))) == 48, len(set(dataset.base_sample_ids.astype(str))))
        add("paired_liftoff_complete", liftoff.paired_liftoff_complete(dataset), True)
        add("split_counts", {k: int(len(v)) for k, v in liftoff.split_indices(dataset).items()} == {"train": 128, "val": 32, "test": 32}, {k: int(len(v)) for k, v in liftoff.split_indices(dataset).items()})
        add("latest_newest_npz_scan", False, "disabled; explicit registry/manifest only")
    except Exception as exc:
        add("dataset_load", False, type(exc).__name__, str(exc))

    try:
        base_manifest, _base_ckpt, _base_model = obs.load_artifact(BASELINE_MANIFEST)
        add("baseline_artifact_load", True, base_manifest.get("artifact_id"))
    except Exception as exc:
        add("baseline_artifact_load", False, type(exc).__name__, str(exc))

    try:
        a2_manifest = load_adapter_manifest(A2_MANIFEST)
        _a2_ckpt, _cfg, _model = load_a2_adapter(a2_manifest)
        add("a2_artifact_load", True, a2_manifest.get("artifact_id"))
    except Exception as exc:
        add("a2_artifact_load", False, type(exc).__name__, str(exc))

    staged = staged_files()
    forbidden = forbidden_staged(staged)
    add("forbidden_staged_files", not forbidden, forbidden or "none")
    add("COMSOL_run", False, "not allowed")
    add("training_run", False, "not allowed")
    add("NPZ_write", False, "not allowed")
    add("CURRENT_BASELINE_update", False, "not allowed")
    add("dataset_loaded_for_preflight", dataset_loaded, dataset_loaded)

    for row in checks:
        summary_lines.append(f"{row['check']}: pass={row['pass']} observed={row['observed']} notes={row['notes']}")
    failed = [row for row in checks if not row["pass"] and row["check"] not in {"COMSOL_run", "training_run", "NPZ_write", "CURRENT_BASELINE_update", "latest_newest_npz_scan"}]
    summary_lines.extend(
        [
            "",
            f"preflight_pass: {not failed}",
            "stop_condition: stop if registry/manifest, baseline artifact, or A2 artifact is missing.",
            "forbidden_submit: data/, NPZ, checkpoint, preview PNG, notes, baseline docs, CURRENT_BASELINE.md, scripts/visualize_current_baseline.py",
        ]
    )
    PRE_FLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PRE_FLIGHT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError("20.96 preflight failed; see " + str(PRE_FLIGHT_SUMMARY))


def add_inference_metadata(
    dataset: liftoff.True3DRBCLiftoffDataset,
    rows: list[dict[str, Any]],
    route_mode: str,
    route_used: np.ndarray,
    out_of_range: np.ndarray,
    tolerance_m: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    expected = np.asarray([route_expected(z, tolerance_m) for z in dataset.sensor_z_m])
    for idx, row in enumerate(rows):
        cur = dict(row)
        cur.update(
            {
                "route_mode": route_mode,
                "route_used": str(route_used[idx]),
                "route_expected_auto": str(expected[idx]),
                "route_correct_for_mode": bool(route_used[idx] == expected[idx]) if route_mode == "auto" else True,
                "sensor_z_m": float(dataset.sensor_z_m[idx]),
                "sensor_z_out_of_range": bool(out_of_range[idx]),
                "base_sample_id": str(dataset.base_sample_ids[idx]),
                "variant_name": str(dataset.variant_name[idx]),
                "factor_group": str(dataset.factor_group[idx]),
                "row_kind": str(dataset.row_kind[idx]),
            }
        )
        out.append(cur)
    return out


def prediction_rows_for_mode(
    dataset: liftoff.True3DRBCLiftoffDataset,
    stats: dict[str, np.ndarray],
    pred_raw: np.ndarray,
    route_mode: str,
    route_used: np.ndarray,
    out_of_range: np.ndarray,
    tolerance_m: float,
) -> list[dict[str, Any]]:
    rows = pilot.evaluate_param_predictions(dataset, pred_raw, {"y_mean": stats["y_mean"], "y_std": stats["y_std"]})
    rows = add_profile_error_rows(dataset, pred_raw, rows)
    return add_inference_metadata(dataset, rows, route_mode, route_used, out_of_range, tolerance_m)


def aggregate_rows(rows: list[dict[str, Any]], route_mode: str, split_name: str, subset_name: str, indices: np.ndarray) -> dict[str, Any]:
    index_set = {int(i) for i in indices.tolist()}
    subset = [row for idx, row in enumerate(rows) if idx in index_set and row["split"] == split_name]
    return {
        "route_mode": route_mode,
        "split": split_name,
        "liftoff_subset": subset_name,
        "sample_count": len(subset),
        "profile_depth_rmse_m": mean(subset, "profile_depth_rmse_m"),
        "er_like_profile_error": mean(subset, "er_like_profile_error"),
        "normalized_param_mae": mean(subset, "normalized_param_mae_mean"),
        "L_mae_mm": mean(subset, "L_mae_mm"),
        "W_mae_mm": mean(subset, "W_mae_mm"),
        "D_mae_mm": mean(subset, "D_mae_mm"),
        "wMAE_auxiliary": mean(subset, "curvature_mae_mean"),
        "wLD_abs_error": mean(subset, "wLD_abs_error"),
        "wWD_abs_error": mean(subset, "wWD_abs_error"),
        "wLW_abs_error": mean(subset, "wLW_abs_error"),
        "projected_mask_iou": mean(subset, "projected_mask_iou"),
        "projected_mask_dice": mean(subset, "projected_mask_dice"),
        "max_depth_error_m": mean(subset, "max_depth_error_m"),
        "volume_proxy_rel_error": mean(subset, "volume_proxy_rel_error"),
        "route_used_accuracy": mean(subset, "route_correct_for_mode"),
        "out_of_range_count": int(sum(bool(row["sensor_z_out_of_range"]) for row in subset)),
    }


def build_aggregate_metrics(dataset: liftoff.True3DRBCLiftoffDataset, all_rows: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_idx = liftoff.split_indices(dataset)
    metrics: list[dict[str, Any]] = []
    by_liftoff: list[dict[str, Any]] = []
    all_indices = np.arange(len(dataset.sample_ids))
    nominal = np.where(np.isclose(dataset.sensor_z_m, NOMINAL_SENSOR_Z_M, atol=1.0e-7))[0]
    non_nominal = np.setdiff1d(all_indices, nominal)
    subsets = {"all": all_indices, "nominal_0p008": nominal, "non_nominal": non_nominal}
    for route_mode, rows in all_rows.items():
        for split_name in ("train", "val", "test"):
            for subset_name, idx in subsets.items():
                split_subset = np.intersect1d(split_idx[split_name], idx)
                metrics.append(aggregate_rows(rows, route_mode, split_name, subset_name, split_subset))
            for z in sorted({round(float(v), 3) for v in dataset.sensor_z_m}):
                level_idx = np.where(np.isclose(dataset.sensor_z_m, z, atol=1.0e-7))[0]
                split_level = np.intersect1d(split_idx[split_name], level_idx)
                row = aggregate_rows(rows, route_mode, split_name, f"z_{z:.3f}", split_level)
                row["sensor_z_m"] = z
                by_liftoff.append(row)
    clean_by_mode = {
        row["route_mode"]: row
        for row in metrics
        if row["split"] == "test" and row["liftoff_subset"] == "all"
    }
    reference = clean_by_mode.get("force_baseline")
    if reference:
        for row in metrics + by_liftoff:
            ref = reference["profile_depth_rmse_m"]
            row["profile_rmse_delta_vs_force_baseline_all_test_pct"] = pct_delta(float(row["profile_depth_rmse_m"]), float(ref))
            row["dice_delta_vs_force_baseline_all_test"] = float(row["projected_mask_dice"]) - float(reference["projected_mask_dice"])
    return metrics, by_liftoff


def failure_rows(dataset: liftoff.True3DRBCLiftoffDataset, all_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for route_mode, rows in all_rows.items():
        test_rows = [row for row in rows if row["split"] == "test"]
        for rank, row in enumerate(sorted(test_rows, key=lambda r: float(r["profile_depth_rmse_m"]), reverse=True)[:8], start=1):
            cur = {
                "case_type": "worst_profile_rmse_test",
                "rank": rank,
                "route_mode": route_mode,
                "sample_id": row["sample_id"],
                "base_sample_id": row["base_sample_id"],
                "sensor_z_m": row["sensor_z_m"],
                "route_used": row["route_used"],
                "profile_depth_rmse_m": row["profile_depth_rmse_m"],
                "er_like_profile_error": row["er_like_profile_error"],
                "projected_mask_dice": row["projected_mask_dice"],
                "L_mae_mm": row["L_mae_mm"],
                "W_mae_mm": row["W_mae_mm"],
                "D_mae_mm": row["D_mae_mm"],
                "wMAE_auxiliary": row["curvature_mae_mean"],
                "curvature_template": row["curvature_template"],
                "depth_bin": row["depth_bin"],
                "aspect_bin": row["aspect_bin"],
            }
            out.append(cur)
    try:
        choose_route(None)
    except ValueError as exc:
        out.append({"case_type": "missing_sensor_z_contract_check", "rank": "", "route_mode": "auto", "sample_id": "", "base_sample_id": "", "sensor_z_m": "", "route_used": "error", "profile_depth_rmse_m": "", "er_like_profile_error": "", "projected_mask_dice": "", "notes": str(exc)})
    for z, expected in [(0.006, False), (0.012, False), (0.005999, True), (0.012001, True), (0.02, True)]:
        route, flag = choose_route(z)
        out.append(
            {
                "case_type": "sensor_z_range_contract_check",
                "rank": "",
                "route_mode": "auto",
                "sample_id": "",
                "base_sample_id": "",
                "sensor_z_m": z,
                "route_used": route,
                "sensor_z_out_of_range": flag,
                "expected_out_of_range": expected,
                "contract_check_pass": bool(flag == expected),
                "notes": "range boundary check for validated liftoff interval [0.006,0.012].",
            }
        )
    return out


def run_smoke(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = liftoff.load_liftoff_dataset(args.dataset_id)
    stats = liftoff.train_normalization(dataset)

    baseline_manifest, baseline_checkpoint, baseline_model = obs.load_artifact(BASELINE_MANIFEST)
    baseline_arrays = adapter_train.baseline_arrays(dataset, stats, baseline_checkpoint, baseline_model)
    baseline_pred = baseline_arrays["pred_raw"]

    a2_manifest = load_adapter_manifest(A2_MANIFEST)
    _a2_checkpoint, config, a2_model = load_a2_adapter(a2_manifest)
    z_norm = liftoff.normalize_sensor_z(dataset, stats)
    x_norm = liftoff.normalize_x(dataset, stats)
    features = adapter_train.features_for(config, baseline_arrays, z_norm)
    adapter_pred_norm, _residual = adapter_train.predict_candidate(
        a2_model,
        config,
        x_norm,
        z_norm,
        features,
        baseline_arrays["pred_norm"],
    )
    adapter_pred = liftoff.denormalize_y(adapter_pred_norm, stats)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for route_mode in ROUTE_MODES:
        route_used = []
        out_of_range = []
        for z in dataset.sensor_z_m:
            route, flag = choose_route(float(z), route_mode, tolerance_m=args.nominal_tolerance_m)
            route_used.append(route)
            out_of_range.append(flag)
        route_used_arr = np.asarray(route_used, dtype=object)
        out_of_range_arr = np.asarray(out_of_range, dtype=bool)
        use_adapter = route_used_arr == "baseline_plus_adapter"
        pred = baseline_pred.copy()
        pred[use_adapter] = adapter_pred[use_adapter]
        all_rows[route_mode] = prediction_rows_for_mode(dataset, stats, pred, route_mode, route_used_arr, out_of_range_arr, args.nominal_tolerance_m)

    metrics, by_liftoff = build_aggregate_metrics(dataset, all_rows)
    failures = failure_rows(dataset, all_rows)

    write_csv(SMOKE_METRICS, metrics)
    write_csv(BY_LIFTOFF, by_liftoff)
    write_csv(FAILURE_CASES, failures)

    test_auto_all = next(row for row in metrics if row["route_mode"] == "auto" and row["split"] == "test" and row["liftoff_subset"] == "all")
    test_auto_nom = next(row for row in metrics if row["route_mode"] == "auto" and row["split"] == "test" and row["liftoff_subset"] == "nominal_0p008")
    test_auto_non = next(row for row in metrics if row["route_mode"] == "auto" and row["split"] == "test" and row["liftoff_subset"] == "non_nominal")
    force_base_non = next(row for row in metrics if row["route_mode"] == "force_baseline" and row["split"] == "test" and row["liftoff_subset"] == "non_nominal")
    force_adapter_non = next(row for row in metrics if row["route_mode"] == "force_adapter" and row["split"] == "test" and row["liftoff_subset"] == "non_nominal")
    summary = [
        "20.96 liftoff-conditioned inference smoke summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"baseline_artifact: {baseline_manifest.get('artifact_id')}",
        f"a2_artifact: {a2_manifest.get('artifact_id')}",
        f"route_rule: auto uses baseline when abs(sensor_z_m-0.008)<{args.nominal_tolerance_m}; otherwise baseline_plus_adapter",
        "override_modes: force_baseline, force_adapter",
        "missing_sensor_z_behavior: ValueError; no guessing",
        "out_of_range_behavior: sensor_z_m outside [0.006,0.012] is flagged",
        "",
        "test auto metrics:",
        f"  all profile_depth_rmse_m={test_auto_all['profile_depth_rmse_m']:.9f}, dice={test_auto_all['projected_mask_dice']:.6f}",
        f"  nominal profile_depth_rmse_m={test_auto_nom['profile_depth_rmse_m']:.9f}, dice={test_auto_nom['projected_mask_dice']:.6f}",
        f"  non_nominal profile_depth_rmse_m={test_auto_non['profile_depth_rmse_m']:.9f}, dice={test_auto_non['projected_mask_dice']:.6f}",
        "20.95 companion replay:",
        f"  force_baseline non_nominal RMSE={force_base_non['profile_depth_rmse_m']:.9f}, dice={force_base_non['projected_mask_dice']:.6f}",
        f"  force_adapter non_nominal RMSE={force_adapter_non['profile_depth_rmse_m']:.9f}, dice={force_adapter_non['projected_mask_dice']:.6f}",
        f"  auto non_nominal equals adapter route: {abs(float(test_auto_non['profile_depth_rmse_m']) - float(force_adapter_non['profile_depth_rmse_m'])) < 1.0e-12}",
        f"route_used_accuracy_auto_test_all: {test_auto_all['route_used_accuracy']}",
        "COMSOL_run: false",
        "training_run: false",
        "NPZ_write: false",
        "checkpoint_write: false",
        "CURRENT_BASELINE_update: false",
        f"outputs: {SMOKE_METRICS.relative_to(ROOT)}, {BY_LIFTOFF.relative_to(ROOT)}, {FAILURE_CASES.relative_to(ROOT)}",
    ]
    SMOKE_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return metrics, by_liftoff, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 20.96 liftoff-conditioned true-3D RBC inference smoke.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--nominal-tolerance-m", type=float, default=DEFAULT_NOMINAL_TOLERANCE_M)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset_id != DATASET_ID:
        raise RuntimeError(f"20.96 only allows dataset_id={DATASET_ID}")
    if not args.skip_preflight:
        preflight(args)
    run_smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
