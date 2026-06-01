#!/usr/bin/env python
"""Audit RBC six-parameter representability on the 25.2 shape-extension pilot.

This script uses only shape labels, depth maps, masks, and geometry metadata.
It does not use Bx/By/Bz for fitting, does not train, does not run COMSOL, and
does not modify generated data or NPZ files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy import ndimage

from load_true_3d_rbc_pilot_dataset import (
    MASK_HEIGHT,
    MASK_WIDTH,
    depth_map_from_params,
    projected_mask_from_params,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_surface_shape_extension_pilot_v1"
BASELINE_ARTIFACT = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
PILOT_MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
PILOT_VALIDATION_SUMMARY = ROOT / "results/summaries/surface_shape_extension_pilot_validation_summary.txt"
PROFILE_GENERATOR = ROOT / "scripts/true_3d_rbc_profile_generator.py"

PREFLIGHT_SUMMARY = ROOT / "results/summaries/surface_shape_extension_baseline_audit_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/surface_shape_extension_rbc_oracle_fit_summary.txt"
METRICS = ROOT / "results/metrics/surface_shape_extension_rbc_oracle_fit_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/surface_shape_extension_rbc_oracle_fit_group_summary.csv"
FAILURE_CASES = ROOT / "results/metrics/surface_shape_extension_rbc_oracle_failure_cases.csv"

ORACLE_PROFILE_RMSE_PASS_M = 4.0e-4
ORACLE_DICE_PASS = 0.80

PARAM_NAMES = ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW"]
NON_RBC_SHAPES = {
    "flat_bottom_pit",
    "sharp_wall_boxy_corrosion",
    "asymmetric_corrosion",
    "elongated_crack_like_surface_defect",
    "multi_pit_two_component_surface_defect",
    "irregular_corrosion_non_rbc",
}
FORBIDDEN_STAGE_PREFIXES = ("data/", "checkpoints/", "results/previews/", "notes/")
FORBIDDEN_STAGE_SUFFIXES = (".npz", ".mph", ".png", ".jpg", ".jpeg", ".stl")

METRIC_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "rbc_compatible",
    "component_count",
    "oracle_fit_success",
    "rbc_representable",
    "oracle_profile_depth_rmse_m",
    "oracle_Er_like_error",
    "oracle_projected_mask_IoU",
    "oracle_projected_mask_Dice",
    "oracle_area_error",
    "oracle_mask_center_error_px",
    "true_area_px",
    "oracle_area_px",
    "oracle_component_count",
    "true_component_count",
    "oracle_L_m",
    "oracle_W_m",
    "oracle_D_m",
    "oracle_wLD",
    "oracle_wWD",
    "oracle_wLW",
    "label_L_m",
    "label_W_m",
    "label_D_m",
    "oracle_L_abs_error_m",
    "oracle_W_abs_error_m",
    "oracle_D_abs_error_m",
    "oracle_objective",
    "fit_message",
]

GROUP_FIELDS = [
    "group_field",
    "group_value",
    "split",
    "sample_count",
    "oracle_fit_success_rate",
    "rbc_representable_rate",
    "oracle_profile_depth_rmse_mean_m",
    "oracle_profile_depth_rmse_p95_m",
    "oracle_Er_like_mean",
    "oracle_projected_mask_Dice_mean",
    "oracle_projected_mask_IoU_mean",
    "oracle_area_error_mean",
    "primary_interpretation",
]

FAILURE_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "oracle_profile_depth_rmse_m",
    "oracle_projected_mask_Dice",
    "oracle_Er_like_error",
    "oracle_area_error",
    "true_component_count",
    "oracle_component_count",
    "failure_reason",
]


@dataclass(frozen=True)
class SurfaceShapeDataset:
    manifest: dict[str, Any]
    registry_entry: dict[str, str]
    npz_path: Path
    sample_ids: np.ndarray
    split: np.ndarray
    shape_type: np.ndarray
    topology_type: np.ndarray
    representation_target: np.ndarray
    rbc_compatible: np.ndarray
    delta_b: np.ndarray
    b_defect: np.ndarray
    b_no_defect: np.ndarray
    depth_grid_m: np.ndarray
    projected_mask_2d: np.ndarray
    L_m: np.ndarray
    W_m: np.ndarray
    D_m: np.ndarray
    center_xyz_m: np.ndarray
    component_count: np.ndarray
    component_params_json: np.ndarray
    aspect_ratio: np.ndarray
    rotation_angle: np.ndarray
    asymmetry_score: np.ndarray
    edge_steepness: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RBC oracle representability for the 25.2 shape-extension pilot.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--registry", type=Path, default=ROOT / "COMSOL_DATA_REGISTRY.md")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def clean_csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.rstrip()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: clean_csv_value(row.get(field, "")) for field in fields} for row in rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def normalize_staged(path: str) -> str:
    return path.replace("\\", "/").strip('"')


def forbidden_staged() -> list[str]:
    staged = [normalize_staged(item) for item in git_value(["diff", "--cached", "--name-only"]).splitlines() if item.strip()]
    bad: list[str] = []
    for path in staged:
        lower = path.lower()
        if path == "CURRENT_BASELINE.md" or path == "scripts/visualize_current_baseline.py":
            bad.append(path)
        if path.startswith(FORBIDDEN_STAGE_PREFIXES) or lower.endswith(FORBIDDEN_STAGE_SUFFIXES):
            bad.append(path)
    return sorted(set(bad))


def parse_registry(path: Path) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1).strip()
            entries[current] = {}
            continue
        if current and line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            entries[current][key.strip()] = value.strip().strip("`")
    return entries


def resolve_surface_dataset(dataset_id: str = DATASET_ID, registry_path: Path = ROOT / "COMSOL_DATA_REGISTRY.md") -> tuple[dict[str, str], dict[str, Any], Path]:
    registry = parse_registry(registry_path)
    if dataset_id not in registry:
        raise RuntimeError(f"dataset_id not found in registry: {dataset_id}")
    entry = registry[dataset_id]
    manifest_path = Path(entry.get("manifest_path", ""))
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_path from registry does not exist: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("dataset_id") != dataset_id:
        raise RuntimeError(f"manifest dataset_id mismatch: {manifest.get('dataset_id')} != {dataset_id}")
    if Path(str(manifest.get("manifest_path", manifest_path))) != manifest_path:
        raise RuntimeError(f"manifest self path mismatch: {manifest.get('manifest_path')} != {manifest_path}")
    npz_path = Path(manifest["generated_npz_path"])
    return entry, manifest, npz_path


def _string_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def load_surface_dataset(dataset_id: str = DATASET_ID, registry_path: Path = ROOT / "COMSOL_DATA_REGISTRY.md") -> SurfaceShapeDataset:
    entry, manifest, npz_path = resolve_surface_dataset(dataset_id, registry_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if manifest.get("npz_sha256") and sha256_file(npz_path) != manifest.get("npz_sha256"):
        raise RuntimeError("surface shape-extension NPZ sha256 mismatch")
    if manifest.get("validation_status") is not True:
        raise RuntimeError("surface shape-extension manifest validation_status is not true")
    if manifest.get("baseline_ready") is not False:
        raise RuntimeError("surface shape-extension pilot must not be baseline_ready")
    if manifest.get("train_ready_candidate") is not False:
        raise RuntimeError("surface shape-extension pilot must not be train_ready_candidate for 25.3")
    with np.load(npz_path, allow_pickle=True) as pack:
        required = [
            "sample_ids",
            "split",
            "shape_type",
            "topology_type",
            "representation_target",
            "rbc_compatible",
            "delta_b",
            "b_defect",
            "b_no_defect",
            "depth_grid_m",
            "projected_mask_2d",
            "L_m",
            "W_m",
            "D_m",
            "center_xyz_m",
            "component_count",
            "component_params_json",
            "aspect_ratio",
            "rotation_angle",
            "asymmetry_score",
            "edge_steepness",
        ]
        missing = [key for key in required if key not in pack.files]
        if missing:
            raise RuntimeError(f"missing surface shape-extension fields: {missing}")
        return SurfaceShapeDataset(
            manifest=manifest,
            registry_entry=entry,
            npz_path=npz_path,
            sample_ids=_string_array(pack["sample_ids"]),
            split=_string_array(pack["split"]),
            shape_type=_string_array(pack["shape_type"]),
            topology_type=_string_array(pack["topology_type"]),
            representation_target=_string_array(pack["representation_target"]),
            rbc_compatible=np.asarray(pack["rbc_compatible"], dtype=bool),
            delta_b=np.asarray(pack["delta_b"], dtype=np.float64),
            b_defect=np.asarray(pack["b_defect"], dtype=np.float64),
            b_no_defect=np.asarray(pack["b_no_defect"], dtype=np.float64),
            depth_grid_m=np.asarray(pack["depth_grid_m"], dtype=np.float32),
            projected_mask_2d=np.asarray(pack["projected_mask_2d"], dtype=np.uint8),
            L_m=np.asarray(pack["L_m"], dtype=np.float64),
            W_m=np.asarray(pack["W_m"], dtype=np.float64),
            D_m=np.asarray(pack["D_m"], dtype=np.float64),
            center_xyz_m=np.asarray(pack["center_xyz_m"], dtype=np.float64),
            component_count=np.asarray(pack["component_count"], dtype=np.int64),
            component_params_json=np.asarray(pack["component_params_json"], dtype=object),
            aspect_ratio=np.asarray(pack["aspect_ratio"], dtype=np.float64),
            rotation_angle=np.asarray(pack["rotation_angle"], dtype=np.float64),
            asymmetry_score=np.asarray(pack["asymmetry_score"], dtype=np.float64),
            edge_steepness=np.asarray(pack["edge_steepness"], dtype=np.float64),
        )


def pose_for_sample(dataset: SurfaceShapeDataset, index: int) -> np.ndarray:
    return np.asarray([dataset.center_xyz_m[index, 0], dataset.center_xyz_m[index, 1], dataset.rotation_angle[index]], dtype=np.float64)


def mask_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)
    intersection = int(np.logical_and(pred, true).sum())
    union = int(np.logical_or(pred, true).sum())
    pred_area = int(pred.sum())
    true_area = int(true.sum())
    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if pred_area + true_area == 0 else (2.0 * intersection) / (pred_area + true_area)
    area_error = 0.0 if true_area == 0 else abs(pred_area - true_area) / true_area
    if pred_area == 0 or true_area == 0:
        center_error = float(max(MASK_WIDTH, MASK_HEIGHT))
    else:
        yy, xx = np.indices(true.shape)
        pred_center = np.array([float(xx[pred].mean()), float(yy[pred].mean())])
        true_center = np.array([float(xx[true].mean()), float(yy[true].mean())])
        center_error = float(np.linalg.norm(pred_center - true_center))
    return {
        "iou": float(iou),
        "dice": float(dice),
        "area_error": float(area_error),
        "center_error_px": center_error,
        "pred_area_px": float(pred_area),
        "true_area_px": float(true_area),
    }


def connected_component_count(mask: np.ndarray) -> int:
    labels, count = ndimage.label(np.asarray(mask).astype(bool))
    _ = labels
    return int(count)


def er_like_profile_error(pred: np.ndarray, true: np.ndarray) -> float:
    pred64 = np.asarray(pred, dtype=np.float64)
    true64 = np.asarray(true, dtype=np.float64)
    denom = float(np.sum(true64 * true64))
    if denom <= 1.0e-20:
        return 0.0 if float(np.sum(pred64 * pred64)) <= 1.0e-20 else math.inf
    return float(math.sqrt(float(np.sum((pred64 - true64) ** 2)) / denom))


def profile_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)) ** 2)))


def row_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in {"", None, ""} and math.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else math.nan


def row_p95(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in {"", None, ""} and math.isfinite(float(row[key]))]
    return float(np.percentile(vals, 95)) if vals else math.nan


def bbox_initial_params(dataset: SurfaceShapeDataset, index: int) -> np.ndarray:
    mask = np.asarray(dataset.projected_mask_2d[index]).astype(bool)
    d0 = max(float(dataset.depth_grid_m[index].max()), float(dataset.D_m[index]), 1.0e-4)
    if not mask.any():
        return np.asarray([dataset.L_m[index], dataset.W_m[index], d0, 0.35, 0.35, 0.35], dtype=np.float64)
    yy, xx = np.where(mask)
    x_m = np.linspace(-0.04, 0.04, MASK_WIDTH)
    y_m = np.linspace(-0.01, 0.01, MASK_HEIGHT)
    l0 = max(float(x_m[xx.max()] - x_m[xx.min()]), float(dataset.L_m[index]), 0.002)
    w0 = max(float(y_m[yy.max()] - y_m[yy.min()]), float(dataset.W_m[index]), 0.001)
    return np.asarray([l0, w0, d0, 0.35, 0.35, 0.35], dtype=np.float64)


def oracle_bounds(dataset: SurfaceShapeDataset) -> list[tuple[float, float]]:
    l_high = max(float(np.nanmax(dataset.L_m) * 1.75), 0.035)
    w_high = max(float(np.nanmax(dataset.W_m) * 1.75), 0.018)
    d_high = max(float(np.nanmax(dataset.D_m) * 1.75), float(np.nanmax(dataset.depth_grid_m)) * 1.75, 0.004)
    return [(0.0015, l_high), (0.00075, w_high), (5.0e-5, d_high), (0.03, 10.0), (0.03, 10.0), (0.03, 10.0)]


def fit_one_sample(dataset: SurfaceShapeDataset, index: int) -> dict[str, Any]:
    true_depth = np.asarray(dataset.depth_grid_m[index], dtype=np.float64)
    true_mask = np.asarray(dataset.projected_mask_2d[index], dtype=np.uint8)
    pose = pose_for_sample(dataset, index)
    bounds = oracle_bounds(dataset)
    scale = max(float(true_depth.max()), float(dataset.D_m[index]), 1.0e-6)

    def evaluate(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
        pred_depth = depth_map_from_params(np.asarray(params, dtype=np.float64), pose)
        pred_mask = projected_mask_from_params(np.asarray(params, dtype=np.float64), pose)
        mm = mask_metrics(pred_mask, true_mask)
        rmse_norm = profile_rmse(pred_depth, true_depth) / scale
        objective = rmse_norm + 0.20 * (1.0 - mm["dice"]) + 0.05 * min(mm["area_error"], 4.0)
        return float(objective), pred_depth, pred_mask, mm

    label_start = np.asarray([dataset.L_m[index], dataset.W_m[index], max(dataset.D_m[index], 1.0e-4), 0.35, 0.35, 0.35], dtype=np.float64)
    bbox_start = bbox_initial_params(dataset, index)
    starts = [
        label_start,
        bbox_start,
        np.asarray([bbox_start[0], bbox_start[1], bbox_start[2], 1.0, 1.0, 1.0], dtype=np.float64),
        np.asarray([bbox_start[0], bbox_start[1], bbox_start[2], 5.0, 5.0, 5.0], dtype=np.float64),
        np.asarray([label_start[0], label_start[1], label_start[2], 0.08, 0.08, 0.08], dtype=np.float64),
    ]

    best: dict[str, Any] | None = None
    messages: list[str] = []
    for start in starts:
        x0 = np.asarray([min(max(float(v), lo), hi) for v, (lo, hi) in zip(start, bounds)], dtype=np.float64)
        try:
            result = minimize(
                lambda x: evaluate(x)[0],
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 120, "ftol": 1.0e-10, "maxls": 20},
            )
            params = np.asarray(result.x, dtype=np.float64)
            obj, pred_depth, pred_mask, mm = evaluate(params)
            messages.append(str(result.message))
            candidate = {
                "objective": obj,
                "params": params,
                "pred_depth": pred_depth,
                "pred_mask": pred_mask,
                "mask_metrics": mm,
                "success": bool(result.success or math.isfinite(obj)),
                "message": str(result.message),
            }
            if best is None or obj < float(best["objective"]):
                best = candidate
        except Exception as exc:  # Fit failures must be visible in output rows.
            messages.append(f"{type(exc).__name__}: {exc}")
    if best is None:
        return {
            "oracle_fit_success": False,
            "fit_message": "; ".join(messages) or "fit_failed_without_message",
        }

    params = np.asarray(best["params"], dtype=np.float64)
    pred_depth = np.asarray(best["pred_depth"], dtype=np.float64)
    pred_mask = np.asarray(best["pred_mask"], dtype=np.uint8)
    mm = dict(best["mask_metrics"])
    rmse = profile_rmse(pred_depth, true_depth)
    er = er_like_profile_error(pred_depth, true_depth)
    representable = bool(rmse <= ORACLE_PROFILE_RMSE_PASS_M and mm["dice"] >= ORACLE_DICE_PASS)
    true_components = int(dataset.component_count[index])
    oracle_components = connected_component_count(pred_mask)
    return {
        "oracle_fit_success": bool(best["success"]),
        "rbc_representable": representable,
        "oracle_profile_depth_rmse_m": rmse,
        "oracle_Er_like_error": er,
        "oracle_projected_mask_IoU": mm["iou"],
        "oracle_projected_mask_Dice": mm["dice"],
        "oracle_area_error": mm["area_error"],
        "oracle_mask_center_error_px": mm["center_error_px"],
        "true_area_px": mm["true_area_px"],
        "oracle_area_px": mm["pred_area_px"],
        "true_component_count": true_components,
        "oracle_component_count": oracle_components,
        "oracle_L_m": float(params[0]),
        "oracle_W_m": float(params[1]),
        "oracle_D_m": float(params[2]),
        "oracle_wLD": float(params[3]),
        "oracle_wWD": float(params[4]),
        "oracle_wLW": float(params[5]),
        "oracle_L_abs_error_m": abs(float(params[0]) - float(dataset.L_m[index])),
        "oracle_W_abs_error_m": abs(float(params[1]) - float(dataset.W_m[index])),
        "oracle_D_abs_error_m": abs(float(params[2]) - float(dataset.D_m[index])),
        "oracle_objective": float(best["objective"]),
        "fit_message": str(best["message"]),
    }


def write_preflight_summary(args: argparse.Namespace) -> None:
    lines: list[str] = ["surface shape-extension baseline audit preflight summary", "stage: 25.3", ""]
    checks: list[tuple[str, bool, str]] = []
    cwd_ok = ROOT == Path(r"C:\Users\19166\Desktop\PINN_project")
    checks.append(("root_directory", cwd_ok, str(ROOT)))
    checks.append(("registry_exists", args.registry.exists(), str(args.registry)))
    checks.append(("pilot_manifest_exists", PILOT_MANIFEST.exists(), str(PILOT_MANIFEST)))
    checks.append(("pilot_validation_summary_exists", PILOT_VALIDATION_SUMMARY.exists(), str(PILOT_VALIDATION_SUMMARY)))
    checks.append(("baseline_artifact_manifest_exists", BASELINE_ARTIFACT.exists(), str(BASELINE_ARTIFACT)))
    checks.append(("profile_generator_exists", PROFILE_GENERATOR.exists(), str(PROFILE_GENERATOR)))
    try:
        dataset = load_surface_dataset(args.dataset_id, args.registry)
        split_counts = dict(Counter(dataset.split.tolist()))
        shape_counts = dict(Counter(dataset.shape_type.tolist()))
        delta_error = float(np.max(np.abs(dataset.delta_b - (dataset.b_defect - dataset.b_no_defect))))
        checks.extend(
            [
                ("pilot_dataset_loaded_by_registry_manifest", True, str(dataset.npz_path)),
                ("pilot_npz_sha256_matches_manifest", True, dataset.manifest.get("npz_sha256", "")),
                ("pilot_validation_status_true", dataset.manifest.get("validation_status") is True, str(dataset.manifest.get("validation_status"))),
                ("pilot_train_ready_false", dataset.manifest.get("train_ready_candidate") is False, str(dataset.manifest.get("train_ready_candidate"))),
                ("pilot_baseline_ready_false", dataset.manifest.get("baseline_ready") is False, str(dataset.manifest.get("baseline_ready"))),
                ("delta_b_shape", dataset.delta_b.shape == (120, 3, 3, 201), str(dataset.delta_b.shape)),
                ("delta_b_equals_b_defect_minus_b_no_defect", delta_error <= 1.0e-12, str(delta_error)),
                ("bxyz_finite", bool(np.isfinite(dataset.delta_b).all()), "delta_b finite"),
                ("split_counts", split_counts == {"train": 72, "val": 24, "test": 24}, str(split_counts)),
                ("shape_coverage", shape_counts == dict(dataset.manifest.get("shape_type_counts", {})), str(shape_counts)),
                ("depth_grid_present", dataset.depth_grid_m.shape == (120, 64, 128), str(dataset.depth_grid_m.shape)),
                ("projected_mask_present", dataset.projected_mask_2d.shape == (120, 64, 128), str(dataset.projected_mask_2d.shape)),
                ("non_rbc_compatibility_false", all(not bool(dataset.rbc_compatible[i]) for i, shape in enumerate(dataset.shape_type) if shape in NON_RBC_SHAPES), "checked"),
            ]
        )
    except Exception as exc:
        checks.append(("pilot_dataset_loaded_by_registry_manifest", False, f"{type(exc).__name__}: {exc}"))
    try:
        artifact = read_json(BASELINE_ARTIFACT)
        checkpoint = Path(artifact.get("checkpoint_path", ""))
        prediction = Path(artifact.get("prediction_artifact_path", ""))
        checks.extend(
            [
                ("baseline_artifact_dataset", artifact.get("dataset_id") == "comsol_true_3d_rbc_imported_watertight_pilot_v3_240", str(artifact.get("dataset_id"))),
                ("baseline_checkpoint_exists", checkpoint.exists(), str(checkpoint)),
                ("baseline_prediction_artifact_exists", prediction.exists(), str(prediction)),
                ("baseline_checkpoint_sha256", checkpoint.exists() and sha256_file(checkpoint) == artifact.get("checkpoint_sha256"), str(artifact.get("checkpoint_sha256"))),
                ("baseline_prediction_sha256", prediction.exists() and sha256_file(prediction) == artifact.get("prediction_artifact_sha256"), str(artifact.get("prediction_artifact_sha256"))),
            ]
        )
    except Exception as exc:
        checks.append(("baseline_artifact_loadable", False, f"{type(exc).__name__}: {exc}"))
    validation_text = PILOT_VALIDATION_SUMMARY.read_text(encoding="utf-8", errors="replace") if PILOT_VALIDATION_SUMMARY.exists() else ""
    checks.append(("pilot_validation_passed_summary", "validation_pass: true" in validation_text, "validation_pass: true"))
    checks.append(("CURRENT_BASELINE_unmodified", not git_value(["diff", "--name-only", "--", "CURRENT_BASELINE.md"]), "git diff CURRENT_BASELINE.md"))
    checks.append(("forbidden_artifacts_not_staged", not forbidden_staged(), str(forbidden_staged())))
    all_pass = all(ok for _name, ok, _observed in checks)
    lines.extend(f"- {name}: pass={str(ok).lower()} observed={observed}" for name, ok, observed in checks)
    lines.extend(
        [
            "",
            "policy:",
            "- COMSOL_run: false",
            "- training_run: false",
            "- data_or_npz_mutation: false",
            "- CURRENT_BASELINE_update: false",
            "- latest_newest_npz_scan: false",
            "- dataset_load: COMSOL_DATA_REGISTRY.md + manifest only",
            "",
            f"preflight_decision: {'pass' if all_pass else 'blocker'}",
        ]
    )
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all_pass:
        raise RuntimeError("25.3 preflight blocker; see surface_shape_extension_baseline_audit_preflight_summary.txt")


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_field in ("shape_type", "topology_type", "representation_target"):
        values = sorted({str(row[group_field]) for row in rows})
        for split_name in ("all", "train", "val", "test"):
            for value in values:
                subset = [row for row in rows if str(row[group_field]) == value and (split_name == "all" or row["split"] == split_name)]
                if not subset:
                    continue
                success_rate = float(np.mean([str(row["oracle_fit_success"]).lower() == "true" for row in subset]))
                repr_rate = float(np.mean([str(row["rbc_representable"]).lower() == "true" for row in subset]))
                primary = "RBC representation adequate" if repr_rate >= 0.75 else "RBC representation failure"
                out.append(
                    {
                        "group_field": group_field,
                        "group_value": value,
                        "split": split_name,
                        "sample_count": len(subset),
                        "oracle_fit_success_rate": success_rate,
                        "rbc_representable_rate": repr_rate,
                        "oracle_profile_depth_rmse_mean_m": row_mean(subset, "oracle_profile_depth_rmse_m"),
                        "oracle_profile_depth_rmse_p95_m": row_p95(subset, "oracle_profile_depth_rmse_m"),
                        "oracle_Er_like_mean": row_mean(subset, "oracle_Er_like_error"),
                        "oracle_projected_mask_Dice_mean": row_mean(subset, "oracle_projected_mask_Dice"),
                        "oracle_projected_mask_IoU_mean": row_mean(subset, "oracle_projected_mask_IoU"),
                        "oracle_area_error_mean": row_mean(subset, "oracle_area_error"),
                        "primary_interpretation": primary,
                    }
                )
    return out


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = [row for row in rows if str(row["rbc_representable"]).lower() != "true" or str(row["oracle_fit_success"]).lower() != "true"]
    out: list[dict[str, Any]] = []
    for row in sorted(failed, key=lambda item: (float(item["oracle_profile_depth_rmse_m"]), -float(item["oracle_projected_mask_Dice"])), reverse=True):
        reason = []
        if str(row["oracle_fit_success"]).lower() != "true":
            reason.append("oracle_fit_failed")
        if float(row["oracle_profile_depth_rmse_m"]) > ORACLE_PROFILE_RMSE_PASS_M:
            reason.append("profile_rmse_above_threshold")
        if float(row["oracle_projected_mask_Dice"]) < ORACLE_DICE_PASS:
            reason.append("mask_dice_below_threshold")
        if int(row["true_component_count"]) > 1 and int(row["oracle_component_count"]) < int(row["true_component_count"]):
            reason.append("component_merge")
        out.append({**{field: row.get(field, "") for field in FAILURE_FIELDS if field != "failure_reason"}, "failure_reason": "|".join(reason) or "unknown"})
    return out


def summary_lines(rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> list[str]:
    total = len(rows)
    representable = sum(1 for row in rows if str(row["rbc_representable"]).lower() == "true")
    non_rbc = [row for row in rows if row["shape_type"] in NON_RBC_SHAPES]
    non_rbc_repr = sum(1 for row in non_rbc if str(row["rbc_representable"]).lower() == "true")
    by_shape = {row["group_value"]: row for row in groups if row["group_field"] == "shape_type" and row["split"] == "all"}
    lines = [
        "surface shape-extension RBC oracle fit summary",
        "stage: 25.3",
        "",
        f"dataset_id: {DATASET_ID}",
        f"sample_count: {total}",
        f"oracle_fit_success_count: {sum(1 for row in rows if str(row['oracle_fit_success']).lower() == 'true')}",
        f"rbc_representable_count: {representable}",
        f"rbc_representable_rate: {representable / total:.6f}",
        f"non_rbc_representable_count: {non_rbc_repr}",
        f"non_rbc_representable_rate: {non_rbc_repr / max(len(non_rbc), 1):.6f}",
        f"representability_thresholds: profile_rmse<={ORACLE_PROFILE_RMSE_PASS_M}, dice>={ORACLE_DICE_PASS}",
        "",
        "by_shape:",
    ]
    for shape in sorted(by_shape):
        row = by_shape[shape]
        lines.append(
            f"- {shape}: n={row['sample_count']} representable_rate={float(row['rbc_representable_rate']):.6f} "
            f"rmse_mean={float(row['oracle_profile_depth_rmse_mean_m']):.9f} dice_mean={float(row['oracle_projected_mask_Dice_mean']):.6f} "
            f"interpretation={row['primary_interpretation']}"
        )
    lines.extend(
        [
            "",
            "interpretation:",
            "- RBC oracle uses labels only; Bx/By/Bz are not used for fitting.",
            "- rbc_representable=false means the old six-parameter RBC surface representation is not sufficient for that sample under the fixed thresholds.",
            "- This is not model inference and not training.",
            "",
            f"metrics: {METRICS}",
            f"group_summary: {GROUP_SUMMARY}",
            f"failure_cases: {FAILURE_CASES}",
        ]
    )
    return lines


def run(args: argparse.Namespace) -> int:
    write_preflight_summary(args)
    if args.preflight_only:
        return 0
    dataset = load_surface_dataset(args.dataset_id, args.registry)
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(dataset.sample_ids):
        fit = fit_one_sample(dataset, idx)
        base = {
            "sample_id": str(sample_id),
            "split": str(dataset.split[idx]),
            "shape_type": str(dataset.shape_type[idx]),
            "topology_type": str(dataset.topology_type[idx]),
            "representation_target": str(dataset.representation_target[idx]),
            "rbc_compatible": bool(dataset.rbc_compatible[idx]),
            "component_count": int(dataset.component_count[idx]),
            "label_L_m": float(dataset.L_m[idx]),
            "label_W_m": float(dataset.W_m[idx]),
            "label_D_m": float(dataset.D_m[idx]),
        }
        if not fit.get("oracle_fit_success", False) and "oracle_profile_depth_rmse_m" not in fit:
            fit = {
                **fit,
                "rbc_representable": False,
                "oracle_profile_depth_rmse_m": math.inf,
                "oracle_Er_like_error": math.inf,
                "oracle_projected_mask_IoU": 0.0,
                "oracle_projected_mask_Dice": 0.0,
                "oracle_area_error": math.inf,
                "oracle_mask_center_error_px": math.inf,
                "true_area_px": float(dataset.projected_mask_2d[idx].sum()),
                "oracle_area_px": 0.0,
                "true_component_count": int(dataset.component_count[idx]),
                "oracle_component_count": 0,
                "oracle_L_m": math.nan,
                "oracle_W_m": math.nan,
                "oracle_D_m": math.nan,
                "oracle_wLD": math.nan,
                "oracle_wWD": math.nan,
                "oracle_wLW": math.nan,
                "oracle_L_abs_error_m": math.nan,
                "oracle_W_abs_error_m": math.nan,
                "oracle_D_abs_error_m": math.nan,
                "oracle_objective": math.inf,
            }
        rows.append({**base, **fit})
    groups = group_summary(rows)
    failures = failure_rows(rows)
    write_csv(METRICS, rows, METRIC_FIELDS)
    write_csv(GROUP_SUMMARY, groups, GROUP_FIELDS)
    write_csv(FAILURE_CASES, failures, FAILURE_FIELDS)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(summary_lines(rows, groups)) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
