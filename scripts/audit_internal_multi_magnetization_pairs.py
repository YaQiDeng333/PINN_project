#!/usr/bin/env python
"""23.5 multi-magnetization paired signal audit.

只显式读取 `comsol_internal_defect_multi_magnetization_pack_v1` 的 registry
与 manifest；不扫描 latest/newest，不运行 COMSOL，不训练正式模型，不写 data/NPZ。
本文件同时提供 23.5 后续诊断脚本复用的 loader、feature 和 metric helper。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_multi_magnetization_pack_v1"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_multi_magnetization_pack_v1.manifest.json"
RICHER_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_richer_observation_pack_v1.manifest.json"
PRE_SUMMARY = ROOT / "results/summaries/internal_multi_magnetization_evaluation_preflight_summary.txt"
PAIR_SUMMARY = ROOT / "results/summaries/internal_multi_magnetization_pair_audit_summary.txt"
PAIR_METRICS = ROOT / "results/metrics/internal_multi_magnetization_pair_metrics.csv"

SHAPE_CLASSES = ["internal_cuboid", "internal_ellipsoid", "internal_sphere"]
PARAM_NAMES = ["L_m", "W_m", "D_m", "burial_depth_m", "center_x_m", "center_y_m", "center_z_m"]
CENTER_SLICE = slice(4, 7)

CONFIGS: dict[str, dict[str, Any]] = {
    "mag_x_5line_only": {"variant": "M1_mag_y_5line_z0p008", "mags": [0], "mag_features": False},
    "dual_mag_xy_5line": {"variant": "M1_mag_y_5line_z0p008", "mags": [0, 1], "mag_features": False},
    "dual_mag_xy_5line_plus_mag_features": {"variant": "M1_mag_y_5line_z0p008", "mags": [0, 1], "mag_features": True},
    "mag_x_9line_only": {"variant": "M2_mag_y_9line_z0p008", "mags": [0], "mag_features": False},
    "dual_mag_xy_9line": {"variant": "M2_mag_y_9line_z0p008", "mags": [0, 1], "mag_features": False},
    "dual_mag_xy_9line_plus_mag_features": {"variant": "M2_mag_y_9line_z0p008", "mags": [0, 1], "mag_features": True},
}


@dataclass(frozen=True)
class MultiMagDataset:
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    npz_path: Path
    base_ids: np.ndarray
    split: np.ndarray
    y: np.ndarray
    shape_label: np.ndarray
    shape_type: np.ndarray
    burial_depth_level: np.ndarray
    size_level: np.ndarray
    aspect_bin: np.ndarray
    row_index: dict[tuple[str, str], int]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def strings(arr: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(arr).reshape(-1).tolist()]


def parse_registry(path: Path = REGISTRY) -> dict[str, dict[str, str]]:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_lines(args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def staged_forbidden() -> list[str]:
    staged = git_lines(["diff", "--cached", "--name-only"])
    return [
        path
        for path in staged
        if path.startswith("data/")
        or path.endswith(".npz")
        or path.endswith(".mph")
        or path.startswith("checkpoints/")
        or path.startswith("results/previews/")
        or path.startswith("notes/")
        or path == "CURRENT_BASELINE.md"
        or path == "scripts/visualize_current_baseline.py"
    ]


def protected_status() -> list[str]:
    return git_lines(
        ["status", "--short", "--", "data", "checkpoints", "results/previews", "notes", "CURRENT_BASELINE.md", "scripts/visualize_current_baseline.py"]
    )


def registry_manifest_checks() -> list[dict[str, Any]]:
    registry = parse_registry()
    entry = registry.get(DATASET_ID, {})
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any, notes: str = "") -> None:
        checks.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})

    add("registry_entry_present", bool(entry), "present" if entry else "missing", "present")
    add("manifest_exists", MANIFEST.exists(), str(MANIFEST), "exists")
    if not MANIFEST.exists():
        return checks
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    npz_path = Path(str(manifest.get("npz_path", "")))
    forbidden = set(manifest.get("forbidden_use", []))
    add("dataset_id", manifest.get("dataset_id") == DATASET_ID, manifest.get("dataset_id"), DATASET_ID)
    add("status", manifest.get("status") == "diagnostic_pack_generated", manifest.get("status"), "diagnostic_pack_generated")
    add("npz_exists", npz_path.exists(), str(npz_path), "exists")
    if npz_path.exists():
        add("npz_sha256", sha256_file(npz_path) == manifest.get("npz_sha256"), sha256_file(npz_path), manifest.get("npz_sha256", ""))
    add("assembled_shape", manifest.get("assembled_delta_shape") == [60, 3, 2, 9, 201], manifest.get("assembled_delta_shape"), [60, 3, 2, 9, 201])
    add("base_count", int(manifest.get("base_count", 0)) == 30, manifest.get("base_count"), 30)
    add("paired_complete", int(manifest.get("complete_base_count", 0)) == 30, manifest.get("complete_base_count"), 30)
    add("direction_names", manifest.get("magnetization_direction_names") == ["mag_x", "mag_y"], manifest.get("magnetization_direction_names"), ["mag_x", "mag_y"])
    add("source_je_changed", bool(manifest.get("source_je_changed")), manifest.get("source_je_changed"), True)
    add("baseline_ready_false", not bool(manifest.get("baseline_ready")), manifest.get("baseline_ready"), False)
    add("train_ready_candidate_false", not bool(manifest.get("train_ready_candidate")), manifest.get("train_ready_candidate"), False)
    add("baseline_forbidden", {"baseline_update", "current_baseline_replacement"}.issubset(forbidden), sorted(forbidden), "baseline forbidden")
    add("richer_manifest_exists", RICHER_MANIFEST.exists(), str(RICHER_MANIFEST), "exists")
    for rel in [
        "results/summaries/internal_multi_magnetization_pack_validation_summary.txt",
        "results/summaries/internal_multi_magnetization_pack_route_decision_summary.txt",
        "results/summaries/internal_multi_scan_direction_evaluation_route_decision_summary.txt",
        "results/summaries/internal_defect_b2_failure_audit_summary.txt",
        "results/summaries/internal_defect_inference_abstention_smoke_summary.txt",
    ]:
        add(f"upstream_exists::{rel}", (ROOT / rel).exists(), rel, "exists", "上游诊断证据；缺失只作为可追溯性风险")
    add("no_forbidden_staged", len(staged_forbidden()) == 0, staged_forbidden(), "[]")
    add("protected_paths_clean", len(protected_status()) == 0, protected_status(), "[]")
    return checks


def load_dataset() -> MultiMagDataset:
    checks = registry_manifest_checks()
    hard_fail_names = {
        "registry_entry_present",
        "manifest_exists",
        "dataset_id",
        "status",
        "npz_exists",
        "npz_sha256",
        "assembled_shape",
        "base_count",
        "paired_complete",
        "direction_names",
        "source_je_changed",
        "baseline_ready_false",
        "train_ready_candidate_false",
        "no_forbidden_staged",
        "protected_paths_clean",
    }
    failed = [row for row in checks if (not row["pass"]) and row["check_name"] in hard_fail_names]
    if failed:
        raise RuntimeError("23.5 preflight hard gate failed: " + json.dumps(failed, ensure_ascii=False))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    npz_path = Path(manifest["npz_path"])
    with np.load(npz_path, allow_pickle=True) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    required = [
        "delta_b",
        "scan_line_mask",
        "magnetization_direction_names",
        "magnetization_mask",
        "base_group_id",
        "mag_y_observation_variant",
        "mag_x_observation_variant",
        "shape_type",
        "burial_depth_level",
        "size_level",
        "aspect_bin",
        "L_m",
        "W_m",
        "D_m_or_cavity_size_m",
        "burial_depth_m",
        "defect_center_xyz_m",
    ]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise RuntimeError(f"multi-magnetization NPZ missing fields: {missing}")
    direction_names = strings(arrays["magnetization_direction_names"])
    if direction_names != ["mag_x", "mag_y"]:
        raise RuntimeError(f"unexpected magnetization_direction_names: {direction_names}")
    row_index: dict[tuple[str, str], int] = {}
    for i, (base, variant) in enumerate(zip(strings(arrays["base_group_id"]), strings(arrays["mag_y_observation_variant"]), strict=True)):
        row_index[(base, variant)] = i
    bases = np.asarray(sorted({base for base, _ in row_index}), dtype="<U80")
    first_idx = np.asarray([row_index[(base, "M1_mag_y_5line_z0p008")] for base in bases], dtype=np.int64)
    center = np.asarray(arrays["defect_center_xyz_m"], dtype=np.float32)[first_idx]
    y = np.column_stack(
        [
            np.asarray(arrays["L_m"], dtype=np.float32)[first_idx],
            np.asarray(arrays["W_m"], dtype=np.float32)[first_idx],
            np.asarray(arrays["D_m_or_cavity_size_m"], dtype=np.float32)[first_idx],
            np.asarray(arrays["burial_depth_m"], dtype=np.float32)[first_idx],
            center[:, 0],
            center[:, 1],
            center[:, 2],
        ]
    ).astype(np.float32)
    shape_type = arrays["shape_type"][first_idx].astype(str)
    shape_label = np.asarray([SHAPE_CLASSES.index(str(v)) for v in shape_type], dtype=np.int64)
    split = stratified_base_split(
        bases,
        shape_type,
        arrays["burial_depth_level"][first_idx].astype(str),
        arrays["size_level"][first_idx].astype(str),
        arrays["aspect_bin"][first_idx].astype(str),
    )
    return MultiMagDataset(
        manifest=manifest,
        arrays=arrays,
        npz_path=npz_path,
        base_ids=bases,
        split=split,
        y=y,
        shape_label=shape_label,
        shape_type=shape_type,
        burial_depth_level=arrays["burial_depth_level"][first_idx].astype(str),
        size_level=arrays["size_level"][first_idx].astype(str),
        aspect_bin=arrays["aspect_bin"][first_idx].astype(str),
        row_index=row_index,
    )


def stratified_base_split(base_ids: np.ndarray, shape: np.ndarray, burial: np.ndarray, size: np.ndarray, aspect: np.ndarray) -> np.ndarray:
    """30 base deterministic split: train/val/test = 20/5/5, no paired leakage."""
    quotas = {"train": 20, "val": 5, "test": 5}
    counts = {key: 0 for key in quotas}
    splits = np.full(len(base_ids), "", dtype="<U8")
    shape_total = Counter(strings(shape))
    burial_total = Counter(strings(burial))
    assigned_shape = {split: Counter() for split in quotas}
    assigned_burial = {split: Counter() for split in quotas}
    order_items = []
    for i, base in enumerate(base_ids):
        order_items.append(
            (
                shape_total[str(shape[i])],
                burial_total[str(burial[i])],
                str(shape[i]),
                str(burial[i]),
                str(size[i]),
                str(aspect[i]),
                str(base),
                i,
            )
        )
    for *_, i in sorted(order_items):
        best_split = ""
        best_score = -1e18
        for split_name in ["val", "test", "train"]:
            if counts[split_name] >= quotas[split_name]:
                continue
            shape_need = 1.0 / (1.0 + assigned_shape[split_name][str(shape[i])])
            burial_need = 1.0 / (1.0 + assigned_burial[split_name][str(burial[i])])
            fill = counts[split_name] / max(1, quotas[split_name])
            score = 3.0 * shape_need + 2.0 * burial_need - fill
            if score > best_score:
                best_score = score
                best_split = split_name
        splits[i] = best_split
        counts[best_split] += 1
        assigned_shape[best_split][str(shape[i])] += 1
        assigned_burial[best_split][str(burial[i])] += 1
    return splits


def split_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.where(split == name)[0] for name in ["train", "val", "test"]}


def train_standardize(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def target_scaler(y: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = y[train_idx].mean(axis=0, keepdims=True)
    std = y[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(float)
    b = b.reshape(-1).astype(float)
    if a.size == 0 or b.size == 0 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def signal_summary(delta_mag: np.ndarray, mask_mag: np.ndarray, sensor_x: np.ndarray, lines: np.ndarray) -> dict[str, float]:
    valid = np.where(mask_mag)[0]
    if valid.size == 0:
        return {key: 0.0 for key in ["absmax", "p2p", "rms", "mean_abs", "grad_rms", "energy_x_center", "energy_x_width", "energy_y_center", "energy_y_width", "peak_x", "peak_y"]}
    d = delta_mag[:, valid, :].astype(np.float64)
    line_vals = lines[valid].astype(np.float64)
    abs_d = np.abs(d)
    energy = abs_d.sum(axis=0)
    line_energy = energy.sum(axis=1)
    x_energy = energy.sum(axis=0)
    total = float(x_energy.sum() + 1e-12)
    peak = np.unravel_index(int(np.argmax(abs_d)), abs_d.shape)
    x_center = float((x_energy * sensor_x).sum() / total)
    x_width = float(np.sqrt(((sensor_x - x_center) ** 2 * x_energy).sum() / total))
    line_total = float(line_energy.sum() + 1e-12)
    y_center = float((line_energy * line_vals).sum() / line_total)
    y_width = float(np.sqrt(((line_vals - y_center) ** 2 * line_energy).sum() / line_total))
    return {
        "absmax": float(abs_d.max()),
        "p2p": float(d.max() - d.min()),
        "rms": float(np.sqrt(np.mean(d**2))),
        "mean_abs": float(abs_d.mean()),
        "grad_rms": float(np.sqrt(np.mean(np.diff(d, axis=-1) ** 2))) if d.shape[-1] > 1 else 0.0,
        "energy_x_center": x_center,
        "energy_x_width": x_width,
        "energy_y_center": y_center,
        "energy_y_width": y_width,
        "peak_x": float(sensor_x[peak[2]]),
        "peak_y": float(line_vals[peak[1]]),
    }


def _features_for_mag(delta: np.ndarray, mask: np.ndarray, sensor_x: np.ndarray, lines: np.ndarray, mag_index: int) -> list[float]:
    valid = np.where(mask[mag_index])[0]
    values: list[float] = []
    summary = signal_summary(delta[:, mag_index], mask[mag_index], sensor_x, lines)
    values.extend(summary.values())
    if valid.size == 0:
        values.extend([0.0] * 18)
        return values
    d = delta[:, mag_index, valid, :].astype(np.float64)
    for axis in range(3):
        a = d[axis]
        values.extend(
            [
                float(np.max(np.abs(a))),
                float(np.ptp(a)),
                float(np.sqrt(np.mean(a**2))),
                float(np.mean(np.abs(a))),
                float(np.sqrt(np.mean(np.diff(a, axis=-1) ** 2))),
            ]
        )
    axis_energy = np.asarray([np.sqrt(np.mean(d[axis] ** 2)) for axis in range(3)], dtype=np.float64)
    values.extend(
        [
            float(axis_energy[0] / (axis_energy[1] + 1e-12)),
            float(axis_energy[0] / (axis_energy[2] + 1e-12)),
            float(axis_energy[1] / (axis_energy[2] + 1e-12)),
        ]
    )
    return values


def build_feature_matrix(dataset: MultiMagDataset, config_name: str) -> tuple[np.ndarray, list[str]]:
    spec = CONFIGS[config_name]
    arrays = dataset.arrays
    sensor_x = np.asarray(arrays["sensor_x"], dtype=np.float64)
    rows: list[list[float]] = []
    for base in dataset.base_ids:
        idx = dataset.row_index[(str(base), str(spec["variant"]))]
        delta = np.asarray(arrays["delta_b"][idx], dtype=np.float32)
        mask = np.asarray(arrays["scan_line_mask"][idx], dtype=bool)
        lines = np.asarray(arrays["scan_line_y"][idx], dtype=np.float32)
        features: list[float] = []
        summaries = []
        for mag_index in spec["mags"]:
            features.extend(_features_for_mag(delta, mask, sensor_x, lines, int(mag_index)))
            summaries.append(signal_summary(delta[:, int(mag_index)], mask[int(mag_index)], sensor_x, lines))
        if bool(spec.get("mag_features")) and len(spec["mags"]) == 2:
            sx, sy = summaries
            x_valid = delta[:, 0, mask[0], :].reshape(-1)
            y_valid = delta[:, 1, mask[1], :].reshape(-1)
            min_len = min(x_valid.size, y_valid.size)
            corr = _safe_corr(x_valid[:min_len], y_valid[:min_len]) if min_len else 0.0
            features.extend(
                [
                    float(sy["rms"] / (sx["rms"] + 1e-12)),
                    float(sy["absmax"] / (sx["absmax"] + 1e-12)),
                    float(abs(sy["energy_x_center"] - sx["energy_x_center"])),
                    float(abs(sy["energy_y_center"] - sx["energy_y_center"])),
                    float(abs(sy["peak_x"] - sx["peak_x"])),
                    float(abs(sy["peak_y"] - sx["peak_y"])),
                    corr,
                    float(1.0 - abs(corr)),
                    float(abs(math.log((sy["rms"] + 1e-12) / (sx["rms"] + 1e-12)))),
                ]
            )
        rows.append(features)
    max_len = max(len(row) for row in rows)
    matrix = np.zeros((len(rows), max_len), dtype=np.float32)
    for i, row in enumerate(rows):
        matrix[i, : len(row)] = np.asarray(row, dtype=np.float32)
    return matrix, [f"f{i:03d}" for i in range(max_len)]


def shape_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    f1s: list[float] = []
    for label in range(len(SHAPE_CLASSES)):
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"shape_accuracy": acc, "shape_macro_f1": float(np.mean(f1s))}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict[str, float]:
    err = np.abs(y_pred - y_true)
    norm = err / np.maximum(y_std.reshape(-1), 1e-8)
    center = np.linalg.norm((y_pred[:, CENTER_SLICE] - y_true[:, CENTER_SLICE]) * 1000.0, axis=1)
    return {
        "total_normalized_mae": float(norm.mean()),
        "L_mae_mm": float(err[:, 0].mean() * 1000.0),
        "W_mae_mm": float(err[:, 1].mean() * 1000.0),
        "D_mae_mm": float(err[:, 2].mean() * 1000.0),
        "burial_depth_mae_mm": float(err[:, 3].mean() * 1000.0),
        "center_xyz_component_mae_mm": float(err[:, CENTER_SLICE].mean() * 1000.0),
        "center_xyz_euclidean_mean_mm": float(center.mean()),
    }


def tail_metrics(y_true: np.ndarray, y_pred: np.ndarray, shape_true: np.ndarray, shape_pred: np.ndarray) -> dict[str, float]:
    dim_err = np.abs((y_pred[:, :3] - y_true[:, :3]) * 1000.0)
    burial = np.abs((y_pred[:, 3] - y_true[:, 3]) * 1000.0)
    center = np.linalg.norm((y_pred[:, CENTER_SLICE] - y_true[:, CENTER_SLICE]) * 1000.0, axis=1)
    rel_dim = dim_err / np.maximum(np.abs(y_true[:, :3] * 1000.0), 1e-6)
    center_out = center > 3.0
    burial_out = burial > 1.0
    dim_out = np.any((dim_err > 2.0) | (rel_dim > 0.30), axis=1)
    shape_miss = shape_true != shape_pred
    geometry = shape_miss & center_out & burial_out
    full_shift = center_out & burial_out
    catastrophic = full_shift | geometry | (dim_out & center_out)
    pct = lambda a, q: float(np.percentile(a, q)) if a.size else 0.0
    return {
        "center_xyz_error_mean_mm": float(center.mean()) if center.size else 0.0,
        "center_xyz_error_median_mm": float(np.median(center)) if center.size else 0.0,
        "center_xyz_error_p90_mm": pct(center, 90),
        "center_xyz_error_p95_mm": pct(center, 95),
        "center_xyz_error_max_mm": float(center.max()) if center.size else 0.0,
        "burial_depth_error_mean_mm": float(burial.mean()) if burial.size else 0.0,
        "burial_depth_error_median_mm": float(np.median(burial)) if burial.size else 0.0,
        "burial_depth_error_p90_mm": pct(burial, 90),
        "burial_depth_error_p95_mm": pct(burial, 95),
        "burial_depth_error_max_mm": float(burial.max()) if burial.size else 0.0,
        "catastrophic_failure_count": int(catastrophic.sum()),
        "catastrophic_failure_rate": float(catastrophic.mean()) if catastrophic.size else 0.0,
        "geometry_branch_failure_count": int(geometry.sum()),
        "geometry_branch_failure_rate": float(geometry.mean()) if geometry.size else 0.0,
        "shape_misclassified_count": int(shape_miss.sum()),
        "full_shift_failure_count": int(full_shift.sum()),
    }


def selection_score(row: dict[str, Any]) -> float:
    return (
        float(row["total_normalized_mae"])
        + 0.10 * float(row["burial_depth_mae_mm"])
        + 0.05 * float(row["center_xyz_component_mae_mm"])
        + 0.50 * float(row["catastrophic_failure_rate"])
        + 0.35 * float(row["geometry_branch_failure_rate"])
        + 0.10 * (1.0 - float(row["shape_macro_f1"]))
    )


def metric_row(
    model: str,
    config: str,
    split_name: str,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shape_true: np.ndarray,
    shape_pred: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, Any]:
    row = {
        "model": model,
        "observation_config": config,
        "split": split_name,
        "sample_count": int(idx.size),
        **regression_metrics(y_true[idx], y_pred[idx], y_std),
        **shape_metrics(shape_true[idx], shape_pred[idx]),
        **tail_metrics(y_true[idx], y_pred[idx], shape_true[idx], shape_pred[idx]),
    }
    row["selection_score"] = selection_score(row)
    return row


def write_preflight_summary(dataset: MultiMagDataset, path: Path = PRE_SUMMARY) -> None:
    checks = registry_manifest_checks()
    hard_fail_names = {
        "registry_entry_present",
        "manifest_exists",
        "dataset_id",
        "status",
        "npz_exists",
        "npz_sha256",
        "assembled_shape",
        "base_count",
        "paired_complete",
        "direction_names",
        "source_je_changed",
        "baseline_ready_false",
        "train_ready_candidate_false",
        "no_forbidden_staged",
        "protected_paths_clean",
    }
    hard_passed = all(row["pass"] for row in checks if row["check_name"] in hard_fail_names)
    lines = [
        "23.5 internal multi-magnetization evaluation preflight summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"manifest: {MANIFEST}",
        f"npz_path: {dataset.npz_path}",
        f"hard_preflight_passed: {str(hard_passed).lower()}",
        f"n_rows: {dataset.arrays['delta_b'].shape[0]}",
        f"base_count: {len(dataset.base_ids)}",
        f"delta_b_shape: {tuple(dataset.arrays['delta_b'].shape)}",
        f"magnetization_direction_names: {strings(dataset.arrays['magnetization_direction_names'])}",
        f"magnetization_mask_shape: {tuple(dataset.arrays['magnetization_mask'].shape)}",
        f"scan_line_mask_shape: {tuple(dataset.arrays['scan_line_mask'].shape)}",
        f"split_counts_for_diagnostic_probes: {dict(Counter(strings(dataset.split)))}",
        f"source_je_changed: {dataset.manifest.get('source_je_changed')}",
        "comsol_run: false",
        "formal_training_run: false",
        "data_npz_mutation: false",
        "current_baseline_update: false",
        "",
        "checks:",
    ]
    lines.extend([f"- {row['check_name']}: {row['pass']} observed={row['observed']} expected={row['expected']}" for row in checks])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paired mag_x/mag_y signals for 23.5.")
    parser.add_argument("--preflight-summary", type=Path, default=PRE_SUMMARY)
    parser.add_argument("--summary", type=Path, default=PAIR_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=PAIR_METRICS)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    write_preflight_summary(dataset, args.preflight_summary)
    arrays = dataset.arrays
    sensor_x = np.asarray(arrays["sensor_x"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for base in dataset.base_ids:
        for variant in ["M1_mag_y_5line_z0p008", "M2_mag_y_9line_z0p008"]:
            idx = dataset.row_index[(str(base), variant)]
            delta = arrays["delta_b"][idx]
            mask = arrays["scan_line_mask"][idx]
            lines = arrays["scan_line_y"][idx]
            sx = signal_summary(delta[:, 0], mask[0], sensor_x, lines)
            sy = signal_summary(delta[:, 1], mask[1], sensor_x, lines)
            x_valid = delta[:, 0, mask[0], :].reshape(-1)
            y_valid = delta[:, 1, mask[1], :].reshape(-1)
            min_len = min(x_valid.size, y_valid.size)
            corr = _safe_corr(x_valid[:min_len], y_valid[:min_len]) if min_len else 0.0
            extra_score = float(
                (1.0 - abs(corr))
                + abs(math.log((sy["rms"] + 1e-12) / (sx["rms"] + 1e-12)))
                + abs(sy["energy_y_center"] - sx["energy_y_center"]) * 1000.0
                + abs(sy["energy_x_center"] - sx["energy_x_center"]) * 1000.0
            )
            rows.append(
                {
                    "base_group_id": str(base),
                    "base_sample_id": str(arrays["base_sample_id"][idx]),
                    "variant": variant,
                    "paired_reference_variant": str(arrays["paired_reference_variant"][idx]),
                    "line_count": int(mask[1].sum()),
                    "shape_type": str(arrays["shape_type"][idx]),
                    "burial_depth_level": str(arrays["burial_depth_level"][idx]),
                    "size_level": str(arrays["size_level"][idx]),
                    "aspect_bin": str(arrays["aspect_bin"][idx]),
                    "mag_x_rms": sx["rms"],
                    "mag_y_rms": sy["rms"],
                    "mag_y_over_mag_x_rms": float(sy["rms"] / (sx["rms"] + 1e-12)),
                    "mag_x_absmax": sx["absmax"],
                    "mag_y_absmax": sy["absmax"],
                    "magnetization_corr": corr,
                    "magnetization_nonredundancy_score": extra_score,
                    "mag_x_peak_x_m": sx["peak_x"],
                    "mag_y_peak_x_m": sy["peak_x"],
                    "mag_x_peak_y_m": sx["peak_y"],
                    "mag_y_peak_y_m": sy["peak_y"],
                    "x_center_difference_mm": float(abs(sy["energy_x_center"] - sx["energy_x_center"]) * 1000.0),
                    "y_center_difference_mm": float(abs(sy["energy_y_center"] - sx["energy_y_center"]) * 1000.0),
                    "adds_nonredundant_signal": extra_score > 0.35,
                }
            )

    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    lines_out = [
        "23.5 internal multi-magnetization paired audit summary",
        "",
        f"base_count: {len(dataset.base_ids)}",
        f"rows: {len(rows)}",
        "magnetization_pairing_complete: true",
        f"source_je_changed: {dataset.manifest.get('source_je_changed')}",
    ]
    for variant, values in sorted(by_variant.items()):
        nonred = sum(bool(v["adds_nonredundant_signal"]) for v in values)
        lines_out.append(
            f"{variant}: mean_mag_y_over_mag_x_rms={np.mean([v['mag_y_over_mag_x_rms'] for v in values]):.6f}, "
            f"mean_corr={np.mean([v['magnetization_corr'] for v in values]):.6f}, "
            f"nonredundant_count={nonred}/{len(values)}"
        )
    lines_out.extend(
        [
            "",
            "结论：mag_y 是真实 source 改向后的 paired observation；是否足以改善 shape/center/burial，需要结合 separability 与 diagnostic probe 判断。",
        ]
    )
    write_csv(
        args.metrics,
        rows,
        [
            "base_group_id",
            "base_sample_id",
            "variant",
            "paired_reference_variant",
            "line_count",
            "shape_type",
            "burial_depth_level",
            "size_level",
            "aspect_bin",
            "mag_x_rms",
            "mag_y_rms",
            "mag_y_over_mag_x_rms",
            "mag_x_absmax",
            "mag_y_absmax",
            "magnetization_corr",
            "magnetization_nonredundancy_score",
            "mag_x_peak_x_m",
            "mag_y_peak_x_m",
            "mag_x_peak_y_m",
            "mag_y_peak_y_m",
            "x_center_difference_mm",
            "y_center_difference_mm",
            "adds_nonredundant_signal",
        ],
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
