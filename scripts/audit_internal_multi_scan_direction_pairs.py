#!/usr/bin/env python
"""23.3 dual-direction paired signal audit.

只读读取 `comsol_internal_defect_multi_scan_direction_pack_v1`，不运行 COMSOL、
不训练、不写 data/NPZ、不更新 CURRENT_BASELINE.md。该文件同时提供 23.3
后续诊断脚本复用的 loader / feature / metric helper。
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
DATASET_ID = "comsol_internal_defect_multi_scan_direction_pack_v1"
RICHER_DATASET_ID = "comsol_internal_defect_richer_observation_pack_v1"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_multi_scan_direction_pack_v1.manifest.json"
RICHER_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_richer_observation_pack_v1.manifest.json"
PRE_SUMMARY = ROOT / "results/summaries/internal_multi_scan_direction_evaluation_preflight_summary.txt"
PAIR_SUMMARY = ROOT / "results/summaries/internal_multi_scan_direction_pair_audit_summary.txt"
PAIR_METRICS = ROOT / "results/metrics/internal_multi_scan_direction_pair_metrics.csv"

SHAPE_CLASSES = ["internal_cuboid", "internal_ellipsoid", "internal_sphere"]
PARAM_NAMES = ["L_m", "W_m", "D_m", "burial_depth_m", "center_x_m", "center_y_m", "center_z_m"]
CENTER_SLICE = slice(4, 7)
CONFIGS: dict[str, dict[str, Any]] = {
    "single_x_5line": {"variant": "D1_y_scan_5line_z0p008", "directions": [0], "direction_features": False},
    "dual_xy_5line": {"variant": "D1_y_scan_5line_z0p008", "directions": [0, 1], "direction_features": False},
    "dual_xy_5line_plus_direction_features": {"variant": "D1_y_scan_5line_z0p008", "directions": [0, 1], "direction_features": True},
    "single_x_9line": {"variant": "D2_y_scan_9line_z0p008", "directions": [0], "direction_features": False},
    "dual_xy_9line": {"variant": "D2_y_scan_9line_z0p008", "directions": [0, 1], "direction_features": False},
    "dual_xy_9line_plus_direction_features": {"variant": "D2_y_scan_9line_z0p008", "directions": [0, 1], "direction_features": True},
}


@dataclass(frozen=True)
class MultiScanDataset:
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
    return git_lines(["status", "--short", "--", "data", "checkpoints", "results/previews", "notes", "CURRENT_BASELINE.md", "scripts/visualize_current_baseline.py"])


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
    add("baseline_ready_false", not bool(manifest.get("baseline_ready")), manifest.get("baseline_ready"), False)
    add("train_ready_candidate_false", not bool(manifest.get("train_ready_candidate")), manifest.get("train_ready_candidate"), False)
    add("baseline_forbidden", {"baseline_update", "current_baseline_replacement"}.issubset(forbidden), sorted(forbidden), "baseline forbidden")
    add("richer_manifest_exists", RICHER_MANIFEST.exists(), str(RICHER_MANIFEST), "exists")
    add("no_forbidden_staged", len(staged_forbidden()) == 0, staged_forbidden(), "[]")
    add("protected_paths_clean", len(protected_status()) == 0, protected_status(), "[]")
    return checks


def load_dataset() -> MultiScanDataset:
    failed = [row for row in registry_manifest_checks() if not row["pass"]]
    if failed:
        raise RuntimeError("23.3 preflight gate failed: " + json.dumps(failed, ensure_ascii=False))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    npz_path = Path(manifest["npz_path"])
    with np.load(npz_path, allow_pickle=True) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    required = [
        "delta_b",
        "scan_line_mask",
        "direction_names",
        "path_coordinate_axis",
        "line_coordinate_axis",
        "path_coordinate_m",
        "line_coordinate_m",
        "base_group_id",
        "y_observation_variant",
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
        raise RuntimeError(f"multi-scan NPZ missing fields: {missing}")
    direction_names = strings(arrays["direction_names"])
    if direction_names != ["x_scan", "y_scan"]:
        raise RuntimeError(f"unexpected direction_names: {direction_names}")
    if strings(arrays["path_coordinate_axis"]) != ["x", "y"] or strings(arrays["line_coordinate_axis"]) != ["y", "x"]:
        raise RuntimeError("direction coordinate axes are not x/y and y/x")
    row_index: dict[tuple[str, str], int] = {}
    for i, (base, variant) in enumerate(zip(strings(arrays["base_group_id"]), strings(arrays["y_observation_variant"]), strict=True)):
        row_index[(base, variant)] = i
    bases = np.asarray(sorted({base for base, _ in row_index}), dtype="<U80")
    first_idx = np.asarray([row_index[(base, "D1_y_scan_5line_z0p008")] for base in bases], dtype=np.int64)
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
    return MultiScanDataset(
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
    rarity = []
    for i, base in enumerate(base_ids):
        rarity.append((shape_total[str(shape[i])], burial_total[str(burial[i])], str(shape[i]), str(burial[i]), str(size[i]), str(aspect[i]), str(base), i))
    order = [item[-1] for item in sorted(rarity)]
    for i in order:
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
        if not best_split:
            best_split = min(counts, key=counts.get)
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


def direction_signal_summary(delta_dir: np.ndarray, mask_dir: np.ndarray, path: np.ndarray, lines: np.ndarray) -> dict[str, float]:
    valid = np.where(mask_dir)[0]
    if valid.size == 0:
        return {key: 0.0 for key in ["absmax", "p2p", "rms", "mean_abs", "grad_rms", "energy_path_center", "energy_path_width", "energy_line_center", "energy_line_width", "peak_path", "peak_line"]}
    d = delta_dir[:, valid, :].astype(np.float64)
    line_vals = lines[valid].astype(np.float64)
    abs_d = np.abs(d)
    energy = abs_d.sum(axis=0)
    line_energy = energy.sum(axis=1)
    path_energy = energy.sum(axis=0)
    total = float(path_energy.sum() + 1e-12)
    peak = np.unravel_index(int(np.argmax(abs_d)), abs_d.shape)
    path_center = float((path_energy * path).sum() / total)
    path_width = float(np.sqrt(((path - path_center) ** 2 * path_energy).sum() / total))
    line_total = float(line_energy.sum() + 1e-12)
    line_center = float((line_energy * line_vals).sum() / line_total)
    line_width = float(np.sqrt(((line_vals - line_center) ** 2 * line_energy).sum() / line_total))
    return {
        "absmax": float(abs_d.max()),
        "p2p": float(d.max() - d.min()),
        "rms": float(np.sqrt(np.mean(d**2))),
        "mean_abs": float(abs_d.mean()),
        "grad_rms": float(np.sqrt(np.mean(np.diff(d, axis=-1) ** 2))) if d.shape[-1] > 1 else 0.0,
        "energy_path_center": path_center,
        "energy_path_width": path_width,
        "energy_line_center": line_center,
        "energy_line_width": line_width,
        "peak_path": float(path[peak[2]]),
        "peak_line": float(line_vals[peak[1]]),
    }


def _features_for_direction(delta: np.ndarray, mask: np.ndarray, path: np.ndarray, lines: np.ndarray, direction: int) -> list[float]:
    valid = np.where(mask[direction])[0]
    values: list[float] = []
    summary = direction_signal_summary(delta[:, direction], mask[direction], path[direction], lines[direction])
    values.extend(summary.values())
    if valid.size == 0:
        values.extend([0.0] * 18)
        return values
    d = delta[:, direction, valid, :].astype(np.float64)
    for axis in range(3):
        a = d[axis]
        values.extend([float(np.max(np.abs(a))), float(np.ptp(a)), float(np.sqrt(np.mean(a**2))), float(np.mean(np.abs(a))), float(np.sqrt(np.mean(np.diff(a, axis=-1) ** 2)))])
    axis_energy = np.asarray([np.sqrt(np.mean(d[axis] ** 2)) for axis in range(3)], dtype=np.float64)
    values.extend(
        [
            float(axis_energy[0] / (axis_energy[1] + 1e-12)),
            float(axis_energy[0] / (axis_energy[2] + 1e-12)),
            float(axis_energy[1] / (axis_energy[2] + 1e-12)),
        ]
    )
    return values


def build_feature_matrix(dataset: MultiScanDataset, config_name: str) -> tuple[np.ndarray, list[str]]:
    spec = CONFIGS[config_name]
    arrays = dataset.arrays
    rows: list[list[float]] = []
    for base in dataset.base_ids:
        idx = dataset.row_index[(str(base), str(spec["variant"]))]
        delta = np.asarray(arrays["delta_b"][idx], dtype=np.float32)
        mask = np.asarray(arrays["scan_line_mask"][idx], dtype=bool)
        path = np.asarray(arrays["path_coordinate_m"][idx], dtype=np.float32)
        lines = np.asarray(arrays["line_coordinate_m"][idx], dtype=np.float32)
        features: list[float] = []
        summaries = []
        for direction in spec["directions"]:
            features.extend(_features_for_direction(delta, mask, path, lines, int(direction)))
            summaries.append(direction_signal_summary(delta[:, int(direction)], mask[int(direction)], path[int(direction)], lines[int(direction)]))
        if bool(spec.get("direction_features")) and len(spec["directions"]) == 2:
            sx, sy = summaries
            x_valid = delta[:, 0, mask[0], :].reshape(-1)
            y_valid = delta[:, 1, mask[1], :].reshape(-1)
            min_len = min(x_valid.size, y_valid.size)
            corr = _safe_corr(x_valid[:min_len], y_valid[:min_len]) if min_len else 0.0
            features.extend(
                [
                    float(sy["rms"] / (sx["rms"] + 1e-12)),
                    float(sy["absmax"] / (sx["absmax"] + 1e-12)),
                    float(abs(sy["energy_path_center"] - sx["energy_path_center"])),
                    float(abs(sy["energy_line_center"] - sx["energy_line_center"])),
                    float(abs(sy["peak_path"] - sx["peak_path"])),
                    float(abs(sy["peak_line"] - sx["peak_line"])),
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
    return float(row["total_normalized_mae"]) + 0.10 * float(row["burial_depth_mae_mm"]) + 0.05 * float(row["center_xyz_component_mae_mm"]) + 0.50 * float(row["catastrophic_failure_rate"]) + 0.35 * float(row["geometry_branch_failure_rate"]) + 0.10 * (1.0 - float(row["shape_macro_f1"]))


def metric_row(model: str, config: str, split_name: str, idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, shape_true: np.ndarray, shape_pred: np.ndarray, y_std: np.ndarray) -> dict[str, Any]:
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


def write_preflight_summary(dataset: MultiScanDataset, path: Path = PRE_SUMMARY) -> None:
    checks = registry_manifest_checks()
    passed = all(row["pass"] for row in checks)
    lines = [
        "23.3 internal multi-scan-direction evaluation preflight summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"manifest: {MANIFEST}",
        f"npz_path: {dataset.npz_path}",
        f"preflight_passed: {str(passed).lower()}",
        f"n_rows: {dataset.arrays['delta_b'].shape[0]}",
        f"base_count: {len(dataset.base_ids)}",
        f"delta_b_shape: {tuple(dataset.arrays['delta_b'].shape)}",
        f"direction_names: {strings(dataset.arrays['direction_names'])}",
        f"path_coordinate_axis: {strings(dataset.arrays['path_coordinate_axis'])}",
        f"line_coordinate_axis: {strings(dataset.arrays['line_coordinate_axis'])}",
        f"split_counts_for_diagnostic_probes: {dict(Counter(strings(dataset.split)))}",
        f"richer_observation_manifest_exists: {RICHER_MANIFEST.exists()}",
        f"23_1_metrics_exists: {(ROOT / 'results/metrics/internal_richer_observation_metrics.csv').exists()}",
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
    parser = argparse.ArgumentParser(description="Audit paired x/y scan signals for 23.3.")
    parser.add_argument("--preflight-summary", type=Path, default=PRE_SUMMARY)
    parser.add_argument("--summary", type=Path, default=PAIR_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=PAIR_METRICS)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    write_preflight_summary(dataset, args.preflight_summary)
    arrays = dataset.arrays
    rows: list[dict[str, Any]] = []
    for base in dataset.base_ids:
        for variant in ["D1_y_scan_5line_z0p008", "D2_y_scan_9line_z0p008"]:
            idx = dataset.row_index[(str(base), variant)]
            delta = arrays["delta_b"][idx]
            mask = arrays["scan_line_mask"][idx]
            path = arrays["path_coordinate_m"][idx]
            lines = arrays["line_coordinate_m"][idx]
            sx = direction_signal_summary(delta[:, 0], mask[0], path[0], lines[0])
            sy = direction_signal_summary(delta[:, 1], mask[1], path[1], lines[1])
            x_valid = delta[:, 0, mask[0], :].reshape(-1)
            y_valid = delta[:, 1, mask[1], :].reshape(-1)
            min_len = min(x_valid.size, y_valid.size)
            corr = _safe_corr(x_valid[:min_len], y_valid[:min_len]) if min_len else 0.0
            extra_score = float((1.0 - abs(corr)) + abs(math.log((sy["rms"] + 1e-12) / (sx["rms"] + 1e-12))) + abs(sy["energy_line_center"] - sx["energy_line_center"]) * 1000.0)
            rows.append(
                {
                    "base_group_id": str(base),
                    "base_sample_id": str(arrays["base_sample_id"][idx]),
                    "variant": variant,
                    "line_count": int(arrays["scan_line_mask"][idx, 1].sum()),
                    "shape_type": str(arrays["shape_type"][idx]),
                    "burial_depth_level": str(arrays["burial_depth_level"][idx]),
                    "size_level": str(arrays["size_level"][idx]),
                    "aspect_bin": str(arrays["aspect_bin"][idx]),
                    "x_rms": sx["rms"],
                    "y_rms": sy["rms"],
                    "y_over_x_rms": float(sy["rms"] / (sx["rms"] + 1e-12)),
                    "x_absmax": sx["absmax"],
                    "y_absmax": sy["absmax"],
                    "direction_corr": corr,
                    "direction_nonredundancy_score": extra_score,
                    "x_peak_path_m": sx["peak_path"],
                    "y_peak_path_m": sy["peak_path"],
                    "x_peak_line_m": sx["peak_line"],
                    "y_peak_line_m": sy["peak_line"],
                    "path_center_difference_mm": float(abs(sy["energy_path_center"] - sx["energy_path_center"]) * 1000.0),
                    "line_center_difference_mm": float(abs(sy["energy_line_center"] - sx["energy_line_center"]) * 1000.0),
                    "adds_nonredundant_signal": extra_score > 0.35,
                }
            )
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    lines_out = [
        "23.3 internal multi-scan-direction paired direction audit summary",
        "",
        f"base_count: {len(dataset.base_ids)}",
        f"rows: {len(rows)}",
        "direction_pairing_complete: true",
    ]
    for variant, values in sorted(by_variant.items()):
        nonred = sum(bool(v["adds_nonredundant_signal"]) for v in values)
        lines_out.append(
            f"{variant}: mean_y_over_x_rms={np.mean([v['y_over_x_rms'] for v in values]):.6f}, "
            f"mean_corr={np.mean([v['direction_corr'] for v in values]):.6f}, "
            f"nonredundant_count={nonred}/{len(values)}"
        )
    lines_out.extend(
        [
            "",
            "结论：y_scan 为每个 base 提供了与 x_scan 可配对的正交观测；是否足以降低 shape/center/burial tail 需要看 separability 和 diagnostic probe。",
        ]
    )
    write_csv(
        args.metrics,
        rows,
        [
            "base_group_id",
            "base_sample_id",
            "variant",
            "line_count",
            "shape_type",
            "burial_depth_level",
            "size_level",
            "aspect_bin",
            "x_rms",
            "y_rms",
            "y_over_x_rms",
            "x_absmax",
            "y_absmax",
            "direction_corr",
            "direction_nonredundancy_score",
            "x_peak_path_m",
            "y_peak_path_m",
            "x_peak_line_m",
            "y_peak_line_m",
            "path_center_difference_mm",
            "line_center_difference_mm",
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
