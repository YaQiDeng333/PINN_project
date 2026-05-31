#!/usr/bin/env python
"""显式加载 internal richer-observation diagnostic pack。

本 loader 只通过 COMSOL_DATA_REGISTRY.md 和 tracked manifest 解析
comsol_internal_defect_richer_observation_pack_v1；没有 latest/newest NPZ scan。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_richer_observation_pack_v1"
REGISTRY_PATH = ROOT / "COMSOL_DATA_REGISTRY.md"
SHAPE_CLASSES = ["internal_cuboid", "internal_ellipsoid", "internal_sphere"]
PARAM_NAMES = ["L_m", "W_m", "D_m", "burial_depth_m", "center_x_m", "center_y_m", "center_z_m"]
CENTER_SLICE = slice(4, 7)
EXPECTED_VARIANTS = [
    "R0_3line_z0p008",
    "R1_5line_z0p008",
    "R1_9line_z0p008",
    "R2_5line_z0p006",
    "R2_5line_z0p010",
    "R2_5line_z0p012",
]
OBSERVATION_CONFIGS: dict[str, list[str]] = {
    "R0_3line_z0p008": ["R0_3line_z0p008"],
    "R1_5line_z0p008": ["R1_5line_z0p008"],
    "R1_9line_z0p008": ["R1_9line_z0p008"],
    "R2_5line_multi_liftoff": [
        "R2_5line_z0p006",
        "R1_5line_z0p008",
        "R2_5line_z0p010",
        "R2_5line_z0p012",
    ],
    "R1_plus_R2_combined": [
        "R1_9line_z0p008",
        "R2_5line_z0p006",
        "R1_5line_z0p008",
        "R2_5line_z0p010",
        "R2_5line_z0p012",
    ],
}


@dataclass(frozen=True)
class RicherObservationDataset:
    dataset_id: str
    manifest: dict[str, Any]
    registry_entry: dict[str, str]
    npz_path: Path
    arrays: dict[str, np.ndarray]
    base_ids: np.ndarray
    split: np.ndarray
    y: np.ndarray
    shape_label: np.ndarray
    shape_type: np.ndarray
    burial_depth_level: np.ndarray
    size_level: np.ndarray
    aspect_bin: np.ndarray
    row_index_by_base_variant: dict[tuple[str, str], int]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, str]]:
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


def parse_list_value(value: str) -> list[str]:
    text = value.strip().strip("`")
    if text.lower() in {"", "none"}:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset(dataset_id: str = DATASET_ID) -> tuple[dict[str, str], dict[str, Any], Path]:
    registry = parse_registry()
    if dataset_id not in registry:
        raise RuntimeError(f"dataset_id not found in COMSOL_DATA_REGISTRY.md: {dataset_id}")
    entry = registry[dataset_id]
    manifest_path = Path(entry.get("manifest_path", ""))
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_path from registry does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != dataset_id:
        raise RuntimeError(f"manifest dataset_id mismatch: {manifest.get('dataset_id')} != {dataset_id}")
    npz_path = Path(manifest["npz_path"])
    return entry, manifest, npz_path


def registry_manifest_checks(entry: dict[str, str], manifest: dict[str, Any], npz_path: Path, dataset_id: str = DATASET_ID) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
        checks.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})

    forbidden = set(manifest.get("forbidden_use", []))
    allowed = set(manifest.get("allowed_use", []))
    entry_forbidden = set(parse_list_value(entry.get("forbidden_use", "")))
    entry_allowed = set(parse_list_value(entry.get("allowed_use", "")))
    add("dataset_id", manifest.get("dataset_id") == dataset_id, manifest.get("dataset_id"), dataset_id)
    add("registry_dataset_id_present", bool(entry), "present", "present")
    add("route", manifest.get("route") == "internal_buried_defect_richer_observation", manifest.get("route"), "internal_buried_defect_richer_observation")
    add("status", manifest.get("status") == "diagnostic_pack_generated", manifest.get("status"), "diagnostic_pack_generated")
    add("validation_passed", bool(manifest.get("validation_passed")), manifest.get("validation_passed"), True)
    add("train_ready_candidate_false", not bool(manifest.get("train_ready_candidate")), manifest.get("train_ready_candidate"), False, "22.9 是 diagnostic pack；23.1 只能作为 training gate，不是 baseline")
    add("baseline_ready_false", (not bool(manifest.get("baseline_ready"))) and (not parse_bool(entry.get("baseline_ready", "true"))), manifest.get("baseline_ready"), False)
    add("explicit_diagnostic_allowed", "explicit_richer_observation_diagnostic" in allowed and "explicit_richer_observation_diagnostic" in entry_allowed, sorted(allowed), "explicit_richer_observation_diagnostic")
    add("baseline_forbidden", {"baseline_update", "current_baseline_replacement"}.issubset(forbidden) and {"baseline_update", "current_baseline_replacement"}.issubset(entry_forbidden), sorted(forbidden), "baseline update forbidden")
    add("latest_newest_forbidden", "latest_newest_auto_discovery" in forbidden and not bool(manifest.get("latest_newest_discovery_allowed")), manifest.get("latest_newest_discovery_allowed"), False)
    add("npz_exists", npz_path.exists(), str(npz_path), "manifest npz_path")
    if npz_path.exists() and manifest.get("npz_sha256"):
        add("npz_sha256", sha256_file(npz_path) == manifest.get("npz_sha256"), sha256_file(npz_path), manifest.get("npz_sha256"))
    return checks


def shape_to_label(shape: np.ndarray) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(SHAPE_CLASSES)}
    return np.asarray([mapping[str(value)] for value in shape], dtype=np.int64)


def stratified_base_split(base_ids: np.ndarray, shape: np.ndarray, burial: np.ndarray, size: np.ndarray, aspect: np.ndarray) -> np.ndarray:
    """30 base 的确定性分组 split；每个 base 的 paired variants 不跨 split。"""
    n = len(base_ids)
    quotas = {"train": max(1, round(n * 2 / 3)), "val": max(1, round(n / 6)), "test": n}
    quotas["test"] = n - quotas["train"] - quotas["val"]
    counts = {name: 0 for name in quotas}
    splits = np.full(n, "", dtype="<U8")
    shape_total = {v: int(np.sum(shape == v)) for v in sorted(set(shape.tolist()))}
    burial_total = {v: int(np.sum(burial == v)) for v in sorted(set(burial.tolist()))}
    assigned_shape = {name: {v: 0 for v in shape_total} for name in quotas}
    assigned_burial = {name: {v: 0 for v in burial_total} for name in quotas}
    rarity = [
        (
            shape_total[str(shape[i])],
            burial_total[str(burial[i])],
            str(shape[i]),
            str(burial[i]),
            str(size[i]),
            str(aspect[i]),
            str(base_ids[i]),
            i,
        )
        for i in range(n)
    ]
    order = [item[-1] for item in sorted(rarity)]
    split_order = ["val", "test", "train"]
    for i in order:
        best_name = ""
        best_score = -1e18
        for name in split_order:
            if counts[name] >= quotas[name]:
                continue
            fill_penalty = counts[name] / max(1, quotas[name])
            shape_need = 1.0 / (1.0 + assigned_shape[name][str(shape[i])])
            burial_need = 1.0 / (1.0 + assigned_burial[name][str(burial[i])])
            score = 3.0 * shape_need + 2.0 * burial_need - fill_penalty
            if score > best_score:
                best_score = score
                best_name = name
        if not best_name:
            best_name = min(counts, key=counts.get)
        splits[i] = best_name
        counts[best_name] += 1
        assigned_shape[best_name][str(shape[i])] += 1
        assigned_burial[best_name][str(burial[i])] += 1
    return splits


def load_dataset(dataset_id: str = DATASET_ID) -> RicherObservationDataset:
    entry, manifest, npz_path = resolve_dataset(dataset_id)
    failed = [row for row in registry_manifest_checks(entry, manifest, npz_path, dataset_id) if not row["pass"]]
    if failed:
        raise RuntimeError("richer-observation registry/manifest gate failed: " + json.dumps(failed, ensure_ascii=False))
    with np.load(npz_path, allow_pickle=True) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    required = [
        "delta_b",
        "b_defect",
        "b_no_defect",
        "base_group_id",
        "observation_variant",
        "scan_line_mask",
        "scan_line_count",
        "sensor_z_m",
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
        raise RuntimeError(f"missing richer-observation NPZ fields: {missing}")
    bases = sorted(set(arrays["base_group_id"].astype(str).tolist()))
    row_index: dict[tuple[str, str], int] = {}
    for i, (base, variant) in enumerate(zip(arrays["base_group_id"].astype(str), arrays["observation_variant"].astype(str), strict=False)):
        row_index[(base, variant)] = i
    missing_pairs = [(base, variant) for base in bases for variant in EXPECTED_VARIANTS if (base, variant) not in row_index]
    if missing_pairs:
        raise RuntimeError(f"incomplete paired variants: {missing_pairs[:5]}")
    first_idx = np.asarray([row_index[(base, EXPECTED_VARIANTS[0])] for base in bases], dtype=np.int64)
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
    shape = arrays["shape_type"].astype(str)[first_idx]
    burial = arrays["burial_depth_level"].astype(str)[first_idx]
    size = arrays["size_level"].astype(str)[first_idx]
    aspect = arrays["aspect_bin"].astype(str)[first_idx]
    split = stratified_base_split(np.asarray(bases), shape, burial, size, aspect)
    return RicherObservationDataset(
        dataset_id=dataset_id,
        manifest=manifest,
        registry_entry=entry,
        npz_path=npz_path,
        arrays=arrays,
        base_ids=np.asarray(bases),
        split=split,
        y=y,
        shape_label=shape_to_label(shape),
        shape_type=shape,
        burial_depth_level=burial,
        size_level=size,
        aspect_bin=aspect,
        row_index_by_base_variant=row_index,
    )


def _line_feature(delta: np.ndarray, mask: np.ndarray, sensor_x: np.ndarray, scan_line_y: np.ndarray) -> list[float]:
    valid = delta[:, mask, :]
    if valid.size == 0:
        return [0.0] * 42
    abs_valid = np.abs(valid)
    features: list[float] = []
    for axis in range(valid.shape[0]):
        a = valid[axis]
        aa = np.abs(a)
        features.extend(
            [
                float(aa.max()),
                float(aa.mean()),
                float(aa.std()),
                float(np.sqrt(np.mean(a * a))),
                float(np.ptp(a)),
                float(sensor_x[np.unravel_index(np.argmax(aa), aa.shape)[1]]),
            ]
        )
    energy_line = np.sum(abs_valid, axis=(0, 2))
    energy_sum = float(energy_line.sum()) + 1e-12
    y_valid = scan_line_y[mask]
    x_grid = np.broadcast_to(sensor_x.reshape(1, 1, -1), valid.shape)
    y_grid = np.broadcast_to(y_valid.reshape(1, -1, 1), valid.shape)
    weights = abs_valid + 1e-12
    features.extend(
        [
            float(np.sum(weights * x_grid) / np.sum(weights)),
            float(np.sum(weights * y_grid) / np.sum(weights)),
            float(energy_line.max() / energy_sum),
            float(energy_line.min() / energy_sum),
            float(np.std(energy_line / energy_sum)),
            float(len(y_valid)),
        ]
    )
    padded = np.zeros(9, dtype=np.float64)
    padded[: energy_line.size] = energy_line / energy_sum
    features.extend(padded.tolist())
    while len(features) < 42:
        features.append(0.0)
    return features[:42]


def build_inputs(dataset: RicherObservationDataset, config_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if config_name not in OBSERVATION_CONFIGS:
        raise KeyError(f"unknown observation config: {config_name}")
    variants = OBSERVATION_CONFIGS[config_name]
    arrays = dataset.arrays
    sensor_x = np.asarray(arrays["sensor_x"], dtype=np.float32)
    delta = np.asarray(arrays["delta_b"], dtype=np.float32)
    scan_line_y = np.asarray(arrays["scan_line_y"], dtype=np.float32)
    scan_mask = np.asarray(arrays["scan_line_mask"], dtype=bool)
    sensor_z = np.asarray(arrays["sensor_z_m"], dtype=np.float32)
    raw_rows: list[np.ndarray] = []
    feature_rows: list[list[float]] = []
    for base in dataset.base_ids.tolist():
        channels: list[np.ndarray] = []
        features: list[float] = []
        energies: list[float] = []
        liftoffs: list[float] = []
        for variant in variants:
            idx = dataset.row_index_by_base_variant[(base, variant)]
            d = delta[idx].copy()
            mask = scan_mask[idx]
            d[:, ~mask, :] = 0.0
            channels.append(d.reshape(27, d.shape[-1]))
            features.extend(_line_feature(delta[idx], mask, sensor_x, scan_line_y[idx]))
            features.extend([float(sensor_z[idx]), float(np.sum(mask)), float(np.max(np.abs(delta[idx]))), float(np.mean(np.abs(delta[idx][:, mask, :])))])
            energies.append(float(np.sqrt(np.mean(delta[idx][:, mask, :] ** 2))))
            liftoffs.append(float(sensor_z[idx]))
        if len(set(round(z, 6) for z in liftoffs)) > 1:
            z = np.asarray(liftoffs, dtype=np.float64)
            e = np.log(np.asarray(energies, dtype=np.float64) + 1e-12)
            slope = float(np.polyfit(z, e, 1)[0])
            features.extend([slope, float(e.max() - e.min())])
        else:
            features.extend([0.0, 0.0])
        raw_rows.append(np.concatenate(channels, axis=0))
        feature_rows.append(features)
    max_len = max(len(row) for row in feature_rows)
    features_arr = np.zeros((len(feature_rows), max_len), dtype=np.float32)
    for i, row in enumerate(feature_rows):
        features_arr[i, : len(row)] = np.asarray(row, dtype=np.float32)
    return np.stack(raw_rows).astype(np.float32), features_arr, variants


def split_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.where(split == name)[0] for name in ["train", "val", "test"]}


def train_scaler(x: np.ndarray, train_idx: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=axes, keepdims=True)
    std = x[train_idx].std(axis=axes, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_matrix(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def target_scaler(y: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = y[train_idx].mean(axis=0, keepdims=True)
    std = y[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((y - mean) / std).astype(np.float32)


def denormalize_y(y_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (y_norm * std + mean).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict[str, float]:
    err = np.abs(y_pred - y_true)
    norm = err / np.maximum(y_std.reshape(-1), 1e-8)
    center_err = np.linalg.norm((y_pred[:, CENTER_SLICE] - y_true[:, CENTER_SLICE]) * 1000.0, axis=1)
    return {
        "total_normalized_mae": float(norm.mean()),
        "L_mae_mm": float(err[:, 0].mean() * 1000.0),
        "W_mae_mm": float(err[:, 1].mean() * 1000.0),
        "D_mae_mm": float(err[:, 2].mean() * 1000.0),
        "burial_depth_mae_mm": float(err[:, 3].mean() * 1000.0),
        "center_xyz_component_mae_mm": float(err[:, CENTER_SLICE].mean() * 1000.0),
        "center_xyz_euclidean_mean_mm": float(center_err.mean()),
    }


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


def metric_row(model: str, config: str, split_name: str, idx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, shape_true: np.ndarray, shape_pred: np.ndarray, y_std: np.ndarray) -> dict[str, Any]:
    reg = regression_metrics(y_true[idx], y_pred[idx], y_std)
    cls = shape_metrics(shape_true[idx], shape_pred[idx])
    tail = tail_metrics(y_true[idx], y_pred[idx], shape_true[idx], shape_pred[idx])
    return {
        "model": model,
        "observation_config": config,
        "split": split_name,
        "sample_count": int(idx.size),
        **reg,
        **cls,
        **tail,
    }


def selection_score(row: dict[str, Any]) -> float:
    return float(
        float(row["total_normalized_mae"])
        + 0.10 * float(row["burial_depth_mae_mm"])
        + 0.05 * float(row["center_xyz_component_mae_mm"])
        + 0.50 * float(row["catastrophic_failure_rate"])
        + 0.35 * float(row["geometry_branch_failure_rate"])
        + 0.10 * (1.0 - float(row["shape_macro_f1"]))
    )
