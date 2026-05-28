#!/usr/bin/env python
"""Explicit loader for the 21.1 internal / buried defect pilot pack.

All dataset resolution goes through COMSOL_DATA_REGISTRY.md and the tracked
manifest. There is intentionally no latest/newest NPZ discovery path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_pilot_pack_v1"
ROUTE = "internal_buried_defect_feasibility"
REGISTRY_PATH = ROOT / "COMSOL_DATA_REGISTRY.md"
PARAM_NAMES = ["L_m", "W_m", "D_m", "burial_depth_m", "center_x_m", "center_y_m", "center_z_m"]
DIMENSION_NAMES = ["L_m", "W_m", "D_m"]
CENTER_NAMES = ["center_x_m", "center_y_m", "center_z_m"]
SHAPE_CLASSES = ["internal_cuboid", "internal_ellipsoid", "internal_sphere"]
CHECK_FIELDS = ["check_name", "pass", "observed", "expected", "notes"]


@dataclass(frozen=True)
class InternalDefectDataset:
    dataset_id: str
    manifest: dict[str, Any]
    registry_entry: dict[str, str]
    npz_path: Path
    delta_b: np.ndarray
    b_defect: np.ndarray
    b_no_defect: np.ndarray
    x_channels: np.ndarray
    y_regression: np.ndarray
    shape_label: np.ndarray
    shape_type: np.ndarray
    sample_ids: np.ndarray
    split: np.ndarray
    axis_names: list[str]
    sensor_x: np.ndarray
    scan_line_y: np.ndarray
    sensor_z_m: np.ndarray
    burial_depth_level: np.ndarray
    size_level: np.ndarray
    aspect_bin: np.ndarray
    cavity_internal: np.ndarray
    ground_truth_method: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate the internal defect pilot dataset.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def parse_list_value(value: str) -> list[str]:
    text = value.strip().strip("`")
    if text.lower() in {"", "none"}:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_registry(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
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


def resolve_dataset(dataset_id: str = DATASET_ID, registry_path: Path = REGISTRY_PATH) -> tuple[dict[str, str], dict[str, Any], Path]:
    registry = parse_registry(registry_path)
    if dataset_id not in registry:
        raise RuntimeError(f"dataset_id not found in registry: {dataset_id}")
    entry = registry[dataset_id]
    manifest_path = Path(entry.get("manifest_path", ""))
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_path from registry does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != dataset_id:
        raise RuntimeError(f"manifest dataset_id mismatch: {manifest.get('dataset_id')} != {dataset_id}")
    npz_path = Path(manifest["npz_path"])
    return entry, manifest, npz_path


def gate_manifest(entry: dict[str, str], manifest: dict[str, Any], npz_path: Path, dataset_id: str = DATASET_ID) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
        checks.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})

    allowed = set(manifest.get("allowed_use", []))
    forbidden = set(manifest.get("forbidden_use", []))
    entry_allowed = set(parse_list_value(entry.get("allowed_use", "")))
    entry_forbidden = set(parse_list_value(entry.get("forbidden_use", "")))
    add("dataset_id", manifest.get("dataset_id") == dataset_id, manifest.get("dataset_id"), dataset_id)
    add("route", manifest.get("route") == ROUTE and entry.get("route") == ROUTE, f"manifest={manifest.get('route')}; registry={entry.get('route')}", ROUTE)
    add("status", manifest.get("status") == "pilot_generated" and entry.get("status") == "pilot_generated", f"manifest={manifest.get('status')}; registry={entry.get('status')}", "pilot_generated")
    add("train_ready_candidate", bool(manifest.get("train_ready_candidate")) and parse_bool(entry.get("train_ready_candidate", "false")), manifest.get("train_ready_candidate"), True)
    add("baseline_ready_false", (not bool(manifest.get("baseline_ready"))) and (not parse_bool(entry.get("baseline_ready", "true"))), manifest.get("baseline_ready"), False)
    add("explicit_internal_training_allowed", "explicit_internal_training_gate" in allowed and "explicit_internal_training_gate" in entry_allowed, sorted(allowed), "explicit_internal_training_gate")
    add("baseline_forbidden", {"baseline_update", "current_baseline_replacement"}.issubset(forbidden) and {"baseline_update", "current_baseline_replacement"}.issubset(entry_forbidden), sorted(forbidden), "baseline update forbidden")
    add("latest_newest_forbidden", "latest_newest_auto_discovery" in forbidden and not bool(manifest.get("latest_newest_discovery_allowed")), manifest.get("latest_newest_discovery_allowed"), False)
    add("auto_discovery_forbidden", not bool(manifest.get("auto_discovery_allowed")), manifest.get("auto_discovery_allowed"), False)
    add("npz_exists", npz_path.exists(), str(npz_path), "NPZ path from manifest")
    if npz_path.exists():
        add("npz_sha256", sha256_file(npz_path) == manifest.get("npz_sha256"), sha256_file(npz_path), manifest.get("npz_sha256"))
    return checks


def split_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    result = {name: np.where(split == name)[0] for name in ["train", "val", "test"]}
    return result


def train_normalization(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = x[train_idx]
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_x(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def train_target_scaler(y: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = y[train_idx].mean(axis=0, keepdims=True)
    std = y[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((y - mean) / std).astype(np.float32)


def denormalize_y(y_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (y_norm * std + mean).astype(np.float32)


def shape_to_label(shape: np.ndarray) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(SHAPE_CLASSES)}
    return np.asarray([mapping[str(value)] for value in shape], dtype=np.int64)


def load_dataset(dataset_id: str = DATASET_ID, registry_path: Path = REGISTRY_PATH) -> InternalDefectDataset:
    entry, manifest, npz_path = resolve_dataset(dataset_id, registry_path)
    failed = [row for row in gate_manifest(entry, manifest, npz_path, dataset_id) if not row["pass"]]
    if failed:
        raise RuntimeError("dataset registry/manifest gate failed: " + json.dumps(failed, ensure_ascii=False))
    with np.load(npz_path, allow_pickle=True) as npz:
        required = [
            "delta_b",
            "b_defect",
            "b_no_defect",
            "axis_names",
            "sensor_x",
            "scan_line_y",
            "sensor_z_m",
            "sample_ids",
            "split",
            "shape_type",
            "burial_depth_level",
            "size_level",
            "aspect_bin",
            "L_m",
            "W_m",
            "D_m",
            "D_m_or_cavity_size_m",
            "burial_depth_m",
            "depth_to_surface_m",
            "defect_center_xyz_m",
            "cavity_internal",
            "ground_truth_method",
        ]
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise RuntimeError(f"missing NPZ fields: {missing}")
        delta_b = np.asarray(npz["delta_b"], dtype=np.float32)
        center = np.asarray(npz["defect_center_xyz_m"], dtype=np.float32)
        y = np.column_stack(
            [
                np.asarray(npz["L_m"], dtype=np.float32),
                np.asarray(npz["W_m"], dtype=np.float32),
                np.asarray(npz["D_m_or_cavity_size_m"], dtype=np.float32),
                np.asarray(npz["burial_depth_m"], dtype=np.float32),
                center[:, 0],
                center[:, 1],
                center[:, 2],
            ]
        ).astype(np.float32)
        shape_type = np.asarray(npz["shape_type"]).astype(str)
        dataset = InternalDefectDataset(
            dataset_id=dataset_id,
            manifest=manifest,
            registry_entry=entry,
            npz_path=npz_path,
            delta_b=delta_b,
            b_defect=np.asarray(npz["b_defect"], dtype=np.float32),
            b_no_defect=np.asarray(npz["b_no_defect"], dtype=np.float32),
            x_channels=delta_b.reshape(delta_b.shape[0], 9, delta_b.shape[-1]).astype(np.float32),
            y_regression=y,
            shape_label=shape_to_label(shape_type),
            shape_type=shape_type,
            sample_ids=np.asarray(npz["sample_ids"]).astype(str),
            split=np.asarray(npz["split"]).astype(str),
            axis_names=[str(x) for x in np.asarray(npz["axis_names"]).reshape(-1).tolist()],
            sensor_x=np.asarray(npz["sensor_x"], dtype=np.float64),
            scan_line_y=np.asarray(npz["scan_line_y"], dtype=np.float32),
            sensor_z_m=np.asarray(npz["sensor_z_m"], dtype=np.float32),
            burial_depth_level=np.asarray(npz["burial_depth_level"]).astype(str),
            size_level=np.asarray(npz["size_level"]).astype(str),
            aspect_bin=np.asarray(npz["aspect_bin"]).astype(str),
            cavity_internal=np.asarray(npz["cavity_internal"], dtype=bool),
            ground_truth_method=np.asarray(npz["ground_truth_method"]).astype(str),
        )
    return dataset


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scale: np.ndarray | None = None) -> dict[str, float]:
    err = np.abs(y_true - y_pred)
    scale = y_scale.reshape(1, -1) if y_scale is not None else np.ones((1, y_true.shape[1]), dtype=np.float32)
    norm_err = err / np.where(scale == 0, 1.0, scale)
    return {
        "total_normalized_mae": float(norm_err.mean()),
        "dimension_mae_mm": float(err[:, :3].mean() * 1000.0),
        "L_mae_mm": float(err[:, 0].mean() * 1000.0),
        "W_mae_mm": float(err[:, 1].mean() * 1000.0),
        "D_mae_mm": float(err[:, 2].mean() * 1000.0),
        "burial_depth_mae_mm": float(err[:, 3].mean() * 1000.0),
        "center_xyz_mae_mm": float(err[:, 4:7].mean() * 1000.0),
        "center_x_mae_mm": float(err[:, 4].mean() * 1000.0),
        "center_y_mae_mm": float(err[:, 5].mean() * 1000.0),
        "center_z_mae_mm": float(err[:, 6].mean() * 1000.0),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> dict[str, float]:
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1s: list[float] = []
    for cls in range(n_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return {"shape_accuracy": acc, "shape_macro_f1": float(np.mean(f1s))}


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset_id, args.registry)
    print(json.dumps({"dataset_id": dataset.dataset_id, "n": int(dataset.delta_b.shape[0]), "split": {k: int(v.size) for k, v in split_indices(dataset.split).items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
