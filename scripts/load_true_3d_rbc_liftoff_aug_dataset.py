#!/usr/bin/env python
"""Explicit loader for the 20.91b true-3D RBC liftoff augmentation pack.

This module intentionally has no latest/newest discovery path. The liftoff pack
is generated data, so callers must resolve it through COMSOL_DATA_REGISTRY.md
and its tracked manifest before reading the ignored NPZ.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import load_true_3d_rbc_pilot_dataset as pilot


ROOT = pilot.ROOT
DATASET_ID = "comsol_true_3d_rbc_liftoff_aug_pack_v1"
ROUTE = "true_3d_piao_style_liftoff_robustness"
STATUS = "diagnostic_pack_generated"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc_liftoff_aug"
LIFTOFF_LEVELS = [0.006, 0.008, 0.010, 0.012]
PARAM_NAMES = pilot.PARAM_NAMES


@dataclass(frozen=True)
class True3DRBCLiftoffDataset:
    dataset_id: str
    manifest: dict[str, Any]
    registry_entry: dict[str, str]
    npz_path: Path
    delta_b: np.ndarray
    b_defect: np.ndarray
    b_no_defect: np.ndarray
    x_channels: np.ndarray
    rbc_params: np.ndarray
    profile_pose: np.ndarray
    projected_mask_2d: np.ndarray
    profile_depth_grid_m: np.ndarray
    profile_depth_map_xy_m: np.ndarray
    sample_ids: np.ndarray
    base_sample_ids: np.ndarray
    source_sample_ids: np.ndarray
    variant_name: np.ndarray
    factor_group: np.ndarray
    row_kind: np.ndarray
    split: np.ndarray
    axis_names: list[str]
    axis_expressions: list[str]
    sensor_x: np.ndarray
    scan_line_y: np.ndarray
    sensor_z_m: np.ndarray
    liftoff_delta_m: np.ndarray
    jscale: np.ndarray
    scan_bundle_y_offset_m: np.ndarray
    curvature_template: np.ndarray
    depth_bin: np.ndarray
    aspect_bin: np.ndarray
    size_bin: np.ndarray


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return pilot.parse_list_value(str(value))


def resolve_liftoff_dataset(
    dataset_id: str = DATASET_ID, registry_path: Path = pilot.REGISTRY_PATH
) -> tuple[dict[str, str], dict[str, Any], Path]:
    entry, manifest, npz_path = pilot.resolve_dataset(dataset_id, registry_path)
    return entry, manifest, npz_path


def gate_liftoff_manifest(
    entry: dict[str, str], manifest: dict[str, Any], npz_path: Path, dataset_id: str = DATASET_ID
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, notes: str = "") -> None:
        checks.append({"check_name": name, "pass": bool(passed), "observed": observed, "notes": notes})

    allowed = set(_as_list(manifest.get("allowed_use", [])))
    forbidden = set(_as_list(manifest.get("forbidden_use", [])))
    entry_allowed = set(pilot.parse_list_value(entry.get("allowed_use", "")))
    entry_forbidden = set(pilot.parse_list_value(entry.get("forbidden_use", "")))
    levels = [round(float(x), 3) for x in manifest.get("liftoff_levels_m", [])]

    add("dataset_id", manifest.get("dataset_id") == dataset_id, manifest.get("dataset_id"))
    add("route", manifest.get("route") == ROUTE and entry.get("route") == ROUTE, f"manifest={manifest.get('route')}; registry={entry.get('route')}")
    add("status", manifest.get("status") == STATUS and entry.get("status") == STATUS, f"manifest={manifest.get('status')}; registry={entry.get('status')}")
    add("schema_version", manifest.get("schema_version") == SCHEMA_VERSION, manifest.get("schema_version"))
    add("train_ready_candidate", bool(manifest.get("train_ready_candidate")) and pilot.parse_bool(entry.get("train_ready_candidate", "false")), manifest.get("train_ready_candidate"))
    add("baseline_ready_false", (not bool(manifest.get("baseline_ready"))) and (not pilot.parse_bool(entry.get("baseline_ready", "true"))), manifest.get("baseline_ready"))
    add("geometry_method", manifest.get("geometry_method") == "imported_watertight_mesh_solid", manifest.get("geometry_method"))
    add("exact_piao_rbc_false", not bool(manifest.get("exact_piao_rbc")), manifest.get("exact_piao_rbc"))
    add("rbc_style_approximation_true", bool(manifest.get("rbc_style_approximation")), manifest.get("rbc_style_approximation"))
    add("explicit_liftoff_training_allowed", "explicit_liftoff_training_gate" in allowed and "explicit_liftoff_training_gate" in entry_allowed, sorted(allowed))
    add("schema_validation_allowed", "schema_validation" in allowed and "schema_validation" in entry_allowed, sorted(allowed))
    add("baseline_forbidden", {"baseline_update", "current_baseline_replacement"}.issubset(forbidden) and {"baseline_update", "current_baseline_replacement"}.issubset(entry_forbidden), sorted(forbidden))
    add("latest_newest_forbidden", "latest_newest_auto_discovery" in forbidden and not bool(manifest.get("latest_newest_discovery_allowed")), manifest.get("latest_newest_discovery_allowed"))
    add("auto_discovery_forbidden", not bool(manifest.get("auto_discovery_allowed")), manifest.get("auto_discovery_allowed"))
    add("row_count", int(manifest.get("n_samples", -1)) == 192, manifest.get("n_samples"))
    add("base_count", int(manifest.get("base_count", -1)) == 48, manifest.get("base_count"))
    add("paired_liftoff_complete", bool(manifest.get("paired_liftoff_complete")), manifest.get("paired_liftoff_complete"))
    add("liftoff_levels", levels == [0.006, 0.008, 0.01, 0.012], levels)
    add("npz_exists", npz_path.exists(), str(npz_path))
    if npz_path.exists():
        add("npz_sha256", pilot.sha256_file(npz_path) == manifest.get("npz_sha256"), manifest.get("npz_sha256"))
    return checks


def load_liftoff_dataset(
    dataset_id: str = DATASET_ID, registry_path: Path = pilot.REGISTRY_PATH
) -> True3DRBCLiftoffDataset:
    entry, manifest, npz_path = resolve_liftoff_dataset(dataset_id, registry_path)
    failed = [row for row in gate_liftoff_manifest(entry, manifest, npz_path, dataset_id) if not row["pass"]]
    if failed:
        raise RuntimeError("liftoff registry/manifest gate failed: " + json.dumps(failed, ensure_ascii=False))
    with np.load(npz_path, allow_pickle=True) as npz:
        required = [
            "delta_b",
            "b_defect",
            "b_no_defect",
            "axis_names",
            "axis_expressions",
            "sensor_x",
            "scan_line_y",
            "sensor_z_m",
            "jscale",
            "scan_bundle_y_offset_m",
            "sample_ids",
            "base_sample_ids",
            "source_sample_ids",
            "variant_name",
            "factor_group",
            "row_kind",
            "split",
            "curvature_template",
            "depth_bin",
            "aspect_bin",
            "size_bin",
            "rbc_params",
            "profile_pose",
            "profile_depth_grid_m",
            "profile_depth_map_xy_m",
            "projected_mask_2d",
            "liftoff_delta_m",
        ]
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise RuntimeError(f"missing liftoff NPZ fields: {missing}")
        delta_b = np.asarray(npz["delta_b"], dtype=np.float32)
        return True3DRBCLiftoffDataset(
            dataset_id=dataset_id,
            manifest=manifest,
            registry_entry=entry,
            npz_path=npz_path,
            delta_b=delta_b,
            b_defect=np.asarray(npz["b_defect"], dtype=np.float32),
            b_no_defect=np.asarray(npz["b_no_defect"], dtype=np.float32),
            x_channels=delta_b.reshape(delta_b.shape[0], 9, delta_b.shape[-1]).astype(np.float32),
            rbc_params=np.asarray(npz["rbc_params"], dtype=np.float32).reshape(delta_b.shape[0], 6),
            profile_pose=np.asarray(npz["profile_pose"], dtype=np.float32).reshape(delta_b.shape[0], 6),
            projected_mask_2d=np.asarray(npz["projected_mask_2d"], dtype=np.uint8),
            profile_depth_grid_m=np.asarray(npz["profile_depth_grid_m"], dtype=np.float32),
            profile_depth_map_xy_m=np.asarray(npz["profile_depth_map_xy_m"], dtype=np.float32),
            sample_ids=np.asarray(npz["sample_ids"]).astype(str),
            base_sample_ids=np.asarray(npz["base_sample_ids"]).astype(str),
            source_sample_ids=np.asarray(npz["source_sample_ids"]).astype(str),
            variant_name=np.asarray(npz["variant_name"]).astype(str),
            factor_group=np.asarray(npz["factor_group"]).astype(str),
            row_kind=np.asarray(npz["row_kind"]).astype(str),
            split=np.asarray(npz["split"]).astype(str),
            axis_names=[str(x) for x in np.asarray(npz["axis_names"]).tolist()],
            axis_expressions=[str(x) for x in np.asarray(npz["axis_expressions"]).tolist()],
            sensor_x=np.asarray(npz["sensor_x"], dtype=np.float32),
            scan_line_y=np.asarray(npz["scan_line_y"], dtype=np.float32),
            sensor_z_m=np.asarray(npz["sensor_z_m"], dtype=np.float32).reshape(delta_b.shape[0]),
            liftoff_delta_m=np.asarray(npz["liftoff_delta_m"], dtype=np.float32).reshape(delta_b.shape[0]),
            jscale=np.asarray(npz["jscale"], dtype=np.float32).reshape(delta_b.shape[0]),
            scan_bundle_y_offset_m=np.asarray(npz["scan_bundle_y_offset_m"], dtype=np.float32).reshape(delta_b.shape[0]),
            curvature_template=np.asarray(npz["curvature_template"]).astype(str),
            depth_bin=np.asarray(npz["depth_bin"]).astype(str),
            aspect_bin=np.asarray(npz["aspect_bin"]).astype(str),
            size_bin=np.asarray(npz["size_bin"]).astype(str),
        )


def split_indices(dataset: True3DRBCLiftoffDataset) -> dict[str, np.ndarray]:
    return {name: np.where(dataset.split == name)[0] for name in ("train", "val", "test")}


def split_base_ids(dataset: True3DRBCLiftoffDataset) -> dict[str, set[str]]:
    return {name: set(dataset.base_sample_ids[idx].astype(str).tolist()) for name, idx in split_indices(dataset).items()}


def train_normalization(dataset: True3DRBCLiftoffDataset) -> dict[str, np.ndarray]:
    train = split_indices(dataset)["train"]
    x_train = dataset.x_channels[train]
    y_train = dataset.rbc_params[train]
    x_mean = x_train.mean(axis=(0, 2), keepdims=True)
    x_std = x_train.std(axis=(0, 2), keepdims=True)
    x_std = np.where(x_std < 1.0e-12, 1.0, x_std)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1.0e-12, 1.0, y_std)
    z_mean = np.array([[dataset.sensor_z_m[train].mean()]], dtype=np.float32)
    z_std = np.array([[dataset.sensor_z_m[train].std()]], dtype=np.float32)
    z_std = np.where(z_std < 1.0e-12, 1.0, z_std)
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "sensor_z_mean": z_mean.astype(np.float32),
        "sensor_z_std": z_std.astype(np.float32),
    }


def normalize_x_raw(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - stats["x_mean"]) / stats["x_std"]).astype(np.float32)


def normalize_x(dataset: True3DRBCLiftoffDataset, stats: dict[str, np.ndarray]) -> np.ndarray:
    return normalize_x_raw(dataset.x_channels, stats)


def normalize_y(dataset: True3DRBCLiftoffDataset, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((dataset.rbc_params - stats["y_mean"]) / stats["y_std"]).astype(np.float32)


def denormalize_y(y_norm: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(y_norm, dtype=np.float32) * stats["y_std"] + stats["y_mean"]).astype(np.float32)


def normalize_sensor_z(dataset: True3DRBCLiftoffDataset, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((dataset.sensor_z_m.reshape(-1, 1) - stats["sensor_z_mean"]) / stats["sensor_z_std"]).astype(np.float32)


def paired_liftoff_complete(dataset: True3DRBCLiftoffDataset) -> bool:
    expected = {round(x, 3) for x in LIFTOFF_LEVELS}
    for base_id in sorted(set(dataset.base_sample_ids.astype(str))):
        idx = np.where(dataset.base_sample_ids == base_id)[0]
        levels = {round(float(x), 3) for x in dataset.sensor_z_m[idx]}
        if levels != expected:
            return False
    return True


def base_split_leakage(dataset: True3DRBCLiftoffDataset) -> list[dict[str, Any]]:
    seen: dict[str, set[str]] = {}
    for base_id, split in zip(dataset.base_sample_ids.astype(str), dataset.split.astype(str)):
        seen.setdefault(base_id, set()).add(split)
    return [{"base_sample_id": base_id, "splits": sorted(splits)} for base_id, splits in seen.items() if len(splits) > 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and smoke-check the explicit 20.91b liftoff augmentation pack.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_liftoff_dataset(args.dataset_id)
    print(f"dataset_id={dataset.dataset_id}")
    print(f"rows={len(dataset.sample_ids)} bases={len(set(dataset.base_sample_ids.astype(str)))}")
    print(f"split_counts={{name: len(idx) for name, idx in split_indices(dataset).items()}}")
    print(f"paired_liftoff_complete={paired_liftoff_complete(dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
