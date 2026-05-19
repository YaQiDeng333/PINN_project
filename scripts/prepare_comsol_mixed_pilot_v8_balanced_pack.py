#!/usr/bin/env python
"""Prepare the balanced COMSOL mixed pilot_v8 pack.

This script combines existing rectangular_notch / rotated_rect / polygon COMSOL
pilot packs with the real pilot_v7 and pilot_v8 top-up samples. It does not run
COMSOL and does not train a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECT_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v2.npz"
DEFAULT_ROT_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v3_rotated_rect.npz"
DEFAULT_POLY_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v5_polygon.npz"
DEFAULT_V7_TOPUP_DIR = (
    PROJECT_ROOT
    / "data/comsol_mfl/generated/comsol_single_defect_multiline_forward_pack_v1_pilot_v7_balanced_three_types"
)
DEFAULT_V8_TOPUP_DIR = (
    PROJECT_ROOT
    / "data/comsol_mfl/generated/comsol_single_defect_multiline_forward_pack_v1_pilot_v8_balanced_three_types"
)
DEFAULT_OUTPUT_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v8_balanced_three_types.npz"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_mixed_pilot_v8_balanced_pack_summary.txt"
DEFAULT_INVENTORY = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v8_balanced_inventory.csv"
V7_TOPUP_MANIFEST_NAME = "manifest_comsol_single_defect_multiline_forward_pack_v1_pilot_v7_topup.csv"
V8_TOPUP_MANIFEST_NAME = "manifest_comsol_single_defect_multiline_forward_pack_v1_pilot_v8_topup.csv"

TARGET_PER_TYPE = 120
SPLIT_TARGET_PER_TYPE = {"train": 80, "val": 20, "test": 20}
TARGET_TOTAL_SPLITS = {"train": 240, "val": 60, "test": 60}

REQUIRED_KEYS = [
    "delta_bz",
    "bz_defect",
    "bz_no_defect",
    "masks",
    "sensor_x",
    "scan_line_y",
    "mask_x",
    "mask_y",
    "defect_types",
    "sample_ids",
    "geometry_params",
    "split",
]

INVENTORY_FIELDS = [
    "sample_id",
    "source_pack",
    "source_sample_id",
    "split",
    "defect_type",
    "center_x",
    "center_y",
    "width",
    "length",
    "depth",
    "angle_deg",
    "vertex_count",
    "polygon_area",
    "n_lines",
    "signal_length",
    "signal_shape",
    "mask_shape",
    "mask_area",
    "delta_bz_min",
    "delta_bz_max",
    "delta_bz_mean",
    "delta_bz_std",
    "has_bz_no_defect",
    "has_bz_defect",
    "has_delta_bz",
    "has_mask",
    "has_coords",
    "delta_matches_defect_minus_reference",
    "notes",
]


class PackValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare balanced COMSOL mixed pilot_v8 pack.")
    parser.add_argument("--rect-npz", type=Path, default=DEFAULT_RECT_NPZ)
    parser.add_argument("--rot-npz", type=Path, default=DEFAULT_ROT_NPZ)
    parser.add_argument("--poly-npz", type=Path, default=DEFAULT_POLY_NPZ)
    parser.add_argument("--v7-topup-dir", type=Path, default=DEFAULT_V7_TOPUP_DIR)
    parser.add_argument("--v8-topup-dir", type=Path, default=DEFAULT_V8_TOPUP_DIR)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise PackValidationError(f"missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def parse_geometry(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, dict):
        return dict(value)
    return json.loads(decode(value))


def ensure_source_pack(name: str, pack: dict[str, np.ndarray], expected_n: int, expected_type: str) -> dict[str, Any]:
    missing = [key for key in REQUIRED_KEYS if key not in pack]
    if missing:
        raise PackValidationError(f"{name} missing keys: {missing}")
    delta = pack["delta_bz"].astype(float)
    bz_defect = pack["bz_defect"].astype(float)
    bz_no_defect = pack["bz_no_defect"].astype(float)
    masks = pack["masks"]
    if delta.shape != (expected_n, 3, 201):
        raise PackValidationError(f"{name} delta_bz expected {(expected_n, 3, 201)}, got {delta.shape}")
    if bz_defect.shape != delta.shape or bz_no_defect.shape != delta.shape:
        raise PackValidationError(f"{name} raw Bz arrays must match delta_bz shape")
    if masks.shape != (expected_n, 64, 128):
        raise PackValidationError(f"{name} masks expected {(expected_n, 64, 128)}, got {masks.shape}")
    for key, shape in {"sensor_x": (201,), "scan_line_y": (3,), "mask_x": (128,), "mask_y": (64,)}.items():
        if pack[key].shape != shape:
            raise PackValidationError(f"{name} {key} expected {shape}, got {pack[key].shape}")
        if not np.all(np.diff(pack[key].astype(float)) > 0):
            raise PackValidationError(f"{name} {key} must be strictly increasing")
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(bz_defect)) or not np.all(np.isfinite(bz_no_defect)):
        raise PackValidationError(f"{name} contains NaN or inf signal")
    delta_error = float(np.max(np.abs(delta - (bz_defect - bz_no_defect))))
    if delta_error > 1e-10:
        raise PackValidationError(f"{name} delta_bz mismatch: {delta_error}")
    if np.any(np.sum(np.abs(delta), axis=(1, 2)) <= 0):
        raise PackValidationError(f"{name} contains all-zero delta sample")
    if np.any(np.sum(masks > 0, axis=(1, 2)) <= 0):
        raise PackValidationError(f"{name} contains empty mask")
    types = [decode(item) for item in pack["defect_types"]]
    if set(types) != {expected_type}:
        raise PackValidationError(f"{name} expected only {expected_type}, got {Counter(types)}")
    return {
        "delta_match_max_abs_error": delta_error,
        "mask_area_min": int(np.min(np.sum(masks > 0, axis=(1, 2)))),
        "mask_area_max": int(np.max(np.sum(masks > 0, axis=(1, 2)))),
    }


def ensure_coordinates_match(reference: dict[str, np.ndarray], packs: list[tuple[str, dict[str, np.ndarray]]]) -> None:
    for name, pack in packs:
        for key in ["sensor_x", "scan_line_y", "mask_x", "mask_y"]:
            if not np.allclose(reference[key].astype(float), pack[key].astype(float)):
                raise PackValidationError(f"coordinate mismatch for {key}: reference vs {name}")


def select_evenly(n: int, k: int) -> list[int]:
    if k > n:
        raise ValueError(f"cannot select {k} from {n}")
    indices = [int(math.floor(i * n / k)) for i in range(k)]
    if len(set(indices)) != len(indices):
        raise RuntimeError("even selection produced duplicate indices")
    return indices


def normalize_geometry(raw: dict[str, Any], defect_type: str, source_pack: str, source_sample_id: str) -> dict[str, Any]:
    angle_deg = raw.get("angle_deg")
    angle_rad = raw.get("angle_rad")
    if defect_type == "rectangular_notch":
        angle_deg = 0.0
        angle_rad = 0.0
    elif defect_type == "rotated_rect":
        if angle_deg is None and angle_rad is not None:
            angle_deg = float(angle_rad) * 180.0 / math.pi
        if angle_rad is None and angle_deg is not None:
            angle_rad = float(angle_deg) * math.pi / 180.0
    else:
        angle_deg = None
        angle_rad = None
    polygon_vertices = raw.get("polygon_vertices") if defect_type == "polygon" else None
    vertex_count = int(raw.get("vertex_count", 0) or 0) if defect_type == "polygon" else 0
    polygon_area = raw.get("polygon_area_m2", raw.get("polygon_area")) if defect_type == "polygon" else None
    return {
        "defect_type": defect_type,
        "center_x": raw.get("center_x_m"),
        "center_y": raw.get("center_y_m"),
        "width": raw.get("width_m"),
        "length": raw.get("length_m"),
        "depth": raw.get("depth_m"),
        "angle": angle_deg,
        "angle_deg": angle_deg,
        "angle_rad": angle_rad,
        "polygon_vertices": polygon_vertices,
        "vertex_count": vertex_count,
        "polygon_area": polygon_area,
        "units": raw.get("units", "coordinates=m, Bz=T"),
        "source_pack": source_pack,
        "source_sample_id": source_sample_id,
    }


def build_existing_samples(
    pack: dict[str, np.ndarray],
    source_pack: str,
    defect_type: str,
    indices: list[int],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index in indices:
        source_id = decode(pack["sample_ids"][index])
        raw_geometry = parse_geometry(pack["geometry_params"][index])
        geometry = normalize_geometry(raw_geometry, defect_type, source_pack, source_id)
        samples.append(
            {
                "source_pack": source_pack,
                "source_sample_id": source_id,
                "defect_type": defect_type,
                "delta_bz": pack["delta_bz"][index].astype(np.float64),
                "bz_defect": pack["bz_defect"][index].astype(np.float64),
                "bz_no_defect": pack["bz_no_defect"][index].astype(np.float64),
                "mask": pack["masks"][index].astype(np.uint8),
                "geometry": geometry,
            }
        )
    return samples


def load_topup_samples(topup_dir: Path, manifest_name: str) -> list[dict[str, Any]]:
    manifest_path = topup_dir / manifest_name
    if not manifest_path.exists():
        raise PackValidationError(f"missing top-up manifest: {manifest_path}")
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    samples: list[dict[str, Any]] = []
    for row in rows:
        defect_type = row["defect_type"]
        if defect_type not in {"rotated_rect", "polygon"}:
            raise PackValidationError(f"unexpected top-up defect_type: {defect_type}")
        delta = np.load(row["delta_bz_path"]).astype(np.float64)
        bz_defect = np.load(row["bz_defect_path"]).astype(np.float64)
        bz_no_defect = np.load(row["bz_no_defect_path"]).astype(np.float64)
        mask = np.load(row["mask_path"]).astype(np.uint8)
        if delta.shape != (3, 201) or bz_defect.shape != delta.shape or bz_no_defect.shape != delta.shape:
            raise PackValidationError(f"{row['sample_id']} signal shape mismatch")
        if mask.shape != (64, 128):
            raise PackValidationError(f"{row['sample_id']} mask shape mismatch")
        if not np.allclose(delta, bz_defect - bz_no_defect, rtol=1e-9, atol=1e-12):
            raise PackValidationError(f"{row['sample_id']} delta_bz mismatch")
        if np.sum(mask > 0) <= 0:
            raise PackValidationError(f"{row['sample_id']} empty mask")
        raw_geometry = json.loads(row["geometry_params_json"])
        source_pack = raw_geometry.get("source_pack", f"{topup_dir.name}_{defect_type}")
        source_id = raw_geometry.get("source_sample_id", row["sample_id"])
        geometry = normalize_geometry(raw_geometry, defect_type, source_pack, source_id)
        samples.append(
            {
                "source_pack": source_pack,
                "source_sample_id": source_id,
                "defect_type": defect_type,
                "delta_bz": delta,
                "bz_defect": bz_defect,
                "bz_no_defect": bz_no_defect,
                "mask": mask,
                "geometry": geometry,
                "topup_coord_paths": {
                    "sensor_x": row["sensor_x_path"],
                    "scan_line_y": row["scan_line_y_path"],
                    "mask_x": row["mask_x_path"],
                    "mask_y": row["mask_y_path"],
                },
            }
        )
    return samples


def check_topup_coordinates(samples: list[dict[str, Any]], reference: dict[str, np.ndarray]) -> None:
    for sample in samples:
        for key, ref in [("sensor_x", reference["sensor_x"]), ("scan_line_y", reference["scan_line_y"]), ("mask_x", reference["mask_x"]), ("mask_y", reference["mask_y"])]:
            arr = np.load(sample["topup_coord_paths"][key]).astype(float)
            if arr.shape != ref.shape or not np.allclose(arr, ref.astype(float)):
                raise PackValidationError(f"top-up coordinate mismatch for {sample['source_sample_id']} {key}")
            if not np.all(np.diff(arr) > 0):
                raise PackValidationError(f"top-up {key} must be strictly increasing")


def split_samples(samples: list[dict[str, Any]], group_fn: Callable[[dict[str, Any]], Any]) -> None:
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[group_fn(sample)].append(index)
    for values in groups.values():
        values.sort(key=lambda i: (samples[i]["geometry"].get("source_sample_id", ""), i))
    split_by_index: dict[int, str] = {}
    remaining = {key: list(value) for key, value in groups.items()}
    ordered_keys = sorted(remaining, key=lambda value: str(value))
    for split_name in ["val", "test"]:
        target = SPLIT_TARGET_PER_TYPE[split_name]
        cursor = 0
        while sum(1 for value in split_by_index.values() if value == split_name) < target:
            key = ordered_keys[cursor % len(ordered_keys)]
            cursor += 1
            if not remaining[key]:
                continue
            index = remaining[key].pop(0 if split_name == "val" else -1)
            if index not in split_by_index:
                split_by_index[index] = split_name
    for index in range(len(samples)):
        if index not in split_by_index:
            split_by_index[index] = "train"
    counts = Counter(split_by_index.values())
    if dict(counts) != SPLIT_TARGET_PER_TYPE:
        raise PackValidationError(f"split target mismatch: {dict(counts)}")
    for index, split in split_by_index.items():
        samples[index]["split"] = split


def split_samples_source_aware(samples: list[dict[str, Any]], group_fn: Callable[[dict[str, Any]], Any]) -> None:
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[(sample["source_pack"], group_fn(sample))].append(index)
    for values in groups.values():
        values.sort(key=lambda i: (samples[i]["source_sample_id"], i))
    split_by_index: dict[int, str] = {}
    remaining = {key: list(value) for key, value in groups.items()}
    ordered_keys = sorted(remaining, key=lambda value: (str(value[0]), str(value[1])))
    for split_name in ["val", "test"]:
        target = SPLIT_TARGET_PER_TYPE[split_name]
        cursor = 0
        while sum(1 for value in split_by_index.values() if value == split_name) < target:
            key = ordered_keys[cursor % len(ordered_keys)]
            cursor += 1
            if not remaining[key]:
                continue
            index = remaining[key].pop(0 if split_name == "val" else -1)
            if index not in split_by_index:
                split_by_index[index] = split_name
    for index in range(len(samples)):
        if index not in split_by_index:
            split_by_index[index] = "train"
    counts = Counter(split_by_index.values())
    if dict(counts) != SPLIT_TARGET_PER_TYPE:
        raise PackValidationError(f"split target mismatch: {dict(counts)}")
    for split_name in ["train", "val", "test"]:
        split_sources = {samples[index]["source_pack"] for index, value in split_by_index.items() if value == split_name}
        if len(split_sources) < 3:
            raise PackValidationError(f"{split_name} source coverage too narrow: {sorted(split_sources)}")
    for index, split in split_by_index.items():
        samples[index]["split"] = split


def assign_ids(samples_by_type: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    prefix_by_source = {
        "pilot_v2_rectangular_notch": "rect",
        "pilot_v3_rotated_rect": "rot_p3",
        "pilot_v7_topup_rotated_rect": "rot_v7",
        "pilot_v8_topup_rotated_rect": "rot_v8",
        "pilot_v5_polygon": "poly_p5",
        "pilot_v7_topup_polygon": "poly_v7",
        "pilot_v8_topup_polygon": "poly_v8",
    }
    counters: Counter[str] = Counter()
    merged: list[dict[str, Any]] = []
    for defect_type in ["rectangular_notch", "rotated_rect", "polygon"]:
        for seq, sample in enumerate(samples_by_type[defect_type], start=1):
            prefix = prefix_by_source.get(sample["source_pack"], defect_type)
            counters[prefix] += 1
            sample["sample_id"] = f"{prefix}_{counters[prefix]:04d}"
            sample["geometry"]["source_pack"] = sample["source_pack"]
            sample["geometry"]["source_sample_id"] = sample["source_sample_id"]
            merged.append(sample)
    if len({sample["sample_id"] for sample in merged}) != len(merged):
        raise PackValidationError("sample_id collision after prefixing")
    return merged


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INVENTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory_row(sample: dict[str, Any]) -> dict[str, Any]:
    delta = sample["delta_bz"]
    mask = sample["mask"]
    geometry = sample["geometry"]
    return {
        "sample_id": sample["sample_id"],
        "source_pack": sample["source_pack"],
        "source_sample_id": sample["source_sample_id"],
        "split": sample["split"],
        "defect_type": sample["defect_type"],
        "center_x": geometry.get("center_x"),
        "center_y": geometry.get("center_y"),
        "width": geometry.get("width"),
        "length": geometry.get("length"),
        "depth": geometry.get("depth"),
        "angle_deg": geometry.get("angle_deg"),
        "vertex_count": geometry.get("vertex_count"),
        "polygon_area": geometry.get("polygon_area"),
        "n_lines": delta.shape[0],
        "signal_length": delta.shape[1],
        "signal_shape": str(tuple(delta.shape)),
        "mask_shape": str(tuple(mask.shape)),
        "mask_area": int(np.sum(mask > 0)),
        "delta_bz_min": float(np.min(delta)),
        "delta_bz_max": float(np.max(delta)),
        "delta_bz_mean": float(np.mean(delta)),
        "delta_bz_std": float(np.std(delta)),
        "has_bz_no_defect": True,
        "has_bz_defect": True,
        "has_delta_bz": True,
        "has_mask": True,
        "has_coords": True,
        "delta_matches_defect_minus_reference": bool(
            np.allclose(sample["delta_bz"], sample["bz_defect"] - sample["bz_no_defect"], rtol=1e-9, atol=1e-12)
        ),
        "notes": "balanced pilot_v8 single-defect mixed sample",
    }


def validate_merged(samples: list[dict[str, Any]]) -> dict[str, Any]:
    defect_counts = Counter(sample["defect_type"] for sample in samples)
    split_counts = Counter(sample["split"] for sample in samples)
    split_type_counts: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        split_type_counts[split] = dict(Counter(sample["defect_type"] for sample in samples if sample["split"] == split))
    if dict(defect_counts) != {"rectangular_notch": 120, "rotated_rect": 120, "polygon": 120}:
        raise PackValidationError(f"defect distribution mismatch: {dict(defect_counts)}")
    if dict(split_counts) != TARGET_TOTAL_SPLITS:
        raise PackValidationError(f"split distribution mismatch: {dict(split_counts)}")
    for split, counts in split_type_counts.items():
        if counts != {"rectangular_notch": SPLIT_TARGET_PER_TYPE[split], "rotated_rect": SPLIT_TARGET_PER_TYPE[split], "polygon": SPLIT_TARGET_PER_TYPE[split]}:
            raise PackValidationError(f"{split} type split mismatch: {counts}")
    delta_bz = np.stack([sample["delta_bz"] for sample in samples], axis=0)
    bz_defect = np.stack([sample["bz_defect"] for sample in samples], axis=0)
    bz_no_defect = np.stack([sample["bz_no_defect"] for sample in samples], axis=0)
    masks = np.stack([sample["mask"] for sample in samples], axis=0)
    if delta_bz.shape != (360, 3, 201) or masks.shape != (360, 64, 128):
        raise PackValidationError(f"merged shape mismatch: {delta_bz.shape}, {masks.shape}")
    if not np.all(np.isfinite(delta_bz)):
        raise PackValidationError("merged delta_bz has NaN or inf")
    delta_error = float(np.max(np.abs(delta_bz - (bz_defect - bz_no_defect))))
    if delta_error > 1e-10:
        raise PackValidationError(f"merged delta mismatch: {delta_error}")
    if np.any(np.sum(masks > 0, axis=(1, 2)) <= 0):
        raise PackValidationError("merged pack contains empty mask")
    angle_counts = Counter(float(sample["geometry"]["angle_deg"]) for sample in samples if sample["defect_type"] == "rotated_rect")
    vertex_counts = Counter(int(sample["geometry"]["vertex_count"]) for sample in samples if sample["defect_type"] == "polygon")
    split_angle_counts: dict[str, dict[float, int]] = {}
    split_vertex_counts: dict[str, dict[int, int]] = {}
    split_source_counts: dict[str, dict[str, int]] = {}
    split_type_source_counts: dict[str, dict[str, dict[str, int]]] = {}
    for split in ["train", "val", "test"]:
        split_angle_counts[split] = dict(Counter(float(sample["geometry"]["angle_deg"]) for sample in samples if sample["split"] == split and sample["defect_type"] == "rotated_rect"))
        split_vertex_counts[split] = dict(Counter(int(sample["geometry"]["vertex_count"]) for sample in samples if sample["split"] == split and sample["defect_type"] == "polygon"))
        split_source_counts[split] = dict(Counter(sample["source_pack"] for sample in samples if sample["split"] == split))
        split_type_source_counts[split] = {}
        for defect_type in ["rectangular_notch", "rotated_rect", "polygon"]:
            split_type_source_counts[split][defect_type] = dict(
                Counter(
                    sample["source_pack"]
                    for sample in samples
                    if sample["split"] == split and sample["defect_type"] == defect_type
                )
            )
        for defect_type in ["rotated_rect", "polygon"]:
            if len(split_type_source_counts[split][defect_type]) < 3:
                raise PackValidationError(
                    f"{split} {defect_type} source_pack distribution is single/narrow: "
                    f"{split_type_source_counts[split][defect_type]}"
                )
    return {
        "defect_counts": dict(defect_counts),
        "split_counts": dict(split_counts),
        "split_type_counts": split_type_counts,
        "angle_counts": dict(sorted(angle_counts.items())),
        "vertex_counts": dict(sorted(vertex_counts.items())),
        "split_angle_counts": split_angle_counts,
        "split_vertex_counts": split_vertex_counts,
        "source_counts": dict(Counter(sample["source_pack"] for sample in samples)),
        "split_source_counts": split_source_counts,
        "split_type_source_counts": split_type_source_counts,
        "delta_error": delta_error,
        "signal_min": float(np.min(delta_bz)),
        "signal_max": float(np.max(delta_bz)),
        "signal_mean": float(np.mean(delta_bz)),
        "signal_std": float(np.std(delta_bz)),
        "mask_area_min": int(np.min(np.sum(masks > 0, axis=(1, 2)))),
        "mask_area_max": int(np.max(np.sum(masks > 0, axis=(1, 2)))),
        "mask_area_mean": float(np.mean(np.sum(masks > 0, axis=(1, 2)))),
    }


def write_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rect_npz = resolve(args.rect_npz)
    rot_npz = resolve(args.rot_npz)
    poly_npz = resolve(args.poly_npz)
    v7_topup_dir = resolve(args.v7_topup_dir)
    v8_topup_dir = resolve(args.v8_topup_dir)
    output_npz = resolve(args.output_npz)
    summary_path = resolve(args.summary)
    inventory_path = resolve(args.inventory)

    rect_pack = load_npz(rect_npz)
    rot_pack = load_npz(rot_npz)
    poly_pack = load_npz(poly_npz)
    source_quality = {
        "pilot_v2_rectangular_notch": ensure_source_pack("pilot_v2_rectangular_notch", rect_pack, 120, "rectangular_notch"),
        "pilot_v3_rotated_rect": ensure_source_pack("pilot_v3_rotated_rect", rot_pack, 48, "rotated_rect"),
        "pilot_v5_polygon": ensure_source_pack("pilot_v5_polygon", poly_pack, 60, "polygon"),
    }
    ensure_coordinates_match(
        rect_pack,
        [("pilot_v3_rotated_rect", rot_pack), ("pilot_v5_polygon", poly_pack)],
    )
    v7_topup_samples = load_topup_samples(v7_topup_dir, V7_TOPUP_MANIFEST_NAME)
    v8_topup_samples = load_topup_samples(v8_topup_dir, V8_TOPUP_MANIFEST_NAME)
    check_topup_coordinates(v7_topup_samples, rect_pack)
    check_topup_coordinates(v8_topup_samples, rect_pack)
    v7_topup_counts = Counter(sample["defect_type"] for sample in v7_topup_samples)
    v8_topup_counts = Counter(sample["defect_type"] for sample in v8_topup_samples)
    if dict(v7_topup_counts) != {"rotated_rect": 32, "polygon": 20}:
        raise PackValidationError(f"pilot_v7 top-up distribution mismatch: {dict(v7_topup_counts)}")
    if dict(v8_topup_counts) != {"rotated_rect": 40, "polygon": 40}:
        raise PackValidationError(f"pilot_v8 top-up distribution mismatch: {dict(v8_topup_counts)}")
    topup_samples = v7_topup_samples + v8_topup_samples

    rect_samples = build_existing_samples(rect_pack, "pilot_v2_rectangular_notch", "rectangular_notch", list(range(120)))
    rotated_samples = build_existing_samples(rot_pack, "pilot_v3_rotated_rect", "rotated_rect", list(range(48))) + [
        sample for sample in topup_samples if sample["defect_type"] == "rotated_rect"
    ]
    polygon_samples = build_existing_samples(poly_pack, "pilot_v5_polygon", "polygon", list(range(60))) + [
        sample for sample in topup_samples if sample["defect_type"] == "polygon"
    ]
    if len(rect_samples) != 120 or len(rotated_samples) != 120 or len(polygon_samples) != 120:
        raise PackValidationError(
            f"composition mismatch: rect={len(rect_samples)}, rotated={len(rotated_samples)}, polygon={len(polygon_samples)}"
        )

    split_samples(rect_samples, lambda item: (round(float(item["geometry"]["width"] or 0.0), 6), round(float(item["geometry"]["depth"] or 0.0), 6)))
    split_samples_source_aware(rotated_samples, lambda item: float(item["geometry"]["angle_deg"]))
    split_samples_source_aware(polygon_samples, lambda item: int(item["geometry"]["vertex_count"]))
    samples = assign_ids(
        {
            "rectangular_notch": rect_samples,
            "rotated_rect": rotated_samples,
            "polygon": polygon_samples,
        }
    )
    validation = validate_merged(samples)

    inventory_rows = [inventory_row(sample) for sample in samples]
    write_csv(inventory_path, inventory_rows)

    delta_bz = np.stack([sample["delta_bz"] for sample in samples], axis=0)
    bz_defect = np.stack([sample["bz_defect"] for sample in samples], axis=0)
    bz_no_defect = np.stack([sample["bz_no_defect"] for sample in samples], axis=0)
    masks = np.stack([sample["mask"] for sample in samples], axis=0).astype(np.uint8)
    defect_types = np.array([sample["defect_type"] for sample in samples], dtype="<U64")
    sample_ids = np.array([sample["sample_id"] for sample in samples], dtype="<U64")
    geometry_params = np.array([json.dumps(sample["geometry"], sort_keys=True) for sample in samples], dtype=object)
    split = np.array([sample["split"] for sample in samples], dtype="<U16")
    metadata = {
        "pack_name": "comsol_single_defect_multiline_forward_pack_v1_pilot_v8_balanced_three_types",
        "source_packs": [
            str(rect_npz),
            str(rot_npz),
            str(poly_npz),
            str(v7_topup_dir / V7_TOPUP_MANIFEST_NAME),
            str(v8_topup_dir / V8_TOPUP_MANIFEST_NAME),
        ],
        "n_total": int(delta_bz.shape[0]),
        "defect_type_counts": validation["defect_counts"],
        "split_counts": validation["split_counts"],
        "signal_shape": [3, 201],
        "mask_shape": [64, 128],
        "scan_line_y": rect_pack["scan_line_y"].astype(float).tolist(),
        "coordinate_convention": "coordinates=m, Bz=T; coordinates inherited from source COMSOL packs",
        "note": "balanced pilot_v8 mixed single-defect-type pack; not official full dataset and not multi_defect",
    }
    if not args.dry_run:
        if output_npz.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing NPZ: {output_npz}")
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            delta_bz=delta_bz,
            bz_defect=bz_defect,
            bz_no_defect=bz_no_defect,
            masks=masks,
            sensor_x=rect_pack["sensor_x"],
            scan_line_y=rect_pack["scan_line_y"],
            mask_x=rect_pack["mask_x"],
            mask_y=rect_pack["mask_y"],
            defect_types=defect_types,
            sample_ids=sample_ids,
            geometry_params=geometry_params,
            split=split,
            metadata=json.dumps(metadata, sort_keys=True),
        )
        merged = load_npz(output_npz)
        ensure_source_pack("pilot_v8_balanced", merged, 360, "rectangular_notch") if False else None
        if merged["delta_bz"].shape != (360, 3, 201) or merged["masks"].shape != (360, 64, 128):
            raise PackValidationError("written NPZ shape check failed")
        if float(np.max(np.abs(merged["delta_bz"] - (merged["bz_defect"] - merged["bz_no_defect"])))) > 1e-10:
            raise PackValidationError("written NPZ delta check failed")

    summary_lines = [
        "# COMSOL mixed pilot_v8 balanced pack summary",
        "",
        f"source packs readable: yes ({rect_npz}, {rot_npz}, {poly_npz})",
        f"pilot_v7 top-up readable: yes ({v7_topup_dir / V7_TOPUP_MANIFEST_NAME})",
        f"pilot_v8 top-up readable: yes ({v8_topup_dir / V8_TOPUP_MANIFEST_NAME})",
        "coordinate consistency: yes (sensor_x, scan_line_y, mask_x, mask_y all match)",
        f"balanced pilot_v8 NPZ generated: {'no, dry-run only' if args.dry_run else 'yes'}",
        f"balanced pilot_v8 NPZ path: {output_npz}",
        f"sample count: {delta_bz.shape[0]}",
        f"split distribution: {validation['split_counts']}",
        f"defect_type distribution: {validation['defect_counts']}",
        f"source_pack distribution: {validation['source_counts']}",
        f"split source_pack distribution: {validation['split_source_counts']}",
        f"split defect_type source_pack distribution: {validation['split_type_source_counts']}",
        "val/test source_pack distribution no longer single-source for rotated_rect and polygon: yes",
        f"rotated_rect angle distribution: {validation['angle_counts']}",
        f"polygon vertex_count distribution: {validation['vertex_counts']}",
        f"split defect_type distribution: {validation['split_type_counts']}",
        f"split rotated angle coverage: {validation['split_angle_counts']}",
        f"split polygon vertex_count coverage: {validation['split_vertex_counts']}",
        f"delta_bz shape: {tuple(delta_bz.shape)}",
        f"masks shape: {tuple(masks.shape)}",
        f"delta_bz equals bz_defect - bz_no_defect max_abs_error: {validation['delta_error']}",
        f"signal min/max/mean/std: {[validation['signal_min'], validation['signal_max'], validation['signal_mean'], validation['signal_std']]}",
        f"mask area min/max/mean: {[validation['mask_area_min'], validation['mask_area_max'], validation['mask_area_mean']]}",
        "geometry_params schema unified: yes (defect_type, center_x, center_y, width, length, depth, angle, angle_deg, angle_rad, polygon_vertices, vertex_count, polygon_area, units, source_pack, source_sample_id)",
        "schema-ready: yes",
        "pilot_v8 train-ready: yes, for balanced pilot smoke/training gate only",
        "current limitations: still pilot-level, single-defect only, no multi_defect, limited COMSOL geometry families",
        "next step: enter balanced pilot_v8 training gate in PINN_project",
        f"source quality: {source_quality}",
        f"inventory path: {inventory_path}",
    ]
    write_summary(summary_path, summary_lines)
    print(
        json.dumps(
            {
                "output_npz": str(output_npz),
                "delta_bz_shape": tuple(delta_bz.shape),
                "masks_shape": tuple(masks.shape),
                "defect_counts": validation["defect_counts"],
                "split_counts": validation["split_counts"],
                "source_counts": validation["source_counts"],
                "split_type_source_counts": validation["split_type_source_counts"],
                "schema_ready": True,
                "pilot_v8_train_ready": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
