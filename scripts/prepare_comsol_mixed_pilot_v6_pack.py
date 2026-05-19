#!/usr/bin/env python
"""Prepare the mixed COMSOL rectangular + rotated + polygon pilot v6 pack.

This script only merges already-generated COMSOL NPZ packs. It does not run
COMSOL, does not train a model, and does not create synthetic samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RECT_NPZ = ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v2.npz"
DEFAULT_ROT_NPZ = ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v3_rotated_rect.npz"
DEFAULT_POLY_NPZ = ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v5_polygon.npz"
DEFAULT_OUT_NPZ = ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v6_mixed_three_types.npz"
DEFAULT_SUMMARY = ROOT / "results/summaries/comsol_mixed_pilot_v6_pack_summary.txt"
DEFAULT_INVENTORY = ROOT / "results/metrics/comsol_mixed_pilot_v6_inventory.csv"

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
    "metadata",
    "split",
]

EXPECTED_COUNTS = {"rectangular_notch": 120, "rotated_rect": 48, "polygon": 60}
EXPECTED_SPLITS = {"train": 152, "val": 38, "test": 38}


class PackValidationError(RuntimeError):
    """Raised when a source pack cannot be safely merged."""


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise PackValidationError(f"Missing input NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {"raw_geometry_params": str(value)}
    return parsed if isinstance(parsed, dict) else {"raw_geometry_params": parsed}


def get_float(params: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        if key in params and params[key] is not None:
            try:
                return float(params[key])
            except (TypeError, ValueError):
                continue
    return default


def ensure_required_keys(name: str, pack: dict[str, np.ndarray]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in pack]
    if missing:
        raise PackValidationError(f"{name} missing required keys: {missing}")


def ensure_shapes(name: str, pack: dict[str, np.ndarray]) -> None:
    delta = pack["delta_bz"]
    masks = pack["masks"]
    n = delta.shape[0]
    if delta.ndim != 3 or delta.shape[1:] != (3, 201):
        raise PackValidationError(f"{name} delta_bz must be (N,3,201), got {delta.shape}")
    if pack["bz_defect"].shape != delta.shape or pack["bz_no_defect"].shape != delta.shape:
        raise PackValidationError(f"{name} raw Bz arrays must match delta_bz shape")
    if masks.shape != (n, 64, 128):
        raise PackValidationError(f"{name} masks must be (N,64,128), got {masks.shape}")
    if pack["sensor_x"].shape != (201,) or pack["scan_line_y"].shape != (3,):
        raise PackValidationError(f"{name} sensor_x / scan_line_y shape mismatch")
    if pack["mask_x"].shape != (128,) or pack["mask_y"].shape != (64,):
        raise PackValidationError(f"{name} mask_x / mask_y shape mismatch")
    for key in ["defect_types", "sample_ids", "geometry_params", "split"]:
        if pack[key].shape[0] != n:
            raise PackValidationError(f"{name} {key} length does not match N={n}")


def ensure_numeric_quality(name: str, pack: dict[str, np.ndarray]) -> dict[str, Any]:
    delta = pack["delta_bz"].astype(float)
    bz_defect = pack["bz_defect"].astype(float)
    bz_no_defect = pack["bz_no_defect"].astype(float)
    masks = pack["masks"]
    for key, arr in {
        "delta_bz": delta,
        "bz_defect": bz_defect,
        "bz_no_defect": bz_no_defect,
        "masks": masks,
        "sensor_x": pack["sensor_x"],
        "scan_line_y": pack["scan_line_y"],
        "mask_x": pack["mask_x"],
        "mask_y": pack["mask_y"],
    }.items():
        if not np.all(np.isfinite(arr)):
            raise PackValidationError(f"{name} {key} contains NaN or inf")
    delta_error = float(np.max(np.abs(delta - (bz_defect - bz_no_defect))))
    if delta_error > 1e-9:
        raise PackValidationError(f"{name} delta_bz mismatch; max abs error={delta_error:.3e}")
    if np.any(np.sum(np.abs(delta), axis=(1, 2)) <= 0):
        raise PackValidationError(f"{name} contains all-zero delta_bz sample")
    if np.any(np.sum(masks > 0, axis=(1, 2)) <= 0):
        raise PackValidationError(f"{name} contains empty mask sample")
    for key in ["sensor_x", "mask_x", "mask_y"]:
        if not np.all(np.diff(pack[key].astype(float)) > 0):
            raise PackValidationError(f"{name} {key} must be strictly increasing")
    return {
        "delta_match_max_abs_error": delta_error,
        "delta_min": float(np.min(delta)),
        "delta_max": float(np.max(delta)),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "mask_area_min": int(np.min(np.sum(masks > 0, axis=(1, 2)))),
        "mask_area_max": int(np.max(np.sum(masks > 0, axis=(1, 2)))),
    }


def ensure_coordinate_consistency(packs: dict[str, dict[str, np.ndarray]]) -> None:
    first_name = next(iter(packs))
    first = packs[first_name]
    for name, pack in packs.items():
        for key in ["sensor_x", "scan_line_y", "mask_x", "mask_y"]:
            if first[key].shape != pack[key].shape or not np.allclose(first[key].astype(float), pack[key].astype(float)):
                raise PackValidationError(f"Coordinate mismatch for {key}: {first_name} vs {name}")


def prefixed_sample_ids(prefix: str, original_ids: np.ndarray) -> np.ndarray:
    return np.array([f"{prefix}_{idx + 1:04d}" for idx in range(len(original_ids))], dtype="<U64")


def normalize_geometry_params(
    raw: Any,
    defect_type: str,
    new_sample_id: str,
    source_sample_id: str,
    source_pack: str,
) -> str:
    params = parse_jsonish(raw)
    angle_deg = get_float(params, "angle_deg", "angle", default=0.0)
    angle_rad = get_float(params, "angle_rad", default=None)
    if defect_type == "rectangular_notch":
        angle_deg = 0.0
        angle_rad = 0.0
    elif angle_rad is None and angle_deg is not None:
        angle_rad = float(np.deg2rad(angle_deg))
    polygon_vertices = params.get("polygon_vertices") if defect_type == "polygon" else None
    vertex_count = int(params.get("vertex_count", 0)) if defect_type == "polygon" else 0
    polygon_area = get_float(params, "polygon_area", "polygon_area_m2") if defect_type == "polygon" else None
    normalized = {
        "sample_id": new_sample_id,
        "source_sample_id": source_sample_id,
        "source_pack": source_pack,
        "defect_type": defect_type,
        "center_x": get_float(params, "center_x", "center_x_m"),
        "center_y": get_float(params, "center_y", "center_y_m"),
        "width": get_float(params, "width", "width_m"),
        "length": get_float(params, "length", "length_m", "height", "height_m"),
        "depth": get_float(params, "depth", "depth_m"),
        "angle": angle_deg if defect_type != "polygon" else 0.0,
        "angle_deg": angle_deg if defect_type != "polygon" else 0.0,
        "angle_rad": angle_rad if defect_type != "polygon" else 0.0,
        "polygon_vertices": polygon_vertices,
        "vertex_count": vertex_count,
        "polygon_area": polygon_area,
        "units": "m",
        "source_geometry_params": params,
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def build_inventory_rows(
    pack: dict[str, np.ndarray],
    source_pack: str,
    new_ids: np.ndarray,
    normalized_geometries: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delta = pack["delta_bz"].astype(float)
    masks = pack["masks"]
    for idx, sample_id in enumerate(new_ids.astype(str).tolist()):
        geom = parse_jsonish(normalized_geometries[idx])
        sample_delta = delta[idx]
        mask_area = int(np.sum(masks[idx] > 0))
        rows.append(
            {
                "sample_id": sample_id,
                "source_pack": source_pack,
                "split": str(pack["split"][idx]),
                "defect_type": str(pack["defect_types"][idx]),
                "center_x": geom.get("center_x"),
                "center_y": geom.get("center_y"),
                "width": geom.get("width"),
                "length": geom.get("length"),
                "depth": geom.get("depth"),
                "angle_deg": geom.get("angle_deg"),
                "vertex_count": geom.get("vertex_count"),
                "polygon_area": geom.get("polygon_area"),
                "n_lines": int(delta.shape[1]),
                "signal_length": int(delta.shape[2]),
                "signal_shape": f"{tuple(sample_delta.shape)}",
                "mask_shape": f"{tuple(masks[idx].shape)}",
                "mask_area": mask_area,
                "delta_bz_min": float(np.min(sample_delta)),
                "delta_bz_max": float(np.max(sample_delta)),
                "delta_bz_mean": float(np.mean(sample_delta)),
                "delta_bz_std": float(np.std(sample_delta)),
                "has_bz_no_defect": True,
                "has_bz_defect": True,
                "has_delta_bz": True,
                "has_mask": mask_area > 0,
                "has_coords": True,
                "delta_matches_defect_minus_reference": True,
                "notes": "merged pilot_v6 single-defect-type sample",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_pack",
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_type_counts(split: np.ndarray, defect_types: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for split_name in sorted(set(split.astype(str).tolist())):
        mask = split.astype(str) == split_name
        out[split_name] = dict(Counter(defect_types[mask].astype(str).tolist()))
    return dict(out)


def angle_counts(defect_types: np.ndarray, geometries: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for defect_type, geom_text in zip(defect_types.astype(str).tolist(), geometries):
        geom = parse_jsonish(geom_text)
        if defect_type not in {"rectangular_notch", "rotated_rect"}:
            continue
        angle = geom.get("angle_deg")
        label = "null" if angle is None else f"{float(angle):.1f}"
        counts[defect_type][label] += 1
    return {key: dict(value) for key, value in counts.items()}


def vertex_counts(defect_types: np.ndarray, geometries: list[str]) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for defect_type, geom_text in zip(defect_types.astype(str).tolist(), geometries):
        if defect_type == "polygon":
            counter[int(parse_jsonish(geom_text).get("vertex_count", 0))] += 1
    return dict(sorted(counter.items()))


def write_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_specs = [
        ("pilot_v2_rectangular_notch", "rect", "rectangular_notch", Path(args.rect_npz)),
        ("pilot_v3_rotated_rect", "rot", "rotated_rect", Path(args.rot_npz)),
        ("pilot_v5_polygon", "poly", "polygon", Path(args.poly_npz)),
    ]
    packs = {name: load_npz(path) for name, _, _, path in source_specs}
    qualities = {}
    for name, pack in packs.items():
        ensure_required_keys(name, pack)
        ensure_shapes(name, pack)
        qualities[name] = ensure_numeric_quality(name, pack)
    ensure_coordinate_consistency(packs)

    merged_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ["delta_bz", "bz_defect", "bz_no_defect", "masks"]}
    all_defect_types: list[np.ndarray] = []
    all_sample_ids: list[np.ndarray] = []
    all_geometry_params: list[str] = []
    all_splits: list[np.ndarray] = []
    inventory_rows: list[dict[str, Any]] = []

    for source_pack, prefix, expected_type, _ in source_specs:
        pack = packs[source_pack]
        defect_types = pack["defect_types"].astype(str)
        if set(defect_types.tolist()) != {expected_type}:
            raise PackValidationError(f"{source_pack} defect_types are not all {expected_type}: {Counter(defect_types.tolist())}")
        ids = prefixed_sample_ids(prefix, pack["sample_ids"])
        geometries = [
            normalize_geometry_params(raw, expected_type, new_id, as_text(old_id), source_pack)
            for raw, new_id, old_id in zip(pack["geometry_params"], ids.astype(str), pack["sample_ids"])
        ]
        for key in merged_arrays:
            merged_arrays[key].append(pack[key])
        all_defect_types.append(defect_types)
        all_sample_ids.append(ids)
        all_geometry_params.extend(geometries)
        all_splits.append(pack["split"].astype(str))
        inventory_rows.extend(build_inventory_rows(pack, source_pack, ids, geometries))

    delta_bz = np.concatenate(merged_arrays["delta_bz"], axis=0)
    bz_defect = np.concatenate(merged_arrays["bz_defect"], axis=0)
    bz_no_defect = np.concatenate(merged_arrays["bz_no_defect"], axis=0)
    masks = np.concatenate(merged_arrays["masks"], axis=0)
    defect_types = np.concatenate(all_defect_types, axis=0)
    sample_ids = np.concatenate(all_sample_ids, axis=0)
    geometry_params = np.array(all_geometry_params, dtype=object)
    split = np.concatenate(all_splits, axis=0)

    if len(set(sample_ids.astype(str).tolist())) != len(sample_ids):
        raise PackValidationError("Merged sample_id values are not unique")

    split_counts = dict(Counter(split.tolist()))
    defect_counts = dict(Counter(defect_types.astype(str).tolist()))
    split_defect_counts = split_type_counts(split, defect_types)
    angle_distribution = angle_counts(defect_types, list(geometry_params))
    polygon_vertex_counts = vertex_counts(defect_types, list(geometry_params))

    metadata = {
        "pack_name": "comsol_single_defect_multiline_forward_pack_v1_pilot_v6_mixed_three_types",
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_packs": [str(path.relative_to(ROOT)) for _, _, _, path in source_specs],
        "output_npz": str(Path(args.output_npz).relative_to(ROOT)),
        "n_total": int(delta_bz.shape[0]),
        "n_rectangular_notch": int(defect_counts.get("rectangular_notch", 0)),
        "n_rotated_rect": int(defect_counts.get("rotated_rect", 0)),
        "n_polygon": int(defect_counts.get("polygon", 0)),
        "split_counts": split_counts,
        "signal_shape": [int(delta_bz.shape[1]), int(delta_bz.shape[2])],
        "mask_shape": [int(masks.shape[1]), int(masks.shape[2])],
        "scan_line_y": packs["pilot_v2_rectangular_notch"]["scan_line_y"].astype(float).tolist(),
        "coordinate_convention": "sensor_x, scan_line_y, mask_x, and mask_y are inherited unchanged from the COMSOL source packs.",
        "note": "mixed single-defect-type pilot_v6; not official full dataset and not multi_defect",
    }

    out_npz = Path(args.output_npz)
    summary_path = Path(args.summary)
    inventory_path = Path(args.inventory)
    summary_lines = [
        "# COMSOL mixed pilot_v6 pack summary",
        "",
        "source packs readable: yes (pilot_v2 rectangular_notch, pilot_v3 rotated_rect, pilot_v5 polygon)",
        "coordinate consistency: yes (sensor_x, scan_line_y, mask_x, mask_y all match)",
        f"mixed NPZ generated: {'no, dry-run only' if args.dry_run else 'yes'}",
        f"mixed NPZ path: {out_npz}",
        f"sample count: {delta_bz.shape[0]}",
        f"split distribution: {split_counts}",
        f"defect_type distribution: {defect_counts}",
        f"split x defect_type distribution: {split_defect_counts}",
        f"rotated_rect angle distribution: {angle_distribution.get('rotated_rect', {})}",
        f"polygon vertex_count distribution: {polygon_vertex_counts}",
        f"delta_bz shape: {tuple(delta_bz.shape)}",
        f"masks shape: {tuple(masks.shape)}",
        f"source pack quality: {qualities}",
        "geometry_params unified: yes; canonical fields include defect_type, center_x, center_y, width, length, depth, angle, angle_deg, angle_rad, polygon_vertices, vertex_count, polygon_area, units, source_pack, source_sample_id.",
        "schema-ready: yes",
        "pilot_v6 train-ready: yes, for mixed single-defect-type pilot training gate only.",
        "",
        "Current limitations:",
        "- Still pilot scale.",
        "- Still single-defect only.",
        "- No multi_defect samples.",
        "- Sample count is not enough for formal generalization claims.",
        "",
        "Next step:",
        "- Enter PINN_project mixed pilot_v6 training gate.",
    ]

    if not args.dry_run:
        if out_npz.exists() and not args.overwrite:
            raise PackValidationError(f"Output NPZ already exists; pass --overwrite to replace: {out_npz}")
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            delta_bz=delta_bz,
            bz_defect=bz_defect,
            bz_no_defect=bz_no_defect,
            masks=masks,
            sensor_x=packs["pilot_v2_rectangular_notch"]["sensor_x"],
            scan_line_y=packs["pilot_v2_rectangular_notch"]["scan_line_y"],
            mask_x=packs["pilot_v2_rectangular_notch"]["mask_x"],
            mask_y=packs["pilot_v2_rectangular_notch"]["mask_y"],
            defect_types=defect_types,
            sample_ids=sample_ids,
            geometry_params=geometry_params,
            split=split,
            suggested_split=split,
            metadata=np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True), dtype=object),
        )
        merged = load_npz(out_npz)
        ensure_required_keys("pilot_v6_mixed", merged)
        ensure_shapes("pilot_v6_mixed", merged)
        ensure_numeric_quality("pilot_v6_mixed", merged)

    write_summary(summary_path, summary_lines)
    write_csv(inventory_path, inventory_rows)
    return {
        "dry_run": args.dry_run,
        "out_npz": str(out_npz),
        "summary": str(summary_path),
        "inventory": str(inventory_path),
        "delta_bz_shape": tuple(delta_bz.shape),
        "masks_shape": tuple(masks.shape),
        "split_counts": split_counts,
        "defect_counts": defect_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rect-npz", default=str(DEFAULT_RECT_NPZ))
    parser.add_argument("--rot-npz", default=str(DEFAULT_ROT_NPZ))
    parser.add_argument("--poly-npz", default=str(DEFAULT_POLY_NPZ))
    parser.add_argument("--output-npz", default=str(DEFAULT_OUT_NPZ))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
