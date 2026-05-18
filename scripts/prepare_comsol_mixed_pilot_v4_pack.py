#!/usr/bin/env python
"""Prepare the mixed COMSOL rectangular_notch + rotated_rect pilot v4 pack.

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
DEFAULT_OUT_NPZ = ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v4_mixed.npz"
DEFAULT_SUMMARY = ROOT / "results/summaries/comsol_mixed_pilot_v4_pack_summary.txt"
DEFAULT_INVENTORY = ROOT / "results/metrics/comsol_mixed_pilot_v4_inventory.csv"

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


class PackValidationError(RuntimeError):
    """Raised when a source pack cannot be safely merged."""


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise PackValidationError(f"Missing input NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def ensure_required_keys(name: str, pack: dict[str, np.ndarray]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in pack]
    if missing:
        raise PackValidationError(f"{name} missing required keys: {missing}")


def ensure_shapes(name: str, pack: dict[str, np.ndarray]) -> None:
    delta = pack["delta_bz"]
    bz_defect = pack["bz_defect"]
    bz_no_defect = pack["bz_no_defect"]
    masks = pack["masks"]
    sensor_x = pack["sensor_x"]
    scan_line_y = pack["scan_line_y"]
    mask_x = pack["mask_x"]
    mask_y = pack["mask_y"]
    n = delta.shape[0]

    if delta.ndim != 3:
        raise PackValidationError(f"{name} delta_bz must be (N, n_lines, L), got {delta.shape}")
    if bz_defect.shape != delta.shape or bz_no_defect.shape != delta.shape:
        raise PackValidationError(f"{name} raw Bz arrays must match delta_bz shape")
    if masks.ndim != 3:
        raise PackValidationError(f"{name} masks must be (N, H, W), got {masks.shape}")
    if masks.shape[0] != n:
        raise PackValidationError(f"{name} masks N does not match delta_bz N")
    if sensor_x.shape != (delta.shape[2],):
        raise PackValidationError(f"{name} sensor_x shape {sensor_x.shape} does not match L={delta.shape[2]}")
    if scan_line_y.shape != (delta.shape[1],):
        raise PackValidationError(f"{name} scan_line_y shape {scan_line_y.shape} does not match n_lines={delta.shape[1]}")
    if mask_x.shape != (masks.shape[2],):
        raise PackValidationError(f"{name} mask_x shape {mask_x.shape} does not match W={masks.shape[2]}")
    if mask_y.shape != (masks.shape[1],):
        raise PackValidationError(f"{name} mask_y shape {mask_y.shape} does not match H={masks.shape[1]}")
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

    delta_error = np.max(np.abs(delta - (bz_defect - bz_no_defect)))
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
        "delta_match_max_abs_error": float(delta_error),
        "delta_min": float(np.min(delta)),
        "delta_max": float(np.max(delta)),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "mask_area_min": int(np.min(np.sum(masks > 0, axis=(1, 2)))),
        "mask_area_max": int(np.max(np.sum(masks > 0, axis=(1, 2)))),
    }


def ensure_coordinate_consistency(rect: dict[str, np.ndarray], rot: dict[str, np.ndarray]) -> None:
    for key in ["sensor_x", "scan_line_y", "mask_x", "mask_y"]:
        if rect[key].shape != rot[key].shape or not np.allclose(rect[key].astype(float), rot[key].astype(float)):
            raise PackValidationError(f"Coordinate mismatch for {key}")


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"raw_geometry_params": value}
    return {"raw_geometry_params": str(value)}


def get_float(params: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        if key in params and params[key] is not None:
            try:
                return float(params[key])
            except (TypeError, ValueError):
                continue
    return default


def normalize_geometry_params(raw: Any, defect_type: str, new_sample_id: str, source_sample_id: str) -> str:
    params = parse_jsonish(raw)
    angle_rad = get_float(params, "angle_rad", default=0.0)
    angle_deg = get_float(params, "angle_deg", default=None)
    if angle_deg is None and angle_rad is not None:
        angle_deg = float(np.degrees(angle_rad))
    if defect_type == "rectangular_notch":
        angle_deg = 0.0
        angle_rad = 0.0

    normalized = {
        "sample_id": new_sample_id,
        "source_sample_id": source_sample_id,
        "defect_type": defect_type,
        "center_x": get_float(params, "center_x", "center_x_m"),
        "center_y": get_float(params, "center_y", "center_y_m"),
        "center_z": get_float(params, "center_z", "center_z_m"),
        "width": get_float(params, "width", "width_m"),
        "length": get_float(params, "length", "length_m", "height", "height_m"),
        "depth": get_float(params, "depth", "depth_m"),
        "angle": angle_deg,
        "angle_deg": angle_deg,
        "angle_rad": angle_rad,
        "units": "m",
        "source_geometry_params": params,
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def prefixed_sample_ids(prefix: str, original_ids: np.ndarray) -> np.ndarray:
    return np.array([f"{prefix}_{idx + 1:04d}" for idx in range(len(original_ids))], dtype="<U64")


def merged_metadata(rect: dict[str, np.ndarray], rot: dict[str, np.ndarray], out_npz: Path) -> str:
    split = np.concatenate([rect["split"].astype(str), rot["split"].astype(str)])
    metadata = {
        "pack_name": "comsol_single_defect_multiline_forward_pack_v1_pilot_v4_mixed",
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_packs": [
            str(DEFAULT_RECT_NPZ.relative_to(ROOT)),
            str(DEFAULT_ROT_NPZ.relative_to(ROOT)),
        ],
        "output_npz": str(out_npz.relative_to(ROOT)),
        "n_total": int(rect["delta_bz"].shape[0] + rot["delta_bz"].shape[0]),
        "n_rectangular_notch": int(rect["delta_bz"].shape[0]),
        "n_rotated_rect": int(rot["delta_bz"].shape[0]),
        "split_counts": dict(Counter(split.tolist())),
        "signal_shape": [int(rect["delta_bz"].shape[1]), int(rect["delta_bz"].shape[2])],
        "mask_shape": [int(rect["masks"].shape[1]), int(rect["masks"].shape[2])],
        "scan_line_y": rect["scan_line_y"].astype(float).tolist(),
        "coordinate_convention": "sensor_x, scan_line_y, mask_x, and mask_y are inherited unchanged from the COMSOL source packs.",
        "note": "pilot_v4 mixed pack; schema-ready pilot data, not official full dataset",
    }
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


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
                "angle": geom.get("angle"),
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
                "notes": "merged pilot_v4 sample",
            }
        )
    return rows


def write_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
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
        "angle",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        angle = geom.get("angle")
        label = "null" if angle is None else f"{float(angle):.1f}"
        counts[defect_type][label] += 1
    return {key: dict(value) for key, value in counts.items()}


def write_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rect_path = Path(args.rect_npz)
    rot_path = Path(args.rot_npz)
    out_npz = Path(args.output_npz)
    summary_path = Path(args.summary)
    inventory_path = Path(args.inventory)

    rect = load_npz(rect_path)
    rot = load_npz(rot_path)

    ensure_required_keys("pilot_v2", rect)
    ensure_required_keys("pilot_v3_rotated_rect", rot)
    ensure_shapes("pilot_v2", rect)
    ensure_shapes("pilot_v3_rotated_rect", rot)
    rect_quality = ensure_numeric_quality("pilot_v2", rect)
    rot_quality = ensure_numeric_quality("pilot_v3_rotated_rect", rot)
    ensure_coordinate_consistency(rect, rot)

    rect_ids = prefixed_sample_ids("rect", rect["sample_ids"])
    rot_ids = prefixed_sample_ids("rot", rot["sample_ids"])
    rect_geoms = [
        normalize_geometry_params(raw, "rectangular_notch", new_id, str(old_id))
        for raw, new_id, old_id in zip(rect["geometry_params"], rect_ids.astype(str), rect["sample_ids"].astype(str))
    ]
    rot_geoms = [
        normalize_geometry_params(raw, "rotated_rect", new_id, str(old_id))
        for raw, new_id, old_id in zip(rot["geometry_params"], rot_ids.astype(str), rot["sample_ids"].astype(str))
    ]

    delta_bz = np.concatenate([rect["delta_bz"], rot["delta_bz"]], axis=0)
    bz_defect = np.concatenate([rect["bz_defect"], rot["bz_defect"]], axis=0)
    bz_no_defect = np.concatenate([rect["bz_no_defect"], rot["bz_no_defect"]], axis=0)
    masks = np.concatenate([rect["masks"], rot["masks"]], axis=0)
    defect_types = np.concatenate([rect["defect_types"].astype(str), rot["defect_types"].astype(str)])
    sample_ids = np.concatenate([rect_ids, rot_ids])
    geometry_params = np.array(rect_geoms + rot_geoms, dtype=object)
    split = np.concatenate([rect["split"].astype(str), rot["split"].astype(str)])
    metadata = np.array(merged_metadata(rect, rot, out_npz), dtype=object)

    if len(set(sample_ids.astype(str).tolist())) != len(sample_ids):
        raise PackValidationError("Merged sample_id values are not unique")

    split_counts = dict(Counter(split.tolist()))
    defect_counts = dict(Counter(defect_types.astype(str).tolist()))
    split_defect_counts = split_type_counts(split, defect_types)
    angle_distribution = angle_counts(defect_types, list(geometry_params))

    inventory_rows = (
        build_inventory_rows(rect, "pilot_v2_rectangular_notch", rect_ids, rect_geoms)
        + build_inventory_rows(rot, "pilot_v3_rotated_rect", rot_ids, rot_geoms)
    )

    summary_lines = [
        "# COMSOL mixed pilot_v4 pack summary",
        "",
        f"pilot_v2 readable: yes ({rect_path})",
        f"pilot_v3_rotated_rect readable: yes ({rot_path})",
        "coordinate consistency: yes (sensor_x, scan_line_y, mask_x, mask_y all match)",
        f"mixed NPZ generated: {'no, dry-run only' if args.dry_run else 'yes'}",
        f"mixed NPZ path: {out_npz}",
        f"sample count: {len(sample_ids)}",
        f"split distribution: {split_counts}",
        f"defect_type distribution: {defect_counts}",
        f"split x defect_type distribution: {split_defect_counts}",
        f"angle distribution: {angle_distribution}",
        f"delta_bz shape: {tuple(delta_bz.shape)}",
        f"masks shape: {tuple(masks.shape)}",
        f"pilot_v2 quality: {rect_quality}",
        f"pilot_v3_rotated_rect quality: {rot_quality}",
        "geometry_params unified: yes; canonical fields include defect_type, center_x, center_y, width, length, depth, angle, angle_deg, angle_rad, units, source_sample_id.",
        "schema-ready: yes",
        "pilot_v4 train-ready: yes, for mixed pilot training gate only; not an official full dataset.",
        "",
        "Current limitations:",
        "- defect_type only includes rectangular_notch and rotated_rect.",
        "- polygon is not included.",
        "- multi_defect is not included.",
        "- sample count is still pilot scale.",
        "",
        "Next step:",
        "- Enter PINN_project mixed pilot_v4 training gate if this pack remains readable and inventory passes review.",
    ]

    if args.dry_run:
        write_summary(summary_path, summary_lines)
        write_inventory(inventory_path, inventory_rows)
        return {
            "dry_run": True,
            "out_npz": str(out_npz),
            "summary": str(summary_path),
            "inventory": str(inventory_path),
            "shape": tuple(delta_bz.shape),
        }

    if out_npz.exists() and not args.overwrite:
        raise PackValidationError(f"Output NPZ already exists; pass --overwrite to replace: {out_npz}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        delta_bz=delta_bz,
        bz_defect=bz_defect,
        bz_no_defect=bz_no_defect,
        masks=masks,
        sensor_x=rect["sensor_x"],
        scan_line_y=rect["scan_line_y"],
        mask_x=rect["mask_x"],
        mask_y=rect["mask_y"],
        defect_types=defect_types,
        sample_ids=sample_ids,
        geometry_params=geometry_params,
        split=split,
        suggested_split=split,
        metadata=metadata,
    )

    merged = load_npz(out_npz)
    ensure_required_keys("pilot_v4_mixed", merged)
    ensure_shapes("pilot_v4_mixed", merged)
    ensure_numeric_quality("pilot_v4_mixed", merged)

    write_summary(summary_path, summary_lines)
    write_inventory(inventory_path, inventory_rows)
    return {
        "dry_run": False,
        "out_npz": str(out_npz),
        "summary": str(summary_path),
        "inventory": str(inventory_path),
        "shape": tuple(delta_bz.shape),
        "mask_shape": tuple(masks.shape),
        "split_counts": split_counts,
        "defect_counts": defect_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rect-npz", default=str(DEFAULT_RECT_NPZ))
    parser.add_argument("--rot-npz", default=str(DEFAULT_ROT_NPZ))
    parser.add_argument("--output-npz", default=str(DEFAULT_OUT_NPZ))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = prepare(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
