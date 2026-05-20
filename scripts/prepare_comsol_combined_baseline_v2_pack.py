from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
MULTI_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_multi_defect_multiline_forward_pack_v3_pilot.npz"
OUTPUT_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_combined_single_multi_defect_baseline_v2_candidate.npz"
SUMMARY_PATH = PROJECT_ROOT / "results/summaries/comsol_combined_baseline_v2_pack_summary.txt"
INVENTORY_PATH = PROJECT_ROOT / "results/metrics/comsol_combined_baseline_v2_inventory.csv"

EXPECTED_SINGLE_SHAPES = {"delta_bz": (600, 3, 201), "masks": (600, 64, 128)}
EXPECTED_MULTI_SHAPES = {"delta_bz": (240, 3, 201), "masks": (240, 64, 128)}
EXPECTED_SPLITS = {"train": 562, "val": 139, "test": 139}
EXPECTED_GROUPS = {"single_defect": 600, "multi_defect": 240}

INVENTORY_FIELDS = [
    "sample_id",
    "source_dataset",
    "split",
    "defect_group",
    "defect_type",
    "component_count",
    "connected_component_count",
    "component_types",
    "source_pack",
    "angle",
    "vertex_count",
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


class CombinedPackError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare combined COMSOL single + multi-defect baseline v2 candidate pack.")
    parser.add_argument("--single-npz", type=Path, default=SINGLE_NPZ)
    parser.add_argument("--multi-npz", type=Path, default=MULTI_NPZ)
    parser.add_argument("--output-npz", type=Path, default=OUTPUT_NPZ)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return str(value)


def parse_json(value: Any) -> Any:
    return json.loads(as_text(value))


def connected_component_count(mask: np.ndarray) -> int:
    binary = mask.astype(bool)
    labels = np.zeros(binary.shape, dtype=bool)
    height, width = binary.shape
    count = 0
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or labels[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            labels[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = True
                        stack.append((ny, nx))
    return count


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_fields(data: np.lib.npyio.NpzFile, fields: list[str], label: str) -> None:
    missing = [field for field in fields if field not in data.files]
    if missing:
        raise CombinedPackError(f"{label} missing fields: {missing}")


def validate_common_arrays(data: np.lib.npyio.NpzFile, expected: dict[str, tuple[int, ...]], label: str) -> None:
    if data["delta_bz"].shape != expected["delta_bz"]:
        raise CombinedPackError(f"{label} delta_bz shape mismatch: {data['delta_bz'].shape}")
    if data["bz_defect"].shape != expected["delta_bz"] or data["bz_no_defect"].shape != expected["delta_bz"]:
        raise CombinedPackError(f"{label} bz_defect / bz_no_defect shape mismatch")
    if data["masks"].shape != expected["masks"]:
        raise CombinedPackError(f"{label} masks shape mismatch: {data['masks'].shape}")
    if data["sensor_x"].shape != (201,) or data["scan_line_y"].shape != (3,):
        raise CombinedPackError(f"{label} sensor coordinate shape mismatch")
    if data["mask_x"].shape != (128,) or data["mask_y"].shape != (64,):
        raise CombinedPackError(f"{label} mask coordinate shape mismatch")
    arrays = ["delta_bz", "bz_defect", "bz_no_defect", "masks", "sensor_x", "scan_line_y", "mask_x", "mask_y"]
    if not all(np.isfinite(data[name]).all() for name in arrays):
        raise CombinedPackError(f"{label} has non-finite arrays")
    if not np.allclose(data["delta_bz"], data["bz_defect"] - data["bz_no_defect"], rtol=1e-9, atol=1e-12):
        raise CombinedPackError(f"{label} delta_bz does not match bz_defect - bz_no_defect")
    if not np.all(data["masks"].reshape(data["masks"].shape[0], -1).sum(axis=1) > 0):
        raise CombinedPackError(f"{label} contains empty masks")
    for coord_name in ("sensor_x", "scan_line_y", "mask_x", "mask_y"):
        if not np.all(np.diff(data[coord_name]) > 0):
            raise CombinedPackError(f"{label} {coord_name} is not strictly increasing")


def single_component_from_geometry(geometry: dict[str, Any], defect_type: str) -> dict[str, Any]:
    depth = geometry.get("depth", geometry.get("depth_m", 0.0))
    component: dict[str, Any] = {
        "component_id": 1,
        "component_type": defect_type,
        "center_x_m": geometry.get("center_x", geometry.get("center_x_m")),
        "center_y_m": geometry.get("center_y", geometry.get("center_y_m")),
        "center_z_m": -float(depth) / 2.0 if depth is not None else None,
        "width_m": geometry.get("width", geometry.get("width_m")),
        "length_m": geometry.get("length", geometry.get("length_m")),
        "depth_m": depth,
        "angle_deg": geometry.get("angle_deg", geometry.get("angle", 0.0)),
        "angle_rad": geometry.get("angle_rad", 0.0),
    }
    if defect_type == "polygon":
        component["polygon_vertices"] = geometry.get("polygon_vertices")
        component["vertex_count"] = geometry.get("vertex_count")
        component["polygon_area_m2"] = geometry.get("polygon_area")
    return component


def build_single_records(data: np.lib.npyio.NpzFile) -> tuple[list[str], list[str], list[str], list[str], np.ndarray, list[str], list[str], list[dict[str, Any]]]:
    sample_ids = [f"single_{as_text(item)}" for item in data["sample_ids"].tolist()]
    splits = [as_text(item) for item in data["split"].tolist()]
    defect_types = [as_text(item) for item in data["defect_types"].tolist()]
    geometries = [parse_json(item) for item in data["geometry_params"].tolist()]
    connected_counts = np.array([connected_component_count(mask) for mask in data["masks"]], dtype=np.int64)
    components_json: list[str] = []
    component_types: list[str] = []
    unified_geometry: list[str] = []
    for geometry, defect_type in zip(geometries, defect_types):
        geometry = dict(geometry)
        geometry["defect_group"] = "single_defect"
        geometry["source_dataset"] = "single_pilot_v9"
        geometry["component_count"] = 1
        component = single_component_from_geometry(geometry, defect_type)
        components_json.append(json.dumps([component], sort_keys=True))
        component_types.append(json.dumps([defect_type], sort_keys=True))
        unified_geometry.append(json.dumps(geometry, sort_keys=True))
    return (
        sample_ids,
        splits,
        defect_types,
        ["single_defect"] * len(sample_ids),
        connected_counts,
        components_json,
        component_types,
        [{"json": value, "raw": geometry} for value, geometry in zip(unified_geometry, geometries)],
    )


def build_multi_records(data: np.lib.npyio.NpzFile) -> tuple[list[str], list[str], list[str], list[str], np.ndarray, list[str], list[str], list[dict[str, Any]]]:
    sample_ids = [f"multi_{as_text(item)}" for item in data["sample_ids"].tolist()]
    splits = [as_text(item) for item in data["split"].tolist()]
    defect_types = [as_text(item) for item in data["defect_types"].tolist()]
    components_json = [json.dumps(parse_json(item), sort_keys=True) for item in data["components_json"].tolist()]
    component_types = [json.dumps(parse_json(item), sort_keys=True) for item in data["component_types"].tolist()]
    connected_counts = data["connected_component_counts"].astype(np.int64)
    unified_geometry: list[dict[str, Any]] = []
    for sample_id, component_json, component_type_json in zip(sample_ids, components_json, component_types):
        geometry = {
            "sample_id": sample_id,
            "defect_group": "multi_defect",
            "defect_type": "multi_defect",
            "source_dataset": "multi_defect_pilot_v3",
            "component_count": 2,
            "components_json": json.loads(component_json),
            "component_types": json.loads(component_type_json),
        }
        unified_geometry.append({"json": json.dumps(geometry, sort_keys=True), "raw": geometry})
    return sample_ids, splits, defect_types, ["multi_defect"] * len(sample_ids), connected_counts, components_json, component_types, unified_geometry


def maybe_angle(geometry: dict[str, Any]) -> Any:
    return geometry.get("angle_deg", geometry.get("angle", ""))


def maybe_vertex_count(geometry: dict[str, Any]) -> Any:
    return geometry.get("vertex_count", "")


def source_pack(geometry: dict[str, Any], source_dataset: str) -> str:
    return as_text(geometry.get("source_pack", source_dataset))


def inventory_rows(
    data: np.lib.npyio.NpzFile,
    sample_ids: list[str],
    splits: list[str],
    defect_types: list[str],
    defect_groups: list[str],
    component_counts: np.ndarray,
    connected_counts: np.ndarray,
    component_types: list[str],
    source_dataset_name: str,
    geometry_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        geometry = geometry_records[index]["raw"]
        rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": source_dataset_name,
                "split": splits[index],
                "defect_group": defect_groups[index],
                "defect_type": defect_types[index],
                "component_count": int(component_counts[index]),
                "connected_component_count": int(connected_counts[index]),
                "component_types": component_types[index],
                "source_pack": source_pack(geometry, source_dataset_name),
                "angle": maybe_angle(geometry),
                "vertex_count": maybe_vertex_count(geometry),
                "mask_area": int(data["masks"][index].sum()),
                "delta_bz_min": float(data["delta_bz"][index].min()),
                "delta_bz_max": float(data["delta_bz"][index].max()),
                "delta_bz_mean": float(data["delta_bz"][index].mean()),
                "delta_bz_std": float(data["delta_bz"][index].std()),
                "has_bz_no_defect": True,
                "has_bz_defect": True,
                "has_delta_bz": True,
                "has_mask": True,
                "has_coords": True,
                "delta_matches_defect_minus_reference": bool(
                    np.allclose(data["delta_bz"][index], data["bz_defect"][index] - data["bz_no_defect"][index], rtol=1e-9, atol=1e-12)
                ),
                "notes": "combined baseline v2 candidate inventory row",
            }
        )
    return rows


def build_summary(
    output_npz: Path,
    single_loaded: bool,
    multi_loaded: bool,
    coords_match: bool,
    combined_generated: bool,
    inventory: list[dict[str, Any]],
    npz_readback: dict[str, Any],
) -> str:
    split_counts = Counter(row["split"] for row in inventory)
    group_counts = Counter(row["defect_group"] for row in inventory)
    defect_counts = Counter(row["defect_type"] for row in inventory)
    component_counts = Counter(int(row["component_count"]) for row in inventory)
    connected_counts = Counter(int(row["connected_component_count"]) for row in inventory)
    split_groups: dict[str, dict[str, int]] = {}
    split_defects: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_rows = [row for row in inventory if row["split"] == split_name]
        split_groups[split_name] = dict(Counter(row["defect_group"] for row in split_rows))
        split_defects[split_name] = dict(Counter(row["defect_type"] for row in split_rows))
    schema_ready = bool(npz_readback["schema_ready"])
    train_ready = schema_ready and dict(split_counts) == EXPECTED_SPLITS and dict(group_counts) == EXPECTED_GROUPS
    lines = [
        "# COMSOL combined single + multi-defect baseline v2 candidate pack summary",
        "",
        f"created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"single_defect_npz_loaded: {single_loaded}",
        f"multi_defect_npz_loaded: {multi_loaded}",
        f"coordinates_matched: {coords_match}",
        f"combined_npz_generated: {combined_generated}",
        f"combined_npz_path: {output_npz}",
        f"N: {len(inventory)}",
        f"split_distribution: {dict(sorted(split_counts.items()))}",
        f"defect_group_distribution: {dict(sorted(group_counts.items()))}",
        f"defect_type_distribution: {dict(sorted(defect_counts.items()))}",
        f"split_defect_group_distribution: {split_groups}",
        f"split_defect_type_distribution: {split_defects}",
        f"delta_bz_shape: {npz_readback['delta_bz_shape']}",
        f"masks_shape: {npz_readback['masks_shape']}",
        f"component_count_distribution: {dict(sorted(component_counts.items()))}",
        f"connected_component_count_distribution: {dict(sorted(connected_counts.items()))}",
        f"delta_matches_bz_defect_minus_reference: {npz_readback['delta_matches']}",
        f"has_nan_or_inf: {not npz_readback['finite']}",
        f"masks_non_empty: {npz_readback['masks_non_empty']}",
        f"sample_ids_unique: {npz_readback['sample_ids_unique']}",
        f"geometry_params_parseable: {npz_readback['geometry_params_parseable']}",
        f"components_json_parseable_where_present: {npz_readback['components_json_parseable']}",
        f"multi_defect_component_count_all_2: {npz_readback['multi_component_counts_all_2']}",
        f"multi_defect_connected_component_count_all_2: {npz_readback['multi_connected_counts_all_2']}",
        f"coordinates_monotonic: {npz_readback['coordinates_monotonic']}",
        f"schema_ready: {schema_ready}",
        f"baseline_v2_candidate_train_ready: {train_ready}",
        "",
        "## Current limitations",
        "",
        "- Combined pack preserves reviewed source splits and does not reshuffle.",
        "- Single-defect samples remain controlled synthetic COMSOL pilot_v9 samples.",
        "- Multi_defect samples remain controlled synthetic component_count=2 pilot_v3 samples.",
        "- The pack is a candidate input for training; it is not itself a baseline result.",
        "- No data, NPZ, checkpoint, or preview PNG should be committed.",
    ]
    return "\n".join(lines) + "\n"


def readback_check(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    required = [
        "delta_bz",
        "bz_defect",
        "bz_no_defect",
        "masks",
        "sensor_x",
        "scan_line_y",
        "mask_x",
        "mask_y",
        "defect_types",
        "defect_group",
        "sample_ids",
        "split",
        "geometry_params",
        "components_json",
        "component_counts",
        "component_types",
        "connected_component_counts",
        "source_dataset",
        "metadata",
    ]
    missing = [field for field in required if field not in data.files]
    geometry_parseable = True
    components_parseable = True
    try:
        for value in data["geometry_params"].tolist():
            parse_json(value)
    except Exception:
        geometry_parseable = False
    try:
        for value in data["components_json"].tolist():
            text = as_text(value)
            if text != "null":
                parse_json(text)
    except Exception:
        components_parseable = False
    groups = np.array([as_text(item) for item in data["defect_group"].tolist()])
    multi_idx = np.where(groups == "multi_defect")[0]
    return {
        "schema_ready": not missing and data["delta_bz"].shape == (840, 3, 201) and data["masks"].shape == (840, 64, 128),
        "missing": missing,
        "delta_bz_shape": tuple(data["delta_bz"].shape),
        "masks_shape": tuple(data["masks"].shape),
        "delta_matches": bool(np.allclose(data["delta_bz"], data["bz_defect"] - data["bz_no_defect"], rtol=1e-9, atol=1e-12)),
        "finite": bool(all(np.isfinite(data[name]).all() for name in ("delta_bz", "bz_defect", "bz_no_defect", "masks"))),
        "masks_non_empty": bool(np.all(data["masks"].reshape(data["masks"].shape[0], -1).sum(axis=1) > 0)),
        "sample_ids_unique": len(set(as_text(item) for item in data["sample_ids"].tolist())) == len(data["sample_ids"]),
        "geometry_params_parseable": geometry_parseable,
        "components_json_parseable": components_parseable,
        "multi_component_counts_all_2": bool(np.all(data["component_counts"][multi_idx] == 2)),
        "multi_connected_counts_all_2": bool(np.all(data["connected_component_counts"][multi_idx] == 2)),
        "coordinates_monotonic": bool(
            np.all(np.diff(data["sensor_x"]) > 0)
            and np.all(np.diff(data["scan_line_y"]) > 0)
            and np.all(np.diff(data["mask_x"]) > 0)
            and np.all(np.diff(data["mask_y"]) > 0)
        ),
    }


def main() -> int:
    args = parse_args()
    single_path = resolve(args.single_npz)
    multi_path = resolve(args.multi_npz)
    output_path = resolve(args.output_npz)
    summary_path = resolve(args.summary)
    inventory_path = resolve(args.inventory)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output NPZ: {output_path}")

    single = np.load(single_path, allow_pickle=True)
    multi = np.load(multi_path, allow_pickle=True)
    require_fields(
        single,
        ["delta_bz", "bz_defect", "bz_no_defect", "masks", "sensor_x", "scan_line_y", "mask_x", "mask_y", "defect_types", "sample_ids", "split", "metadata", "geometry_params"],
        "single",
    )
    require_fields(
        multi,
        [
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
            "split",
            "metadata",
            "components_json",
            "component_counts",
            "component_types",
            "connected_component_counts",
        ],
        "multi",
    )
    validate_common_arrays(single, EXPECTED_SINGLE_SHAPES, "single")
    validate_common_arrays(multi, EXPECTED_MULTI_SHAPES, "multi")
    coords_match = all(np.array_equal(single[name], multi[name]) for name in ("sensor_x", "scan_line_y", "mask_x", "mask_y"))
    if not coords_match:
        raise CombinedPackError("source coordinate arrays do not match")

    single_records = build_single_records(single)
    multi_records = build_multi_records(multi)
    single_sample_ids, single_splits, single_defects, single_groups, single_connected, single_components_json, single_component_types, single_geometry = single_records
    multi_sample_ids, multi_splits, multi_defects, multi_groups, multi_connected, multi_components_json, multi_component_types, multi_geometry = multi_records

    sample_ids = single_sample_ids + multi_sample_ids
    if len(set(sample_ids)) != len(sample_ids):
        raise CombinedPackError("combined sample_ids are not unique")

    component_counts = np.concatenate(
        [np.ones(len(single_sample_ids), dtype=np.int64), multi["component_counts"].astype(np.int64)]
    )
    connected_counts = np.concatenate([single_connected, multi_connected])
    splits = np.array(single_splits + multi_splits, dtype="<U16")
    defect_types = np.array(single_defects + multi_defects, dtype="<U64")
    defect_groups = np.array(single_groups + multi_groups, dtype="<U32")
    source_dataset_values = np.array(["single_pilot_v9"] * len(single_sample_ids) + ["multi_defect_pilot_v3"] * len(multi_sample_ids), dtype="<U64")
    geometry_params = np.array([record["json"] for record in single_geometry + multi_geometry], dtype=object)
    components_json = np.array(single_components_json + multi_components_json, dtype=object)
    component_types = np.array(single_component_types + multi_component_types, dtype=object)
    metadata = {
        "pack_name": "comsol_combined_single_multi_defect_baseline_v2_candidate",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_npz": [str(single_path), str(multi_path)],
        "n_total": 840,
        "n_single_defect": 600,
        "n_multi_defect": 240,
        "split_counts": EXPECTED_SPLITS,
        "signal_shape": [3, 201],
        "mask_shape": [64, 128],
        "note": "combined COMSOL baseline v2 candidate pack; not a committed dataset artifact",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        delta_bz=np.concatenate([single["delta_bz"], multi["delta_bz"]], axis=0),
        bz_defect=np.concatenate([single["bz_defect"], multi["bz_defect"]], axis=0),
        bz_no_defect=np.concatenate([single["bz_no_defect"], multi["bz_no_defect"]], axis=0),
        masks=np.concatenate([single["masks"], multi["masks"]], axis=0),
        sensor_x=single["sensor_x"],
        scan_line_y=single["scan_line_y"],
        mask_x=single["mask_x"],
        mask_y=single["mask_y"],
        defect_types=defect_types,
        defect_group=defect_groups,
        sample_ids=np.array(sample_ids, dtype="<U128"),
        split=splits,
        geometry_params=geometry_params,
        components_json=components_json,
        component_counts=component_counts,
        component_types=component_types,
        connected_component_counts=connected_counts,
        source_dataset=source_dataset_values,
        metadata=np.array(json.dumps(metadata, sort_keys=True), dtype=object),
    )

    inventory = []
    inventory.extend(
        inventory_rows(
            single,
            single_sample_ids,
            single_splits,
            single_defects,
            single_groups,
            np.ones(len(single_sample_ids), dtype=np.int64),
            single_connected,
            single_component_types,
            "single_pilot_v9",
            single_geometry,
        )
    )
    inventory.extend(
        inventory_rows(
            multi,
            multi_sample_ids,
            multi_splits,
            multi_defects,
            multi_groups,
            multi["component_counts"].astype(np.int64),
            multi_connected,
            multi_component_types,
            "multi_defect_pilot_v3",
            multi_geometry,
        )
    )
    readback = readback_check(output_path)
    write_csv(inventory_path, inventory, INVENTORY_FIELDS)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        build_summary(output_path, True, True, coords_match, output_path.exists(), inventory, readback),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "combined_npz": str(output_path),
                "delta_bz_shape": readback["delta_bz_shape"],
                "masks_shape": readback["masks_shape"],
                "split_distribution": dict(Counter(row["split"] for row in inventory)),
                "defect_group_distribution": dict(Counter(row["defect_group"] for row in inventory)),
                "schema_ready": readback["schema_ready"],
                "summary": str(summary_path),
                "inventory": str(inventory_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
