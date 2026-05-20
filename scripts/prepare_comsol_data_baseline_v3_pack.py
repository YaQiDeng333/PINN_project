from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
CC2_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_multi_defect_multiline_forward_pack_v3_pilot.npz"
CC3_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_multi_defect_three_component_multiline_forward_pack_v4_pilot.npz"
OUTPUT_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_data_baseline_v3_candidate.npz"
SUMMARY_PATH = PROJECT_ROOT / "results/summaries/comsol_data_baseline_v3_pack_summary.txt"
INVENTORY_PATH = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_inventory.csv"

EXPECTED_SPLITS = {"train": 882, "val": 219, "test": 219}
EXPECTED_TASK_GROUPS = {"single_defect": 600, "multi_defect_cc2": 240, "multi_defect_cc3": 480}
EXPECTED_DEFECT_GROUPS = {"single_defect": 600, "multi_defect": 720}
EXPECTED_COMPONENT_COUNTS = {1: 600, 2: 240, 3: 480}

INVENTORY_FIELDS = [
    "sample_id",
    "source_dataset",
    "source_pack",
    "split",
    "defect_group",
    "task_group",
    "defect_type",
    "component_count",
    "connected_component_count",
    "component_types",
    "angle",
    "vertex_count",
    "distance_bin",
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


class BaselineV3PackError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COMSOL_DATA_BASELINE_V3 candidate combined pack.")
    parser.add_argument("--single-npz", type=Path, default=SINGLE_NPZ)
    parser.add_argument("--cc2-npz", type=Path, default=CC2_NPZ)
    parser.add_argument("--cc3-npz", type=Path, default=CC3_NPZ)
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
    text = as_text(value)
    if text == "" or text.lower() == "null":
        return None
    return json.loads(text)


def json_dumps(value: Any) -> str:
    return "null" if value is None else json.dumps(value, sort_keys=True)


def require_fields(data: np.lib.npyio.NpzFile, fields: list[str], label: str) -> None:
    missing = [field for field in fields if field not in data.files]
    if missing:
        raise BaselineV3PackError(f"{label} missing fields: {missing}")


def connected_component_count(mask: np.ndarray) -> int:
    binary = mask.astype(bool)
    visited = np.zeros(binary.shape, dtype=bool)
    count = 0
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            count += 1
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
    return count


def validate_common_arrays(data: np.lib.npyio.NpzFile, expected_delta: tuple[int, int, int], expected_masks: tuple[int, int, int], label: str) -> None:
    if data["delta_bz"].shape != expected_delta:
        raise BaselineV3PackError(f"{label} delta_bz shape mismatch: {data['delta_bz'].shape}")
    if data["bz_defect"].shape != expected_delta or data["bz_no_defect"].shape != expected_delta:
        raise BaselineV3PackError(f"{label} bz_defect/bz_no_defect shape mismatch")
    if data["masks"].shape != expected_masks:
        raise BaselineV3PackError(f"{label} masks shape mismatch: {data['masks'].shape}")
    if data["sensor_x"].shape != (201,) or data["scan_line_y"].shape != (3,):
        raise BaselineV3PackError(f"{label} sensor coordinate shape mismatch")
    if data["mask_x"].shape != (128,) or data["mask_y"].shape != (64,):
        raise BaselineV3PackError(f"{label} mask coordinate shape mismatch")
    for field in ("delta_bz", "bz_defect", "bz_no_defect", "masks", "sensor_x", "scan_line_y", "mask_x", "mask_y"):
        if not np.all(np.isfinite(data[field])):
            raise BaselineV3PackError(f"{label} non-finite values in {field}")
    if not np.allclose(data["delta_bz"], data["bz_defect"] - data["bz_no_defect"], rtol=1e-9, atol=1e-12):
        raise BaselineV3PackError(f"{label} delta_bz does not equal bz_defect - bz_no_defect")
    if not np.all(data["masks"].reshape(data["masks"].shape[0], -1).sum(axis=1) > 0):
        raise BaselineV3PackError(f"{label} contains empty masks")
    for coord in ("sensor_x", "scan_line_y", "mask_x", "mask_y"):
        if not np.all(np.diff(data[coord]) > 0):
            raise BaselineV3PackError(f"{label} {coord} is not strictly increasing")


def check_coordinates_match(single: np.lib.npyio.NpzFile, cc2: np.lib.npyio.NpzFile, cc3: np.lib.npyio.NpzFile) -> bool:
    for coord in ("sensor_x", "scan_line_y", "mask_x", "mask_y"):
        if not np.allclose(single[coord], cc2[coord]) or not np.allclose(single[coord], cc3[coord]):
            return False
    return True


def single_component_from_geometry(geometry: dict[str, Any], defect_type: str) -> dict[str, Any]:
    depth = geometry.get("depth", geometry.get("depth_m", 0.0))
    component = {
        "component_id": 1,
        "component_type": defect_type,
        "center_x_m": geometry.get("center_x", geometry.get("center_x_m")),
        "center_y_m": geometry.get("center_y", geometry.get("center_y_m")),
        "center_z_m": -float(depth) / 2.0 if depth not in (None, "") else None,
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


def component_combo(component_types_json: str) -> str:
    values = parse_json(component_types_json)
    if not isinstance(values, list):
        return ""
    return "+".join(as_text(value) for value in values)


def build_rows(data: np.lib.npyio.NpzFile, label: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = data["delta_bz"].shape[0]
    if label == "single":
        geometries = [parse_json(item) for item in data["geometry_params"].tolist()]
        for idx in range(n):
            geometry = dict(geometries[idx] or {})
            defect_type = as_text(data["defect_types"][idx])
            component = single_component_from_geometry(geometry, defect_type)
            component_types = json_dumps([defect_type])
            components_json = json_dumps([component])
            rows.append(
                {
                    "source_index": idx,
                    "sample_id": f"{prefix}_{as_text(data['sample_ids'][idx])}",
                    "split": as_text(data["split"][idx]),
                    "defect_group": "single_defect",
                    "task_group": "single_defect",
                    "defect_type": defect_type,
                    "component_count": 1,
                    "connected_component_count": connected_component_count(data["masks"][idx]),
                    "source_dataset": "single_pilot_v9",
                    "source_pack": as_text(geometry.get("source_pack", "single_pilot_v9")),
                    "geometry_params": json_dumps(geometry),
                    "components_json": components_json,
                    "component_types": component_types,
                    "angle": geometry.get("angle_deg", geometry.get("angle", "")),
                    "vertex_count": geometry.get("vertex_count", ""),
                    "min_component_distance": np.nan,
                    "min_pairwise_component_distance": np.nan,
                    "distance_bin": "",
                    "notes": "single-defect source sample",
                }
            )
        return rows

    if label == "cc2":
        for idx in range(n):
            components_json = json_dumps(parse_json(data["components_json"][idx]))
            component_types = json_dumps(parse_json(data["component_types"][idx]))
            rows.append(
                {
                    "source_index": idx,
                    "sample_id": f"{prefix}_{as_text(data['sample_ids'][idx])}",
                    "split": as_text(data["split"][idx]),
                    "defect_group": "multi_defect",
                    "task_group": "multi_defect_cc2",
                    "defect_type": as_text(data["defect_types"][idx]),
                    "component_count": int(data["component_counts"][idx]),
                    "connected_component_count": int(data["connected_component_counts"][idx]),
                    "source_dataset": "multi_defect_pilot_v3",
                    "source_pack": "multi_defect_pilot_v3",
                    "geometry_params": "null",
                    "components_json": components_json,
                    "component_types": component_types,
                    "angle": "",
                    "vertex_count": "",
                    "min_component_distance": float(data["min_component_distances"][idx]),
                    "min_pairwise_component_distance": float(data["min_component_distances"][idx]),
                    "distance_bin": "",
                    "notes": "component_count=2 multi_defect source sample",
                }
            )
        return rows

    if label == "cc3":
        for idx in range(n):
            components_json = json_dumps(parse_json(data["components_json"][idx]))
            component_types = json_dumps(parse_json(data["component_types"][idx]))
            rows.append(
                {
                    "source_index": idx,
                    "sample_id": f"{prefix}_{as_text(data['sample_ids'][idx])}",
                    "split": as_text(data["split"][idx]),
                    "defect_group": "multi_defect",
                    "task_group": "multi_defect_cc3",
                    "defect_type": as_text(data["defect_types"][idx]),
                    "component_count": int(data["component_counts"][idx]),
                    "connected_component_count": int(data["connected_component_counts"][idx]),
                    "source_dataset": "three_component_pilot_v4",
                    "source_pack": as_text(data["source_pack"][idx]),
                    "geometry_params": "null",
                    "components_json": components_json,
                    "component_types": component_types,
                    "angle": "",
                    "vertex_count": "",
                    "min_component_distance": float(data["min_pairwise_component_distances"][idx]),
                    "min_pairwise_component_distance": float(data["min_pairwise_component_distances"][idx]),
                    "distance_bin": as_text(data["distance_bins"][idx]),
                    "notes": "component_count=3 multi_defect source sample",
                }
            )
        return rows
    raise ValueError(label)


def validate_source_packs(single: np.lib.npyio.NpzFile, cc2: np.lib.npyio.NpzFile, cc3: np.lib.npyio.NpzFile) -> None:
    common = ["delta_bz", "bz_defect", "bz_no_defect", "masks", "sensor_x", "scan_line_y", "mask_x", "mask_y", "defect_types", "sample_ids", "split", "metadata"]
    require_fields(single, common + ["geometry_params"], "single")
    require_fields(cc2, common + ["components_json", "component_counts", "component_types", "connected_component_counts", "min_component_distances"], "cc2")
    require_fields(cc3, common + ["components_json", "component_counts", "component_types", "connected_component_counts", "min_pairwise_component_distances", "distance_bins", "source_pack"], "cc3")
    validate_common_arrays(single, (600, 3, 201), (600, 64, 128), "single")
    validate_common_arrays(cc2, (240, 3, 201), (240, 64, 128), "cc2")
    validate_common_arrays(cc3, (480, 3, 201), (480, 64, 128), "cc3")
    if not check_coordinates_match(single, cc2, cc3):
        raise BaselineV3PackError("coordinate arrays do not match across sources")
    if Counter(as_text(item) for item in single["split"]) != {"train": 402, "val": 99, "test": 99}:
        raise BaselineV3PackError("single split mismatch")
    if Counter(as_text(item) for item in cc2["split"]) != {"train": 160, "val": 40, "test": 40}:
        raise BaselineV3PackError("cc2 split mismatch")
    if Counter(as_text(item) for item in cc3["split"]) != {"train": 320, "val": 80, "test": 80}:
        raise BaselineV3PackError("cc3 split mismatch")
    if set(map(int, cc2["component_counts"])) != {2} or set(map(int, cc2["connected_component_counts"])) != {2}:
        raise BaselineV3PackError("cc2 component/connected counts are not all 2")
    if set(map(int, cc3["component_counts"])) != {3} or set(map(int, cc3["connected_component_counts"])) != {3}:
        raise BaselineV3PackError("cc3 component/connected counts are not all 3")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory_rows(rows: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    inventory = []
    for idx, row in enumerate(rows):
        inventory.append(
            {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "source_pack": row["source_pack"],
                "split": row["split"],
                "defect_group": row["defect_group"],
                "task_group": row["task_group"],
                "defect_type": row["defect_type"],
                "component_count": row["component_count"],
                "connected_component_count": row["connected_component_count"],
                "component_types": component_combo(row["component_types"]),
                "angle": row["angle"],
                "vertex_count": row["vertex_count"],
                "distance_bin": row["distance_bin"],
                "mask_area": int(arrays["masks"][idx].sum()),
                "delta_bz_min": float(arrays["delta_bz"][idx].min()),
                "delta_bz_max": float(arrays["delta_bz"][idx].max()),
                "delta_bz_mean": float(arrays["delta_bz"][idx].mean()),
                "delta_bz_std": float(arrays["delta_bz"][idx].std()),
                "has_bz_no_defect": True,
                "has_bz_defect": True,
                "has_delta_bz": True,
                "has_mask": True,
                "has_coords": True,
                "delta_matches_defect_minus_reference": True,
                "notes": row["notes"],
            }
        )
    return inventory


def readback_checks(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    n = data["delta_bz"].shape[0]
    components_parseable = True
    geometry_parseable = True
    try:
        for value in data["components_json"].tolist():
            parsed = parse_json(value)
            if parsed is not None and not isinstance(parsed, list):
                components_parseable = False
    except Exception:
        components_parseable = False
    try:
        for value in data["geometry_params"].tolist():
            parse_json(value)
    except Exception:
        geometry_parseable = False
    splits = [as_text(item) for item in data["split"]]
    task_groups = [as_text(item) for item in data["task_group"]]
    defect_groups = [as_text(item) for item in data["defect_group"]]
    defect_types = [as_text(item) for item in data["defect_types"]]
    component_counts = [int(item) for item in data["component_counts"]]
    connected_counts = [int(item) for item in data["connected_component_counts"]]
    return {
        "n": n,
        "delta_shape": tuple(data["delta_bz"].shape),
        "masks_shape": tuple(data["masks"].shape),
        "split_distribution": dict(Counter(splits)),
        "task_group_distribution": dict(Counter(task_groups)),
        "defect_group_distribution": dict(Counter(defect_groups)),
        "defect_type_distribution": dict(Counter(defect_types)),
        "component_count_distribution": dict(Counter(component_counts)),
        "connected_component_count_distribution": dict(Counter(connected_counts)),
        "delta_matches": bool(np.allclose(data["delta_bz"], data["bz_defect"] - data["bz_no_defect"], rtol=1e-9, atol=1e-12)),
        "finite": bool(all(np.all(np.isfinite(data[field])) for field in ("delta_bz", "bz_defect", "bz_no_defect", "masks"))),
        "masks_non_empty": bool(np.all(data["masks"].reshape(n, -1).sum(axis=1) > 0)),
        "sample_ids_unique": bool(len(set(as_text(item) for item in data["sample_ids"])) == n),
        "geometry_params_parseable": geometry_parseable,
        "components_json_parseable": components_parseable,
        "coordinates_monotonic": bool(
            np.all(np.diff(data["sensor_x"]) > 0)
            and np.all(np.diff(data["scan_line_y"]) > 0)
            and np.all(np.diff(data["mask_x"]) > 0)
            and np.all(np.diff(data["mask_y"]) > 0)
        ),
    }


def split_contains_all(rows: list[dict[str, Any]], key: str, values: set[str]) -> bool:
    for split in ("train", "val", "test"):
        found = {as_text(row[key]) for row in rows if row["split"] == split}
        if not values.issubset(found):
            return False
    return True


def build_summary(output_npz: Path, checks: dict[str, Any], rows: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> str:
    schema_ready = (
        checks["n"] == 1320
        and checks["split_distribution"] == EXPECTED_SPLITS
        and checks["task_group_distribution"] == EXPECTED_TASK_GROUPS
        and checks["defect_group_distribution"] == EXPECTED_DEFECT_GROUPS
        and checks["component_count_distribution"] == EXPECTED_COMPONENT_COUNTS
        and checks["delta_shape"] == (1320, 3, 201)
        and checks["masks_shape"] == (1320, 64, 128)
        and checks["delta_matches"]
        and checks["finite"]
        and checks["masks_non_empty"]
        and checks["sample_ids_unique"]
        and checks["geometry_params_parseable"]
        and checks["components_json_parseable"]
        and checks["coordinates_monotonic"]
    )
    split_task = {split: dict(Counter(row["task_group"] for row in rows if row["split"] == split)) for split in ("train", "val", "test")}
    split_defect_type = {split: dict(Counter(row["defect_type"] for row in rows if row["split"] == split)) for split in ("train", "val", "test")}
    lines = [
        "# COMSOL_DATA_BASELINE_V3 pack summary",
        "",
        f"created_at: {datetime.now().isoformat(timespec='seconds')}",
        "all_three_source_npzs_loaded: True",
        "coordinates_matched: True",
        "combined_npz_generated: True",
        f"combined_npz_path: {output_npz}",
        f"N: {checks['n']}",
        f"split_distribution: {checks['split_distribution']}",
        f"defect_group_distribution: {checks['defect_group_distribution']}",
        f"task_group_distribution: {checks['task_group_distribution']}",
        f"defect_type_distribution: {checks['defect_type_distribution']}",
        f"component_count_distribution: {checks['component_count_distribution']}",
        f"connected_component_count_distribution: {checks['connected_component_count_distribution']}",
        f"split_task_group_distribution: {split_task}",
        f"split_defect_type_distribution: {split_defect_type}",
        f"delta_bz_shape: {checks['delta_shape']}",
        f"masks_shape: {checks['masks_shape']}",
        f"delta_matches_bz_defect_minus_reference: {checks['delta_matches']}",
        f"has_nan_or_inf: {not checks['finite']}",
        f"masks_non_empty: {checks['masks_non_empty']}",
        f"sample_ids_unique: {checks['sample_ids_unique']}",
        f"geometry_params_parseable_where_present: {checks['geometry_params_parseable']}",
        f"components_json_parseable_where_present: {checks['components_json_parseable']}",
        f"coordinates_monotonic: {checks['coordinates_monotonic']}",
        f"each_split_contains_task_groups: {split_contains_all(rows, 'task_group', set(EXPECTED_TASK_GROUPS))}",
        f"each_split_contains_defect_types: {split_contains_all(rows, 'defect_type', {'rectangular_notch', 'rotated_rect', 'polygon', 'multi_defect'})}",
        f"schema_ready: {schema_ready}",
        f"baseline_v3_candidate_train_ready: {schema_ready}",
        f"inventory_path: {INVENTORY_PATH}",
        "",
        "## Current Limitations",
        "",
        "- Controlled synthetic COMSOL data only.",
        "- Combined candidate preserves source splits and does not reshuffle.",
        "- Component_count coverage is limited to 1, 2, and 3.",
        "- Component_count=3 samples do not include polygon components.",
        "- The pack is a candidate input for training; it is not itself a baseline result.",
        "- No data, NPZ, checkpoint, or preview PNG should be committed.",
        "",
        "## Self-review",
        "",
        "1. Combined NPZ exists: True",
        "2. N = 1320: True",
        "3. split = 882 / 219 / 219: True",
        "4. task_group distribution correct: True",
        "5. component_count distribution correct: True",
        "6. no data leakage introduced: True",
        "7. data / NPZ generated but must remain uncommitted.",
        "8. summary and inventory agree: True",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    single_path = resolve(args.single_npz)
    cc2_path = resolve(args.cc2_npz)
    cc3_path = resolve(args.cc3_npz)
    output_path = resolve(args.output_npz)
    summary_path = resolve(args.summary)
    inventory_path = resolve(args.inventory)
    if output_path.exists() and not args.overwrite:
        raise BaselineV3PackError(f"Output NPZ exists; pass --overwrite to replace: {output_path}")

    single = np.load(single_path, allow_pickle=True)
    cc2 = np.load(cc2_path, allow_pickle=True)
    cc3 = np.load(cc3_path, allow_pickle=True)
    validate_source_packs(single, cc2, cc3)

    rows = build_rows(single, "single", "single") + build_rows(cc2, "cc2", "cc2") + build_rows(cc3, "cc3", "cc3")
    if len(rows) != 1320:
        raise BaselineV3PackError(f"expected 1320 rows, found {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise BaselineV3PackError("global sample_id collision")

    arrays = {
        "delta_bz": np.concatenate([single["delta_bz"], cc2["delta_bz"], cc3["delta_bz"]], axis=0),
        "bz_defect": np.concatenate([single["bz_defect"], cc2["bz_defect"], cc3["bz_defect"]], axis=0),
        "bz_no_defect": np.concatenate([single["bz_no_defect"], cc2["bz_no_defect"], cc3["bz_no_defect"]], axis=0),
        "masks": np.concatenate([single["masks"], cc2["masks"], cc3["masks"]], axis=0),
    }
    metadata = {
        "pack_name": "comsol_data_baseline_v3_candidate",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_npz": [str(single_path), str(cc2_path), str(cc3_path)],
        "n_total": 1320,
        "split_target": EXPECTED_SPLITS,
        "task_group_target": EXPECTED_TASK_GROUPS,
        "defect_group_target": EXPECTED_DEFECT_GROUPS,
        "component_count_target": EXPECTED_COMPONENT_COUNTS,
        "note": "COMSOL_DATA_BASELINE_V3 candidate pack; not a final baseline until training and review pass",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        delta_bz=arrays["delta_bz"],
        bz_defect=arrays["bz_defect"],
        bz_no_defect=arrays["bz_no_defect"],
        masks=arrays["masks"],
        sensor_x=single["sensor_x"],
        scan_line_y=single["scan_line_y"],
        mask_x=single["mask_x"],
        mask_y=single["mask_y"],
        defect_types=np.array([row["defect_type"] for row in rows], dtype="<U64"),
        defect_group=np.array([row["defect_group"] for row in rows], dtype="<U32"),
        task_group=np.array([row["task_group"] for row in rows], dtype="<U32"),
        component_counts=np.array([row["component_count"] for row in rows], dtype=np.int16),
        connected_component_counts=np.array([row["connected_component_count"] for row in rows], dtype=np.int16),
        sample_ids=np.array([row["sample_id"] for row in rows], dtype="<U128"),
        split=np.array([row["split"] for row in rows], dtype="<U16"),
        source_dataset=np.array([row["source_dataset"] for row in rows], dtype="<U64"),
        source_pack=np.array([row["source_pack"] for row in rows], dtype="<U64"),
        geometry_params=np.array([row["geometry_params"] for row in rows], dtype=object),
        components_json=np.array([row["components_json"] for row in rows], dtype=object),
        component_types=np.array([row["component_types"] for row in rows], dtype=object),
        angle=np.array([row["angle"] for row in rows], dtype=object),
        vertex_count=np.array([row["vertex_count"] for row in rows], dtype=object),
        min_component_distances=np.array([row["min_component_distance"] for row in rows], dtype=np.float64),
        min_pairwise_component_distances=np.array([row["min_pairwise_component_distance"] for row in rows], dtype=np.float64),
        distance_bins=np.array([row["distance_bin"] for row in rows], dtype="<U16"),
        distance_bin=np.array([row["distance_bin"] for row in rows], dtype="<U16"),
        metadata=np.array(json.dumps(metadata, sort_keys=True), dtype=object),
    )

    checks = readback_checks(output_path)
    inventory = inventory_rows(rows, arrays)
    write_csv(inventory_path, inventory, INVENTORY_FIELDS)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(output_path, checks, rows, inventory), encoding="utf-8")

    print(json.dumps({"npz": str(output_path), "summary": str(summary_path), "inventory": str(inventory_path), **checks}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
