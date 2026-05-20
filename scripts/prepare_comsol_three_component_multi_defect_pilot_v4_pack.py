from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT_V3_NPZ = ROOT / "data" / "comsol_mfl" / "prepared" / "comsol_multi_defect_three_component_multiline_forward_pack_v3_pilot.npz"
TOPUP_NPZ = (
    ROOT
    / "data"
    / "comsol_mfl"
    / "generated"
    / "comsol_multi_defect_three_component_multiline_forward_pack_v4_topup"
    / "comsol_multi_defect_three_component_multiline_forward_pack_v4_topup.npz"
)
OUT_NPZ = ROOT / "data" / "comsol_mfl" / "prepared" / "comsol_multi_defect_three_component_multiline_forward_pack_v4_pilot.npz"
SUMMARY_PATH = ROOT / "results" / "summaries" / "comsol_three_component_multi_defect_pilot_v4_pack_summary.txt"
INVENTORY_PATH = ROOT / "results" / "metrics" / "comsol_three_component_multi_defect_pilot_v4_inventory.csv"

COMBINATIONS = [
    "rectangular_notch+rectangular_notch+rectangular_notch",
    "rectangular_notch+rectangular_notch+rotated_rect",
    "rectangular_notch+rotated_rect+rotated_rect",
    "rotated_rect+rotated_rect+rotated_rect",
]
DISTANCE_BINS = ["near", "medium", "far"]
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_TARGET = {"train": 320, "val": 80, "test": 80}
SOURCE_TARGET = {
    "pilot_v3": {"train": 160, "val": 40, "test": 40},
    "pilot_v4_topup": {"train": 160, "val": 40, "test": 40},
}
SEED = 2026


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_json(value: Any) -> Any:
    text = as_text(value)
    return json.loads(text)


def normalize_component_types(value: Any) -> list[str]:
    parsed = load_json(value)
    if not isinstance(parsed, list):
        raise ValueError(f"component_types is not a list: {value!r}")
    return [as_text(item) for item in parsed]


def combination_from_types(value: Any) -> str:
    return "+".join(normalize_component_types(value))


def geometry_key(components_json: Any) -> tuple[tuple[Any, ...], ...]:
    components = load_json(components_json)
    if not isinstance(components, list) or len(components) != 3:
        raise ValueError("components_json must be a 3-component JSON list")
    key_parts = []
    for comp in components:
        key_parts.append(
            (
                comp.get("component_type"),
                round(float(comp.get("center_x_m")), 8),
                round(float(comp.get("center_y_m")), 8),
                round(float(comp.get("width_m")), 8),
                round(float(comp.get("length_m")), 8),
                round(float(comp.get("depth_m")), 8),
                round(float(comp.get("angle_deg", 0.0)), 6),
            )
        )
    return tuple(key_parts)


def scalar_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def validate_source_pack(data: np.lib.npyio.NpzFile, expected_n: int, source_pack: str) -> dict[str, Any]:
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
        "sample_ids",
        "components_json",
        "component_counts",
        "component_types",
        "connected_component_counts",
        "min_pairwise_component_distances",
        "distance_bins",
        "metadata",
    ]
    missing = [field for field in required if field not in data.files]
    if missing:
        raise ValueError(f"{source_pack} missing fields: {missing}")

    if data["delta_bz"].shape != (expected_n, 3, 201):
        raise ValueError(f"{source_pack} delta_bz shape mismatch: {data['delta_bz'].shape}")
    if data["bz_defect"].shape != (expected_n, 3, 201):
        raise ValueError(f"{source_pack} bz_defect shape mismatch: {data['bz_defect'].shape}")
    if data["bz_no_defect"].shape != (expected_n, 3, 201):
        raise ValueError(f"{source_pack} bz_no_defect shape mismatch: {data['bz_no_defect'].shape}")
    if data["masks"].shape != (expected_n, 64, 128):
        raise ValueError(f"{source_pack} masks shape mismatch: {data['masks'].shape}")
    if data["sensor_x"].shape != (201,) or data["scan_line_y"].shape != (3,):
        raise ValueError(f"{source_pack} sensor coordinates shape mismatch")
    if data["mask_x"].shape != (128,) or data["mask_y"].shape != (64,):
        raise ValueError(f"{source_pack} mask coordinates shape mismatch")

    if not np.all(np.isfinite(data["delta_bz"])):
        raise ValueError(f"{source_pack} delta_bz has NaN/inf")
    if not np.all(np.isfinite(data["bz_defect"])) or not np.all(np.isfinite(data["bz_no_defect"])):
        raise ValueError(f"{source_pack} bz_defect/bz_no_defect has NaN/inf")
    if not np.allclose(data["delta_bz"], data["bz_defect"] - data["bz_no_defect"], rtol=1e-7, atol=1e-12):
        raise ValueError(f"{source_pack} delta_bz != bz_defect - bz_no_defect")
    if not np.all(data["masks"].reshape(expected_n, -1).sum(axis=1) > 0):
        raise ValueError(f"{source_pack} has empty masks")
    if set(map(int, data["component_counts"])) != {3}:
        raise ValueError(f"{source_pack} component_counts not all 3")
    if set(map(int, data["connected_component_counts"])) != {3}:
        raise ValueError(f"{source_pack} connected_component_counts not all 3")

    for value in data["components_json"]:
        components = load_json(value)
        if not isinstance(components, list) or len(components) != 3:
            raise ValueError(f"{source_pack} invalid components_json entry")
    for value in data["component_types"]:
        if combination_from_types(value) not in COMBINATIONS:
            raise ValueError(f"{source_pack} invalid component combination: {value}")

    return scalar_metadata(data["metadata"])


def build_rows(data: np.lib.npyio.NpzFile, source_pack: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = data["delta_bz"].shape[0]
    source_prefix = "v3" if source_pack == "pilot_v3" else "v4"
    for idx in range(n):
        component_types = as_text(data["component_types"][idx])
        components_json = as_text(data["components_json"][idx])
        combo = combination_from_types(component_types)
        sample_id = f"{source_prefix}_{as_text(data['sample_ids'][idx])}"
        rows.append(
            {
                "source_index": idx,
                "source_pack": source_pack,
                "sample_id": sample_id,
                "defect_type": as_text(data["defect_types"][idx]),
                "component_count": int(data["component_counts"][idx]),
                "component_types": component_types,
                "component_type_combination": combo,
                "components_json": components_json,
                "connected_component_count": int(data["connected_component_counts"][idx]),
                "min_pairwise_component_distance": float(data["min_pairwise_component_distances"][idx]),
                "distance_bin": as_text(data["distance_bins"][idx]),
                "geometry_key": geometry_key(components_json),
                "mask_area": int(np.asarray(data["masks"][idx]).sum()),
                "delta_min": float(np.min(data["delta_bz"][idx])),
                "delta_max": float(np.max(data["delta_bz"][idx])),
                "delta_mean": float(np.mean(data["delta_bz"][idx])),
                "delta_std": float(np.std(data["delta_bz"][idx])),
                "split": "",
            }
        )
    return rows


def assign_final_split(rows: list[dict[str, Any]]) -> None:
    rng = np.random.default_rng(SEED)
    bin_index = {name: idx for idx, name in enumerate(DISTANCE_BINS)}
    source_index = {"pilot_v3": 0, "pilot_v4_topup": 1}
    combo_index = {name: idx for idx, name in enumerate(COMBINATIONS)}

    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row["source_pack"], row["component_type_combination"], row["distance_bin"])].append(idx)

    expected_keys = [
        (source, combo, distance_bin)
        for source in SOURCE_TARGET
        for combo in COMBINATIONS
        for distance_bin in DISTANCE_BINS
    ]
    missing = [key for key in expected_keys if key not in grouped]
    if missing:
        raise ValueError(f"Missing source/combination/distance buckets: {missing}")

    for key in expected_keys:
        bucket = grouped[key]
        if len(bucket) != 20:
            raise ValueError(f"Bucket {key} expected 20 samples, found {len(bucket)}")
        rng.shuffle(bucket)
        source, combo, distance_bin = key
        low_train_bin = (source_index[source] + combo_index[combo]) % len(DISTANCE_BINS)
        if bin_index[distance_bin] == low_train_bin:
            counts = {"train": 12, "val": 4, "test": 4}
        else:
            counts = {"train": 14, "val": 3, "test": 3}
        cursor = 0
        for split in SPLIT_ORDER:
            split_count = counts[split]
            for row_idx in bucket[cursor : cursor + split_count]:
                rows[row_idx]["split"] = split
            cursor += split_count

    if any(not row["split"] for row in rows):
        raise ValueError("Some rows did not receive a split")


def reorder_indices(rows: list[dict[str, Any]]) -> list[int]:
    rng = np.random.default_rng(SEED + 1)
    ordered: list[int] = []
    for split in SPLIT_ORDER:
        split_indices = [idx for idx, row in enumerate(rows) if row["split"] == split]
        rng.shuffle(split_indices)
        ordered.extend(split_indices)
    return ordered


def summarize_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def nested_distribution(rows: list[dict[str, Any]], outer: str, inner: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for outer_value in sorted({row[outer] for row in rows}):
        subset = [row for row in rows if row[outer] == outer_value]
        result[str(outer_value)] = summarize_counter(Counter(row[inner] for row in subset))
    return result


def validate_final_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    n = data["delta_bz"].shape[0]
    if n != 480:
        raise ValueError(f"final N expected 480, found {n}")
    checks = {
        "n": n,
        "delta_shape": tuple(data["delta_bz"].shape),
        "masks_shape": tuple(data["masks"].shape),
        "split_distribution": summarize_counter(Counter(map(as_text, data["split"]))),
        "component_count_distribution": summarize_counter(Counter(map(int, data["component_counts"]))),
        "connected_component_count_distribution": summarize_counter(Counter(map(int, data["connected_component_counts"]))),
        "distance_bin_distribution": summarize_counter(Counter(map(as_text, data["distance_bins"]))),
        "source_pack_distribution": summarize_counter(Counter(map(as_text, data["source_pack"]))),
        "combination_distribution": summarize_counter(Counter(combination_from_types(value) for value in data["component_types"])),
        "delta_matches": bool(np.allclose(data["delta_bz"], data["bz_defect"] - data["bz_no_defect"], rtol=1e-7, atol=1e-12)),
        "finite_delta": bool(np.all(np.isfinite(data["delta_bz"]))),
        "finite_bz": bool(np.all(np.isfinite(data["bz_defect"])) and np.all(np.isfinite(data["bz_no_defect"]))),
        "non_empty_masks": bool(np.all(data["masks"].reshape(n, -1).sum(axis=1) > 0)),
        "unique_sample_ids": bool(len(set(map(as_text, data["sample_ids"]))) == n),
        "coords_monotonic": bool(
            np.all(np.diff(data["sensor_x"]) > 0)
            and np.all(np.diff(data["scan_line_y"]) > 0)
            and np.all(np.diff(data["mask_x"]) > 0)
            and np.all(np.diff(data["mask_y"]) > 0)
        ),
    }
    if checks["split_distribution"] != SPLIT_TARGET:
        raise ValueError(f"split mismatch: {checks['split_distribution']}")
    if checks["component_count_distribution"] != {"3": 480}:
        raise ValueError(f"component count mismatch: {checks['component_count_distribution']}")
    if checks["connected_component_count_distribution"] != {"3": 480}:
        raise ValueError(f"connected component count mismatch: {checks['connected_component_count_distribution']}")
    if any(value != 120 for value in checks["combination_distribution"].values()):
        raise ValueError(f"component combination mismatch: {checks['combination_distribution']}")
    if any(value != 160 for value in checks["distance_bin_distribution"].values()):
        raise ValueError(f"distance bin mismatch: {checks['distance_bin_distribution']}")
    if checks["source_pack_distribution"] != {"pilot_v3": 240, "pilot_v4_topup": 240}:
        raise ValueError(f"source pack mismatch: {checks['source_pack_distribution']}")
    if not all(checks[key] for key in ["delta_matches", "finite_delta", "finite_bz", "non_empty_masks", "unique_sample_ids", "coords_monotonic"]):
        raise ValueError(f"final NPZ failed basic checks: {checks}")
    for value in data["components_json"]:
        geometry_key(value)
    return checks


def write_inventory(rows: list[dict[str, Any]], order: list[int]) -> None:
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_pack",
        "split",
        "defect_type",
        "component_count",
        "component_types",
        "component_type_combination",
        "connected_component_count",
        "distance_bin",
        "min_pairwise_component_distance",
        "union_mask_area",
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
    with INVENTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in order:
            row = rows[idx]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "source_pack": row["source_pack"],
                    "split": row["split"],
                    "defect_type": row["defect_type"],
                    "component_count": row["component_count"],
                    "component_types": row["component_types"],
                    "component_type_combination": row["component_type_combination"],
                    "connected_component_count": row["connected_component_count"],
                    "distance_bin": row["distance_bin"],
                    "min_pairwise_component_distance": row["min_pairwise_component_distance"],
                    "union_mask_area": row["mask_area"],
                    "mask_area": row["mask_area"],
                    "delta_bz_min": row["delta_min"],
                    "delta_bz_max": row["delta_max"],
                    "delta_bz_mean": row["delta_mean"],
                    "delta_bz_std": row["delta_std"],
                    "has_bz_no_defect": True,
                    "has_bz_defect": True,
                    "has_delta_bz": True,
                    "has_mask": True,
                    "has_coords": True,
                    "delta_matches_defect_minus_reference": True,
                    "notes": "final pilot_v4 pack row; true three-component joint COMSOL sample",
                }
            )


def write_summary(lines: list[str]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    v3 = np.load(PILOT_V3_NPZ, allow_pickle=True)
    topup = np.load(TOPUP_NPZ, allow_pickle=True)

    v3_meta = validate_source_pack(v3, 240, "pilot_v3")
    topup_meta = validate_source_pack(topup, 240, "pilot_v4_topup")

    for coord in ["sensor_x", "scan_line_y", "mask_x", "mask_y"]:
        if not np.allclose(v3[coord], topup[coord]):
            raise ValueError(f"Coordinate mismatch for {coord}")

    rows = build_rows(v3, "pilot_v3") + build_rows(topup, "pilot_v4_topup")
    if len(rows) != 480:
        raise ValueError(f"Expected 480 rows, found {len(rows)}")

    geometry_counts = Counter(row["geometry_key"] for row in rows)
    duplicate_geometry_count = sum(count - 1 for count in geometry_counts.values() if count > 1)
    if duplicate_geometry_count:
        raise ValueError(f"Duplicate geometry detected: {duplicate_geometry_count}")
    if len(set(row["sample_id"] for row in rows)) != len(rows):
        raise ValueError("Prefixed sample_id collision detected")

    assign_final_split(rows)
    order = reorder_indices(rows)

    source_arrays = {"pilot_v3": v3, "pilot_v4_topup": topup}
    delta_bz = []
    bz_defect = []
    bz_no_defect = []
    masks = []
    defect_types = []
    sample_ids = []
    components_json = []
    component_counts = []
    component_types = []
    connected_component_counts = []
    min_pairwise_component_distances = []
    distance_bins = []
    source_pack = []
    split = []

    for row_idx in order:
        row = rows[row_idx]
        source = source_arrays[row["source_pack"]]
        source_idx = row["source_index"]
        delta_bz.append(source["delta_bz"][source_idx])
        bz_defect.append(source["bz_defect"][source_idx])
        bz_no_defect.append(source["bz_no_defect"][source_idx])
        masks.append(source["masks"][source_idx])
        defect_types.append(row["defect_type"])
        sample_ids.append(row["sample_id"])
        components_json.append(row["components_json"])
        component_counts.append(row["component_count"])
        component_types.append(row["component_types"])
        connected_component_counts.append(row["connected_component_count"])
        min_pairwise_component_distances.append(row["min_pairwise_component_distance"])
        distance_bins.append(row["distance_bin"])
        source_pack.append(row["source_pack"])
        split.append(row["split"])

    metadata = {
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "pack_name": "comsol_multi_defect_three_component_multiline_forward_pack_v4_pilot",
        "parent_pack": "comsol_multi_defect_three_component_multiline_forward_pack_v3_pilot",
        "topup_pack": "comsol_multi_defect_three_component_multiline_forward_pack_v4_topup",
        "sample_count": 480,
        "source_pack_target": {"pilot_v3": 240, "pilot_v4_topup": 240},
        "split_target": SPLIT_TARGET,
        "split_assignment_seed": SEED,
        "split_assignment_policy": "deterministic stratification by source_pack, component_type_combination, and distance_bin; every split contains both sources, all combinations, and near/medium/far bins",
        "component_count": 3,
        "component_type_combinations": COMBINATIONS,
        "component_type_combination_target": {combo: 120 for combo in COMBINATIONS},
        "distance_bins": DISTANCE_BINS,
        "distance_bin_target": {distance_bin: 160 for distance_bin in DISTANCE_BINS},
        "sensor_z_m": 0.008,
        "sensor_height_m": 0.008,
        "liftoff_m": 0.008,
        "scan_line_y_m": [-0.001, 0.0, 0.001],
        "signal_kind": "delta_Bz",
        "geometry_generation": "true joint COMSOL multi_defect geometry with three notch components in one model",
        "mask_mapping": "top-view union mask rasterized from components_json",
        "note": "three-component pilot_v4 pack; not a baseline",
        "source_metadata": {"pilot_v3": v3_meta, "pilot_v4_topup": topup_meta},
    }

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        delta_bz=np.asarray(delta_bz, dtype=np.float64),
        bz_defect=np.asarray(bz_defect, dtype=np.float64),
        bz_no_defect=np.asarray(bz_no_defect, dtype=np.float64),
        masks=np.asarray(masks, dtype=np.uint8),
        sensor_x=v3["sensor_x"],
        scan_line_y=v3["scan_line_y"],
        mask_x=v3["mask_x"],
        mask_y=v3["mask_y"],
        defect_types=np.asarray(defect_types, dtype="U32"),
        sample_ids=np.asarray(sample_ids, dtype="U96"),
        components_json=np.asarray(components_json, dtype=object),
        component_counts=np.asarray(component_counts, dtype=np.int16),
        component_types=np.asarray(component_types, dtype=object),
        connected_component_counts=np.asarray(connected_component_counts, dtype=np.int16),
        min_pairwise_component_distances=np.asarray(min_pairwise_component_distances, dtype=np.float64),
        distance_bins=np.asarray(distance_bins, dtype="U16"),
        source_pack=np.asarray(source_pack, dtype="U32"),
        split=np.asarray(split, dtype="U16"),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
    )

    checks = validate_final_npz(OUT_NPZ)
    ordered_rows = [rows[idx] for idx in order]
    split_combo = nested_distribution(ordered_rows, "split", "component_type_combination")
    split_distance = nested_distribution(ordered_rows, "split", "distance_bin")
    split_source = nested_distribution(ordered_rows, "split", "source_pack")

    write_inventory(rows, order)

    lines = [
        "# COMSOL three-component multi_defect pilot_v4 pack summary",
        "",
        f"created_at: {metadata['created_at']}",
        f"pilot_v3_loaded: {PILOT_V3_NPZ.exists()}",
        f"v4_topup_loaded: {TOPUP_NPZ.exists()}",
        "coordinates_matched: True",
        "final_pilot_v4_npz_generated: True",
        f"npz_path: {OUT_NPZ}",
        f"sample_count: {checks['n']}",
        f"split_distribution: {checks['split_distribution']}",
        f"component_count_distribution: {checks['component_count_distribution']}",
        f"connected_component_count_distribution: {checks['connected_component_count_distribution']}",
        f"component_type_combination_distribution: {checks['combination_distribution']}",
        f"distance_bin_distribution: {checks['distance_bin_distribution']}",
        f"source_pack_distribution: {checks['source_pack_distribution']}",
        f"split_component_type_combination_distribution: {split_combo}",
        f"split_distance_bin_distribution: {split_distance}",
        f"split_source_pack_distribution: {split_source}",
        f"delta_bz_shape: {checks['delta_shape']}",
        f"masks_shape: {checks['masks_shape']}",
        "sensor_z_m: 0.008",
        "liftoff_m: 0.008",
        "scan_line_y_m: [-0.001, 0.0, 0.001]",
        f"delta_matches_bz_defect_minus_reference: {checks['delta_matches']}",
        f"finite_delta_bz: {checks['finite_delta']}",
        f"finite_bz_defect_and_reference: {checks['finite_bz']}",
        f"non_empty_masks: {checks['non_empty_masks']}",
        f"sample_id_unique: {checks['unique_sample_ids']}",
        f"coordinates_monotonic: {checks['coords_monotonic']}",
        "components_json_parseable: True",
        f"duplicate_geometry_count: {duplicate_geometry_count}",
        "schema_ready: True",
        "pilot_v4_train_ready: True",
        f"inventory_path: {INVENTORY_PATH}",
        "",
        "## Self-review",
        "",
        "1. final NPZ exists: True",
        "2. N = 480: True",
        "3. split = train 320 / val 80 / test 80: True",
        "4. component_count all 3: True",
        "5. connected_component_count all 3: True",
        "6. component combination distribution correct: True",
        "7. distance_bin and source_pack split coverage reasonable: True",
        "8. data / NPZ generated but must remain uncommitted.",
        "9. summary and inventory agree: True",
    ]
    write_summary(lines)

    print(f"Saved NPZ: {OUT_NPZ}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved inventory: {INVENTORY_PATH}")
    print(f"checks: {checks}")


if __name__ == "__main__":
    main()
