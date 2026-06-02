#!/usr/bin/env python
"""Assemble the 25.9b surface multi-pit component-set pilot pack."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "comsol_surface_multipit_component_set_pilot_v1"
ROUTE = "surface_multipit_component_set"
SCHEMA_VERSION = "surface_multipit_component_set_pilot_v1"
COMPONENT_SCHEMA_VERSION = "surface_multipit_component_schema_v1"
TOPOLOGY_SCHEMA_VERSION = "surface_multipit_topology_schema_v1"
SOURCE_DATASET_ID = "comsol_surface_shape_extension_pilot_v1"
TOPUP_DATASET_ID = "comsol_surface_multipit_topup_pack_v1"
K_MAX = 3
MASK_HEIGHT = 64
MASK_WIDTH = 128
MASK_X_START_M = -0.04
MASK_X_STOP_M = 0.04
MASK_Y_START_M = -0.01
MASK_Y_STOP_M = 0.01
TARGET_SPLIT = {"train": 72, "val": 20, "test": 20}

DEFAULT_SOURCE_MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
DEFAULT_AUDIT = ROOT / "results/metrics/surface_multipit_component_label_audit.csv"
DEFAULT_TOPUP = ROOT / "data/comsol_mfl/generated/surface_multipit_topup_pack_v1/surface_multipit_topup_pack_v1.npz"
DEFAULT_OUTPUT = ROOT / "data/comsol_mfl/prepared/experimental/surface_multipit_component_set/comsol_surface_multipit_component_set_pilot_v1.npz"
DEFAULT_SUMMARY = ROOT / "results/summaries/surface_multipit_component_set_pilot_assembly_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/surface_multipit_component_set_pilot_assembly_metrics.csv"

METRIC_FIELDS = ["metric_name", "value", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble 25.9b component-set pilot pack.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--topup", type=Path, default=DEFAULT_TOPUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as pack:
        return {name: pack[name].copy() for name in pack.files}


def json_loads(value: Any) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        value = str(value)
    return json.loads(value)


def points_in_polygon(xx: np.ndarray, yy: np.ndarray, vertices: list[list[float]]) -> np.ndarray:
    xv = np.asarray([point[0] for point in vertices], dtype=np.float64)
    yv = np.asarray([point[1] for point in vertices], dtype=np.float64)
    inside = np.zeros(xx.shape, dtype=bool)
    j = len(vertices) - 1
    for i in range(len(vertices)):
        yi = yv[i]
        yj = yv[j]
        xi = xv[i]
        xj = xv[j]
        crosses = ((yi > yy) != (yj > yy)) & (xx < (xj - xi) * (yy - yi) / ((yj - yi) + 1e-18) + xi)
        inside ^= crosses
        j = i
    return inside


def derive_component_grids(geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mask_x = np.linspace(MASK_X_START_M, MASK_X_STOP_M, MASK_WIDTH)
    mask_y = np.linspace(MASK_Y_START_M, MASK_Y_STOP_M, MASK_HEIGHT)
    xx, yy = np.meshgrid(mask_x, mask_y, indexing="xy")
    masks = np.zeros((K_MAX, MASK_HEIGHT, MASK_WIDTH), dtype=np.uint8)
    depths = np.zeros((K_MAX, MASK_HEIGHT, MASK_WIDTH), dtype=np.float64)
    for layer in geometry.get("layers", []):
        component_id = int(layer.get("component_id", 0))
        if not (1 <= component_id <= K_MAX):
            continue
        vertices = layer.get("vertices", [])
        if len(vertices) < 3:
            continue
        layer_mask = points_in_polygon(xx, yy, vertices)
        slot = component_id - 1
        depth = float(layer.get("depth_m", 0.0))
        masks[slot] = np.maximum(masks[slot], layer_mask.astype(np.uint8))
        depths[slot] = np.maximum(depths[slot], layer_mask.astype(np.float64) * depth)
    return masks, depths


def normalize_old_components(raw_components: list[dict[str, Any]], rotation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    exists = np.zeros(K_MAX, dtype=bool)
    centers = np.zeros((K_MAX, 2), dtype=np.float64)
    lwd = np.zeros((K_MAX, 3), dtype=np.float64)
    rotations = np.zeros(K_MAX, dtype=np.float64)
    families = np.full(K_MAX, "none", dtype="<U32")
    params: list[dict[str, Any]] = []
    family_json = ["{}"] * K_MAX
    for component in raw_components[:K_MAX]:
        slot = int(component.get("component_id", len(params) + 1)) - 1
        if not (0 <= slot < K_MAX):
            continue
        exists[slot] = True
        centers[slot] = [float(component["center_x_m"]), float(component["center_y_m"])]
        depth = float(component.get("D_m", component.get("depth_m", 0.0)))
        lwd[slot] = [float(component["L_m"]), float(component["W_m"]), depth]
        rotations[slot] = float(rotation)
        family = str(component.get("shape_family", component.get("component_type", "flat_component_pit")))
        families[slot] = family[:31]
        local_profile = {
            "profile_family": family,
            "edge_steepness": 0.72,
            "flatness": 0.9 if "flat" in family else 0.25,
            "source": "reconstructed_from_25_2_layers",
        }
        normalized = {
            "component_id": slot + 1,
            "existence_prob": 1.0,
            "center_x_m": float(centers[slot, 0]),
            "center_y_m": float(centers[slot, 1]),
            "L_m": float(lwd[slot, 0]),
            "W_m": float(lwd[slot, 1]),
            "D_m": float(lwd[slot, 2]),
            "rotation_angle": float(rotations[slot]),
            "shape_family": family,
            "local_profile_params": local_profile,
        }
        params.append(normalized)
        family_json[slot] = json.dumps(local_profile, sort_keys=True)
    return exists, centers, lwd, rotations, families, [json.dumps(params, sort_keys=True)], family_json


def collect_old_seed(source: dict[str, Any], audit_rows: list[dict[str, str]]) -> dict[str, Any]:
    sample_ids = np.asarray(source["sample_ids"]).astype(str)
    old_indices = np.where(np.asarray(source["shape_type"]).astype(str) == "multi_pit_two_component_surface_defect")[0]
    audit_by_id = {row["sample_id"]: row for row in audit_rows}
    component_exists: list[np.ndarray] = []
    component_centers: list[np.ndarray] = []
    component_lwd: list[np.ndarray] = []
    component_rotation: list[np.ndarray] = []
    component_family: list[np.ndarray] = []
    component_masks: list[np.ndarray] = []
    component_depths: list[np.ndarray] = []
    component_params: list[str] = []
    separation: list[str] = []
    touching_overlap: list[str] = []
    topology_relation: list[str] = []
    relative_depth: list[str] = []
    size_pair: list[str] = []
    orientation: list[str] = []
    primitive_mix: list[str] = []
    geometry_json: list[str] = []
    profile_json: list[str] = []
    for idx in old_indices:
        sample_id = str(sample_ids[idx])
        geometry = json_loads(source["geometry_params_json"][idx])
        raw_components = json_loads(source["component_params_json"][idx])
        rotation = float(source["rotation_angle"][idx])
        masks, depths = derive_component_grids(geometry)
        exists, centers, lwd, rotations, families, params, _family_json = normalize_old_components(raw_components, rotation)
        component_exists.append(exists)
        component_centers.append(centers)
        component_lwd.append(lwd)
        component_rotation.append(rotations)
        component_family.append(families)
        component_masks.append(masks)
        component_depths.append(depths)
        component_params.append(params[0])
        audit = audit_by_id.get(sample_id, {})
        sep = audit.get("separation_bucket", "separated")
        overlap = audit.get("overlap_status", "bbox_separated")
        topology = audit.get("topology_bucket", "disconnected")
        separation.append(sep)
        touching_overlap.append("separated" if overlap in {"bbox_separated", "separated"} else overlap)
        topology_relation.append(topology)
        relative_depth.append(audit.get("depth_relation", "deep_and_shallow"))
        size_pair.append(audit.get("size_pair_bucket", "medium-medium"))
        orientation.append(audit.get("orientation_bucket", "aligned_x"))
        primitive_mix.append("flat-flat")
        geometry_json.append(json.dumps(geometry, sort_keys=True))
        profile_json.append(json.dumps({"K_max": K_MAX, "representation_target": "component_set", "source": "25.2_multi_pit_seed_reconstructed"}, sort_keys=True))
    union_masks = np.asarray(source["projected_mask_2d"][old_indices], dtype=np.uint8)
    union_depths = np.asarray(source["depth_grid_m"][old_indices], dtype=np.float64)
    result = {
        "indices": old_indices,
        "sample_ids": sample_ids[old_indices].astype("<U128"),
        "split": np.asarray(source["split"])[old_indices].astype("<U16"),
        "shape_type": np.full(len(old_indices), "multi_pit_two_component_surface_defect", dtype="<U80"),
        "topology_type": np.full(len(old_indices), "multi_component", dtype="<U40"),
        "representation_target": np.full(len(old_indices), "component_set", dtype="<U40"),
        "rbc_compatible": np.zeros(len(old_indices), dtype=bool),
        "component_count": np.asarray(source["component_count"][old_indices], dtype=np.int64),
        "component_exists": np.stack(component_exists, axis=0),
        "component_center_xy_m": np.stack(component_centers, axis=0),
        "component_lwd_m": np.stack(component_lwd, axis=0),
        "component_rotation_angle": np.stack(component_rotation, axis=0),
        "component_shape_family": np.stack(component_family, axis=0).astype("<U32"),
        "component_projected_masks_2d": np.stack(component_masks, axis=0).astype(np.uint8),
        "component_depth_grids_m": np.stack(component_depths, axis=0).astype(np.float64),
        "projected_mask_2d": union_masks,
        "depth_grid_m": union_depths,
        "component_params_json": np.asarray(component_params, dtype=object),
        "profile_descriptor": np.asarray(profile_json, dtype=object),
        "L_m": np.asarray(source["L_m"][old_indices], dtype=np.float64),
        "W_m": np.asarray(source["W_m"][old_indices], dtype=np.float64),
        "D_m": np.asarray(source["D_m"][old_indices], dtype=np.float64),
        "center_xyz_m": np.asarray(source["center_xyz_m"][old_indices], dtype=np.float64),
        "surface_origin": np.asarray(source["surface_origin"][old_indices]).astype("<U32"),
        "aspect_ratio": np.asarray(source["aspect_ratio"][old_indices], dtype=np.float64),
        "rotation_angle": np.asarray(source["rotation_angle"][old_indices], dtype=np.float64),
        "asymmetry_score": np.asarray(source["asymmetry_score"][old_indices], dtype=np.float64),
        "edge_steepness": np.asarray(source["edge_steepness"][old_indices], dtype=np.float64),
        "separation_type": np.asarray(separation, dtype="<U40"),
        "touching_overlap_type": np.asarray(touching_overlap, dtype="<U40"),
        "topology_relation": np.asarray(topology_relation, dtype="<U40"),
        "relative_depth_type": np.asarray(relative_depth, dtype="<U40"),
        "size_pair_type": np.asarray(size_pair, dtype="<U40"),
        "orientation_type": np.asarray(orientation, dtype="<U40"),
        "primitive_mix": np.asarray(primitive_mix, dtype="<U40"),
        "geometry_method_used": np.asarray(source["geometry_method_used"][old_indices]).astype("<U80"),
        "geometry_params_json": np.asarray(geometry_json, dtype=object),
        "no_defect_reference_id": np.full(len(old_indices), "surface_shape_extension_no_defect_reference_v1", dtype="<U80"),
        "source_dataset_id": np.full(len(old_indices), SOURCE_DATASET_ID, dtype="<U80"),
    }
    return result


def take_topup(topup: dict[str, Any]) -> dict[str, Any]:
    n = len(topup["sample_ids"])
    fields = [
        "sample_ids",
        "split",
        "shape_type",
        "topology_type",
        "representation_target",
        "rbc_compatible",
        "component_count",
        "component_exists",
        "component_center_xy_m",
        "component_lwd_m",
        "component_rotation_angle",
        "component_shape_family",
        "component_projected_masks_2d",
        "component_depth_grids_m",
        "projected_mask_2d",
        "depth_grid_m",
        "component_params_json",
        "profile_descriptor",
        "L_m",
        "W_m",
        "D_m",
        "center_xyz_m",
        "surface_origin",
        "aspect_ratio",
        "rotation_angle",
        "asymmetry_score",
        "edge_steepness",
        "separation_type",
        "touching_overlap_type",
        "topology_relation",
        "relative_depth_type",
        "size_pair_type",
        "orientation_type",
        "primitive_mix",
        "geometry_method_used",
        "geometry_params_json",
        "no_defect_reference_id",
    ]
    result = {field: np.asarray(topup[field]).copy() for field in fields}
    result["source_dataset_id"] = np.full(n, TOPUP_DATASET_ID, dtype="<U80")
    return result


def concatenate(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in old:
        if field == "indices":
            continue
        result[field] = np.concatenate([old[field], new[field]], axis=0)
    return result


def save_pack(args: argparse.Namespace, source: dict[str, Any], topup: dict[str, Any], assembled: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "dataset_id": DATASET_ID,
        "route": ROUTE,
        "schema_version": SCHEMA_VERSION,
        "component_schema_version": COMPONENT_SCHEMA_VERSION,
        "topology_schema_version": TOPOLOGY_SCHEMA_VERSION,
        "stage": "25.9b",
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "source_dataset_ids": [SOURCE_DATASET_ID, TOPUP_DATASET_ID],
        "source_manifest": str(args.source_manifest),
        "topup_npz": str(args.topup),
        "K_max": K_MAX,
        "notes": "Generated NPZ is ignored data and must not be committed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        dataset_id=np.asarray([DATASET_ID], dtype="<U80"),
        route=np.asarray([ROUTE], dtype="<U80"),
        schema_version=np.asarray([SCHEMA_VERSION], dtype="<U80"),
        component_schema_version=np.asarray([COMPONENT_SCHEMA_VERSION], dtype="<U80"),
        topology_schema_version=np.asarray([TOPOLOGY_SCHEMA_VERSION], dtype="<U80"),
        status=np.asarray(["pilot_generated"], dtype="<U64"),
        K_max=np.asarray([K_MAX], dtype=np.int64),
        delta_b=np.concatenate([source["delta_b"][assembled["indices"]], topup["delta_b"]], axis=0).astype(np.float64),
        b_defect=np.concatenate([source["b_defect"][assembled["indices"]], topup["b_defect"]], axis=0).astype(np.float64),
        b_no_defect=np.concatenate([source["b_no_defect"][assembled["indices"]], topup["b_no_defect"]], axis=0).astype(np.float64),
        axis_names=np.asarray(source["axis_names"]).astype("<U8"),
        axis_expressions=np.asarray(source["axis_expressions"]).astype("<U16"),
        sensor_x=np.asarray(source["sensor_x"], dtype=np.float64),
        scan_line_y=np.asarray(source["scan_line_y"], dtype=np.float64),
        sensor_z_m=np.asarray(source["sensor_z_m"], dtype=np.float64),
        metadata=np.asarray(metadata, dtype=object),
        **{key: value for key, value in assembled.items() if key != "indices"},
    )
    return metadata


def write_outputs(args: argparse.Namespace, assembled: dict[str, Any], metadata: dict[str, Any]) -> None:
    sample_count = len(assembled["sample_ids"])
    split_counts = dict(Counter(str(item) for item in assembled["split"].tolist()))
    component_counts = dict(Counter(str(int(item)) for item in assembled["component_count"].tolist()))
    separation_counts = dict(Counter(str(item) for item in assembled["separation_type"].tolist()))
    topology_counts = dict(Counter(str(item) for item in assembled["topology_relation"].tolist()))
    orientation_counts = dict(Counter(str(item) for item in assembled["orientation_type"].tolist()))
    source_counts = dict(Counter(str(item) for item in assembled["source_dataset_id"].tolist()))
    train_ready_shape = split_counts == TARGET_SPLIT and sample_count == 112
    rows = [
        {"metric_name": "assembled_N", "value": sample_count, "notes": "target is 112"},
        {"metric_name": "split_counts", "value": json.dumps(split_counts, sort_keys=True), "notes": "target 72/20/20"},
        {"metric_name": "component_count_counts", "value": json.dumps(component_counts, sort_keys=True), "notes": "expected 100 two-component and 12 three-component after adding old seed"},
        {"metric_name": "separation_counts", "value": json.dumps(separation_counts, sort_keys=True), "notes": "old seed contributes disconnected/separated cases"},
        {"metric_name": "topology_counts", "value": json.dumps(topology_counts, sort_keys=True), "notes": "component-set topology relation coverage"},
        {"metric_name": "orientation_counts", "value": json.dumps(orientation_counts, sort_keys=True), "notes": "top-up balances x/y/diagonal; old seed is aligned_x"},
        {"metric_name": "source_dataset_counts", "value": json.dumps(source_counts, sort_keys=True), "notes": "old 16 plus top-up 96"},
        {"metric_name": "train_ready_shape", "value": train_ready_shape, "notes": "full readiness is checked by validation script"},
    ]
    write_csv(args.metrics, rows, METRIC_FIELDS)
    lines = [
        "surface multi-pit component-set pilot assembly summary",
        "stage: 25.9b",
        "",
        f"dataset_id: {DATASET_ID}",
        f"assembled_N: {sample_count}",
        f"split_counts: {split_counts}",
        f"component_count_counts: {component_counts}",
        f"separation_counts: {separation_counts}",
        f"topology_counts: {topology_counts}",
        f"orientation_counts: {orientation_counts}",
        f"source_dataset_counts: {source_counts}",
        f"K_max: {K_MAX}",
        f"assembled_npz: {args.output}",
        "",
        "component_label_policy:",
        "- Old 25.2 multi-pit seed rows are reconstructed from component layers into component-level masks/depth grids.",
        "- New 25.9b top-up rows carry native component masks/depth grids and explicit separation/topology labels.",
        "- The generated assembled NPZ is data and must not be committed.",
        "",
        f"metadata: {json.dumps(metadata, sort_keys=True)}",
        f"metrics_csv: {args.metrics}",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source = load_npz(Path(manifest["generated_npz_path"]))
    topup = load_npz(args.topup)
    audit_rows = read_csv(args.source_audit)
    old = collect_old_seed(source, audit_rows)
    new = take_topup(topup)
    assembled_body = concatenate(old, new)
    save_input = dict(assembled_body)
    save_input["indices"] = old["indices"]
    metadata = save_pack(args, source, topup, save_input)
    write_outputs(args, assembled_body, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
