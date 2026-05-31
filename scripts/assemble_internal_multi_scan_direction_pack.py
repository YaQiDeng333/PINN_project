#!/usr/bin/env python
"""组装 23.2b internal dual-direction diagnostic pack。

输入为 22.9 既有 x_scan richer-observation NPZ 与 23.2b 新生成 y_scan top-up NPZ。
输出 assembled NPZ 到 ignored data 路径，并创建可提交 manifest/registry summary。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_multi_scan_direction_pack_v1"
Y_DATASET_ID = "comsol_internal_defect_multi_scan_direction_y_scan_pack_v1"
RICHER_DATASET_ID = "comsol_internal_defect_richer_observation_pack_v1"
SOURCE_DATASET_ID = "comsol_internal_defect_pilot_pack_v3_hardcase"
RICHER_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_richer_observation_pack_v1.manifest.json"
Y_SCAN_NPZ = ROOT / "data/comsol_mfl/generated/internal_multi_scan_direction_pack/internal_multi_scan_direction_y_scan_pack_v1.npz"
OUTPUT_NPZ = ROOT / "data/comsol_mfl/generated/internal_multi_scan_direction_pack/comsol_internal_defect_multi_scan_direction_pack_v1.npz"
SUMMARY = ROOT / "results/summaries/internal_multi_scan_direction_pack_assembly_summary.txt"
METRICS = ROOT / "results/metrics/internal_multi_scan_direction_pack_assembly_metrics.csv"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_multi_scan_direction_pack_v1.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
REGISTRY_SUMMARY = ROOT / "results/summaries/internal_multi_scan_direction_pack_registry_summary.txt"

ROUTE = "internal_buried_defect_multi_scan_direction"
SCHEMA_VERSION = "internal_defect_multi_scan_direction_v1"
ALLOWED_USE = ["schema_validation", "explicit_multi_scan_direction_diagnostic"]
FORBIDDEN_USE = [
    "automatic_mainline_training",
    "baseline_update",
    "current_baseline_replacement",
    "latest_newest_auto_discovery",
    "direct_training_without_manifest_gate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="组装 internal dual-direction diagnostic pack。")
    parser.add_argument("--richer-manifest", type=Path, default=RICHER_MANIFEST)
    parser.add_argument("--y-scan-npz", type=Path, default=Y_SCAN_NPZ)
    parser.add_argument("--output-npz", type=Path, default=OUTPUT_NPZ)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--registry-summary", type=Path, default=REGISTRY_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def strings(arr: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(arr).reshape(-1).tolist()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("拒绝覆盖已存在文件；如需重跑请显式加 --overwrite:\n" + "\n".join(str(path) for path in existing))


def richer_npz_from_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != RICHER_DATASET_ID:
        raise ValueError(f"richer manifest dataset_id 不匹配: {payload.get('dataset_id')}")
    npz_path = Path(str(payload.get("npz_path", "")))
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    return payload, npz_path


def build_x_index(x_arrays: dict[str, np.ndarray]) -> dict[tuple[str, str], int]:
    index: dict[tuple[str, str], int] = {}
    for idx, (base, variant) in enumerate(zip(strings(x_arrays["base_group_id"]), strings(x_arrays["observation_variant"]), strict=True)):
        index[(base, variant)] = idx
    return index


def complete_base_count(base_ids: list[str], variants: list[str]) -> int:
    by_base: dict[str, set[str]] = defaultdict(set)
    for base, variant in zip(base_ids, variants, strict=True):
        by_base[base].add(variant)
    return sum(1 for values in by_base.values() if {"D1_y_scan_5line_z0p008", "D2_y_scan_9line_z0p008"}.issubset(values))


def counts(arrays: dict[str, np.ndarray], field: str) -> dict[str, int]:
    return dict(Counter(strings(arrays[field]))) if field in arrays else {}


def registry_entry(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"## {manifest['dataset_id']}",
            "",
            f"- dataset_role: {manifest['dataset_role']}",
            f"- status: {manifest['status']}",
            f"- route: {manifest['route']}",
            f"- stage: {manifest['stage']}",
            f"- schema_version: {manifest['schema_version']}",
            f"- internal_surface_mixed: {str(manifest['internal_surface_mixed']).lower()}",
            f"- path: `{manifest['npz_path']}`",
            f"- manifest_path: `{manifest['manifest_path']}`",
            f"- n_samples: {manifest['n_samples']}",
            f"- base_count: {manifest['base_count']}",
            f"- complete_base_count: {manifest['complete_base_count']}",
            f"- assembled_delta_shape: {manifest['assembled_delta_shape']}",
            f"- direction_names: {manifest['direction_names']}",
            f"- observation_variants: {manifest['observation_variants']}",
            f"- paired_x_variants: {manifest['paired_x_variants']}",
            f"- shape_counts: {manifest['shape_counts']}",
            f"- burial_depth_counts: {manifest['burial_depth_counts']}",
            f"- train_ready_candidate: {str(manifest['train_ready_candidate']).lower()}",
            f"- baseline_ready: {str(manifest['baseline_ready']).lower()}",
            f"- auto_discovery_allowed: {str(manifest['auto_discovery_allowed']).lower()}",
            f"- latest_newest_discovery_allowed: {str(manifest['latest_newest_discovery_allowed']).lower()}",
            f"- allowed_use: {', '.join(manifest['allowed_use'])}",
            f"- forbidden_use: {', '.join(manifest['forbidden_use'])}",
            f"- source_dataset_ids: {', '.join(manifest['source_dataset_ids'])}",
            f"- npz_sha256: {manifest['npz_sha256']}",
            "- notes: Dual-direction diagnostic pack only. Generated NPZ/data files are not committed; not a baseline.",
            "",
        ]
    )


def update_registry(path: Path, manifest: dict[str, Any]) -> None:
    entry = registry_entry(manifest)
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else "# COMSOL Data Registry\n\n"
    heading = f"## {manifest['dataset_id']}"
    if heading in text:
        start = text.index(heading)
        next_start = text.find("\n## ", start + 1)
        if next_start == -1:
            text = text[:start].rstrip() + "\n\n" + entry
        else:
            text = text[:start].rstrip() + "\n\n" + entry + text[next_start:].lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + entry
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.output_npz, args.summary, args.metrics, args.manifest, args.registry_summary], args.overwrite)
    richer_manifest, richer_npz = richer_npz_from_manifest(args.richer_manifest)
    x_arrays = load_npz(richer_npz)
    y_arrays = load_npz(args.y_scan_npz)
    x_index = build_x_index(x_arrays)

    y_n = int(len(y_arrays["sample_ids"]))
    rows: list[dict[str, Any]] = []
    delta_rows: list[np.ndarray] = []
    b_defect_rows: list[np.ndarray] = []
    b_no_rows: list[np.ndarray] = []
    direction_masks: list[np.ndarray] = []
    scan_masks: list[np.ndarray] = []
    path_coords: list[np.ndarray] = []
    line_coords: list[np.ndarray] = []
    x_sample_ids: list[str] = []
    y_sample_ids: list[str] = []

    x_path = np.asarray(x_arrays["sensor_x"], dtype=np.float32)
    for y_idx in range(y_n):
        base = str(y_arrays["base_group_id"][y_idx])
        x_variant = str(y_arrays["paired_existing_x_variant"][y_idx])
        key = (base, x_variant)
        if key not in x_index:
            raise KeyError(f"找不到配对 x_scan: base={base}; variant={x_variant}")
        x_idx = x_index[key]
        delta_rows.append(np.stack([x_arrays["delta_b"][x_idx], y_arrays["delta_b"][y_idx]], axis=1))
        b_defect_rows.append(np.stack([x_arrays["b_defect"][x_idx], y_arrays["b_defect"][y_idx]], axis=1))
        b_no_rows.append(np.stack([x_arrays["b_no_defect"][x_idx], y_arrays["b_no_defect"][y_idx]], axis=1))
        direction_masks.append(np.asarray([True, True], dtype=bool))
        scan_masks.append(np.stack([x_arrays["scan_line_mask"][x_idx], y_arrays["scan_line_mask"][y_idx]], axis=0).astype(bool))
        path_coords.append(np.stack([x_path, y_arrays["path_coordinate_m"][y_idx]], axis=0).astype(np.float32))
        line_coords.append(np.stack([x_arrays["scan_line_y"][x_idx], y_arrays["line_coordinate_m"][y_idx]], axis=0).astype(np.float32))
        x_sample_ids.append(str(x_arrays["sample_ids"][x_idx]))
        y_sample_ids.append(str(y_arrays["sample_ids"][y_idx]))
        rows.append({"base_group_id": base, "y_variant": str(y_arrays["observation_variant"][y_idx]), "x_variant": x_variant})

    out: dict[str, np.ndarray] = {
        "dataset_id": np.asarray(DATASET_ID, dtype=object),
        "source_dataset_ids": np.asarray([RICHER_DATASET_ID, Y_DATASET_ID, SOURCE_DATASET_ID], dtype=object),
        "delta_b": np.stack(delta_rows, axis=0).astype(np.float32),
        "b_defect": np.stack(b_defect_rows, axis=0).astype(np.float32),
        "b_no_defect": np.stack(b_no_rows, axis=0).astype(np.float32),
        "direction_mask": np.stack(direction_masks, axis=0),
        "scan_line_mask": np.stack(scan_masks, axis=0),
        "path_coordinate_m": np.stack(path_coords, axis=0).astype(np.float32),
        "line_coordinate_m": np.stack(line_coords, axis=0).astype(np.float32),
        "direction_names": np.asarray(["x_scan", "y_scan"], dtype="<U16"),
        "path_coordinate_axis": np.asarray(["x", "y"], dtype="<U8"),
        "line_coordinate_axis": np.asarray(["y", "x"], dtype="<U8"),
        "axis_names": np.asarray(["Bx", "By", "Bz"], dtype="<U8"),
        "axis_expressions": np.asarray(["mf.Bx", "mf.By", "mf.Bz"], dtype="<U16"),
        "sample_ids": np.asarray([f"{str(y_arrays['base_group_id'][i])}_{str(y_arrays['observation_family'][i])}_dual_direction" for i in range(y_n)], dtype="<U128"),
        "base_group_id": y_arrays["base_group_id"],
        "base_sample_id": y_arrays["base_sample_id"],
        "x_scan_sample_id": np.asarray(x_sample_ids, dtype="<U128"),
        "y_scan_sample_id": np.asarray(y_sample_ids, dtype="<U128"),
        "x_observation_variant": y_arrays["paired_existing_x_variant"],
        "y_observation_variant": y_arrays["observation_variant"],
        "observation_variant": np.asarray([f"dual_direction_{str(v).split('_')[0]}_z0p008" for v in y_arrays["observation_variant"]], dtype="<U64"),
        "sensor_z_m": y_arrays["sensor_z_m"],
        "scan_direction": np.asarray(["dual_direction"] * y_n, dtype="<U32"),
    }

    for key in [
        "source_split",
        "source_subset",
        "true_shape_type",
        "pred_shape_type",
        "shape_type",
        "burial_depth_level",
        "size_level",
        "aspect_bin",
        "hardcase_target_id",
        "failure_tags",
        "L_m",
        "W_m",
        "D_m",
        "D_m_or_cavity_size_m",
        "burial_depth_m",
        "depth_to_surface_m",
        "defect_center_xyz_m",
        "cavity_internal",
        "ground_truth_method",
        "material",
        "specimen_geometry_json",
        "row_origin",
        "source_dataset_id_per_row",
    ]:
        if key in y_arrays:
            out[key] = y_arrays[key]

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    base_ids = strings(out["base_group_id"])
    variants = strings(out["y_observation_variant"])
    base_count = len(set(base_ids))
    complete = complete_base_count(base_ids, variants)
    status = "diagnostic_pack_generated" if y_n == 60 and complete == 30 else ("partial_diagnostic_pack_generated" if y_n >= 48 and complete >= 24 else "blocked")
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_role": "internal_defect_multi_scan_direction_diagnostic_pack",
        "stage": "23.2b",
        "route": ROUTE,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "npz_path": str(args.output_npz),
        "npz_sha256": sha256_file(args.output_npz),
        "manifest_path": str(args.manifest),
        "n_samples": y_n,
        "base_count": base_count,
        "complete_base_count": complete,
        "assembled_delta_shape": list(out["delta_b"].shape),
        "direction_names": ["x_scan", "y_scan"],
        "direction_mask_shape": list(out["direction_mask"].shape),
        "scan_line_mask_shape": list(out["scan_line_mask"].shape),
        "observation_variants": dict(Counter(strings(out["y_observation_variant"]))),
        "paired_x_variants": dict(Counter(strings(out["x_observation_variant"]))),
        "shape_counts": counts(out, "shape_type"),
        "burial_depth_counts": counts(out, "burial_depth_level"),
        "size_counts": counts(out, "size_level"),
        "aspect_counts": counts(out, "aspect_bin"),
        "source_dataset_ids": [RICHER_DATASET_ID, Y_DATASET_ID, SOURCE_DATASET_ID],
        "source_richer_manifest": str(args.richer_manifest),
        "source_richer_npz": str(richer_npz),
        "source_richer_sha256": richer_manifest.get("npz_sha256", ""),
        "source_y_scan_npz": str(args.y_scan_npz),
        "train_ready_candidate": False,
        "baseline_ready": False,
        "internal_surface_mixed": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "allowed_use": ALLOWED_USE,
        "forbidden_use": FORBIDDEN_USE,
        "assembly_script": "scripts/assemble_internal_multi_scan_direction_pack.py",
        "validation_script": "scripts/validate_internal_multi_scan_direction_pack.py",
        "current_baseline_update": False,
        "notes": "Dual-direction diagnostic only: x_scan comes from 22.9 richer pack; y_scan comes from 23.2b direction-aware generation. Not training data and not baseline.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_registry(args.registry, manifest)

    metric_rows = [
        {"metric": "assembled_rows", "value": y_n},
        {"metric": "base_count", "value": base_count},
        {"metric": "complete_base_count", "value": complete},
        {"metric": "delta_shape", "value": str(out["delta_b"].shape)},
        {"metric": "status", "value": status},
    ]
    write_csv(args.metrics, metric_rows, ["metric", "value"])
    lines = [
        "23.2b internal multi-scan-direction assembly summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"source_richer_npz: {richer_npz}",
        f"source_y_scan_npz: {args.y_scan_npz}",
        f"assembled_npz: {args.output_npz}",
        f"assembled_rows: {y_n}",
        f"base_count: {base_count}",
        f"complete_base_count: {complete}",
        f"assembled_delta_shape: {tuple(out['delta_b'].shape)}",
        "direction_names: ['x_scan', 'y_scan']",
        f"status: {status}",
        "train_ready_candidate: false",
        "baseline_ready: false",
        "",
        "结论: assembled pack 只用于 23.3 dual-direction diagnostic evaluation；不替代 CURRENT_BASELINE。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = [
        "23.2b internal multi-scan-direction registry summary",
        f"dataset_id: {DATASET_ID}",
        f"status: {status}",
        f"manifest: {args.manifest}",
        f"npz_path: {args.output_npz}",
        f"npz_sha256: {manifest['npz_sha256']}",
        "allowed_use: schema_validation, explicit_multi_scan_direction_diagnostic",
        "forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate",
    ]
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")
    return 0 if status != "blocked" else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
