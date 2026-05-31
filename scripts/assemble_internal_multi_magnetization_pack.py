#!/usr/bin/env python
"""组装 23.4 internal multi-magnetization diagnostic pack。

输入为 22.9 richer-observation pack 中的 nominal/mag_x reference rows，以及
23.4 新生成的 mag_y rows。输出 NPZ 放在 ignored data 路径，只提交 manifest、
registry summary 和 metrics。
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
DATASET_ID = "comsol_internal_defect_multi_magnetization_pack_v1"
MAG_Y_DATASET_ID = "comsol_internal_defect_multi_magnetization_mag_y_pack_v1"
RICHER_DATASET_ID = "comsol_internal_defect_richer_observation_pack_v1"
SOURCE_DATASET_ID = "comsol_internal_defect_pilot_pack_v3_hardcase"
RICHER_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_richer_observation_pack_v1.manifest.json"
MAG_Y_NPZ = ROOT / "data/comsol_mfl/generated/internal_multi_magnetization_pack/internal_multi_magnetization_mag_y_pack_v1.npz"
OUTPUT_NPZ = ROOT / "data/comsol_mfl/generated/internal_multi_magnetization_pack/comsol_internal_defect_multi_magnetization_pack_v1.npz"
SUMMARY = ROOT / "results/summaries/internal_multi_magnetization_pack_assembly_summary.txt"
METRICS = ROOT / "results/metrics/internal_multi_magnetization_pack_assembly_metrics.csv"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_multi_magnetization_pack_v1.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
REGISTRY_SUMMARY = ROOT / "results/summaries/internal_multi_magnetization_pack_registry_summary.txt"

ROUTE = "internal_buried_defect_multi_magnetization"
SCHEMA_VERSION = "internal_defect_multi_magnetization_v1"
ALLOWED_USE = ["schema_validation", "explicit_multi_magnetization_diagnostic"]
FORBIDDEN_USE = [
    "automatic_mainline_training",
    "baseline_update",
    "current_baseline_replacement",
    "latest_newest_auto_discovery",
    "direct_training_without_manifest_gate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="组装 internal multi-magnetization diagnostic pack。")
    parser.add_argument("--richer-manifest", type=Path, default=RICHER_MANIFEST)
    parser.add_argument("--mag-y-npz", type=Path, default=MAG_Y_NPZ)
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
        raise FileExistsError("拒绝覆盖既有文件；如需重跑请显式添加 --overwrite:\n" + "\n".join(str(path) for path in existing))


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
    for idx, (base, variant) in enumerate(
        zip(strings(x_arrays["base_group_id"]), strings(x_arrays["observation_variant"]), strict=True)
    ):
        index[(base, variant)] = idx
    return index


def complete_base_count(base_ids: list[str], variants: list[str]) -> int:
    by_base: dict[str, set[str]] = defaultdict(set)
    for base, variant in zip(base_ids, variants, strict=True):
        by_base[base].add(variant)
    return sum(1 for values in by_base.values() if {"M1_mag_y_5line_z0p008", "M2_mag_y_9line_z0p008"}.issubset(values))


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
            f"- magnetization_direction_names: {manifest['magnetization_direction_names']}",
            f"- observation_variants: {manifest['observation_variants']}",
            f"- paired_reference_variants: {manifest['paired_reference_variants']}",
            f"- nominal_source_je: {manifest['nominal_source_je']}",
            f"- orthogonal_source_je: {manifest['orthogonal_source_je']}",
            f"- source_je_changed: {str(manifest['source_je_changed']).lower()}",
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
            "- notes: Multi-magnetization diagnostic pack only. Generated NPZ/data files are not committed; not a baseline.",
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
    y_arrays = load_npz(args.mag_y_npz)
    x_index = build_x_index(x_arrays)

    y_n = int(len(y_arrays["sample_ids"]))
    delta_rows: list[np.ndarray] = []
    b_defect_rows: list[np.ndarray] = []
    b_no_rows: list[np.ndarray] = []
    magnetization_masks: list[np.ndarray] = []
    scan_masks: list[np.ndarray] = []
    x_sample_ids: list[str] = []
    y_sample_ids: list[str] = []
    mag_x_source_je: list[list[str]] = []
    mag_y_source_je: list[list[str]] = []

    for y_idx in range(y_n):
        base = str(y_arrays["base_group_id"][y_idx])
        x_variant = str(y_arrays["paired_reference_variant"][y_idx])
        key = (base, x_variant)
        if key not in x_index:
            raise KeyError(f"找不到配对 mag_x/reference row: base={base}; variant={x_variant}")
        x_idx = x_index[key]
        delta_rows.append(np.stack([x_arrays["delta_b"][x_idx], y_arrays["delta_b"][y_idx]], axis=1))
        b_defect_rows.append(np.stack([x_arrays["b_defect"][x_idx], y_arrays["b_defect"][y_idx]], axis=1))
        b_no_rows.append(np.stack([x_arrays["b_no_defect"][x_idx], y_arrays["b_no_defect"][y_idx]], axis=1))
        magnetization_masks.append(np.asarray([True, True], dtype=bool))
        scan_masks.append(np.stack([x_arrays["scan_line_mask"][x_idx], y_arrays["scan_line_mask"][y_idx]], axis=0).astype(bool))
        x_sample_ids.append(str(x_arrays["sample_ids"][x_idx]))
        y_sample_ids.append(str(y_arrays["sample_ids"][y_idx]))
        mag_x_source_je.append(json.loads(str(y_arrays["nominal_source_je_json"][y_idx])))
        mag_y_source_je.append(json.loads(str(y_arrays["orthogonal_source_je_json"][y_idx])))

    out: dict[str, np.ndarray] = {
        "dataset_id": np.asarray(DATASET_ID, dtype=object),
        "source_dataset_ids": np.asarray([RICHER_DATASET_ID, MAG_Y_DATASET_ID, SOURCE_DATASET_ID], dtype=object),
        "delta_b": np.stack(delta_rows, axis=0).astype(np.float32),
        "b_defect": np.stack(b_defect_rows, axis=0).astype(np.float32),
        "b_no_defect": np.stack(b_no_rows, axis=0).astype(np.float32),
        "magnetization_mask": np.stack(magnetization_masks, axis=0),
        "direction_mask": np.stack(magnetization_masks, axis=0),
        "scan_line_mask": np.stack(scan_masks, axis=0),
        "magnetization_direction_names": np.asarray(["mag_x", "mag_y"], dtype="<U16"),
        "direction_names": np.asarray(["mag_x", "mag_y"], dtype="<U16"),
        "scan_direction": np.asarray(["x_scan"] * y_n, dtype="<U32"),
        "axis_names": np.asarray(["Bx", "By", "Bz"], dtype="<U8"),
        "axis_expressions": np.asarray(["mf.Bx", "mf.By", "mf.Bz"], dtype="<U16"),
        "sample_ids": np.asarray(
            [
                f"{str(y_arrays['base_group_id'][i])}_{str(y_arrays['observation_variant'][i]).replace('M', 'multi_mag_M')}"
                for i in range(y_n)
            ],
            dtype="<U160",
        ),
        "base_group_id": y_arrays["base_group_id"],
        "base_sample_id": y_arrays["base_sample_id"],
        "mag_x_sample_id": np.asarray(x_sample_ids, dtype="<U128"),
        "mag_y_sample_id": np.asarray(y_sample_ids, dtype="<U128"),
        "mag_x_observation_variant": y_arrays["paired_reference_variant"],
        "mag_y_observation_variant": y_arrays["observation_variant"],
        "paired_reference_variant": y_arrays["paired_reference_variant"],
        "observation_variant": np.asarray(
            [f"multi_magnetization_{str(v).split('_')[0]}_z0p008" for v in y_arrays["observation_variant"]], dtype="<U80"
        ),
        "sensor_z_m": y_arrays["sensor_z_m"],
        "sensor_x": y_arrays["sensor_x"],
        "scan_line_y": np.stack([x_arrays["scan_line_y"][x_index[(str(y_arrays["base_group_id"][i]), str(y_arrays["paired_reference_variant"][i]))]] for i in range(y_n)], axis=0).astype(np.float32),
        "magnetization_source_je_json": np.asarray(
            [[json.dumps(mag_x_source_je[i], separators=(",", ":")), json.dumps(mag_y_source_je[i], separators=(",", ":"))] for i in range(y_n)],
            dtype="<U80",
        ),
        "nominal_source_je_json": y_arrays["nominal_source_je_json"],
        "orthogonal_source_je_json": y_arrays["orthogonal_source_je_json"],
        "source_je_changed": y_arrays["source_je_changed"],
        "mag_y_source_direction": y_arrays["source_direction"],
        "mag_y_J_direction": y_arrays["J_direction"],
        "mag_y_H_direction": y_arrays["H_direction"],
    }

    for key in [
        "source_split",
        "source_subset",
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
        "material",
        "specimen_geometry_json",
        "row_origin",
        "source_dataset_id_per_row",
    ]:
        if key in y_arrays:
            out[key] = y_arrays[key]

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **out)

    n = int(out["delta_b"].shape[0])
    variants = [str(v) for v in y_arrays["observation_variant"]]
    base_ids = [str(v) for v in y_arrays["base_group_id"]]
    base_count = len(set(base_ids))
    complete = complete_base_count(base_ids, variants)
    status = "diagnostic_pack_generated" if n == 60 and complete == 30 else "partial_diagnostic_pack_generated"
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_role": "internal_defect_multi_magnetization_diagnostic_pack",
        "stage": "23.4",
        "route": ROUTE,
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "npz_path": str(args.output_npz),
        "npz_sha256": sha256_file(args.output_npz),
        "manifest_path": str(args.manifest),
        "source_dataset_ids": [RICHER_DATASET_ID, MAG_Y_DATASET_ID, SOURCE_DATASET_ID],
        "source_richer_manifest": str(args.richer_manifest),
        "source_richer_npz": str(richer_npz),
        "source_mag_y_npz": str(args.mag_y_npz),
        "comsol_summary": str(Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\internal_multi_magnetization_pack_summary.txt")),
        "comsol_inventory": str(Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\inventory_internal_multi_magnetization_pack.csv")),
        "n_samples": n,
        "base_count": base_count,
        "complete_base_count": complete,
        "assembled_delta_shape": list(out["delta_b"].shape),
        "b_defect_shape": list(out["b_defect"].shape),
        "b_no_defect_shape": list(out["b_no_defect"].shape),
        "magnetization_direction_names": ["mag_x", "mag_y"],
        "magnetization_mask_shape": list(out["magnetization_mask"].shape),
        "scan_line_mask_shape": list(out["scan_line_mask"].shape),
        "scan_direction": "x_scan",
        "observation_variants": dict(Counter(strings(y_arrays["observation_variant"]))),
        "paired_reference_variants": dict(Counter(strings(y_arrays["paired_reference_variant"]))),
        "shape_counts": counts(y_arrays, "shape_type"),
        "burial_depth_counts": counts(y_arrays, "burial_depth_level"),
        "size_counts": counts(y_arrays, "size_level"),
        "aspect_counts": counts(y_arrays, "aspect_bin"),
        "nominal_source_je": mag_x_source_je[0] if mag_x_source_je else [],
        "orthogonal_source_je": mag_y_source_je[0] if mag_y_source_je else [],
        "source_je_changed": bool(np.asarray(y_arrays["source_je_changed"]).astype(bool).all()),
        "train_ready_candidate": False,
        "baseline_ready": False,
        "internal_surface_mixed": False,
        "surface_rbc_baseline_update": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "allowed_use": ALLOWED_USE,
        "forbidden_use": FORBIDDEN_USE,
        "validation_script": "scripts/validate_internal_multi_magnetization_pack.py",
        "assembly_script": "scripts/assemble_internal_multi_magnetization_pack.py",
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    update_registry(args.registry, manifest)

    checks = [
        {"check_name": "assembled_rows", "pass": n == 60, "observed": n, "expected": 60},
        {"check_name": "base_count", "pass": base_count == 30, "observed": base_count, "expected": 30},
        {"check_name": "complete_base_count", "pass": complete == 30, "observed": complete, "expected": 30},
        {"check_name": "assembled_delta_shape", "pass": list(out["delta_b"].shape) == [60, 3, 2, 9, 201], "observed": list(out["delta_b"].shape), "expected": [60, 3, 2, 9, 201]},
        {"check_name": "source_je_changed", "pass": manifest["source_je_changed"], "observed": manifest["source_je_changed"], "expected": True},
        {"check_name": "train_ready_candidate_false", "pass": not manifest["train_ready_candidate"], "observed": manifest["train_ready_candidate"], "expected": False},
        {"check_name": "baseline_ready_false", "pass": not manifest["baseline_ready"], "observed": manifest["baseline_ready"], "expected": False},
    ]
    write_csv(args.metrics, checks, ["check_name", "pass", "observed", "expected"])

    lines = [
        "23.4 internal multi-magnetization pack assembly summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"status: {status}",
        f"n_samples: {n}",
        f"base_count: {base_count}",
        f"complete_base_count: {complete}",
        f"assembled_delta_shape: {list(out['delta_b'].shape)}",
        "magnetization_direction_names: ['mag_x', 'mag_y']",
        f"nominal_source_je: {manifest['nominal_source_je']}",
        f"orthogonal_source_je: {manifest['orthogonal_source_je']}",
        f"source_je_changed: {str(manifest['source_je_changed']).lower()}",
        f"manifest_path: {args.manifest}",
        f"npz_path_ignored: {args.output_npz}",
        "train_ready_candidate: false",
        "baseline_ready: false",
        "current_baseline_updated: false",
        "",
        "结论：已将既有 nominal/mag_x reference 与新生成 mag_y rows 配对为 multi-magnetization diagnostic pack；NPZ 位于 ignored data 路径，不作为 baseline。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = [
        "23.4 internal multi-magnetization registry summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"registry_path: {args.registry}",
        f"manifest_path: {args.manifest}",
        f"status: {status}",
        "route: internal_buried_defect_multi_magnetization",
        "allowed_use: schema_validation, explicit_multi_magnetization_diagnostic",
        "forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate",
        "train_ready_candidate: false",
        "baseline_ready: false",
        "surface_rbc_baseline_update: false",
    ]
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
