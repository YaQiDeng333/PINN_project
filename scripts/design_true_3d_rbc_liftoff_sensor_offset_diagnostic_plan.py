#!/usr/bin/env python
"""Design the 20.90 true-3D RBC liftoff/sensor-offset diagnostic pack.

This is a planning script only. It reads the explicit v3_240 registry/manifest,
existing plan/mesh metadata, and the frozen baseline artifact manifest. It does
not read latest/newest NPZ paths, run COMSOL, train, or write data artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from load_true_3d_rbc_pilot_dataset import (
    ROOT,
    V3_240_DATASET_ID,
    gate_manifest,
    load_dataset,
    resolve_dataset,
)


ARTIFACT_MANIFEST = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
GAIN_METRICS = ROOT / "results/metrics/true_3d_rbc_gain_calibration_strategy_metrics.csv"
FORMAL_PROFILE = ROOT / "results/metrics/true_3d_rbc_formal_benchmark_20_77_profile_metrics.csv"

PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_sensor_offset_preflight_summary.txt"
PLAN_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_sensor_offset_plan_summary.txt"
PLAN_CSV = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_plan.csv"

SOURCE_PLAN_MESH = [
    (
        ROOT / "results/metrics/true_3d_rbc_pilot_pack_plan.csv",
        ROOT / "results/metrics/true_3d_rbc_pilot_watertight_mesh_metrics.csv",
        "v1_pilot",
    ),
    (
        ROOT / "results/metrics/true_3d_rbc_pilot_topup_plan.csv",
        ROOT / "results/metrics/true_3d_rbc_pilot_topup_watertight_mesh_metrics.csv",
        "v1_topup",
    ),
    (
        ROOT / "results/metrics/true_3d_rbc_dataset_120_topup_plan.csv",
        ROOT / "results/metrics/true_3d_rbc_dataset_120_topup_mesh_metrics.csv",
        "v2_120_topup",
    ),
    (
        ROOT / "results/metrics/true_3d_rbc_dataset_240_topup_plan.csv",
        ROOT / "results/metrics/true_3d_rbc_dataset_240_topup_mesh_metrics.csv",
        "v3_240_topup",
    ),
]

COMSOL_VARIANTS = [
    ("nominal", "nominal", 0.008, 0.0, 1.0),
    ("liftoff_z_0p006", "liftoff", 0.006, 0.0, 1.0),
    ("liftoff_z_0p010", "liftoff", 0.010, 0.0, 1.0),
    ("liftoff_z_0p012", "liftoff", 0.012, 0.0, 1.0),
    ("scan_offset_m0p0005", "scan_line_offset", 0.008, -0.0005, 1.0),
    ("scan_offset_p0p0005", "scan_line_offset", 0.008, 0.0005, 1.0),
    ("source_jscale_0p8", "source_amplitude", 0.008, 0.0, 0.8),
    ("source_jscale_1p2", "source_amplitude", 0.008, 0.0, 1.2),
]

POSTPROCESS_VARIANTS = [
    ("axis_misalignment_x_light", "axis_misalignment_postprocess", {"Bx": 1, "By": -1, "Bz": 0}),
    ("axis_misalignment_x_hard", "axis_misalignment_postprocess", {"Bx": 2, "By": -2, "Bz": 1}),
    ("axis_misalignment_x_reverse", "axis_misalignment_postprocess", {"Bx": -1, "By": 1, "Bz": 0}),
]

PLAN_FIELDS = [
    "diagnostic_row_id",
    "base_index",
    "base_sample_id",
    "source_sample_id",
    "row_kind",
    "requires_comsol",
    "variant_name",
    "factor_group",
    "sensor_z_m",
    "liftoff_delta_m",
    "scan_line_y_json",
    "scan_bundle_y_offset_m",
    "jscale",
    "source_variant_name",
    "misalignment_shifts_json",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "size_bin",
    "L_m",
    "W_m",
    "D_m",
    "wLD",
    "wWD",
    "wLW",
    "profile_pose_json",
    "rbc_params_json",
    "profile_depth_grid_shape_json",
    "profile_depth_grid_m_json",
    "profile_depth_map_xy_shape_json",
    "profile_depth_map_xy_m_json",
    "projected_mask_2d_shape_json",
    "projected_mask_2d_json",
    "projection_threshold_m",
    "mesh_path",
    "mesh_validation_pass",
    "source_plan_csv",
    "source_mesh_csv",
    "selection_reason",
    "geometry_params_json",
    "exact_piao_rbc",
    "rbc_style_approximation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.90 liftoff/sensor-offset diagnostic pack.")
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--preflight-summary", type=Path, default=PREFLIGHT_SUMMARY)
    parser.add_argument("--summary", type=Path, default=PLAN_SUMMARY)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def nearest_to_median(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    dims = np.asarray([[float(r["L_m"]), float(r["W_m"]), float(r["D_m"])] for r in candidates], dtype=float)
    median = np.median(dims, axis=0)
    scale = np.maximum(np.ptp(dims, axis=0), 1.0e-9)
    scores = np.sqrt(np.sum(((dims - median) / scale) ** 2, axis=1))
    return candidates[int(np.argmin(scores))]


def load_source_rows(sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for plan_path, mesh_path, source_name in SOURCE_PLAN_MESH:
        plan_rows = read_csv(plan_path)
        mesh_rows = {row.get("sample_id", ""): row for row in read_csv(mesh_path)}
        for plan in plan_rows:
            sample_id = plan.get("sample_id", "")
            if sample_id not in sample_ids or sample_id in merged:
                continue
            mesh = mesh_rows.get(sample_id)
            if mesh is None:
                continue
            mesh_candidate = mesh.get("export_path") or plan.get("temp_mesh_output_path", "")
            if not mesh_candidate:
                continue
            merged[sample_id] = {
                **plan,
                "_mesh": mesh,
                "_mesh_path": mesh_candidate,
                "_source_plan_csv": str(plan_path),
                "_source_mesh_csv": str(mesh_path),
                "_source_name": source_name,
                "_mesh_validation_pass": bool_text(mesh.get("mesh_validation_pass", "")),
                "_mesh_exists": Path(mesh_candidate).exists(),
            }
    return merged


def selected_profile_sentinels(source_by_id: dict[str, dict[str, Any]], used: set[str]) -> list[tuple[str, str]]:
    rows = [row for row in read_csv(FORMAL_PROFILE) if row.get("selected_seed", "").lower() == "true"]
    rows = [row for row in rows if row.get("sample_id") in source_by_id and row.get("sample_id") not in used]
    if not rows:
        return []
    sorted_best = sorted(rows, key=lambda r: float(r.get("profile_depth_rmse_m", "inf")))
    sorted_worst = sorted(rows, key=lambda r: float(r.get("profile_depth_rmse_m", "-inf")), reverse=True)
    picks: list[tuple[str, str]] = []
    for row, reason in [(sorted_best[0], "clean_profile_best_sentinel"), (sorted_worst[0], "clean_profile_worst_sentinel")]:
        sample_id = row["sample_id"]
        if sample_id in used or any(sample_id == existing for existing, _ in picks):
            continue
        picks.append((sample_id, reason))
    return picks


def select_base_rows(dataset: Any, source_by_id: dict[str, dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    sample_to_dataset = {str(sid): i for i, sid in enumerate(dataset.sample_ids)}
    candidates = []
    for sample_id, row in source_by_id.items():
        if not row.get("_mesh_validation_pass") or not row.get("_mesh_exists"):
            continue
        idx = sample_to_dataset[sample_id]
        enriched = dict(row)
        enriched["_dataset_index"] = idx
        candidates.append(enriched)
    selected: list[tuple[dict[str, Any], str]] = []
    used: set[str] = set()
    templates = ["sharp", "round", "boxy", "LD_dominant", "WD_dominant"]
    depths = ["shallow", "deep"]
    for template in templates:
        for depth in depths:
            pool = [r for r in candidates if r["curvature_template"] == template and r["depth_bin"] == depth and r["sample_id"] not in used]
            if not pool:
                pool = [r for r in candidates if r["curvature_template"] == template and r["sample_id"] not in used]
            if not pool:
                pool = [r for r in candidates if r["sample_id"] not in used]
            priority_pool: list[dict[str, Any]] = []
            for split in ("test", "val", "train"):
                priority_pool = [r for r in pool if r["split"] == split]
                if priority_pool:
                    break
            choice = nearest_to_median(priority_pool or pool)
            used.add(choice["sample_id"])
            selected.append((choice, f"template_depth_pair_{template}_{depth}_priority_{choice['split']}"))
    for sample_id, reason in selected_profile_sentinels(source_by_id, used):
        row = dict(source_by_id[sample_id])
        row["_dataset_index"] = sample_to_dataset[sample_id]
        selected.append((row, reason))
        used.add(sample_id)
        if len(selected) >= 12:
            break
    if len(selected) < 12:
        medium = [r for r in candidates if r["sample_id"] not in used and r["depth_bin"] == "medium"]
        while len(selected) < 12 and medium:
            choice = nearest_to_median(medium)
            selected.append((choice, f"medium_depth_median_backfill_{choice['split']}"))
            used.add(choice["sample_id"])
            medium = [r for r in medium if r["sample_id"] not in used]
    if len(selected) != 12:
        raise RuntimeError(f"expected 12 base geometries, selected {len(selected)}")
    return selected


def base_row_payload(row: dict[str, Any], selection_reason: str, base_index: int, variant: dict[str, Any]) -> dict[str, Any]:
    scan_line_y = [-0.001 + variant["scan_offset"], 0.0 + variant["scan_offset"], 0.001 + variant["scan_offset"]]
    diagnostic_row_id = f"diag20_90_b{base_index:02d}_{row['sample_id']}__{variant['variant_name']}"
    return {
        "diagnostic_row_id": diagnostic_row_id,
        "base_index": base_index,
        "base_sample_id": row["sample_id"],
        "source_sample_id": row.get("source_sample_id", row["sample_id"]) or row["sample_id"],
        "row_kind": variant["row_kind"],
        "requires_comsol": str(bool(variant["requires_comsol"])),
        "variant_name": variant["variant_name"],
        "factor_group": variant["factor_group"],
        "sensor_z_m": variant["sensor_z_m"],
        "liftoff_delta_m": variant["sensor_z_m"] - 0.008,
        "scan_line_y_json": json_dumps(scan_line_y),
        "scan_bundle_y_offset_m": variant["scan_offset"],
        "jscale": variant["jscale"],
        "source_variant_name": variant.get("source_variant_name", ""),
        "misalignment_shifts_json": json_dumps(variant.get("misalignment_shifts", {})),
        "split": row["split"],
        "curvature_template": row["curvature_template"],
        "depth_bin": row["depth_bin"],
        "aspect_bin": row["aspect_bin"],
        "size_bin": row["size_bin"],
        "L_m": row["L_m"],
        "W_m": row["W_m"],
        "D_m": row["D_m"],
        "wLD": row["wLD"],
        "wWD": row["wWD"],
        "wLW": row["wLW"],
        "profile_pose_json": row["profile_pose_json"],
        "rbc_params_json": row["rbc_params_json"],
        "profile_depth_grid_shape_json": row["profile_depth_grid_shape_json"],
        "profile_depth_grid_m_json": row["profile_depth_grid_m_json"],
        "profile_depth_map_xy_shape_json": row["profile_depth_map_xy_shape_json"],
        "profile_depth_map_xy_m_json": row["profile_depth_map_xy_m_json"],
        "projected_mask_2d_shape_json": row["projected_mask_2d_shape_json"],
        "projected_mask_2d_json": row["projected_mask_2d_json"],
        "projection_threshold_m": row.get("projection_threshold_m", ""),
        "mesh_path": row["_mesh_path"],
        "mesh_validation_pass": str(row["_mesh_validation_pass"]),
        "source_plan_csv": row["_source_plan_csv"],
        "source_mesh_csv": row["_source_mesh_csv"],
        "selection_reason": selection_reason,
        "geometry_params_json": row["geometry_params_json"],
        "exact_piao_rbc": "False",
        "rbc_style_approximation": "True",
    }


def build_plan_rows(selected: list[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_index, (row, reason) in enumerate(selected, start=1):
        for name, group, sensor_z, scan_offset, jscale in COMSOL_VARIANTS:
            rows.append(
                base_row_payload(
                    row,
                    reason,
                    base_index,
                    {
                        "variant_name": name,
                        "factor_group": group,
                        "sensor_z_m": sensor_z,
                        "scan_offset": scan_offset,
                        "jscale": jscale,
                        "row_kind": "comsol",
                        "requires_comsol": True,
                        "source_variant_name": "",
                    },
                )
            )
        for name, group, shifts in POSTPROCESS_VARIANTS:
            rows.append(
                base_row_payload(
                    row,
                    reason,
                    base_index,
                    {
                        "variant_name": name,
                        "factor_group": group,
                        "sensor_z_m": 0.008,
                        "scan_offset": 0.0,
                        "jscale": 1.0,
                        "row_kind": "postprocess",
                        "requires_comsol": False,
                        "source_variant_name": "nominal",
                        "misalignment_shifts": shifts,
                    },
                )
            )
    return rows


def write_preflight(path: Path, dataset: Any, checks: list[dict[str, Any]], source_count: int) -> None:
    failed = [row for row in checks if not row["pass"]]
    artifact = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8")) if ARTIFACT_MANIFEST.exists() else {}
    lines = [
        "20.90 true 3D RBC liftoff / sensor-offset diagnostic preflight",
        "",
        f"dataset_id: {dataset.dataset_id}",
        f"registry_manifest_gate: {'PASS' if not failed else 'FAIL'}",
        f"n_samples: {len(dataset.sample_ids)}",
        f"delta_b_shape: {list(dataset.delta_b.shape)}",
        f"split_counts: train={int((dataset.split == 'train').sum())}, val={int((dataset.split == 'val').sum())}, test={int((dataset.split == 'test').sum())}",
        f"artifact_manifest_exists: {ARTIFACT_MANIFEST.exists()}",
        f"checkpoint_path: {artifact.get('checkpoint_path', '')}",
        f"gain_metrics_exists: {GAIN_METRICS.exists()}",
        f"source_plan_mesh_rows_available: {source_count}",
        "subagent_method: liftoff/scan offset/source variation are the correct Layer-2 diagnostic factors; calibration is a caveat only.",
        "subagent_comsol: reuse imported_watertight_mesh_solid and 20.70 material/domain/solver protocol; parameterize sensor_z, scan_line_y, and Jscale.",
        "subagent_pinn: use v3_240 registry/manifest gate, 20.88a artifact manifest, and fixed per_axis_rms_train_stats calibration.",
        "subagent_experiment: fixed size 12 base geometries, 96 COMSOL rows, 36 postprocess misalignment rows, 132 evaluation rows.",
        "subagent_safety: do not submit data/NPZ/.mph/raw CSV/checkpoints/previews/notes/temp STL/CURRENT_BASELINE/baseline docs.",
        "stop_conditions: registry/manifest fail; artifact manifest missing; source COMSOL base script missing; plan cannot select 12 base geometries; COMSOL success <92/96; nominal replay drift.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, selected: list[tuple[dict[str, Any], str]], plan_rows: list[dict[str, Any]]) -> None:
    base_lines = [
        f"{idx:02d}. {row['sample_id']} split={row['split']} template={row['curvature_template']} depth={row['depth_bin']} reason={reason}"
        for idx, (row, reason) in enumerate(selected, start=1)
    ]
    lines = [
        "20.90 true 3D RBC liftoff / sensor-offset diagnostic plan",
        "",
        f"dataset_id: {V3_240_DATASET_ID}",
        "baseline_artifact: results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json",
        "calibration_protocol: per_axis_rms_train_stats from 20.89, diagnostic only, not baseline replacement",
        f"base_geometry_count: {len(selected)}",
        f"evaluation_rows: {len(plan_rows)}",
        f"comsol_rows: {sum(str(row['requires_comsol']).lower() == 'true' for row in plan_rows)}",
        f"postprocess_rows: {sum(str(row['requires_comsol']).lower() == 'false' for row in plan_rows)}",
        "COMSOL variants per base: nominal; liftoff z=0.006/0.010/0.012 m; scan bundle offset +/-0.0005 m; Jscale 0.8/1.2.",
        "Postprocess variants per base: (+1,-1,0), (+2,-2,+1), (-1,+1,0) sensor_x sample shifts for Bx/By/Bz.",
        "Boundary: diagnostic pack only; no training; no baseline update; generated data/NPZ/.mph/raw CSV/temp files remain uncommitted.",
        "",
        "Selected base geometries:",
        *base_lines,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    check_overwrite([args.preflight_summary, args.summary, args.plan_csv], args.overwrite)
    entry, manifest, npz_path = resolve_dataset(args.dataset_id)
    checks = gate_manifest(entry, manifest, npz_path, args.dataset_id)
    dataset = load_dataset(args.dataset_id)
    source_by_id = load_source_rows(set(map(str, dataset.sample_ids)))
    write_preflight(args.preflight_summary, dataset, checks, len(source_by_id))
    failed = [row for row in checks if not row["pass"]]
    if failed:
        raise RuntimeError(f"registry/manifest gate failed: {failed}")
    if not ARTIFACT_MANIFEST.exists():
        raise FileNotFoundError(ARTIFACT_MANIFEST)
    if not GAIN_METRICS.exists():
        raise FileNotFoundError(GAIN_METRICS)
    if len(source_by_id) < 12:
        raise RuntimeError(f"too few source rows with mesh metadata: {len(source_by_id)}")
    selected = select_base_rows(dataset, source_by_id)
    plan_rows = build_plan_rows(selected)
    if len(plan_rows) != 132:
        raise RuntimeError(f"expected 132 plan rows, got {len(plan_rows)}")
    write_csv(args.plan_csv, plan_rows, PLAN_FIELDS)
    write_summary(args.summary, selected, plan_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
