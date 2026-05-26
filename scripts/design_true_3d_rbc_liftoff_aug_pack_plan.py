#!/usr/bin/env python
"""Design the 20.91 true-3D RBC liftoff augmentation pack.

This is a planning script only. It reads the explicit v3_240 registry/manifest,
existing mesh metadata, 20.88a artifact manifest, and 20.90 liftoff metrics. It
does not run COMSOL, train, or write generated data/NPZ artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
LIFTOFF_ROBUSTNESS_METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_sensor_offset_robustness_metrics.csv"
LIFTOFF_DECISION_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_sensor_offset_route_decision_summary.txt"

PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_aug_pack_preflight_summary.txt"
PLAN_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_aug_pack_plan_summary.txt"
PLAN_CSV = ROOT / "results/metrics/true_3d_rbc_liftoff_aug_pack_plan.csv"

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

LIFTOFF_LEVELS_M = [0.006, 0.008, 0.010, 0.012]
TARGET_BASE_COUNT = 48
MINIMUM_BASE_COUNT = 32
WORST_SENTINEL_COUNT = 6
SPLIT_TARGETS = {"train": 32, "val": 8, "test": 8}
TEMPLATE_TARGETS = {"sharp": 10, "round": 10, "boxy": 10, "LD_dominant": 9, "WD_dominant": 9}
DEPTH_TARGETS = {"shallow": 16, "medium": 16, "deep": 16}
ASPECT_TARGETS = {"compact": 12, "balanced": 12, "wide": 12, "narrow": 12}

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
    "liftoff_sentinel_rank",
    "liftoff_sentinel_max_raw_degradation_pct",
    "geometry_params_json",
    "exact_piao_rbc",
    "rbc_style_approximation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.91 true-3D RBC liftoff augmentation pack.")
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--target-base-count", type=int, default=TARGET_BASE_COUNT)
    parser.add_argument("--minimum-base-count", type=int, default=MINIMUM_BASE_COUNT)
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


def load_liftoff_sentinel_scores() -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in read_csv(LIFTOFF_ROBUSTNESS_METRICS)
        if row.get("input_mode") == "raw" and row.get("factor_group") == "liftoff"
    ]
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("base_sample_id", "")
        try:
            degradation = float(row.get("profile_depth_rmse_degradation_pct_vs_nominal", "nan"))
        except ValueError:
            continue
        previous = best.get(sample_id)
        if previous is None or degradation > float(previous["max_raw_degradation_pct"]):
            best[sample_id] = {
                "sample_id": sample_id,
                "max_raw_degradation_pct": degradation,
                "worst_variant": row.get("variant_name", ""),
            }
    ranked = sorted(best.values(), key=lambda r: float(r["max_raw_degradation_pct"]), reverse=True)
    return {row["sample_id"]: {**row, "rank": idx} for idx, row in enumerate(ranked, start=1)}


def enrich_candidates(dataset: Any, source_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    sample_to_dataset = {str(sid): i for i, sid in enumerate(dataset.sample_ids)}
    candidates: list[dict[str, Any]] = []
    for sample_id, row in source_by_id.items():
        if not row.get("_mesh_validation_pass") or not row.get("_mesh_exists"):
            continue
        enriched = dict(row)
        enriched["_dataset_index"] = sample_to_dataset[sample_id]
        candidates.append(enriched)
    return candidates


def pick_from_pool(pool: list[dict[str, Any]], selected_ids: set[str], split_counts: Counter[str]) -> dict[str, Any]:
    available = [row for row in pool if row["sample_id"] not in selected_ids]
    if not available:
        raise RuntimeError("empty candidate pool")
    under_split = [row for row in available if split_counts[row["split"]] < SPLIT_TARGETS.get(row["split"], 0)]
    return nearest_to_median(under_split or available)


def greedy_fill_score(row: dict[str, Any], counts: dict[str, Counter[str]]) -> tuple[int, int, int, int, float]:
    template_need = max(0, TEMPLATE_TARGETS.get(row["curvature_template"], 0) - counts["template"][row["curvature_template"]])
    depth_need = max(0, DEPTH_TARGETS.get(row["depth_bin"], 0) - counts["depth"][row["depth_bin"]])
    aspect_need = max(0, ASPECT_TARGETS.get(row["aspect_bin"], 0) - counts["aspect"][row["aspect_bin"]])
    split_need = max(0, SPLIT_TARGETS.get(row["split"], 0) - counts["split"][row["split"]])
    centrality = -float(row["L_m"]) - float(row["W_m"]) - float(row["D_m"])
    return (template_need, depth_need, aspect_need, split_need, centrality)


def add_selection(
    selected: list[tuple[dict[str, Any], str]],
    selected_ids: set[str],
    counts: dict[str, Counter[str]],
    row: dict[str, Any],
    reason: str,
) -> None:
    if row["sample_id"] in selected_ids:
        return
    selected.append((row, reason))
    selected_ids.add(row["sample_id"])
    counts["template"][row["curvature_template"]] += 1
    counts["depth"][row["depth_bin"]] += 1
    counts["aspect"][row["aspect_bin"]] += 1
    counts["split"][row["split"]] += 1


def select_base_rows(candidates: list[dict[str, Any]], target_base_count: int) -> list[tuple[dict[str, Any], str]]:
    if len(candidates) < target_base_count:
        raise RuntimeError(f"too few candidates with watertight mesh metadata: {len(candidates)} < {target_base_count}")
    sentinel_scores = load_liftoff_sentinel_scores()
    selected: list[tuple[dict[str, Any], str]] = []
    selected_ids: set[str] = set()
    counts: dict[str, Counter[str]] = {
        "template": Counter(),
        "depth": Counter(),
        "aspect": Counter(),
        "split": Counter(),
    }

    sentinel_pool = [row for row in candidates if row["sample_id"] in sentinel_scores]
    sentinel_pool.sort(key=lambda row: float(sentinel_scores[row["sample_id"]]["max_raw_degradation_pct"]), reverse=True)
    for row in sentinel_pool[:WORST_SENTINEL_COUNT]:
        score = sentinel_scores[row["sample_id"]]
        reason = f"20_90_worst_liftoff_rank_{score['rank']}_{score['worst_variant']}"
        row["_liftoff_sentinel_rank"] = score["rank"]
        row["_liftoff_sentinel_max_raw_degradation_pct"] = score["max_raw_degradation_pct"]
        add_selection(selected, selected_ids, counts, row, reason)

    for template in TEMPLATE_TARGETS:
        for depth in DEPTH_TARGETS:
            if any(row["curvature_template"] == template and row["depth_bin"] == depth for row, _ in selected):
                continue
            pool = [row for row in candidates if row["curvature_template"] == template and row["depth_bin"] == depth]
            if pool:
                add_selection(selected, selected_ids, counts, pick_from_pool(pool, selected_ids, counts["split"]), f"coverage_template_depth_{template}_{depth}")

    for aspect in ASPECT_TARGETS:
        for depth in DEPTH_TARGETS:
            if any(row["aspect_bin"] == aspect and row["depth_bin"] == depth for row, _ in selected):
                continue
            pool = [row for row in candidates if row["aspect_bin"] == aspect and row["depth_bin"] == depth]
            if pool:
                add_selection(selected, selected_ids, counts, pick_from_pool(pool, selected_ids, counts["split"]), f"coverage_aspect_depth_{aspect}_{depth}")

    while len(selected) < target_base_count:
        available = [row for row in candidates if row["sample_id"] not in selected_ids]
        if not available:
            break
        available.sort(key=lambda row: greedy_fill_score(row, counts), reverse=True)
        row = available[0]
        add_selection(selected, selected_ids, counts, row, "balanced_fill_template_depth_aspect_split")

    return selected


def base_row_payload(row: dict[str, Any], selection_reason: str, base_index: int, sensor_z_m: float) -> dict[str, Any]:
    variant_name = f"liftoff_z_{sensor_z_m:.3f}".replace(".", "p")
    diagnostic_row_id = f"liftoff20_91_b{base_index:02d}_{row['sample_id']}__{variant_name}"
    scan_line_y = [-0.001, 0.0, 0.001]
    return {
        "diagnostic_row_id": diagnostic_row_id,
        "base_index": base_index,
        "base_sample_id": row["sample_id"],
        "source_sample_id": row.get("source_sample_id", row["sample_id"]) or row["sample_id"],
        "row_kind": "comsol",
        "requires_comsol": "True",
        "variant_name": variant_name,
        "factor_group": "liftoff_aug",
        "sensor_z_m": sensor_z_m,
        "liftoff_delta_m": sensor_z_m - 0.008,
        "scan_line_y_json": json_dumps(scan_line_y),
        "scan_bundle_y_offset_m": 0.0,
        "jscale": 1.0,
        "source_variant_name": "nominal_0p008" if abs(sensor_z_m - 0.008) > 1.0e-12 else "",
        "misalignment_shifts_json": json_dumps({}),
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
        "liftoff_sentinel_rank": row.get("_liftoff_sentinel_rank", ""),
        "liftoff_sentinel_max_raw_degradation_pct": row.get("_liftoff_sentinel_max_raw_degradation_pct", ""),
        "geometry_params_json": row["geometry_params_json"],
        "exact_piao_rbc": "False",
        "rbc_style_approximation": "True",
    }


def build_plan_rows(selected: list[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_index, (row, reason) in enumerate(selected, start=1):
        for sensor_z_m in LIFTOFF_LEVELS_M:
            rows.append(base_row_payload(row, reason, base_index, sensor_z_m))
    return rows


def coverage_lines(selected: list[tuple[dict[str, Any], str]]) -> list[str]:
    rows = [row for row, _ in selected]
    counters = {
        "split": Counter(row["split"] for row in rows),
        "curvature_template": Counter(row["curvature_template"] for row in rows),
        "depth_bin": Counter(row["depth_bin"] for row in rows),
        "aspect_bin": Counter(row["aspect_bin"] for row in rows),
    }
    return [f"{name}: {dict(counter)}" for name, counter in counters.items()]


def write_preflight(path: Path, dataset: Any, checks: list[dict[str, Any]], source_count: int) -> None:
    failed = [row for row in checks if not row["pass"]]
    lines = [
        "20.91 true 3D RBC liftoff augmentation pack preflight",
        "",
        f"dataset_id: {dataset.dataset_id}",
        f"registry_path: {ROOT / 'COMSOL_DATA_REGISTRY.md'}",
        f"manifest_path: {dataset.manifest.get('manifest_path', '')}",
        f"registry_manifest_gate: {'PASS' if not failed else 'FAIL'}",
        f"n_samples: {len(dataset.sample_ids)}",
        f"delta_b_shape: {list(dataset.delta_b.shape)}",
        f"split_counts: train={int((dataset.split == 'train').sum())}, val={int((dataset.split == 'val').sum())}, test={int((dataset.split == 'test').sum())}",
        f"artifact_manifest_exists: {ARTIFACT_MANIFEST.exists()}",
        f"gain_metrics_exists: {GAIN_METRICS.exists()}",
        f"liftoff_20_90_metrics_exists: {LIFTOFF_ROBUSTNESS_METRICS.exists()}",
        f"liftoff_20_90_decision_exists: {LIFTOFF_DECISION_SUMMARY.exists()}",
        f"source_plan_mesh_rows_available: {source_count}",
        "method_preflight: 20.90 identified liftoff as the main unresolved blocker; source/amplitude calibration remains diagnostic only.",
        "comsol_preflight: reuse imported_watertight_mesh_solid and 20.70 material/domain/solver protocol; only sensor_z_m varies.",
        "pinn_preflight: use v3_240 registry/manifest and 20.88a artifact manifest; no latest/newest scan.",
        "experiment_preflight: target 48 base geometries x 4 liftoff levels = 192 COMSOL rows; minimum fallback is 32 x 4 = 128 rows.",
        "safety_preflight: no COMSOL execution in this stage; no data/NPZ/checkpoint/preview/notes/temp STL/CURRENT_BASELINE/baseline docs submission.",
        "stop_conditions: registry/manifest fail; artifact manifest missing; 20.90 liftoff metrics missing; fewer than 32 source geometries with valid mesh metadata.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, selected: list[tuple[dict[str, Any], str]], plan_rows: list[dict[str, Any]], target_met: bool) -> None:
    base_lines = [
        f"{idx:02d}. {row['sample_id']} split={row['split']} template={row['curvature_template']} depth={row['depth_bin']} aspect={row['aspect_bin']} reason={reason}"
        for idx, (row, reason) in enumerate(selected, start=1)
    ]
    lines = [
        "20.91 true 3D RBC liftoff augmentation pack plan",
        "",
        f"dataset_id: {V3_240_DATASET_ID}",
        f"registry_path: {ROOT / 'COMSOL_DATA_REGISTRY.md'}",
        f"manifest_path: {selected[0][0].get('_manifest_path', '') if selected else ''}",
        "baseline_artifact: results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json",
        "calibration_protocol: per_axis_rms_train_stats from 20.89, diagnostic caveat only, not baseline replacement",
        f"target_base_count: {TARGET_BASE_COUNT}",
        f"selected_base_count: {len(selected)}",
        f"target_met: {target_met}",
        f"minimum_base_count: {MINIMUM_BASE_COUNT}",
        f"evaluation_rows: {len(plan_rows)}",
        f"liftoff_levels_m: {LIFTOFF_LEVELS_M}",
        "nominal_liftoff_m: 0.008",
        "COMSOL rows per base: 4",
        "planned_COMSOL_rows: " + str(len(plan_rows)),
        "requires_COMSOL_for_generation: true",
        "training_run: false",
        "baseline_update: false",
        "CURRENT_BASELINE_update: false",
        "sensor_z_condition_recommendation: plan 20.92 ablation comparing unconditioned model vs scalar sensor_z_m conditioned model.",
        "internal_defect_feasibility: deferred until liftoff robustness is characterized.",
        "",
        "Coverage:",
        *coverage_lines(selected),
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
    for required_path in [ARTIFACT_MANIFEST, GAIN_METRICS, LIFTOFF_ROBUSTNESS_METRICS, LIFTOFF_DECISION_SUMMARY]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    candidates = enrich_candidates(dataset, source_by_id)
    for candidate in candidates:
        candidate["_manifest_path"] = dataset.manifest.get("manifest_path", "")
    if len(candidates) < args.minimum_base_count:
        raise RuntimeError(f"fewer than minimum viable candidates: {len(candidates)} < {args.minimum_base_count}")
    target_count = args.target_base_count if len(candidates) >= args.target_base_count else args.minimum_base_count
    selected = select_base_rows(candidates, target_count)
    if len(selected) < args.minimum_base_count:
        raise RuntimeError(f"selected fewer than minimum base count: {len(selected)}")
    plan_rows = build_plan_rows(selected)
    expected_rows = len(selected) * len(LIFTOFF_LEVELS_M)
    if len(plan_rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} plan rows, got {len(plan_rows)}")
    write_csv(args.plan_csv, plan_rows, PLAN_FIELDS)
    write_summary(args.summary, selected, plan_rows, len(selected) == args.target_base_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
