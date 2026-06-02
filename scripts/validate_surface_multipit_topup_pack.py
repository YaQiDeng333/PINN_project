#!/usr/bin/env python
"""Validate the 25.9b surface multi-pit COMSOL top-up pack."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMSOL_ROOT = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")

DATASET_ID = "comsol_surface_multipit_topup_pack_v1"
ROUTE = "surface_multipit_component_set"
SCHEMA_VERSION = "surface_multipit_component_set_topup_v1"
COMPONENT_SCHEMA_VERSION = "surface_multipit_component_schema_v1"
TOPOLOGY_SCHEMA_VERSION = "surface_multipit_topology_schema_v1"
TARGET_N = 96
FALLBACK_N = 60
TARGET_SPLIT = {"train": 63, "val": 17, "test": 16}
TARGET_COMPONENT_COUNTS = {"2": 84, "3": 12}
TARGET_SEPARATION = {"separated": 24, "close": 24, "touching": 24, "partially_overlapping": 24}
TARGET_TOPOLOGY = {"disconnected": 48, "touching_boundary": 24, "partially_overlapping": 24}
TARGET_ORIENTATION = {"aligned_x": 32, "aligned_y": 32, "diagonal": 32}
TARGET_RELATIVE_DEPTH = {"similar_depth": 48, "one_deep_one_shallow": 48}
TARGET_PRIMITIVE_MIX = {"smooth-smooth": 24, "flat-flat": 24, "smooth-flat": 24, "asymmetric_mix": 24}

DEFAULT_NPZ = ROOT / "data/comsol_mfl/generated/surface_multipit_topup_pack_v1/surface_multipit_topup_pack_v1.npz"
DEFAULT_INVENTORY = COMSOL_ROOT / "results/inventory_surface_multipit_topup_pack.csv"
DEFAULT_COMSOL_SUMMARY = COMSOL_ROOT / "results/surface_multipit_topup_pack_summary.txt"
DEFAULT_PREFLIGHT = ROOT / "results/summaries/surface_multipit_topup_pack_preflight_summary.txt"
DEFAULT_SUMMARY = ROOT / "results/summaries/surface_multipit_topup_pack_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/surface_multipit_topup_pack_validation_metrics.csv"
DEFAULT_GROUPS = ROOT / "results/metrics/surface_multipit_topup_pack_group_summary.csv"
DEFAULT_MISSING = ROOT / "results/metrics/surface_multipit_topup_pack_missing_fields.csv"

METRIC_FIELDS = ["check_name", "pass", "observed", "required", "notes"]
GROUP_FIELDS = ["group_field", "group_value", "split", "count", "notes"]
MISSING_FIELDS = ["sample_id", "field_name", "severity", "notes"]
FORBIDDEN_STAGE_PREFIXES = ("data/", "checkpoints/", "notes/", "results/previews/")
FORBIDDEN_STAGE_SUFFIXES = (".npz", ".mph", ".png", ".jpg", ".jpeg", ".stl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 25.9b surface multi-pit top-up pack.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--comsol-summary", type=Path, default=DEFAULT_COMSOL_SUMMARY)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--missing-fields", type=Path, default=DEFAULT_MISSING)
    parser.add_argument("--preflight-only", action="store_true")
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


def git_value(cwd: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip('"')


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def forbidden_staged() -> list[str]:
    staged = [normalize_path(item) for item in git_value(ROOT, ["diff", "--cached", "--name-only"]).splitlines() if item.strip()]
    bad: list[str] = []
    for path in staged:
        lower = path.lower()
        if path in {"CURRENT_BASELINE.md", "scripts/visualize_current_baseline.py"}:
            bad.append(path)
        if path.startswith(FORBIDDEN_STAGE_PREFIXES) or lower.endswith(FORBIDDEN_STAGE_SUFFIXES):
            bad.append(path)
    return sorted(set(bad))


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as pack:
        return {name: pack[name].copy() for name in pack.files}


def as_list(values: np.ndarray) -> list[str]:
    return [str(item) for item in values.tolist()]


def count_strings(values: np.ndarray | list[str]) -> dict[str, int]:
    if isinstance(values, np.ndarray):
        values = as_list(values)
    return dict(Counter(str(item) for item in values))


def target_match(observed: dict[str, int], required: dict[str, int]) -> bool:
    return all(int(observed.get(key, 0)) == int(value) for key, value in required.items())


def write_preflight(args: argparse.Namespace) -> None:
    required = [
        ROOT / "results/metrics/surface_multipit_component_branch_decision_matrix.csv",
        ROOT / "results/metrics/surface_multipit_dataset_topup_plan.csv",
        ROOT / "results/metrics/surface_multipit_comsol_generation_feasibility.csv",
        ROOT / "results/metrics/surface_multipit_component_set_representation_matrix.csv",
        ROOT / "results/metrics/surface_shape_extension_oracle_vs_baseline_matrix.csv",
        ROOT / "results/summaries/surface_forward_consistency_refinement_route_decision_summary.txt",
        ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json",
        COMSOL_ROOT / "scripts/generate_mfl_surface_multipit_topup_pack.py",
    ]
    pinn_status = git_value(ROOT, ["status", "--short"])
    comsol_status = git_value(COMSOL_ROOT, ["status", "--short"])
    lines = [
        "surface multi-pit top-up pack preflight summary",
        "stage: 25.9b",
        "",
        "scope: surface multi-pit component-set COMSOL top-up generation only; no training, no model gate, no CURRENT_BASELINE update.",
        "",
        "required_inputs:",
        *[f"- {path}: exists={bool_text(path.exists())}" for path in required],
        "",
        "route_boundary:",
        "- 25.3/25.8 classify multi-pit as a representation failure, not a six-parameter RBC refinement target.",
        "- 25.9 defines fixed_K_component_set with K=3 and explicit per-component labels.",
        "- COMSOL generation route is multi-component surface-connected Boolean subtract.",
        "",
        "forbidden_output_policy:",
        "- Generated data/NPZ/MPH/raw solver files/checkpoints/preview PNG/notes/temp STL are not commit artifacts.",
        "- Registry/manifest/summary/metrics/scripts are the commit-facing outputs.",
        "",
        "git_status_PINN_project:",
        pinn_status or "clean",
        "",
        "git_status_COMSOL_Multiphysics_MCP:",
        comsol_status or "clean",
        "",
        "preflight_decision: generation may proceed only if the 25.9 plan, label audit, top-up plan, COMSOL feasibility, and 25.2 manifest are present.",
    ]
    args.preflight_summary.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_pack(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not args.npz.exists():
        raise FileNotFoundError(args.npz)
    if not args.inventory.exists():
        raise FileNotFoundError(args.inventory)
    pack = load_npz(args.npz)
    inventory = read_csv(args.inventory)
    sample_ids = as_list(pack["sample_ids"])
    n = len(sample_ids)
    split = as_list(pack["split"])
    shape_type = as_list(pack["shape_type"])
    representation_target = as_list(pack["representation_target"])
    component_count = np.asarray(pack["component_count"], dtype=np.int64)
    component_exists = np.asarray(pack["component_exists"], dtype=bool)
    component_center = np.asarray(pack["component_center_xy_m"], dtype=np.float64)
    component_lwd = np.asarray(pack["component_lwd_m"], dtype=np.float64)
    component_rot = np.asarray(pack["component_rotation_angle"], dtype=np.float64)
    component_masks = np.asarray(pack["component_projected_masks_2d"], dtype=np.uint8)
    component_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float64)
    union_masks = np.asarray(pack["projected_mask_2d"], dtype=np.uint8)
    union_depths = np.asarray(pack["depth_grid_m"], dtype=np.float64)
    delta_b = np.asarray(pack["delta_b"], dtype=np.float64)
    b_defect = np.asarray(pack["b_defect"], dtype=np.float64)
    b_no_defect = np.asarray(pack["b_no_defect"], dtype=np.float64)
    separation = as_list(pack["separation_type"])
    topology_relation = as_list(pack["topology_relation"])
    relative_depth = as_list(pack["relative_depth_type"])
    orientation = as_list(pack["orientation_type"])
    primitive_mix = as_list(pack["primitive_mix"])
    topology_type = as_list(pack["topology_type"])
    delta_error = float(np.max(np.abs(delta_b - (b_defect - b_no_defect)))) if n else float("nan")
    split_counts = count_strings(split)
    component_counts = count_strings([str(int(v)) for v in component_count.tolist()])
    separation_counts = count_strings(separation)
    topology_counts = count_strings(topology_relation)
    orientation_counts = count_strings(orientation)
    relative_depth_counts = count_strings(relative_depth)
    primitive_mix_counts = count_strings(primitive_mix)
    successful_inventory = [row for row in inventory if str(row.get("status", "")).strip().lower() == "pass"]
    boolean_mesh_solve = all(
        str(row.get(key, "")).strip().lower() == "true"
        for row in successful_inventory
        for key in ["boolean_subtract_success", "mesh_precheck_success", "solve_success"]
    )
    active_mask_counts = [
        int(component_masks[i, j].sum())
        for i in range(n)
        for j in range(component_masks.shape[1])
        if component_exists[i, j]
    ]
    active_depth_max = [
        float(np.max(component_depths[i, j]))
        for i in range(n)
        for j in range(component_depths.shape[1])
        if component_exists[i, j]
    ]
    missing_rows: list[dict[str, Any]] = []
    for i, sample_id in enumerate(sample_ids):
        if int(component_count[i]) != int(component_exists[i].sum()):
            missing_rows.append({"sample_id": sample_id, "field_name": "component_exists", "severity": "blocking", "notes": "component_count does not match active slots"})
        for slot in range(component_masks.shape[1]):
            if not component_exists[i, slot]:
                continue
            if not np.isfinite(component_center[i, slot]).all():
                missing_rows.append({"sample_id": sample_id, "field_name": f"component_{slot}_center_xy_m", "severity": "blocking", "notes": "non-finite center"})
            if not np.isfinite(component_lwd[i, slot]).all() or float(np.min(component_lwd[i, slot])) <= 0.0:
                missing_rows.append({"sample_id": sample_id, "field_name": f"component_{slot}_lwd_m", "severity": "blocking", "notes": "invalid L/W/D"})
            if not np.isfinite(component_rot[i, slot]):
                missing_rows.append({"sample_id": sample_id, "field_name": f"component_{slot}_rotation_angle", "severity": "blocking", "notes": "non-finite rotation"})
            if int(component_masks[i, slot].sum()) <= 0:
                missing_rows.append({"sample_id": sample_id, "field_name": f"component_{slot}_projected_mask_2d", "severity": "blocking", "notes": "empty component mask"})
            if not np.isfinite(component_depths[i, slot]).all() or float(np.max(component_depths[i, slot])) <= 0.0:
                missing_rows.append({"sample_id": sample_id, "field_name": f"component_{slot}_depth_grid_m", "severity": "blocking", "notes": "invalid component depth grid"})
    staged_bad = forbidden_staged()
    checks = [
        ("topup_npz_exists", args.npz.exists(), str(args.npz), "generated NPZ must remain uncommitted"),
        ("target_N", n == TARGET_N, n, TARGET_N),
        ("fallback_N", n >= FALLBACK_N, n, f">={FALLBACK_N}"),
        ("split_63_17_16", split_counts == TARGET_SPLIT, split_counts, TARGET_SPLIT),
        ("component_count_84_12", target_match(component_counts, TARGET_COMPONENT_COUNTS), component_counts, TARGET_COMPONENT_COUNTS),
        ("separation_coverage", target_match(separation_counts, TARGET_SEPARATION), separation_counts, TARGET_SEPARATION),
        ("topology_coverage", target_match(topology_counts, TARGET_TOPOLOGY), topology_counts, TARGET_TOPOLOGY),
        ("orientation_coverage", target_match(orientation_counts, TARGET_ORIENTATION), orientation_counts, TARGET_ORIENTATION),
        ("relative_depth_coverage", target_match(relative_depth_counts, TARGET_RELATIVE_DEPTH), relative_depth_counts, TARGET_RELATIVE_DEPTH),
        ("primitive_mix_coverage", target_match(primitive_mix_counts, TARGET_PRIMITIVE_MIX), primitive_mix_counts, TARGET_PRIMITIVE_MIX),
        ("all_samples_component_set", set(representation_target) == {"component_set"}, sorted(set(representation_target)), "component_set only"),
        ("all_samples_multi_component", set(topology_type) == {"multi_component"}, sorted(set(topology_type)), "multi_component only"),
        ("finite_Bx_By_Bz", np.isfinite(delta_b).all() and np.isfinite(b_defect).all() and np.isfinite(b_no_defect).all(), str(delta_b.shape), "finite Bx/By/Bz arrays"),
        ("delta_b_consistency", delta_error <= 1e-12, f"{delta_error:.3e}", "<=1e-12"),
        ("union_masks_nonempty", bool(np.all(union_masks.reshape(n, -1).sum(axis=1) > 0)), int(union_masks.sum()), "each sample has projected mask"),
        ("union_depth_valid", np.isfinite(union_depths).all() and bool(np.all(union_depths.reshape(n, -1).max(axis=1) > 0.0)), str(union_depths.shape), "finite positive depth grids"),
        ("component_masks_nonempty", bool(active_mask_counts) and min(active_mask_counts) > 0, min(active_mask_counts) if active_mask_counts else 0, "each active component has mask"),
        ("component_depths_valid", bool(active_depth_max) and min(active_depth_max) > 0.0, min(active_depth_max) if active_depth_max else 0.0, "each active component has depth grid"),
        ("component_params_complete", not missing_rows, len(missing_rows), "0 blocking missing fields"),
        ("inventory_success_count", len(successful_inventory) == TARGET_N, len(successful_inventory), TARGET_N),
        ("boolean_mesh_solve_pass", boolean_mesh_solve, boolean_mesh_solve, "all pass rows report boolean/mesh/solve true"),
        ("no_duplicate_sample_ids", len(set(sample_ids)) == n, len(set(sample_ids)), n),
        ("no_forbidden_staged_artifacts", not staged_bad, "|".join(staged_bad), "no data/NPZ/CURRENT_BASELINE/preview/checkpoint staged"),
    ]
    metric_rows = [
        {"check_name": name, "pass": bool_text(bool(passed)), "observed": json.dumps(observed, sort_keys=True) if isinstance(observed, dict) else observed, "required": json.dumps(required, sort_keys=True) if isinstance(required, dict) else required, "notes": notes}
        for name, passed, observed, required, *maybe_notes in checks
        for notes in [maybe_notes[0] if maybe_notes else ""]
    ]
    group_rows: list[dict[str, Any]] = []
    dimensions = {
        "split": split,
        "shape_type": shape_type,
        "component_count": [str(int(v)) for v in component_count.tolist()],
        "separation_type": separation,
        "topology_relation": topology_relation,
        "relative_depth_type": relative_depth,
        "orientation_type": orientation,
        "primitive_mix": primitive_mix,
    }
    for field, values in dimensions.items():
        for (value, split_name), count in Counter(zip(values, split)).items():
            group_rows.append({"group_field": field, "group_value": value, "split": split_name, "count": count, "notes": "top-up pack"})
    context = {
        "dataset_id": DATASET_ID,
        "n": n,
        "split_counts": split_counts,
        "component_counts": component_counts,
        "separation_counts": separation_counts,
        "topology_counts": topology_counts,
        "relative_depth_counts": relative_depth_counts,
        "orientation_counts": orientation_counts,
        "primitive_mix_counts": primitive_mix_counts,
        "delta_error": delta_error,
        "missing_count": len(missing_rows),
        "inventory_success_count": len(successful_inventory),
        "validation_pass": all(row["pass"] == "true" for row in metric_rows),
        "npz_path": str(args.npz),
        "inventory_path": str(args.inventory),
        "comsol_summary_path": str(args.comsol_summary),
    }
    return metric_rows, group_rows, missing_rows, context


def write_summary(args: argparse.Namespace, context: dict[str, Any]) -> None:
    lines = [
        "surface multi-pit top-up pack validation summary",
        "stage: 25.9b",
        "",
        f"dataset_id: {context['dataset_id']}",
        f"route: {ROUTE}",
        f"schema_version: {SCHEMA_VERSION}",
        f"component_schema_version: {COMPONENT_SCHEMA_VERSION}",
        f"topology_schema_version: {TOPOLOGY_SCHEMA_VERSION}",
        f"planned_topup_N: {TARGET_N}",
        f"fallback_topup_N: {FALLBACK_N}",
        f"successful_topup_N: {context['n']}",
        f"split_counts: {context['split_counts']}",
        f"component_count_counts: {context['component_counts']}",
        f"separation_counts: {context['separation_counts']}",
        f"topology_counts: {context['topology_counts']}",
        f"relative_depth_counts: {context['relative_depth_counts']}",
        f"orientation_counts: {context['orientation_counts']}",
        f"primitive_mix_counts: {context['primitive_mix_counts']}",
        f"inventory_success_count: {context['inventory_success_count']}",
        f"delta_b_max_abs_error: {context['delta_error']:.3e}",
        f"component_label_missing_count: {context['missing_count']}",
        f"validation_pass: {context['validation_pass']}",
        "",
        "boundary:",
        "- This validates generated component-set labels only; it does not train or update a baseline.",
        "- Generated NPZ/data remain ignored artifacts and must not be committed.",
        "- Multi-pit remains excluded from six-parameter RBC refinement success credit.",
        "",
        f"generated_npz: {context['npz_path']}",
        f"comsol_inventory: {context['inventory_path']}",
        f"comsol_summary: {context['comsol_summary_path']}",
        f"metrics_csv: {args.metrics}",
        f"group_summary_csv: {args.group_summary}",
        f"missing_fields_csv: {args.missing_fields}",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    write_preflight(args)
    if args.preflight_only:
        return 0
    metrics, groups, missing, context = validate_pack(args)
    write_csv(args.metrics, metrics, METRIC_FIELDS)
    write_csv(args.group_summary, groups, GROUP_FIELDS)
    write_csv(args.missing_fields, missing, MISSING_FIELDS)
    write_summary(args, context)
    return 0 if context["validation_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
