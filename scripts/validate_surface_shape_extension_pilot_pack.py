#!/usr/bin/env python
"""Validate and register the 25.2 surface shape-extension pilot pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMSOL_ROOT = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")

DATASET_ID = "comsol_surface_shape_extension_pilot_v1"
ROUTE = "surface_shape_extension_non_rbc"
SCHEMA_VERSION = "surface_shape_extension_v1"
TAXONOMY_VERSION = "surface_shape_extension_taxonomy_v1"
LABEL_SCHEMA_VERSION = "surface_shape_extension_label_schema_v1"
TARGET_N = 120
FALLBACK_N = 84
TARGET_SPLIT = {"train": 72, "val": 24, "test": 24}
TARGET_SHAPE_COUNTS = {
    "rbc_like_smooth_pit": 24,
    "flat_bottom_pit": 16,
    "sharp_wall_boxy_corrosion": 16,
    "asymmetric_corrosion": 16,
    "elongated_crack_like_surface_defect": 16,
    "multi_pit_two_component_surface_defect": 16,
    "irregular_corrosion_non_rbc": 16,
}
NON_RBC_SHAPES = {shape for shape in TARGET_SHAPE_COUNTS if shape != "rbc_like_smooth_pit"}
FORBIDDEN_STAGE_PREFIXES = (
    "data/",
    "checkpoints/",
    "results/previews/",
    "notes/",
)
FORBIDDEN_STAGE_SUFFIXES = (".npz", ".mph", ".png", ".jpg", ".jpeg", ".stl")

DEFAULT_NPZ = ROOT / "data/comsol_mfl/prepared/experimental/surface_shape_extension/comsol_surface_shape_extension_pilot_v1.npz"
DEFAULT_INVENTORY = COMSOL_ROOT / "results/inventory_surface_shape_extension_pilot_pack.csv"
DEFAULT_PREFLIGHT = ROOT / "results/summaries/surface_shape_extension_pilot_preflight_summary.txt"
DEFAULT_SUMMARY = ROOT / "results/summaries/surface_shape_extension_pilot_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/surface_shape_extension_pilot_validation_metrics.csv"
DEFAULT_GROUPS = ROOT / "results/metrics/surface_shape_extension_pilot_group_summary.csv"
DEFAULT_MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DEFAULT_REGISTRY_SUMMARY = ROOT / "results/summaries/surface_shape_extension_pilot_registry_summary.txt"

METRIC_FIELDS = ["check_name", "pass", "observed", "required", "notes"]
GROUP_FIELDS = ["group_field", "group_value", "split", "count", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 25.2 surface shape-extension pilot pack.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--registry-summary", type=Path, default=DEFAULT_REGISTRY_SUMMARY)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def git_value(cwd: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_staged(path: str) -> str:
    return path.replace("\\", "/").strip('"')


def forbidden_staged() -> list[str]:
    staged = [normalize_staged(item) for item in git_value(ROOT, ["diff", "--cached", "--name-only"]).splitlines() if item.strip()]
    bad: list[str] = []
    for path in staged:
        if path == "CURRENT_BASELINE.md" or path == "scripts/visualize_current_baseline.py":
            bad.append(path)
        if path.startswith(FORBIDDEN_STAGE_PREFIXES) or path.lower().endswith(FORBIDDEN_STAGE_SUFFIXES):
            bad.append(path)
    return sorted(set(bad))


def write_preflight(args: argparse.Namespace) -> None:
    required = [
        ROOT / "results/metrics/surface_shape_extension_dataset_plan.csv",
        ROOT / "results/metrics/surface_shape_extension_label_schema.csv",
        ROOT / "results/metrics/surface_shape_extension_comsol_feasibility_matrix.csv",
        ROOT / "results/metrics/surface_shape_extension_taxonomy_matrix.csv",
        COMSOL_ROOT / "scripts/generate_mfl_surface_shape_extension_pilot_pack.py",
    ]
    pinn_status = git_value(ROOT, ["status", "--short"])
    comsol_status = git_value(COMSOL_ROOT, ["status", "--short"])
    args.preflight_summary.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_summary.write_text(
        "\n".join(
            [
                "surface shape-extension pilot preflight summary",
                "stage: 25.2",
                "",
                "scope: preflight before/around COMSOL pilot generation; COMSOL is allowed in 25.2, training and CURRENT_BASELINE updates remain forbidden.",
                "",
                "required_inputs:",
                *[f"- {path}: exists={bool_text(path.exists())}" for path in required],
                "",
                "protocol_reuse:",
                "- 20.70 dynamic material/domain/solver protocol is reused through existing COMSOL MagneticFields helper path.",
                "- Surface subtract is implemented as surface-connected polygon-prism material loss from top_z_0.",
                "- shape_type, topology_type, representation_target, rbc_compatible, component labels, depth grids, and masks are required metadata.",
                "",
                "git_status_PINN_project:",
                pinn_status or "clean",
                "",
                "git_status_COMSOL_Multiphysics_MCP:",
                comsol_status or "clean",
                "",
                "preflight_decision: stop if taxonomy/schema/feasibility inputs are missing; otherwise generation may proceed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def string_list(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def load_pack(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as pack:
        return {name: pack[name].copy() for name in pack.files}


def validate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    if not args.npz.exists():
        raise FileNotFoundError(args.npz)
    if not args.inventory.exists():
        raise FileNotFoundError(args.inventory)
    pack = load_pack(args.npz)
    inventory = read_csv(args.inventory)
    sample_ids = string_list(pack["sample_ids"])
    n = len(sample_ids)
    split = string_list(pack["split"])
    shape_type = string_list(pack["shape_type"])
    topology_type = string_list(pack["topology_type"])
    representation_target = string_list(pack["representation_target"])
    rbc_compatible = np.asarray(pack["rbc_compatible"], dtype=bool)
    component_count = np.asarray(pack["component_count"], dtype=np.int64)
    delta = np.asarray(pack["delta_b"], dtype=np.float64)
    b_defect = np.asarray(pack["b_defect"], dtype=np.float64)
    b_no = np.asarray(pack["b_no_defect"], dtype=np.float64)
    depth_grid = np.asarray(pack["depth_grid_m"], dtype=np.float64)
    masks = np.asarray(pack["projected_mask_2d"], dtype=np.uint8)
    aspect = np.asarray(pack["aspect_ratio"], dtype=np.float64)
    rotation = np.asarray(pack["rotation_angle"], dtype=np.float64)
    asymmetry = np.asarray(pack["asymmetry_score"], dtype=np.float64)
    edge = np.asarray(pack["edge_steepness"], dtype=np.float64)
    split_counts = dict(Counter(split))
    shape_counts = dict(Counter(shape_type))
    topology_counts = dict(Counter(topology_type))
    target_counts = dict(Counter(representation_target))
    delta_error = float(np.max(np.abs(delta - (b_defect - b_no)))) if n else float("nan")
    staged_bad = forbidden_staged()
    successful_inventory = [row for row in inventory if row.get("status") == "pass"]
    bool_mesh_solve = all(
        str(row.get(key, "")).strip().lower() == "true"
        for row in successful_inventory
        for key in ["boolean_subtract_success", "mesh_precheck_success", "solve_success"]
    )
    checks = [
        ("npz_exists", args.npz.exists(), str(args.npz), "NPZ exists but remains generated data"),
        ("npz_not_staged", normalize_staged(str(args.npz.relative_to(ROOT))) not in [normalize_staged(x) for x in git_value(ROOT, ["diff", "--cached", "--name-only"]).splitlines()], str(args.npz), "generated NPZ must not be staged"),
        ("sample_count_target", n == TARGET_N, n, TARGET_N),
        ("sample_count_fallback", n >= FALLBACK_N, n, FALLBACK_N),
        ("split_counts", split_counts == TARGET_SPLIT if n == TARGET_N else all(split_counts.get(name, 0) > 0 for name in TARGET_SPLIT), split_counts, TARGET_SPLIT),
        ("delta_shape", tuple(delta.shape[1:]) == (3, 3, 201), tuple(delta.shape), "(N,3,3,201)"),
        ("bxyz_finite", bool(np.isfinite(delta).all() and np.isfinite(b_defect).all() and np.isfinite(b_no).all()), "finite", "all finite"),
        ("delta_b_check", delta_error <= 1.0e-12, delta_error, "<=1e-12"),
        ("shape_type_coverage", all(shape_counts.get(shape, 0) > 0 for shape in TARGET_SHAPE_COUNTS), shape_counts, "all seven shape families"),
        ("target_shape_counts", all(shape_counts.get(shape, 0) >= target for shape, target in TARGET_SHAPE_COUNTS.items()) if n == TARGET_N else True, shape_counts, TARGET_SHAPE_COUNTS),
        ("topology_coverage", all(topology_counts.get(name, 0) > 0 for name in ["single_component", "multi_component", "elongated", "irregular"]), topology_counts, "single/multi/elongated/irregular"),
        ("representation_target_coverage", all(target_counts.get(name, 0) > 0 for name in ["six_param_rbc", "profile_basis", "depth_grid", "component_set", "polygon_or_contour"]), target_counts, "all representation targets"),
        ("depth_grid_finite", bool(np.isfinite(depth_grid).all() and depth_grid.shape[0] == n), list(depth_grid.shape), "finite depth_grid_m per sample"),
        ("projected_mask_nonempty", bool(np.all(masks.reshape(n, -1).sum(axis=1) > 0)), "nonempty", "all masks nonempty"),
        ("component_count_valid", bool(np.all(component_count >= 1)), component_count.tolist()[:10], ">=1"),
        ("multi_pit_component_labels", all(component_count[i] >= 2 and representation_target[i] == "component_set" for i, shape in enumerate(shape_type) if shape == "multi_pit_two_component_surface_defect"), "checked", "multi-pit component_count>=2 and component_set"),
        ("crack_like_aspect_rotation", all(aspect[i] >= 4.0 and np.isfinite(rotation[i]) for i, shape in enumerate(shape_type) if shape == "elongated_crack_like_surface_defect"), "checked", "aspect>=4 and finite rotation label"),
        ("irregular_depth_grid_target", all(representation_target[i] == "depth_grid" and depth_grid[i].max() > 0 for i, shape in enumerate(shape_type) if shape == "irregular_corrosion_non_rbc"), "checked", "irregular uses depth_grid target"),
        ("non_rbc_not_rbc_compatible", all(not bool(rbc_compatible[i]) for i, shape in enumerate(shape_type) if shape in NON_RBC_SHAPES), "checked", "non-RBC rbc_compatible=false"),
        ("sharpness_asymmetry_finite", bool(np.isfinite(asymmetry).all() and np.isfinite(edge).all()), "finite", "finite scores"),
        ("comsol_boolean_mesh_solve_inventory", bool_mesh_solve, "checked", "Boolean/mesh/solve true for successful rows"),
        ("no_forbidden_artifacts_staged", not staged_bad, staged_bad, "no data/NPZ/.mph/raw/checkpoint/preview/notes/temp STL/CURRENT_BASELINE staged"),
    ]
    for name, ok, observed, required in checks:
        rows.append({"check_name": name, "pass": bool(ok), "observed": json.dumps(observed, ensure_ascii=False, sort_keys=True) if isinstance(observed, (dict, list, tuple)) else observed, "required": json.dumps(required, ensure_ascii=False, sort_keys=True) if isinstance(required, (dict, list, tuple)) else required, "notes": ""})
    for field_name, values in [("shape_type", shape_type), ("topology_type", topology_type), ("representation_target", representation_target)]:
        for (value, split_name), count in sorted(Counter(zip(values, split)).items()):
            groups.append({"group_field": field_name, "group_value": value, "split": split_name, "count": count, "notes": ""})
    metadata = {
        "n_samples": n,
        "status": "pilot_generated" if n >= TARGET_N else "partial_pilot_generated" if n >= FALLBACK_N else "partial_under_fallback",
        "split_counts": split_counts,
        "shape_type_counts": shape_counts,
        "topology_counts": topology_counts,
        "representation_target_counts": target_counts,
        "validation_pass": all(bool(row["pass"]) for row in rows if row["check_name"] not in {"sample_count_target", "target_shape_counts"} or n >= TARGET_N),
        "target_n_pass": n == TARGET_N,
        "fallback_n_pass": n >= FALLBACK_N,
        "delta_max_abs_error": delta_error,
    }
    return rows, groups, metadata


def manifest(metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_role": "surface_shape_extension_pilot",
        "status": metadata["status"],
        "route": ROUTE,
        "stage": "25.2",
        "schema_version": SCHEMA_VERSION,
        "shape_taxonomy_version": TAXONOMY_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "n_samples": metadata["n_samples"],
        "target_N": TARGET_N,
        "fallback_N": FALLBACK_N,
        "split_counts": metadata["split_counts"],
        "shape_type_counts": metadata["shape_type_counts"],
        "topology_counts": metadata["topology_counts"],
        "representation_target_counts": metadata["representation_target_counts"],
        "axes": ["Bx", "By", "Bz"],
        "sensor_z_m": 0.008,
        "scan_line_y": [-0.001, 0.0, 0.001],
        "sensor_x_count": 201,
        "generated_npz_path": str(args.npz),
        "npz_sha256": sha256_file(args.npz),
        "comsol_generator_script": str(COMSOL_ROOT / "scripts/generate_mfl_surface_shape_extension_pilot_pack.py"),
        "comsol_inventory_path": str(args.inventory),
        "validation_script": "scripts/validate_surface_shape_extension_pilot_pack.py",
        "validation_status": bool(metadata["validation_pass"]),
        "target_n_pass": bool(metadata["target_n_pass"]),
        "fallback_n_pass": bool(metadata["fallback_n_pass"]),
        "train_ready_candidate": False,
        "baseline_ready": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "allowed_use": ["schema_validation", "explicit_surface_shape_extension_audit"],
        "forbidden_use": [
            "automatic_mainline_training",
            "baseline_update",
            "current_baseline_replacement",
            "latest_newest_auto_discovery",
            "direct_training_without_manifest_gate",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_commit": git_value(ROOT, ["rev-parse", "HEAD"]),
        "comsol_commit": git_value(COMSOL_ROOT, ["rev-parse", "HEAD"]),
        "notes": "Generated data/NPZ are ignored artifacts and must not be committed.",
    }


def update_registry(path: Path, manifest_data: dict[str, Any]) -> None:
    section = "\n".join(
        [
            f"## {DATASET_ID}",
            "",
            "- dataset_role: surface_shape_extension_pilot",
            f"- status: {manifest_data['status']}",
            f"- route: {ROUTE}",
            "- stage: 25.2",
            f"- schema_version: {SCHEMA_VERSION}",
            f"- shape_taxonomy_version: {TAXONOMY_VERSION}",
            f"- label_schema_version: {LABEL_SCHEMA_VERSION}",
            "- geometry_method: surface_connected_polygon_prism_boolean_subtract / stacked_layer_control",
            f"- path: `{manifest_data['generated_npz_path']}`",
            f"- manifest_path: `{manifest_data['manifest_path']}`",
            f"- n_samples: {manifest_data['n_samples']}",
            f"- split_counts: {manifest_data['split_counts']}",
            f"- shape_type_counts: {manifest_data['shape_type_counts']}",
            f"- topology_counts: {manifest_data['topology_counts']}",
            f"- representation_target_counts: {manifest_data['representation_target_counts']}",
            "- train_ready_candidate: false",
            "- baseline_ready: false",
            "- auto_discovery_allowed: false",
            "- latest_newest_discovery_allowed: false",
            "- allowed_use: schema_validation, explicit_surface_shape_extension_audit",
            "- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate",
            "- generator_script: `scripts/generate_mfl_surface_shape_extension_pilot_pack.py`",
            "- validation_script: `scripts/validate_surface_shape_extension_pilot_pack.py`",
            f"- npz_sha256: {manifest_data['npz_sha256']}",
            "- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.",
            "",
        ]
    )
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else "# COMSOL Data Registry\n\n"
    marker = f"## {DATASET_ID}"
    if marker in text:
        start = text.index(marker)
        next_start = text.find("\n## ", start + 1)
        if next_start == -1:
            text = text[:start].rstrip() + "\n\n" + section
        else:
            text = text[:start].rstrip() + "\n\n" + section + text[next_start:].lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    write_preflight(args)
    required_inputs = [
        ROOT / "results/metrics/surface_shape_extension_dataset_plan.csv",
        ROOT / "results/metrics/surface_shape_extension_label_schema.csv",
        ROOT / "results/metrics/surface_shape_extension_comsol_feasibility_matrix.csv",
    ]
    missing = [path for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("missing 25.1 inputs: " + ", ".join(str(path) for path in missing))
    if args.preflight_only:
        return 0
    rows, groups, metadata = validate(args)
    write_csv(args.metrics, rows, METRIC_FIELDS)
    write_csv(args.group_summary, groups, GROUP_FIELDS)
    manifest_data = manifest(metadata, args)
    manifest_data["manifest_path"] = str(args.manifest)
    write_json(args.manifest, manifest_data)
    update_registry(args.registry, manifest_data)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface shape-extension pilot validation summary",
                "stage: 25.2",
                "",
                f"dataset_id: {DATASET_ID}",
                f"status: {metadata['status']}",
                f"validation_pass: {bool_text(bool(metadata['validation_pass']))}",
                f"target_n_pass: {bool_text(bool(metadata['target_n_pass']))}",
                f"fallback_n_pass: {bool_text(bool(metadata['fallback_n_pass']))}",
                f"n_samples: {metadata['n_samples']}",
                f"split_counts: {metadata['split_counts']}",
                f"shape_type_counts: {metadata['shape_type_counts']}",
                f"topology_counts: {metadata['topology_counts']}",
                f"representation_target_counts: {metadata['representation_target_counts']}",
                f"delta_max_abs_error: {metadata['delta_max_abs_error']}",
                "train_ready_candidate: false",
                "baseline_ready: false",
                "CURRENT_BASELINE_update: false",
                f"manifest_path: {args.manifest}",
                f"validation_metrics: {args.metrics}",
                f"group_summary: {args.group_summary}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text(
        "\n".join(
            [
                "surface shape-extension pilot registry summary",
                "stage: 25.2",
                "",
                f"dataset_id: {DATASET_ID}",
                f"registry_updated: {bool_text(DATASET_ID in args.registry.read_text(encoding='utf-8', errors='replace'))}",
                f"manifest_path: {args.manifest}",
                "train_ready_candidate: false",
                "baseline_ready: false",
                "allowed_use: schema_validation, explicit_surface_shape_extension_audit",
                "forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement",
                "data_policy: generated NPZ/data remain ignored and uncommitted.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not metadata["validation_pass"]:
        raise RuntimeError("surface shape-extension pilot validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
