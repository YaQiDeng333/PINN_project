#!/usr/bin/env python
"""Validate and register the 25.9b surface multi-pit component-set pilot pack."""

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

DATASET_ID = "comsol_surface_multipit_component_set_pilot_v1"
ROUTE = "surface_multipit_component_set"
SCHEMA_VERSION = "surface_multipit_component_set_pilot_v1"
COMPONENT_SCHEMA_VERSION = "surface_multipit_component_schema_v1"
TOPOLOGY_SCHEMA_VERSION = "surface_multipit_topology_schema_v1"
SOURCE_DATASET_IDS = ["comsol_surface_shape_extension_pilot_v1", "comsol_surface_multipit_topup_pack_v1"]
TARGET_N = 112
TARGET_SPLIT = {"train": 72, "val": 20, "test": 20}
TARGET_K = 3
TARGET_COMPONENT_COUNTS = {"2": 100, "3": 12}
REQUIRED_SEPARATION = {"separated", "close", "touching", "partially_overlapping"}
REQUIRED_TOPOLOGY = {"disconnected", "touching_boundary", "partially_overlapping"}
REQUIRED_ORIENTATION = {"aligned_x", "aligned_y", "diagonal"}

DEFAULT_NPZ = ROOT / "data/comsol_mfl/prepared/experimental/surface_multipit_component_set/comsol_surface_multipit_component_set_pilot_v1.npz"
DEFAULT_TOPUP_VALIDATION = ROOT / "results/metrics/surface_multipit_topup_pack_validation_metrics.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/surface_multipit_component_set_pilot_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/surface_multipit_component_set_pilot_validation_metrics.csv"
DEFAULT_GROUPS = ROOT / "results/metrics/surface_multipit_component_set_pilot_group_summary.csv"
DEFAULT_MANIFEST = ROOT / "results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json"
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DEFAULT_REGISTRY_SUMMARY = ROOT / "results/summaries/surface_multipit_component_set_pilot_registry_summary.txt"

METRIC_FIELDS = ["check_name", "pass", "observed", "required", "notes"]
GROUP_FIELDS = ["group_field", "group_value", "split", "count", "notes"]
FORBIDDEN_STAGE_PREFIXES = ("data/", "checkpoints/", "notes/", "results/previews/")
FORBIDDEN_STAGE_SUFFIXES = (".npz", ".mph", ".png", ".jpg", ".jpeg", ".stl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate/register 25.9b component-set pilot pack.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--topup-validation", type=Path, default=DEFAULT_TOPUP_VALIDATION)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--registry-summary", type=Path, default=DEFAULT_REGISTRY_SUMMARY)
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def strings(values: np.ndarray) -> list[str]:
    return [str(item) for item in values.tolist()]


def counts(values: list[str] | np.ndarray) -> dict[str, int]:
    if isinstance(values, np.ndarray):
        values = strings(values)
    return dict(Counter(str(item) for item in values))


def metric(name: str, passed: bool, observed: Any, required: Any, notes: str = "") -> dict[str, Any]:
    def encode(value: Any) -> Any:
        return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple, set)) else value

    return {"check_name": name, "pass": bool_text(passed), "observed": encode(observed), "required": encode(required), "notes": notes}


def validate_pack(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not args.npz.exists():
        raise FileNotFoundError(args.npz)
    pack = load_npz(args.npz)
    n = len(pack["sample_ids"])
    split = strings(pack["split"])
    component_count = np.asarray(pack["component_count"], dtype=np.int64)
    component_exists = np.asarray(pack["component_exists"], dtype=bool)
    component_centers = np.asarray(pack["component_center_xy_m"], dtype=np.float64)
    component_lwd = np.asarray(pack["component_lwd_m"], dtype=np.float64)
    component_rot = np.asarray(pack["component_rotation_angle"], dtype=np.float64)
    component_masks = np.asarray(pack["component_projected_masks_2d"], dtype=np.uint8)
    component_depths = np.asarray(pack["component_depth_grids_m"], dtype=np.float64)
    union_masks = np.asarray(pack["projected_mask_2d"], dtype=np.uint8)
    union_depths = np.asarray(pack["depth_grid_m"], dtype=np.float64)
    delta_b = np.asarray(pack["delta_b"], dtype=np.float64)
    b_defect = np.asarray(pack["b_defect"], dtype=np.float64)
    b_no_defect = np.asarray(pack["b_no_defect"], dtype=np.float64)
    separation = strings(pack["separation_type"])
    topology = strings(pack["topology_relation"])
    orientation = strings(pack["orientation_type"])
    relative_depth = strings(pack["relative_depth_type"])
    source_dataset_id = strings(pack["source_dataset_id"])
    split_counts = counts(split)
    component_counts = counts([str(int(item)) for item in component_count.tolist()])
    separation_counts = counts(separation)
    topology_counts = counts(topology)
    orientation_counts = counts(orientation)
    relative_depth_counts = counts(relative_depth)
    source_counts = counts(source_dataset_id)
    delta_error = float(np.max(np.abs(delta_b - (b_defect - b_no_defect)))) if n else float("nan")
    active_depth_valid = []
    active_mask_valid = []
    active_param_valid = []
    for i in range(n):
        for slot in range(TARGET_K):
            if not component_exists[i, slot]:
                continue
            active_mask_valid.append(int(component_masks[i, slot].sum()) > 0)
            active_depth_valid.append(np.isfinite(component_depths[i, slot]).all() and float(component_depths[i, slot].max()) > 0.0)
            active_param_valid.append(
                np.isfinite(component_centers[i, slot]).all()
                and np.isfinite(component_lwd[i, slot]).all()
                and float(component_lwd[i, slot].min()) > 0.0
                and np.isfinite(component_rot[i, slot])
            )
    component_exists_match = bool(np.all(component_exists.sum(axis=1) == component_count))
    missing_fields = int(active_mask_valid.count(False) + active_depth_valid.count(False) + active_param_valid.count(False))
    staged_bad = forbidden_staged()
    topup_metrics = read_csv(args.topup_validation) if args.topup_validation.exists() else []
    topup_pass = bool(topup_metrics) and all(row.get("pass") == "true" for row in topup_metrics)
    checks = [
        metric("assembled_npz_exists", args.npz.exists(), str(args.npz), "generated NPZ exists but remains uncommitted"),
        metric("assembled_N", n == TARGET_N, n, TARGET_N),
        metric("split_72_20_20", split_counts == TARGET_SPLIT, split_counts, TARGET_SPLIT),
        metric("K_max_3", int(np.asarray(pack["K_max"])[0]) == TARGET_K and component_exists.shape[1] == TARGET_K, component_exists.shape, f"K={TARGET_K}"),
        metric("component_count_100_12", component_counts == TARGET_COMPONENT_COUNTS, component_counts, TARGET_COMPONENT_COUNTS),
        metric("required_separation_coverage", REQUIRED_SEPARATION.issubset(set(separation)), sorted(set(separation)), sorted(REQUIRED_SEPARATION)),
        metric("required_topology_coverage", REQUIRED_TOPOLOGY.issubset(set(topology)), sorted(set(topology)), sorted(REQUIRED_TOPOLOGY)),
        metric("required_orientation_coverage", REQUIRED_ORIENTATION.issubset(set(orientation)), sorted(set(orientation)), sorted(REQUIRED_ORIENTATION)),
        metric("source_dataset_counts", source_counts == {"comsol_surface_multipit_topup_pack_v1": 96, "comsol_surface_shape_extension_pilot_v1": 16}, source_counts, "16 old seed + 96 top-up"),
        metric("component_exists_count_match", component_exists_match, component_exists.sum(axis=1).tolist()[:10], "component_exists sums equal component_count"),
        metric("component_params_valid", all(active_param_valid), missing_fields, "0 invalid active component labels"),
        metric("component_masks_valid", all(active_mask_valid), active_mask_valid.count(False), "0 empty active component masks"),
        metric("component_depth_grids_valid", all(active_depth_valid), active_depth_valid.count(False), "0 invalid active component depth grids"),
        metric("union_masks_valid", bool(np.all(union_masks.reshape(n, -1).sum(axis=1) > 0)), int(union_masks.sum()), "each sample has projected union mask"),
        metric("union_depth_grids_valid", np.isfinite(union_depths).all() and bool(np.all(union_depths.reshape(n, -1).max(axis=1) > 0.0)), str(union_depths.shape), "finite positive union depth grids"),
        metric("finite_Bx_By_Bz", np.isfinite(delta_b).all() and np.isfinite(b_defect).all() and np.isfinite(b_no_defect).all(), str(delta_b.shape), "finite Bx/By/Bz arrays"),
        metric("delta_b_consistency", delta_error <= 1e-12, f"{delta_error:.3e}", "<=1e-12"),
        metric("topup_validation_pass", topup_pass, topup_pass, "top-up validation metrics all pass"),
        metric("CURRENT_BASELINE_unchanged", not bool(git_value(ROOT, ["diff", "--name-only", "--", "CURRENT_BASELINE.md"])), git_value(ROOT, ["diff", "--name-only", "--", "CURRENT_BASELINE.md"]) or "clean", "no CURRENT_BASELINE diff"),
        metric("no_forbidden_staged_artifacts", not staged_bad, "|".join(staged_bad), "no generated data/NPZ/checkpoint/preview/CURRENT_BASELINE staged"),
    ]
    train_ready_candidate = all(row["pass"] == "true" for row in checks)
    groups: list[dict[str, Any]] = []
    dimensions = {
        "split": split,
        "component_count": [str(int(item)) for item in component_count.tolist()],
        "separation_type": separation,
        "topology_relation": topology,
        "relative_depth_type": relative_depth,
        "orientation_type": orientation,
        "source_dataset_id": source_dataset_id,
    }
    for field, values in dimensions.items():
        for (value, split_name), count in Counter(zip(values, split)).items():
            groups.append({"group_field": field, "group_value": value, "split": split_name, "count": count, "notes": "assembled component-set pilot"})
    context = {
        "n": n,
        "split_counts": split_counts,
        "component_counts": component_counts,
        "separation_counts": separation_counts,
        "topology_counts": topology_counts,
        "orientation_counts": orientation_counts,
        "relative_depth_counts": relative_depth_counts,
        "source_counts": source_counts,
        "missing_fields": missing_fields,
        "delta_error": delta_error,
        "train_ready_candidate": train_ready_candidate,
        "baseline_ready": False,
        "npz_path": str(args.npz),
        "npz_sha256": sha256_file(args.npz),
    }
    return checks, groups, context


def write_manifest(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_role": "surface_multipit_component_set_pilot",
        "stage": "25.9b",
        "status": "pilot_generated" if context["train_ready_candidate"] else "pilot_generated_needs_review",
        "route": ROUTE,
        "schema_version": SCHEMA_VERSION,
        "component_schema_version": COMPONENT_SCHEMA_VERSION,
        "topology_schema_version": TOPOLOGY_SCHEMA_VERSION,
        "path": context["npz_path"],
        "manifest_path": str(args.manifest),
        "n_samples": context["n"],
        "split_counts": context["split_counts"],
        "component_count_counts": context["component_counts"],
        "separation_counts": context["separation_counts"],
        "topology_counts": context["topology_counts"],
        "orientation_counts": context["orientation_counts"],
        "relative_depth_counts": context["relative_depth_counts"],
        "source_dataset_counts": context["source_counts"],
        "K_max": TARGET_K,
        "train_ready_candidate": context["train_ready_candidate"],
        "baseline_ready": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "allowed_use": ["schema_validation", "explicit_component_set_training_gate"],
        "forbidden_use": [
            "automatic_mainline_training",
            "baseline_update",
            "current_baseline_replacement",
            "latest_newest_auto_discovery",
            "direct_training_without_manifest_gate",
            "six_parameter_rbc_success_credit",
        ],
        "source_dataset_ids": SOURCE_DATASET_IDS,
        "topup_generator_script": str(COMSOL_ROOT / "scripts/generate_mfl_surface_multipit_topup_pack.py"),
        "validation_script": "scripts/validate_surface_multipit_component_set_pilot_pack.py",
        "npz_sha256": context["npz_sha256"],
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "notes": "Metadata only. Generated NPZ/data files are ignored artifacts and must be loaded only by explicit dataset_id + manifest.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def registry_section(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"## {DATASET_ID}",
            "",
            "- dataset_role: surface_multipit_component_set_pilot",
            f"- status: {manifest['status']}",
            f"- route: {ROUTE}",
            "- stage: 25.9b",
            f"- schema_version: {SCHEMA_VERSION}",
            f"- component_schema_version: {COMPONENT_SCHEMA_VERSION}",
            f"- topology_schema_version: {TOPOLOGY_SCHEMA_VERSION}",
            "- representation: fixed_K_component_set",
            f"- K_max: {TARGET_K}",
            f"- path: `{manifest['path']}`",
            f"- manifest_path: `{manifest['manifest_path']}`",
            f"- n_samples: {manifest['n_samples']}",
            f"- split_counts: {manifest['split_counts']}",
            f"- component_count_counts: {manifest['component_count_counts']}",
            f"- separation_counts: {manifest['separation_counts']}",
            f"- topology_counts: {manifest['topology_counts']}",
            f"- orientation_counts: {manifest['orientation_counts']}",
            f"- train_ready_candidate: {str(manifest['train_ready_candidate']).lower()}",
            "- baseline_ready: false",
            "- auto_discovery_allowed: false",
            "- latest_newest_discovery_allowed: false",
            "- allowed_use: schema_validation, explicit_component_set_training_gate",
            "- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate, six_parameter_rbc_success_credit",
            f"- source_dataset_ids: {', '.join(SOURCE_DATASET_IDS)}",
            f"- generator_script: `scripts/generate_mfl_surface_multipit_topup_pack.py`",
            "- validation_script: `scripts/validate_surface_multipit_component_set_pilot_pack.py`",
            f"- npz_sha256: {manifest['npz_sha256']}",
            "- notes: Metadata only. Generated NPZ/data files are not committed; multi-pit is component-set data, not a CURRENT_BASELINE transition.",
            "",
        ]
    )


def upsert_registry(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    section = registry_section(manifest)
    text = args.registry.read_text(encoding="utf-8") if args.registry.exists() else "# COMSOL Data Registry\n\n"
    marker = f"## {DATASET_ID}"
    if marker in text:
        start = text.index(marker)
        next_start = text.find("\n## ", start + 1)
        if next_start == -1:
            text = text[:start].rstrip() + "\n\n" + section
        else:
            text = text[:start].rstrip() + "\n\n" + section + text[next_start:]
    else:
        text = text.rstrip() + "\n\n" + section
    args.registry.write_text(text, encoding="utf-8")
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text(
        "\n".join(
            [
                "surface multi-pit component-set pilot registry summary",
                "stage: 25.9b",
                "",
                f"dataset_id: {DATASET_ID}",
                f"manifest_path: {args.manifest}",
                f"registry_path: {args.registry}",
                f"train_ready_candidate: {manifest['train_ready_candidate']}",
                "baseline_ready: false",
                "auto_discovery_allowed: false",
                "latest_newest_discovery_allowed: false",
                "forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, six_parameter_rbc_success_credit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_summary(args: argparse.Namespace, context: dict[str, Any]) -> None:
    lines = [
        "surface multi-pit component-set pilot validation summary",
        "stage: 25.9b",
        "",
        f"dataset_id: {DATASET_ID}",
        f"assembled_N: {context['n']}",
        f"assembled_split_counts: {context['split_counts']}",
        f"component_count_counts: {context['component_counts']}",
        f"separation_counts: {context['separation_counts']}",
        f"topology_counts: {context['topology_counts']}",
        f"orientation_counts: {context['orientation_counts']}",
        f"relative_depth_counts: {context['relative_depth_counts']}",
        f"source_dataset_counts: {context['source_counts']}",
        f"component_label_missing_count: {context['missing_fields']}",
        f"delta_b_max_abs_error: {context['delta_error']:.3e}",
        f"train_ready_candidate: {context['train_ready_candidate']}",
        "baseline_ready: false",
        "",
        "route_boundary:",
        "- This pack can only enter the explicit 25.10 component-set training gate after review.",
        "- It cannot be auto-discovered as the current baseline and cannot give six-parameter RBC success credit.",
        "- CURRENT_BASELINE.md remains unchanged.",
        "",
        f"assembled_npz: {context['npz_path']}",
        f"manifest: {args.manifest}",
        f"metrics_csv: {args.metrics}",
        f"group_summary_csv: {args.group_summary}",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    checks, groups, context = validate_pack(args)
    write_csv(args.metrics, checks, METRIC_FIELDS)
    write_csv(args.group_summary, groups, GROUP_FIELDS)
    manifest = write_manifest(args, context)
    upsert_registry(args, manifest)
    write_summary(args, context)
    return 0 if context["train_ready_candidate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
