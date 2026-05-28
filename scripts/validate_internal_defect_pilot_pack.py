#!/usr/bin/env python
"""Validate and register the 21.1 internal / buried defect COMSOL pilot pack.

The generated NPZ is consumed from an explicit path. This script does not scan
latest/newest data, train models, run COMSOL, or update CURRENT_BASELINE.md.
"""

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
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DATASET_ID = "comsol_internal_defect_pilot_pack_v1"
SOURCE_DATASET_ID = "comsol_internal_defect_smoke_pack_v1"
ROUTE = "internal_buried_defect_feasibility"
SCHEMA_VERSION = "internal_defect_feasibility_v1"
PLAN_CSV = ROOT / "results/metrics/internal_defect_pilot_pack_plan.csv"
PACK_NPZ = ROOT / "data/comsol_mfl/generated/internal_defect_pilot_pack/internal_defect_pilot_pack_v1.npz"
COMSOL_INVENTORY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\inventory_internal_defect_pilot_pack.csv")
COMSOL_SUMMARY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\internal_defect_pilot_pack_summary.txt")
SUMMARY = ROOT / "results/summaries/internal_defect_pilot_pack_validation_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_pilot_pack_validation_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_pilot_pack_group_summary.csv"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v1.manifest.json"
REGISTRY_SUMMARY = ROOT / "results/summaries/internal_defect_pilot_pack_registry_summary.txt"

AXIS_NAMES = ["Bx", "By", "Bz"]
SCAN_LINE_Y = [-0.001, 0.0, 0.001]
FORBIDDEN_USE = [
    "automatic_mainline_training",
    "baseline_update",
    "current_baseline_replacement",
    "latest_newest_auto_discovery",
    "direct_training_without_manifest_gate",
]
ALLOWED_USE = ["schema_validation", "explicit_internal_training_gate"]
CHECK_FIELDS = ["check_name", "pass", "observed", "expected", "notes"]
GROUP_FIELDS = ["group_field", "group_value", "row_count", "success_count", "failure_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and register internal defect pilot pack.")
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--pack-npz", type=Path, default=PACK_NPZ)
    parser.add_argument("--comsol-inventory", type=Path, default=COMSOL_INVENTORY)
    parser.add_argument("--comsol-summary", type=Path, default=COMSOL_SUMMARY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--registry-summary", type=Path, default=REGISTRY_SUMMARY)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_lines(cwd: Path, args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def as_strings(value: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(value).reshape(-1).tolist()]


def status_from_success(success_count: int, planned_count: int) -> str:
    if success_count == planned_count and planned_count == 96:
        return "pilot_generated"
    if success_count >= 72:
        return "partial_pilot_generated"
    return "blocked"


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
            f"- geometry_method: {manifest['geometry_method']}",
            f"- exact_piao_rbc: {str(manifest['exact_piao_rbc']).lower()}",
            f"- rbc_style_approximation: {str(manifest['rbc_style_approximation']).lower()}",
            f"- path: `{manifest['npz_path']}`",
            f"- manifest_path: `{manifest['manifest_path']}`",
            f"- n_samples: {manifest['n_samples']}",
            f"- planned_samples: {manifest['planned_samples']}",
            f"- split_counts: {manifest['split_counts']}",
            f"- shape_counts: {manifest['shape_counts']}",
            f"- burial_depth_counts: {manifest['burial_depth_counts']}",
            f"- train_ready_candidate: {str(manifest['train_ready_candidate']).lower()}",
            f"- baseline_ready: {str(manifest['baseline_ready']).lower()}",
            f"- auto_discovery_allowed: {str(manifest['auto_discovery_allowed']).lower()}",
            f"- latest_newest_discovery_allowed: {str(manifest['latest_newest_discovery_allowed']).lower()}",
            f"- allowed_use: {', '.join(manifest['allowed_use'])}",
            f"- forbidden_use: {', '.join(manifest['forbidden_use'])}",
            f"- source_dataset_ids: {', '.join(manifest['source_dataset_ids'])}",
            f"- generator_script: `{manifest['generator_script']}`",
            f"- validation_script: `{manifest['validation_script']}`",
            f"- npz_sha256: {manifest['npz_sha256']}",
            "- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.",
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


def no_forbidden_staged(root: Path) -> tuple[bool, str]:
    staged = git_lines(root, ["diff", "--cached", "--name-only"])
    forbidden = [
        path
        for path in staged
        if path.startswith("data/")
        or path.endswith(".npz")
        or path.endswith(".mph")
        or path.endswith(".pt")
        or path.endswith(".pth")
        or path.startswith("checkpoints/")
        or path.startswith("results/previews/")
        or path.startswith("notes/")
        or path == "CURRENT_BASELINE.md"
    ]
    return (not forbidden, ",".join(forbidden))


def group_summary(plan_rows: list[dict[str, str]], inventory_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    status_by_id = {row.get("sample_id", ""): row.get("status", "missing") for row in inventory_rows}
    joined = [{**row, "status": status_by_id.get(row.get("sample_id", ""), "missing")} for row in plan_rows]
    groups: list[dict[str, Any]] = []
    for field in ["split", "shape_type", "burial_depth_level", "size_level", "aspect_bin"]:
        for value in sorted({row.get(field, "") for row in joined}):
            subset = [row for row in joined if row.get(field, "") == value]
            groups.append(
                {
                    "group_field": field,
                    "group_value": value,
                    "row_count": len(subset),
                    "success_count": sum(1 for row in subset if row["status"] == "success"),
                    "failure_count": sum(1 for row in subset if row["status"] != "success"),
                }
            )
    return groups


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics, args.group_summary, args.manifest, args.registry_summary], args.overwrite)
    checks: list[dict[str, Any]] = []
    plan_rows = read_csv(args.plan_csv)
    inventory_rows = read_csv(args.comsol_inventory)
    arrays = load_npz(args.pack_npz)
    success_rows = [row for row in inventory_rows if row.get("status") == "success"]
    success_count = len(success_rows)

    add(checks, "plan_csv_exists", args.plan_csv.exists(), str(args.plan_csv), "internal pilot plan CSV")
    add(checks, "planned_rows_96", len(plan_rows) == 96, len(plan_rows), 96)
    add(checks, "comsol_inventory_exists", args.comsol_inventory.exists(), str(args.comsol_inventory), "COMSOL inventory")
    add(checks, "comsol_summary_exists", args.comsol_summary.exists(), str(args.comsol_summary), "COMSOL summary")
    add(checks, "pack_npz_exists", args.pack_npz.exists(), str(args.pack_npz), "ignored generated NPZ")
    add(checks, "success_count_fallback_minimum", success_count >= 72, success_count, ">=72")

    required_fields = [
        "delta_b",
        "b_defect",
        "b_no_defect",
        "axis_names",
        "sensor_x",
        "scan_line_y",
        "sensor_z_m",
        "sample_ids",
        "split",
        "shape_type",
        "burial_depth_level",
        "size_level",
        "aspect_bin",
        "burial_depth_m",
        "depth_to_surface_m",
        "defect_center_xyz_m",
        "L_m",
        "W_m",
        "D_m",
        "D_m_or_cavity_size_m",
        "ground_truth_method",
        "cavity_internal",
        "no_defect_reference_id",
    ]
    missing = [key for key in required_fields if key not in arrays]
    add(checks, "npz_required_fields", not missing, ",".join(missing), "no missing required fields")
    if arrays:
        delta = arrays.get("delta_b")
        b_defect = arrays.get("b_defect")
        b_no = arrays.get("b_no_defect")
        expected_shape = (success_count, 3, 3, 201)
        add(checks, "delta_b_shape", delta is not None and tuple(delta.shape) == expected_shape, getattr(delta, "shape", None), expected_shape)
        add(checks, "b_defect_shape", b_defect is not None and tuple(b_defect.shape) == expected_shape, getattr(b_defect, "shape", None), expected_shape)
        add(checks, "b_no_defect_shape", b_no is not None and tuple(b_no.shape) == expected_shape, getattr(b_no, "shape", None), expected_shape)
        finite = bool(delta is not None and b_defect is not None and b_no is not None and np.isfinite(delta).all() and np.isfinite(b_defect).all() and np.isfinite(b_no).all())
        add(checks, "bxyz_finite", finite, finite, True)
        if delta is not None and b_defect is not None and b_no is not None:
            max_err = float(np.max(np.abs(delta - (b_defect - b_no))))
            add(checks, "delta_b_equals_defect_minus_no_defect", max_err <= 1.0e-7, max_err, "<=1e-7")
            add(checks, "delta_b_nonzero", bool(np.any(np.abs(delta) > 0)), float(np.max(np.abs(delta))), ">0")
        add(checks, "axis_names", as_strings(arrays.get("axis_names", np.asarray([]))) == AXIS_NAMES, as_strings(arrays.get("axis_names", np.asarray([]))), AXIS_NAMES)
        add(checks, "sensor_x_count", arrays.get("sensor_x", np.asarray([])).shape == (201,), arrays.get("sensor_x", np.asarray([])).shape, "(201,)")
        add(checks, "scan_line_y_shape", arrays.get("scan_line_y", np.asarray([])).shape == (success_count, 3), arrays.get("scan_line_y", np.asarray([])).shape, f"({success_count}, 3)")
        add(checks, "sensor_z_nominal", bool(np.allclose(arrays.get("sensor_z_m", np.asarray([])), 0.008)), arrays.get("sensor_z_m", np.asarray([])).tolist(), 0.008)
        internal = arrays.get("cavity_internal")
        add(checks, "cavity_internal_true", internal is not None and bool(np.asarray(internal, dtype=bool).all()), "" if internal is None else np.asarray(internal).tolist(), True)
        depths = arrays.get("depth_to_surface_m")
        d_m = arrays.get("D_m")
        add(checks, "internal_depth_positive", depths is not None and bool(np.all(depths > 0)), "" if depths is None else depths.tolist(), ">0")
        add(checks, "internal_depth_within_block", depths is not None and d_m is not None and bool(np.all(depths + d_m <= 0.0056 + 1e-9)), "" if depths is None or d_m is None else np.max(depths + d_m), "<=0.0056")

    shape_counts = Counter(as_strings(arrays.get("shape_type", np.asarray([])))) if arrays else Counter()
    burial_counts = Counter(as_strings(arrays.get("burial_depth_level", np.asarray([])))) if arrays else Counter()
    size_counts = Counter(as_strings(arrays.get("size_level", np.asarray([])))) if arrays else Counter()
    aspect_counts = Counter(as_strings(arrays.get("aspect_bin", np.asarray([])))) if arrays else Counter()
    split_counts = Counter(as_strings(arrays.get("split", np.asarray([])))) if arrays else Counter()
    add(checks, "shape_type_coverage", set(shape_counts) == {"internal_sphere", "internal_ellipsoid", "internal_cuboid"}, dict(shape_counts), "3 shape types")
    add(checks, "burial_depth_coverage", set(burial_counts) == {"shallow", "medium", "deep", "deep_plus"}, dict(burial_counts), "4 burial levels")
    add(checks, "size_level_coverage", set(size_counts) == {"small", "medium", "large"}, dict(size_counts), "3 size levels")
    add(checks, "aspect_bin_coverage", {"compact", "elongated_x", "elongated_y"}.issubset(set(aspect_counts)), dict(aspect_counts), "compact/elongated_x/elongated_y")
    add(checks, "split_full_or_fallback", (split_counts == Counter({"train": 64, "val": 16, "test": 16})) or (split_counts["train"] >= 48 and split_counts["val"] >= 12 and split_counts["test"] >= 12), dict(split_counts), "64/16/16 or >=48/12/12")

    status = status_from_success(success_count, len(plan_rows))
    blocker_names = {
        "plan_csv_exists",
        "planned_rows_96",
        "comsol_inventory_exists",
        "comsol_summary_exists",
        "pack_npz_exists",
        "success_count_fallback_minimum",
        "npz_required_fields",
        "delta_b_shape",
        "b_defect_shape",
        "b_no_defect_shape",
        "bxyz_finite",
        "delta_b_equals_defect_minus_no_defect",
        "axis_names",
        "sensor_x_count",
        "scan_line_y_shape",
        "cavity_internal_true",
        "internal_depth_positive",
        "internal_depth_within_block",
        "shape_type_coverage",
        "burial_depth_coverage",
        "size_level_coverage",
        "split_full_or_fallback",
    }
    failed_blockers = [row["check_name"] for row in checks if row["check_name"] in blocker_names and not row["pass"]]
    if failed_blockers:
        status = "blocked"
    train_ready = status in {"pilot_generated", "partial_pilot_generated"} and not failed_blockers

    npz_sha = sha256_file(args.pack_npz) if args.pack_npz.exists() else ""
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_role": "internal_defect_feasibility_pilot_pack",
        "status": status,
        "route": ROUTE,
        "stage": "21.1",
        "schema_version": SCHEMA_VERSION,
        "geometry_method": "internal_cavity_comsol_solid",
        "exact_piao_rbc": False,
        "rbc_style_approximation": False,
        "internal_surface_mixed": False,
        "npz_path": str(args.pack_npz),
        "manifest_path": str(args.manifest),
        "n_samples": success_count,
        "planned_samples": len(plan_rows),
        "split_counts": dict(split_counts),
        "shape_counts": dict(shape_counts),
        "burial_depth_counts": dict(burial_counts),
        "size_counts": dict(size_counts),
        "aspect_counts": dict(aspect_counts),
        "axis_names": AXIS_NAMES,
        "sensor_x_count": 201,
        "scan_line_y_m": SCAN_LINE_Y,
        "sensor_z_m": 0.008,
        "label_fields": [
            "L_m",
            "W_m",
            "D_m",
            "burial_depth_m",
            "depth_to_surface_m",
            "defect_center_xyz_m",
            "shape_type",
            "aspect_bin",
            "ground_truth_method",
        ],
        "train_ready_candidate": train_ready,
        "baseline_ready": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "allowed_use": ALLOWED_USE,
        "forbidden_use": FORBIDDEN_USE,
        "source_dataset_ids": [SOURCE_DATASET_ID],
        "generator_script": str(Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\scripts\generate_mfl_internal_defect_pilot_pack.py")),
        "validation_script": "scripts/validate_internal_defect_pilot_pack.py",
        "npz_sha256": npz_sha,
        "surface_rbc_baseline_update": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(args.manifest, manifest)
    update_registry(args.registry, manifest)
    ok_staged, staged_forbidden = no_forbidden_staged(ROOT)
    add(checks, "no_forbidden_data_staged", ok_staged, staged_forbidden, "no forbidden staged paths")
    write_csv(args.metrics, checks, CHECK_FIELDS)
    groups = group_summary(plan_rows, inventory_rows)
    write_csv(args.group_summary, groups, GROUP_FIELDS)

    failed_checks = [row["check_name"] for row in checks if not row["pass"]]
    lines = [
        "21.1 内部/埋藏缺陷 pilot pack validation summary",
        "",
        f"dataset_id: {DATASET_ID}",
        f"status: {status}",
        f"planned_samples: {len(plan_rows)}",
        f"successful_samples: {success_count}",
        f"train_ready_candidate: {str(train_ready).lower()}",
        f"validation_blockers: {failed_blockers if failed_blockers else 'none'}",
        f"shape_counts: {dict(shape_counts)}",
        f"burial_depth_counts: {dict(burial_counts)}",
        f"size_counts: {dict(size_counts)}",
        f"aspect_counts: {dict(aspect_counts)}",
        f"split_counts: {dict(split_counts)}",
        f"npz_path_ignored: {args.pack_npz}",
        f"manifest_path: {args.manifest}",
        "baseline_update: false",
        "current_surface_rbc_baseline_mixed: false",
        "latest_newest_npz_scan: false",
        "",
        "failed_checks:",
        *(f"- {name}" for name in failed_checks),
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry_lines = [
        "21.1 internal defect pilot pack registry summary",
        "",
        f"registry_path: {args.registry}",
        f"manifest_path: {args.manifest}",
        f"dataset_id: {DATASET_ID}",
        f"status: {status}",
        f"allowed_use: {', '.join(ALLOWED_USE)}",
        f"forbidden_use: {', '.join(FORBIDDEN_USE)}",
        f"train_ready_candidate: {str(train_ready).lower()}",
        "baseline_ready: false",
        "auto_discovery_allowed: false",
        "latest_newest_discovery_allowed: false",
    ]
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")
    if failed_blockers:
        raise RuntimeError(f"internal defect pilot validation blockers: {failed_blockers}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
