#!/usr/bin/env python
"""Validate the surface RBC targeted +120 top-up pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_true_3d_rbc_dataset_120_pack as v120  # noqa: E402


DATASET_ID = "comsol_true_3d_rbc_surface_targeted_topup_v1_120"
SOURCE_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc"
NPZ_ROUTE = "true_3d_piao_style"
REGISTRY_ROUTE = "true_3d_rbc_surface_targeted_expansion"
COMSOL_ROOT = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")

DEFAULT_NPZ = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_surface_targeted_topup_v1_120.npz"
DEFAULT_PLAN = ROOT / "results/metrics/surface_rbc_targeted_expansion_plan.csv"
DEFAULT_EXPECTED_COVERAGE = ROOT / "results/metrics/surface_rbc_targeted_expansion_expected_coverage.csv"
DEFAULT_SOURCE_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json"
DEFAULT_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_surface_targeted_topup_v1_120.manifest.json"
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DEFAULT_SUMMARY = ROOT / "results/summaries/surface_rbc_targeted_expansion_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/surface_rbc_targeted_expansion_validation_metrics.csv"
DEFAULT_GROUP_SUMMARY = ROOT / "results/metrics/surface_rbc_targeted_expansion_group_summary.csv"
DEFAULT_REGISTRY_SUMMARY = ROOT / "results/summaries/surface_rbc_targeted_expansion_registry_summary.txt"

METRIC_FIELDS = [
    "sample_id",
    "split",
    "targeted_role",
    "edge_position_bin",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "schema_pass",
    "delta_max_abs_error",
    "defect_signal_norm",
    "projected_mask_area_px",
    "coverage_signature",
]
GROUP_FIELDS = ["group_key", "group_value", "sample_count", "schema_pass_count", "mean_delta_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate surface RBC targeted top-up NPZ.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--expected-coverage", type=Path, default=DEFAULT_EXPECTED_COVERAGE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--registry-summary", type=Path, default=DEFAULT_REGISTRY_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def scalar_str(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(-1)[0])


def string_list(value: np.ndarray) -> list[str]:
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def scalar_bool(value: np.ndarray, idx: int) -> bool:
    arr = np.asarray(value)
    return bool(arr.reshape(-1)[idx if arr.size > 1 else 0])


def scalar_float(value: np.ndarray, idx: int) -> float:
    arr = np.asarray(value)
    return float(arr.reshape(-1)[idx if arr.size > 1 else 0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(cwd: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def npz_staged(path: Path) -> bool:
    staged = git_value(ROOT, ["diff", "--cached", "--name-only"]).splitlines()
    try:
        rel = str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        rel = str(path)
    return rel in staged


def plan_signature_map(plan_csv: Path) -> dict[str, tuple[str, str, str, str, str]]:
    out: dict[str, tuple[str, str, str, str, str]] = {}
    for row in read_csv(plan_csv):
        out[row["sample_id"]] = (
            row["targeted_role"],
            row["depth_bin"],
            row["aspect_bin"],
            row["curvature_template"],
            row["edge_position_bin"],
        )
    return out


def validate_pack(path: Path, plan_csv: Path) -> tuple[list[dict[str, Any]], bool, dict[str, int], dict[str, int], dict[str, int]]:
    signatures = plan_signature_map(plan_csv)
    with np.load(path, allow_pickle=True) as npz:
        required = [
            "dataset_id",
            "schema_version",
            "route",
            "status",
            "delta_b",
            "b_defect",
            "b_no_defect",
            "axis_names",
            "sensor_x",
            "scan_line_y",
            "sensor_z_m",
            "sample_ids",
            "split",
            "rbc_params",
            "profile_pose",
            "profile_depth_grid_m",
            "profile_depth_map_xy_m",
            "projected_mask_2d",
            "geometry_method_used",
            "selected_solver_protocol",
            "mesh_auto_size",
            "material_fix_applied",
            "domain_material_audit_pass",
            "solver_probe_pass",
            "full_source_jscale",
            "exact_piao_rbc",
            "rbc_style_approximation",
            "depth_bin",
            "aspect_bin",
            "curvature_template",
            "targeted_role",
            "edge_position_bin",
        ]
        missing = [name for name in required if name not in npz.files]
        if missing:
            raise RuntimeError(f"missing NPZ fields: {missing}")
        dataset_id = scalar_str(npz["dataset_id"])
        schema_version = scalar_str(npz["schema_version"])
        route = scalar_str(npz["route"])
        sample_ids = string_list(npz["sample_ids"])
        split = string_list(npz["split"])
        curv = string_list(npz["curvature_template"])
        depth = string_list(npz["depth_bin"])
        aspect = string_list(npz["aspect_bin"])
        role = string_list(npz["targeted_role"])
        edge = string_list(npz["edge_position_bin"])
        methods = string_list(npz["geometry_method_used"])
        protocols = string_list(npz["selected_solver_protocol"])
        axis_names = string_list(npz["axis_names"])
        scan_line_y = [float(x) for x in np.asarray(npz["scan_line_y"]).reshape(-1).tolist()]
        sensor_z_m = float(np.asarray(npz["sensor_z_m"]).reshape(-1)[0])
        rows: list[dict[str, Any]] = []
        for idx, sample_id in enumerate(sample_ids):
            delta = np.asarray(npz["delta_b"][idx], dtype=float)
            defect = np.asarray(npz["b_defect"][idx], dtype=float)
            no_defect = np.asarray(npz["b_no_defect"][idx], dtype=float)
            delta_error = float(np.max(np.abs(delta - (defect - no_defect))))
            finite = bool(
                np.isfinite(delta).all()
                and np.isfinite(defect).all()
                and np.isfinite(no_defect).all()
                and np.isfinite(npz["rbc_params"][idx]).all()
                and np.isfinite(npz["profile_depth_grid_m"][idx]).all()
                and np.isfinite(npz["profile_depth_map_xy_m"][idx]).all()
            )
            norm = float(np.linalg.norm(delta))
            mask_area = int(np.asarray(npz["projected_mask_2d"][idx]).sum())
            signature = (role[idx], depth[idx], aspect[idx], curv[idx], edge[idx])
            plan_signature = signatures.get(sample_id)
            replacement_ok = sample_id.endswith("_repl01") or sample_id.endswith("_repl02") or sample_id.endswith("_repl03")
            signature_known = plan_signature == signature or replacement_ok
            schema_pass = (
                dataset_id == DATASET_ID
                and schema_version == SCHEMA_VERSION
                and route == NPZ_ROUTE
                and delta.shape == (3, 3, 201)
                and axis_names == ["Bx", "By", "Bz"]
                and len(npz["sensor_x"]) == 201
                and scan_line_y == [-0.001, 0.0, 0.001]
                and abs(sensor_z_m - 0.008) <= 1.0e-12
                and methods[idx] == "imported_watertight_mesh_solid"
                and protocols[idx] == "default"
                and int(np.asarray(npz["mesh_auto_size"]).reshape(-1)[idx]) == 5
                and abs(scalar_float(npz["full_source_jscale"], idx) - 1.0) <= 1.0e-12
                and scalar_bool(npz["material_fix_applied"], idx)
                and scalar_bool(npz["domain_material_audit_pass"], idx)
                and scalar_bool(npz["solver_probe_pass"], idx)
                and not scalar_bool(npz["exact_piao_rbc"], idx)
                and scalar_bool(npz["rbc_style_approximation"], idx)
                and finite
                and delta_error <= 1.0e-12
                and norm > 0.0
                and mask_area > 0
                and signature_known
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split[idx],
                    "targeted_role": role[idx],
                    "edge_position_bin": edge[idx],
                    "curvature_template": curv[idx],
                    "depth_bin": depth[idx],
                    "aspect_bin": aspect[idx],
                    "schema_pass": schema_pass,
                    "delta_max_abs_error": delta_error,
                    "defect_signal_norm": norm,
                    "projected_mask_area_px": mask_area,
                    "coverage_signature": "|".join(signature),
                }
            )
    validation_pass = len(sample_ids) == 120 and len(sample_ids) == len(set(sample_ids)) and all(bool(row["schema_pass"]) for row in rows)
    return rows, validation_pass, dict(Counter(split)), dict(Counter(curv)), dict(Counter(role))


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("split", "targeted_role", "edge_position_bin", "curvature_template", "depth_bin", "aspect_bin"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row[key])].append(row)
        for value, subset in sorted(buckets.items()):
            out.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "sample_count": len(subset),
                    "schema_pass_count": sum(1 for item in subset if bool(item["schema_pass"])),
                    "mean_delta_norm": float(np.mean([float(item["defect_signal_norm"]) for item in subset])) if subset else 0.0,
                }
            )
    return out


def build_manifest(
    *,
    n_success: int,
    validation_pass: bool,
    npz_sha256: str,
    pinn_commit: str,
    comsol_commit: str,
    npz_path: Path = DEFAULT_NPZ,
    manifest_path: Path = DEFAULT_MANIFEST,
    split_counts: dict[str, int] | None = None,
    curvature_counts: dict[str, int] | None = None,
    role_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_role": "topup_source",
        "status": "topup_generated" if n_success == 120 and validation_pass else "partial_topup_generated",
        "route": REGISTRY_ROUTE,
        "stage": "surface_rbc_targeted_expansion_v1",
        "schema_version": SCHEMA_VERSION,
        "geometry_method": "imported_watertight_mesh_solid",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "n_samples": n_success,
        "split_counts": split_counts or {},
        "curvature_counts": curvature_counts or {},
        "targeted_role_counts": role_counts or {},
        "axes": ["Bx", "By", "Bz"],
        "sensor_z_m": 0.008,
        "scan_line_y": [-0.001, 0.0, 0.001],
        "sensor_x_count": 201,
        "allowed_use": ["schema_validation", "explicit_surface_rbc_expansion_training_gate"],
        "forbidden_use": [
            "automatic_mainline_training",
            "baseline_update",
            "current_baseline_replacement",
            "latest_newest_auto_discovery",
            "direct_training_without_manifest_gate",
        ],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_commit": pinn_commit,
        "comsol_commit": comsol_commit,
        "generator_script": "COMSOL_Multiphysics_MCP/scripts/generate_mfl_surface_rbc_targeted_topup_pack.py",
        "validation_script": "PINN_project/scripts/validate_surface_rbc_targeted_expansion_pack.py",
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "npz_sha256": npz_sha256,
        "source_dataset_ids": [SOURCE_DATASET_ID],
        "source_manifest_paths": [str(DEFAULT_SOURCE_MANIFEST)],
        "merge_policy": "none_topup_source_only",
        "validation_pass": validation_pass,
        "train_ready_candidate": validation_pass and n_success == 120,
        "baseline_ready": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "creates_assembled_dataset": False,
        "notes": "Top-up source only. Later +120 training gate must explicitly assemble v3_240 + topup_v1_120.",
    }


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit(f"Run from PINN_project root: {ROOT}")
    check_no_overwrite([args.manifest, args.summary, args.metrics, args.group_summary, args.registry_summary], args.overwrite)
    rows, validation_pass, split_counts, curvature_counts, role_counts = validate_pack(args.npz, args.plan_csv)
    groups = group_rows(rows)
    write_csv(args.metrics, rows, METRIC_FIELDS)
    write_csv(args.group_summary, groups, GROUP_FIELDS)
    if not args.source_manifest.exists():
        raise FileNotFoundError(args.source_manifest)
    if npz_staged(args.npz):
        raise RuntimeError(f"NPZ is staged and must not be committed: {args.npz}")
    manifest = build_manifest(
        n_success=len(rows),
        validation_pass=validation_pass,
        npz_sha256=sha256_file(args.npz),
        pinn_commit=git_value(ROOT, ["rev-parse", "HEAD"]),
        comsol_commit=git_value(COMSOL_ROOT, ["rev-parse", "HEAD"]),
        npz_path=args.npz,
        manifest_path=args.manifest,
        split_counts=split_counts,
        curvature_counts=curvature_counts,
        role_counts=role_counts,
    )
    write_json(args.manifest, manifest)
    v120.update_registry(args.registry, [manifest])
    registry_text = args.registry.read_text(encoding="utf-8", errors="replace")
    registry_valid = DATASET_ID in registry_text and "baseline_ready: false" in registry_text and "explicit_surface_rbc_expansion_training_gate" in registry_text
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface RBC targeted expansion validation summary",
                "",
                f"dataset_id: {DATASET_ID}",
                "dataset_role: topup_source",
                f"n_success: {len(rows)}",
                f"validation_pass: {validation_pass}",
                f"split_counts: {split_counts}",
                f"curvature_counts: {curvature_counts}",
                f"targeted_role_counts: {role_counts}",
                f"npz_staged: {npz_staged(args.npz)}",
                "creates_assembled_dataset: false",
                "baseline_ready: false",
                f"manifest_path: {args.manifest}",
                f"npz_sha256: {manifest['npz_sha256']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text(
        "\n".join(
            [
                "surface RBC targeted expansion registry summary",
                "",
                f"registry_validation_pass: {registry_valid}",
                f"dataset_id: {DATASET_ID}",
                "dataset_role: topup_source",
                "baseline_ready: false",
                "auto_discovery_allowed: false",
                "latest_newest_discovery_allowed: false",
                "assembled_manifest_created: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not validation_pass or not registry_valid:
        raise RuntimeError("surface RBC targeted top-up validation failed")
    print(f"validation_pass={validation_pass} n_success={len(rows)}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
