#!/usr/bin/env python
"""Validate the 20.71 true-3D RBC imported-watertight pilot pack.

This script also writes the lightweight tracked registry/manifest metadata.
It never stages or commits generated data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMSOL_ROOT = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")
DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc"
ROUTE = "true_3d_piao_style"
DEFAULT_NPZ = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v1.npz"
DEFAULT_PLAN = ROOT / "results/metrics/true_3d_rbc_pilot_pack_plan.csv"
DEFAULT_MESH_METRICS = ROOT / "results/metrics/true_3d_rbc_pilot_watertight_mesh_metrics.csv"
DEFAULT_COMSOL_INVENTORY = COMSOL_ROOT / "results/inventory_true_3d_rbc_pilot_pack_v1.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_pack_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_rbc_pilot_pack_validation_metrics.csv"
DEFAULT_GROUP_SUMMARY = ROOT / "results/metrics/true_3d_rbc_pilot_pack_group_summary.csv"
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DEFAULT_REGISTRY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_dataset_registry_summary.txt"
DEFAULT_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1.manifest.json"
DEFAULT_ROUTE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_pack_route_decision_summary.txt"
DEFAULT_ROUTE_MATRIX = ROOT / "results/metrics/true_3d_rbc_pilot_pack_route_decision_matrix.csv"
DEFAULT_LOCAL_SIDECAR = DEFAULT_NPZ.with_suffix(".manifest.json")

METRIC_FIELDS = [
    "sample_id",
    "split",
    "schema_pass",
    "geometry_method_used",
    "exact_piao_rbc",
    "rbc_style_approximation",
    "delta_b_shape",
    "b_defect_shape",
    "b_no_defect_shape",
    "axis_names",
    "axis_expressions",
    "all_values_finite",
    "delta_max_abs_error",
    "defect_signal_norm",
    "defect_signal_nonzero",
    "Bx_norm",
    "By_norm",
    "Bz_norm",
    "projected_mask_area_px",
    "depth_max_m",
    "param_D_m",
    "profile_depth_max_error_vs_param_D",
    "selected_solver_protocol",
    "mesh_auto_size",
    "material_fix_applied",
    "domain_material_audit_pass",
    "solver_probe_pass",
    "full_source_jscale",
    "no_defect_reused",
    "mesh_source",
    "mesh_units",
    "depth_bin",
    "size_bin",
    "aspect_bin",
    "curvature_template",
    "notes",
]

GROUP_FIELDS = ["group_key", "group_value", "sample_count", "schema_pass_count", "mean_delta_norm"]
ROUTE_FIELDS = ["decision_option", "selected", "condition", "observed", "next_step"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 20.71 true-3D RBC pilot NPZ.")
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--mesh-metrics", type=Path, default=DEFAULT_MESH_METRICS)
    parser.add_argument("--comsol-inventory", type=Path, default=DEFAULT_COMSOL_INVENTORY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--registry-summary", type=Path, default=DEFAULT_REGISTRY_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--local-sidecar", type=Path, default=DEFAULT_LOCAL_SIDECAR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def git_value(cwd: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def arr_shape(value: np.ndarray) -> str:
    return json_dumps(list(value.shape))


def string_list(array: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(array).tolist()]


def scalar_bool(array: np.ndarray, index: int) -> bool:
    return bool(np.asarray(array).reshape(-1)[index])


def scalar_float(array: np.ndarray, index: int) -> float:
    return float(np.asarray(array).reshape(-1)[index])


def scalar_int(array: np.ndarray, index: int) -> int:
    return int(np.asarray(array).reshape(-1)[index])


def scalar_str(array: np.ndarray, index: int) -> str:
    return str(np.asarray(array).reshape(-1)[index])


def update_registry(path: Path, manifest: dict[str, Any]) -> None:
    entry = "\n".join(
        [
            f"## {manifest['dataset_id']}",
            "",
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
            f"- split_counts: {manifest['split_counts']}",
            f"- allowed_use: {', '.join(manifest['allowed_use'])}",
            f"- forbidden_use: {', '.join(manifest['forbidden_use'])}",
            f"- generator_script: `{manifest['generator_script']}`",
            f"- validation_script: `{manifest['validation_script']}`",
            f"- pinn_commit: {manifest['pinn_commit']}",
            f"- comsol_commit: {manifest['comsol_commit']}",
            f"- inventory_status_counts: {manifest['inventory_status_counts']}",
            f"- missing_curvature_templates: {', '.join(manifest['missing_curvature_templates']) or 'none'}",
            f"- comsol_worktree_status_entries: {len(manifest['comsol_git_status_short'])}",
            f"- npz_sha256: {manifest['npz_sha256']}",
            "- notes: Pilot pack metadata only. This dataset is not a baseline and must not be loaded by latest/newest auto-discovery. See manifest for worktree status and partial-pack blockers.",
            "",
        ]
    )
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        pattern = rf"\n?## {re.escape(manifest['dataset_id'])}\n.*?(?=\n## |\Z)"
        text = re.sub(pattern, "\n", text, flags=re.S)
        if not text.strip():
            text = "# COMSOL Data Registry\n\n"
        if not text.startswith("#"):
            text = "# COMSOL Data Registry\n\n" + text
        new_text = text.rstrip() + "\n\n" + entry
    else:
        new_text = "# COMSOL Data Registry\n\nThis registry records generated COMSOL dataset identities and allowed usage. It is not a baseline document.\n\n" + entry
    path.write_text(new_text, encoding="utf-8")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate(args: argparse.Namespace) -> int:
    check_no_overwrite(
        [
            args.summary,
            args.metrics,
            args.group_summary,
            args.registry_summary,
            args.manifest,
            args.route_summary,
            args.route_matrix,
            args.local_sidecar,
        ],
        args.overwrite,
    )
    required = [
        "dataset_id",
        "schema_version",
        "route",
        "delta_b",
        "b_defect",
        "b_no_defect",
        "axis_names",
        "axis_expressions",
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
        "no_defect_reused",
        "exact_piao_rbc",
        "rbc_style_approximation",
        "mesh_source",
        "mesh_units",
        "depth_bin",
        "size_bin",
        "aspect_bin",
        "curvature_template",
    ]
    with np.load(args.npz_path, allow_pickle=True) as npz:
        missing = [name for name in required if name not in npz.files]
        if missing:
            raise RuntimeError(f"missing NPZ fields: {missing}")
        dataset_id = scalar_str(npz["dataset_id"], 0)
        schema_version = scalar_str(npz["schema_version"], 0)
        route = scalar_str(npz["route"], 0)
        delta_b = npz["delta_b"]
        b_defect = npz["b_defect"]
        b_no = npz["b_no_defect"]
        axis_names = string_list(npz["axis_names"])
        axis_expr = string_list(npz["axis_expressions"])
        sample_ids = string_list(npz["sample_ids"])
        split = string_list(npz["split"])
        rbc_params = npz["rbc_params"]
        depth_grids = npz["profile_depth_grid_m"]
        depth_maps = npz["profile_depth_map_xy_m"]
        masks = npz["projected_mask_2d"]
        methods = string_list(npz["geometry_method_used"])
        depth_bins = string_list(npz["depth_bin"])
        size_bins = string_list(npz["size_bin"])
        aspect_bins = string_list(npz["aspect_bin"])
        curvatures = string_list(npz["curvature_template"])
        metrics: list[dict[str, Any]] = []
        for idx, sample_id in enumerate(sample_ids):
            delta = delta_b[idx]
            defect = b_defect[idx]
            no = b_no[idx]
            recomputed = defect - no
            delta_error = float(np.max(np.abs(delta - recomputed)))
            finite = bool(
                np.isfinite(delta).all()
                and np.isfinite(defect).all()
                and np.isfinite(no).all()
                and np.isfinite(depth_grids[idx]).all()
                and np.isfinite(depth_maps[idx]).all()
            )
            norm = float(np.linalg.norm(delta))
            axis_norms = [float(np.linalg.norm(delta[axis_index])) for axis_index in range(delta.shape[0])]
            params = rbc_params[idx]
            param_d = float(params.reshape(-1)[2])
            depth_max = float(max(float(depth_grids[idx].max()), float(depth_maps[idx].max())))
            schema_pass = (
                dataset_id == DATASET_ID
                and schema_version == SCHEMA_VERSION
                and route == ROUTE
                and delta.shape == (3, 3, 201)
                and defect.shape == (3, 3, 201)
                and no.shape == (3, 3, 201)
                and axis_names == ["Bx", "By", "Bz"]
                and methods[idx] == "imported_watertight_mesh_solid"
                and not scalar_bool(npz["exact_piao_rbc"], idx)
                and scalar_bool(npz["rbc_style_approximation"], idx)
                and scalar_str(npz["selected_solver_protocol"], idx) == "default"
                and scalar_int(npz["mesh_auto_size"], idx) == 5
                and scalar_bool(npz["material_fix_applied"], idx)
                and scalar_bool(npz["domain_material_audit_pass"], idx)
                and scalar_bool(npz["solver_probe_pass"], idx)
                and abs(scalar_float(npz["full_source_jscale"], idx) - 1.0) <= 1.0e-12
                and scalar_bool(npz["no_defect_reused"], idx)
                and masks[idx].shape == (64, 128)
                and int(masks[idx].sum()) > 0
                and depth_grids[idx].shape == (33, 17)
                and depth_maps[idx].shape == (64, 128)
                and finite
                and delta_error <= 1.0e-12
                and norm > 0.0
                and abs(depth_max - param_d) <= 0.03 * param_d
            )
            metrics.append(
                {
                    "sample_id": sample_id,
                    "split": split[idx],
                    "schema_pass": schema_pass,
                    "geometry_method_used": methods[idx],
                    "exact_piao_rbc": scalar_bool(npz["exact_piao_rbc"], idx),
                    "rbc_style_approximation": scalar_bool(npz["rbc_style_approximation"], idx),
                    "delta_b_shape": arr_shape(delta),
                    "b_defect_shape": arr_shape(defect),
                    "b_no_defect_shape": arr_shape(no),
                    "axis_names": json_dumps(axis_names),
                    "axis_expressions": json_dumps(axis_expr),
                    "all_values_finite": finite,
                    "delta_max_abs_error": delta_error,
                    "defect_signal_norm": norm,
                    "defect_signal_nonzero": norm > 0.0,
                    "Bx_norm": axis_norms[0],
                    "By_norm": axis_norms[1],
                    "Bz_norm": axis_norms[2],
                    "projected_mask_area_px": int(masks[idx].sum()),
                    "depth_max_m": depth_max,
                    "param_D_m": param_d,
                    "profile_depth_max_error_vs_param_D": abs(depth_max - param_d),
                    "selected_solver_protocol": scalar_str(npz["selected_solver_protocol"], idx),
                    "mesh_auto_size": scalar_int(npz["mesh_auto_size"], idx),
                    "material_fix_applied": scalar_bool(npz["material_fix_applied"], idx),
                    "domain_material_audit_pass": scalar_bool(npz["domain_material_audit_pass"], idx),
                    "solver_probe_pass": scalar_bool(npz["solver_probe_pass"], idx),
                    "full_source_jscale": scalar_float(npz["full_source_jscale"], idx),
                    "no_defect_reused": scalar_bool(npz["no_defect_reused"], idx),
                    "mesh_source": scalar_str(npz["mesh_source"], idx),
                    "mesh_units": scalar_str(npz["mesh_units"], idx),
                    "depth_bin": depth_bins[idx],
                    "size_bin": size_bins[idx],
                    "aspect_bin": aspect_bins[idx],
                    "curvature_template": curvatures[idx],
                    "notes": "projected mask is comparator; 3D label is RBC params plus depth grid/map",
                }
            )
    write_csv(args.metrics, metrics, METRIC_FIELDS)
    pass_count = sum(1 for row in metrics if row["schema_pass"])
    split_counts = dict(Counter(row["split"] for row in metrics))
    group_rows: list[dict[str, Any]] = []
    for key in ("split", "depth_bin", "size_bin", "curvature_template"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in metrics:
            groups[str(row[key])].append(row)
        for value, rows in groups.items():
            group_rows.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "sample_count": len(rows),
                    "schema_pass_count": sum(1 for row in rows if row["schema_pass"]),
                    "mean_delta_norm": float(np.mean([float(row["defect_signal_norm"]) for row in rows])),
                }
            )
    write_csv(args.group_summary, group_rows, GROUP_FIELDS)
    plan_rows = read_csv(args.plan_csv)
    inventory_rows = read_csv(args.comsol_inventory)
    inventory_status_counts = dict(Counter(row.get("status", "") for row in inventory_rows))
    expected_curvatures = sorted({row["curvature_template"] for row in plan_rows})
    observed_curvatures = sorted({row["curvature_template"] for row in metrics})
    missing_curvatures = [name for name in expected_curvatures if name not in observed_curvatures]
    pinn_git_status = [line for line in git_value(ROOT, ["status", "--short"]).splitlines() if line.strip()]
    comsol_git_status = [line for line in git_value(COMSOL_ROOT, ["status", "--short"]).splitlines() if line.strip()]
    validation_pass = bool(metrics) and pass_count == len(metrics)
    status = "pilot_generated" if validation_pass and len(metrics) >= 54 else "partial_pilot_generated" if validation_pass and len(metrics) >= 30 else "validation_failed"
    train_ready = status == "pilot_generated"
    baseline_ready = False
    npz_sha = sha256_file(args.npz_path)
    sample_ids_sha = hashlib.sha256("\n".join(sorted(row["sample_id"] for row in metrics)).encode("utf-8")).hexdigest()
    manifest = {
        "dataset_id": DATASET_ID,
        "route": ROUTE,
        "stage": "20.71",
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "geometry_method": "imported_watertight_mesh_solid",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "n_samples": len(metrics),
        "schema_pass_count": pass_count,
        "split_counts": split_counts,
        "inventory_status_counts": inventory_status_counts,
        "expected_curvature_templates": expected_curvatures,
        "observed_curvature_templates": observed_curvatures,
        "missing_curvature_templates": missing_curvatures,
        "axes": ["Bx", "By", "Bz"],
        "sensor_z_m": 0.008,
        "scan_line_y": [-0.001, 0.0, 0.001],
        "sensor_x_count": 201,
        "allowed_use": ["schema_validation", "explicit_pilot_training_gate"],
        "forbidden_use": ["automatic_mainline_training", "baseline_update", "current_baseline_replacement"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_commit": git_value(ROOT, ["rev-parse", "HEAD"]),
        "pinn_branch": git_value(ROOT, ["branch", "--show-current"]),
        "pinn_git_status_short": pinn_git_status,
        "comsol_commit": git_value(COMSOL_ROOT, ["rev-parse", "HEAD"]),
        "comsol_branch": git_value(COMSOL_ROOT, ["branch", "--show-current"]),
        "comsol_git_status_short": comsol_git_status,
        "generator_script": "COMSOL_Multiphysics_MCP/scripts/generate_mfl_true_3d_rbc_pilot_pack.py",
        "validation_script": "PINN_project/scripts/validate_true_3d_rbc_pilot_pack.py",
        "plan_csv": str(args.plan_csv),
        "mesh_metrics_csv": str(args.mesh_metrics),
        "comsol_inventory_csv": str(args.comsol_inventory),
        "validation_summary_path": str(args.summary),
        "validation_metrics_path": str(args.metrics),
        "manifest_path": str(args.manifest),
        "npz_path": str(args.npz_path),
        "npz_sha256": npz_sha,
        "sample_ids_sha256": sample_ids_sha,
        "train_ready": train_ready,
        "baseline_ready": baseline_ready,
        "notes": "Generated data remains under data/ and must not be committed. This is a partial pilot pack; missing curvature templates and not-attempted inventory rows require top-up before any training gate.",
    }
    write_manifest(args.manifest, manifest)
    write_manifest(args.local_sidecar, manifest)
    update_registry(args.registry, manifest)
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text(
        "\n".join(
            [
                "20.71 true 3D RBC pilot dataset registry summary",
                "",
                f"dataset_id: {DATASET_ID}",
                f"registry_path: {args.registry}",
                f"manifest_path: {args.manifest}",
                f"local_sidecar_manifest: {args.local_sidecar}",
                f"status: {status}",
                f"inventory_status_counts: {inventory_status_counts}",
                f"observed_curvature_templates: {observed_curvatures}",
                f"missing_curvature_templates: {missing_curvatures}",
                f"allowed_use: {manifest['allowed_use']}",
                f"forbidden_use: {manifest['forbidden_use']}",
                f"pinn_git_status_short: {pinn_git_status}",
                f"comsol_git_status_short: {comsol_git_status}",
                f"npz_sha256: {npz_sha}",
                "",
                "Boundary: registry/manifest is metadata only; NPZ and sidecar under data/ are not committed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    route_rows = [
        {
            "decision_option": "A_full_pilot_pack_validates",
            "selected": status == "pilot_generated",
            "condition": "N>=54 and validation passes",
            "observed": f"N={len(metrics)}, schema_pass={pass_count}, status={status}",
            "next_step": "true 3D training gate",
        },
        {
            "decision_option": "B_partial_pilot_top_up",
            "selected": status == "partial_pilot_generated",
            "condition": "30<=N<54 and validation passes",
            "observed": f"N={len(metrics)}, schema_pass={pass_count}, status={status}",
            "next_step": "top-up generation before training",
        },
        {
            "decision_option": "E_registry_or_schema_fix",
            "selected": status == "validation_failed",
            "condition": "schema, registry, or manifest validation fails",
            "observed": f"N={len(metrics)}, schema_pass={pass_count}, status={status}",
            "next_step": "fix schema/registry/manifest before training",
        },
    ]
    write_csv(args.route_matrix, route_rows, ROUTE_FIELDS)
    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    args.route_summary.write_text(
        "\n".join(
            [
                "20.71 true 3D RBC pilot pack route decision summary",
                "",
                f"dataset_id: {DATASET_ID}",
                f"pack_generated: {status in {'pilot_generated', 'partial_pilot_generated'}}",
                f"pack_status: {status}",
                f"n_samples: {len(metrics)}",
                f"schema_pass_count: {pass_count}",
                f"split_counts: {split_counts}",
                f"inventory_status_counts: {inventory_status_counts}",
                f"observed_curvature_templates: {observed_curvatures}",
                f"missing_curvature_templates: {missing_curvatures}",
                f"train_ready: {train_ready}",
                f"baseline_ready: {baseline_ready}",
                "registry_in_place: True",
                "manifest_in_place: True",
                "next_step: true 3D training gate"
                if train_ready
                else "next_step: top-up generation"
                if status == "partial_pilot_generated"
                else "next_step: fix schema/registry/manifest",
                "",
                "Boundary: no training, no refinement, no baseline update, no data artifact commit.",
                "Top-up blocker: missing curvature templates must be generated before any training gate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "20.71 true 3D RBC pilot pack validation summary",
                "",
                f"dataset_id: {DATASET_ID}",
                f"status: {status}",
                f"sample_count: {len(metrics)}",
                f"schema_pass_count: {pass_count}",
                f"split_counts: {split_counts}",
                f"inventory_status_counts: {inventory_status_counts}",
                f"observed_curvature_templates: {observed_curvatures}",
                f"missing_curvature_templates: {missing_curvatures}",
                "geometry_method: imported_watertight_mesh_solid",
                "exact_piao_rbc: False",
                "rbc_style_approximation: True",
                "axis_names: [Bx, By, Bz]",
                "selected_solver_protocol: default",
                "mesh_auto_size: 5",
                "material_fix_applied: True",
                "full_source_jscale: 1.0",
                "no_defect_reused: True",
                f"npz_sha256: {npz_sha}",
                "",
                "Validation result:",
                "PASS" if validation_pass else "FAIL",
                "",
                "Boundary: this validation authorizes at most an explicit pilot training gate; it does not update CURRENT_BASELINE.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not validation_pass or len(metrics) < 30:
        raise RuntimeError(f"20.71 validation failed: n={len(metrics)}, pass_count={pass_count}")
    return 0


def main() -> int:
    return validate(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
