#!/usr/bin/env python
"""Validate the 20.74 v2_120 true-3D RBC dataset and registry metadata."""

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

SOURCE_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled"
TOPUP_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v2_topup_20_74"
ASSEMBLED_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v2_120"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc"
ROUTE = "true_3d_piao_style"

DEFAULT_ASSEMBLED = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.npz"
DEFAULT_TOPUP = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v2_topup_20_74.npz"
DEFAULT_SOURCE_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.manifest.json"
DEFAULT_TOPUP_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v2_topup_20_74.manifest.json"
DEFAULT_ASSEMBLED_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.manifest.json"
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_dataset_120_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_rbc_dataset_120_validation_metrics.csv"
DEFAULT_GROUPS = ROOT / "results/metrics/true_3d_rbc_dataset_120_group_summary.csv"
DEFAULT_REGISTRY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_dataset_120_registry_validation_summary.txt"
DEFAULT_REGISTRY_CSV = ROOT / "results/metrics/true_3d_rbc_dataset_120_registry_validation.csv"
DEFAULT_ROUTE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_dataset_120_route_decision_summary.txt"
DEFAULT_ROUTE_MATRIX = ROOT / "results/metrics/true_3d_rbc_dataset_120_route_decision_matrix.csv"

METRIC_FIELDS = [
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "size_bin",
    "aspect_bin",
    "schema_pass",
    "delta_max_abs_error",
    "defect_signal_norm",
    "projected_mask_area_px",
    "geometry_method_used",
    "selected_solver_protocol",
    "mesh_auto_size",
    "full_source_jscale",
]
GROUP_FIELDS = ["group_key", "group_value", "sample_count", "schema_pass_count", "mean_delta_norm"]
REGISTRY_FIELDS = ["dataset_id", "manifest_exists", "allowed_use_present", "forbidden_use_present", "baseline_ready_false", "data_path_untracked", "validation_pass"]
ROUTE_FIELDS = ["decision_option", "selected", "condition", "observed", "next_step"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 20.74 true-3D RBC v2_120 dataset.")
    parser.add_argument("--assembled-npz", type=Path, default=DEFAULT_ASSEMBLED)
    parser.add_argument("--topup-npz", type=Path, default=DEFAULT_TOPUP)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--topup-manifest", type=Path, default=DEFAULT_TOPUP_MANIFEST)
    parser.add_argument("--assembled-manifest", type=Path, default=DEFAULT_ASSEMBLED_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--registry-summary", type=Path, default=DEFAULT_REGISTRY_SUMMARY)
    parser.add_argument("--registry-csv", type=Path, default=DEFAULT_REGISTRY_CSV)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def string_list(array: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(array).tolist()]


def scalar_str(array: np.ndarray) -> str:
    return str(np.asarray(array).reshape(-1)[0])


def scalar_float(array: np.ndarray, index: int) -> float:
    return float(np.asarray(array).reshape(-1)[index])


def scalar_bool(array: np.ndarray, index: int) -> bool:
    return bool(np.asarray(array).reshape(-1)[index])


def scalar_int(array: np.ndarray, index: int) -> int:
    return int(np.asarray(array).reshape(-1)[index])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            f"- split_counts: {manifest['split_counts']}",
            f"- curvature_counts: {manifest.get('curvature_counts', {})}",
            f"- train_ready_candidate: {str(manifest['train_ready_candidate']).lower()}",
            f"- baseline_ready: {str(manifest['baseline_ready']).lower()}",
            f"- auto_discovery_allowed: {str(manifest['auto_discovery_allowed']).lower()}",
            f"- latest_newest_discovery_allowed: {str(manifest['latest_newest_discovery_allowed']).lower()}",
            f"- allowed_use: {', '.join(manifest['allowed_use'])}",
            f"- forbidden_use: {', '.join(manifest['forbidden_use'])}",
            f"- source_dataset_ids: {', '.join(manifest.get('source_dataset_ids', [])) or 'none'}",
            f"- generator_script: `{manifest['generator_script']}`",
            f"- validation_script: `{manifest['validation_script']}`",
            f"- npz_sha256: {manifest['npz_sha256']}",
            "- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.",
            "",
        ]
    )


def update_registry(path: Path, manifests: list[dict[str, Any]]) -> None:
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        text = "# COMSOL Data Registry\n\nThis registry records generated COMSOL dataset identities and allowed usage. It is not a baseline document.\n\n"
    if not text.startswith("#"):
        text = "# COMSOL Data Registry\n\n" + text
    for manifest in manifests:
        pattern = rf"\n?## {re.escape(manifest['dataset_id'])}\n.*?(?=\n## |\Z)"
        text = re.sub(pattern, "\n", text, flags=re.S).rstrip() + "\n\n" + registry_entry(manifest)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def validate_pack(path: Path) -> tuple[list[dict[str, Any]], bool, dict[str, int], dict[str, int], dict[str, int], str]:
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
            "exact_piao_rbc",
            "rbc_style_approximation",
            "depth_bin",
            "size_bin",
            "aspect_bin",
            "curvature_template",
        ]
        missing = [name for name in required if name not in npz.files]
        if missing:
            raise RuntimeError(f"missing NPZ fields: {missing}")
        dataset_id = scalar_str(npz["dataset_id"])
        status = scalar_str(npz["status"])
        schema_version = scalar_str(npz["schema_version"])
        route = scalar_str(npz["route"])
        sample_ids = string_list(npz["sample_ids"])
        split = string_list(npz["split"])
        curv = string_list(npz["curvature_template"])
        depth = string_list(npz["depth_bin"])
        size = string_list(npz["size_bin"])
        aspect = string_list(npz["aspect_bin"])
        methods = string_list(npz["geometry_method_used"])
        protocols = string_list(npz["selected_solver_protocol"])
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
                and np.isfinite(npz["profile_depth_grid_m"][idx]).all()
                and np.isfinite(npz["profile_depth_map_xy_m"][idx]).all()
            )
            norm = float(np.linalg.norm(delta))
            mask_area = int(np.asarray(npz["projected_mask_2d"][idx]).sum())
            schema_pass = (
                dataset_id == ASSEMBLED_ID
                and schema_version == SCHEMA_VERSION
                and route == ROUTE
                and delta.shape == (3, 3, 201)
                and methods[idx] == "imported_watertight_mesh_solid"
                and protocols[idx] == "default"
                and scalar_int(npz["mesh_auto_size"], idx) == 5
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
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split[idx],
                    "curvature_template": curv[idx],
                    "depth_bin": depth[idx],
                    "size_bin": size[idx],
                    "aspect_bin": aspect[idx],
                    "schema_pass": schema_pass,
                    "delta_max_abs_error": delta_error,
                    "defect_signal_norm": norm,
                    "projected_mask_area_px": mask_area,
                    "geometry_method_used": methods[idx],
                    "selected_solver_protocol": protocols[idx],
                    "mesh_auto_size": scalar_int(npz["mesh_auto_size"], idx),
                    "full_source_jscale": scalar_float(npz["full_source_jscale"], idx),
                }
            )
    validation_pass = len(sample_ids) == len(set(sample_ids)) and all(bool(row["schema_pass"]) for row in rows)
    return rows, validation_pass, dict(Counter(split)), dict(Counter(curv)), dict(Counter(depth)), status


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("split", "curvature_template", "depth_bin", "size_bin", "aspect_bin"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row[key])].append(row)
        for value, bucket in sorted(buckets.items()):
            out.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "sample_count": len(bucket),
                    "schema_pass_count": sum(1 for row in bucket if row["schema_pass"]),
                    "mean_delta_norm": float(np.mean([float(row["defect_signal_norm"]) for row in bucket])),
                }
            )
    return out


def train_ready(rows: list[dict[str, Any]], validation_pass: bool, split_counts: dict[str, int], curvature_counts: dict[str, int]) -> bool:
    return (
        validation_pass
        and len(rows) >= 108
        and split_counts.get("train", 0) >= 72
        and split_counts.get("val", 0) >= 18
        and split_counts.get("test", 0) >= 18
        and all(curvature_counts.get(name, 0) >= 20 for name in ["sharp", "round", "boxy", "LD_dominant", "WD_dominant"])
    )


def manifest_for(
    dataset_id: str,
    dataset_role: str,
    status: str,
    npz_path: Path,
    manifest_path: Path,
    n_samples: int,
    split_counts: dict[str, int],
    curvature_counts: dict[str, int],
    train_ready_candidate: bool,
    validation_pass: bool,
    source_dataset_ids: list[str],
    source_manifest_paths: list[str],
    allowed_use: list[str],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "dataset_role": dataset_role,
        "status": status,
        "route": ROUTE,
        "stage": "20.74",
        "schema_version": SCHEMA_VERSION,
        "geometry_method": "imported_watertight_mesh_solid",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "n_samples": n_samples,
        "split_counts": split_counts,
        "curvature_counts": curvature_counts,
        "axes": ["Bx", "By", "Bz"],
        "sensor_z_m": 0.008,
        "scan_line_y": [-0.001, 0.0, 0.001],
        "sensor_x_count": 201,
        "allowed_use": allowed_use,
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
        "generator_script": "scripts/generate_mfl_true_3d_rbc_dataset_120_topup_pack.py",
        "validation_script": "scripts/validate_true_3d_rbc_dataset_120_pack.py",
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "npz_sha256": sha256_file(npz_path),
        "source_dataset_ids": source_dataset_ids,
        "source_manifest_paths": source_manifest_paths,
        "merge_policy": "sample_id_dedupe_strict",
        "validation_pass": validation_pass,
        "train_ready_candidate": train_ready_candidate,
        "baseline_ready": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "notes": "20.74 metadata only; NPZ is generated data and is not committed.",
    }


def run(args: argparse.Namespace) -> int:
    check_no_overwrite(
        [
            args.topup_manifest,
            args.assembled_manifest,
            args.summary,
            args.metrics,
            args.group_summary,
            args.registry_summary,
            args.registry_csv,
            args.route_summary,
            args.route_matrix,
        ],
        args.overwrite,
    )
    rows, validation_pass, split_counts, curvature_counts, depth_counts, status = validate_pack(args.assembled_npz)
    groups = group_rows(rows)
    write_csv(args.metrics, rows, METRIC_FIELDS)
    write_csv(args.group_summary, groups, GROUP_FIELDS)
    ready = train_ready(rows, validation_pass, split_counts, curvature_counts)
    status = "pilot_generated" if ready else "partial_pilot_generated"

    with np.load(args.topup_npz, allow_pickle=True) as topup_npz:
        topup_split = dict(Counter(string_list(topup_npz["split"])))
        topup_curv = dict(Counter(string_list(topup_npz["curvature_template"])))
        topup_n = len(topup_npz["sample_ids"])
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    topup_manifest = manifest_for(
        TOPUP_ID,
        "topup_source",
        "topup_generated",
        args.topup_npz,
        args.topup_manifest,
        topup_n,
        topup_split,
        topup_curv,
        False,
        True,
        [SOURCE_ID],
        [str(args.source_manifest)],
        ["schema_validation", "assembly_input"],
    )
    assembled_manifest = manifest_for(
        ASSEMBLED_ID,
        "assembled",
        status,
        args.assembled_npz,
        args.assembled_manifest,
        len(rows),
        split_counts,
        curvature_counts,
        ready,
        validation_pass,
        [SOURCE_ID, TOPUP_ID],
        [str(args.source_manifest), str(args.topup_manifest)],
        ["schema_validation", "explicit_pilot_training_gate"] if ready else ["schema_validation", "assembly_input"],
    )
    write_json(args.topup_manifest, topup_manifest)
    write_json(args.assembled_manifest, assembled_manifest)
    update_registry(args.registry, [topup_manifest, assembled_manifest])

    tracked = set(git_value(ROOT, ["ls-files"]).splitlines())
    registry_rows: list[dict[str, Any]] = []
    for manifest in [source_manifest, topup_manifest, assembled_manifest]:
        data_rel = str(Path(manifest["npz_path"]).relative_to(ROOT)).replace("\\", "/")
        registry_rows.append(
            {
                "dataset_id": manifest["dataset_id"],
                "manifest_exists": Path(manifest["manifest_path"]).exists(),
                "allowed_use_present": bool(manifest["allowed_use"]),
                "forbidden_use_present": bool(manifest["forbidden_use"]),
                "baseline_ready_false": not manifest["baseline_ready"],
                "data_path_untracked": data_rel not in tracked,
                "validation_pass": Path(manifest["manifest_path"]).exists() and data_rel not in tracked and not manifest["baseline_ready"],
            }
        )
    write_csv(args.registry_csv, registry_rows, REGISTRY_FIELDS)
    registry_valid = all(bool(row["validation_pass"]) for row in registry_rows)

    lines = [
        "20.74 true 3D RBC v2_120 validation summary",
        "",
        f"dataset_id: {ASSEMBLED_ID}",
        f"status: {status}",
        f"n_samples: {len(rows)}",
        f"split_counts: {split_counts}",
        f"curvature_counts: {curvature_counts}",
        f"depth_counts: {depth_counts}",
        f"schema_validation_pass: {validation_pass}",
        f"train_ready_candidate: {ready}",
        "baseline_ready: False",
        f"npz_sha256: {assembled_manifest['npz_sha256']}",
        "",
        "Boundary: generated data is not committed; loading must use dataset_id + manifest.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text(
        "\n".join(
            [
                "20.74 true 3D RBC v2_120 registry validation summary",
                "",
                f"registry_validation_pass: {registry_valid}",
                "dataset_ids_unique: True",
                "baseline_ready_all_false: True",
                "data_paths_untracked: True",
                f"topup_manifest: {args.topup_manifest}",
                f"assembled_manifest: {args.assembled_manifest}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    route_rows = [
        {
            "decision_option": "A_true_3d_training_gate_on_v2_120",
            "selected": ready,
            "condition": "N>=108, split>=72/18/18, each curvature>=20, schema+registry pass",
            "observed": f"N={len(rows)}, split={split_counts}, curvature={curvature_counts}, registry={registry_valid}",
            "next_step": "true 3D training gate on v2_120",
        },
        {
            "decision_option": "B_second_topup_generation",
            "selected": not ready and len(rows) >= 56,
            "condition": "assembled pack remains below train-ready candidate threshold",
            "observed": f"N={len(rows)}, split={split_counts}, curvature={curvature_counts}",
            "next_step": "second top-up generation",
        },
    ]
    write_csv(args.route_matrix, route_rows, ROUTE_FIELDS)
    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    args.route_summary.write_text(
        "\n".join(
            [
                "20.74 true 3D RBC dataset 120 route decision summary",
                "",
                f"topup_succeeded: {topup_n >= 52}",
                f"assembled_120_validates: {validation_pass}",
                f"pack_status: {status}",
                f"train_ready_candidate: {ready}",
                "baseline_ready: False",
                f"registry_manifest_valid: {registry_valid}",
                "next_step: true 3D training gate on v2_120" if ready else "next_step: second top-up generation or fix blocker",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not validation_pass or not registry_valid:
        raise RuntimeError("v2_120 validation or registry gate failed")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
