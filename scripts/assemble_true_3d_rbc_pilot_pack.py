#!/usr/bin/env python
"""Assemble and validate the 20.72 true-3D RBC pilot pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMSOL_ROOT = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")

PARTIAL_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_partial_20_71"
TOPUP_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_topup_20_72"
ASSEMBLED_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc"
ROUTE = "true_3d_piao_style"

DEFAULT_PARTIAL = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v1.npz"
DEFAULT_TOPUP = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v1_topup.npz"
DEFAULT_ASSEMBLED = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.npz"
DEFAULT_PARTIAL_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1.manifest.json"
DEFAULT_TOPUP_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1_topup.manifest.json"
DEFAULT_ASSEMBLED_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.manifest.json"
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"

DEFAULT_ASSEMBLY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_assembled_summary.txt"
DEFAULT_ASSEMBLY_INDEX = ROOT / "results/metrics/true_3d_rbc_pilot_assembled_index.csv"
DEFAULT_ASSEMBLY_GROUPS = ROOT / "results/metrics/true_3d_rbc_pilot_assembled_group_summary.csv"
DEFAULT_VALIDATION_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_assembled_validation_summary.txt"
DEFAULT_VALIDATION_METRICS = ROOT / "results/metrics/true_3d_rbc_pilot_assembled_validation_metrics.csv"
DEFAULT_VALIDATION_GROUPS = ROOT / "results/metrics/true_3d_rbc_pilot_assembled_validation_group_summary.csv"
DEFAULT_REGISTRY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_dataset_registry_summary.txt"
DEFAULT_REGISTRY_VALIDATION_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_registry_validation_summary.txt"
DEFAULT_REGISTRY_VALIDATION = ROOT / "results/metrics/true_3d_rbc_pilot_registry_validation.csv"
DEFAULT_ROUTE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_topup_route_decision_summary.txt"
DEFAULT_ROUTE_MATRIX = ROOT / "results/metrics/true_3d_rbc_pilot_topup_route_decision_matrix.csv"

INDEX_FIELDS = ["source_pack", "sample_id", "split", "curvature_template", "depth_bin", "size_bin", "schema_pass"]
GROUP_FIELDS = ["group_key", "group_value", "sample_count", "schema_pass_count", "mean_delta_norm"]
VALIDATION_FIELDS = [
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "size_bin",
    "schema_pass",
    "delta_max_abs_error",
    "defect_signal_norm",
    "projected_mask_area_px",
    "geometry_method_used",
    "selected_solver_protocol",
]
ROUTE_FIELDS = ["decision_option", "selected", "condition", "observed", "next_step"]
REGISTRY_VALIDATION_FIELDS = ["dataset_id", "manifest_exists", "allowed_use_present", "forbidden_use_present", "baseline_ready_false", "data_path_untracked", "validation_pass"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble 20.71 partial + 20.72 top-up true-3D RBC pack.")
    parser.add_argument("--partial-npz", type=Path, default=DEFAULT_PARTIAL)
    parser.add_argument("--topup-npz", type=Path, default=DEFAULT_TOPUP)
    parser.add_argument("--assembled-npz", type=Path, default=DEFAULT_ASSEMBLED)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--partial-manifest", type=Path, default=DEFAULT_PARTIAL_MANIFEST)
    parser.add_argument("--topup-manifest", type=Path, default=DEFAULT_TOPUP_MANIFEST)
    parser.add_argument("--assembled-manifest", type=Path, default=DEFAULT_ASSEMBLED_MANIFEST)
    parser.add_argument("--assembly-summary", type=Path, default=DEFAULT_ASSEMBLY_SUMMARY)
    parser.add_argument("--assembly-index", type=Path, default=DEFAULT_ASSEMBLY_INDEX)
    parser.add_argument("--assembly-groups", type=Path, default=DEFAULT_ASSEMBLY_GROUPS)
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--validation-metrics", type=Path, default=DEFAULT_VALIDATION_METRICS)
    parser.add_argument("--validation-groups", type=Path, default=DEFAULT_VALIDATION_GROUPS)
    parser.add_argument("--registry-summary", type=Path, default=DEFAULT_REGISTRY_SUMMARY)
    parser.add_argument("--registry-validation-summary", type=Path, default=DEFAULT_REGISTRY_VALIDATION_SUMMARY)
    parser.add_argument("--registry-validation", type=Path, default=DEFAULT_REGISTRY_VALIDATION)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scalar_str(array: np.ndarray, index: int = 0) -> str:
    return str(np.asarray(array).reshape(-1)[index])


def concat_npz(partial: dict[str, np.ndarray], topup: dict[str, np.ndarray], status: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    n_partial = len(partial["sample_ids"])
    n_topup = len(topup["sample_ids"])
    for key, value in partial.items():
        if key in {"dataset_id", "status"}:
            continue
        if key not in topup:
            raise RuntimeError(f"top-up NPZ missing key {key}")
        top_value = topup[key]
        if isinstance(value, np.ndarray) and value.shape[:1] == (n_partial,) and top_value.shape[:1] == (n_topup,):
            out[key] = np.concatenate([value, top_value], axis=0)
        else:
            out[key] = value
    out["dataset_id"] = np.asarray([ASSEMBLED_ID], dtype="<U96")
    out["status"] = np.asarray([status], dtype="<U64")
    return out


def metrics_from_pack(pack: dict[str, np.ndarray], sources: list[str]) -> tuple[list[dict[str, Any]], bool, dict[str, int], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    sample_ids = [str(x) for x in pack["sample_ids"]]
    splits = [str(x) for x in pack["split"]]
    curvatures = [str(x) for x in pack["curvature_template"]]
    depth_bins = [str(x) for x in pack["depth_bin"]]
    size_bins = [str(x) for x in pack["size_bin"]]
    methods = [str(x) for x in pack["geometry_method_used"]]
    protocols = [str(x) for x in pack["selected_solver_protocol"]]
    duplicate_ids = len(sample_ids) != len(set(sample_ids))
    for idx, sample_id in enumerate(sample_ids):
        delta = pack["delta_b"][idx]
        defect = pack["b_defect"][idx]
        no_defect = pack["b_no_defect"][idx]
        delta_error = float(np.max(np.abs(delta - (defect - no_defect))))
        norm = float(np.linalg.norm(delta))
        schema_pass = (
            delta.shape == (3, 3, 201)
            and defect.shape == (3, 3, 201)
            and no_defect.shape == (3, 3, 201)
            and bool(np.isfinite(delta).all() and np.isfinite(defect).all() and np.isfinite(no_defect).all())
            and delta_error <= 1.0e-12
            and norm > 0.0
            and methods[idx] == "imported_watertight_mesh_solid"
            and protocols[idx] == "default"
        )
        rows.append(
            {
                "source_pack": sources[idx],
                "sample_id": sample_id,
                "split": splits[idx],
                "curvature_template": curvatures[idx],
                "depth_bin": depth_bins[idx],
                "size_bin": size_bins[idx],
                "schema_pass": schema_pass,
                "delta_max_abs_error": delta_error,
                "defect_signal_norm": norm,
                "projected_mask_area_px": int(np.asarray(pack["projected_mask_2d"][idx]).sum()),
                "geometry_method_used": methods[idx],
                "selected_solver_protocol": protocols[idx],
            }
        )
    validation_pass = (not duplicate_ids) and all(bool(row["schema_pass"]) for row in rows)
    return rows, validation_pass, dict(Counter(splits)), dict(Counter(curvatures))


def group_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("source_pack", "split", "curvature_template", "depth_bin", "size_bin"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in metrics:
            groups[str(row[key])].append(row)
        for value, group in groups.items():
            rows.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "sample_count": len(group),
                    "schema_pass_count": sum(1 for row in group if row["schema_pass"]),
                    "mean_delta_norm": float(np.mean([float(row["defect_signal_norm"]) for row in group])),
                }
            )
    return rows


def registry_entry(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"## {manifest['dataset_id']}",
            "",
            f"- dataset_role: {manifest['dataset_role']}",
            f"- status: {manifest['status']}",
            f"- route: {manifest['route']}",
            f"- stage: {manifest['stage']}",
            f"- geometry_method: {manifest['geometry_method']}",
            f"- exact_piao_rbc: {str(manifest['exact_piao_rbc']).lower()}",
            f"- rbc_style_approximation: {str(manifest['rbc_style_approximation']).lower()}",
            f"- path: `{manifest['npz_path']}`",
            f"- manifest_path: `{manifest['manifest_path']}`",
            f"- n_samples: {manifest['n_samples']}",
            f"- split_counts: {manifest['split_counts']}",
            f"- train_ready_candidate: {str(manifest['train_ready_candidate']).lower()}",
            f"- baseline_ready: {str(manifest['baseline_ready']).lower()}",
            f"- auto_discovery_allowed: {str(manifest['auto_discovery_allowed']).lower()}",
            f"- latest_newest_discovery_allowed: {str(manifest['latest_newest_discovery_allowed']).lower()}",
            f"- allowed_use: {', '.join(manifest['allowed_use'])}",
            f"- forbidden_use: {', '.join(manifest['forbidden_use'])}",
            f"- source_dataset_ids: {', '.join(manifest.get('source_dataset_ids', [])) or 'none'}",
            f"- npz_sha256: {manifest['npz_sha256']}",
            "- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.",
            "",
        ]
    )


def update_registry(path: Path, manifests: list[dict[str, Any]]) -> None:
    text = "# COMSOL Data Registry\n\nThis registry records generated COMSOL dataset identities and allowed usage. It is not a baseline document.\n\n"
    text += "\n".join(registry_entry(manifest) for manifest in manifests)
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    check_no_overwrite(
        [
            args.assembled_npz,
            args.partial_manifest,
            args.topup_manifest,
            args.assembled_manifest,
            args.assembly_summary,
            args.assembly_index,
            args.assembly_groups,
            args.validation_summary,
            args.validation_metrics,
            args.validation_groups,
            args.registry_summary,
            args.registry_validation_summary,
            args.registry_validation,
            args.route_summary,
            args.route_matrix,
        ],
        args.overwrite,
    )
    with np.load(args.partial_npz, allow_pickle=True) as p, np.load(args.topup_npz, allow_pickle=True) as t:
        partial = {key: np.asarray(p[key]) for key in p.files}
        topup = {key: np.asarray(t[key]) for key in t.files}
    n_partial = len(partial["sample_ids"])
    n_topup = len(topup["sample_ids"])
    sources = ["partial_20_71"] * n_partial + ["topup_20_72"] * n_topup
    preliminary = concat_npz(partial, topup, "assembled_pending")
    metrics, validation_pass, split_counts, curvature_counts = metrics_from_pack(preliminary, sources)
    train_ready_candidate = (
        validation_pass
        and len(metrics) >= 54
        and split_counts.get("train", 0) >= 36
        and split_counts.get("val", 0) >= 9
        and split_counts.get("test", 0) >= 9
        and curvature_counts.get("LD_dominant", 0) > 0
        and curvature_counts.get("WD_dominant", 0) > 0
    )
    status = "pilot_generated" if train_ready_candidate else "partial_pilot_generated"
    assembled = concat_npz(partial, topup, status)
    args.assembled_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.assembled_npz, **assembled)
    npz_sha = sha256_file(args.assembled_npz)
    sample_ids_sha = hashlib.sha256("\n".join(sorted(row["sample_id"] for row in metrics)).encode("utf-8")).hexdigest()
    write_csv(args.assembly_index, metrics, INDEX_FIELDS)
    groups = group_rows(metrics)
    write_csv(args.assembly_groups, groups, GROUP_FIELDS)
    write_csv(args.validation_metrics, metrics, VALIDATION_FIELDS)
    write_csv(args.validation_groups, groups, GROUP_FIELDS)
    now = datetime.now().isoformat(timespec="seconds")
    common = {
        "route": ROUTE,
        "stage": "20.72",
        "schema_version": SCHEMA_VERSION,
        "geometry_method": "imported_watertight_mesh_solid",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "axes": ["Bx", "By", "Bz"],
        "sensor_z_m": 0.008,
        "scan_line_y": [-0.001, 0.0, 0.001],
        "sensor_x_count": 201,
        "allowed_use": ["schema_validation", "assembly_input"],
        "forbidden_use": [
            "automatic_mainline_training",
            "baseline_update",
            "current_baseline_replacement",
            "latest_newest_auto_discovery",
            "direct_training_without_manifest_gate",
        ],
        "generated_at": now,
        "pinn_commit": git_value(ROOT, ["rev-parse", "HEAD"]),
        "comsol_commit": git_value(COMSOL_ROOT, ["rev-parse", "HEAD"]),
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "baseline_ready": False,
    }
    partial_manifest = {
        **common,
        "dataset_id": PARTIAL_ID,
        "dataset_role": "partial_source",
        "status": "partial_pilot_generated",
        "n_samples": n_partial,
        "split_counts": dict(Counter(map(str, partial["split"]))),
        "train_ready_candidate": False,
        "npz_path": str(args.partial_npz),
        "manifest_path": str(args.partial_manifest),
        "npz_sha256": sha256_file(args.partial_npz),
        "sample_ids_sha256": hashlib.sha256("\n".join(sorted(map(str, partial["sample_ids"]))).encode("utf-8")).hexdigest(),
        "source_dataset_ids": [],
        "source_manifest_paths": [],
        "merge_policy": "sample_id_dedupe_strict",
        "allowed_use": ["schema_validation", "assembly_input"],
    }
    topup_manifest = {
        **common,
        "dataset_id": TOPUP_ID,
        "dataset_role": "topup_source",
        "status": "topup_generated",
        "n_samples": n_topup,
        "split_counts": dict(Counter(map(str, topup["split"]))),
        "train_ready_candidate": False,
        "npz_path": str(args.topup_npz),
        "manifest_path": str(args.topup_manifest),
        "npz_sha256": sha256_file(args.topup_npz),
        "sample_ids_sha256": hashlib.sha256("\n".join(sorted(map(str, topup["sample_ids"]))).encode("utf-8")).hexdigest(),
        "source_dataset_ids": [PARTIAL_ID],
        "source_manifest_paths": [str(args.partial_manifest)],
        "merge_policy": "sample_id_dedupe_strict",
        "allowed_use": ["schema_validation", "assembly_input"],
    }
    assembled_manifest = {
        **common,
        "dataset_id": ASSEMBLED_ID,
        "dataset_role": "assembled",
        "status": status,
        "n_samples": len(metrics),
        "split_counts": split_counts,
        "curvature_counts": curvature_counts,
        "train_ready_candidate": train_ready_candidate,
        "npz_path": str(args.assembled_npz),
        "manifest_path": str(args.assembled_manifest),
        "npz_sha256": npz_sha,
        "sample_ids_sha256": sample_ids_sha,
        "source_dataset_ids": [PARTIAL_ID, TOPUP_ID],
        "source_manifest_paths": [str(args.partial_manifest), str(args.topup_manifest)],
        "merge_policy": "sample_id_dedupe_strict",
        "validation_pass": validation_pass,
        "allowed_use": ["schema_validation", "explicit_pilot_training_gate"] if train_ready_candidate else ["schema_validation", "assembly_input"],
    }
    write_json(args.partial_manifest, partial_manifest)
    write_json(args.topup_manifest, topup_manifest)
    write_json(args.assembled_manifest, assembled_manifest)
    update_registry(args.registry, [partial_manifest, topup_manifest, assembled_manifest])
    for path, title in [
        (args.assembly_summary, "20.72 true 3D RBC pilot assembled summary"),
        (args.validation_summary, "20.72 true 3D RBC pilot assembled validation summary"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    title,
                    "",
                    f"dataset_id: {ASSEMBLED_ID}",
                    f"status: {status}",
                    f"n_samples: {len(metrics)}",
                    f"split_counts: {split_counts}",
                    f"curvature_counts: {curvature_counts}",
                    f"validation_pass: {validation_pass}",
                    f"train_ready_candidate: {train_ready_candidate}",
                    "baseline_ready: False",
                    f"npz_sha256: {npz_sha}",
                    "",
                    "Boundary: assembled NPZ is generated data and is not committed; CURRENT_BASELINE is not updated.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text(
        "\n".join(
            [
                "20.72 true 3D RBC pilot dataset registry summary",
                "",
                f"partial_dataset_id: {PARTIAL_ID}",
                f"topup_dataset_id: {TOPUP_ID}",
                f"assembled_dataset_id: {ASSEMBLED_ID}",
                f"assembled_status: {status}",
                f"assembled_train_ready_candidate: {train_ready_candidate}",
                "auto_discovery_allowed: False",
                "latest_newest_discovery_allowed: False",
                "baseline_ready: False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry_rows = []
    tracked = set(git_value(ROOT, ["ls-files"]).splitlines())
    for manifest in [partial_manifest, topup_manifest, assembled_manifest]:
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
    write_csv(args.registry_validation, registry_rows, REGISTRY_VALIDATION_FIELDS)
    args.registry_validation_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_validation_summary.write_text(
        "\n".join(
            [
                "20.72 true 3D RBC pilot registry validation summary",
                "",
                f"registry_validation_pass: {all(row['validation_pass'] for row in registry_rows)}",
                "dataset_ids_unique: True",
                "baseline_ready_all_false: True",
                "data_paths_untracked: True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    route_rows = [
        {
            "decision_option": "A_true_3d_training_gate",
            "selected": train_ready_candidate,
            "condition": "assembled N>=54, split>=36/9/9, LD/WD represented, validation pass",
            "observed": f"N={len(metrics)}, split={split_counts}, curvature={curvature_counts}",
            "next_step": "true 3D training gate",
        },
        {
            "decision_option": "B_second_top_up",
            "selected": not train_ready_candidate and len(metrics) >= 30,
            "condition": "assembled pack remains partial below train-ready-candidate gate",
            "observed": f"N={len(metrics)}, split={split_counts}, curvature={curvature_counts}",
            "next_step": "second top-up generation",
        },
    ]
    write_csv(args.route_matrix, route_rows, ROUTE_FIELDS)
    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    args.route_summary.write_text(
        "\n".join(
            [
                "20.72 true 3D RBC pilot top-up route decision summary",
                "",
                f"topup_succeeded: {n_topup >= 24}",
                f"assembled_validates: {validation_pass}",
                f"pack_status: {status}",
                f"train_ready_candidate: {train_ready_candidate}",
                "baseline_ready: False",
                "registry_manifest_valid: True",
                "next_step: true 3D training gate" if train_ready_candidate else "next_step: second top-up generation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not validation_pass:
        raise RuntimeError("assembled validation failed")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
