"""Validate the 20.76 v3_240 true-3D RBC dataset and registry metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import validate_true_3d_rbc_dataset_120_pack as v120


ROOT = Path(__file__).resolve().parents[1]
COMSOL_ROOT = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")

SOURCE_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v2_120"
TOPUP_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76"
ASSEMBLED_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc"
ROUTE = "true_3d_piao_style"

DEFAULT_ASSEMBLED = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.npz"
DEFAULT_TOPUP = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76.npz"
DEFAULT_SOURCE_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.manifest.json"
DEFAULT_TOPUP_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76.manifest.json"
DEFAULT_ASSEMBLED_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json"
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_dataset_240_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_rbc_dataset_240_validation_metrics.csv"
DEFAULT_GROUPS = ROOT / "results/metrics/true_3d_rbc_dataset_240_group_summary.csv"
DEFAULT_REGISTRY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_dataset_240_registry_validation_summary.txt"
DEFAULT_REGISTRY_CSV = ROOT / "results/metrics/true_3d_rbc_dataset_240_registry_validation.csv"
DEFAULT_ROUTE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_dataset_240_route_decision_summary.txt"
DEFAULT_ROUTE_MATRIX = ROOT / "results/metrics/true_3d_rbc_dataset_240_route_decision_matrix.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 20.76 true-3D RBC v3_240 dataset.")
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


def patch_v120_constants() -> None:
    v120.SOURCE_ID = SOURCE_ID
    v120.TOPUP_ID = TOPUP_ID
    v120.ASSEMBLED_ID = ASSEMBLED_ID
    v120.SCHEMA_VERSION = SCHEMA_VERSION
    v120.ROUTE = ROUTE


def train_ready_240(
    rows: list[dict[str, Any]], validation_pass: bool, split_counts: dict[str, int], curvature_counts: dict[str, int]
) -> bool:
    return (
        validation_pass
        and len(rows) >= 216
        and split_counts.get("train", 0) >= 144
        and split_counts.get("val", 0) >= 36
        and split_counts.get("test", 0) >= 36
        and all(curvature_counts.get(name, 0) >= 43 for name in ["sharp", "round", "boxy", "LD_dominant", "WD_dominant"])
    )


def manifest_for_240(
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
        "stage": "20.76",
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
        "pinn_commit": v120.git_value(ROOT, ["rev-parse", "HEAD"]),
        "comsol_commit": v120.git_value(COMSOL_ROOT, ["rev-parse", "HEAD"]),
        "generator_script": "scripts/generate_mfl_true_3d_rbc_dataset_240_topup_pack.py",
        "validation_script": "scripts/validate_true_3d_rbc_dataset_240_pack.py",
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "npz_sha256": v120.sha256_file(npz_path),
        "source_dataset_ids": source_dataset_ids,
        "source_manifest_paths": source_manifest_paths,
        "merge_policy": "sample_id_dedupe_strict",
        "validation_pass": validation_pass,
        "train_ready_candidate": train_ready_candidate,
        "baseline_ready": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "notes": "20.76 metadata only; NPZ is generated data and is not committed.",
    }


def run(args: argparse.Namespace) -> int:
    patch_v120_constants()
    v120.check_no_overwrite(
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
    rows, validation_pass, split_counts, curvature_counts, depth_counts, status = v120.validate_pack(args.assembled_npz)
    groups = v120.group_rows(rows)
    v120.write_csv(args.metrics, rows, v120.METRIC_FIELDS)
    v120.write_csv(args.group_summary, groups, v120.GROUP_FIELDS)
    ready = train_ready_240(rows, validation_pass, split_counts, curvature_counts)
    status = "pilot_generated" if ready else "partial_pilot_generated"

    with np.load(args.topup_npz, allow_pickle=True) as topup_npz:
        topup_split = dict(Counter(v120.string_list(topup_npz["split"])))
        topup_curv = dict(Counter(v120.string_list(topup_npz["curvature_template"])))
        topup_n = len(topup_npz["sample_ids"])
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    topup_manifest = manifest_for_240(
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
    assembled_manifest = manifest_for_240(
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
    v120.write_json(args.topup_manifest, topup_manifest)
    v120.write_json(args.assembled_manifest, assembled_manifest)
    v120.update_registry(args.registry, [topup_manifest, assembled_manifest])

    tracked = set(v120.git_value(ROOT, ["ls-files"]).splitlines())
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
    v120.write_csv(args.registry_csv, registry_rows, v120.REGISTRY_FIELDS)
    registry_valid = all(bool(row["validation_pass"]) for row in registry_rows)

    lines = [
        "20.76 true 3D RBC v3_240 validation summary",
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
                "20.76 true 3D RBC v3_240 registry validation summary",
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
            "decision_option": "A_true_3d_training_gate_on_v3_240",
            "selected": ready,
            "condition": "N>=216, split>=144/36/36, each curvature>=43, schema+registry pass",
            "observed": f"N={len(rows)}, split={split_counts}, curvature={curvature_counts}, registry={registry_valid}",
            "next_step": "true 3D training gate on v3_240",
        },
        {
            "decision_option": "B_second_topup_generation",
            "selected": not ready and len(rows) >= 112,
            "condition": "assembled pack remains below v3_240 train-ready candidate threshold",
            "observed": f"N={len(rows)}, split={split_counts}, curvature={curvature_counts}",
            "next_step": "second top-up generation",
        },
    ]
    v120.write_csv(args.route_matrix, route_rows, v120.ROUTE_FIELDS)
    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    args.route_summary.write_text(
        "\n".join(
            [
                "20.76 true 3D RBC dataset 240 route decision summary",
                "",
                f"topup_succeeded: {topup_n >= 104}",
                f"assembled_240_validates: {validation_pass}",
                f"pack_status: {status}",
                f"train_ready_candidate: {ready}",
                "baseline_ready: False",
                f"registry_manifest_valid: {registry_valid}",
                "next_step: true 3D training gate on v3_240" if ready else "next_step: second top-up generation or fix blocker",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not validation_pass or not registry_valid:
        raise RuntimeError("v3_240 validation or registry gate failed")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
