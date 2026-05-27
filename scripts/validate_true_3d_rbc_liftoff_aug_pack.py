#!/usr/bin/env python
"""Validate and register the 20.91b true-3D RBC liftoff augmentation pack.

The generated NPZ is read from the explicit 20.91b path. This script does not
discover latest/newest data, train a model, run COMSOL, or update the current
baseline.
"""

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

from load_true_3d_rbc_pilot_dataset import (
    ROOT,
    REGISTRY_PATH,
    V3_240_DATASET_ID,
    gate_manifest,
    resolve_dataset,
    sha256_file,
    write_csv,
)


DATASET_ID = "comsol_true_3d_rbc_liftoff_aug_pack_v1"
ROUTE = "true_3d_piao_style_liftoff_robustness"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc_liftoff_aug"
PLAN_CSV = ROOT / "results/metrics/true_3d_rbc_liftoff_aug_pack_plan.csv"
PACK_NPZ = ROOT / "data/comsol_mfl/generated/true_3d_rbc_liftoff_aug_pack/true_3d_rbc_liftoff_aug_pack.npz"
COMSOL_INVENTORY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\inventory_true_3d_rbc_liftoff_aug_pack.csv")
COMSOL_SUMMARY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\true_3d_rbc_liftoff_aug_pack_summary.txt")
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_aug_pack_validation_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_aug_pack_validation_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/true_3d_rbc_liftoff_aug_pack_group_summary.csv"
MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json"
REGISTRY_SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_aug_pack_registry_summary.txt"

EXPECTED_ROW_COUNT = 192
EXPECTED_BASE_COUNT = 48
LIFTOFF_LEVELS = {0.006, 0.008, 0.010, 0.012}
FORBIDDEN_USE = [
    "automatic_mainline_training",
    "baseline_update",
    "current_baseline_replacement",
    "latest_newest_auto_discovery",
    "direct_training_without_manifest_gate",
]
ALLOWED_USE = ["schema_validation", "explicit_liftoff_training_gate"]

CHECK_FIELDS = ["check_name", "pass", "observed", "expected", "notes"]
GROUP_FIELDS = ["group_field", "group_value", "row_count", "base_count", "success_count", "failure_count", "complete_paired_base_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and register 20.91b true-3D RBC liftoff augmentation pack.")
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--pack-npz", type=Path, default=PACK_NPZ)
    parser.add_argument("--comsol-inventory", type=Path, default=COMSOL_INVENTORY)
    parser.add_argument("--comsol-summary", type=Path, default=COMSOL_SUMMARY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--registry-summary", type=Path, default=REGISTRY_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def check_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def git_value(cwd: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status_counts(inventory: list[dict[str, str]]) -> Counter:
    return Counter(row.get("status", "") for row in inventory)


def paired_base_sets(rows: list[dict[str, str]], only_success: bool = False) -> dict[str, set[float]]:
    out: dict[str, set[float]] = defaultdict(set)
    for row in rows:
        if only_success and row.get("status") != "success":
            continue
        try:
            out[row["base_sample_id"]].add(round(float(row["sensor_z_m"]), 3))
        except Exception:
            pass
    return out


def complete_count(base_levels: dict[str, set[float]]) -> int:
    expected = {round(x, 3) for x in LIFTOFF_LEVELS}
    return sum(1 for levels in base_levels.values() if levels == expected)


def no_data_staged(root: Path) -> tuple[bool, str]:
    staged = git_value(root, ["diff", "--cached", "--name-only"]).splitlines()
    forbidden = [
        path
        for path in staged
        if path.startswith("data/")
        or path.endswith(".npz")
        or path.endswith(".mph")
        or path.startswith("checkpoints/")
        or path.startswith("results/previews/")
        or path.startswith("notes/")
    ]
    return (not forbidden, ",".join(forbidden))


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
            f"- base_count: {manifest['base_count']}",
            f"- paired_liftoff_complete: {str(manifest['paired_liftoff_complete']).lower()}",
            f"- liftoff_levels_m: {manifest['liftoff_levels_m']}",
            f"- split_counts: {manifest['split_counts']}",
            f"- curvature_counts: {manifest['curvature_counts']}",
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
    entry_text = registry_entry(manifest)
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else "# COMSOL Data Registry\n\n"
    heading = f"## {manifest['dataset_id']}"
    if heading in text:
        start = text.index(heading)
        next_start = text.find("\n## ", start + 1)
        if next_start == -1:
            text = text[:start].rstrip() + "\n\n" + entry_text
        else:
            text = text[:start].rstrip() + "\n\n" + entry_text + text[next_start:].lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + entry_text
    path.write_text(text, encoding="utf-8")


def group_rows(plan_rows: list[dict[str, str]], inventory: list[dict[str, str]]) -> list[dict[str, Any]]:
    inv_by_row = {row.get("diagnostic_row_id", ""): row for row in inventory}
    joined = [{**row, **{"status": inv_by_row.get(row["diagnostic_row_id"], {}).get("status", "missing")}} for row in plan_rows]
    groups: list[dict[str, Any]] = []
    for field in ["split", "curvature_template", "depth_bin", "aspect_bin", "sensor_z_m"]:
        for value in sorted({str(row.get(field, "")) for row in joined}):
            subset = [row for row in joined if str(row.get(field, "")) == value]
            base_levels = paired_base_sets(subset)
            success = [row for row in subset if row.get("status") == "success"]
            groups.append(
                {
                    "group_field": field,
                    "group_value": value,
                    "row_count": len(subset),
                    "base_count": len({row.get("base_sample_id", "") for row in subset}),
                    "success_count": len(success),
                    "failure_count": len([row for row in subset if row.get("status") != "success"]),
                    "complete_paired_base_count": complete_count(base_levels) if field != "sensor_z_m" else "",
                }
            )
    return groups


def clean_counts(values: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter([str(x) for x in values.tolist()]).items()}


def run(args: argparse.Namespace) -> int:
    check_overwrite([args.summary, args.metrics, args.group_summary, args.manifest, args.registry_summary], args.overwrite)
    checks: list[dict[str, Any]] = []
    entry, source_manifest, source_npz = resolve_dataset(args.dataset_id)
    gate = gate_manifest(entry, source_manifest, source_npz, args.dataset_id)
    add(checks, "source_registry_manifest_gate", not [row for row in gate if not row["pass"]], len([row for row in gate if not row["pass"]]), 0)
    add(checks, "plan_csv_exists", args.plan_csv.exists(), str(args.plan_csv), "20.91 plan CSV")
    add(checks, "pack_npz_exists", args.pack_npz.exists(), str(args.pack_npz), "generated ignored NPZ path")
    add(checks, "comsol_inventory_exists", args.comsol_inventory.exists(), str(args.comsol_inventory), "COMSOL inventory")
    add(checks, "comsol_summary_exists", args.comsol_summary.exists(), str(args.comsol_summary), "COMSOL summary")

    plan_rows = read_csv(args.plan_csv)
    inventory = read_csv(args.comsol_inventory)
    arrays = load_npz(args.pack_npz)
    required = [
        "delta_b",
        "b_defect",
        "b_no_defect",
        "sample_ids",
        "base_sample_ids",
        "variant_name",
        "split",
        "rbc_params",
        "profile_pose",
        "profile_depth_grid_m",
        "profile_depth_map_xy_m",
        "projected_mask_2d",
        "sensor_x",
        "scan_line_y",
        "sensor_z_m",
        "axis_names",
    ]
    missing = [key for key in required if key not in arrays]
    add(checks, "npz_required_fields", not missing, ",".join(missing) if missing else "none", "all required fields")
    add(checks, "plan_row_count", len(plan_rows) == EXPECTED_ROW_COUNT, len(plan_rows), EXPECTED_ROW_COUNT)
    add(checks, "plan_base_count", len({row.get("base_sample_id", "") for row in plan_rows}) == EXPECTED_BASE_COUNT, len({row.get("base_sample_id", "") for row in plan_rows}), EXPECTED_BASE_COUNT)
    add(checks, "plan_all_requires_comsol", all(row.get("requires_comsol", "").lower() == "true" for row in plan_rows), Counter(row.get("requires_comsol", "") for row in plan_rows), "all true")
    plan_levels = paired_base_sets(plan_rows)
    add(checks, "plan_paired_liftoff_complete", complete_count(plan_levels) == EXPECTED_BASE_COUNT, complete_count(plan_levels), EXPECTED_BASE_COUNT)

    success_inventory = [row for row in inventory if row.get("status") == "success"]
    success_levels = paired_base_sets(inventory, only_success=True)
    full_pack = len(success_inventory) == EXPECTED_ROW_COUNT and complete_count(success_levels) == EXPECTED_BASE_COUNT
    pack_status = "diagnostic_pack_generated" if full_pack else "partial_diagnostic_pack_generated"
    add(checks, "inventory_row_count", len(inventory) == len(plan_rows), len(inventory), len(plan_rows), "no silent skip")
    add(checks, "success_row_count", len(success_inventory) == EXPECTED_ROW_COUNT, len(success_inventory), EXPECTED_ROW_COUNT)
    add(checks, "paired_liftoff_success_complete", complete_count(success_levels) == EXPECTED_BASE_COUNT, complete_count(success_levels), EXPECTED_BASE_COUNT)
    add(checks, "inventory_status_counts", True, dict(status_counts(inventory)), "")

    if arrays:
        n = int(arrays.get("delta_b", np.empty((0,))).shape[0])
        add(checks, "npz_row_count_matches_success", n == len(success_inventory), n, len(success_inventory))
        add(checks, "delta_b_shape", arrays["delta_b"].shape[1:] == (3, 3, 201), list(arrays["delta_b"].shape), "(n,3,3,201)")
        add(checks, "finite_bxyz", bool(np.isfinite(arrays["delta_b"]).all() and np.isfinite(arrays["b_defect"]).all() and np.isfinite(arrays["b_no_defect"]).all()), "finite", "all finite")
        delta_err = float(np.max(np.abs(arrays["delta_b"] - (arrays["b_defect"] - arrays["b_no_defect"])))) if n else float("nan")
        add(checks, "delta_recompute_error", delta_err <= 1.0e-7, f"{delta_err:.6e}", "<=1e-7", "float32 NPZ storage")
        observed_levels = {round(float(x), 3) for x in np.asarray(arrays["sensor_z_m"]).reshape(-1)}
        add(checks, "npz_liftoff_levels", observed_levels == {round(x, 3) for x in LIFTOFF_LEVELS}, sorted(observed_levels), sorted(LIFTOFF_LEVELS))
        add(checks, "sample_ids_unique", len(set(arrays["sample_ids"].astype(str))) == n, len(set(arrays["sample_ids"].astype(str))), n)
        add(checks, "base_count_npz", len(set(arrays["base_sample_ids"].astype(str))) == (EXPECTED_BASE_COUNT if full_pack else len(success_levels)), len(set(arrays["base_sample_ids"].astype(str))), EXPECTED_BASE_COUNT)
        add(checks, "axis_names", [str(x) for x in arrays["axis_names"].tolist()] == ["Bx", "By", "Bz"], [str(x) for x in arrays["axis_names"].tolist()], "Bx,By,Bz")
        add(checks, "sensor_x_count", int(np.asarray(arrays["sensor_x"]).shape[0]) == 201, int(np.asarray(arrays["sensor_x"]).shape[0]), 201)
        add(checks, "scan_line_y_shape", np.asarray(arrays["scan_line_y"]).shape[1:] == (3,), list(np.asarray(arrays["scan_line_y"]).shape), "(n,3)")
        add(checks, "split_coverage", {"train", "val", "test"}.issubset(set(arrays["split"].astype(str))), sorted(set(arrays["split"].astype(str))), "train,val,test")

    no_staged, staged_forbidden = no_data_staged(ROOT)
    add(checks, "no_forbidden_data_staged", no_staged, staged_forbidden or "none", "no staged data/NPZ/checkpoint/preview/notes")

    validation_pass = all(bool(row["pass"]) for row in checks if row["check_name"] not in {"success_row_count", "paired_liftoff_success_complete"})
    train_ready = validation_pass and full_pack
    split_counts = clean_counts(arrays["split"].astype(str)) if arrays else {}
    curvature_counts = clean_counts(arrays["curvature_template"].astype(str)) if arrays and "curvature_template" in arrays else {}
    source_manifest_path = Path(source_manifest["manifest_path"])
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_role": "liftoff_augmentation_diagnostic_pack",
        "status": pack_status,
        "route": ROUTE,
        "stage": "20.91b",
        "schema_version": SCHEMA_VERSION,
        "geometry_method": "imported_watertight_mesh_solid",
        "exact_piao_rbc": False,
        "rbc_style_approximation": True,
        "n_samples": int(arrays["delta_b"].shape[0]) if arrays else 0,
        "planned_rows": EXPECTED_ROW_COUNT,
        "base_count": len(set(arrays["base_sample_ids"].astype(str))) if arrays and "base_sample_ids" in arrays else 0,
        "planned_base_count": EXPECTED_BASE_COUNT,
        "paired_liftoff_complete": full_pack,
        "complete_paired_base_count": complete_count(success_levels),
        "incomplete_pair_count": EXPECTED_BASE_COUNT - complete_count(success_levels),
        "liftoff_levels_m": sorted(LIFTOFF_LEVELS),
        "split_counts": split_counts,
        "curvature_counts": curvature_counts,
        "axes": ["Bx", "By", "Bz"],
        "sensor_x_count": 201,
        "allowed_use": ALLOWED_USE,
        "forbidden_use": FORBIDDEN_USE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_commit": git_value(ROOT, ["rev-parse", "HEAD"]),
        "comsol_commit": git_value(Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP"), ["rev-parse", "HEAD"]),
        "generator_script": "scripts/generate_mfl_true_3d_rbc_liftoff_aug_pack.py",
        "validation_script": "scripts/validate_true_3d_rbc_liftoff_aug_pack.py",
        "npz_path": str(args.pack_npz),
        "manifest_path": str(args.manifest),
        "npz_sha256": sha256(args.pack_npz) if args.pack_npz.exists() else "",
        "source_dataset_ids": [args.dataset_id],
        "source_manifest_paths": [str(source_manifest_path)],
        "source_plan_csv": str(args.plan_csv),
        "comsol_inventory": str(args.comsol_inventory),
        "validation_pass": validation_pass,
        "train_ready_candidate": train_ready,
        "baseline_ready": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "notes": "20.91b metadata only; NPZ is generated data and is not committed. Use only through explicit dataset_id + manifest.",
    }
    write_json(args.manifest, manifest)
    update_registry(args.registry, manifest)
    groups = group_rows(plan_rows, inventory)
    write_csv(args.metrics, checks, CHECK_FIELDS)
    write_csv(args.group_summary, groups, GROUP_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "20.91b true 3D RBC liftoff augmentation pack validation summary",
                "",
                f"dataset_id: {DATASET_ID}",
                f"source_dataset_id: {args.dataset_id}",
                f"pack_status: {pack_status}",
                f"validation_pass: {validation_pass}",
                f"train_ready_candidate: {train_ready}",
                f"planned_rows: {len(plan_rows)}",
                f"successful_rows: {len(success_inventory)}",
                f"planned_base_count: {EXPECTED_BASE_COUNT}",
                f"observed_base_count: {manifest['base_count']}",
                f"complete_paired_base_count: {manifest['complete_paired_base_count']}",
                f"incomplete_pair_count: {manifest['incomplete_pair_count']}",
                f"liftoff_levels_m: {sorted(LIFTOFF_LEVELS)}",
                f"npz_path_ignored: {args.pack_npz}",
                f"manifest_path: {args.manifest}",
                "latest_newest_npz_scan: false",
                "COMSOL_run_by_this_script: false",
                "training_run: false",
                "baseline_update: false",
                "",
                "Failed checks:",
                *[
                    f"- {row['check_name']}: observed={row['observed']} expected={row['expected']} notes={row['notes']}"
                    for row in checks
                    if not row["pass"]
                ],
                *([] if [row for row in checks if not row["pass"]] else ["- none"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text(
        "\n".join(
            [
                "20.91b true 3D RBC liftoff augmentation pack registry summary",
                "",
                f"dataset_id: {DATASET_ID}",
                f"registry_path: {args.registry}",
                f"manifest_path: {args.manifest}",
                f"status: {pack_status}",
                f"allowed_use: {', '.join(ALLOWED_USE)}",
                f"forbidden_use: {', '.join(FORBIDDEN_USE)}",
                "baseline_ready: false",
                "auto_discovery_allowed: false",
                "latest_newest_discovery_allowed: false",
                "CURRENT_BASELINE_update: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not validation_pass:
        raise RuntimeError("20.91b liftoff augmentation pack validation failed; see validation summary")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
