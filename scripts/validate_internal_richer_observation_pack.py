#!/usr/bin/env python
"""验证并注册 22.9 internal richer-observation diagnostic pack。

脚本只读取显式路径，不扫描 latest/newest；生成 manifest/registry/summary，
不训练、不运行 COMSOL、不更新 CURRENT_BASELINE.md。
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


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_richer_observation_pack_v1"
SOURCE_DATASET_IDS = ["comsol_internal_defect_pilot_pack_v3_hardcase"]
ROUTE = "internal_buried_defect_richer_observation"
SCHEMA_VERSION = "internal_defect_richer_observation_v1"
PACK_NPZ = ROOT / "data/comsol_mfl/generated/internal_richer_observation_pack/comsol_internal_defect_richer_observation_pack_v1.npz"
PLAN_CSV = ROOT / "results/metrics/internal_richer_observation_diagnostic_pack_plan.csv"
SOURCE_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json"
COMSOL_INVENTORY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\inventory_internal_richer_observation_pack.csv")
COMSOL_SUMMARY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\internal_richer_observation_pack_summary.txt")
SUMMARY = ROOT / "results/summaries/internal_richer_observation_pack_validation_summary.txt"
METRICS = ROOT / "results/metrics/internal_richer_observation_pack_validation_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_richer_observation_pack_group_summary.csv"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_richer_observation_pack_v1.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
REGISTRY_SUMMARY = ROOT / "results/summaries/internal_richer_observation_pack_registry_summary.txt"

EXPECTED_VARIANTS = [
    "R0_3line_z0p008",
    "R1_5line_z0p008",
    "R1_9line_z0p008",
    "R2_5line_z0p006",
    "R2_5line_z0p010",
    "R2_5line_z0p012",
]
EXPECTED_SCAN_COUNTS = {3, 5, 9}
EXPECTED_LIFTOFF = {0.006, 0.008, 0.010, 0.012}
EXPECTED_SHAPES = {"internal_sphere", "internal_ellipsoid", "internal_cuboid"}
EXPECTED_BURIAL = {"shallow", "medium", "deep", "deep_plus"}
EXPECTED_SIZE = {"small", "medium", "large"}
EXPECTED_ASPECT = {"compact", "elongated_x", "elongated_y"}
ALLOWED_USE = ["schema_validation", "explicit_richer_observation_diagnostic"]
FORBIDDEN_USE = [
    "automatic_mainline_training",
    "baseline_update",
    "current_baseline_replacement",
    "latest_newest_auto_discovery",
    "direct_training_without_manifest_gate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 internal richer-observation diagnostic pack。")
    parser.add_argument("--pack-npz", type=Path, default=PACK_NPZ)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--comsol-inventory", type=Path, default=COMSOL_INVENTORY)
    parser.add_argument("--comsol-summary", type=Path, default=COMSOL_SUMMARY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--registry-summary", type=Path, default=REGISTRY_SUMMARY)
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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def strings(value: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(value).reshape(-1).tolist()]


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append({"check_name": name, "pass": bool(passed), "observed": observed, "expected": expected, "notes": notes})


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
        or path == "scripts/visualize_current_baseline.py"
    ]
    return not forbidden, ",".join(forbidden)


def data_not_staged(root: Path) -> tuple[bool, str]:
    lines = git_lines(root, ["status", "--short", "--", "data", "checkpoints", "results/previews", "notes"])
    return len(lines) == 0, "; ".join(lines)


def count_field(arrays: dict[str, np.ndarray], field: str) -> Counter[str]:
    if field not in arrays:
        return Counter()
    return Counter(strings(arrays[field]))


def complete_bases(arrays: dict[str, np.ndarray]) -> tuple[int, int, dict[str, list[str]]]:
    if not {"base_group_id", "observation_variant"}.issubset(arrays):
        return 0, 0, {}
    variants_by_base: dict[str, set[str]] = defaultdict(set)
    for base, variant in zip(strings(arrays["base_group_id"]), strings(arrays["observation_variant"]), strict=False):
        variants_by_base[base].add(variant)
    complete = {base: sorted(variants) for base, variants in variants_by_base.items() if set(EXPECTED_VARIANTS).issubset(variants)}
    return len(variants_by_base), len(complete), complete


def sorted_float_set(values: np.ndarray) -> list[float]:
    return sorted({round(float(x), 6) for x in np.asarray(values).reshape(-1).tolist()})


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
            f"- internal_surface_mixed: {str(manifest['internal_surface_mixed']).lower()}",
            f"- path: `{manifest['npz_path']}`",
            f"- manifest_path: `{manifest['manifest_path']}`",
            f"- n_samples: {manifest['n_samples']}",
            f"- planned_rows: {manifest['planned_rows']}",
            f"- complete_base_count: {manifest['complete_base_count']}",
            f"- observation_variants: {manifest['observation_variants']}",
            f"- scan_line_counts: {manifest['scan_line_counts']}",
            f"- sensor_z_levels_m: {manifest['sensor_z_levels_m']}",
            f"- shape_counts: {manifest['shape_counts']}",
            f"- burial_depth_counts: {manifest['burial_depth_counts']}",
            f"- size_counts: {manifest['size_counts']}",
            f"- aspect_counts: {manifest['aspect_counts']}",
            f"- train_ready_candidate: {str(manifest['train_ready_candidate']).lower()}",
            f"- baseline_ready: {str(manifest['baseline_ready']).lower()}",
            f"- auto_discovery_allowed: {str(manifest['auto_discovery_allowed']).lower()}",
            f"- latest_newest_discovery_allowed: {str(manifest['latest_newest_discovery_allowed']).lower()}",
            f"- allowed_use: {', '.join(manifest['allowed_use'])}",
            f"- forbidden_use: {', '.join(manifest['forbidden_use'])}",
            f"- source_dataset_ids: {', '.join(manifest['source_dataset_ids'])}",
            f"- validation_script: `{manifest['validation_script']}`",
            f"- npz_sha256: {manifest['npz_sha256']}",
            "- notes: Richer-observation diagnostic pack only; generated NPZ/data files are not committed; not a baseline.",
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


def group_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not arrays:
        return rows
    base = strings(arrays["base_group_id"]) if "base_group_id" in arrays else [""] * int(len(arrays.get("sample_ids", [])))
    for field in ["observation_variant", "scan_line_count", "sensor_z_m", "shape_type", "burial_depth_level", "size_level", "aspect_bin"]:
        if field not in arrays:
            continue
        values = arrays[field]
        counter: dict[str, list[int]] = defaultdict(list)
        for idx, value in enumerate(np.asarray(values).reshape(-1).tolist()):
            key = str(round(float(value), 6)) if field == "sensor_z_m" else str(value)
            counter[key].append(idx)
        for key, indices in sorted(counter.items()):
            rows.append(
                {
                    "group_type": field,
                    "group_value": key,
                    "row_count": len(indices),
                    "base_count": len({base[i] for i in indices}),
                }
            )
    return rows


def run(args: argparse.Namespace) -> int:
    plan_rows = read_csv(args.plan_csv)
    inventory = read_csv(args.comsol_inventory)
    arrays = load_npz(args.pack_npz)
    checks: list[dict[str, Any]] = []

    pack_exists = args.pack_npz.exists()
    add(checks, "pack_npz_exists", pack_exists, args.pack_npz, "NPZ exists")
    add(checks, "plan_rows_180", len(plan_rows) == 180, len(plan_rows), 180)
    add(checks, "source_manifest_exists", args.source_manifest.exists(), args.source_manifest, "source v3_hardcase manifest")
    add(checks, "comsol_inventory_exists", args.comsol_inventory.exists(), args.comsol_inventory, "COMSOL inventory")
    add(checks, "comsol_summary_exists", args.comsol_summary.exists(), args.comsol_summary, "COMSOL summary")

    success_inventory = [row for row in inventory if row.get("status") == "success"]
    n = int(len(arrays.get("sample_ids", []))) if arrays else 0
    base_count, complete_base_count, complete_map = complete_bases(arrays)
    status = "blocked"
    if n == 180 and complete_base_count == 30:
        status = "diagnostic_pack_generated"
    elif n >= 144 and complete_base_count >= 24:
        status = "partial_diagnostic_pack_generated"

    add(checks, "success_rows_match_inventory", n == len(success_inventory), f"npz={n}; inventory_success={len(success_inventory)}", "same")
    add(checks, "success_min_144", n >= 144, n, ">=144")
    add(checks, "full_rows_180", n == 180, n, 180)
    add(checks, "base_count_30", base_count == 30, base_count, 30)
    add(checks, "complete_base_count_min_24", complete_base_count >= 24, complete_base_count, ">=24")
    add(checks, "complete_base_count_30", complete_base_count == 30, complete_base_count, 30)

    if arrays:
        delta = arrays.get("delta_b")
        b_defect = arrays.get("b_defect")
        b_no = arrays.get("b_no_defect")
        shape_ok = (
            delta is not None
            and b_defect is not None
            and b_no is not None
            and delta.shape == b_defect.shape == b_no.shape
            and len(delta.shape) == 4
            and delta.shape[1:] == (3, 9, 201)
        )
        add(checks, "signal_shape_3_9_201", shape_ok, getattr(delta, "shape", None), "(N,3,9,201)")
        finite_ok = bool(delta is not None and b_defect is not None and b_no is not None and np.isfinite(delta).all() and np.isfinite(b_defect).all() and np.isfinite(b_no).all())
        add(checks, "bxyz_finite", finite_ok, finite_ok, True)
        max_delta_error = float(np.max(np.abs(delta - (b_defect - b_no)))) if shape_ok else float("nan")
        add(checks, "delta_check", bool(shape_ok and max_delta_error <= 1e-8), max_delta_error, "<=1e-8")
        sample_ids = strings(arrays["sample_ids"]) if "sample_ids" in arrays else []
        add(checks, "sample_id_unique", len(sample_ids) == len(set(sample_ids)), f"{len(sample_ids)} rows / {len(set(sample_ids))} unique", "unique")
        variant_counts = count_field(arrays, "observation_variant")
        add(checks, "variant_coverage", set(variant_counts) == set(EXPECTED_VARIANTS), dict(variant_counts), EXPECTED_VARIANTS)
        scan_counts = {int(x) for x in np.asarray(arrays.get("scan_line_count", [])).reshape(-1).tolist()}
        add(checks, "scan_line_count_coverage", scan_counts == EXPECTED_SCAN_COUNTS, sorted(scan_counts), sorted(EXPECTED_SCAN_COUNTS))
        sensor_z_levels = set(sorted_float_set(arrays.get("sensor_z_m", np.asarray([]))))
        add(checks, "liftoff_level_coverage", sensor_z_levels == EXPECTED_LIFTOFF, sorted(sensor_z_levels), sorted(EXPECTED_LIFTOFF))
        if "scan_line_mask" in arrays and "scan_line_count" in arrays:
            mask_counts = np.asarray(arrays["scan_line_mask"], dtype=bool).sum(axis=1).astype(int)
            expected_counts = np.asarray(arrays["scan_line_count"]).astype(int)
            add(checks, "scan_line_mask_matches_count", bool(np.array_equal(mask_counts, expected_counts)), f"mismatches={(mask_counts != expected_counts).sum()}", "0 mismatches")
        add(checks, "shape_coverage", set(count_field(arrays, "shape_type")) == EXPECTED_SHAPES, dict(count_field(arrays, "shape_type")), EXPECTED_SHAPES)
        add(checks, "burial_coverage", set(count_field(arrays, "burial_depth_level")) == EXPECTED_BURIAL, dict(count_field(arrays, "burial_depth_level")), EXPECTED_BURIAL)
        add(checks, "size_coverage", set(count_field(arrays, "size_level")) == EXPECTED_SIZE, dict(count_field(arrays, "size_level")), EXPECTED_SIZE)
        add(checks, "aspect_coverage", EXPECTED_ASPECT.issubset(set(count_field(arrays, "aspect_bin"))), dict(count_field(arrays, "aspect_bin")), EXPECTED_ASPECT)
        add(checks, "cavity_internal_all", bool("cavity_internal" in arrays and np.asarray(arrays["cavity_internal"], dtype=bool).all()), "all true" if "cavity_internal" in arrays else "missing", True)
    else:
        max_delta_error = float("nan")

    no_data_staged, data_status = data_not_staged(ROOT)
    no_forbidden, forbidden = no_forbidden_staged(ROOT)
    add(checks, "data_not_staged", no_data_staged, data_status, "no data/checkpoint/preview/notes staged")
    add(checks, "forbidden_not_staged", no_forbidden, forbidden, "no forbidden staged artifacts")

    blocking_checks = {
        "pack_npz_exists",
        "plan_rows_180",
        "source_manifest_exists",
        "comsol_inventory_exists",
        "success_min_144",
        "complete_base_count_min_24",
        "signal_shape_3_9_201",
        "bxyz_finite",
        "delta_check",
        "sample_id_unique",
        "variant_coverage",
        "scan_line_count_coverage",
        "liftoff_level_coverage",
        "shape_coverage",
        "burial_coverage",
        "size_coverage",
        "data_not_staged",
        "forbidden_not_staged",
    }
    failed_blockers = [row["check_name"] for row in checks if row["check_name"] in blocking_checks and not row["pass"]]
    validation_passed = not failed_blockers
    if failed_blockers:
        status = "blocked"

    shape_counts = dict(count_field(arrays, "shape_type"))
    burial_counts = dict(count_field(arrays, "burial_depth_level"))
    size_counts = dict(count_field(arrays, "size_level"))
    aspect_counts = dict(count_field(arrays, "aspect_bin"))
    variant_counts = dict(count_field(arrays, "observation_variant"))
    scan_line_counts = {str(k): int(v) for k, v in Counter(np.asarray(arrays.get("scan_line_count", [])).astype(int).tolist()).items()} if arrays else {}
    sensor_z_levels = sorted_float_set(arrays.get("sensor_z_m", np.asarray([]))) if arrays else []
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_role": "internal_defect_richer_observation_diagnostic_pack",
        "route": ROUTE,
        "stage": "22.9",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "npz_path": str(args.pack_npz),
        "npz_sha256": sha256_file(args.pack_npz) if pack_exists else "",
        "manifest_path": str(args.manifest),
        "source_dataset_ids": SOURCE_DATASET_IDS,
        "source_manifest": str(args.source_manifest),
        "source_plan": str(args.plan_csv),
        "comsol_inventory": str(args.comsol_inventory),
        "comsol_summary": str(args.comsol_summary),
        "n_samples": n,
        "planned_rows": len(plan_rows),
        "success_rows": n,
        "base_count": base_count,
        "complete_base_count": complete_base_count,
        "complete_base_ids": sorted(complete_map),
        "observation_variants": variant_counts,
        "scan_line_counts": scan_line_counts,
        "sensor_z_levels_m": sensor_z_levels,
        "shape_counts": shape_counts,
        "burial_depth_counts": burial_counts,
        "size_counts": size_counts,
        "aspect_counts": aspect_counts,
        "validation_passed": validation_passed,
        "failed_blockers": failed_blockers,
        "train_ready_candidate": False,
        "baseline_ready": False,
        "internal_surface_mixed": False,
        "surface_rbc_baseline_update": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "allowed_use": ALLOWED_USE,
        "forbidden_use": FORBIDDEN_USE,
        "validation_script": "scripts/validate_internal_richer_observation_pack.py",
        "max_delta_error": max_delta_error,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_registry(args.registry, manifest)

    write_csv(args.metrics, checks, ["check_name", "pass", "observed", "expected", "notes"])
    write_csv(args.group_summary, group_rows(arrays), ["group_type", "group_value", "row_count", "base_count"])

    summary_lines = [
        "22.9 internal richer-observation pack validation summary",
        f"dataset_id: {DATASET_ID}",
        f"status: {status}",
        f"validation_passed: {str(validation_passed).lower()}",
        f"planned_rows: {len(plan_rows)}",
        f"success_rows: {n}",
        f"base_count: {base_count}",
        f"complete_base_count: {complete_base_count}",
        f"observation_variants: {variant_counts}",
        f"scan_line_counts: {scan_line_counts}",
        f"sensor_z_levels_m: {sensor_z_levels}",
        f"shape_counts: {shape_counts}",
        f"burial_depth_counts: {burial_counts}",
        f"size_counts: {size_counts}",
        f"aspect_counts: {aspect_counts}",
        f"max_delta_error: {max_delta_error}",
        f"failed_blockers: {failed_blockers}",
        "baseline_update: false",
        "train_ready_candidate: false",
        "结论：该数据包只用于 richer-observation schema/diagnostic，不能自动训练，不能替换 CURRENT_BASELINE。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    registry_lines = [
        "22.9 internal richer-observation registry summary",
        f"dataset_id: {DATASET_ID}",
        f"registry_path: {args.registry}",
        f"manifest_path: {args.manifest}",
        f"status: {status}",
        f"allowed_use: {', '.join(ALLOWED_USE)}",
        f"forbidden_use: {', '.join(FORBIDDEN_USE)}",
        "baseline_ready: false",
        "train_ready_candidate: false",
        "latest_newest_discovery_allowed: false",
    ]
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")
    if not validation_passed:
        raise RuntimeError(f"validation blockers: {failed_blockers}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
