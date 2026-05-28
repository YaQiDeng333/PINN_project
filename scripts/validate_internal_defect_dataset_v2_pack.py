#!/usr/bin/env python
"""验证并注册 21.3b internal defect v2_240 数据包。

本脚本只读取显式 assembled NPZ 路径，不扫描 latest/newest，不训练，
不运行 COMSOL，不更新 CURRENT_BASELINE.md。
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
DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SOURCE_DATASET_IDS = ["comsol_internal_defect_pilot_pack_v1", "comsol_internal_defect_dataset_topup_pack_v1"]
ROUTE = "internal_buried_defect_feasibility"
SCHEMA_VERSION = "internal_defect_feasibility_v2"
PACK_NPZ = ROOT / "data/comsol_mfl/generated/internal_defect_pilot_pack_v2_240/comsol_internal_defect_pilot_pack_v2_240.npz"
PLAN_CSV = ROOT / "results/metrics/internal_defect_dataset_expansion_plan.csv"
COMSOL_INVENTORY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\inventory_internal_defect_dataset_topup_pack.csv")
COMSOL_SUMMARY = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\internal_defect_dataset_topup_pack_summary.txt")
SUMMARY = ROOT / "results/summaries/internal_defect_dataset_v2_validation_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_dataset_v2_validation_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_dataset_v2_group_summary.csv"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
REGISTRY_SUMMARY = ROOT / "results/summaries/internal_defect_dataset_v2_registry_summary.txt"

AXIS_NAMES = ["Bx", "By", "Bz"]
SCAN_LINE_Y = [-0.001, 0.0, 0.001]
SPLITS = ["train", "val", "test"]
SHAPES = ["internal_sphere", "internal_ellipsoid", "internal_cuboid"]
BURIALS = ["shallow", "medium", "deep", "deep_plus"]
SIZES = ["small", "medium", "large"]
ASPECTS = ["compact", "elongated_x", "elongated_y"]
SPLIT_TARGET = {"train": 160, "val": 40, "test": 40}
SHAPE_TARGET = {"internal_sphere": 80, "internal_ellipsoid": 80, "internal_cuboid": 80}
BURIAL_TARGET = {"shallow": 60, "medium": 60, "deep": 60, "deep_plus": 60}
SIZE_TARGET = {"small": 80, "medium": 80, "large": 80}
ALLOWED_USE = ["schema_validation", "explicit_internal_training_gate"]
FORBIDDEN_USE = [
    "automatic_mainline_training",
    "baseline_update",
    "current_baseline_replacement",
    "latest_newest_auto_discovery",
    "direct_training_without_manifest_gate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证并注册 internal defect v2_240 pack。")
    parser.add_argument("--pack-npz", type=Path, default=PACK_NPZ)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
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
    ]
    return not forbidden, ",".join(forbidden)


def count(arrays: dict[str, np.ndarray], field: str) -> Counter[str]:
    if field not in arrays:
        return Counter()
    return Counter(strings(arrays[field]))


def split_field_counts(arrays: dict[str, np.ndarray], field: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: {split: 0 for split in SPLITS})
    if field not in arrays or "split" not in arrays:
        return {}
    for split, value in zip(strings(arrays["split"]), strings(arrays[field]), strict=False):
        if split in SPLITS:
            out[value][split] += 1
    return dict(out)


def split_contains(arrays: dict[str, np.ndarray], field: str, expected: list[str]) -> bool:
    if field not in arrays or "split" not in arrays:
        return False
    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split, value in zip(strings(arrays["split"]), strings(arrays[field]), strict=False):
        if split in by_split:
            by_split[split].add(value)
    return all(set(expected).issubset(by_split[split]) for split in SPLITS)


def ellipsoid_cuboid_aspect_coverage(arrays: dict[str, np.ndarray]) -> bool:
    if not {"split", "shape_type", "aspect_bin"}.issubset(arrays):
        return False
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split, shape, aspect in zip(strings(arrays["split"]), strings(arrays["shape_type"]), strings(arrays["aspect_bin"]), strict=False):
        if split in SPLITS and shape in {"internal_ellipsoid", "internal_cuboid"}:
            seen[(split, shape)].add(aspect)
    return all(set(ASPECTS).issubset(seen[(split, shape)]) for split in SPLITS for shape in ["internal_ellipsoid", "internal_cuboid"])


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
            f"- internal_surface_mixed: {str(manifest['internal_surface_mixed']).lower()}",
            f"- path: `{manifest['npz_path']}`",
            f"- manifest_path: `{manifest['manifest_path']}`",
            f"- n_samples: {manifest['n_samples']}",
            f"- split_counts: {manifest['split_counts']}",
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
            "- notes: Generated NPZ/data files are not committed. Use only explicit dataset_id + manifest; not a baseline.",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def group_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in ["split", "shape_type", "burial_depth_level", "size_level", "aspect_bin", "row_origin"]:
        counts = count(arrays, field)
        for value, row_count in sorted(counts.items()):
            rows.append({"group_field": field, "group_value": value, "row_count": row_count})
    for field in ["shape_type", "burial_depth_level", "size_level", "aspect_bin"]:
        for value, split_counts in sorted(split_field_counts(arrays, field).items()):
            rows.append(
                {
                    "group_field": f"{field}_by_split",
                    "group_value": value,
                    "row_count": sum(split_counts.values()),
                    "train_count": split_counts.get("train", 0),
                    "val_count": split_counts.get("val", 0),
                    "test_count": split_counts.get("test", 0),
                }
            )
    return rows


def run(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    plan_rows = read_csv(args.plan_csv)
    inventory_rows = read_csv(args.comsol_inventory)
    arrays = load_npz(args.pack_npz)
    n = int(len(arrays.get("sample_ids", []))) if arrays else 0
    success_count = sum(1 for row in inventory_rows if row.get("status") == "success")

    add(checks, "plan_csv_exists", args.plan_csv.exists(), str(args.plan_csv), "21.3 expansion plan")
    add(checks, "planned_topup_rows_168", len(plan_rows) == 168, len(plan_rows), 168)
    add(checks, "comsol_inventory_exists", args.comsol_inventory.exists(), str(args.comsol_inventory), "top-up inventory")
    add(checks, "comsol_summary_exists", args.comsol_summary.exists(), str(args.comsol_summary), "top-up summary")
    add(checks, "topup_success_target", success_count >= 144, success_count, ">=144")
    add(checks, "pack_npz_exists", args.pack_npz.exists(), str(args.pack_npz), "assembled v2 NPZ")
    add(checks, "assembled_n_240", n == 240, n, 240)

    required = [
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
        "L_m",
        "W_m",
        "D_m",
        "D_m_or_cavity_size_m",
        "burial_depth_m",
        "depth_to_surface_m",
        "defect_center_xyz_m",
        "ground_truth_method",
        "cavity_internal",
        "row_origin",
        "source_dataset_id_per_row",
    ]
    missing = [key for key in required if key not in arrays]
    add(checks, "npz_required_fields", not missing, ",".join(missing), "no missing required fields")

    if arrays:
        delta = arrays.get("delta_b")
        b_defect = arrays.get("b_defect")
        b_no = arrays.get("b_no_defect")
        expected_shape = (n, 3, 3, 201)
        add(checks, "delta_b_shape", delta is not None and tuple(delta.shape) == expected_shape, getattr(delta, "shape", None), expected_shape)
        add(checks, "b_defect_shape", b_defect is not None and tuple(b_defect.shape) == expected_shape, getattr(b_defect, "shape", None), expected_shape)
        add(checks, "b_no_defect_shape", b_no is not None and tuple(b_no.shape) == expected_shape, getattr(b_no, "shape", None), expected_shape)
        finite = bool(delta is not None and b_defect is not None and b_no is not None and np.isfinite(delta).all() and np.isfinite(b_defect).all() and np.isfinite(b_no).all())
        add(checks, "bxyz_finite", finite, finite, True)
        if delta is not None and b_defect is not None and b_no is not None:
            max_err = float(np.max(np.abs(delta - (b_defect - b_no))))
            add(checks, "delta_b_equals_defect_minus_no_defect", max_err <= 1.0e-7, max_err, "<=1e-7")
        add(checks, "axis_names", strings(arrays.get("axis_names", np.asarray([]))) == AXIS_NAMES, strings(arrays.get("axis_names", np.asarray([]))), AXIS_NAMES)
        add(checks, "sensor_x_count", arrays.get("sensor_x", np.asarray([])).shape == (201,), arrays.get("sensor_x", np.asarray([])).shape, "(201,)")
        add(checks, "scan_line_y_shape", arrays.get("scan_line_y", np.asarray([])).shape == (n, 3), arrays.get("scan_line_y", np.asarray([])).shape, f"({n}, 3)")
        add(checks, "sensor_z_nominal", bool(np.allclose(arrays.get("sensor_z_m", np.asarray([])), 0.008)), "allclose 0.008", 0.008)
        add(checks, "sample_id_unique", len(set(strings(arrays["sample_ids"]))) == n, len(set(strings(arrays["sample_ids"]))), n)
        internal = arrays.get("cavity_internal")
        add(checks, "cavity_internal_true", internal is not None and bool(np.asarray(internal, dtype=bool).all()), "" if internal is None else "all true", True)
        depths = arrays.get("depth_to_surface_m")
        d_m = arrays.get("D_m")
        add(checks, "internal_depth_positive", depths is not None and bool(np.all(depths > 0)), "" if depths is None else float(np.min(depths)), ">0")
        add(checks, "internal_depth_within_block", depths is not None and d_m is not None and bool(np.all(depths + d_m <= 0.0056 + 1e-9)), "" if depths is None or d_m is None else float(np.max(depths + d_m)), "<=0.0056")

    split_counts = count(arrays, "split")
    shape_counts = count(arrays, "shape_type")
    burial_counts = count(arrays, "burial_depth_level")
    size_counts = count(arrays, "size_level")
    aspect_counts = count(arrays, "aspect_bin")
    add(checks, "split_counts_160_40_40", dict(split_counts) == SPLIT_TARGET, dict(split_counts), SPLIT_TARGET)
    add(checks, "shape_counts_balanced", dict(shape_counts) == SHAPE_TARGET, dict(shape_counts), SHAPE_TARGET)
    add(checks, "burial_counts_balanced", dict(burial_counts) == BURIAL_TARGET, dict(burial_counts), BURIAL_TARGET)
    add(checks, "size_counts_balanced", dict(size_counts) == SIZE_TARGET, dict(size_counts), SIZE_TARGET)
    add(checks, "aspect_coverage", set(ASPECTS).issubset(set(aspect_counts)), dict(aspect_counts), ASPECTS)
    add(checks, "each_split_has_all_shapes", split_contains(arrays, "shape_type", SHAPES), split_field_counts(arrays, "shape_type"), SHAPES)
    add(checks, "each_split_has_all_burials", split_contains(arrays, "burial_depth_level", BURIALS), split_field_counts(arrays, "burial_depth_level"), BURIALS)
    add(checks, "each_split_has_all_sizes", split_contains(arrays, "size_level", SIZES), split_field_counts(arrays, "size_level"), SIZES)
    add(checks, "ellipsoid_cuboid_aspect_each_split", ellipsoid_cuboid_aspect_coverage(arrays), split_field_counts(arrays, "aspect_bin"), "compact/elongated_x/elongated_y for ellipsoid and cuboid in each split")
    staged_ok, staged_forbidden = no_forbidden_staged(ROOT)
    add(checks, "no_forbidden_staged", staged_ok, staged_forbidden, "no data/NPZ/checkpoint/CURRENT_BASELINE staged")

    blocker_names = {
        "plan_csv_exists",
        "planned_topup_rows_168",
        "comsol_inventory_exists",
        "comsol_summary_exists",
        "topup_success_target",
        "pack_npz_exists",
        "assembled_n_240",
        "npz_required_fields",
        "delta_b_shape",
        "b_defect_shape",
        "b_no_defect_shape",
        "bxyz_finite",
        "delta_b_equals_defect_minus_no_defect",
        "axis_names",
        "sensor_x_count",
        "scan_line_y_shape",
        "sample_id_unique",
        "cavity_internal_true",
        "internal_depth_positive",
        "internal_depth_within_block",
        "split_counts_160_40_40",
        "shape_counts_balanced",
        "burial_counts_balanced",
        "size_counts_balanced",
        "each_split_has_all_shapes",
        "each_split_has_all_burials",
        "each_split_has_all_sizes",
        "ellipsoid_cuboid_aspect_each_split",
        "no_forbidden_staged",
    }
    failed_blockers = [row["check_name"] for row in checks if row["check_name"] in blocker_names and not row["pass"]]
    status = "pilot_generated" if not failed_blockers else "blocked"
    train_ready = status == "pilot_generated"

    npz_sha = sha256_file(args.pack_npz) if args.pack_npz.exists() else ""
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_role": "internal_defect_feasibility_pilot_pack_v2_240",
        "status": status,
        "route": ROUTE,
        "stage": "21.3b",
        "schema_version": SCHEMA_VERSION,
        "geometry_method": "internal_cavity_comsol_solid",
        "internal_surface_mixed": False,
        "npz_path": str(args.pack_npz),
        "manifest_path": str(args.manifest),
        "n_samples": n,
        "split_counts": dict(split_counts),
        "shape_counts": dict(shape_counts),
        "burial_depth_counts": dict(burial_counts),
        "size_counts": dict(size_counts),
        "aspect_counts": dict(aspect_counts),
        "axis_names": AXIS_NAMES,
        "sensor_x_count": 201,
        "scan_line_y_m": SCAN_LINE_Y,
        "sensor_z_m": 0.008,
        "train_ready_candidate": train_ready,
        "baseline_ready": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "allowed_use": ALLOWED_USE,
        "forbidden_use": FORBIDDEN_USE,
        "source_dataset_ids": SOURCE_DATASET_IDS,
        "validation_script": "scripts/validate_internal_defect_dataset_v2_pack.py",
        "npz_sha256": npz_sha,
        "surface_rbc_baseline_update": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "failed_blockers": failed_blockers,
    }
    write_json(args.manifest, manifest)
    update_registry(args.registry, manifest)
    write_csv(args.metrics, checks, ["check_name", "pass", "observed", "expected", "notes"])
    write_csv(args.group_summary, group_rows(arrays), ["group_field", "group_value", "row_count", "train_count", "val_count", "test_count"])
    summary_lines = [
        "21.3b internal defect v2_240 validation summary",
        f"dataset_id: {DATASET_ID}",
        f"status: {status}",
        f"train_ready_candidate: {str(train_ready).lower()}",
        f"n_samples: {n}",
        f"split_counts: {dict(split_counts)}",
        f"shape_counts: {dict(shape_counts)}",
        f"burial_depth_counts: {dict(burial_counts)}",
        f"size_counts: {dict(size_counts)}",
        f"aspect_counts: {dict(aspect_counts)}",
        f"topup_success_count: {success_count}",
        f"failed_blockers: {failed_blockers if failed_blockers else 'none'}",
        f"manifest_path: {args.manifest}",
        f"npz_path: {args.pack_npz}",
        "",
        "结论：该数据包仅用于 internal/buried defect 显式训练 gate；baseline_ready=false，不替换 surface/RBC CURRENT_BASELINE。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    registry_lines = [
        "21.3b registry / manifest summary",
        f"dataset_id: {DATASET_ID}",
        f"registry_path: {args.registry}",
        f"manifest_path: {args.manifest}",
        f"status: {status}",
        f"train_ready_candidate: {str(train_ready).lower()}",
        "forbidden_use: " + ", ".join(FORBIDDEN_USE),
        "说明：registry 仅允许 schema_validation / explicit_internal_training_gate，禁止 automatic_mainline_training / baseline_update / current_baseline_replacement。",
    ]
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")
    return 0 if not failed_blockers else 2


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
