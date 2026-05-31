#!/usr/bin/env python
"""验证 23.4 internal multi-magnetization mag_y diagnostic pack。

本脚本只读取显式路径，不扫描 latest/newest；不训练、不运行 COMSOL、不写
data/NPZ、不更新 CURRENT_BASELINE.md。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAG_Y_NPZ = ROOT / "data/comsol_mfl/generated/internal_multi_magnetization_pack/internal_multi_magnetization_mag_y_pack_v1.npz"
COMSOL_INVENTORY = Path(
    r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\inventory_internal_multi_magnetization_pack.csv"
)
COMSOL_SUMMARY = Path(
    r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\internal_multi_magnetization_pack_summary.txt"
)
PREFLIGHT_SUMMARY = ROOT / "results/summaries/internal_multi_magnetization_pack_preflight_summary.txt"
SUMMARY = ROOT / "results/summaries/internal_multi_magnetization_pack_validation_summary.txt"
METRICS = ROOT / "results/metrics/internal_multi_magnetization_pack_validation_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_multi_magnetization_pack_group_summary.csv"

EXPECTED_VARIANTS = {"M1_mag_y_5line_z0p008", "M2_mag_y_9line_z0p008"}
EXPECTED_REFERENCES = {"R1_5line_z0p008", "R1_9line_z0p008"}
EXPECTED_LINE_COUNTS = {5, 9}
EXPECTED_SENSOR_Z = 0.008
EXPECTED_NOMINAL_JE = ["0", "1e6[A/m^2]", "0"]
EXPECTED_ORTHOGONAL_JE = ["1e6[A/m^2]", "0", "0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 internal multi-magnetization mag_y pack。")
    parser.add_argument("--mag-y-npz", type=Path, default=MAG_Y_NPZ)
    parser.add_argument("--comsol-inventory", type=Path, default=COMSOL_INVENTORY)
    parser.add_argument("--comsol-summary", type=Path, default=COMSOL_SUMMARY)
    parser.add_argument("--preflight-summary", type=Path, default=PREFLIGHT_SUMMARY)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
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


def strings(arr: np.ndarray | None) -> list[str]:
    if arr is None:
        return []
    return [str(x) for x in np.asarray(arr).reshape(-1).tolist()]


def as_bool_array(arr: np.ndarray | None) -> np.ndarray:
    if arr is None:
        return np.asarray([], dtype=bool)
    if np.asarray(arr).dtype == np.bool_:
        return np.asarray(arr, dtype=bool).reshape(-1)
    return np.asarray([str(x).lower() == "true" for x in np.asarray(arr).reshape(-1).tolist()], dtype=bool)


def git_lines(args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def add(rows: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any = "", notes: str = "") -> None:
    rows.append(
        {
            "check_name": name,
            "pass": bool(passed),
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def no_forbidden_staged() -> tuple[bool, str]:
    staged = git_lines(["diff", "--cached", "--name-only"])
    forbidden = [
        path
        for path in staged
        if path.startswith("data/")
        or path.endswith(".npz")
        or path.endswith(".mph")
        or path.startswith("checkpoints/")
        or path.startswith("results/previews/")
        or path.startswith("notes/")
        or path == "CURRENT_BASELINE.md"
        or path == "scripts/visualize_current_baseline.py"
    ]
    return not forbidden, "; ".join(forbidden)


def protected_paths_clean() -> tuple[bool, str]:
    lines = git_lines(
        [
            "status",
            "--short",
            "--",
            "data",
            "checkpoints",
            "results/previews",
            "notes",
            "CURRENT_BASELINE.md",
            "scripts/visualize_current_baseline.py",
        ]
    )
    return len(lines) == 0, "; ".join(lines)


def complete_bases(arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    by_base: dict[str, set[str]] = defaultdict(set)
    for base, variant in zip(strings(arrays.get("base_group_id")), strings(arrays.get("observation_variant")), strict=False):
        by_base[base].add(variant)
    complete = sum(1 for variants in by_base.values() if EXPECTED_VARIANTS.issubset(variants))
    return len(by_base), complete


def parse_je_values(values: list[str]) -> set[str]:
    parsed: set[str] = set()
    for value in values:
        try:
            parsed.add(json.dumps(json.loads(value), ensure_ascii=False, separators=(",", ":")))
        except Exception:
            parsed.add(value)
    return parsed


def group_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(len(arrays.get("sample_ids", []))) if arrays else 0
    base = strings(arrays.get("base_group_id")) or [""] * n
    for field in [
        "observation_variant",
        "paired_reference_variant",
        "magnetization_direction",
        "source_direction",
        "scan_line_count",
        "shape_type",
        "burial_depth_level",
        "size_level",
        "aspect_bin",
        "source_split",
    ]:
        if field not in arrays:
            continue
        buckets: dict[str, list[int]] = defaultdict(list)
        for idx, value in enumerate(strings(arrays[field])):
            buckets[value].append(idx)
        for value, indices in sorted(buckets.items()):
            rows.append(
                {
                    "group_type": field,
                    "group_value": value,
                    "row_count": len(indices),
                    "base_count": len({base[i] for i in indices}),
                }
            )
    return rows


def run(args: argparse.Namespace) -> int:
    inventory = read_csv(args.comsol_inventory)
    arrays = load_npz(args.mag_y_npz)
    checks: list[dict[str, Any]] = []

    success_inventory = [row for row in inventory if row.get("status") == "success"]
    n = int(len(arrays.get("sample_ids", []))) if arrays else 0
    base_count, complete_base_count = complete_bases(arrays) if arrays else (0, 0)

    add(checks, "preflight_summary_exists", args.preflight_summary.exists(), args.preflight_summary, "preflight summary")
    add(checks, "comsol_summary_exists", args.comsol_summary.exists(), args.comsol_summary, "COMSOL summary")
    add(checks, "comsol_inventory_exists", args.comsol_inventory.exists(), args.comsol_inventory, "COMSOL inventory")
    add(checks, "mag_y_npz_exists", args.mag_y_npz.exists(), args.mag_y_npz, "ignored NPZ exists")
    add(checks, "planned_success_rows_60", n == 60, n, 60)
    add(checks, "success_rows_match_inventory", n == len(success_inventory), f"npz={n}; inventory={len(success_inventory)}", "match")
    add(checks, "base_count_30", base_count == 30, base_count, 30)
    add(checks, "m1_m2_complete_bases_30", complete_base_count == 30, complete_base_count, 30)

    max_delta_err = float("nan")
    variants: set[str] = set()
    source_je_changed_all = False
    nominal_je_ok = False
    orthogonal_je_ok = False
    if arrays:
        expected_shape = (n, 3, 9, 201)
        for key in ["delta_b", "b_defect", "b_no_defect"]:
            arr = arrays.get(key)
            add(checks, f"{key}_shape", arr is not None and arr.shape == expected_shape, "" if arr is None else arr.shape, expected_shape)
            add(checks, f"{key}_finite", arr is not None and np.isfinite(arr).all(), "" if arr is None else bool(np.isfinite(arr).all()), "finite")
        if {"delta_b", "b_defect", "b_no_defect"}.issubset(arrays):
            max_delta_err = float(np.max(np.abs(arrays["delta_b"] - (arrays["b_defect"] - arrays["b_no_defect"]))))
        add(checks, "delta_b_matches_difference", np.isfinite(max_delta_err) and max_delta_err < 1e-7, max_delta_err, "<1e-7")

        variants = set(strings(arrays.get("observation_variant")))
        references = set(strings(arrays.get("paired_reference_variant")))
        line_counts = {int(x) for x in np.asarray(arrays.get("scan_line_count", [])).reshape(-1).tolist()}
        sensor_z = {round(float(x), 6) for x in np.asarray(arrays.get("sensor_z_m", [])).reshape(-1).tolist()}
        magnetization = set(strings(arrays.get("magnetization_direction")))
        source_direction = set(strings(arrays.get("source_direction")))
        j_direction = set(strings(arrays.get("J_direction")))
        scan_direction = set(strings(arrays.get("scan_direction")))
        axis_names = strings(arrays.get("axis_names"))
        source_je_changed_all = bool(as_bool_array(arrays.get("source_je_changed")).all())
        nominal_je = parse_je_values(strings(arrays.get("nominal_source_je_json")))
        orthogonal_je = parse_je_values(strings(arrays.get("orthogonal_source_je_json")))
        nominal_je_ok = nominal_je == {json.dumps(EXPECTED_NOMINAL_JE, separators=(",", ":"))}
        orthogonal_je_ok = orthogonal_je == {json.dumps(EXPECTED_ORTHOGONAL_JE, separators=(",", ":"))}

        add(checks, "variants_m1_m2_only", variants == EXPECTED_VARIANTS, variants, EXPECTED_VARIANTS)
        add(checks, "paired_references_r1_only", references == EXPECTED_REFERENCES, references, EXPECTED_REFERENCES)
        add(checks, "line_counts_5_and_9", line_counts == EXPECTED_LINE_COUNTS, line_counts, EXPECTED_LINE_COUNTS)
        add(checks, "sensor_z_nominal", sensor_z == {EXPECTED_SENSOR_Z}, sensor_z, {EXPECTED_SENSOR_Z})
        add(checks, "magnetization_direction_mag_y", magnetization == {"mag_y"}, magnetization, {"mag_y"})
        add(checks, "source_direction_records_orthogonal", source_direction == {"orthogonal_source_x_from_nominal_y"}, source_direction, "orthogonal source")
        add(checks, "j_direction_records_jx", j_direction == {"Jx"}, j_direction, {"Jx"})
        add(checks, "scan_direction_x_scan", scan_direction == {"x_scan"}, scan_direction, {"x_scan"})
        add(checks, "axis_names_bxyz", axis_names == ["Bx", "By", "Bz"], axis_names, ["Bx", "By", "Bz"])
        add(checks, "source_je_changed_all_true", source_je_changed_all, source_je_changed_all, "all true")
        add(checks, "nominal_source_je_expected", nominal_je_ok, nominal_je, EXPECTED_NOMINAL_JE)
        add(checks, "orthogonal_source_je_expected", orthogonal_je_ok, orthogonal_je, EXPECTED_ORTHOGONAL_JE)
        add(checks, "source_je_not_metadata_only", source_je_changed_all and nominal_je_ok and orthogonal_je_ok, f"{nominal_je} -> {orthogonal_je}", "true source vector change")

        sample_ids = strings(arrays.get("sample_ids"))
        add(checks, "sample_id_unique", len(sample_ids) == len(set(sample_ids)), f"{len(sample_ids)} rows/{len(set(sample_ids))} unique", "unique")
        required_labels = {
            "shape_type",
            "L_m",
            "W_m",
            "D_m",
            "D_m_or_cavity_size_m",
            "burial_depth_m",
            "depth_to_surface_m",
            "defect_center_xyz_m",
            "cavity_internal",
            "ground_truth_method",
        }
        add(checks, "required_labels_present", required_labels.issubset(arrays), sorted(required_labels.intersection(arrays)), sorted(required_labels))
        if "cavity_internal" in arrays:
            cavity = np.asarray(arrays["cavity_internal"]).astype(bool)
            add(checks, "cavity_internal_true", bool(cavity.all()), bool(cavity.all()), "all true")

    staged_ok, staged_notes = no_forbidden_staged()
    protected_ok, protected_notes = protected_paths_clean()
    add(checks, "no_forbidden_staged", staged_ok, staged_notes, "no forbidden staged files")
    add(checks, "protected_artifact_paths_clean", protected_ok, protected_notes, "no tracked forbidden path changes")

    validation_passed = all(bool(row["pass"]) for row in checks)
    write_csv(args.metrics, checks, ["check_name", "pass", "observed", "expected", "notes"])
    write_csv(args.group_summary, group_rows(arrays), ["group_type", "group_value", "row_count", "base_count"])

    lines = [
        "23.4 internal multi-magnetization pack validation summary",
        "",
        f"planned_success_rows: {n}",
        f"base_count: {base_count}",
        f"m1_m2_complete_base_count: {complete_base_count}",
        f"variant_counts: {dict(Counter(strings(arrays.get('observation_variant')))) if arrays else {}}",
        f"validation_passed: {str(validation_passed).lower()}",
        f"max_delta_error: {max_delta_err}",
        f"source_je_changed_all_true: {str(source_je_changed_all).lower()}",
        f"nominal_source_je: {EXPECTED_NOMINAL_JE}",
        f"orthogonal_source_je: {EXPECTED_ORTHOGONAL_JE}",
        "magnetization_metadata_only: false",
        "training_run: false",
        "comsol_run_by_this_script: false",
        "data_npz_modified_by_this_script: false",
        "current_baseline_updated: false",
        "",
        "结论：mag_y diagnostic pack 通过显式 COMSOL source Je 改向生成，可进入组装验证；本轮不是训练集，也不是 baseline。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if validation_passed else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
