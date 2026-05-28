#!/usr/bin/env python
"""Validate a manifest-only real-data intake contract for true 3D RBC inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/templates/real_data_intake_manifest_template.json"
PREFLIGHT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_real_data_schema_preflight_summary.txt"
VALIDATION_SUMMARY = ROOT / "results/summaries/true_3d_rbc_real_data_schema_validation_summary.txt"
VALIDATION_MATRIX = ROOT / "results/metrics/true_3d_rbc_real_data_schema_validation_matrix.csv"

REQUIRED_20_96 = [
    ROOT / "results/summaries/true_3d_rbc_liftoff_inference_metadata_contract.md",
    ROOT / "results/summaries/true_3d_rbc_liftoff_conditioned_inference_smoke_summary.txt",
    ROOT / "scripts/run_true_3d_rbc_liftoff_conditioned_inference.py",
    ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json",
    ROOT / "results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json",
    ROOT / "COMSOL_DATA_REGISTRY.md",
    ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json",
    ROOT / "results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json",
    ROOT / "CURRENT_BASELINE.md",
]
AXIS_ORDER = ["Bx", "By", "Bz"]
SCAN_LINE_Y_M = [-0.001, 0.0, 0.001]
SENSOR_Z_MIN = 0.006
SENSOR_Z_MAX = 0.012
NOMINAL_SENSOR_Z = 0.008
NOMINAL_TOL = 0.0005
RANGE_TOL = 1.0e-9


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def staged_files() -> list[str]:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def forbidden_staged(paths: list[str]) -> list[str]:
    forbidden: list[str] = []
    for path in paths:
        if (
            path.startswith(("data/", "checkpoints/", "notes/", "results/previews/"))
            or path.endswith((".npz", ".pt", ".pth", ".ckpt", ".png", ".mph"))
            or path in {"CURRENT_BASELINE.md", "scripts/visualize_current_baseline.py"}
        ):
            forbidden.append(path)
    return forbidden


def write_preflight() -> None:
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, notes: str = "") -> None:
        rows.append({"check": name, "pass": bool(passed), "observed": observed, "notes": notes})

    for path in REQUIRED_20_96:
        add(f"exists:{path.relative_to(ROOT)}", path.exists(), str(path))
    current = (ROOT / "CURRENT_BASELINE.md").read_text(encoding="utf-8", errors="replace") if (ROOT / "CURRENT_BASELINE.md").exists() else ""
    add("current_baseline_true_3d_rbc", "true 3D RBC" in current and "20.85" in current, "CURRENT_BASELINE.md")
    registry = (ROOT / "COMSOL_DATA_REGISTRY.md").read_text(encoding="utf-8", errors="replace") if (ROOT / "COMSOL_DATA_REGISTRY.md").exists() else ""
    add("registry_has_v3_240", "comsol_true_3d_rbc_imported_watertight_pilot_v3_240" in registry, "COMSOL_DATA_REGISTRY.md")
    add("registry_has_liftoff_pack", "comsol_true_3d_rbc_liftoff_aug_pack_v1" in registry, "COMSOL_DATA_REGISTRY.md")
    staged = staged_files()
    blocked = forbidden_staged(staged)
    add("forbidden_staged_artifacts", not blocked, blocked or "none")
    add("no_COMSOL_run", True, "not run")
    add("no_training_run", True, "not run")
    add("no_data_npz_mutation", True, "not modified")
    add("no_CURRENT_BASELINE_update", True, "not modified")
    failed = [row for row in rows if not row["pass"]]
    lines = [
        "20.97 true 3D RBC real-data schema intake preflight",
        "",
        *[f"{row['check']}: pass={row['pass']} observed={row['observed']} notes={row['notes']}" for row in rows],
        "",
        f"preflight_pass: {not failed}",
        "latest_newest_npz_scan: false",
        "real_data_generation: false",
        "NPZ_write: false",
        "CURRENT_BASELINE_update: false",
        "stop_condition: stop if 20.96 inference artifacts, registry/manifest records, or CURRENT_BASELINE state are missing.",
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError("preflight failed; see " + str(PREFLIGHT_SUMMARY))


def parse_shape(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        try:
            return [int(x) for x in value]
        except Exception:
            return None
    if isinstance(value, str):
        cleaned = value.replace("|", ",").replace("x", ",").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        try:
            return [int(part.strip()) for part in cleaned.split(",") if part.strip()]
        except Exception:
            return None
    return None


def valid_signal_shape(shape: list[int] | None) -> bool:
    if shape is None:
        return False
    return shape == [3, 3, 201] or (len(shape) == 4 and shape[-3:] == [3, 3, 201])


def has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped.startswith("<") and stripped.endswith(">")
    if isinstance(value, dict):
        return any(has_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(has_placeholder(item) for item in value)
    return False


def parse_axis_order(value: Any) -> list[str] | None:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    if isinstance(value, str) and value.strip():
        sep = "|" if "|" in value else ","
        return [part.strip() for part in value.split(sep) if part.strip()]
    return None


def is_unknownish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "unknown", "optional/unknown", "unk", "n/a", "na", "none"}
    return False


def is_positive(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "yes", "y", "1", "available", "present"}


def route_for_sensor_z(sensor_z_m: float | None) -> tuple[str, bool]:
    if sensor_z_m is None or not math.isfinite(sensor_z_m):
        return "blocker_missing_sensor_z_m", True
    out = sensor_z_m < SENSOR_Z_MIN - RANGE_TOL or sensor_z_m > SENSOR_Z_MAX + RANGE_TOL
    if out:
        return "out_of_range_blocker", True
    if abs(sensor_z_m - NOMINAL_SENSOR_Z) < NOMINAL_TOL or abs(sensor_z_m - NOMINAL_SENSOR_Z) < 1.0e-12:
        return "baseline", False
    return "baseline_plus_adapter", False


def add_check(rows: list[dict[str, Any]], severity: str, item: str, field: str, passed: bool, observed: Any, message: str, route: str = "") -> None:
    rows.append(
        {
            "severity": severity,
            "item": item,
            "field": field,
            "pass": bool(passed),
            "observed": observed,
            "message": message,
            "route": route,
        }
    )


def table_samples(path: Path) -> list[dict[str, Any]]:
    if not path:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_manifest(manifest: dict[str, Any], sample_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fmt = manifest.get("format")
    add_check(rows, "blocker", "manifest", "format", fmt in {"prepared_delta_b", "raw_defect_no_defect"}, fmt, "format must be prepared_delta_b or raw_defect_no_defect")
    defect_location_type = manifest.get("defect_location_type")
    internal_like = str(defect_location_type).strip().lower() in {"internal_or_buried", "internal", "buried", "subsurface"}
    add_check(rows, "blocker", "manifest", "defect_location_type", not internal_like, defect_location_type, "internal/buried 缺陷需要独立 schema，不能直接进入当前 surface RBC baseline")
    if str(manifest.get("schema_branch_recommendation", "")).strip().lower() == "internal_defect_feasibility_required":
        add_check(rows, "warning", "manifest", "schema_branch_recommendation", False, manifest.get("schema_branch_recommendation"), "路线建议指向 internal defect feasibility 分支")
    if not is_positive(manifest.get("data_available", True)):
        add_check(rows, "blocker", "manifest", "data_available", False, manifest.get("data_available"), "没有真实信号数组，不能推理")
    if not is_positive(manifest.get("bxyz_available", True)):
        add_check(rows, "blocker", "manifest", "bxyz_available", False, manifest.get("bxyz_available"), "必须提供三轴 Bx/By/Bz")
    if not is_positive(manifest.get("no_defect_reference_available", True)):
        add_check(rows, "blocker", "manifest", "no_defect_reference_available", False, manifest.get("no_defect_reference_available"), "必须提供匹配的 no-defect reference")
    if has_placeholder(manifest.get("data_file")):
        add_check(rows, "warning", "manifest", "data_file", False, manifest.get("data_file"), "data_file still contains a template placeholder")
    meta = manifest.get("global_metadata", {})
    axis_order = meta.get("axis_order")
    add_check(rows, "blocker", "global_metadata", "axis_order", axis_order == AXIS_ORDER, axis_order, "axis_order 必须是 [Bx, By, Bz]")
    add_check(rows, "blocker", "global_metadata", "axis_order_bz_only", axis_order != ["Bz"], axis_order, "只有 Bz 是当前 true 3D RBC 路线的 blocker")
    scan_line = meta.get("scan_line_y_m")
    scan_ok = isinstance(scan_line, list) and len(scan_line) == 3 and all(abs(float(a) - b) < 1.0e-9 for a, b in zip(scan_line, SCAN_LINE_Y_M))
    add_check(rows, "blocker", "global_metadata", "scan_line_y_m", scan_ok, scan_line, "scan_line_y_m 必须映射为三条扫描线，推荐 [-0.001,0,0.001]")
    sensor_x_count = meta.get("sensor_x_m_count")
    sensor_x = meta.get("sensor_x_m")
    sensor_x_ok = sensor_x_count == 201 or (isinstance(sensor_x, list) and len(sensor_x) == 201)
    add_check(rows, "blocker", "global_metadata", "sensor_x_m", sensor_x_ok, sensor_x_count if sensor_x_count is not None else type(sensor_x).__name__, "sensor_x_m 必须有 201 个采样点，或提供 sensor_x_m_count=201")
    if has_placeholder(sensor_x):
        add_check(rows, "warning", "global_metadata", "sensor_x_m", False, sensor_x, "sensor_x_m still contains a template placeholder")
    unit = meta.get("delta_b_unit") or meta.get("b_unit")
    add_check(rows, "blocker", "global_metadata", "unit", unit == "Tesla", unit, "磁场单位必须是 Tesla")
    for field in ("coordinate_system", "no_defect_reference_method", "sensor_alignment_status", "gain_calibration_status", "material", "specimen_info", "magnetization_setup"):
        known = bool(meta.get(field)) and not is_unknownish(meta.get(field))
        add_check(rows, "blocker" if field in {"coordinate_system", "no_defect_reference_method", "sensor_alignment_status", "gain_calibration_status", "magnetization_setup"} else "warning", "global_metadata", field, known, meta.get(field, ""), f"必须提供 {field}")
        if has_placeholder(meta.get(field)):
            add_check(rows, "warning", "global_metadata", field, False, meta.get(field), f"{field} still contains a template placeholder")
    if meta.get("gain_calibration_status") in {"unknown", "", None}:
        add_check(rows, "warning", "global_metadata", "gain_calibration_status", False, meta.get("gain_calibration_status"), "gain/amplitude calibration unknown; report as diagnostic risk")
    if meta.get("sensor_alignment_status") not in {"verified_aligned", "aligned"}:
        add_check(rows, "warning", "global_metadata", "sensor_alignment_status", False, meta.get("sensor_alignment_status"), "sensor alignment is not explicitly verified")

    samples = sample_table or manifest.get("samples", [])
    add_check(rows, "blocker", "manifest", "samples", bool(samples), len(samples) if isinstance(samples, list) else type(samples).__name__, "at least one sample row is required")
    if not isinstance(samples, list):
        samples = []
    for idx, sample in enumerate(samples):
        item = str(sample.get("sample_id") or f"sample_{idx}")
        if sample_table:
            sample_format = sample.get("format") or fmt
            add_check(rows, "blocker", item, "sample_table.format_present", bool(sample.get("format")), sample.get("format", ""), "sample table format column must be present and non-empty")
            add_check(rows, "blocker", item, "sample_table.format", sample_format == fmt, sample_format, "sample table format must match manifest format")
            sample_axis = parse_axis_order(sample.get("axis_order"))
            add_check(rows, "blocker", item, "sample_table.axis_order_present", sample_axis is not None, sample.get("axis_order", ""), "sample table axis_order column must be present and non-empty")
            if sample_axis is not None:
                add_check(rows, "blocker", item, "sample_table.axis_order", sample_axis == AXIS_ORDER, sample_axis, "sample table axis_order must be Bx|By|Bz")
            sample_unit = sample.get("delta_b_unit")
            add_check(rows, "blocker", item, "sample_table.delta_b_unit_present", bool(sample_unit), sample_unit or "", "sample table delta_b_unit column must be present and non-empty")
            if sample_unit:
                add_check(rows, "blocker", item, "sample_table.delta_b_unit", sample_unit == "Tesla", sample_unit, "sample table delta_b_unit must be Tesla")
            if str(sample.get("format", "")) == "raw_defect_no_defect" and fmt == "prepared_delta_b":
                add_check(rows, "blocker", item, "sample_table.format_conflict", False, sample.get("format"), "sample table raw rows cannot be validated against prepared_delta_b manifest")
        add_check(rows, "blocker", item, "sample_id", bool(sample.get("sample_id")) and not is_unknownish(sample.get("sample_id")), sample.get("sample_id", ""), "sample_id is required")
        add_check(rows, "blocker", item, "specimen_id", bool(sample.get("specimen_id")) and not is_unknownish(sample.get("specimen_id")), sample.get("specimen_id", ""), "specimen_id is required")
        add_check(rows, "blocker", item, "no_defect_reference_id", bool(sample.get("no_defect_reference_id")) and not is_unknownish(sample.get("no_defect_reference_id")), sample.get("no_defect_reference_id", ""), "必须提供 no_defect_reference_id")
        for field in ("sample_id", "specimen_id", "no_defect_reference_id"):
            if has_placeholder(sample.get(field)):
                add_check(rows, "warning", item, field, False, sample.get(field), f"{field} still contains a template placeholder")
        try:
            z = float(sample["sensor_z_m"]) if sample.get("sensor_z_m") not in (None, "") else None
        except Exception:
            z = None
        route, route_blocker = route_for_sensor_z(z)
        add_check(rows, "blocker", item, "sensor_z_m", not route_blocker, z, "sensor_z_m 必须提供，且必须在 [0.006,0.012] 内", route)
        if fmt == "prepared_delta_b":
            shape = parse_shape(sample.get("delta_b_shape") or manifest.get("delta_b_shape"))
            add_check(rows, "blocker", item, "delta_b_shape", valid_signal_shape(shape), shape, "delta_b shape 必须是 (3,3,201) 或 (N,3,3,201)", route)
        elif fmt == "raw_defect_no_defect":
            defect = parse_shape(sample.get("b_defect_shape") or manifest.get("b_defect_shape"))
            ref = parse_shape(sample.get("b_no_defect_shape") or manifest.get("b_no_defect_shape"))
            add_check(rows, "blocker", item, "b_defect_shape", valid_signal_shape(defect), defect, "b_defect shape must be (3,3,201) or (N,3,3,201)", route)
            add_check(rows, "blocker", item, "b_no_defect_shape", valid_signal_shape(ref), ref, "b_no_defect shape must match (3,3,201) or (N,3,3,201)", route)
        if str(sample.get("ground_truth_LWD_available", "")).lower() in {"false", "0", "no"} or sample.get("ground_truth_LWD", {}).get("available") is False:
            add_check(rows, "warning", item, "ground_truth_LWD", True, "not_available", "inference can run but L/W/D scoring will be unavailable", route)
        if str(sample.get("profile_depth_ground_truth_available", "")).lower() in {"false", "0", "no"} or sample.get("profile_depth_ground_truth", {}).get("available") is False:
            add_check(rows, "warning", item, "profile_depth_ground_truth", True, "not_available", "inference can run but profile scoring will be unavailable", route)
    return rows


def write_validation_summary(
    rows: list[dict[str, Any]],
    manifest_path: Path,
    sample_table_path: Path | None,
    summary_path: Path = VALIDATION_SUMMARY,
) -> None:
    blockers = [row for row in rows if row["severity"] == "blocker" and not row["pass"]]
    warnings = [row for row in rows if row["severity"] == "warning" and not row["pass"]]
    placeholder_warnings = [row for row in warnings if "placeholder" in str(row.get("message", "")).lower()]
    ready = not blockers and not placeholder_warnings
    routes = sorted({str(row.get("route", "")) for row in rows if row.get("route")})
    lines = [
        "true 3D RBC 真实数据接入 schema 验证",
        "",
        f"manifest: {manifest_path}",
        f"sample_table: {sample_table_path if sample_table_path else '未提供'}",
        f"ready_for_inference: {str(ready).lower()}",
        f"blocker_count: {len(blockers)}",
        f"warning_count: {len(warnings)}",
        f"placeholder_warning_count: {len(placeholder_warnings)}",
        f"routes_detected: {routes}",
        "",
        "主要 hard blockers:",
    ]
    if blockers:
        lines.extend([f"- {row['item']}::{row['field']} observed={row['observed']} message={row['message']}" for row in blockers[:20]])
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "validator_scope: 本阶段允许 manifest-only 验证，不要求真实数据文件存在。",
            "COMSOL_run: false",
            "training_run: false",
            "NPZ_write: false",
            "CURRENT_BASELINE_update: false",
        ]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate true 3D RBC real-data intake manifest/schema.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-table", type=Path)
    parser.add_argument("--summary", type=Path, default=VALIDATION_SUMMARY)
    parser.add_argument("--matrix", type=Path, default=VALIDATION_MATRIX)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_preflight:
        write_preflight()
    manifest = read_json(args.manifest)
    table = table_samples(args.sample_table) if args.sample_table else []
    rows = validate_manifest(manifest, table)
    write_csv(args.matrix, rows)
    write_validation_summary(rows, args.manifest, args.sample_table, args.summary)
    blockers = [row for row in rows if row["severity"] == "blocker" and not row["pass"]]
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
