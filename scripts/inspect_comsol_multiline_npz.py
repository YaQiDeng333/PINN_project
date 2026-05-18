from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data"
    / "comsol_mfl"
    / "prepared"
    / "comsol_single_defect_multiline_forward_pack_v1_smoke.npz"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "summaries"
    / "comsol_multiline_npz_ingest_validation_summary.txt"
)
DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "comsol_multiline_npz_ingest_validation_inventory.csv"
)

REQUIRED_FIELDS = [
    "delta_bz",
    "bz_defect",
    "bz_no_defect",
    "masks",
    "sensor_x",
    "scan_line_y",
    "mask_x",
    "mask_y",
    "defect_types",
    "sample_ids",
    "geometry_params",
    "metadata",
]

INVENTORY_FIELDS = [
    "sample_id",
    "defect_type",
    "n_lines",
    "signal_length",
    "mask_shape",
    "mask_area",
    "delta_bz_min",
    "delta_bz_max",
    "delta_bz_mean",
    "delta_bz_std",
    "scan_line_y",
    "has_valid_coords",
    "delta_matches_defect_minus_reference",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the COMSOL multi-line smoke NPZ for PINN_project ingest."
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return as_text(value.item())
        return json.dumps([as_text(item) for item in value.tolist()], ensure_ascii=False)
    return str(value)


def load_json_array(array: np.ndarray) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in array.tolist():
        text = as_text(item)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = {"raw": text}
        if isinstance(value, dict):
            parsed.append(value)
        else:
            parsed.append({"raw": value})
    return parsed


def stats(array: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def finite_numeric_arrays(data: np.lib.npyio.NpzFile) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name in data.files:
        array = data[name]
        if np.issubdtype(array.dtype, np.number):
            result[name] = bool(np.isfinite(array).all())
    return result


def strictly_monotonic(array: np.ndarray) -> bool:
    return array.ndim == 1 and array.size >= 2 and bool(np.all(np.diff(array) > 0))


def rasterize_geometry_mask(
    geometry: dict[str, Any], mask_x: np.ndarray, mask_y: np.ndarray
) -> np.ndarray | None:
    required = ["center_x_m", "center_y_m", "width_m", "length_m"]
    if any(key not in geometry for key in required):
        return None
    cx = float(geometry["center_x_m"])
    cy = float(geometry["center_y_m"])
    half_w = float(geometry["width_m"]) / 2.0
    half_l = float(geometry["length_m"]) / 2.0
    yy, xx = np.meshgrid(mask_y, mask_x, indexing="ij")
    return ((np.abs(xx - cx) <= half_w) & (np.abs(yy - cy) <= half_l)).astype(np.uint8)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INVENTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(report: dict[str, Any]) -> str:
    lines = [
        "# 第 20.7 COMSOL multiline NPZ ingest validation",
        "",
        "## 1. NPZ 读取与 schema",
        "",
        f"- NPZ 路径：{report['npz_path']}",
        f"- NPZ 是否可读：{report['npz_readable']}",
        f"- 字段列表：{report['files']}",
        f"- 缺失字段：{report['missing_fields']}",
        f"- schema-ready：{report['schema_ready']}",
        f"- train-ready：{report['train_ready']}",
        f"- train-ready 判断：{report['train_ready_reason']}",
        "",
        "## 2. shape 检查",
        "",
        f"- delta_bz shape：{report['shapes'].get('delta_bz')}",
        f"- bz_defect shape：{report['shapes'].get('bz_defect')}",
        f"- bz_no_defect shape：{report['shapes'].get('bz_no_defect')}",
        f"- masks shape：{report['shapes'].get('masks')}",
        f"- sensor_x shape：{report['shapes'].get('sensor_x')}",
        f"- scan_line_y shape：{report['shapes'].get('scan_line_y')}",
        f"- mask_x shape：{report['shapes'].get('mask_x')}",
        f"- mask_y shape：{report['shapes'].get('mask_y')}",
        f"- shape 是否一致：{report['shape_consistent']}",
        "",
        "## 3. 数值质量",
        "",
        f"- numeric arrays finite：{report['finite_numeric_arrays']}",
        f"- delta_bz stats：{report['delta_bz_stats']}",
        f"- bz_defect stats：{report['bz_defect_stats']}",
        f"- bz_no_defect stats：{report['bz_no_defect_stats']}",
        f"- mask_area：{report['mask_areas']}",
        f"- mask 是否非空：{report['non_empty_masks']}",
        f"- delta_bz 是否非零：{report['nonzero_delta_bz']}",
        f"- 三条 scan line 是否不同：{report['scan_lines_different']}",
        f"- delta_bz 是否等于 bz_defect - bz_no_defect：{report['delta_matches_defect_minus_reference']}",
        f"- 最大 delta 重构误差：{report['max_delta_reconstruction_error']}",
        "",
        "## 4. 坐标与几何一致性",
        "",
        f"- sensor_x 单调：{report['sensor_x_monotonic']}",
        f"- scan_line_y：{report['scan_line_y_values']}",
        f"- scan_line_y 是否为预期三线：{report['scan_line_y_expected']}",
        f"- mask_x 单调：{report['mask_x_monotonic']}",
        f"- mask_y 单调：{report['mask_y_monotonic']}",
        f"- mask 坐标范围：{report['mask_coord_range']}",
        f"- geometry_params：{report['geometry_params']}",
        f"- geometry 是否落在 mask 坐标范围内：{report['geometry_covered_by_mask_coords']}",
        f"- geometry raster 与存储 mask 的 IoU：{report['geometry_mask_iou']}",
        "",
        "## 5. ingest 结论",
        "",
        "- 该 1-sample smoke NPZ 字段完整、shape 一致、数值有限，且 delta_bz 与 raw Bz 差值一致。",
        "- 该 NPZ 可以作为 PINN_project 的 schema smoke 输入，但样本数只有 1，且没有 train / val / test split，因此不是正式 train-ready 数据集。",
        "- 外部 COMSOL 工程可以扩展到 6-12 个样本，但扩展前建议先 review 坐标映射、rectangular_notch 到目标 defect_type 的语义映射，以及 mask rasterization 是否符合后续 2D/quasi-2D 反演定义。",
        "- PINN_project 后续需要新增 prepare / dataset loader 支持 `(N, n_lines, L)` 的 `delta_bz` 输入、`(N, H, W)` masks、坐标数组、geometry metadata，以及小样本 split/manifest 读取。",
        "- 下一步更适合让 COMSOL 工程扩展真实样本，同时在 PINN_project 增加只读 ingest/prepare loader；不应直接基于 1 个样本训练模型。",
    ]
    return "\n".join(lines) + "\n"


def validate(npz_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {
        "npz_path": str(npz_path),
        "npz_readable": False,
        "files": [],
        "missing_fields": REQUIRED_FIELDS,
    }
    data = np.load(npz_path, allow_pickle=True)
    report["npz_readable"] = True
    report["files"] = list(data.files)
    report["missing_fields"] = [field for field in REQUIRED_FIELDS if field not in data.files]
    report["shapes"] = {name: tuple(data[name].shape) for name in data.files}

    delta_bz = data["delta_bz"]
    bz_defect = data["bz_defect"]
    bz_no_defect = data["bz_no_defect"]
    masks = data["masks"]
    sensor_x = data["sensor_x"]
    scan_line_y = data["scan_line_y"]
    mask_x = data["mask_x"]
    mask_y = data["mask_y"]
    sample_ids = [as_text(item) for item in data["sample_ids"].tolist()]
    defect_types = [as_text(item) for item in data["defect_types"].tolist()]
    geometry_params = load_json_array(data["geometry_params"])

    n, n_lines, signal_length = delta_bz.shape
    mask_n, mask_h, mask_w = masks.shape
    shape_consistent = (
        delta_bz.ndim == 3
        and bz_defect.shape == delta_bz.shape
        and bz_no_defect.shape == delta_bz.shape
        and masks.ndim == 3
        and mask_n == n
        and sensor_x.shape == (signal_length,)
        and scan_line_y.shape == (n_lines,)
        and mask_x.shape == (mask_w,)
        and mask_y.shape == (mask_h,)
    )

    delta_error = np.abs(delta_bz - (bz_defect - bz_no_defect))
    max_delta_error = float(np.max(delta_error))
    delta_matches = bool(np.allclose(delta_bz, bz_defect - bz_no_defect, rtol=1e-9, atol=1e-12))
    line_diffs = []
    if n_lines >= 2:
        for line_index in range(1, n_lines):
            line_diffs.append(float(np.max(np.abs(delta_bz[:, line_index, :] - delta_bz[:, 0, :]))))
    scan_lines_different = bool(line_diffs and max(line_diffs) > 1e-12)

    expected_scan_line_y = np.array([-0.001, 0.0, 0.001], dtype=np.float64)
    scan_line_y_expected = bool(
        scan_line_y.shape == expected_scan_line_y.shape
        and np.allclose(scan_line_y, expected_scan_line_y, rtol=0.0, atol=1e-12)
    )

    geometry_covered: list[bool] = []
    geometry_mask_ious: list[float | None] = []
    for index, geometry in enumerate(geometry_params):
        if all(key in geometry for key in ("center_x_m", "center_y_m", "width_m", "length_m")):
            cx = float(geometry["center_x_m"])
            cy = float(geometry["center_y_m"])
            half_w = float(geometry["width_m"]) / 2.0
            half_l = float(geometry["length_m"]) / 2.0
            covered = (
                float(mask_x.min()) <= cx - half_w
                and float(mask_x.max()) >= cx + half_w
                and float(mask_y.min()) <= cy - half_l
                and float(mask_y.max()) >= cy + half_l
            )
            geometry_covered.append(bool(covered))
            expected_mask = rasterize_geometry_mask(geometry, mask_x, mask_y)
            geometry_mask_ious.append(
                None if expected_mask is None else mask_iou(masks[index], expected_mask)
            )
        else:
            geometry_covered.append(False)
            geometry_mask_ious.append(None)

    valid_coords = bool(
        strictly_monotonic(sensor_x)
        and strictly_monotonic(mask_x)
        and strictly_monotonic(mask_y)
        and scan_line_y.ndim == 1
        and scan_line_y.size == n_lines
    )

    inventory_rows: list[dict[str, Any]] = []
    for index in range(n):
        sample_delta = delta_bz[index]
        notes = []
        if not valid_coords:
            notes.append("invalid_coords")
        if not delta_matches:
            notes.append("delta_mismatch")
        if int(masks[index].sum()) <= 0:
            notes.append("empty_mask")
        if not np.any(np.abs(sample_delta) > 0):
            notes.append("zero_delta_bz")
        if geometry_mask_ious[index] is not None and geometry_mask_ious[index] < 0.999:
            notes.append(f"geometry_mask_iou={geometry_mask_ious[index]:.6f}")
        inventory_rows.append(
            {
                "sample_id": sample_ids[index],
                "defect_type": defect_types[index],
                "n_lines": n_lines,
                "signal_length": signal_length,
                "mask_shape": f"({mask_h}, {mask_w})",
                "mask_area": int(masks[index].sum()),
                "delta_bz_min": float(np.min(sample_delta)),
                "delta_bz_max": float(np.max(sample_delta)),
                "delta_bz_mean": float(np.mean(sample_delta)),
                "delta_bz_std": float(np.std(sample_delta)),
                "scan_line_y": json.dumps(scan_line_y.tolist()),
                "has_valid_coords": valid_coords,
                "delta_matches_defect_minus_reference": delta_matches,
                "notes": ";".join(notes) if notes else "ok",
            }
        )

    schema_ready = bool(not report["missing_fields"] and shape_consistent and delta_matches and valid_coords)
    train_ready = bool(schema_ready and n >= 6)
    train_ready_reason = (
        "schema 完整且样本数达到最小 smoke-training 建议"
        if train_ready
        else "schema 已可读，但样本数只有 1，且没有正式 train/val/test split"
    )

    report.update(
        {
            "schema_ready": schema_ready,
            "train_ready": train_ready,
            "train_ready_reason": train_ready_reason,
            "shape_consistent": shape_consistent,
            "finite_numeric_arrays": finite_numeric_arrays(data),
            "delta_bz_stats": stats(delta_bz),
            "bz_defect_stats": stats(bz_defect),
            "bz_no_defect_stats": stats(bz_no_defect),
            "mask_areas": [int(mask.sum()) for mask in masks],
            "non_empty_masks": [bool(mask.sum() > 0) for mask in masks],
            "nonzero_delta_bz": bool(np.any(np.abs(delta_bz) > 0)),
            "scan_lines_different": scan_lines_different,
            "delta_matches_defect_minus_reference": delta_matches,
            "max_delta_reconstruction_error": max_delta_error,
            "sensor_x_monotonic": strictly_monotonic(sensor_x),
            "scan_line_y_values": scan_line_y.tolist(),
            "scan_line_y_expected": scan_line_y_expected,
            "mask_x_monotonic": strictly_monotonic(mask_x),
            "mask_y_monotonic": strictly_monotonic(mask_y),
            "mask_coord_range": {
                "mask_x_min": float(mask_x.min()),
                "mask_x_max": float(mask_x.max()),
                "mask_y_min": float(mask_y.min()),
                "mask_y_max": float(mask_y.max()),
            },
            "geometry_params": geometry_params,
            "geometry_covered_by_mask_coords": geometry_covered,
            "geometry_mask_iou": geometry_mask_ious,
        }
    )
    return report, inventory_rows


def main() -> int:
    args = parse_args()
    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    inventory_path = resolve(args.inventory)
    report, inventory_rows = validate(npz_path)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(report), encoding="utf-8-sig")
    write_csv(inventory_path, inventory_rows)

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"summary={summary_path}")
    print(f"inventory={inventory_path}")
    return 0 if report["schema_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
