"""Diagnose COMSOL polygon inverse held-out generalization failures."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


SPLITS = ("train", "val", "test")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: str | float | int, default: float = 0.0) -> float:
    if value == "" or value is None:
        return default
    return float(value)


def _as_bool(value: str | bool | int | float) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return True
    if text in {"false", "no", "n", ""}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return False


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _min(values: Iterable[float]) -> float:
    vals = list(values)
    return float(min(vals)) if vals else 0.0


def _max(values: Iterable[float]) -> float:
    vals = list(values)
    return float(max(vals)) if vals else 0.0


def _shoelace(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _component_vertices(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [[_as_float(row[f"{prefix}_x{i}"]), _as_float(row[f"{prefix}_y{i}"])] for i in range(4)],
        dtype=np.float64,
    )


def _polygon_vertices_from_params(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [[_as_float(row[f"norm_x{i}"]), _as_float(row[f"norm_y{i}"])] for i in range(4)],
        dtype=np.float64,
    )


def _signal_stats(npz_path: Path) -> dict[int, dict[str, float]]:
    with np.load(npz_path, allow_pickle=True) as data:
        signals = data["signals"].astype(np.float64)
    if signals.ndim != 3:
        raise ValueError(f"signals must have shape [N,C,L], got {signals.shape}")
    stats: dict[int, dict[str, float]] = {}
    for idx, sample in enumerate(signals):
        channel_std = sample.std(axis=1)
        channel_ptp = np.ptp(sample, axis=1)
        ratio = float(channel_std[-1] / channel_std[0]) if channel_std[0] != 0.0 else 0.0
        stats[idx] = {
            "signal_std_mean": float(channel_std.mean()),
            "signal_std_min": float(channel_std.min()),
            "signal_std_max": float(channel_std.max()),
            "signal_peak_to_peak_mean": float(channel_ptp.mean()),
            "signal_peak_to_peak_min": float(channel_ptp.min()),
            "signal_peak_to_peak_max": float(channel_ptp.max()),
            "lift_off_std_ratio_last_first": ratio,
        }
    return stats


def _sample_geometry(split: str, raw_root: Path, converted_dir: Path) -> list[dict]:
    defect_rows = _read_csv(raw_root / split / "defect_params.csv")
    polygon_rows = _read_csv(raw_root / split / "polygon_params.csv")
    signal_by_sample = _signal_stats(converted_dir / f"{split}_comsol_v3_polygon_hard_case.npz")
    polygons_by_sample: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in polygon_rows:
        if _as_bool(row.get("presence", "1")):
            polygons_by_sample[int(row["sample_index"])].append(row)
    out = []
    for row in defect_rows:
        sample_index = int(row["sample_index"])
        polys = polygons_by_sample.get(sample_index, [])
        vertices = [_polygon_vertices_from_params(item) for item in polys]
        if vertices:
            all_vertices = np.concatenate(vertices, axis=0)
            areas = [abs(_shoelace(item)) for item in vertices]
            signed_areas = [_shoelace(item) for item in vertices]
            bbox_width = float(all_vertices[:, 0].max() - all_vertices[:, 0].min())
            bbox_height = float(all_vertices[:, 1].max() - all_vertices[:, 1].min())
            first_edge = vertices[0][1] - vertices[0][0]
            orientation_deg = float(math.degrees(math.atan2(first_edge[1], first_edge[0])))
            vertex_x_min = float(all_vertices[:, 0].min())
            vertex_x_max = float(all_vertices[:, 0].max())
            vertex_y_min = float(all_vertices[:, 1].min())
            vertex_y_max = float(all_vertices[:, 1].max())
        else:
            areas = [0.0]
            signed_areas = [0.0]
            bbox_width = bbox_height = orientation_deg = 0.0
            vertex_x_min = vertex_x_max = vertex_y_min = vertex_y_max = 0.0
        signal = signal_by_sample[sample_index]
        out.append(
            {
                "split": split,
                "sample_index": sample_index,
                "hard_case_type": row["hard_case_type"],
                "defect_type": row.get("defect_type", ""),
                "component_type_combination": row.get("component_type_combination", ""),
                "true_rotated": _as_bool(row.get("true_rotated_geometry", "false")),
                "true_multi_component": _as_bool(row.get("true_multi_component_geometry", "false")),
                "component_count": len(polys),
                "center_x": _as_float(row.get("defect_center_x", 0.0)),
                "center_y": _as_float(row.get("defect_center_y", 0.0)),
                "axis_x": _as_float(row.get("defect_axis_x", 0.0)),
                "axis_y": _as_float(row.get("defect_axis_y", 0.0)),
                "rotation_angle": _as_float(row.get("rotation_angle", 0.0)),
                "polygon_area_sum": float(sum(areas)),
                "polygon_area_max": float(max(areas)),
                "polygon_signed_area_flip_count": sum(1 for value in signed_areas if value > 0.0),
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "orientation_deg": orientation_deg,
                "vertex_x_min": vertex_x_min,
                "vertex_x_max": vertex_x_max,
                "vertex_y_min": vertex_y_min,
                "vertex_y_max": vertex_y_max,
                **signal,
            }
        )
    return out


def _sample_prediction_rows(split: str, prediction_dir: Path, geometry_rows: dict[tuple[str, int], dict]) -> list[dict]:
    metrics = _read_csv(prediction_dir / f"{split}_polygon_mask_metrics.csv")
    pred_rows = _read_csv(prediction_dir / f"{split}_polygon_predictions.csv")
    comp_rows = _component_prediction_rows(split, pred_rows)
    comp_by_sample: dict[int, list[dict]] = defaultdict(list)
    for row in comp_rows:
        comp_by_sample[int(row["sample_index"])].append(row)
    out = []
    for row in metrics:
        sample_index = int(row["sample_index"])
        comps = comp_by_sample.get(sample_index, [])
        present_comps = [item for item in comps if item["presence_true"]]
        geom = geometry_rows[(split, sample_index)]
        presence_correct = all(item["presence_correct"] for item in comps) if comps else True
        present_type_correct = all(item["type_correct"] for item in present_comps) if present_comps else True
        out.append(
            {
                "split": split,
                "sample_index": sample_index,
                "hard_case_type": geom["hard_case_type"],
                "true_rotated": geom["true_rotated"],
                "true_multi_component": geom["true_multi_component"],
                "component_count": geom["component_count"],
                "polygon_mask_iou": _as_float(row["polygon_mask_iou"]),
                "polygon_dice": _as_float(row["polygon_dice"]),
                "target_area": int(_as_float(row["target_area"])),
                "pred_area": int(_as_float(row["pred_area"])),
                "area_diff": int(_as_float(row["pred_area"]) - _as_float(row["target_area"])),
                "true_component_count": int(_as_float(row["true_component_count"])),
                "pred_component_count": int(_as_float(row["pred_component_count"])),
                "presence_correct": presence_correct,
                "present_type_correct": present_type_correct,
                "present_vertex_mae_mean": _mean(item["vertex_mae"] for item in present_comps),
                "present_vertex_mae_max": _max(item["vertex_mae"] for item in present_comps),
                "out_of_grid_vertex_count": sum(item["out_of_grid_vertex_count"] for item in present_comps),
                "signed_area_flip_count": sum(1 for item in present_comps if item["signed_area_flip"]),
                "degenerate_pred_component_count": sum(1 for item in present_comps if item["degenerate_pred"]),
                "pred_component_out_of_grid_count": sum(1 for item in present_comps if item["out_of_grid_vertex_count"] > 0),
            }
        )
    return out


def _component_prediction_rows(split: str, pred_rows: list[dict[str, str]]) -> list[dict]:
    out = []
    x_min, x_max = -0.04, 0.04
    y_min, y_max = -0.01, 0.01
    for row in pred_rows:
        pred_vertices = _component_vertices(row, "pred")
        true_vertices = _component_vertices(row, "true")
        pred_signed = _shoelace(pred_vertices)
        true_signed = _shoelace(true_vertices)
        out_of_grid = int(
            np.sum(
                (pred_vertices[:, 0] < x_min)
                | (pred_vertices[:, 0] > x_max)
                | (pred_vertices[:, 1] < y_min)
                | (pred_vertices[:, 1] > y_max)
            )
        )
        presence_true = _as_bool(row["presence_true"])
        presence_pred = _as_bool(row["presence_pred"])
        type_true = int(_as_float(row["type_true"], -1))
        type_pred = int(_as_float(row["type_pred"], -1))
        sign_flip = bool(presence_true and abs(true_signed) > 1e-12 and (pred_signed * true_signed < 0.0))
        item = {
                "split": split,
                "sample_index": int(row["sample_index"]),
                "component_slot": int(row["component_slot"]),
                "presence_true": presence_true,
                "presence_pred": presence_pred,
                "presence_correct": presence_true == presence_pred,
                "type_true": type_true,
                "type_pred": type_pred,
                "type_correct": (not presence_true) or (type_true == type_pred),
                "vertex_mae": _as_float(row["vertex_mae"]),
                "out_of_grid_vertex_count": out_of_grid,
                "pred_signed_area": pred_signed,
                "true_signed_area": true_signed,
                "signed_area_flip": sign_flip,
                "degenerate_pred": bool(presence_true and abs(pred_signed) < 1e-12),
                "pred_area_norm_abs": abs(pred_signed),
                "true_area_norm_abs": abs(true_signed),
                "pred_bbox_width": float(pred_vertices[:, 0].max() - pred_vertices[:, 0].min()),
                "pred_bbox_height": float(pred_vertices[:, 1].max() - pred_vertices[:, 1].min()),
                "true_bbox_width": float(true_vertices[:, 0].max() - true_vertices[:, 0].min()),
                "true_bbox_height": float(true_vertices[:, 1].max() - true_vertices[:, 1].min()),
        }
        for idx in range(4):
            item[f"pred_x{idx}"] = float(pred_vertices[idx, 0])
            item[f"pred_y{idx}"] = float(pred_vertices[idx, 1])
            item[f"true_x{idx}"] = float(true_vertices[idx, 0])
            item[f"true_y{idx}"] = float(true_vertices[idx, 1])
        out.append(item)
    return out


def _aggregate(rows: list[dict], keys: list[str], numeric_fields: list[str], bool_fields: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key, items in sorted(groups.items()):
        result = {name: value for name, value in zip(keys, key)}
        result["sample_count"] = len(items)
        for field in numeric_fields:
            vals = [_as_float(item[field]) for item in items]
            result[f"{field}_mean"] = _mean(vals)
            result[f"{field}_min"] = _min(vals)
            result[f"{field}_max"] = _max(vals)
        for field in bool_fields:
            vals = [bool(item[field]) for item in items]
            result[f"{field}_rate"] = _mean(1.0 if value else 0.0 for value in vals)
        out.append(result)
    return out


def run(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    converted_dir = Path(args.converted_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry_rows = []
    for split in SPLITS:
        geometry_rows.extend(_sample_geometry(split, raw_root, converted_dir))
    geometry_by_sample = {(row["split"], int(row["sample_index"])): row for row in geometry_rows}

    prediction_rows = []
    component_rows = []
    for split in SPLITS:
        pred_csv = _read_csv(prediction_dir / f"{split}_polygon_predictions.csv")
        for row in _component_prediction_rows(split, pred_csv):
            geom = geometry_by_sample[(split, int(row["sample_index"]))]
            row["hard_case_type"] = geom["hard_case_type"]
            row["true_rotated"] = geom["true_rotated"]
            row["true_multi_component"] = geom["true_multi_component"]
            row["component_count"] = geom["component_count"]
            component_rows.append(row)
        prediction_rows.extend(_sample_prediction_rows(split, prediction_dir, geometry_by_sample))

    _write_csv(output_dir / "geometry_signal_per_sample.csv", geometry_rows)
    _write_csv(
        output_dir / "split_geometry_signal_distribution.csv",
        _aggregate(
            geometry_rows,
            ["split"],
            [
                "center_x",
                "center_y",
                "axis_x",
                "axis_y",
                "rotation_angle",
                "polygon_area_sum",
                "bbox_width",
                "bbox_height",
                "vertex_x_min",
                "vertex_x_max",
                "signal_std_mean",
                "signal_peak_to_peak_mean",
                "lift_off_std_ratio_last_first",
            ],
            ["true_rotated", "true_multi_component"],
        ),
    )
    _write_csv(
        output_dir / "grouped_geometry_signal_distribution.csv",
        _aggregate(
            geometry_rows,
            ["split", "hard_case_type"],
            [
                "center_x",
                "center_y",
                "axis_x",
                "axis_y",
                "polygon_area_sum",
                "bbox_width",
                "bbox_height",
                "signal_std_mean",
                "signal_peak_to_peak_mean",
            ],
            ["true_rotated", "true_multi_component"],
        ),
    )
    _write_csv(output_dir / "prediction_failure_per_sample.csv", prediction_rows)
    _write_csv(output_dir / "prediction_failure_per_component.csv", component_rows)
    _write_csv(
        output_dir / "grouped_prediction_failures.csv",
        _aggregate(
            prediction_rows,
            ["split", "hard_case_type"],
            [
                "polygon_mask_iou",
                "present_vertex_mae_mean",
                "area_diff",
                "out_of_grid_vertex_count",
                "signed_area_flip_count",
                "degenerate_pred_component_count",
            ],
            ["presence_correct", "present_type_correct", "true_rotated", "true_multi_component"],
        ),
    )
    worst = sorted(
        [row for row in prediction_rows if row["split"] in {"val", "test"}],
        key=lambda item: _as_float(item["polygon_mask_iou"]),
    )[:12]
    _write_csv(output_dir / "worst_val_test_polygon_samples.csv", worst)
    _write_summary(output_dir, geometry_rows, prediction_rows)


def _write_summary(output_dir: Path, geometry_rows: list[dict], prediction_rows: list[dict]) -> None:
    split_rows = _aggregate(
        prediction_rows,
        ["split"],
        ["polygon_mask_iou", "present_vertex_mae_mean", "area_diff", "out_of_grid_vertex_count", "signed_area_flip_count"],
        ["presence_correct", "present_type_correct"],
    )
    geometry_split = _aggregate(
        geometry_rows,
        ["split"],
        ["center_x", "polygon_area_sum", "signal_std_mean", "signal_peak_to_peak_mean"],
        ["true_rotated", "true_multi_component"],
    )
    lines = ["# COMSOL polygon generalization diagnostics", "", "## Prediction Summary", ""]
    for row in split_rows:
        lines.append(
            f"- {row['split']}: IoU mean/min/max "
            f"`{row['polygon_mask_iou_mean']:.6f}` / `{row['polygon_mask_iou_min']:.6f}` / `{row['polygon_mask_iou_max']:.6f}`, "
            f"vertex MAE mean `{row['present_vertex_mae_mean_mean']:.6e}`, "
            f"presence/type sample correctness `{row['presence_correct_rate']:.6f}` / `{row['present_type_correct_rate']:.6f}`."
        )
    lines.extend(["", "## Geometry And Signal Summary", ""])
    for row in geometry_split:
        lines.append(
            f"- {row['split']}: center_x mean/min/max "
            f"`{row['center_x_mean']:.6f}` / `{row['center_x_min']:.6f}` / `{row['center_x_max']:.6f}`, "
            f"signal std mean `{row['signal_std_mean_mean']:.6e}`, "
            f"rotated/multi rates `{row['true_rotated_rate']:.3f}` / `{row['true_multi_component_rate']:.3f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This diagnostic is read-only. It is intended to distinguish split geometry coverage, prediction shape pathology, and small-N memorization before any new training experiment.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/raw")
    parser.add_argument("--converted-dir", default="experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/converted")
    parser.add_argument("--prediction-dir", default="experiments/dual_network/S282_comsol_v3_polygon_train30_repair_quick_gate/longer_train30")
    parser.add_argument("--output-dir", default="experiments/dual_network/S285_comsol_v3_polygon_generalization_distribution_diagnostics")
    args = parser.parse_args(argv)
    run(args)
    print(f"Saved COMSOL polygon generalization diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
