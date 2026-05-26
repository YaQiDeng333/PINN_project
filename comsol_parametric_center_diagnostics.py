"""Diagnose COMSOL parametric center localization errors."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from comsol_parametric_oracle_ablation import (
    RAW_SCHEMA,
    load_prediction_data,
    load_target_data,
    write_csv,
)


PRESENT_EPS = 1e-12


def _load_grid(npz_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(npz_path, allow_pickle=True) as data:
        for key in ["x", "y", "masks"]:
            if key not in data:
                raise ValueError(f"{npz_path} missing required field: {key}")
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        masks = data["masks"]
        units = str(data["geometry_units"]) if "geometry_units" in data else ""
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D coordinate grids.")
    if masks.ndim != 3 or masks.shape[1] != y.shape[0] or masks.shape[2] != x.shape[0]:
        raise ValueError("masks shape must align with y/x grid dimensions.")
    if not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0):
        raise ValueError("x and y grids must be strictly increasing.")
    return x, y, units


def _grid_spacing(values: np.ndarray, name: str) -> float:
    diffs = np.diff(values.astype(np.float64))
    mean = float(np.mean(diffs))
    if mean <= 0 or not np.isfinite(mean):
        raise ValueError(f"{name} grid spacing is invalid: {mean}")
    if float(np.max(np.abs(diffs - mean))) > max(abs(mean) * 1e-3, 1e-9):
        raise ValueError(f"{name} grid spacing is not approximately uniform.")
    return mean


def _load_mask_metrics(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            rows[int(float(row["sample_index"]))] = row
    return rows


def _float_or_nan(row: dict, key: str) -> float:
    if key not in row or row[key] == "":
        return float("nan")
    try:
        return float(row[key])
    except ValueError:
        return float("nan")


def _rotation_bin(angle_deg: float) -> str:
    value = abs(float(angle_deg))
    if value <= 5:
        return "0-5"
    if value <= 10:
        return "5-10"
    if value <= 20:
        return "10-20"
    if value <= 30:
        return "20-30"
    return ">30"


def _area_bins(mask_metrics: dict[int, dict]) -> dict[int, str]:
    areas = []
    for sample_index, row in mask_metrics.items():
        area = _float_or_nan(row, "target_area")
        if np.isfinite(area):
            areas.append((sample_index, area))
    if not areas:
        return {}
    values = np.asarray([area for _idx, area in areas], dtype=np.float64)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    out = {}
    for sample_index, area in areas:
        if area <= q1:
            out[sample_index] = "small"
        elif area <= q2:
            out[sample_index] = "medium"
        else:
            out[sample_index] = "large"
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _correlation(x_values: np.ndarray, y_values: np.ndarray, *, method: str) -> float:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x_values[mask].astype(np.float64)
    y = y_values[mask].astype(np.float64)
    if method == "spearman":
        x = _rankdata(x)
        y = _rankdata(y)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _summarize_group(rows: list[dict], group_by: str, group_value: str) -> dict:
    return {
        "group_by": group_by,
        "group_value": group_value,
        "count": len(rows),
        "center_x_grid_mae": _mean([float(row["center_x_error_grid"]) for row in rows]),
        "center_y_grid_mae": _mean([float(row["center_y_error_grid"]) for row in rows]),
        "center_l2_grid_mae": _mean([float(row["center_error_grid_l2"]) for row in rows]),
        "center_x_axis_relative_mae": _mean([float(row["center_x_error_axis_relative"]) for row in rows]),
        "center_y_axis_relative_mae": _mean([float(row["center_y_error_axis_relative"]) for row in rows]),
        "center_axis_relative_l2_mae": _mean([float(row["center_error_axis_relative_l2"]) for row in rows]),
        "mask_iou_mean": _mean([float(row["pred_mask_iou"]) for row in rows]),
    }


def build_center_rows(args) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    npz_path = Path(args.npz_path)
    x, y, geometry_units = _load_grid(npz_path)
    dx_grid = _grid_spacing(x, "x")
    dy_grid = _grid_spacing(y, "y")
    target = load_target_data(npz_path, Path(args.targets_path), max_components=3)
    pred = load_prediction_data(Path(args.predictions_csv), target, args.split)
    mask_metrics = _load_mask_metrics(Path(args.mask_metrics_csv) if args.mask_metrics_csv else None)
    area_bin_by_sample = _area_bins(mask_metrics)
    type_vocab = target.type_vocab

    center_x_idx = RAW_SCHEMA.index("center_x")
    center_y_idx = RAW_SCHEMA.index("center_y")
    axis_x_idx = RAW_SCHEMA.index("axis_x")
    axis_y_idx = RAW_SCHEMA.index("axis_y")
    rotation_idx = RAW_SCHEMA.index("rotation_angle")

    component_rows = []
    sample_groups: dict[int, list[dict]] = defaultdict(list)
    for sample_pos, sample_index_raw in enumerate(target.sample_indices):
        sample_index = int(sample_index_raw)
        metric = mask_metrics.get(sample_index, {})
        area_bin = area_bin_by_sample.get(sample_index, "unknown")
        pred_mask_iou = _float_or_nan(metric, "pred_mask_iou")
        oracle_mask_iou = _float_or_nan(metric, "oracle_mask_iou")
        oracle_gap = _float_or_nan(metric, "oracle_gap")
        target_area = _float_or_nan(metric, "target_area")
        for slot in range(target.presence.shape[1]):
            if target.presence[sample_pos, slot] <= 0.5:
                continue
            true_raw = target.continuous_raw[sample_pos, slot]
            pred_raw = pred.continuous_raw[sample_pos, slot]
            delta_x = float(pred_raw[center_x_idx] - true_raw[center_x_idx])
            delta_y = float(pred_raw[center_y_idx] - true_raw[center_y_idx])
            abs_x = abs(delta_x)
            abs_y = abs(delta_y)
            x_grid = abs_x / dx_grid
            y_grid = abs_y / dy_grid
            l2_grid = float(np.sqrt((delta_x / dx_grid) ** 2 + (delta_y / dy_grid) ** 2))
            axis_x = max(abs(float(true_raw[axis_x_idx])), PRESENT_EPS)
            axis_y = max(abs(float(true_raw[axis_y_idx])), PRESENT_EPS)
            x_rel = abs_x / axis_x
            y_rel = abs_y / axis_y
            rel_l2 = float(np.sqrt((delta_x / axis_x) ** 2 + (delta_y / axis_y) ** 2))
            type_id = int(target.type_targets[sample_pos, slot])
            row = {
                "split": args.split,
                "sample_index": sample_index,
                "component_slot": slot,
                "type_true": type_vocab[type_id] if type_id >= 0 else "",
                "rotation_true": float(true_raw[rotation_idx]),
                "rotation_bin": _rotation_bin(float(true_raw[rotation_idx])),
                "target_area_bin": area_bin,
                "center_x_true": float(true_raw[center_x_idx]),
                "center_x_pred": float(pred_raw[center_x_idx]),
                "center_y_true": float(true_raw[center_y_idx]),
                "center_y_pred": float(pred_raw[center_y_idx]),
                "center_x_error_abs": abs_x,
                "center_y_error_abs": abs_y,
                "center_error_l2": float(np.sqrt(delta_x * delta_x + delta_y * delta_y)),
                "center_x_error_grid": x_grid,
                "center_y_error_grid": y_grid,
                "center_error_grid_l2": l2_grid,
                "center_x_error_axis_relative": x_rel,
                "center_y_error_axis_relative": y_rel,
                "center_error_axis_relative_l2": rel_l2,
                "axis_x_true": float(true_raw[axis_x_idx]),
                "axis_y_true": float(true_raw[axis_y_idx]),
                "pred_mask_iou": pred_mask_iou,
                "oracle_mask_iou": oracle_mask_iou,
                "oracle_gap": oracle_gap,
                "target_area": target_area,
            }
            component_rows.append(row)
            sample_groups[sample_index].append(row)

    sample_rows = []
    for sample_index in sorted(sample_groups):
        rows = sample_groups[sample_index]
        first = rows[0]
        sample_rows.append(
            {
                "split": args.split,
                "sample_index": sample_index,
                "component_count": len(rows),
                "center_x_grid_mae": _mean([float(row["center_x_error_grid"]) for row in rows]),
                "center_y_grid_mae": _mean([float(row["center_y_error_grid"]) for row in rows]),
                "center_l2_grid_mae": _mean([float(row["center_error_grid_l2"]) for row in rows]),
                "center_x_axis_relative_mae": _mean([float(row["center_x_error_axis_relative"]) for row in rows]),
                "center_y_axis_relative_mae": _mean([float(row["center_y_error_axis_relative"]) for row in rows]),
                "center_axis_relative_l2_mae": _mean([float(row["center_error_axis_relative_l2"]) for row in rows]),
                "pred_mask_iou": float(first["pred_mask_iou"]),
                "oracle_mask_iou": float(first["oracle_mask_iou"]),
                "oracle_gap": float(first["oracle_gap"]),
                "target_area": float(first["target_area"]),
                "target_area_bin": str(first["target_area_bin"]),
            }
        )

    grouped_rows = []
    for group_by in ["component_slot", "type_true", "rotation_bin", "target_area_bin"]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in component_rows:
            grouped[str(row[group_by])].append(row)
        for group_value, rows in sorted(grouped.items()):
            grouped_rows.append(_summarize_group(rows, group_by, group_value))

    x_grid = np.asarray([row["center_l2_grid_mae"] for row in sample_rows], dtype=np.float64)
    rel_grid = np.asarray([row["center_axis_relative_l2_mae"] for row in sample_rows], dtype=np.float64)
    iou = np.asarray([row["pred_mask_iou"] for row in sample_rows], dtype=np.float64)
    correlation_rows = []
    for metric_name, values in [
        ("center_l2_grid_mae", x_grid),
        ("center_axis_relative_l2_mae", rel_grid),
    ]:
        correlation_rows.append(
            {
                "split": args.split,
                "center_error_metric": metric_name,
                "mask_metric": "pred_mask_iou",
                "pearson": _correlation(values, iou, method="pearson"),
                "spearman": _correlation(values, iou, method="spearman"),
                "samples": int(np.isfinite(values).sum()),
            }
        )

    meta = {
        "dx_grid": dx_grid,
        "dy_grid": dy_grid,
        "geometry_units": geometry_units,
        "component_count": len(component_rows),
        "sample_count": len(sample_rows),
    }
    return component_rows, sample_rows, grouped_rows, correlation_rows, meta


def write_summary(path: Path, split: str, sample_rows: list[dict], grouped_rows: list[dict], correlation_rows: list[dict], meta: dict) -> None:
    all_group = _summarize_group(
        [
            {
                "center_x_error_grid": row["center_x_grid_mae"],
                "center_y_error_grid": row["center_y_grid_mae"],
                "center_error_grid_l2": row["center_l2_grid_mae"],
                "center_x_error_axis_relative": row["center_x_axis_relative_mae"],
                "center_y_error_axis_relative": row["center_y_axis_relative_mae"],
                "center_error_axis_relative_l2": row["center_axis_relative_l2_mae"],
                "pred_mask_iou": row["pred_mask_iou"],
            }
            for row in sample_rows
        ],
        "all_samples",
        split,
    )
    worst_slot = max(
        (row for row in grouped_rows if row["group_by"] == "component_slot"),
        key=lambda row: row["center_l2_grid_mae"],
    )
    corr_text = "; ".join(
        f"{row['center_error_metric']}: Pearson={row['pearson']:.6e}, Spearman={row['spearman']:.6e}"
        for row in correlation_rows
    )
    recommendation = (
        "axis_relative_smoothl1"
        if all_group["center_x_axis_relative_mae"] > 0.2 or all_group["center_y_axis_relative_mae"] > 0.2
        else "grid_mse"
    )
    lines = [
        f"# S161 center diagnostics split summary: {split}",
        "",
        "## Grid and units",
        "",
        f"- geometry_units: `{meta['geometry_units']}`",
        f"- dx_grid: `{meta['dx_grid']:.6e}`",
        f"- dy_grid: `{meta['dy_grid']:.6e}`",
        f"- samples: `{meta['sample_count']}`",
        f"- present_components: `{meta['component_count']}`",
        "",
        "## Aggregate center errors",
        "",
        f"- center_x_grid_mae: `{all_group['center_x_grid_mae']:.6e}`",
        f"- center_y_grid_mae: `{all_group['center_y_grid_mae']:.6e}`",
        f"- center_l2_grid_mae: `{all_group['center_l2_grid_mae']:.6e}`",
        f"- center_x_axis_relative_mae: `{all_group['center_x_axis_relative_mae']:.6e}`",
        f"- center_y_axis_relative_mae: `{all_group['center_y_axis_relative_mae']:.6e}`",
        f"- center_axis_relative_l2_mae: `{all_group['center_axis_relative_l2_mae']:.6e}`",
        "",
        "## Correlation with mask IoU",
        "",
        f"- {corr_text}",
        "",
        "## Interpretation",
        "",
        f"- worst component_slot by center_l2_grid_mae: `{worst_slot['group_value']}` = `{worst_slot['center_l2_grid_mae']:.6e}`.",
        f"- recommended center loss mode: `{recommendation}`.",
        "- Negative correlation means larger center error tends to lower mask IoU.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    component_rows, sample_rows, grouped_rows, correlation_rows, meta = build_center_rows(args)
    write_csv(output_dir / "per_component_center_errors.csv", component_rows)
    write_csv(output_dir / "per_sample_center_error_summary.csv", sample_rows)
    write_csv(output_dir / "grouped_center_errors.csv", grouped_rows)
    write_csv(output_dir / "center_error_correlation.csv", correlation_rows)
    write_summary(output_dir / "summary.md", args.split, sample_rows, grouped_rows, correlation_rows, meta)
    print(f"Saved COMSOL parametric center diagnostics to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", default="")
    parser.add_argument("--targets-path", default="")
    parser.add_argument("--predictions-csv", default="")
    parser.add_argument("--mask-metrics-csv", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--split", default="")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.npz_path or not args.targets_path or not args.predictions_csv or not args.output_dir or not args.split:
        parser.print_help()
        print(
            "\nExample: python comsol_parametric_center_diagnostics.py "
            "--npz-path val.npz --targets-path parametric_targets.npz "
            "--predictions-csv val_predictions.csv --mask-metrics-csv val_prediction_mask_metrics.csv "
            "--output-dir out --split val"
        )
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
