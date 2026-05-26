"""Diagnose sample-level center-bin failure patterns from prediction exports."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "val", "test")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _parse_summary_values(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("- `") or "`:" not in text:
            continue
        key = text.split("`", 2)[1]
        value_text = text.split(":", 1)[1].strip().strip("`")
        try:
            values[key] = float(value_text)
        except ValueError:
            continue
    return values


def _bin_index(value: float, min_value: float, bin_width: float, bins: int) -> int | None:
    if not math.isfinite(value) or bins <= 0 or bin_width <= 0:
        return None
    idx = int(math.floor((value - min_value) / bin_width))
    return max(0, min(bins - 1, idx))


def _bin_center(idx: int | None, min_value: float, bin_width: float) -> float:
    if idx is None:
        return float("nan")
    return min_value + (idx + 0.5) * bin_width


def _safe_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _mask_iou_bin(value: float) -> str:
    if not math.isfinite(value):
        return "unknown"
    if value < 0.40:
        return "<0.40"
    if value < 0.50:
        return "0.40-0.50"
    if value < 0.60:
        return "0.50-0.60"
    return ">=0.60"


def _area_bin(value: float) -> str:
    if not math.isfinite(value):
        return "unknown"
    if value < 500:
        return "<500"
    if value < 1000:
        return "500-1000"
    if value < 1500:
        return "1000-1500"
    return ">=1500"


def _rotation_bin(value: float) -> str:
    if not math.isfinite(value):
        return "unknown"
    abs_value = abs(value)
    if abs_value < 15:
        return "<15deg"
    if abs_value < 45:
        return "15-45deg"
    if abs_value < 75:
        return "45-75deg"
    return ">=75deg"


def _mean(values: Iterable[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _rate(values: Iterable[bool]) -> float:
    items = list(values)
    return float(sum(1 for item in items if item) / len(items)) if items else float("nan")


def _aggregate_group(rows: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, ""))].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        out.append(
            {
                "group_by": group_key,
                "group_value": key,
                "count": len(items),
                "mean_mask_iou": _mean(float(item["mask_iou"]) for item in items),
                "mean_center_grid_error": _mean(float(item["center_grid_error"]) for item in items),
                "x_bin_wrong_rate": 1.0 - _rate(item["x_bin_correct"] == "true" for item in items),
                "y_bin_wrong_rate": 1.0 - _rate(item["y_bin_correct"] == "true" for item in items),
                "both_bins_correct_rate": _rate(item["both_bins_correct"] == "true" for item in items),
                "mean_abs_offset_x_error": _mean(abs(float(item["center_offset_x_error"])) for item in items),
                "mean_abs_offset_y_error": _mean(abs(float(item["center_offset_y_error"])) for item in items),
            }
        )
    return out


def diagnose(prediction_dir: Path, output_dir: Path, label: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_values = _parse_summary_values(prediction_dir / "run_summary.md")
    x_min = summary_values.get("x_min", float("nan"))
    y_min = summary_values.get("y_min", float("nan"))
    dx = summary_values.get("dx", float("nan"))
    dy = summary_values.get("dy", float("nan"))
    bin_width_x = summary_values.get("bin_width_x", float("nan"))
    bin_width_y = summary_values.get("bin_width_y", float("nan"))
    center_x_bins = int(summary_values.get("center_x_bins", 0))
    center_y_bins = int(summary_values.get("center_y_bins", 0))

    missing = []
    per_component: list[dict] = []
    per_sample: list[dict] = []
    diagnosed_splits: list[str] = []

    for split in SPLITS:
        pred_path = prediction_dir / f"{split}_predictions.csv"
        metric_path = prediction_dir / f"{split}_prediction_mask_metrics.csv"
        if not pred_path.exists() or not metric_path.exists():
            missing.append(split)
            continue
        pred_rows = _read_csv(pred_path)
        metric_rows = _read_csv(metric_path)
        metric_by_sample = {str(_int(row, "sample_index")): row for row in metric_rows}
        if not pred_rows or not metric_rows:
            missing.append(split)
            continue
        diagnosed_splits.append(split)
        sample_items: dict[str, list[dict]] = defaultdict(list)
        for row in pred_rows:
            sample_index = str(_int(row, "sample_index"))
            metric = metric_by_sample.get(sample_index, {})
            cx_true = _float(row, "center_x_true")
            cx_pred = _float(row, "center_x_pred")
            cy_true = _float(row, "center_y_true")
            cy_pred = _float(row, "center_y_pred")
            x_bin_true = _bin_index(cx_true, x_min, bin_width_x, center_x_bins)
            x_bin_pred = _bin_index(cx_pred, x_min, bin_width_x, center_x_bins)
            y_bin_true = _bin_index(cy_true, y_min, bin_width_y, center_y_bins)
            y_bin_pred = _bin_index(cy_pred, y_min, bin_width_y, center_y_bins)
            x_bin_correct = None if x_bin_true is None or x_bin_pred is None else x_bin_true == x_bin_pred
            y_bin_correct = None if y_bin_true is None or y_bin_pred is None else y_bin_true == y_bin_pred
            both_bins_correct = None if x_bin_correct is None or y_bin_correct is None else x_bin_correct and y_bin_correct
            x_offset_true = (cx_true - _bin_center(x_bin_true, x_min, bin_width_x)) / bin_width_x if math.isfinite(bin_width_x) and bin_width_x > 0 else float("nan")
            x_offset_pred = (cx_pred - _bin_center(x_bin_pred, x_min, bin_width_x)) / bin_width_x if math.isfinite(bin_width_x) and bin_width_x > 0 else float("nan")
            y_offset_true = (cy_true - _bin_center(y_bin_true, y_min, bin_width_y)) / bin_width_y if math.isfinite(bin_width_y) and bin_width_y > 0 else float("nan")
            y_offset_pred = (cy_pred - _bin_center(y_bin_pred, y_min, bin_width_y)) / bin_width_y if math.isfinite(bin_width_y) and bin_width_y > 0 else float("nan")
            if math.isfinite(dx) and math.isfinite(dy) and dx > 0 and dy > 0:
                center_grid_error = math.sqrt(((cx_pred - cx_true) / dx) ** 2 + ((cy_pred - cy_true) / dy) ** 2)
            else:
                center_grid_error = _float(row, "center_error")
            axis_error = _float(row, "axis_error")
            rotation_true = _float(row, "rotation_true")
            out_row = {
                "label": label,
                "split": split,
                "sample_index": sample_index,
                "component_slot": _int(row, "component_slot"),
                "type_true": row.get("type_true", ""),
                "type_pred": row.get("type_pred", ""),
                "center_x_true": cx_true,
                "center_x_pred": cx_pred,
                "center_y_true": cy_true,
                "center_y_pred": cy_pred,
                "center_x_bin_true": "" if x_bin_true is None else x_bin_true,
                "center_x_bin_pred": "" if x_bin_pred is None else x_bin_pred,
                "center_y_bin_true": "" if y_bin_true is None else y_bin_true,
                "center_y_bin_pred": "" if y_bin_pred is None else y_bin_pred,
                "x_bin_correct": _safe_bool(x_bin_correct),
                "y_bin_correct": _safe_bool(y_bin_correct),
                "both_bins_correct": _safe_bool(both_bins_correct),
                "center_offset_x_error": x_offset_pred - x_offset_true,
                "center_offset_y_error": y_offset_pred - y_offset_true,
                "center_grid_error": center_grid_error,
                "rotation_true": rotation_true,
                "rotation_pred": _float(row, "rotation_pred"),
                "rotation_error": _float(row, "rotation_error"),
                "axis_x_true": _float(row, "axis_x_true"),
                "axis_x_pred": _float(row, "axis_x_pred"),
                "axis_error": axis_error,
                "mask_iou": _float(metric, "pred_mask_iou"),
                "pred_area": _float(metric, "pred_area"),
                "target_area": _float(metric, "target_area"),
                "mask_iou_bin": _mask_iou_bin(_float(metric, "pred_mask_iou")),
                "target_area_bin": _area_bin(_float(metric, "target_area")),
                "rotation_bin": _rotation_bin(rotation_true),
            }
            per_component.append(out_row)
            sample_items[sample_index].append(out_row)

        for sample_index, items in sorted(sample_items.items(), key=lambda kv: int(kv[0])):
            metric = metric_by_sample.get(sample_index, {})
            x_correct_values = [item["x_bin_correct"] == "true" for item in items if item["x_bin_correct"]]
            y_correct_values = [item["y_bin_correct"] == "true" for item in items if item["y_bin_correct"]]
            center_errors = [float(item["center_grid_error"]) for item in items]
            per_sample.append(
                {
                    "label": label,
                    "split": split,
                    "sample_index": sample_index,
                    "mask_iou": _float(metric, "pred_mask_iou"),
                    "pred_area": _float(metric, "pred_area"),
                    "target_area": _float(metric, "target_area"),
                    "all_x_bins_correct": _safe_bool(all(x_correct_values) if x_correct_values else None),
                    "all_y_bins_correct": _safe_bool(all(y_correct_values) if y_correct_values else None),
                    "any_x_bin_wrong": _safe_bool(any(not value for value in x_correct_values) if x_correct_values else None),
                    "any_y_bin_wrong": _safe_bool(any(not value for value in y_correct_values) if y_correct_values else None),
                    "mean_center_grid_error": _mean(center_errors),
                    "max_center_grid_error": max([v for v in center_errors if math.isfinite(v)], default=float("nan")),
                    "type_sequence_true": metric.get("type_sequence_true", ""),
                    "type_sequence_pred": metric.get("type_sequence_pred", ""),
                }
            )

    if not any(row["split"] in {"val", "test"} for row in per_component):
        raise ValueError(f"{prediction_dir} does not contain diagnosable val/test prediction exports.")

    grouped = []
    for key in [
        "split",
        "component_slot",
        "type_true",
        "x_bin_correct",
        "y_bin_correct",
        "both_bins_correct",
        "mask_iou_bin",
        "target_area_bin",
        "rotation_bin",
    ]:
        grouped.extend(_aggregate_group(per_component, key))

    worst_samples = sorted(per_sample, key=lambda row: (row["split"] != "val", float(row["mask_iou"])))[:20]

    _write_csv(output_dir / "per_component_center_bin_errors.csv", per_component)
    _write_csv(output_dir / "per_sample_center_bin_errors.csv", per_sample)
    _write_csv(output_dir / "grouped_center_bin_errors.csv", grouped)
    _write_csv(output_dir / "worst_samples.csv", worst_samples)

    val_rows = [row for row in per_component if row["split"] == "val"]
    test_rows = [row for row in per_component if row["split"] == "test"]
    sample_val = [row for row in per_sample if row["split"] == "val"]
    sample_test = [row for row in per_sample if row["split"] == "test"]

    def wrong_rate(rows: list[dict], key: str) -> float:
        values = [row[key] == "true" for row in rows if row[key]]
        return 1.0 - _rate(values)

    summary = {
        "label": label,
        "diagnosed_splits": diagnosed_splits,
        "missing_splits": missing,
        "val_x_wrong_rate": wrong_rate(val_rows, "x_bin_correct"),
        "val_y_wrong_rate": wrong_rate(val_rows, "y_bin_correct"),
        "test_x_wrong_rate": wrong_rate(test_rows, "x_bin_correct"),
        "test_y_wrong_rate": wrong_rate(test_rows, "y_bin_correct"),
        "val_mean_iou": _mean(float(row["mask_iou"]) for row in sample_val),
        "test_mean_iou": _mean(float(row["mask_iou"]) for row in sample_test),
        "val_mean_center_grid_error": _mean(float(row["mean_center_grid_error"]) for row in sample_val),
        "test_mean_center_grid_error": _mean(float(row["mean_center_grid_error"]) for row in sample_test),
        "worst_val_sample_count_below_050": sum(1 for row in sample_val if float(row["mask_iou"]) < 0.50),
    }

    driver = "x-bin" if summary["val_x_wrong_rate"] >= summary["val_y_wrong_rate"] else "y-bin"
    lines = [
        f"# Center-bin failure diagnostics: {label}",
        "",
        f"- prediction_dir: `{prediction_dir}`",
        f"- diagnosed_splits: `{', '.join(diagnosed_splits)}`",
        f"- missing_splits: `{', '.join(missing) if missing else 'none'}`",
        f"- val_mean_iou: `{summary['val_mean_iou']:.6f}`",
        f"- test_mean_iou: `{summary['test_mean_iou']:.6f}`",
        f"- val_x_wrong_rate: `{summary['val_x_wrong_rate']:.6f}`",
        f"- val_y_wrong_rate: `{summary['val_y_wrong_rate']:.6f}`",
        f"- test_x_wrong_rate: `{summary['test_x_wrong_rate']:.6f}`",
        f"- test_y_wrong_rate: `{summary['test_y_wrong_rate']:.6f}`",
        f"- val_mean_center_grid_error: `{summary['val_mean_center_grid_error']:.6f}`",
        f"- test_mean_center_grid_error: `{summary['test_mean_center_grid_error']:.6f}`",
        f"- worst_val_sample_count_below_050: `{summary['worst_val_sample_count_below_050']}`",
        "",
        "## Answers",
        "",
        f"1. x-bin vs y-bin: `{driver}` has the higher val wrong-rate in this run.",
        "2. Val fluctuation should be checked through `worst_samples.csv`; the count below IoU 0.50 is listed above.",
        "3. The most error-prone component slot can be read from `grouped_center_bin_errors.csv` where `group_by=component_slot`.",
        "4. Type / rotation / area bin sensitivity is reported in grouped rows for `type_true`, `rotation_bin`, and `target_area_bin`.",
        "5. Auxiliary-head usefulness is decided by comparing this summary across labels, not within a single run.",
        "6. Preferred next action depends on S196 aggregate comparison; do not infer a new model route from this single-run summary alone.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--label", default="")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.prediction_dir or not args.output_dir or not args.label:
        parser.print_help()
        print("\nExample: python comsol_center_bin_failure_diagnostics.py --prediction-dir run --output-dir out --label candidate")
        return 0
    diagnose(Path(args.prediction_dir), Path(args.output_dir), args.label)
    print(f"Saved center-bin diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
