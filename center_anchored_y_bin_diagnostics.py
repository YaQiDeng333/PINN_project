"""Diagnose held-out y-bin errors for center-anchored polygon inverse runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


DEFAULT_RESPLIT_ROOT = Path("experiments/dual_network/S299_comsol_polygon_matched_coverage_resplit")
DEFAULT_PREDICTION_DIR = Path("experiments/dual_network/S300_center_anchored_polygon_matched_coverage_probe/matched_coverage_train30")
DEFAULT_OUTPUT_DIR = Path("experiments/dual_network/S304_center_anchored_y_bin_diagnostics")
SPLITS = ("train", "val", "test")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: str | bool | int | float | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def _load_targets(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_split_metadata(root: Path, split: str) -> tuple[dict[int, dict[str, str]], dict[tuple[int, int], dict[str, str]]]:
    defect_rows = _read_csv(root / split / "defect_params.csv")
    polygon_rows = _read_csv(root / split / "polygon_params.csv")
    defects = {_int(row, "sample_index"): row for row in defect_rows}
    polygons = {(_int(row, "sample_index"), _int(row, "component_index")): row for row in polygon_rows}
    return defects, polygons


def _load_predictions(prediction_dir: Path, split: str) -> tuple[dict[tuple[int, int], dict[str, str]], dict[int, dict[str, str]]]:
    comp_path = prediction_dir / f"{split}_center_anchored_polygon_predictions.csv"
    mask_path = prediction_dir / f"{split}_center_anchored_polygon_mask_metrics.csv"
    component_rows = _read_csv(comp_path)
    mask_rows = _read_csv(mask_path)
    components = {(_int(row, "sample_index"), _int(row, "component_slot")): row for row in component_rows}
    masks = {_int(row, "sample_index"): row for row in mask_rows}
    return components, masks


def _coverage_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    return {(row["new_split"], _int(row, "new_sample_index")): row for row in rows}


def _target_bbox_height_grid(vertices: np.ndarray, vertex_mask: np.ndarray, grid_dy: float) -> float:
    valid = vertex_mask > 0.5
    if not valid.any():
        return 0.0
    ys = vertices[valid, 1]
    return float((ys.max() - ys.min()) / grid_dy)


def _summarize_group(rows: list[dict], group_key: str) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], str(row[group_key]))].append(row)
    out = []
    for (split, key), items in sorted(grouped.items()):
        out.append(
            {
                "split": split,
                group_key: key,
                "component_count": len(items),
                "x_bin_acc": _mean_bool(items, "x_bin_correct"),
                "y_bin_acc": _mean_bool(items, "y_bin_correct"),
                "both_bins_acc": _mean_bool(items, "both_bins_correct"),
                "mean_abs_y_bin_error": mean(float(item["abs_y_bin_error"]) for item in items),
                "mean_polygon_iou": mean(float(item["polygon_iou"]) for item in items),
                "zero_iou_rate": mean(float(item["zero_iou"]) for item in items),
                "mean_center_offset_y_abs": mean(float(item["center_offset_y_abs"]) for item in items),
                "mean_target_bbox_height_grid": mean(float(item["target_bbox_height_grid"]) for item in items),
                "mean_defect_axis_y": mean(float(item["defect_axis_y"]) for item in items),
                "mean_rotation_angle": mean(float(item["rotation_angle"]) for item in items),
                "true_rotated_rate": mean(float(item["is_true_rotated"]) for item in items),
                "true_multi_component_rate": mean(float(item["is_true_multi_component"]) for item in items),
            }
        )
    return out


def _mean_bool(rows: list[dict], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def build_diagnostics(resplit_root: Path, prediction_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = _coverage_rows(resplit_root / "coverage_report.csv")
    per_component = []
    per_sample = []
    for split in SPLITS:
        targets = _load_targets(resplit_root / split / "center_anchored_polygon_targets.npz")
        defects, polygons = _load_split_metadata(resplit_root, split)
        predictions, mask_metrics = _load_predictions(prediction_dir, split)
        sample_indices = targets["sample_indices"].astype(int)
        grid_dy = float(targets["grid_dy"])
        for row_idx, sample_index in enumerate(sample_indices):
            defect = defects.get(int(sample_index), {})
            mask_row = mask_metrics.get(int(sample_index))
            if mask_row is None:
                raise ValueError(f"Missing mask metrics for {split} sample {sample_index}.")
            sample_components = []
            for slot in range(targets["presence_targets"].shape[1]):
                if float(targets["presence_targets"][row_idx, slot]) <= 0.5:
                    continue
                pred_row = predictions.get((int(sample_index), slot))
                if pred_row is None:
                    raise ValueError(f"Missing prediction row for {split} sample {sample_index} slot {slot}.")
                polygon = polygons.get((int(sample_index), slot), {})
                true_x = _int(pred_row, "center_x_bin_true")
                pred_x = _int(pred_row, "center_x_bin_pred")
                true_y = _int(pred_row, "center_y_bin_true")
                pred_y = _int(pred_row, "center_y_bin_pred")
                x_error = pred_x - true_x
                y_error = pred_y - true_y
                target_offset_y = float(targets["center_offset_targets"][row_idx, slot, 1])
                coverage_row = coverage.get((split, int(sample_index)), {})
                diag = {
                    "split": split,
                    "sample_index": int(sample_index),
                    "component_slot": slot,
                    "hard_case_type": defect.get("hard_case_type", polygon.get("hard_case_type", "")),
                    "component_type": polygon.get("component_type", ""),
                    "true_center_x_bin": true_x,
                    "pred_center_x_bin": pred_x,
                    "true_center_y_bin": true_y,
                    "pred_center_y_bin": pred_y,
                    "x_bin_error": x_error,
                    "y_bin_error": y_error,
                    "abs_x_bin_error": abs(x_error),
                    "abs_y_bin_error": abs(y_error),
                    "x_bin_correct": int(x_error == 0),
                    "y_bin_correct": int(y_error == 0),
                    "both_bins_correct": int(x_error == 0 and y_error == 0),
                    "center_offset_mae": _float(pred_row, "center_offset_mae"),
                    "center_offset_y_abs": abs(target_offset_y),
                    "center_offset_y_margin_to_boundary": 0.5 - abs(target_offset_y),
                    "local_vertex_grid_mae": _float(pred_row, "local_vertex_mae_grid"),
                    "decoded_vertex_mae": _float(pred_row, "decoded_vertex_mae"),
                    "target_bbox_height_grid": _target_bbox_height_grid(
                        targets["polygon_vertices_norm"][row_idx, slot],
                        targets["polygon_vertex_mask"][row_idx, slot],
                        grid_dy,
                    ),
                    "defect_axis_y": _float(defect, "defect_axis_y"),
                    "rotation_angle": _float(defect, "rotation_angle"),
                    "is_true_rotated": int(_bool(polygon.get("is_true_rotated", defect.get("true_rotated_geometry", "")))),
                    "is_true_multi_component": int(_bool(polygon.get("is_true_multi_component", defect.get("true_multi_component_geometry", "")))),
                    "type_true": _int(pred_row, "type_true"),
                    "type_pred": _int(pred_row, "type_pred"),
                    "type_correct": int(_int(pred_row, "type_true") == _int(pred_row, "type_pred")),
                    "polygon_iou": _float(mask_row, "polygon_mask_iou"),
                    "zero_iou": int(_float(mask_row, "polygon_mask_iou") <= 0.0),
                    "pred_area": _int(mask_row, "pred_area"),
                    "target_area": _int(mask_row, "target_area"),
                    "area_diff": _int(mask_row, "pred_area") - _int(mask_row, "target_area"),
                    "all_bins_exactly_covered": int(_bool(coverage_row.get("all_bins_exactly_covered", ""))),
                    "all_bins_within_distance1": int(_bool(coverage_row.get("all_bins_within_distance1", ""))),
                    "max_center_bin_distance_to_train": _float(coverage_row, "max_center_bin_distance_to_train"),
                }
                per_component.append(diag)
                sample_components.append(diag)
            if sample_components:
                per_sample.append(
                    {
                        "split": split,
                        "sample_index": int(sample_index),
                        "hard_case_type": sample_components[0]["hard_case_type"],
                        "component_count": len(sample_components),
                        "polygon_iou": _float(mask_row, "polygon_mask_iou"),
                        "zero_iou": int(_float(mask_row, "polygon_mask_iou") <= 0.0),
                        "any_x_bin_error": int(any(item["x_bin_correct"] == 0 for item in sample_components)),
                        "any_y_bin_error": int(any(item["y_bin_correct"] == 0 for item in sample_components)),
                        "all_bins_correct": int(all(item["both_bins_correct"] == 1 for item in sample_components)),
                        "mean_abs_y_bin_error": mean(float(item["abs_y_bin_error"]) for item in sample_components),
                        "mean_local_vertex_grid_mae": mean(float(item["local_vertex_grid_mae"]) for item in sample_components),
                        "pred_area": _int(mask_row, "pred_area"),
                        "target_area": _int(mask_row, "target_area"),
                        "area_diff": _int(mask_row, "pred_area") - _int(mask_row, "target_area"),
                        "is_true_rotated": int(any(item["is_true_rotated"] for item in sample_components)),
                        "is_true_multi_component": int(any(item["is_true_multi_component"] for item in sample_components)),
                        "all_bins_exactly_covered": sample_components[0]["all_bins_exactly_covered"],
                        "all_bins_within_distance1": sample_components[0]["all_bins_within_distance1"],
                        "max_center_bin_distance_to_train": sample_components[0]["max_center_bin_distance_to_train"],
                    }
                )
    _write_csv(output_dir / "per_component_y_bin_diagnostics.csv", per_component)
    _write_csv(output_dir / "per_sample_y_bin_diagnostics.csv", per_sample)
    _write_csv(output_dir / "grouped_by_y_bin.csv", _summarize_group(per_component, "true_center_y_bin"))
    _write_csv(output_dir / "grouped_by_hard_case_type.csv", _summarize_group(per_component, "hard_case_type"))
    _write_csv(output_dir / "grouped_by_component_slot.csv", _summarize_group(per_component, "component_slot"))
    _write_csv(output_dir / "grouped_by_true_rotated.csv", _summarize_group(per_component, "is_true_rotated"))
    _write_csv(output_dir / "grouped_by_multi_component.csv", _summarize_group(per_component, "is_true_multi_component"))
    confusion = _confusion_rows(per_component)
    histogram = _histogram_rows(per_component)
    _write_csv(output_dir / "y_bin_confusion.csv", confusion)
    _write_csv(output_dir / "y_bin_error_histogram.csv", histogram)
    summary = _summary(per_component, per_sample)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {
        "component_count": len(per_component),
        "sample_count": len(per_sample),
        "summary": summary,
    }


def _confusion_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], int(row["true_center_y_bin"]), int(row["pred_center_y_bin"]))].append(row)
    out = []
    for (split, true_y, pred_y), items in sorted(grouped.items()):
        out.append(
            {
                "split": split,
                "true_center_y_bin": true_y,
                "pred_center_y_bin": pred_y,
                "count": len(items),
                "mean_polygon_iou": mean(float(item["polygon_iou"]) for item in items),
                "zero_iou_count": sum(int(item["zero_iou"]) for item in items),
            }
        )
    return out


def _histogram_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], int(row["abs_y_bin_error"]))].append(row)
    out = []
    for (split, abs_err), items in sorted(grouped.items()):
        out.append(
            {
                "split": split,
                "abs_y_bin_error": abs_err,
                "count": len(items),
                "mean_polygon_iou": mean(float(item["polygon_iou"]) for item in items),
                "zero_iou_count": sum(int(item["zero_iou"]) for item in items),
            }
        )
    return out


def _split_rows(rows: list[dict], split: str) -> list[dict]:
    return [row for row in rows if row["split"] == split]


def _summary(per_component: list[dict], per_sample: list[dict]) -> str:
    lines = [
        "# Center-Anchored Polygon Y-Bin Diagnostics",
        "",
        "This diagnostic reads existing S300 matched-coverage predictions only; it does not run training.",
        "",
        "## Split Summary",
        "",
        "| split | samples | zero_iou | x_bin_acc | y_bin_acc | y_within1 | mean_abs_y_error | mean_iou |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        comps = _split_rows(per_component, split)
        samples = _split_rows(per_sample, split)
        if not comps:
            continue
        abs_y = [float(row["abs_y_bin_error"]) for row in comps]
        lines.append(
            f"| {split} | `{len(samples)}` | `{sum(int(row['zero_iou']) for row in samples)}` | "
            f"`{_mean_bool(comps, 'x_bin_correct'):.6f}` | `{_mean_bool(comps, 'y_bin_correct'):.6f}` | "
            f"`{mean(1.0 if err <= 1.0 else 0.0 for err in abs_y):.6f}` | `{mean(abs_y):.6f}` | "
            f"`{mean(float(row['polygon_iou']) for row in samples):.6f}` |"
        )
    heldout_comps = [row for row in per_component if row["split"] in {"val", "test"}]
    heldout_zero = [row for row in heldout_comps if int(row["zero_iou"]) == 1]
    y_wrong_zero = sum(int(row["y_bin_correct"]) == 0 for row in heldout_zero)
    x_wrong_zero = sum(int(row["x_bin_correct"]) == 0 for row in heldout_zero)
    adjacent = sum(int(row["abs_y_bin_error"]) == 1 for row in heldout_comps if int(row["y_bin_correct"]) == 0)
    far = sum(int(row["abs_y_bin_error"]) >= 2 for row in heldout_comps if int(row["y_bin_correct"]) == 0)
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- heldout zero-IoU present components: `{len(heldout_zero)}`",
            f"- zero-IoU components with y-bin error: `{y_wrong_zero}`",
            f"- zero-IoU components with x-bin error: `{x_wrong_zero}`",
            f"- heldout y-bin adjacent errors: `{adjacent}`",
            f"- heldout y-bin distance >=2 errors: `{far}`",
            "",
            "Y-bin error is reported as ordered distance, so adjacent-bin and far-bin failures are separated instead of collapsed into ordinary classification errors.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resplit-root", type=Path, default=DEFAULT_RESPLIT_ROOT)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = build_diagnostics(args.resplit_root, args.prediction_dir, args.output_dir)
    print(f"Wrote {result['component_count']} component diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
