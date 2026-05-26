"""Diagnose center-anchored polygon local-shape targets and predictions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


SPLITS = ("train", "val", "test")
DEFAULT_RESPLIT_ROOT = Path("experiments/dual_network/S299_comsol_polygon_matched_coverage_resplit")
DEFAULT_PREDICTION_DIR = Path("experiments/dual_network/S306_center_anchored_y_bin_repair_quick_gate/current_reference")
DEFAULT_OUTPUT_DIR = Path("experiments/dual_network/S309_center_anchored_local_shape_diagnostics")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
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


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" or value is None else float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value == "" or value is None else int(round(float(value)))


def _bool(value: str | int | float | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0", ""}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return False


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing target NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def _edges(vertices: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _safe_min(values: list[float]) -> float:
    return float(min(values)) if values else 0.0


def _component_meta(resplit_root: Path, split: str) -> tuple[dict[int, dict[str, str]], dict[tuple[int, int], dict[str, str]]]:
    defects: dict[int, dict[str, str]] = {}
    for row in _read_csv(resplit_root / split / "defect_params.csv"):
        key = _int(row, "sample_index")
        if key in defects:
            raise ValueError(f"Duplicate defect_params row for split={split}, sample_index={key}")
        defects[key] = row
    polygons: dict[tuple[int, int], dict[str, str]] = {}
    for row in _read_csv(resplit_root / split / "polygon_params.csv"):
        key = (_int(row, "sample_index"), _int(row, "component_index"))
        if key in polygons:
            raise ValueError(f"Duplicate polygon_params row for split={split}, sample_index={key[0]}, component_index={key[1]}")
        polygons[key] = row
    return defects, polygons


def _prediction_rows(prediction_dir: Path, split: str) -> tuple[dict[tuple[int, int], dict[str, str]], dict[int, dict[str, str]]]:
    components: dict[tuple[int, int], dict[str, str]] = {}
    for row in _read_csv(prediction_dir / f"{split}_center_anchored_polygon_predictions.csv"):
        key = (_int(row, "sample_index"), _int(row, "component_slot"))
        if key in components:
            raise ValueError(f"Duplicate prediction row for split={split}, sample_index={key[0]}, component_slot={key[1]}")
        components[key] = row
    samples: dict[int, dict[str, str]] = {}
    for row in _read_csv(prediction_dir / f"{split}_center_anchored_polygon_mask_metrics.csv"):
        key = _int(row, "sample_index")
        if key in samples:
            raise ValueError(f"Duplicate mask metric row for split={split}, sample_index={key}")
        samples[key] = row
    return components, samples


def _component_prediction_local(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [[_float(row, f"{prefix}_local_x{i}"), _float(row, f"{prefix}_local_y{i}")] for i in range(4)],
        dtype=np.float64,
    )


def _target_rows_for_split(resplit_root: Path, split: str) -> list[dict]:
    targets = _load_npz(resplit_root / split / "center_anchored_polygon_targets.npz")
    defects, polygons = _component_meta(resplit_root, split)
    local_vertices = targets["local_vertices_grid"].astype(np.float64)
    vertex_mask = targets["polygon_vertex_mask"] > 0.5
    presence = targets["presence_targets"] > 0.5
    sample_indices = targets["sample_indices"].astype(int)
    rows = []
    for row_idx, sample_index in enumerate(sample_indices):
        defect = defects[int(sample_index)]
        for slot in range(local_vertices.shape[1]):
            if not presence[row_idx, slot]:
                continue
            valid = vertex_mask[row_idx, slot]
            vertices = local_vertices[row_idx, slot, valid]
            if vertices.shape[0] != 4:
                continue
            poly = polygons.get((int(sample_index), slot), {})
            xs = vertices[:, 0]
            ys = vertices[:, 1]
            edge_lengths = _edges(vertices)
            rows.append(
                {
                    "split": split,
                    "sample_index": int(sample_index),
                    "component_slot": slot,
                    "hard_case_type": defect.get("hard_case_type", ""),
                    "component_type": poly.get("component_type", ""),
                    "is_true_rotated": int(_bool(poly.get("is_true_rotated", defect.get("is_true_rotated", "0")))),
                    "is_true_multi_component": int(_bool(poly.get("is_true_multi_component", defect.get("is_true_multi_component", "0")))),
                    "local_abs_x_max": float(np.max(np.abs(xs))),
                    "local_abs_y_max": float(np.max(np.abs(ys))),
                    "local_bbox_width_grid": float(xs.max() - xs.min()),
                    "local_bbox_height_grid": float(ys.max() - ys.min()),
                    "local_area_grid": _area(vertices),
                    "local_edge_min_grid": float(edge_lengths.min()),
                    "local_edge_mean_grid": float(edge_lengths.mean()),
                    "local_edge_max_grid": float(edge_lengths.max()),
                }
            )
    return rows


def _prediction_rows_for_split(resplit_root: Path, prediction_dir: Path, split: str) -> tuple[list[dict], list[dict]]:
    defects, polygons = _component_meta(resplit_root, split)
    component_predictions, sample_predictions = _prediction_rows(prediction_dir, split)
    component_rows = []
    sample_rows = []
    by_sample: dict[int, list[dict]] = defaultdict(list)
    for (sample_index, slot), row in sorted(component_predictions.items()):
        if _float(row, "presence_true") <= 0.5:
            continue
        defect = defects[int(sample_index)]
        poly = polygons.get((int(sample_index), slot), {})
        valid = np.asarray([_bool(row.get(f"vertex{i}_valid", "0")) for i in range(4)])
        pred_local = _component_prediction_local(row, "pred")
        true_local = _component_prediction_local(row, "true")
        pred_raw = pred_local
        if "pred_local_raw_x0" in row:
            pred_raw = np.asarray(
                [[_float(row, f"pred_local_raw_x{i}"), _float(row, f"pred_local_raw_y{i}")] for i in range(4)],
                dtype=np.float64,
            )
        local_errors = np.abs(pred_local[valid] - true_local[valid]) if valid.any() else np.zeros((0, 2))
        sample_mask = sample_predictions[int(sample_index)]
        pred_area = _int(sample_mask, "pred_area")
        target_area = _int(sample_mask, "target_area")
        x_bin_true = _int(row, "center_x_bin_true")
        x_bin_pred = _int(row, "center_x_bin_pred")
        y_bin_true = _int(row, "center_y_bin_true")
        y_bin_pred = _int(row, "center_y_bin_pred")
        out_row = {
            "split": split,
            "sample_index": int(sample_index),
            "component_slot": slot,
            "hard_case_type": defect.get("hard_case_type", ""),
            "component_type": poly.get("component_type", ""),
            "is_true_rotated": int(_bool(poly.get("is_true_rotated", defect.get("is_true_rotated", "0")))),
            "is_true_multi_component": int(_bool(poly.get("is_true_multi_component", defect.get("is_true_multi_component", "0")))),
            "polygon_iou": _float(sample_mask, "polygon_mask_iou"),
            "zero_iou": int(_float(sample_mask, "polygon_mask_iou") <= 0.0),
            "target_area": target_area,
            "pred_area": pred_area,
            "area_diff": pred_area - target_area,
            "x_bin_correct": int(x_bin_true == x_bin_pred),
            "y_bin_correct": int(y_bin_true == y_bin_pred),
            "both_bins_correct": int(x_bin_true == x_bin_pred and y_bin_true == y_bin_pred),
            "x_bin_abs_error": abs(x_bin_pred - x_bin_true),
            "y_bin_abs_error": abs(y_bin_pred - y_bin_true),
            "local_vertex_mae_grid": float(local_errors.mean()) if local_errors.size else 0.0,
            "local_vertex_max_error_grid": float(local_errors.max()) if local_errors.size else 0.0,
            "pred_local_abs_x_max": float(np.max(np.abs(pred_local[valid, 0]))) if valid.any() else 0.0,
            "pred_local_abs_y_max": float(np.max(np.abs(pred_local[valid, 1]))) if valid.any() else 0.0,
            "pred_local_raw_abs_x_max": float(np.max(np.abs(pred_raw[valid, 0]))) if valid.any() else 0.0,
            "pred_local_raw_abs_y_max": float(np.max(np.abs(pred_raw[valid, 1]))) if valid.any() else 0.0,
            "presence_correct": int(_float(row, "presence_true") == _float(row, "presence_pred")),
            "type_correct": int(_int(row, "type_true") == _int(row, "type_pred")),
            "signed_area_flip": _int(row, "signed_area_flip"),
            "out_of_grid_vertex_count": _int(sample_mask, "out_of_grid_vertex_count"),
        }
        component_rows.append(out_row)
        by_sample[int(sample_index)].append(out_row)
    for sample_index, items in sorted(by_sample.items()):
        mask_row = sample_predictions[int(sample_index)]
        sample_rows.append(
            {
                "split": split,
                "sample_index": sample_index,
                "hard_case_type": defects[sample_index].get("hard_case_type", ""),
                "component_count": len(items),
                "polygon_iou": _float(mask_row, "polygon_mask_iou"),
                "zero_iou": int(_float(mask_row, "polygon_mask_iou") <= 0.0),
                "target_area": _int(mask_row, "target_area"),
                "pred_area": _int(mask_row, "pred_area"),
                "area_diff": _int(mask_row, "pred_area") - _int(mask_row, "target_area"),
                "all_x_bins_correct": int(all(item["x_bin_correct"] for item in items)),
                "all_y_bins_correct": int(all(item["y_bin_correct"] for item in items)),
                "all_bins_correct": int(all(item["both_bins_correct"] for item in items)),
                "mean_local_vertex_mae_grid": _safe_mean([float(item["local_vertex_mae_grid"]) for item in items]),
                "max_local_vertex_error_grid": _safe_max([float(item["local_vertex_max_error_grid"]) for item in items]),
                "true_rotated_component_count": sum(int(item["is_true_rotated"]) for item in items),
                "true_multi_component_count": sum(int(item["is_true_multi_component"]) for item in items),
                "out_of_grid_vertex_count": _int(mask_row, "out_of_grid_vertex_count"),
            }
        )
    return component_rows, sample_rows


def _summarize_target(rows: list[dict], group_key: str) -> list[dict]:
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
                "local_abs_x_max": _safe_max([float(item["local_abs_x_max"]) for item in items]),
                "local_abs_x_p95": float(np.percentile([float(item["local_abs_x_max"]) for item in items], 95)),
                "local_abs_y_max": _safe_max([float(item["local_abs_y_max"]) for item in items]),
                "local_abs_y_p95": float(np.percentile([float(item["local_abs_y_max"]) for item in items], 95)),
                "bbox_width_mean": _safe_mean([float(item["local_bbox_width_grid"]) for item in items]),
                "bbox_height_mean": _safe_mean([float(item["local_bbox_height_grid"]) for item in items]),
                "area_mean": _safe_mean([float(item["local_area_grid"]) for item in items]),
                "edge_mean": _safe_mean([float(item["local_edge_mean_grid"]) for item in items]),
            }
        )
    return out


def _summarize_predictions(rows: list[dict], group_key: str) -> list[dict]:
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
                "mean_polygon_iou": _safe_mean([float(item["polygon_iou"]) for item in items]),
                "zero_iou_rate": _safe_mean([float(item["zero_iou"]) for item in items]),
                "x_bin_acc": _safe_mean([float(item["x_bin_correct"]) for item in items]),
                "y_bin_acc": _safe_mean([float(item["y_bin_correct"]) for item in items]),
                "both_bins_acc": _safe_mean([float(item["both_bins_correct"]) for item in items]),
                "local_vertex_mae_grid": _safe_mean([float(item["local_vertex_mae_grid"]) for item in items]),
                "local_vertex_max_error_grid": _safe_mean([float(item["local_vertex_max_error_grid"]) for item in items]),
                "area_diff_mean": _safe_mean([float(item["area_diff"]) for item in items]),
                "area_diff_abs_mean": _safe_mean([abs(float(item["area_diff"])) for item in items]),
                "pred_local_abs_x_max": _safe_max([float(item["pred_local_abs_x_max"]) for item in items]),
                "pred_local_abs_y_max": _safe_max([float(item["pred_local_abs_y_max"]) for item in items]),
            }
        )
    return out


def run(resplit_root: Path, prediction_dir: Path, output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_rows: list[dict] = []
    component_rows: list[dict] = []
    sample_rows: list[dict] = []
    for split in SPLITS:
        target_rows.extend(_target_rows_for_split(resplit_root, split))
        comp, sample = _prediction_rows_for_split(resplit_root, prediction_dir, split)
        component_rows.extend(comp)
        sample_rows.extend(sample)
    _write_csv(output_dir / "local_shape_target_components.csv", target_rows)
    _write_csv(output_dir / "local_shape_target_stats_by_split.csv", _summarize_target(target_rows, "split"))
    grouped_targets = []
    for key in ["hard_case_type", "component_slot", "is_true_rotated", "is_true_multi_component"]:
        for row in _summarize_target(target_rows, key):
            row["group_key"] = key
            row["group_value"] = row.pop(key)
            grouped_targets.append(row)
    _write_csv(output_dir / "local_shape_target_stats_by_group.csv", grouped_targets)
    _write_csv(output_dir / "local_shape_prediction_diagnostics_per_component.csv", component_rows)
    _write_csv(output_dir / "local_shape_prediction_diagnostics_per_sample.csv", sample_rows)
    grouped_predictions = []
    for key in ["hard_case_type", "component_slot", "is_true_rotated", "is_true_multi_component", "both_bins_correct"]:
        for row in _summarize_predictions(component_rows, key):
            row["group_key"] = key
            row["group_value"] = row.pop(key)
            grouped_predictions.append(row)
    _write_csv(output_dir / "local_shape_prediction_diagnostics_by_group.csv", grouped_predictions)
    worst = sorted(
        [row for row in sample_rows if row["split"] in {"val", "test"}],
        key=lambda row: (float(row["polygon_iou"]), -float(row["mean_local_vertex_mae_grid"])),
    )[:10]
    _write_csv(output_dir / "worst_heldout_local_shape_samples.csv", worst)
    summary = _summary_lines(target_rows, component_rows, sample_rows, prediction_dir)
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {
        "component_rows": float(len(component_rows)),
        "sample_rows": float(len(sample_rows)),
        "heldout_zero_iou": float(sum(int(row["zero_iou"]) for row in sample_rows if row["split"] in {"val", "test"})),
    }


def _split_values(rows: list[dict], split: str, key: str) -> list[float]:
    return [float(row[key]) for row in rows if row["split"] == split]


def _summary_lines(target_rows: list[dict], component_rows: list[dict], sample_rows: list[dict], prediction_dir: Path) -> list[str]:
    lines = [
        "# S309 center-anchored local-shape diagnostics",
        "",
        f"- prediction_dir: `{prediction_dir}`",
        "- scope: read-only diagnostics; no training, model changes, or COMSOL generation.",
        "",
        "## Target local-shape range",
        "",
    ]
    for split in SPLITS:
        xvals = _split_values(target_rows, split, "local_abs_x_max")
        yvals = _split_values(target_rows, split, "local_abs_y_max")
        areas = _split_values(target_rows, split, "local_area_grid")
        lines.append(
            f"- {split}: components `{len(xvals)}`, local_abs_x max/p95 `{_safe_max(xvals):.6f}` / `{np.percentile(xvals, 95):.6f}`, "
            f"local_abs_y max/p95 `{_safe_max(yvals):.6f}` / `{np.percentile(yvals, 95):.6f}`, area mean/max `{_safe_mean(areas):.6f}` / `{_safe_max(areas):.6f}`."
        )
    lines.extend(["", "## Reference prediction linkage", ""])
    for split in SPLITS:
        samples = [row for row in sample_rows if row["split"] == split]
        comps = [row for row in component_rows if row["split"] == split]
        lines.append(
            f"- {split}: mean IoU `{_safe_mean([float(row['polygon_iou']) for row in samples]):.6f}`, zero-IoU `{sum(int(row['zero_iou']) for row in samples)}/{len(samples)}`, "
            f"local_vertex_mae_grid `{_safe_mean([float(row['local_vertex_mae_grid']) for row in comps]):.6f}`, both-bin acc `{_safe_mean([float(row['both_bins_correct']) for row in comps]):.6f}`."
        )
    correct = [row for row in component_rows if int(row["both_bins_correct"])]
    wrong = [row for row in component_rows if not int(row["both_bins_correct"])]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- both-bin-correct components local_vertex_mae_grid: `{_safe_mean([float(row['local_vertex_mae_grid']) for row in correct]):.6f}`.",
            f"- bin-wrong components local_vertex_mae_grid: `{_safe_mean([float(row['local_vertex_mae_grid']) for row in wrong]):.6f}`.",
            "- The bounded-output gate should preserve the center-bin path and only constrain effective local vertices used by loss, decode, metrics, and prediction export.",
        ]
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resplit-root", type=Path, default=DEFAULT_RESPLIT_ROOT)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    stats = run(args.resplit_root, args.prediction_dir, args.output_dir)
    print(
        f"Saved local-shape diagnostics to {args.output_dir} "
        f"(components={int(stats['component_rows'])}, samples={int(stats['sample_rows'])}, heldout_zero_iou={int(stats['heldout_zero_iou'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
