"""Diagnose polygon vertex errors that are amplified by hard rasterization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from comsol_polygon_rasterizer import mask_iou_dice, rasterize_polygon_components


def _usage() -> str:
    return (
        "Usage: python comsol_polygon_raster_sensitivity_diagnostics.py "
        "--predictions-csv train_polygon_predictions.csv "
        "--mask-metrics-csv train_polygon_mask_metrics.csv "
        "--npz-path subset.npz --polygon-targets subset_polygon_targets.npz "
        "--output-dir out [--label label]"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _poly_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def _edge_lengths(vertices: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)


def _bbox(vertices: np.ndarray) -> tuple[float, float, float, float]:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])


def _prediction_arrays(rows: list[dict[str, str]], targets: dict) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], dict]]:
    vertices = np.zeros_like(targets["vertices"], dtype=np.float32)
    presence = np.zeros_like(targets["presence"], dtype=np.float32)
    by_key: dict[tuple[int, int], dict] = {}
    index_to_row = {int(sample_index): row_idx for row_idx, sample_index in enumerate(targets["sample_indices"].tolist())}
    for row in rows:
        sample_index = int(row["sample_index"])
        slot = int(row["component_slot"])
        if sample_index not in index_to_row:
            raise ValueError(f"Prediction sample_index {sample_index} is absent from polygon targets.")
        row_idx = index_to_row[sample_index]
        presence[row_idx, slot] = float(row["presence_pred"])
        for vertex_idx in range(vertices.shape[2]):
            vertices[row_idx, slot, vertex_idx, 0] = float(row[f"pred_x{vertex_idx}"])
            vertices[row_idx, slot, vertex_idx, 1] = float(row[f"pred_y{vertex_idx}"])
        by_key[(sample_index, slot)] = row
    return vertices, presence, by_key


def _load_targets(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {
            "vertices": data["polygon_vertices_norm"].astype(np.float32),
            "vertex_mask": data["polygon_vertex_mask"].astype(np.float32),
            "presence": data["presence_targets"].astype(np.float32),
            "type_targets": data["type_targets"].astype(np.int64),
            "sample_indices": data["sample_indices"].astype(np.int64),
        }


def _grid_spacing(values: np.ndarray, name: str) -> float:
    if values.size < 2:
        raise ValueError(f"{name} grid must have at least two points.")
    spacing = float(np.mean(np.diff(values.astype(np.float64))))
    if spacing == 0.0:
        raise ValueError(f"{name} grid spacing must be non-zero.")
    return abs(spacing)


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.npz_path, allow_pickle=True) as data:
        masks = data["masks"].astype(np.float32)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
    targets = _load_targets(Path(args.polygon_targets))
    prediction_rows = _read_csv(Path(args.predictions_csv))
    mask_metric_rows = _read_csv(Path(args.mask_metrics_csv)) if args.mask_metrics_csv else []
    pred_vertices, pred_presence, prediction_by_key = _prediction_arrays(prediction_rows, targets)
    dx = _grid_spacing(x, "x")
    dy = _grid_spacing(y, "y")
    true_vertices = targets["vertices"]
    true_vertex_mask = targets["vertex_mask"]
    true_presence = targets["presence"]
    pred_vertex_mask = np.ones_like(true_vertex_mask, dtype=np.float32)
    pred_masks = rasterize_polygon_components(pred_vertices, pred_vertex_mask, pred_presence, x, y)
    target_poly_masks = rasterize_polygon_components(true_vertices, true_vertex_mask, true_presence, x, y)
    pred_ious, pred_dices = mask_iou_dice(pred_masks, masks)
    oracle_ious, _oracle_dices = mask_iou_dice(target_poly_masks, masks)
    vertex_rows = []
    component_rows = []
    for row_idx, sample_index in enumerate(targets["sample_indices"].tolist()):
        for slot in range(true_presence.shape[1]):
            if true_presence[row_idx, slot] <= 0.5:
                continue
            key = (int(sample_index), int(slot))
            if key not in prediction_by_key:
                raise ValueError(f"Missing prediction row for sample={sample_index} slot={slot}.")
            valid = true_vertex_mask[row_idx, slot] > 0.5
            pred_valid = pred_vertices[row_idx, slot, valid].astype(np.float64)
            true_valid = true_vertices[row_idx, slot, valid].astype(np.float64)
            delta = pred_valid - true_valid
            cell_delta = np.column_stack([delta[:, 0] / dx, delta[:, 1] / dy])
            pred_grid = np.column_stack([pred_valid[:, 0] / dx, pred_valid[:, 1] / dy])
            true_grid = np.column_stack([true_valid[:, 0] / dx, true_valid[:, 1] / dy])
            for local_idx, vertex_idx in enumerate(np.flatnonzero(valid)):
                vertex_rows.append(
                    {
                        "label": args.label,
                        "sample_index": int(sample_index),
                        "component_slot": int(slot),
                        "vertex_index": int(vertex_idx),
                        "pred_x": float(pred_valid[local_idx, 0]),
                        "pred_y": float(pred_valid[local_idx, 1]),
                        "true_x": float(true_valid[local_idx, 0]),
                        "true_y": float(true_valid[local_idx, 1]),
                        "dx_norm": float(delta[local_idx, 0]),
                        "dy_norm": float(delta[local_idx, 1]),
                        "dx_cells": float(cell_delta[local_idx, 0]),
                        "dy_cells": float(cell_delta[local_idx, 1]),
                        "l2_cells": float(np.linalg.norm(cell_delta[local_idx])),
                    }
                )
            pred_bbox = _bbox(pred_grid)
            true_bbox = _bbox(true_grid)
            pred_edges = _edge_lengths(pred_grid)
            true_edges = _edge_lengths(true_grid)
            pred_area_grid = _poly_area(pred_grid)
            true_area_grid = _poly_area(true_grid)
            component_rows.append(
                {
                    "label": args.label,
                    "sample_index": int(sample_index),
                    "component_slot": int(slot),
                    "max_abs_dx_cells": float(np.max(np.abs(cell_delta[:, 0]))),
                    "max_abs_dy_cells": float(np.max(np.abs(cell_delta[:, 1]))),
                    "mean_l2_cells": float(np.mean(np.linalg.norm(cell_delta, axis=1))),
                    "max_l2_cells": float(np.max(np.linalg.norm(cell_delta, axis=1))),
                    "pred_area_grid": float(pred_area_grid),
                    "true_area_grid": float(true_area_grid),
                    "area_grid_diff": float(pred_area_grid - true_area_grid),
                    "area_grid_ratio": float(pred_area_grid / true_area_grid) if true_area_grid else 0.0,
                    "pred_width_cells": float(pred_bbox[2] - pred_bbox[0]),
                    "true_width_cells": float(true_bbox[2] - true_bbox[0]),
                    "pred_height_cells": float(pred_bbox[3] - pred_bbox[1]),
                    "true_height_cells": float(true_bbox[3] - true_bbox[1]),
                    "max_edge_diff_cells": float(np.max(np.abs(pred_edges - true_edges))),
                    "mean_edge_diff_cells": float(np.mean(np.abs(pred_edges - true_edges))),
                }
            )
    sample_rows = []
    for row_idx, sample_index in enumerate(targets["sample_indices"].tolist()):
        pred_mask = pred_masks[row_idx] > 0.5
        true_mask = masks[row_idx] > 0.5
        target_poly_mask = target_poly_masks[row_idx] > 0.5
        false_positive = int(np.logical_and(pred_mask, ~true_mask).sum())
        false_negative = int(np.logical_and(~pred_mask, true_mask).sum())
        target_area = int(true_mask.sum())
        pred_area = int(pred_mask.sum())
        sample_rows.append(
            {
                "label": args.label,
                "sample_index": int(sample_index),
                "polygon_mask_iou": float(pred_ious[row_idx]),
                "polygon_dice": float(pred_dices[row_idx]),
                "oracle_iou": float(oracle_ious[row_idx]),
                "target_area": target_area,
                "pred_area": pred_area,
                "area_diff_pixels": int(pred_area - target_area),
                "area_ratio": float(pred_area / target_area) if target_area else 0.0,
                "false_positive_pixels": false_positive,
                "false_negative_pixels": false_negative,
                "pixel_disagreement": int(np.logical_xor(pred_mask, true_mask).sum()),
                "target_polygon_area": int(target_poly_mask.sum()),
            }
        )
    _write_csv(output_dir / "vertex_raster_sensitivity.csv", vertex_rows)
    _write_csv(output_dir / "component_raster_sensitivity.csv", component_rows)
    _write_csv(output_dir / "sample_raster_sensitivity.csv", sample_rows)
    if mask_metric_rows:
        _write_csv(output_dir / "input_mask_metrics_copy.csv", mask_metric_rows)
    worst_sample = min(sample_rows, key=lambda row: row["polygon_mask_iou"])
    worst_component = max(component_rows, key=lambda row: max(row["max_abs_dx_cells"], row["max_abs_dy_cells"]))
    summary = [
        "# COMSOL polygon vertex-to-raster sensitivity summary",
        "",
        f"- label: `{args.label}`",
        f"- samples: `{len(sample_rows)}`",
        f"- x grid spacing: `{dx:.12g}`",
        f"- y grid spacing: `{dy:.12g}`",
        f"- mean hard polygon IoU: `{float(np.mean(pred_ious)):.6f}`",
        f"- min hard polygon IoU: `{float(np.min(pred_ious)):.6f}`",
        f"- mean oracle IoU from target vertices: `{float(np.mean(oracle_ious)):.6f}`",
        f"- min oracle IoU from target vertices: `{float(np.min(oracle_ious)):.6f}`",
        f"- worst sample: `{worst_sample['sample_index']}` with IoU `{worst_sample['polygon_mask_iou']:.6f}` and area diff `{worst_sample['area_diff_pixels']}` pixels",
        f"- worst component max dx/dy cells: `{worst_component['max_abs_dx_cells']:.6f}` / `{worst_component['max_abs_dy_cells']:.6f}`",
        "",
        "Interpretation: if oracle IoU remains 1.0 while predicted area and grid-cell vertex errors are non-trivial, the failure is a vertex-to-hard-raster sensitivity / loss-alignment issue rather than a polygon target or rasterizer alignment failure.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Saved polygon raster sensitivity diagnostics to {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv")
    parser.add_argument("--mask-metrics-csv")
    parser.add_argument("--npz-path")
    parser.add_argument("--polygon-targets")
    parser.add_argument("--output-dir")
    parser.add_argument("--label", default="polygon_run")
    args = parser.parse_args(argv)
    if not args.predictions_csv or not args.npz_path or not args.polygon_targets or not args.output_dir:
        print(_usage())
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
