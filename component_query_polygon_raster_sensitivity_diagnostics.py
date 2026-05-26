"""Offline raster-sensitivity diagnostics for a component-query polygon overfit run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from comsol_polygon_rasterizer import mask_iou_dice, rasterize_polygon_components, write_csv


DEFAULT_PREDICTION_CSV = (
    "experiments/dual_network/S330_component_query_polygon_overfit_gates/"
    "one_sample_overfit/train_center_anchored_polygon_predictions.csv"
)
DEFAULT_MASK_METRICS_CSV = (
    "experiments/dual_network/S330_component_query_polygon_overfit_gates/"
    "one_sample_overfit/train_center_anchored_polygon_mask_metrics.csv"
)
DEFAULT_NPZ = "experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/converted/train_comsol_v3_polygon_hard_case.npz"
DEFAULT_TARGETS = (
    "experiments/dual_network/S290_comsol_v3_center_anchored_polygon_targets/train/"
    "center_anchored_polygon_targets.npz"
)
DEFAULT_OUTPUT = "experiments/dual_network/S334_component_query_1sample_raster_sensitivity_diagnostics"


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float(row: dict, key: str) -> float:
    if key not in row:
        raise KeyError(f"Missing required prediction field `{key}`")
    value = row[key]
    if value == "":
        return 0.0
    return float(value)


def _as_int(row: dict, key: str) -> int:
    return int(round(_as_float(row, key)))


def _signed_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _centroid(vertices: np.ndarray) -> np.ndarray:
    return vertices.mean(axis=0)


def _edge_lengths(vertices: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)


def _bbox(vertices: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(vertices[:, 0].min()),
        float(vertices[:, 1].min()),
        float(vertices[:, 0].max()),
        float(vertices[:, 1].max()),
    )


def _load_sample(npz_path: Path, targets_path: Path, sample_index: int) -> dict:
    with np.load(npz_path, allow_pickle=True) as data:
        masks = data["masks"].astype(np.float64)
        x = data["x"].astype(np.float64)
        y = data["y"].astype(np.float64)
    with np.load(targets_path, allow_pickle=True) as data:
        targets = {key: data[key] for key in data.files}
    sample_indices = targets["sample_indices"].astype(np.int64)
    matches = np.where(sample_indices == sample_index)[0]
    if matches.size != 1:
        raise ValueError(f"Expected exactly one target row for sample_index={sample_index}, found {matches.size}")
    row = int(matches[0])
    return {
        "row": row,
        "mask": masks[row],
        "x": x,
        "y": y,
        "vertices": targets["polygon_vertices_norm"][row : row + 1].astype(np.float64),
        "vertex_mask": targets["polygon_vertex_mask"][row : row + 1].astype(np.float64),
        "presence": targets["presence_targets"][row : row + 1].astype(np.float64),
        "center": targets["center_targets_norm"][row : row + 1].astype(np.float64),
        "local": targets["local_vertices_grid"][row : row + 1].astype(np.float64),
        "grid_dx": float(targets["grid_dx"]),
        "grid_dy": float(targets["grid_dy"]),
    }


def _prediction_arrays(prediction_csv: Path, sample_index: int, max_components: int, max_vertices: int) -> dict:
    rows = [row for row in _read_csv(prediction_csv) if _as_int(row, "sample_index") == sample_index]
    if len(rows) != max_components:
        raise ValueError(f"Expected {max_components} prediction rows for sample_index={sample_index}, found {len(rows)}")
    by_slot = {_as_int(row, "component_slot"): row for row in rows}
    required = set(range(max_components))
    if set(by_slot) != required:
        raise ValueError(f"Prediction slots for sample_index={sample_index} are {sorted(by_slot)}, expected {sorted(required)}")
    vertices = np.zeros((1, max_components, max_vertices, 2), dtype=np.float64)
    local = np.zeros_like(vertices)
    centers = np.zeros((1, max_components, 2), dtype=np.float64)
    presence = np.zeros((1, max_components), dtype=np.float64)
    type_pred = np.zeros((1, max_components), dtype=np.int64)
    vertex_mask = np.zeros((1, max_components, max_vertices), dtype=np.float64)
    for slot in range(max_components):
        row = by_slot[slot]
        presence[0, slot] = _as_float(row, "presence_pred")
        type_pred[0, slot] = _as_int(row, "type_pred")
        centers[0, slot] = [_as_float(row, "hard_center_x_pred"), _as_float(row, "hard_center_y_pred")]
        for vertex in range(max_vertices):
            vertex_mask[0, slot, vertex] = _as_float(row, f"vertex{vertex}_valid")
            vertices[0, slot, vertex] = [_as_float(row, f"pred_x{vertex}"), _as_float(row, f"pred_y{vertex}")]
            local[0, slot, vertex] = [_as_float(row, f"pred_local_x{vertex}"), _as_float(row, f"pred_local_y{vertex}")]
    return {
        "vertices": vertices,
        "local": local,
        "center": centers,
        "presence": presence,
        "type_pred": type_pred,
        "vertex_mask": vertex_mask,
        "rows": by_slot,
    }


def _vertices_from_center_local(center: np.ndarray, local: np.ndarray, grid_dx: float, grid_dy: float) -> np.ndarray:
    out = np.zeros_like(local, dtype=np.float64)
    out[..., 0] = center[..., None, 0] + local[..., 0] * grid_dx
    out[..., 1] = center[..., None, 1] + local[..., 1] * grid_dy
    return out


def _scale_about_centroid(vertices: np.ndarray, scale: float) -> np.ndarray:
    center = _centroid(vertices)
    return center + (vertices - center) * scale


def _translate_to_centroid(vertices: np.ndarray, target_vertices: np.ndarray) -> np.ndarray:
    return vertices + (_centroid(target_vertices) - _centroid(vertices))


def _component_geometry_rows(sample_index: int, pred: dict, target: dict) -> tuple[list[dict], list[dict]]:
    per_vertex: list[dict] = []
    edge_rows: list[dict] = []
    max_components = target["presence"].shape[1]
    max_vertices = target["vertices"].shape[2]
    for slot in range(max_components):
        if target["presence"][0, slot] <= 0.5:
            continue
        valid = target["vertex_mask"][0, slot] > 0.5
        pred_v = pred["vertices"][0, slot, valid]
        target_v = target["vertices"][0, slot, valid]
        pred_edges = _edge_lengths(pred_v)
        target_edges = _edge_lengths(target_v)
        for vertex_pos, vertex_idx in enumerate(np.where(valid)[0]):
            dx = pred_v[vertex_pos, 0] - target_v[vertex_pos, 0]
            dy = pred_v[vertex_pos, 1] - target_v[vertex_pos, 1]
            per_vertex.append(
                {
                    "sample_index": sample_index,
                    "component_slot": slot,
                    "vertex_index": int(vertex_idx),
                    "pred_x": float(pred_v[vertex_pos, 0]),
                    "pred_y": float(pred_v[vertex_pos, 1]),
                    "target_x": float(target_v[vertex_pos, 0]),
                    "target_y": float(target_v[vertex_pos, 1]),
                    "dx_norm": float(dx),
                    "dy_norm": float(dy),
                    "dx_cells": float(dx / target["grid_dx"]),
                    "dy_cells": float(dy / target["grid_dy"]),
                    "l2_cells": float(np.hypot(dx / target["grid_dx"], dy / target["grid_dy"])),
                }
            )
        for edge_idx in range(len(pred_edges)):
            edge_rows.append(
                {
                    "sample_index": sample_index,
                    "component_slot": slot,
                    "edge_index": edge_idx,
                    "pred_edge_len_norm": float(pred_edges[edge_idx]),
                    "target_edge_len_norm": float(target_edges[edge_idx]),
                    "edge_len_error_norm": float(pred_edges[edge_idx] - target_edges[edge_idx]),
                    "edge_len_error_cells": float((pred_edges[edge_idx] - target_edges[edge_idx]) / target["grid_dx"]),
                }
            )
    return per_vertex, edge_rows


def _variant_vertices(pred: dict, target: dict) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {}
    pred_vertices = pred["vertices"].copy()
    target_vertices = target["vertices"].copy()
    variants["pred_polygon"] = pred_vertices
    variants["gt_polygon"] = target_vertices
    variants["pred_center_gt_local_vertices"] = _vertices_from_center_local(
        pred["center"], target["local"], target["grid_dx"], target["grid_dy"]
    )
    variants["gt_center_pred_local_vertices"] = _vertices_from_center_local(
        target["center"], pred["local"], target["grid_dx"], target["grid_dy"]
    )
    area_scaled = pred_vertices.copy()
    centroid_aligned = pred_vertices.copy()
    edge_scaled = pred_vertices.copy()

    for slot in range(target["presence"].shape[1]):
        if target["presence"][0, slot] <= 0.5:
            continue
        valid = target["vertex_mask"][0, slot] > 0.5
        pv = pred_vertices[0, slot, valid]
        tv = target_vertices[0, slot, valid]
        pred_area = abs(_signed_area(pv))
        target_area = abs(_signed_area(tv))
        pred_edge = _edge_lengths(pv)
        target_edge = _edge_lengths(tv)
        area_scale = np.sqrt(target_area / pred_area) if pred_area > 0.0 else 1.0
        edge_scale = float(np.mean(target_edge / np.clip(pred_edge, 1e-12, None)))

        area_scaled[0, slot, valid] = _scale_about_centroid(pv, area_scale)
        centroid_aligned[0, slot, valid] = _translate_to_centroid(pv, tv)
        edge_scaled[0, slot, valid] = _scale_about_centroid(pv, edge_scale)
    variants["pred_polygon_area_scaled_to_target"] = area_scaled
    variants["pred_polygon_centroid_aligned_to_target"] = centroid_aligned
    variants["pred_polygon_edge_length_scaled_to_target"] = edge_scaled

    for alpha in (0.25, 0.5, 0.75):
        variants[f"pred_polygon_interpolate_gt_alpha_{alpha:.2f}"] = (1.0 - alpha) * pred_vertices + alpha * target_vertices
    return variants


def _mask_metrics(mask: np.ndarray, target_mask: np.ndarray) -> dict:
    pred = mask > 0.5
    target = target_mask > 0.5
    intersection = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    false_positive = int(np.logical_and(pred, ~target).sum())
    false_negative = int(np.logical_and(~pred, target).sum())
    pred_area = int(pred.sum())
    target_area = int(target.sum())
    dice_denom = pred_area + target_area
    return {
        "intersection": intersection,
        "union": union,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "symmetric_diff_pixels": false_positive + false_negative,
        "pred_area": pred_area,
        "target_area": target_area,
        "area_diff": pred_area - target_area,
        "iou": float(intersection / union if union else 1.0),
        "dice": float(2.0 * intersection / dice_denom if dice_denom else 1.0),
    }


def _variant_rows(sample_index: int, pred: dict, target: dict) -> list[dict]:
    rows: list[dict] = []
    variant_map = _variant_vertices(pred, target)
    for name, vertices in variant_map.items():
        masks = rasterize_polygon_components(vertices, target["vertex_mask"], target["presence"], target["x"], target["y"])
        metrics = _mask_metrics(masks[0], target["mask"])
        present_slot = int(np.where(target["presence"][0] > 0.5)[0][0])
        valid = target["vertex_mask"][0, present_slot] > 0.5
        geom_vertices = vertices[0, present_slot, valid]
        target_vertices = target["vertices"][0, present_slot, valid]
        bbox = _bbox(geom_vertices)
        target_bbox = _bbox(target_vertices)
        centroid_error = np.linalg.norm(_centroid(geom_vertices) - _centroid(target_vertices))
        edge_error = np.abs(_edge_lengths(geom_vertices) - _edge_lengths(target_vertices))
        rows.append(
            {
                "sample_index": sample_index,
                "variant": name,
                **metrics,
                "signed_area": _signed_area(geom_vertices),
                "target_signed_area": _signed_area(target_vertices),
                "geom_area_abs": abs(_signed_area(geom_vertices)),
                "target_geom_area_abs": abs(_signed_area(target_vertices)),
                "centroid_error_norm": float(centroid_error),
                "centroid_error_cells": float(centroid_error / target["grid_dx"]),
                "bbox_xmin_error": float(bbox[0] - target_bbox[0]),
                "bbox_ymin_error": float(bbox[1] - target_bbox[1]),
                "bbox_xmax_error": float(bbox[2] - target_bbox[2]),
                "bbox_ymax_error": float(bbox[3] - target_bbox[3]),
                "max_edge_length_abs_error_norm": float(edge_error.max()),
                "mean_edge_length_abs_error_norm": float(edge_error.mean()),
            }
        )
    return rows


def _write_summary(output_dir: Path, variant_rows: list[dict], vertex_rows: list[dict], edge_rows: list[dict]) -> None:
    by_variant = {row["variant"]: row for row in variant_rows}
    pred = by_variant["pred_polygon"]
    area_scaled = by_variant.get("pred_polygon_area_scaled_to_target", pred)
    centroid = by_variant.get("pred_polygon_centroid_aligned_to_target", pred)
    edge_scaled = by_variant.get("pred_polygon_edge_length_scaled_to_target", pred)
    alphas = [row for row in variant_rows if row["variant"].startswith("pred_polygon_interpolate_gt_alpha")]
    best_alpha = max(alphas, key=lambda row: float(row["iou"])) if alphas else pred
    max_vertex_l2 = max(float(row["l2_cells"]) for row in vertex_rows) if vertex_rows else 0.0
    max_edge_err = max(abs(float(row["edge_len_error_cells"])) for row in edge_rows) if edge_rows else 0.0
    fp = int(pred["false_positive_pixels"])
    fn = int(pred["false_negative_pixels"])
    primary = "false-positive" if fp > fn else "false-negative" if fn > fp else "balanced FP/FN"
    five_area_only = abs(int(pred["area_diff"])) == abs(fp - fn)
    small_alpha_pass = any(float(row["iou"]) >= 0.99 and row["variant"].endswith("0.25") for row in alphas)
    area_pass = float(area_scaled["iou"]) >= 0.99
    centroid_pass = float(centroid["iou"]) >= 0.99
    edge_pass = float(edge_scaled["iou"]) >= 0.99
    if small_alpha_pass:
        recommendation = "1-sample repair quick gate with precision-focused local/area-edge refinement; do not enter 5-sample yet."
    elif area_pass or centroid_pass or edge_pass:
        recommendation = "1-sample repair quick gate with the corresponding centroid/center auxiliary; do not enter 5-sample yet."
    else:
        recommendation = "Do not enter 5-sample; first repair component-query one-sample local-shape precision."
    lines = [
        "# S334 Component-Query 1-Sample Raster Sensitivity Diagnostic",
        "",
        f"- pred IoU: `{float(pred['iou']):.6f}`",
        f"- pred Dice: `{float(pred['dice']):.6f}`",
        f"- pred / target area: `{int(pred['pred_area'])}` / `{int(pred['target_area'])}`",
        f"- area diff: `{int(pred['area_diff'])}`",
        f"- false-positive / false-negative pixels: `{fp}` / `{fn}`",
        f"- symmetric diff pixels: `{int(pred['symmetric_diff_pixels'])}`",
        f"- max vertex error: `{max_vertex_l2:.6f}` grid cells",
        f"- max edge length error: `{max_edge_err:.6f}` x-grid cells",
        "",
        "## Findings",
        "",
        f"1. The `0.974227` IoU is primarily `{primary}` driven.",
        f"2. The 5-pixel raster area surplus {'matches' if five_area_only else 'does not fully explain'} the FP/FN balance.",
        f"3. Area-scaled IoU is `{float(area_scaled['iou']):.6f}`, centroid-aligned IoU is `{float(centroid['iou']):.6f}`, and edge-scaled IoU is `{float(edge_scaled['iou']):.6f}`.",
        f"4. Best alpha interpolation variant is `{best_alpha['variant']}` with IoU `{float(best_alpha['iou']):.6f}`.",
        f"5. Recommendation: {recommendation}",
        "",
        "## Variant Table",
        "",
        "| variant | IoU | pred area | area diff | FP | FN | sym diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in variant_rows:
        lines.append(
            f"| {row['variant']} | `{float(row['iou']):.6f}` | `{int(row['pred_area'])}` | "
            f"`{int(row['area_diff'])}` | `{int(row['false_positive_pixels'])}` | "
            f"`{int(row['false_negative_pixels'])}` | `{int(row['symmetric_diff_pixels'])}` |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostics(
    prediction_csv: Path,
    mask_metrics_csv: Path,
    npz_path: Path,
    targets_path: Path,
    output_dir: Path,
    sample_index: int,
) -> dict:
    if not prediction_csv.exists():
        raise FileNotFoundError(prediction_csv)
    if not mask_metrics_csv.exists():
        raise FileNotFoundError(mask_metrics_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = _load_sample(npz_path, targets_path, sample_index)
    pred = _prediction_arrays(
        prediction_csv,
        sample_index,
        max_components=target["vertices"].shape[1],
        max_vertices=target["vertices"].shape[2],
    )
    variant_rows = _variant_rows(sample_index, pred, target)
    pred_row = next(row for row in variant_rows if row["variant"] == "pred_polygon")
    exported = _read_csv(mask_metrics_csv)
    exported_row = next(row for row in exported if int(float(row["sample_index"])) == sample_index)
    if abs(float(exported_row["polygon_mask_iou"]) - float(pred_row["iou"])) > 1e-9:
        raise ValueError("Offline pred_polygon IoU does not reproduce exported mask metrics.")
    if int(float(exported_row["pred_area"])) != int(pred_row["pred_area"]):
        raise ValueError("Offline pred_polygon area does not reproduce exported mask metrics.")
    vertex_rows, edge_rows = _component_geometry_rows(sample_index, pred, target)
    mask_summary = [
        {
            key: pred_row[key]
            for key in [
                "sample_index",
                "variant",
                "intersection",
                "union",
                "false_positive_pixels",
                "false_negative_pixels",
                "symmetric_diff_pixels",
                "pred_area",
                "target_area",
                "area_diff",
                "iou",
                "dice",
            ]
        }
    ]
    write_csv(output_dir / "raster_sensitivity_variants.csv", variant_rows)
    write_csv(output_dir / "per_vertex_errors.csv", vertex_rows)
    write_csv(output_dir / "edge_length_errors.csv", edge_rows)
    write_csv(output_dir / "mask_diff_summary.csv", mask_summary)
    _write_summary(output_dir, variant_rows, vertex_rows, edge_rows)
    return {
        "variant_rows": variant_rows,
        "vertex_rows": vertex_rows,
        "edge_rows": edge_rows,
        "mask_summary": mask_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-csv", default=DEFAULT_PREDICTION_CSV)
    parser.add_argument("--mask-metrics-csv", default=DEFAULT_MASK_METRICS_CSV)
    parser.add_argument("--npz-path", default=DEFAULT_NPZ)
    parser.add_argument("--targets-path", default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args(argv)
    run_diagnostics(
        Path(args.prediction_csv),
        Path(args.mask_metrics_csv),
        Path(args.npz_path),
        Path(args.targets_path),
        Path(args.output_dir),
        args.sample_index,
    )
    print(f"Saved component-query raster sensitivity diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
