"""Hard polygon rasterization utilities for COMSOL V3 polygon targets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _usage() -> str:
    return (
        "Usage: python comsol_polygon_rasterizer.py --npz-path data.npz "
        "--polygon-targets polygon_targets.npz --output-dir out [--vertex-space norm|raw]"
    )


def _point_in_polygon(grid_x: np.ndarray, grid_y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Return mask for points inside or on polygon boundary."""

    x = grid_x.astype(np.float64)
    y = grid_y.astype(np.float64)
    verts = vertices.astype(np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    on_edge = np.zeros(x.shape, dtype=bool)
    eps = 1e-12
    n = verts.shape[0]
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        within_x = (np.minimum(x1, x2) - eps <= x) & (x <= np.maximum(x1, x2) + eps)
        within_y = (np.minimum(y1, y2) - eps <= y) & (y <= np.maximum(y1, y2) + eps)
        on_edge |= (np.abs(cross) <= eps) & within_x & within_y
        intersects = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) + eps) + x1
        )
        inside ^= intersects
    return inside | on_edge


def rasterize_polygon_components(
    polygon_vertices: np.ndarray,
    polygon_vertex_mask: np.ndarray,
    presence: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Rasterize polygon component union masks."""

    if polygon_vertices.ndim != 4 or polygon_vertices.shape[-1] != 2:
        raise ValueError("polygon_vertices must have shape [N,K,V,2].")
    if polygon_vertex_mask.shape != polygon_vertices.shape[:3]:
        raise ValueError("polygon_vertex_mask must have shape [N,K,V].")
    if presence.shape != polygon_vertices.shape[:2]:
        raise ValueError("presence must have shape [N,K].")
    grid_x, grid_y = np.meshgrid(x.astype(np.float64), y.astype(np.float64))
    masks = np.zeros((polygon_vertices.shape[0], len(y), len(x)), dtype=bool)
    for sample in range(polygon_vertices.shape[0]):
        for slot in range(polygon_vertices.shape[1]):
            if presence[sample, slot] <= 0.5:
                continue
            valid = polygon_vertex_mask[sample, slot] > 0.5
            vertices = polygon_vertices[sample, slot, valid]
            if vertices.shape[0] < 3:
                raise ValueError(f"sample={sample} slot={slot} has fewer than 3 valid vertices.")
            masks[sample] |= _point_in_polygon(grid_x, grid_y, vertices)
    return masks


def mask_iou_dice(pred_masks: np.ndarray, true_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = true_masks > 0.5
    pred = pred_masks > 0.5
    ious = np.zeros(pred.shape[0], dtype=np.float64)
    dices = np.zeros(pred.shape[0], dtype=np.float64)
    for i in range(pred.shape[0]):
        intersection = np.logical_and(pred[i], true[i]).sum()
        union = np.logical_or(pred[i], true[i]).sum()
        denom = pred[i].sum() + true[i].sum()
        ious[i] = intersection / union if union else 1.0
        dices[i] = 2.0 * intersection / denom if denom else 1.0
    return ious, dices


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.npz_path, allow_pickle=True) as data:
        masks = data["masks"].astype(np.float32)
        raw_x = data["x"].astype(np.float32)
        raw_y = data["y"].astype(np.float32)
    with np.load(args.polygon_targets, allow_pickle=True) as data:
        if args.vertex_space == "norm":
            vertices = data["polygon_vertices_norm"].astype(np.float32)
            x = data["x_norm"].astype(np.float32)
            y = data["y_norm"].astype(np.float32)
        else:
            vertices = data["polygon_vertices_raw"].astype(np.float32)
            x = raw_x
            y = raw_y
        vertex_mask = data["polygon_vertex_mask"].astype(np.float32)
        presence = data["presence_targets"].astype(np.float32)
        sample_indices = data["sample_indices"].astype(np.int64)
    if masks.shape[0] != vertices.shape[0]:
        raise ValueError("NPZ masks and polygon target sample counts do not match.")
    pred = rasterize_polygon_components(vertices, vertex_mask, presence, x, y)
    ious, dices = mask_iou_dice(pred, masks)
    rows = []
    for i, sample_index in enumerate(sample_indices):
        rows.append(
            {
                "sample_index": int(sample_index),
                "polygon_iou": float(ious[i]),
                "polygon_dice": float(dices[i]),
                "target_area": int((masks[i] > 0.5).sum()),
                "raster_area": int(pred[i].sum()),
                "area_diff": int(pred[i].sum() - (masks[i] > 0.5).sum()),
            }
        )
    write_csv(output_dir / "polygon_oracle_metrics.csv", rows)
    summary = [
        "# COMSOL polygon rasterization oracle summary",
        "",
        f"- vertex_space: `{args.vertex_space}`",
        f"- samples: `{len(rows)}`",
        f"- mean IoU: `{float(np.mean(ious)):.6f}`",
        f"- min IoU: `{float(np.min(ious)):.6f}`",
        f"- max IoU: `{float(np.max(ious)):.6f}`",
        f"- mean Dice: `{float(np.mean(dices)):.6f}`",
        "",
        "Gate guidance: true COMSOL polygon smoke should have every sample IoU >= 0.95.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Saved polygon oracle metrics to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--npz-path")
    parser.add_argument("--polygon-targets")
    parser.add_argument("--output-dir")
    parser.add_argument("--vertex-space", choices=["norm", "raw"], default="norm")
    args = parser.parse_args()
    if not args.npz_path or not args.polygon_targets or not args.output_dir:
        print(_usage())
        return
    run(args)


if __name__ == "__main__":
    main()
