"""Build center-anchored polygon targets for COMSOL V3 geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _usage() -> str:
    return (
        "Usage: python comsol_center_anchored_polygon_targets.py "
        "--npz-path data.npz --polygon-targets polygon_targets.npz "
        "--output-dir out [--center-bin-size-cells 8]"
    )


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


ORDERING_ALIASES = {
    "clockwise_top_left": "clockwise_top_left",
    "clockwise_start_min_y_then_min_x_in_normalized_space": "clockwise_top_left",
}


def _string_scalar(value) -> str:
    if isinstance(value, np.ndarray):
        return str(value.tolist())
    return str(value)


def normalize_vertex_ordering(value) -> str:
    text = _string_scalar(value)
    if text not in ORDERING_ALIASES:
        raise ValueError(f"Unsupported vertex_ordering: {value}")
    return ORDERING_ALIASES[text]


def mean_grid_spacing(values: np.ndarray, axis_name: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{axis_name} grid must be a one-dimensional array with at least two points.")
    diffs = np.diff(arr)
    if np.any(diffs <= 0):
        raise ValueError(f"{axis_name} grid must be strictly increasing.")
    return float(np.mean(diffs))


def build_center_bin_info(x: np.ndarray, y: np.ndarray, center_bin_size_cells: int) -> dict:
    if center_bin_size_cells <= 0:
        raise ValueError("center-bin-size-cells must be positive.")
    dx = mean_grid_spacing(x, "x")
    dy = mean_grid_spacing(y, "y")
    x_min = float(np.asarray(x, dtype=np.float64)[0])
    x_max = float(np.asarray(x, dtype=np.float64)[-1])
    y_min = float(np.asarray(y, dtype=np.float64)[0])
    y_max = float(np.asarray(y, dtype=np.float64)[-1])
    bin_width_x = float(center_bin_size_cells * dx)
    bin_width_y = float(center_bin_size_cells * dy)
    center_x_bins = int(np.ceil((x_max - x_min) / bin_width_x))
    center_y_bins = int(np.ceil((y_max - y_min) / bin_width_y))
    if center_x_bins <= 0 or center_y_bins <= 0:
        raise ValueError("center bin configuration produced no bins.")
    x_centers = (x_min + (np.arange(center_x_bins, dtype=np.float32) + 0.5) * bin_width_x).astype(np.float32)
    y_centers = (y_min + (np.arange(center_y_bins, dtype=np.float32) + 0.5) * bin_width_y).astype(np.float32)
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "dx": dx,
        "dy": dy,
        "bin_size_cells": int(center_bin_size_cells),
        "bin_width_x": bin_width_x,
        "bin_width_y": bin_width_y,
        "center_x_bins": center_x_bins,
        "center_y_bins": center_y_bins,
        "x_centers": x_centers,
        "y_centers": y_centers,
    }


def build_center_anchored_targets(polygon_targets: dict, x: np.ndarray, y: np.ndarray, center_bin_size_cells: int) -> dict:
    vertices = polygon_targets["polygon_vertices_norm"].astype(np.float32)
    vertex_mask = polygon_targets["polygon_vertex_mask"].astype(np.float32)
    presence = polygon_targets["presence_targets"].astype(np.float32)
    if vertices.ndim != 4 or vertices.shape[-1] != 2:
        raise ValueError(f"polygon_vertices_norm must have shape [N,K,V,2], got {vertices.shape}")
    if vertex_mask.shape != vertices.shape[:3]:
        raise ValueError("polygon_vertex_mask shape does not match polygon_vertices_norm.")
    if presence.shape != vertices.shape[:2]:
        raise ValueError("presence_targets shape does not match polygon_vertices_norm.")
    ordering = normalize_vertex_ordering(polygon_targets.get("vertex_ordering", "unknown"))
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    bin_info = build_center_bin_info(x, y, center_bin_size_cells)
    valid_weights = vertex_mask[..., None]
    denom = valid_weights.sum(axis=2).clip(min=1.0)
    centers = (vertices * valid_weights).sum(axis=2) / denom
    present = presence > 0.5
    center_x = centers[..., 0].astype(np.float64)
    center_y = centers[..., 1].astype(np.float64)
    if present.any():
        if np.any(center_x[present] < bin_info["x_min"]) or np.any(center_x[present] > bin_info["x_max"]):
            raise ValueError("center_x target is outside the x grid range.")
        if np.any(center_y[present] < bin_info["y_min"]) or np.any(center_y[present] > bin_info["y_max"]):
            raise ValueError("center_y target is outside the y grid range.")
    x_bin = np.floor((center_x - bin_info["x_min"]) / bin_info["bin_width_x"]).astype(np.int64)
    y_bin = np.floor((center_y - bin_info["y_min"]) / bin_info["bin_width_y"]).astype(np.int64)
    x_bin = np.clip(x_bin, 0, bin_info["center_x_bins"] - 1)
    y_bin = np.clip(y_bin, 0, bin_info["center_y_bins"] - 1)
    x_center = bin_info["x_centers"][x_bin]
    y_center = bin_info["y_centers"][y_bin]
    offsets = np.stack(
        [
            ((center_x - x_center) / bin_info["bin_width_x"]).astype(np.float32),
            ((center_y - y_center) / bin_info["bin_width_y"]).astype(np.float32),
        ],
        axis=-1,
    )
    if present.any():
        max_abs_offset = float(np.max(np.abs(offsets[present])))
        if max_abs_offset > 0.5001:
            raise ValueError(f"center bin offset target outside [-0.5, 0.5]: {max_abs_offset:.6f}")
    local = np.zeros_like(vertices, dtype=np.float32)
    local[..., 0] = (vertices[..., 0] - centers[..., None, 0]) / bin_info["dx"]
    local[..., 1] = (vertices[..., 1] - centers[..., None, 1]) / bin_info["dy"]
    local *= vertex_mask[..., None]
    absent = presence <= 0.5
    x_bin[absent] = 0
    y_bin[absent] = 0
    offsets[absent] = 0.0
    centers[absent] = 0.0
    local[absent] = 0.0
    out = dict(polygon_targets)
    out.update(
        {
            "center_x_bin_targets": x_bin.astype(np.int64),
            "center_y_bin_targets": y_bin.astype(np.int64),
            "center_offset_targets": offsets.astype(np.float32),
            "center_targets_norm": centers.astype(np.float32),
            "local_vertices_grid": local.astype(np.float32),
            "center_bin_x_centers": bin_info["x_centers"].astype(np.float32),
            "center_bin_y_centers": bin_info["y_centers"].astype(np.float32),
            "center_bin_width_x": np.array(bin_info["bin_width_x"], dtype=np.float32),
            "center_bin_width_y": np.array(bin_info["bin_width_y"], dtype=np.float32),
            "grid_dx": np.array(bin_info["dx"], dtype=np.float32),
            "grid_dy": np.array(bin_info["dy"], dtype=np.float32),
            "center_bin_size_cells": np.array(center_bin_size_cells, dtype=np.int64),
            "center_x_bins": np.array(bin_info["center_x_bins"], dtype=np.int64),
            "center_y_bins": np.array(bin_info["center_y_bins"], dtype=np.int64),
            "center_anchored_schema": np.array(
                ["center_x_bin", "center_y_bin", "center_offset_x", "center_offset_y", "local_vertices_grid"],
                dtype="U64",
            ),
            "vertex_ordering": np.array(ordering, dtype="U64"),
            "source_vertex_ordering": np.array(_string_scalar(polygon_targets.get("vertex_ordering", "unknown")), dtype="U128"),
            "center_bin_info_json": np.array(
                json.dumps(
                    {
                        key: value
                        for key, value in bin_info.items()
                        if key not in {"x_centers", "y_centers"}
                    },
                    sort_keys=True,
                ),
                dtype="U512",
            ),
        }
    )
    return out


def decode_center_anchored_vertices(targets: dict) -> np.ndarray:
    x_bin = targets["center_x_bin_targets"].astype(np.int64)
    y_bin = targets["center_y_bin_targets"].astype(np.int64)
    offsets = targets["center_offset_targets"].astype(np.float32)
    local = targets["local_vertices_grid"].astype(np.float32)
    x_centers = targets["center_bin_x_centers"].astype(np.float32)
    y_centers = targets["center_bin_y_centers"].astype(np.float32)
    center_x = x_centers[x_bin] + offsets[..., 0] * float(targets["center_bin_width_x"])
    center_y = y_centers[y_bin] + offsets[..., 1] * float(targets["center_bin_width_y"])
    vertices = np.zeros_like(local, dtype=np.float32)
    vertices[..., 0] = center_x[..., None] + local[..., 0] * float(targets["grid_dx"])
    vertices[..., 1] = center_y[..., None] + local[..., 1] * float(targets["grid_dy"])
    vertices *= targets["polygon_vertex_mask"][..., None].astype(np.float32)
    return vertices.astype(np.float32)


def write_summary(output_dir: Path, targets: dict) -> None:
    present = targets["presence_targets"] > 0.5
    offsets = targets["center_offset_targets"]
    local = targets["local_vertices_grid"]
    vertex_mask = (targets["presence_targets"][..., None] * targets["polygon_vertex_mask"]) > 0.5
    decoded = decode_center_anchored_vertices(targets)
    true_vertices = targets["polygon_vertices_norm"].astype(np.float32)
    max_decode_error = float(np.max(np.abs((decoded - true_vertices)[vertex_mask]))) if vertex_mask.any() else 0.0
    lines = [
        "# COMSOL center-anchored polygon target summary",
        "",
        f"- samples: `{targets['presence_targets'].shape[0]}`",
        f"- max_components: `{targets['presence_targets'].shape[1]}`",
        f"- max_vertices: `{targets['polygon_vertex_mask'].shape[2]}`",
        f"- center_bin_size_cells: `{int(targets['center_bin_size_cells'])}`",
        f"- center_x_bins: `{int(targets['center_x_bins'])}`",
        f"- center_y_bins: `{int(targets['center_y_bins'])}`",
        f"- grid_dx: `{float(targets['grid_dx']):.9e}`",
        f"- grid_dy: `{float(targets['grid_dy']):.9e}`",
        f"- present_components: `{int(present.sum())}`",
        f"- max_abs_center_offset: `{float(np.max(np.abs(offsets[present])) if present.any() else 0.0):.6f}`",
        f"- max_abs_local_vertex_grid: `{float(np.max(np.abs(local[vertex_mask])) if vertex_mask.any() else 0.0):.6f}`",
        f"- max_decode_abs_error: `{max_decode_error:.9e}`",
        "",
        "The primary training target is center-bin localization plus grid-cell local vertices.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz = _load_npz(Path(args.npz_path))
    polygon_targets = _load_npz(Path(args.polygon_targets))
    targets = build_center_anchored_targets(
        polygon_targets,
        npz["x"].astype(np.float32),
        npz["y"].astype(np.float32),
        args.center_bin_size_cells,
    )
    if int(npz["signals"].shape[0]) != int(targets["presence_targets"].shape[0]):
        raise ValueError("NPZ signals and center-anchored target sample counts do not match.")
    np.savez_compressed(output_dir / "center_anchored_polygon_targets.npz", **targets)
    write_summary(output_dir, targets)
    print(f"Saved center-anchored polygon targets to {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path")
    parser.add_argument("--polygon-targets")
    parser.add_argument("--output-dir")
    parser.add_argument("--center-bin-size-cells", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.npz_path or not args.polygon_targets or not args.output_dir:
        print(_usage())
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
