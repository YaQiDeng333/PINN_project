"""Rasterize COMSOL parametric component targets and score mask overlap."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


RAW_SCHEMA = [
    "center_x",
    "center_y",
    "axis_x",
    "axis_y",
    "depth_or_shape_param",
    "rotation_angle",
]


def _schema_list(values) -> list[str]:
    return [str(v) for v in values]


def _angle_to_degrees(angle: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(angle)
    if arr.size and np.nanmax(np.abs(arr)) <= 2 * np.pi + 1e-6:
        out = np.rad2deg(arr)
    else:
        out = arr
    return float(out) if np.isscalar(angle) else out


def continuous_to_raw(
    continuous: np.ndarray,
    target_schema,
    *,
    angle_encoding: str | None = None,
) -> np.ndarray:
    """Convert raw or sin/cos continuous targets to RAW_SCHEMA order."""

    schema = _schema_list(target_schema)
    raw = np.zeros((*continuous.shape[:2], len(RAW_SCHEMA)), dtype=np.float32)
    for name in ["center_x", "center_y", "axis_x", "axis_y", "depth_or_shape_param"]:
        if name not in schema:
            raise ValueError(f"target_schema missing required field: {name}")
        raw[:, :, RAW_SCHEMA.index(name)] = continuous[:, :, schema.index(name)]

    if "rotation_angle" in schema:
        raw[:, :, RAW_SCHEMA.index("rotation_angle")] = _angle_to_degrees(
            continuous[:, :, schema.index("rotation_angle")]
        )
    elif "rotation_sin" in schema and "rotation_cos" in schema:
        sin_v = continuous[:, :, schema.index("rotation_sin")]
        cos_v = continuous[:, :, schema.index("rotation_cos")]
        raw[:, :, RAW_SCHEMA.index("rotation_angle")] = np.rad2deg(np.arctan2(sin_v, cos_v))
    else:
        if angle_encoding == "sincos":
            raise ValueError("angle_encoding=sincos but rotation_sin/rotation_cos are absent.")
        raw[:, :, RAW_SCHEMA.index("rotation_angle")] = 0.0
    return raw


def rasterize_components(
    continuous: np.ndarray,
    type_targets: np.ndarray,
    presence: np.ndarray,
    target_schema,
    type_vocab,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Rasterize component union masks using full width/height axis semantics."""

    if continuous.ndim != 3:
        raise ValueError(f"continuous must have shape [N,K,P], got {continuous.shape}")
    if type_targets.shape != presence.shape or type_targets.shape != continuous.shape[:2]:
        raise ValueError("type_targets, presence and continuous component dimensions must align.")
    raw = continuous_to_raw(continuous, target_schema)
    type_vocab = _schema_list(type_vocab)
    grid_x, grid_y = np.meshgrid(x.astype(np.float32), y.astype(np.float32))
    masks = np.zeros((continuous.shape[0], len(y), len(x)), dtype=bool)
    for sample in range(continuous.shape[0]):
        for slot in range(continuous.shape[1]):
            if presence[sample, slot] <= 0.5:
                continue
            type_id = int(type_targets[sample, slot])
            if type_id < 0:
                continue
            if type_id >= len(type_vocab):
                raise ValueError(f"type id {type_id} is out of range for type_vocab={type_vocab}")
            component_type = type_vocab[type_id]
            if component_type not in {"rectangular_notch", "rotated_rect"}:
                raise ValueError(f"Unsupported component_type for rasterization: {component_type}")
            center_x, center_y, axis_x, axis_y, _depth, rotation_angle = raw[sample, slot]
            half_x = max(abs(float(axis_x)) * 0.5, 1e-12)
            half_y = max(abs(float(axis_y)) * 0.5, 1e-12)
            theta = np.deg2rad(float(rotation_angle))
            dx = grid_x - float(center_x)
            dy = grid_y - float(center_y)
            xr = np.cos(theta) * dx + np.sin(theta) * dy
            yr = -np.sin(theta) * dx + np.cos(theta) * dy
            masks[sample] |= (np.abs(xr) <= half_x) & (np.abs(yr) <= half_y)
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


def load_npz_and_targets(npz_path: Path, targets_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=True) as data:
        masks = data["masks"].astype(np.float32)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
    with np.load(targets_path, allow_pickle=True) as data:
        continuous = data["continuous_targets_raw"].astype(np.float32) if "continuous_targets_raw" in data else data["continuous_targets"].astype(np.float32)
        target_schema = data["raw_target_schema"] if "raw_target_schema" in data else data["target_schema"]
        type_targets = data["type_targets"].astype(np.int64)
        presence = data["presence_targets"].astype(np.float32)
        sample_indices = data["sample_indices"].astype(np.int64) if "sample_indices" in data else np.arange(continuous.shape[0])
        type_vocab = data["type_vocab"]
    if masks.shape[0] != continuous.shape[0]:
        raise ValueError("NPZ masks and parametric targets sample counts do not match.")
    return {
        "masks": masks,
        "x": x,
        "y": y,
        "continuous": continuous,
        "target_schema": target_schema,
        "type_targets": type_targets,
        "presence": presence,
        "sample_indices": sample_indices,
        "type_vocab": type_vocab,
    }


def _type_sequence(type_targets: np.ndarray, presence: np.ndarray, type_vocab) -> str:
    vocab = _schema_list(type_vocab)
    names = []
    for type_id, present in zip(type_targets, presence):
        if present > 0.5 and int(type_id) >= 0:
            names.append(vocab[int(type_id)])
    return "|".join(names)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> None:
    npz_path = Path(args.npz_path)
    targets_path = Path(args.parametric_targets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_npz_and_targets(npz_path, targets_path)
    pred_masks = rasterize_components(
        loaded["continuous"],
        loaded["type_targets"],
        loaded["presence"],
        loaded["target_schema"],
        loaded["type_vocab"],
        loaded["x"],
        loaded["y"],
    )
    ious, dices = mask_iou_dice(pred_masks, loaded["masks"])
    target_areas = (loaded["masks"] > 0.5).sum(axis=(1, 2))
    raster_areas = pred_masks.sum(axis=(1, 2))
    rows = []
    for i, sample_index in enumerate(loaded["sample_indices"]):
        rows.append(
            {
                "sample_index": int(sample_index),
                "oracle_mask_iou": float(ious[i]),
                "oracle_dice": float(dices[i]),
                "target_area": int(target_areas[i]),
                "raster_area": int(raster_areas[i]),
                "area_diff": int(raster_areas[i] - target_areas[i]),
                "component_count": int((loaded["presence"][i] > 0.5).sum()),
                "type_sequence": _type_sequence(loaded["type_targets"][i], loaded["presence"][i], loaded["type_vocab"]),
            }
        )

    dataset_name = output_dir.name
    aggregate = {
        "split_or_dataset": dataset_name,
        "samples": len(rows),
        "avg_oracle_iou": float(np.mean(ious)),
        "min_oracle_iou": float(np.min(ious)),
        "max_oracle_iou": float(np.max(ious)),
        "avg_oracle_dice": float(np.mean(dices)),
        "avg_target_area": float(np.mean(target_areas)),
        "avg_raster_area": float(np.mean(raster_areas)),
        "avg_abs_area_diff": float(np.mean(np.abs(raster_areas - target_areas))),
    }
    write_csv(output_dir / "oracle_parametric_mask_metrics.csv", rows)
    write_csv(output_dir / "oracle_parametric_mask_aggregate.csv", [aggregate])

    type_counts = Counter(row["type_sequence"] for row in rows)
    lines = [
        "# COMSOL parametric rasterization oracle summary",
        "",
        f"- npz_path: `{npz_path}`",
        f"- parametric_targets: `{targets_path}`",
        "- axis semantics: `axis_x` / `axis_y` are treated as full width / full height.",
        "- `rectangular_notch` and `rotated_rect` are both approximated as rotated rectangles.",
        f"- samples: `{aggregate['samples']}`",
        f"- avg_oracle_iou: `{aggregate['avg_oracle_iou']:.6e}`",
        f"- min_oracle_iou: `{aggregate['min_oracle_iou']:.6e}`",
        f"- max_oracle_iou: `{aggregate['max_oracle_iou']:.6e}`",
        f"- avg_oracle_dice: `{aggregate['avg_oracle_dice']:.6e}`",
        f"- avg_target_area: `{aggregate['avg_target_area']:.6e}`",
        f"- avg_raster_area: `{aggregate['avg_raster_area']:.6e}`",
        f"- avg_abs_area_diff: `{aggregate['avg_abs_area_diff']:.6e}`",
        "",
        "## Type sequences",
        "",
    ]
    lines.extend(f"- `{name}`: `{count}` samples" for name, count in sorted(type_counts.items()))
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            "- Gate passes when avg oracle IoU is at least 0.70 for train / val / test.",
            "- Low oracle IoU would indicate target schema, axis semantics or rasterizer mismatch before any model training.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved oracle rasterization metrics to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", default="")
    parser.add_argument("--parametric-targets", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-components", type=int, default=3)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.npz_path or not args.parametric_targets or not args.output_dir:
        parser.print_help()
        print("\nExample: python comsol_parametric_rasterizer.py --npz-path train.npz --parametric-targets parametric_targets.npz --output-dir out")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
