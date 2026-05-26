"""Offline oracle ablations for center-anchored polygon predictions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from comsol_polygon_rasterizer import mask_iou_dice, rasterize_polygon_components, write_csv


SPLITS = ("train", "val", "test")
VARIANTS = (
    "pred_all",
    "gt_center_bin",
    "gt_offset",
    "gt_center_bin_offset",
    "gt_local",
    "gt_center_bin_offset_local",
)


def _usage() -> str:
    return (
        "Usage: python center_anchored_polygon_oracle_ablation.py "
        "--prediction-dir run_dir --resplit-root experiments/dual_network/S299_comsol_polygon_matched_coverage_resplit "
        "--output-dir out --run-name current_reference"
    )


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_targets(split_dir: Path) -> dict:
    target_path = split_dir / "center_anchored_polygon_targets.npz"
    npz_path = split_dir / "comsol_v3_polygon_matched_coverage.npz"
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(target_path, allow_pickle=True) as data:
        targets = {
            "vertices": data["polygon_vertices_norm"].astype(np.float64),
            "vertex_mask": data["polygon_vertex_mask"].astype(np.float64),
            "presence": data["presence_targets"].astype(np.float64),
            "type_targets": data["type_targets"].astype(np.int64),
            "sample_indices": data["sample_indices"].astype(np.int64),
            "x_bins": data["center_x_bin_targets"].astype(np.int64),
            "y_bins": data["center_y_bin_targets"].astype(np.int64),
            "offsets": data["center_offset_targets"].astype(np.float64),
            "local_vertices": data["local_vertices_grid"].astype(np.float64),
            "x_centers": data["center_bin_x_centers"].astype(np.float64),
            "y_centers": data["center_bin_y_centers"].astype(np.float64),
            "bin_width_x": float(data["center_bin_width_x"]),
            "bin_width_y": float(data["center_bin_width_y"]),
            "grid_dx": float(data["grid_dx"]),
            "grid_dy": float(data["grid_dy"]),
            "x": data["x_norm"].astype(np.float64),
            "y": data["y_norm"].astype(np.float64),
        }
    with np.load(npz_path, allow_pickle=False) as data:
        targets["masks"] = data["masks"].astype(np.float64)
    return targets


def _rows_by_sample_slot(rows: Iterable[dict]) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for row in rows:
        key = (int(row["sample_index"]), int(row["component_slot"]))
        out[key] = row
    return out


def _as_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def _as_int(row: dict, key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def _prediction_arrays(prediction_csv: Path, targets: dict) -> dict:
    rows = _rows_by_sample_slot(_read_csv(prediction_csv))
    n, max_components, max_vertices, _coord = targets["vertices"].shape
    pred_vertices = np.zeros_like(targets["vertices"], dtype=np.float64)
    pred_local = np.zeros_like(targets["local_vertices"], dtype=np.float64)
    pred_presence = np.zeros((n, max_components), dtype=np.float64)
    pred_type = np.zeros((n, max_components), dtype=np.int64)
    pred_x_bins = np.zeros((n, max_components), dtype=np.int64)
    pred_y_bins = np.zeros((n, max_components), dtype=np.int64)
    true_x_bins = np.zeros((n, max_components), dtype=np.int64)
    true_y_bins = np.zeros((n, max_components), dtype=np.int64)
    pred_offsets = np.zeros((n, max_components, 2), dtype=np.float64)
    sample_to_row = {int(sample_index): row_idx for row_idx, sample_index in enumerate(targets["sample_indices"])}

    for sample_index, row_idx in sample_to_row.items():
        for slot in range(max_components):
            row = rows.get((sample_index, slot))
            if row is None:
                raise ValueError(f"Missing prediction row sample={sample_index} slot={slot}: {prediction_csv}")
            pred_presence[row_idx, slot] = _as_float(row, "presence_pred")
            pred_type[row_idx, slot] = _as_int(row, "type_pred")
            pred_x_bins[row_idx, slot] = _as_int(row, "center_x_bin_pred")
            pred_y_bins[row_idx, slot] = _as_int(row, "center_y_bin_pred")
            true_x_bins[row_idx, slot] = _as_int(row, "center_x_bin_true")
            true_y_bins[row_idx, slot] = _as_int(row, "center_y_bin_true")
            inferred_centers = []
            for vertex_idx in range(max_vertices):
                px = _as_float(row, f"pred_x{vertex_idx}")
                py = _as_float(row, f"pred_y{vertex_idx}")
                lx = _as_float(row, f"pred_local_x{vertex_idx}")
                ly = _as_float(row, f"pred_local_y{vertex_idx}")
                pred_vertices[row_idx, slot, vertex_idx] = [px, py]
                pred_local[row_idx, slot, vertex_idx] = [lx, ly]
                if _as_float(row, f"vertex{vertex_idx}_valid") > 0.5:
                    inferred_centers.append([px - lx * targets["grid_dx"], py - ly * targets["grid_dy"]])
            if inferred_centers:
                center = np.asarray(inferred_centers, dtype=np.float64).mean(axis=0)
                x_bin = np.clip(pred_x_bins[row_idx, slot], 0, len(targets["x_centers"]) - 1)
                y_bin = np.clip(pred_y_bins[row_idx, slot], 0, len(targets["y_centers"]) - 1)
                pred_offsets[row_idx, slot, 0] = (center[0] - targets["x_centers"][x_bin]) / targets["bin_width_x"]
                pred_offsets[row_idx, slot, 1] = (center[1] - targets["y_centers"][y_bin]) / targets["bin_width_y"]
    if not np.array_equal(true_x_bins, targets["x_bins"]) or not np.array_equal(true_y_bins, targets["y_bins"]):
        raise ValueError(f"Prediction true center-bin labels do not match targets: {prediction_csv}")
    return {
        "vertices": pred_vertices,
        "local": pred_local,
        "presence": pred_presence,
        "type": pred_type,
        "x_bins": pred_x_bins,
        "y_bins": pred_y_bins,
        "offsets": pred_offsets,
    }


def _decode_vertices(x_bins: np.ndarray, y_bins: np.ndarray, offsets: np.ndarray, local: np.ndarray, targets: dict) -> np.ndarray:
    x_idx = np.clip(x_bins.astype(np.int64), 0, len(targets["x_centers"]) - 1)
    y_idx = np.clip(y_bins.astype(np.int64), 0, len(targets["y_centers"]) - 1)
    center_x = targets["x_centers"][x_idx] + offsets[..., 0] * targets["bin_width_x"]
    center_y = targets["y_centers"][y_idx] + offsets[..., 1] * targets["bin_width_y"]
    vertices = np.zeros_like(local, dtype=np.float64)
    vertices[..., 0] = center_x[..., None] + local[..., 0] * targets["grid_dx"]
    vertices[..., 1] = center_y[..., None] + local[..., 1] * targets["grid_dy"]
    return vertices


def _variant_arrays(name: str, pred: dict, targets: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "pred_all":
        return pred["vertices"], np.ones_like(targets["vertex_mask"], dtype=np.float64), pred["presence"]
    if name == "gt_center_bin":
        vertices = _decode_vertices(targets["x_bins"], targets["y_bins"], pred["offsets"], pred["local"], targets)
        return vertices, np.ones_like(targets["vertex_mask"], dtype=np.float64), pred["presence"]
    if name == "gt_offset":
        vertices = _decode_vertices(pred["x_bins"], pred["y_bins"], targets["offsets"], pred["local"], targets)
        return vertices, np.ones_like(targets["vertex_mask"], dtype=np.float64), pred["presence"]
    if name == "gt_center_bin_offset":
        vertices = _decode_vertices(targets["x_bins"], targets["y_bins"], targets["offsets"], pred["local"], targets)
        return vertices, np.ones_like(targets["vertex_mask"], dtype=np.float64), pred["presence"]
    if name == "gt_local":
        vertices = _decode_vertices(pred["x_bins"], pred["y_bins"], pred["offsets"], targets["local_vertices"], targets)
        return vertices, targets["vertex_mask"], pred["presence"]
    if name == "gt_center_bin_offset_local":
        return targets["vertices"], targets["vertex_mask"], targets["presence"]
    raise ValueError(f"Unknown ablation variant: {name}")


def _summarize(values: list[dict]) -> dict:
    ious = np.asarray([float(row["polygon_iou"]) for row in values], dtype=np.float64)
    dices = np.asarray([float(row["polygon_dice"]) for row in values], dtype=np.float64)
    area_diffs = np.asarray([float(row["area_diff"]) for row in values], dtype=np.float64)
    return {
        "samples": len(values),
        "polygon_iou_mean": float(ious.mean()),
        "polygon_iou_min": float(ious.min()),
        "polygon_dice_mean": float(dices.mean()),
        "zero_iou_count": int((ious <= 0.0).sum()),
        "area_diff_mean": float(area_diffs.mean()),
        "area_abs_diff_mean": float(np.abs(area_diffs).mean()),
    }


def _component_rows(run_name: str, split: str, variant: str, pred: dict, targets: dict) -> list[dict]:
    rows = []
    present = targets["presence"] > 0.5
    true_local = targets["local_vertices"]
    pred_local = pred["local"]
    for row_idx, sample_index in enumerate(targets["sample_indices"]):
        for slot in range(targets["presence"].shape[1]):
            valid = targets["vertex_mask"][row_idx, slot] > 0.5
            local_mae = 0.0
            if valid.any():
                local_mae = float(np.abs(pred_local[row_idx, slot, valid] - true_local[row_idx, slot, valid]).mean())
            rows.append(
                {
                    "run": run_name,
                    "split": split,
                    "variant": variant,
                    "sample_index": int(sample_index),
                    "component_slot": int(slot),
                    "presence_true": float(targets["presence"][row_idx, slot]),
                    "presence_pred": float(pred["presence"][row_idx, slot]),
                    "type_true": int(targets["type_targets"][row_idx, slot]),
                    "type_pred": int(pred["type"][row_idx, slot]),
                    "true_center_x_bin": int(targets["x_bins"][row_idx, slot]),
                    "pred_center_x_bin": int(pred["x_bins"][row_idx, slot]),
                    "true_center_y_bin": int(targets["y_bins"][row_idx, slot]),
                    "pred_center_y_bin": int(pred["y_bins"][row_idx, slot]),
                    "x_bin_correct": int(pred["x_bins"][row_idx, slot] == targets["x_bins"][row_idx, slot]),
                    "y_bin_correct": int(pred["y_bins"][row_idx, slot] == targets["y_bins"][row_idx, slot]),
                    "both_bins_correct": int(
                        (pred["x_bins"][row_idx, slot] == targets["x_bins"][row_idx, slot])
                        and (pred["y_bins"][row_idx, slot] == targets["y_bins"][row_idx, slot])
                    ),
                    "center_offset_mae": float(
                        np.abs(pred["offsets"][row_idx, slot] - targets["offsets"][row_idx, slot]).mean()
                    ),
                    "local_vertex_mae_grid": local_mae if present[row_idx, slot] else 0.0,
                }
            )
    return rows


def run_ablation(prediction_dir: Path, resplit_root: Path, output_dir: Path, run_name: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample: list[dict] = []
    per_component: list[dict] = []
    summary_rows: list[dict] = []
    parity_rows: list[dict] = []

    for split in SPLITS:
        targets = _load_targets(resplit_root / split)
        pred = _prediction_arrays(prediction_dir / f"{split}_center_anchored_polygon_predictions.csv", targets)
        exported_mask_rows = {
            int(row["sample_index"]): row
            for row in _read_csv(prediction_dir / f"{split}_center_anchored_polygon_mask_metrics.csv")
        }
        for variant in VARIANTS:
            vertices, vertex_mask, presence = _variant_arrays(variant, pred, targets)
            masks = rasterize_polygon_components(vertices, vertex_mask, presence, targets["x"], targets["y"])
            ious, dices = mask_iou_dice(masks, targets["masks"])
            variant_rows: list[dict] = []
            for row_idx, sample_index in enumerate(targets["sample_indices"]):
                target_area = int((targets["masks"][row_idx] > 0.5).sum())
                pred_area = int(masks[row_idx].sum())
                row = {
                    "run": run_name,
                    "split": split,
                    "variant": variant,
                    "sample_index": int(sample_index),
                    "polygon_iou": float(ious[row_idx]),
                    "polygon_dice": float(dices[row_idx]),
                    "zero_iou": int(ious[row_idx] <= 0.0),
                    "target_area": target_area,
                    "pred_area": pred_area,
                    "area_diff": pred_area - target_area,
                }
                if variant == "pred_all":
                    exported = exported_mask_rows[int(sample_index)]
                    exported_iou = float(exported["polygon_mask_iou"])
                    row["exported_iou"] = exported_iou
                    row["exported_iou_abs_diff"] = abs(float(ious[row_idx]) - exported_iou)
                    row["exported_pred_area"] = int(float(exported["pred_area"]))
                    row["exported_pred_area_diff"] = pred_area - int(float(exported["pred_area"]))
                else:
                    row["exported_iou"] = ""
                    row["exported_iou_abs_diff"] = ""
                    row["exported_pred_area"] = ""
                    row["exported_pred_area_diff"] = ""
                per_sample.append(row)
                variant_rows.append(row)
            summary = _summarize(variant_rows)
            if variant == "pred_all":
                parity_rows.extend(variant_rows)
            present = targets["presence"] > 0.5
            x_acc = float((pred["x_bins"][present] == targets["x_bins"][present]).mean()) if present.any() else 1.0
            y_acc = float((pred["y_bins"][present] == targets["y_bins"][present]).mean()) if present.any() else 1.0
            summary_rows.append(
                {
                    "run": run_name,
                    "split": split,
                    "variant": variant,
                    **summary,
                    "pred_x_bin_acc": x_acc,
                    "pred_y_bin_acc": y_acc,
                }
            )
        per_component.extend(_component_rows(run_name, split, "pred_component_diagnostics", pred, targets))

    write_csv(output_dir / "ablation_per_sample.csv", per_sample)
    write_csv(output_dir / "ablation_per_component.csv", per_component)
    write_csv(output_dir / "ablation_summary.csv", summary_rows)
    _write_summary_md(output_dir / "summary.md", run_name, summary_rows, parity_rows)
    return {
        "summary_rows": summary_rows,
        "per_sample_rows": per_sample,
    }


def _find_summary(rows: list[dict], split: str, variant: str) -> dict:
    for row in rows:
        if row["split"] == split and row["variant"] == variant:
            return row
    raise KeyError((split, variant))


def _write_summary_md(path: Path, run_name: str, rows: list[dict], parity_rows: list[dict]) -> None:
    max_pred_iou_diff = max(float(row["exported_iou_abs_diff"]) for row in parity_rows)
    max_area_diff = max(abs(int(row["exported_pred_area_diff"])) for row in parity_rows)
    lines = [
        f"# Center-anchored oracle ablation: {run_name}",
        "",
        f"- pred_all max exported IoU diff: `{max_pred_iou_diff:.6e}`",
        f"- pred_all max exported pred_area diff: `{max_area_diff}`",
        "",
        "| split | variant | mean IoU | min IoU | zero-IoU | mean area abs diff |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        for variant in VARIANTS:
            row = _find_summary(rows, split, variant)
            lines.append(
                f"| {split} | {variant} | `{row['polygon_iou_mean']:.6f}` | "
                f"`{row['polygon_iou_min']:.6f}` | `{row['zero_iou_count']}` | "
                f"`{row['area_abs_diff_mean']:.6f}` |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir")
    parser.add_argument("--resplit-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name", default="run")
    args = parser.parse_args()
    if not args.prediction_dir or not args.resplit_root or not args.output_dir:
        print(_usage())
        return 0
    run_ablation(Path(args.prediction_dir), Path(args.resplit_root), Path(args.output_dir), args.run_name)
    print(f"Saved center-anchored oracle ablation to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
