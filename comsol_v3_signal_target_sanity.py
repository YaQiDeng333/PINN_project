"""Read-only sanity checks for normalized COMSOL V3 signal/target alignment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")
RAW_FIELDS = [
    "center_x",
    "center_y",
    "axis_x",
    "axis_y",
    "depth_or_shape_param",
    "rotation_angle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose normalized V3 signal/target sanity.")
    parser.add_argument("--npz-root", help="Root containing split converted NPZ files.")
    parser.add_argument("--targets-root", help="Root containing split/parametric_targets.npz files.")
    parser.add_argument("--defect-root", help="Root containing split/defect_params.csv files.")
    parser.add_argument("--output-dir", help="Directory for CSV and summary outputs.")
    parser.add_argument("--center-bin-size-cells", type=int, default=8)
    parser.add_argument("--std-floor", type=float, default=1e-8)
    return parser.parse_args()


def _usage_and_exit() -> int:
    print(
        "Usage: python comsol_v3_signal_target_sanity.py "
        "--npz-root NPZ_ROOT --targets-root TARGETS_ROOT --defect-root DEFECT_ROOT --output-dir OUTPUT_DIR"
    )
    return 0


def _find_split_npz(root: Path, split: str) -> Path:
    candidates = sorted(root.glob(f"{split}_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No NPZ found for split {split} under {root}")
    return candidates[0]


def _schema(values) -> list[str]:
    return [str(v) for v in values]


def _mean_spacing(values: np.ndarray, name: str) -> float:
    diffs = np.diff(values.astype(np.float64))
    if diffs.size == 0 or not np.all(np.isfinite(diffs)):
        raise ValueError(f"{name} grid spacing is invalid.")
    if np.any(diffs <= 0):
        raise ValueError(f"{name} grid must be strictly increasing.")
    return float(np.mean(diffs))


def _bbox(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    yy, xx = np.where(mask > 0.5)
    if len(xx) == 0:
        return {
            "bbox_center_x": float("nan"),
            "bbox_center_y": float("nan"),
            "bbox_axis_x": float("nan"),
            "bbox_axis_y": float("nan"),
            "mask_area": 0.0,
        }
    x_vals = x[xx].astype(np.float64)
    y_vals = y[yy].astype(np.float64)
    return {
        "bbox_center_x": float(0.5 * (x_vals.min() + x_vals.max())),
        "bbox_center_y": float(0.5 * (y_vals.min() + y_vals.max())),
        "bbox_axis_x": float(x_vals.max() - x_vals.min()),
        "bbox_axis_y": float(y_vals.max() - y_vals.min()),
        "mask_area": float(len(xx)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _signal_energy_center_x(signals: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    centers = []
    for sample in signals:
        centered = sample.astype(np.float64) - sample.astype(np.float64).mean(axis=1, keepdims=True)
        energy = np.abs(centered).sum(axis=0)
        total = float(energy.sum())
        centers.append(float((energy * x).sum() / total) if total > 0 else float("nan"))
    return np.asarray(centers, dtype=np.float64)


def _center_bin_stats(x: np.ndarray, y: np.ndarray, targets: dict, center_bin_size_cells: int) -> dict:
    dx = _mean_spacing(x, "x")
    dy = _mean_spacing(y, "y")
    x_min = float(x[0])
    x_max = float(x[-1])
    y_min = float(y[0])
    y_max = float(y[-1])
    bin_width_x = dx * center_bin_size_cells
    bin_width_y = dy * center_bin_size_cells
    x_bins = int(np.ceil((x_max - x_min) / bin_width_x))
    y_bins = int(np.ceil((y_max - y_min) / bin_width_y))
    schema = targets["schema"]
    cx = schema.index("center_x")
    cy = schema.index("center_y")
    centers = targets["continuous"]
    presence = targets["presence"] > 0.5
    center_x = centers[:, :, cx].astype(np.float64)
    center_y = centers[:, :, cy].astype(np.float64)
    in_range = True
    if presence.any():
        in_range = bool(
            np.all(center_x[presence] >= x_min)
            and np.all(center_x[presence] <= x_max)
            and np.all(center_y[presence] >= y_min)
            and np.all(center_y[presence] <= y_max)
        )
    x_bin = np.clip(np.floor((center_x - x_min) / bin_width_x).astype(np.int64), 0, x_bins - 1)
    y_bin = np.clip(np.floor((center_y - y_min) / bin_width_y).astype(np.int64), 0, y_bins - 1)
    x_centers = x_min + (np.arange(x_bins, dtype=np.float64) + 0.5) * bin_width_x
    y_centers = y_min + (np.arange(y_bins, dtype=np.float64) + 0.5) * bin_width_y
    x_offset = (center_x - x_centers[x_bin]) / bin_width_x
    y_offset = (center_y - y_centers[y_bin]) / bin_width_y
    if not presence.any():
        offset_min = offset_max = float("nan")
    else:
        offsets = np.concatenate([x_offset[presence], y_offset[presence]])
        offset_min = float(np.min(offsets))
        offset_max = float(np.max(offsets))
    return {
        "center_x_bins": x_bins,
        "center_y_bins": y_bins,
        "center_bin_targets_in_range": in_range,
        "center_bin_offset_min": offset_min,
        "center_bin_offset_max": offset_max,
        "center_x_bin_min": int(x_bin[presence].min()) if presence.any() else -1,
        "center_x_bin_max": int(x_bin[presence].max()) if presence.any() else -1,
        "center_y_bin_min": int(y_bin[presence].min()) if presence.any() else -1,
        "center_y_bin_max": int(y_bin[presence].max()) if presence.any() else -1,
        "max_abs_offset": float(max(abs(offset_min), abs(offset_max))) if np.isfinite(offset_min) else float("nan"),
    }


def _target_vs_defects(
    targets: dict,
    defects: pd.DataFrame,
) -> dict[str, float]:
    schema = targets["schema"]
    continuous = targets["continuous"]
    presence = targets["presence"] > 0.5
    diffs: dict[str, float] = {}
    field_map = {
        "center_x": "defect_center_x",
        "center_y": "defect_center_y",
        "axis_x": "defect_axis_x",
        "axis_y": "defect_axis_y",
        "depth_or_shape_param": "defect_depth_or_shape_param",
        "rotation_angle": "rotation_angle",
    }
    for field, defect_col in field_map.items():
        if field not in schema or defect_col not in defects.columns:
            continue
        values = []
        for i in range(continuous.shape[0]):
            slots = np.where(presence[i])[0]
            if len(slots) == 0:
                continue
            values.append(abs(float(continuous[i, slots[0], schema.index(field)]) - float(defects.iloc[i][defect_col])))
        diffs[f"max_abs_target_vs_defect_{field}"] = float(max(values)) if values else float("nan")
    return diffs


def analyze_split(
    split: str,
    npz_path: Path,
    targets_path: Path,
    defect_path: Path,
    center_bin_size_cells: int,
    std_floor: float,
) -> tuple[list[dict], dict]:
    with np.load(npz_path, allow_pickle=True) as data:
        signals = data["signals"].astype(np.float64)
        masks = data["masks"].astype(np.float32)
        mu_maps = data["mu_maps"].astype(np.float32)
        x = data["x"].astype(np.float64)
        y = data["y"].astype(np.float64)
        source_sample_ids = data["source_sample_ids"] if "source_sample_ids" in data else np.arange(signals.shape[0])
        source_global_indices = data["source_global_indices"] if "source_global_indices" in data else np.arange(signals.shape[0])
        csv_sample_indices = data["csv_sample_indices"] if "csv_sample_indices" in data else np.arange(signals.shape[0])
    with np.load(targets_path, allow_pickle=True) as data:
        target_schema = _schema(data["raw_target_schema"] if "raw_target_schema" in data else data["target_schema"])
        targets = {
            "continuous": (
                data["continuous_targets_raw"].astype(np.float64)
                if "continuous_targets_raw" in data
                else data["continuous_targets"].astype(np.float64)
            ),
            "presence": data["presence_targets"].astype(np.float32),
            "sample_indices": data["sample_indices"].astype(np.int64),
            "schema": target_schema,
        }
    defects = pd.read_csv(defect_path)
    if len(defects) != signals.shape[0]:
        raise ValueError(f"{defect_path} rows do not match signals count.")

    flat = signals.reshape(signals.shape[0], -1)
    means = flat.mean(axis=1)
    stds = flat.std(axis=1)
    mins = flat.min(axis=1)
    maxs = flat.max(axis=1)
    ranges = maxs - mins
    peak_abs = np.max(np.abs(flat), axis=1)
    floor_triggered = stds < std_floor
    after_floor_std = np.where(floor_triggered, 1.0, stds)
    zscore_norm = np.linalg.norm((flat - means[:, None]) / after_floor_std[:, None], axis=1)
    energy_center_x = _signal_energy_center_x(signals, x)

    dx = _mean_spacing(x, "x")
    dy = _mean_spacing(y, "y")
    rows = []
    threshold_mismatch = (masks > 0.5) != (mu_maps < 500)
    bbox_center_x_errors = []
    bbox_center_y_errors = []
    bbox_axis_x_errors = []
    bbox_axis_y_errors = []
    for i in range(signals.shape[0]):
        bbox = _bbox(masks[i], x, y)
        defect = defects.iloc[i]
        cx_err_cells = abs(bbox["bbox_center_x"] - float(defect["defect_center_x"])) / dx
        cy_err_cells = abs(bbox["bbox_center_y"] - float(defect["defect_center_y"])) / dy
        ax_err_cells = abs(bbox["bbox_axis_x"] - float(defect["defect_axis_x"])) / dx
        ay_err_cells = abs(bbox["bbox_axis_y"] - float(defect["defect_axis_y"])) / dy
        bbox_center_x_errors.append(cx_err_cells)
        bbox_center_y_errors.append(cy_err_cells)
        bbox_axis_x_errors.append(ax_err_cells)
        bbox_axis_y_errors.append(ay_err_cells)
        rows.append(
            {
                "split": split,
                "sample_pos": i,
                "sample_index": int(targets["sample_indices"][i]),
                "csv_sample_index": int(csv_sample_indices[i]),
                "source_sample_id": str(source_sample_ids[i]),
                "source_global_index": int(source_global_indices[i]),
                "hard_case_type": str(defect.get("hard_case_type", "")),
                "signal_mean": float(means[i]),
                "signal_std": float(stds[i]),
                "signal_min": float(mins[i]),
                "signal_max": float(maxs[i]),
                "signal_range": float(ranges[i]),
                "signal_peak_abs": float(peak_abs[i]),
                "std_floor_triggered": bool(floor_triggered[i]),
                "zscore_input_norm_after_runner_floor": float(zscore_norm[i]),
                "signal_energy_center_x": float(energy_center_x[i]),
                "defect_center_x": float(defect["defect_center_x"]),
                "defect_center_y": float(defect["defect_center_y"]),
                "bbox_center_x": bbox["bbox_center_x"],
                "bbox_center_y": bbox["bbox_center_y"],
                "bbox_center_x_error_cells": float(cx_err_cells),
                "bbox_center_y_error_cells": float(cy_err_cells),
                "bbox_axis_x_error_cells": float(ax_err_cells),
                "bbox_axis_y_error_cells": float(ay_err_cells),
                "mask_area": bbox["mask_area"],
            }
        )
    bin_stats = _center_bin_stats(x, y, targets, center_bin_size_cells)
    target_diffs = _target_vs_defects(targets, defects)
    summary = {
        "split": split,
        "samples": int(signals.shape[0]),
        "signals_shape": "x".join(str(v) for v in signals.shape),
        "signal_std_min": float(stds.min()),
        "signal_std_max": float(stds.max()),
        "signal_std_mean": float(stds.mean()),
        "std_floor": float(std_floor),
        "std_floor_trigger_rate": float(np.mean(floor_triggered)),
        "all_samples_trigger_std_floor": bool(np.all(floor_triggered)),
        "signal_range_mean": float(ranges.mean()),
        "signal_peak_abs_min": float(peak_abs.min()),
        "signal_peak_abs_max": float(peak_abs.max()),
        "unique_signal_rows": int(np.unique(flat, axis=0).shape[0]),
        "mask_mu_threshold_mismatch_pixels": int(threshold_mismatch.sum()),
        "mask_mu_threshold_mismatch_rate": float(threshold_mismatch.mean()),
        "max_bbox_center_x_error_cells": float(np.nanmax(bbox_center_x_errors)),
        "max_bbox_center_y_error_cells": float(np.nanmax(bbox_center_y_errors)),
        "max_bbox_axis_x_error_cells": float(np.nanmax(bbox_axis_x_errors)),
        "max_bbox_axis_y_error_cells": float(np.nanmax(bbox_axis_y_errors)),
        "energy_center_x_vs_defect_center_x_corr": _corr(energy_center_x, defects["defect_center_x"].to_numpy()),
        **bin_stats,
        **target_diffs,
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summaries: list[dict]) -> None:
    train_summary = next((row for row in summaries if row["split"] == "train"), None)
    train_all_floor = bool(train_summary and train_summary["all_samples_trigger_std_floor"])
    all_floor = all(bool(row["all_samples_trigger_std_floor"]) for row in summaries)
    lines = [
        "# S227 normalized V3 signal-target sanity",
        "",
        "S227 checks normalized V3 data before any tiny-overfit run.",
        "",
        "## Split Summary",
        "",
        "| split | samples | signal_std_min | signal_std_max | floor_rate | mask_threshold_mismatch | max_bbox_center_error_cells | center_bin_in_range | offset_min | offset_max |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in summaries:
        max_center = max(row["max_bbox_center_x_error_cells"], row["max_bbox_center_y_error_cells"])
        lines.append(
            f"| {row['split']} | {row['samples']} | `{row['signal_std_min']:.6e}` | "
            f"`{row['signal_std_max']:.6e}` | `{row['std_floor_trigger_rate']:.6f}` | "
            f"`{row['mask_mu_threshold_mismatch_pixels']}` | `{max_center:.6f}` | "
            f"`{row['center_bin_targets_in_range']}` | `{row['center_bin_offset_min']:.6f}` | "
            f"`{row['center_bin_offset_max']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if train_all_floor:
        prefix = "All splits trigger" if all_floor else "The train split triggers"
        lines.append(
            f"{prefix} the runner `std < 1e-8` signal floor for every sample. "
            "Tiny-overfit training is therefore skipped; the next check should target COMSOL signal export / "
            "probe height / field expression / signal scaling / runner normalization floor."
        )
    else:
        lines.append(
            "At least one split has samples above the runner signal std floor. Tiny-overfit can proceed."
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Mask threshold alignment, bbox/defect alignment, and center-bin target ranges are reported separately from signal scale.",
            "- This diagnostic does not train and does not modify the normalized V3 data.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.npz_root or not args.targets_root or not args.defect_root or not args.output_dir:
        return _usage_and_exit()
    npz_root = Path(args.npz_root)
    targets_root = Path(args.targets_root)
    defect_root = Path(args.defect_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for split in SPLITS:
        rows, summary = analyze_split(
            split,
            _find_split_npz(npz_root, split),
            targets_root / split / "parametric_targets.npz",
            defect_root / split / "defect_params.csv",
            args.center_bin_size_cells,
            args.std_floor,
        )
        all_rows.extend(rows)
        summaries.append(summary)
    _write_csv(output_dir / "per_sample_signal_target_sanity.csv", all_rows)
    _write_csv(output_dir / "split_signal_target_sanity.csv", summaries)
    (output_dir / "sanity_stats.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    _write_summary(output_dir / "summary.md", summaries)
    print(f"Saved normalized V3 signal-target sanity diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
