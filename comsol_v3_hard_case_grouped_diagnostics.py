"""Group COMSOL V3 hard-case prediction exports by hard_case_type.

This script is read-only: it consumes existing prediction CSVs, mask metrics,
defect parameter CSVs, and converted NPZ grids. It does not train or write model
artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group V3 hard-case prediction exports by hard_case_type.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec as label=prediction_dir. Can be repeated.",
    )
    parser.add_argument(
        "--defect-root",
        default="experiments/dual_network/S208_comsol_v3_hard_case_ingest/raw",
        help="Root containing split defect_params.csv files.",
    )
    parser.add_argument(
        "--npz-root",
        default="experiments/dual_network/S208_comsol_v3_hard_case_ingest/converted",
        help="Root containing converted split NPZ files.",
    )
    parser.add_argument("--output-dir", help="Directory for diagnostic CSV and summary outputs.")
    parser.add_argument("--center-bin-size-cells", type=int, default=8)
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to diagnose. Use val,test for zero-shot runs trained on non-V3 data.",
    )
    return parser.parse_args()


def _usage_and_exit() -> int:
    print(
        "Usage: python comsol_v3_hard_case_grouped_diagnostics.py "
        "--run label=prediction_dir --output-dir OUTPUT_DIR"
    )
    return 0


def parse_run_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Run spec must be label=path, got: {spec}")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Run label is empty in spec: {spec}")
        parsed.append((label, Path(path)))
    return parsed


def parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(splits) - set(SPLITS))
    if invalid:
        raise ValueError(f"Unsupported split(s): {invalid}; expected subset of {SPLITS}")
    if not splits:
        raise ValueError("At least one split is required.")
    return splits


def split_npz_path(npz_root: Path, split: str) -> Path:
    candidates = sorted(npz_root.glob(f"{split}_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No converted NPZ found for split {split} under {npz_root}")
    return candidates[0]


def bin_index(values: np.ndarray, min_value: float, bin_width: float) -> np.ndarray:
    return np.floor((values - min_value) / bin_width).astype(np.int64)


def bin_offset(values: np.ndarray, bins: np.ndarray, min_value: float, bin_width: float) -> np.ndarray:
    centers = min_value + (bins.astype(np.float64) + 0.5) * bin_width
    return (values.astype(np.float64) - centers) / bin_width


def load_split_frame(
    label: str,
    run_dir: Path,
    split: str,
    defect_root: Path,
    npz_root: Path,
    center_bin_size_cells: int,
) -> pd.DataFrame | None:
    pred_path = run_dir / f"{split}_predictions.csv"
    metric_path = run_dir / f"{split}_prediction_mask_metrics.csv"
    defect_path = defect_root / split / "defect_params.csv"
    if not pred_path.exists() or not metric_path.exists() or not defect_path.exists():
        return None

    pred = pd.read_csv(pred_path)
    metrics = pd.read_csv(metric_path)
    defects = pd.read_csv(defect_path)
    npz = np.load(split_npz_path(npz_root, split), allow_pickle=True)
    x = np.asarray(npz["x"], dtype=np.float64)
    y = np.asarray(npz["y"], dtype=np.float64)
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    bin_width_x = dx * center_bin_size_cells
    bin_width_y = dy * center_bin_size_cells

    required_pred = {
        "sample_index",
        "component_slot",
        "presence_true",
        "center_x_true",
        "center_x_pred",
        "center_y_true",
        "center_y_pred",
    }
    missing = sorted(required_pred - set(pred.columns))
    if missing:
        raise ValueError(f"{pred_path} missing columns: {missing}")
    if "hard_case_type" not in defects.columns:
        raise ValueError(f"{defect_path} missing hard_case_type")
    if "pred_mask_iou" not in metrics.columns:
        raise ValueError(f"{metric_path} missing pred_mask_iou")

    pred = pred[pred["presence_true"].astype(float) > 0.5].copy()
    pred["run_label"] = label
    pred["split"] = split
    pred["center_x_bin_true_derived"] = bin_index(pred["center_x_true"].to_numpy(), float(x.min()), bin_width_x)
    pred["center_x_bin_pred_derived"] = bin_index(pred["center_x_pred"].to_numpy(), float(x.min()), bin_width_x)
    pred["center_y_bin_true_derived"] = bin_index(pred["center_y_true"].to_numpy(), float(y.min()), bin_width_y)
    pred["center_y_bin_pred_derived"] = bin_index(pred["center_y_pred"].to_numpy(), float(y.min()), bin_width_y)
    pred["x_bin_correct"] = pred["center_x_bin_true_derived"] == pred["center_x_bin_pred_derived"]
    pred["y_bin_correct"] = pred["center_y_bin_true_derived"] == pred["center_y_bin_pred_derived"]
    pred["both_bins_correct"] = pred["x_bin_correct"] & pred["y_bin_correct"]
    pred["center_x_grid_error"] = np.abs(pred["center_x_pred"] - pred["center_x_true"]) / abs(dx)
    pred["center_y_grid_error"] = np.abs(pred["center_y_pred"] - pred["center_y_true"]) / abs(dy)
    pred["center_grid_error"] = np.sqrt(pred["center_x_grid_error"] ** 2 + pred["center_y_grid_error"] ** 2)
    true_x_offset = bin_offset(
        pred["center_x_true"].to_numpy(),
        pred["center_x_bin_true_derived"].to_numpy(),
        float(x.min()),
        bin_width_x,
    )
    pred_x_offset = bin_offset(
        pred["center_x_pred"].to_numpy(),
        pred["center_x_bin_pred_derived"].to_numpy(),
        float(x.min()),
        bin_width_x,
    )
    true_y_offset = bin_offset(
        pred["center_y_true"].to_numpy(),
        pred["center_y_bin_true_derived"].to_numpy(),
        float(y.min()),
        bin_width_y,
    )
    pred_y_offset = bin_offset(
        pred["center_y_pred"].to_numpy(),
        pred["center_y_bin_pred_derived"].to_numpy(),
        float(y.min()),
        bin_width_y,
    )
    pred["center_offset_x_abs_error"] = np.abs(pred_x_offset - true_x_offset)
    pred["center_offset_y_abs_error"] = np.abs(pred_y_offset - true_y_offset)
    pred["center_offset_mae"] = 0.5 * (
        pred["center_offset_x_abs_error"] + pred["center_offset_y_abs_error"]
    )

    sample_cols = [
        "sample_index",
        "pred_mask_iou",
        "pred_area",
        "target_area",
        "area_diff",
    ]
    joined = pred.merge(metrics[sample_cols], on="sample_index", how="left")
    defect_cols = ["sample_index", "hard_case_type"]
    for optional in ("defect_type", "rotation_angle", "component_type_combination"):
        if optional in defects.columns:
            defect_cols.append(optional)
    joined = joined.merge(defects[defect_cols], on="sample_index", how="left")
    return joined


def summarize(frames: list[pd.DataFrame], output_dir: Path) -> None:
    all_components = pd.concat(frames, ignore_index=True)
    all_components.to_csv(output_dir / "per_component_v3_hard_case_diagnostics.csv", index=False)

    sample = (
        all_components.groupby(["run_label", "split", "sample_index", "hard_case_type"], dropna=False)
        .agg(
            mask_iou=("pred_mask_iou", "first"),
            center_grid_mae=("center_grid_error", "mean"),
            max_center_grid_error=("center_grid_error", "max"),
            x_bin_acc=("x_bin_correct", "mean"),
            y_bin_acc=("y_bin_correct", "mean"),
            both_bin_acc=("both_bins_correct", "mean"),
            center_offset_mae=("center_offset_mae", "mean"),
            pred_area=("pred_area", "first"),
            target_area=("target_area", "first"),
            area_diff=("area_diff", "first"),
        )
        .reset_index()
    )
    sample.to_csv(output_dir / "per_sample_v3_hard_case_diagnostics.csv", index=False)

    grouped = (
        sample.groupby(["run_label", "split", "hard_case_type"], dropna=False)
        .agg(
            sample_count=("sample_index", "count"),
            mask_iou=("mask_iou", "mean"),
            center_grid_mae=("center_grid_mae", "mean"),
            x_bin_acc=("x_bin_acc", "mean"),
            y_bin_acc=("y_bin_acc", "mean"),
            center_offset_mae=("center_offset_mae", "mean"),
            pred_area=("pred_area", "mean"),
            target_area=("target_area", "mean"),
        )
        .reset_index()
    )
    grouped.to_csv(output_dir / "grouped_by_hard_case_type.csv", index=False)

    worst = sample.sort_values(["mask_iou", "center_grid_mae"], ascending=[True, False]).head(30)
    worst.to_csv(output_dir / "worst_v3_samples.csv", index=False)

    lines = [
        "# S215 V3 hard-case grouped diagnostics",
        "",
        "This summary is generated from existing prediction exports and defect parameters only.",
        "",
        "## Runs",
        "",
    ]
    for label in sorted(sample["run_label"].unique()):
        sub = sample[sample["run_label"] == label]
        lines.append(
            f"- `{label}`: splits={','.join(sorted(sub['split'].unique()))}, "
            f"mean IoU={sub['mask_iou'].mean():.6f}, "
            f"mean center_grid_mae={sub['center_grid_mae'].mean():.6f}"
        )
    lines.extend(["", "## Hardest groups", ""])
    for (label, split), sub in grouped.groupby(["run_label", "split"]):
        hardest = sub.sort_values("mask_iou").iloc[0]
        lines.append(
            f"- `{label}` `{split}` hardest: `{hardest['hard_case_type']}` "
            f"(count={int(hardest['sample_count'])}, IoU={hardest['mask_iou']:.6f}, "
            f"center_grid_mae={hardest['center_grid_mae']:.6f}, "
            f"x_bin_acc={hardest['x_bin_acc']:.6f}, y_bin_acc={hardest['y_bin_acc']:.6f})"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The script reports grouped evidence; the stage-level conclusion should compare these tables with S213/S214 summaries.",
            "- `center_offset_mae` is derived from decoded center coordinates and bin-normalized residuals because raw offset logits are not exported.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.run or not args.output_dir:
        return _usage_and_exit()
    runs = parse_run_specs(args.run)
    splits = parse_splits(args.splits)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    defect_root = Path(args.defect_root)
    npz_root = Path(args.npz_root)
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for label, run_dir in runs:
        for split in splits:
            frame = load_split_frame(
                label,
                run_dir,
                split,
                defect_root,
                npz_root,
                args.center_bin_size_cells,
            )
            if frame is None:
                skipped.append(f"{label}:{split}")
            else:
                frames.append(frame)
    if not frames:
        raise ValueError("No diagnostic splits were available.")
    summarize(frames, output_dir)
    if skipped:
        (output_dir / "skipped_splits.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
    print(f"Saved V3 hard-case grouped diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
