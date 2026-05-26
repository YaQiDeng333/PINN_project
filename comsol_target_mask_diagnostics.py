"""Diagnose consistency between COMSOL mu_maps and provided masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _flatten_samples(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError(f"{name} must include a sample dimension and map dimensions, got {array.shape}")
    return array.reshape(array.shape[0], -1)


def _check_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return 0.0 if union == 0 else float(intersection / union)


def run_diagnostics(npz_path, output_dir, mu_threshold: float = 500.0):
    path = Path(npz_path)
    if not path.exists():
        raise ValueError(f"npz_path does not exist: {path}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        has_mu_maps = "mu_maps" in files
        has_masks = "masks" in files
        if not has_mu_maps and not has_masks:
            raise ValueError("diagnostics requires at least one of 'mu_maps' or 'masks'")

        mu_maps = data["mu_maps"] if has_mu_maps else None
        masks = data["masks"] if has_masks else None
        if has_mu_maps:
            _check_finite(mu_maps, "mu_maps")
            threshold_mask = _flatten_samples(mu_maps, "mu_maps") < mu_threshold
            sample_count = threshold_mask.shape[0]
        else:
            threshold_mask = None
            sample_count = masks.shape[0]
        if has_masks:
            _check_finite(masks, "masks")
            provided_mask = _flatten_samples(masks, "masks") > 0.5
            if provided_mask.shape[0] != sample_count:
                raise ValueError(
                    "masks and mu_maps must have the same sample count: "
                    f"got {provided_mask.shape[0]} and {sample_count}"
                )
        else:
            provided_mask = None

    rows = []
    for sample_index in range(sample_count):
        threshold_area = int(threshold_mask[sample_index].sum()) if threshold_mask is not None else ""
        provided_area = int(provided_mask[sample_index].sum()) if provided_mask is not None else ""
        if threshold_mask is not None and provided_mask is not None:
            area_diff = int(provided_area - threshold_area)
            mask_iou = _mask_iou(threshold_mask[sample_index], provided_mask[sample_index])
            mismatch_count = int(np.not_equal(threshold_mask[sample_index], provided_mask[sample_index]).sum())
        else:
            area_diff = ""
            mask_iou = ""
            mismatch_count = ""
        rows.append(
            {
                "sample_index": sample_index,
                "threshold_area": threshold_area,
                "provided_mask_area": provided_area,
                "area_diff": area_diff,
                "mask_iou": mask_iou,
                "mismatch_count": mismatch_count,
                "has_mu_maps": has_mu_maps,
                "has_masks": has_masks,
            }
        )

    threshold_areas = [int(row["threshold_area"]) for row in rows if row["threshold_area"] != ""]
    provided_areas = [int(row["provided_mask_area"]) for row in rows if row["provided_mask_area"] != ""]
    abs_area_diffs = [abs(int(row["area_diff"])) for row in rows if row["area_diff"] != ""]
    mask_ious = [float(row["mask_iou"]) for row in rows if row["mask_iou"] != ""]
    mismatch_counts = [int(row["mismatch_count"]) for row in rows if row["mismatch_count"] != ""]
    aggregate = {
        "samples": sample_count,
        "avg_threshold_area": float(np.mean(threshold_areas)) if threshold_areas else "",
        "avg_provided_mask_area": float(np.mean(provided_areas)) if provided_areas else "",
        "avg_abs_area_diff": float(np.mean(abs_area_diffs)) if abs_area_diffs else "",
        "avg_mask_iou": float(np.mean(mask_ious)) if mask_ious else "",
        "min_mask_iou": float(np.min(mask_ious)) if mask_ious else "",
        "max_mask_iou": float(np.max(mask_ious)) if mask_ious else "",
        "total_mismatch_count": int(np.sum(mismatch_counts)) if mismatch_counts else "",
    }

    per_sample_path = out_dir / "per_sample_mask_consistency.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "threshold_area",
                "provided_mask_area",
                "area_diff",
                "mask_iou",
                "mismatch_count",
                "has_mu_maps",
                "has_masks",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    aggregate_path = out_dir / "aggregate_mask_consistency.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate.keys()))
        writer.writeheader()
        writer.writerow(aggregate)

    summary_lines = [
        "# COMSOL target / mask consistency diagnostics",
        "",
        f"- npz_path: `{path}`",
        f"- mu_threshold: `{mu_threshold}`",
        f"- has_mu_maps: `{has_mu_maps}`",
        f"- has_masks: `{has_masks}`",
        f"- samples: `{sample_count}`",
        "",
        "## Aggregate metrics",
        "",
    ]
    for key, value in aggregate.items():
        if isinstance(value, float):
            summary_lines.append(f"- {key}: `{value:.6e}`")
        else:
            summary_lines.append(f"- {key}: `{value}`")
    summary_lines.append("")
    if has_mu_maps and has_masks:
        total_mismatch = int(aggregate["total_mismatch_count"])
        avg_iou = float(aggregate["avg_mask_iou"])
        if total_mismatch == 0:
            summary_lines.append("`mu_maps < mu_threshold` 与 provided `masks` 完全一致。")
            summary_lines.append("当前建议：`mask_source=mu_threshold` 和 `mask_source=masks` 等价，可继续使用默认 `mu_threshold`。")
        else:
            summary_lines.append("`mu_maps < mu_threshold` 与 provided `masks` 存在差异。")
            summary_lines.append(f"平均 mask IoU 为 `{avg_iou:.6e}`，total mismatch count 为 `{total_mismatch}`。")
            summary_lines.append("当前建议：在 conditional runner 中显式比较 `mask_source=mu_threshold` 与 `mask_source=masks`。")
    elif has_mu_maps:
        summary_lines.append("数据只包含 `mu_maps`，当前只能使用 `mu_threshold` 构造 mask label。")
    else:
        summary_lines.append("数据只包含 `masks`，当前 runner 若需要 `mu_mse` / `mu_mae` 仍应补充 `mu_maps`。")
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return {
        "rows": rows,
        "aggregate": aggregate,
        "per_sample_path": per_sample_path,
        "aggregate_path": aggregate_path,
        "summary_path": out_dir / "summary.md",
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--mu-threshold", type=float, default=500.0)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.npz_path or not args.output_dir:
        print("comsol_target_mask_diagnostics.py diagnoses mu_maps / masks consistency.")
        print("Usage: python comsol_target_mask_diagnostics.py --npz-path data.npz --output-dir diagnostics")
        return 0
    result = run_diagnostics(args.npz_path, args.output_dir, args.mu_threshold)
    print(f"Saved per-sample diagnostics to {result['per_sample_path']}")
    print(f"Saved aggregate diagnostics to {result['aggregate_path']}")
    print(f"Saved summary to {result['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
