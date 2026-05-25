#!/usr/bin/env python
"""Export a small visual gallery for Stage 20.83 profile-primary predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_true_3d_rbc_pilot_dataset import (
    V3_240_DATASET_ID,
    PARAM_NAMES,
    ROOT,
    depth_grid_from_params,
    load_dataset,
    projected_mask_from_params,
    write_csv,
)


PROFILE_CANDIDATE = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_candidate_profile_metrics.csv"
PROFILE_TRAINING = ROOT / "results/metrics/true_3d_rbc_v3_240_profile_primary_profile_metrics.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_profile_primary_loss_gallery_summary.txt"
INDEX = ROOT / "results/metrics/true_3d_rbc_profile_primary_loss_gallery_index.csv"
SAMPLE_METRICS = ROOT / "results/metrics/true_3d_rbc_profile_primary_loss_gallery_sample_metrics.csv"
PREVIEW_DIR = ROOT / "results/previews/true_3d_rbc_profile_primary_loss_gallery"

INDEX_FIELDS = ["rank", "selection_bucket", "sample_id", "split", "png_path", "profile_depth_rmse_m", "er_like_profile_error", "projected_mask_dice", "notes"]
SAMPLE_FIELDS = [
    "selection_bucket",
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "projected_mask_iou",
    "projected_mask_dice",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "png_path",
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    return [row for row in rows if row.get("split") == "test" and str(row.get("selected_by_validation", "")).lower() == "true"]


def selected_profile_path() -> Path:
    return PROFILE_TRAINING if PROFILE_TRAINING.exists() else PROFILE_CANDIDATE


def choose_rows(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    best_profile = sorted(rows, key=lambda row: float(row["profile_depth_rmse_m"]))[:3]
    worst_profile = sorted(rows, key=lambda row: float(row["profile_depth_rmse_m"]), reverse=True)[:3]
    best_dice = sorted(rows, key=lambda row: float(row["projected_mask_dice"]), reverse=True)[:3]
    worst_curv = sorted(rows, key=lambda row: float(row["wLD_abs_error"]) + float(row["wWD_abs_error"]) + float(row["wLW_abs_error"]), reverse=True)[:3]
    seen: set[str] = set()
    out: list[tuple[str, dict[str, Any]]] = []
    for bucket, bucket_rows in (
        ("best_profile", best_profile),
        ("worst_profile", worst_profile),
        ("best_dice", best_dice),
        ("curvature_risk", worst_curv),
    ):
        for row in bucket_rows:
            sid = row["sample_id"]
            if sid in seen:
                continue
            seen.add(sid)
            out.append((bucket, row))
            if len(out) >= 12:
                return out
    return out


def params_from_row(row: dict[str, Any], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_{name}"]) for name in PARAM_NAMES], dtype=np.float32)


def plot_sample(dataset: Any, row: dict[str, Any], bucket: str, path: Path) -> None:
    idx = int(np.where(dataset.sample_ids == row["sample_id"])[0][0])
    true_params = params_from_row(row, "true")
    pred_params = params_from_row(row, "pred")
    true_depth = dataset.profile_depth_grid_m[idx]
    pred_depth = depth_grid_from_params(pred_params)
    err_depth = pred_depth - true_depth
    true_mask = dataset.projected_mask_2d[idx]
    pred_mask = projected_mask_from_params(pred_params, dataset.profile_pose[idx])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    x = np.asarray(dataset.sensor_x)
    for axis_idx, axis_name in enumerate(dataset.axis_names):
        for line_idx, y in enumerate(dataset.scan_line_y):
            axes[0, 0].plot(x, dataset.delta_b[idx, axis_idx, line_idx], linewidth=0.8, label=f"{axis_name} y{line_idx}")
    axes[0, 0].set_title("delta_b Bx/By/Bz")
    axes[0, 0].set_xlabel("x_m")
    axes[0, 0].legend(fontsize=6, ncol=2)
    axes[0, 1].imshow(true_mask, cmap="gray")
    axes[0, 1].set_title("true mask")
    axes[0, 2].imshow(pred_mask, cmap="gray")
    axes[0, 2].set_title("pred mask")
    overlay = np.zeros((*true_mask.shape, 3), dtype=np.float32)
    overlay[..., 1] = true_mask
    overlay[..., 0] = pred_mask
    axes[0, 3].imshow(overlay)
    axes[0, 3].set_title("mask overlay")
    im1 = axes[1, 0].imshow(true_depth.T, origin="lower", aspect="auto")
    axes[1, 0].set_title("true depth grid")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    im2 = axes[1, 1].imshow(pred_depth.T, origin="lower", aspect="auto")
    axes[1, 1].set_title("pred depth grid")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    im3 = axes[1, 2].imshow(err_depth.T, origin="lower", aspect="auto", cmap="coolwarm")
    axes[1, 2].set_title("depth error")
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    text = [
        f"bucket={bucket}",
        f"sample={row['sample_id']}",
        f"profile_rmse={float(row['profile_depth_rmse_m']):.6g}",
        f"Er_like={float(row['er_like_profile_error']):.4f}",
        f"IoU/Dice={float(row['projected_mask_iou']):.4f}/{float(row['projected_mask_dice']):.4f}",
        f"L/W/D mm={float(row['L_mae_mm']):.3f}/{float(row['W_mae_mm']):.3f}/{float(row['D_mae_mm']):.3f}",
        f"w aux={float(row['wLD_abs_error']):.3f}/{float(row['wWD_abs_error']):.3f}/{float(row['wLW_abs_error']):.3f}",
        "true params: " + ", ".join(f"{v:.4g}" for v in true_params),
        "pred params: " + ", ".join(f"{v:.4g}" for v in pred_params),
    ]
    axes[1, 3].axis("off")
    axes[1, 3].text(0.0, 1.0, "\n".join(text), va="top", fontsize=8)
    for ax in axes.ravel()[:7]:
        ax.tick_params(labelsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    profile_path = args.profile_metrics or selected_profile_path()
    rows = read_rows(profile_path)
    if not rows:
        raise RuntimeError(f"no selected test rows found in {profile_path}")
    dataset = load_dataset(args.dataset_id)
    chosen = choose_rows(rows)
    index_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for rank, (bucket, row) in enumerate(chosen, start=1):
        png_path = args.preview_dir / f"{rank:02d}_{bucket}_{row['sample_id']}.png"
        plot_sample(dataset, row, bucket, png_path)
        index_rows.append(
            {
                "rank": rank,
                "selection_bucket": bucket,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "png_path": str(png_path),
                "profile_depth_rmse_m": row["profile_depth_rmse_m"],
                "er_like_profile_error": row["er_like_profile_error"],
                "projected_mask_dice": row["projected_mask_dice"],
                "notes": "held-out test visualization; not used for selection",
            }
        )
        out_row = {field: row.get(field, "") for field in SAMPLE_FIELDS}
        out_row["selection_bucket"] = bucket
        out_row["png_path"] = str(png_path)
        sample_rows.append(out_row)
    write_csv(args.index, index_rows, INDEX_FIELDS)
    write_csv(args.sample_metrics, sample_rows, SAMPLE_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "true_3d_rbc profile-primary loss gallery summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"profile_metrics_source: {profile_path}",
                f"preview_dir: {args.preview_dir}",
                f"png_count: {len(index_rows)}",
                "selection_policy: best profile, worst profile, best Dice, and curvature-risk samples from held-out test rows.",
                "selection_boundary: gallery was not used for model or candidate selection.",
                "artifact_boundary: preview PNG files are local inspection artifacts and must not be committed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=V3_240_DATASET_ID)
    parser.add_argument("--profile-metrics", type=Path)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--sample-metrics", type=Path, default=SAMPLE_METRICS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

