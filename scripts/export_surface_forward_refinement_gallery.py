#!/usr/bin/env python
"""Export the 25.8 surface forward-refinement gallery index and optional PNGs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from audit_surface_shape_extension_rbc_oracle_fit import DATASET_ID, ROOT, load_surface_dataset, pose_for_sample
from build_surface_forward_refinement_target_set import PARAM_NAMES, REGISTRY, as_float, read_csv, write_csv
from load_true_3d_rbc_pilot_dataset import depth_map_from_params, projected_mask_from_params
from run_surface_forward_refinement_inference import METRICS as RUNNER_METRICS


SUMMARY = ROOT / "results/summaries/surface_forward_refinement_gallery_summary.txt"
INDEX = ROOT / "results/metrics/surface_forward_refinement_gallery_index.csv"
PREVIEW_DIR = ROOT / "results/previews/surface_forward_refinement_gallery"

FIELDS = [
    "gallery_bucket",
    "rank",
    "sample_id",
    "sample_index",
    "split",
    "shape_type",
    "representation_target",
    "target_role",
    "diagnosis",
    "failure_reason",
    "baseline_profile_rmse_m",
    "refined_profile_rmse_m",
    "oracle_profile_rmse_m",
    "profile_rmse_improvement_m",
    "baseline_Dice",
    "refined_Dice",
    "oracle_Dice",
    "Dice_improvement",
    "baseline_IoU",
    "refined_IoU",
    "oracle_IoU",
    "IoU_improvement",
    "feature_residual_before",
    "feature_residual_after",
    "feature_residual_improvement",
    "suitability_flag",
    "preview_path",
    "preview_generated",
    "preview_note",
]


def try_matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def params_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([as_float(row[f"{prefix}_{name}"]) for name in PARAM_NAMES], dtype=np.float64)


def failure_reason(row: dict[str, str]) -> str:
    if row.get("eligibility_status") == "not_suitable_for_rbc_refinement":
        return row.get("not_suitable_reason") or "not_suitable_for_rbc_refinement"
    if row.get("target_role") == "refinement_target":
        reasons: list[str] = []
        if as_float(row["profile_rmse_delta_m"]) > 0.0:
            reasons.append("profile_rmse_not_improved")
        if as_float(row["Dice_delta"]) < 0.0:
            reasons.append("dice_not_improved")
        if as_float(row["Er_like_delta"]) > 0.0:
            reasons.append("er_like_not_improved")
        return "|".join(reasons) if reasons else row.get("diagnosis", "")
    return row.get("diagnosis", "")


def add_unique(selected: list[tuple[str, int, dict[str, str]]], bucket: str, rows: list[dict[str, str]], count: int, key: Any, reverse: bool) -> None:
    seen = {(bucket, row["sample_id"]) for bucket, _rank, row in selected}
    rank = 1
    for row in sorted(rows, key=key, reverse=reverse):
        if (bucket, row["sample_id"]) in seen:
            continue
        selected.append((bucket, rank, row))
        seen.add((bucket, row["sample_id"]))
        rank += 1
        if rank > count:
            break


def select_gallery_rows(rows: list[dict[str, str]]) -> list[tuple[str, int, dict[str, str]]]:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in rows if row["shape_type"] == "rbc_like_smooth_pit"]
    negative = [row for row in rows if row["target_role"] == "excluded_negative_control"]
    degraded_targets = [
        row
        for row in targets
        if as_float(row["profile_rmse_delta_m"]) > 0.0
        or as_float(row["Dice_delta"]) < 0.0
        or as_float(row["Er_like_delta"]) > 0.0
    ]
    selected: list[tuple[str, int, dict[str, str]]] = []
    add_unique(selected, "best_5_profile_rmse_improvement", targets, 5, lambda row: as_float(row["profile_rmse_delta_m"]), False)
    add_unique(selected, "best_5_dice_improvement", targets, 5, lambda row: as_float(row["Dice_delta"]), True)
    add_unique(selected, "worst_5_remaining_failures", targets, 5, lambda row: as_float(row["refined_profile_depth_rmse_m"]), True)
    for rank, row in enumerate(sorted(degraded_targets, key=lambda item: as_float(item["profile_rmse_delta_m"]), reverse=True), start=1):
        selected.append(("all_degraded_target_rows", rank, row))
    add_unique(selected, "representative_rbc_like_control_rows", rbc_like, 5, lambda row: as_float(row["refined_profile_depth_rmse_m"]), True)
    add_unique(selected, "representative_multi_pit_negative_controls", negative, 5, lambda row: as_float(row["baseline_profile_depth_rmse_m"]), True)
    return selected


def render_preview(plt: Any, dataset: Any, row: dict[str, str], preview_path: Path) -> None:
    idx = int(row["sample_index"])
    pose = pose_for_sample(dataset, idx)
    true_depth = np.asarray(dataset.depth_grid_m[idx], dtype=np.float64)
    baseline_depth = depth_map_from_params(params_from_row(row, "initial"), pose)
    refined_depth = depth_map_from_params(params_from_row(row, "refined"), pose)
    oracle_depth = depth_map_from_params(params_from_row(row, "oracle"), pose)
    baseline_mask = projected_mask_from_params(params_from_row(row, "initial"), pose)
    refined_mask = projected_mask_from_params(params_from_row(row, "refined"), pose)
    oracle_mask = projected_mask_from_params(params_from_row(row, "oracle"), pose)
    true_mask = np.asarray(dataset.projected_mask_2d[idx], dtype=np.uint8)

    v_max = max(float(np.max(true_depth)), float(np.max(baseline_depth)), float(np.max(refined_depth)), float(np.max(oracle_depth)), 1.0e-9)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    panels = [
        ("true depth", true_depth),
        ("baseline depth", baseline_depth),
        ("refined depth", refined_depth),
        ("oracle depth", oracle_depth),
    ]
    for ax, (title, data) in zip(axes[0], panels):
        image = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=v_max)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    mask_panels = [
        ("true mask", true_mask),
        ("baseline mask", baseline_mask),
        ("refined mask", refined_mask),
        ("oracle mask", oracle_mask),
    ]
    for ax, (title, data) in zip(axes[1], mask_panels):
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{row['sample_id']} | {row['shape_type']} | RMSE {as_float(row['baseline_profile_depth_rmse_m']):.3g}->{as_float(row['refined_profile_depth_rmse_m']):.3g}",
        fontsize=9,
    )
    _ = image
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path, dpi=140)
    plt.close(fig)


def index_row(bucket: str, rank: int, row: dict[str, str], preview_path: str, preview_generated: bool, preview_note: str) -> dict[str, Any]:
    baseline_rmse = as_float(row["baseline_profile_depth_rmse_m"])
    refined_rmse = as_float(row["refined_profile_depth_rmse_m"])
    baseline_dice = as_float(row["baseline_projected_mask_Dice"])
    refined_dice = as_float(row["refined_projected_mask_Dice"])
    baseline_iou = as_float(row["baseline_projected_mask_IoU"])
    refined_iou = as_float(row["refined_projected_mask_IoU"])
    before = as_float(row["feature_residual_mse_before"])
    after = as_float(row["feature_residual_mse_after"])
    return {
        "gallery_bucket": bucket,
        "rank": rank,
        "sample_id": row["sample_id"],
        "sample_index": row["sample_index"],
        "split": row["split"],
        "shape_type": row["shape_type"],
        "representation_target": row["representation_target"],
        "target_role": row["target_role"],
        "diagnosis": row["diagnosis"],
        "failure_reason": failure_reason(row),
        "baseline_profile_rmse_m": baseline_rmse,
        "refined_profile_rmse_m": refined_rmse,
        "oracle_profile_rmse_m": as_float(row["oracle_profile_depth_rmse_m"]),
        "profile_rmse_improvement_m": baseline_rmse - refined_rmse,
        "baseline_Dice": baseline_dice,
        "refined_Dice": refined_dice,
        "oracle_Dice": as_float(row["oracle_projected_mask_Dice"]),
        "Dice_improvement": refined_dice - baseline_dice,
        "baseline_IoU": baseline_iou,
        "refined_IoU": refined_iou,
        "oracle_IoU": as_float(row["oracle_projected_mask_IoU"]),
        "IoU_improvement": refined_iou - baseline_iou,
        "feature_residual_before": before,
        "feature_residual_after": after,
        "feature_residual_improvement": before - after,
        "suitability_flag": row.get("eligibility_status") != "not_suitable_for_rbc_refinement",
        "preview_path": preview_path,
        "preview_generated": preview_generated,
        "preview_note": preview_note,
    }


def main() -> int:
    if not RUNNER_METRICS.exists():
        raise FileNotFoundError(RUNNER_METRICS)
    rows = read_csv(RUNNER_METRICS)
    selected = select_gallery_rows(rows)
    plt = try_matplotlib()
    dataset = load_surface_dataset(DATASET_ID, REGISTRY) if plt is not None else None
    index_rows: list[dict[str, Any]] = []
    generated_count = 0
    for bucket, rank, row in selected:
        safe_sample = row["sample_id"].replace("/", "_").replace("\\", "_")
        preview_path = PREVIEW_DIR / f"{bucket}_{rank:02d}_{safe_sample}.png"
        preview_note = "preview generated"
        generated = False
        if plt is not None and dataset is not None:
            render_preview(plt, dataset, row, preview_path)
            generated = True
            generated_count += 1
        else:
            preview_note = "matplotlib unavailable; index-only gallery row"
        index_rows.append(index_row(bucket, rank, row, str(preview_path) if generated else "", generated, preview_note))
    write_csv(INDEX, index_rows, FIELDS)
    bucket_counts: dict[str, int] = {}
    for bucket, _rank, _row in selected:
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    lines = [
        "25.8 surface forward-refinement gallery",
        "",
        f"gallery_index_csv: {INDEX}",
        f"ignored_preview_dir: {PREVIEW_DIR}",
        f"index_row_count: {len(index_rows)}",
        f"preview_png_generated_count: {generated_count}",
        f"preview_png_commit_allowed: false",
        f"matplotlib_available: {plt is not None}",
        f"bucket_counts: {bucket_counts}",
        "",
        "gallery_policy:",
        "- PNG previews, when generated, live only in ignored results/previews/surface_forward_refinement_gallery/.",
        "- Index rows are the commit-safe artifact.",
        "- Labels/masks/depth maps are used only for visualization annotation and comparison panels.",
        "- Multi-pit rows are displayed as negative controls, not RBC refinement success.",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
