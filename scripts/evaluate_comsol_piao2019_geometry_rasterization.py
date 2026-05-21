from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
DEFAULT_PREDICTIONS = PROJECT_ROOT / "results/metrics/comsol_piao2019_geometry_parameter_prediction_errors.csv"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_piao2019_geometry_parameter_poc_metrics.csv"
DEFAULT_GROUP_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_piao2019_geometry_parameter_group_summary.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_piao2019_geometry_parameter_poc_summary.txt"
DEFAULT_AUDIT = PROJECT_ROOT / "results/summaries/comsol_piao2019_geometry_parameter_failure_audit_summary.txt"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_piao2019_geometry_parameter_poc"

DENSE_SINGLE_BASELINE_IOU = 0.6515
DENSE_SINGLE_BASELINE_DICE = 0.7861
MAIN_TYPES = {"rectangular_notch", "rotated_rect"}
MAIN_FEATURE_SET = "nls_style_main"
BASELINE_FEATURE_SET = "generic_only"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _raster_rect(mask_x: np.ndarray, mask_y: np.ndarray, cx: float, cy: float, width: float, length: float, angle_rad: float) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(mask_x, mask_y)
    dx = x_grid - cx
    dy = y_grid - cy
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    xr = dx * ca + dy * sa
    yr = -dx * sa + dy * ca
    return ((np.abs(xr) <= width / 2.0) & (np.abs(yr) <= length / 2.0)).astype(np.uint8)


def _mask_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred_bool = pred.astype(bool)
    true_bool = true.astype(bool)
    inter = int(np.logical_and(pred_bool, true_bool).sum())
    union = int(np.logical_or(pred_bool, true_bool).sum())
    pred_area = int(pred_bool.sum())
    true_area = int(true_bool.sum())
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred_area + true_area) if (pred_area + true_area) else 1.0
    area_error = abs(pred_area - true_area) / max(true_area, 1)
    center_error = _center_error(pred_bool, true_bool)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "area_error": float(area_error),
        "center_error": float(center_error),
        "pred_area": float(pred_area),
        "true_area": float(true_area),
        "pred_area_zero": float(pred_area == 0),
    }


def _center_error(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.sum() == 0 or true.sum() == 0:
        return float("nan")
    py, px = np.argwhere(pred).mean(axis=0)
    ty, tx = np.argwhere(true).mean(axis=0)
    return float(math.hypot(px - tx, py - ty))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_to_float(row.get(key, "")) for row in rows]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else math.nan


def _std(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_to_float(row.get(key, "")) for row in rows]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.std(vals)) if vals else math.nan


def _summarize_group(rows: list[dict[str, Any]], group_name: str, group_value: str) -> dict[str, Any]:
    return {
        "group_name": group_name,
        "group_value": group_value,
        "n": len(rows),
        "iou_mean": _mean(rows, "iou"),
        "dice_mean": _mean(rows, "dice"),
        "area_error_mean": _mean(rows, "area_error"),
        "center_error_mean": _mean(rows, "center_error"),
        "type_accuracy": _mean(rows, "type_correct"),
        "center_mae": _mean(rows, "center_error_param"),
        "width_mae": _mean(rows, "width_abs_error"),
        "length_mae": _mean(rows, "length_abs_error"),
        "depth_mae": _mean(rows, "depth_abs_error"),
        "angle_mae_deg": _mean(rows, "angle_abs_error_deg"),
        "pred_area_zero_mean": _mean(rows, "pred_area_zero"),
    }


def _failure_category(row: dict[str, Any]) -> str:
    if row["defect_type"] == "polygon":
        return "polygon_unsupported"
    if row["pred_defect_type"] not in MAIN_TYPES:
        return "wrong_type"
    angle_error = _to_float(row["angle_abs_error_deg"])
    center_error_px = _to_float(row["center_error"])
    area_error = _to_float(row["area_error"])
    size_error = max(_to_float(row["width_abs_error"], 0.0), _to_float(row["length_abs_error"], 0.0))
    if row["defect_type"] == "rotated_rect" and angle_error > 15:
        return "wrong_angle"
    if center_error_px > 8:
        return "wrong_center"
    if area_error > 0.5 or size_error > 0.004:
        return "wrong_size"
    if _to_float(row["iou"]) < 0.4:
        return "rasterizer_or_parameter_mismatch"
    return "acceptable_or_minor_error"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_to_mask(row: dict[str, Any], mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    if row["pred_defect_type"] not in MAIN_TYPES:
        return np.zeros((mask_y.shape[0], mask_x.shape[0]), dtype=np.uint8)
    cx = _to_float(row["pred_center_x"])
    cy = _to_float(row["pred_center_y"])
    width = max(_to_float(row["pred_width"]), 1e-6)
    length = max(_to_float(row["pred_length"]), 1e-6)
    angle = math.radians(_to_float(row["pred_angle_deg"], 0.0)) if row["pred_defect_type"] == "rotated_rect" else 0.0
    return _raster_rect(mask_x, mask_y, cx, cy, width, length, angle)


def _preview(metrics_rows: list[dict[str, Any]], npz_data: Any, preview_dir: Path, max_count: int = 24) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = npz_data["sample_ids"].astype(str)
    masks = npz_data["masks"].astype(bool)
    mask_x = npz_data["mask_x"].astype(float)
    mask_y = npz_data["mask_y"].astype(float)
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    main_test = [
        row
        for row in metrics_rows
        if row["feature_set"] == MAIN_FEATURE_SET and row["split"] == "test" and row["defect_type"] in MAIN_TYPES
    ]
    if not main_test:
        return 0
    sorted_iou = sorted(main_test, key=lambda row: _to_float(row["iou"]))
    sorted_angle = sorted(
        [row for row in main_test if row["defect_type"] == "rotated_rect"],
        key=lambda row: _to_float(row["angle_abs_error_deg"]),
        reverse=True,
    )
    chosen = sorted_iou[:8] + sorted_iou[-8:] + sorted_angle[:8]
    seen: set[str] = set()
    count = 0
    for row in chosen:
        if count >= max_count or row["sample_id"] in seen:
            continue
        seen.add(row["sample_id"])
        idx = id_to_idx[row["sample_id"]]
        true_mask = masks[idx]
        pred_mask = _prediction_to_mask(row, mask_x, mask_y)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(true_mask, origin="lower", cmap="gray")
        axes[0].set_title("true")
        axes[1].imshow(pred_mask, origin="lower", cmap="gray")
        axes[1].set_title("pred geometry")
        axes[2].imshow(true_mask, origin="lower", cmap="gray", alpha=0.6)
        axes[2].imshow(pred_mask, origin="lower", cmap="Reds", alpha=0.45)
        axes[2].set_title(f"IoU {float(row['iou']):.3f}")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{row['sample_id']} {row['defect_type']} pred={row['pred_defect_type']}")
        out = preview_dir / f"{count:02d}_{row['sample_id']}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        count += 1
    return count


def evaluate(npz_path: Path, predictions_path: Path, metrics_path: Path, group_path: Path, preview_dir: Path) -> dict[str, Any]:
    predictions = _read_csv(predictions_path)
    data = np.load(npz_path, allow_pickle=True)
    sample_ids = data["sample_ids"].astype(str)
    masks = data["masks"].astype(np.uint8)
    mask_x = data["mask_x"].astype(float)
    mask_y = data["mask_y"].astype(float)
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}

    metrics_rows: list[dict[str, Any]] = []
    for row in predictions:
        out: dict[str, Any] = dict(row)
        sample_id = row["sample_id"]
        if sample_id not in id_to_idx:
            raise KeyError(f"Prediction sample_id not in NPZ: {sample_id}")
        true_mask = masks[id_to_idx[sample_id]]
        if row["defect_type"] in MAIN_TYPES:
            pred_mask = _prediction_to_mask(row, mask_x, mask_y)
            mask_metric = _mask_metrics(pred_mask, true_mask)
        else:
            mask_metric = {
                "iou": "",
                "dice": "",
                "area_error": "",
                "center_error": "",
                "pred_area": "",
                "true_area": float(true_mask.sum()),
                "pred_area_zero": "",
            }
        out.update(mask_metric)
        out["center_error_param"] = row.get("center_error", "")
        out["failure_category"] = _failure_category(out)
        metrics_rows.append(out)

    _write_csv(metrics_rows, metrics_path)

    group_rows: list[dict[str, Any]] = []
    for feature_set in sorted(set(row["feature_set"] for row in metrics_rows)):
        fs_rows = [row for row in metrics_rows if row["feature_set"] == feature_set]
        for split in ["train", "val", "test"]:
            group_rows.append(
                _summarize_group(
                    [row for row in fs_rows if row["split"] == split and row["defect_type"] in MAIN_TYPES],
                    "feature_set_split_main_rect_rotated",
                    f"{feature_set}:{split}",
                )
            )
        for defect_type in ["rectangular_notch", "rotated_rect", "polygon"]:
            group_rows.append(
                _summarize_group(
                    [row for row in fs_rows if row["defect_type"] == defect_type],
                    "feature_set_defect_type_all_splits",
                    f"{feature_set}:{defect_type}",
                )
            )
        for category in sorted(set(row["failure_category"] for row in fs_rows)):
            group_rows.append(
                _summarize_group(
                    [row for row in fs_rows if row["failure_category"] == category],
                    "feature_set_failure_category",
                    f"{feature_set}:{category}",
                )
            )
    _write_csv(group_rows, group_path)
    preview_count = _preview(metrics_rows, data, preview_dir)
    return {
        "metrics_rows": metrics_rows,
        "group_rows": group_rows,
        "preview_count": preview_count,
        "metrics_path": str(metrics_path),
        "group_path": str(group_path),
        "preview_dir": str(preview_dir),
    }


def _split_metrics(metrics_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for feature_set in sorted(set(row["feature_set"] for row in metrics_rows)):
        out[feature_set] = {}
        fs_rows = [row for row in metrics_rows if row["feature_set"] == feature_set]
        for split in ["train", "val", "test"]:
            rows = [row for row in fs_rows if row["split"] == split and row["defect_type"] in MAIN_TYPES]
            all_rows = [row for row in fs_rows if row["split"] == split]
            rot_rows = [row for row in rows if row["defect_type"] == "rotated_rect"]
            out[feature_set][split] = {
                "n_main_rect_rotated": float(len(rows)),
                "type_accuracy_all3": _mean(all_rows, "type_correct"),
                "iou": _mean(rows, "iou"),
                "dice": _mean(rows, "dice"),
                "area_error": _mean(rows, "area_error"),
                "center_error_mask_px": _mean(rows, "center_error"),
                "center_mae_m": _mean(rows, "center_error_param"),
                "width_mae_m": _mean(rows, "width_abs_error"),
                "length_mae_m": _mean(rows, "length_abs_error"),
                "depth_mae_m": _mean(rows, "depth_abs_error"),
                "rotated_angle_mae_deg": _mean(rot_rows, "angle_abs_error_deg"),
            }
    return out


def _gain(main: float, baseline: float) -> float:
    if math.isnan(main) or math.isnan(baseline):
        return math.nan
    return main - baseline


def write_summaries(summary_path: Path, audit_path: Path, evaluation: dict[str, Any], npz_path: Path, predictions_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    rows = evaluation["metrics_rows"]
    split_metrics = _split_metrics(rows)
    main_test = [row for row in rows if row["feature_set"] == MAIN_FEATURE_SET and row["split"] == "test" and row["defect_type"] in MAIN_TYPES]
    generic_test = [row for row in rows if row["feature_set"] == BASELINE_FEATURE_SET and row["split"] == "test" and row["defect_type"] in MAIN_TYPES]
    main_all_test = [row for row in rows if row["feature_set"] == MAIN_FEATURE_SET and row["split"] == "test"]
    main_rot_test = [row for row in main_test if row["defect_type"] == "rotated_rect"]
    main_rect_test = [row for row in main_test if row["defect_type"] == "rectangular_notch"]
    iou_gain = _gain(_mean(main_test, "iou"), _mean(generic_test, "iou"))
    dice_gain = _gain(_mean(main_test, "dice"), _mean(generic_test, "dice"))
    promising = (
        _mean(main_all_test, "type_correct") > 0.55
        and _mean(main_test, "center_error_param") < 0.006
        and _mean(main_rot_test, "angle_abs_error_deg") < 25.0
        and _mean(main_test, "iou") > 0.25
        and _mean(main_test, "dice") > 0.40
        and (iou_gain > 0 or dice_gain > 0 or _mean(main_rot_test, "angle_abs_error_deg") < 20.0)
    )

    lines = [
        "COMSOL Piao-2019-inspired Bz-only NLS-style geometry parameter inversion POC summary",
        "",
        "Method alignment note:",
        "- This is not a Piao 2019 full reproduction.",
        "- This is a weak Piao-2019-inspired Bz-only 2D/quasi-2D engineering adaptation.",
        "- Paper RBC / tri-axis NLS / LS-SVM are not fully implemented here.",
        "- The revised POC adds Bz-only NLS-style exponential fitting features inspired by the paper's local/global exponential feature idea.",
        "- Project targets are rectangular_notch / rotated_rect geometry parameters from geometry_params, then hard rasterized into masks.",
        "",
        f"Input NPZ: {npz_path}",
        f"Prediction CSV: {predictions_path}",
        f"Metrics CSV: {evaluation['metrics_path']}",
        f"Group summary CSV: {evaluation['group_path']}",
        f"Preview directory: {evaluation['preview_dir']}",
        f"Preview PNG generated: {evaluation['preview_count']} (not for submission)",
        "",
        "Scope:",
        "- Input features use delta_bz / sensor_x / scan_line_y only.",
        "- Geometry labels come from geometry_params, not predicted masks.",
        "- Main reconstruction result is rect+rotated only.",
        "- Polygon participates in all-3 type classification but is excluded from main geometry regression and mask metrics.",
        "- No dense mask decoder, no COMSOL run, no new simulation data, and no baseline document update.",
        "",
        "Models:",
        "- Type classifier: LogisticRegression with train-fit StandardScaler.",
        "- Geometry regressors: validation-selected KernelRidge(RBF), SVR(RBF), or Ridge substitute for LS-SVM.",
        "- Main setting: features -> classifier + geometry regressors, no true defect_type input.",
        "- Validation split selects regressor hyperparameters; test split is final evaluation only.",
        "",
        "Train / val / test metrics:",
        json.dumps(split_metrics, indent=2, sort_keys=True),
        "",
        "Test headline for nls_style_main:",
        f"- all-3 type accuracy: {_mean(main_all_test, 'type_correct'):.4f}",
        f"- rect test IoU/Dice: {_mean(main_rect_test, 'iou'):.4f} / {_mean(main_rect_test, 'dice'):.4f}",
        f"- rotated test IoU/Dice: {_mean(main_rot_test, 'iou'):.4f} / {_mean(main_rot_test, 'dice'):.4f}",
        f"- rect+rotated test IoU/Dice: {_mean(main_test, 'iou'):.4f} / {_mean(main_test, 'dice'):.4f}",
        f"- test center MAE (m): {_mean(main_test, 'center_error_param'):.6f}",
        f"- test width / length / depth MAE (m): {_mean(main_test, 'width_abs_error'):.6f} / {_mean(main_test, 'length_abs_error'):.6f} / {_mean(main_test, 'depth_abs_error'):.6f}",
        f"- rotated_rect test angle MAE (deg): {_mean(main_rot_test, 'angle_abs_error_deg'):.4f}",
        "",
        "Bz-only NLS-style feature benefit vs generic_only:",
        f"- test IoU gain: {iou_gain:.4f}",
        f"- test Dice gain: {dice_gain:.4f}",
        f"- generic_only test IoU/Dice: {_mean(generic_test, 'iou'):.4f} / {_mean(generic_test, 'dice'):.4f}",
        f"- nls_style_main test IoU/Dice: {_mean(main_test, 'iou'):.4f} / {_mean(main_test, 'dice'):.4f}",
        "",
        "Dense decoder comparison:",
        f"- COMSOL single-defect pilot_v9 dense baseline test IoU/Dice: {DENSE_SINGLE_BASELINE_IOU:.4f} / {DENSE_SINGLE_BASELINE_DICE:.4f}",
        f"- This POC nls_style_main rect+rotated test IoU/Dice: {_mean(main_test, 'iou'):.4f} / {_mean(main_test, 'dice'):.4f}",
        "- This POC is not expected to beat the dense baseline; value is interpretability and parameterized failure attribution.",
        "",
        "POC acceptance judgement:",
        f"- Promising Piao-style geometry-parameter route: {promising}",
        "- Acceptance considers type accuracy, parameter errors, non-trivial mask IoU/Dice, NLS-style feature benefit or interpretability, and coherent failure categories.",
        "",
        "Failure source judgement:",
        "- If performance is weak, the primary suspects are Bz-only information limits, NLS-style feature instability, and KernelRidge/SVR/Ridge substitute capacity.",
        "- Labels and rect/rotated rasterization are expected to be stable because geometry_params exactly rasterize to the source masks for those shapes.",
        "- A future neural geometry head with differentiable rasterization may be a better continuation than further hand-crafted feature tuning.",
        "",
        "Next recommended direction:",
        "- B. Add differentiable rasterizer + neural geometry head.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = Counter(row["failure_category"] for row in rows if row["feature_set"] == MAIN_FEATURE_SET)
    worst = sorted(
        [row for row in rows if row["feature_set"] == MAIN_FEATURE_SET and row["split"] == "test" and row["defect_type"] in MAIN_TYPES],
        key=lambda row: _to_float(row["iou"]),
    )[:20]
    audit_lines = [
        "COMSOL Piao-2019-inspired Bz-only NLS-style geometry parameter failure audit",
        "",
        f"Failure category counts for nls_style_main: {dict(failures)}",
        "",
        "Interpretation:",
        "- wrong_type: classifier chose an unsupported or wrong geometry family.",
        "- wrong_center / wrong_size / wrong_angle: parameter estimation error.",
        "- rasterizer_or_parameter_mismatch: predicted parameters rasterized poorly without a simpler dominant error.",
        "- polygon_unsupported: expected in this revised POC and excluded from main reconstruction metrics.",
        "",
        "Worst test rect/rotated nls_style_main cases:",
    ]
    for row in worst:
        audit_lines.append(
            f"- {row['sample_id']} {row['defect_type']} pred={row['pred_defect_type']} "
            f"IoU={_to_float(row['iou']):.4f} Dice={_to_float(row['dice']):.4f} "
            f"center_err_px={_to_float(row['center_error']):.2f} "
            f"angle_err_deg={_to_float(row['angle_abs_error_deg']):.2f} "
            f"category={row['failure_category']}"
        )
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP_SUMMARY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    args = parser.parse_args()

    evaluation = evaluate(args.npz, args.predictions, args.metrics, args.group_summary, args.preview_dir)
    write_summaries(args.summary, args.audit, evaluation, args.npz, args.predictions)


if __name__ == "__main__":
    main()
