"""Diagnose center decode errors for center-anchored polygon inverse runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None or value == "":
        return default
    return int(float(value))


def _require_fields(row: dict[str, str], fields: list[str], path: Path) -> None:
    missing = [field for field in fields if field not in row or row[field] == ""]
    if missing:
        raise ValueError(f"{path} is missing required center-decode fields: {', '.join(missing)}")


def _load_targets(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "sample_indices": data["sample_indices"].astype(np.int64),
            "presence": data["presence_targets"].astype(np.float32),
            "center_targets": data["center_targets_norm"].astype(np.float32),
            "center_x_bin": data["center_x_bin_targets"].astype(np.int64),
            "center_y_bin": data["center_y_bin_targets"].astype(np.int64),
            "center_offset": data["center_offset_targets"].astype(np.float32),
            "vertex_mask": data["polygon_vertex_mask"].astype(np.float32),
            "grid_dx": float(data["grid_dx"]),
            "grid_dy": float(data["grid_dy"]),
        }


def _defect_hard_cases(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    cases = {}
    for row in _read_csv(path):
        if "sample_index" in row and "hard_case_type" in row:
            cases[_int(row, "sample_index")] = row["hard_case_type"]
    return cases


def _infer_hard_center(row: dict[str, str], grid_dx: float, grid_dy: float) -> tuple[float, float]:
    xs = []
    ys = []
    for vertex_idx in range(4):
        if _float(row, f"vertex{vertex_idx}_valid", 0.0) <= 0.5:
            continue
        pred_x = _float(row, f"pred_x{vertex_idx}")
        pred_y = _float(row, f"pred_y{vertex_idx}")
        pred_local_x = _float(row, f"pred_local_x{vertex_idx}")
        pred_local_y = _float(row, f"pred_local_y{vertex_idx}")
        if all(math.isfinite(value) for value in [pred_x, pred_y, pred_local_x, pred_local_y]):
            xs.append(pred_x - pred_local_x * grid_dx)
            ys.append(pred_y - pred_local_y * grid_dy)
    if not xs:
        return math.nan, math.nan
    return float(np.median(xs)), float(np.median(ys))


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def _safe_max(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.max(finite)) if finite else math.nan


def diagnose_split(prediction_dir: Path, resplit_root: Path, split: str, run_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    prediction_path = prediction_dir / f"{split}_center_anchored_polygon_predictions.csv"
    metrics_path = prediction_dir / f"{split}_center_anchored_polygon_mask_metrics.csv"
    target_path = resplit_root / split / "center_anchored_polygon_targets.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    targets = _load_targets(target_path)
    sample_to_row = {int(sample_index): idx for idx, sample_index in enumerate(targets["sample_indices"])}
    hard_cases = _defect_hard_cases(resplit_root / split / "defect_params.csv")
    sample_metrics = {_int(row, "sample_index"): row for row in _read_csv(metrics_path)}
    component_rows = []
    sample_errors: dict[int, list[dict]] = {}
    required_prediction_fields = [
        "sample_index",
        "component_slot",
        "presence_true",
        "presence_pred",
        "center_x_bin_true",
        "center_x_bin_pred",
        "center_y_bin_true",
        "center_y_bin_pred",
        "center_offset_mae",
        "local_vertex_mae_grid",
        "decoded_vertex_mae",
    ]
    for row in _read_csv(prediction_path):
        _require_fields(row, required_prediction_fields, prediction_path)
        sample_index = _int(row, "sample_index")
        slot = _int(row, "component_slot")
        if sample_index not in sample_to_row:
            raise ValueError(f"sample_index {sample_index} missing from {target_path}")
        target_row = sample_to_row[sample_index]
        presence_true = float(targets["presence"][target_row, slot])
        true_center = targets["center_targets"][target_row, slot]
        true_offset = targets["center_offset"][target_row, slot]
        grid_dx = float(targets["grid_dx"])
        grid_dy = float(targets["grid_dy"])
        hard_x = _float(row, "hard_center_x_pred")
        hard_y = _float(row, "hard_center_y_pred")
        if not math.isfinite(hard_x) or not math.isfinite(hard_y):
            hard_x, hard_y = _infer_hard_center(row, grid_dx, grid_dy)
        soft_x = _float(row, "soft_center_x_pred")
        soft_y = _float(row, "soft_center_y_pred")
        hard_x_err = (hard_x - float(true_center[0])) / grid_dx if math.isfinite(hard_x) else math.nan
        hard_y_err = (hard_y - float(true_center[1])) / grid_dy if math.isfinite(hard_y) else math.nan
        soft_x_err = (soft_x - float(true_center[0])) / grid_dx if math.isfinite(soft_x) else math.nan
        soft_y_err = (soft_y - float(true_center[1])) / grid_dy if math.isfinite(soft_y) else math.nan
        x_bin_true = _int(row, "center_x_bin_true", int(targets["center_x_bin"][target_row, slot]))
        y_bin_true = _int(row, "center_y_bin_true", int(targets["center_y_bin"][target_row, slot]))
        x_bin_pred = _int(row, "center_x_bin_pred")
        y_bin_pred = _int(row, "center_y_bin_pred")
        metric = sample_metrics.get(sample_index, {})
        polygon_iou = _float(metric, "polygon_mask_iou")
        out = {
            "run": run_name,
            "split": split,
            "sample_index": sample_index,
            "component_slot": slot,
            "hard_case_type": hard_cases.get(sample_index, ""),
            "presence_true": presence_true,
            "presence_pred": _float(row, "presence_pred"),
            "polygon_iou": polygon_iou,
            "zero_iou": int(math.isfinite(polygon_iou) and polygon_iou <= 0.0),
            "x_bin_true": x_bin_true,
            "x_bin_pred": x_bin_pred,
            "y_bin_true": y_bin_true,
            "y_bin_pred": y_bin_pred,
            "x_bin_error": x_bin_pred - x_bin_true,
            "y_bin_error": y_bin_pred - y_bin_true,
            "x_bin_abs_error": abs(x_bin_pred - x_bin_true),
            "y_bin_abs_error": abs(y_bin_pred - y_bin_true),
            "both_bins_correct": int(x_bin_pred == x_bin_true and y_bin_pred == y_bin_true),
            "center_offset_x_true": float(true_offset[0]),
            "center_offset_x_pred": _float(row, "center_offset_x_pred"),
            "center_offset_y_true": float(true_offset[1]),
            "center_offset_y_pred": _float(row, "center_offset_y_pred"),
            "center_offset_mae": _float(row, "center_offset_mae"),
            "hard_center_x_error_grid": hard_x_err,
            "hard_center_y_error_grid": hard_y_err,
            "hard_center_l2_error_grid": float(np.linalg.norm([hard_x_err, hard_y_err])) if math.isfinite(hard_x_err) and math.isfinite(hard_y_err) else math.nan,
            "soft_center_x_error_grid": soft_x_err,
            "soft_center_y_error_grid": soft_y_err,
            "soft_center_l2_error_grid": float(np.linalg.norm([soft_x_err, soft_y_err])) if math.isfinite(soft_x_err) and math.isfinite(soft_y_err) else math.nan,
            "center_x_bin_prob_top1": _float(row, "center_x_bin_prob_top1"),
            "center_x_bin_prob_top2": _float(row, "center_x_bin_prob_top2"),
            "center_x_bin_prob_margin": _float(row, "center_x_bin_prob_margin"),
            "center_y_bin_prob_top1": _float(row, "center_y_bin_prob_top1"),
            "center_y_bin_prob_top2": _float(row, "center_y_bin_prob_top2"),
            "center_y_bin_prob_margin": _float(row, "center_y_bin_prob_margin"),
            "local_vertex_mae_grid": _float(row, "local_vertex_mae_grid"),
            "decoded_vertex_mae": _float(row, "decoded_vertex_mae"),
            "signed_area_flip": _int(row, "signed_area_flip"),
        }
        component_rows.append(out)
        if presence_true > 0.5:
            sample_errors.setdefault(sample_index, []).append(out)
    sample_rows = []
    for sample_index, rows in sorted(sample_errors.items()):
        metric = sample_metrics.get(sample_index, {})
        sample_rows.append(
            {
                "run": run_name,
                "split": split,
                "sample_index": sample_index,
                "hard_case_type": hard_cases.get(sample_index, ""),
                "polygon_iou": _float(metric, "polygon_mask_iou"),
                "zero_iou": int(_float(metric, "polygon_mask_iou") <= 0.0),
                "component_count": len(rows),
                "all_bins_correct": int(all(row["both_bins_correct"] for row in rows)),
                "max_x_bin_abs_error": _safe_max([row["x_bin_abs_error"] for row in rows]),
                "max_y_bin_abs_error": _safe_max([row["y_bin_abs_error"] for row in rows]),
                "mean_hard_center_l2_error_grid": _safe_mean([row["hard_center_l2_error_grid"] for row in rows]),
                "mean_soft_center_l2_error_grid": _safe_mean([row["soft_center_l2_error_grid"] for row in rows]),
                "mean_local_vertex_mae_grid": _safe_mean([row["local_vertex_mae_grid"] for row in rows]),
                "mean_decoded_vertex_mae": _safe_mean([row["decoded_vertex_mae"] for row in rows]),
            }
        )
    summary_rows = []
    present_rows = [row for row in component_rows if float(row["presence_true"]) > 0.5]
    if present_rows:
        summary_rows.append(
            {
                "run": run_name,
                "split": split,
                "components": len(present_rows),
                "samples": len(sample_rows),
                "mean_iou": _safe_mean([row["polygon_iou"] for row in sample_rows]),
                "zero_iou_count": sum(int(row["zero_iou"]) for row in sample_rows),
                "x_bin_acc": _safe_mean([1.0 if row["x_bin_abs_error"] == 0 else 0.0 for row in present_rows]),
                "y_bin_acc": _safe_mean([1.0 if row["y_bin_abs_error"] == 0 else 0.0 for row in present_rows]),
                "x_bin_abs_error": _safe_mean([row["x_bin_abs_error"] for row in present_rows]),
                "y_bin_abs_error": _safe_mean([row["y_bin_abs_error"] for row in present_rows]),
                "hard_center_l2_error_grid": _safe_mean([row["hard_center_l2_error_grid"] for row in present_rows]),
                "soft_center_l2_error_grid": _safe_mean([row["soft_center_l2_error_grid"] for row in present_rows]),
                "local_vertex_mae_grid": _safe_mean([row["local_vertex_mae_grid"] for row in present_rows]),
            }
        )
    return component_rows, sample_rows, summary_rows


def write_summary(path: Path, run_name: str, summary_rows: list[dict]) -> None:
    lines = [
        f"# Center decode diagnostics: {run_name}",
        "",
        "| split | mean_iou | zero_iou | x_bin_acc | y_bin_acc | hard_center_l2 | soft_center_l2 | local_vertex_mae_grid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        soft_value = row["soft_center_l2_error_grid"]
        soft_text = "nan" if not math.isfinite(soft_value) else f"{soft_value:.6f}"
        lines.append(
            f"| {row['split']} | `{row['mean_iou']:.6f}` | `{int(row['zero_iou_count'])}` | "
            f"`{row['x_bin_acc']:.6f}` | `{row['y_bin_acc']:.6f}` | `{row['hard_center_l2_error_grid']:.6f}` | "
            f"`{soft_text}` | `{row['local_vertex_mae_grid']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "Soft expected-center fields are `nan` when the prediction export predates S323 support.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--resplit-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="run")
    args = parser.parse_args(argv)
    prediction_dir = Path(args.prediction_dir)
    resplit_root = Path(args.resplit_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_components: list[dict] = []
    all_samples: list[dict] = []
    all_summary: list[dict] = []
    for split in SPLITS:
        component_rows, sample_rows, summary_rows = diagnose_split(prediction_dir, resplit_root, split, args.run_name)
        all_components.extend(component_rows)
        all_samples.extend(sample_rows)
        all_summary.extend(summary_rows)
    _write_csv(output_dir / "center_decode_per_component.csv", all_components)
    _write_csv(output_dir / "center_decode_per_sample.csv", all_samples)
    _write_csv(output_dir / "center_decode_summary.csv", all_summary)
    write_summary(output_dir / "summary.md", args.run_name, all_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
