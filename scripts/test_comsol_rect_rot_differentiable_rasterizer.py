from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
DEFAULT_LABELS = PROJECT_ROOT / "results/metrics/comsol_single_defect_geometry_labels.csv"
DEFAULT_METRICS = (
    PROJECT_ROOT / "results/metrics/comsol_rect_rot_differentiable_rasterizer_validation_metrics.csv"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_rect_rot_differentiable_rasterizer_validation_summary.txt"
)

MAIN_TYPES = {"rectangular_notch", "rotated_rect"}
TEMPERATURE_M = 5.0e-4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def soft_rect_mask(
    mask_x: torch.Tensor,
    mask_y: torch.Tensor,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    width: torch.Tensor,
    length: torch.Tensor,
    angle_rad: torch.Tensor,
    temperature: float = TEMPERATURE_M,
) -> torch.Tensor:
    x_grid, y_grid = torch.meshgrid(mask_x, mask_y, indexing="xy")
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)
    cx = center_x.view(-1, 1, 1)
    cy = center_y.view(-1, 1, 1)
    half_w = (width.view(-1, 1, 1).clamp_min(1e-6)) / 2.0
    half_l = (length.view(-1, 1, 1).clamp_min(1e-6)) / 2.0
    angle = angle_rad.view(-1, 1, 1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    dx0 = x_grid - cx
    dy0 = y_grid - cy
    x_rot = dx0 * cos_a + dy0 * sin_a
    y_rot = -dx0 * sin_a + dy0 * cos_a
    dx = torch.abs(x_rot) - half_w
    dy = torch.abs(y_rot) - half_l
    outside = torch.sqrt(torch.clamp(dx, min=0.0).square() + torch.clamp(dy, min=0.0).square() + 1e-18)
    inside = torch.clamp(torch.maximum(dx, dy), max=0.0)
    signed_distance = outside + inside
    return torch.sigmoid(-signed_distance / temperature)


def mask_metrics(pred_prob: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = pred_prob >= threshold
    true = true_mask > 0
    inter = int(np.logical_and(pred, true).sum())
    union = int(np.logical_or(pred, true).sum())
    pred_area = int(pred.sum())
    true_area = int(true.sum())
    iou = inter / union if union else 1.0
    dice = 2.0 * inter / (pred_area + true_area) if (pred_area + true_area) else 1.0
    area_error = abs(pred_area - true_area) / max(true_area, 1)
    return {
        "raster_iou": float(iou),
        "raster_dice": float(dice),
        "raster_area_error": float(area_error),
        "pred_area": float(pred_area),
        "true_area": float(true_area),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [to_float(row[key]) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return float(np.mean(values)) if values else math.nan


def validate(npz_path: Path, labels_path: Path, metrics_path: Path, summary_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    labels = read_csv(labels_path)
    sample_ids = data["sample_ids"].astype(str)
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    masks = data["masks"].astype(np.uint8)
    mask_x = torch.tensor(data["mask_x"].astype(np.float32))
    mask_y = torch.tensor(data["mask_y"].astype(np.float32))

    rows: list[dict[str, Any]] = []
    for row in labels:
        if row["defect_type"] not in MAIN_TYPES:
            continue
        sample_id = row["sample_id"]
        idx = id_to_idx[sample_id]
        angle = 0.0 if row["defect_type"] == "rectangular_notch" else to_float(row["angle_rad"], 0.0)
        with torch.no_grad():
            prob = soft_rect_mask(
                mask_x,
                mask_y,
                torch.tensor([to_float(row["center_x"])], dtype=torch.float32),
                torch.tensor([to_float(row["center_y"])], dtype=torch.float32),
                torch.tensor([to_float(row["width"])], dtype=torch.float32),
                torch.tensor([to_float(row["length"])], dtype=torch.float32),
                torch.tensor([angle], dtype=torch.float32),
            )[0].cpu().numpy()
        metric = mask_metrics(prob, masks[idx])
        rows.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "defect_type": row["defect_type"],
                "temperature_m": TEMPERATURE_M,
                **metric,
            }
        )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["defect_type"]].append(row)
        by_split[row["split"]].append(row)

    cx = torch.tensor([0.0], requires_grad=True)
    cy = torch.tensor([0.0], requires_grad=True)
    width = torch.tensor([0.01], requires_grad=True)
    length = torch.tensor([0.007], requires_grad=True)
    angle = torch.tensor([0.2], requires_grad=True)
    prob = soft_rect_mask(mask_x, mask_y, cx, cy, width, length, angle)
    prob.mean().backward()
    grads = [cx.grad, cy.grad, width.grad, length.grad, angle.grad]
    gradient_ok = all(grad is not None and torch.isfinite(grad).all().item() for grad in grads)

    diagnostics = {
        "n": len(rows),
        "temperature_m": TEMPERATURE_M,
        "overall_iou": mean(rows, "raster_iou"),
        "overall_dice": mean(rows, "raster_dice"),
        "rect_iou": mean(by_type["rectangular_notch"], "raster_iou"),
        "rot_iou": mean(by_type["rotated_rect"], "raster_iou"),
        "by_type": {
            key: {
                "n": len(value),
                "iou": mean(value, "raster_iou"),
                "dice": mean(value, "raster_dice"),
                "area_error": mean(value, "raster_area_error"),
            }
            for key, value in sorted(by_type.items())
        },
        "by_split": {
            key: {
                "n": len(value),
                "iou": mean(value, "raster_iou"),
                "dice": mean(value, "raster_dice"),
                "area_error": mean(value, "raster_area_error"),
            }
            for key, value in sorted(by_split.items())
        },
        "gradient_ok": gradient_ok,
    }
    passed = (
        diagnostics["rect_iou"] >= 0.90
        and diagnostics["rot_iou"] >= 0.85
        and diagnostics["overall_iou"] >= 0.87
        and gradient_ok
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "COMSOL rect/rot differentiable rasterizer validation summary",
        "",
        f"Input NPZ: {npz_path}",
        f"Input labels: {labels_path}",
        f"Metrics CSV: {metrics_path}",
        f"Supported defect types: {sorted(MAIN_TYPES)}",
        "Polygon policy: parsed by label extraction but excluded from this rasterizer validation.",
        "",
        "Rasterizer formula:",
        "- Rotate mask grid by predicted center_x / center_y / angle_rad.",
        "- dx = abs(x_rot) - width / 2; dy = abs(y_rot) - length / 2.",
        "- signed_distance = outside_distance + inside_distance.",
        "- mask_prob = sigmoid(-signed_distance / temperature).",
        f"- temperature_m = {TEMPERATURE_M}",
        "",
        f"N rect+rotated: {diagnostics['n']}",
        f"Overall raster IoU/Dice: {diagnostics['overall_iou']:.6f} / {diagnostics['overall_dice']:.6f}",
        f"Rectangular raster IoU: {diagnostics['rect_iou']:.6f}",
        f"Rotated raster IoU: {diagnostics['rot_iou']:.6f}",
        f"By defect type: {diagnostics['by_type']}",
        f"By split: {diagnostics['by_split']}",
        f"Differentiability gradient sanity check passed: {gradient_ok}",
        "",
        "Acceptance thresholds:",
        "- rectangular_notch raster IoU mean >= 0.90",
        "- rotated_rect raster IoU mean >= 0.85",
        "- overall rect+rotated raster IoU mean >= 0.87",
        f"Rasterizer validation passed: {passed}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    diagnostics["passed"] = passed
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    diagnostics = validate(args.npz, args.labels, args.metrics, args.summary)
    if not diagnostics["passed"]:
        raise SystemExit("Differentiable rasterizer validation failed")


if __name__ == "__main__":
    main()
