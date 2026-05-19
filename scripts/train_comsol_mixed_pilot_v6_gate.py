from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import train_comsol_multiline_tiny_smoke as tiny
import train_comsol_polygon_pilot_v5_gate as poly_gate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data"
    / "comsol_mfl"
    / "prepared"
    / "comsol_single_defect_multiline_forward_pack_v1_pilot_v6_mixed_three_types.npz"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_mixed_pilot_v6_training_gate_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v6_training_gate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v6_training_gate_epoch_log.csv"
DEFAULT_DEFECT_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v6_defect_type_summary.csv"
DEFAULT_ANGLE_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v6_angle_summary.csv"
DEFAULT_VERTEX_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_mixed_pilot_v6_vertex_count_summary.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_mixed_pilot_v6_gate"

THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPECTED_SPLITS = {"train": 152, "val": 38, "test": 38}
EXPECTED_DEFECTS = {"rectangular_notch": 120, "rotated_rect": 48, "polygon": 60}
EXPECTED_ROT_ANGLES = {-30.0, -20.0, -10.0, 10.0, 20.0, 30.0}
EXPECTED_VERTEX_COUNTS = {4: 20, 5: 20, 6: 20}

METRIC_FIELDS = [
    "source_index",
    "sample_id",
    "split",
    "defect_type",
    "angle_deg",
    "vertex_count",
    "threshold",
    "iou",
    "dice",
    "area_error",
    "center_error",
    "pred_area",
    "true_area",
    "pred_area_zero",
    "bce_loss",
    "dice_loss",
    "total_loss",
    "prob_min",
    "prob_max",
    "prob_mean",
    "notes",
]

GROUP_FIELDS = [
    "group",
    "split",
    "sample_count",
    "iou_mean",
    "dice_mean",
    "area_error_mean",
    "center_error_mean",
    "pred_area_mean",
    "true_area_mean",
    "pred_area_zero_sum",
    "total_loss_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mixed three-type COMSOL pilot_v6 training gate.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--defect-summary", type=Path, default=DEFAULT_DEFECT_SUMMARY)
    parser.add_argument("--angle-summary", type=Path, default=DEFAULT_ANGLE_SUMMARY)
    parser.add_argument("--vertex-summary", type=Path, default=DEFAULT_VERTEX_SUMMARY)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_geometry(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    parsed = json.loads(tiny.as_text(value))
    if not isinstance(parsed, dict):
        raise RuntimeError("geometry_params entry is not a JSON object")
    return parsed


def rasterize_rect(geometry: dict[str, Any], mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    angle_rad = float(geometry.get("angle_rad") or 0.0)
    center_x = float(geometry["center_x"])
    center_y = float(geometry["center_y"])
    width = float(geometry["width"])
    length = float(geometry["length"])
    xx, yy = np.meshgrid(mask_x, mask_y, indexing="xy")
    dx = xx - center_x
    dy = yy - center_y
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return (np.abs(local_x) <= width / 2.0) & (np.abs(local_y) <= length / 2.0)


def validate_mixed_v6_npz(npz_path: Path) -> dict[str, Any]:
    required = [
        "delta_bz",
        "bz_defect",
        "bz_no_defect",
        "masks",
        "sensor_x",
        "scan_line_y",
        "mask_x",
        "mask_y",
        "defect_types",
        "sample_ids",
        "geometry_params",
        "metadata",
        "split",
    ]
    data = np.load(npz_path, allow_pickle=True)
    missing = [field for field in required if field not in data.files]
    if missing:
        raise RuntimeError(f"missing NPZ fields: {missing}")

    delta_bz = data["delta_bz"]
    bz_defect = data["bz_defect"]
    bz_no_defect = data["bz_no_defect"]
    masks = data["masks"]
    sensor_x = data["sensor_x"]
    scan_line_y = data["scan_line_y"]
    mask_x = data["mask_x"]
    mask_y = data["mask_y"]
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    split_values = np.array([tiny.as_text(item) for item in data["split"].tolist()])
    geometries = [parse_geometry(item) for item in data["geometry_params"].tolist()]

    if delta_bz.shape != (228, 3, 201):
        raise RuntimeError(f"unexpected delta_bz shape: {delta_bz.shape}")
    if bz_defect.shape != delta_bz.shape or bz_no_defect.shape != delta_bz.shape:
        raise RuntimeError("bz_defect / bz_no_defect shape mismatch with delta_bz")
    if masks.shape != (228, 64, 128):
        raise RuntimeError(f"unexpected masks shape: {masks.shape}")
    if sensor_x.shape != (201,) or scan_line_y.shape != (3,) or mask_x.shape != (128,) or mask_y.shape != (64,):
        raise RuntimeError("coordinate shape mismatch")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise RuntimeError("sample_id values are not unique")

    split_counts = {name: int(np.sum(split_values == name)) for name in ("train", "val", "test")}
    defect_counts = dict(Counter(defect_types.tolist()))
    if split_counts != EXPECTED_SPLITS:
        raise RuntimeError(f"unexpected split counts: {split_counts}")
    if defect_counts != EXPECTED_DEFECTS:
        raise RuntimeError(f"unexpected defect_type distribution: {defect_counts}")

    for name, arr in {
        "delta_bz": delta_bz,
        "bz_defect": bz_defect,
        "bz_no_defect": bz_no_defect,
        "masks": masks,
        "sensor_x": sensor_x,
        "scan_line_y": scan_line_y,
        "mask_x": mask_x,
        "mask_y": mask_y,
    }.items():
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} contains NaN or inf")
    delta_matches = bool(np.allclose(delta_bz, bz_defect - bz_no_defect, rtol=1e-9, atol=1e-12))
    if not delta_matches:
        raise RuntimeError("delta_bz does not match bz_defect - bz_no_defect")
    if np.any(np.sum(masks > 0, axis=(1, 2)) <= 0):
        raise RuntimeError("one or more masks are empty")
    if not np.any(np.abs(delta_bz) > 0):
        raise RuntimeError("delta_bz is all zero")
    for name, coords in (("sensor_x", sensor_x), ("scan_line_y", scan_line_y), ("mask_x", mask_x), ("mask_y", mask_y)):
        if not np.all(np.diff(coords) > 0):
            raise RuntimeError(f"{name} is not strictly increasing")
    max_line_diff = max(
        float(np.max(np.abs(delta_bz[:, line_index, :] - delta_bz[:, 0, :])))
        for line_index in range(1, delta_bz.shape[1])
    )
    if max_line_diff <= 1e-12:
        raise RuntimeError("scan lines are numerically identical")

    split_defect_counts = {
        split_name: {
            defect_name: int(np.sum((split_values == split_name) & (defect_types == defect_name)))
            for defect_name in EXPECTED_DEFECTS
        }
        for split_name in ("train", "val", "test")
    }
    if any(any(count <= 0 for count in values.values()) for values in split_defect_counts.values()):
        raise RuntimeError(f"each split must contain all three defect types: {split_defect_counts}")

    angles = np.zeros(len(defect_types), dtype=np.float32)
    vertex_counts = np.zeros(len(defect_types), dtype=np.int32)
    geom_ious: list[float] = []
    for index, geometry in enumerate(geometries):
        required_keys = {
            "defect_type",
            "center_x",
            "center_y",
            "width",
            "length",
            "depth",
            "angle",
            "angle_deg",
            "angle_rad",
            "polygon_vertices",
            "vertex_count",
            "polygon_area",
            "units",
            "source_pack",
            "source_sample_id",
        }
        missing_geom = sorted(required_keys - set(geometry))
        if missing_geom:
            raise RuntimeError(f"geometry_params missing keys at sample {index}: {missing_geom}")
        if geometry["defect_type"] != defect_types[index]:
            raise RuntimeError(f"geometry defect_type mismatch at sample {index}")
        angle = float(geometry.get("angle_deg") or 0.0)
        angles[index] = angle
        vertex_counts[index] = int(geometry.get("vertex_count") or 0)
        if defect_types[index] == "rectangular_notch" and abs(angle) > 1e-6:
            raise RuntimeError(f"rectangular_notch angle must be 0 at sample {index}")
        if defect_types[index] in {"rectangular_notch", "rotated_rect"}:
            raster = rasterize_rect(geometry, mask_x, mask_y)
        else:
            vertices = np.array(geometry["polygon_vertices"], dtype=np.float64)
            raster = poly_gate.rasterize_polygon(vertices, mask_x, mask_y)
            if vertex_counts[index] not in {4, 5, 6}:
                raise RuntimeError(f"invalid polygon vertex_count at sample {index}: {vertex_counts[index]}")
        stored = masks[index].astype(bool)
        union = np.logical_or(raster, stored).sum()
        geom_ious.append(1.0 if union == 0 else float(np.logical_and(raster, stored).sum() / union))
    if min(geom_ious) < 0.999:
        raise RuntimeError(f"geometry_params do not explain masks: min IoU={min(geom_ious):.6f}")

    rotated_angles = set(float(value) for value in angles[defect_types == "rotated_rect"].tolist())
    if rotated_angles != EXPECTED_ROT_ANGLES:
        raise RuntimeError(f"unexpected rotated angle values: {sorted(rotated_angles)}")
    polygon_vertices = dict(Counter(vertex_counts[defect_types == "polygon"].tolist()))
    if polygon_vertices != EXPECTED_VERTEX_COUNTS:
        raise RuntimeError(f"unexpected polygon vertex_count distribution: {polygon_vertices}")

    splits = {name: np.where(split_values == name)[0].tolist() for name in ("train", "val", "test")}
    return {
        "data": data,
        "missing": missing,
        "split_counts": split_counts,
        "defect_counts": defect_counts,
        "split_defect_counts": split_defect_counts,
        "delta_matches": delta_matches,
        "max_line_diff": max_line_diff,
        "geometry_mask_ious": geom_ious,
        "angles": angles,
        "vertex_counts": vertex_counts,
        "rotated_angle_distribution": dict(Counter(angles[defect_types == "rotated_rect"].tolist())),
        "polygon_vertex_distribution": polygon_vertices,
        "geometries": geometries,
        "splits": splits,
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    threshold: float,
    sample_ids: np.ndarray,
    defect_types: np.ndarray,
    angles: np.ndarray,
    vertex_counts: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    probs: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for signals, masks, indices in loader:
            signals = signals.to(device)
            masks = masks.to(device)
            logits = model(signals)
            total, bce, dice = tiny.loss_components(logits, masks)
            prob = torch.sigmoid(logits).cpu().numpy()[0]
            target = masks.cpu().numpy()[0]
            index = int(indices.item())
            probs[index] = prob
            metrics = tiny.sample_metrics(prob, target, threshold)
            rows.append(
                {
                    "source_index": index,
                    "sample_id": tiny.as_text(sample_ids[index]),
                    "defect_type": tiny.as_text(defect_types[index]),
                    "angle_deg": float(angles[index]),
                    "vertex_count": int(vertex_counts[index]),
                    "threshold": threshold,
                    **metrics,
                    "bce_loss": float(bce.item()),
                    "dice_loss": float(dice.item()),
                    "total_loss": float(total.item()),
                    "prob_min": float(prob.min()),
                    "prob_max": float(prob.max()),
                    "prob_mean": float(prob.mean()),
                    "notes": "mixed_pilot_v6_training_gate_only",
                }
            )
    return rows, probs


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def summarize_rows(rows: list[dict[str, Any]], group_name: str, split_name: str) -> dict[str, Any]:
    return {
        "group": group_name,
        "split": split_name,
        "sample_count": len(rows),
        "iou_mean": mean_or_nan([float(row["iou"]) for row in rows]),
        "dice_mean": mean_or_nan([float(row["dice"]) for row in rows]),
        "area_error_mean": mean_or_nan([float(row["area_error"]) for row in rows]),
        "center_error_mean": mean_or_nan([float(row["center_error"]) for row in rows if str(row["center_error"]).lower() != "nan"]),
        "pred_area_mean": mean_or_nan([float(row["pred_area"]) for row in rows]),
        "true_area_mean": mean_or_nan([float(row["true_area"]) for row in rows]),
        "pred_area_zero_sum": int(sum(int(row["pred_area_zero"]) for row in rows)),
        "total_loss_mean": mean_or_nan([float(row["total_loss"]) for row in rows]),
    }


def split_summary_rows(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        split_name: summarize_rows([row for row in metric_rows if row["split"] == split_name], "all", split_name)
        for split_name in ("train", "val", "test")
    }


def group_summary(metric_rows: list[dict[str, Any]], key: str, values: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "all"):
        source = metric_rows if split_name == "all" else [row for row in metric_rows if row["split"] == split_name]
        for value in values:
            selected = [row for row in source if row[key] == value]
            rows.append(summarize_rows(selected, f"{key}={value}", split_name))
    return rows


def angle_summary(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = group_summary(
        [row for row in metric_rows if row["defect_type"] in {"rectangular_notch", "rotated_rect"}],
        "angle_deg",
        [0.0, -30.0, -20.0, -10.0, 10.0, 20.0, 30.0],
    )
    return rows


def choose_preview_indices(metric_rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected:
            selected.append(index)

    val_test = [row for row in metric_rows if row["split"] in {"val", "test"}]
    for defect_type in ("rectangular_notch", "rotated_rect", "polygon"):
        rows = [row for row in val_test if row["defect_type"] == defect_type]
        if rows:
            add(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
            add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    for angle in [-30.0, -20.0, -10.0, 10.0, 20.0, 30.0]:
        rows = [row for row in val_test if row["defect_type"] == "rotated_rect" and float(row["angle_deg"]) == angle]
        if rows:
            add(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
    for vertex_count in [4, 5, 6]:
        rows = [row for row in val_test if row["defect_type"] == "polygon" and int(row["vertex_count"]) == vertex_count]
        if rows:
            add(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
    for reverse in (True, False):
        for row in sorted(val_test, key=lambda item: float(item["dice"]), reverse=reverse)[:8]:
            add(int(row["source_index"]))
    return selected[:24]


def pixel_vertices(vertices: np.ndarray, mask_x: np.ndarray, mask_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = (vertices[:, 0] - mask_x[0]) / (mask_x[-1] - mask_x[0]) * (len(mask_x) - 1)
    py = (vertices[:, 1] - mask_y[0]) / (mask_y[-1] - mask_y[0]) * (len(mask_y) - 1)
    return np.r_[px, px[0]], np.r_[py, py[0]]


def make_previews(
    preview_dir: Path,
    probs: dict[int, np.ndarray],
    masks: np.ndarray,
    signals: np.ndarray,
    sensor_x: np.ndarray,
    scan_line_y: np.ndarray,
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    geometries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    threshold: float,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    rows_by_index = {int(row["source_index"]): row for row in rows}
    for index, prob in probs.items():
        row = rows_by_index[index]
        pred = prob >= threshold
        true = masks[index] >= 0.5
        overlay = np.zeros((*true.shape, 3), dtype=np.float32)
        overlay[..., 1] = true.astype(np.float32)
        overlay[..., 0] = pred.astype(np.float32)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for line_index, y in enumerate(scan_line_y):
            axes[0, 0].plot(sensor_x, signals[index, line_index], label=f"y={y:.4g} m")
        axes[0, 0].set_title("delta_bz scan lines")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].imshow(true, cmap="gray")
        axes[0, 1].set_title("true mask")
        axes[0, 2].imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[0, 2].set_title("predicted probability")
        axes[1, 0].imshow(pred, cmap="gray")
        axes[1, 0].set_title(f"pred mask @ {threshold:.2f}")
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("overlay red=pred green=true")
        axes[1, 2].axis("off")
        geometry = geometries[index]
        if row["defect_type"] == "polygon" and geometry.get("polygon_vertices") is not None:
            vx, vy = pixel_vertices(np.array(geometry["polygon_vertices"], dtype=np.float64), mask_x, mask_y)
            for ax in (axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
                ax.plot(vx, vy, color="cyan", linewidth=1.0)
        extra = f"v={row['vertex_count']}" if row["defect_type"] == "polygon" else f"angle={float(row['angle_deg']):.1f}"
        axes[1, 2].text(
            0.0,
            0.95,
            "\n".join(
                [
                    f"sample_id: {row['sample_id']}",
                    f"split: {row['split']}",
                    f"type: {row['defect_type']}",
                    extra,
                    f"IoU: {float(row['iou']):.4f}",
                    f"Dice: {float(row['dice']):.4f}",
                    f"area_error: {float(row['area_error']):.4f}",
                ]
            ),
            va="top",
            fontsize=9,
        )
        for ax in axes.flat:
            if ax is not axes[0, 0] and ax is not axes[1, 2]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(preview_dir / f"{row['sample_id']}_{row['split']}_{row['defect_type']}.png", dpi=140)
        plt.close(fig)


def build_summary(context: dict[str, Any]) -> str:
    lines = [
        "# Stage 20.20 COMSOL mixed pilot_v6 training gate",
        "",
        "## 1. NPZ / schema checks",
        "",
        f"- mixed pilot_v6 NPZ readable: {context['npz_readable']}",
        f"- schema complete: {context['schema_complete']}",
        f"- split is 152 / 38 / 38: {context['split_is_152_38_38']} ({context['split_counts']})",
        f"- defect_type distribution correct: {context['defect_distribution_ok']} ({context['defect_distribution']})",
        f"- each split has all three defect types: {context['each_split_has_three_types']} ({context['split_defect_counts']})",
        f"- rotated_rect angle distribution: {context['rotated_angle_distribution']}",
        f"- polygon vertex_count distribution: {context['polygon_vertex_distribution']}",
        f"- geometry_params explain masks: {context['geometry_mask_ious_summary']}",
        f"- delta_bz input shape: {context['delta_bz_shape']}",
        f"- mask output shape: {context['masks_shape']}",
        f"- delta_bz equals bz_defect - bz_no_defect: {context['delta_matches']}",
        f"- scan lines different: {context['scan_lines_different']}",
        "",
        "## 2. Normalization / loader",
        "",
        "- Normalization uses only train split delta_bz statistics, per n_lines channel over train samples and signal length.",
        f"- train mean shape: {context['train_mean_shape']}",
        f"- train std shape: {context['train_std_shape']}",
        f"- normalization train-only: {context['normalization_train_only']}",
        "",
        "## 3. Training gate",
        "",
        "- Model: lightweight Conv1d encoder for `(3, 201)` delta_bz, latent projection to `4x8`, ConvTranspose2d decoder to `(64, 128)` mask logits.",
        "- Loss: BCEWithLogits + soft Dice. No defect_type, angle, vertex_count, or geometry parameter supervision.",
        f"- epochs: {context['epochs']}",
        f"- batch_size: {context['batch_size']}",
        f"- selected threshold: {context['threshold']}",
        f"- validation threshold scores: {context['threshold_scores']}",
        f"- best validation epoch: {context['best_epoch']}",
        f"- train loop ok: {context['train_loop_ok']}",
        f"- train loss decreased: {context['train_loss_decreased']} (initial={context['initial_train_loss']:.6f}, final={context['final_train_loss']:.6f})",
        f"- can fit 152 train samples: {context['can_fit_train_samples']}",
        "",
        "## 4. Pilot metrics",
        "",
        f"- train: {context['train_metrics']}",
        f"- val: {context['val_metrics']}",
        f"- test: {context['test_metrics']}",
        f"- defect_type summary: {context['defect_summary_path']}",
        f"- angle summary: {context['angle_summary_path']}",
        f"- vertex_count summary: {context['vertex_summary_path']}",
        f"- type imbalance issue: {context['type_imbalance_issue']}",
        f"- per-angle issue: {context['per_angle_issue']}",
        f"- vertex_count issue: {context['vertex_count_issue']}",
        f"- empty predictions: {context['has_empty_prediction']}",
        f"- full-image predictions: {context['has_full_prediction']}",
        f"- NaN detected: {context['has_nan']}",
        "",
        "## 5. Preview",
        "",
        f"- preview generated: {context['preview_generated']}",
        f"- preview dir: {context['preview_dir']}",
        f"- preview sample ids: {context['preview_sample_ids']}",
        "",
        "## 6. Conclusion",
        "",
        "- This result validates the mixed single-defect-type pilot_v6 read -> dataset loader -> train-only normalization -> training -> validation threshold selection -> test smoke evaluation -> preview chain.",
        "- It is not a v3_complex formal model result, not a candidate, and does not update CURRENT_BASELINE.",
        "- The 228-sample mixed pack is usable for next-stage data expansion, but remains pilot-level and single-defect only.",
        f"- Recommended next step: {context['next_step_recommendation']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("--epochs must be between 1 and 200 for this mixed pilot_v6 gate.")
    tiny.set_seed(args.seed)

    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    metrics_path = resolve(args.metrics)
    epoch_log_path = resolve(args.epoch_log)
    defect_summary_path = resolve(args.defect_summary)
    angle_summary_path = resolve(args.angle_summary)
    vertex_summary_path = resolve(args.vertex_summary)
    preview_dir = resolve(args.preview_dir)

    validation = validate_mixed_v6_npz(npz_path)
    data = validation["data"]
    splits = validation["splits"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    angles = validation["angles"]
    vertex_counts = validation["vertex_counts"]
    geometries = validation["geometries"]
    scan_line_y = data["scan_line_y"].astype(np.float64)
    sensor_x = data["sensor_x"].astype(np.float64)
    mask_x = data["mask_x"].astype(np.float64)
    mask_y = data["mask_y"].astype(np.float64)

    train_mean = delta_bz[splits["train"]].mean(axis=(0, 2), keepdims=True)
    train_std = np.maximum(delta_bz[splits["train"]].std(axis=(0, 2), keepdims=True), 1e-8)
    normalized = (delta_bz - train_mean) / train_std

    train_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["train"])
    val_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["val"])
    test_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["test"])
    all_dataset = tiny.ComsolSmokeDataset(normalized, masks, list(range(delta_bz.shape[0])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tiny.TinyComsolMaskDecoder(delta_bz.shape[1], masks.shape[1], masks.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    epoch_rows: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    initial_train_loss: float | None = None
    final_train_loss: float | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_totals: list[float] = []
        batch_bces: list[float] = []
        batch_dices: list[float] = []
        for signals, target, _ in train_loader:
            signals = signals.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            total, bce, dice = tiny.loss_components(logits, target)
            total.backward()
            optimizer.step()
            batch_totals.append(float(total.item()))
            batch_bces.append(float(bce.item()))
            batch_dices.append(float(dice.item()))

        train_loss = float(np.mean(batch_totals))
        train_bce = float(np.mean(batch_bces))
        train_dice_loss = float(np.mean(batch_dices))
        val_loss, val_bce, val_dice_loss = tiny.evaluate_loss(model, val_dataset, device)
        val_rows_at_half, _ = evaluate_model(model, val_dataset, device, 0.5, sample_ids, defect_types, angles, vertex_counts)
        val_iou = float(np.mean([row["iou"] for row in val_rows_at_half]))
        val_dice_metric = float(np.mean([row["dice"] for row in val_rows_at_half]))
        val_area_error = float(np.mean([row["area_error"] for row in val_rows_at_half]))
        val_score = val_iou + val_dice_metric - val_area_error
        if val_score > best_score:
            best_score = val_score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        if initial_train_loss is None:
            initial_train_loss = train_loss
        final_train_loss = train_loss
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_bce": train_bce,
                "train_dice_loss": train_dice_loss,
                "val_loss": val_loss,
                "val_bce": val_bce,
                "val_dice_loss": val_dice_loss,
                "val_iou_at_0_5": val_iou,
                "val_dice_at_0_5": val_dice_metric,
                "val_area_error_at_0_5": val_area_error,
                "val_score_at_0_5": val_score,
            }
        )

    model.load_state_dict(best_state)
    selected_threshold, threshold_scores = tiny.select_threshold(model, val_dataset, device, THRESHOLD_CANDIDATES)

    metric_rows: list[dict[str, Any]] = []
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        rows, _ = evaluate_model(model, dataset, device, selected_threshold, sample_ids, defect_types, angles, vertex_counts)
        for row in rows:
            row["split"] = split_name
        metric_rows.extend(rows)

    write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    write_csv(epoch_log_path, epoch_rows, tiny.EPOCH_FIELDS)
    defect_rows = group_summary(metric_rows, "defect_type", ["rectangular_notch", "rotated_rect", "polygon"])
    angle_rows = angle_summary(metric_rows)
    vertex_rows = group_summary([row for row in metric_rows if row["defect_type"] == "polygon"], "vertex_count", [4, 5, 6])
    write_csv(defect_summary_path, defect_rows, GROUP_FIELDS)
    write_csv(angle_summary_path, angle_rows, GROUP_FIELDS)
    write_csv(vertex_summary_path, vertex_rows, GROUP_FIELDS)

    selected_preview_indices = choose_preview_indices(metric_rows)
    all_rows, all_probs = evaluate_model(model, all_dataset, device, selected_threshold, sample_ids, defect_types, angles, vertex_counts)
    split_by_index = {}
    for split_name, indices in splits.items():
        for index in indices:
            split_by_index[index] = split_name
    for row in all_rows:
        row["split"] = split_by_index[int(row["source_index"])]
    selected_probs = {index: all_probs[index] for index in selected_preview_indices}
    make_previews(preview_dir, selected_probs, masks, delta_bz, sensor_x, scan_line_y, mask_x, mask_y, geometries, all_rows, selected_threshold)

    split_metrics = split_summary_rows(metric_rows)
    train_metrics = split_metrics["train"]
    val_metrics = split_metrics["val"]
    test_metrics = split_metrics["test"]
    train_loss_decreased = bool(final_train_loss is not None and initial_train_loss is not None and final_train_loss < initial_train_loss)
    can_fit_train_samples = bool(
        train_loss_decreased
        and float(train_metrics.get("dice_mean", 0.0)) > 0.70
        and float(train_metrics.get("iou_mean", 0.0)) > 0.55
    )
    full_area = masks.shape[1] * masks.shape[2]
    all_defect = {row["group"]: row for row in defect_rows if row["split"] == "all"}
    defect_dices = [float(row["dice_mean"]) for row in all_defect.values()]
    type_imbalance_issue = bool(max(defect_dices) - min(defect_dices) > 0.25)
    per_angle_issue = any(float(row["dice_mean"]) < 0.45 or int(row["pred_area_zero_sum"]) > 0 for row in angle_rows if row["split"] == "all")
    vertex_count_issue = any(float(row["dice_mean"]) < 0.45 or int(row["pred_area_zero_sum"]) > 0 for row in vertex_rows if row["split"] == "all")
    next_step = (
        "Expand sample count for the three single-defect types before adding true multi_defect."
        if not (type_imbalance_issue or per_angle_issue or vertex_count_issue)
        else "Inspect weak defect_type/angle/vertex groups before deciding whether to expand samples or model capacity."
    )

    context = {
        "npz_readable": True,
        "schema_complete": len(validation["missing"]) == 0,
        "split_is_152_38_38": validation["split_counts"] == EXPECTED_SPLITS,
        "split_counts": validation["split_counts"],
        "defect_distribution_ok": validation["defect_counts"] == EXPECTED_DEFECTS,
        "defect_distribution": validation["defect_counts"],
        "each_split_has_three_types": all(all(count > 0 for count in values.values()) for values in validation["split_defect_counts"].values()),
        "split_defect_counts": validation["split_defect_counts"],
        "rotated_angle_distribution": validation["rotated_angle_distribution"],
        "polygon_vertex_distribution": validation["polygon_vertex_distribution"],
        "geometry_mask_ious_summary": {
            "min": float(np.min(validation["geometry_mask_ious"])),
            "max": float(np.max(validation["geometry_mask_ious"])),
            "mean": float(np.mean(validation["geometry_mask_ious"])),
        },
        "delta_bz_shape": tuple(delta_bz.shape),
        "masks_shape": tuple(masks.shape),
        "delta_matches": validation["delta_matches"],
        "scan_lines_different": validation["max_line_diff"] > 1e-12,
        "train_mean_shape": tuple(train_mean.shape),
        "train_std_shape": tuple(train_std.shape),
        "normalization_train_only": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "threshold": selected_threshold,
        "threshold_scores": threshold_scores,
        "best_epoch": best_epoch,
        "train_loop_ok": True,
        "train_loss_decreased": train_loss_decreased,
        "initial_train_loss": float(initial_train_loss if initial_train_loss is not None else float("nan")),
        "final_train_loss": float(final_train_loss if final_train_loss is not None else float("nan")),
        "can_fit_train_samples": can_fit_train_samples,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "defect_summary_path": str(defect_summary_path),
        "angle_summary_path": str(angle_summary_path),
        "vertex_summary_path": str(vertex_summary_path),
        "type_imbalance_issue": type_imbalance_issue,
        "per_angle_issue": bool(per_angle_issue),
        "vertex_count_issue": bool(vertex_count_issue),
        "has_empty_prediction": any(int(row["pred_area_zero"]) == 1 for row in metric_rows),
        "has_full_prediction": any(int(row["pred_area"]) >= full_area for row in metric_rows),
        "has_nan": any(not np.isfinite(float(row["total_loss"])) for row in metric_rows),
        "preview_generated": True,
        "preview_dir": str(preview_dir),
        "preview_sample_ids": [tiny.as_text(sample_ids[index]) for index in selected_preview_indices],
        "next_step_recommendation": next_step,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(context), encoding="utf-8")
    print(json.dumps(context, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
