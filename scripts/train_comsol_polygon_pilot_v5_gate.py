from __future__ import annotations

import argparse
import csv
import json
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data"
    / "comsol_mfl"
    / "prepared"
    / "comsol_single_defect_multiline_forward_pack_v1_pilot_v5_polygon.npz"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_polygon_pilot_v5_training_gate_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_polygon_pilot_v5_training_gate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_polygon_pilot_v5_training_gate_epoch_log.csv"
DEFAULT_VERTEX_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_polygon_pilot_v5_vertex_count_summary.csv"
DEFAULT_AREA_BIN_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_polygon_pilot_v5_area_bin_summary.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_polygon_pilot_v5_gate"

THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPECTED_SPLIT_COUNTS = {"train": 40, "val": 10, "test": 10}
EXPECTED_VERTEX_COUNTS = {4: 20, 5: 20, 6: 20}

METRIC_FIELDS = [
    "source_index",
    "sample_id",
    "split",
    "defect_type",
    "vertex_count",
    "polygon_area",
    "area_bin",
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
    parser = argparse.ArgumentParser(description="Run a polygon COMSOL pilot_v5 training gate.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--vertex-summary", type=Path, default=DEFAULT_VERTEX_SUMMARY)
    parser.add_argument("--area-bin-summary", type=Path, default=DEFAULT_AREA_BIN_SUMMARY)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
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


def polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def segment_intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def polygon_self_intersects(vertices: np.ndarray) -> bool:
    count = len(vertices)
    for i in range(count):
        a = vertices[i]
        b = vertices[(i + 1) % count]
        for j in range(i + 1, count):
            if abs(i - j) <= 1 or (i == 0 and j == count - 1):
                continue
            c = vertices[j]
            d = vertices[(j + 1) % count]
            if segment_intersects(a, b, c, d):
                return True
    return False


def rasterize_polygon(vertices: np.ndarray, mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    yy, xx = np.meshgrid(mask_y, mask_x, indexing="ij")
    inside = np.zeros(xx.shape, dtype=bool)
    xj = vertices[-1, 0]
    yj = vertices[-1, 1]
    for xi, yi in vertices:
        crosses = ((yi > yy) != (yj > yy)) & (xx < (xj - xi) * (yy - yi) / ((yj - yi) + 1e-30) + xi)
        inside ^= crosses
        xj, yj = xi, yi
    return inside


def area_bin_labels(areas: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    q1, q2 = np.quantile(areas, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.empty(areas.shape, dtype=object)
    labels[areas <= q1] = "small"
    labels[(areas > q1) & (areas <= q2)] = "medium"
    labels[areas > q2] = "large"
    return labels, {"small_max": float(q1), "medium_max": float(q2)}


def validate_polygon_npz(npz_path: Path) -> dict[str, Any]:
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
    geometry_params = [parse_geometry(item) for item in data["geometry_params"].tolist()]

    if delta_bz.shape != (60, 3, 201):
        raise RuntimeError(f"unexpected delta_bz shape: {delta_bz.shape}")
    if bz_defect.shape != delta_bz.shape or bz_no_defect.shape != delta_bz.shape:
        raise RuntimeError("bz_defect / bz_no_defect shape mismatch with delta_bz")
    if masks.shape != (60, 64, 128):
        raise RuntimeError(f"unexpected masks shape: {masks.shape}")
    if sensor_x.shape != (201,) or scan_line_y.shape != (3,):
        raise RuntimeError("sensor_x / scan_line_y shape mismatch")
    if mask_x.shape != (128,) or mask_y.shape != (64,):
        raise RuntimeError("mask_x / mask_y shape mismatch")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise RuntimeError("sample_id values are not unique")

    split_counts = {name: int(np.sum(split_values == name)) for name in ("train", "val", "test")}
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"unexpected split counts: {split_counts}")
    defect_counts = {name: int(np.sum(defect_types == name)) for name in sorted(set(defect_types.tolist()))}
    if defect_counts != {"polygon": 60}:
        raise RuntimeError(f"unexpected defect_type distribution: {defect_counts}")

    numeric_arrays = {
        "delta_bz": delta_bz,
        "bz_defect": bz_defect,
        "bz_no_defect": bz_no_defect,
        "masks": masks,
        "sensor_x": sensor_x,
        "scan_line_y": scan_line_y,
        "mask_x": mask_x,
        "mask_y": mask_y,
    }
    finite = {name: bool(np.isfinite(value).all()) for name, value in numeric_arrays.items()}
    if not all(finite.values()):
        raise RuntimeError(f"non-finite numeric arrays: {finite}")
    delta_matches = bool(np.allclose(delta_bz, bz_defect - bz_no_defect, rtol=1e-9, atol=1e-12))
    if not delta_matches:
        raise RuntimeError("delta_bz does not match bz_defect - bz_no_defect")
    max_line_diff = max(
        float(np.max(np.abs(delta_bz[:, line_index, :] - delta_bz[:, 0, :])))
        for line_index in range(1, delta_bz.shape[1])
    )
    if max_line_diff <= 1e-12:
        raise RuntimeError("scan lines are numerically identical")
    if not np.all(masks.reshape(masks.shape[0], -1).sum(axis=1) > 0):
        raise RuntimeError("one or more masks are empty")
    if not np.any(np.abs(delta_bz) > 0):
        raise RuntimeError("delta_bz is all zero")
    for name, coords in (("sensor_x", sensor_x), ("scan_line_y", scan_line_y), ("mask_x", mask_x), ("mask_y", mask_y)):
        if not np.all(np.diff(coords) > 0):
            raise RuntimeError(f"{name} is not strictly increasing")

    vertex_counts: list[int] = []
    polygon_areas: list[float] = []
    geometry_mask_ious: list[float] = []
    invalid_polygons: list[int] = []
    out_of_bounds: list[int] = []
    for index, geometry in enumerate(geometry_params):
        if geometry.get("defect_type") != "polygon" or defect_types[index] != "polygon":
            raise RuntimeError(f"defect_type mismatch at sample {index}")
        for key in ("polygon_vertices", "vertex_count", "polygon_area_m2", "center_x_m", "center_y_m", "width_m", "length_m", "depth_m"):
            if key not in geometry:
                raise RuntimeError(f"geometry_params missing {key} at sample {index}")
        vertices = np.array(geometry["polygon_vertices"], dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise RuntimeError(f"polygon_vertices shape invalid at sample {index}: {vertices.shape}")
        vertex_count = int(geometry["vertex_count"])
        if vertex_count != vertices.shape[0] or vertex_count not in {4, 5, 6}:
            raise RuntimeError(f"vertex_count invalid at sample {index}: {vertex_count}")
        area = float(geometry["polygon_area_m2"])
        raster = rasterize_polygon(vertices, mask_x, mask_y)
        stored = masks[index].astype(bool)
        union = np.logical_or(raster, stored).sum()
        iou = 1.0 if union == 0 else float(np.logical_and(raster, stored).sum() / union)
        geometry_mask_ious.append(iou)
        if polygon_self_intersects(vertices) or polygon_area(vertices) <= 2.0e-5:
            invalid_polygons.append(index)
        if (
            vertices[:, 0].min() < mask_x.min()
            or vertices[:, 0].max() > mask_x.max()
            or vertices[:, 1].min() < mask_y.min()
            or vertices[:, 1].max() > mask_y.max()
        ):
            out_of_bounds.append(index)
        vertex_counts.append(vertex_count)
        polygon_areas.append(area)
    vertex_distribution = {count: vertex_counts.count(count) for count in sorted(set(vertex_counts))}
    if vertex_distribution != EXPECTED_VERTEX_COUNTS:
        raise RuntimeError(f"unexpected vertex_count distribution: {vertex_distribution}")
    if min(geometry_mask_ious) < 0.999:
        raise RuntimeError(f"polygon_vertices do not explain masks: min IoU={min(geometry_mask_ious):.6f}")
    if invalid_polygons:
        raise RuntimeError(f"invalid/self-intersecting polygon samples: {invalid_polygons}")
    if out_of_bounds:
        raise RuntimeError(f"polygon vertices out of mask coordinate range: {out_of_bounds}")

    split_vertex_counts = {
        split_name: {
            count: int(
                np.sum(
                    (split_values == split_name)
                    & (np.array(vertex_counts, dtype=np.int32) == count)
                )
            )
            for count in (4, 5, 6)
        }
        for split_name in ("train", "val", "test")
    }
    if any(any(value <= 0 for value in counts.values()) for counts in split_vertex_counts.values()):
        raise RuntimeError(f"each split must contain vertex_count 4/5/6: {split_vertex_counts}")
    splits = {name: np.where(split_values == name)[0].tolist() for name in ("train", "val", "test")}
    area_bins, area_bin_edges = area_bin_labels(np.array(polygon_areas, dtype=np.float64))
    return {
        "data": data,
        "missing": missing,
        "finite": finite,
        "delta_matches": delta_matches,
        "max_line_diff": max_line_diff,
        "geometry_params": geometry_params,
        "geometry_mask_ious": geometry_mask_ious,
        "vertex_counts": np.array(vertex_counts, dtype=np.int32),
        "polygon_areas": np.array(polygon_areas, dtype=np.float64),
        "vertex_distribution": vertex_distribution,
        "split_vertex_counts": split_vertex_counts,
        "area_bins": area_bins,
        "area_bin_edges": area_bin_edges,
        "splits": splits,
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    threshold: float,
    sample_ids: np.ndarray,
    defect_types: np.ndarray,
    vertex_counts: np.ndarray,
    polygon_areas: np.ndarray,
    area_bins: np.ndarray,
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
                    "vertex_count": int(vertex_counts[index]),
                    "polygon_area": float(polygon_areas[index]),
                    "area_bin": tiny.as_text(area_bins[index]),
                    "threshold": threshold,
                    **metrics,
                    "bce_loss": float(bce.item()),
                    "dice_loss": float(dice.item()),
                    "total_loss": float(total.item()),
                    "prob_min": float(prob.min()),
                    "prob_max": float(prob.max()),
                    "prob_mean": float(prob.mean()),
                    "notes": "polygon_pilot_v5_training_gate_only",
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


def group_summary(metric_rows: list[dict[str, Any]], key: str, values: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "all"):
        source = metric_rows if split_name == "all" else [row for row in metric_rows if row["split"] == split_name]
        for value in values:
            selected = [row for row in source if row[key] == value]
            rows.append(summarize_rows(selected, f"{key}={value}", split_name))
    return rows


def split_summary_rows(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        split_name: summarize_rows([row for row in metric_rows if row["split"] == split_name], "all", split_name)
        for split_name in ("train", "val", "test")
    }


def choose_preview_indices(metric_rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected:
            selected.append(index)

    val_test = [row for row in metric_rows if row["split"] in {"val", "test"}]
    for vertex_count in (4, 5, 6):
        rows = [row for row in val_test if int(row["vertex_count"]) == vertex_count]
        if rows:
            add(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
            add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    for split_name in ("val", "test"):
        rows = [row for row in metric_rows if row["split"] == split_name]
        if rows:
            add(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
            add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    train_rows = [row for row in metric_rows if row["split"] == "train"]
    for reverse in (True, False):
        for row in sorted(train_rows, key=lambda item: float(item["dice"]), reverse=reverse)[:3]:
            add(int(row["source_index"]))
    return selected[:16]


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
        vertices = np.array(geometries[index]["polygon_vertices"], dtype=np.float64)
        vx, vy = pixel_vertices(vertices, mask_x, mask_y)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for line_index, y in enumerate(scan_line_y):
            axes[0, 0].plot(sensor_x, signals[index, line_index], label=f"y={y:.4g} m")
        axes[0, 0].set_title("delta_bz scan lines")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].imshow(true, cmap="gray")
        axes[0, 1].plot(vx, vy, color="cyan", linewidth=1.3)
        axes[0, 1].set_title("true mask + polygon")
        axes[0, 2].imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[0, 2].plot(vx, vy, color="white", linewidth=1.0)
        axes[0, 2].set_title("predicted probability")
        axes[1, 0].imshow(pred, cmap="gray")
        axes[1, 0].plot(vx, vy, color="cyan", linewidth=1.0)
        axes[1, 0].set_title(f"pred mask @ {threshold:.2f}")
        axes[1, 1].imshow(overlay)
        axes[1, 1].plot(vx, vy, color="cyan", linewidth=1.0)
        axes[1, 1].set_title("overlay red=pred green=true")
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.0,
            0.95,
            "\n".join(
                [
                    f"sample_id: {row['sample_id']}",
                    f"split: {row['split']}",
                    f"vertex_count: {row['vertex_count']}",
                    f"IoU: {float(row['iou']):.4f}",
                    f"Dice: {float(row['dice']):.4f}",
                    f"area_error: {float(row['area_error']):.4f}",
                ]
            ),
            va="top",
            fontsize=10,
        )
        for ax in axes.flat:
            if ax is not axes[0, 0] and ax is not axes[1, 2]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(preview_dir / f"{row['sample_id']}_{row['split']}_v{row['vertex_count']}.png", dpi=140)
        plt.close(fig)


def build_summary(context: dict[str, Any]) -> str:
    lines = [
        "# Stage 20.19 COMSOL polygon pilot_v5 training gate",
        "",
        "## 1. NPZ / schema / polygon checks",
        "",
        f"- polygon pilot_v5 NPZ readable: {context['npz_readable']}",
        f"- schema complete: {context['schema_complete']}",
        f"- split is 40 / 10 / 10: {context['split_is_40_10_10']} ({context['split_counts']})",
        f"- vertex_count distribution correct: {context['vertex_distribution_ok']} ({context['vertex_distribution']})",
        f"- split vertex_count coverage: {context['split_vertex_counts']}",
        f"- polygon_vertices present and parseable: {context['polygon_vertices_parseable']}",
        f"- mask expresses polygon vertices: {context['mask_polygon_shape_ok']} ({context['geometry_mask_ious_summary']})",
        f"- polygon area range: {context['polygon_area_range']}",
        f"- delta_bz input shape: {context['delta_bz_shape']}",
        f"- mask output shape: {context['masks_shape']}",
        f"- delta_bz equals bz_defect - bz_no_defect: {context['delta_matches']}",
        f"- scan lines different: {context['scan_lines_different']}",
        f"- coordinates valid and monotonic: {context['coords_valid']}",
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
        "- Loss: BCEWithLogits + soft Dice. No vertex_count, polygon_vertices, or geometry parameter supervision.",
        f"- epochs: {context['epochs']}",
        f"- batch_size: {context['batch_size']}",
        "- Checkpoint selection: validation score = IoU + Dice - area_error. Test is final smoke evaluation only.",
        f"- selected threshold: {context['threshold']}",
        f"- validation threshold scores: {context['threshold_scores']}",
        f"- best validation epoch: {context['best_epoch']}",
        f"- train loop ok: {context['train_loop_ok']}",
        f"- train loss decreased: {context['train_loss_decreased']} (initial={context['initial_train_loss']:.6f}, final={context['final_train_loss']:.6f})",
        f"- can fit 40 train samples: {context['can_fit_train_samples']}",
        "",
        "## 4. Pilot metrics",
        "",
        f"- train: {context['train_metrics']}",
        f"- val: {context['val_metrics']}",
        f"- test: {context['test_metrics']}",
        f"- vertex_count summary: {context['vertex_summary_path']}",
        f"- area bin summary: {context['area_bin_summary_path']}",
        f"- vertex_count issue: {context['vertex_count_issue']}",
        f"- empty predictions: {context['has_empty_prediction']}",
        f"- full-image predictions: {context['has_full_prediction']}",
        f"- NaN detected: {context['has_nan']}",
        f"- visually still round/blob-like: {context['visually_round_blob_like']}",
        "",
        "## 5. Preview",
        "",
        f"- preview generated: {context['preview_generated']}",
        f"- preview dir: {context['preview_dir']}",
        f"- preview sample ids: {context['preview_sample_ids']}",
        "",
        "## 6. Conclusion",
        "",
        "- This result only validates the polygon pilot_v5 read -> dataset loader -> train-only normalization -> training -> validation threshold selection -> test smoke evaluation -> preview chain.",
        "- It is not a v3_complex formal model result, not a candidate, and does not update CURRENT_BASELINE.",
        "- The 60-sample polygon pack is usable for the next defect_type diversity stage, but remains pilot-level: polygon-only, no mixed rectangular/rotated/polygon training yet, no multi_defect, and limited polygon complexity.",
        f"- Recommended next step: {context['next_step_recommendation']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("--epochs must be between 1 and 200 for this polygon pilot_v5 gate.")
    tiny.set_seed(args.seed)

    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    metrics_path = resolve(args.metrics)
    epoch_log_path = resolve(args.epoch_log)
    vertex_summary_path = resolve(args.vertex_summary)
    area_bin_summary_path = resolve(args.area_bin_summary)
    preview_dir = resolve(args.preview_dir)

    validation = validate_polygon_npz(npz_path)
    data = validation["data"]
    splits = validation["splits"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    vertex_counts = validation["vertex_counts"]
    polygon_areas = validation["polygon_areas"]
    area_bins = validation["area_bins"]
    geometries = validation["geometry_params"]
    scan_line_y = data["scan_line_y"].astype(np.float64)
    sensor_x = data["sensor_x"].astype(np.float64)
    mask_x = data["mask_x"].astype(np.float64)
    mask_y = data["mask_y"].astype(np.float64)

    # Train-only normalization. Validation/test never contribute statistics.
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
        val_rows_at_half, _ = evaluate_model(
            model,
            val_dataset,
            device,
            0.5,
            sample_ids,
            defect_types,
            vertex_counts,
            polygon_areas,
            area_bins,
        )
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
        rows, _ = evaluate_model(
            model,
            dataset,
            device,
            selected_threshold,
            sample_ids,
            defect_types,
            vertex_counts,
            polygon_areas,
            area_bins,
        )
        for row in rows:
            row["split"] = split_name
        metric_rows.extend(rows)

    write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    write_csv(epoch_log_path, epoch_rows, tiny.EPOCH_FIELDS)
    vertex_rows = group_summary(metric_rows, "vertex_count", [4, 5, 6])
    area_rows = group_summary(metric_rows, "area_bin", ["small", "medium", "large"])
    write_csv(vertex_summary_path, vertex_rows, GROUP_FIELDS)
    write_csv(area_bin_summary_path, area_rows, GROUP_FIELDS)

    selected_preview_indices = choose_preview_indices(metric_rows)
    all_rows, all_probs = evaluate_model(
        model,
        all_dataset,
        device,
        selected_threshold,
        sample_ids,
        defect_types,
        vertex_counts,
        polygon_areas,
        area_bins,
    )
    split_by_index = {}
    for split_name, indices in splits.items():
        for index in indices:
            split_by_index[index] = split_name
    for row in all_rows:
        row["split"] = split_by_index[int(row["source_index"])]
    selected_probs = {index: all_probs[index] for index in selected_preview_indices}
    make_previews(
        preview_dir,
        selected_probs,
        masks,
        delta_bz,
        sensor_x,
        scan_line_y,
        mask_x,
        mask_y,
        geometries,
        all_rows,
        selected_threshold,
    )

    split_metrics = split_summary_rows(metric_rows)
    train_metrics = split_metrics["train"]
    val_metrics = split_metrics["val"]
    test_metrics = split_metrics["test"]
    train_loss_decreased = bool(
        final_train_loss is not None and initial_train_loss is not None and final_train_loss < initial_train_loss
    )
    can_fit_train_samples = bool(
        train_loss_decreased
        and float(train_metrics.get("dice_mean", 0.0)) > 0.70
        and float(train_metrics.get("iou_mean", 0.0)) > 0.55
    )
    full_area = masks.shape[1] * masks.shape[2]
    all_vertex_rows = [row for row in vertex_rows if row["split"] == "all"]
    vertex_count_issue = any(
        float(row["dice_mean"]) < 0.45 or int(row["pred_area_zero_sum"]) > 0
        for row in all_vertex_rows
    )
    visually_round_blob_like = bool(
        float(test_metrics.get("dice_mean", 0.0)) < 0.50
        or float(test_metrics.get("area_error_mean", 999.0)) > 0.75
    )
    next_step = (
        "Merge rectangular_notch + rotated_rect + polygon into a mixed pilot pack and run a mixed defect_type gate."
        if not vertex_count_issue
        else "Inspect weak vertex_count groups before merging, then decide whether to expand polygon samples or adjust model capacity."
    )

    context = {
        "npz_readable": True,
        "schema_complete": len(validation["missing"]) == 0,
        "split_is_40_10_10": {name: len(indices) for name, indices in splits.items()} == EXPECTED_SPLIT_COUNTS,
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "vertex_distribution_ok": validation["vertex_distribution"] == EXPECTED_VERTEX_COUNTS,
        "vertex_distribution": validation["vertex_distribution"],
        "split_vertex_counts": validation["split_vertex_counts"],
        "polygon_vertices_parseable": True,
        "mask_polygon_shape_ok": min(validation["geometry_mask_ious"]) >= 0.999,
        "geometry_mask_ious_summary": {
            "min": float(np.min(validation["geometry_mask_ious"])),
            "max": float(np.max(validation["geometry_mask_ious"])),
            "mean": float(np.mean(validation["geometry_mask_ious"])),
        },
        "polygon_area_range": [float(np.min(polygon_areas)), float(np.max(polygon_areas))],
        "delta_bz_shape": tuple(delta_bz.shape),
        "masks_shape": tuple(masks.shape),
        "delta_matches": validation["delta_matches"],
        "scan_lines_different": validation["max_line_diff"] > 1e-12,
        "coords_valid": True,
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
        "vertex_summary_path": str(vertex_summary_path),
        "area_bin_summary_path": str(area_bin_summary_path),
        "vertex_count_issue": bool(vertex_count_issue),
        "has_empty_prediction": any(int(row["pred_area_zero"]) == 1 for row in metric_rows),
        "has_full_prediction": any(int(row["pred_area"]) >= full_area for row in metric_rows),
        "has_nan": any(not np.isfinite(float(row["total_loss"])) for row in metric_rows),
        "visually_round_blob_like": visually_round_blob_like,
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
