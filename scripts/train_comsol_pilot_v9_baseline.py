from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_multiline_tiny_smoke as tiny  # noqa: E402
import train_comsol_polygon_pilot_v5_gate as poly_gate  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_pilot_v9_baseline_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_epoch_log.csv"
DEFAULT_SEED_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_seed_summary.csv"
DEFAULT_DEFECT_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_defect_type_summary.csv"
DEFAULT_ANGLE_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_angle_summary.csv"
DEFAULT_VERTEX_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_vertex_count_summary.csv"
DEFAULT_SOURCE_PACK_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_pilot_v9_baseline_source_pack_summary.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_pilot_v9_baseline"

THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
DEFAULT_SEEDS = [42, 123, 2026]
EXPECTED_SPLITS = {"train": 402, "val": 99, "test": 99}
EXPECTED_DEFECTS = {"rectangular_notch": 200, "rotated_rect": 200, "polygon": 200}
EXPECTED_ROT_ANGLES = {-30.0, -20.0, -10.0, 10.0, 20.0, 30.0}
EXPECTED_VERTEX_COUNTS = {4, 5, 6}

METRIC_FIELDS = [
    "seed",
    "source_index",
    "sample_id",
    "split",
    "defect_type",
    "source_pack",
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

EPOCH_FIELDS = tiny.EPOCH_FIELDS + [
    "seed",
    "best_val_threshold",
    "best_val_iou",
    "best_val_dice",
    "best_val_area_error",
    "best_val_score",
]

SEED_SUMMARY_FIELDS = [
    "seed",
    "best_epoch",
    "selected_threshold",
    "best_val_score",
    "train_loss_initial",
    "train_loss_final",
    "train_loss_decreased",
    "can_fit_train_samples",
    "split",
    "iou_mean",
    "dice_mean",
    "area_error_mean",
    "center_error_mean",
    "pred_area_mean",
    "true_area_mean",
    "pred_area_zero_sum",
    "total_loss_mean",
]

GROUP_FIELDS = [
    "seed",
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


class BaselineValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train COMSOL pilot_v9 data-domain baseline.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--seed-summary", type=Path, default=DEFAULT_SEED_SUMMARY)
    parser.add_argument("--defect-summary", type=Path, default=DEFAULT_DEFECT_SUMMARY)
    parser.add_argument("--angle-summary", type=Path, default=DEFAULT_ANGLE_SUMMARY)
    parser.add_argument("--vertex-summary", type=Path, default=DEFAULT_VERTEX_SUMMARY)
    parser.add_argument("--source-pack-summary", type=Path, default=DEFAULT_SOURCE_PACK_SUMMARY)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
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
        raise BaselineValidationError("geometry_params entry is not a JSON object")
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
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return (np.abs(local_x) <= width / 2.0) & (np.abs(local_y) <= length / 2.0)


def validate_pilot_v9_npz(npz_path: Path) -> dict[str, Any]:
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
        raise BaselineValidationError(f"missing NPZ fields: {missing}")

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

    if delta_bz.shape != (600, 3, 201):
        raise BaselineValidationError(f"unexpected delta_bz shape: {delta_bz.shape}")
    if bz_defect.shape != delta_bz.shape or bz_no_defect.shape != delta_bz.shape:
        raise BaselineValidationError("bz_defect / bz_no_defect shape mismatch")
    if masks.shape != (600, 64, 128):
        raise BaselineValidationError(f"unexpected masks shape: {masks.shape}")
    if sensor_x.shape != (201,) or scan_line_y.shape != (3,) or mask_x.shape != (128,) or mask_y.shape != (64,):
        raise BaselineValidationError("coordinate shape mismatch")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise BaselineValidationError("sample_id values are not unique")

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
            raise BaselineValidationError(f"{name} contains NaN or inf")
    if not np.allclose(delta_bz, bz_defect - bz_no_defect, rtol=1e-9, atol=1e-12):
        raise BaselineValidationError("delta_bz does not match bz_defect - bz_no_defect")
    if np.any(np.sum(masks > 0, axis=(1, 2)) <= 0):
        raise BaselineValidationError("one or more masks are empty")
    if np.sum(np.abs(delta_bz), axis=(1, 2)).min() <= 0:
        raise BaselineValidationError("one or more delta_bz samples are all zero")
    for name, coords in (("sensor_x", sensor_x), ("scan_line_y", scan_line_y), ("mask_x", mask_x), ("mask_y", mask_y)):
        if not np.all(np.diff(coords) > 0):
            raise BaselineValidationError(f"{name} is not strictly increasing")
    max_line_diff = max(
        float(np.max(np.abs(delta_bz[:, line_index, :] - delta_bz[:, 0, :])))
        for line_index in range(1, delta_bz.shape[1])
    )
    if max_line_diff <= 1e-12:
        raise BaselineValidationError("scan lines are numerically identical")

    split_counts = {name: int(np.sum(split_values == name)) for name in ("train", "val", "test")}
    defect_counts = dict(Counter(defect_types.tolist()))
    if split_counts != EXPECTED_SPLITS:
        raise BaselineValidationError(f"unexpected split counts: {split_counts}")
    if defect_counts != EXPECTED_DEFECTS:
        raise BaselineValidationError(f"unexpected defect_type distribution: {defect_counts}")

    angles = np.zeros(len(defect_types), dtype=np.float32)
    vertex_counts = np.zeros(len(defect_types), dtype=np.int32)
    source_packs: list[str] = []
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
            raise BaselineValidationError(f"geometry_params missing keys at sample {index}: {missing_geom}")
        if geometry["defect_type"] != defect_types[index]:
            raise BaselineValidationError(f"geometry defect_type mismatch at sample {index}")
        source_packs.append(tiny.as_text(geometry.get("source_pack", "")))
        angle = float(geometry.get("angle_deg") or 0.0)
        angles[index] = angle
        vertex_counts[index] = int(geometry.get("vertex_count") or 0)
        if defect_types[index] == "rectangular_notch" and abs(angle) > 1e-6:
            raise BaselineValidationError(f"rectangular_notch angle must be 0 at sample {index}")
        if defect_types[index] == "polygon":
            vertices = np.array(geometry["polygon_vertices"], dtype=np.float64)
            if vertex_counts[index] not in EXPECTED_VERTEX_COUNTS or vertices.shape[1] != 2:
                raise BaselineValidationError(f"invalid polygon vertices at sample {index}")
            raster = poly_gate.rasterize_polygon(vertices, mask_x, mask_y)
        else:
            raster = rasterize_rect(geometry, mask_x, mask_y)
        stored = masks[index].astype(bool)
        union = np.logical_or(raster, stored).sum()
        geom_ious.append(1.0 if union == 0 else float(np.logical_and(raster, stored).sum() / union))
    if min(geom_ious) < 0.999:
        raise BaselineValidationError(f"geometry_params do not explain masks: min IoU={min(geom_ious):.6f}")

    rotated_angles = set(float(value) for value in angles[defect_types == "rotated_rect"].tolist())
    if rotated_angles != EXPECTED_ROT_ANGLES:
        raise BaselineValidationError(f"unexpected rotated angle values: {sorted(rotated_angles)}")
    polygon_vertices = set(int(value) for value in vertex_counts[defect_types == "polygon"].tolist())
    if polygon_vertices != EXPECTED_VERTEX_COUNTS:
        raise BaselineValidationError(f"unexpected polygon vertex_count values: {sorted(polygon_vertices)}")

    split_defect_counts = {
        split_name: {
            defect_name: int(np.sum((split_values == split_name) & (defect_types == defect_name)))
            for defect_name in EXPECTED_DEFECTS
        }
        for split_name in ("train", "val", "test")
    }
    split_type_source_counts: dict[str, dict[str, dict[str, int]]] = {}
    for split_name in ("train", "val", "test"):
        split_type_source_counts[split_name] = {}
        for defect_name in EXPECTED_DEFECTS:
            selected_sources = [
                source_packs[index]
                for index in range(len(source_packs))
                if split_values[index] == split_name and defect_types[index] == defect_name
            ]
            split_type_source_counts[split_name][defect_name] = dict(Counter(selected_sources))
            if split_name in {"val", "test"} and len(split_type_source_counts[split_name][defect_name]) < 2:
                raise BaselineValidationError(
                    f"{split_name} {defect_name} source_pack coverage is too narrow: "
                    f"{split_type_source_counts[split_name][defect_name]}"
                )

    return {
        "data": data,
        "missing": missing,
        "split_counts": split_counts,
        "defect_counts": defect_counts,
        "split_defect_counts": split_defect_counts,
        "split_type_source_counts": split_type_source_counts,
        "max_line_diff": max_line_diff,
        "geometry_mask_ious": geom_ious,
        "angles": angles,
        "vertex_counts": vertex_counts,
        "source_packs": np.array(source_packs),
        "source_pack_distribution": dict(Counter(source_packs)),
        "rotated_angle_distribution": dict(Counter(angles[defect_types == "rotated_rect"].tolist())),
        "polygon_vertex_distribution": dict(Counter(vertex_counts[defect_types == "polygon"].tolist())),
        "geometries": geometries,
        "splits": {name: np.where(split_values == name)[0].tolist() for name in ("train", "val", "test")},
    }


def evaluate_model(
    model: torch.nn.Module,
    dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    threshold: float,
    seed: int,
    sample_ids: np.ndarray,
    defect_types: np.ndarray,
    source_packs: np.ndarray,
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
                    "seed": seed,
                    "source_index": index,
                    "sample_id": tiny.as_text(sample_ids[index]),
                    "defect_type": tiny.as_text(defect_types[index]),
                    "source_pack": tiny.as_text(source_packs[index]),
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
                    "notes": "COMSOL_DATA_BASELINE_pilot_v9_only",
                }
            )
    return rows, probs


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std_or_nan(values: list[float]) -> float:
    return float(np.std(values, ddof=0)) if values else float("nan")


def summarize_rows(rows: list[dict[str, Any]], group_name: str, split_name: str, seed: Any) -> dict[str, Any]:
    return {
        "seed": seed,
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


def split_summary_rows(metric_rows: list[dict[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    return {
        split_name: summarize_rows([row for row in metric_rows if row["split"] == split_name], "all", split_name, seed)
        for split_name in ("train", "val", "test")
    }


def group_summary(metric_rows: list[dict[str, Any]], key: str, values: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in metric_rows}):
        seed_rows = [row for row in metric_rows if int(row["seed"]) == seed]
        for split_name in ("train", "val", "test", "all"):
            source = seed_rows if split_name == "all" else [row for row in seed_rows if row["split"] == split_name]
            for value in values:
                selected = [row for row in source if row[key] == value]
                rows.append(summarize_rows(selected, f"{key}={value}", split_name, seed))
    return rows


def angle_summary(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = group_summary(
        [row for row in metric_rows if row["defect_type"] in {"rectangular_notch", "rotated_rect"}],
        "angle_deg",
        [0.0, -30.0, -20.0, -10.0, 10.0, 20.0, 30.0],
    )
    return rows


def seed_metric_summary(seed_summaries: list[dict[str, Any]]) -> dict[str, dict[str, tuple[float, float]]]:
    output: dict[str, dict[str, tuple[float, float]]] = {}
    for split_name in ("train", "val", "test"):
        rows = [row for row in seed_summaries if row["split"] == split_name]
        output[split_name] = {}
        for key in ("iou_mean", "dice_mean", "area_error_mean", "center_error_mean", "pred_area_zero_sum"):
            values = [float(row[key]) for row in rows if str(row[key]).lower() != "nan"]
            output[split_name][key] = (mean_or_nan(values), std_or_nan(values))
    return output


def select_threshold_for_epoch(
    model: torch.nn.Module,
    val_dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    seed: int,
    sample_ids: np.ndarray,
    defect_types: np.ndarray,
    source_packs: np.ndarray,
    angles: np.ndarray,
    vertex_counts: np.ndarray,
) -> dict[str, float]:
    del seed, sample_ids, defect_types, source_packs, angles, vertex_counts
    model.eval()
    with torch.no_grad():
        signals, masks, _ = next(iter(DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)))
        prob = torch.sigmoid(model(signals.to(device))).cpu().numpy()
        target = masks.numpy()
    best = {
        "threshold": THRESHOLD_CANDIDATES[0],
        "iou": 0.0,
        "dice": 0.0,
        "area_error": float("inf"),
        "score": -float("inf"),
    }
    for threshold in THRESHOLD_CANDIDATES:
        metrics = [tiny.sample_metrics(prob[index], target[index], threshold) for index in range(prob.shape[0])]
        iou = float(np.mean([row["iou"] for row in metrics]))
        dice = float(np.mean([row["dice"] for row in metrics]))
        area_error = float(np.mean([row["area_error"] for row in metrics]))
        score = iou + dice - area_error
        if score > best["score"]:
            best = {"threshold": threshold, "iou": iou, "dice": dice, "area_error": area_error, "score": score}
    return best


def train_one_seed(
    seed: int,
    args: argparse.Namespace,
    normalized: np.ndarray,
    masks: np.ndarray,
    splits: dict[str, list[int]],
    sample_ids: np.ndarray,
    defect_types: np.ndarray,
    source_packs: np.ndarray,
    angles: np.ndarray,
    vertex_counts: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    tiny.set_seed(seed)
    train_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["train"])
    val_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["val"])
    test_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["test"])
    model = tiny.TinyComsolMaskDecoder(normalized.shape[1], masks.shape[1], masks.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    best_epoch_threshold = THRESHOLD_CANDIDATES[0]
    epoch_rows: list[dict[str, Any]] = []
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
        threshold_result = select_threshold_for_epoch(
            model, val_dataset, device, seed, sample_ids, defect_types, source_packs, angles, vertex_counts
        )
        if threshold_result["score"] > best_score:
            best_score = float(threshold_result["score"])
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_epoch_threshold = float(threshold_result["threshold"])
        if initial_train_loss is None:
            initial_train_loss = train_loss
        final_train_loss = train_loss
        epoch_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_bce": train_bce,
                "train_dice_loss": train_dice_loss,
                "val_loss": val_loss,
                "val_bce": val_bce,
                "val_dice_loss": val_dice_loss,
                "val_iou_at_0_5": float("nan"),
                "val_dice_at_0_5": float("nan"),
                "val_area_error_at_0_5": float("nan"),
                "val_score_at_0_5": float("nan"),
                "best_val_threshold": threshold_result["threshold"],
                "best_val_iou": threshold_result["iou"],
                "best_val_dice": threshold_result["dice"],
                "best_val_area_error": threshold_result["area_error"],
                "best_val_score": threshold_result["score"],
            }
        )

    model.load_state_dict(best_state)
    selected_threshold, threshold_scores = tiny.select_threshold(model, val_dataset, device, THRESHOLD_CANDIDATES)
    metric_rows: list[dict[str, Any]] = []
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        rows, _ = evaluate_model(
            model, dataset, device, selected_threshold, seed, sample_ids, defect_types, source_packs, angles, vertex_counts
        )
        for row in rows:
            row["split"] = split_name
        metric_rows.extend(rows)
    split_metrics = split_summary_rows(metric_rows, seed)
    train_loss_decreased = bool(final_train_loss is not None and initial_train_loss is not None and final_train_loss < initial_train_loss)
    can_fit_train_samples = bool(
        train_loss_decreased
        and float(split_metrics["train"].get("dice_mean", 0.0)) > 0.70
        and float(split_metrics["train"].get("iou_mean", 0.0)) > 0.55
    )
    return {
        "seed": seed,
        "model_state": deepcopy(model.state_dict()),
        "best_epoch": best_epoch,
        "best_epoch_threshold": best_epoch_threshold,
        "selected_threshold": selected_threshold,
        "threshold_scores": threshold_scores,
        "best_score": best_score,
        "epoch_rows": epoch_rows,
        "metric_rows": metric_rows,
        "split_metrics": split_metrics,
        "initial_train_loss": float(initial_train_loss if initial_train_loss is not None else float("nan")),
        "final_train_loss": float(final_train_loss if final_train_loss is not None else float("nan")),
        "train_loss_decreased": train_loss_decreased,
        "can_fit_train_samples": can_fit_train_samples,
    }


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
                    f"seed: {row['seed']}",
                    f"sample_id: {row['sample_id']}",
                    f"split: {row['split']}",
                    f"type: {row['defect_type']}",
                    f"source: {row['source_pack']}",
                    extra,
                    f"IoU: {float(row['iou']):.4f}",
                    f"Dice: {float(row['dice']):.4f}",
                    f"area_error: {float(row['area_error']):.4f}",
                ]
            ),
            va="top",
            fontsize=8,
        )
        for ax in axes.flat:
            if ax is not axes[0, 0] and ax is not axes[1, 2]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(preview_dir / f"seed{row['seed']}_{row['sample_id']}_{row['split']}_{row['defect_type']}.png", dpi=140)
        plt.close(fig)


def format_mean_std(summary: dict[str, dict[str, tuple[float, float]]]) -> dict[str, str]:
    return {
        split: ", ".join(f"{key}={mean:.4f}+/-{std:.4f}" for key, (mean, std) in values.items())
        for split, values in summary.items()
    }


def build_summary(context: dict[str, Any]) -> str:
    lines = [
        "# COMSOL pilot_v9 data-domain baseline",
        "",
        "## Scope",
        "",
        "- Baseline name: COMSOL_DATA_BASELINE / COMSOL single-defect pilot_v9 mask-only multi-line baseline.",
        "- Dataset domain: COMSOL multi-line delta_Bz controlled synthetic single-defect data.",
        "- This does not replace CURRENT_BASELINE.md and is not a v3_complex baseline.",
        "",
        "## Schema",
        "",
        f"- pilot_v9 NPZ readable: {context['npz_readable']}",
        f"- schema complete: {context['schema_complete']}",
        f"- split is 402 / 99 / 99: {context['split_is_402_99_99']} ({context['split_counts']})",
        f"- defect_type distribution: {context['defect_counts']}",
        f"- every split has three defect types: {context['each_split_has_three_types']} ({context['split_defect_counts']})",
        f"- val/test source_pack not single-source: {context['val_test_source_pack_not_single']}",
        f"- delta_bz input shape: {context['delta_bz_shape']}",
        f"- mask output shape: {context['masks_shape']}",
        f"- rotated_rect angle distribution: {context['rotated_angle_distribution']}",
        f"- polygon vertex_count distribution: {context['polygon_vertex_distribution']}",
        f"- geometry/mask IoU min/mean/max: {context['geometry_iou_summary']}",
        "",
        "## Normalization And Model",
        "",
        "- Normalization: per-channel delta_bz mean/std computed only from train split over train samples and signal length.",
        f"- train mean shape: {context['train_mean_shape']}",
        f"- train std shape: {context['train_std_shape']}",
        "- Model: mask-only grid decoder; Conv1d encoder for `(3, 201)` delta_bz and ConvTranspose2d decoder to `(64, 128)` mask logits.",
        "- Loss: BCEWithLogits + soft Dice.",
        "- No bz_defect / bz_no_defect input, no geometry input, no defect_type conditional input, no forward consistency.",
        "",
        "## Three-Seed Training",
        "",
        f"- seeds: {context['seeds']}",
        f"- epochs: {context['epochs']}",
        f"- batch_size: {context['batch_size']}",
        "- Checkpoint selection: each epoch scans validation thresholds and uses best IoU + Dice - area_error.",
        f"- all seeds completed: {context['all_seeds_completed']}",
        f"- per-seed best epoch / threshold: {context['seed_best']}",
        f"- train loss decreased by seed: {context['train_loss_decreased_by_seed']}",
        f"- can fit 402 train samples by seed: {context['can_fit_by_seed']}",
        "",
        "## Metrics Mean Plus Std",
        "",
        f"- train: {context['mean_std']['train']}",
        f"- val: {context['mean_std']['val']}",
        f"- test: {context['mean_std']['test']}",
        "",
        "## Group Diagnostics",
        "",
        f"- defect_type summary path: {context['defect_summary_path']}",
        f"- angle summary path: {context['angle_summary_path']}",
        f"- vertex_count summary path: {context['vertex_summary_path']}",
        f"- source_pack summary path: {context['source_pack_summary_path']}",
        f"- source_pack issue: {context['source_pack_issue']}",
        f"- per-angle issue: {context['per_angle_issue']}",
        f"- vertex_count issue: {context['vertex_count_issue']}",
        f"- empty predictions: {context['has_empty_prediction']}",
        f"- full-image predictions: {context['has_full_prediction']}",
        f"- NaN detected: {context['has_nan']}",
        "",
        "## Preview",
        "",
        f"- preview generated: {context['preview_generated']}",
        f"- preview dir: {context['preview_dir']}",
        f"- preview sample ids: {context['preview_sample_ids']}",
        "",
        "## Conclusion",
        "",
        f"- Recommend recording as COMSOL_DATA_BASELINE: {context['recommend_baseline']}",
        "- Limitations: pilot-level controlled synthetic data, single-defect only, no multi_defect, no real experimental data, not a v3_complex baseline.",
        f"- Recommended next step: {context['next_step']}",
    ]
    return "\n".join(lines) + "\n"


def write_baseline_doc(path: Path, context: dict[str, Any]) -> None:
    lines = [
        "# COMSOL_DATA_BASELINE",
        "",
        "## Baseline",
        "",
        "- Name: COMSOL single-defect pilot_v9 mask-only multi-line baseline.",
        "- Dataset: `comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect`.",
        f"- NPZ path: `{context['npz_path']}`.",
        "- Input: `delta_bz`, shape `(N, 3, 201)`.",
        "- Output: 2D / quasi-2D defect mask, shape `(64, 128)`.",
        "- Model family: mask-only grid decoder with Conv1d Bz encoder and ConvTranspose2d mask decoder.",
        "- Seeds: 42, 123, 2026.",
        "- Threshold selection: validation-only global threshold per seed from candidates `0.30..0.90`.",
        "- Checkpoint selection: validation `IoU + Dice - area_error`, scanning threshold candidates each epoch.",
        "",
        "## Results",
        "",
        f"- Train mean +/- std: {context['mean_std']['train']}",
        f"- Val mean +/- std: {context['mean_std']['val']}",
        f"- Test mean +/- std: {context['mean_std']['test']}",
        f"- Per-seed best epoch / threshold: {context['seed_best']}",
        "",
        "## Group Checks",
        "",
        f"- Defect type summary: `{context['defect_summary_path']}`.",
        f"- Angle summary: `{context['angle_summary_path']}`.",
        f"- Vertex count summary: `{context['vertex_summary_path']}`.",
        f"- Source pack summary: `{context['source_pack_summary_path']}`.",
        "",
        "## Scope And Limitations",
        "",
        "- This is a COMSOL data-domain baseline only.",
        "- It does not replace `CURRENT_BASELINE.md`.",
        "- It is not a `v3_complex` baseline and should not be compared as a formal replacement.",
        "- It is pilot-level controlled synthetic data.",
        "- It covers single-defect rectangular_notch, rotated_rect, and polygon only.",
        "- It does not demonstrate multi_defect capability or real experimental data generalization.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("--epochs must be between 1 and 200.")
    if args.seeds != DEFAULT_SEEDS:
        raise ValueError("Stage 20.28 baseline must use seeds 42, 123, 2026.")

    npz_path = resolve(args.npz)
    summary_path = resolve(args.summary)
    metrics_path = resolve(args.metrics)
    epoch_log_path = resolve(args.epoch_log)
    seed_summary_path = resolve(args.seed_summary)
    defect_summary_path = resolve(args.defect_summary)
    angle_summary_path = resolve(args.angle_summary)
    vertex_summary_path = resolve(args.vertex_summary)
    source_pack_summary_path = resolve(args.source_pack_summary)
    preview_dir = resolve(args.preview_dir)

    validation = validate_pilot_v9_npz(npz_path)
    data = validation["data"]
    splits = validation["splits"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    source_packs = validation["source_packs"]
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_epoch_rows: list[dict[str, Any]] = []
    seed_summary_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        result = train_one_seed(
            seed,
            args,
            normalized,
            masks,
            splits,
            sample_ids,
            defect_types,
            source_packs,
            angles,
            vertex_counts,
            device,
        )
        seed_results.append(result)
        all_metric_rows.extend(result["metric_rows"])
        all_epoch_rows.extend(result["epoch_rows"])
        for split_name in ("train", "val", "test"):
            row = dict(result["split_metrics"][split_name])
            seed_summary_rows.append(
                {
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "selected_threshold": result["selected_threshold"],
                    "best_val_score": result["best_score"],
                    "train_loss_initial": result["initial_train_loss"],
                    "train_loss_final": result["final_train_loss"],
                    "train_loss_decreased": result["train_loss_decreased"],
                    "can_fit_train_samples": result["can_fit_train_samples"],
                    "split": split_name,
                    "iou_mean": row["iou_mean"],
                    "dice_mean": row["dice_mean"],
                    "area_error_mean": row["area_error_mean"],
                    "center_error_mean": row["center_error_mean"],
                    "pred_area_mean": row["pred_area_mean"],
                    "true_area_mean": row["true_area_mean"],
                    "pred_area_zero_sum": row["pred_area_zero_sum"],
                    "total_loss_mean": row["total_loss_mean"],
                }
            )

    write_csv(metrics_path, all_metric_rows, METRIC_FIELDS)
    write_csv(epoch_log_path, all_epoch_rows, EPOCH_FIELDS)
    write_csv(seed_summary_path, seed_summary_rows, SEED_SUMMARY_FIELDS)
    defect_rows = group_summary(all_metric_rows, "defect_type", ["rectangular_notch", "rotated_rect", "polygon"])
    angle_rows = angle_summary(all_metric_rows)
    vertex_rows = group_summary([row for row in all_metric_rows if row["defect_type"] == "polygon"], "vertex_count", [4, 5, 6])
    source_pack_values = sorted(set(tiny.as_text(value) for value in source_packs.tolist()))
    source_pack_rows = group_summary(all_metric_rows, "source_pack", source_pack_values)
    write_csv(defect_summary_path, defect_rows, GROUP_FIELDS)
    write_csv(angle_summary_path, angle_rows, GROUP_FIELDS)
    write_csv(vertex_summary_path, vertex_rows, GROUP_FIELDS)
    write_csv(source_pack_summary_path, source_pack_rows, GROUP_FIELDS)

    mean_std = seed_metric_summary(seed_summary_rows)
    best_seed = max(seed_results, key=lambda item: item["split_metrics"]["val"]["dice_mean"])
    all_dataset = tiny.ComsolSmokeDataset(normalized, masks, list(range(delta_bz.shape[0])))
    model = tiny.TinyComsolMaskDecoder(delta_bz.shape[1], masks.shape[1], masks.shape[2]).to(device)
    model.load_state_dict(best_seed["model_state"])
    all_rows, all_probs = evaluate_model(
        model,
        all_dataset,
        device,
        float(best_seed["selected_threshold"]),
        int(best_seed["seed"]),
        sample_ids,
        defect_types,
        source_packs,
        angles,
        vertex_counts,
    )
    split_by_index = {}
    for split_name, indices in splits.items():
        for index in indices:
            split_by_index[index] = split_name
    for row in all_rows:
        row["split"] = split_by_index[int(row["source_index"])]
    selected_preview_indices = choose_preview_indices(all_rows)
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
        float(best_seed["selected_threshold"]),
    )

    full_area = masks.shape[1] * masks.shape[2]
    source_pack_issue = any(float(row["dice_mean"]) < 0.45 for row in source_pack_rows if row["split"] == "all")
    per_angle_issue = any(float(row["dice_mean"]) < 0.45 for row in angle_rows if row["split"] == "all" and row["sample_count"] > 0)
    vertex_count_issue = any(float(row["dice_mean"]) < 0.45 for row in vertex_rows if row["split"] == "all" and row["sample_count"] > 0)
    has_empty_prediction = any(int(row["pred_area_zero"]) == 1 for row in all_metric_rows)
    has_full_prediction = any(int(row["pred_area"]) >= full_area for row in all_metric_rows)
    has_nan = any(not np.isfinite(float(row["total_loss"])) for row in all_metric_rows)
    recommend_baseline = bool(
        all(result["train_loss_decreased"] for result in seed_results)
        and not has_nan
        and not has_full_prediction
        and mean_std["test"]["dice_mean"][0] > 0.70
    )
    next_step = (
        "Expand toward multi_defect after reviewing COMSOL_DATA_BASELINE visual failures."
        if recommend_baseline
        else "Inspect weak groups and consider more samples or model capacity before multi_defect."
    )

    context = {
        "npz_path": str(npz_path),
        "npz_readable": True,
        "schema_complete": len(validation["missing"]) == 0,
        "split_is_402_99_99": validation["split_counts"] == EXPECTED_SPLITS,
        "split_counts": validation["split_counts"],
        "defect_counts": validation["defect_counts"],
        "each_split_has_three_types": all(all(count > 0 for count in values.values()) for values in validation["split_defect_counts"].values()),
        "split_defect_counts": validation["split_defect_counts"],
        "val_test_source_pack_not_single": all(
            len(validation["split_type_source_counts"][split_name][defect_name]) >= 2
            for split_name in ("val", "test")
            for defect_name in ("rectangular_notch", "rotated_rect", "polygon")
        ),
        "delta_bz_shape": tuple(delta_bz.shape),
        "masks_shape": tuple(masks.shape),
        "rotated_angle_distribution": validation["rotated_angle_distribution"],
        "polygon_vertex_distribution": validation["polygon_vertex_distribution"],
        "geometry_iou_summary": {
            "min": float(np.min(validation["geometry_mask_ious"])),
            "mean": float(np.mean(validation["geometry_mask_ious"])),
            "max": float(np.max(validation["geometry_mask_ious"])),
        },
        "train_mean_shape": tuple(train_mean.shape),
        "train_std_shape": tuple(train_std.shape),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "all_seeds_completed": len(seed_results) == len(args.seeds),
        "seed_best": {
            int(result["seed"]): {
                "best_epoch": int(result["best_epoch"]),
                "checkpoint_threshold": float(result["best_epoch_threshold"]),
                "selected_threshold": float(result["selected_threshold"]),
                "best_val_score": float(result["best_score"]),
            }
            for result in seed_results
        },
        "train_loss_decreased_by_seed": {int(result["seed"]): bool(result["train_loss_decreased"]) for result in seed_results},
        "can_fit_by_seed": {int(result["seed"]): bool(result["can_fit_train_samples"]) for result in seed_results},
        "mean_std": format_mean_std(mean_std),
        "defect_summary_path": str(defect_summary_path),
        "angle_summary_path": str(angle_summary_path),
        "vertex_summary_path": str(vertex_summary_path),
        "source_pack_summary_path": str(source_pack_summary_path),
        "source_pack_issue": bool(source_pack_issue),
        "per_angle_issue": bool(per_angle_issue),
        "vertex_count_issue": bool(vertex_count_issue),
        "has_empty_prediction": bool(has_empty_prediction),
        "has_full_prediction": bool(has_full_prediction),
        "has_nan": bool(has_nan),
        "preview_generated": True,
        "preview_dir": str(preview_dir),
        "preview_sample_ids": [tiny.as_text(sample_ids[index]) for index in selected_preview_indices],
        "recommend_baseline": recommend_baseline,
        "next_step": next_step,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(context), encoding="utf-8")
    write_baseline_doc(PROJECT_ROOT / "COMSOL_DATA_BASELINE.md", context)
    print(json.dumps(context, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
