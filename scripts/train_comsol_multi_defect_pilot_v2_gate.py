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
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_multiline_tiny_smoke as tiny  # noqa: E402
import train_comsol_multi_defect_pilot_gate as pilot_v1  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_multi_defect_multiline_forward_pack_v2_pilot.npz"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multi_defect_pilot_v2_training_gate_summary.txt"
DEFAULT_AUDIT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_multi_defect_pilot_v2_failure_audit_summary.txt"
DEFAULT_PLAN = PROJECT_ROOT / "results/summaries/comsol_multi_defect_pilot_v3_expansion_plan.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_v2_training_gate_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_v2_epoch_log.csv"
DEFAULT_SEED_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_v2_seed_summary.csv"
DEFAULT_COMBO_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_v2_component_combination_summary.csv"
DEFAULT_CONNECTED_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_v2_connected_component_summary.csv"
DEFAULT_FAILURE_CASES = PROJECT_ROOT / "results/metrics/comsol_multi_defect_pilot_v2_failure_audit_cases.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_multi_defect_pilot_v2_gate"

SEEDS = [42, 123, 2026]
EXPECTED_SPLITS = {"train": 80, "val": 20, "test": 20}
EXPECTED_COMBOS = {
    "rectangular_notch+rectangular_notch",
    "rectangular_notch+rotated_rect",
    "rotated_rect+rotated_rect",
}
THRESHOLD_CANDIDATES = pilot_v1.THRESHOLD_CANDIDATES

METRIC_FIELDS = [
    "seed",
    *pilot_v1.METRIC_FIELDS,
    "min_component_distance",
    "min_component_distance_bin",
    "union_mask_area",
    "union_mask_area_bin",
]
EPOCH_FIELDS = [
    "seed",
    *pilot_v1.EPOCH_FIELDS,
]
SUMMARY_FIELDS = [
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
    "pred_connected_component_count_mean",
    "predicted_component_count_is_2_rate",
    "missed_component_rate",
    "merged_component_rate",
    "split_component_rate",
    "component_recall_mean",
]
SEED_SUMMARY_FIELDS = [
    "seed",
    "split",
    "best_epoch",
    "selected_threshold",
    "best_val_score",
    "initial_train_loss",
    "final_train_loss",
    "train_loss_decreased",
    "sample_count",
    "iou_mean",
    "dice_mean",
    "area_error_mean",
    "center_error_mean",
    "pred_area_zero_sum",
    "pred_connected_component_count_mean",
    "predicted_component_count_is_2_rate",
    "missed_component_rate",
    "merged_component_rate",
    "split_component_rate",
    "component_recall_mean",
]
FAILURE_FIELDS = pilot_v1.FAILURE_FIELDS + ["seed"]


class MultiDefectV2ValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COMSOL multi_defect pilot_v2 3-seed training gate.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--expansion-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--seed-summary", type=Path, default=DEFAULT_SEED_SUMMARY)
    parser.add_argument("--component-summary", type=Path, default=DEFAULT_COMBO_SUMMARY)
    parser.add_argument("--connected-summary", type=Path, default=DEFAULT_CONNECTED_SUMMARY)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE_CASES)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
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


def bin_by_tertiles(values: np.ndarray, labels: tuple[str, str, str]) -> list[str]:
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    bins: list[str] = []
    for value in values:
        if value <= q1:
            bins.append(labels[0])
        elif value <= q2:
            bins.append(labels[1])
        else:
            bins.append(labels[2])
    return bins


def validate_npz(npz_path: Path) -> dict[str, Any]:
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
        "components_json",
        "component_counts",
        "component_types",
        "connected_component_counts",
        "split",
        "metadata",
    ]
    data = np.load(npz_path, allow_pickle=True)
    missing = [field for field in required if field not in data.files]
    if missing:
        raise MultiDefectV2ValidationError(f"missing NPZ fields: {missing}")

    delta_bz = data["delta_bz"]
    masks = data["masks"]
    bz_defect = data["bz_defect"]
    bz_no_defect = data["bz_no_defect"]
    sensor_x = data["sensor_x"]
    scan_line_y = data["scan_line_y"]
    mask_x = data["mask_x"]
    mask_y = data["mask_y"]
    defect_types = np.array([tiny.as_text(item) for item in data["defect_types"].tolist()])
    sample_ids = np.array([tiny.as_text(item) for item in data["sample_ids"].tolist()])
    splits = np.array([tiny.as_text(item) for item in data["split"].tolist()])
    component_counts = data["component_counts"].astype(np.int64)
    connected_counts = data["connected_component_counts"].astype(np.int64)
    components = [pilot_v1.as_json(item) for item in data["components_json"].tolist()]
    component_types = [pilot_v1.as_json(item) for item in data["component_types"].tolist()]

    if delta_bz.shape != (120, 3, 201):
        raise MultiDefectV2ValidationError(f"unexpected delta_bz shape: {delta_bz.shape}")
    if bz_defect.shape != delta_bz.shape or bz_no_defect.shape != delta_bz.shape:
        raise MultiDefectV2ValidationError("bz_defect / bz_no_defect shape mismatch")
    if masks.shape != (120, 64, 128):
        raise MultiDefectV2ValidationError(f"unexpected masks shape: {masks.shape}")
    if sensor_x.shape != (201,) or scan_line_y.shape != (3,) or mask_x.shape != (128,) or mask_y.shape != (64,):
        raise MultiDefectV2ValidationError("coordinate shape mismatch")
    if not (
        np.all(np.diff(sensor_x) > 0)
        and np.all(np.diff(scan_line_y) > 0)
        and np.all(np.diff(mask_x) > 0)
        and np.all(np.diff(mask_y) > 0)
    ):
        raise MultiDefectV2ValidationError("coordinates are not strictly increasing")
    if not all(np.isfinite(array).all() for array in (delta_bz, bz_defect, bz_no_defect, masks, sensor_x, scan_line_y, mask_x, mask_y)):
        raise MultiDefectV2ValidationError("non-finite arrays found")
    if not np.allclose(delta_bz, bz_defect - bz_no_defect, rtol=1e-9, atol=1e-12):
        raise MultiDefectV2ValidationError("delta_bz mismatch")
    if float(delta_bz.std()) <= 0.0:
        raise MultiDefectV2ValidationError("delta_bz is zero")
    if np.max(np.abs(delta_bz[:, 1:, :] - delta_bz[:, :1, :])) <= 1e-12:
        raise MultiDefectV2ValidationError("scan lines are identical")
    if not np.all(masks.reshape(masks.shape[0], -1).sum(axis=1) > 0):
        raise MultiDefectV2ValidationError("empty mask found")
    if set(defect_types.tolist()) != {"multi_defect"}:
        raise MultiDefectV2ValidationError(f"unexpected defect_types: {Counter(defect_types.tolist())}")
    if not np.all(component_counts == 2):
        raise MultiDefectV2ValidationError(f"component_counts not all 2: {Counter(component_counts.tolist())}")
    if not np.all(connected_counts == 2):
        raise MultiDefectV2ValidationError(f"connected_component_counts not all 2: {Counter(connected_counts.tolist())}")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise MultiDefectV2ValidationError("sample_ids are not unique")
    split_counts = Counter(splits.tolist())
    if dict(split_counts) != EXPECTED_SPLITS:
        raise MultiDefectV2ValidationError(f"unexpected split counts: {dict(split_counts)}")

    component_type_combos: list[str] = []
    min_distances: list[float] = []
    geometry_mask_ious: list[float] = []
    for index, component_list in enumerate(components):
        if not isinstance(component_list, list) or len(component_list) != 2:
            raise MultiDefectV2ValidationError(f"components_json index {index} is not a 2-component list")
        combo = "+".join(tiny.as_text(component["component_type"]) for component in component_list)
        component_type_combos.append(combo)
        if combo not in EXPECTED_COMBOS:
            raise MultiDefectV2ValidationError(f"unexpected component combo: {combo}")
        union = np.zeros(masks[index].shape, dtype=bool)
        centers = []
        for component in component_list:
            component_mask = pilot_v1.rasterize_component(component, mask_x, mask_y).astype(bool)
            if int(component_mask.sum()) <= 0:
                raise MultiDefectV2ValidationError(f"empty component mask at sample {index}")
            union |= component_mask
            centers.append((float(component["center_x_m"]), float(component["center_y_m"])))
        stored = masks[index].astype(bool)
        denom = int(np.logical_or(union, stored).sum())
        geometry_mask_ious.append(1.0 if denom == 0 else float(np.logical_and(union, stored).sum() / denom))
        cc, _, _ = pilot_v1.connected_components(stored)
        if cc != 2:
            raise MultiDefectV2ValidationError(f"mask connected component count at {index} is {cc}")
        min_distances.append(float(math.hypot(centers[0][0] - centers[1][0], centers[0][1] - centers[1][1])))
    if min(geometry_mask_ious) < 0.999:
        raise MultiDefectV2ValidationError(f"components_json does not explain masks: min IoU {min(geometry_mask_ious)}")

    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64)
    distance_bins = np.array(bin_by_tertiles(np.array(min_distances), ("near", "medium", "far")))
    area_bins = np.array(bin_by_tertiles(areas, ("small", "medium", "large")))
    return {
        "data": data,
        "sample_ids": sample_ids,
        "splits_array": splits,
        "splits": {name: np.where(splits == name)[0].tolist() for name in ("train", "val", "test")},
        "components": components,
        "component_types_raw": component_types,
        "component_type_combos": np.array(component_type_combos),
        "connected_counts": connected_counts,
        "geometry_mask_ious": geometry_mask_ious,
        "min_distances": np.array(min_distances),
        "distance_bins": distance_bins,
        "union_areas": areas,
        "area_bins": area_bins,
        "split_counts": dict(split_counts),
        "combo_counts": dict(Counter(component_type_combos)),
        "distance_bin_counts": dict(Counter(distance_bins.tolist())),
        "area_bin_counts": dict(Counter(area_bins.tolist())),
    }


def summarize(rows: list[dict[str, Any]], group: str, split: str, seed: Any = "all") -> dict[str, Any]:
    base = pilot_v1.summarize_rows(rows, group, split)
    return {"seed": seed, **base}


def grouped_summary(metric_rows: list[dict[str, Any]], key: str, values: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "all"):
        source = metric_rows if split_name == "all" else [row for row in metric_rows if row["split"] == split_name]
        for value in values:
            selected = [row for row in source if row[key] == value]
            rows.append(summarize(selected, f"{key}={value}", split_name))
    return rows


def train_one_seed(
    seed: int,
    args: argparse.Namespace,
    validation: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    tiny.set_seed(seed)
    data = validation["data"]
    delta_bz = data["delta_bz"].astype(np.float32)
    masks = data["masks"].astype(np.float32)
    splits = validation["splits"]
    train_mean = delta_bz[splits["train"]].mean(axis=(0, 2), keepdims=True)
    train_std = np.maximum(delta_bz[splits["train"]].std(axis=(0, 2), keepdims=True), 1e-8)
    normalized = (delta_bz - train_mean) / train_std

    train_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["train"])
    val_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["val"])
    test_dataset = tiny.ComsolSmokeDataset(normalized, masks, splits["test"])
    all_dataset = tiny.ComsolSmokeDataset(normalized, masks, list(range(delta_bz.shape[0])))

    model = tiny.TinyComsolMaskDecoder(delta_bz.shape[1], masks.shape[1], masks.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_totals: list[float] = []
        batch_bces: list[float] = []
        batch_dices: list[float] = []
        for signals, targets, _ in train_loader:
            signals = signals.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            total, bce, dice = tiny.loss_components(logits, targets)
            total.backward()
            optimizer.step()
            batch_totals.append(float(total.item()))
            batch_bces.append(float(bce.item()))
            batch_dices.append(float(dice.item()))
        train_loss = {
            "total_loss": float(np.mean(batch_totals)),
            "bce_loss": float(np.mean(batch_bces)),
            "dice_loss": float(np.mean(batch_dices)),
        }
        val_total, val_bce, val_dice = tiny.evaluate_loss(model, val_dataset, device)
        val_loss = {"total_loss": float(val_total), "bce_loss": float(val_bce), "dice_loss": float(val_dice)}
        best_val = pilot_v1.select_threshold_for_epoch(model, val_dataset, device)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss["total_loss"],
            "train_bce": train_loss["bce_loss"],
            "train_dice_loss": train_loss["dice_loss"],
            "val_loss": val_loss["total_loss"],
            "val_bce": val_loss["bce_loss"],
            "val_dice_loss": val_loss["dice_loss"],
            "best_val_threshold": best_val["threshold"],
            "best_val_iou": best_val["iou"],
            "best_val_dice": best_val["dice"],
            "best_val_area_error": best_val["area_error"],
            "best_val_score": best_val["score"],
        }
        epoch_rows.append(row)
        if best_val["score"] > best_score:
            best_score = best_val["score"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    threshold_metrics = pilot_v1.select_threshold_for_epoch(model, val_dataset, device)
    threshold = float(threshold_metrics["threshold"])
    metric_rows, all_probs = pilot_v1.evaluate_model(
        model,
        all_dataset,
        device,
        threshold,
        validation["sample_ids"],
        validation["splits_array"],
        validation["component_type_combos"],
        validation["components"],
        data["mask_x"].astype(np.float64),
        data["mask_y"].astype(np.float64),
    )
    for row in metric_rows:
        idx = int(row["source_index"])
        row["seed"] = seed
        row["min_component_distance"] = float(validation["min_distances"][idx])
        row["min_component_distance_bin"] = tiny.as_text(validation["distance_bins"][idx])
        row["union_mask_area"] = int(validation["union_areas"][idx])
        row["union_mask_area_bin"] = tiny.as_text(validation["area_bins"][idx])

    selected_probs = {}
    if seed == SEEDS[0]:
        preview_indices = choose_preview_indices(metric_rows)
        selected_probs = {index: all_probs[index] for index in preview_indices if index in all_probs}
    split_summaries = {
        split_name: summarize([row for row in metric_rows if row["split"] == split_name], "all", split_name, seed)
        for split_name in ("train", "val", "test")
    }
    return {
        "seed": seed,
        "model": model,
        "normalized": normalized,
        "train_mean": train_mean,
        "train_std": train_std,
        "threshold": threshold,
        "threshold_metrics": threshold_metrics,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "epoch_rows": epoch_rows,
        "metric_rows": metric_rows,
        "selected_probs": selected_probs,
        "split_summaries": split_summaries,
        "initial_train_loss": float(epoch_rows[0]["train_loss"]),
        "final_train_loss": float(epoch_rows[-1]["train_loss"]),
        "train_loss_decreased": float(epoch_rows[-1]["train_loss"]) < float(epoch_rows[0]["train_loss"]),
    }


def choose_preview_indices(metric_rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected:
            selected.append(index)

    val_test = [row for row in metric_rows if row["split"] in {"val", "test"}]
    for combo in sorted(EXPECTED_COMBOS):
        rows = [row for row in val_test if row["component_types"] == combo]
        if rows:
            add(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
            add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    for bin_name in ("near", "medium", "far"):
        rows = [row for row in val_test if row["min_component_distance_bin"] == bin_name]
        if rows:
            add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    for pred_cc in (1, 2, 3):
        rows = [row for row in metric_rows if int(row["pred_connected_component_count"]) == pred_cc]
        if rows:
            add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    for reverse in (True, False):
        for row in sorted(val_test, key=lambda item: float(item["dice"]), reverse=reverse)[:8]:
            add(int(row["source_index"]))
    return selected[:24]


def mean_std(seed_results: list[dict[str, Any]], split: str, field: str) -> tuple[float, float]:
    values = [float(result["split_summaries"][split][field]) for result in seed_results]
    return float(np.mean(values)), float(np.std(values, ddof=0))


def format_mean_std(seed_results: list[dict[str, Any]], split: str) -> str:
    items = []
    for field, label in (
        ("iou_mean", "IoU"),
        ("dice_mean", "Dice"),
        ("area_error_mean", "area_error"),
        ("center_error_mean", "center_error"),
        ("predicted_component_count_is_2_rate", "pred_cc_is_2"),
        ("missed_component_rate", "missed"),
        ("merged_component_rate", "merged"),
        ("split_component_rate", "split"),
    ):
        mean, std = mean_std(seed_results, split, field)
        items.append(f"{label}={mean:.4f}+/-{std:.4f}")
    return ", ".join(items)


def seed_summary_rows(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in seed_results:
        for split_name in ("train", "val", "test"):
            rows.append(
                {
                    "seed": result["seed"],
                    "split": split_name,
                    "best_epoch": result["best_epoch"],
                    "selected_threshold": result["threshold"],
                    "best_val_score": result["best_val_score"],
                    "initial_train_loss": result["initial_train_loss"],
                    "final_train_loss": result["final_train_loss"],
                    "train_loss_decreased": result["train_loss_decreased"],
                    **{key: value for key, value in result["split_summaries"][split_name].items() if key not in {"seed", "group", "split"}},
                }
            )
    return rows


def build_group_rows(metric_rows: list[dict[str, Any]], key: str, values: list[Any]) -> list[dict[str, Any]]:
    return grouped_summary(metric_rows, key, values)


def failure_cases(metric_rows: list[dict[str, Any]], preview_paths: dict[str, str]) -> list[dict[str, Any]]:
    seed42_rows = [row for row in metric_rows if int(row["seed"]) == SEEDS[0]]
    cases = pilot_v1.build_failure_cases(seed42_rows, preview_paths)
    for row in cases:
        row["seed"] = SEEDS[0]
    return cases


def dominant_failure(cases: list[dict[str, Any]]) -> str:
    counts = Counter(tiny.as_text(row["failure_category"]) for row in cases)
    return counts.most_common(1)[0][0] if counts else "none"


def hardest_combo(combo_rows: list[dict[str, Any]]) -> str:
    all_rows = [row for row in combo_rows if row["split"] == "test" and int(row["sample_count"]) > 0]
    if not all_rows:
        all_rows = [row for row in combo_rows if row["split"] == "all" and int(row["sample_count"]) > 0]
    if not all_rows:
        return "n/a"
    row = min(all_rows, key=lambda item: float(item["dice_mean"]))
    return tiny.as_text(row["group"])


def build_training_summary(context: dict[str, Any]) -> str:
    lines = [
        "# COMSOL true multi_defect pilot_v2 3-seed training gate",
        "",
        "## Schema",
        "",
        f"- pilot_v2 NPZ readable: {context['npz_readable']}",
        f"- schema complete: {context['schema_complete']}",
        f"- split is 80 / 20 / 20: {context['split_ok']} ({context['split_counts']})",
        f"- component_count all 2: {context['component_count_all_2']}",
        f"- true connected count all 2: {context['true_cc_all_2']}",
        f"- component_type combination distribution: {context['combo_counts']}",
        f"- min_component_distance bins: {context['distance_bin_counts']}",
        f"- union_mask_area bins: {context['area_bin_counts']}",
        f"- delta_bz input shape: {context['delta_bz_shape']}",
        f"- mask output shape: {context['masks_shape']}",
        f"- components_json/mask IoU min/mean/max: {context['geometry_iou_summary']}",
        "",
        "## Normalization And Model",
        "",
        "- Normalization: per-channel delta_bz mean/std computed only from train split for each seed run.",
        f"- train mean shape: {context['train_mean_shape']}",
        f"- train std shape: {context['train_std_shape']}",
        "- Model: lightweight mask-only Conv1d encoder for `(3, 201)` delta_bz and ConvTranspose2d decoder to `(64, 128)` union-mask logits.",
        "- Loss: BCEWithLogits + soft Dice. No component_count, component_type, geometry, bz_defect, or bz_no_defect input.",
        "",
        "## Training Gate",
        "",
        f"- seeds: {context['seeds']}",
        f"- epochs: {context['epochs']}",
        f"- batch_size: {context['batch_size']}",
        "- Checkpoint selection: each epoch scans validation thresholds and uses best validation IoU + Dice - area_error.",
        f"- seed best epochs / thresholds: {context['seed_best']}",
        f"- all seeds completed: {context['all_seeds_ok']}",
        f"- train loss decreased all seeds: {context['train_loss_decreased_all']}",
        f"- can fit 80 train samples: {context['can_fit_train_samples']}",
        "",
        "## 3-Seed Pilot Metrics",
        "",
        f"- train mean+/-std: {context['train_mean_std']}",
        f"- val mean+/-std: {context['val_mean_std']}",
        f"- test mean+/-std: {context['test_mean_std']}",
        f"- predicted connected component count summary: {context['connected_summary_path']}",
        f"- component combination summary: {context['component_summary_path']}",
        f"- missed / merged / split rates on test: {context['test_component_error_summary']}",
        f"- hardest component_type combination: {context['hardest_combo']}",
        f"- empty predictions: {context['has_empty_prediction']}",
        f"- full-image predictions: {context['has_full_prediction']}",
        f"- NaN detected: {context['has_nan']}",
        "",
        "## Preview And Audit",
        "",
        f"- preview generated: {context['preview_generated']}",
        f"- preview dir: {context['preview_dir']}",
        f"- preview sample ids: {context['preview_sample_ids']}",
        f"- failure audit summary: {context['audit_summary_path']}",
        f"- failure cases CSV: {context['failure_cases_path']}",
        f"- failure audit main conclusion: {context['failure_main']}",
        "",
        "## Conclusion",
        "",
        "- This is a multi_defect pilot_v2 gate only, not a baseline and not a v3_complex comparison.",
        f"- recommend expanding multi_defect data: {context['recommend_expand']}",
        f"- recommend polygon components: {context['recommend_polygon']}",
        f"- recommend multi_defect COMSOL baseline now: {context['recommend_baseline']}",
        f"- current limitations: {context['limitations']}",
    ]
    return "\n".join(lines) + "\n"


def build_audit_summary(context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# COMSOL true multi_defect pilot_v2 failure audit",
            "",
            f"- model predicts non-empty union masks: {not context['has_empty_prediction']}",
            f"- missed second component rate on test: {context['test_missed_rate']:.4f}",
            f"- merged one-blob rate on test: {context['test_merged_rate']:.4f}",
            f"- extra fragment / split rate on test: {context['test_split_rate']:.4f}",
            f"- hardest component combination: {context['hardest_combo']}",
            f"- dominant failure category: {context['dominant_failure_category']}",
            f"- near-distance issue: {context['near_distance_issue']}",
            f"- schema / mask / components_json issue found: False",
            f"- preview dir: {context['preview_dir']}",
            "",
            "## Interpretation",
            "",
            context["interpretation"],
            "",
            "## Recommendation",
            "",
            context["recommendation"],
        ]
    ) + "\n"


def build_expansion_plan(context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# COMSOL multi_defect pilot_v3 expansion plan",
            "",
            "- Recommended scale: 240 true multi_defect samples as the next bounded pilot_v3; use 160 train / 40 val / 40 test.",
            "- Keep component_count=2 for pilot_v3; do not jump to 3+ components until 2-component validation/test behavior is stable.",
            "- Balance rectangular_notch+rectangular_notch, rectangular_notch+rotated_rect, and rotated_rect+rotated_rect across train/val/test.",
            "- Add polygon components only after a separate 1-sample smoke and a 24-sample mini pilot; suggested first polygon combinations are rectangular_notch+polygon and rotated_rect+polygon.",
            "- Stratify min_component_distance into near / medium / far bins and ensure each split covers all bins.",
            "- Stratify relative component area and depth so one component is not always dominant.",
            "- Preserve train-only normalization and validation-only threshold/checkpoint selection.",
            "- Keep component-aware metrics: predicted connected component count, missed component flag, merge flag, extra fragment flag, and component recall heuristic.",
            "- Do not merge with single-defect data until a multi_defect-only 240-sample gate is reviewed.",
            "- Acceptance condition: schema complete, no data corruption, no systematic missed second component, and val/test contain all combinations and distance bins.",
            "- Stop condition: COMSOL geometry instability, components_json/mask mismatch, or frequent predicted connected count collapse to 1 on val/test.",
            f"- Current pilot_v2 signal: {context['recommendation']}",
        ]
    ) + "\n"


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.epochs > 200:
        raise ValueError("--epochs must be between 1 and 200.")
    paths = {
        "npz": resolve(args.npz),
        "summary": resolve(args.summary),
        "audit_summary": resolve(args.audit_summary),
        "plan": resolve(args.expansion_plan),
        "metrics": resolve(args.metrics),
        "epoch_log": resolve(args.epoch_log),
        "seed_summary": resolve(args.seed_summary),
        "component_summary": resolve(args.component_summary),
        "connected_summary": resolve(args.connected_summary),
        "failure_cases": resolve(args.failure_cases),
        "preview_dir": resolve(args.preview_dir),
    }
    validation = validate_npz(paths["npz"])
    data = validation["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = [train_one_seed(seed, args, validation, device) for seed in SEEDS]
    all_metric_rows = [row for result in seed_results for row in result["metric_rows"]]
    all_epoch_rows = [row for result in seed_results for row in result["epoch_rows"]]

    component_rows = build_group_rows(all_metric_rows, "component_types", sorted(EXPECTED_COMBOS))
    connected_values = sorted({int(row["pred_connected_component_count"]) for row in all_metric_rows})
    connected_rows = build_group_rows(all_metric_rows, "pred_connected_component_count", connected_values)
    distance_rows = build_group_rows(all_metric_rows, "min_component_distance_bin", ["near", "medium", "far"])
    area_rows = build_group_rows(all_metric_rows, "union_mask_area_bin", ["small", "medium", "large"])
    component_rows.extend(distance_rows)
    component_rows.extend(area_rows)

    preview_result = seed_results[0]
    preview_paths = pilot_v1.make_previews(
        paths["preview_dir"],
        preview_result["selected_probs"],
        data["masks"].astype(np.float32),
        data["delta_bz"].astype(np.float32),
        data["sensor_x"].astype(np.float64),
        data["scan_line_y"].astype(np.float64),
        [row for row in preview_result["metric_rows"] if int(row["source_index"]) in preview_result["selected_probs"]],
        preview_result["threshold"],
    )
    cases = failure_cases(all_metric_rows, preview_paths)

    write_csv(paths["metrics"], all_metric_rows, METRIC_FIELDS)
    write_csv(paths["epoch_log"], all_epoch_rows, EPOCH_FIELDS)
    write_csv(paths["seed_summary"], seed_summary_rows(seed_results), SEED_SUMMARY_FIELDS)
    write_csv(paths["component_summary"], component_rows, SUMMARY_FIELDS)
    write_csv(paths["connected_summary"], connected_rows, SUMMARY_FIELDS)
    write_csv(paths["failure_cases"], cases, FAILURE_FIELDS)

    test_rows = [row for row in all_metric_rows if row["split"] == "test"]
    test_missed_rate = float(np.mean([float(row["missed_component_flag"]) for row in test_rows]))
    test_merged_rate = float(np.mean([float(row["merged_component_flag"]) for row in test_rows]))
    test_split_rate = float(np.mean([float(row["split_component_flag"]) for row in test_rows]))
    test_cc2_rate = float(np.mean([float(row["predicted_component_count_is_2"]) for row in test_rows]))
    has_empty = any(int(row["pred_area_zero"]) for row in all_metric_rows)
    has_full = any(float(row["pred_area"]) >= 0.95 * (64 * 128) for row in all_metric_rows)
    has_nan = any(math.isnan(float(row["iou"])) or math.isnan(float(row["dice"])) for row in all_metric_rows)
    hardest = hardest_combo(component_rows)
    dominant = dominant_failure(cases)
    near_rows = [row for row in test_rows if row["min_component_distance_bin"] == "near"]
    far_rows = [row for row in test_rows if row["min_component_distance_bin"] == "far"]
    near_issue = False
    if near_rows and far_rows:
        near_issue = float(np.mean([float(row["dice"]) for row in near_rows])) + 0.05 < float(np.mean([float(row["dice"]) for row in far_rows]))
    recommend_expand = True
    recommend_polygon = test_missed_rate < 0.10 and test_merged_rate < 0.10 and test_cc2_rate > 0.80
    recommend_baseline = False
    can_fit = all(result["split_summaries"]["train"]["dice_mean"] > 0.75 for result in seed_results)

    context = {
        "npz_readable": True,
        "schema_complete": True,
        "split_ok": validation["split_counts"] == EXPECTED_SPLITS,
        "split_counts": validation["split_counts"],
        "component_count_all_2": True,
        "true_cc_all_2": True,
        "combo_counts": validation["combo_counts"],
        "distance_bin_counts": validation["distance_bin_counts"],
        "area_bin_counts": validation["area_bin_counts"],
        "delta_bz_shape": tuple(data["delta_bz"].shape),
        "masks_shape": tuple(data["masks"].shape),
        "geometry_iou_summary": [
            float(min(validation["geometry_mask_ious"])),
            float(np.mean(validation["geometry_mask_ious"])),
            float(max(validation["geometry_mask_ious"])),
        ],
        "train_mean_shape": tuple(seed_results[0]["train_mean"].shape),
        "train_std_shape": tuple(seed_results[0]["train_std"].shape),
        "seeds": SEEDS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed_best": {
            result["seed"]: {
                "best_epoch": result["best_epoch"],
                "selected_threshold": result["threshold"],
                "best_val_score": result["best_val_score"],
            }
            for result in seed_results
        },
        "all_seeds_ok": len(seed_results) == len(SEEDS),
        "train_loss_decreased_all": all(result["train_loss_decreased"] for result in seed_results),
        "can_fit_train_samples": can_fit,
        "train_mean_std": format_mean_std(seed_results, "train"),
        "val_mean_std": format_mean_std(seed_results, "val"),
        "test_mean_std": format_mean_std(seed_results, "test"),
        "connected_summary_path": paths["connected_summary"],
        "component_summary_path": paths["component_summary"],
        "test_component_error_summary": {
            "missed_rate": test_missed_rate,
            "merged_rate": test_merged_rate,
            "split_rate": test_split_rate,
            "predicted_component_count_is_2_rate": test_cc2_rate,
        },
        "hardest_combo": hardest,
        "has_empty_prediction": has_empty,
        "has_full_prediction": has_full,
        "has_nan": has_nan,
        "preview_generated": bool(preview_paths),
        "preview_dir": paths["preview_dir"],
        "preview_sample_ids": sorted(preview_paths),
        "audit_summary_path": paths["audit_summary"],
        "failure_cases_path": paths["failure_cases"],
        "failure_main": (
            "Pilot_v2 is trainable and component diagnostics are meaningful; dominant failures remain "
            f"{dominant}, with test missed/merged/split rates {test_missed_rate:.3f}/{test_merged_rate:.3f}/{test_split_rate:.3f}."
        ),
        "recommend_expand": recommend_expand,
        "recommend_polygon": recommend_polygon,
        "recommend_baseline": recommend_baseline,
        "limitations": "pilot_v2 only, component_count=2 only, no polygon components, no real experimental data, not a baseline",
    }
    audit_context = {
        **context,
        "test_missed_rate": test_missed_rate,
        "test_merged_rate": test_merged_rate,
        "test_split_rate": test_split_rate,
        "dominant_failure_category": dominant,
        "near_distance_issue": near_issue,
        "interpretation": context["failure_main"],
        "recommendation": (
            "Expand to a 240-sample multi_defect pilot_v3 before declaring a baseline; add a small polygon-component "
            "smoke/mini-pack if geometry remains stable."
            if recommend_polygon
            else "Expand rect/rotated multi_defect first; defer polygon components until component-count behavior stabilizes."
        ),
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["audit_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["plan"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(build_training_summary(context), encoding="utf-8")
    paths["audit_summary"].write_text(build_audit_summary(audit_context), encoding="utf-8")
    paths["plan"].write_text(build_expansion_plan(audit_context), encoding="utf-8")

    print(
        json.dumps(
            {
                "all_seeds_ok": context["all_seeds_ok"],
                "seed_best": context["seed_best"],
                "train": context["train_mean_std"],
                "val": context["val_mean_std"],
                "test": context["test_mean_std"],
                "hardest_combo": hardest,
                "test_component_error_summary": context["test_component_error_summary"],
                "summary": str(paths["summary"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
