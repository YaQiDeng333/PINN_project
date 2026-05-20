from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_multiline_tiny_smoke as tiny  # noqa: E402
import train_comsol_multi_defect_pilot_gate as pilot_v1  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_data_baseline_v3_candidate.npz"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_data_baseline_v3_topology_gated_v2_poc_summary.txt"
DEFAULT_AUDIT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_data_baseline_v3_topology_gated_v2_poc_failure_audit_summary.txt"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_topology_gated_v2_poc_metrics.csv"
DEFAULT_EPOCH_LOG = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_topology_gated_v2_poc_epoch_log.csv"
DEFAULT_SEED_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_topology_gated_v2_poc_seed_summary.csv"
DEFAULT_DEFECT_GROUP_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_topology_gated_v2_poc_defect_group_summary.csv"
DEFAULT_TASK_GROUP_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_topology_gated_v2_poc_task_group_summary.csv"
DEFAULT_CONNECTED_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_topology_gated_v2_poc_connected_component_summary.csv"
DEFAULT_COMPONENT_COUNT_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_topology_gated_v2_poc_component_count_summary.csv"
DEFAULT_GATE_SUMMARY = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_topology_gated_v2_poc_gate_summary.csv"
DEFAULT_FAILURE_CASES = PROJECT_ROOT / "results/metrics/comsol_data_baseline_v3_topology_gated_v2_poc_failure_cases.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_data_baseline_v3_topology_gated_v2_poc"

SEEDS = [42]
EXPECTED_SPLITS = {"train": 882, "val": 219, "test": 219}
EXPECTED_TASK_GROUPS = {"single_defect": 600, "multi_defect_cc2": 240, "multi_defect_cc3": 480}
THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
TASK_LABELS = ["single_defect", "multi_defect_cc2", "multi_defect_cc3"]
COUNT_LABELS = [1, 2, 3]
LAMBDA_TASK = 0.10
LAMBDA_COUNT = 0.05
LAMBDA_RESIDUAL = 1e-4
RESIDUAL_SCALE = 0.35
RESIDUAL_DOMINANCE_LIMIT = 0.75
MAJORITY_TASK_BASELINE = 480 / 1320
LIGHTWEIGHT_V3_REFERENCE = {
    "overall_iou": 0.6461,
    "overall_dice": 0.7735,
    "single_iou": 0.6602,
    "single_dice": 0.7917,
    "cc2_iou": 0.5598,
    "cc2_dice": 0.6996,
    "cc3_iou": 0.6717,
    "cc3_dice": 0.7879,
    "cc2_pred_cc_is_2": 0.8583,
    "cc3_pred_cc_is_3": 0.9542,
}
SINGLE_BASELINE_TEST = {"iou": 0.6515, "dice": 0.7861}
MULTI_BASELINE_TEST = {"iou": 0.6118, "dice": 0.7573, "pred_cc_is_2": 1.0}
THREE_COMPONENT_BASELINE_TEST = {"iou": 0.6761, "dice": 0.7958, "pred_cc_is_3": 0.9875}
SUBSTANTIAL_DROP = 0.03
_BASE_RASTERIZE_COMPONENT = pilot_v1.rasterize_component

METRIC_FIELDS = [
    "seed",
    "source_index",
    "sample_id",
    "split",
    "defect_group",
    "task_group",
    "defect_type",
    "component_count",
    "component_types",
    "component_type_combination",
    "source_dataset",
    "source_pack",
    "angle_deg",
    "vertex_count",
    "distance_bin",
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
    "residual_l2_loss",
    "base_abs_mean",
    "weighted_residual_abs_mean",
    "scaled_residual_abs_mean",
    "residual_to_base_abs_ratio",
    "residual_dominates_base",
    "prob_min",
    "prob_max",
    "prob_mean",
    "true_task_label",
    "pred_task_label",
    "task_gate_correct",
    "gate_prob_single",
    "gate_prob_cc2",
    "gate_prob_cc3",
    "true_count_label",
    "pred_count_label",
    "count_head_correct",
    "true_connected_component_count",
    "pred_connected_component_count",
    "predicted_component_count_correct",
    "predicted_component_count_is_1",
    "predicted_component_count_is_2",
    "predicted_component_count_is_3",
    "component_count_error",
    "missed_component_flag",
    "merged_component_flag",
    "split_component_flag",
    "extra_fragment_flag",
    "largest_component_area_ratio",
    "second_largest_component_area_ratio",
    "third_largest_component_area_ratio",
    "component_recall_heuristic",
    "polygon_component_present",
    "notes",
]

EPOCH_FIELDS = [
    "seed",
    *tiny.EPOCH_FIELDS,
    "best_val_threshold",
    "best_val_iou",
    "best_val_dice",
    "best_val_area_error",
    "best_val_score",
    "best_val_group_score_mean",
    "best_val_group_score_std",
    "train_task_ce",
    "train_count_ce",
    "train_residual_l2",
    "val_task_ce",
    "val_count_ce",
    "val_residual_l2",
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
    "residual_l2_loss_mean",
    "residual_to_base_abs_ratio_mean",
    "residual_dominance_rate",
    "pred_connected_component_count_mean",
    "predicted_component_count_correct_rate",
    "predicted_component_count_is_1_rate",
    "predicted_component_count_is_2_rate",
    "predicted_component_count_is_3_rate",
    "missed_component_rate",
    "merged_component_rate",
    "split_component_rate",
    "extra_fragment_rate",
    "component_recall_mean",
    "task_gate_accuracy",
    "count_head_accuracy",
]

SEED_SUMMARY_FIELDS = [
    "seed",
    "split",
    "best_epoch",
    "selected_threshold",
    "best_val_score",
    "best_val_group_score_mean",
    "best_val_group_score_std",
    "initial_train_loss",
    "final_train_loss",
    "train_loss_decreased",
    "sample_count",
    "iou_mean",
    "dice_mean",
    "area_error_mean",
    "center_error_mean",
    "pred_area_zero_sum",
    "total_loss_mean",
    "pred_connected_component_count_mean",
    "predicted_component_count_correct_rate",
    "predicted_component_count_is_1_rate",
    "predicted_component_count_is_2_rate",
    "predicted_component_count_is_3_rate",
    "missed_component_rate",
    "merged_component_rate",
    "split_component_rate",
    "extra_fragment_rate",
    "component_recall_mean",
    "task_gate_accuracy",
    "count_head_accuracy",
]

FAILURE_FIELDS = [
    "seed",
    "sample_id",
    "split",
    "defect_group",
    "task_group",
    "defect_type",
    "component_count",
    "component_type_combination",
    "source_dataset",
    "source_pack",
    "true_connected_component_count",
    "pred_connected_component_count",
    "IoU",
    "Dice",
    "area_error",
    "center_error",
    "pred_area",
    "true_area",
    "failure_category",
    "distance_bin",
    "short_note",
    "preview_path",
]

GATE_SUMMARY_FIELDS = [
    "seed",
    "split",
    "true_task_group",
    "pred_task_group",
    "sample_count",
    "task_gate_accuracy",
    "count_head_accuracy",
    "mean_gate_single",
    "mean_gate_cc2",
    "mean_gate_cc3",
    "gate_entropy_mean",
    "gate_collapse_flag",
]


class CombinedTrainingError(RuntimeError):
    pass


class GridDecoder(torch.nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 128 * 4 * 8),
            torch.nn.ReLU(inplace=True),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        feature = self.project(latent).view(latent.shape[0], 128, 4, 8)
        return self.decoder(feature).squeeze(1)


class TopologyResidualGatedMaskDecoder(torch.nn.Module):
    def __init__(self, n_lines: int, mask_h: int, mask_w: int):
        super().__init__()
        if (mask_h, mask_w) != (64, 128):
            raise ValueError("Topology residual gated decoder expects 64x128 masks.")
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(n_lines, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool1d(16),
            torch.nn.Flatten(),
        )
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(128 * 16, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.task_head = torch.nn.Linear(512, len(TASK_LABELS))
        self.count_head = torch.nn.Linear(512, len(COUNT_LABELS))
        self.base_decoder = GridDecoder(512)
        self.residual_adapters = torch.nn.ModuleList([GridDecoder(512) for _ in TASK_LABELS])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.latent(self.encoder(x))
        task_logits = self.task_head(latent)
        count_logits = self.count_head(latent)
        gate_probs = torch.softmax(task_logits, dim=1)
        base_mask_logits = self.base_decoder(latent)
        residual_logits = torch.stack([decoder(latent) for decoder in self.residual_adapters], dim=1)
        weighted_residual = (residual_logits * gate_probs[:, :, None, None]).sum(dim=1)
        mask_logits = base_mask_logits + RESIDUAL_SCALE * weighted_residual
        return {
            "mask_logits": mask_logits,
            "base_mask_logits": base_mask_logits,
            "weighted_residual": weighted_residual,
            "task_logits": task_logits,
            "count_logits": count_logits,
            "gate_probs": gate_probs,
            "residual_logits": residual_logits,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train COMSOL_DATA_BASELINE_V3 topology-aware residual gated decoder v2 candidate.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--seed-summary", type=Path, default=DEFAULT_SEED_SUMMARY)
    parser.add_argument("--defect-group-summary", type=Path, default=DEFAULT_DEFECT_GROUP_SUMMARY)
    parser.add_argument("--task-group-summary", type=Path, default=DEFAULT_TASK_GROUP_SUMMARY)
    parser.add_argument("--connected-summary", type=Path, default=DEFAULT_CONNECTED_SUMMARY)
    parser.add_argument("--component-count-summary", type=Path, default=DEFAULT_COMPONENT_COUNT_SUMMARY)
    parser.add_argument("--gate-summary", type=Path, default=DEFAULT_GATE_SUMMARY)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE_CASES)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--stage-label", default="POC")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return str(value)


def parse_json(value: Any) -> Any:
    return json.loads(as_text(value))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rasterize_component(component: dict[str, Any], mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    if as_text(component.get("component_type")) != "polygon":
        return _BASE_RASTERIZE_COMPONENT(component, mask_x, mask_y)
    vertices = np.asarray(component.get("polygon_vertices", []), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        return np.zeros((mask_y.shape[0], mask_x.shape[0]), dtype=np.uint8)
    yy, xx = np.meshgrid(mask_y, mask_x, indexing="ij")
    inside = np.zeros(xx.shape, dtype=bool)
    xj = vertices[-1, 0]
    yj = vertices[-1, 1]
    for xi, yi in vertices:
        crosses = ((yi > yy) != (yj > yy)) & (xx < (xj - xi) * (yy - yi) / ((yj - yi) + 1e-30) + xi)
        inside ^= crosses
        xj, yj = xi, yi
    return inside.astype(np.uint8)


def connected_components(mask: np.ndarray) -> tuple[int, list[int], np.ndarray]:
    return pilot_v1.connected_components(mask)


def component_recall(pred: np.ndarray, components: list[dict[str, Any]], mask_x: np.ndarray, mask_y: np.ndarray) -> float:
    hits = 0
    for component in components:
        component_mask = rasterize_component(component, mask_x, mask_y).astype(bool)
        area = int(component_mask.sum())
        if area == 0:
            continue
        overlap_fraction = float(np.logical_and(pred, component_mask).sum() / area)
        if overlap_fraction >= 0.20:
            hits += 1
    return float(hits / max(1, len(components)))


def extended_sample_metrics(
    prob: np.ndarray,
    target: np.ndarray,
    threshold: float,
    components: list[dict[str, Any]],
    mask_x: np.ndarray,
    mask_y: np.ndarray,
) -> dict[str, Any]:
    base = tiny.sample_metrics(prob, target, threshold)
    pred = prob >= threshold
    true = target >= 0.5
    true_cc, _, _ = connected_components(true)
    pred_cc, pred_areas, _ = connected_components(pred)
    sorted_areas = sorted(pred_areas, reverse=True)
    pred_area = max(1, int(pred.sum()))
    largest_ratio = float(sorted_areas[0] / pred_area) if sorted_areas else 0.0
    second_ratio = float(sorted_areas[1] / pred_area) if len(sorted_areas) > 1 else 0.0
    third_ratio = float(sorted_areas[2] / pred_area) if len(sorted_areas) > 2 else 0.0
    recall = component_recall(pred, components, mask_x, mask_y)
    return {
        **base,
        "true_connected_component_count": true_cc,
        "pred_connected_component_count": pred_cc,
        "predicted_component_count_correct": int(pred_cc == true_cc),
        "predicted_component_count_is_1": int(pred_cc == 1),
        "predicted_component_count_is_2": int(pred_cc == 2),
        "predicted_component_count_is_3": int(pred_cc == 3),
        "component_count_error": abs(pred_cc - true_cc),
        "missed_component_flag": int(recall < 1.0),
        "merged_component_flag": int(true_cc >= 2 and pred_cc == 1 and recall >= 1.0),
        "split_component_flag": int(pred_cc > true_cc),
        "extra_fragment_flag": int(pred_cc > true_cc),
        "largest_component_area_ratio": largest_ratio,
        "second_largest_component_area_ratio": second_ratio,
        "third_largest_component_area_ratio": third_ratio,
        "component_recall_heuristic": recall,
    }


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
        "defect_group",
        "task_group",
        "sample_ids",
        "split",
        "geometry_params",
        "components_json",
        "component_counts",
        "component_types",
        "connected_component_counts",
        "source_dataset",
        "source_pack",
        "distance_bin",
        "metadata",
    ]
    data = np.load(npz_path, allow_pickle=True)
    missing = [field for field in required if field not in data.files]
    if missing:
        raise CombinedTrainingError(f"missing fields: {missing}")
    delta = data["delta_bz"]
    masks = data["masks"]
    if delta.shape != (1320, 3, 201):
        raise CombinedTrainingError(f"unexpected delta_bz shape: {delta.shape}")
    if data["bz_defect"].shape != delta.shape or data["bz_no_defect"].shape != delta.shape:
        raise CombinedTrainingError("bz_defect / bz_no_defect shape mismatch")
    if masks.shape != (1320, 64, 128):
        raise CombinedTrainingError(f"unexpected masks shape: {masks.shape}")
    for name, shape in {"sensor_x": (201,), "scan_line_y": (3,), "mask_x": (128,), "mask_y": (64,)}.items():
        if data[name].shape != shape:
            raise CombinedTrainingError(f"{name} shape mismatch: {data[name].shape}")
        if not np.all(np.diff(data[name]) > 0):
            raise CombinedTrainingError(f"{name} is not strictly increasing")
    if not all(np.isfinite(data[name]).all() for name in ("delta_bz", "bz_defect", "bz_no_defect", "masks")):
        raise CombinedTrainingError("non-finite arrays")
    if not np.allclose(delta, data["bz_defect"] - data["bz_no_defect"], rtol=1e-9, atol=1e-12):
        raise CombinedTrainingError("delta_bz mismatch")
    if not np.all(masks.reshape(masks.shape[0], -1).sum(axis=1) > 0):
        raise CombinedTrainingError("empty mask found")
    sample_ids = np.array([as_text(item) for item in data["sample_ids"].tolist()])
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise CombinedTrainingError("sample_ids are not unique")
    splits = np.array([as_text(item) for item in data["split"].tolist()])
    split_counts = dict(Counter(splits.tolist()))
    if split_counts != EXPECTED_SPLITS:
        raise CombinedTrainingError(f"unexpected split counts: {split_counts}")
    defect_group = np.array([as_text(item) for item in data["defect_group"].tolist()])
    task_group = np.array([as_text(item) for item in data["task_group"].tolist()])
    defect_types = np.array([as_text(item) for item in data["defect_types"].tolist()])
    source_dataset = np.array([as_text(item) for item in data["source_dataset"].tolist()])
    source_pack = np.array([as_text(item) for item in data["source_pack"].tolist()])
    distance_bins = np.array([as_text(item) for item in data["distance_bin"].tolist()])
    component_counts = data["component_counts"].astype(np.int64)
    connected_counts = data["connected_component_counts"].astype(np.int64)
    components = [parse_json(item) for item in data["components_json"].tolist()]
    component_types_json = [parse_json(item) for item in data["component_types"].tolist()]
    geometries = [parse_json(item) for item in data["geometry_params"].tolist()]
    if dict(Counter(defect_group.tolist())) != {"single_defect": 600, "multi_defect": 720}:
        raise CombinedTrainingError(f"unexpected defect_group distribution: {Counter(defect_group.tolist())}")
    if dict(Counter(task_group.tolist())) != EXPECTED_TASK_GROUPS:
        raise CombinedTrainingError(f"unexpected task_group distribution: {Counter(task_group.tolist())}")
    if dict(Counter(component_counts.tolist())) != {1: 600, 2: 240, 3: 480}:
        raise CombinedTrainingError(f"unexpected component_count distribution: {Counter(component_counts.tolist())}")
    for expected_group, expected_count in (("single_defect", 1), ("multi_defect_cc2", 2), ("multi_defect_cc3", 3)):
        idx = np.where(task_group == expected_group)[0]
        if not np.all(component_counts[idx] == expected_count) or not np.all(connected_counts[idx] == expected_count):
            raise CombinedTrainingError(f"{expected_group} component/connected counts are not all {expected_count}")
    component_combos = np.array(["+".join(as_text(value) for value in values) for values in component_types_json])
    task_to_label = {name: index for index, name in enumerate(TASK_LABELS)}
    count_to_label = {count: index for index, count in enumerate(COUNT_LABELS)}
    task_labels = np.array([task_to_label[name] for name in task_group.tolist()], dtype=np.int64)
    count_labels = np.array([count_to_label[int(count)] for count in component_counts.tolist()], dtype=np.int64)
    angles: list[Any] = []
    vertex_counts: list[Any] = []
    polygon_present: list[str] = []
    for geometry, combo in zip(geometries, component_combos):
        geometry_dict = geometry if isinstance(geometry, dict) else {}
        angles.append(geometry_dict.get("angle_deg", geometry_dict.get("angle", "")))
        vertex_counts.append(geometry_dict.get("vertex_count", ""))
        polygon_present.append("yes" if "polygon" in combo else "no")
    return {
        "data": data,
        "sample_ids": sample_ids,
        "splits_array": splits,
        "splits": {name: np.where(splits == name)[0].tolist() for name in ("train", "val", "test")},
        "defect_group": defect_group,
        "task_group": task_group,
        "task_labels": task_labels,
        "defect_types": defect_types,
        "source_dataset": source_dataset,
        "source_pack": source_pack,
        "distance_bins": distance_bins,
        "components": components,
        "component_combos": component_combos,
        "component_counts": component_counts,
        "count_labels": count_labels,
        "connected_counts": connected_counts,
        "angles": np.array(angles, dtype=object),
        "vertex_counts": np.array(vertex_counts, dtype=object),
        "polygon_component_present": np.array(polygon_present),
        "split_counts": split_counts,
        "defect_group_counts": dict(Counter(defect_group.tolist())),
        "task_group_counts": dict(Counter(task_group.tolist())),
        "defect_type_counts": dict(Counter(defect_types.tolist())),
        "component_combo_counts": dict(Counter(component_combos.tolist())),
    }


def label_tensors(indices: torch.Tensor, validation: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = indices.cpu().numpy().astype(np.int64)
    task_labels = torch.from_numpy(validation["task_labels"][idx]).long().to(device)
    count_labels = torch.from_numpy(validation["count_labels"][idx]).long().to(device)
    return task_labels, count_labels


def gated_loss_components(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    task_labels: torch.Tensor,
    count_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_total, bce, dice = tiny.loss_components(outputs["mask_logits"], target)
    task_ce = F.cross_entropy(outputs["task_logits"], task_labels)
    count_ce = F.cross_entropy(outputs["count_logits"], count_labels)
    residual_l2 = torch.mean(outputs["weighted_residual"].pow(2))
    total = mask_total + LAMBDA_TASK * task_ce + LAMBDA_COUNT * count_ce + LAMBDA_RESIDUAL * residual_l2
    return total, bce, dice, task_ce, count_ce, residual_l2


def evaluate_gated_loss(
    model: torch.nn.Module,
    dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    validation: dict[str, Any],
) -> tuple[float, float, float, float, float, float]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    totals: list[float] = []
    bces: list[float] = []
    dices: list[float] = []
    task_ces: list[float] = []
    count_ces: list[float] = []
    residual_l2s: list[float] = []
    with torch.no_grad():
        for signals, targets, indices in loader:
            signals = signals.to(device)
            targets = targets.to(device)
            task_labels, count_labels = label_tensors(indices, validation, device)
            outputs = model(signals)
            total, bce, dice, task_ce, count_ce, residual_l2 = gated_loss_components(outputs, targets, task_labels, count_labels)
            totals.append(float(total.item()))
            bces.append(float(bce.item()))
            dices.append(float(dice.item()))
            task_ces.append(float(task_ce.item()))
            count_ces.append(float(count_ce.item()))
            residual_l2s.append(float(residual_l2.item()))
    return (
        float(np.mean(totals)),
        float(np.mean(bces)),
        float(np.mean(dices)),
        float(np.mean(task_ces)),
        float(np.mean(count_ces)),
        float(np.mean(residual_l2s)),
    )


def select_threshold_for_epoch(
    model: torch.nn.Module,
    val_dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    validation: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        signals, masks, indices = next(iter(DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)))
        prob = torch.sigmoid(model(signals.to(device))["mask_logits"]).cpu().numpy()
        target = masks.numpy()
        source_indices = indices.cpu().numpy().astype(np.int64)
    best = {
        "threshold": THRESHOLD_CANDIDATES[0],
        "iou": 0.0,
        "dice": 0.0,
        "area_error": float("inf"),
        "score": -float("inf"),
        "group_score_mean": -float("inf"),
        "group_score_std": float("inf"),
    }
    for threshold in THRESHOLD_CANDIDATES:
        metrics = [tiny.sample_metrics(prob[index], target[index], threshold) for index in range(prob.shape[0])]
        iou = float(np.mean([row["iou"] for row in metrics]))
        dice = float(np.mean([row["dice"] for row in metrics]))
        area_error = float(np.mean([row["area_error"] for row in metrics]))
        group_scores = []
        for task_group in TASK_LABELS:
            group_indices = [idx for idx, source_index in enumerate(source_indices) if as_text(validation["task_group"][source_index]) == task_group]
            if not group_indices:
                continue
            group_iou = float(np.mean([metrics[idx]["iou"] for idx in group_indices]))
            group_dice = float(np.mean([metrics[idx]["dice"] for idx in group_indices]))
            group_area_error = float(np.mean([metrics[idx]["area_error"] for idx in group_indices]))
            group_scores.append(group_iou + group_dice - group_area_error)
        group_score_mean = float(np.mean(group_scores)) if group_scores else -float("inf")
        group_score_std = float(np.std(group_scores, ddof=0)) if group_scores else float("inf")
        score = group_score_mean - 0.25 * group_score_std
        if score > best["score"]:
            best = {
                "threshold": threshold,
                "iou": iou,
                "dice": dice,
                "area_error": area_error,
                "score": score,
                "group_score_mean": group_score_mean,
                "group_score_std": group_score_std,
            }
    return best


def evaluate_model(
    model: torch.nn.Module,
    dataset: tiny.ComsolSmokeDataset,
    device: torch.device,
    threshold: float,
    validation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    data = validation["data"]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    probs: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for signals, masks, indices in loader:
            signals = signals.to(device)
            masks = masks.to(device)
            task_labels, count_labels = label_tensors(indices, validation, device)
            outputs = model(signals)
            total, bce, dice, task_ce, count_ce, residual_l2 = gated_loss_components(outputs, masks, task_labels, count_labels)
            prob = torch.sigmoid(outputs["mask_logits"]).cpu().numpy()[0]
            gate_probs = outputs["gate_probs"].cpu().numpy()[0]
            pred_task_label = int(np.argmax(gate_probs))
            pred_count_label = int(torch.argmax(outputs["count_logits"], dim=1).cpu().numpy()[0])
            base_abs_mean = float(outputs["base_mask_logits"].abs().mean().item())
            weighted_residual_abs_mean = float(outputs["weighted_residual"].abs().mean().item())
            scaled_residual_abs_mean = RESIDUAL_SCALE * weighted_residual_abs_mean
            residual_to_base_ratio = scaled_residual_abs_mean / max(base_abs_mean, 1e-8)
            target = masks.cpu().numpy()[0]
            index = int(indices.item())
            probs[index] = prob
            metrics = extended_sample_metrics(
                prob,
                target,
                threshold,
                validation["components"][index],
                data["mask_x"].astype(np.float64),
                data["mask_y"].astype(np.float64),
            )
            rows.append(
                {
                    "source_index": index,
                    "sample_id": as_text(validation["sample_ids"][index]),
                    "split": as_text(validation["splits_array"][index]),
                    "defect_group": as_text(validation["defect_group"][index]),
                    "task_group": as_text(validation["task_group"][index]),
                    "defect_type": as_text(validation["defect_types"][index]),
                    "component_count": int(validation["component_counts"][index]),
                    "component_types": as_text(validation["component_combos"][index]),
                    "component_type_combination": as_text(validation["component_combos"][index]),
                    "source_dataset": as_text(validation["source_dataset"][index]),
                    "source_pack": as_text(validation["source_pack"][index]),
                    "angle_deg": as_text(validation["angles"][index]),
                    "vertex_count": as_text(validation["vertex_counts"][index]),
                    "distance_bin": as_text(validation["distance_bins"][index]),
                    "threshold": threshold,
                    **metrics,
                    "bce_loss": float(bce.item()),
                    "dice_loss": float(dice.item()),
                    "total_loss": float(total.item()),
                    "residual_l2_loss": float(residual_l2.item()),
                    "base_abs_mean": base_abs_mean,
                    "weighted_residual_abs_mean": weighted_residual_abs_mean,
                    "scaled_residual_abs_mean": scaled_residual_abs_mean,
                    "residual_to_base_abs_ratio": residual_to_base_ratio,
                    "residual_dominates_base": int(residual_to_base_ratio > RESIDUAL_DOMINANCE_LIMIT),
                    "prob_min": float(prob.min()),
                    "prob_max": float(prob.max()),
                    "prob_mean": float(prob.mean()),
                    "true_task_label": int(validation["task_labels"][index]),
                    "pred_task_label": pred_task_label,
                    "task_gate_correct": int(pred_task_label == int(validation["task_labels"][index])),
                    "gate_prob_single": float(gate_probs[0]),
                    "gate_prob_cc2": float(gate_probs[1]),
                    "gate_prob_cc3": float(gate_probs[2]),
                    "true_count_label": int(validation["count_labels"][index]),
                    "pred_count_label": pred_count_label,
                    "count_head_correct": int(pred_count_label == int(validation["count_labels"][index])),
                    "polygon_component_present": as_text(validation["polygon_component_present"][index]),
                    "notes": "topology_gated_v2_candidate_eval",
                }
            )
    return rows, probs


def train_one_seed(seed: int, args: argparse.Namespace, validation: dict[str, Any], device: torch.device) -> dict[str, Any]:
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
    all_dataset = tiny.ComsolSmokeDataset(normalized, masks, list(range(delta_bz.shape[0])))
    model = TopologyResidualGatedMaskDecoder(delta_bz.shape[1], masks.shape[1], masks.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    best_state = deepcopy(model.state_dict())
    best_score = -float("inf")
    best_group_score_mean = -float("inf")
    best_group_score_std = float("inf")
    best_epoch = 0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_totals: list[float] = []
        batch_bces: list[float] = []
        batch_dices: list[float] = []
        batch_task_ces: list[float] = []
        batch_count_ces: list[float] = []
        batch_residual_l2s: list[float] = []
        for signals, targets, indices in train_loader:
            signals = signals.to(device)
            targets = targets.to(device)
            task_labels, count_labels = label_tensors(indices, validation, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(signals)
            total, bce, dice, task_ce, count_ce, residual_l2 = gated_loss_components(outputs, targets, task_labels, count_labels)
            total.backward()
            optimizer.step()
            batch_totals.append(float(total.item()))
            batch_bces.append(float(bce.item()))
            batch_dices.append(float(dice.item()))
            batch_task_ces.append(float(task_ce.item()))
            batch_count_ces.append(float(count_ce.item()))
            batch_residual_l2s.append(float(residual_l2.item()))
        train_loss = {
            "total_loss": float(np.mean(batch_totals)),
            "bce_loss": float(np.mean(batch_bces)),
            "dice_loss": float(np.mean(batch_dices)),
            "task_ce": float(np.mean(batch_task_ces)),
            "count_ce": float(np.mean(batch_count_ces)),
            "residual_l2": float(np.mean(batch_residual_l2s)),
        }
        val_total, val_bce, val_dice, val_task_ce, val_count_ce, val_residual_l2 = evaluate_gated_loss(model, val_dataset, device, validation)
        best_val = select_threshold_for_epoch(model, val_dataset, device, validation)
        epoch_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss["total_loss"],
                "train_bce": train_loss["bce_loss"],
                "train_dice_loss": train_loss["dice_loss"],
                "val_loss": float(val_total),
                "val_bce": float(val_bce),
                "val_dice_loss": float(val_dice),
                "best_val_threshold": best_val["threshold"],
                "best_val_iou": best_val["iou"],
                "best_val_dice": best_val["dice"],
                "best_val_area_error": best_val["area_error"],
                "best_val_score": best_val["score"],
                "best_val_group_score_mean": best_val["group_score_mean"],
                "best_val_group_score_std": best_val["group_score_std"],
                "train_task_ce": train_loss["task_ce"],
                "train_count_ce": train_loss["count_ce"],
                "train_residual_l2": train_loss["residual_l2"],
                "val_task_ce": float(val_task_ce),
                "val_count_ce": float(val_count_ce),
                "val_residual_l2": float(val_residual_l2),
            }
        )
        if best_val["score"] > best_score:
            best_score = best_val["score"]
            best_group_score_mean = best_val["group_score_mean"]
            best_group_score_std = best_val["group_score_std"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected = select_threshold_for_epoch(model, val_dataset, device, validation)
    metric_rows, all_probs = evaluate_model(model, all_dataset, device, float(selected["threshold"]), validation)
    for row in metric_rows:
        row["seed"] = seed
    preview_probs = {}
    if seed == SEEDS[0]:
        preview_indices = choose_preview_indices(metric_rows)
        preview_probs = {index: all_probs[index] for index in preview_indices if index in all_probs}
    split_summaries = {
        split_name: summarize([row for row in metric_rows if row["split"] == split_name], "all", split_name, seed)
        for split_name in ("train", "val", "test")
    }
    return {
        "seed": seed,
        "model": model,
        "threshold": float(selected["threshold"]),
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "best_val_group_score_mean": best_group_score_mean,
        "best_val_group_score_std": best_group_score_std,
        "epoch_rows": epoch_rows,
        "metric_rows": metric_rows,
        "selected_probs": preview_probs,
        "split_summaries": split_summaries,
        "initial_train_loss": float(epoch_rows[0]["train_loss"]),
        "final_train_loss": float(epoch_rows[-1]["train_loss"]),
        "train_loss_decreased": float(epoch_rows[-1]["train_loss"]) < float(epoch_rows[0]["train_loss"]),
        "train_mean_shape": tuple(train_mean.shape),
        "train_std_shape": tuple(train_std.shape),
    }


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def summarize(rows: list[dict[str, Any]], group: str, split: str, seed: Any = "all") -> dict[str, Any]:
    return {
        "seed": seed,
        "group": group,
        "split": split,
        "sample_count": len(rows),
        "iou_mean": mean_or_nan([float(row["iou"]) for row in rows]),
        "dice_mean": mean_or_nan([float(row["dice"]) for row in rows]),
        "area_error_mean": mean_or_nan([float(row["area_error"]) for row in rows]),
        "center_error_mean": mean_or_nan([float(row["center_error"]) for row in rows if as_text(row["center_error"]).lower() != "nan"]),
        "pred_area_mean": mean_or_nan([float(row["pred_area"]) for row in rows]),
        "true_area_mean": mean_or_nan([float(row["true_area"]) for row in rows]),
        "pred_area_zero_sum": int(sum(int(row["pred_area_zero"]) for row in rows)),
        "total_loss_mean": mean_or_nan([float(row["total_loss"]) for row in rows]),
        "residual_l2_loss_mean": mean_or_nan([float(row["residual_l2_loss"]) for row in rows]),
        "residual_to_base_abs_ratio_mean": mean_or_nan([float(row["residual_to_base_abs_ratio"]) for row in rows]),
        "residual_dominance_rate": mean_or_nan([float(row["residual_dominates_base"]) for row in rows]),
        "pred_connected_component_count_mean": mean_or_nan([float(row["pred_connected_component_count"]) for row in rows]),
        "predicted_component_count_correct_rate": mean_or_nan([float(row["predicted_component_count_correct"]) for row in rows]),
        "predicted_component_count_is_1_rate": mean_or_nan([float(row["predicted_component_count_is_1"]) for row in rows]),
        "predicted_component_count_is_2_rate": mean_or_nan([float(row["predicted_component_count_is_2"]) for row in rows]),
        "predicted_component_count_is_3_rate": mean_or_nan([float(row["predicted_component_count_is_3"]) for row in rows]),
        "missed_component_rate": mean_or_nan([float(row["missed_component_flag"]) for row in rows]),
        "merged_component_rate": mean_or_nan([float(row["merged_component_flag"]) for row in rows]),
        "split_component_rate": mean_or_nan([float(row["split_component_flag"]) for row in rows]),
        "extra_fragment_rate": mean_or_nan([float(row["extra_fragment_flag"]) for row in rows]),
        "component_recall_mean": mean_or_nan([float(row["component_recall_heuristic"]) for row in rows]),
        "task_gate_accuracy": mean_or_nan([float(row["task_gate_correct"]) for row in rows]),
        "count_head_accuracy": mean_or_nan([float(row["count_head_correct"]) for row in rows]),
    }


def grouped_summary(metric_rows: list[dict[str, Any]], key: str, values: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test", "all"):
        source = metric_rows if split_name == "all" else [row for row in metric_rows if row["split"] == split_name]
        for value in values:
            selected = [row for row in source if row[key] == value]
            rows.append(summarize(selected, f"{key}={value}", split_name))
    return rows


def build_gate_summary(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in metric_rows}):
        seed_rows = [row for row in metric_rows if int(row["seed"]) == seed]
        for split_name in ("train", "val", "test", "all"):
            split_rows = seed_rows if split_name == "all" else [row for row in seed_rows if row["split"] == split_name]
            for true_task in [*TASK_LABELS, "all"]:
                source = split_rows if true_task == "all" else [row for row in split_rows if row["task_group"] == true_task]
                if not source:
                    continue
                gate_probs = np.asarray(
                    [[float(row["gate_prob_single"]), float(row["gate_prob_cc2"]), float(row["gate_prob_cc3"])] for row in source],
                    dtype=np.float64,
                )
                gate_entropy = -np.sum(gate_probs * np.log(np.maximum(gate_probs, 1e-12)), axis=1)
                predicted = [TASK_LABELS[int(row["pred_task_label"])] for row in source]
                for pred_task in [*TASK_LABELS, "all"]:
                    selected = source if pred_task == "all" else [row for row in source if TASK_LABELS[int(row["pred_task_label"])] == pred_task]
                    if pred_task != "all" and not selected:
                        continue
                    selected_probs = np.asarray(
                        [[float(row["gate_prob_single"]), float(row["gate_prob_cc2"]), float(row["gate_prob_cc3"])] for row in selected],
                        dtype=np.float64,
                    )
                    selected_entropy = -np.sum(selected_probs * np.log(np.maximum(selected_probs, 1e-12)), axis=1)
                    rows.append(
                        {
                            "seed": seed,
                            "split": split_name,
                            "true_task_group": true_task,
                            "pred_task_group": pred_task,
                            "sample_count": len(selected),
                            "task_gate_accuracy": mean_or_nan([float(row["task_gate_correct"]) for row in selected]),
                            "count_head_accuracy": mean_or_nan([float(row["count_head_correct"]) for row in selected]),
                            "mean_gate_single": float(selected_probs[:, 0].mean()),
                            "mean_gate_cc2": float(selected_probs[:, 1].mean()),
                            "mean_gate_cc3": float(selected_probs[:, 2].mean()),
                            "gate_entropy_mean": float(selected_entropy.mean()),
                            "gate_collapse_flag": int(max(Counter(predicted).values()) / len(predicted) > 0.95),
                        }
                    )
    return rows


def mean_std(seed_results: list[dict[str, Any]], split: str, field: str) -> tuple[float, float]:
    values = [float(result["split_summaries"][split][field]) for result in seed_results]
    return float(np.mean(values)), float(np.std(values, ddof=0))


def format_mean_std(seed_results: list[dict[str, Any]], split: str) -> str:
    parts = []
    for field, label in (
        ("iou_mean", "IoU"),
        ("dice_mean", "Dice"),
        ("area_error_mean", "area_error"),
        ("center_error_mean", "center_error"),
        ("predicted_component_count_correct_rate", "pred_cc_correct"),
        ("predicted_component_count_is_2_rate", "pred_cc_is_2"),
        ("predicted_component_count_is_3_rate", "pred_cc_is_3"),
        ("missed_component_rate", "missed"),
        ("merged_component_rate", "merged"),
        ("split_component_rate", "split"),
        ("extra_fragment_rate", "extra"),
    ):
        mean, std = mean_std(seed_results, split, field)
        parts.append(f"{label}={mean:.4f}+/-{std:.4f}")
    return ", ".join(parts)


def seed_summary_rows(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in seed_results:
        for split_name in ("train", "val", "test"):
            summary = result["split_summaries"][split_name]
            rows.append(
                {
                    "seed": result["seed"],
                    "split": split_name,
                    "best_epoch": result["best_epoch"],
                    "selected_threshold": result["threshold"],
                    "best_val_score": result["best_val_score"],
                    "best_val_group_score_mean": result["best_val_group_score_mean"],
                    "best_val_group_score_std": result["best_val_group_score_std"],
                    "initial_train_loss": result["initial_train_loss"],
                    "final_train_loss": result["final_train_loss"],
                    "train_loss_decreased": result["train_loss_decreased"],
                    **{key: value for key, value in summary.items() if key not in {"seed", "group", "split"}},
                }
            )
    return rows


def choose_preview_indices(metric_rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected:
            selected.append(index)

    val_test = [row for row in metric_rows if row["split"] in {"val", "test"}]
    for key, values in (
        ("defect_group", ["single_defect", "multi_defect"]),
        ("task_group", ["single_defect", "multi_defect_cc2", "multi_defect_cc3"]),
        ("defect_type", ["rectangular_notch", "rotated_rect", "polygon", "multi_defect"]),
    ):
        for value in values:
            rows = [row for row in val_test if row[key] == value]
            if rows:
                add(int(max(rows, key=lambda row: float(row["dice"]))["source_index"]))
                add(int(min(rows, key=lambda row: float(row["dice"]))["source_index"]))
    cc_mismatch = [row for row in val_test if int(row["predicted_component_count_correct"]) == 0]
    for row in sorted(cc_mismatch, key=lambda item: float(item["dice"]))[:8]:
        add(int(row["source_index"]))
    for reverse in (True, False):
        for row in sorted(val_test, key=lambda item: float(item["dice"]), reverse=reverse)[:12]:
            add(int(row["source_index"]))
    return selected[:36]


def make_previews(
    preview_dir: Path,
    probs: dict[int, np.ndarray],
    masks: np.ndarray,
    delta_bz: np.ndarray,
    sensor_x: np.ndarray,
    scan_line_y: np.ndarray,
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, str]:
    return pilot_v1.make_previews(preview_dir, probs, masks, delta_bz, sensor_x, scan_line_y, rows, threshold)


def failure_category(row: dict[str, Any]) -> tuple[str, str]:
    if int(row["pred_area_zero"]):
        return "empty_prediction", "predicted area is zero"
    if int(row["predicted_component_count_correct"]) == 0:
        return "component_count_mismatch", f"pred_cc={row['pred_connected_component_count']}, true_cc={row['true_connected_component_count']}"
    if float(row["area_error"]) > 0.35:
        return "area_error", "large area mismatch"
    if float(row["center_error"]) > 8.0:
        return "localization", "large center error"
    return "boundary/shape smoothing", "mask area and component count are plausible but boundary remains coarse"


def build_failure_cases(metric_rows: list[dict[str, Any]], preview_paths: dict[str, str]) -> list[dict[str, Any]]:
    seed_rows = [row for row in metric_rows if int(row["seed"]) == SEEDS[0] and row["split"] in {"val", "test"}]
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any]) -> None:
        if not any(existing["sample_id"] == row["sample_id"] for existing in selected):
            selected.append(row)

    for row in sorted(seed_rows, key=lambda item: float(item["dice"]))[:12]:
        add(row)
    for row in sorted(seed_rows, key=lambda item: float(item["area_error"]), reverse=True)[:8]:
        add(row)
    for row in sorted(seed_rows, key=lambda item: float(item["center_error"]), reverse=True)[:8]:
        add(row)
    for row in [row for row in seed_rows if int(row["predicted_component_count_correct"]) == 0][:8]:
        add(row)
    cases: list[dict[str, Any]] = []
    for row in selected[:36]:
        category, note = failure_category(row)
        cases.append(
            {
                "seed": row["seed"],
                "sample_id": row["sample_id"],
                "split": row["split"],
                "defect_group": row["defect_group"],
                "task_group": row["task_group"],
                "defect_type": row["defect_type"],
                "component_count": row["component_count"],
                "component_type_combination": row["component_type_combination"],
                "source_dataset": row["source_dataset"],
                "source_pack": row["source_pack"],
                "true_connected_component_count": row["true_connected_component_count"],
                "pred_connected_component_count": row["pred_connected_component_count"],
                "IoU": row["iou"],
                "Dice": row["dice"],
                "area_error": row["area_error"],
                "center_error": row["center_error"],
                "pred_area": row["pred_area"],
                "true_area": row["true_area"],
                "failure_category": category,
                "distance_bin": row["distance_bin"],
                "short_note": note,
                "preview_path": preview_paths.get(row["sample_id"], ""),
            }
        )
    return cases


def summary_lookup(rows: list[dict[str, Any]], group: str, split: str) -> dict[str, Any]:
    matches = [row for row in rows if row["group"] == group and row["split"] == split]
    return matches[0] if matches else summarize([], group, split)


def build_training_summary(context: dict[str, Any]) -> str:
    lines = [
        f"# COMSOL_DATA_BASELINE_V3 topology-aware residual gated decoder v2 {context['stage_label']} summary",
        "",
        f"- combined NPZ readable: {context['npz_readable']}",
        f"- schema complete: {context['schema_complete']}",
        f"- split distribution: {context['split_counts']}",
        f"- input shape: {context['input_shape']}",
        f"- output shape: {context['output_shape']}",
        f"- normalization train-only: {context['normalization_train_only']}",
        f"- model: shared Conv1d BzEncoder + shared base decoder + predicted task gate + 3 residual adapters, delta_bz input only",
        f"- final logits: base_mask_logits + residual_scale * predicted_gate_weighted_residual",
        f"- residual_scale: {RESIDUAL_SCALE}",
        f"- auxiliary loss: task_group CE lambda={LAMBDA_TASK}, component_count CE lambda={LAMBDA_COUNT}, residual L2 lambda={LAMBDA_RESIDUAL}",
        f"- checkpoint score: mean(task_group val scores) - 0.25 * std(task_group val scores), validation only",
        f"- seeds completed: {context['seeds_completed']}",
        f"- best epoch / threshold per seed: {context['seed_best']}",
        f"- train mean+/-std: {context['train_mean_std']}",
        f"- val mean+/-std: {context['val_mean_std']}",
        f"- test mean+/-std: {context['test_mean_std']}",
        "",
        "## Group Metrics",
        "",
        f"- single_defect test: {context['single_test']}",
        f"- multi_defect_cc2 test: {context['cc2_test']}",
        f"- multi_defect_cc3 test: {context['cc3_test']}",
        f"- multi_defect combined test: {context['multi_test']}",
        f"- task_group test summary: {context['task_group_test']}",
        f"- defect_type test summary: {context['defect_type_test']}",
        f"- connected-component behavior: {context['cc_behavior_test']}",
        f"- missed / merged / split / extra behavior: {context['multi_error_test']}",
        "",
        "## Gate Diagnostics",
        "",
        f"- task gate accuracy: {context['task_gate_accuracy']}",
        f"- component_count head accuracy: {context['count_head_accuracy']}",
        f"- majority task baseline: {MAJORITY_TASK_BASELINE}",
        f"- gate above majority baseline: {context['gate_above_majority']}",
        f"- gate predicted distribution: {context['gate_pred_distribution']}",
        f"- gate collapse: {context['gate_collapse']}",
        "",
        "## Residual Diagnostics",
        "",
        f"- residual scale: {context['residual_scale']}",
        f"- test residual L2 mean: {context['test_residual_l2_mean']}",
        f"- test residual/base absolute ratio mean: {context['test_residual_to_base_ratio_mean']}",
        f"- residual dominance rate: {context['residual_dominance_rate']}",
        f"- residual dominates base logits: {context['residual_dominates_base']}",
        "",
        "## Comparison To 20.42 Lightweight Candidate",
        "",
        f"- 20.42 reference: {context['lightweight_reference']}",
        f"- cc2 improvement: {context['cc2_improvement']}",
        f"- cc2 pred_cc_is_2 tolerance pass: {context['cc2_pred_ok']}",
        f"- single_defect IoU delta: {context['single_iou_delta']}",
        f"- cc3 IoU delta: {context['cc3_iou_delta']}",
        f"- single_defect tolerance pass: {context['single_ok']}",
        f"- cc3 tolerance pass: {context['cc3_ok']}",
        f"- overall tolerance pass: {context['overall_ok']}",
        f"- residual dominance pass: {context['residual_ok']}",
        f"- finite metrics / no NaN: {context['finite_metrics']}",
        f"- {context['stage_label']} acceptance passed: {context['acceptance_passed']}",
        "",
        "## Standalone Baseline Comparison",
        "",
        f"- single_defect substantial degradation vs COMSOL_DATA_BASELINE: {context['single_degraded']}",
        f"- cc2 substantial degradation vs COMSOL_MULTI_DEFECT_DATA_BASELINE: {context['cc2_degraded']}",
        f"- cc3 substantial degradation vs COMSOL_THREE_COMPONENT_DATA_BASELINE: {context['cc3_degraded']}",
        f"- preview generated: {context['preview_generated']}",
        f"- preview dir: {context['preview_dir']}",
        f"- should run Stage B 3-seed: {context['should_run_stage_b']}",
        "",
        "## Limitations",
        "",
        "- COMSOL data-domain only; not v3_complex and not CURRENT_BASELINE.",
        "- Controlled synthetic pilot data only.",
        "- Component_count coverage is limited to 1, 2, and 3.",
        "- Component_count=3 samples do not include polygon components.",
        "- No real experimental data.",
    ]
    return "\n".join(lines) + "\n"


def build_audit_summary(context: dict[str, Any]) -> str:
    lines = [
        f"# COMSOL_DATA_BASELINE_V3 topology-aware residual gated decoder v2 {context['stage_label']} failure audit",
        "",
        f"- {context['stage_label']} acceptance passed: {context['acceptance_passed']}",
        f"- cc2 improved vs 20.42: {context['cc2_improved']}",
        f"- cc2 pred_cc tolerance pass: {context['cc2_pred_ok']}",
        f"- single_defect tolerance pass: {context['single_ok']}",
        f"- cc3 tolerance pass: {context['cc3_ok']}",
        f"- overall tolerance pass: {context['overall_ok']}",
        f"- task gate accuracy: {context['task_gate_accuracy']}",
        f"- gate collapse: {context['gate_collapse']}",
        f"- residual scale: {context['residual_scale']}",
        f"- residual/base absolute ratio mean: {context['test_residual_to_base_ratio_mean']}",
        f"- residual dominance rate: {context['residual_dominance_rate']}",
        f"- residual dominates base logits: {context['residual_dominates_base']}",
        f"- single_defect degradation vs COMSOL_DATA_BASELINE: {context['single_degraded']}",
        f"- cc2 degradation vs COMSOL_MULTI_DEFECT_DATA_BASELINE: {context['cc2_degraded']}",
        f"- cc3 degradation vs COMSOL_THREE_COMPONENT_DATA_BASELINE: {context['cc3_degraded']}",
        f"- model still predicts correct component counts for cc2/cc3: {context['multi_pred_cc_ok']}",
        f"- hardest group: {context['hardest_group']}",
        f"- dominant failure mode: {context['dominant_failure']}",
        f"- source_dataset/source_pack issue: {context['source_issue']}",
        f"- recommended next step: {context['recommended_next_step']}",
        "",
        "## Interpretation",
        "",
        context["interpretation"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    paths = {
        "npz": resolve(args.npz),
        "summary": resolve(args.summary),
        "audit_summary": resolve(args.audit_summary),
        "metrics": resolve(args.metrics),
        "epoch_log": resolve(args.epoch_log),
        "seed_summary": resolve(args.seed_summary),
        "defect_group_summary": resolve(args.defect_group_summary),
        "task_group_summary": resolve(args.task_group_summary),
        "connected_summary": resolve(args.connected_summary),
        "component_count_summary": resolve(args.component_count_summary),
        "gate_summary": resolve(args.gate_summary),
        "failure_cases": resolve(args.failure_cases),
        "preview_dir": resolve(args.preview_dir),
    }
    validation = validate_npz(paths["npz"])
    pilot_v1.rasterize_component = rasterize_component
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = [train_one_seed(seed, args, validation, device) for seed in args.seeds]
    all_metric_rows = [row for result in seed_results for row in result["metric_rows"]]
    all_epoch_rows = [row for result in seed_results for row in result["epoch_rows"]]

    defect_group_rows = grouped_summary(all_metric_rows, "defect_group", ["single_defect", "multi_defect"])
    task_group_rows = grouped_summary(all_metric_rows, "task_group", ["single_defect", "multi_defect_cc2", "multi_defect_cc3"])
    defect_type_rows = grouped_summary(all_metric_rows, "defect_type", ["rectangular_notch", "rotated_rect", "polygon", "multi_defect"])
    connected_rows = grouped_summary(
        all_metric_rows,
        "pred_connected_component_count",
        sorted({int(row["pred_connected_component_count"]) for row in all_metric_rows}),
    )
    component_count_rows = grouped_summary(
        all_metric_rows,
        "component_count",
        sorted({int(row["component_count"]) for row in all_metric_rows}),
    )
    component_rows = grouped_summary(
        all_metric_rows,
        "component_type_combination",
        sorted({row["component_type_combination"] for row in all_metric_rows}),
    )
    component_rows.extend(grouped_summary(all_metric_rows, "polygon_component_present", ["no", "yes"]))
    component_rows.extend(grouped_summary(all_metric_rows, "distance_bin", sorted(set(row["distance_bin"] for row in all_metric_rows))))
    gate_rows = build_gate_summary(all_metric_rows)

    preview_result = seed_results[0]
    preview_indices = choose_preview_indices(preview_result["metric_rows"])
    preview_probs = {index: preview_result["selected_probs"].get(index) for index in preview_indices if index in preview_result["selected_probs"]}
    preview_rows = [row for row in preview_result["metric_rows"] if int(row["source_index"]) in preview_probs]
    data = validation["data"]
    preview_paths = make_previews(
        paths["preview_dir"],
        preview_probs,
        data["masks"].astype(np.float32),
        data["delta_bz"].astype(np.float32),
        data["sensor_x"].astype(np.float64),
        data["scan_line_y"].astype(np.float64),
        preview_rows,
        preview_result["threshold"],
    )
    failure_cases = build_failure_cases(all_metric_rows, preview_paths)

    write_csv(paths["metrics"], all_metric_rows, METRIC_FIELDS)
    write_csv(paths["epoch_log"], all_epoch_rows, EPOCH_FIELDS)
    write_csv(paths["seed_summary"], seed_summary_rows(seed_results), SEED_SUMMARY_FIELDS)
    write_csv(paths["defect_group_summary"], defect_group_rows, SUMMARY_FIELDS)
    write_csv(paths["task_group_summary"], task_group_rows, SUMMARY_FIELDS)
    write_csv(paths["connected_summary"], connected_rows, SUMMARY_FIELDS)
    write_csv(paths["component_count_summary"], component_count_rows, SUMMARY_FIELDS)
    write_csv(paths["gate_summary"], gate_rows, GATE_SUMMARY_FIELDS)
    write_csv(paths["failure_cases"], failure_cases, FAILURE_FIELDS)

    single_test = summary_lookup(defect_group_rows, "defect_group=single_defect", "test")
    multi_test = summary_lookup(defect_group_rows, "defect_group=multi_defect", "test")
    cc2_test = summary_lookup(task_group_rows, "task_group=multi_defect_cc2", "test")
    cc3_test = summary_lookup(task_group_rows, "task_group=multi_defect_cc3", "test")
    single_degraded = (
        SINGLE_BASELINE_TEST["iou"] - float(single_test["iou_mean"]) > SUBSTANTIAL_DROP
        or SINGLE_BASELINE_TEST["dice"] - float(single_test["dice_mean"]) > SUBSTANTIAL_DROP
    )
    cc2_degraded = (
        MULTI_BASELINE_TEST["iou"] - float(cc2_test["iou_mean"]) > SUBSTANTIAL_DROP
        or MULTI_BASELINE_TEST["dice"] - float(cc2_test["dice_mean"]) > SUBSTANTIAL_DROP
    )
    cc3_degraded = (
        THREE_COMPONENT_BASELINE_TEST["iou"] - float(cc3_test["iou_mean"]) > SUBSTANTIAL_DROP
        or THREE_COMPONENT_BASELINE_TEST["dice"] - float(cc3_test["dice_mean"]) > SUBSTANTIAL_DROP
    )
    cc2_rows = [row for row in all_metric_rows if row["split"] == "test" and row["task_group"] == "multi_defect_cc2"]
    cc3_rows = [row for row in all_metric_rows if row["split"] == "test" and row["task_group"] == "multi_defect_cc3"]
    test_rows = [row for row in all_metric_rows if row["split"] == "test"]
    multi_rows = [row for row in all_metric_rows if row["split"] == "test" and row["defect_group"] == "multi_defect"]
    cc2_pred_cc2 = float(np.mean([float(row["predicted_component_count_is_2"]) for row in cc2_rows])) if cc2_rows else float("nan")
    cc3_pred_cc3 = float(np.mean([float(row["predicted_component_count_is_3"]) for row in cc3_rows])) if cc3_rows else float("nan")
    residual_to_base_ratio_mean = mean_or_nan([float(row["residual_to_base_abs_ratio"]) for row in test_rows])
    residual_l2_mean = mean_or_nan([float(row["residual_l2_loss"]) for row in test_rows])
    residual_dominance_rate = mean_or_nan([float(row["residual_dominates_base"]) for row in test_rows])
    residual_dominates_base = bool(np.isfinite(residual_to_base_ratio_mean) and residual_to_base_ratio_mean > RESIDUAL_DOMINANCE_LIMIT) or (
        bool(np.isfinite(residual_dominance_rate)) and residual_dominance_rate > 0.05
    )
    residual_ok = not residual_dominates_base
    multi_errors = {
        "missed": float(np.mean([float(row["missed_component_flag"]) for row in multi_rows])) if multi_rows else float("nan"),
        "merged": float(np.mean([float(row["merged_component_flag"]) for row in multi_rows])) if multi_rows else float("nan"),
        "split": float(np.mean([float(row["split_component_flag"]) for row in multi_rows])) if multi_rows else float("nan"),
        "extra": float(np.mean([float(row["extra_fragment_flag"]) for row in multi_rows])) if multi_rows else float("nan"),
    }
    test_group_rows = [row for row in [*task_group_rows, *defect_type_rows, *component_rows] if row["split"] == "test" and int(row["sample_count"]) > 0]
    hardest = min(test_group_rows, key=lambda row: float(row["dice_mean"])) if test_group_rows else {"group": "n/a", "dice_mean": "nan"}
    failure_counts = Counter(row["failure_category"] for row in failure_cases)
    dominant_failure = failure_counts.most_common(1)[0][0] if failure_counts else "none"
    source_issue = False
    for task_group in sorted(set(row["task_group"] for row in all_metric_rows)):
        task_test = [row for row in all_metric_rows if row["split"] == "test" and row["task_group"] == task_group]
        task_sources = sorted(set(row["source_pack"] for row in task_test))
        source_dices = []
        for source_pack in task_sources:
            source_rows_for_task = [row for row in task_test if row["source_pack"] == source_pack]
            if source_rows_for_task:
                source_dices.append(mean_or_nan([float(row["dice"]) for row in source_rows_for_task]))
        if len(source_dices) > 1 and max(source_dices) - min(source_dices) > 0.08:
            source_issue = True
    multi_pred_cc_ok = cc2_pred_cc2 >= 0.90 and cc3_pred_cc3 >= 0.90
    task_gate_accuracy = mean_or_nan([float(row["task_gate_correct"]) for row in test_rows])
    count_head_accuracy = mean_or_nan([float(row["count_head_correct"]) for row in test_rows])
    pred_task_counts = Counter(TASK_LABELS[int(row["pred_task_label"])] for row in test_rows)
    gate_collapse = bool(test_rows and max(pred_task_counts.values()) / len(test_rows) > 0.95)
    cc2_iou_delta = float(cc2_test["iou_mean"]) - LIGHTWEIGHT_V3_REFERENCE["cc2_iou"]
    cc2_dice_delta = float(cc2_test["dice_mean"]) - LIGHTWEIGHT_V3_REFERENCE["cc2_dice"]
    cc2_improved = float(cc2_test["iou_mean"]) >= 0.5700 or float(cc2_test["dice_mean"]) >= 0.7120
    cc2_pred_delta = cc2_pred_cc2 - LIGHTWEIGHT_V3_REFERENCE["cc2_pred_cc_is_2"]
    cc2_pred_ok = cc2_pred_delta >= -0.02
    single_iou_delta = float(single_test["iou_mean"]) - LIGHTWEIGHT_V3_REFERENCE["single_iou"]
    cc3_iou_delta = float(cc3_test["iou_mean"]) - LIGHTWEIGHT_V3_REFERENCE["cc3_iou"]
    single_ok = float(single_test["iou_mean"]) >= 0.6450
    cc3_ok = float(cc3_test["iou_mean"]) >= 0.6550 and float(cc3_test["dice_mean"]) >= 0.7750
    gate_above_majority = task_gate_accuracy > MAJORITY_TASK_BASELINE + 0.05
    finite_metrics = all(np.isfinite(float(row["iou"])) and np.isfinite(float(row["dice"])) for row in all_metric_rows)
    overall_test = summarize(test_rows, "all", "test")
    poc_overall_ok = (
        float(overall_test["iou_mean"]) >= 0.6400
        or float(overall_test["dice_mean"]) >= 0.7680
    )
    candidate_overall_ok = (
        float(overall_test["iou_mean"]) >= LIGHTWEIGHT_V3_REFERENCE["overall_iou"] - 0.01
        and float(overall_test["dice_mean"]) >= LIGHTWEIGHT_V3_REFERENCE["overall_dice"] - 0.01
    )
    overall_ok = candidate_overall_ok
    seed_collapse = False
    gate_stable = True
    for seed in args.seeds:
        seed_test_rows = [row for row in test_rows if int(row["seed"]) == seed]
        seed_cc2_rows = [row for row in seed_test_rows if row["task_group"] == "multi_defect_cc2"]
        seed_cc3_rows = [row for row in seed_test_rows if row["task_group"] == "multi_defect_cc3"]
        seed_gate_acc = mean_or_nan([float(row["task_gate_correct"]) for row in seed_test_rows])
        gate_stable = gate_stable and seed_gate_acc > MAJORITY_TASK_BASELINE + 0.05
        if mean_or_nan([float(row["dice"]) for row in seed_test_rows]) < LIGHTWEIGHT_V3_REFERENCE["overall_dice"] - 0.05:
            seed_collapse = True
        if mean_or_nan([float(row["dice"]) for row in seed_cc2_rows]) < LIGHTWEIGHT_V3_REFERENCE["cc2_dice"] - 0.05:
            seed_collapse = True
        if mean_or_nan([float(row["dice"]) for row in seed_cc3_rows]) < LIGHTWEIGHT_V3_REFERENCE["cc3_dice"] - 0.05:
            seed_collapse = True
    is_three_seed = len(args.seeds) > 1
    poc_acceptance_passed = (
        cc2_improved
        and cc2_pred_ok
        and single_ok
        and cc3_ok
        and poc_overall_ok
        and gate_above_majority
        and not gate_collapse
        and residual_ok
        and finite_metrics
    )
    candidate_acceptance_passed = (
        cc2_improved
        and cc2_pred_ok
        and single_ok
        and cc3_ok
        and candidate_overall_ok
        and gate_stable
        and not gate_collapse
        and residual_ok
        and not seed_collapse
        and finite_metrics
    )
    acceptance_passed = candidate_acceptance_passed if is_three_seed else poc_acceptance_passed
    context = {
        "stage_label": args.stage_label,
        "npz_readable": True,
        "schema_complete": True,
        "split_counts": validation["split_counts"],
        "input_shape": tuple(data["delta_bz"].shape),
        "output_shape": tuple(data["masks"].shape),
        "normalization_train_only": True,
        "seeds_completed": len(seed_results) == len(args.seeds),
        "seed_best": {
            result["seed"]: {
                "best_epoch": result["best_epoch"],
                "selected_threshold": result["threshold"],
                "best_val_score": result["best_val_score"],
                "best_val_group_score_mean": result["best_val_group_score_mean"],
                "best_val_group_score_std": result["best_val_group_score_std"],
            }
            for result in seed_results
        },
        "train_mean_std": format_mean_std(seed_results, "train"),
        "val_mean_std": format_mean_std(seed_results, "val"),
        "test_mean_std": format_mean_std(seed_results, "test"),
        "single_test": {key: single_test[key] for key in ("iou_mean", "dice_mean", "area_error_mean", "sample_count")},
        "cc2_test": {key: cc2_test[key] for key in ("iou_mean", "dice_mean", "area_error_mean", "sample_count")},
        "cc3_test": {key: cc3_test[key] for key in ("iou_mean", "dice_mean", "area_error_mean", "sample_count")},
        "multi_test": {key: multi_test[key] for key in ("iou_mean", "dice_mean", "area_error_mean", "sample_count")},
        "task_group_test": {
            row["group"]: {"iou": row["iou_mean"], "dice": row["dice_mean"], "n": row["sample_count"]}
            for row in task_group_rows
            if row["split"] == "test"
        },
        "defect_type_test": {
            row["group"]: {"iou": row["iou_mean"], "dice": row["dice_mean"], "n": row["sample_count"]}
            for row in defect_type_rows
            if row["split"] == "test"
        },
        "cc_behavior_test": {
            "cc2_pred_cc_is_2": cc2_pred_cc2,
            "cc3_pred_cc_is_3": cc3_pred_cc3,
            "multi_pred_cc_correct": float(np.mean([float(row["predicted_component_count_correct"]) for row in multi_rows])) if multi_rows else float("nan"),
        },
        "multi_error_test": multi_errors,
        "lightweight_reference": LIGHTWEIGHT_V3_REFERENCE,
        "cc2_improvement": {
            "iou_delta": cc2_iou_delta,
            "dice_delta": cc2_dice_delta,
            "pred_cc_is_2_delta": cc2_pred_delta,
        },
        "cc2_improved": cc2_improved,
        "cc2_pred_ok": cc2_pred_ok,
        "single_iou_delta": single_iou_delta,
        "cc3_iou_delta": cc3_iou_delta,
        "single_ok": single_ok,
        "cc3_ok": cc3_ok,
        "task_gate_accuracy": task_gate_accuracy,
        "count_head_accuracy": count_head_accuracy,
        "gate_above_majority": gate_above_majority,
        "gate_pred_distribution": dict(pred_task_counts),
        "gate_collapse": gate_collapse,
        "gate_stable": gate_stable,
        "residual_scale": RESIDUAL_SCALE,
        "test_residual_l2_mean": residual_l2_mean,
        "test_residual_to_base_ratio_mean": residual_to_base_ratio_mean,
        "residual_dominance_rate": residual_dominance_rate,
        "residual_dominates_base": residual_dominates_base,
        "residual_ok": residual_ok,
        "finite_metrics": finite_metrics,
        "overall_ok": candidate_overall_ok if is_three_seed else poc_overall_ok,
        "seed_collapse": seed_collapse,
        "poc_acceptance_passed": poc_acceptance_passed,
        "candidate_acceptance_passed": candidate_acceptance_passed,
        "acceptance_passed": acceptance_passed,
        "should_run_stage_b": poc_acceptance_passed and not is_three_seed,
        "single_degraded": single_degraded,
        "cc2_degraded": cc2_degraded,
        "cc3_degraded": cc3_degraded,
        "preview_generated": bool(preview_paths),
        "preview_dir": str(paths["preview_dir"]),
        "multi_pred_cc_ok": multi_pred_cc_ok,
        "hardest_group": f"{hardest['group']} dice={float(hardest['dice_mean']):.4f}" if hardest["group"] != "n/a" else "n/a",
        "dominant_failure": dominant_failure,
        "source_issue": source_issue,
        "interpretation": (
            (
                "Topology-aware residual gated decoder v2 POC is promising against the 20.42 lightweight reference and can advance to 3-seed validation."
                if not is_three_seed
                else "Topology-aware residual gated decoder v2 3-seed candidate passes acceptance against the 20.42 lightweight reference."
            )
            if acceptance_passed
            else (
                "Topology-aware residual gated decoder v2 POC is not promising enough for 3-seed validation; do not create a candidate baseline document."
                if not is_three_seed
                else "Topology-aware residual gated decoder v2 3-seed candidate does not pass acceptance; do not create a candidate baseline document."
            )
        ),
        "recommended_next_step": (
            "run 3-seed topology-aware residual gated v2 candidate"
            if poc_acceptance_passed and not is_three_seed
            else ("create topology-aware residual gated v2 candidate doc for human confirmation" if acceptance_passed else "stop and record rejection")
        ),
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["audit_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(build_training_summary(context), encoding="utf-8")
    paths["audit_summary"].write_text(build_audit_summary(context), encoding="utf-8")
    print(
        json.dumps(
            {
                "seeds_completed": context["seeds_completed"],
                "seed_best": context["seed_best"],
                "train": context["train_mean_std"],
                "val": context["val_mean_std"],
                "test": context["test_mean_std"],
                "single_test": context["single_test"],
                "cc2_test": context["cc2_test"],
                "cc3_test": context["cc3_test"],
                "multi_test": context["multi_test"],
                "cc_behavior_test": context["cc_behavior_test"],
                "task_gate_accuracy": context["task_gate_accuracy"],
                "count_head_accuracy": context["count_head_accuracy"],
                "gate_collapse": context["gate_collapse"],
                "residual_scale": context["residual_scale"],
                "test_residual_to_base_ratio_mean": context["test_residual_to_base_ratio_mean"],
                "residual_dominates_base": context["residual_dominates_base"],
                "cc2_improvement": context["cc2_improvement"],
                "poc_acceptance_passed": context["poc_acceptance_passed"],
                "candidate_acceptance_passed": context["candidate_acceptance_passed"],
                "acceptance_passed": context["acceptance_passed"],
                "summary": str(paths["summary"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
