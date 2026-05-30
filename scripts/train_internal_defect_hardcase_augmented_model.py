#!/usr/bin/env python
"""22.3 hard-case augmented B2 training gate.

正式候选只使用 delta_b/BxByBz 与 delta_b-derived features。shape、burial、
size、aspect、split、sample_id、hard-case target 等字段只用于 supervision、
validation selection 和 metrics grouping，不进入模型输入。
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from internal_defect_hardcase_utils import (
    DATASET_ID,
    METRIC_FIELDS,
    PREDICTION_FIELDS,
    TAIL_FIELDS,
    load_old_b2_on_dataset,
    metric_row,
    metric_rows_for_model,
    prediction_rows,
    prepare_dataset,
    safe_float,
    selection_score,
    tail_row,
)
from load_internal_defect_pilot_dataset import ROOT, classification_metrics, denormalize_y, regression_metrics, write_csv
from train_internal_defect_burial_depth_candidates import BurialDepthNet, predict


SUMMARY = ROOT / "results/summaries/internal_defect_hardcase_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/internal_defect_hardcase_seed_summary.csv"
METRICS = ROOT / "results/metrics/internal_defect_hardcase_metrics.csv"
TAIL = ROOT / "results/metrics/internal_defect_hardcase_tail_metrics.csv"
GROUP_SUMMARY = ROOT / "results/metrics/internal_defect_hardcase_group_summary.csv"
VS_B2 = ROOT / "results/metrics/internal_defect_hardcase_vs_b2_reference.csv"
SELECTED_PRED = ROOT / "results/metrics/internal_defect_hardcase_selected_predictions.csv"

MODEL_NAME = "H1_B2_feature_fusion_burial_head_hardcase_aug"
H2_MODEL_NAME = "H2_B2_hardcase_tail_weighted"

SEED_FIELDS = [
    "model",
    "selected_model",
    "seed",
    "best_epoch",
    "best_val_selection_score",
    "train_total_normalized_mae",
    "val_total_normalized_mae",
    "test_total_normalized_mae",
    "test_L_mae_mm",
    "test_W_mae_mm",
    "test_D_mae_mm",
    "test_burial_depth_mae_mm",
    "test_center_xyz_component_mae_mm",
    "test_shape_accuracy",
    "test_shape_macro_f1",
    "test_catastrophic_failure_count",
    "test_catastrophic_failure_rate",
    "test_geometry_branch_failure_count",
    "test_geometry_branch_failure_rate",
    "test_center_p95_mm",
    "test_center_max_mm",
    "test_burial_p95_mm",
    "test_burial_max_mm",
]

GROUP_FIELDS = [
    "model",
    "selected_model",
    "split",
    "subset",
    "group_field",
    "group_value",
    "sample_count",
    "total_normalized_mae",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "burial_depth_mae_mm",
    "center_xyz_component_mae_mm",
    "shape_accuracy",
    "shape_macro_f1",
    "catastrophic_failure_count",
    "geometry_branch_failure_count",
]

VS_FIELDS = ["metric", "old_B2_reference", "hardcase_augmented", "delta_augmented_minus_b2", "passes_gate", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hard-case augmented internal defect B2 model.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed-summary", type=Path, default=SEED_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--tail", type=Path, default=TAIL)
    parser.add_argument("--group-summary", type=Path, default=GROUP_SUMMARY)
    parser.add_argument("--vs-b2", type=Path, default=VS_B2)
    parser.add_argument("--selected-predictions", type=Path, default=SELECTED_PRED)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = indices.copy()
    rng.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, order.size, batch_size)]


def train_candidate(seed: int, epochs: int, batch_size: int, prepared: dict[str, Any], model_name: str, hardcase_weighted: bool = False) -> dict[str, Any]:
    dataset = prepared["dataset"]
    splits = prepared["splits"]
    x = prepared["x"]
    features = prepared["features"]
    y_norm = prepared["y_norm"]
    y_true = prepared["y"]
    y_mean = prepared["y_mean"]
    y_std = prepared["y_std"]
    shape = dataset.shape_label
    set_seed(seed)
    model = BurialDepthNet(feature_dim=int(features.shape[1]), feature_fusion=True, shape_conditioned=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    x_t = torch.from_numpy(x)
    f_t = torch.from_numpy(features.astype(np.float32))
    y_t = torch.from_numpy(y_norm.astype(np.float32))
    shape_t = torch.from_numpy(shape.astype(np.int64))
    param_w = torch.tensor([1.0, 1.0, 1.0, 3.5, 1.0, 1.0, 1.0], dtype=torch.float32)
    sample_weight_np = np.ones(dataset.sample_ids.shape[0], dtype=np.float32)
    if hardcase_weighted:
        train_idx = splits["train"]
        train_hard = train_idx[dataset.row_origin[train_idx] == "hardcase_topup_v1"]
        sample_weight_np[train_hard] = 1.35
    sample_w = torch.from_numpy(sample_weight_np)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metric: dict[str, Any] = {}
    best_val_tail: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx in batches(splits["train"], batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            reg, logits = model(x_t[batch_idx], f_t[batch_idx])
            raw_reg = nn.functional.smooth_l1_loss(reg, y_t[batch_idx], reduction="none")
            reg_loss = (raw_reg * param_w.reshape(1, -1) * sample_w[batch_idx].reshape(-1, 1)).mean()
            loss = reg_loss + 0.35 * ce(logits, shape_t[batch_idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        pred_norm, shape_pred = predict(model, x, features)
        pred = denormalize_y(pred_norm, y_mean, y_std)
        val_metric = metric_row(model_name, False, seed, "val", "all", splits["val"], y_true, pred, shape, shape_pred, y_std.reshape(-1))
        val_tail = tail_row(model_name, False, seed, "val", "all", splits["val"], y_true, pred, shape, shape_pred, y_std.reshape(-1))
        score = selection_score(val_metric, val_tail)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metric = val_metric
            best_val_tail = val_tail
    if best_state is None:
        raise RuntimeError("no H1 best state selected")
    model.load_state_dict(best_state)
    pred_norm, shape_pred = predict(model, x, features)
    pred = denormalize_y(pred_norm, y_mean, y_std)
    return {
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_metric": best_val_metric,
        "best_val_tail": best_val_tail,
        "pred": pred,
        "shape_pred": shape_pred,
    }


def group_rows(model: str, selected: bool, dataset: Any, y_pred: np.ndarray, shape_pred: np.ndarray, y_std: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_values = {
        "shape_type": dataset.shape_type,
        "burial_depth_level": dataset.burial_depth_level,
        "size_level": dataset.size_level,
        "aspect_bin": dataset.aspect_bin,
        "row_origin": dataset.row_origin,
        "hardcase_target_id": dataset.hardcase_target_id,
    }
    for split_name in ["train", "val", "test"]:
        split_idx = np.where(dataset.split == split_name)[0]
        for subset in ["all", "source_v2", "hardcase_topup"]:
            if subset == "source_v2":
                idx_base = split_idx[dataset.row_origin[split_idx] == "source_v2_240"]
            elif subset == "hardcase_topup":
                idx_base = split_idx[dataset.row_origin[split_idx] == "hardcase_topup_v1"]
            else:
                idx_base = split_idx
            for field, values in group_values.items():
                for value in sorted(set(values[idx_base].tolist())):
                    idx = idx_base[values[idx_base] == value]
                    if idx.size == 0:
                        continue
                    reg = regression_metrics(dataset.y_regression[idx], y_pred[idx], y_std)
                    cls = classification_metrics(dataset.shape_label[idx], shape_pred[idx])
                    from internal_defect_hardcase_utils import tail_row

                    tail = tail_row(model, selected, "", split_name, subset, idx, dataset.y_regression, y_pred, dataset.shape_label, shape_pred, y_std)
                    rows.append(
                        {
                            "model": model,
                            "selected_model": selected,
                            "split": split_name,
                            "subset": subset,
                            "group_field": field,
                            "group_value": value,
                            "sample_count": int(idx.size),
                            "total_normalized_mae": reg["total_normalized_mae"],
                            "L_mae_mm": reg["L_mae_mm"],
                            "W_mae_mm": reg["W_mae_mm"],
                            "D_mae_mm": reg["D_mae_mm"],
                            "burial_depth_mae_mm": reg["burial_depth_mae_mm"],
                            "center_xyz_component_mae_mm": reg["center_xyz_mae_mm"],
                            "shape_accuracy": cls["shape_accuracy"],
                            "shape_macro_f1": cls["shape_macro_f1"],
                            "catastrophic_failure_count": tail["catastrophic_failure_count"],
                            "geometry_branch_failure_count": tail["geometry_branch_failure_count"],
                        }
                    )
    return rows


def build_vs_rows(b2_metric: dict[str, Any], b2_tail: dict[str, Any], sel_metric: dict[str, Any], sel_tail: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("total_normalized_mae", b2_metric["total_normalized_mae"], sel_metric["total_normalized_mae"], safe_float(sel_metric["total_normalized_mae"]) <= safe_float(b2_metric["total_normalized_mae"]) * 1.10, "mean total should not clearly regress"),
        ("L_mae_mm", b2_metric["L_mae_mm"], sel_metric["L_mae_mm"], safe_float(sel_metric["L_mae_mm"]) <= safe_float(b2_metric["L_mae_mm"]) * 1.15, "L MAE guard"),
        ("W_mae_mm", b2_metric["W_mae_mm"], sel_metric["W_mae_mm"], safe_float(sel_metric["W_mae_mm"]) <= safe_float(b2_metric["W_mae_mm"]) * 1.15, "W MAE guard"),
        ("D_mae_mm", b2_metric["D_mae_mm"], sel_metric["D_mae_mm"], safe_float(sel_metric["D_mae_mm"]) <= safe_float(b2_metric["D_mae_mm"]) * 1.15, "D MAE guard"),
        ("burial_depth_mae_mm", b2_metric["burial_depth_mae_mm"], sel_metric["burial_depth_mae_mm"], safe_float(sel_metric["burial_depth_mae_mm"]) <= safe_float(b2_metric["burial_depth_mae_mm"]) * 1.10, "burial mean guard"),
        ("center_xyz_component_mae_mm", b2_metric["center_xyz_component_mae_mm"], sel_metric["center_xyz_component_mae_mm"], safe_float(sel_metric["center_xyz_component_mae_mm"]) <= safe_float(b2_metric["center_xyz_component_mae_mm"]) * 1.10, "center component MAE guard"),
        ("shape_accuracy", b2_metric["shape_accuracy"], sel_metric["shape_accuracy"], safe_float(sel_metric["shape_accuracy"]) >= safe_float(b2_metric["shape_accuracy"]) - 0.05, "shape should not clearly regress"),
        ("catastrophic_failure_rate", b2_tail["catastrophic_failure_rate"], sel_tail["catastrophic_failure_rate"], safe_float(sel_tail["catastrophic_failure_rate"]) < safe_float(b2_tail["catastrophic_failure_rate"]) and safe_float(sel_tail["catastrophic_failure_rate"]) <= 0.05, "target <=5%"),
        ("geometry_branch_failure_count", b2_tail["geometry_branch_failure_count"], sel_tail["geometry_branch_failure_count"], safe_float(sel_tail["geometry_branch_failure_count"]) < safe_float(b2_tail["geometry_branch_failure_count"]), "target below old B2"),
        ("center_xyz_error_p95_mm", b2_tail["center_xyz_error_p95_mm"], sel_tail["center_xyz_error_p95_mm"], safe_float(sel_tail["center_xyz_error_p95_mm"]) < safe_float(b2_tail["center_xyz_error_p95_mm"]), "tail center p95 lower"),
        ("center_xyz_error_max_mm", b2_tail["center_xyz_error_max_mm"], sel_tail["center_xyz_error_max_mm"], safe_float(sel_tail["center_xyz_error_max_mm"]) < safe_float(b2_tail["center_xyz_error_max_mm"]), "tail center max lower"),
        ("burial_depth_error_p95_mm", b2_tail["burial_depth_error_p95_mm"], sel_tail["burial_depth_error_p95_mm"], safe_float(sel_tail["burial_depth_error_p95_mm"]) < safe_float(b2_tail["burial_depth_error_p95_mm"]), "tail burial p95 lower"),
        ("burial_depth_error_max_mm", b2_tail["burial_depth_error_max_mm"], sel_tail["burial_depth_error_max_mm"], safe_float(sel_tail["burial_depth_error_max_mm"]) < safe_float(b2_tail["burial_depth_error_max_mm"]), "tail burial max lower"),
    ]
    return [
        {
            "metric": metric,
            "old_B2_reference": b2,
            "hardcase_augmented": val,
            "delta_augmented_minus_b2": safe_float(val) - safe_float(b2),
            "passes_gate": passed,
            "notes": notes,
        }
        for metric, b2, val, passed, notes in specs
    ]


def main() -> int:
    args = parse_args()
    prepared = prepare_dataset(args.dataset_id)
    dataset = prepared["dataset"]
    splits = prepared["splits"]
    y_std = prepared["y_std"].reshape(-1)
    old_pred, old_shape_pred, _manifest = load_old_b2_on_dataset(prepared)
    old_metric_rows, old_tail_rows = metric_rows_for_model("old_B2_v2_artifact", False, 2026, dataset, splits, old_pred, old_shape_pred, y_std)
    old_test_metric = next(row for row in old_metric_rows if row["split"] == "test" and row["subset"] == "all")
    old_test_tail = next(row for row in old_tail_rows if row["split"] == "test" and row["subset"] == "all")

    results = [train_candidate(seed, args.epochs, args.batch_size, prepared, MODEL_NAME, False) for seed in [42, 123, 2026]]
    results.extend(train_candidate(seed, args.epochs, args.batch_size, prepared, H2_MODEL_NAME, True) for seed in [42, 123, 2026])
    selected = min(results, key=lambda item: item["best_score"])
    metric_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    group_summary: list[dict[str, Any]] = []
    selected_metric: dict[str, Any] | None = None
    selected_tail: dict[str, Any] | None = None
    for result in results:
        is_selected = result["model"] == selected["model"] and int(result["seed"]) == int(selected["seed"])
        m_rows, t_rows = metric_rows_for_model(result["model"], is_selected, result["seed"], dataset, splits, result["pred"], result["shape_pred"], y_std, result["best_score"], result["best_epoch"])
        metric_rows.extend(m_rows)
        tail_rows.extend(t_rows)
        train_metric = next(row for row in m_rows if row["split"] == "train" and row["subset"] == "all")
        val_metric = next(row for row in m_rows if row["split"] == "val" and row["subset"] == "all")
        test_metric = next(row for row in m_rows if row["split"] == "test" and row["subset"] == "all")
        test_tail = next(row for row in t_rows if row["split"] == "test" and row["subset"] == "all")
        if is_selected:
            selected_metric = test_metric
            selected_tail = test_tail
            group_summary = group_rows(result["model"], True, dataset, result["pred"], result["shape_pred"], y_std)
            write_csv(args.selected_predictions, prediction_rows(result["model"], result["seed"], dataset, result["pred"], result["shape_pred"], y_std), PREDICTION_FIELDS)
        seed_rows.append(
            {
                "model": result["model"],
                "selected_model": is_selected,
                "seed": result["seed"],
                "best_epoch": result["best_epoch"],
                "best_val_selection_score": result["best_score"],
                "train_total_normalized_mae": train_metric["total_normalized_mae"],
                "val_total_normalized_mae": val_metric["total_normalized_mae"],
                "test_total_normalized_mae": test_metric["total_normalized_mae"],
                "test_L_mae_mm": test_metric["L_mae_mm"],
                "test_W_mae_mm": test_metric["W_mae_mm"],
                "test_D_mae_mm": test_metric["D_mae_mm"],
                "test_burial_depth_mae_mm": test_metric["burial_depth_mae_mm"],
                "test_center_xyz_component_mae_mm": test_metric["center_xyz_component_mae_mm"],
                "test_shape_accuracy": test_metric["shape_accuracy"],
                "test_shape_macro_f1": test_metric["shape_macro_f1"],
                "test_catastrophic_failure_count": test_tail["catastrophic_failure_count"],
                "test_catastrophic_failure_rate": test_tail["catastrophic_failure_rate"],
                "test_geometry_branch_failure_count": test_tail["geometry_branch_failure_count"],
                "test_geometry_branch_failure_rate": test_tail["geometry_branch_failure_rate"],
                "test_center_p95_mm": test_tail["center_xyz_error_p95_mm"],
                "test_center_max_mm": test_tail["center_xyz_error_max_mm"],
                "test_burial_p95_mm": test_tail["burial_depth_error_p95_mm"],
                "test_burial_max_mm": test_tail["burial_depth_error_max_mm"],
            }
        )
    if selected_metric is None or selected_tail is None:
        raise RuntimeError("selected metrics missing")
    vs_rows = build_vs_rows(old_test_metric, old_test_tail, selected_metric, selected_tail)
    write_csv(args.seed_summary, seed_rows, SEED_FIELDS)
    write_csv(args.metrics, metric_rows, METRIC_FIELDS)
    write_csv(args.tail, tail_rows, TAIL_FIELDS)
    write_csv(args.group_summary, group_summary, GROUP_FIELDS)
    write_csv(args.vs_b2, vs_rows, VS_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "22.3 hard-case augmented internal defect training summary",
                f"dataset_id: {args.dataset_id}",
                f"selected_model: {selected['model']}",
                f"selected_seed: {selected['seed']}",
                f"selected_best_epoch: {selected['best_epoch']}",
                "architecture: fixed 21.7 B2_feature_fusion_burial_head; H2 only adds train-split hard-case sample weighting",
                "input_policy: delta_b/BxByBz plus delta_b-derived features only",
                "metadata_policy: hard-case/source flags are not model input; H2 uses train split row_origin only for sample weighting and metrics grouping",
                "selection_protocol: validation-only seed selection; test final only",
                f"test_total_normalized_mae: {safe_float(selected_metric['total_normalized_mae']):.6f}",
                f"test_LWD_mae_mm: {safe_float(selected_metric['L_mae_mm']):.3f} / {safe_float(selected_metric['W_mae_mm']):.3f} / {safe_float(selected_metric['D_mae_mm']):.3f}",
                f"test_burial_depth_mae_mm: {safe_float(selected_metric['burial_depth_mae_mm']):.3f}",
                f"test_center_xyz_component_mae_mm: {safe_float(selected_metric['center_xyz_component_mae_mm']):.3f}",
                f"test_shape_accuracy_f1: {safe_float(selected_metric['shape_accuracy']):.6f} / {safe_float(selected_metric['shape_macro_f1']):.6f}",
                f"test_catastrophic_failure_count_rate: {selected_tail['catastrophic_failure_count']} / {safe_float(selected_tail['catastrophic_failure_rate']):.6f}",
                f"test_geometry_branch_failure_count_rate: {selected_tail['geometry_branch_failure_count']} / {safe_float(selected_tail['geometry_branch_failure_rate']):.6f}",
                f"test_center_p95_max_mm: {safe_float(selected_tail['center_xyz_error_p95_mm']):.3f} / {safe_float(selected_tail['center_xyz_error_max_mm']):.3f}",
                f"test_burial_p95_max_mm: {safe_float(selected_tail['burial_depth_error_p95_mm']):.3f} / {safe_float(selected_tail['burial_depth_error_max_mm']):.3f}",
                f"old_B2_test_catastrophic_failure_count_rate: {old_test_tail['catastrophic_failure_count']} / {safe_float(old_test_tail['catastrophic_failure_rate']):.6f}",
                f"old_B2_test_geometry_branch_failure_count_rate: {old_test_tail['geometry_branch_failure_count']} / {safe_float(old_test_tail['geometry_branch_failure_rate']):.6f}",
                f"old_B2_test_center_p95_max_mm: {safe_float(old_test_tail['center_xyz_error_p95_mm']):.3f} / {safe_float(old_test_tail['center_xyz_error_max_mm']):.3f}",
                f"old_B2_test_burial_p95_max_mm: {safe_float(old_test_tail['burial_depth_error_p95_mm']):.3f} / {safe_float(old_test_tail['burial_depth_error_max_mm']):.3f}",
                f"stable_inference_gate_passed: {all(str(row['passes_gate']).lower() == 'true' for row in vs_rows if row['metric'] in {'catastrophic_failure_rate', 'geometry_branch_failure_count', 'center_xyz_error_p95_mm', 'burial_depth_error_p95_mm'})}",
                "checkpoint_saved: false",
                "npz_written: false",
                "current_baseline_update: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
