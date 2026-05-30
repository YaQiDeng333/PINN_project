#!/usr/bin/env python
"""训练 22.6 internal defect tail-risk abstention gate。

只使用 audit_internal_defect_tail_risk_signals.py 生成的推理时可得风险信号。
真实标签和分层字段只用于构造风险目标与评估，不作为 gate 输入。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from audit_internal_defect_tail_risk_signals import (
    DATASET_ID,
    METADATA_COLUMNS,
    RISK_FEATURE_COLUMNS,
    SIGNAL_AUDIT_CSV,
    TARGET_COLUMNS,
    as_bool,
    build_risk_signal_rows,
    prepare_dataset,
    safe_float,
    train_label_ranges_mm,
)
from load_internal_defect_pilot_dataset import ROOT, write_csv


SUMMARY_PATH = ROOT / "results/summaries/internal_defect_tail_risk_gate_summary.txt"
METRICS_PATH = ROOT / "results/metrics/internal_defect_tail_risk_gate_metrics.csv"
THRESHOLDS_PATH = ROOT / "results/metrics/internal_defect_tail_risk_gate_thresholds.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_signal_csv() -> list[dict[str, Any]]:
    if not SIGNAL_AUDIT_CSV.exists():
        rows, _, _ = build_risk_signal_rows()
        return rows
    return read_csv(SIGNAL_AUDIT_CSV)


def row_mask(rows: list[dict[str, Any]], split: str) -> np.ndarray:
    return np.asarray([row["split"] == split for row in rows], dtype=bool)


def build_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x = np.asarray([[safe_float(row[col]) for col in RISK_FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    y = np.asarray([1 if as_bool(row["bad_tail_target"]) else 0 for row in rows], dtype=np.int64)
    targets = {col: np.asarray([1 if as_bool(row[col]) else 0 for row in rows], dtype=np.int64) for col in TARGET_COLUMNS}
    return x, y, targets


def train_scale(x: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train].mean(axis=0, keepdims=True)
    std = x[train].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x - mean) / std).astype(np.float64), mean.reshape(-1), std.reshape(-1)


def model_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        raw = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)
        return 1.0 / (1.0 + np.exp(-raw))
    pred = model.predict(x)
    return np.asarray(pred, dtype=np.float64)


def candidate_models() -> list[tuple[str, Any]]:
    return [
        (
            "logistic_regression_balanced",
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42),
        ),
        ("ridge_classifier_balanced", RidgeClassifier(class_weight="balanced")),
        (
            "random_forest_small",
            RandomForestClassifier(
                n_estimators=128,
                max_depth=4,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]


def quantile(values: list[float] | np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def metric_for_scores(rows: list[dict[str, Any]], indices: np.ndarray, scores: np.ndarray, threshold: float, model_name: str, split: str) -> dict[str, Any]:
    selected_rows = [rows[i] for i in indices.tolist()]
    split_scores = scores[indices]
    high_risk = split_scores >= threshold
    bad = np.asarray([as_bool(row["bad_tail_target"]) for row in selected_rows], dtype=bool)
    catastrophic = np.asarray([as_bool(row["catastrophic_failure"]) for row in selected_rows], dtype=bool)
    geometry = np.asarray([as_bool(row["geometry_branch_failure"]) for row in selected_rows], dtype=bool)
    clean = ~bad
    accepted = ~high_risk
    n = len(selected_rows)
    flagged = int(high_risk.sum())

    def recall(mask: np.ndarray) -> float:
        positives = int(mask.sum())
        if positives == 0:
            return 1.0
        return float((high_risk & mask).sum() / positives)

    precision = float((high_risk & bad).sum() / flagged) if flagged else 0.0
    false_alarm_rate = float((high_risk & clean).sum() / clean.sum()) if int(clean.sum()) else 0.0
    coverage = float(accepted.sum() / n) if n else 0.0

    center = np.asarray([safe_float(row["f2_center_xyz_error_mm"]) for row in selected_rows], dtype=np.float64)
    burial = np.asarray([safe_float(row["f2_burial_depth_error_mm"]) for row in selected_rows], dtype=np.float64)
    total = np.asarray([safe_float(row["f2_total_abs_normalized_error"]) for row in selected_rows], dtype=np.float64)

    accepted_center = center[accepted]
    accepted_burial = burial[accepted]
    accepted_total = total[accepted]
    before_center_p95 = quantile(center, 95)
    before_burial_p95 = quantile(burial, 95)
    after_center_p95 = quantile(accepted_center, 95)
    after_burial_p95 = quantile(accepted_burial, 95)

    return {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "sample_count": n,
        "high_risk_count": flagged,
        "bad_tail_count": int(bad.sum()),
        "catastrophic_failure_count": int(catastrophic.sum()),
        "geometry_branch_failure_count": int(geometry.sum()),
        "high_risk_precision": precision,
        "bad_tail_recall": recall(bad),
        "catastrophic_failure_recall": recall(catastrophic),
        "geometry_branch_failure_recall": recall(geometry),
        "false_alarm_rate": false_alarm_rate,
        "coverage_retained": coverage,
        "before_total_mean": float(total.mean()) if n else 0.0,
        "accepted_total_mean": float(accepted_total.mean()) if accepted_total.size else 0.0,
        "before_center_p95_mm": before_center_p95,
        "accepted_center_p95_mm": after_center_p95,
        "before_center_max_mm": float(center.max()) if center.size else 0.0,
        "accepted_center_max_mm": float(accepted_center.max()) if accepted_center.size else 0.0,
        "before_burial_p95_mm": before_burial_p95,
        "accepted_burial_p95_mm": after_burial_p95,
        "before_burial_max_mm": float(burial.max()) if burial.size else 0.0,
        "accepted_burial_max_mm": float(accepted_burial.max()) if accepted_burial.size else 0.0,
        "center_p95_reduction_mm": before_center_p95 - after_center_p95,
        "burial_p95_reduction_mm": before_burial_p95 - after_burial_p95,
    }


def threshold_score(metric: dict[str, Any]) -> float:
    false_alarm = safe_float(metric["false_alarm_rate"])
    coverage = safe_float(metric["coverage_retained"])
    coverage_penalty = max(0.0, 0.35 - coverage) + 0.5 * max(0.0, coverage - 0.65)
    operating_bonus = 0.5 if false_alarm <= 0.50 and coverage >= 0.35 else 0.0
    return float(
        4.0 * safe_float(metric["catastrophic_failure_recall"])
        + 4.0 * safe_float(metric["geometry_branch_failure_recall"])
        + 1.5 * safe_float(metric["bad_tail_recall"])
        + 0.25 * safe_float(metric["center_p95_reduction_mm"])
        + 0.25 * safe_float(metric["burial_p95_reduction_mm"])
        - 3.0 * false_alarm
        - 1.5 * coverage_penalty
        + operating_bonus
    )


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(scores, 8))
    candidates = list(np.linspace(0.05, 0.95, 19))
    candidates.extend(float(x) for x in np.quantile(scores, np.linspace(0.05, 0.95, 19)))
    candidates.extend(float(x) for x in unique)
    return np.asarray(sorted(set(max(0.0, min(1.0, x)) for x in candidates)), dtype=np.float64)


def main() -> int:
    rows = ensure_signal_csv()
    x, y, _ = build_arrays(rows)
    train = row_mask(rows, "train")
    val = row_mask(rows, "val")
    test = row_mask(rows, "test")
    if y[train].sum() == 0:
        raise RuntimeError("train split 没有 bad_tail_target，无法训练 risk gate")

    x_scaled, x_mean, x_std = train_scale(x, train)
    split_indices = {name: np.where(mask)[0] for name, mask in {"train": train, "val": val, "test": test}.items()}

    threshold_rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[Any, np.ndarray]] = {}
    best: tuple[float, str, float, dict[str, Any]] | None = None
    for model_name, model in candidate_models():
        model.fit(x_scaled[train], y[train])
        scores = model_scores(model, x_scaled)
        fitted[model_name] = (model, scores)
        for threshold in threshold_grid(scores[val]):
            metric = metric_for_scores(rows, split_indices["val"], scores, float(threshold), model_name, "val")
            score = threshold_score(metric)
            metric["selection_score"] = score
            metric["selected"] = False
            threshold_rows.append(metric)
            if best is None or score > best[0]:
                best = (score, model_name, float(threshold), metric)

    if best is None:
        raise RuntimeError("未能选择 risk gate threshold")
    _, selected_model, selected_threshold, _ = best

    metrics_rows: list[dict[str, Any]] = []
    for model_name, (_, scores) in fitted.items():
        is_selected = model_name == selected_model
        for split_name, idx in split_indices.items():
            metric = metric_for_scores(rows, idx, scores, selected_threshold if is_selected else 0.5, model_name, split_name)
            metric["selected"] = is_selected
            metric["selection_threshold"] = selected_threshold if is_selected else 0.5
            metric["selection_score"] = threshold_score(metric) if split_name == "val" else ""
            metrics_rows.append(metric)

    for row in threshold_rows:
        if row["model"] == selected_model and abs(safe_float(row["threshold"]) - selected_threshold) < 1e-9:
            row["selected"] = True

    metric_fields = [
        "model",
        "selected",
        "split",
        "threshold",
        "selection_threshold",
        "selection_score",
        "sample_count",
        "high_risk_count",
        "bad_tail_count",
        "catastrophic_failure_count",
        "geometry_branch_failure_count",
        "high_risk_precision",
        "bad_tail_recall",
        "catastrophic_failure_recall",
        "geometry_branch_failure_recall",
        "false_alarm_rate",
        "coverage_retained",
        "before_total_mean",
        "accepted_total_mean",
        "before_center_p95_mm",
        "accepted_center_p95_mm",
        "before_center_max_mm",
        "accepted_center_max_mm",
        "before_burial_p95_mm",
        "accepted_burial_p95_mm",
        "before_burial_max_mm",
        "accepted_burial_max_mm",
        "center_p95_reduction_mm",
        "burial_p95_reduction_mm",
    ]
    write_csv(METRICS_PATH, metrics_rows, metric_fields)
    write_csv(
        THRESHOLDS_PATH,
        threshold_rows,
        [
            "model",
            "selected",
            "split",
            "threshold",
            "selection_score",
            "sample_count",
            "high_risk_count",
            "bad_tail_count",
            "catastrophic_failure_count",
            "geometry_branch_failure_count",
            "high_risk_precision",
            "bad_tail_recall",
            "catastrophic_failure_recall",
            "geometry_branch_failure_recall",
            "false_alarm_rate",
            "coverage_retained",
            "center_p95_reduction_mm",
            "burial_p95_reduction_mm",
        ],
    )

    selected_test = next(row for row in metrics_rows if row["model"] == selected_model and row["split"] == "test")
    selected_val = next(row for row in metrics_rows if row["model"] == selected_model and row["split"] == "val")
    prepared = prepare_dataset(DATASET_ID)
    label_low_mm, label_high_mm = train_label_ranges_mm(prepared)
    contract = {
        "selected_model": selected_model,
        "threshold": selected_threshold,
        "risk_feature_columns": RISK_FEATURE_COLUMNS,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "train_label_range_feature": "f2_pred_train_range_violation_count",
        "train_label_param_order": ["L_m", "W_m", "D_m", "burial_depth_m", "center_x_m", "center_y_m", "center_z_m"],
        "train_label_low_mm_with_5pct_margin": label_low_mm.tolist(),
        "train_label_high_mm_with_5pct_margin": label_high_mm.tolist(),
    }
    contract_path = ROOT / "results/metrics/internal_defect_tail_risk_gate_model_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "22.6 internal defect tail-risk gate 摘要",
        "",
        f"- 选择方式：只用 validation split 选择模型和阈值；test 只做最终评估。",
        f"- selected risk gate：{selected_model}，threshold={selected_threshold:.6f}。",
        f"- validation：catastrophic recall={safe_float(selected_val['catastrophic_failure_recall']):.3f}，geometry recall={safe_float(selected_val['geometry_branch_failure_recall']):.3f}，false alarm={safe_float(selected_val['false_alarm_rate']):.3f}，coverage={safe_float(selected_val['coverage_retained']):.3f}。",
        f"- test：catastrophic recall={safe_float(selected_test['catastrophic_failure_recall']):.3f}，geometry recall={safe_float(selected_test['geometry_branch_failure_recall']):.3f}，false alarm={safe_float(selected_test['false_alarm_rate']):.3f}，coverage={safe_float(selected_test['coverage_retained']):.3f}。",
        f"- test accepted center p95/max：{safe_float(selected_test['accepted_center_p95_mm']):.3f}/{safe_float(selected_test['accepted_center_max_mm']):.3f} mm，原始为 {safe_float(selected_test['before_center_p95_mm']):.3f}/{safe_float(selected_test['before_center_max_mm']):.3f} mm。",
        f"- test accepted burial p95/max：{safe_float(selected_test['accepted_burial_p95_mm']):.3f}/{safe_float(selected_test['accepted_burial_max_mm']):.3f} mm，原始为 {safe_float(selected_test['before_burial_p95_mm']):.3f}/{safe_float(selected_test['before_burial_max_mm']):.3f} mm。",
        "",
        "结论：risk gate 只给出高风险/abstain 标记，不把 F2 或 internal branch 升级为 baseline。",
    ]
    write_text(SUMMARY_PATH, "\n".join(lines) + "\n")
    print(json.dumps({"selected_model": selected_model, "threshold": selected_threshold, "metrics": str(METRICS_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
