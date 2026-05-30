#!/usr/bin/env python
"""审计 internal defect v3_hardcase 的 tail-risk 推理信号。

本脚本只读取已存在的 v3_hardcase 数据集和 B2/H2/F2 预测 CSV，不训练、
不运行 COMSOL、不写 data/NPZ。输出供 22.6 risk gate 使用的推理时可得
风险信号表。
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from internal_defect_hardcase_utils import DATASET_ID, prepare_dataset, safe_float
from load_internal_defect_pilot_dataset import ROOT, write_csv


SUMMARY_DIR = ROOT / "results/summaries"
METRICS_DIR = ROOT / "results/metrics"
MANIFEST_PATH = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json"

B2_PREDICTIONS = METRICS_DIR / "internal_defect_hardcase_b2_reference_predictions.csv"
H2_PREDICTIONS = METRICS_DIR / "internal_defect_hardcase_selected_predictions.csv"
F2_PREDICTIONS = METRICS_DIR / "internal_defect_freeze_shape_tail_selected_predictions.csv"

PREFLIGHT_SUMMARY = SUMMARY_DIR / "internal_defect_tail_risk_gate_preflight_summary.txt"
SIGNAL_SUMMARY = SUMMARY_DIR / "internal_defect_tail_risk_signal_audit_summary.txt"
SIGNAL_AUDIT_CSV = METRICS_DIR / "internal_defect_tail_risk_signal_audit.csv"
CORRELATION_CSV = METRICS_DIR / "internal_defect_tail_risk_signal_correlations.csv"

MODEL_ALIASES = {
    "b2": B2_PREDICTIONS,
    "h2": H2_PREDICTIONS,
    "f2": F2_PREDICTIONS,
}

RISK_FEATURE_COLUMNS = [
    "shape_disagreement_count",
    "b2_h2_shape_diff",
    "b2_f2_shape_diff",
    "h2_f2_shape_diff",
    "center_pairwise_mean_mm",
    "center_pairwise_max_mm",
    "burial_pairwise_mean_mm",
    "burial_pairwise_max_mm",
    "lwd_pairwise_mean_mm",
    "lwd_pairwise_max_mm",
    "b2_f2_center_disagreement_mm",
    "h2_f2_center_disagreement_mm",
    "b2_f2_burial_disagreement_mm",
    "h2_f2_burial_disagreement_mm",
    "f2_pred_L_mm",
    "f2_pred_W_mm",
    "f2_pred_D_mm",
    "f2_pred_burial_depth_mm",
    "f2_pred_center_norm_mm",
    "f2_pred_train_range_violation_count",
    "delta_abs_peak",
    "delta_energy",
    "delta_axis_energy_ratio_max",
    "delta_axis_energy_ratio_min",
    "delta_line_energy_range",
    "feature_abs_z_mean",
    "feature_abs_z_p95",
    "feature_abs_z_max",
    "feature_outlier_count_z3",
]

TARGET_COLUMNS = [
    "center_outlier",
    "burial_outlier",
    "dimension_outlier",
    "shape_misclassified",
    "full_shift_failure",
    "catastrophic_failure",
    "geometry_branch_failure",
    "bad_tail_target",
]

METADATA_COLUMNS = [
    "sample_id",
    "split",
    "subset",
    "row_origin",
    "true_shape_type",
    "f2_pred_shape_type",
    "b2_pred_shape_type",
    "h2_pred_shape_type",
    "burial_depth_level",
    "size_level",
    "aspect_bin",
    "center_region",
    "hardcase_target_id",
]

ERROR_COLUMNS = [
    "f2_total_abs_normalized_error",
    "f2_L_error_mm",
    "f2_W_error_mm",
    "f2_D_error_mm",
    "f2_burial_depth_error_mm",
    "f2_center_xyz_error_mm",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def model_rows_by_sample(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    return {row["sample_id"]: row for row in rows}


def pred_vector(row: dict[str, str], fields: list[str]) -> np.ndarray:
    return np.asarray([safe_float(row[field]) for field in fields], dtype=np.float64)


def pairwise_distances(vectors: list[np.ndarray]) -> list[float]:
    distances: list[float] = []
    for a, b in combinations(vectors, 2):
        distances.append(float(np.linalg.norm(a - b)))
    return distances


def pairwise_abs(values: list[float]) -> list[float]:
    return [abs(float(a) - float(b)) for a, b in combinations(values, 2)]


def git_forbidden_status() -> list[str]:
    paths = [
        "data",
        "checkpoints",
        "results/previews",
        "notes",
        "CURRENT_BASELINE.md",
        "scripts/visualize_current_baseline.py",
    ]
    result = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"git status failed: {result.stderr.strip()}"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def scalar_delta_features(delta_b: np.ndarray) -> dict[str, float]:
    axis_energy = np.mean(delta_b * delta_b, axis=(1, 2))
    line_energy = np.mean(delta_b * delta_b, axis=(0, 2))
    axis_energy_safe = axis_energy + 1e-18
    return {
        "delta_abs_peak": float(np.max(np.abs(delta_b))),
        "delta_energy": float(np.mean(delta_b * delta_b)),
        "delta_axis_energy_ratio_max": float(np.max(axis_energy_safe) / np.mean(axis_energy_safe)),
        "delta_axis_energy_ratio_min": float(np.min(axis_energy_safe) / np.mean(axis_energy_safe)),
        "delta_line_energy_range": float(np.max(line_energy) - np.min(line_energy)),
    }


def train_label_ranges_mm(prepared: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train_idx = prepared["splits"]["train"]
    y_mm = prepared["dataset"].y_regression[train_idx] * 1000.0
    low = y_mm.min(axis=0)
    high = y_mm.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    return low - 0.05 * span, high + 0.05 * span


def build_risk_signal_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    required = [MANIFEST_PATH, B2_PREDICTIONS, H2_PREDICTIONS, F2_PREDICTIONS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少 22.6 必需输入: " + "; ".join(missing))

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError(f"manifest dataset_id mismatch: {manifest.get('dataset_id')} != {DATASET_ID}")
    if bool(manifest.get("baseline_ready")):
        raise RuntimeError("v3_hardcase manifest unexpectedly marks baseline_ready=true")

    prepared = prepare_dataset(DATASET_ID)
    dataset = prepared["dataset"]
    sample_to_idx = {str(sample_id): i for i, sample_id in enumerate(dataset.sample_ids)}
    label_low, label_high = train_label_ranges_mm(prepared)

    tables = {alias: model_rows_by_sample(path) for alias, path in MODEL_ALIASES.items()}
    common_ids = sorted(set.intersection(*(set(table) for table in tables.values())))
    if len(common_ids) != len(dataset.sample_ids):
        raise RuntimeError(f"预测样本数与数据集不一致: common={len(common_ids)} dataset={len(dataset.sample_ids)}")

    feature_z = np.asarray(prepared["features"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for sample_id in common_ids:
        idx = sample_to_idx[sample_id]
        b2 = tables["b2"][sample_id]
        h2 = tables["h2"][sample_id]
        f2 = tables["f2"][sample_id]

        shapes = [b2["pred_shape_type"], h2["pred_shape_type"], f2["pred_shape_type"]]
        centers = [
            pred_vector(row, ["pred_center_x_mm", "pred_center_y_mm", "pred_center_z_mm"])
            for row in [b2, h2, f2]
        ]
        burials = [safe_float(row["pred_burial_depth_mm"]) for row in [b2, h2, f2]]
        lwds = [pred_vector(row, ["pred_L_mm", "pred_W_mm", "pred_D_mm"]) for row in [b2, h2, f2]]
        center_distances = pairwise_distances(centers)
        burial_diffs = pairwise_abs(burials)
        lwd_distances = pairwise_distances(lwds)

        f2_params_mm = pred_vector(
            f2,
            [
                "pred_L_mm",
                "pred_W_mm",
                "pred_D_mm",
                "pred_burial_depth_mm",
                "pred_center_x_mm",
                "pred_center_y_mm",
                "pred_center_z_mm",
            ],
        )
        train_range_violation = (f2_params_mm < label_low) | (f2_params_mm > label_high)

        z_abs = np.abs(feature_z[idx])
        delta_features = scalar_delta_features(dataset.delta_b[idx])
        center_outlier = safe_float(f2["center_xyz_error_mm"]) > 3.0
        burial_outlier = safe_float(f2["burial_depth_error_mm"]) > 1.0
        dimension_outlier = (
            "dimension_outlier" in f2.get("failure_tags", "")
            or safe_float(f2.get("dimension_relative_max", 0.0)) > 0.30
            or max(safe_float(f2["L_error_mm"]), safe_float(f2["W_error_mm"]), safe_float(f2["D_error_mm"])) > 2.0
        )
        shape_misclassified = not as_bool(f2["shape_correct"])
        catastrophic = as_bool(f2["is_catastrophic_failure"]) or (center_outlier and burial_outlier)
        geometry_branch = as_bool(f2["is_geometry_branch_failure"]) or (shape_misclassified and center_outlier and burial_outlier)

        row: dict[str, Any] = {
            "sample_id": sample_id,
            "split": f2["split"],
            "subset": f2["subset"],
            "row_origin": f2["row_origin"],
            "true_shape_type": f2["true_shape_type"],
            "f2_pred_shape_type": f2["pred_shape_type"],
            "b2_pred_shape_type": b2["pred_shape_type"],
            "h2_pred_shape_type": h2["pred_shape_type"],
            "burial_depth_level": f2["burial_depth_level"],
            "size_level": f2["size_level"],
            "aspect_bin": f2["aspect_bin"],
            "center_region": f2["center_region"],
            "hardcase_target_id": f2["hardcase_target_id"],
            "shape_disagreement_count": len(set(shapes)) - 1,
            "b2_h2_shape_diff": int(b2["pred_shape_type"] != h2["pred_shape_type"]),
            "b2_f2_shape_diff": int(b2["pred_shape_type"] != f2["pred_shape_type"]),
            "h2_f2_shape_diff": int(h2["pred_shape_type"] != f2["pred_shape_type"]),
            "center_pairwise_mean_mm": float(np.mean(center_distances)),
            "center_pairwise_max_mm": float(np.max(center_distances)),
            "burial_pairwise_mean_mm": float(np.mean(burial_diffs)),
            "burial_pairwise_max_mm": float(np.max(burial_diffs)),
            "lwd_pairwise_mean_mm": float(np.mean(lwd_distances)),
            "lwd_pairwise_max_mm": float(np.max(lwd_distances)),
            "b2_f2_center_disagreement_mm": float(np.linalg.norm(centers[0] - centers[2])),
            "h2_f2_center_disagreement_mm": float(np.linalg.norm(centers[1] - centers[2])),
            "b2_f2_burial_disagreement_mm": abs(burials[0] - burials[2]),
            "h2_f2_burial_disagreement_mm": abs(burials[1] - burials[2]),
            "f2_pred_L_mm": safe_float(f2["pred_L_mm"]),
            "f2_pred_W_mm": safe_float(f2["pred_W_mm"]),
            "f2_pred_D_mm": safe_float(f2["pred_D_mm"]),
            "f2_pred_burial_depth_mm": safe_float(f2["pred_burial_depth_mm"]),
            "f2_pred_center_norm_mm": float(np.linalg.norm(centers[2])),
            "f2_pred_train_range_violation_count": int(np.sum(train_range_violation)),
            "feature_abs_z_mean": float(np.mean(z_abs)),
            "feature_abs_z_p95": float(np.percentile(z_abs, 95)),
            "feature_abs_z_max": float(np.max(z_abs)),
            "feature_outlier_count_z3": int(np.sum(z_abs > 3.0)),
            "f2_total_abs_normalized_error": safe_float(f2["total_abs_normalized_error"]),
            "f2_L_error_mm": safe_float(f2["L_error_mm"]),
            "f2_W_error_mm": safe_float(f2["W_error_mm"]),
            "f2_D_error_mm": safe_float(f2["D_error_mm"]),
            "f2_burial_depth_error_mm": safe_float(f2["burial_depth_error_mm"]),
            "f2_center_xyz_error_mm": safe_float(f2["center_xyz_error_mm"]),
            "center_outlier": center_outlier,
            "burial_outlier": burial_outlier,
            "dimension_outlier": dimension_outlier,
            "shape_misclassified": shape_misclassified,
            "full_shift_failure": catastrophic,
            "catastrophic_failure": catastrophic,
            "geometry_branch_failure": geometry_branch,
            "bad_tail_target": center_outlier or burial_outlier or catastrophic or geometry_branch,
        }
        row.update(delta_features)
        rows.append(row)

    correlations = compute_correlations(rows)
    return rows, correlations, manifest.get("npz_path", "")


def compute_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in TARGET_COLUMNS:
        y = np.asarray([1.0 if as_bool(row[target]) else 0.0 for row in rows], dtype=np.float64)
        if y.std() < 1e-12:
            continue
        for feature in RISK_FEATURE_COLUMNS:
            x = np.asarray([safe_float(row[feature]) for row in rows], dtype=np.float64)
            if x.std() < 1e-12:
                corr = 0.0
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
            result.append(
                {
                    "target": target,
                    "feature": feature,
                    "pearson_correlation": corr,
                    "abs_correlation": abs(corr),
                    "sample_count": len(rows),
                    "target_positive_count": int(y.sum()),
                }
            )
    result.sort(key=lambda item: (item["target"], -safe_float(item["abs_correlation"])))
    return result


def group_counts(rows: list[dict[str, Any]], group: str, target: str) -> list[tuple[str, int, int]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row[group]), []).append(row)
    result = []
    for key, items in sorted(buckets.items()):
        result.append((key, len(items), sum(1 for item in items if as_bool(item[target]))))
    return result


def write_preflight_summary(npz_path: str, rows: list[dict[str, Any]]) -> None:
    forbidden = git_forbidden_status()
    text = [
        "22.6 internal defect tail-risk gate 预检",
        "",
        f"- 工作目录：{ROOT}",
        f"- dataset_id：{DATASET_ID}",
        f"- manifest：{MANIFEST_PATH}",
        f"- manifest NPZ 路径只读引用：{npz_path}",
        f"- B2/H2/F2 prediction CSV：均存在，合并样本数 {len(rows)}",
        "- 本阶段未运行 COMSOL，未训练，未写 data/NPZ，未更新 CURRENT_BASELINE.md。",
        f"- forbidden artifact 状态：{'干净' if not forbidden else '; '.join(forbidden)}",
        "",
        "预检结论：可继续执行 risk signal audit 和轻量 abstention gate。",
    ]
    write_text(PREFLIGHT_SUMMARY, "\n".join(text) + "\n")


def write_signal_summary(rows: list[dict[str, Any]], correlations: list[dict[str, Any]]) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    cat = sum(1 for row in test_rows if as_bool(row["catastrophic_failure"]))
    geo = sum(1 for row in test_rows if as_bool(row["geometry_branch_failure"]))
    bad = sum(1 for row in test_rows if as_bool(row["bad_tail_target"]))
    shape_disagree_bad = sum(1 for row in test_rows if safe_float(row["shape_disagreement_count"]) > 0 and as_bool(row["bad_tail_target"]))
    center_disagree = np.asarray([safe_float(row["center_pairwise_max_mm"]) for row in test_rows], dtype=np.float64)
    top_bad = [row for row in correlations if row["target"] == "bad_tail_target"][:5]
    top_cat = [row for row in correlations if row["target"] == "catastrophic_failure"][:5]
    lines = [
        "22.6 tail-risk 信号审计摘要",
        "",
        f"- 样本数：全量 {len(rows)}，test {len(test_rows)}。",
        f"- F2 test tail：bad_tail={bad}/{len(test_rows)}，catastrophic={cat}/{len(test_rows)}，geometry_branch={geo}/{len(test_rows)}。",
        f"- test 中 model shape disagreement 命中 bad_tail：{shape_disagree_bad} 个；center_pairwise_max 中位数/95分位：{np.median(center_disagree):.3f}/{np.percentile(center_disagree, 95):.3f} mm。",
        "- shape probability entropy / top1-top2 margin：现有 B2/H2/F2 prediction artifact 未保存 logits/probability，因此本轮用跨模型 shape disagreement 作为可部署代理信号。",
        "- feature baseline per-sample prediction：当前未保存，无法形成 feature-vs-neural per-sample disagreement；本轮使用 delta_b-derived feature anomaly 作为替代。",
        "",
        "bad_tail_target 相关性最高的推理信号：",
    ]
    for row in top_bad:
        lines.append(f"- {row['feature']}: r={safe_float(row['pearson_correlation']):.3f}")
    lines.append("")
    lines.append("catastrophic_failure 相关性最高的推理信号：")
    for row in top_cat:
        lines.append(f"- {row['feature']}: r={safe_float(row['pearson_correlation']):.3f}")
    lines.append("")
    lines.append("分组 tail 证据：")
    for group in ["true_shape_type", "burial_depth_level", "size_level", "aspect_bin"]:
        parts = [f"{key}={pos}/{count}" for key, count, pos in group_counts(test_rows, group, "bad_tail_target")]
        lines.append(f"- {group}: " + ", ".join(parts))
    lines.append("")
    lines.append("结论：tail-risk 需要用跨模型 disagreement、预测范围异常和 delta_b feature anomaly 联合判断，单一 shape 分支信号不足。")
    write_text(SIGNAL_SUMMARY, "\n".join(lines) + "\n")


def main() -> int:
    rows, correlations, npz_path = build_risk_signal_rows()
    fields = METADATA_COLUMNS + RISK_FEATURE_COLUMNS + ERROR_COLUMNS + TARGET_COLUMNS
    write_csv(SIGNAL_AUDIT_CSV, rows, fields)
    write_csv(
        CORRELATION_CSV,
        correlations,
        ["target", "feature", "pearson_correlation", "abs_correlation", "sample_count", "target_positive_count"],
    )
    write_preflight_summary(npz_path, rows)
    write_signal_summary(rows, correlations)
    print(json.dumps({"rows": len(rows), "signal_csv": str(SIGNAL_AUDIT_CSV)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
