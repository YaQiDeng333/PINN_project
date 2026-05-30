#!/usr/bin/env python
"""22.7 internal defect inference smoke with abstention.

本脚本不训练 internal 主模型、不运行 COMSOL、不写 data/NPZ。它显式加载
v3_hardcase dataset、B2 inference artifact 和 22.6 tail-risk gate，在 test
split 上输出 no-abstention B2 与带 abstention 的推理 smoke 指标。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from audit_internal_defect_tail_risk_signals import (
    DATASET_ID,
    RISK_FEATURE_COLUMNS,
    SIGNAL_AUDIT_CSV,
    build_risk_signal_rows,
    as_bool,
)
from internal_defect_hardcase_utils import (
    B2_MANIFEST,
    DATASET_ID as HARDCASE_DATASET_ID,
    load_old_b2_on_dataset,
    prepare_dataset,
    prediction_rows,
    safe_float,
    sha256_file,
)
from load_internal_defect_pilot_dataset import ROOT, SHAPE_CLASSES, classification_metrics, write_csv


SUMMARY_DIR = ROOT / "results/summaries"
METRICS_DIR = ROOT / "results/metrics"
MANIFEST_DIR = ROOT / "results/manifests"

RISK_CONTRACT_PATH = METRICS_DIR / "internal_defect_tail_risk_gate_model_contract.json"
RISK_GATE_ARTIFACT_DIR = ROOT / "checkpoints/internal_defect_tail_risk_gate_artifacts"
RISK_GATE_PICKLE = RISK_GATE_ARTIFACT_DIR / "internal_defect_tail_risk_gate_random_forest_v22_6.pkl"
RISK_GATE_MANIFEST = MANIFEST_DIR / "internal_defect_tail_risk_gate_artifact_manifest.json"

PREFLIGHT_SUMMARY = SUMMARY_DIR / "internal_defect_inference_abstention_preflight_summary.txt"
SMOKE_SUMMARY = SUMMARY_DIR / "internal_defect_inference_abstention_smoke_summary.txt"

METRICS_CSV = METRICS_DIR / "internal_defect_inference_abstention_metrics.csv"
ACCEPTED_CSV = METRICS_DIR / "internal_defect_inference_abstention_accepted_subset.csv"
ABSTAINED_CSV = METRICS_DIR / "internal_defect_inference_abstention_abstained_subset.csv"
FAILURE_CSV = METRICS_DIR / "internal_defect_inference_abstention_failure_cases.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    if not RISK_CONTRACT_PATH.exists():
        raise FileNotFoundError(RISK_CONTRACT_PATH)
    contract = json.loads(RISK_CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("selected_model") != "random_forest_small":
        raise RuntimeError(f"unsupported risk gate model: {contract.get('selected_model')}")
    if list(contract.get("risk_feature_columns", [])) != RISK_FEATURE_COLUMNS:
        raise RuntimeError("risk feature column order mismatch")
    return contract


def ensure_signal_rows() -> list[dict[str, Any]]:
    if SIGNAL_AUDIT_CSV.exists():
        return read_csv(SIGNAL_AUDIT_CSV)
    rows, _, _ = build_risk_signal_rows()
    write_csv(SIGNAL_AUDIT_CSV, rows, list(rows[0].keys()))
    return rows


def recover_risk_gate_artifact(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    """按 22.6 固定协议确定性恢复 risk gate artifact。

    这是 22.7 preflight 允许的 artifact recovery，不是 internal defect 主模型训练。
    """
    if RISK_GATE_MANIFEST.exists():
        manifest = json.loads(RISK_GATE_MANIFEST.read_text(encoding="utf-8"))
        artifact = Path(manifest["artifact_path"])
        if artifact.exists() and file_sha256(artifact) == manifest.get("artifact_sha256"):
            return manifest

    x = np.asarray([[safe_float(row[col]) for col in RISK_FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    y = np.asarray([1 if as_bool(row["bad_tail_target"]) else 0 for row in rows], dtype=np.int64)
    split = np.asarray([row["split"] for row in rows])
    train_mask = split == "train"
    mean = np.asarray(contract["x_mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(contract["x_std"], dtype=np.float64).reshape(1, -1)
    x_scaled = (x - mean) / np.where(std < 1e-8, 1.0, std)

    model = RandomForestClassifier(
        n_estimators=128,
        max_depth=4,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_scaled[train_mask], y[train_mask])
    payload = {
        "model": model,
        "model_type": "random_forest_small",
        "dataset_id": DATASET_ID,
        "threshold": float(contract["threshold"]),
        "risk_feature_columns": RISK_FEATURE_COLUMNS,
        "x_mean": contract["x_mean"],
        "x_std": contract["x_std"],
        "train_label_range_feature": contract.get("train_label_range_feature"),
        "train_label_param_order": contract.get("train_label_param_order"),
        "train_label_low_mm_with_5pct_margin": contract.get("train_label_low_mm_with_5pct_margin"),
        "train_label_high_mm_with_5pct_margin": contract.get("train_label_high_mm_with_5pct_margin"),
        "protocol": "22.6 validation-selected deterministic recovery for 22.7 inference smoke",
    }
    RISK_GATE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with RISK_GATE_PICKLE.open("wb") as f:
        pickle.dump(payload, f)
    manifest = {
        "artifact_type": "internal_defect_tail_risk_gate",
        "stage": "22.7",
        "source_stage": "22.6",
        "dataset_id": DATASET_ID,
        "model_type": "random_forest_small",
        "threshold": float(contract["threshold"]),
        "artifact_path": str(RISK_GATE_PICKLE),
        "artifact_sha256": file_sha256(RISK_GATE_PICKLE),
        "risk_contract_path": str(RISK_CONTRACT_PATH),
        "risk_signal_audit_csv": str(SIGNAL_AUDIT_CSV),
        "selection_protocol": "train fit, validation threshold/model selection, test final only in 22.6",
        "labels_as_inference_input": False,
        "allowed_use": ["internal_inference_smoke_with_abstention"],
        "forbidden_use": ["baseline_update", "current_baseline_replacement", "stable_all_sample_inference_claim"],
    }
    RISK_GATE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RISK_GATE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_risk_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    artifact = Path(manifest["artifact_path"])
    if not artifact.exists():
        raise FileNotFoundError(artifact)
    if file_sha256(artifact) != manifest.get("artifact_sha256"):
        raise RuntimeError("risk gate artifact sha256 mismatch")
    with artifact.open("rb") as f:
        payload = pickle.load(f)
    if payload["risk_feature_columns"] != RISK_FEATURE_COLUMNS:
        raise RuntimeError("risk gate feature column mismatch")
    return payload


def risk_scores(rows: list[dict[str, Any]], payload: dict[str, Any]) -> np.ndarray:
    x = np.asarray([[safe_float(row[col]) for col in RISK_FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    mean = np.asarray(payload["x_mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(payload["x_std"], dtype=np.float64).reshape(1, -1)
    x_scaled = (x - mean) / np.where(std < 1e-8, 1.0, std)
    return payload["model"].predict_proba(x_scaled)[:, 1]


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def build_inference_rows(b2_rows: list[dict[str, Any]], signal_rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    signal_by_id = {row["sample_id"]: row for row in signal_rows}
    score_by_id = {row["sample_id"]: float(score) for row, score in zip(signal_rows, scores)}
    output: list[dict[str, Any]] = []
    for row in b2_rows:
        sample_id = row["sample_id"]
        signal = signal_by_id[sample_id]
        score = score_by_id[sample_id]
        high = score >= threshold
        tags = row.get("failure_tags", "")
        center_outlier = safe_float(row["center_xyz_error_mm"]) > 3.0
        burial_outlier = safe_float(row["burial_depth_error_mm"]) > 1.0
        shape_mis = not bool(row["shape_correct"])
        catastrophic = bool(row["is_catastrophic_failure"]) or (center_outlier and burial_outlier)
        geometry = bool(row["is_geometry_branch_failure"]) or (shape_mis and center_outlier and burial_outlier)
        bad = center_outlier or burial_outlier or catastrophic or geometry
        risk_reason = []
        for key in [
            "shape_disagreement_count",
            "center_pairwise_max_mm",
            "burial_pairwise_max_mm",
            "feature_abs_z_max",
            "f2_pred_train_range_violation_count",
        ]:
            if safe_float(signal.get(key, 0.0)) > 0:
                risk_reason.append(f"{key}={safe_float(signal[key]):.3f}")
        output.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "subset": row["subset"],
                "row_origin": row["row_origin"],
                "inference_status": "abstain_need_review" if high else "accepted_prediction",
                "risk_score": score,
                "risk_threshold": threshold,
                "risk_reason": "; ".join(risk_reason[:5]),
                "true_shape_type": row["true_shape_type"],
                "pred_shape_type": row["pred_shape_type"],
                "shape_correct": row["shape_correct"],
                "burial_depth_level": row["burial_depth_level"],
                "size_level": row["size_level"],
                "aspect_bin": row["aspect_bin"],
                "hardcase_target_id": row["hardcase_target_id"],
                "pred_L_mm": row["pred_L_mm"],
                "pred_W_mm": row["pred_W_mm"],
                "pred_D_mm": row["pred_D_mm"],
                "pred_burial_depth_mm": row["pred_burial_depth_mm"],
                "pred_center_x_mm": row["pred_center_x_mm"],
                "pred_center_y_mm": row["pred_center_y_mm"],
                "pred_center_z_mm": row["pred_center_z_mm"],
                "L_error_mm": row["L_error_mm"],
                "W_error_mm": row["W_error_mm"],
                "D_error_mm": row["D_error_mm"],
                "burial_depth_error_mm": row["burial_depth_error_mm"],
                "center_xyz_error_mm": row["center_xyz_error_mm"],
                "total_abs_normalized_error": row["total_abs_normalized_error"],
                "center_outlier": center_outlier,
                "burial_outlier": burial_outlier,
                "catastrophic_failure": catastrophic,
                "geometry_branch_failure": geometry,
                "bad_tail_target": bad,
                "failure_tags": tags,
                "stable_prediction_claim": False if high else True,
            }
        )
    return output


def shape_metrics(rows: list[dict[str, Any]]) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    mapping = {name: i for i, name in enumerate(SHAPE_CLASSES)}
    true = np.asarray([mapping[str(row["true_shape_type"])] for row in rows], dtype=np.int64)
    pred = np.asarray([mapping[str(row["pred_shape_type"])] for row in rows], dtype=np.int64)
    metrics = classification_metrics(true, pred)
    return metrics["shape_accuracy"], metrics["shape_macro_f1"]


def metric_row(name: str, rows: list[dict[str, Any]], high_risk_reference: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    n = len(rows)
    center = np.asarray([safe_float(row["center_xyz_error_mm"]) for row in rows], dtype=np.float64)
    burial = np.asarray([safe_float(row["burial_depth_error_mm"]) for row in rows], dtype=np.float64)
    total = np.asarray([safe_float(row["total_abs_normalized_error"]) for row in rows], dtype=np.float64)
    center_q = quantiles(center)
    burial_q = quantiles(burial)
    total_q = quantiles(total)
    shape_acc, shape_f1 = shape_metrics(rows)

    all_rows = high_risk_reference or rows
    high = np.asarray([row["inference_status"] == "abstain_need_review" for row in all_rows], dtype=bool)
    bad = np.asarray([as_bool(row["bad_tail_target"]) for row in all_rows], dtype=bool)
    cat = np.asarray([as_bool(row["catastrophic_failure"]) for row in all_rows], dtype=bool)
    geo = np.asarray([as_bool(row["geometry_branch_failure"]) for row in all_rows], dtype=bool)
    clean = ~bad

    def recall(mask: np.ndarray) -> float:
        positives = int(mask.sum())
        return float((high & mask).sum() / positives) if positives else 1.0

    false_alarm = float((high & clean).sum() / clean.sum()) if int(clean.sum()) else 0.0
    if high_risk_reference is not None and name == "abstention_gate_test":
        coverage_retained = float((~high).sum() / len(all_rows)) if all_rows else 0.0
    elif high_risk_reference is not None:
        coverage_retained = float(n / len(all_rows)) if all_rows else 0.0
    else:
        coverage_retained = 1.0 if n else 0.0
    return {
        "metric_scope": name,
        "sample_count": n,
        "coverage_retained": coverage_retained,
        "high_risk_count": int(high.sum()) if high_risk_reference is not None else 0,
        "bad_tail_count": int(bad.sum()) if high_risk_reference is not None else int(sum(as_bool(row["bad_tail_target"]) for row in rows)),
        "catastrophic_failure_count": int(cat.sum()) if high_risk_reference is not None else int(sum(as_bool(row["catastrophic_failure"]) for row in rows)),
        "geometry_branch_failure_count": int(geo.sum()) if high_risk_reference is not None else int(sum(as_bool(row["geometry_branch_failure"]) for row in rows)),
        "catastrophic_failure_recall": recall(cat) if high_risk_reference is not None else "",
        "geometry_branch_failure_recall": recall(geo) if high_risk_reference is not None else "",
        "false_alarm_rate": false_alarm if high_risk_reference is not None else "",
        "total_error_mean": total_q["mean"],
        "total_error_median": total_q["median"],
        "total_error_p95": total_q["p95"],
        "total_error_max": total_q["max"],
        "center_error_mean_mm": center_q["mean"],
        "center_error_median_mm": center_q["median"],
        "center_error_p75_mm": center_q["p75"],
        "center_error_p90_mm": center_q["p90"],
        "center_error_p95_mm": center_q["p95"],
        "center_error_max_mm": center_q["max"],
        "burial_error_mean_mm": burial_q["mean"],
        "burial_error_median_mm": burial_q["median"],
        "burial_error_p75_mm": burial_q["p75"],
        "burial_error_p90_mm": burial_q["p90"],
        "burial_error_p95_mm": burial_q["p95"],
        "burial_error_max_mm": burial_q["max"],
        "shape_accuracy": shape_acc,
        "shape_macro_f1": shape_f1,
    }


def write_preflight_summary(risk_manifest: dict[str, Any], recovered: bool) -> None:
    text = [
        "22.7 internal defect inference with abstention 预检",
        "",
        f"- 工作目录：{ROOT}",
        f"- dataset_id：{DATASET_ID}",
        f"- v3_hardcase manifest：{ROOT / 'results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json'}",
        f"- B2 artifact manifest：{B2_MANIFEST}",
        f"- risk gate manifest：{RISK_GATE_MANIFEST}",
        f"- risk gate artifact：{risk_manifest['artifact_path']}",
        f"- risk gate artifact recovery：{'本轮确定性恢复并写入 ignored checkpoints' if recovered else '已存在并通过 sha256'}",
        "- 本阶段未训练 internal 主模型，未运行 COMSOL，未写 data/NPZ，未更新 CURRENT_BASELINE.md。",
        "",
        "预检结论：abstention inference smoke 可以执行。",
    ]
    write_text(PREFLIGHT_SUMMARY, "\n".join(text) + "\n")


def write_smoke_summary(metrics: list[dict[str, Any]], risk_manifest: dict[str, Any]) -> None:
    full = next(row for row in metrics if row["metric_scope"] == "no_abstention_b2_full_test")
    accepted = next(row for row in metrics if row["metric_scope"] == "abstention_b2_accepted_test")
    route = next(row for row in metrics if row["metric_scope"] == "abstention_gate_test")
    lines = [
        "22.7 internal defect inference with abstention smoke 摘要",
        "",
        f"- risk gate：{risk_manifest['model_type']}，threshold={safe_float(risk_manifest['threshold']):.8f}。",
        f"- test coverage retained={safe_float(route['coverage_retained']):.3f}，high-risk count={int(route['high_risk_count'])}/{int(route['sample_count'])}。",
        f"- catastrophic recall={safe_float(route['catastrophic_failure_recall']):.3f}，geometry_branch recall={safe_float(route['geometry_branch_failure_recall']):.3f}，false alarm={safe_float(route['false_alarm_rate']):.3f}。",
        f"- no_abstention B2 center p95/max={safe_float(full['center_error_p95_mm']):.3f}/{safe_float(full['center_error_max_mm']):.3f} mm；accepted subset={safe_float(accepted['center_error_p95_mm']):.3f}/{safe_float(accepted['center_error_max_mm']):.3f} mm。",
        f"- no_abstention B2 burial p95/max={safe_float(full['burial_error_p95_mm']):.3f}/{safe_float(full['burial_error_max_mm']):.3f} mm；accepted subset={safe_float(accepted['burial_error_p95_mm']):.3f}/{safe_float(accepted['burial_error_max_mm']):.3f} mm。",
        f"- accepted shape accuracy/F1={safe_float(accepted['shape_accuracy']):.3f}/{safe_float(accepted['shape_macro_f1']):.3f}。",
        "",
        "结论：runner 可用，但只能作为带 abstention 的 smoke。高风险样本只输出 raw prediction + risk flag，不给稳定 center/burial 结论。",
    ]
    write_text(SMOKE_SUMMARY, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run internal defect inference smoke with abstention.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset_id != HARDCASE_DATASET_ID:
        raise ValueError(f"22.7 只允许 dataset_id={HARDCASE_DATASET_ID}")

    prepared = prepare_dataset(args.dataset_id)
    dataset = prepared["dataset"]
    if not B2_MANIFEST.exists():
        raise FileNotFoundError(B2_MANIFEST)
    # 显式加载 B2 checkpoint，保证 no-abstention path 来自 artifact，而不是 latest scan。
    b2_pred, b2_shape, _ = load_old_b2_on_dataset(prepared)
    b2_rows = prediction_rows("B2_no_abstention_artifact", "2026", dataset, b2_pred, b2_shape, prepared["y_std"].reshape(-1))

    signal_rows = ensure_signal_rows()
    contract = load_contract()
    existed_before = RISK_GATE_MANIFEST.exists()
    risk_manifest = recover_risk_gate_artifact(signal_rows, contract)
    risk_payload = load_risk_gate(risk_manifest)
    scores = risk_scores(signal_rows, risk_payload)
    threshold = float(risk_payload["threshold"])
    inference_rows = build_inference_rows(b2_rows, signal_rows, scores, threshold)
    test_rows = [row for row in inference_rows if row["split"] == "test"]
    accepted = [row for row in test_rows if row["inference_status"] == "accepted_prediction"]
    abstained = [row for row in test_rows if row["inference_status"] == "abstain_need_review"]
    failures = [
        row
        for row in test_rows
        if as_bool(row["bad_tail_target"]) or as_bool(row["catastrophic_failure"]) or as_bool(row["geometry_branch_failure"])
    ]

    fields = [
        "sample_id",
        "split",
        "subset",
        "row_origin",
        "inference_status",
        "risk_score",
        "risk_threshold",
        "risk_reason",
        "true_shape_type",
        "pred_shape_type",
        "shape_correct",
        "burial_depth_level",
        "size_level",
        "aspect_bin",
        "hardcase_target_id",
        "pred_L_mm",
        "pred_W_mm",
        "pred_D_mm",
        "pred_burial_depth_mm",
        "pred_center_x_mm",
        "pred_center_y_mm",
        "pred_center_z_mm",
        "L_error_mm",
        "W_error_mm",
        "D_error_mm",
        "burial_depth_error_mm",
        "center_xyz_error_mm",
        "total_abs_normalized_error",
        "center_outlier",
        "burial_outlier",
        "catastrophic_failure",
        "geometry_branch_failure",
        "bad_tail_target",
        "failure_tags",
        "stable_prediction_claim",
    ]
    write_csv(ACCEPTED_CSV, accepted, fields)
    write_csv(ABSTAINED_CSV, abstained, fields)
    write_csv(FAILURE_CSV, failures, fields)

    metrics = [
        metric_row("no_abstention_b2_full_test", test_rows),
        metric_row("abstention_gate_test", test_rows, test_rows),
        metric_row("abstention_b2_accepted_test", accepted, test_rows),
        metric_row("abstention_b2_abstained_test", abstained, test_rows),
    ]
    metric_fields = [
        "metric_scope",
        "sample_count",
        "coverage_retained",
        "high_risk_count",
        "bad_tail_count",
        "catastrophic_failure_count",
        "geometry_branch_failure_count",
        "catastrophic_failure_recall",
        "geometry_branch_failure_recall",
        "false_alarm_rate",
        "total_error_mean",
        "total_error_median",
        "total_error_p95",
        "total_error_max",
        "center_error_mean_mm",
        "center_error_median_mm",
        "center_error_p75_mm",
        "center_error_p90_mm",
        "center_error_p95_mm",
        "center_error_max_mm",
        "burial_error_mean_mm",
        "burial_error_median_mm",
        "burial_error_p75_mm",
        "burial_error_p90_mm",
        "burial_error_p95_mm",
        "burial_error_max_mm",
        "shape_accuracy",
        "shape_macro_f1",
    ]
    write_csv(METRICS_CSV, metrics, metric_fields)
    write_preflight_summary(risk_manifest, recovered=not existed_before)
    write_smoke_summary(metrics, risk_manifest)
    print(json.dumps({"test_rows": len(test_rows), "accepted": len(accepted), "abstained": len(abstained)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
