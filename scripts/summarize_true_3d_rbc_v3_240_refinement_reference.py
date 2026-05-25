"""Freeze the 20.77/20.78 reference metrics for v3_240 refinement."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"

MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json"
BENCHMARK_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_benchmark_candidate_metrics.csv"
CURVATURE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_curvature_failure_audit_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_seed_summary.csv"
PARAM_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_metrics.csv"
FEATURE_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_baseline_metrics.csv"

SUMMARY_OUT = ROOT / "results/summaries/true_3d_rbc_v3_240_curvature_refinement_reference_summary.txt"
METRICS_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refinement_reference_metrics.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["section", "metric", "value", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def selected_row(path: Path, **criteria: str) -> dict[str, str]:
    for row in read_csv(path):
        if all(row.get(k) == v for k, v in criteria.items()):
            return row
    raise RuntimeError(f"row not found in {path}: {criteria}")


def benchmark_metric(name: str) -> str:
    for row in read_csv(BENCHMARK_METRICS):
        if row.get("metric") == name:
            return row.get("value", "")
    raise RuntimeError(f"benchmark metric missing: {name}")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError("manifest dataset_id mismatch")
    seed = selected_row(SEED_SUMMARY, selected_seed="True")
    feature = selected_row(FEATURE_METRICS, model="svr_rbf_C10", split="test", selected_by_val="True")
    params = {
        name: selected_row(PARAM_METRICS, selected_seed="True", split="test", param=name)
        for name in ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW", "CURVATURE_MEAN"]
    }
    rows = [
        {"section": "identity", "metric": "dataset_id", "value": DATASET_ID, "notes": "explicit registry/manifest dataset"},
        {"section": "identity", "metric": "status", "value": manifest.get("status"), "notes": ""},
        {"section": "identity", "metric": "train_ready_candidate", "value": manifest.get("train_ready_candidate"), "notes": ""},
        {"section": "identity", "metric": "baseline_ready", "value": manifest.get("baseline_ready"), "notes": "must remain false"},
        {"section": "identity", "metric": "exact_piao_rbc", "value": manifest.get("exact_piao_rbc"), "notes": "RBC-style approximation only"},
        {"section": "data", "metric": "n_samples", "value": manifest.get("n_samples"), "notes": ""},
        {"section": "data", "metric": "split_counts", "value": json.dumps(manifest.get("split_counts"), sort_keys=True), "notes": ""},
        {"section": "reference", "metric": "selected_seed", "value": seed.get("seed"), "notes": "20.77 validation-selected seed"},
        {"section": "reference", "metric": "train_val_test_normalized_mae", "value": f"{seed.get('train_normalized_param_mae')}/{seed.get('val_normalized_param_mae')}/{seed.get('test_normalized_param_mae')}", "notes": ""},
        {"section": "reference", "metric": "L_W_D_mae_mm", "value": f"{params['L_m'].get('physical_mae_mm')}/{params['W_m'].get('physical_mae_mm')}/{params['D_m'].get('physical_mae_mm')}", "notes": ""},
        {"section": "reference", "metric": "curvature_mae", "value": params["CURVATURE_MEAN"].get("physical_mae"), "notes": ""},
        {"section": "reference", "metric": "wLD_wWD_wLW_mae", "value": f"{params['wLD'].get('physical_mae')}/{params['wWD'].get('physical_mae')}/{params['wLW'].get('physical_mae')}", "notes": ""},
        {"section": "reference", "metric": "mask_iou_dice", "value": f"{seed.get('test_projected_mask_iou')}/{seed.get('test_projected_mask_dice')}", "notes": ""},
        {"section": "reference", "metric": "profile_depth_rmse_m", "value": seed.get("test_profile_depth_rmse_m"), "notes": ""},
        {"section": "feature", "metric": "feature_test_normalized_mae", "value": feature.get("normalized_param_mae_mean_mean"), "notes": "Piao-inspired feature sanity only"},
        {"section": "feature", "metric": "feature_curvature_mae", "value": feature.get("curvature_mae_mean_mean"), "notes": ""},
        {"section": "benchmark", "metric": "candidate_decision", "value": benchmark_metric("overall_decision"), "notes": "20.78 audit"},
    ]
    write_csv(METRICS_OUT, rows)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 curvature refinement reference summary",
                "",
                f"dataset_id: {DATASET_ID}",
                f"status: {manifest.get('status')}",
                f"train_ready_candidate: {manifest.get('train_ready_candidate')}",
                f"baseline_ready: {manifest.get('baseline_ready')}",
                f"input_shape: delta_b=(240,3,3,201), Conv1D=(240,9,201)",
                f"split: {manifest.get('split_counts')}",
                f"selected_20_77_seed: {seed.get('seed')}",
                f"20_77_neural_train_val_test_mae: {seed.get('train_normalized_param_mae')}/{seed.get('val_normalized_param_mae')}/{seed.get('test_normalized_param_mae')}",
                f"20_77_LWD_mae_mm: {params['L_m'].get('physical_mae_mm')}/{params['W_m'].get('physical_mae_mm')}/{params['D_m'].get('physical_mae_mm')}",
                f"20_77_curvature_mae: {params['CURVATURE_MEAN'].get('physical_mae')}",
                f"20_77_wLD_wWD_wLW_mae: {params['wLD'].get('physical_mae')}/{params['wWD'].get('physical_mae')}/{params['wLW'].get('physical_mae')}",
                f"20_77_mask_iou_dice: {seed.get('test_projected_mask_iou')}/{seed.get('test_projected_mask_dice')}",
                f"20_77_profile_depth_rmse_m: {seed.get('test_profile_depth_rmse_m')}",
                f"feature_baseline_test_mae: {feature.get('normalized_param_mae_mean_mean')}",
                "",
                "curvature_failure_audit:",
                CURVATURE_SUMMARY.read_text(encoding="utf-8").strip(),
                "",
                "boundary: fixed reference only; no COMSOL, no new data, no NPZ modification, no retraining.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
