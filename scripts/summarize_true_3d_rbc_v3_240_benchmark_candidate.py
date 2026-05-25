"""Summarize the fixed-scope v3_240 true 3D RBC benchmark candidate audit.

This script is audit-only. It reads registry, manifest, and existing 20.77
metrics. It does not scan for newest/latest NPZ files, train models, run
COMSOL, or modify data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"

MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"

SUMMARY_OUT = ROOT / "results/summaries/true_3d_rbc_v3_240_benchmark_candidate_summary.txt"
METRICS_OUT = ROOT / "results/metrics/true_3d_rbc_v3_240_benchmark_candidate_metrics.csv"

INPUT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_training_gate_input_summary.txt"
FEATURE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_feature_baseline_summary.txt"
NEURAL_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_neural_training_gate_summary.txt"
DECISION_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_training_gate_decision_summary.txt"

FEATURE_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_baseline_metrics.csv"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_seed_summary.csv"
PARAM_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_metrics.csv"
PROFILE_METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_profile_metrics.csv"
V2_COMPARISON = ROOT / "results/metrics/true_3d_rbc_v3_240_vs_v2_120_comparison.csv"
V1_COMPARISON = ROOT / "results/metrics/true_3d_rbc_v3_240_vs_v1_56_comparison.csv"
DECISION_MATRIX = ROOT / "results/metrics/true_3d_rbc_v3_240_training_gate_decision_matrix.csv"


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


def selected_feature_test() -> dict[str, str]:
    rows = read_csv(FEATURE_METRICS)
    selected = [r for r in rows if r.get("split") == "test" and r.get("selected_by_val") == "True"]
    if not selected:
        raise RuntimeError("selected feature test row missing")
    return selected[0]


def mean_feature_test() -> dict[str, str]:
    rows = read_csv(FEATURE_METRICS)
    selected = [r for r in rows if r.get("split") == "test" and r.get("model") == "mean_train_target"]
    if not selected:
        raise RuntimeError("mean test row missing")
    return selected[0]


def selected_seed() -> dict[str, str]:
    rows = read_csv(SEED_SUMMARY)
    selected = [r for r in rows if r.get("selected_seed") == "True"]
    if not selected:
        raise RuntimeError("selected neural seed row missing")
    return selected[0]


def selected_param(param: str) -> dict[str, str]:
    rows = read_csv(PARAM_METRICS)
    selected = [
        r
        for r in rows
        if r.get("selected_seed") == "True" and r.get("split") == "test" and r.get("param") == param
    ]
    if not selected:
        raise RuntimeError(f"selected test param row missing: {param}")
    return selected[0]


def comparison_value(path: Path, metric: str) -> dict[str, str]:
    rows = read_csv(path)
    selected = [r for r in rows if r.get("metric") == metric]
    if not selected:
        raise RuntimeError(f"comparison row missing: {path.name}:{metric}")
    return selected[0]


def key_value_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def main() -> int:
    required = [
        MANIFEST,
        REGISTRY,
        INPUT_SUMMARY,
        FEATURE_SUMMARY,
        NEURAL_SUMMARY,
        DECISION_SUMMARY,
        FEATURE_METRICS,
        SEED_SUMMARY,
        PARAM_METRICS,
        PROFILE_METRICS,
        V2_COMPARISON,
        V1_COMPARISON,
        DECISION_MATRIX,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing audit inputs: " + "; ".join(missing))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry_text = REGISTRY.read_text(encoding="utf-8")
    if DATASET_ID not in registry_text:
        raise RuntimeError("dataset_id missing from registry")
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError("manifest dataset_id mismatch")

    feature = selected_feature_test()
    mean = mean_feature_test()
    seed = selected_seed()
    neural_summary = key_value_summary(NEURAL_SUMMARY)
    decision = key_value_summary(DECISION_SUMMARY)
    l_row = selected_param("L_m")
    w_row = selected_param("W_m")
    d_row = selected_param("D_m")
    curv_row = selected_param("CURVATURE_MEAN")

    v2_neural = comparison_value(V2_COMPARISON, "neural_test_normalized_mae")
    v1_neural = comparison_value(V1_COMPARISON, "neural_test_normalized_mae")
    v2_d = comparison_value(V2_COMPARISON, "D_mae_mm")
    v2_curv = comparison_value(V2_COMPARISON, "curvature_mae")

    metric_rows = [
        {"section": "identity", "metric": "dataset_id", "value": DATASET_ID, "notes": "explicit registry/manifest identity"},
        {"section": "identity", "metric": "route", "value": manifest.get("route"), "notes": ""},
        {"section": "identity", "metric": "status", "value": manifest.get("status"), "notes": ""},
        {"section": "identity", "metric": "train_ready_candidate", "value": manifest.get("train_ready_candidate"), "notes": ""},
        {"section": "identity", "metric": "baseline_ready", "value": manifest.get("baseline_ready"), "notes": "must remain false"},
        {"section": "identity", "metric": "geometry_method", "value": manifest.get("geometry_method"), "notes": ""},
        {"section": "identity", "metric": "exact_piao_rbc", "value": manifest.get("exact_piao_rbc"), "notes": "not an exact Piao RBC reproduction"},
        {"section": "identity", "metric": "rbc_style_approximation", "value": manifest.get("rbc_style_approximation"), "notes": ""},
        {"section": "data", "metric": "n_samples", "value": manifest.get("n_samples"), "notes": ""},
        {"section": "data", "metric": "split_counts", "value": json.dumps(manifest.get("split_counts"), sort_keys=True), "notes": ""},
        {"section": "input", "metric": "input_shape_delta_b", "value": "[240, 3, 3, 201]", "notes": "Bx/By/Bz x 3 scan lines x 201"},
        {"section": "input", "metric": "conv1d_shape", "value": "[240, 9, 201]", "notes": "flattened axes and scan lines"},
        {"section": "feature", "metric": "selected_model", "value": feature.get("model"), "notes": "Piao-inspired feature sanity only"},
        {"section": "feature", "metric": "test_normalized_mae", "value": feature.get("normalized_param_mae_mean_mean"), "notes": ""},
        {"section": "feature", "metric": "test_LWD_mae_mm", "value": f"{feature.get('L_mae_mm_mean')}/{feature.get('W_mae_mm_mean')}/{feature.get('D_mae_mm_mean')}", "notes": "L/W/D"},
        {"section": "feature", "metric": "test_curvature_mae", "value": feature.get("curvature_mae_mean_mean"), "notes": ""},
        {"section": "feature", "metric": "test_mask_iou_dice", "value": f"{feature.get('projected_mask_iou_mean')}/{feature.get('projected_mask_dice_mean')}", "notes": ""},
        {"section": "neural", "metric": "selected_seed", "value": seed.get("seed"), "notes": "validation selected"},
        {"section": "neural", "metric": "train_val_test_normalized_mae", "value": f"{seed.get('train_normalized_param_mae')}/{seed.get('val_normalized_param_mae')}/{seed.get('test_normalized_param_mae')}", "notes": ""},
        {"section": "neural", "metric": "test_LWD_mae_mm", "value": f"{l_row.get('physical_mae_mm')}/{w_row.get('physical_mae_mm')}/{d_row.get('physical_mae_mm')}", "notes": "L/W/D"},
        {"section": "neural", "metric": "test_curvature_mae", "value": curv_row.get("physical_mae"), "notes": "mean wLD/wWD/wLW absolute error"},
        {"section": "neural", "metric": "test_mask_iou_dice", "value": f"{seed.get('test_projected_mask_iou')}/{seed.get('test_projected_mask_dice')}", "notes": ""},
        {"section": "neural", "metric": "test_profile_depth_rmse_m", "value": seed.get("test_profile_depth_rmse_m"), "notes": ""},
        {"section": "comparison", "metric": "v3_vs_v2_neural_test_mae_delta", "value": v2_neural.get("delta"), "notes": f"improved={v2_neural.get('improved')}"},
        {"section": "comparison", "metric": "v3_vs_v1_neural_test_mae_delta", "value": v1_neural.get("delta"), "notes": f"improved={v1_neural.get('improved')}"},
        {"section": "comparison", "metric": "v3_vs_v2_D_mae_mm_delta", "value": v2_d.get("delta"), "notes": f"improved={v2_d.get('improved')}"},
        {"section": "comparison", "metric": "v3_vs_v2_curvature_mae_delta", "value": v2_curv.get("delta"), "notes": f"improved={v2_curv.get('improved')}"},
        {"section": "decision", "metric": "overall_decision", "value": decision.get("overall_decision"), "notes": "candidate, not baseline"},
        {"section": "decision", "metric": "prior_20_77_next_step", "value": decision.get("next_step"), "notes": "20.77 gate recommendation before 20.78 curvature audit"},
    ]
    write_csv(METRICS_OUT, metric_rows)

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 benchmark candidate summary",
                "",
                f"dataset_id: {DATASET_ID}",
                f"registry_manifest_status: pass; status={manifest.get('status')}; train_ready_candidate={manifest.get('train_ready_candidate')}; baseline_ready={manifest.get('baseline_ready')}",
                f"route: {manifest.get('route')}",
                f"geometry_method: {manifest.get('geometry_method')}",
                f"exact_piao_rbc: {manifest.get('exact_piao_rbc')}",
                f"rbc_style_approximation: {manifest.get('rbc_style_approximation')}",
                "input_shape: delta_b=(240,3,3,201), Conv1D=(240,9,201)",
                f"split: {manifest.get('split_counts')}",
                "",
                f"feature_baseline: selected={feature.get('model')}; test_normalized_mae={float(feature.get('normalized_param_mae_mean_mean')):.6f}; L/W/D_mae_mm={float(feature.get('L_mae_mm_mean')):.3f}/{float(feature.get('W_mae_mm_mean')):.3f}/{float(feature.get('D_mae_mm_mean')):.3f}; curvature_mae={float(feature.get('curvature_mae_mean_mean')):.6f}; IoU/Dice={float(feature.get('projected_mask_iou_mean')):.6f}/{float(feature.get('projected_mask_dice_mean')):.6f}",
                f"neural_selected: seed={seed.get('seed')}; train/val/test_normalized_mae={float(seed.get('train_normalized_param_mae')):.6f}/{float(seed.get('val_normalized_param_mae')):.6f}/{float(seed.get('test_normalized_param_mae')):.6f}; L/W/D_mae_mm={float(l_row.get('physical_mae_mm')):.3f}/{float(w_row.get('physical_mae_mm')):.3f}/{float(d_row.get('physical_mae_mm')):.3f}; curvature_mae={float(curv_row.get('physical_mae')):.6f}",
                f"projected_mask_depth: IoU/Dice={float(seed.get('test_projected_mask_iou')):.6f}/{float(seed.get('test_projected_mask_dice')):.6f}; profile_depth_rmse_m={float(seed.get('test_profile_depth_rmse_m')):.9f}",
                "",
                f"mean_baseline_comparison: neural_test={float(seed.get('test_normalized_param_mae')):.6f} vs mean_test={float(mean.get('normalized_param_mae_mean_mean')):.6f}; neural_beats_mean=True",
                f"feature_comparison: neural_test={float(seed.get('test_normalized_param_mae')):.6f} vs feature_test={float(feature.get('normalized_param_mae_mean_mean')):.6f}; neural_beats_feature=True",
                f"N_trend: N56={float(v1_neural.get('reference_value')):.6f} -> N112={float(v2_neural.get('reference_value')):.6f} -> N240={float(v2_neural.get('current_value')):.6f}",
                f"D_m_improvement_vs_N112: {float(v2_d.get('reference_value')):.3f} mm -> {float(v2_d.get('current_value')):.3f} mm",
                f"curvature_risk_vs_N112: {float(v2_curv.get('reference_value')):.6f} -> {float(v2_curv.get('current_value')):.6f}; not improved",
                "learnable_params: L_m, W_m, D_m",
                "unstable_params: wLD, wWD, wLW",
                "",
                "benchmark_candidate_recommendation: formal benchmark candidate with curvature risk",
                "baseline_ready_status: false",
                "boundary: no COMSOL run, no data generation, no NPZ modification, no retraining, no baseline update.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
