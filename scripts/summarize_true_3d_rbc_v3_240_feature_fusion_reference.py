#!/usr/bin/env python
"""Freeze 20.77/20.80/20.79 references for Stage 20.81 feature fusion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from load_true_3d_rbc_pilot_dataset import V3_240_DATASET_ID, ROOT, load_dataset, split_indices, write_csv


MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json"
NEURAL_SEED = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_seed_summary.csv"
NEURAL_PARAMS = ROOT / "results/metrics/true_3d_rbc_v3_240_neural_training_gate_metrics.csv"
FEATURE_ONLY = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_regression_metrics.csv"
REFINED_SEED = ROOT / "results/metrics/true_3d_rbc_v3_240_curvature_refined_seed_summary.csv"
FEATURE_QUALITY = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_feature_quality.csv"
FEATURES = ROOT / "results/metrics/true_3d_rbc_v3_240_piao_nls_features.csv"

SUMMARY = ROOT / "results/summaries/true_3d_rbc_v3_240_feature_fusion_reference_summary.txt"
METRICS = ROOT / "results/metrics/true_3d_rbc_v3_240_feature_fusion_reference_metrics.csv"

FIELDS = ["section", "reference", "metric", "value", "source", "notes"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def selected_row(path: Path, selected_key: str = "selected_seed") -> dict[str, str]:
    for row in read_csv(path):
        if str(row.get(selected_key, "")).lower() == "true":
            return row
    raise RuntimeError(f"selected row not found in {path}")


def row_where(path: Path, **criteria: str) -> dict[str, str]:
    for row in read_csv(path):
        if all(row.get(k) == v for k, v in criteria.items()):
            return row
    raise RuntimeError(f"row not found in {path}: {criteria}")


def add(rows: list[dict[str, Any]], section: str, reference: str, metric: str, value: Any, source: Path, notes: str = "") -> None:
    rows.append(
        {
            "section": section,
            "reference": reference,
            "metric": metric,
            "value": value,
            "source": str(source),
            "notes": notes,
        }
    )


def feature_counts() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_csv(FEATURE_QUALITY):
        out[row["feature_group"]] = row["feature_count"]
    return out


def main() -> int:
    dataset = load_dataset(V3_240_DATASET_ID)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    splits = split_indices(dataset)
    if len(dataset.sample_ids) != 240:
        raise RuntimeError(f"expected N=240, observed {len(dataset.sample_ids)}")
    observed_split = {key: len(value) for key, value in splits.items()}
    if observed_split != {"train": 162, "val": 39, "test": 39}:
        raise RuntimeError(f"unexpected split: {observed_split}")
    with FEATURES.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    feature_cols = [name for name in header if name not in {"sample_id", "split"}]
    if not all(name.startswith(("F0__", "F1__", "F2__", "F3__", "F4__", "F5__")) for name in feature_cols):
        raise RuntimeError("feature CSV contains non-allowlisted model input columns")

    neural = selected_row(NEURAL_SEED)
    feature = row_where(FEATURE_ONLY, split="test", selected_by_validation="True")
    refined = selected_row(REFINED_SEED)
    params = {
        name: row_where(NEURAL_PARAMS, selected_seed="True", split="test", param=name)
        for name in ("L_m", "W_m", "D_m", "wLD", "wWD", "wLW", "CURVATURE_MEAN")
    }
    counts = feature_counts()

    rows: list[dict[str, Any]] = []
    add(rows, "identity", "v3_240", "dataset_id", dataset.dataset_id, MANIFEST)
    add(rows, "identity", "v3_240", "status", manifest.get("status"), MANIFEST)
    add(rows, "identity", "v3_240", "train_ready_candidate", manifest.get("train_ready_candidate"), MANIFEST)
    add(rows, "identity", "v3_240", "baseline_ready", manifest.get("baseline_ready"), MANIFEST, "must remain false")
    add(rows, "identity", "v3_240", "input_shape_delta_b", list(dataset.delta_b.shape), MANIFEST)
    add(rows, "identity", "v3_240", "input_shape_conv1d", list(dataset.x_channels.shape), MANIFEST)
    add(rows, "identity", "v3_240", "split_counts", observed_split, MANIFEST)
    add(rows, "features", "20.80", "feature_columns", len(feature_cols), FEATURES)
    add(rows, "features", "20.80", "feature_group_counts", json.dumps(counts, sort_keys=True), FEATURE_QUALITY)
    add(rows, "features", "20.80", "selected_feature_set", "F0_F1_F2_basic_physical", FEATURE_ONLY)
    add(rows, "features", "20.80", "selected_model", "svr_rbf_C10_eps0.03", FEATURE_ONLY)

    add(rows, "reference", "20.77_neural", "selected_seed", neural["seed"], NEURAL_SEED)
    add(rows, "reference", "20.77_neural", "train_val_test_total_mae", f"{neural['train_normalized_param_mae']}/{neural['val_normalized_param_mae']}/{neural['test_normalized_param_mae']}", NEURAL_SEED)
    add(rows, "reference", "20.77_neural", "test_total_mae", neural["test_normalized_param_mae"], NEURAL_SEED)
    add(rows, "reference", "20.77_neural", "test_LWD_mae_mm", f"{neural['test_L_mae_mm']}/{neural['test_W_mae_mm']}/{neural['test_D_mae_mm']}", NEURAL_SEED)
    add(rows, "reference", "20.77_neural", "test_curvature_mae", neural["test_curvature_mae"], NEURAL_SEED)
    add(rows, "reference", "20.77_neural", "test_wLD_wWD_wLW", f"{params['wLD']['physical_mae']}/{params['wWD']['physical_mae']}/{params['wLW']['physical_mae']}", NEURAL_PARAMS)
    add(rows, "reference", "20.77_neural", "test_iou_dice", f"{neural['test_projected_mask_iou']}/{neural['test_projected_mask_dice']}", NEURAL_SEED)
    add(rows, "reference", "20.77_neural", "test_profile_depth_rmse_m", neural["test_profile_depth_rmse_m"], NEURAL_SEED)

    add(rows, "reference", "20.80_feature_only", "test_total_mae", feature["normalized_param_mae"], FEATURE_ONLY)
    add(rows, "reference", "20.80_feature_only", "test_LWD_mae_mm", f"{feature['L_mae_mm']}/{feature['W_mae_mm']}/{feature['D_mae_mm']}", FEATURE_ONLY)
    add(rows, "reference", "20.80_feature_only", "test_curvature_mae", feature["curvature_mae"], FEATURE_ONLY)
    add(rows, "reference", "20.80_feature_only", "test_wLD_wWD_wLW", f"{feature['wLD_abs_error']}/{feature['wWD_abs_error']}/{feature['wLW_abs_error']}", FEATURE_ONLY)
    add(rows, "reference", "20.80_feature_only", "test_iou_dice", f"{feature['projected_mask_iou']}/{feature['projected_mask_dice']}", FEATURE_ONLY)
    add(rows, "reference", "20.80_feature_only", "test_profile_depth_rmse_m", feature["profile_depth_rmse_m"], FEATURE_ONLY)

    add(rows, "reference", "20.79_failed_refinement", "selected_variant", refined["variant"], REFINED_SEED)
    add(rows, "reference", "20.79_failed_refinement", "test_total_mae", refined["test_normalized_param_mae"], REFINED_SEED)
    add(rows, "reference", "20.79_failed_refinement", "test_LWD_mae_mm", f"{refined['test_L_mae_mm']}/{refined['test_W_mae_mm']}/{refined['test_D_mae_mm']}", REFINED_SEED)
    add(rows, "reference", "20.79_failed_refinement", "test_curvature_mae", refined["test_curvature_mae"], REFINED_SEED)
    add(rows, "reference", "20.79_failed_refinement", "test_wLD_wWD_wLW", f"{refined['test_wLD_abs_error']}/{refined['test_wWD_abs_error']}/{refined['test_wLW_abs_error']}", REFINED_SEED)
    add(rows, "reference", "20.79_failed_refinement", "test_iou_dice", f"{refined['test_projected_mask_iou']}/{refined['test_projected_mask_dice']}", REFINED_SEED)
    add(rows, "reference", "20.79_failed_refinement", "test_profile_depth_rmse_m", refined["test_profile_depth_rmse_m"], REFINED_SEED)

    write_csv(METRICS, rows, FIELDS)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "true_3d_rbc_v3_240 feature-fusion reference summary",
                "",
                f"dataset_id: {dataset.dataset_id}",
                f"status: {manifest.get('status')}",
                f"train_ready_candidate: {manifest.get('train_ready_candidate')}",
                f"baseline_ready: {manifest.get('baseline_ready')}",
                f"input_shape_delta_b: {tuple(dataset.delta_b.shape)}",
                f"input_shape_conv1d: {tuple(dataset.x_channels.shape)}",
                f"split_counts: {observed_split}",
                f"feature_csv: {FEATURES}",
                f"feature_columns: {len(feature_cols)}",
                f"feature_group_counts: {counts}",
                "selected_20_80_feature_set: F0_F1_F2_basic_physical",
                "selected_20_80_feature_model: svr_rbf_C10_eps0.03",
                "",
                f"20.77 neural test total MAE: {neural['test_normalized_param_mae']}",
                f"20.77 neural L/W/D MAE mm: {neural['test_L_mae_mm']}/{neural['test_W_mae_mm']}/{neural['test_D_mae_mm']}",
                f"20.77 neural curvature MAE: {neural['test_curvature_mae']}",
                f"20.77 neural wLD/wWD/wLW: {params['wLD']['physical_mae']}/{params['wWD']['physical_mae']}/{params['wLW']['physical_mae']}",
                f"20.77 neural IoU/Dice: {neural['test_projected_mask_iou']}/{neural['test_projected_mask_dice']}",
                f"20.77 neural profile depth RMSE m: {neural['test_profile_depth_rmse_m']}",
                "",
                f"20.80 feature-only test total MAE: {feature['normalized_param_mae']}",
                f"20.80 feature-only L/W/D MAE mm: {feature['L_mae_mm']}/{feature['W_mae_mm']}/{feature['D_mae_mm']}",
                f"20.80 feature-only curvature MAE: {feature['curvature_mae']}",
                f"20.80 feature-only wLD/wWD/wLW: {feature['wLD_abs_error']}/{feature['wWD_abs_error']}/{feature['wLW_abs_error']}",
                f"20.80 feature-only IoU/Dice: {feature['projected_mask_iou']}/{feature['projected_mask_dice']}",
                f"20.80 feature-only profile depth RMSE m: {feature['profile_depth_rmse_m']}",
                "",
                f"20.79 failed refinement variant: {refined['variant']}",
                f"20.79 failed refinement test total MAE: {refined['test_normalized_param_mae']}",
                f"20.79 failed refinement curvature MAE: {refined['test_curvature_mae']}",
                "",
                "boundary: no COMSOL, no new data, no NPZ modification, no baseline update. Stage 20.81 may only use feature columns with F0__..F5__ prefixes and delta_b tensors from explicit dataset_id loading.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
