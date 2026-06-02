#!/usr/bin/env python
"""Run the 25.7 surface forward-refinement inference runner.

Runtime refinement inputs are limited to observed delta_b-derived features,
the frozen 20.85 baseline prediction, and the exported 25.7 artifact. Labels
are used only for optional metrics on labeled pilot samples.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_surface_shape_extension_current_baseline_inference import (
    baseline_train_bounds,
    load_artifact as load_baseline_artifact,
    predict_params as predict_baseline_params,
)
from audit_surface_shape_extension_rbc_oracle_fit import DATASET_ID, ROOT, load_surface_dataset, pose_for_sample
from build_surface_forward_refinement_target_set import (
    PARAM_NAMES,
    REGISTRY,
    TARGET_MATERIALIZED,
    as_bool,
    as_float,
    read_csv,
    write_csv,
)
from fit_surface_feature_space_forward_surrogate import RidgeSurrogate, observed_feature_matrix
from load_true_3d_rbc_pilot_dataset import depth_map_from_params, projected_mask_from_params
from run_surface_forward_refinement_formal_benchmark import (
    FIXED_LAMBDA,
    FORMAL_FIELDS,
    enrich_with_oracle,
)
from run_surface_rbc_forward_consistency_refinement import (
    build_metric_row,
    clip_params,
    evaluate_params,
    feature_residual,
    metric_mean,
    optimize_params,
)


DEFAULT_MANIFEST = ROOT / "results/manifests/surface_forward_refinement_inference_artifact_manifest.json"
SUMMARY = ROOT / "results/summaries/surface_forward_refinement_inference_runner_summary.txt"
METRICS = ROOT / "results/metrics/surface_forward_refinement_inference_metrics.csv"
BY_SHAPE = ROOT / "results/metrics/surface_forward_refinement_inference_by_shape.csv"
FAILURES = ROOT / "results/metrics/surface_forward_refinement_inference_failure_cases.csv"

INFERENCE_EXTRA_FIELDS = [
    "runner_protocol",
    "artifact_id",
    "input_dataset_id",
    "baseline_prediction_source",
    "eligibility_status",
    "not_suitable_reason",
    "metadata_used_for_eligibility_only",
    "label_or_oracle_used_for_refinement",
    "runtime_refinement_inputs",
    "generated_profile_shape",
    "generated_projected_mask_shape",
    "generated_projected_mask_area_px",
    "baseline_generated_projected_mask_area_px",
]
INFERENCE_FIELDS = [*INFERENCE_EXTRA_FIELDS, *FORMAL_FIELDS]

GROUP_FIELDS = [
    "shape_type",
    "target_role",
    "split",
    "sample_count",
    "refinement_applied_count",
    "baseline_profile_rmse_mean_m",
    "refined_profile_rmse_mean_m",
    "oracle_profile_rmse_mean_m",
    "baseline_Er_like_mean",
    "refined_Er_like_mean",
    "oracle_Er_like_mean",
    "baseline_IoU_mean",
    "refined_IoU_mean",
    "oracle_IoU_mean",
    "baseline_Dice_mean",
    "refined_Dice_mean",
    "oracle_Dice_mean",
    "feature_residual_before_mean",
    "feature_residual_after_mean",
    "profile_rmse_improved_rate",
    "forward_residual_improved_rate",
    "success_credit_allowed",
]

FAILURE_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "target_role",
    "eligibility_status",
    "failure_mode",
    "feature_residual_mse_before",
    "feature_residual_mse_after",
    "baseline_profile_depth_rmse_m",
    "refined_profile_depth_rmse_m",
    "oracle_profile_depth_rmse_m",
    "baseline_projected_mask_Dice",
    "refined_projected_mask_Dice",
    "oracle_projected_mask_Dice",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 25.7 surface forward-refinement inference.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-ids", help="Optional comma-separated sample_ids from the selected dataset.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_refinement_artifact(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], RidgeSurrogate]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"artifact manifest missing; run export script first: {manifest_path}")
    manifest = read_json(manifest_path)
    artifact_path = Path(str(manifest["artifact_path"]))
    if not artifact_path.exists():
        raise FileNotFoundError(f"ignored runtime artifact missing; run export script first: {artifact_path}")
    expected_sha = str(manifest.get("artifact_sha256", ""))
    if expected_sha and sha256_file(artifact_path) != expected_sha:
        raise RuntimeError("surface refinement artifact sha256 mismatch")
    artifact = read_json(artifact_path)
    payload = artifact["surrogate"]
    surrogate = RidgeSurrogate(
        candidate_id=str(payload["candidate_id"]),
        descriptor_kind=str(payload["descriptor_kind"]),
        feature_mode=str(payload["feature_mode"]),
        alpha=float(payload["alpha"]),
        x_mean=np.asarray(payload["x_mean"], dtype=np.float64),
        x_std=np.asarray(payload["x_std"], dtype=np.float64),
        y_mean=np.asarray(payload["y_mean"], dtype=np.float64),
        y_std=np.asarray(payload["y_std"], dtype=np.float64),
        coef=np.asarray(payload["coef"], dtype=np.float64),
        descriptor_names=[str(x) for x in artifact["feature_columns"]["descriptor_names"]],
        model_feature_names=[str(x) for x in artifact["feature_columns"]["model_feature_names"]],
        observed_feature_names=[str(x) for x in artifact["feature_columns"]["observed_feature_names"]],
    )
    return manifest, artifact, surrogate


def selected_indices(dataset: Any, sample_ids_arg: str | None) -> list[int]:
    if not sample_ids_arg:
        return list(range(len(dataset.sample_ids)))
    requested = [item.strip() for item in sample_ids_arg.split(",") if item.strip()]
    index_by_id = {str(sample_id): idx for idx, sample_id in enumerate(dataset.sample_ids)}
    missing = [sample_id for sample_id in requested if sample_id not in index_by_id]
    if missing:
        raise RuntimeError(f"requested sample_id(s) not found in dataset: {missing}")
    return [index_by_id[sample_id] for sample_id in requested]


def load_target_rows() -> dict[str, dict[str, str]]:
    if not TARGET_MATERIALIZED.exists():
        raise FileNotFoundError(f"target materialized CSV missing: {TARGET_MATERIALIZED}")
    return {row["sample_id"]: row for row in read_csv(TARGET_MATERIALIZED)}


def baseline_predictions(dataset: Any) -> tuple[np.ndarray, str]:
    _manifest, checkpoint, model = load_baseline_artifact()
    pred_raw = predict_baseline_params(dataset, checkpoint, model)
    low, high = baseline_train_bounds()
    pred = np.clip(pred_raw, low[None, :], high[None, :]).astype(np.float64)
    return pred, "frozen_20_85_baseline_checkpoint_with_train_bounds_clip"


def metadata_eligibility(dataset: Any, index: int) -> tuple[bool, str]:
    reasons: list[str] = []
    if int(dataset.component_count[index]) > 1:
        reasons.append("component_count_gt_1")
    if str(dataset.topology_type[index]) == "multi_component":
        reasons.append("multi_component_topology")
    if str(dataset.representation_target[index]) == "component_set":
        reasons.append("component_set_representation_target")
    if str(dataset.shape_type[index]) == "multi_pit_two_component_surface_defect":
        reasons.append("multi_pit_shape_type")
    if reasons:
        return False, "|".join(reasons)
    return True, ""


def updated_materialized_row(base: dict[str, str], baseline_params: np.ndarray, baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = dict(base)
    row.update(
        {
            "baseline_profile_depth_rmse_m": float(baseline_metrics["profile_depth_rmse_m"]),
            "baseline_Er_like_error": float(baseline_metrics["Er_like_error"]),
            "baseline_projected_mask_IoU": float(baseline_metrics["projected_mask_IoU"]),
            "baseline_projected_mask_Dice": float(baseline_metrics["projected_mask_Dice"]),
            "baseline_area_error": float(baseline_metrics["area_error"]),
            "baseline_pred_component_count": int(baseline_metrics["component_count"]),
        }
    )
    for i, name in enumerate(PARAM_NAMES):
        row[f"pred_{name}"] = float(baseline_params[i])
    return row


def generated_shapes_and_area(dataset: Any, index: int, params: np.ndarray) -> tuple[str, str, int]:
    pose = pose_for_sample(dataset, index)
    profile = depth_map_from_params(params, pose)
    mask = projected_mask_from_params(params, pose)
    return str(tuple(int(x) for x in profile.shape)), str(tuple(int(x) for x in mask.shape)), int(mask.sum())


def run_inference(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, artifact, surrogate = load_refinement_artifact(args.artifact_manifest)
    if args.dataset_id != artifact["dataset_id"]:
        raise RuntimeError(f"artifact dataset_id mismatch: {artifact['dataset_id']} != {args.dataset_id}")
    dataset = load_surface_dataset(args.dataset_id, REGISTRY)
    observed_features, observed_names = observed_feature_matrix(dataset.delta_b)
    if observed_names != surrogate.observed_feature_names:
        raise RuntimeError("observed feature column order mismatch with exported artifact")
    baseline_params_all, baseline_source = baseline_predictions(dataset)
    target_by_id = load_target_rows()
    indices = selected_indices(dataset, args.sample_ids)

    metric_rows: list[dict[str, Any]] = []
    for idx in indices:
        sample_id = str(dataset.sample_ids[idx])
        if sample_id not in target_by_id:
            raise RuntimeError(f"target/evaluation metadata row missing for sample_id={sample_id}")
        base_row = target_by_id[sample_id]
        init_params = np.asarray(baseline_params_all[idx], dtype=np.float64)
        metadata_suitable, not_suitable_reason = metadata_eligibility(dataset, idx)
        baseline_metrics = evaluate_params(dataset, idx, init_params)
        row_for_metrics = updated_materialized_row(base_row, init_params, baseline_metrics)

        if metadata_suitable:
            refined_params, success, iters, message = optimize_params(surrogate, observed_features[idx], init_params, FIXED_LAMBDA)
            eligibility_status = "rbc_refinement_applied"
            refinement_applied = True
        else:
            refined_params, success, iters, message = clip_params(init_params), False, 0, "not_suitable_for_rbc_refinement"
            eligibility_status = "not_suitable_for_rbc_refinement"
            refinement_applied = False

        refined_metrics = evaluate_params(dataset, idx, refined_params)
        before = feature_residual(surrogate, init_params, observed_features[idx])
        after = feature_residual(surrogate, refined_params, observed_features[idx])
        meta = {
            "optimizer_success": success,
            "optimizer_iterations": iters,
            "optimizer_message": message,
            "feature_residual_mse_before": before,
            "feature_residual_mse_after": after,
        }
        base_metric_row = build_metric_row(
            row_for_metrics,
            refined_params,
            meta,
            refined_metrics,
            FIXED_LAMBDA,
            surrogate.candidate_id,
            refinement_applied,
        )
        enriched = enrich_with_oracle(base_metric_row, row_for_metrics)
        generated_profile_shape, generated_mask_shape, generated_mask_area = generated_shapes_and_area(dataset, idx, refined_params)
        _baseline_profile_shape, _baseline_mask_shape, baseline_mask_area = generated_shapes_and_area(dataset, idx, init_params)
        enriched.update(
            {
                "runner_protocol": "25.7_surface_forward_refinement_inference_runner",
                "artifact_id": artifact["artifact_id"],
                "input_dataset_id": args.dataset_id,
                "baseline_prediction_source": baseline_source,
                "eligibility_status": eligibility_status,
                "not_suitable_reason": not_suitable_reason,
                "metadata_used_for_eligibility_only": True,
                "label_or_oracle_used_for_refinement": False,
                "runtime_refinement_inputs": "delta_b_features+frozen_20_85_predicted_six_params+exported_artifact",
                "generated_profile_shape": generated_profile_shape,
                "generated_projected_mask_shape": generated_mask_shape,
                "generated_projected_mask_area_px": generated_mask_area,
                "baseline_generated_projected_mask_area_px": baseline_mask_area,
            }
        )
        metric_rows.append(enriched)
    context = {
        "manifest": manifest,
        "artifact": artifact,
        "surrogate": surrogate,
        "dataset": dataset,
        "indices": indices,
        "baseline_source": baseline_source,
    }
    return metric_rows, context


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shape in sorted({str(row["shape_type"]) for row in rows} | {"all"}):
        for role in sorted({str(row["target_role"]) for row in rows} | {"all"}):
            for split in ["all", "train", "val", "test"]:
                subset = [
                    row
                    for row in rows
                    if (shape == "all" or row["shape_type"] == shape)
                    and (role == "all" or row["target_role"] == role)
                    and (split == "all" or row["split"] == split)
                ]
                if not subset:
                    continue
                out.append(
                    {
                        "shape_type": shape,
                        "target_role": role,
                        "split": split,
                        "sample_count": len(subset),
                        "refinement_applied_count": sum(bool(row["refinement_applied"]) for row in subset),
                        "baseline_profile_rmse_mean_m": metric_mean(subset, "baseline_profile_depth_rmse_m"),
                        "refined_profile_rmse_mean_m": metric_mean(subset, "refined_profile_depth_rmse_m"),
                        "oracle_profile_rmse_mean_m": metric_mean(subset, "oracle_profile_depth_rmse_m"),
                        "baseline_Er_like_mean": metric_mean(subset, "baseline_Er_like_error"),
                        "refined_Er_like_mean": metric_mean(subset, "refined_Er_like_error"),
                        "oracle_Er_like_mean": metric_mean(subset, "oracle_Er_like_error"),
                        "baseline_IoU_mean": metric_mean(subset, "baseline_projected_mask_IoU"),
                        "refined_IoU_mean": metric_mean(subset, "refined_projected_mask_IoU"),
                        "oracle_IoU_mean": metric_mean(subset, "oracle_projected_mask_IoU"),
                        "baseline_Dice_mean": metric_mean(subset, "baseline_projected_mask_Dice"),
                        "refined_Dice_mean": metric_mean(subset, "refined_projected_mask_Dice"),
                        "oracle_Dice_mean": metric_mean(subset, "oracle_projected_mask_Dice"),
                        "feature_residual_before_mean": metric_mean(subset, "feature_residual_mse_before"),
                        "feature_residual_after_mean": metric_mean(subset, "feature_residual_mse_after"),
                        "profile_rmse_improved_rate": float(np.mean([bool(row["profile_rmse_improved"]) for row in subset])),
                        "forward_residual_improved_rate": float(np.mean([bool(row["feature_residual_improved"]) for row in subset])),
                        "success_credit_allowed": bool(subset) and all(bool(row["include_in_success_gate"]) for row in subset),
                    }
                )
    return out


def failure_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        modes: list[str] = []
        if row["target_role"] == "refinement_target":
            if not bool(row["profile_rmse_improved"]):
                modes.append("profile_rmse_not_improved")
            if not bool(row["Dice_improved"]):
                modes.append("dice_not_improved")
            if bool(row["feature_residual_improved"]) and not bool(row["profile_rmse_improved"]):
                modes.append("forward_residual_improved_without_profile_improvement")
        if row["target_role"] == "already_pass_reference" and float(row["profile_rmse_delta_m"]) > 0.0:
            modes.append("already_pass_reference_profile_degraded_monitoring")
        if row["eligibility_status"] == "not_suitable_for_rbc_refinement" and row["target_role"] != "excluded_negative_control":
            modes.append("unexpected_non_negative_control_exclusion")
        if not modes:
            continue
        out.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "shape_type": row["shape_type"],
                "target_role": row["target_role"],
                "eligibility_status": row["eligibility_status"],
                "failure_mode": "|".join(modes),
                "feature_residual_mse_before": row["feature_residual_mse_before"],
                "feature_residual_mse_after": row["feature_residual_mse_after"],
                "baseline_profile_depth_rmse_m": row["baseline_profile_depth_rmse_m"],
                "refined_profile_depth_rmse_m": row["refined_profile_depth_rmse_m"],
                "oracle_profile_depth_rmse_m": row["oracle_profile_depth_rmse_m"],
                "baseline_projected_mask_Dice": row["baseline_projected_mask_Dice"],
                "refined_projected_mask_Dice": row["refined_projected_mask_Dice"],
                "oracle_projected_mask_Dice": row["oracle_projected_mask_Dice"],
                "notes": "runner failure/monitoring row only; no baseline transition",
            }
        )
    out.sort(key=lambda item: float(item["refined_profile_depth_rmse_m"]) - float(item["baseline_profile_depth_rmse_m"]), reverse=True)
    return out


def write_summary(rows: list[dict[str, Any]], context: dict[str, Any]) -> None:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    rbc_like = [row for row in rows if row["shape_type"] == "rbc_like_smooth_pit"]
    negative = [row for row in rows if row["target_role"] == "excluded_negative_control"]
    pass_refs = [row for row in rows if row["target_role"] == "already_pass_reference"]
    role_counts = Counter(row["target_role"] for row in rows)
    lines = [
        "25.7 surface forward-refinement inference runner",
        "",
        f"dataset_id: {context['artifact']['dataset_id']}",
        f"artifact_id: {context['artifact']['artifact_id']}",
        f"artifact_manifest: {DEFAULT_MANIFEST}",
        f"selected_surrogate: {context['surrogate'].candidate_id}",
        f"lambda_param: {FIXED_LAMBDA:g}",
        f"baseline_prediction_source: {context['baseline_source']}",
        "runtime_refinement_inputs: observed delta_b-derived features + frozen 20.85 predicted six params + exported artifact",
        "label_or_oracle_used_for_refinement: false",
        "labels_used_for_metric_only: true for this labeled pilot report",
        "",
        f"sample_count: {len(rows)}",
        f"target_role_counts: {dict(role_counts)}",
        f"refinement_target_count: {len(targets)}",
        f"already_pass_reference_count: {len(pass_refs)}",
        f"excluded_negative_control_count: {len(negative)}",
        f"refinement_applied_count: {sum(bool(row['refinement_applied']) for row in rows)}",
        f"not_suitable_for_rbc_refinement_count: {sum(row['eligibility_status'] == 'not_suitable_for_rbc_refinement' for row in rows)}",
        "",
        f"target_baseline_profile_rmse_mean_m: {metric_mean(targets, 'baseline_profile_depth_rmse_m'):.12g}",
        f"target_refined_profile_rmse_mean_m: {metric_mean(targets, 'refined_profile_depth_rmse_m'):.12g}",
        f"target_oracle_profile_rmse_mean_m: {metric_mean(targets, 'oracle_profile_depth_rmse_m'):.12g}",
        f"target_baseline_Er_like_mean: {metric_mean(targets, 'baseline_Er_like_error'):.12g}",
        f"target_refined_Er_like_mean: {metric_mean(targets, 'refined_Er_like_error'):.12g}",
        f"target_baseline_IoU_mean: {metric_mean(targets, 'baseline_projected_mask_IoU'):.12g}",
        f"target_refined_IoU_mean: {metric_mean(targets, 'refined_projected_mask_IoU'):.12g}",
        f"target_baseline_Dice_mean: {metric_mean(targets, 'baseline_projected_mask_Dice'):.12g}",
        f"target_refined_Dice_mean: {metric_mean(targets, 'refined_projected_mask_Dice'):.12g}",
        f"target_feature_residual_before_mean: {metric_mean(targets, 'feature_residual_mse_before'):.12g}",
        f"target_feature_residual_after_mean: {metric_mean(targets, 'feature_residual_mse_after'):.12g}",
        "",
        f"rbc_like_control_baseline_rmse_mean_m: {metric_mean(rbc_like, 'baseline_profile_depth_rmse_m'):.12g}",
        f"rbc_like_control_refined_rmse_mean_m: {metric_mean(rbc_like, 'refined_profile_depth_rmse_m'):.12g}",
        f"rbc_like_control_baseline_Dice_mean: {metric_mean(rbc_like, 'baseline_projected_mask_Dice'):.12g}",
        f"rbc_like_control_refined_Dice_mean: {metric_mean(rbc_like, 'refined_projected_mask_Dice'):.12g}",
        "multi_pit_handling: metadata marks multi-pit/component-set rows as not_suitable_for_rbc_refinement; no RBC success credit.",
        "",
        f"metrics_csv: {METRICS}",
        f"by_shape_csv: {BY_SHAPE}",
        f"failure_cases_csv: {FAILURES}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows, context = run_inference(args)
    groups = group_summary(rows)
    failures = failure_cases(rows)
    write_csv(METRICS, rows, INFERENCE_FIELDS)
    write_csv(BY_SHAPE, groups, GROUP_FIELDS)
    write_csv(FAILURES, failures, FAILURE_FIELDS)
    write_summary(rows, context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
