#!/usr/bin/env python
"""Design the 25.4 surface forward-refinement target set.

This script is plan-only. It reads 25.3 audit CSV/summaries and writes
summary/metrics artifacts. It does not train, run COMSOL, load or write NPZ
data, save checkpoints, or update CURRENT_BASELINE.md.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS_MATRIX = ROOT / "results/metrics/surface_shape_extension_oracle_vs_baseline_matrix.csv"
FAILURE_BY_SHAPE = ROOT / "results/metrics/surface_shape_extension_failure_mode_by_shape.csv"
ORACLE_SUMMARY = ROOT / "results/summaries/surface_shape_extension_rbc_oracle_fit_summary.txt"
BASELINE_SUMMARY = ROOT / "results/summaries/surface_shape_extension_current_baseline_inference_summary.txt"
DIAGNOSIS_SUMMARY = ROOT / "results/summaries/surface_shape_extension_oracle_vs_baseline_diagnosis_summary.txt"
PILOT_MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
BASELINE_ARTIFACT = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
PROFILE_GENERATOR = ROOT / "scripts/load_true_3d_rbc_pilot_dataset.py"
NLS_LITE_SUMMARY = ROOT / "results/summaries/surface_rbc_nls_lite_feature_summary.txt"
FEATURE_BASELINE_SUMMARY = ROOT / "results/summaries/surface_rbc_piao_style_feature_baseline_summary.txt"

PREFLIGHT_SUMMARY = ROOT / "results/summaries/surface_forward_consistency_refinement_preflight_summary.txt"
TARGET_SUMMARY = ROOT / "results/summaries/surface_forward_refinement_target_set_summary.txt"
TARGET_CSV = ROOT / "results/metrics/surface_forward_refinement_target_set.csv"

MULTI_PIT_SHAPE = "multi_pit_two_component_surface_defect"

TARGET_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "diagnosis",
    "target_role",
    "suitable_for_six_param_refinement",
    "include_in_success_gate",
    "include_in_rbc_control_gate",
    "include_as_negative_control",
    "exclude_reason",
    "failure_reason",
    "oracle_profile_depth_rmse_m",
    "oracle_projected_mask_Dice",
    "baseline_profile_depth_rmse_m",
    "baseline_projected_mask_Dice",
    "baseline_minus_oracle_rmse_m",
    "true_component_count",
    "pred_component_count",
    "component_recall_proxy",
    "merge_component_proxy",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def b(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def classify(row: dict[str, str]) -> tuple[str, bool, bool, bool, bool, str]:
    diagnosis = row["diagnosis"]
    shape_type = row["shape_type"]
    representation_target = row["representation_target"]
    if shape_type == MULTI_PIT_SHAPE or representation_target == "component_set":
        return (
            "excluded_negative_control",
            False,
            False,
            False,
            True,
            "multi-pit/component-set is not a six-parameter RBC refinement success target",
        )
    if diagnosis == "rbc_representable_but_model_fail":
        return ("refinement_target", True, True, shape_type == "rbc_like_smooth_pit", False, "")
    if diagnosis == "rbc_representable_and_model_pass":
        return ("already_pass_reference", True, False, shape_type == "rbc_like_smooth_pit", False, "")
    if diagnosis == "rbc_not_representable":
        return ("excluded_negative_control", False, False, False, True, "RBC oracle representation failure")
    return ("excluded_label_or_geometry_issue", False, False, False, True, "label_or_geometry_issue")


def build_target_rows(source_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        role, suitable, success, rbc_control, negative, exclude_reason = classify(row)
        baseline_rmse = f(row, "baseline_profile_depth_rmse_m")
        oracle_rmse = f(row, "oracle_profile_depth_rmse_m")
        rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "shape_type": row["shape_type"],
                "topology_type": row["topology_type"],
                "representation_target": row["representation_target"],
                "diagnosis": row["diagnosis"],
                "target_role": role,
                "suitable_for_six_param_refinement": suitable,
                "include_in_success_gate": success,
                "include_in_rbc_control_gate": rbc_control,
                "include_as_negative_control": negative,
                "exclude_reason": exclude_reason,
                "failure_reason": row.get("primary_reason", ""),
                "oracle_profile_depth_rmse_m": oracle_rmse,
                "oracle_projected_mask_Dice": f(row, "oracle_projected_mask_Dice"),
                "baseline_profile_depth_rmse_m": baseline_rmse,
                "baseline_projected_mask_Dice": f(row, "baseline_projected_mask_Dice"),
                "baseline_minus_oracle_rmse_m": baseline_rmse - oracle_rmse,
                "true_component_count": row.get("true_component_count", ""),
                "pred_component_count": row.get("pred_component_count", ""),
                "component_recall_proxy": row.get("component_recall_proxy", ""),
                "merge_component_proxy": row.get("merge_component_proxy", ""),
            }
        )
    return rows


def counter(rows: list[dict[str, Any]], key: str, where_key: str | None = None, where_value: Any | None = None) -> dict[str, int]:
    selected = rows if where_key is None else [row for row in rows if row.get(where_key) == where_value]
    return dict(Counter(str(row[key]) for row in selected))


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return sum(values) / len(values) if values else float("nan")


def write_preflight() -> None:
    required = [
        DIAGNOSIS_MATRIX,
        FAILURE_BY_SHAPE,
        ORACLE_SUMMARY,
        BASELINE_SUMMARY,
        DIAGNOSIS_SUMMARY,
        PILOT_MANIFEST,
        BASELINE_ARTIFACT,
        REGISTRY,
        PROFILE_GENERATOR,
        NLS_LITE_SUMMARY,
        FEATURE_BASELINE_SUMMARY,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing 25.4 preflight input(s): " + ", ".join(str(path) for path in missing))
    pilot = read_json(PILOT_MANIFEST)
    baseline = read_json(BASELINE_ARTIFACT)
    forbidden_diff = git_value(
        ["diff", "--name-only", "--", "CURRENT_BASELINE.md", "data", "checkpoints", "notes", "results/previews", "scripts/visualize_current_baseline.py"]
    )
    lines = [
        "25.4 surface forward-consistency refinement preflight",
        "",
        "scope: plan only; no COMSOL, no training, no data/NPZ generation or mutation, no CURRENT_BASELINE.md update.",
        f"dataset_id: {pilot.get('dataset_id')}",
        f"dataset_status: {pilot.get('status')}",
        f"shape_extension_sample_count: {pilot.get('n_samples')}",
        f"baseline_artifact_id: {baseline.get('artifact_id')}",
        f"baseline_model_family: {baseline.get('model_family')}",
        "25.3_evidence: oracle fit, frozen baseline inference, and oracle-vs-baseline diagnosis summaries are present.",
        "nls_context: NLS-lite and Piao-style feature baseline summaries are present as feature-space references only.",
        "profile_generator_context: RBC depth/profile helpers are available through load_true_3d_rbc_pilot_dataset.py.",
        f"forbidden_diff_present: {bool(forbidden_diff)}",
        "forbidden_diff_paths: " + (forbidden_diff if forbidden_diff else "none"),
        "",
        "guardrails:",
        "- Do not train.",
        "- Do not run COMSOL.",
        "- Do not create or modify data/ or NPZ.",
        "- Do not update CURRENT_BASELINE.md.",
        "- Do not stage checkpoints, previews, notes, baseline docs, or scripts/visualize_current_baseline.py.",
        "- Do not use git add . and do not push.",
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows: list[dict[str, Any]]) -> None:
    targets = [row for row in rows if row["target_role"] == "refinement_target"]
    negative = [row for row in rows if row["target_role"] == "excluded_negative_control"]
    pass_refs = [row for row in rows if row["target_role"] == "already_pass_reference"]
    lines = [
        "25.4 surface forward-refinement target set",
        "",
        "source: results/metrics/surface_shape_extension_oracle_vs_baseline_matrix.csv",
        f"sample_count: {len(rows)}",
        f"refinement_target_count: {len(targets)}",
        f"excluded_negative_control_count: {len(negative)}",
        f"already_pass_reference_count: {len(pass_refs)}",
        "",
        "target_role_rules:",
        "- refinement_target: diagnosis=rbc_representable_but_model_fail and not component_set/multi-pit.",
        "- excluded_negative_control: rbc_not_representable or multi-pit/component-set; excluded from RBC success gates.",
        "- already_pass_reference: rbc_representable_and_model_pass; used as non-collapse reference.",
        "",
        f"refinement_targets_by_shape: {counter(rows, 'shape_type', 'target_role', 'refinement_target')}",
        f"refinement_targets_by_split: {counter(rows, 'split', 'target_role', 'refinement_target')}",
        f"refinement_targets_by_representation_target: {counter(rows, 'representation_target', 'target_role', 'refinement_target')}",
        f"negative_controls_by_shape: {counter(rows, 'shape_type', 'target_role', 'excluded_negative_control')}",
        "",
        f"target_baseline_profile_rmse_mean_m: {mean(targets, 'baseline_profile_depth_rmse_m'):.12g}",
        f"target_oracle_profile_rmse_mean_m: {mean(targets, 'oracle_profile_depth_rmse_m'):.12g}",
        f"target_baseline_dice_mean: {mean(targets, 'baseline_projected_mask_Dice'):.12g}",
        f"target_oracle_dice_mean: {mean(targets, 'oracle_projected_mask_Dice'):.12g}",
        "",
        "success_gate_policy:",
        "- Multi-pit/component-set samples are recorded as negative controls only.",
        "- RBC-like control failures are included in the refinement target and also flagged for control-gate tracking.",
        "- The 25.5 diagnostic must report all roles separately.",
        f"target_csv: {TARGET_CSV}",
    ]
    TARGET_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    TARGET_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    write_preflight()
    source_rows = read_csv(DIAGNOSIS_MATRIX)
    rows = build_target_rows(source_rows)
    write_csv(TARGET_CSV, rows, TARGET_FIELDS)
    write_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
