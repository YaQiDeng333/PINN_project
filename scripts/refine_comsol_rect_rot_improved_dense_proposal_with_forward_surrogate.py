from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import refine_comsol_rect_rot_dense_initializer_with_forward_surrogate as dense_refine  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NPZ = dense_refine.DEFAULT_NPZ
DEFAULT_LABELS = dense_refine.DEFAULT_LABELS
DEFAULT_FEATURES = dense_refine.DEFAULT_FEATURES
DEFAULT_DENSE_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_proposal_extraction_selected_geometry.csv"
DEFAULT_INITIAL_GEOMETRY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_proposal_extraction_selected_geometry.csv"

DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_improved_dense_priewald_refinement_summary.txt"
DEFAULT_AUDIT = PROJECT_ROOT / "results/summaries/comsol_rect_rot_improved_dense_priewald_refinement_failure_audit_summary.txt"
DEFAULT_CONFIG_SWEEP = PROJECT_ROOT / "results/metrics/comsol_rect_rot_improved_dense_priewald_refinement_config_sweep.csv"
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_rect_rot_improved_dense_priewald_refinement_metrics.csv"
DEFAULT_GROUP = PROJECT_ROOT / "results/metrics/comsol_rect_rot_improved_dense_priewald_refinement_group_summary.csv"
DEFAULT_GEOMETRY = PROJECT_ROOT / "results/metrics/comsol_rect_rot_improved_dense_priewald_refinement_geometry_summary.csv"
DEFAULT_FORWARD = PROJECT_ROOT / "results/metrics/comsol_rect_rot_improved_dense_priewald_refinement_forward_summary.csv"
DEFAULT_FAILURE = PROJECT_ROOT / "results/metrics/comsol_rect_rot_improved_dense_priewald_refinement_failure_cases.csv"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "results/previews/comsol_rect_rot_improved_dense_priewald_refinement"

TEMP_INPUT_SUMMARY = PROJECT_ROOT / "results/summaries/_tmp_20_54_improved_dense_refinement_input_check_summary.txt"
TEMP_INPUT_CHECK = PROJECT_ROOT / "results/metrics/_tmp_20_54_improved_dense_refinement_input_check.csv"

REF_2053_DENSE_IOU = 0.5664
REF_2053_DENSE_DICE = 0.7179
REF_2053_EXTRACTED_IOU = 0.5652
REF_2053_EXTRACTED_DICE = 0.7169
REF_2053_EXTRACTED_AREA = 0.3804
REF_2053_REFINED_IOU = 0.5810
REF_2053_REFINED_DICE = 0.7300
REF_2053_FORWARD_PRE = 0.4869
REF_2053_FORWARD_POST = 0.3641
DENSE_SINGLE_BASELINE_IOU = dense_refine.DENSE_SINGLE_BASELINE_IOU
DENSE_SINGLE_BASELINE_DICE = dense_refine.DENSE_SINGLE_BASELINE_DICE


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [as_float(row, key) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(sum(values) / len(values)) if values else math.nan


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def split_stats(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    subset = split_rows(rows, split)
    return {
        "count": float(len(subset)),
        "dense_iou": mean(subset, "dense_iou"),
        "dense_dice": mean(subset, "dense_dice"),
        "dense_area_error": mean(subset, "dense_area_error"),
        "initial_iou": mean(subset, "initial_iou"),
        "initial_dice": mean(subset, "initial_dice"),
        "initial_area_error": mean(subset, "initial_area_error"),
        "refined_iou": mean(subset, "refined_iou"),
        "refined_dice": mean(subset, "refined_dice"),
        "refined_area_error": mean(subset, "refined_area_error"),
        "delta_iou": mean(subset, "delta_iou"),
        "delta_dice": mean(subset, "delta_dice"),
        "delta_area_error": mean(subset, "delta_area_error"),
        "initial_forward_nrmse": mean(subset, "initial_forward_nrmse"),
        "refined_forward_nrmse": mean(subset, "refined_forward_nrmse"),
        "forward_nrmse_reduction": mean(subset, "forward_nrmse_reduction"),
        "initial_angle_mae": mean([row for row in subset if row["defect_type"] == "rotated_rect"], "initial_angle_abs_error_deg"),
        "refined_angle_mae": mean([row for row in subset if row["defect_type"] == "rotated_rect"], "refined_angle_abs_error_deg"),
        "angle_error_delta": mean([row for row in subset if row["defect_type"] == "rotated_rect"], "angle_error_delta"),
        "parameter_drift_norm": mean(subset, "parameter_drift_norm"),
    }


def selected_config_row(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    return sorted(
        rows,
        key=lambda row: (
            as_float(row, "val_refinement_score"),
            as_float(row, "delta_mask_iou"),
            as_float(row, "delta_mask_dice"),
            as_float(row, "forward_nrmse_reduction"),
        ),
        reverse=True,
    )[0]


def write_stage_2054_summary(args: argparse.Namespace, result: dict[str, Any]) -> None:
    rows = read_csv(args.metrics)
    config = selected_config_row(args.config_sweep)
    stats = {split: split_stats(rows, split) for split in ["train", "val", "test"]}
    test = stats["test"]
    strong_dense_ok = test["dense_iou"] > REF_2053_DENSE_IOU and test["dense_dice"] > REF_2053_DENSE_DICE
    extraction_ok = test["initial_iou"] >= REF_2053_EXTRACTED_IOU + 0.02 or test["initial_dice"] >= REF_2053_EXTRACTED_DICE + 0.015
    refine_ok = test["delta_iou"] >= 0.01 or test["delta_dice"] >= 0.008
    forward_ok = stats["val"]["forward_nrmse_reduction"] > 0 and test["forward_nrmse_reduction"] > 0
    area_ok = test["delta_area_error"] <= 0.03
    promising = bool(strong_dense_ok and extraction_ok and refine_ok and forward_ok and area_ok)
    surrogate_mismatch = bool(test["forward_nrmse_reduction"] > 0.02 and (test["delta_iou"] < -0.005 or test["delta_dice"] < -0.004))
    if not strong_dense_ok:
        recommendation = "E. Pause geometry route until a stronger dense initializer is available."
    elif not extraction_ok:
        recommendation = "B. Improve dense-to-geometry proposal extraction."
    elif not refine_ok and surrogate_mismatch:
        recommendation = "A. Improve forward surrogate."
    elif not refine_ok:
        recommendation = "C. Do mask/profile basis refinement."
    else:
        recommendation = "A. Improve forward surrogate before any larger refinement run."

    lines = [
        "COMSOL rect/rot improved dense proposal + Priewald-style refinement retry summary",
        "",
        f"Input NPZ: {args.npz}",
        f"Improved proposal source: {args.initial_geometry}",
        "No COMSOL run; no new data; no baseline update.",
        "Dense initializer is only a proposal generator, not a new baseline.",
        "Refinement uses observed normalized delta_bz and a frozen lightweight forward surrogate.",
        "True mask / true geometry are not used in optimization; they are metrics and validation-selection references only.",
        "",
        "Forward surrogate:",
        "- source: retrained in-memory through the reused 20.53 refinement implementation; no checkpoint written",
        "- protocol: train split fitting, validation checkpoint, test final only",
        "",
        "Selected validation refinement config:",
        f"- config = {config['config_name']}",
        f"- steps/lr/lambda_prior = {config['steps']} / {config['lr']} / {config['lambda_prior']}",
        f"- val_refinement_score = {as_float(config, 'val_refinement_score'):.6f}",
        f"- val delta geometry-raster IoU/Dice = {as_float(config, 'delta_mask_iou'):.6f} / {as_float(config, 'delta_mask_dice'):.6f}",
        f"- val forward NRMSE reduction = {as_float(config, 'forward_nrmse_reduction'):.6f}",
        "",
        "Dense mask vs extracted geometry vs refined geometry:",
    ]
    for split in ["train", "val", "test"]:
        s = stats[split]
        lines.extend(
            [
                f"- {split} dense mask IoU/Dice/area = {s['dense_iou']:.4f} / {s['dense_dice']:.4f} / {s['dense_area_error']:.4f}",
                f"- {split} extracted geometry IoU/Dice/area = {s['initial_iou']:.4f} / {s['initial_dice']:.4f} / {s['initial_area_error']:.4f}",
                f"- {split} refined geometry IoU/Dice/area = {s['refined_iou']:.4f} / {s['refined_dice']:.4f} / {s['refined_area_error']:.4f}",
                f"- {split} forward NRMSE = {s['initial_forward_nrmse']:.4f} -> {s['refined_forward_nrmse']:.4f} (reduction {s['forward_nrmse_reduction']:.4f})",
                f"- {split} angle MAE = {s['initial_angle_mae']:.4f} -> {s['refined_angle_mae']:.4f} (delta {s['angle_error_delta']:.4f})",
                f"- {split} parameter drift norm = {s['parameter_drift_norm']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "Reference comparison:",
            f"- 20.53 dense initializer test IoU/Dice = {REF_2053_DENSE_IOU:.4f} / {REF_2053_DENSE_DICE:.4f}",
            f"- 20.53 extracted geometry test IoU/Dice/area = {REF_2053_EXTRACTED_IOU:.4f} / {REF_2053_EXTRACTED_DICE:.4f} / {REF_2053_EXTRACTED_AREA:.4f}",
            f"- 20.53 refined geometry test IoU/Dice = {REF_2053_REFINED_IOU:.4f} / {REF_2053_REFINED_DICE:.4f}",
            f"- 20.53 forward NRMSE test = {REF_2053_FORWARD_PRE:.4f} -> {REF_2053_FORWARD_POST:.4f}",
            f"- dense single-defect baseline test IoU/Dice = {DENSE_SINGLE_BASELINE_IOU:.4f} / {DENSE_SINGLE_BASELINE_DICE:.4f}",
            "",
            "Acceptance check:",
            f"- strong dense initializer improves over 20.53: {strong_dense_ok}",
            f"- extracted geometry proposal improves over 20.53 by required margin: {extraction_ok}",
            f"- post-refinement improves over extracted proposal by required margin: {refine_ok}",
            f"- forward NRMSE decreases on val and test: {forward_ok}",
            f"- area_error not materially worse: {area_ok}",
            f"- surrogate mismatch risk: {surrogate_mismatch}",
            f"- 20.54 promising: {promising}",
            f"- next recommendation: {recommendation}",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    categories: dict[str, int] = defaultdict(int)
    for row in split_rows(rows, "test"):
        categories[row["refinement_category"]] += 1
    audit_lines = [
        "COMSOL rect/rot improved dense Priewald refinement failure audit summary",
        "",
        f"Selected config: {config['config_name']}",
        f"Test refinement categories: {dict(sorted(categories.items()))}",
        f"Surrogate mismatch risk: {surrogate_mismatch}",
        f"20.54 promising: {promising}",
        f"Next recommendation: {recommendation}",
        "",
        "Interpretation:",
        "- Dense initializer and extracted geometry proposal are much stronger than 20.53.",
        "- The refinement step is judged separately from proposal quality.",
        "- If forward residual improves but raster metrics fail to improve, the blocker is surrogate/refinement mismatch rather than initializer quality.",
    ]
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def cleanup_temp_files() -> None:
    for path in [TEMP_INPUT_SUMMARY, TEMP_INPUT_CHECK]:
        if path.exists():
            path.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    inner_args = argparse.Namespace(
        npz=args.npz,
        labels=args.labels,
        features=args.features,
        dense_metrics=args.dense_metrics,
        initial_geometry=args.initial_geometry,
        input_summary=TEMP_INPUT_SUMMARY,
        input_check=TEMP_INPUT_CHECK,
        summary=args.summary,
        audit=args.audit,
        config_sweep=args.config_sweep,
        metrics=args.metrics,
        group_summary=args.group_summary,
        geometry_summary=args.geometry_summary,
        forward_summary=args.forward_summary,
        failure_cases=args.failure_cases,
        preview_dir=args.preview_dir,
        seed=args.seed,
        forward_epochs=args.forward_epochs,
        forward_batch_size=args.forward_batch_size,
        forward_lr=args.forward_lr,
        cpu=args.cpu,
    )
    try:
        result = dense_refine.run(inner_args)
        write_stage_2054_summary(args, result)
        return result
    finally:
        cleanup_temp_files()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--dense-metrics", type=Path, default=DEFAULT_DENSE_METRICS)
    parser.add_argument("--initial-geometry", type=Path, default=DEFAULT_INITIAL_GEOMETRY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--config-sweep", type=Path, default=DEFAULT_CONFIG_SWEEP)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUP)
    parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--forward-summary", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--failure-cases", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward-epochs", type=int, default=300)
    parser.add_argument("--forward-batch-size", type=int, default=32)
    parser.add_argument("--forward-lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
