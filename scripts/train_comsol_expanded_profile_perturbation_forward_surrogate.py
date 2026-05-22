#!/usr/bin/env python
"""Train 20.61 expanded profile perturbation forward surrogates.

The implementation reuses the 20.60 profile-perturbation training code, but
keeps the expanded-pack inputs, candidate names, summaries, and stricter
20.61 residual-ordering gates separate. It does not run refinement and does
not write checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_comsol_profile_perturbation_forward_surrogate as base  # noqa: E402


DEFAULT_PACK = (
    PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_rect_rot_expanded_profile_perturbation_forward_pack_v1.npz"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_expanded_profile_perturbation_forward_surrogate_summary.txt"
DEFAULT_AUDIT_SUMMARY = (
    PROJECT_ROOT / "results/summaries/comsol_expanded_profile_perturbation_residual_objective_audit_summary.txt"
)
DEFAULT_CANDIDATES = (
    PROJECT_ROOT / "results/metrics/comsol_expanded_profile_perturbation_forward_surrogate_candidates.csv"
)
DEFAULT_METRICS = PROJECT_ROOT / "results/metrics/comsol_expanded_profile_perturbation_forward_surrogate_metrics.csv"
DEFAULT_EPOCH_LOG = (
    PROJECT_ROOT / "results/metrics/comsol_expanded_profile_perturbation_forward_surrogate_epoch_log.csv"
)
DEFAULT_ORDERING = (
    PROJECT_ROOT / "results/metrics/comsol_expanded_profile_perturbation_forward_surrogate_ordering_audit.csv"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT / "results/metrics/comsol_expanded_profile_perturbation_residual_objective_audit.csv"
)

OLD_2060_VAL_ORDERING = 0.6607
OLD_2060_TEST_ORDERING = 0.2143
OLD_2060_VAL_MISMATCH = 0.3393
OLD_2060_TEST_MISMATCH = 0.7857
OLD_2060_VAL_CORR = 0.5703
OLD_2060_TEST_CORR = -0.7167

EPPF1 = "EPPF1_profile_station_mlp"
EPPF2 = "EPPF2_profile_raster_sequence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train expanded profile perturbation forward surrogates.")
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--epoch-log", type=Path, default=DEFAULT_EPOCH_LOG)
    parser.add_argument("--ordering-audit", type=Path, default=DEFAULT_ORDERING)
    parser.add_argument("--residual-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return math.nan


def install_expanded_training(args: argparse.Namespace) -> None:
    base.CANDIDATES = (EPPF1, EPPF2)

    def make_model(name: str, vector_dim: int) -> Any:
        if name == EPPF1:
            return base.PPF1ProfileStationMLP(vector_dim)
        if name == EPPF2:
            return base.PPF2ProfileRasterSequence(vector_dim)
        raise ValueError(f"unknown candidate: {name}")

    base.make_model = make_model
    sys.argv = [
        str(Path(__file__).resolve()),
        "--pack",
        str(args.pack),
        "--summary",
        str(args.summary),
        "--audit-summary",
        str(args.audit_summary),
        "--candidates",
        str(args.candidates),
        "--metrics",
        str(args.metrics),
        "--epoch-log",
        str(args.epoch_log),
        "--ordering-audit",
        str(args.ordering_audit),
        "--residual-audit",
        str(args.residual_audit),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
    ]
    if args.cpu:
        sys.argv.append("--cpu")


def strict_gate(candidate: dict[str, Any]) -> bool:
    return bool(
        as_float(candidate, "val_nrmse") < 0.75
        and as_float(candidate, "val_ordering_accuracy") > 0.65
        and as_float(candidate, "test_ordering_accuracy") > 0.65
        and as_float(candidate, "test_mismatch_rate") < 0.35
        and as_float(candidate, "val_oracle_ordering_accuracy") >= 0.55
        and as_float(candidate, "test_residual_error_correlation") >= 0.0
    )


def rewrite_gate_and_summaries(args: argparse.Namespace) -> None:
    candidates = read_csv(args.candidates)
    metrics = read_csv(args.metrics)
    ordering = read_csv(args.ordering_audit)
    residual = read_csv(args.residual_audit)
    for row in candidates:
        row["gate_pass"] = str(strict_gate(row))
        row["notes"] = (
            "20.61 validation-only selection; strict gate requires val/test ordering > 0.65, "
            "test mismatch < 0.35, and non-negative test residual-error correlation; no refinement run"
        )
    selected = next(row for row in candidates if str(row.get("selected", "")).lower() == "true")
    gate_pass = strict_gate(selected)
    for row in residual:
        row["stage_c_gate_passed"] = str(gate_pass)
    write_csv(args.candidates, candidates)
    write_csv(args.residual_audit, residual)

    sel_name = selected["candidate"]
    metric_by_split = {
        row["split"]: row for row in metrics if row["candidate"] == sel_name and row["split"] in {"train", "val", "test"}
    }
    order_by_split = {
        row["split"]: row
        for row in ordering
        if row["candidate"] == sel_name and row["split"] in {"train", "val", "test"}
    }
    oracle_by_split = {
        row["split"]: row
        for row in ordering
        if row["candidate"] == "COMSOL_oracle" and row["split"] in {"train", "val", "test"}
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        f.write("COMSOL expanded profile perturbation forward surrogate calibration summary\n\n")
        f.write("Stage 20.61 only trains/audits profile-compatible forward surrogates; it does not run profile refinement, inverse training, checkpoints, or baseline updates.\n")
        f.write("Surrogate inputs are profile station/global features and optional rasterized profile features. Observed base delta_bz is only used for residual-ordering audit, not as model input.\n\n")
        f.write(f"pack: {args.pack}\n")
        f.write("expanded_pack_total_rows: 288\n")
        f.write("expanded_pack_reused_original_rows: 36\n")
        f.write("expanded_pack_real_comsol_forward_rows: 252\n\n")
        f.write(f"selected_candidate: {sel_name}\n")
        f.write(f"selected_by_validation_score: {float(selected['selection_score']):.6f}\n")
        f.write(f"stage_c_gate_passed: {gate_pass}\n")
        f.write("\nWaveform metrics for selected candidate:\n")
        for split_name in ("train", "val", "test"):
            row = metric_by_split[split_name]
            f.write(
                f"- {split_name}: NRMSE={float(row['nrmse']):.6f}, corr={float(row['correlation']):.6f}, "
                f"MAE={float(row['mae']):.6e}, peak_index_error={float(row['peak_index_error']):.3f}\n"
            )
        f.write("\nOrdering metrics for selected candidate:\n")
        for split_name in ("train", "val", "test"):
            row = order_by_split[split_name]
            f.write(
                f"- {split_name}: oracle_ordering={float(row['oracle_ordering_accuracy']):.6f}, "
                f"surrogate_ordering={float(row['surrogate_ordering_accuracy']):.6f}, "
                f"mismatch_rate={float(row['mismatch_rate']):.6f}, "
                f"residual_error_corr={float(row['surrogate_residual_error_correlation']):.6f}\n"
            )
        f.write("\n20.60 reference:\n")
        f.write(f"- val/test surrogate ordering = {OLD_2060_VAL_ORDERING:.4f} / {OLD_2060_TEST_ORDERING:.4f}\n")
        f.write(f"- val/test mismatch = {OLD_2060_VAL_MISMATCH:.4f} / {OLD_2060_TEST_MISMATCH:.4f}\n")
        f.write(f"- val/test residual-error corr = {OLD_2060_VAL_CORR:.4f} / {OLD_2060_TEST_CORR:.4f}\n")
        f.write("\nStrict 20.61 usability gate:\n")
        f.write("- val and test surrogate ordering accuracy both > 0.65.\n")
        f.write("- test mismatch_rate < 0.35.\n")
        f.write("- test residual-error correlation non-negative, preferably > 0.20.\n")
        f.write("- oracle residual ordering is not poor and no split leakage is allowed.\n")
        f.write("\nConclusion:\n")
        if gate_pass:
            f.write("Expanded profile perturbation data produced a usable surrogate; next stage can retry profile-forward refinement with validation-only selection.\n")
        else:
            f.write("Expanded profile perturbation data did not satisfy all strict gates; do not run profile refinement in 20.61.\n")

    with args.audit_summary.open("w", encoding="utf-8") as f:
        f.write("COMSOL expanded profile perturbation residual objective audit summary\n\n")
        f.write("Question 1: Does expanded profile perturbation data fix 20.60 test collapse?\n")
        test_order = float(order_by_split["test"]["surrogate_ordering_accuracy"])
        test_mismatch = float(order_by_split["test"]["mismatch_rate"])
        test_corr = float(order_by_split["test"]["surrogate_residual_error_correlation"])
        if test_order > OLD_2060_TEST_ORDERING and test_mismatch < OLD_2060_TEST_MISMATCH:
            f.write("- Yes directionally: test ordering/mismatch improved versus 20.60.\n")
        else:
            f.write("- No: test ordering/mismatch did not improve enough versus 20.60.\n")
        f.write("Question 2: Does real COMSOL oracle residual rank profile quality?\n")
        f.write(
            f"- Oracle ordering train/val/test = {float(oracle_by_split['train']['oracle_ordering_accuracy']):.6f} / "
            f"{float(oracle_by_split['val']['oracle_ordering_accuracy']):.6f} / "
            f"{float(oracle_by_split['test']['oracle_ordering_accuracy']):.6f}.\n"
        )
        f.write("Question 3: Does surrogate residual approximate oracle ordering?\n")
        f.write(
            f"- Surrogate ordering train/val/test = {float(order_by_split['train']['surrogate_ordering_accuracy']):.6f} / "
            f"{float(order_by_split['val']['surrogate_ordering_accuracy']):.6f} / {test_order:.6f}; "
            f"test mismatch = {test_mismatch:.6f}; test residual-error corr = {test_corr:.6f}.\n"
        )
        f.write("Question 4: Is profile perturbation data enough to make surrogate useful?\n")
        f.write(f"- Strict gate passed: {gate_pass}.\n")
        f.write("Question 5: Next step?\n")
        if gate_pass:
            f.write("- Use the calibrated expanded profile surrogate in a controlled profile-forward refinement retry.\n")
        elif float(oracle_by_split["test"]["oracle_ordering_accuracy"]) < 0.55:
            f.write("- Oracle ordering remains weak; prioritize richer observations / non-identifiability analysis.\n")
        elif test_order < 0.65:
            f.write("- Oracle is usable but surrogate ordering is weak; use more data or a stronger profile surrogate, without refinement in this stage.\n")
        else:
            f.write("- Ordering improved but gate remains incomplete; expand data or improve architecture before refinement.\n")


def main() -> None:
    args = parse_args()
    install_expanded_training(args)
    base.main()
    rewrite_gate_and_summaries(args)
    print(f"strict_gate_summary={args.summary}")


if __name__ == "__main__":
    main()
