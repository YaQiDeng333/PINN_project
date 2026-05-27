#!/usr/bin/env python
"""20.94 multi-seed training for the selected liftoff adapter candidate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import load_true_3d_rbc_liftoff_aug_dataset as liftoff
from train_true_3d_rbc_liftoff_adapter_candidates import run_training


ROOT = liftoff.ROOT
SCREEN_SELECTED = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_candidate_screen_selected.csv"
MODEL_SELECTED = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_model_selected.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_liftoff_adapter_training_summary.txt"
SEED_SUMMARY = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_seed_summary.csv"
METRICS = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_metrics.csv"
BY_LIFTOFF = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_by_liftoff.csv"
VS_BASELINE = ROOT / "results/metrics/true_3d_rbc_liftoff_adapter_vs_baseline.csv"


def read_selected(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty selected candidate file: {path}")
    return rows[-1]["selected_candidate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=liftoff.DATASET_ID)
    parser.add_argument("--selected-candidate", default="")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--include-full-model", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.selected_candidate or read_selected(SCREEN_SELECTED)
    run_training(SUMMARY, METRICS, BY_LIFTOFF, SEED_SUMMARY, VS_BASELINE, MODEL_SELECTED, selected, args.seeds, args)
    print(f"selected_candidate={selected}")
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
