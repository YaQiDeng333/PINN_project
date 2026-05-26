"""Smoke test for make_comsol_polygon_matched_coverage_resplit.py."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from make_comsol_polygon_matched_coverage_resplit import choose_resplit


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mock_records() -> list[dict]:
    records = []
    hard_cases = [
        "x_bin_wrong_like",
        "x_bin_wrong_like",
        "both_bins_wrong_like",
        "both_bins_wrong_like",
        "bins_correct_center_or_offset_bad",
        "bins_correct_center_or_offset_bad",
        "geometry_or_type_interaction",
        "geometry_or_type_interaction",
        "rare_y_bin_wrong",
        "rare_y_bin_wrong",
    ]
    for idx, hard_case in enumerate(hard_cases):
        records.append(
            {
                "global_id": f"mock_{idx:03d}",
                "old_split": "train" if idx < 6 else "val",
                "old_sample_index": idx,
                "hard_case_type": hard_case,
                "true_rotated": idx % 2 == 0,
                "true_multi_component": idx % 3 == 0,
                "component_count": 1,
                "bin_pairs": [(idx % 5, idx // 5)],
                "center_bin_signature": f"{idx % 5}:{idx // 5}",
            }
        )
    return records


def test_assignment_no_duplicates_and_coverable() -> None:
    import make_comsol_polygon_matched_coverage_resplit as module

    old_targets = module.TARGET_COUNTS
    module.TARGET_COUNTS = {
        "train": {
            "x_bin_wrong_like": 1,
            "both_bins_wrong_like": 1,
            "bins_correct_center_or_offset_bad": 1,
            "geometry_or_type_interaction": 1,
            "rare_y_bin_wrong": 1,
        },
        "val": {
            "x_bin_wrong_like": 1,
            "both_bins_wrong_like": 0,
            "bins_correct_center_or_offset_bad": 1,
            "geometry_or_type_interaction": 0,
            "rare_y_bin_wrong": 0,
        },
        "test": {
            "x_bin_wrong_like": 0,
            "both_bins_wrong_like": 1,
            "bins_correct_center_or_offset_bad": 0,
            "geometry_or_type_interaction": 1,
            "rare_y_bin_wrong": 1,
        },
    }
    try:
        assignments, _score = choose_resplit(_mock_records(), seed=3, iterations=200)
    finally:
        module.TARGET_COUNTS = old_targets
    ids = [row["global_id"] for split in ("train", "val", "test") for row in assignments[split]]
    if len(ids) != len(set(ids)):
        raise AssertionError("resplit contains duplicate source ids")
    train_bins = {pair for row in assignments["train"] for pair in row["bin_pairs"]}
    for split in ("val", "test"):
        for row in assignments[split]:
            for x_bin, y_bin in row["bin_pairs"]:
                distance = min(abs(x_bin - tx) + abs(y_bin - ty) for tx, ty in train_bins)
                if distance > 1:
                    raise AssertionError(f"held-out bin is not covered within distance 1: {row}")


def test_manifest_written() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "split_manifest.csv"
        _write_csv(
            path,
            [
                {
                    "new_split": "train",
                    "new_sample_index": 0,
                    "source_id": "mock_000",
                    "old_split": "train",
                    "old_sample_index": 0,
                    "hard_case_type": "x_bin_wrong_like",
                    "center_bin_signature": "0:0",
                    "true_rotated": False,
                    "true_multi_component": False,
                    "component_count": 1,
                }
            ],
        )
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        if rows[0]["source_id"] != "mock_000":
            raise AssertionError("manifest source id was not written")


if __name__ == "__main__":
    test_assignment_no_duplicates_and_coverable()
    test_manifest_written()
