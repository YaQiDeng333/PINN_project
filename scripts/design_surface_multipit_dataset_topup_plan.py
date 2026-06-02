#!/usr/bin/env python
"""Design the 25.9 surface multi-pit dataset top-up plan."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
AUDIT_CSV = ROOT / "results/metrics/surface_multipit_component_label_audit.csv"

SUMMARY = ROOT / "results/summaries/surface_multipit_dataset_topup_plan_summary.txt"
PLAN_CSV = ROOT / "results/metrics/surface_multipit_dataset_topup_plan.csv"
COVERAGE_CSV = ROOT / "results/metrics/surface_multipit_expected_coverage.csv"

TOPUP_N = 96
FALLBACK_TOPUP_N = 60
EXISTING_N = 16
ASSEMBLED_N = EXISTING_N + TOPUP_N

PLAN_FIELDS = ["plan_item", "value", "train", "val", "test", "notes"]
COVERAGE_FIELDS = ["dimension", "level", "target_count", "topup_or_assembled", "rationale"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_plan_rows() -> list[dict[str, Any]]:
    return [
        {"plan_item": "existing_seed_samples", "value": EXISTING_N, "train": 9, "val": 3, "test": 4, "notes": "from 25.2 pilot manifest/audit; read-only seed"},
        {"plan_item": "target_topup_samples", "value": TOPUP_N, "train": 63, "val": 17, "test": 16, "notes": "default top-up to reach assembled 72/20/20 split"},
        {"plan_item": "assembled_target_samples", "value": ASSEMBLED_N, "train": 72, "val": 20, "test": 20, "notes": "default train/val/test split for first component-set branch"},
        {"plan_item": "fallback_topup_samples", "value": FALLBACK_TOPUP_N, "train": 39, "val": 11, "test": 10, "notes": "fallback assembled N=76 with 48/14/14 split; feasibility only"},
        {"plan_item": "primary_component_count", "value": "2", "train": 56, "val": 15, "test": 13, "notes": "main training coverage for current two-pit branch"},
        {"plan_item": "future_negative_component_count", "value": "3", "train": 7, "val": 2, "test": 3, "notes": "small amount of K=3 future/negative coverage inside K=3 output contract"},
    ]


def coverage_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("component_count", [("2", 84), ("3", 12)], "topup", "K=3 fixed slots with two-component main coverage and small three-component stress coverage"),
        ("separation", [("separated", 24), ("close", 24), ("touching", 24), ("partially_overlapping", 24)], "topup", "component merge and missed-component failure modes need balanced spacing"),
        ("relative_depth", [("similar_depth", 48), ("deep_and_shallow", 48)], "topup", "depth contrast tests whether secondary shallow pits disappear"),
        ("orientation", [("aligned_x", 32), ("aligned_y", 32), ("diagonal", 32)], "topup", "avoid orientation leakage from scan-line geometry"),
        ("topology", [("disconnected", 36), ("merged_projected_mask", 24), ("touching_boundary", 24), ("partially_overlapping", 12)], "topup", "stress topology where union mask no longer cleanly separates instances"),
        ("size_pair", [("small-small", 24), ("small-large", 24), ("medium-medium", 24), ("medium-large", 24)], "topup", "cover small secondary pits and balanced medium pairs"),
        ("primitive_mix", [("smooth-smooth", 36), ("flat-flat", 24), ("smooth-flat", 18), ("asymmetric_mix", 18)], "topup", "connect 25.1 non-RBC taxonomy to component-set branch"),
    ]
    for dimension, levels, scope, rationale in specs:
        for level, count in levels:
            rows.append({"dimension": dimension, "level": level, "target_count": count, "topup_or_assembled": scope, "rationale": rationale})
    return rows


def write_summary(audit_rows: list[dict[str, str]], manifest: dict[str, Any]) -> None:
    split_counts = dict(Counter(row["split"] for row in audit_rows))
    lines = [
        "25.9 surface multi-pit dataset top-up plan",
        "",
        f"source_dataset_id: {manifest.get('dataset_id')}",
        f"existing_multi_pit_seed_N: {len(audit_rows)}",
        f"existing_split_counts: {split_counts}",
        f"target_topup_N: {TOPUP_N}",
        f"fallback_topup_N: {FALLBACK_TOPUP_N}",
        f"assembled_target_N: {ASSEMBLED_N}",
        "assembled_split: 72/20/20",
        "topup_split: 63/17/16",
        "fallback_assembled_split: 48/14/14",
        "",
        "coverage_policy: balance separation, topology, depth contrast, size pair, orientation, and primitive mix.",
        "schema_policy: top-up must add per-component rotation, component-level projected masks/depth grids, and explicit separation/topology labels.",
        "training_policy: this stage does not train; top-up is the next route only after review.",
        f"topup_plan_csv: {PLAN_CSV}",
        f"expected_coverage_csv: {COVERAGE_CSV}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not AUDIT_CSV.exists():
        raise FileNotFoundError("run audit_surface_multipit_component_labels.py before dataset top-up design")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit_rows = read_csv(AUDIT_CSV)
    write_csv(PLAN_CSV, build_plan_rows(), PLAN_FIELDS)
    write_csv(COVERAGE_CSV, coverage_rows(), COVERAGE_FIELDS)
    write_summary(audit_rows, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
