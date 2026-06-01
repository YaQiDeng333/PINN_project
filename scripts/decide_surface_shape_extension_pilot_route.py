#!/usr/bin/env python
"""Decide the route after validating the 25.2 surface shape-extension pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_surface_shape_extension_pilot_v1"
TARGET_N = 120
FALLBACK_N = 84
TARGET_SHAPES = [
    "rbc_like_smooth_pit",
    "flat_bottom_pit",
    "sharp_wall_boxy_corrosion",
    "asymmetric_corrosion",
    "elongated_crack_like_surface_defect",
    "multi_pit_two_component_surface_defect",
    "irregular_corrosion_non_rbc",
]

DEFAULT_MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
DEFAULT_VALIDATION_METRICS = ROOT / "results/metrics/surface_shape_extension_pilot_validation_metrics.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/surface_shape_extension_pilot_route_decision_summary.txt"
DEFAULT_MATRIX = ROOT / "results/metrics/surface_shape_extension_pilot_decision_matrix.csv"

FIELDS = ["question", "answer", "decision", "evidence", "next_action"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide the post-25.2 surface shape-extension route.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-metrics", type=Path, default=DEFAULT_VALIDATION_METRICS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pass_value(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def run(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = {row["check_name"]: row for row in read_csv(args.validation_metrics)}
    n_samples = int(manifest.get("n_samples", 0))
    shape_counts = dict(manifest.get("shape_type_counts", {}))
    representation_counts = dict(manifest.get("representation_target_counts", {}))
    target_pass = n_samples >= TARGET_N
    fallback_pass = n_samples >= FALLBACK_N
    all_shapes_present = all(int(shape_counts.get(shape, 0)) > 0 for shape in TARGET_SHAPES)
    full_shape_counts = all(int(shape_counts.get(shape, 0)) >= (24 if shape == "rbc_like_smooth_pit" else 16) for shape in TARGET_SHAPES)
    labels_usable = all(
        pass_value(checks.get(name, {}).get("pass", "false"))
        for name in [
            "non_rbc_not_rbc_compatible",
            "multi_pit_component_labels",
            "crack_like_aspect_rotation",
            "irregular_depth_grid_target",
            "depth_grid_finite",
            "projected_mask_nonempty",
        ]
    )
    validation_pass = bool(manifest.get("validation_status", False))
    needs_topup = not target_pass or not full_shape_counts
    route = (
        "enter_25_3_current_baseline_generalization_audit"
        if validation_pass and fallback_pass and all_shapes_present and labels_usable
        else "top_up_or_revise_surface_shape_extension_pilot"
    )
    rows = [
        {
            "question": "Did pilot reach target N=120?",
            "answer": "yes" if target_pass else "no",
            "decision": "full_pilot" if target_pass else "partial_pilot",
            "evidence": f"n_samples={n_samples}",
            "next_action": "use full pilot for audit" if target_pass else "continue only if fallback and coverage gates pass",
        },
        {
            "question": "If partial, did pilot reach fallback N=84?",
            "answer": "yes" if fallback_pass else "no",
            "decision": "fallback_pass" if fallback_pass else "fallback_fail",
            "evidence": f"n_samples={n_samples}; fallback={FALLBACK_N}",
            "next_action": "may audit as partial pilot" if fallback_pass else "top up failed generation first",
        },
        {
            "question": "Is each shape_type covered?",
            "answer": "yes" if all_shapes_present else "no",
            "decision": "shape_coverage_present" if all_shapes_present else "shape_coverage_missing",
            "evidence": json.dumps(shape_counts, ensure_ascii=False, sort_keys=True),
            "next_action": "audit all families" if all_shapes_present else "top up missing shape families",
        },
        {
            "question": "Are non-RBC-like labels usable?",
            "answer": "yes" if labels_usable else "no",
            "decision": "non_rbc_label_gate_pass" if labels_usable else "non_rbc_label_gate_fail",
            "evidence": json.dumps(representation_counts, ensure_ascii=False, sort_keys=True),
            "next_action": "preserve representation_target in 25.3 audit",
        },
        {
            "question": "Are depth/profile/mask labels audit-ready?",
            "answer": "yes" if labels_usable else "no",
            "decision": "profile_mask_labels_ready" if labels_usable else "profile_mask_labels_not_ready",
            "evidence": "depth_grid_finite/projected_mask_nonempty/component/crack/irregular checks",
            "next_action": "use profile/depth as governing audit metrics",
        },
        {
            "question": "Should 25.3 current baseline generalization audit start?",
            "answer": "yes" if route.startswith("enter_25_3") else "no",
            "decision": route,
            "evidence": f"validation_pass={validation_pass}; fallback_pass={fallback_pass}; all_shapes_present={all_shapes_present}; labels_usable={labels_usable}",
            "next_action": "run frozen 20.85 baseline audit; no training" if route.startswith("enter_25_3") else "repair/top-up first",
        },
        {
            "question": "Is failed-family top-up needed?",
            "answer": "yes" if needs_topup else "no",
            "decision": "topup_needed" if needs_topup else "topup_not_needed",
            "evidence": f"target_pass={target_pass}; full_shape_counts={full_shape_counts}",
            "next_action": "only top up if full pilot target is required before audit" if needs_topup else "no top-up before 25.3",
        },
        {
            "question": "Are training and baseline update still forbidden?",
            "answer": "yes",
            "decision": "training_and_baseline_update_forbidden",
            "evidence": "manifest train_ready_candidate=false; baseline_ready=false; CURRENT_BASELINE unchanged",
            "next_action": "do not train until a later explicit 25.4 training gate",
        },
    ]
    write_csv(args.matrix, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface shape-extension pilot route decision summary",
                "stage: 25.2",
                "",
                f"dataset_id: {DATASET_ID}",
                f"n_samples: {n_samples}",
                f"target_n_pass: {str(target_pass).lower()}",
                f"fallback_n_pass: {str(fallback_pass).lower()}",
                f"shape_coverage_present: {str(all_shapes_present).lower()}",
                f"non_rbc_labels_usable: {str(labels_usable).lower()}",
                f"decision: {route}",
                "next_step: 25.3 current baseline generalization audit" if route.startswith("enter_25_3") else "next_step: top up or revise surface shape-extension pilot",
                "training_allowed: false",
                "baseline_update_allowed: false",
                f"decision_matrix: {args.matrix}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not validation_pass or not fallback_pass:
        raise RuntimeError("pilot route decision blocked by validation/fallback failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
