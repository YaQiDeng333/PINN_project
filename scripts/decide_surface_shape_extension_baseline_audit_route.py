#!/usr/bin/env python
"""Decide the next route after the 25.3 shape-extension baseline audit."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

from audit_surface_shape_extension_rbc_oracle_fit import DATASET_ID, ROOT, write_csv
from diagnose_surface_shape_extension_oracle_vs_baseline import FAILURE_MODE_BY_SHAPE, MATRIX


SUMMARY = ROOT / "results/summaries/surface_shape_extension_baseline_audit_route_decision_summary.txt"
DECISION_MATRIX = ROOT / "results/metrics/surface_shape_extension_baseline_audit_decision_matrix.csv"

DECISION_FIELDS = ["question", "answer", "decision", "evidence", "next_action"]

OPTIONS = {
    "A": "A. profile-basis / depth-grid decoder plan",
    "B": "B. component-set decoder plan",
    "C": "C. geometry-aware contour / primitive decoder plan",
    "D": "D. forward-consistency refinement plan",
    "E": "E. train 20.85-style model on shape-extension data",
    "F": "F. revise dataset/schema",
    "G": "G. stop shape-extension",
}


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Decide route after 25.3 baseline audit.").parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def shape(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    for row in rows:
        if row.get("shape_type") == name and row.get("split") == "all":
            return row
    return {}


def decide(matrix: list[dict[str, str]], shapes: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    total = len(matrix)
    counts = Counter(row["diagnosis"] for row in matrix)
    non_rbc = [row for row in matrix if row["shape_type"] != "rbc_like_smooth_pit"]
    non_counts = Counter(row["diagnosis"] for row in non_rbc)
    oracle_failure_rate = non_counts.get("rbc_not_representable", 0) / max(len(non_rbc), 1)
    model_failure_rate = non_counts.get("rbc_representable_but_model_fail", 0) / max(len(non_rbc), 1)
    pass_rate = non_counts.get("rbc_representable_and_model_pass", 0) / max(len(non_rbc), 1)

    rbc = shape(shapes, "rbc_like_smooth_pit")
    flat = shape(shapes, "flat_bottom_pit")
    sharp = shape(shapes, "sharp_wall_boxy_corrosion")
    asym = shape(shapes, "asymmetric_corrosion")
    crack = shape(shapes, "elongated_crack_like_surface_defect")
    multi = shape(shapes, "multi_pit_two_component_surface_defect")
    irregular = shape(shapes, "irregular_corrosion_non_rbc")

    shape_oracle_failures = {
        row["shape_type"]: 1.0 - f(row.get("oracle_representable_rate", "nan"))
        for row in shapes
        if row.get("split") == "all" and row.get("shape_type") != "rbc_like_smooth_pit"
    }
    broad_profile_failure = sum(rate >= 0.50 for rate in shape_oracle_failures.values()) >= 4
    multi_prominent = (1.0 - f(multi.get("oracle_representable_rate", "nan"))) >= 0.75 and f(multi.get("merge_component_proxy_rate", "0")) >= 0.50
    contour_prominent = sum((1.0 - f(row.get("oracle_representable_rate", "nan"))) >= 0.50 for row in [flat, sharp, crack]) >= 2
    rbc_control_ok = f(rbc.get("model_pass_rate", "0")) >= 0.75 and f(rbc.get("oracle_representable_rate", "0")) >= 0.75

    rows: list[dict[str, Any]] = [
        {
            "question": "Did the audit produce rows?",
            "answer": "yes" if total else "no",
            "decision": "audit_outputs_present" if total else "blocked",
            "evidence": f"rows={total}; diagnosis_counts={dict(counts)}",
            "next_action": "continue" if total else OPTIONS["F"],
        },
        {
            "question": "Is RBC-like control stable enough for audit separation?",
            "answer": "yes" if rbc_control_ok else "no",
            "decision": "rbc_control_ok" if rbc_control_ok else "rbc_control_unstable",
            "evidence": f"model_pass_rate={rbc.get('model_pass_rate')}; oracle_repr_rate={rbc.get('oracle_representable_rate')}",
            "next_action": "use RBC-like as control" if rbc_control_ok else "treat 20.85 inversion as unstable on its own control",
        },
        {
            "question": "Do most non-RBC failures start at the RBC oracle representation?",
            "answer": "yes" if oracle_failure_rate >= 0.50 else "no",
            "decision": "representation_failure_dominant" if oracle_failure_rate >= 0.50 else "model_failure_or_mixed",
            "evidence": f"non_rbc_oracle_failure_rate={oracle_failure_rate:.6f}; model_failure_rate={model_failure_rate:.6f}; pass_rate={pass_rate:.6f}",
            "next_action": "do not keep six-parameter RBC as non-RBC route" if oracle_failure_rate >= 0.50 else "consider model/forward refinement routes",
        },
        {
            "question": "Is multi-pit component failure the dominant unique blocker?",
            "answer": "yes" if multi_prominent and not broad_profile_failure else "no",
            "decision": "component_set_needed" if multi_prominent else "component_set_secondary",
            "evidence": f"multi_oracle_repr_rate={multi.get('oracle_representable_rate')}; multi_merge_proxy={multi.get('merge_component_proxy_rate')}",
            "next_action": OPTIONS["B"] if multi_prominent and not broad_profile_failure else "keep component-set route as sub-branch",
        },
        {
            "question": "Are flat/sharp/crack contour families prominent failures?",
            "answer": "yes" if contour_prominent else "no",
            "decision": "contour_decoder_needed" if contour_prominent else "contour_decoder_secondary",
            "evidence": f"flat_repr={flat.get('oracle_representable_rate')}; sharp_repr={sharp.get('oracle_representable_rate')}; crack_repr={crack.get('oracle_representable_rate')}",
            "next_action": OPTIONS["C"] if contour_prominent and not broad_profile_failure else "keep contour/primitive route as sub-branch",
        },
        {
            "question": "Is profile/depth failure broad across non-RBC families?",
            "answer": "yes" if broad_profile_failure else "no",
            "decision": "profile_depth_decoder_primary" if broad_profile_failure else "profile_depth_not_primary",
            "evidence": f"shape_oracle_failure_rates={shape_oracle_failures}; asym_repr={asym.get('oracle_representable_rate')}; irregular_repr={irregular.get('oracle_representable_rate')}",
            "next_action": OPTIONS["A"] if broad_profile_failure else "select narrower route based on dominant failure",
        },
        {
            "question": "Can 20.85 transition to a non-RBC-like baseline now?",
            "answer": "no",
            "decision": "baseline_transition_forbidden",
            "evidence": "CURRENT_BASELINE update forbidden; non-RBC pass rate not a formal benchmark transition",
            "next_action": "keep CURRENT_BASELINE unchanged",
        },
    ]

    if not total:
        selected = "F"
    elif oracle_failure_rate >= 0.50 and broad_profile_failure:
        selected = "A"
    elif oracle_failure_rate >= 0.50 and multi_prominent:
        selected = "B"
    elif oracle_failure_rate >= 0.50 and contour_prominent:
        selected = "C"
    elif model_failure_rate > oracle_failure_rate and pass_rate < 0.50:
        selected = "D"
    elif model_failure_rate > 0.50 and pass_rate < 0.50:
        selected = "E"
    else:
        selected = "A"

    rows.append(
        {
            "question": "Unique next route",
            "answer": OPTIONS[selected],
            "decision": f"select_{selected}",
            "evidence": f"oracle_failure_rate={oracle_failure_rate:.6f}; model_failure_rate={model_failure_rate:.6f}; broad_profile_failure={broad_profile_failure}; multi_prominent={multi_prominent}; contour_prominent={contour_prominent}",
            "next_action": OPTIONS[selected],
        }
    )
    return OPTIONS[selected], rows


def run(_args: argparse.Namespace) -> int:
    if not MATRIX.exists():
        raise FileNotFoundError(MATRIX)
    if not FAILURE_MODE_BY_SHAPE.exists():
        raise FileNotFoundError(FAILURE_MODE_BY_SHAPE)
    matrix = read_csv(MATRIX)
    shapes = read_csv(FAILURE_MODE_BY_SHAPE)
    selected, rows = decide(matrix, shapes)
    write_csv(DECISION_MATRIX, rows, DECISION_FIELDS)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "surface shape-extension baseline audit route decision summary",
                "stage: 25.3",
                "",
                f"dataset_id: {DATASET_ID}",
                f"decision: {selected}",
                "training_allowed: false",
                "baseline_update_allowed: false",
                "CURRENT_BASELINE_update: false",
                "direct_20_85_baseline_transition_allowed: false",
                f"decision_matrix: {DECISION_MATRIX}",
                "",
                "rationale:",
                *[f"- {row['question']}: {row['answer']} ({row['decision']}); evidence={row['evidence']}" for row in rows],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
