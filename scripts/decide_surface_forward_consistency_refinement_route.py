#!/usr/bin/env python
"""Decide the 25.4 surface forward-consistency refinement route."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_CSV = ROOT / "results/metrics/surface_forward_refinement_target_set.csv"
SURROGATE_MATRIX = ROOT / "results/metrics/surface_forward_consistency_surrogate_matrix.csv"
STRATEGY_MATRIX = ROOT / "results/metrics/surface_rbc_parameter_refinement_strategy_matrix.csv"
ACCEPTANCE_MATRIX = ROOT / "results/metrics/surface_forward_refinement_acceptance_gate_matrix.csv"
SUMMARY = ROOT / "results/summaries/surface_forward_consistency_refinement_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/surface_forward_consistency_refinement_decision_matrix.csv"

FIELDS = ["option", "selected", "decision", "evidence", "blocked_by"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    for path in [TARGET_CSV, SURROGATE_MATRIX, STRATEGY_MATRIX, ACCEPTANCE_MATRIX]:
        if not path.exists():
            raise FileNotFoundError(path)
    return read_csv(TARGET_CSV), read_csv(SURROGATE_MATRIX), read_csv(STRATEGY_MATRIX), read_csv(ACCEPTANCE_MATRIX)


def main() -> int:
    target_rows, surrogate_rows, strategy_rows, acceptance_rows = require_inputs()
    role_counts = Counter(row["target_role"] for row in target_rows)
    selected_surrogate = [row for row in surrogate_rows if str(row.get("selected_for_25_5", "")).lower() == "true"]
    selected_strategy = [row for row in strategy_rows if str(row.get("recommended_for_25_5", "")).lower() == "true"]
    if len(selected_surrogate) != 1:
        raise RuntimeError("expected exactly one selected surrogate route")
    if len(selected_strategy) != 1:
        raise RuntimeError("expected exactly one selected refinement strategy")
    target_count = role_counts["refinement_target"]
    negative_count = role_counts["excluded_negative_control"]
    pass_count = role_counts["already_pass_reference"]
    options = [
        {
            "option": "A. execute 25.5 feature-space forward-consistency refinement diagnostic",
            "selected": True,
            "decision": "recommended_unique_next_step",
            "evidence": (
                f"refinement_target={target_count}; negative_control={negative_count}; pass_reference={pass_count}; "
                f"surrogate={selected_surrogate[0]['route_id']}; strategy={selected_strategy[0]['strategy_id']}; "
                f"acceptance_gates={len(acceptance_rows)}"
            ),
            "blocked_by": "",
        },
        {
            "option": "B. first train neural forward surrogate",
            "selected": False,
            "decision": "defer",
            "evidence": "F1 requires training and surrogate validation",
            "blocked_by": "25.4 and 25.5 first diagnostic are no-training routes",
        },
        {
            "option": "C. first build RBC parameter bounds / optimizer only",
            "selected": False,
            "decision": "already_planned_inside_A",
            "evidence": "R1 bounds and stop criteria are defined in the strategy matrix",
            "blocked_by": "insufficient alone without F0 consistency residual and acceptance gates",
        },
        {
            "option": "D. switch to profile-basis decoder plan",
            "selected": False,
            "decision": "not_primary_for_25_5",
            "evidence": "25.3 oracle shows most non-multi failures are RBC-representable",
            "blocked_by": "model-failure dominates representation failure outside multi-pit",
        },
        {
            "option": "E. switch to component-set decoder for multi-pit",
            "selected": False,
            "decision": "future_parallel_branch",
            "evidence": "multi-pit remains representation failure and excluded_negative_control",
            "blocked_by": "does not address the 82 RBC-representable model failures",
        },
        {
            "option": "F. pause shape-extension route",
            "selected": False,
            "decision": "not_recommended",
            "evidence": "25.4 has a low-cost diagnostic path using existing 25.3 evidence",
            "blocked_by": "no current blocker for plan-only F0/R1 diagnostic design",
        },
    ]
    write_csv(MATRIX, options)
    lines = [
        "25.4 surface forward-consistency refinement route decision",
        "",
        "decision: A. execute 25.5 feature-space forward-consistency refinement diagnostic",
        "selected_surrogate: F0_feature_space_consistency",
        "selected_refinement_strategy: R1_low_dim_param_refinement",
        "training_allowed_this_stage: false",
        "COMSOL_allowed_this_stage: false",
        "data_npz_generation_allowed_this_stage: false",
        "CURRENT_BASELINE_update: false",
        "baseline_transition_allowed: false",
        "",
        f"target_role_counts: {dict(role_counts)}",
        "rationale:",
        "- Most 25.3 failures are rbc_representable_but_model_fail, so forward-consistency refinement is the first diagnostic.",
        "- Multi-pit is recorded as a future component-set branch and does not block A.",
        "- Neural surrogate training and component-set decoding are later branches, not 25.5 first execution.",
        f"decision_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
