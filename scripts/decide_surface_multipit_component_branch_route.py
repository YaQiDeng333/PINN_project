#!/usr/bin/env python
"""Decide the 25.9 surface multi-pit component branch route."""

from __future__ import annotations

import csv
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CSV = ROOT / "results/metrics/surface_multipit_component_label_audit.csv"
MISSING_FIELDS_CSV = ROOT / "results/metrics/surface_multipit_component_label_missing_fields.csv"
REPRESENTATION_MATRIX = ROOT / "results/metrics/surface_multipit_component_set_representation_matrix.csv"
TOPUP_PLAN = ROOT / "results/metrics/surface_multipit_dataset_topup_plan.csv"
COMSOL_FEASIBILITY = ROOT / "results/metrics/surface_multipit_comsol_generation_feasibility.csv"
GATES = ROOT / "results/metrics/surface_multipit_acceptance_gate_matrix.csv"

SUMMARY = ROOT / "results/summaries/surface_multipit_component_branch_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/surface_multipit_component_branch_decision_matrix.csv"

FIELDS = ["option", "selected", "decision", "evidence", "blocked_by"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def git_value(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def decide() -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    audit = read_csv(AUDIT_CSV)
    missing = read_csv(MISSING_FIELDS_CSV)
    reps = read_csv(REPRESENTATION_MATRIX)
    topup = read_csv(TOPUP_PLAN)
    comsol = read_csv(COMSOL_FEASIBILITY)
    gates = read_csv(GATES)

    c1_seed_ready = bool(audit) and all(row["label_sufficiency"] == "sufficient_for_C1_seed_with_schema_gaps" for row in audit)
    all_component_count_two = bool(audit) and all(row["component_count"] == "2" and row["component_json_count"] == "2" for row in audit)
    component_fields_present = bool(audit) and all(
        as_bool(row["component_centers_present"])
        and as_bool(row["component_lwd_present"])
        and as_bool(row["component_depth_present"])
        and as_bool(row["projected_mask_present"])
        and as_bool(row["depth_grid_present"])
        for row in audit
    )
    c1_recommended = any(row["route_id"] == "C1" and as_bool(row["selected_first"]) for row in reps)
    topup_ready = any(row["plan_item"] == "target_topup_samples" and row["value"] == "96" for row in topup)
    comsol_ready = any(row["route_id"] == "G1" and as_bool(row["recommended"]) and as_bool(row["feasible_for_25_10"]) for row in comsol)
    depth_grid_complete = bool(audit) and all(as_bool(row["depth_grid_present"]) for row in audit)
    hard_label_blocker = not (c1_seed_ready and all_component_count_two and component_fields_present)
    current_baseline_diff = git_value(["diff", "--name-only", "--", "CURRENT_BASELINE.md"])
    forbidden_diff = git_value(["diff", "--name-only", "--", "data", "checkpoints", "notes", "results/previews", "scripts/visualize_current_baseline.py"])

    if hard_label_blocker:
        selected = "B. revise component label schema"
    elif not comsol_ready and depth_grid_complete:
        selected = "D. use depth-grid decoder instead"
    elif c1_recommended and topup_ready and comsol_ready and not current_baseline_diff and not forbidden_diff:
        selected = "A. execute multi-pit COMSOL top-up generation"
    elif not c1_recommended:
        selected = "C. implement component-set decoder before more data"
    else:
        selected = "E. pause multi-pit branch"

    options = [
        {
            "option": "A. execute multi-pit COMSOL top-up generation",
            "selected": selected.startswith("A."),
            "decision": "selected" if selected.startswith("A.") else "not_selected",
            "evidence": f"c1_seed_ready={c1_seed_ready}; topup_ready={topup_ready}; comsol_ready={comsol_ready}; CURRENT_BASELINE_diff={bool(current_baseline_diff)}",
            "blocked_by": "" if selected.startswith("A.") else "label, top-up, COMSOL feasibility, or protected-path gate not satisfied",
        },
        {
            "option": "B. revise component label schema",
            "selected": selected.startswith("B."),
            "decision": "selected" if selected.startswith("B.") else "not_selected",
            "evidence": f"hard_label_blocker={hard_label_blocker}; c1_seed_ready={c1_seed_ready}; component_fields_present={component_fields_present}",
            "blocked_by": "" if selected.startswith("B.") else "existing labels are sufficient for seed audit; gaps are documented for top-up",
        },
        {
            "option": "C. implement component-set decoder before more data",
            "selected": selected.startswith("C."),
            "decision": "selected" if selected.startswith("C.") else "not_selected",
            "evidence": f"c1_recommended={c1_recommended}; audit_rows={len(audit)}",
            "blocked_by": "" if selected.startswith("C.") else "current 16 samples are too small; top-up should precede training/decoder implementation",
        },
        {
            "option": "D. use depth-grid decoder instead",
            "selected": selected.startswith("D."),
            "decision": "selected" if selected.startswith("D.") else "not_selected",
            "evidence": f"depth_grid_complete={depth_grid_complete}; comsol_ready={comsol_ready}",
            "blocked_by": "" if selected.startswith("D.") else "component labels and COMSOL feasibility are adequate for component-set top-up",
        },
        {
            "option": "E. pause multi-pit branch",
            "selected": selected.startswith("E."),
            "decision": "selected" if selected.startswith("E.") else "not_selected",
            "evidence": f"gates_defined={len(gates)}; missing_field_rows={len(missing)}; forbidden_diff={bool(forbidden_diff)}",
            "blocked_by": "" if selected.startswith("E.") else "route has a concrete next generation step",
        },
    ]
    context = {
        "selected": selected,
        "audit_rows": len(audit),
        "split_counts": dict(Counter(row["split"] for row in audit)),
        "separation_counts": dict(Counter(row["separation_bucket"] for row in audit)),
        "topology_counts": dict(Counter(row["topology_bucket"] for row in audit)),
        "c1_seed_ready": c1_seed_ready,
        "component_fields_present": component_fields_present,
        "c1_recommended": c1_recommended,
        "topup_ready": topup_ready,
        "comsol_ready": comsol_ready,
        "depth_grid_complete": depth_grid_complete,
        "gate_count": len(gates),
        "current_baseline_unchanged": not bool(current_baseline_diff),
        "forbidden_diff_present": bool(forbidden_diff),
        "blocking_schema_gaps": [row["field_name"] for row in missing if row["severity"] == "blocking"],
    }
    return selected, options, context


def write_summary(context: dict[str, Any]) -> None:
    lines = [
        "25.9 surface multi-pit component branch route decision",
        "",
        f"decision: {context['selected']}",
        f"multi_pit_audit_rows: {context['audit_rows']}",
        f"split_counts: {context['split_counts']}",
        f"separation_counts: {context['separation_counts']}",
        f"topology_counts: {context['topology_counts']}",
        f"C1_seed_ready: {context['c1_seed_ready']}",
        f"component_fields_present: {context['component_fields_present']}",
        f"C1_recommended: {context['c1_recommended']}",
        f"topup_ready: {context['topup_ready']}",
        f"COMSOL_feasibility_ready: {context['comsol_ready']}",
        f"depth_grid_complete: {context['depth_grid_complete']}",
        f"acceptance_gate_count: {context['gate_count']}",
        f"CURRENT_BASELINE_unchanged: {context['current_baseline_unchanged']}",
        f"forbidden_diff_present: {context['forbidden_diff_present']}",
        f"blocking_schema_gaps: {context['blocking_schema_gaps']}",
        "",
        "interpretation: existing 16 multi-pit rows are enough to define and audit C1, but insufficient for training; next stage should generate the multi-pit top-up pack if review passes.",
        "baseline_policy: no CURRENT_BASELINE.md transition and no six-parameter RBC success credit for multi-pit.",
        f"decision_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in [AUDIT_CSV, MISSING_FIELDS_CSV, REPRESENTATION_MATRIX, TOPUP_PLAN, COMSOL_FEASIBILITY, GATES]:
        if not path.exists():
            raise FileNotFoundError(f"missing route prerequisite: {path}")
    _selected, rows, context = decide()
    write_csv(MATRIX, rows, FIELDS)
    write_summary(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
