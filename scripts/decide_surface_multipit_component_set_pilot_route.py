#!/usr/bin/env python
"""Decide the post-25.9b route for the surface multi-pit component-set pilot."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json"
TOPUP_METRICS = ROOT / "results/metrics/surface_multipit_topup_pack_validation_metrics.csv"
PILOT_METRICS = ROOT / "results/metrics/surface_multipit_component_set_pilot_validation_metrics.csv"
SUMMARY = ROOT / "results/summaries/surface_multipit_component_set_pilot_route_decision_summary.txt"
MATRIX = ROOT / "results/metrics/surface_multipit_component_set_pilot_decision_matrix.csv"

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


def git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def all_pass(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(row.get("pass") == "true" for row in rows)


def metric_value(rows: list[dict[str, str]], check_name: str) -> str:
    for row in rows:
        if row.get("check_name") == check_name:
            return row.get("observed", "")
    return ""


def decide() -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    topup_rows = read_csv(TOPUP_METRICS)
    pilot_rows = read_csv(PILOT_METRICS)
    topup_pass = all_pass(topup_rows)
    pilot_pass = all_pass(pilot_rows)
    train_ready = bool(manifest.get("train_ready_candidate"))
    baseline_ready = bool(manifest.get("baseline_ready"))
    current_baseline_diff = git_value(["diff", "--name-only", "--", "CURRENT_BASELINE.md"])
    forbidden_diff = git_value(["diff", "--name-only", "--", "checkpoints", "notes", "results/previews", "scripts/visualize_current_baseline.py"])
    data_staged = git_value(["diff", "--cached", "--name-only", "--", "data"])
    if topup_pass and pilot_pass and train_ready and not baseline_ready and not current_baseline_diff and not data_staged:
        selected = "A. enter 25.10 component-set training gate"
    elif not topup_pass:
        selected = "B. generate failed/missing multi-pit top-up strata"
    elif not pilot_pass or not train_ready:
        selected = "C. revise component-set assembly/label extraction"
    else:
        selected = "D. pause before any training gate"
    options = [
        {
            "option": "A. enter 25.10 component-set training gate",
            "selected": selected.startswith("A."),
            "decision": "selected" if selected.startswith("A.") else "not_selected",
            "evidence": f"topup_pass={topup_pass}; pilot_pass={pilot_pass}; train_ready_candidate={train_ready}; baseline_ready={baseline_ready}; assembled_N={manifest.get('n_samples')}",
            "blocked_by": "" if selected.startswith("A.") else "top-up, validation, train-ready, or protected-path gate not satisfied",
        },
        {
            "option": "B. generate failed/missing multi-pit top-up strata",
            "selected": selected.startswith("B."),
            "decision": "selected" if selected.startswith("B.") else "not_selected",
            "evidence": f"topup_pass={topup_pass}; topup_N={metric_value(topup_rows, 'target_N')}",
            "blocked_by": "" if selected.startswith("B.") else "top-up generated full 96/96 with required coverage",
        },
        {
            "option": "C. revise component-set assembly/label extraction",
            "selected": selected.startswith("C."),
            "decision": "selected" if selected.startswith("C.") else "not_selected",
            "evidence": f"pilot_pass={pilot_pass}; train_ready_candidate={train_ready}; component_check={metric_value(pilot_rows, 'component_params_valid')}",
            "blocked_by": "" if selected.startswith("C.") else "assembled pilot validation passed",
        },
        {
            "option": "D. pause before any training gate",
            "selected": selected.startswith("D."),
            "decision": "selected" if selected.startswith("D.") else "not_selected",
            "evidence": f"CURRENT_BASELINE_diff={bool(current_baseline_diff)}; forbidden_diff={bool(forbidden_diff)}; data_staged={bool(data_staged)}",
            "blocked_by": "" if selected.startswith("D.") else "route has a single explicit next gate",
        },
    ]
    context = {
        "selected": selected,
        "dataset_id": manifest.get("dataset_id"),
        "assembled_N": manifest.get("n_samples"),
        "split_counts": manifest.get("split_counts"),
        "component_count_counts": manifest.get("component_count_counts"),
        "separation_counts": manifest.get("separation_counts"),
        "topology_counts": manifest.get("topology_counts"),
        "train_ready_candidate": train_ready,
        "baseline_ready": baseline_ready,
        "topup_pass": topup_pass,
        "pilot_pass": pilot_pass,
        "current_baseline_unchanged": not bool(current_baseline_diff),
        "data_staged": bool(data_staged),
    }
    return selected, options, context


def write_summary(context: dict[str, Any]) -> None:
    lines = [
        "surface multi-pit component-set pilot route decision",
        "stage: 25.9b",
        "",
        f"decision: {context['selected']}",
        f"dataset_id: {context['dataset_id']}",
        f"assembled_N: {context['assembled_N']}",
        f"split_counts: {context['split_counts']}",
        f"component_count_counts: {context['component_count_counts']}",
        f"separation_counts: {context['separation_counts']}",
        f"topology_counts: {context['topology_counts']}",
        f"train_ready_candidate: {context['train_ready_candidate']}",
        f"baseline_ready: {context['baseline_ready']}",
        f"topup_validation_pass: {context['topup_pass']}",
        f"pilot_validation_pass: {context['pilot_pass']}",
        f"CURRENT_BASELINE_unchanged: {context['current_baseline_unchanged']}",
        f"data_staged: {context['data_staged']}",
        "",
        "policy: 25.10 may start only as an explicit component-set training gate; no CURRENT_BASELINE.md transition is allowed here.",
        f"decision_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    _selected, rows, context = decide()
    write_csv(MATRIX, rows, FIELDS)
    write_summary(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
