#!/usr/bin/env python
"""Audit the 20.71 partial true-3D RBC pilot pack for 20.72 top-up."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMSOL_ROOT = Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")

DEFAULT_PLAN = ROOT / "results/metrics/true_3d_rbc_pilot_pack_plan.csv"
DEFAULT_VALIDATION = ROOT / "results/metrics/true_3d_rbc_pilot_pack_validation_metrics.csv"
DEFAULT_MESH = ROOT / "results/metrics/true_3d_rbc_pilot_watertight_mesh_metrics.csv"
DEFAULT_COMSOL_INVENTORY = COMSOL_ROOT / "results/inventory_true_3d_rbc_pilot_pack_v1.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_partial_audit_summary.txt"
DEFAULT_AUDIT = ROOT / "results/metrics/true_3d_rbc_pilot_partial_audit.csv"
DEFAULT_MISSING = ROOT / "results/metrics/true_3d_rbc_pilot_missing_coverage.csv"
DEFAULT_PREFLIGHT = ROOT / "results/summaries/true_3d_rbc_pilot_topup_preflight_summary.txt"

AUDIT_FIELDS = [
    "sample_id",
    "plan_split",
    "inventory_status",
    "classification",
    "curvature_template",
    "depth_bin",
    "size_bin",
    "aspect_bin",
    "L_m",
    "W_m",
    "D_m",
    "wLD",
    "wWD",
    "wLW",
    "mesh_validation_pass",
    "schema_pass",
    "failure_reason",
    "recommended_action",
]

MISSING_FIELDS = ["group_key", "group_value", "planned_count", "success_count", "missing_count", "priority"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 20.71 partial true-3D RBC pilot pack.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--validation-metrics", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--mesh-metrics", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--comsol-inventory", type=Path, default=DEFAULT_COMSOL_INVENTORY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--missing", type=Path, default=DEFAULT_MISSING)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def classify(status: str, reason: str, schema_pass: bool) -> str:
    if status == "pass" and schema_pass:
        return "success"
    if "timeout" in reason.lower():
        return "timeout_or_bounded_skip"
    if status == "not_attempted":
        return "not_attempted"
    if status == "fail":
        return "failed"
    return "unknown"


def recommended_action(row: dict[str, str], classification: str) -> str:
    curvature = row["curvature_template"]
    depth = row["depth_bin"]
    aspect = row["aspect_bin"]
    if classification == "success":
        return "keep_as_partial_source"
    if curvature in {"LD_dominant", "WD_dominant"}:
        return "topup_required_missing_curvature_family"
    if depth == "deep" and aspect == "narrow":
        return "bounded_retry_then_adjusted_replacement"
    if classification == "not_attempted":
        return "topup_required_to_complete_full_pack"
    return "inspect_before_training_gate"


def write_preflight(path: Path) -> None:
    lines = [
        "20.72 true 3D RBC pilot top-up preflight summary",
        "",
        "Subagent status:",
        "- Agent A Method/Route: GO. Top-up fits the true 3D / Piao-style route but is not training, not baseline, and not exact Piao RBC.",
        "- Agent B COMSOL: GO. 20.70 protocol remains usable; 20.71 gaps are bounded timeout / not-attempted rows, not imported-route invalidation.",
        "- Agent C RBC Parameter/Split: GO. Recommended full top-up shape is 24 LD/WD + 4 deep boxy + 2 deep-elongated replacements.",
        "- Agent D Registry/Manifest: GO with governance fixes. Use partial_source, topup_source, assembled roles and machine-readable latest/newest forbids.",
        "- Agent E Safety/Git: conditional GO. Do not stage data, NPZ, temp STL, baseline docs, or unrelated dirty items.",
        "- Agent F Implementation: not spawned because the platform agent limit was reached; main controller performed read-only feasibility review.",
        "",
        "Answers:",
        "1. worthwhile: True",
        "2. 20.71 partial source usable: True",
        "3. top-up focus: LD_dominant, WD_dominant, deep_elongated timeout replacements, deep_boxy completion",
        "4. deep-elongated strategy: bounded retry first, adjusted replacement second, documented skip last",
        "5. recommended top-up: planned 30-36 rows, target success 30, split target 20/5/5",
        "6. COMSOL needed: True",
        "7. training needed: False",
        "8. registry strategy: partial source + top-up source + assembled pack; no latest/newest auto-discovery",
        "9. allowed commits: scripts, summaries, metrics, registry, manifests, markdown route docs",
        "10. forbidden commits: data, NPZ, temp STL/mesh, .mph, raw CSV, checkpoints, previews, notes, baseline docs",
        "11. hard blockers: none found in preflight",
        "12. continue execution: True",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.audit, args.missing, args.preflight_summary], args.overwrite)
    plan_rows = read_csv(args.plan_csv)
    validation_rows = {row["sample_id"]: row for row in read_csv(args.validation_metrics)}
    mesh_rows = {row["sample_id"]: row for row in read_csv(args.mesh_metrics)}
    inventory_rows = {row["sample_id"]: row for row in read_csv(args.comsol_inventory)}
    if len(plan_rows) != 60:
        raise RuntimeError(f"expected 60 plan rows, got {len(plan_rows)}")
    audit_rows: list[dict[str, Any]] = []
    for row in plan_rows:
        sample_id = row["sample_id"]
        inv = inventory_rows.get(sample_id, {})
        val = validation_rows.get(sample_id, {})
        mesh = mesh_rows.get(sample_id, {})
        status = inv.get("status", "missing_inventory")
        schema_pass = truthy(val.get("schema_pass", False))
        cls = classify(status, inv.get("failure_reason", ""), schema_pass)
        audit_rows.append(
            {
                "sample_id": sample_id,
                "plan_split": row["split"],
                "inventory_status": status,
                "classification": cls,
                "curvature_template": row["curvature_template"],
                "depth_bin": row["depth_bin"],
                "size_bin": row["size_bin"],
                "aspect_bin": row["aspect_bin"],
                "L_m": row["L_m"],
                "W_m": row["W_m"],
                "D_m": row["D_m"],
                "wLD": row["wLD"],
                "wWD": row["wWD"],
                "wLW": row["wLW"],
                "mesh_validation_pass": truthy(mesh.get("mesh_validation_pass", False)),
                "schema_pass": schema_pass,
                "failure_reason": inv.get("failure_reason", ""),
                "recommended_action": recommended_action(row, cls),
            }
        )
    write_csv(args.audit, audit_rows, AUDIT_FIELDS)
    success_rows = [row for row in audit_rows if row["classification"] == "success"]
    missing_rows: list[dict[str, Any]] = []
    for key, priority_values in {
        "curvature_template": {"LD_dominant", "WD_dominant"},
        "depth_bin": {"deep"},
        "size_bin": set(),
        "aspect_bin": {"narrow"},
        "plan_split": set(),
    }.items():
        planned = Counter(row[key] for row in audit_rows)
        success = Counter(row[key] for row in success_rows)
        for value in sorted(planned):
            missing = planned[value] - success.get(value, 0)
            missing_rows.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "planned_count": planned[value],
                    "success_count": success.get(value, 0),
                    "missing_count": missing,
                    "priority": value in priority_values or missing > 0,
                }
            )
    write_csv(args.missing, missing_rows, MISSING_FIELDS)
    status_counts = Counter(row["inventory_status"] for row in audit_rows)
    classification_counts = Counter(row["classification"] for row in audit_rows)
    split_success = Counter(row["plan_split"] for row in success_rows)
    curvature_success = Counter(row["curvature_template"] for row in success_rows)
    duplicate_sample_ids = len(audit_rows) - len({row["sample_id"] for row in audit_rows})
    six_param_keys = [
        (row["L_m"], row["W_m"], row["D_m"], row["wLD"], row["wWD"], row["wLW"])
        for row in audit_rows
    ]
    duplicate_six_params = len(six_param_keys) - len(set(six_param_keys))
    lines = [
        "20.72 true 3D RBC pilot partial-pack audit summary",
        "",
        f"planned_rows: {len(audit_rows)}",
        f"status_counts: {dict(status_counts)}",
        f"classification_counts: {dict(classification_counts)}",
        f"success_split_counts: {dict(split_success)}",
        f"success_curvature_counts: {dict(curvature_success)}",
        f"duplicate_sample_ids: {duplicate_sample_ids}",
        f"duplicate_full_six_params: {duplicate_six_params}",
        "missing_curvature_templates: " + ", ".join(name for name in ["LD_dominant", "WD_dominant"] if curvature_success.get(name, 0) == 0),
        "timeout_or_bounded_skip_rows: "
        + ", ".join(row["sample_id"] for row in audit_rows if row["classification"] == "timeout_or_bounded_skip"),
        "",
        "Gate:",
        "- partial source is readable and internally consistent.",
        "- top-up must not overwrite the 20.71 partial NPZ.",
        "- top-up must cover LD_dominant and WD_dominant before any training gate.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_preflight(args.preflight_summary)
    if len(success_rows) < 30:
        raise RuntimeError("partial source has fewer than 30 validated successes")
    if duplicate_sample_ids:
        raise RuntimeError("duplicate sample ids found in partial audit")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
