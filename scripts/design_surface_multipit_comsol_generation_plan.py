#!/usr/bin/env python
"""Design the 25.9 surface multi-pit COMSOL generation plan.

This script records feasibility only. It does not open COMSOL, invoke a
generator, create data, or mutate NPZ files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
SOURCE_FEASIBILITY = ROOT / "results/metrics/surface_shape_extension_comsol_feasibility_matrix.csv"
TOPUP_PLAN = ROOT / "results/metrics/surface_multipit_dataset_topup_plan.csv"

SUMMARY = ROOT / "results/summaries/surface_multipit_comsol_generation_plan_summary.txt"
MATRIX = ROOT / "results/metrics/surface_multipit_comsol_generation_feasibility.csv"

FIELDS = [
    "route_id",
    "generation_route",
    "recommended",
    "feasible_for_25_10",
    "boolean_risk",
    "mesh_risk",
    "solver_risk",
    "label_extraction",
    "no_defect_reference_reuse",
    "BxByBz_export_required",
    "failure_recording_required",
    "decision",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_multi_pit_row() -> dict[str, str]:
    rows = read_csv(SOURCE_FEASIBILITY)
    for row in rows:
        if row.get("shape_type") == "multi_pit_two_component_surface_defect":
            return row
    raise RuntimeError("missing multi-pit row in surface_shape_extension_comsol_feasibility_matrix.csv")


def build_rows(source: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "route_id": "G1",
            "generation_route": "two_or_three_component_boolean_subtract",
            "recommended": True,
            "feasible_for_25_10": True,
            "boolean_risk": "medium for separated/close; high for touching/overlap",
            "mesh_risk": source.get("mesh_risk", "medium_high_when_components_close"),
            "solver_risk": source.get("solver_risk", "medium_high"),
            "label_extraction": "component_params_json plus per-component masks/depth grids plus union projected_mask_2d/depth_grid_m",
            "no_defect_reference_reuse": True,
            "BxByBz_export_required": True,
            "failure_recording_required": "record Boolean failure, mesh failure, solver failure, merged-domain label mismatch, and component-count mismatch",
            "decision": "recommended_first",
        },
        {
            "route_id": "G2",
            "generation_route": "touching_overlap_stress_subset",
            "recommended": True,
            "feasible_for_25_10": True,
            "boolean_risk": "high",
            "mesh_risk": "high",
            "solver_risk": "medium_high",
            "label_extraction": "explicit topology_relation plus component masks are mandatory",
            "no_defect_reference_reuse": True,
            "BxByBz_export_required": True,
            "failure_recording_required": "keep failed cases in feasibility ledger; do not silently replace with easy separated cases",
            "decision": "include_as_coverage_stress_not_first_smoke_only",
        },
        {
            "route_id": "G3",
            "generation_route": "three_component_future_negative",
            "recommended": True,
            "feasible_for_25_10": True,
            "boolean_risk": "medium_high",
            "mesh_risk": "medium_high",
            "solver_risk": "medium_high",
            "label_extraction": "K=3 component labels; all components require existence/center/L/W/D/rotation/shape_family",
            "no_defect_reference_reuse": True,
            "BxByBz_export_required": True,
            "failure_recording_required": "record slot count, component id stability, and merge/miss ambiguity",
            "decision": "small_fraction_only",
        },
        {
            "route_id": "G4",
            "generation_route": "imported_watertight_mesh_per_component",
            "recommended": False,
            "feasible_for_25_10": False,
            "boolean_risk": "lower Boolean dependency but higher mesh authoring burden",
            "mesh_risk": "high",
            "solver_risk": "medium",
            "label_extraction": "component mesh metadata must be preserved manually",
            "no_defect_reference_reuse": True,
            "BxByBz_export_required": True,
            "failure_recording_required": "record import/mesh metadata mismatch",
            "decision": "fallback_only",
        },
        {
            "route_id": "G5",
            "generation_route": "height_depth_map_solid_union_only",
            "recommended": False,
            "feasible_for_25_10": False,
            "boolean_risk": "low",
            "mesh_risk": "medium",
            "solver_risk": "medium",
            "label_extraction": "union depth grid easy; component identity weak",
            "no_defect_reference_reuse": True,
            "BxByBz_export_required": True,
            "failure_recording_required": "record loss of component identity",
            "decision": "not_recommended_for_component_set_labels",
        },
    ]


def write_summary(manifest: dict[str, Any], source: dict[str, str]) -> None:
    generator = Path(manifest.get("comsol_generator_script", ""))
    lines = [
        "25.9 surface multi-pit COMSOL generation plan",
        "",
        "plan_only: true",
        "COMSOL_run: false",
        "data_or_NPZ_generation: false",
        f"source_dataset_id: {manifest.get('dataset_id')}",
        f"read_only_generator_reference_exists: {generator.exists()}",
        f"read_only_generator_reference: {generator}",
        f"25.1_preferred_route: {source.get('preferred_generation_route')}",
        f"25.1_boolean_feasibility: {source.get('boolean_feasibility')}",
        f"25.1_mesh_risk: {source.get('mesh_risk')}",
        f"25.1_solver_risk: {source.get('solver_risk')}",
        "",
        "recommended_generation_route: G1 two_or_three_component_boolean_subtract",
        "required_exports: Bx, By, Bz, b_no_defect, delta_b, union projected_mask_2d, union depth_grid_m, component_params_json, component-level masks/depth grids.",
        "required_failure_ledger: Boolean, mesh, solver, component-count mismatch, merged-domain, label/union mismatch.",
        "no_defect_reference_policy: reusable across component top-up rows only when material/domain/solver settings match.",
        "20.70_material_domain_solver_fix_applies: true",
        f"feasibility_csv: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not TOPUP_PLAN.exists():
        raise FileNotFoundError("run design_surface_multipit_dataset_topup_plan.py before COMSOL generation design")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = source_multi_pit_row()
    rows = build_rows(source)
    write_csv(MATRIX, rows, FIELDS)
    write_summary(manifest, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
