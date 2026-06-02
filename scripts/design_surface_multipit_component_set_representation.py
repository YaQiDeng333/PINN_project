#!/usr/bin/env python
"""Design 25.9 surface multi-pit component-set representations.

This script is plan-only. It reads the 25.9 label audit and writes
representation planning artifacts only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CSV = ROOT / "results/metrics/surface_multipit_component_label_audit.csv"
MISSING_FIELDS_CSV = ROOT / "results/metrics/surface_multipit_component_label_missing_fields.csv"

SUMMARY = ROOT / "results/summaries/surface_multipit_component_set_representation_summary.txt"
MATRIX = ROOT / "results/metrics/surface_multipit_component_set_representation_matrix.csv"

FIELDS = [
    "route_id",
    "representation",
    "selected_first",
    "output_contract",
    "required_labels",
    "matching_policy",
    "loss_terms",
    "metrics",
    "advantages",
    "blockers_or_deferred_work",
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


def build_rows() -> list[dict[str, Any]]:
    return [
        {
            "route_id": "C1",
            "representation": "fixed_K_component_set",
            "selected_first": True,
            "output_contract": "K=3 slots; per slot existence_prob, center_x_m, center_y_m, L_m, W_m, D_m, rotation_angle, shape_family, local_profile_params",
            "required_labels": "component_count; component_params_json; component centers/L/W/D; rotation; shape_family; projected_mask_2d; depth_grid_m",
            "matching_policy": "Hungarian matching over active components using weighted center distance, size/depth error, shape_family mismatch, and optional component-mask IoU",
            "loss_terms": "existence BCE; matched component SmoothL1 for center/L/W/D/rotation; shape_family CE; local_profile_params SmoothL1; union mask Dice/BCE; depth-grid Huber",
            "metrics": "component recall; missed/merged/extra component rate; matched center error; L/W/D relative MAE; union mask Dice/IoU; depth-grid RMSE",
            "advantages": "directly addresses two-component representation failure while preserving RBC branch boundaries",
            "blockers_or_deferred_work": "needs top-up labels for per-component rotation and component-level masks/depth grids",
            "decision": "recommended_first",
        },
        {
            "route_id": "C2",
            "representation": "component_set_plus_raster",
            "selected_first": False,
            "output_contract": "C1 component slots plus auxiliary union projected_mask_2d and depth_grid_m decoder heads",
            "required_labels": "all C1 labels plus reliable raster targets for mask/depth supervision",
            "matching_policy": "Hungarian for components; raster losses applied to union targets after slot rasterization",
            "loss_terms": "C1 losses plus projected mask Dice/BCE and depth-grid RMSE/Huber as auxiliary heads",
            "metrics": "C1 metrics plus raster mask/depth agreement",
            "advantages": "reduces valid-parameter but wrong-union failures",
            "blockers_or_deferred_work": "more moving parts; should follow C1 if close/touching cases remain weak",
            "decision": "second_route_after_C1_diagnostic",
        },
        {
            "route_id": "C3",
            "representation": "component_heatmap_plus_param_head",
            "selected_first": False,
            "output_contract": "component-center heatmap with local param regression for each detected peak",
            "required_labels": "component centers; component existence; per-component L/W/D/depth/rotation; optionally component masks",
            "matching_policy": "peak-to-label matching with Hungarian or local nearest-neighbor after non-max suppression",
            "loss_terms": "heatmap focal/MSE; param SmoothL1 on matched peaks; union mask/depth auxiliary losses",
            "metrics": "center detection recall; duplicate/extra peaks; component param error",
            "advantages": "better suited if fixed slots confuse close or partially overlapping components",
            "blockers_or_deferred_work": "requires detector-style implementation; not needed before top-up evidence",
            "decision": "defer",
        },
        {
            "route_id": "C4",
            "representation": "depth_grid_decoder_baseline",
            "selected_first": False,
            "output_contract": "predict full depth_grid_m and projected_mask_2d without explicit component identities",
            "required_labels": "depth_grid_m; projected_mask_2d; optional component labels only for evaluation",
            "matching_policy": "no component matching in training; post-hoc connected-component extraction for metrics",
            "loss_terms": "depth-grid RMSE/Huber; mask Dice/BCE; edge regularity",
            "metrics": "mask Dice/IoU; depth-grid RMSE; post-hoc component merge/missed rate",
            "advantages": "fallback when component labels are unstable or COMSOL component IDs are unreliable",
            "blockers_or_deferred_work": "does not solve component identity by itself and should not get component-set success credit",
            "decision": "fallback_only",
        },
    ]


def write_summary(audit_rows: list[dict[str, str]], missing_rows: list[dict[str, str]]) -> None:
    c1_seed_ready = bool(audit_rows) and all(row["label_sufficiency"] == "sufficient_for_C1_seed_with_schema_gaps" for row in audit_rows)
    missing_names = [row["field_name"] for row in missing_rows if row.get("severity") != "not_blocking"]
    lines = [
        "25.9 surface multi-pit component-set representation design",
        "",
        "primary_representation: C1 fixed_K_component_set",
        "K_default: 3",
        "current_seed_component_count: 2",
        f"current_16_C1_seed_ready: {c1_seed_ready}",
        "per_component_output: existence_prob, center_x_m, center_y_m, L_m, W_m, D_m, rotation_angle, shape_family, local_profile_params",
        "matching: Hungarian matching with permutation-invariant component losses.",
        "loss: existence BCE + component parameter SmoothL1 + shape-family CE + union projected-mask Dice/BCE + depth-grid Huber.",
        "negative_control_policy: 20.85 single-RBC and forward-refinement runner remain comparators only for multi-pit.",
        "",
        "schema_gaps_to_close_in_topup:",
        "- " + "\n- ".join(missing_names),
        "",
        "route_order:",
        "1. C1 fixed-K component-set.",
        "2. C2 component-set plus raster if C1 geometry is valid but union mask/depth is weak.",
        "3. C3 heatmap plus param head if close/touching cases cause slot confusion.",
        "4. C4 depth-grid decoder baseline only if component labels cannot be stabilized.",
        f"representation_matrix: {MATRIX}",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not AUDIT_CSV.exists() or not MISSING_FIELDS_CSV.exists():
        raise FileNotFoundError("run audit_surface_multipit_component_labels.py before representation design")
    audit_rows = read_csv(AUDIT_CSV)
    missing_rows = read_csv(MISSING_FIELDS_CSV)
    rows = build_rows()
    write_csv(MATRIX, rows, FIELDS)
    write_summary(audit_rows, missing_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
