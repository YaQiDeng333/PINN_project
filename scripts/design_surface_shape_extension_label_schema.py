#!/usr/bin/env python
"""Design the 25.1 surface shape-extension label schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summaries/surface_shape_extension_label_schema_summary.txt"
SCHEMA_CSV = ROOT / "results/metrics/surface_shape_extension_label_schema.csv"

FIELDS = [
    "label_name",
    "required",
    "dtype_or_shape",
    "applies_to",
    "allowed_values_or_rule",
    "leakage_policy",
    "representation_policy",
    "notes",
]

SCHEMA_ROWS: list[dict[str, Any]] = [
    {
        "label_name": "sample_id",
        "required": True,
        "dtype_or_shape": "string",
        "applies_to": "all",
        "allowed_values_or_rule": "unique stable id; no latest/newest discovery",
        "leakage_policy": "join_key_only_not_model_input",
        "representation_policy": "metadata",
        "notes": "must remain explicit in manifest and metrics",
    },
    {
        "label_name": "shape_type",
        "required": True,
        "dtype_or_shape": "categorical string",
        "applies_to": "all",
        "allowed_values_or_rule": "taxonomy shape_type values only",
        "leakage_policy": "allowed for stratification/audit; model input only in explicitly shape-conditioned routes",
        "representation_policy": "taxonomy",
        "notes": "separates RBC-like controls from non-RBC-like surface defects",
    },
    {
        "label_name": "topology_type",
        "required": True,
        "dtype_or_shape": "categorical string",
        "applies_to": "all",
        "allowed_values_or_rule": "single_component|multi_component|elongated|irregular",
        "leakage_policy": "target/audit label; not implicit input",
        "representation_policy": "topology",
        "notes": "used for split coverage and topology failure metrics",
    },
    {
        "label_name": "L_m",
        "required": True,
        "dtype_or_shape": "float meters",
        "applies_to": "all",
        "allowed_values_or_rule": "positive footprint major length",
        "leakage_policy": "supervision/metric only",
        "representation_policy": "global_size",
        "notes": "for non-RBC-like shapes this is a bounding descriptor, not a sufficient representation",
    },
    {
        "label_name": "W_m",
        "required": True,
        "dtype_or_shape": "float meters",
        "applies_to": "all",
        "allowed_values_or_rule": "positive footprint minor width",
        "leakage_policy": "supervision/metric only",
        "representation_policy": "global_size",
        "notes": "crack-like cases must preserve high aspect_ratio separately",
    },
    {
        "label_name": "D_m",
        "required": True,
        "dtype_or_shape": "float meters",
        "applies_to": "all",
        "allowed_values_or_rule": "positive max depth",
        "leakage_policy": "supervision/metric only",
        "representation_policy": "global_depth",
        "notes": "not enough for flat-bottom/asymmetric/irregular profile shape",
    },
    {
        "label_name": "center_xyz_m",
        "required": True,
        "dtype_or_shape": "3 floats meters",
        "applies_to": "all",
        "allowed_values_or_rule": "[center_x_m,center_y_m,center_z_m] in COMSOL coordinate convention",
        "leakage_policy": "supervision/metric only unless route explicitly conditions on known pose",
        "representation_policy": "pose",
        "notes": "surface origin remains z=0 scan surface",
    },
    {
        "label_name": "surface_origin",
        "required": True,
        "dtype_or_shape": "string or 3 floats",
        "applies_to": "all",
        "allowed_values_or_rule": "top_z_0 with explicit coordinate convention",
        "leakage_policy": "metadata",
        "representation_policy": "coordinate_frame",
        "notes": "prevents mixing surface and internal-defect conventions",
    },
    {
        "label_name": "depth_grid_m",
        "required": True,
        "dtype_or_shape": "2D float array meters",
        "applies_to": "all",
        "allowed_values_or_rule": "finite nonnegative depth grid; max approximately D_m",
        "leakage_policy": "target only; never input feature for inverse model",
        "representation_policy": "depth_grid",
        "notes": "mandatory for profile/depth metrics and irregular corrosion",
    },
    {
        "label_name": "projected_mask_2d",
        "required": True,
        "dtype_or_shape": "2D uint8/bool array",
        "applies_to": "all",
        "allowed_values_or_rule": "depth_grid_m > threshold; nonempty for defect sample",
        "leakage_policy": "target/QA only",
        "representation_policy": "projected_mask",
        "notes": "IoU/Dice remain QA, not replacement for profile metrics",
    },
    {
        "label_name": "profile_descriptor",
        "required": True,
        "dtype_or_shape": "JSON object",
        "applies_to": "all",
        "allowed_values_or_rule": "family-specific profile basis, contour, or depth-grid descriptor",
        "leakage_policy": "target metadata",
        "representation_policy": "profile_basis_or_descriptor",
        "notes": "RBC-like may store six-param descriptor; non-RBC stores richer target",
    },
    {
        "label_name": "component_count",
        "required": True,
        "dtype_or_shape": "integer",
        "applies_to": "all",
        "allowed_values_or_rule": ">=1; multi-pit requires 2 for first pilot",
        "leakage_policy": "target/audit label",
        "representation_policy": "component_set",
        "notes": "used for component recall and merge-rate metrics",
    },
    {
        "label_name": "component_params_json",
        "required": True,
        "dtype_or_shape": "JSON list/object",
        "applies_to": "multi_component and all component-capable shapes",
        "allowed_values_or_rule": "component-level L/W/D/center/profile fields",
        "leakage_policy": "target metadata; not model input unless explicit component-conditioned route",
        "representation_policy": "component_set",
        "notes": "multi-pit must keep component-level labels",
    },
    {
        "label_name": "edge_steepness",
        "required": True,
        "dtype_or_shape": "float proxy",
        "applies_to": "all",
        "allowed_values_or_rule": "finite edge gradient or wall steepness proxy",
        "leakage_policy": "target/metric only",
        "representation_policy": "edge_metric",
        "notes": "also covers sharpness proxy requested by plan",
    },
    {
        "label_name": "asymmetry_score",
        "required": True,
        "dtype_or_shape": "float [0,1] preferred",
        "applies_to": "all",
        "allowed_values_or_rule": "0 for symmetric controls; higher for skewed depth distribution",
        "leakage_policy": "target/stratification only",
        "representation_policy": "profile_metric",
        "notes": "asymmetric corrosion must have nonzero expected range",
    },
    {
        "label_name": "aspect_ratio",
        "required": True,
        "dtype_or_shape": "float",
        "applies_to": "all",
        "allowed_values_or_rule": "L_m / W_m after rotation-aware major/minor ordering",
        "leakage_policy": "target/stratification only",
        "representation_policy": "shape_metric",
        "notes": "crack-like defects require high aspect_ratio coverage",
    },
    {
        "label_name": "rotation_angle",
        "required": True,
        "dtype_or_shape": "float radians or degrees with unit suffix",
        "applies_to": "elongated, boxy, flat-bottom, multi-component where applicable",
        "allowed_values_or_rule": "finite angle; may be 0.0 for axis-aligned controls",
        "leakage_policy": "target/metric only",
        "representation_policy": "pose",
        "notes": "required for crack-like and rotated polygon/primitive routes",
    },
    {
        "label_name": "rbc_compatible",
        "required": True,
        "dtype_or_shape": "boolean",
        "applies_to": "all",
        "allowed_values_or_rule": "true only for rbc_like_smooth_pit",
        "leakage_policy": "target/route gating label",
        "representation_policy": "route_boundary",
        "notes": "prevents non-RBC-like cases from being forced into six-param RBC",
    },
    {
        "label_name": "representation_target",
        "required": True,
        "dtype_or_shape": "categorical string",
        "applies_to": "all",
        "allowed_values_or_rule": "six_param_rbc|profile_basis|depth_grid|component_set|polygon_or_contour",
        "leakage_policy": "target/route selection label",
        "representation_policy": "output_contract",
        "notes": "model output family is chosen from this field",
    },
    {
        "label_name": "forward_consistency_required",
        "required": True,
        "dtype_or_shape": "boolean",
        "applies_to": "all",
        "allowed_values_or_rule": "true for pilot generation and model gate rows",
        "leakage_policy": "gate metadata",
        "representation_policy": "physics_gate",
        "notes": "predicted profile -> forward surrogate -> Bx/By/Bz residual in later stages",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 25.1 surface shape-extension label schema.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--schema", type=Path, default=SCHEMA_CSV)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    write_csv(args.schema, SCHEMA_ROWS, FIELDS)
    required_count = sum(1 for row in SCHEMA_ROWS if row["required"])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "surface shape-extension label schema summary",
                "stage: 25.1",
                "",
                f"label_count: {len(SCHEMA_ROWS)}",
                f"required_label_count: {required_count}",
                "rbc_like_policy: rbc_like_smooth_pit may continue six_param_rbc labels.",
                "non_rbc_policy: non-RBC-like shape families must not be forced into L/W/D/wLD/wWD/wLW as a complete representation.",
                "multi_pit_policy: component_count and component_params_json are mandatory component-level labels.",
                "crack_like_policy: aspect_ratio and rotation_angle are mandatory.",
                "irregular_policy: depth_grid_m and profile_descriptor remain the primary target.",
                "forward_consistency_policy: every pilot-ready sample must be capable of later forward residual evaluation.",
                "leakage_policy: sample_id, split, shape_type, topology_type, component labels, and representation_target are targets/metadata unless a later route explicitly authorizes them as inputs.",
                "",
                f"schema_csv: {args.schema}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
