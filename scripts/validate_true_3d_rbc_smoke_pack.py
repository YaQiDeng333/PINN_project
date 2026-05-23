#!/usr/bin/env python
"""Validate the 20.66 true-3D RBC-style smoke NPZ schema.

This script audits only the smoke pack schema and label/field consistency. It
does not train a model, does not run refinement, and does not update baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "data/comsol_mfl/prepared/true_3d_rbc_smoke_pack_v1.npz"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_smoke_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_rbc_smoke_validation_metrics.csv"
DEFAULT_ROUTE_SUMMARY = ROOT / "results/summaries/true_3d_rbc_smoke_route_decision_summary.txt"
DEFAULT_ROUTE_MATRIX = ROOT / "results/metrics/true_3d_rbc_smoke_route_decision_matrix.csv"

AXIS_NAMES = ("Bx", "By", "Bz")
MIN_SUCCESSFUL_SAMPLES = 3

METRIC_FIELDS = [
    "sample_id",
    "schema_pass",
    "delta_b_shape",
    "b_defect_shape",
    "b_no_defect_shape",
    "projected_mask_shape",
    "profile_depth_grid_shape",
    "profile_depth_map_shape",
    "rbc_params_shape",
    "axis_names",
    "axis_expressions",
    "all_values_finite",
    "delta_max_abs_error",
    "projected_mask_area_px",
    "depth_max_m",
    "param_D_m",
    "profile_depth_max_error_vs_param_D",
    "depth_volume_proxy_m3",
    "Bx_norm",
    "By_norm",
    "Bz_norm",
    "Bx_ptp",
    "By_ptp",
    "Bz_ptp",
    "no_defect_vs_defect_nonzero",
    "stepped_depth_approximation",
    "exact_piao_rbc",
    "notes",
]

ROUTE_FIELDS = [
    "decision_option",
    "selected",
    "condition",
    "observed",
    "next_step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate true-3D RBC smoke NPZ.")
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing files:\n{joined}")


def load_metadata(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    raw = npz["metadata"]
    if raw.shape == ():
        return json.loads(str(raw.item()))
    return json.loads(str(raw.tolist()))


def arr_shape(value: np.ndarray) -> str:
    return json.dumps(list(value.shape), separators=(",", ":"))


def validate_sample(
    index: int,
    sample_id: str,
    delta_b: np.ndarray,
    b_defect: np.ndarray,
    b_no_defect: np.ndarray,
    mask: np.ndarray,
    depth_grid: np.ndarray,
    depth_map: np.ndarray,
    rbc_params: np.ndarray,
    stepped: bool,
    exact_piao: bool,
    axis_names: list[str],
    axis_expressions: list[str],
) -> dict[str, Any]:
    delta_error = float(np.max(np.abs(delta_b - (b_defect - b_no_defect))))
    all_finite = bool(
        np.isfinite(delta_b).all()
        and np.isfinite(b_defect).all()
        and np.isfinite(b_no_defect).all()
        and np.isfinite(depth_grid).all()
        and np.isfinite(depth_map).all()
    )
    param_d = float(rbc_params[0, 2]) if rbc_params.ndim == 2 else float(rbc_params[2])
    depth_max = float(max(float(depth_grid.max()), float(depth_map.max())))
    depth_error = abs(depth_max - param_d)
    norms = [float(np.linalg.norm(delta_b[i])) for i in range(delta_b.shape[0])]
    ptp = [float(np.ptp(delta_b[i])) for i in range(delta_b.shape[0])]
    expected_signal_shape = (3, 3, 201)
    schema_pass = (
        delta_b.shape == expected_signal_shape
        and b_defect.shape == expected_signal_shape
        and b_no_defect.shape == expected_signal_shape
        and mask.shape == (64, 128)
        and depth_map.shape == (64, 128)
        and depth_grid.ndim == 2
        and rbc_params.shape == (1, 6)
        and axis_names == list(AXIS_NAMES)
        and all_finite
        and delta_error <= 1.0e-12
        and int(mask.sum()) > 0
        and depth_error / max(param_d, 1.0e-12) <= 0.05
        and bool(np.any(np.abs(delta_b) > 0.0))
    )
    return {
        "sample_id": sample_id,
        "schema_pass": schema_pass,
        "delta_b_shape": arr_shape(delta_b),
        "b_defect_shape": arr_shape(b_defect),
        "b_no_defect_shape": arr_shape(b_no_defect),
        "projected_mask_shape": arr_shape(mask),
        "profile_depth_grid_shape": arr_shape(depth_grid),
        "profile_depth_map_shape": arr_shape(depth_map),
        "rbc_params_shape": arr_shape(rbc_params),
        "axis_names": json.dumps(axis_names),
        "axis_expressions": json.dumps(axis_expressions),
        "all_values_finite": all_finite,
        "delta_max_abs_error": delta_error,
        "projected_mask_area_px": int(mask.sum()),
        "depth_max_m": depth_max,
        "param_D_m": param_d,
        "profile_depth_max_error_vs_param_D": depth_error,
        "depth_volume_proxy_m3": float(depth_map.sum()),
        "Bx_norm": norms[0],
        "By_norm": norms[1],
        "Bz_norm": norms[2],
        "Bx_ptp": ptp[0],
        "By_ptp": ptp[1],
        "Bz_ptp": ptp[2],
        "no_defect_vs_defect_nonzero": bool(np.linalg.norm(delta_b) > 0.0),
        "stepped_depth_approximation": stepped,
        "exact_piao_rbc": exact_piao,
        "notes": "projected_mask_2d is comparator only; 3D label uses rbc_params plus depth grid/map",
    }


def determine_status(metrics: list[dict[str, Any]], metadata: dict[str, Any], stepped: np.ndarray) -> str:
    pass_count = sum(1 for row in metrics if str(row["schema_pass"]) == "True" or row["schema_pass"] is True)
    smooth_verified = bool(metadata.get("smooth_variable_depth_solid_verified", False))
    if pass_count < MIN_SUCCESSFUL_SAMPLES:
        return "failed"
    if smooth_verified:
        return "variable_depth_pass"
    if bool(np.asarray(stepped, dtype=bool).all()):
        return "stepped_depth_smoke_pass"
    return "failed"


def write_summary(path: Path, metrics: list[dict[str, Any]], metadata: dict[str, Any], smoke_status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pass_count = sum(1 for row in metrics if str(row["schema_pass"]) == "True" or row["schema_pass"] is True)
    lines = [
        "20.66 true 3D RBC-style smoke validation summary",
        "",
        f"sample_count: {len(metrics)}",
        f"schema_pass_count: {pass_count}",
        f"minimum_acceptable_samples: {MIN_SUCCESSFUL_SAMPLES}",
        f"smoke_status: {smoke_status}",
        f"geometry_implementation: {metadata.get('geometry_implementation')}",
        f"smooth_variable_depth_solid_verified: {metadata.get('smooth_variable_depth_solid_verified')}",
        f"stepped_depth_approximation: {metadata.get('stepped_depth_approximation')}",
        f"constant_depth_extrusion_used_as_success: {metadata.get('constant_depth_extrusion_used_as_success')}",
        f"exact_piao_rbc: {metadata.get('exact_piao_rbc')}",
        "axis_names: [Bx, By, Bz]",
        "sensor_z_m: 0.008",
        "",
        "Validation result:",
        "- The pack is a stepped-depth smoke pass, not a smooth variable-depth pass." if smoke_status == "stepped_depth_smoke_pass" else f"- {smoke_status}",
        "- No training, surrogate, refinement, or baseline update is performed by this validation.",
        "- projected_mask_2d is retained only as a 2D comparator; the 3D label remains rbc_params plus depth/profile map metadata.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def route_rows(smoke_status: str) -> list[dict[str, Any]]:
    return [
        {
            "decision_option": "A_variable_depth_pass",
            "selected": smoke_status == "variable_depth_pass",
            "condition": "smooth true variable-depth solid succeeds with schema validation",
            "observed": smoke_status,
            "next_step": "design 60-sample true 3D RBC pilot",
        },
        {
            "decision_option": "B_stepped_depth_smoke_pass",
            "selected": smoke_status == "stepped_depth_smoke_pass",
            "condition": "stepped-depth layered approximation succeeds but smooth variable-depth is not verified",
            "observed": smoke_status,
            "next_step": "decide whether to improve smooth variable-depth geometry or accept stepped-depth as pilot approximation",
        },
        {
            "decision_option": "C_geometry_failed",
            "selected": smoke_status == "failed",
            "condition": "constant-depth only, fewer than 3 valid samples, geometry construction failure, or schema failure",
            "observed": smoke_status,
            "next_step": "fix COMSOL geometry generator before any pilot or training",
        },
    ]


def write_route_summary(path: Path, smoke_status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if smoke_status == "variable_depth_pass":
        recommendation = "A. Generate a 60-sample true 3D RBC pilot."
    elif smoke_status == "stepped_depth_smoke_pass":
        recommendation = "B. Decide whether to improve smooth variable-depth COMSOL geometry or accept stepped-depth as a clearly labeled pilot approximation."
    else:
        recommendation = "C. Fix the COMSOL geometry/schema blocker before any pilot or training."
    lines = [
        "20.66 true 3D RBC-style smoke route decision",
        "",
        f"smoke_status: {smoke_status}",
        f"next_step_unique_recommendation: {recommendation}",
        "",
        "Answers:",
        "1. true 3D / Piao-style route is technically feasible only at the stepped-depth smoke level unless smoke_status is variable_depth_pass.",
        "2. COMSOL smooth variable-depth geometry remains the main blocker when smoke_status is stepped_depth_smoke_pass.",
        "3. Output schema is ready if validation schema_pass_count >= 3.",
        "4. Dense mask baseline remains comparator only and is not updated.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    check_no_overwrite([args.summary, args.metrics, args.route_summary, args.route_matrix], args.overwrite)
    if not args.npz_path.exists():
        raise FileNotFoundError(f"missing smoke NPZ: {args.npz_path}")
    with np.load(args.npz_path, allow_pickle=True) as npz:
        required = [
            "delta_b",
            "b_defect",
            "b_no_defect",
            "rbc_params",
            "profile_pose",
            "profile_depth_grid_m",
            "profile_depth_map_xy_m",
            "projected_mask_2d",
            "stepped_depth_approximation",
            "depth_levels_m",
            "axis_names",
            "axis_expressions",
            "sample_ids",
            "metadata",
        ]
        missing = [name for name in required if name not in npz.files]
        if missing:
            raise RuntimeError(f"missing required arrays: {missing}")
        delta_b = npz["delta_b"]
        b_defect = npz["b_defect"]
        b_no_defect = npz["b_no_defect"]
        masks = npz["projected_mask_2d"]
        depth_grids = npz["profile_depth_grid_m"]
        depth_maps = npz["profile_depth_map_xy_m"]
        rbc_params = npz["rbc_params"]
        stepped = npz["stepped_depth_approximation"]
        exact = npz["exact_piao_rbc"]
        sample_ids = [str(value) for value in npz["sample_ids"]]
        axis_names = [str(value) for value in npz["axis_names"]]
        axis_expressions = [str(value) for value in npz["axis_expressions"]]
        metadata = load_metadata(npz)

        metrics = [
            validate_sample(
                index,
                sample_id,
                delta_b[index],
                b_defect[index],
                b_no_defect[index],
                masks[index],
                depth_grids[index],
                depth_maps[index],
                rbc_params[index],
                bool(stepped[index]),
                bool(exact[index]),
                axis_names,
                axis_expressions,
            )
            for index, sample_id in enumerate(sample_ids)
        ]

    smoke_status = determine_status(metrics, metadata, stepped)
    write_csv(args.metrics, metrics, METRIC_FIELDS)
    write_summary(args.summary, metrics, metadata, smoke_status)
    decisions = route_rows(smoke_status)
    write_csv(args.route_matrix, decisions, ROUTE_FIELDS)
    write_route_summary(args.route_summary, smoke_status)
    if smoke_status == "failed":
        raise RuntimeError("20.66 true 3D RBC smoke validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
