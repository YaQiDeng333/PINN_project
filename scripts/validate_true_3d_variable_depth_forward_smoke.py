#!/usr/bin/env python
"""Validate the 20.67 variable-depth true-3D forward smoke NPZ.

This validation does not train a model, does not run refinement, and does not
update baselines. It only checks schema, field consistency, and route status.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "data/comsol_mfl/prepared/true_3d_variable_depth_forward_smoke_v1.npz"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_variable_depth_forward_smoke_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_variable_depth_forward_smoke_validation_metrics.csv"
DEFAULT_ROUTE_SUMMARY = ROOT / "results/summaries/true_3d_variable_depth_geometry_route_decision_summary.txt"
DEFAULT_ROUTE_MATRIX = ROOT / "results/metrics/true_3d_variable_depth_geometry_route_decision_matrix.csv"

METRIC_FIELDS = [
    "sample_id",
    "schema_pass",
    "delta_b_shape",
    "b_defect_shape",
    "b_no_defect_shape",
    "axis_names",
    "axis_expressions",
    "geometry_method_used",
    "variable_depth_pass",
    "near_smooth_pass",
    "high_layer_pass",
    "constant_depth_extrusion",
    "depth_levels_count",
    "depth_variation_m",
    "projected_mask_shape",
    "projected_mask_area_px",
    "profile_depth_grid_shape",
    "profile_depth_map_shape",
    "rbc_params_shape",
    "all_values_finite",
    "delta_max_abs_error",
    "defect_signal_norm",
    "defect_signal_nonzero",
    "Bx_norm",
    "By_norm",
    "Bz_norm",
    "depth_max_m",
    "param_D_m",
    "profile_depth_max_error_vs_param_D",
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
    parser = argparse.ArgumentParser(description="Validate 20.67 variable-depth forward smoke.")
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    return json_dumps(list(value.shape))


def bool_at(array: np.ndarray, index: int = 0) -> bool:
    return bool(np.asarray(array).reshape(-1)[index])


def determine_status(metrics: list[dict[str, Any]]) -> str:
    if not metrics or not all(bool(row["schema_pass"]) for row in metrics):
        return "failed"
    row = metrics[0]
    if bool(row["constant_depth_extrusion"]):
        return "failed"
    if bool(row["variable_depth_pass"]):
        return "variable_depth_pass"
    if bool(row["near_smooth_pass"]):
        return "near_smooth_pass"
    if bool(row["high_layer_pass"]):
        return "high_layer_pass"
    return "failed"


def route_rows(status: str) -> list[dict[str, Any]]:
    return [
        {
            "decision_option": "A_variable_depth_pass",
            "selected": status == "variable_depth_pass",
            "condition": "smooth continuous variable-depth closed solid plus forward/schema validation passes",
            "observed": status,
            "next_step": "generate 60-sample true 3D RBC pilot",
        },
        {
            "decision_option": "B_near_smooth_pass",
            "selected": status == "near_smooth_pass",
            "condition": "near-smooth loft/import/interpolated approximation passes but is not exact smooth",
            "observed": status,
            "next_step": "consider pilot only with explicit approximation label",
        },
        {
            "decision_option": "C_high_layer_pass",
            "selected": status == "high_layer_pass",
            "condition": "12/16-layer high-layer approximation passes and improves beyond 20.66 five-layer smoke",
            "observed": status,
            "next_step": "ask for human confirmation before accepting approximation and expanding samples",
        },
        {
            "decision_option": "D_failed",
            "selected": status == "failed",
            "condition": "constant-depth only, geometry failure, field export failure, delta failure, or schema failure",
            "observed": status,
            "next_step": "fix COMSOL geometry builder before expanding or training",
        },
    ]


def validate(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics, args.route_summary, args.route_matrix], args.overwrite)
    required = [
        "delta_b",
        "b_defect",
        "b_no_defect",
        "axis_names",
        "axis_expressions",
        "sample_ids",
        "rbc_params",
        "profile_pose",
        "profile_depth_grid_m",
        "profile_depth_map_xy_m",
        "projected_mask_2d",
        "depth_levels_m",
        "geometry_method_used",
        "high_layer_pass",
        "near_smooth_pass",
        "variable_depth_pass",
        "constant_depth_extrusion",
        "metadata",
    ]
    metrics: list[dict[str, Any]] = []
    with np.load(args.npz_path, allow_pickle=True) as npz:
        missing = [name for name in required if name not in npz.files]
        if missing:
            raise RuntimeError(f"missing NPZ fields: {missing}")
        metadata = load_metadata(npz)
        delta_b = npz["delta_b"]
        b_defect = npz["b_defect"]
        b_no = npz["b_no_defect"]
        masks = npz["projected_mask_2d"]
        depth_grids = npz["profile_depth_grid_m"]
        depth_maps = npz["profile_depth_map_xy_m"]
        rbc_params = npz["rbc_params"]
        axis_names = [str(value) for value in npz["axis_names"].tolist()]
        axis_expr = [str(value) for value in npz["axis_expressions"].tolist()]
        sample_ids = [str(value) for value in npz["sample_ids"].tolist()]
        methods = [str(value) for value in npz["geometry_method_used"].tolist()]
        high_layer = npz["high_layer_pass"]
        near_smooth = npz["near_smooth_pass"]
        variable_depth = npz["variable_depth_pass"]
        constant_depth = npz["constant_depth_extrusion"]
        if delta_b.ndim != 4:
            raise RuntimeError(f"delta_b expected ndim=4, got {delta_b.ndim}")
        for index, sample_id in enumerate(sample_ids):
            delta = delta_b[index]
            defect = b_defect[index]
            no = b_no[index]
            mask = masks[index]
            depth_grid = depth_grids[index]
            depth_map = depth_maps[index]
            params = rbc_params[index]
            levels = np.asarray(npz["depth_levels_m"][index], dtype=np.float64)
            recomputed = defect - no
            delta_error = float(np.max(np.abs(delta - recomputed)))
            finite = bool(
                np.isfinite(delta).all()
                and np.isfinite(defect).all()
                and np.isfinite(no).all()
                and np.isfinite(depth_grid).all()
                and np.isfinite(depth_map).all()
            )
            norm = float(np.linalg.norm(delta))
            axis_norms = [float(np.linalg.norm(delta[axis_index])) for axis_index in range(delta.shape[0])]
            param_d = float(params.reshape(-1)[2])
            depth_max = float(max(float(depth_grid.max()), float(depth_map.max())))
            status_flags_valid = not bool_at(constant_depth, index) and (
                bool_at(variable_depth, index) or bool_at(near_smooth, index) or bool_at(high_layer, index)
            )
            schema_pass = (
                delta.shape == (3, 3, 201)
                and defect.shape == (3, 3, 201)
                and no.shape == (3, 3, 201)
                and mask.shape == (64, 128)
                and int(mask.sum()) > 0
                and depth_grid.ndim == 2
                and depth_map.shape == (64, 128)
                and params.shape == (1, 6)
                and axis_names == ["Bx", "By", "Bz"]
                and finite
                and delta_error <= 1.0e-12
                and norm > 0.0
                and len(levels[np.isfinite(levels)]) > 5
                and float(np.nanmax(levels) - np.nanmin(levels)) > 0.0
                and status_flags_valid
            )
            metrics.append(
                {
                    "sample_id": sample_id,
                    "schema_pass": schema_pass,
                    "delta_b_shape": arr_shape(delta),
                    "b_defect_shape": arr_shape(defect),
                    "b_no_defect_shape": arr_shape(no),
                    "axis_names": json_dumps(axis_names),
                    "axis_expressions": json_dumps(axis_expr),
                    "geometry_method_used": methods[index],
                    "variable_depth_pass": bool_at(variable_depth, index),
                    "near_smooth_pass": bool_at(near_smooth, index),
                    "high_layer_pass": bool_at(high_layer, index),
                    "constant_depth_extrusion": bool_at(constant_depth, index),
                    "depth_levels_count": int(len(levels[np.isfinite(levels)])),
                    "depth_variation_m": float(np.nanmax(levels) - np.nanmin(levels)),
                    "projected_mask_shape": arr_shape(mask),
                    "projected_mask_area_px": int(mask.sum()),
                    "profile_depth_grid_shape": arr_shape(depth_grid),
                    "profile_depth_map_shape": arr_shape(depth_map),
                    "rbc_params_shape": arr_shape(params),
                    "all_values_finite": finite,
                    "delta_max_abs_error": delta_error,
                    "defect_signal_norm": norm,
                    "defect_signal_nonzero": norm > 0.0,
                    "Bx_norm": axis_norms[0],
                    "By_norm": axis_norms[1],
                    "Bz_norm": axis_norms[2],
                    "depth_max_m": depth_max,
                    "param_D_m": param_d,
                    "profile_depth_max_error_vs_param_D": abs(depth_max - param_d),
                    "notes": "projected_mask_2d is comparator only; 3D label uses RBC params plus depth grid/map",
                }
            )
    status = determine_status(metrics)
    write_csv(args.metrics, metrics, METRIC_FIELDS)
    write_csv(args.route_matrix, route_rows(status), ROUTE_FIELDS)
    pass_count = sum(1 for row in metrics if bool(row["schema_pass"]))
    lines = [
        "20.67 true 3D variable-depth forward smoke validation summary",
        "",
        f"sample_count: {len(metrics)}",
        f"schema_pass_count: {pass_count}",
        f"route_status: {status}",
        f"geometry_method_used: {metrics[0]['geometry_method_used'] if metrics else 'none'}",
        f"variable_depth_pass: {metrics[0]['variable_depth_pass'] if metrics else False}",
        f"near_smooth_pass: {metrics[0]['near_smooth_pass'] if metrics else False}",
        f"high_layer_pass: {metrics[0]['high_layer_pass'] if metrics else False}",
        f"constant_depth_extrusion: {metrics[0]['constant_depth_extrusion'] if metrics else False}",
        f"depth_levels_count: {metrics[0]['depth_levels_count'] if metrics else 0}",
        "baseline_20_66_depth_levels: 5",
        "axis_names: [Bx, By, Bz]",
        "",
        "Validation result:",
    ]
    if status == "variable_depth_pass":
        lines.append("- Smooth variable-depth geometry passed the one-sample forward/schema gate.")
    elif status == "near_smooth_pass":
        lines.append("- Near-smooth approximation passed; keep approximation label before any pilot.")
    elif status == "high_layer_pass":
        lines.append("- High-layer fallback passed; this is beyond 20.66's 5-layer smoke but is not smooth variable-depth.")
    else:
        lines.append("- Validation failed; do not expand samples or train.")
    lines.extend(
        [
            "- No training, surrogate, refinement, or baseline update is performed by this validation.",
            "- projected_mask_2d is retained only as a 2D comparator; it does not replace the 3D profile label.",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    route_lines = [
        "20.67 true 3D variable-depth geometry route decision summary",
        "",
        f"selected_status: {status}",
        "",
    ]
    if status == "variable_depth_pass":
        route_lines.append("Recommendation: A. Generate a 60-sample true 3D RBC pilot.")
    elif status == "near_smooth_pass":
        route_lines.append("Recommendation: B. Consider a pilot only with explicit near-smooth approximation labeling.")
    elif status == "high_layer_pass":
        route_lines.append("Recommendation: C. Ask for human confirmation before accepting high-layer approximation and expanding samples.")
    else:
        route_lines.append("Recommendation: D. Fix the COMSOL geometry builder before expanding or training.")
    route_lines.extend(
        [
            "",
            "Answers:",
            f"1. Did smooth variable-depth pass? {status == 'variable_depth_pass'}",
            f"2. Did near-smooth pass? {status == 'near_smooth_pass'}",
            f"3. Is high-layer stepped approximation materially beyond 20.66? {status == 'high_layer_pass'}",
            f"4. Is it acceptable for pilot without human confirmation? {status in {'variable_depth_pass', 'near_smooth_pass'}}",
            "5. Should next step be pilot generation or geometry improvement? pilot only after status-specific confirmation; high_layer_pass requires explicit human acceptance.",
        ]
    )
    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    args.route_summary.write_text("\n".join(route_lines) + "\n", encoding="utf-8")
    if status == "failed":
        raise RuntimeError("20.67 variable-depth validation failed")
    return 0


def main() -> int:
    return validate(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
