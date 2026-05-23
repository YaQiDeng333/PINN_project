#!/usr/bin/env python
"""Validate the 20.69 imported watertight forward smoke NPZ."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "data/comsol_mfl/prepared/true_3d_imported_watertight_forward_smoke_v1.npz"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_imported_watertight_forward_smoke_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_imported_watertight_forward_smoke_validation_metrics.csv"

METRIC_FIELDS = [
    "sample_id",
    "schema_pass",
    "delta_b_shape",
    "b_defect_shape",
    "b_no_defect_shape",
    "axis_names",
    "axis_expressions",
    "geometry_method_used",
    "imported_watertight_solid_pass",
    "mesh_units",
    "mesh_source",
    "top_cap_plane",
    "depth_sign_convention",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 20.69 imported watertight forward smoke.")
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
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
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def arr_shape(value: np.ndarray) -> str:
    return json_dumps(list(value.shape))


def scalar_bool(array: np.ndarray) -> bool:
    return bool(np.asarray(array).reshape(-1)[0])


def scalar_str(array: np.ndarray) -> str:
    return str(np.asarray(array).reshape(-1)[0])


def validate(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics], args.overwrite)
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
        "geometry_method_used",
        "imported_watertight_solid_pass",
        "mesh_units",
        "mesh_source",
        "top_cap_plane",
        "depth_sign_convention",
        "metadata",
    ]
    metrics: list[dict[str, Any]] = []
    with np.load(args.npz_path, allow_pickle=True) as npz:
        missing = [name for name in required if name not in npz.files]
        if missing:
            raise RuntimeError(f"missing NPZ fields: {missing}")
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
        method = scalar_str(npz["geometry_method_used"])
        imported_pass = scalar_bool(npz["imported_watertight_solid_pass"])
        mesh_units = scalar_str(npz["mesh_units"])
        mesh_source = scalar_str(npz["mesh_source"])
        top_cap = scalar_str(npz["top_cap_plane"])
        depth_sign = scalar_str(npz["depth_sign_convention"])
        for index, sample_id in enumerate(sample_ids):
            delta = delta_b[index]
            defect = b_defect[index]
            no = b_no[index]
            mask = masks[index]
            depth_grid = depth_grids[index]
            depth_map = depth_maps[index]
            params = rbc_params[index]
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
                and method == "imported_watertight_mesh_solid"
                and imported_pass
                and mesh_units == "m"
                and mesh_source == "triangulated_depth_grid"
                and finite
                and delta_error <= 1.0e-12
                and norm > 0.0
                and abs(depth_max - param_d) <= 0.03 * param_d
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
                    "geometry_method_used": method,
                    "imported_watertight_solid_pass": imported_pass,
                    "mesh_units": mesh_units,
                    "mesh_source": mesh_source,
                    "top_cap_plane": top_cap,
                    "depth_sign_convention": depth_sign,
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
                    "notes": "projected_mask_2d is comparator only; imported watertight solid is the geometry route under validation",
                }
            )
    write_csv(args.metrics, metrics, METRIC_FIELDS)
    pass_count = sum(1 for row in metrics if bool(row["schema_pass"]))
    lines = [
        "20.69 imported watertight forward smoke validation summary",
        "",
        f"sample_count: {len(metrics)}",
        f"schema_pass_count: {pass_count}",
        f"geometry_method_used: {metrics[0]['geometry_method_used'] if metrics else 'none'}",
        f"imported_watertight_solid_pass: {metrics[0]['imported_watertight_solid_pass'] if metrics else False}",
        f"mesh_units: {metrics[0]['mesh_units'] if metrics else ''}",
        f"mesh_source: {metrics[0]['mesh_source'] if metrics else ''}",
        "axis_names: [Bx, By, Bz]",
        "",
        "Validation result:",
    ]
    if pass_count == len(metrics) and metrics:
        lines.append("- PASS. The imported watertight solid forward smoke is schema-ready for this one-sample feasibility gate.")
    else:
        lines.append("- FAIL. Do not expand samples or train.")
    lines.extend(
        [
            "- No training, surrogate, refinement, or baseline update is performed by this validation.",
            "- The NPZ is generated data and must not be committed.",
        ]
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if pass_count != len(metrics) or not metrics:
        raise RuntimeError("20.69 imported watertight forward smoke validation failed")
    return 0


def main() -> int:
    return validate(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
