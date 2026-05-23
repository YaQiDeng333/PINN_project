#!/usr/bin/env python
"""Validate the 20.70 imported watertight forward smoke NPZ."""

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
DEFAULT_ROUTE_SUMMARY = ROOT / "results/summaries/true_3d_imported_solid_solver_route_decision_summary.txt"
DEFAULT_ROUTE_MATRIX = ROOT / "results/metrics/true_3d_imported_solid_solver_route_decision_matrix.csv"

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
    "selected_solver_protocol",
    "mesh_auto_size",
    "domain_material_audit_pass",
    "solver_probe_pass",
    "full_source_jscale",
    "direct_solver_used",
    "pilot_scalability_risk",
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
    parser = argparse.ArgumentParser(description="Validate 20.70 imported watertight forward smoke.")
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
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def arr_shape(value: np.ndarray) -> str:
    return json_dumps(list(value.shape))


def scalar_bool(array: np.ndarray) -> bool:
    return bool(np.asarray(array).reshape(-1)[0])


def scalar_str(array: np.ndarray) -> str:
    return str(np.asarray(array).reshape(-1)[0])


def scalar_int(array: np.ndarray) -> int:
    return int(np.asarray(array).reshape(-1)[0])


def scalar_float(array: np.ndarray) -> float:
    return float(np.asarray(array).reshape(-1)[0])


def write_route_decision(summary: Path, matrix: Path, metrics: list[dict[str, Any]], pass_count: int) -> None:
    validation_pass = bool(metrics) and pass_count == len(metrics)
    protocol = str(metrics[0]["selected_solver_protocol"]) if metrics else "none"
    direct = bool(metrics[0]["direct_solver_used"]) if metrics else False
    risk = str(metrics[0]["pilot_scalability_risk"]) if metrics else ""
    if validation_pass:
        decision = "A_imported_solid_solve_forward_validation_pass"
        next_step = "smooth/mesh-based true 3D RBC pilot generation"
        evidence = "Full-source Jscale=1.0 imported-solid defect solve, Bx/By/Bz export, delta check, and NPZ validation passed."
        if direct:
            next_step = "evaluate direct-solver cost on a tiny batch before smooth/mesh-based pilot scaling"
            evidence += " Direct solver was required, so pilot scalability risk is recorded."
    else:
        decision = "E_imported_solid_forward_validation_fail"
        next_step = "continue imported-solid solver robustness"
        evidence = "NPZ/schema validation failed or no full-source forward smoke is available."
    rows = [
        {"decision_key": decision, "selected": True, "evidence": evidence, "next_step": next_step},
        {
            "decision_key": "high_layer_fallback",
            "selected": False,
            "evidence": "20.67 high_layer_approx_12 is a comparator only and was not used as 20.70 success.",
            "next_step": "not used",
        },
    ]
    write_csv(matrix, rows, ["decision_key", "selected", "evidence", "next_step"])
    lines = [
        "20.70 imported-solid solver route decision summary",
        "",
        f"selected_decision: {decision}",
        f"next_step: {next_step}",
        f"validation_pass: {validation_pass}",
        f"selected_solver_protocol: {protocol}",
        f"direct_solver_used: {direct}",
        f"pilot_scalability_risk: {risk}",
        "",
        "Evidence:",
        f"- {evidence}",
        "- Full-source means Jscale=1.0; lower source-scale probes remain diagnostic only.",
        "- high_layer_approx_12 was not used as fallback success.",
    ]
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "geometry_method_used",
        "imported_watertight_solid_pass",
        "mesh_units",
        "mesh_source",
        "top_cap_plane",
        "depth_sign_convention",
        "selected_solver_protocol",
        "mesh_auto_size",
        "domain_material_audit_pass",
        "solver_probe_pass",
        "full_source_jscale",
        "direct_solver_used",
        "pilot_scalability_risk",
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
        selected_solver_protocol = scalar_str(npz["selected_solver_protocol"])
        mesh_auto_size = scalar_int(npz["mesh_auto_size"])
        domain_material_audit_pass = scalar_bool(npz["domain_material_audit_pass"])
        solver_probe_pass = scalar_bool(npz["solver_probe_pass"])
        full_source_jscale = scalar_float(npz["full_source_jscale"])
        direct_solver_used = scalar_bool(npz["direct_solver_used"])
        pilot_scalability_risk = scalar_str(npz["pilot_scalability_risk"])
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
                and selected_solver_protocol in {"default", "direct_solver", "ramp_continuation"}
                and mesh_auto_size in {4, 5, 6}
                and domain_material_audit_pass
                and solver_probe_pass
                and abs(full_source_jscale - 1.0) <= 1.0e-12
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
                    "selected_solver_protocol": selected_solver_protocol,
                    "mesh_auto_size": mesh_auto_size,
                    "domain_material_audit_pass": domain_material_audit_pass,
                    "solver_probe_pass": solver_probe_pass,
                    "full_source_jscale": full_source_jscale,
                    "direct_solver_used": direct_solver_used,
                    "pilot_scalability_risk": pilot_scalability_risk,
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
                    "notes": "projected_mask_2d is comparator only; imported watertight solid full-source solve is the route under validation",
                }
            )
    write_csv(args.metrics, metrics, METRIC_FIELDS)
    pass_count = sum(1 for row in metrics if bool(row["schema_pass"]))
    write_route_decision(args.route_summary, args.route_matrix, metrics, pass_count)
    lines = [
        "20.70 imported watertight forward smoke validation summary",
        "",
        f"sample_count: {len(metrics)}",
        f"schema_pass_count: {pass_count}",
        f"geometry_method_used: {metrics[0]['geometry_method_used'] if metrics else 'none'}",
        f"imported_watertight_solid_pass: {metrics[0]['imported_watertight_solid_pass'] if metrics else False}",
        f"mesh_units: {metrics[0]['mesh_units'] if metrics else ''}",
        f"mesh_source: {metrics[0]['mesh_source'] if metrics else ''}",
        f"selected_solver_protocol: {metrics[0]['selected_solver_protocol'] if metrics else 'none'}",
        f"full_source_jscale: {metrics[0]['full_source_jscale'] if metrics else 'none'}",
        f"direct_solver_used: {metrics[0]['direct_solver_used'] if metrics else False}",
        f"pilot_scalability_risk: {metrics[0]['pilot_scalability_risk'] if metrics else ''}",
        "axis_names: [Bx, By, Bz]",
        "",
        "Validation result:",
    ]
    if pass_count == len(metrics) and metrics:
        lines.append("- PASS. The imported watertight solid full-source forward smoke is schema-ready for this one-sample feasibility gate.")
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
        raise RuntimeError("20.70 imported watertight forward smoke validation failed")
    return 0


def main() -> int:
    return validate(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
