#!/usr/bin/env python
"""Generate and validate the 20.69 medium_round watertight RBC-style mesh.

The mesh is a closed positive-depth void volume, not an open surface. It uses
the stored RBC-style depth grid from the 20.69 plan, keeps units in meters, and
exports a temporary STL for COMSOL import testing. The STL is generated under
data/comsol_mfl/generated/... and is explicitly forbidden from git commits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

csv.field_size_limit(2**31 - 1)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "results/metrics/true_3d_watertight_mesh_builder_plan.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_watertight_mesh_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_watertight_mesh_validation_metrics.csv"
DEFAULT_ROUTE_SUMMARY = ROOT / "results/summaries/true_3d_watertight_imported_solid_route_decision_summary.txt"
DEFAULT_ROUTE_MATRIX = ROOT / "results/metrics/true_3d_watertight_imported_solid_route_decision_matrix.csv"
DEFAULT_COMSOL_INVENTORY = Path(
    r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\true_3d_imported_watertight_solid_test_inventory.csv"
)
DEFAULT_FORWARD_INVENTORY = Path(
    r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\inventory_true_3d_imported_watertight_forward_smoke.csv"
)
DEFAULT_FORWARD_SUMMARY = Path(
    r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\results\true_3d_imported_watertight_forward_smoke_summary.txt"
)

METRIC_FIELDS = [
    "sample_id",
    "mesh_validation_pass",
    "mesh_units",
    "mesh_source",
    "surface_continuity_assumption",
    "top_cap_plane",
    "depth_sign_convention",
    "profile_pose_to_comsol_json",
    "steel_surface_z_m",
    "steel_z_min_m",
    "steel_z_max_m",
    "defect_void_embedded_in_steel",
    "defect_intersects_top_surface",
    "bbox_inside_steel",
    "is_watertight",
    "edge_incidence_all_two",
    "nonmanifold_edges_count",
    "zero_area_triangles_count",
    "volume_m3",
    "signed_volume_m3",
    "volume_positive",
    "bounds_min_json",
    "bounds_max_json",
    "max_depth_m",
    "target_D_m",
    "max_depth_abs_error_m",
    "depth_rmse_vs_target",
    "projected_footprint_area_m2",
    "vertex_count",
    "face_count",
    "top_cap_face_count",
    "bottom_face_count",
    "side_face_count",
    "top_normals_mean_z",
    "bottom_normals_mean_z",
    "inverted_normals_flag",
    "export_path",
    "export_format",
    "exact_piao_rbc",
    "rbc_style_approximation",
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
    parser = argparse.ArgumentParser(description="Generate watertight RBC-style STL for 20.69.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--route-matrix", type=Path, default=DEFAULT_ROUTE_MATRIX)
    parser.add_argument("--comsol-inventory", type=Path, default=DEFAULT_COMSOL_INVENTORY)
    parser.add_argument("--forward-inventory", type=Path, default=DEFAULT_FORWARD_INVENTORY)
    parser.add_argument("--forward-summary", type=Path, default=DEFAULT_FORWARD_SUMMARY)
    parser.add_argument("--route-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def selected_plan_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"20.69 expects exactly one plan row, got {len(rows)}")
    return rows[0]


def local_to_world(row: dict[str, str], u_value: float, v_value: float) -> tuple[float, float]:
    x_local = 0.5 * float(row["L_m"]) * u_value
    y_local = 0.5 * float(row["W_m"]) * v_value
    angle = float(row["angle_rad"])
    ca = math.cos(angle)
    sa = math.sin(angle)
    x = float(row["center_x_m"]) + ca * x_local - sa * y_local
    y = float(row["center_y_m"]) + sa * x_local + ca * y_local
    return x, y


def build_mesh(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    depth = np.asarray(json.loads(row["profile_depth_grid_m_json"]), dtype=np.float64)
    if depth.ndim != 2:
        raise RuntimeError(f"profile_depth_grid_m must be 2D, got {depth.shape}")
    if not np.isfinite(depth).all():
        raise RuntimeError("depth grid contains non-finite values")
    u_count, v_count = depth.shape
    threshold = max(1.0e-6, 0.01 * float(row["D_m"]))
    active_cells = np.zeros((u_count - 1, v_count - 1), dtype=bool)
    for i in range(u_count - 1):
        for j in range(v_count - 1):
            corners = depth[i : i + 2, j : j + 2]
            active_cells[i, j] = bool(np.all(corners > threshold))
    if not active_cells.any():
        raise RuntimeError("no positive-depth grid cells passed the mesh threshold")

    vertex_map: dict[tuple[str, int, int], int] = {}
    vertices: list[list[float]] = []

    def add_vertex(kind: str, i: int, j: int) -> int:
        key = (kind, i, j)
        if key in vertex_map:
            return vertex_map[key]
        u_value = -1.0 + 2.0 * i / max(u_count - 1, 1)
        v_value = -1.0 + 2.0 * j / max(v_count - 1, 1)
        x, y = local_to_world(row, u_value, v_value)
        z = 0.0 if kind == "top" else -float(depth[i, j])
        vertex_map[key] = len(vertices)
        vertices.append([x, y, z])
        return vertex_map[key]

    faces: list[list[int]] = []
    face_roles: list[str] = []

    def add_face(role: str, a: int, b: int, c: int) -> None:
        faces.append([a, b, c])
        face_roles.append(role)

    for i in range(u_count - 1):
        for j in range(v_count - 1):
            if not active_cells[i, j]:
                continue
            t00 = add_vertex("top", i, j)
            t10 = add_vertex("top", i + 1, j)
            t11 = add_vertex("top", i + 1, j + 1)
            t01 = add_vertex("top", i, j + 1)
            b00 = add_vertex("bottom", i, j)
            b10 = add_vertex("bottom", i + 1, j)
            b11 = add_vertex("bottom", i + 1, j + 1)
            b01 = add_vertex("bottom", i, j + 1)
            add_face("top", t00, t10, t11)
            add_face("top", t00, t11, t01)
            add_face("bottom", b00, b11, b10)
            add_face("bottom", b00, b01, b11)
            if i == 0 or not active_cells[i - 1, j]:
                add_face("side", t00, b01, b00)
                add_face("side", t00, t01, b01)
            if i == u_count - 2 or not active_cells[i + 1, j]:
                add_face("side", t10, b10, b11)
                add_face("side", t10, b11, t11)
            if j == 0 or not active_cells[i, j - 1]:
                add_face("side", t00, b00, b10)
                add_face("side", t00, b10, t10)
            if j == v_count - 2 or not active_cells[i, j + 1]:
                add_face("side", t01, b11, b01)
                add_face("side", t01, t11, b11)

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), {
        "face_roles": face_roles,
        "active_cell_count": int(active_cells.sum()),
        "threshold_m": threshold,
    }


def face_normal(vertices: np.ndarray, face: np.ndarray) -> np.ndarray:
    a, b, c = vertices[face]
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return normal / norm


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    total = 0.0
    for face in faces:
        a, b, c = vertices[face]
        total += float(np.dot(a, np.cross(b, c))) / 6.0
    return total


def edge_counts(faces: np.ndarray) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            counts[tuple(sorted((int(u), int(v))))] += 1
    return counts


def zero_area_count(vertices: np.ndarray, faces: np.ndarray) -> int:
    count = 0
    for face in faces:
        a, b, c = vertices[face]
        area2 = float(np.linalg.norm(np.cross(b - a, c - a)))
        if area2 <= 1.0e-18:
            count += 1
    return count


def write_ascii_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("solid medium_round_watertight_depth_surface\n")
        for face in faces:
            normal = face_normal(vertices, face)
            f.write(f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}\n")
            f.write("    outer loop\n")
            for index in face:
                x, y, z = vertices[int(index)]
                f.write(f"      vertex {x:.9e} {y:.9e} {z:.9e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid medium_round_watertight_depth_surface\n")


def validate_mesh(row: dict[str, str], vertices: np.ndarray, faces: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    counts = edge_counts(faces)
    nonmanifold = sum(1 for value in counts.values() if value != 2)
    is_watertight = nonmanifold == 0 and len(counts) > 0
    zero_faces = zero_area_count(vertices, faces)
    signed = signed_volume(vertices, faces)
    volume = abs(signed)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    max_depth = float(max(0.0, -bounds_min[2]))
    target_d = float(row["D_m"])
    bottom_depths = -vertices[vertices[:, 2] < -1.0e-12, 2]
    source_depth = np.asarray(json.loads(row["profile_depth_grid_m_json"]), dtype=np.float64)
    source_positive = source_depth[source_depth > meta["threshold_m"]]
    depth_rmse = float(np.sqrt(np.mean((np.sort(bottom_depths)[: min(len(bottom_depths), len(source_positive))] - np.sort(source_positive)[: min(len(bottom_depths), len(source_positive))]) ** 2))) if len(bottom_depths) and len(source_positive) else float("nan")
    roles = meta["face_roles"]
    normals = np.asarray([face_normal(vertices, face) for face in faces], dtype=np.float64)
    top_normals = normals[[role == "top" for role in roles]]
    bottom_normals = normals[[role == "bottom" for role in roles]]
    top_mean_z = float(np.mean(top_normals[:, 2])) if len(top_normals) else 0.0
    bottom_mean_z = float(np.mean(bottom_normals[:, 2])) if len(bottom_normals) else 0.0
    steel = {
        "x_min": float(row["steel_x_min_m"]),
        "x_max": float(row["steel_x_max_m"]),
        "y_min": float(row["steel_y_min_m"]),
        "y_max": float(row["steel_y_max_m"]),
        "z_min": float(row["steel_z_min_m"]),
        "z_max": float(row["steel_z_max_m"]),
        "surface": float(row["steel_surface_z_m"]),
    }
    tol = 1.0e-9
    bbox_inside = bool(
        bounds_min[0] >= steel["x_min"] - tol
        and bounds_max[0] <= steel["x_max"] + tol
        and bounds_min[1] >= steel["y_min"] - tol
        and bounds_max[1] <= steel["y_max"] + tol
        and bounds_min[2] >= steel["z_min"] - tol
        and bounds_max[2] <= steel["z_max"] + tol
    )
    intersects_top = bool(abs(bounds_max[2] - steel["surface"]) <= 1.0e-12)
    embedded = bool(bbox_inside and intersects_top and bounds_min[2] < steel["surface"] - 1.0e-9)
    inverted = bool(top_mean_z <= 0.5 or bottom_mean_z >= -0.5)
    projected_area = float(row["target_footprint_area_m2"])
    validation_pass = bool(
        np.isfinite(vertices).all()
        and is_watertight
        and nonmanifold == 0
        and zero_faces == 0
        and volume > 0.0
        and abs(max_depth - target_d) <= 0.03 * target_d
        and bbox_inside
        and intersects_top
        and embedded
        and not inverted
    )
    return {
        "sample_id": row["sample_id"],
        "mesh_validation_pass": validation_pass,
        "mesh_units": row["mesh_units"],
        "mesh_source": row["mesh_source"],
        "surface_continuity_assumption": row["surface_continuity_assumption"],
        "top_cap_plane": row["top_cap_plane"],
        "depth_sign_convention": row["depth_sign_convention"],
        "profile_pose_to_comsol_json": row["profile_pose_to_comsol_json"],
        "steel_surface_z_m": row["steel_surface_z_m"],
        "steel_z_min_m": row["steel_z_min_m"],
        "steel_z_max_m": row["steel_z_max_m"],
        "defect_void_embedded_in_steel": embedded,
        "defect_intersects_top_surface": intersects_top,
        "bbox_inside_steel": bbox_inside,
        "is_watertight": is_watertight,
        "edge_incidence_all_two": all(value == 2 for value in counts.values()),
        "nonmanifold_edges_count": nonmanifold,
        "zero_area_triangles_count": zero_faces,
        "volume_m3": volume,
        "signed_volume_m3": signed,
        "volume_positive": volume > 0.0,
        "bounds_min_json": json_dumps([float(v) for v in bounds_min]),
        "bounds_max_json": json_dumps([float(v) for v in bounds_max]),
        "max_depth_m": max_depth,
        "target_D_m": target_d,
        "max_depth_abs_error_m": abs(max_depth - target_d),
        "depth_rmse_vs_target": depth_rmse,
        "projected_footprint_area_m2": projected_area,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "top_cap_face_count": sum(1 for role in roles if role == "top"),
        "bottom_face_count": sum(1 for role in roles if role == "bottom"),
        "side_face_count": sum(1 for role in roles if role == "side"),
        "top_normals_mean_z": top_mean_z,
        "bottom_normals_mean_z": bottom_mean_z,
        "inverted_normals_flag": inverted,
        "export_path": row["temp_mesh_output_path"],
        "export_format": "ascii_stl",
        "exact_piao_rbc": row["exact_piao_rbc"],
        "rbc_style_approximation": row["rbc_style_approximation"],
        "notes": f"active_cells={meta['active_cell_count']}; temp STL is generated artifact and must not be committed",
    }


def run_mesh_generation(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics], args.overwrite)
    row = selected_plan_row(args.plan_csv)
    vertices, faces, meta = build_mesh(row)
    metric = validate_mesh(row, vertices, faces, meta)
    if not metric["mesh_validation_pass"]:
        write_csv(args.metrics, [metric], METRIC_FIELDS)
        write_summary(args.summary, metric)
        raise RuntimeError("watertight mesh validation failed; see metrics")
    write_ascii_stl(Path(row["temp_mesh_output_path"]), vertices, faces)
    write_csv(args.metrics, [metric], METRIC_FIELDS)
    write_summary(args.summary, metric)
    return 0


def write_summary(path: Path, metric: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "20.69 watertight RBC mesh validation summary",
        "",
        f"sample_id: {metric['sample_id']}",
        f"mesh_validation_pass: {metric['mesh_validation_pass']}",
        f"mesh_units: {metric['mesh_units']}",
        f"mesh_source: {metric['mesh_source']}",
        f"surface_continuity_assumption: {metric['surface_continuity_assumption']}",
        f"top_cap_plane: {metric['top_cap_plane']}",
        f"depth_sign_convention: {metric['depth_sign_convention']}",
        f"defect_void_embedded_in_steel: {metric['defect_void_embedded_in_steel']}",
        f"defect_intersects_top_surface: {metric['defect_intersects_top_surface']}",
        f"bbox_inside_steel: {metric['bbox_inside_steel']}",
        f"is_watertight: {metric['is_watertight']}",
        f"edge_incidence_all_two: {metric['edge_incidence_all_two']}",
        f"nonmanifold_edges_count: {metric['nonmanifold_edges_count']}",
        f"zero_area_triangles_count: {metric['zero_area_triangles_count']}",
        f"volume_m3: {metric['volume_m3']}",
        f"max_depth_m: {metric['max_depth_m']}",
        f"target_D_m: {metric['target_D_m']}",
        f"depth_rmse_vs_target: {metric['depth_rmse_vs_target']}",
        f"vertex_count: {metric['vertex_count']}",
        f"face_count: {metric['face_count']}",
        f"export_path: {metric['export_path']}",
        "",
        "Coordinate / placement boundary:",
        "- mesh_units=m",
        "- steel surface is z=0, defect top cap lies on z=0, and bottom surface uses z=-depth.",
        "- The mesh must be embedded in the steel block and intersect the top surface before COMSOL Boolean is attempted.",
        "",
        "Git boundary:",
        "- The exported STL is a generated temp artifact under data/ and must not be committed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bool_value(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def forward_status(args: argparse.Namespace) -> tuple[str, str]:
    if args.forward_inventory.exists():
        rows = read_csv(args.forward_inventory)
        if rows:
            row = rows[0]
            if str(row.get("status", "")).lower() == "pass" and bool_value(row, "imported_watertight_solid_pass"):
                return "forward_pass", "one-sample imported watertight forward smoke passed"
            return "forward_failed", "one-sample imported watertight forward inventory did not pass"
        return "forward_failed", "one-sample imported watertight forward smoke ran but produced no successful inventory rows"
    if args.forward_summary.exists():
        text = args.forward_summary.read_text(encoding="utf-8", errors="replace")
        if "successful_rows: 0" in text or "Failures:" in text:
            return "forward_failed", "one-sample imported watertight forward smoke failed; see COMSOL summary"
    return "forward_not_run", "one-sample imported watertight forward smoke did not run"


def route_status_from_inventory(args: argparse.Namespace) -> tuple[str, str, dict[str, str] | None]:
    path = args.comsol_inventory
    if not path.exists():
        return "D_mesh_pass_no_comsol_probe", "COMSOL imported solid probe did not run", None
    rows = read_csv(path)
    rbc_rows = [row for row in rows if row.get("probe_kind") == "rbc_watertight_mesh"]
    sanity_rows = [row for row in rows if row.get("probe_kind") == "known_sanity_prism"]
    if not rbc_rows:
        return "B_no_rbc_probe_row", "no RBC imported solid row found", sanity_rows[0] if sanity_rows else None
    rbc = rbc_rows[0]
    f_status, f_observed = forward_status(args)
    if f_status == "forward_pass":
        return "A_imported_forward_pass", "imported watertight solid plus forward smoke passed", rbc
    if bool_value(rbc, "mesh_precheck_success"):
        return "C_import_boolean_pass_forward_not_run_or_failed", f"import/subtract/mesh passed but {f_observed}", rbc
    if bool_value(rbc, "import_success") and not bool_value(rbc, "form_solid_success"):
        return "B_mesh_pass_import_form_solid_failed", "RBC STL imported but did not form a usable solid domain", rbc
    if bool_value(rbc, "import_success") and not bool_value(rbc, "boolean_subtract_success"):
        return "B_mesh_pass_boolean_failed", "RBC STL imported/form-solid path did not pass Boolean subtract", rbc
    return "B_mesh_pass_comsol_import_failed", "Python watertight mesh passed but COMSOL import/form-solid/Boolean gate failed", rbc


def write_route(args: argparse.Namespace) -> int:
    check_no_overwrite([args.route_summary, args.route_matrix], args.overwrite)
    if not args.metrics.exists():
        raise RuntimeError(f"mesh metrics missing: {args.metrics}")
    mesh_metric = read_csv(args.metrics)[0]
    mesh_pass = bool_value(mesh_metric, "mesh_validation_pass")
    status, observed, rbc_row = route_status_from_inventory(args)
    if not mesh_pass:
        status = "D_mesh_generation_failed"
        observed = "Python watertight mesh validation failed"
    f_status, f_observed = forward_status(args)
    route_rows = [
        {
            "decision_option": "A_imported_watertight_solid_forward_pass",
            "selected": status == "A_imported_forward_pass",
            "condition": "Python watertight mesh, COMSOL import/form solid/Boolean/mesh precheck, and one-sample forward all pass",
            "observed": observed,
            "next_step": "design smooth/mesh-based true 3D RBC pilot",
        },
        {
            "decision_option": "B_mesh_pass_comsol_import_or_form_solid_fails",
            "selected": status.startswith("B_") or status == "D_mesh_pass_no_comsol_probe",
            "condition": "Python mesh passes but COMSOL import/form solid/Boolean/mesh precheck does not pass",
            "observed": observed,
            "next_step": "fix COMSOL STL import/repair/form-solid workflow",
        },
        {
            "decision_option": "C_import_boolean_pass_forward_fails",
            "selected": status.startswith("C_"),
            "condition": "COMSOL geometry gate passes but Bx/By/Bz forward or schema validation fails",
            "observed": observed,
            "next_step": "fix COMSOL solve/export/schema for imported solid",
        },
        {
            "decision_option": "D_python_mesh_generation_fails",
            "selected": status == "D_mesh_generation_failed",
            "condition": "Watertight mesh generation or validation fails",
            "observed": observed,
            "next_step": "fix watertight RBC mesh generator",
        },
        {
            "decision_option": "E_all_fail_or_accept_high_layer",
            "selected": False,
            "condition": "Imported route remains blocked and user chooses not to continue builder hardening",
            "observed": "not selected by this script",
            "next_step": "ask whether to accept high-layer approximation or pause smooth builder",
        },
    ]
    write_csv(args.route_matrix, route_rows, ROUTE_FIELDS)
    known_status = "not_run"
    rbc_status = "not_run"
    if args.comsol_inventory.exists():
        rows = read_csv(args.comsol_inventory)
        known = [row for row in rows if row.get("probe_kind") == "known_sanity_prism"]
        rbc = [row for row in rows if row.get("probe_kind") == "rbc_watertight_mesh"]
        if known:
            known_status = known[0].get("probe_status", "unknown")
        if rbc:
            rbc_status = rbc[0].get("probe_status", "unknown")
    lines = [
        "20.69 watertight imported solid route decision summary",
        "",
        f"selected_status: {status}",
        f"python_watertight_mesh_pass: {mesh_pass}",
        f"known_sanity_probe_status: {known_status}",
        f"rbc_watertight_mesh_status: {'pass' if mesh_pass else 'failed'}",
        f"rbc_imported_solid_status: {rbc_status}",
        f"forward_smoke_status: {f_status}",
        f"observed: {observed}",
        "",
        "Separated gate accounting:",
    ]
    if rbc_row is None:
        lines.append("- RBC COMSOL row: not available")
    else:
        for key in [
            "import_success",
            "repair_success",
            "form_solid_success",
            "imported_domain_count",
            "boolean_subtract_success",
            "steel_notched_domain_count",
            "mesh_precheck_success",
        ]:
            lines.append(f"- {key}: {rbc_row.get(key, '')}")
        lines.append(f"- forward_smoke_executed: {f_status in {'forward_pass', 'forward_failed'}}")
        lines.append(f"- forward_smoke_pass: {f_status == 'forward_pass'}")
        lines.append(f"- forward_smoke_observed: {f_observed}")
    lines.extend(
        [
            "",
            "Decision:",
        ]
    )
    if status == "A_imported_forward_pass":
        lines.append("- Imported watertight mesh solid route passed geometry and one-sample forward smoke; next step can be smooth/mesh-based pilot design.")
    elif status.startswith("B_") or status == "D_mesh_pass_no_comsol_probe":
        lines.append("- Python mesh passed, but imported solid route is not complete; fix COMSOL import/repair/form-solid/Boolean workflow before pilot expansion.")
    elif status.startswith("C_"):
        lines.append("- Imported solid geometry gate passed, but forward/schema did not; fix solve/export/schema before pilot expansion.")
    elif status == "D_mesh_generation_failed":
        lines.append("- Python watertight mesh failed; fix mesh generator before COMSOL work.")
    else:
        lines.append("- Imported route remains blocked; user must decide whether to continue builder hardening or accept high-layer approximation.")
    lines.extend(
        [
            "",
            "Boundary:",
            "- known cube/prism sanity success does not count as RBC imported solid success.",
            "- high_layer_control_24 is only historical reference and is not selected by 20.69.",
            "- No training, pilot expansion, or baseline update is authorized by this route decision.",
        ]
    )
    args.route_summary.parent.mkdir(parents=True, exist_ok=True)
    args.route_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()
    if args.route_only:
        return write_route(args)
    return run_mesh_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
