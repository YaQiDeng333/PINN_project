#!/usr/bin/env python
"""Audit 25.9 surface multi-pit component labels.

This script is plan/audit-only. It reads the explicit 25.2 manifest and the
referenced NPZ labels, then writes summary/CSV artifacts. It does not train,
run COMSOL, generate data, mutate NPZ files, or update CURRENT_BASELINE.md.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json"
TAXONOMY = ROOT / "results/metrics/surface_shape_extension_taxonomy_matrix.csv"
LABEL_SCHEMA = ROOT / "results/metrics/surface_shape_extension_label_schema.csv"
DIAGNOSIS_MATRIX = ROOT / "results/metrics/surface_shape_extension_oracle_vs_baseline_matrix.csv"
COMSOL_FEASIBILITY = ROOT / "results/metrics/surface_shape_extension_comsol_feasibility_matrix.csv"
REPORT_ROUTE_SUMMARY = ROOT / "results/summaries/surface_forward_refinement_report_route_decision_summary.txt"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
CURRENT_BASELINE = ROOT / "CURRENT_BASELINE.md"

PREFLIGHT_SUMMARY = ROOT / "results/summaries/surface_multipit_component_branch_preflight_summary.txt"
AUDIT_SUMMARY = ROOT / "results/summaries/surface_multipit_component_label_audit_summary.txt"
AUDIT_CSV = ROOT / "results/metrics/surface_multipit_component_label_audit.csv"
MISSING_FIELDS_CSV = ROOT / "results/metrics/surface_multipit_component_label_missing_fields.csv"

MULTI_PIT_SHAPE = "multi_pit_two_component_surface_defect"
REQUIRED_COMPONENT_KEYS = ["component_id", "center_x_m", "center_y_m", "L_m", "W_m", "depth_m", "component_type"]
STRONG_SCHEMA_FIELDS = [
    "per_component_rotation_angle",
    "per_component_projected_mask_2d",
    "per_component_depth_grid_m",
    "component_separation_m",
    "component_overlap_status",
    "component_topology_relation",
]

AUDIT_FIELDS = [
    "sample_id",
    "split",
    "shape_type",
    "topology_type",
    "representation_target",
    "component_count",
    "component_json_count",
    "component_ids",
    "component_centers_present",
    "component_lwd_present",
    "component_depth_present",
    "component_primitive_present",
    "global_rotation_angle_present",
    "per_component_rotation_present",
    "projected_mask_present",
    "depth_grid_present",
    "component_level_masks_present",
    "component_level_depth_grids_present",
    "projected_connected_components",
    "component_separation_m",
    "separation_bucket",
    "overlap_status",
    "depth_relation",
    "size_pair_bucket",
    "orientation_bucket",
    "topology_bucket",
    "label_sufficiency",
    "schema_gaps",
]

MISSING_FIELDS = [
    "field_name",
    "scope",
    "current_status",
    "severity",
    "required_for",
    "recommended_resolution",
]


def git_value(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def parse_components(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
    if not value:
        return []
    parsed = json.loads(str(value))
    if isinstance(parsed, list):
        return [dict(item) for item in parsed]
    if isinstance(parsed, dict) and "components" in parsed:
        return [dict(item) for item in parsed["components"]]
    return []


def connected_components(mask: np.ndarray) -> int:
    arr = np.asarray(mask) > 0
    if arr.ndim != 2 or not arr.any():
        return 0
    visited = np.zeros(arr.shape, dtype=bool)
    components = 0
    rows, cols = arr.shape
    for r, c in np.argwhere(arr):
        if visited[r, c]:
            continue
        components += 1
        stack = [(int(r), int(c))]
        visited[r, c] = True
        while stack:
            cr, cc = stack.pop()
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and arr[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
    return components


def component_separation(components: list[dict[str, Any]]) -> float:
    if len(components) < 2:
        return float("nan")
    c0, c1 = components[0], components[1]
    return float(math.hypot(float(c1["center_x_m"]) - float(c0["center_x_m"]), float(c1["center_y_m"]) - float(c0["center_y_m"])))


def edge_gap_and_overlap(components: list[dict[str, Any]]) -> tuple[float, str]:
    if len(components) < 2:
        return float("nan"), "not_applicable"
    c0, c1 = components[0], components[1]
    dx = abs(float(c1["center_x_m"]) - float(c0["center_x_m"]))
    dy = abs(float(c1["center_y_m"]) - float(c0["center_y_m"]))
    hx = (float(c0["L_m"]) + float(c1["L_m"])) / 2.0
    hy = (float(c0["W_m"]) + float(c1["W_m"])) / 2.0
    gap_x = dx - hx
    gap_y = dy - hy
    if gap_x < 0 and gap_y < 0:
        return min(gap_x, gap_y), "bbox_overlap_or_projected_merge_risk"
    positive = [max(gap_x, 0.0), max(gap_y, 0.0)]
    return float(math.hypot(*positive)), "bbox_separated"


def separation_bucket(edge_gap_m: float, overlap_status: str) -> str:
    if overlap_status.startswith("bbox_overlap"):
        return "partially_overlapping"
    if not math.isfinite(edge_gap_m):
        return "unknown"
    if edge_gap_m <= 2.5e-4:
        return "touching"
    if edge_gap_m <= 2.0e-3:
        return "close"
    return "separated"


def depth_relation(components: list[dict[str, Any]]) -> str:
    if len(components) < 2:
        return "unknown"
    depths = sorted(float(comp["depth_m"]) for comp in components)
    if depths[0] <= 0:
        return "unknown"
    return "similar_depth" if depths[1] / depths[0] <= 1.25 else "deep_and_shallow"


def size_label(comp: dict[str, Any]) -> str:
    area = float(comp["L_m"]) * float(comp["W_m"])
    if area < 1.8e-5:
        return "small"
    if area < 3.0e-5:
        return "medium"
    return "large"


def size_pair_bucket(components: list[dict[str, Any]]) -> str:
    if len(components) < 2:
        return "unknown"
    return "-".join(sorted(size_label(comp) for comp in components))


def orientation_bucket(components: list[dict[str, Any]]) -> str:
    if len(components) < 2:
        return "unknown"
    dx = abs(float(components[1]["center_x_m"]) - float(components[0]["center_x_m"]))
    dy = abs(float(components[1]["center_y_m"]) - float(components[0]["center_y_m"]))
    if dx >= 2.0 * max(dy, 1e-12):
        return "aligned_x"
    if dy >= 2.0 * max(dx, 1e-12):
        return "aligned_y"
    return "diagonal"


def topology_bucket(cc_count: int, sep_bucket: str, overlap_status: str) -> str:
    if overlap_status.startswith("bbox_overlap") or cc_count < 2:
        return "merged_projected_mask"
    if sep_bucket == "touching":
        return "touching_boundary"
    return "disconnected"


def component_key_status(components: list[dict[str, Any]], keys: list[str]) -> bool:
    return bool(components) and all(all(key in comp and comp[key] not in {"", None} for key in keys) for comp in components)


def build_audit_rows(npz_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(npz_path, allow_pickle=True) as data:
        required_keys = [
            "sample_ids",
            "split",
            "shape_type",
            "topology_type",
            "representation_target",
            "component_count",
            "component_params_json",
            "projected_mask_2d",
            "depth_grid_m",
            "rotation_angle",
        ]
        missing_keys = [key for key in required_keys if key not in data.files]
        if missing_keys:
            raise KeyError("missing NPZ keys for 25.9 label audit: " + ", ".join(missing_keys))
        indices = [idx for idx, shape in enumerate(data["shape_type"]) if str(shape) == MULTI_PIT_SHAPE]
        for idx in indices:
            components = parse_components(data["component_params_json"][idx])
            comp_count = int(data["component_count"][idx])
            comp_json_count = len(components)
            cc_count = connected_components(data["projected_mask_2d"][idx])
            separation_m = component_separation(components)
            edge_gap_m, overlap = edge_gap_and_overlap(components)
            sep_bucket = separation_bucket(edge_gap_m, overlap)
            topo = topology_bucket(cc_count, sep_bucket, overlap)
            centers_present = component_key_status(components, ["center_x_m", "center_y_m"])
            lwd_present = component_key_status(components, ["L_m", "W_m", "depth_m"])
            primitive_present = component_key_status(components, ["component_type"])
            per_component_rotation = component_key_status(components, ["rotation_angle"])
            projected_mask_present = np.asarray(data["projected_mask_2d"][idx]).ndim == 2 and bool(np.asarray(data["projected_mask_2d"][idx]).any())
            depth_grid_present = np.asarray(data["depth_grid_m"][idx]).ndim == 2 and bool(np.isfinite(data["depth_grid_m"][idx]).all())
            component_masks_present = "component_projected_masks_2d" in data.files or "component_masks_2d" in data.files
            component_depths_present = "component_depth_grids_m" in data.files
            gaps: list[str] = []
            if not per_component_rotation:
                gaps.append("per_component_rotation_angle_missing")
            if not component_masks_present:
                gaps.append("component_level_projected_masks_missing")
            if not component_depths_present:
                gaps.append("component_level_depth_grids_missing")
            gaps.append("explicit_separation_touching_overlap_labels_missing")
            c1_ready = comp_count == 2 and comp_json_count == 2 and centers_present and lwd_present and primitive_present and projected_mask_present and depth_grid_present
            rows.append(
                {
                    "sample_id": str(data["sample_ids"][idx]),
                    "split": str(data["split"][idx]),
                    "shape_type": str(data["shape_type"][idx]),
                    "topology_type": str(data["topology_type"][idx]),
                    "representation_target": str(data["representation_target"][idx]),
                    "component_count": comp_count,
                    "component_json_count": comp_json_count,
                    "component_ids": "|".join(str(comp.get("component_id", "")) for comp in components),
                    "component_centers_present": centers_present,
                    "component_lwd_present": lwd_present,
                    "component_depth_present": component_key_status(components, ["depth_m"]),
                    "component_primitive_present": primitive_present,
                    "global_rotation_angle_present": bool(np.isfinite(float(data["rotation_angle"][idx]))),
                    "per_component_rotation_present": per_component_rotation,
                    "projected_mask_present": projected_mask_present,
                    "depth_grid_present": depth_grid_present,
                    "component_level_masks_present": component_masks_present,
                    "component_level_depth_grids_present": component_depths_present,
                    "projected_connected_components": cc_count,
                    "component_separation_m": separation_m,
                    "separation_bucket": sep_bucket,
                    "overlap_status": overlap,
                    "depth_relation": depth_relation(components),
                    "size_pair_bucket": size_pair_bucket(components),
                    "orientation_bucket": orientation_bucket(components),
                    "topology_bucket": topo,
                    "label_sufficiency": "sufficient_for_C1_seed_with_schema_gaps" if c1_ready else "insufficient_for_component_set_seed",
                    "schema_gaps": "|".join(gaps),
                }
            )
    return rows


def build_missing_rows(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_c1_ready = all(row["label_sufficiency"] == "sufficient_for_C1_seed_with_schema_gaps" for row in audit_rows)
    return [
        {
            "field_name": "per_component_rotation_angle",
            "scope": "existing_16_multi_pit_samples",
            "current_status": "missing_from_component_params_json; global rotation_angle exists",
            "severity": "topup_required_not_blocking_seed_audit",
            "required_for": "rotation-aware component-set decoder and close/touching diagnostics",
            "recommended_resolution": "add rotation_angle per component in top-up labels; derive global default only for legacy audit rows",
        },
        {
            "field_name": "per_component_projected_mask_2d",
            "scope": "existing_16_multi_pit_samples",
            "current_status": "missing; union projected_mask_2d exists",
            "severity": "important_schema_gap",
            "required_for": "Hungarian mask cost, component recall, merge/missed metrics",
            "recommended_resolution": "export component-level masks in top-up pack and keep union mask for compatibility",
        },
        {
            "field_name": "per_component_depth_grid_m",
            "scope": "existing_16_multi_pit_samples",
            "current_status": "missing; union depth_grid_m exists",
            "severity": "important_schema_gap",
            "required_for": "component depth-grid loss and overlapping/touching ambiguity checks",
            "recommended_resolution": "export component-level depth grids in top-up pack",
        },
        {
            "field_name": "component_separation_m/component_overlap_status/component_topology_relation",
            "scope": "existing_16_multi_pit_samples",
            "current_status": "not explicit; derived in this audit from component centers and bounding boxes",
            "severity": "derivable_but_should_be_manifest_label",
            "required_for": "split coverage, close/touching gates, merge-rate metrics",
            "recommended_resolution": "write explicit separation/topology labels during top-up generation",
        },
        {
            "field_name": "C1_seed_status",
            "scope": "existing_16_multi_pit_samples",
            "current_status": "ready" if all_c1_ready else "not_ready",
            "severity": "blocking" if not all_c1_ready else "not_blocking",
            "required_for": "route decision A versus B",
            "recommended_resolution": "use existing labels as seed if ready; otherwise revise schema before top-up",
        },
    ]


def write_preflight(manifest: dict[str, Any]) -> None:
    required = [MANIFEST, TAXONOMY, LABEL_SCHEMA, DIAGNOSIS_MATRIX, COMSOL_FEASIBILITY, REPORT_ROUTE_SUMMARY, REGISTRY, CURRENT_BASELINE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing 25.9 preflight input(s): " + ", ".join(missing))
    npz_path = Path(manifest["generated_npz_path"])
    generator_path = Path(manifest["comsol_generator_script"])
    diagnosis_rows = read_csv(DIAGNOSIS_MATRIX)
    multi_rows = [row for row in diagnosis_rows if row.get("shape_type") == MULTI_PIT_SHAPE]
    decision_text = REPORT_ROUTE_SUMMARY.read_text(encoding="utf-8")
    forbidden_diff = git_value(["diff", "--name-only", "--", "CURRENT_BASELINE.md", "data", "checkpoints", "notes", "results/previews", "scripts/visualize_current_baseline.py"])
    lines = [
        "25.9 surface multi-pit component branch preflight",
        "",
        "scope: plan/audit only; no COMSOL, no training, no data/NPZ mutation, no CURRENT_BASELINE.md update.",
        f"dataset_id: {manifest.get('dataset_id')}",
        f"manifest_path: {MANIFEST}",
        f"npz_path_exists_for_read_only_audit: {npz_path.exists()}",
        f"npz_path: {npz_path}",
        f"shape_type_count_multi_pit_manifest: {manifest.get('shape_type_counts', {}).get(MULTI_PIT_SHAPE)}",
        f"representation_target_component_set_manifest: {manifest.get('representation_target_counts', {}).get('component_set')}",
        f"diagnosis_multi_pit_rows: {len(multi_rows)}",
        f"diagnosis_multi_pit_rbc_not_representable_rows: {sum(1 for row in multi_rows if row.get('diagnosis') == 'rbc_not_representable')}",
        f"25_8_route_contains_component_set_branch: {'A. component-set branch for multi-pit' in decision_text}",
        f"comsol_generator_reference_exists_read_only: {generator_path.exists()}",
        f"comsol_generator_reference_path: {generator_path}",
        f"forbidden_diff_present_before_25_9_outputs: {bool(forbidden_diff)}",
        "forbidden_diff_paths_before_25_9_outputs: " + (forbidden_diff if forbidden_diff else "none"),
        "",
        "checked_inputs:",
        f"- taxonomy_csv: {TAXONOMY}",
        f"- label_schema_csv: {LABEL_SCHEMA}",
        f"- oracle_vs_baseline_matrix: {DIAGNOSIS_MATRIX}",
        f"- comsol_feasibility_matrix: {COMSOL_FEASIBILITY}",
        f"- 25.8_route_summary: {REPORT_ROUTE_SUMMARY}",
        f"- registry: {REGISTRY}",
        "",
        "guardrails:",
        "- Multi-pit must not be counted as six-parameter RBC refinement success.",
        "- COMSOL_Multiphysics_MCP is read-only context in this stage.",
        "- Generated data/NPZ/checkpoints/previews/notes/baseline docs are forbidden.",
    ]
    PREFLIGHT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(audit_rows: list[dict[str, Any]], missing_rows: list[dict[str, Any]]) -> None:
    split_counts = Counter(str(row["split"]) for row in audit_rows)
    sufficiency_counts = Counter(str(row["label_sufficiency"]) for row in audit_rows)
    separation_counts = Counter(str(row["separation_bucket"]) for row in audit_rows)
    topology_counts = Counter(str(row["topology_bucket"]) for row in audit_rows)
    orientation_counts = Counter(str(row["orientation_bucket"]) for row in audit_rows)
    lines = [
        "25.9 surface multi-pit component label audit",
        "",
        f"multi_pit_sample_count: {len(audit_rows)}",
        f"split_counts: {dict(split_counts)}",
        f"component_count_values: {dict(Counter(str(row['component_count']) for row in audit_rows))}",
        f"component_json_count_values: {dict(Counter(str(row['component_json_count']) for row in audit_rows))}",
        f"label_sufficiency_counts: {dict(sufficiency_counts)}",
        f"separation_bucket_counts: {dict(separation_counts)}",
        f"topology_bucket_counts: {dict(topology_counts)}",
        f"orientation_bucket_counts: {dict(orientation_counts)}",
        "",
        "audit_result: existing 16 multi-pit samples are usable as a C1 fixed-K seed if all rows remain sufficient_for_C1_seed_with_schema_gaps.",
        "schema_gap_policy: rotation, component-level masks/depth grids, and explicit separation/topology labels must be added to the top-up schema.",
        "rbc_boundary: these rows stay excluded_negative_control for six-parameter RBC refinement success accounting.",
        "",
        f"audit_csv: {AUDIT_CSV}",
        f"missing_fields_csv: {MISSING_FIELDS_CSV}",
        "",
        "missing_field_summary:",
    ]
    for row in missing_rows:
        lines.append(f"- {row['field_name']}: {row['current_status']} ({row['severity']})")
    AUDIT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    manifest = load_manifest()
    write_preflight(manifest)
    npz_path = Path(manifest["generated_npz_path"])
    if not npz_path.exists():
        raise FileNotFoundError(f"explicit manifest NPZ path is missing for read-only label audit: {npz_path}")
    audit_rows = build_audit_rows(npz_path)
    if len(audit_rows) != 16:
        raise RuntimeError(f"expected 16 multi-pit samples, found {len(audit_rows)}")
    missing_rows = build_missing_rows(audit_rows)
    write_csv(AUDIT_CSV, audit_rows, AUDIT_FIELDS)
    write_csv(MISSING_FIELDS_CSV, missing_rows, MISSING_FIELDS)
    write_summary(audit_rows, missing_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
