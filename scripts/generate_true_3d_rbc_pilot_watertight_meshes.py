#!/usr/bin/env python
"""Generate batch watertight STL meshes for the 20.71 RBC pilot plan.

The STL files are generated artifacts under data/ and must not be committed.
Only the summary and validation metrics are intended for git.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_true_3d_watertight_rbc_mesh as meshlib  # noqa: E402


DEFAULT_PLAN = ROOT / "results/metrics/true_3d_rbc_pilot_pack_plan.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_watertight_mesh_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_rbc_pilot_watertight_mesh_metrics.csv"

MIN_SUCCESS = 30
FULL_SUCCESS = 54


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 20.71 batch watertight pilot meshes.")
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--max-samples", type=int, default=60)
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


def bool_value(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def write_summary(path: Path, rows: list[dict[str, Any]], failures: list[str], planned_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pass_rows = [row for row in rows if bool_value(row, "mesh_validation_pass")]
    split_counts = Counter(row.get("split", "") for row in pass_rows)
    depth_counts = Counter(row.get("depth_bin", "") for row in pass_rows)
    curvature_counts = Counter(row.get("curvature_template", "") for row in pass_rows)
    status = "full_mesh_pass" if len(pass_rows) >= FULL_SUCCESS else "partial_mesh_pass" if len(pass_rows) >= MIN_SUCCESS else "failed"
    lines = [
        "20.71 true 3D RBC pilot watertight mesh summary",
        "",
        f"planned_count: {planned_count}",
        f"mesh_pass_count: {len(pass_rows)}",
        f"mesh_fail_count: {planned_count - len(pass_rows)}",
        f"mesh_stage_status: {status}",
        f"split_counts_pass: {dict(split_counts)}",
        f"depth_bin_counts_pass: {dict(depth_counts)}",
        f"curvature_template_counts_pass: {dict(curvature_counts)}",
        "mesh_units: m",
        "top_cap_plane: z=0",
        "depth_sign_convention: bottom surface z=-depth",
        "mesh_source: triangulated_depth_grid",
        "",
        "Gate:",
        f"- full threshold: >= {FULL_SUCCESS}",
        f"- minimum partial threshold: >= {MIN_SUCCESS}",
        "- high-layer fallback is not used by this stage.",
        "",
        "Failures:",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics], args.overwrite)
    plan_rows = read_csv(args.plan_csv)[: args.max_samples]
    if not plan_rows:
        raise RuntimeError(f"plan CSV has no rows: {args.plan_csv}")
    metrics: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in plan_rows:
        sample_id = row["sample_id"]
        try:
            vertices, faces, meta = meshlib.build_mesh(row)
            metric = meshlib.validate_mesh(row, vertices, faces, meta)
            metric["dataset_id"] = row.get("dataset_id", "")
            metric["split"] = row.get("split", row.get("split_tag", ""))
            metric["depth_bin"] = row.get("depth_bin", "")
            metric["size_bin"] = row.get("size_bin", "")
            metric["aspect_bin"] = row.get("aspect_bin", "")
            metric["curvature_template"] = row.get("curvature_template", "")
            metric["geometry_method"] = row.get("geometry_method", "imported_watertight_mesh_solid")
            if metric["mesh_validation_pass"]:
                meshlib.write_ascii_stl(Path(row["temp_mesh_output_path"]), vertices, faces)
            else:
                failures.append(f"{sample_id}: mesh validation failed")
            metrics.append(metric)
            print(f"MESH {sample_id} pass={metric['mesh_validation_pass']} vertices={metric['vertex_count']} faces={metric['face_count']}")
        except Exception as exc:
            failure = f"{sample_id}: {exc}"
            failures.append(failure)
            metric = {
                "sample_id": sample_id,
                "mesh_validation_pass": False,
                "dataset_id": row.get("dataset_id", ""),
                "split": row.get("split", row.get("split_tag", "")),
                "depth_bin": row.get("depth_bin", ""),
                "size_bin": row.get("size_bin", ""),
                "aspect_bin": row.get("aspect_bin", ""),
                "curvature_template": row.get("curvature_template", ""),
                "geometry_method": row.get("geometry_method", "imported_watertight_mesh_solid"),
                "notes": failure,
            }
            metrics.append(metric)
            print(f"MESH_FAIL {failure}")
    fields = list(dict.fromkeys(meshlib.METRIC_FIELDS + ["dataset_id", "split", "depth_bin", "size_bin", "aspect_bin", "curvature_template", "geometry_method"]))
    write_csv(args.metrics, metrics, fields)
    write_summary(args.summary, metrics, failures, len(plan_rows))
    pass_count = sum(1 for row in metrics if bool_value(row, "mesh_validation_pass"))
    if pass_count < MIN_SUCCESS:
        raise RuntimeError(f"mesh stage failed: pass_count={pass_count} < {MIN_SUCCESS}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
