#!/usr/bin/env python
"""Design the 20.74 top-up plan for the true-3D RBC v2_120 dataset."""

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

import design_true_3d_rbc_pilot_pack_plan as base_plan  # noqa: E402
import design_true_3d_rbc_pilot_topup_plan as topup72  # noqa: E402


TOPUP_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v2_topup_20_74"
ASSEMBLED_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v2_120"
SOURCE_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled"
MESH_TEMP_DIR = ROOT / "data/comsol_mfl/generated/temp_true_3d_rbc_dataset_120_topup_meshes"

DEFAULT_AUDIT = ROOT / "results/metrics/true_3d_rbc_v1_dataset_expansion_audit.csv"
DEFAULT_TARGETS = ROOT / "results/metrics/true_3d_rbc_v1_missing_expansion_targets.csv"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_dataset_120_topup_plan_summary.txt"
DEFAULT_PLAN = ROOT / "results/metrics/true_3d_rbc_dataset_120_topup_plan.csv"
DEFAULT_COVERAGE = ROOT / "results/metrics/true_3d_rbc_dataset_120_expected_coverage.csv"

COVERAGE_FIELDS = ["group_key", "group_value", "source_count", "topup_planned", "assembled_expected", "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.74 true-3D RBC v2_120 top-up plan.")
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--targets-csv", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--topup-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
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


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def depth_bounds(depth_bin: str) -> tuple[float, float]:
    return {
        "shallow": (0.0010, 0.0018),
        "medium": (0.0022, 0.0035),
        "deep": (0.00435, 0.0060),
    }[depth_bin]


def perturb(spec: base_plan.PilotSpec, variant: int) -> tuple[float, float, float, float, float, float, str]:
    patterns = [
        (0.985, 1.015, 0.98, -0.030, 0.020, -0.020),
        (1.015, 0.985, 1.02, 0.025, -0.030, 0.020),
        (0.970, 1.030, 0.96, -0.045, 0.030, 0.000),
        (1.030, 0.970, 1.04, 0.035, -0.045, -0.010),
    ]
    l_scale, w_scale, d_scale, dwld, dwwd, dwlw = patterns[(variant - 1) % len(patterns)]
    jitter = 1.0 + 0.0015 * max(0, variant - 1)
    lo_d, hi_d = depth_bounds(spec.depth_bin)
    l_m = clip(spec.L_m * l_scale * jitter, 0.010, 0.030)
    w_m = clip(spec.W_m * w_scale / jitter, 0.006, 0.020)
    d_m = clip(spec.D_m * d_scale * (1.0 + 0.0008 * max(0, variant - 1)), lo_d, hi_d)
    reason = "cell-balancing perturbation"
    if spec.depth_bin == "deep" and spec.aspect_bin == "narrow":
        l_m = clip(spec.L_m * 0.93, 0.010, 0.0275)
        w_m = clip(spec.W_m * 1.08, 0.0065, 0.0110)
        d_m = clip(spec.D_m * 0.90, 0.00435, 0.0053)
        reason = "controlled deep-elongated replacement to reduce timeout risk while preserving bin"
    if spec.curvature_template == "WD_dominant" and spec.depth_bin in {"medium", "deep"}:
        d_m = clip(d_m * 0.97, lo_d, hi_d)
        w_m = clip(w_m * 0.98, 0.006, 0.020)
        reason = "controlled WD replacement after prior mesh/domain failures"
    wld = clip(spec.wLD + dwld + 0.001 * variant, 0.55, 1.20)
    wwd = clip(spec.wWD + dwwd - 0.001 * variant, 0.55, 1.20)
    wlw = clip(spec.wLW + dwlw + 0.0005 * variant, 0.55, 1.20)
    return l_m, w_m, d_m, wld, wwd, wlw, reason


def split_for(counter: Counter[str], target: dict[str, int]) -> str:
    deficits = {key: target[key] - counter.get(key, 0) for key in target}
    return max(deficits, key=lambda key: (deficits[key], key == "train"))


def build_cell_specs() -> dict[tuple[str, str, str], base_plan.PilotSpec]:
    return {(spec.curvature_template, spec.depth_bin, spec.size_bin): spec for spec in base_plan.pilot_specs()}


def normalize_row(row: dict[str, Any], source_cell: tuple[str, str, str], source_count: int, order: int, risky: bool) -> dict[str, Any]:
    row = dict(row)
    row["dataset_id"] = TOPUP_DATASET_ID
    row["source_stage"] = "20.74_expansion_topup"
    row["intended_split"] = row["split"]
    row["source_sample_id"] = ""
    row["replacement_for_failure_pattern"] = "deep_elongated_or_wd_mesh_risk" if risky else ""
    row["replacement_reason"] = row.get("replacement_reason", "")
    row["exact_piao_rbc"] = "False"
    row["rbc_style_approximation"] = "True"
    row["geometry_method"] = "imported_watertight_mesh_solid"
    row["allowed_use"] = "schema_validation, assembly_input"
    row["forbidden_use"] = "automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate"
    row["temp_mesh_output_path"] = str(MESH_TEMP_DIR / f"{row['sample_id']}.stl")
    params = json.loads(row["geometry_params_json"])
    params.update(
        {
            "dataset_id": TOPUP_DATASET_ID,
            "assembled_dataset_id": ASSEMBLED_DATASET_ID,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_stage": "20.74_expansion_topup",
            "source_coverage_cell": "|".join(source_cell),
            "source_cell_current_count": source_count,
            "execution_order": order,
            "risk_bucket": "buffer" if risky else "target",
            "exact_piao_rbc": False,
            "rbc_style_approximation": True,
            "allowed_use": ["schema_validation", "assembly_input"],
            "forbidden_use": [
                "automatic_mainline_training",
                "baseline_update",
                "current_baseline_replacement",
                "latest_newest_auto_discovery",
                "direct_training_without_manifest_gate",
            ],
        }
    )
    row["geometry_params_json"] = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return row


def make_row(spec: base_plan.PilotSpec, sample_id: str, split: str, variant: int, source_count: int, order: int, risky: bool) -> dict[str, Any]:
    l_m, w_m, d_m, wld, wwd, wlw, reason = perturb(spec, variant)
    row = topup72.custom_replacement_row(
        sample_id,
        split,
        spec.depth_bin,
        spec.size_bin,
        spec.aspect_bin,
        spec.curvature_template,
        l_m,
        w_m,
        d_m,
        wld,
        wwd,
        wlw,
        reason,
    )
    return normalize_row(row, (spec.curvature_template, spec.depth_bin, spec.size_bin), source_count, order, risky)


def build_topup_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    source_counts = Counter((row["curvature_template"], row["depth_bin"], row["size_bin"]) for row in audit_rows)
    cells = build_cell_specs()
    target_rows: list[tuple[base_plan.PilotSpec, int, int]] = []
    for cell, spec in cells.items():
        current = source_counts.get(cell, 0)
        missing = max(0, 2 - current)
        for variant in range(1, missing + 1):
            target_rows.append((spec, current, variant))
    low_risk: list[tuple[base_plan.PilotSpec, int, int]] = []
    risky_rows: list[tuple[base_plan.PilotSpec, int, int]] = []
    for item in target_rows:
        spec = item[0]
        risky = spec.depth_bin == "deep" and spec.aspect_bin == "narrow" or (
            spec.curvature_template == "WD_dominant" and spec.depth_bin in {"medium", "deep"} and spec.size_bin in {"small_compact", "elongated"}
        )
        (risky_rows if risky else low_risk).append(item)

    rows: list[dict[str, Any]] = []
    split_counter: Counter[str] = Counter()
    target_split = {"train": 44, "val": 10, "test": 10}
    ordered_target = (low_risk + risky_rows)[:64]
    for order, (spec, source_count, variant) in enumerate(ordered_target, start=1):
        split = split_for(split_counter, target_split)
        split_counter[split] += 1
        sample_id = f"rbc_v2topup_{order:03d}_{spec.depth_bin}_{spec.size_bin}_{spec.curvature_template}"
        risky = spec.depth_bin == "deep" and spec.aspect_bin == "narrow" or (
            spec.curvature_template == "WD_dominant" and spec.depth_bin in {"medium", "deep"} and spec.size_bin in {"small_compact", "elongated"}
        )
        rows.append(make_row(spec, sample_id, split, variant, source_count, order, risky))

    buffer_pool = risky_rows + low_risk
    buffer_target = {"train": 56, "val": 12, "test": 12}
    buffer_index = 0
    while len(rows) < 80:
        spec, source_count, variant = buffer_pool[buffer_index % len(buffer_pool)]
        buffer_index += 1
        split = split_for(split_counter, buffer_target)
        split_counter[split] += 1
        order = len(rows) + 1
        sample_id = f"rbc_v2topup_{order:03d}_buffer_{spec.depth_bin}_{spec.size_bin}_{spec.curvature_template}"
        rows.append(make_row(spec, sample_id, split, variant + 2 + buffer_index, source_count, order, True))
    return rows


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.topup_plan, args.coverage], args.overwrite)
    topup72.TOPUP_DATASET_ID = TOPUP_DATASET_ID
    topup72.MESH_TEMP_DIR = MESH_TEMP_DIR
    audit_rows = read_csv(args.audit_csv)
    rows = build_topup_rows(audit_rows)
    if len(rows) != 80:
        raise RuntimeError(f"expected 80 top-up rows, got {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate sample_id in top-up plan")
    source_keys = {(row["L_m"], row["W_m"], row["D_m"], row["wLD"], row["wWD"], row["wLW"]) for row in audit_rows}
    topup_keys = {(row["L_m"], row["W_m"], row["D_m"], row["wLD"], row["wWD"], row["wLW"]) for row in rows}
    if source_keys & topup_keys:
        raise RuntimeError("top-up duplicates a v1 successful six-parameter tuple")
    if len(topup_keys) != len(rows):
        raise RuntimeError("duplicate six-parameter tuple inside top-up plan")

    fields = list(dict.fromkeys(base_plan.PLAN_FIELDS + ["source_stage", "intended_split", "source_sample_id", "replacement_for_failure_pattern", "replacement_reason"]))
    write_csv(args.topup_plan, rows, fields)

    source = {
        "split": Counter(row["split"] for row in audit_rows),
        "curvature_template": Counter(row["curvature_template"] for row in audit_rows),
        "depth_bin": Counter(row["depth_bin"] for row in audit_rows),
        "size_bin": Counter(row["size_bin"] for row in audit_rows),
    }
    topup = {
        "split": Counter(row["split"] for row in rows),
        "curvature_template": Counter(row["curvature_template"] for row in rows[:64]),
        "depth_bin": Counter(row["depth_bin"] for row in rows[:64]),
        "size_bin": Counter(row["size_bin"] for row in rows[:64]),
    }
    targets = {
        "split": {"train": 80, "val": 20, "test": 20},
        "curvature_template": {"sharp": 24, "round": 24, "boxy": 24, "LD_dominant": 24, "WD_dominant": 24},
        "depth_bin": {"shallow": 40, "medium": 40, "deep": 40},
        "size_bin": {"small_compact": 30, "medium_balanced": 30, "large_wide": 30, "elongated": 30},
    }
    coverage_rows: list[dict[str, Any]] = []
    for group_key, target_map in targets.items():
        for group_value, target in target_map.items():
            source_count = source[group_key].get(group_value, 0)
            topup_count = topup[group_key].get(group_value, 0)
            coverage_rows.append(
                {
                    "group_key": group_key,
                    "group_value": group_value,
                    "source_count": source_count,
                    "topup_planned": topup_count,
                    "assembled_expected": source_count + topup_count,
                    "target": target,
                }
            )
    write_csv(args.coverage, coverage_rows, COVERAGE_FIELDS)

    split_counts = Counter(row["split"] for row in rows)
    curv_counts = Counter(row["curvature_template"] for row in rows[:64])
    depth_counts = Counter(row["depth_bin"] for row in rows[:64])
    risky_count = sum(1 for row in rows if row["replacement_for_failure_pattern"])
    lines = [
        "20.74 true 3D RBC v2_120 top-up plan summary",
        "",
        f"topup_dataset_id: {TOPUP_DATASET_ID}",
        f"assembled_dataset_id: {ASSEMBLED_DATASET_ID}",
        f"planned_rows: {len(rows)}",
        f"planned_split_counts: {dict(split_counts)}",
        f"first_64_curvature_counts: {dict(curv_counts)}",
        f"first_64_depth_counts: {dict(depth_counts)}",
        f"buffer_or_risk_rows: {risky_count}",
        "target_success_rows: 64",
        "target_success_split: train=44, val=10, test=10",
        "exact_piao_rbc: False",
        "rbc_style_approximation: True",
        "rotation: disabled; angle_rad=0 for all rows",
        "",
        "Strategy:",
        "- The plan prioritizes source coverage cells with fewer than two successful samples.",
        "- Low-risk rows are placed before high-risk deep elongated / WD rows.",
        "- Rows 65-80 are buffer rows for bounded replacement during COMSOL generation.",
        "- No high-layer fallback or lower-source solve is permitted.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if split_counts != Counter({"train": 56, "val": 12, "test": 12}):
        raise RuntimeError(f"unexpected top-up split plan: {dict(split_counts)}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
