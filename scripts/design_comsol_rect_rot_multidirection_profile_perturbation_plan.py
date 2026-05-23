from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PLAN = PROJECT_ROOT / "results/metrics/comsol_rect_rot_multiaxis_profile_perturbation_plan.csv"
DEFAULT_SOURCE_PACK = (
    PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_rect_rot_multiaxis_profile_perturbation_forward_pack_v1.npz"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_multidirection_profile_perturbation_plan_summary.txt"
DEFAULT_PLAN = PROJECT_ROOT / "results/metrics/comsol_rect_rot_multidirection_profile_perturbation_plan.csv"

SENSOR_Z_M = 0.008
AXIS_NAMES = ("Bx", "By", "Bz")
AXIS_EXPRESSIONS = ("mf.Bx", "mf.By", "mf.Bz")
DIRECTION_NAMES = ("direction_0", "direction_45", "direction_90")
DIRECTION_ANGLES_DEG = (0.0, 45.0, 90.0)
DIRECTION_JE = {
    "direction_0": ["0", "1e6[A/m^2]", "0"],
    "direction_45": ["7.071067811865475e5[A/m^2]", "7.071067811865475e5[A/m^2]", "0"],
    "direction_90": ["1e6[A/m^2]", "0", "0"],
}
VARIANT_TYPES = (
    "true_reference",
    "profile_extracted_reference",
    "half_width_shrink_local",
    "half_width_expand_local",
    "smooth_global_width_scale",
    "centerline_offset_small",
    "roughness_noise",
    "mixed_profile_perturbation",
)
TARGET_COUNTS = {
    ("train", "rectangular_notch"): 4,
    ("train", "rotated_rect"): 4,
    ("val", "rectangular_notch"): 1,
    ("val", "rotated_rect"): 1,
    ("test", "rectangular_notch"): 1,
    ("test", "rotated_rect"): 1,
}
MIN_COUNTS = {
    ("train", "rectangular_notch"): 2,
    ("train", "rotated_rect"): 2,
    ("val", "rectangular_notch"): 1,
    ("val", "rotated_rect"): 1,
    ("test", "rectangular_notch"): 1,
    ("test", "rotated_rect"): 1,
}
MIN_BASE_SAMPLES = 8
MIN_PROFILE_ROWS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.64 multi-direction profile perturbation plan.")
    parser.add_argument("--source-plan", type=Path, default=DEFAULT_SOURCE_PLAN)
    parser.add_argument("--source-pack", type=Path, default=DEFAULT_SOURCE_PACK)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--plan-out", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--target-base-samples", type=int, default=12)
    parser.add_argument("--minimum-base-samples", type=int, default=MIN_BASE_SAMPLES)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def residual_nrmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    ref = np.asarray(b, dtype=np.float64)
    denom = float(np.std(ref))
    if denom <= 1.0e-12:
        denom = float(np.sqrt(np.mean(ref**2)))
    return float(np.sqrt(np.mean(diff**2)) / max(denom, 1.0e-12))


def base_oracle_scores(rows: list[dict[str, str]], pack: np.lib.npyio.NpzFile) -> dict[str, dict[str, float]]:
    sample_ids = pack["sample_ids"].astype(str)
    id_to_idx = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    axis_names = pack["axis_names"].astype(str).tolist()
    bz_idx = axis_names.index("Bz")
    by_base: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_base.setdefault(row["base_sample_id"], []).append(row)

    scores: dict[str, dict[str, float]] = {}
    for base_id, local_rows in by_base.items():
        true_rows = [row for row in local_rows if row["variant_type"] == "true_reference"]
        if not true_rows or true_rows[0]["sample_id"] not in id_to_idx:
            continue
        reference = pack["delta_b_multi_axis"][id_to_idx[true_rows[0]["sample_id"]], bz_idx]
        residuals: list[float] = []
        quality_scores: list[float] = []
        for row in local_rows:
            if row["sample_id"] not in id_to_idx:
                continue
            residuals.append(residual_nrmse(pack["delta_b_multi_axis"][id_to_idx[row["sample_id"]], bz_idx], reference))
            iou = safe_float(row["quality_iou_vs_true"])
            dice = safe_float(row["quality_dice_vs_true"])
            area = safe_float(row["quality_area_error_vs_true"])
            quality_scores.append(iou + dice - area)
        ok = total = 0
        for i in range(len(residuals)):
            for j in range(i + 1, len(residuals)):
                if abs(quality_scores[i] - quality_scores[j]) <= 1.0e-12:
                    continue
                ok += int((quality_scores[i] > quality_scores[j]) == (residuals[i] < residuals[j]))
                total += 1
        scores[base_id] = {
            "singleheight_0p008_base_oracle_ordering": ok / total if total else math.nan,
            "singleheight_0p008_base_pair_count": float(total),
        }
    return scores


def select_extremes(group: list[str], scores: dict[str, dict[str, float]], count: int, prefer_high_first: bool) -> list[str]:
    def score(base_id: str) -> float:
        value = scores.get(base_id, {}).get("singleheight_0p008_base_oracle_ordering", math.nan)
        return 0.5 if not np.isfinite(value) else float(value)

    ordered = sorted(group, key=score)
    selected: list[str] = []
    lo, hi = 0, len(ordered) - 1
    take_high = prefer_high_first
    while len(selected) < count and lo <= hi:
        candidate = ordered[hi] if take_high else ordered[lo]
        if take_high:
            hi -= 1
        else:
            lo += 1
        if candidate not in selected:
            selected.append(candidate)
        take_high = not take_high
    return selected


def select_bases(
    rows: list[dict[str, str]],
    scores: dict[str, dict[str, float]],
    target_counts: dict[tuple[str, str], int],
) -> list[str]:
    by_group: dict[tuple[str, str], list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["split"], row["source_defect_type"], row["base_sample_id"])
        if row["variant_type"] != "true_reference" or key in seen:
            continue
        seen.add(key)
        by_group.setdefault((row["split"], row["source_defect_type"]), []).append(row["base_sample_id"])
    selected: list[str] = []
    for idx, (key, count) in enumerate(target_counts.items()):
        group = by_group.get(key, [])
        if len(group) < count:
            raise ValueError(f"Not enough bases for {key}: need {count}, found {len(group)}")
        selected.extend(select_extremes(group, scores, count, prefer_high_first=bool(idx % 2)))
    return selected


def plan_fields(input_fields: list[str]) -> list[str]:
    extras = [
        "source_20_63_sample_id",
        "source_20_63_geometry_hash",
        "sensor_z_m",
        "axis_names_json",
        "axis_expressions_json",
        "axis_count",
        "direction_names_json",
        "direction_angles_deg_json",
        "direction_je_json",
        "direction_count_target",
        "direction_count_minimum",
        "profile_row_count_unit",
        "total_direction_axis_observations_for_row_target",
        "reuse_direction0_from_20_63_available",
        "true_reference_requires_new_direction_forward",
        "planned_real_comsol_forward_directions_target",
        "planned_real_comsol_forward_directions_minimum",
        "planned_real_comsol_forward_rows",
        "planned_real_comsol_forward_observations",
        "samepack_direction0_bz_base_oracle_ordering",
        "samepack_direction0_bz_base_pair_count",
        "multidirection_plan_stage",
    ]
    fields = list(input_fields)
    insert_at = fields.index("sample_id") + 1
    for extra in reversed(extras):
        if extra not in fields:
            fields.insert(insert_at, extra)
    return fields


def make_plan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    source_rows = read_csv(args.source_plan)
    pack = np.load(args.source_pack, allow_pickle=True)
    scores = base_oracle_scores(source_rows, pack)
    target_used = True
    try:
        selected_bases = select_bases(source_rows, scores, TARGET_COUNTS)
    except ValueError:
        selected_bases = select_bases(source_rows, scores, MIN_COUNTS)
        target_used = False
    selected_set = set(selected_bases)
    selected_rows = [row for row in source_rows if row["base_sample_id"] in selected_set]

    rows: list[dict[str, Any]] = []
    for base_pos, base_id in enumerate(selected_bases):
        local = [row for row in selected_rows if row["base_sample_id"] == base_id]
        local = sorted(local, key=lambda row: VARIANT_TYPES.index(row["variant_type"]))
        for variant_pos, row in enumerate(local):
            out = dict(row)
            out["source_20_63_sample_id"] = row["sample_id"]
            out["source_20_63_geometry_hash"] = row.get("geometry_hash", "")
            out["sample_id"] = f"multidirection_profile_perturb_{base_pos:03d}_{variant_pos:02d}"
            out["sensor_z_m"] = SENSOR_Z_M
            out["axis_names_json"] = json.dumps(list(AXIS_NAMES))
            out["axis_expressions_json"] = json.dumps(list(AXIS_EXPRESSIONS))
            out["axis_count"] = len(AXIS_NAMES)
            out["direction_names_json"] = json.dumps(list(DIRECTION_NAMES))
            out["direction_angles_deg_json"] = json.dumps(list(DIRECTION_ANGLES_DEG))
            out["direction_je_json"] = json.dumps(DIRECTION_JE, sort_keys=True)
            out["direction_count_target"] = len(DIRECTION_NAMES)
            out["direction_count_minimum"] = 2
            out["profile_row_count_unit"] = 1
            out["total_direction_axis_observations_for_row_target"] = len(AXIS_NAMES) * len(DIRECTION_NAMES)
            out["reuse_direction0_from_20_63_available"] = 1
            out["true_reference_requires_new_direction_forward"] = 1
            out["planned_real_comsol_forward_directions_target"] = len(DIRECTION_NAMES) - 1
            out["planned_real_comsol_forward_directions_minimum"] = 1
            out["planned_real_comsol_forward_rows"] = 1
            out["planned_real_comsol_forward_observations"] = (len(DIRECTION_NAMES) - 1) * len(AXIS_NAMES)
            out["samepack_direction0_bz_base_oracle_ordering"] = scores.get(base_id, {}).get(
                "singleheight_0p008_base_oracle_ordering", math.nan
            )
            out["samepack_direction0_bz_base_pair_count"] = scores.get(base_id, {}).get(
                "singleheight_0p008_base_pair_count", 0.0
            )
            out["multidirection_plan_stage"] = "20.64_multidirection_profile_perturbation"
            rows.append(out)

    summary = {
        "selected_base_samples": len(selected_bases),
        "profile_rows": len(rows),
        "direction_count_target": len(DIRECTION_NAMES),
        "direction_count_minimum": 2,
        "axis_count": len(AXIS_NAMES),
        "total_direction_axis_observations_target": len(rows) * len(DIRECTION_NAMES) * len(AXIS_NAMES),
        "reuse_direction0_from_20_63_rows": len(rows),
        "planned_real_comsol_forward_rows": len(rows),
        "planned_real_comsol_forward_observations_target": len(rows) * (len(DIRECTION_NAMES) - 1) * len(AXIS_NAMES),
        "split_distribution": dict(Counter(row["split"] for row in rows)),
        "source_defect_type_distribution": dict(Counter(row["source_defect_type"] for row in rows)),
        "variant_distribution": dict(Counter(row["variant_type"] for row in rows)),
        "base_split_type_distribution": dict(
            Counter((row["split"], row["source_defect_type"]) for row in rows if row["variant_type"] == "true_reference")
        ),
        "polygon_valid_rows": sum(int(row["polygon_valid"]) for row in rows),
        "mask_non_empty_rows": sum(int(row["mask_non_empty"]) for row in rows),
    }
    return rows, summary, target_used


def write_summary(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace, target_used: bool) -> None:
    base_ids = sorted({row["base_sample_id"] for row in rows})
    variants_complete = all(sum(1 for row in rows if row["base_sample_id"] == base_id) == len(VARIANT_TYPES) for base_id in base_ids)
    minimum_gate = (
        len(base_ids) >= args.minimum_base_samples
        and len(rows) >= MIN_PROFILE_ROWS
        and len(DIRECTION_NAMES) >= 2
        and "direction_90" in DIRECTION_NAMES
        and len(AXIS_NAMES) >= 1
        and {"train", "val", "test"}.issubset(set(row["split"] for row in rows))
        and {"rectangular_notch", "rotated_rect"}.issubset(set(row["source_defect_type"] for row in rows))
        and variants_complete
        and summary["polygon_valid_rows"] == len(rows)
        and summary["mask_non_empty_rows"] == len(rows)
    )
    target_gate = (
        len(base_ids) >= args.target_base_samples
        and len(rows) >= args.target_base_samples * len(VARIANT_TYPES)
        and variants_complete
    )
    lines = [
        "COMSOL rect/rot multi-direction profile perturbation plan summary",
        "",
        f"source_plan: {args.source_plan}",
        f"source_pack: {args.source_pack}",
        f"sensor_z_m: {SENSOR_Z_M}",
        f"direction_names: {list(DIRECTION_NAMES)}",
        f"direction_angles_deg: {list(DIRECTION_ANGLES_DEG)}",
        f"direction_je: {DIRECTION_JE}",
        f"axis_names: {list(AXIS_NAMES)}",
        f"axis_expressions: {list(AXIS_EXPRESSIONS)}",
        f"target_base_samples: {args.target_base_samples}",
        f"minimum_base_samples: {args.minimum_base_samples}",
        f"target_selection_used: {target_used}",
        f"represented_base_samples: {len(base_ids)}",
        f"profile_rows: {summary['profile_rows']}",
        f"direction_count_target: {summary['direction_count_target']}",
        f"direction_count_minimum: {summary['direction_count_minimum']}",
        f"axis_count: {summary['axis_count']}",
        f"total_direction_axis_observations_target: {summary['total_direction_axis_observations_target']}",
        f"reuse_direction0_from_20_63_rows: {summary['reuse_direction0_from_20_63_rows']}",
        f"planned_real_comsol_forward_rows: {summary['planned_real_comsol_forward_rows']}",
        f"planned_real_comsol_forward_observations_target: {summary['planned_real_comsol_forward_observations_target']}",
        f"split_distribution: {summary['split_distribution']}",
        f"source_defect_type_distribution: {summary['source_defect_type_distribution']}",
        f"variant_distribution: {summary['variant_distribution']}",
        f"base_split_type_distribution: {summary['base_split_type_distribution']}",
        f"polygon_valid_rows: {summary['polygon_valid_rows']}",
        f"mask_non_empty_rows: {summary['mask_non_empty_rows']}",
        f"all_variants_complete_per_base: {variants_complete}",
        f"target_gate_passed: {target_gate}",
        f"minimum_gate_passed: {minimum_gate}",
        "",
        "Selection note:",
        "- Bases are selected from the 20.63 multi-axis profile perturbation plan for same-geometry comparison.",
        "- Selection alternates easy and hard same-pack direction_0 Bz oracle-ordering bases within each split/source type.",
        "- direction_0 can reuse exact 20.63 default-direction Bx/By/Bz rows in the 20.64 pack.",
        "- direction_90 must be generated by real COMSOL forward with changed Je and is a hard blocker if unavailable.",
        "- direction_45 is attempted only when the direction probe is stable; otherwise the pack continues with direction_0/direction_90.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not minimum_gate:
        raise RuntimeError("multi-direction profile plan failed minimum gate")


def main() -> None:
    args = parse_args()
    rows, summary, target_used = make_plan(args)
    fields = plan_fields(list(read_csv(args.source_plan)[0].keys()))
    write_csv(args.plan_out, rows, fields)
    write_summary(args.summary, rows, summary, args, target_used)
    print(f"Wrote {len(rows)} rows to {args.plan_out}")


if __name__ == "__main__":
    main()
