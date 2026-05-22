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
DEFAULT_EXPANDED_PLAN = PROJECT_ROOT / "results/metrics/comsol_rect_rot_expanded_profile_perturbation_plan.csv"
DEFAULT_EXPANDED_PACK = (
    PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_rect_rot_expanded_profile_perturbation_forward_pack_v1.npz"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_multiheight_profile_perturbation_plan_summary.txt"
DEFAULT_PLAN = PROJECT_ROOT / "results/metrics/comsol_rect_rot_multiheight_profile_perturbation_plan.csv"

HEIGHTS = (0.004, 0.008, 0.012)
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
MIN_BASE_SAMPLES = 8
MIN_PROFILE_ROWS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design 20.62 multi-height profile perturbation plan.")
    parser.add_argument("--expanded-plan", type=Path, default=DEFAULT_EXPANDED_PLAN)
    parser.add_argument("--expanded-pack", type=Path, default=DEFAULT_EXPANDED_PACK)
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
    denom = float(np.std(b))
    if denom <= 1.0e-12:
        denom = float(np.sqrt(np.mean(np.asarray(b, dtype=np.float64) ** 2)))
    return float(np.sqrt(np.mean(diff**2)) / max(denom, 1.0e-12))


def base_oracle_scores(rows: list[dict[str, str]], pack: np.lib.npyio.NpzFile) -> dict[str, dict[str, float]]:
    sample_ids = pack["sample_ids"].astype(str)
    id_to_idx = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    by_base: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_base.setdefault(row["base_sample_id"], []).append(row)
    scores: dict[str, dict[str, float]] = {}
    for base_id, local_rows in by_base.items():
        true_rows = [row for row in local_rows if row["variant_type"] == "true_reference"]
        if not true_rows or true_rows[0]["sample_id"] not in id_to_idx:
            continue
        true_idx = id_to_idx[true_rows[0]["sample_id"]]
        reference = pack["delta_bz"][true_idx]
        residuals: list[float] = []
        quality_scores: list[float] = []
        for row in local_rows:
            if row["sample_id"] not in id_to_idx:
                continue
            idx = id_to_idx[row["sample_id"]]
            residuals.append(residual_nrmse(pack["delta_bz"][idx], reference))
            iou = safe_float(row["quality_iou_vs_true"])
            dice = safe_float(row["quality_dice_vs_true"])
            area = safe_float(row["quality_area_error_vs_true"])
            quality_scores.append(iou + dice - area)
        ok = total = 0
        for i in range(len(residuals)):
            for j in range(i + 1, len(residuals)):
                if abs(quality_scores[i] - quality_scores[j]) <= 1.0e-12:
                    continue
                i_better = quality_scores[i] > quality_scores[j]
                residual_prefers_i = residuals[i] < residuals[j]
                ok += int(i_better == residual_prefers_i)
                total += 1
        vals = [safe_float(row["quality_iou_vs_true"]) for row in local_rows]
        scores[base_id] = {
            "singleheight_0p008_base_oracle_ordering": ok / total if total else math.nan,
            "singleheight_0p008_base_pair_count": float(total),
            "mean_quality_iou": float(np.nanmean(vals)) if vals else math.nan,
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
        if take_high:
            candidate = ordered[hi]
            hi -= 1
        else:
            candidate = ordered[lo]
            lo += 1
        if candidate not in selected:
            selected.append(candidate)
        take_high = not take_high
    return selected


def select_bases(rows: list[dict[str, str]], scores: dict[str, dict[str, float]]) -> list[str]:
    by_group: dict[tuple[str, str], list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["split"], row["source_defect_type"], row["base_sample_id"])
        if row["variant_type"] != "true_reference" or key in seen:
            continue
        seen.add(key)
        by_group.setdefault((row["split"], row["source_defect_type"]), []).append(row["base_sample_id"])
    selected: list[str] = []
    for idx, (key, count) in enumerate(TARGET_COUNTS.items()):
        group = by_group.get(key, [])
        if len(group) < count:
            raise ValueError(f"Not enough bases for {key}: need {count}, found {len(group)}")
        selected.extend(select_extremes(group, scores, count, prefer_high_first=bool(idx % 2)))
    if len(set(selected)) < 12:
        raise ValueError("Selected base sample count below target")
    return selected


def plan_fields(input_fields: list[str]) -> list[str]:
    extras = [
        "source_20_61_sample_id",
        "source_20_61_geometry_hash",
        "sensor_z_heights_m",
        "height_count",
        "profile_row_count_unit",
        "total_height_observations_for_row",
        "singleheight_0p008_reuse_available",
        "reuse_source_0p008",
        "planned_real_comsol_forward_observations",
        "observed_delta_bz_reference_by_height_json",
        "singleheight_0p008_base_oracle_ordering",
        "singleheight_0p008_base_pair_count",
        "multiheight_plan_stage",
    ]
    fields = list(input_fields)
    insert_at = fields.index("sample_id") + 1
    for extra in reversed(extras):
        fields.insert(insert_at, extra)
    return fields


def make_plan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded_rows = read_csv(args.expanded_plan)
    pack = np.load(args.expanded_pack, allow_pickle=True)
    scores = base_oracle_scores(expanded_rows, pack)
    selected_bases = select_bases(expanded_rows, scores)
    selected_set = set(selected_bases)
    selected_rows = [row for row in expanded_rows if row["base_sample_id"] in selected_set]
    rows: list[dict[str, Any]] = []
    for base_pos, base_id in enumerate(selected_bases):
        local = [row for row in selected_rows if row["base_sample_id"] == base_id]
        local = sorted(local, key=lambda row: VARIANT_TYPES.index(row["variant_type"]))
        for variant_pos, row in enumerate(local):
            out = dict(row)
            out["source_20_61_sample_id"] = row["sample_id"]
            out["source_20_61_geometry_hash"] = row.get("geometry_hash", "")
            out["sample_id"] = f"multiheight_profile_perturb_{base_pos:03d}_{variant_pos:02d}"
            out["sensor_z_heights_m"] = json.dumps(list(HEIGHTS))
            out["height_count"] = len(HEIGHTS)
            out["profile_row_count_unit"] = 1
            out["total_height_observations_for_row"] = len(HEIGHTS)
            out["singleheight_0p008_reuse_available"] = 1
            out["reuse_source_0p008"] = "20.61_expanded_profile_pack"
            out["planned_real_comsol_forward_observations"] = len(HEIGHTS) - 1
            out["observed_delta_bz_reference_by_height_json"] = json.dumps(
                {
                    "0.004": "to_be_generated_true_reference",
                    "0.008": row["observed_delta_bz_reference_sample_id"],
                    "0.012": "to_be_generated_true_reference",
                },
                sort_keys=True,
            )
            out["singleheight_0p008_base_oracle_ordering"] = scores.get(base_id, {}).get(
                "singleheight_0p008_base_oracle_ordering", math.nan
            )
            out["singleheight_0p008_base_pair_count"] = scores.get(base_id, {}).get(
                "singleheight_0p008_base_pair_count", 0.0
            )
            out["multiheight_plan_stage"] = "20.62_multiheight_profile_perturbation"
            rows.append(out)
    summary = {
        "selected_base_samples": len(selected_bases),
        "profile_rows": len(rows),
        "height_count": len(HEIGHTS),
        "total_height_observations": len(rows) * len(HEIGHTS),
        "reused_singleheight_0p008_observations": len(rows),
        "planned_real_comsol_forward_observations": len(rows) * (len(HEIGHTS) - 1),
        "split_distribution": dict(Counter(row["split"] for row in rows)),
        "source_defect_type_distribution": dict(Counter(row["source_defect_type"] for row in rows)),
        "variant_distribution": dict(Counter(row["variant_type"] for row in rows)),
        "base_split_type_distribution": dict(
            Counter((row["split"], row["source_defect_type"]) for row in rows if row["variant_type"] == "true_reference")
        ),
        "polygon_valid_rows": sum(int(row["polygon_valid"]) for row in rows),
        "mask_non_empty_rows": sum(int(row["mask_non_empty"]) for row in rows),
    }
    return rows, summary


def write_summary(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace) -> None:
    base_ids = sorted({row["base_sample_id"] for row in rows})
    variants_complete = all(sum(1 for row in rows if row["base_sample_id"] == base_id) == len(VARIANT_TYPES) for base_id in base_ids)
    minimum_gate = (
        len(base_ids) >= args.minimum_base_samples
        and len(rows) >= MIN_PROFILE_ROWS
        and len(HEIGHTS) == 3
        and {"train", "val", "test"}.issubset(set(row["split"] for row in rows))
        and {"rectangular_notch", "rotated_rect"}.issubset(set(row["source_defect_type"] for row in rows))
        and variants_complete
    )
    target_gate = len(base_ids) >= args.target_base_samples and len(rows) >= 96 and variants_complete
    lines = [
        "COMSOL rect/rot multi-height profile perturbation plan summary",
        "",
        f"expanded_plan: {args.expanded_plan}",
        f"expanded_pack: {args.expanded_pack}",
        f"sensor_z_heights_m: {list(HEIGHTS)}",
        f"target_base_samples: {args.target_base_samples}",
        f"minimum_base_samples: {args.minimum_base_samples}",
        f"represented_base_samples: {len(base_ids)}",
        f"profile_rows: {summary['profile_rows']}",
        f"height_count: {summary['height_count']}",
        f"total_height_observations: {summary['total_height_observations']}",
        f"reused_singleheight_0p008_observations: {summary['reused_singleheight_0p008_observations']}",
        f"planned_real_comsol_forward_observations: {summary['planned_real_comsol_forward_observations']}",
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
        "- Bases are selected from the 20.61 expanded profile perturbation plan.",
        "- Within each split/source type, selection alternates easy and hard 0.008m single-height oracle-ordering bases when available.",
        "- 0.008m observations are marked reusable from the 20.61 expanded pack; 0.004m and 0.012m require new real COMSOL forward observations.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not minimum_gate:
        raise RuntimeError("multi-height profile plan failed minimum gate")


def main() -> None:
    args = parse_args()
    rows, summary = make_plan(args)
    fields = plan_fields(list(read_csv(args.expanded_plan)[0].keys()))
    write_csv(args.plan_out, rows, fields)
    write_summary(args.summary, rows, summary, args)
    print(f"Wrote {len(rows)} rows to {args.plan_out}")


if __name__ == "__main__":
    main()
