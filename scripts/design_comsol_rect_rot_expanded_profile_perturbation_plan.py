from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import design_comsol_rect_rot_profile_perturbation_plan as base_plan  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_NPZ = base_plan.DEFAULT_SOURCE_NPZ
DEFAULT_PROFILE_CSV = base_plan.DEFAULT_PROFILE_CSV
DEFAULT_2060_PLAN = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_perturbation_plan.csv"
DEFAULT_2060_INVENTORY = (
    Path(r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP")
    / "results/inventory_comsol_rect_rot_profile_perturbation_forward_pack_v1.csv"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_expanded_profile_perturbation_plan_summary.txt"
DEFAULT_PLAN = PROJECT_ROOT / "results/metrics/comsol_rect_rot_expanded_profile_perturbation_plan.csv"

TARGET_COUNTS = {
    ("train", "rectangular_notch"): 12,
    ("train", "rotated_rect"): 12,
    ("val", "rectangular_notch"): 3,
    ("val", "rotated_rect"): 3,
    ("test", "rectangular_notch"): 3,
    ("test", "rotated_rect"): 3,
}
MIN_COUNTS = {
    ("train", "rectangular_notch"): 8,
    ("train", "rotated_rect"): 8,
    ("val", "rectangular_notch"): 2,
    ("val", "rotated_rect"): 2,
    ("test", "rectangular_notch"): 2,
    ("test", "rotated_rect"): 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design expanded rect/rot profile perturbation COMSOL forward plan.")
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_SOURCE_NPZ)
    parser.add_argument("--profile-csv", type=Path, default=DEFAULT_PROFILE_CSV)
    parser.add_argument("--plan-20-60", type=Path, default=DEFAULT_2060_PLAN)
    parser.add_argument("--inventory-20-60", type=Path, default=DEFAULT_2060_INVENTORY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--plan-out", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--target-base-samples", type=int, default=36)
    parser.add_argument("--minimum-base-samples", type=int, default=24)
    parser.add_argument("--allow-reuse-20-60-bases", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def stable_geometry_hash(row: dict[str, Any]) -> str:
    vertices = np.asarray(json.loads(str(row["polygon_vertices_json"])), dtype=np.float64)
    profile = json.loads(str(row["profile_params_json"]))
    payload = {
        "vertices": np.round(vertices, 10).tolist(),
        "depth": round(float(row["depth"]), 10),
        "variant_type": row["variant_type"],
        "base_sample_id": row["base_sample_id"],
        "profile_center": [
            round(float(profile["center_x"]), 10),
            round(float(profile["center_y"]), 10),
            round(float(profile["angle_rad"]), 10),
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def inventory_reuse_keys(plan20_rows: list[dict[str, str]], inventory_rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    inv_real = {
        row["perturb_sample_id"]
        for row in inventory_rows
        if str(row.get("generated_real_forward", "")).lower() == "true"
    }
    inv_reused = {
        row["perturb_sample_id"]
        for row in inventory_rows
        if str(row.get("reused_original", "")).lower() == "true"
    }
    keys: set[tuple[str, str, str]] = set()
    for row in plan20_rows:
        if row["perturb_sample_id"] not in inv_real and row["perturb_sample_id"] not in inv_reused:
            continue
        keys.add((row["base_sample_id"], row["variant_type"], stable_geometry_hash(row)))
    return keys


def choose_diverse_excluding(rows: list[dict[str, Any]], count: int, excluded: set[str]) -> list[dict[str, Any]]:
    fresh = [row for row in rows if row["sample_id"] not in excluded]
    if len(fresh) >= count:
        return base_plan.choose_diverse(fresh, count)
    selected = base_plan.choose_diverse(fresh, len(fresh)) if fresh else []
    remaining = [row for row in rows if row["sample_id"] in excluded]
    selected.extend(base_plan.choose_diverse(remaining, count - len(selected)))
    return selected


def select_bases(profiles: list[dict[str, Any]], target_counts: dict[tuple[str, str], int], excluded: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for key, count in target_counts.items():
        split, defect_type = key
        group = [row for row in profiles if row.get("split") == split and row.get("defect_type") == defect_type]
        selected.extend(choose_diverse_excluding(group, count, excluded))
    return selected


def plan_fields() -> list[str]:
    fields = base_plan.plan_fields()
    insert_after = fields.index("reused_original") + 1
    extras = [
        "reused_from_20_60",
        "reused_base_from_20_60",
        "planned_real_comsol_forward",
        "geometry_hash",
        "plan_generation_stage",
    ]
    for extra in reversed(extras):
        fields.insert(insert_after, extra)
    return fields


def make_plan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    data = np.load(args.source_npz, allow_pickle=True)
    profiles = read_csv(args.profile_csv)
    profiles = [row for row in profiles if row.get("defect_type") in base_plan.MAIN_TYPES]
    plan20 = read_csv(args.plan_20_60)
    inv20 = read_csv(args.inventory_20_60)
    generated20_keys = inventory_reuse_keys(plan20, inv20)
    generated20_base_ids = {row["base_sample_id"] for row in inv20}
    plan20_base_ids = {row["base_sample_id"] for row in plan20}
    exclude = generated20_base_ids if not args.allow_reuse_20_60_bases else set()
    selected = select_bases(profiles, TARGET_COUNTS, exclude)
    if len({row["sample_id"] for row in selected}) < args.minimum_base_samples:
        raise ValueError("Selected base samples below minimum")

    sample_ids = data["sample_ids"].astype(str)
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    mask_x = data["mask_x"].astype(np.float64)
    mask_y = data["mask_y"].astype(np.float64)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    rank = {
        "true_reference": 0,
        "profile_extracted_reference": 1,
        "half_width_shrink_local": 2,
        "half_width_expand_local": 2,
        "smooth_global_width_scale": 2,
        "centerline_offset_small": 3,
        "roughness_noise": 3,
        "mixed_profile_perturbation": 4,
    }
    for base_i, profile_row in enumerate(selected):
        sample_id = profile_row["sample_id"]
        if sample_id not in id_to_idx:
            raise KeyError(f"sample_id missing from source NPZ: {sample_id}")
        idx = int(id_to_idx[sample_id])
        geom = base_plan.parse_json(data["geometry_params"][idx])
        true_mask = data["masks"][idx].astype(np.float32)
        extracted = base_plan.profile_from_row(profile_row)
        true_prof = base_plan.true_profile_from_geometry(geom)
        base_area = base_plan.profile_area(extracted)
        base_rough = base_plan.profile_roughness(extracted)
        for variant_i, variant in enumerate(base_plan.VARIANT_TYPES):
            if variant == "true_reference":
                prof = true_prof
                generated_geometry_type = "original_rect_rot_reference"
                requires_forward = False
                reused = True
                level = "reference"
            else:
                prof = base_plan.perturb_profile(extracted, variant, seed=idx * 31 + variant_i)
                generated_geometry_type = "profile_polygon_notch"
                requires_forward = True
                reused = False
                level = "reference" if variant == "profile_extracted_reference" else "small" if variant != "mixed_profile_perturbation" else "mixed"
            vertices = base_plan.polygon_from_profile(prof)
            mask = base_plan.rasterize_polygon(vertices, mask_x, mask_y)
            v = base_plan.validity(vertices, mask)
            if not v["polygon_valid"]:
                warnings.append(f"invalid polygon {sample_id} {variant}: {v}")
            q = base_plan.metric(mask, true_mask)
            prof_area = base_plan.profile_area(prof)
            prof_rough = base_plan.profile_roughness(prof)
            l2_delta = float(
                np.sqrt(
                    np.mean((np.asarray(prof["half_width"]) - np.asarray(extracted["half_width"])) ** 2)
                    + np.mean((np.asarray(prof["offset"]) - np.asarray(extracted["offset"])) ** 2)
                )
            )
            temp_row = {
                "base_sample_id": sample_id,
                "variant_type": variant,
                "polygon_vertices_json": json.dumps(vertices.tolist(), sort_keys=True),
                "profile_params_json": base_plan.profile_json(prof),
                "depth": float(prof["depth_proxy"]),
            }
            geom_hash = stable_geometry_hash(temp_row)
            reuse20 = int((sample_id, variant, geom_hash) in generated20_keys)
            planned_real = int((not reused) and not reuse20)
            row: dict[str, Any] = {
                "sample_id": f"expanded_profile_perturb_{base_i:03d}_{variant_i:02d}",
                "base_sample_id": sample_id,
                "perturb_sample_id": f"{sample_id}__{variant}",
                "source_index": idx,
                "split": str(data["split"][idx]),
                "source_defect_type": str(data["defect_types"][idx]),
                "source_pack": geom.get("source_pack", profile_row.get("source_pack", "")),
                "generated_geometry_type": generated_geometry_type,
                "variant_type": variant,
                "perturb_level": level,
                "expected_quality_rank": rank[variant],
                "requires_comsol_forward": int(requires_forward),
                "reused_original": int(reused),
                "reused_from_20_60": reuse20,
                "reused_base_from_20_60": int(sample_id in plan20_base_ids),
                "planned_real_comsol_forward": planned_real,
                "geometry_hash": geom_hash,
                "plan_generation_stage": "20.61_expanded_profile_perturbation",
                "center_x": float(prof["center_x"]),
                "center_y": float(prof["center_y"]),
                "angle_rad": float(prof["angle_rad"]),
                "angle_deg": math.degrees(float(prof["angle_rad"])),
                "length": float(prof["length"]),
                "depth": float(prof["depth_proxy"]),
                "profile_params_json": base_plan.profile_json(prof),
                "polygon_vertices_json": json.dumps(vertices.tolist(), sort_keys=True),
                "vertex_count": int(vertices.shape[0]),
                "polygon_area_m2": v["polygon_area_m2"],
                "polygon_valid": int(v["polygon_valid"]),
                "polygon_finite": int(v["polygon_finite"]),
                "polygon_in_bounds": int(v["polygon_in_bounds"]),
                "polygon_self_intersects": int(v["polygon_self_intersects"]),
                "mask_non_empty": int(v["mask_non_empty"]),
                "component_count": int(v["component_count"]),
                "profile_mask_area": float(mask.sum()),
                "quality_iou_vs_true": q["iou"],
                "quality_dice_vs_true": q["dice"],
                "quality_area_error_vs_true": q["area_error"],
                "quality_center_error_px_vs_true": q["center_error_px"],
                "profile_l2_delta": l2_delta,
                "roughness_delta": prof_rough - base_rough,
                "area_delta_ratio": (prof_area - base_area) / max(abs(base_area), 1.0e-9),
                "observed_delta_bz_reference_sample_id": sample_id,
                "notes": (
                    "true_reference reuses original NPZ arrays"
                    if reused
                    else "exact 20.60 generated row can be reused"
                    if reuse20
                    else "new real COMSOL forward required"
                ),
            }
            for i in range(base_plan.K_STATIONS):
                row[f"u_station_{i}"] = float(np.asarray(prof["u"])[i])
                row[f"half_width_{i}"] = float(np.asarray(prof["half_width"])[i])
                row[f"center_offset_{i}"] = float(np.asarray(prof["offset"])[i])
                row[f"occupancy_{i}"] = float(np.asarray(prof["occupancy"])[i])
            rows.append(row)
    return rows, warnings


def write_summary(path: Path, rows: list[dict[str, Any]], warnings: list[str], args: argparse.Namespace) -> None:
    split_counts = Counter(row["split"] for row in rows)
    type_counts = Counter(row["source_defect_type"] for row in rows)
    variant_counts = Counter(row["variant_type"] for row in rows)
    base_ids = sorted({row["base_sample_id"] for row in rows})
    base_split_type = Counter(
        (row["split"], row["source_defect_type"]) for row in rows if row["variant_type"] == "true_reference"
    )
    reused_original = [row for row in rows if int(row["reused_original"]) == 1]
    reused20 = [row for row in rows if int(row["reused_from_20_60"]) == 1]
    planned_real = [row for row in rows if int(row["planned_real_comsol_forward"]) == 1]
    valid_rows = [row for row in rows if int(row["polygon_valid"]) == 1]
    min_split_type_ok = all(base_split_type.get(key, 0) >= value for key, value in MIN_COUNTS.items())
    target_split_type_ok = all(base_split_type.get(key, 0) >= value for key, value in TARGET_COUNTS.items())
    row_accounting_ok = len(rows) == len(reused_original) + len(reused20) + len(planned_real)
    lines = [
        "COMSOL rect/rot expanded profile perturbation plan summary",
        "",
        f"source_npz: {args.source_npz}",
        f"profile_csv: {args.profile_csv}",
        f"plan_20_60: {args.plan_20_60}",
        f"inventory_20_60: {args.inventory_20_60}",
        f"target_base_samples: {args.target_base_samples}",
        f"minimum_base_samples: {args.minimum_base_samples}",
        f"represented_base_samples: {len(base_ids)}",
        f"total_rows: {len(rows)}",
        f"reused_original_rows: {len(reused_original)}",
        f"reused_from_20_60_rows: {len(reused20)}",
        f"planned_real_comsol_forward_rows: {len(planned_real)}",
        f"row_accounting_ok: {row_accounting_ok}",
        f"split_distribution: {dict(split_counts)}",
        f"source_defect_type_distribution: {dict(type_counts)}",
        f"base_split_type_distribution: {dict(base_split_type)}",
        f"variant_distribution: {dict(variant_counts)}",
        f"all_variants_complete_per_base: {all(sum(1 for row in rows if row['base_sample_id'] == base) == len(base_plan.VARIANT_TYPES) for base in base_ids)}",
        f"polygon_valid_rows: {len(valid_rows)}",
        f"invalid_polygon_rows: {len(rows) - len(valid_rows)}",
        f"mask_non_empty_rows: {sum(int(row['mask_non_empty']) for row in rows)}",
        f"target_gate_base_samples_ge_36: {len(base_ids) >= 36}",
        f"target_gate_total_rows_ge_288: {len(rows) >= 288}",
        f"target_gate_real_forward_plus_reuse_ge_252: {len(planned_real) + len(reused20) >= 252}",
        f"target_gate_split_type_ok: {target_split_type_ok}",
        f"minimum_gate_base_samples_ge_24: {len(base_ids) >= 24}",
        f"minimum_gate_total_rows_ge_192: {len(rows) >= 192}",
        f"minimum_gate_real_forward_plus_reuse_ge_168: {len(planned_real) + len(reused20) >= 168}",
        f"minimum_gate_split_type_ok: {min_split_type_ok}",
        f"warnings_count: {len(warnings)}",
        "",
        "Row accounting semantics:",
        "- reused_original rows are true_reference anchors copied from pilot_v9 source arrays.",
        "- reused_from_20_60 rows may reuse exact 20.60 generated arrays if base/variant/geometry_hash match.",
        "- planned_real_comsol_forward rows require new profile polygon COMSOL forward solves.",
        "",
        "Warnings:",
    ]
    lines.extend(warnings[:60] if warnings else ["- none"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, warnings = make_plan(args)
    fields = plan_fields()
    write_csv(args.plan_out, rows, fields)
    write_summary(args.summary, rows, warnings, args)
    invalid = [row for row in rows if int(row["polygon_valid"]) != 1]
    if invalid:
        raise SystemExit(f"expanded profile plan has {len(invalid)} invalid polygon rows; see summary")
    if len({row["base_sample_id"] for row in rows}) < args.minimum_base_samples:
        raise SystemExit("expanded profile plan below minimum base sample count")
    print(f"Wrote {len(rows)} rows to {args.plan_out}")


if __name__ == "__main__":
    main()
