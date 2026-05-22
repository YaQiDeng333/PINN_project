from __future__ import annotations

import argparse
import csv
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

import extract_comsol_rect_rot_profile_basis_from_dense as profile_extract  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
DEFAULT_PERTURB_NPZ = PROJECT_ROOT / "data/comsol_mfl/prepared/comsol_rect_rot_local_perturbation_forward_pack_v1.npz"
DEFAULT_PROFILES = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_selected_profiles.csv"
DEFAULT_PLAN = PROJECT_ROOT / "results/metrics/comsol_rect_rot_local_perturbation_plan.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_profile_forward_dataset_summary.txt"
DEFAULT_ORIGINAL_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_forward_original_dataset.csv"
DEFAULT_PERTURB_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_forward_perturb_dataset.csv"
DEFAULT_SCHEMA_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_forward_dataset_schema.csv"

K_STATIONS = 8
MAIN_TYPES = {"rectangular_notch", "rotated_rect"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build profile-forward CSV datasets from existing COMSOL artifacts.")
    parser.add_argument("--original-npz", type=Path, default=DEFAULT_ORIGINAL_NPZ)
    parser.add_argument("--perturb-npz", type=Path, default=DEFAULT_PERTURB_NPZ)
    parser.add_argument("--profile-csv", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--perturb-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--original-out", type=Path, default=DEFAULT_ORIGINAL_CSV)
    parser.add_argument("--perturb-out", type=Path, default=DEFAULT_PERTURB_CSV)
    parser.add_argument("--schema-out", type=Path, default=DEFAULT_SCHEMA_CSV)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except Exception:
        return default


def profile_station_value(profile: dict[str, Any], prefix: str, index: int, default: float = math.nan) -> float:
    explicit = f"{prefix}_{index}"
    if explicit in profile:
        return f(profile, explicit, default)
    array_key = {
        "u_station": "u_stations",
        "half_width": "half_widths",
        "center_offset": "center_offsets",
        "occupancy": "occupancy",
    }[prefix]
    values = profile.get(array_key)
    if values is None:
        return default
    try:
        return float(np.asarray(values, dtype=np.float64)[index])
    except Exception:
        return default


def profile_feature_fields() -> list[str]:
    fields = [
        "center_x",
        "center_y",
        "angle_rad",
        "angle_sin",
        "angle_cos",
        "length",
        "depth_proxy",
        "area_from_profile_params",
        "roughness_penalty",
    ]
    fields += [f"u_station_{i}" for i in range(K_STATIONS)]
    fields += [f"half_width_{i}" for i in range(K_STATIONS)]
    fields += [f"center_offset_{i}" for i in range(K_STATIONS)]
    fields += [f"occupancy_{i}" for i in range(K_STATIONS)]
    fields += [
        "mean_half_width",
        "max_half_width",
        "min_half_width",
        "std_half_width",
        "mean_abs_offset",
        "max_abs_offset",
        "profile_area_proxy",
    ]
    return fields


def add_profile_features(row: dict[str, Any], profile: dict[str, Any]) -> None:
    angle = f(profile, "angle_rad", 0.0)
    row["center_x"] = f(profile, "center_x")
    row["center_y"] = f(profile, "center_y")
    row["angle_rad"] = angle
    row["angle_sin"] = math.sin(angle)
    row["angle_cos"] = math.cos(angle)
    row["length"] = f(profile, "length")
    row["depth_proxy"] = f(profile, "depth_proxy")
    row["area_from_profile_params"] = f(profile, "area_from_profile_params")
    row["roughness_penalty"] = f(profile, "roughness_penalty", 0.0)
    half_widths: list[float] = []
    offsets: list[float] = []
    for i in range(K_STATIONS):
        u = profile_station_value(profile, "u_station", i)
        hw = max(profile_station_value(profile, "half_width", i), 1.0e-6)
        off = profile_station_value(profile, "center_offset", i, 0.0)
        occ = profile_station_value(profile, "occupancy", i, 1.0)
        row[f"u_station_{i}"] = u
        row[f"half_width_{i}"] = hw
        row[f"center_offset_{i}"] = off
        row[f"occupancy_{i}"] = occ
        half_widths.append(hw)
        offsets.append(off)
    hw_arr = np.asarray(half_widths, dtype=np.float64)
    off_arr = np.asarray(offsets, dtype=np.float64)
    row["mean_half_width"] = float(hw_arr.mean())
    row["max_half_width"] = float(hw_arr.max())
    row["min_half_width"] = float(hw_arr.min())
    row["std_half_width"] = float(hw_arr.std())
    row["mean_abs_offset"] = float(np.abs(off_arr).mean())
    row["max_abs_offset"] = float(np.abs(off_arr).max())
    row["profile_area_proxy"] = float(np.trapz(2.0 * hw_arr, np.asarray([row[f"u_station_{i}"] for i in range(K_STATIONS)])))


def mask_quality(mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    return profile_extract.metric(mask.astype(np.float32), true_mask.astype(np.float32), threshold=0.5)


def geometry_profile_from_mask(
    mask: np.ndarray,
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    depth_proxy: float,
) -> dict[str, Any]:
    profile = profile_extract.extract_one(
        mask.astype(np.float32),
        method="P1_hardmask_profile",
        threshold=0.5,
        mask_x=mask_x,
        mask_y=mask_y,
        depth_proxy=depth_proxy,
    )
    prob = profile_extract.rasterize_profile_np(
        mask_x,
        mask_y,
        profile["center_x"],
        profile["center_y"],
        profile["angle_rad"],
        profile["u_stations"],
        profile["half_widths"],
        profile["center_offsets"],
    )
    profile["profile_prob"] = prob
    return profile


def original_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = np.load(args.original_npz, allow_pickle=True)
    profiles = read_csv(args.profile_csv)
    selected = {row["sample_id"]: row for row in profiles if row.get("selected_method", "") in {"", row.get("method", "")}}
    sample_ids = data["sample_ids"].astype(str)
    sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    rows: list[dict[str, Any]] = []
    for sample_id, profile in selected.items():
        if sample_id not in sample_to_idx:
            raise KeyError(f"Profile sample_id not found in original NPZ: {sample_id}")
        idx = int(sample_to_idx[sample_id])
        defect_type = str(data["defect_types"][idx])
        if defect_type not in MAIN_TYPES:
            continue
        geom = parse_json(data["geometry_params"][idx])
        row: dict[str, Any] = {
            "dataset": "original",
            "sample_id": sample_id,
            "base_sample_id": sample_id,
            "source_index": idx,
            "split": str(data["split"][idx]),
            "defect_type": defect_type,
            "variant_type": "original_profile",
            "expected_quality_rank": 0,
            "generated_real_forward": True,
            "target_delta_bz_source": "pilot_v9_original_delta_bz",
            "reference_observed_available": False,
            "source_pack": geom.get("source_pack", ""),
            "true_mask_iou": f(profile, "profile_iou"),
            "true_mask_dice": f(profile, "profile_dice"),
            "true_area_error": f(profile, "profile_area_error"),
            "dense_iou": f(profile, "dense_iou"),
            "dense_dice": f(profile, "dense_dice"),
            "dense_area_error": f(profile, "dense_area_error"),
            "target_index": idx,
        }
        add_profile_features(row, profile)
        if not all(math.isfinite(float(row[field])) for field in profile_feature_fields()):
            raise ValueError(f"Non-finite original profile feature for {sample_id}")
        rows.append(row)
    diagnostics = {
        "n": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "type_counts": dict(Counter(row["defect_type"] for row in rows)),
    }
    return rows, diagnostics


def perturb_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.perturb_npz.exists():
        return [], {"available": False, "reason": f"missing {args.perturb_npz}"}
    data = np.load(args.perturb_npz, allow_pickle=True)
    mask_x = data["mask_x"].astype(np.float64)
    mask_y = data["mask_y"].astype(np.float64)
    rows: list[dict[str, Any]] = []
    for idx in range(int(data["delta_bz"].shape[0])):
        defect_type = str(data["defect_types"][idx])
        if defect_type not in MAIN_TYPES:
            continue
        geom = parse_json(data["geometry_params"][idx])
        quality = parse_json(data["geometry_quality_to_true"][idx])
        depth = float(geom.get("depth_m", geom.get("depth", 0.0015)))
        profile = geometry_profile_from_mask(data["masks"][idx].astype(np.float32), mask_x, mask_y, depth)
        metric = mask_quality(profile["profile_prob"], data["masks"][idx])
        row = {
            "dataset": "perturb",
            "sample_id": str(data["sample_ids"][idx]),
            "base_sample_id": str(data["base_sample_ids"][idx]),
            "source_index": idx,
            "split": str(data["split"][idx]),
            "defect_type": defect_type,
            "variant_type": str(data["variant_types"][idx]),
            "expected_quality_rank": int(data["expected_quality_rank"][idx]),
            "generated_real_forward": bool(data["generated_real_forward"][idx]),
            "target_delta_bz_source": "comsol_perturb_delta_bz",
            "reference_observed_available": True,
            "source_pack": "",
            "true_mask_iou": float(quality.get("mask_iou", quality.get("geometry_mask_iou_vs_true", math.nan))),
            "true_mask_dice": float(quality.get("mask_dice", quality.get("geometry_mask_dice_vs_true", math.nan))),
            "true_area_error": float(quality.get("area_error", math.nan)),
            "dense_iou": math.nan,
            "dense_dice": math.nan,
            "dense_area_error": math.nan,
            "target_index": idx,
            "profile_reconstruct_iou_vs_perturb_mask": metric["iou"],
            "profile_reconstruct_dice_vs_perturb_mask": metric["dice"],
            "profile_reconstruct_area_error_vs_perturb_mask": metric["area_error"],
        }
        add_profile_features(row, profile)
        if not all(math.isfinite(float(row[field])) for field in profile_feature_fields()):
            raise ValueError(f"Non-finite perturb profile feature for {row['sample_id']}")
        rows.append(row)
    diagnostics = {
        "available": True,
        "n": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "type_counts": dict(Counter(row["defect_type"] for row in rows)),
        "variant_counts": dict(Counter(row["variant_type"] for row in rows)),
    }
    return rows, diagnostics


def schema_rows() -> list[dict[str, str]]:
    rows = [
        {"field": "dataset", "description": "original or perturb", "used_as_model_input": "no"},
        {"field": "sample_id", "description": "stable sample id", "used_as_model_input": "no"},
        {"field": "split", "description": "preserved train/val/test split", "used_as_model_input": "no"},
        {"field": "defect_type", "description": "metrics grouping only; not a model input", "used_as_model_input": "no"},
        {"field": "target_index", "description": "row index into source NPZ target arrays", "used_as_model_input": "no"},
    ]
    rows += [
        {"field": field, "description": "profile-derived scalar/station feature", "used_as_model_input": "yes"}
        for field in profile_feature_fields()
    ]
    rows += [
        {"field": "true_mask_iou", "description": "metrics / ordering audit only", "used_as_model_input": "no"},
        {"field": "true_mask_dice", "description": "metrics / ordering audit only", "used_as_model_input": "no"},
        {"field": "true_area_error", "description": "metrics / ordering audit only", "used_as_model_input": "no"},
    ]
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    original, original_diag = original_rows(args)
    perturb, perturb_diag = perturb_rows(args)
    if original_diag["n"] != 400:
        raise RuntimeError(f"Expected 400 original rect/rot rows, got {original_diag['n']}")
    if original_diag["split_counts"] != {"train": 268, "val": 66, "test": 66}:
        raise RuntimeError(f"Unexpected original split counts: {original_diag['split_counts']}")
    write_csv(args.original_out, original)
    if perturb:
        write_csv(args.perturb_out, perturb)
    else:
        args.perturb_out.parent.mkdir(parents=True, exist_ok=True)
        args.perturb_out.write_text("dataset_status,reason\nmissing," + perturb_diag["reason"] + "\n", encoding="utf-8")
    write_csv(args.schema_out, schema_rows())
    lines = [
        "COMSOL rect/rot profile forward dataset summary",
        "",
        "No COMSOL run and no new data generation. This stage converts existing NPZ/CSV artifacts into tracked CSV descriptors only.",
        "Original dataset uses 20.58 selected predicted profile representation with pilot_v9 observed delta_bz as target.",
        "Perturbation dataset converts existing 20.56 perturbation masks/geometries into the same K=8 profile representation.",
        "",
        f"Original rows / split / type: {original_diag['n']} / {original_diag['split_counts']} / {original_diag['type_counts']}",
        f"Perturbation available: {perturb_diag.get('available', False)}",
        f"Perturbation rows / split / type: {perturb_diag.get('n', 0)} / {perturb_diag.get('split_counts', {})} / {perturb_diag.get('type_counts', {})}",
        f"Perturbation variant distribution: {perturb_diag.get('variant_counts', {})}",
        "",
        "Leakage controls:",
        "- split is copied from source artifacts and not recomputed.",
        "- defect_type and variant_type are metrics-only fields, not model inputs.",
        "- observed delta_bz is a target/residual reference only, not a model input.",
        "- polygon samples are excluded from both datasets.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"original": original_diag, "perturb": perturb_diag}


def main() -> None:
    result = run(parse_args())
    print(f"Original rows: {result['original']['n']}; perturb rows: {result['perturb'].get('n', 0)}")


if __name__ == "__main__":
    main()
