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
DEFAULT_SOURCE_NPZ = (
    PROJECT_ROOT
    / "data/comsol_mfl/prepared/comsol_single_defect_multiline_forward_pack_v1_pilot_v9_balanced_single_defect.npz"
)
DEFAULT_PROFILE_CSV = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_basis_selected_profiles.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/summaries/comsol_rect_rot_profile_perturbation_plan_summary.txt"
DEFAULT_PLAN = PROJECT_ROOT / "results/metrics/comsol_rect_rot_profile_perturbation_plan.csv"

K_STATIONS = 8
MAIN_TYPES = ("rectangular_notch", "rotated_rect")
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
MASK_X_MIN, MASK_X_MAX = -0.04, 0.04
MASK_Y_MIN, MASK_Y_MAX = -0.01, 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design a rect/rot profile perturbation COMSOL forward plan.")
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_SOURCE_NPZ)
    parser.add_argument("--profile-csv", type=Path, default=DEFAULT_PROFILE_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--plan-out", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--target-base-samples", type=int, default=24)
    parser.add_argument("--minimum-base-samples", type=int, default=12)
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


def f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def segment_intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def polygon_self_intersects(vertices: np.ndarray) -> bool:
    n = len(vertices)
    for i in range(n):
        a = vertices[i]
        b = vertices[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if segment_intersects(a, b, vertices[j], vertices[(j + 1) % n]):
                return True
    return False


def points_in_polygon(x: np.ndarray, y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    inside = np.zeros_like(x, dtype=bool)
    xj = vertices[-1, 0]
    yj = vertices[-1, 1]
    for xi, yi in vertices:
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1.0e-30) + xi)
        inside ^= crosses
        xj, yj = xi, yi
    return inside


def rasterize_polygon(vertices: np.ndarray, mask_x: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    yy, xx = np.meshgrid(mask_y, mask_x, indexing="ij")
    return points_in_polygon(xx, yy, vertices).astype(np.float32)


def largest_component_count(mask: np.ndarray) -> int:
    mask = mask.astype(bool)
    if not mask.any():
        return 0
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return count


def profile_from_row(row: dict[str, Any]) -> dict[str, np.ndarray | float]:
    return {
        "center_x": f(row, "center_x"),
        "center_y": f(row, "center_y"),
        "angle_rad": f(row, "angle_rad", 0.0),
        "length": f(row, "length"),
        "depth_proxy": f(row, "depth_proxy", 0.0015),
        "u": np.array([f(row, f"u_station_{i}") for i in range(K_STATIONS)], dtype=np.float64),
        "half_width": np.array([max(f(row, f"half_width_{i}"), 2.5e-4) for i in range(K_STATIONS)], dtype=np.float64),
        "offset": np.array([f(row, f"center_offset_{i}", 0.0) for i in range(K_STATIONS)], dtype=np.float64),
        "occupancy": np.array([f(row, f"occupancy_{i}", 1.0) for i in range(K_STATIONS)], dtype=np.float64),
    }


def true_profile_from_geometry(geom: dict[str, Any]) -> dict[str, np.ndarray | float]:
    center_x = float(geom["center_x"])
    center_y = float(geom["center_y"])
    angle = float(geom.get("angle_rad", geom.get("angle", 0.0)))
    length = float(geom["length"])
    width = float(geom["width"])
    u = np.linspace(-0.5 * length, 0.5 * length, K_STATIONS)
    return {
        "center_x": center_x,
        "center_y": center_y,
        "angle_rad": angle,
        "length": length,
        "depth_proxy": float(geom["depth"]),
        "u": u,
        "half_width": np.full(K_STATIONS, max(0.5 * width, 2.5e-4), dtype=np.float64),
        "offset": np.zeros(K_STATIONS, dtype=np.float64),
        "occupancy": np.ones(K_STATIONS, dtype=np.float64),
    }


def polygon_from_profile(profile: dict[str, Any]) -> np.ndarray:
    u = np.asarray(profile["u"], dtype=np.float64)
    half_width = np.maximum(np.asarray(profile["half_width"], dtype=np.float64), 2.5e-4)
    offset = np.asarray(profile["offset"], dtype=np.float64)
    top = np.stack([u, offset + half_width], axis=1)
    bottom = np.stack([u[::-1], (offset - half_width)[::-1]], axis=1)
    local = np.concatenate([top, bottom], axis=0)
    angle = float(profile["angle_rad"])
    ca = math.cos(angle)
    sa = math.sin(angle)
    rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    shifted = local @ rot.T
    shifted[:, 0] += float(profile["center_x"])
    shifted[:, 1] += float(profile["center_y"])
    return shifted


def profile_area(profile: dict[str, Any]) -> float:
    u = np.asarray(profile["u"], dtype=np.float64)
    hw = np.asarray(profile["half_width"], dtype=np.float64)
    order = np.argsort(u)
    return float(np.trapz(2.0 * hw[order], u[order]))


def profile_roughness(profile: dict[str, Any]) -> float:
    hw = np.asarray(profile["half_width"], dtype=np.float64)
    if hw.size < 3:
        return 0.0
    return float(np.mean(np.diff(hw, n=2) ** 2))


def perturb_profile(base: dict[str, Any], variant: str, seed: int) -> dict[str, Any]:
    out = {
        "center_x": float(base["center_x"]),
        "center_y": float(base["center_y"]),
        "angle_rad": float(base["angle_rad"]),
        "length": float(base["length"]),
        "depth_proxy": float(base["depth_proxy"]),
        "u": np.asarray(base["u"], dtype=np.float64).copy(),
        "half_width": np.asarray(base["half_width"], dtype=np.float64).copy(),
        "offset": np.asarray(base["offset"], dtype=np.float64).copy(),
        "occupancy": np.asarray(base["occupancy"], dtype=np.float64).copy(),
    }
    rng = np.random.default_rng(seed)
    idx = np.array([3, 4]) if seed % 2 == 0 else np.array([2, 5])
    if variant == "half_width_shrink_local":
        out["half_width"][idx] *= 0.86
    elif variant == "half_width_expand_local":
        out["half_width"][idx] *= 1.14
    elif variant == "smooth_global_width_scale":
        out["half_width"] *= 1.08 if seed % 2 == 0 else 0.92
    elif variant == "centerline_offset_small":
        phase = (seed % 5) * 0.35
        out["offset"] += 6.0e-4 * np.sin(np.linspace(0.0, math.pi, K_STATIONS) + phase)
    elif variant == "roughness_noise":
        pattern = np.array([-1.0, 0.6, -0.4, 0.9, -0.8, 0.5, -0.3, 0.7], dtype=np.float64)
        out["half_width"] *= 1.0 + 0.10 * np.roll(pattern, seed % K_STATIONS)
    elif variant == "mixed_profile_perturbation":
        out["center_x"] += float(rng.choice([-1.0, 1.0]) * rng.uniform(4.0e-4, 8.0e-4))
        out["center_y"] += float(rng.choice([-1.0, 1.0]) * rng.uniform(2.5e-4, 6.0e-4))
        out["angle_rad"] += math.radians(float(rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 10.0)))
        out["u"] *= float(rng.uniform(0.92, 1.10))
        out["half_width"] *= float(rng.uniform(0.90, 1.12))
        out["half_width"][idx] *= float(rng.uniform(0.82, 1.20))
    elif variant in {"true_reference", "profile_extracted_reference"}:
        pass
    else:
        raise ValueError(f"unknown variant: {variant}")
    out["half_width"] = np.clip(out["half_width"], 2.5e-4, 0.012)
    out["offset"] = np.clip(out["offset"], -0.0035, 0.0035)
    out["angle_rad"] = float(np.clip(out["angle_rad"], math.radians(-35.0), math.radians(35.0)))
    return out


def metric(mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    return profile_extract.metric(mask.astype(np.float32), true_mask.astype(np.float32), threshold=0.5)


def choose_diverse(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Need {count} rows, got {len(rows)}")
    values = []
    for row in rows:
        values.append(
            (
                f(row, "profile_iou", 0.0),
                f(row, "profile_area_error", 0.0),
                f(row, "roughness_penalty", 0.0),
                abs(f(row, "angle_deg", 0.0)) / 35.0,
            )
        )
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0)
    if len(rows) == count:
        return rows
    score = arr @ np.array([1.0, -0.5, 0.2, 0.3], dtype=np.float64)
    order = np.argsort(score)
    positions = np.linspace(0, len(order) - 1, count)
    selected_idx = []
    used = set()
    for pos in positions:
        idx = int(order[int(round(pos))])
        if idx in used:
            for alt in order:
                if int(alt) not in used:
                    idx = int(alt)
                    break
        used.add(idx)
        selected_idx.append(idx)
    return [rows[idx] for idx in selected_idx]


def select_bases(profiles: list[dict[str, Any]], target: int, minimum: int) -> list[dict[str, Any]]:
    target_counts = {
        ("train", "rectangular_notch"): 8,
        ("train", "rotated_rect"): 8,
        ("val", "rectangular_notch"): 2,
        ("val", "rotated_rect"): 2,
        ("test", "rectangular_notch"): 2,
        ("test", "rotated_rect"): 2,
    }
    if target != 24:
        scale = target / 24.0
        target_counts = {key: max(1, int(round(value * scale))) for key, value in target_counts.items()}
    selected: list[dict[str, Any]] = []
    for key, count in target_counts.items():
        split, defect_type = key
        group = [row for row in profiles if row.get("split") == split and row.get("defect_type") == defect_type]
        selected.extend(choose_diverse(group, count))
    if len(selected) < minimum:
        raise ValueError(f"Selected only {len(selected)} base rows, below minimum {minimum}")
    return selected


def validity(vertices: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    finite = bool(np.isfinite(vertices).all())
    area = polygon_area(vertices) if finite else 0.0
    in_bounds = bool(
        finite
        and np.all(vertices[:, 0] >= MASK_X_MIN)
        and np.all(vertices[:, 0] <= MASK_X_MAX)
        and np.all(vertices[:, 1] >= MASK_Y_MIN)
        and np.all(vertices[:, 1] <= MASK_Y_MAX)
    )
    self_intersects = bool(polygon_self_intersects(vertices)) if finite else True
    component_count = largest_component_count(mask)
    non_empty = bool(mask.sum() > 0)
    return {
        "polygon_finite": finite,
        "polygon_area_m2": area,
        "polygon_in_bounds": in_bounds,
        "polygon_self_intersects": self_intersects,
        "mask_non_empty": non_empty,
        "component_count": component_count,
        "polygon_valid": bool(finite and area > 1.0e-6 and in_bounds and not self_intersects and non_empty),
    }


def profile_json(profile: dict[str, Any]) -> str:
    payload = {
        "center_x": float(profile["center_x"]),
        "center_y": float(profile["center_y"]),
        "angle_rad": float(profile["angle_rad"]),
        "angle_deg": math.degrees(float(profile["angle_rad"])),
        "length": float(profile["length"]),
        "depth_proxy": float(profile["depth_proxy"]),
        "u_stations": np.asarray(profile["u"], dtype=float).tolist(),
        "half_widths": np.asarray(profile["half_width"], dtype=float).tolist(),
        "center_offsets": np.asarray(profile["offset"], dtype=float).tolist(),
        "occupancy": np.asarray(profile["occupancy"], dtype=float).tolist(),
        "area_proxy": profile_area(profile),
        "roughness": profile_roughness(profile),
    }
    return json.dumps(payload, sort_keys=True)


def plan_fields() -> list[str]:
    fields = [
        "sample_id",
        "base_sample_id",
        "perturb_sample_id",
        "source_index",
        "split",
        "source_defect_type",
        "source_pack",
        "generated_geometry_type",
        "variant_type",
        "perturb_level",
        "expected_quality_rank",
        "requires_comsol_forward",
        "reused_original",
        "center_x",
        "center_y",
        "angle_rad",
        "angle_deg",
        "length",
        "depth",
        "profile_params_json",
        "polygon_vertices_json",
        "vertex_count",
        "polygon_area_m2",
        "polygon_valid",
        "polygon_finite",
        "polygon_in_bounds",
        "polygon_self_intersects",
        "mask_non_empty",
        "component_count",
        "profile_mask_area",
        "quality_iou_vs_true",
        "quality_dice_vs_true",
        "quality_area_error_vs_true",
        "quality_center_error_px_vs_true",
        "profile_l2_delta",
        "roughness_delta",
        "area_delta_ratio",
        "observed_delta_bz_reference_sample_id",
        "notes",
    ]
    for prefix in ("u_station", "half_width", "center_offset", "occupancy"):
        fields += [f"{prefix}_{i}" for i in range(K_STATIONS)]
    return fields


def make_plan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    data = np.load(args.source_npz, allow_pickle=True)
    profiles = read_csv(args.profile_csv)
    profiles = [row for row in profiles if row.get("defect_type") in MAIN_TYPES]
    selected = select_bases(profiles, args.target_base_samples, args.minimum_base_samples)
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
        geom = parse_json(data["geometry_params"][idx])
        true_mask = data["masks"][idx].astype(np.float32)
        extracted = profile_from_row(profile_row)
        true_prof = true_profile_from_geometry(geom)
        base_area = profile_area(extracted)
        base_rough = profile_roughness(extracted)
        for variant_i, variant in enumerate(VARIANT_TYPES):
            if variant == "true_reference":
                prof = true_prof
                generated_geometry_type = "original_rect_rot_reference"
                requires_forward = False
                reused = True
                level = "reference"
            else:
                prof = perturb_profile(extracted, variant, seed=idx * 31 + variant_i)
                generated_geometry_type = "profile_polygon_notch"
                requires_forward = True
                reused = False
                level = "reference" if variant == "profile_extracted_reference" else "small" if variant != "mixed_profile_perturbation" else "mixed"
            vertices = polygon_from_profile(prof)
            mask = rasterize_polygon(vertices, mask_x, mask_y)
            v = validity(vertices, mask)
            if not v["polygon_valid"]:
                warnings.append(f"invalid polygon {sample_id} {variant}: {v}")
            q = metric(mask, true_mask)
            prof_area = profile_area(prof)
            prof_rough = profile_roughness(prof)
            l2_delta = float(
                np.sqrt(
                    np.mean((np.asarray(prof["half_width"]) - np.asarray(extracted["half_width"])) ** 2)
                    + np.mean((np.asarray(prof["offset"]) - np.asarray(extracted["offset"])) ** 2)
                )
            )
            row: dict[str, Any] = {
                "sample_id": f"profile_perturb_{base_i:03d}_{variant_i:02d}",
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
                "center_x": float(prof["center_x"]),
                "center_y": float(prof["center_y"]),
                "angle_rad": float(prof["angle_rad"]),
                "angle_deg": math.degrees(float(prof["angle_rad"])),
                "length": float(prof["length"]),
                "depth": float(prof["depth_proxy"]),
                "profile_params_json": profile_json(prof),
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
                "notes": "true_reference reuses original NPZ arrays" if reused else "real COMSOL forward required",
            }
            for i in range(K_STATIONS):
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
    real_rows = [row for row in rows if int(row["requires_comsol_forward"]) == 1]
    reused_rows = [row for row in rows if int(row["reused_original"]) == 1]
    valid_rows = [row for row in rows if int(row["polygon_valid"]) == 1]
    lines = [
        "COMSOL rect/rot profile perturbation plan summary",
        "",
        f"source_npz: {args.source_npz}",
        f"profile_csv: {args.profile_csv}",
        f"target_base_samples: {args.target_base_samples}",
        f"minimum_base_samples: {args.minimum_base_samples}",
        f"represented_base_samples: {len(base_ids)}",
        f"total_rows: {len(rows)}",
        f"reused_original_rows: {len(reused_rows)}",
        f"real_comsol_forward_rows_planned: {len(real_rows)}",
        f"split_distribution: {dict(split_counts)}",
        f"source_defect_type_distribution: {dict(type_counts)}",
        f"variant_distribution: {dict(variant_counts)}",
        f"all_variants_complete_per_base: {all(sum(1 for row in rows if row['base_sample_id'] == base) == len(VARIANT_TYPES) for base in base_ids)}",
        f"polygon_valid_rows: {len(valid_rows)}",
        f"invalid_polygon_rows: {len(rows) - len(valid_rows)}",
        f"mask_non_empty_rows: {sum(int(row['mask_non_empty']) for row in rows)}",
        f"minimum_partial_gate_total_rows_ge_96: {len(rows) >= 96}",
        f"minimum_partial_gate_real_forward_rows_ge_84: {len(real_rows) >= 84}",
        f"minimum_partial_gate_base_samples_ge_12: {len(base_ids) >= 12}",
        f"warnings_count: {len(warnings)}",
        "",
        "Variant semantics:",
        "- true_reference rows reuse original pilot_v9 arrays and are not counted as real COMSOL forward rows.",
        "- all other variants require profile polygon COMSOL forward solve.",
        "- generated_geometry_type=profile_polygon_notch indicates top-view profile footprint with constant depth extrusion.",
        "",
        "Warnings:",
    ]
    lines.extend(warnings[:40] if warnings else ["- none"])
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
        raise SystemExit(f"profile perturbation plan has {len(invalid)} invalid polygon rows; see summary")
    print(f"Wrote {len(rows)} rows to {args.plan_out}")


if __name__ == "__main__":
    main()
