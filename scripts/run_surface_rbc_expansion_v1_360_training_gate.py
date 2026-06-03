#!/usr/bin/env python
"""Surface RBC +120 assembly/training gate.

This gate explicitly assembles the current v3_240 nominal surface RBC pack and
the validated +120 targeted top-up pack, trains the same 20.85-style Conv1D
six-parameter model on the assembled train split, and compares it with the
frozen 20.85 checkpoint. It does not update CURRENT_BASELINE and does not
commit generated NPZ/checkpoint artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import train_true_3d_rbc_neural_parameter_gate as gate
from load_true_3d_rbc_pilot_dataset import (
    PARAM_NAMES,
    ROOT,
    depth_grid_from_params,
    depth_map_from_params,
    denormalize_y,
    mask_metrics,
    normalize_x,
    normalize_y,
    projected_mask_from_params,
    sha256_file,
    split_indices,
    train_normalization,
)


SOURCE_DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"
TOPUP_DATASET_ID = "comsol_true_3d_rbc_surface_targeted_topup_v1_120"
ASSEMBLED_DATASET_ID = "comsol_true_3d_rbc_surface_expansion_v1_360"
SCHEMA_VERSION = "true3d_profile_v1_piao_rbc"
NPZ_ROUTE = "true_3d_piao_style"
MANIFEST_ROUTE = "true_3d_rbc_surface_expansion"
NOMINAL_SENSOR_Z_M = 0.008
CURVATURE_WEIGHT_MIN = 0.55
CURVATURE_WEIGHT_MAX = 1.200001

SOURCE_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json"
TOPUP_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_surface_targeted_topup_v1_120.manifest.json"
BASELINE_ARTIFACT_MANIFEST = ROOT / "results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json"
ASSEMBLED_NPZ = (
    ROOT
    / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/"
    / f"{ASSEMBLED_DATASET_ID}.npz"
)
METRICS_JSON = ROOT / "results/metrics/surface_rbc_expansion_v1_360_training_gate_metrics.json"
SUMMARY_MD = ROOT / "results/summaries/surface_rbc_expansion_v1_360_training_gate_summary.md"
GATE_MANIFEST = ROOT / "results/manifests/surface_rbc_expansion_v1_360_training_gate_manifest.json"

OLD_TEST_THRESHOLDS = {
    "profile_depth_rmse_m": {"op": "<=", "value": 0.00039936911},
    "er_like_profile_error": {"op": "<=", "value": 0.3575712},
    "L_mae_mm": {"op": "<=", "value": 1.9866},
    "W_mae_mm": {"op": "<=", "value": 2.2953},
    "D_mae_mm": {"op": "<=", "value": 0.8400},
    "projected_mask_dice": {"op": ">=", "value": 0.837727},
}
PRIMARY_IMPROVEMENT_METRICS = [
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "D_mae_mm",
    "projected_mask_dice",
]
LOWER_IS_BETTER = {
    "profile_depth_rmse_m": True,
    "er_like_profile_error": True,
    "L_mae_mm": True,
    "W_mae_mm": True,
    "D_mae_mm": True,
    "projected_mask_iou": False,
    "projected_mask_dice": False,
    "normalized_param_mae": True,
    "dimension_mae_norm": True,
    "curvature_mae_norm": True,
    "wLD_abs_error": True,
    "wWD_abs_error": True,
    "wLW_abs_error": True,
    "wMAE_auxiliary": True,
    "max_depth_error_m": True,
    "volume_proxy_rel_error": True,
}

FORBIDDEN_STAGE_PREFIXES = (
    "data/",
    "checkpoints/",
    "notes/",
    "results/previews/",
)
FORBIDDEN_STAGE_SUFFIXES = (
    ".npz",
    ".mph",
    ".pt",
    ".pth",
    ".ckpt",
    ".png",
    ".jpg",
    ".jpeg",
    ".stl",
)
FORBIDDEN_STAGE_EXACT = {
    "CURRENT_BASELINE.md",
    "scripts/visualize_current_baseline.py",
}


@dataclass
class Pack:
    dataset_id: str
    manifest: dict[str, Any]
    npz_path: Path
    arrays: dict[str, np.ndarray]


@dataclass
class ExpansionDataset:
    dataset_id: str
    manifest: dict[str, Any]
    npz_path: Path
    delta_b: np.ndarray
    b_defect: np.ndarray
    b_no_defect: np.ndarray
    x_channels: np.ndarray
    rbc_params: np.ndarray
    profile_pose: np.ndarray
    projected_mask_2d: np.ndarray
    profile_depth_grid_m: np.ndarray
    profile_depth_map_xy_m: np.ndarray
    sample_ids: np.ndarray
    split: np.ndarray
    axis_names: list[str]
    sensor_x: np.ndarray
    scan_line_y: np.ndarray
    sensor_z_m: float
    curvature_template: np.ndarray
    depth_bin: np.ndarray
    aspect_bin: np.ndarray
    size_bin: np.ndarray
    source_dataset: np.ndarray
    targeted_role: np.ndarray
    edge_position_bin: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--topup-manifest", type=Path, default=TOPUP_MANIFEST)
    parser.add_argument("--baseline-artifact-manifest", type=Path, default=BASELINE_ARTIFACT_MANIFEST)
    parser.add_argument("--assembled-npz", type=Path, default=ASSEMBLED_NPZ)
    parser.add_argument("--metrics-json", type=Path, default=METRICS_JSON)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    parser.add_argument("--gate-manifest", type=Path, default=GATE_MANIFEST)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError("non-finite JSON float")
        return out
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON float")
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def staged_files() -> list[str]:
    return [line.replace("\\", "/") for line in git_value(["diff", "--cached", "--name-only"]).splitlines() if line.strip()]


def forbidden_staged(paths: list[str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        if path in FORBIDDEN_STAGE_EXACT:
            out.append(path)
            continue
        if any(path.startswith(prefix) for prefix in FORBIDDEN_STAGE_PREFIXES):
            out.append(path)
            continue
        if any(path.lower().endswith(suffix) for suffix in FORBIDDEN_STAGE_SUFFIXES):
            out.append(path)
    return out


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, observed: Any, notes: str = "") -> None:
    checks.append({"name": name, "pass": bool(passed), "observed": observed, "notes": notes})


def scalar_str(arrays: dict[str, np.ndarray], key: str) -> str:
    return str(np.asarray(arrays[key]).reshape(-1)[0])


def scalar_float(arrays: dict[str, np.ndarray], key: str) -> float:
    return float(np.asarray(arrays[key]).reshape(-1)[0])


def string_array(arrays: dict[str, np.ndarray], key: str, n: int | None = None, fill: str = "") -> np.ndarray:
    if key in arrays:
        return np.asarray(arrays[key]).astype(str).reshape(-1)
    if n is None:
        raise KeyError(key)
    return np.asarray([fill] * n, dtype=f"<U{max(1, len(fill))}")


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pack(manifest_path: Path, expected_id: str) -> Pack:
    manifest = read_manifest(manifest_path)
    if manifest.get("dataset_id") != expected_id:
        raise RuntimeError(f"manifest dataset_id mismatch: {manifest.get('dataset_id')} != {expected_id}")
    npz_path = Path(str(manifest["npz_path"]))
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if sha256_file(npz_path) != manifest.get("npz_sha256"):
        raise RuntimeError(f"npz sha256 mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as npz:
        arrays = {name: np.array(npz[name]) for name in npz.files}
    return Pack(expected_id, manifest, npz_path, arrays)


def split_counts(split: np.ndarray) -> dict[str, int]:
    counts = Counter(np.asarray(split).astype(str).tolist())
    return {name: int(counts.get(name, 0)) for name in ("train", "val", "test")}


def validate_pack_preflight(source: Pack, topup: Pack) -> tuple[list[dict[str, Any]], bool]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, notes: str = "") -> None:
        add_check(checks, name, passed, observed, notes)

    staged = staged_files()
    add("forbidden_staged_files_before_gate", not forbidden_staged(staged), forbidden_staged(staged) or "none")
    add("source_manifest_exists", True, str(source.npz_path))
    add("topup_manifest_exists", True, str(topup.npz_path))
    add("baseline_artifact_manifest_exists", BASELINE_ARTIFACT_MANIFEST.exists(), str(BASELINE_ARTIFACT_MANIFEST))

    source_delta = np.asarray(source.arrays.get("delta_b"), dtype=np.float64)
    topup_delta = np.asarray(topup.arrays.get("delta_b"), dtype=np.float64)
    add("source_shape", source_delta.shape == (240, 3, 3, 201), list(source_delta.shape))
    add("topup_shape", topup_delta.shape == (120, 3, 3, 201), list(topup_delta.shape))
    add("source_model_view_shape", source_delta.reshape(240, 9, 201).shape == (240, 9, 201), [240, 9, 201])
    add("topup_model_view_shape", topup_delta.reshape(120, 9, 201).shape == (120, 9, 201), [120, 9, 201])
    add("source_split_counts", split_counts(source.arrays["split"]) == {"train": 162, "val": 39, "test": 39}, split_counts(source.arrays["split"]))
    add("topup_split_counts_materialized", split_counts(topup.arrays["split"]) == {"train": 80, "val": 20, "test": 20}, split_counts(topup.arrays["split"]))

    source_axis = [str(x) for x in np.asarray(source.arrays["axis_names"]).reshape(-1).tolist()]
    topup_axis = [str(x) for x in np.asarray(topup.arrays["axis_names"]).reshape(-1).tolist()]
    add("source_axis_names", source_axis == ["Bx", "By", "Bz"], source_axis)
    add("topup_axis_names", topup_axis == ["Bx", "By", "Bz"], topup_axis)
    add("source_sensor_z_nominal", abs(scalar_float(source.arrays, "sensor_z_m") - NOMINAL_SENSOR_Z_M) <= 1.0e-12, scalar_float(source.arrays, "sensor_z_m"))
    add("topup_sensor_z_nominal", abs(scalar_float(topup.arrays, "sensor_z_m") - NOMINAL_SENSOR_Z_M) <= 1.0e-12, scalar_float(topup.arrays, "sensor_z_m"))
    add("source_no_non_nominal_liftoff", True, "sensor_z_m=0.008 scalar")
    add("topup_no_non_nominal_liftoff", True, "sensor_z_m=0.008 scalar")

    for pack_name, pack, expected_n in (("source", source, 240), ("topup", topup, 120)):
        arrays = pack.arrays
        params = np.asarray(arrays.get("rbc_params"), dtype=np.float64).reshape(expected_n, 6)
        defect = np.asarray(arrays.get("b_defect"), dtype=np.float64)
        no_defect = np.asarray(arrays.get("b_no_defect"), dtype=np.float64)
        delta = np.asarray(arrays.get("delta_b"), dtype=np.float64)
        label_source = "explicit" if "rbc_param_names" in arrays else f"schema_implied:{SCHEMA_VERSION}"
        add(f"{pack_name}_six_labels", params.shape == (expected_n, 6), {"shape": list(params.shape), "label_order": PARAM_NAMES, "source": label_source})
        add(f"{pack_name}_delta_finite", bool(np.isfinite(delta).all()), "finite")
        add(f"{pack_name}_labels_finite", bool(np.isfinite(params).all()), "finite")
        add(f"{pack_name}_delta_equals_defect_minus_no_defect", float(np.max(np.abs(delta - (defect - no_defect)))) <= 1.0e-12, float(np.max(np.abs(delta - (defect - no_defect)))))
        add(f"{pack_name}_mask_nonempty", bool((np.asarray(arrays["projected_mask_2d"]).sum(axis=(1, 2)) > 0).all()), int(np.asarray(arrays["projected_mask_2d"]).sum()))

    topup_manifest = topup.manifest
    add("topup_status", topup_manifest.get("status") == "topup_generated", topup_manifest.get("status"))
    add("topup_dataset_role", topup_manifest.get("dataset_role") == "topup_source", topup_manifest.get("dataset_role"))
    add("topup_train_ready_candidate", bool(topup_manifest.get("train_ready_candidate")), topup_manifest.get("train_ready_candidate"))
    add("topup_baseline_ready_false", not bool(topup_manifest.get("baseline_ready")), topup_manifest.get("baseline_ready"))
    add("topup_creates_assembled_dataset_false", not bool(topup_manifest.get("creates_assembled_dataset")), topup_manifest.get("creates_assembled_dataset"))
    add("topup_allowed_training_gate", "explicit_surface_rbc_expansion_training_gate" in set(topup_manifest.get("allowed_use", [])), topup_manifest.get("allowed_use", []))

    source_ids = set(np.asarray(source.arrays["sample_ids"]).astype(str).tolist())
    topup_ids = set(np.asarray(topup.arrays["sample_ids"]).astype(str).tolist())
    add("source_sample_ids_unique", len(source_ids) == 240, len(source_ids))
    add("topup_sample_ids_unique", len(topup_ids) == 120, len(topup_ids))
    add("no_source_topup_sample_id_collision", not (source_ids & topup_ids), sorted(source_ids & topup_ids)[:5])

    failed = [row for row in checks if not row["pass"]]
    return checks, not failed


def concatenate_arrays(source: Pack, topup: Pack) -> dict[str, np.ndarray]:
    s = source.arrays
    t = topup.arrays
    n_source = int(np.asarray(s["delta_b"]).shape[0])
    n_topup = int(np.asarray(t["delta_b"]).shape[0])
    common_fields = [
        "delta_b",
        "b_defect",
        "b_no_defect",
        "delta_b_multi_axis",
        "b_defect_multi_axis",
        "b_no_defect_multi_axis",
        "sample_ids",
        "split",
        "defect_types",
        "rbc_params",
        "profile_pose",
        "profile_depth_grid_m",
        "profile_depth_map_xy_m",
        "projected_mask_2d",
        "profile_footprint_mask",
        "footprint_mask",
        "masks",
        "geometry_method_used",
        "mesh_source",
        "mesh_units",
        "top_cap_plane",
        "depth_sign_convention",
        "selected_solver_protocol",
        "mesh_auto_size",
        "material_fix_applied",
        "domain_material_audit_pass",
        "solver_probe_pass",
        "full_source_jscale",
        "no_defect_reused",
        "exact_piao_rbc",
        "rbc_style_approximation",
        "depth_bin",
        "size_bin",
        "aspect_bin",
        "curvature_template",
        "geometry_params_json",
        "mesh_metrics_json",
        "source_metadata_json",
    ]
    out: dict[str, np.ndarray] = {
        "dataset_id": np.asarray([ASSEMBLED_DATASET_ID]),
        "schema_version": np.asarray([SCHEMA_VERSION]),
        "route": np.asarray([NPZ_ROUTE]),
        "status": np.asarray(["assembled_training_gate_candidate"]),
        "axis_names": np.asarray(["Bx", "By", "Bz"]),
        "axis_expressions": np.asarray(["mf.Bx", "mf.By", "mf.Bz"]),
        "sensor_x": np.asarray(s["sensor_x"]),
        "scan_line_y": np.asarray(s["scan_line_y"]),
        "sensor_z_m": np.asarray([NOMINAL_SENSOR_Z_M], dtype=np.float64),
        "rbc_param_names": np.asarray(PARAM_NAMES),
        "source_dataset_ids": np.asarray([SOURCE_DATASET_ID, TOPUP_DATASET_ID]),
        "source_manifest_paths": np.asarray([str(SOURCE_MANIFEST), str(TOPUP_MANIFEST)]),
        "source_dataset": np.concatenate(
            [
                np.asarray([SOURCE_DATASET_ID] * n_source),
                np.asarray([TOPUP_DATASET_ID] * n_topup),
            ]
        ),
        "source_row_index": np.concatenate([np.arange(n_source, dtype=np.int32), np.arange(n_topup, dtype=np.int32)]),
    }
    for field in common_fields:
        if field in s and field in t:
            out[field] = np.concatenate([np.asarray(s[field]), np.asarray(t[field])], axis=0)
    out["targeted_role"] = np.concatenate(
        [
            np.asarray(["v3_240_source"] * n_source),
            string_array(t, "targeted_role", n_topup, "topup_unspecified"),
        ]
    )
    out["edge_position_bin"] = np.concatenate(
        [
            np.asarray(["legacy_source_unspecified"] * n_source),
            string_array(t, "edge_position_bin", n_topup, "topup_unspecified"),
        ]
    )
    source_sig = [
        "|".join(
            [
                "v3_240_source",
                str(np.asarray(s["depth_bin"]).astype(str)[idx]),
                str(np.asarray(s["aspect_bin"]).astype(str)[idx]),
                str(np.asarray(s["curvature_template"]).astype(str)[idx]),
                "legacy_source_unspecified",
            ]
        )
        for idx in range(n_source)
    ]
    out["coverage_signature"] = np.concatenate(
        [
            np.asarray(source_sig),
            string_array(t, "coverage_signature", n_topup, "topup_unspecified"),
        ]
    )
    return out


def save_assembled_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def make_dataset(dataset_id: str, manifest: dict[str, Any], npz_path: Path, arrays: dict[str, np.ndarray]) -> ExpansionDataset:
    delta = np.asarray(arrays["delta_b"], dtype=np.float64)
    n = int(delta.shape[0])
    return ExpansionDataset(
        dataset_id=dataset_id,
        manifest=manifest,
        npz_path=npz_path,
        delta_b=delta,
        b_defect=np.asarray(arrays["b_defect"], dtype=np.float64),
        b_no_defect=np.asarray(arrays["b_no_defect"], dtype=np.float64),
        x_channels=delta.reshape(n, 9, delta.shape[-1]).astype(np.float32),
        rbc_params=np.asarray(arrays["rbc_params"], dtype=np.float32).reshape(n, 6),
        profile_pose=np.asarray(arrays["profile_pose"], dtype=np.float32).reshape(n, 6),
        projected_mask_2d=np.asarray(arrays["projected_mask_2d"], dtype=np.uint8),
        profile_depth_grid_m=np.asarray(arrays["profile_depth_grid_m"], dtype=np.float32),
        profile_depth_map_xy_m=np.asarray(arrays["profile_depth_map_xy_m"], dtype=np.float32),
        sample_ids=np.asarray(arrays["sample_ids"]).astype(str),
        split=np.asarray(arrays["split"]).astype(str),
        axis_names=[str(x) for x in np.asarray(arrays["axis_names"]).reshape(-1).tolist()],
        sensor_x=np.asarray(arrays["sensor_x"], dtype=np.float32),
        scan_line_y=np.asarray(arrays["scan_line_y"], dtype=np.float32),
        sensor_z_m=float(np.asarray(arrays["sensor_z_m"]).reshape(-1)[0]),
        curvature_template=string_array(arrays, "curvature_template", n, "unknown"),
        depth_bin=string_array(arrays, "depth_bin", n, "unknown"),
        aspect_bin=string_array(arrays, "aspect_bin", n, "unknown"),
        size_bin=string_array(arrays, "size_bin", n, "unknown"),
        source_dataset=string_array(arrays, "source_dataset", n, dataset_id),
        targeted_role=string_array(arrays, "targeted_role", n, "unspecified"),
        edge_position_bin=string_array(arrays, "edge_position_bin", n, "unspecified"),
    )


def edge_or_interior(values: np.ndarray) -> np.ndarray:
    out: list[str] = []
    for raw in np.asarray(values).astype(str):
        value = raw.lower()
        if value == "interior":
            out.append("interior")
        elif "edge" in value or value.startswith("near_") or value in {"left", "right", "upper", "lower"}:
            out.append("edge")
        elif value == "legacy_source_unspecified":
            out.append("legacy_source_unspecified")
        else:
            out.append(raw)
    return np.asarray(out)


def validate_assembled(dataset: ExpansionDataset, source: Pack, topup: Pack) -> tuple[dict[str, Any], bool]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, notes: str = "") -> None:
        add_check(checks, name, passed, observed, notes)

    splits = split_indices(dataset)
    source_mask = dataset.source_dataset == SOURCE_DATASET_ID
    topup_mask = dataset.source_dataset == TOPUP_DATASET_ID
    add("N", len(dataset.sample_ids) == 360, len(dataset.sample_ids))
    add("shape", dataset.delta_b.shape == (360, 3, 3, 201), list(dataset.delta_b.shape))
    add("view_shape", dataset.x_channels.shape == (360, 9, 201), list(dataset.x_channels.shape))
    add("split_counts", {k: len(v) for k, v in splits.items()} == {"train": 242, "val": 59, "test": 59}, {k: len(v) for k, v in splits.items()}, "top-up manifest split 80/20/20 preserved")
    add("source_split_preserved", split_counts(dataset.split[source_mask]) == {"train": 162, "val": 39, "test": 39}, split_counts(dataset.split[source_mask]))
    add("topup_split_preserved", split_counts(dataset.split[topup_mask]) == {"train": 80, "val": 20, "test": 20}, split_counts(dataset.split[topup_mask]))
    add("split_sum", sum(len(v) for v in splits.values()) == 360, sum(len(v) for v in splits.values()))
    add("split_values", set(dataset.split.tolist()) == {"train", "val", "test"}, sorted(set(dataset.split.tolist())))
    add("sample_ids_unique", len(set(dataset.sample_ids.tolist())) == 360, len(set(dataset.sample_ids.tolist())))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(dataset.sample_ids[splits[a]].tolist()) & set(dataset.sample_ids[splits[b]].tolist())
        add(f"no_{a}_{b}_leakage", not overlap, sorted(overlap)[:5])
    add("source_topup_no_collision", not (set(source.arrays["sample_ids"].astype(str).tolist()) & set(topup.arrays["sample_ids"].astype(str).tolist())), "none")
    add("delta_finite", bool(np.isfinite(dataset.delta_b).all()), "finite")
    add("b_defect_finite", bool(np.isfinite(dataset.b_defect).all()), "finite")
    add("b_no_defect_finite", bool(np.isfinite(dataset.b_no_defect).all()), "finite")
    add("labels_finite", bool(np.isfinite(dataset.rbc_params).all()), "finite")
    add("delta_equals_defect_minus_no_defect", float(np.max(np.abs(dataset.delta_b - (dataset.b_defect - dataset.b_no_defect)))) <= 1.0e-12, float(np.max(np.abs(dataset.delta_b - (dataset.b_defect - dataset.b_no_defect)))))
    add("axis_complete", dataset.axis_names == ["Bx", "By", "Bz"], dataset.axis_names)
    add("sensor_metadata_nominal", abs(dataset.sensor_z_m - NOMINAL_SENSOR_Z_M) <= 1.0e-12 and len(dataset.sensor_x) == 201 and np.allclose(dataset.scan_line_y, [-0.001, 0.0, 0.001]), {"sensor_z_m": dataset.sensor_z_m, "sensor_x_count": len(dataset.sensor_x), "scan_line_y": dataset.scan_line_y.tolist()})
    add("no_non_nominal_liftoff", abs(dataset.sensor_z_m - NOMINAL_SENSOR_Z_M) <= 1.0e-12, dataset.sensor_z_m)
    params = dataset.rbc_params
    add("LWD_positive", bool((params[:, :3] > 0.0).all()), {"min_LWD": params[:, :3].min(axis=0).tolist()})
    add(
        "curvature_params_in_schema_range",
        bool(((params[:, 3:] >= CURVATURE_WEIGHT_MIN) & (params[:, 3:] <= CURVATURE_WEIGHT_MAX)).all()),
        {
            "min": params[:, 3:].min(axis=0).tolist(),
            "max": params[:, 3:].max(axis=0).tolist(),
            "valid_range": [CURVATURE_WEIGHT_MIN, CURVATURE_WEIGHT_MAX],
        },
        "surface RBC plan clamps wLD/wWD/wLW to approximately 0.55..1.20",
    )
    add("projected_mask_nonempty", bool((dataset.projected_mask_2d.sum(axis=(1, 2)) > 0).all()), int(dataset.projected_mask_2d.sum()))

    max_grid_error = 0.0
    max_map_error = 0.0
    mask_match_count = 0
    for idx in range(len(dataset.sample_ids)):
        expected_grid = depth_grid_from_params(dataset.rbc_params[idx])
        expected_map = depth_map_from_params(dataset.rbc_params[idx], dataset.profile_pose[idx])
        expected_mask = projected_mask_from_params(dataset.rbc_params[idx], dataset.profile_pose[idx])
        max_grid_error = max(max_grid_error, float(np.max(np.abs(expected_grid - dataset.profile_depth_grid_m[idx]))))
        max_map_error = max(max_map_error, float(np.max(np.abs(expected_map - dataset.profile_depth_map_xy_m[idx]))))
        mask_match_count += int(np.array_equal(expected_mask, dataset.projected_mask_2d[idx]))
    add("profile_depth_grid_reconstruct", max_grid_error <= 1.0e-7, max_grid_error)
    add("profile_depth_map_reconstruct", max_map_error <= 1.0e-7, max_map_error)
    add("projected_mask_reconstruct", mask_match_count == len(dataset.sample_ids), {"matched": mask_match_count, "N": len(dataset.sample_ids)})

    failed = [row for row in checks if not row["pass"]]
    return {"checks": checks, "pass": not failed, "failed_checks": [row["name"] for row in failed]}, not failed


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def counts(values: np.ndarray) -> dict[str, int]:
    counter = Counter(np.asarray(values).astype(str).tolist())
    return {str(k): int(v) for k, v in sorted(counter.items())}


def coverage_report(dataset: ExpansionDataset) -> dict[str, Any]:
    params = dataset.rbc_params
    edge_category = edge_or_interior(dataset.edge_position_bin)
    return {
        "numeric": {
            "L_m": numeric_summary(params[:, 0]),
            "W_m": numeric_summary(params[:, 1]),
            "D_m": numeric_summary(params[:, 2]),
            "aspect_ratio_L_over_W": numeric_summary(params[:, 0] / params[:, 1]),
            "wLD": numeric_summary(params[:, 3]),
            "wWD": numeric_summary(params[:, 4]),
            "wLW": numeric_summary(params[:, 5]),
        },
        "bins": {
            "source_dataset": counts(dataset.source_dataset),
            "profile_family_curvature_template": counts(dataset.curvature_template),
            "depth_bin": counts(dataset.depth_bin),
            "aspect_bin": counts(dataset.aspect_bin),
            "size_bin": counts(dataset.size_bin),
            "targeted_role": counts(dataset.targeted_role),
            "edge_position_bin": counts(dataset.edge_position_bin),
            "edge_or_interior_category": counts(edge_category),
        },
    }


def train_bounds(dataset: ExpansionDataset) -> tuple[np.ndarray, np.ndarray]:
    train = split_indices(dataset)["train"]
    y = dataset.rbc_params[train]
    return y.min(axis=0), y.max(axis=0)


def clip_to_bounds(params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    low, high = bounds
    clipped = np.clip(np.asarray(params, dtype=np.float32), low[None, :], high[None, :])
    clipped_flag = np.any(np.abs(clipped - params) > 1.0e-12, axis=1)
    return clipped.astype(np.float32), clipped_flag


def evaluate_predictions(
    dataset: ExpansionDataset,
    pred_params_raw: np.ndarray,
    stats: dict[str, np.ndarray],
    clip_bounds: tuple[np.ndarray, np.ndarray],
    model_name: str,
) -> list[dict[str, Any]]:
    pred_params, clipped = clip_to_bounds(np.asarray(pred_params_raw, dtype=np.float32), clip_bounds)
    pred_norm = (pred_params - stats["y_mean"]) / stats["y_std"]
    true_norm = (dataset.rbc_params - stats["y_mean"]) / stats["y_std"]
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(dataset.sample_ids):
        pred_depth = depth_grid_from_params(pred_params[idx])
        true_depth = dataset.profile_depth_grid_m[idx]
        diff = pred_depth - true_depth
        denom = float(np.sum(true_depth**2))
        er_like = 0.0 if denom <= 1.0e-20 else float(np.sqrt(np.sum(diff**2) / denom))
        pred_mask = projected_mask_from_params(pred_params[idx], dataset.profile_pose[idx])
        mask_row = mask_metrics(pred_mask, dataset.projected_mask_2d[idx])
        true_volume = float(dataset.profile_depth_map_xy_m[idx].sum())
        pred_volume = float(depth_map_from_params(pred_params[idx], dataset.profile_pose[idx]).sum())
        volume_error = 0.0 if abs(true_volume) < 1.0e-12 else abs(pred_volume - true_volume) / abs(true_volume)
        param_abs = np.abs(pred_params[idx] - dataset.rbc_params[idx])
        norm_abs = np.abs(pred_norm[idx] - true_norm[idx])
        rows.append(
            {
                "model": model_name,
                "sample_id": str(sample_id),
                "source_dataset": str(dataset.source_dataset[idx]),
                "split": str(dataset.split[idx]),
                "targeted_role": str(dataset.targeted_role[idx]),
                "edge_position_bin": str(dataset.edge_position_bin[idx]),
                "edge_or_interior_category": str(edge_or_interior(np.asarray([dataset.edge_position_bin[idx]]))[0]),
                "curvature_template": str(dataset.curvature_template[idx]),
                "depth_bin": str(dataset.depth_bin[idx]),
                "aspect_bin": str(dataset.aspect_bin[idx]),
                "size_bin": str(dataset.size_bin[idx]),
                "clip_applied": bool(clipped[idx]),
                "normalized_param_mae": float(np.mean(norm_abs)),
                "dimension_mae_norm": float(np.mean(norm_abs[:3])),
                "curvature_mae_norm": float(np.mean(norm_abs[3:])),
                "L_mae_mm": float(param_abs[0] * 1000.0),
                "W_mae_mm": float(param_abs[1] * 1000.0),
                "D_mae_mm": float(param_abs[2] * 1000.0),
                "wLD_abs_error": float(param_abs[3]),
                "wWD_abs_error": float(param_abs[4]),
                "wLW_abs_error": float(param_abs[5]),
                "wMAE_auxiliary": float(np.mean(param_abs[3:])),
                "projected_mask_iou": mask_row["iou"],
                "projected_mask_dice": mask_row["dice"],
                "projected_mask_area_error": mask_row["area_error"],
                "projected_mask_center_error_px": mask_row["center_error"],
                "profile_depth_rmse_m": float(np.sqrt(np.mean(diff**2))),
                "er_like_profile_error": er_like,
                "max_depth_error_m": float(abs(pred_params[idx, 2] - dataset.rbc_params[idx, 2])),
                "volume_proxy_rel_error": float(volume_error),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]], indices: np.ndarray, name: str) -> dict[str, Any]:
    subset = [rows[int(idx)] for idx in indices.tolist()]
    metric_keys = [
        "normalized_param_mae",
        "dimension_mae_norm",
        "curvature_mae_norm",
        "L_mae_mm",
        "W_mae_mm",
        "D_mae_mm",
        "wLD_abs_error",
        "wWD_abs_error",
        "wLW_abs_error",
        "wMAE_auxiliary",
        "projected_mask_iou",
        "projected_mask_dice",
        "projected_mask_area_error",
        "projected_mask_center_error_px",
        "profile_depth_rmse_m",
        "er_like_profile_error",
        "max_depth_error_m",
        "volume_proxy_rel_error",
        "clip_applied",
    ]
    out: dict[str, Any] = {"eval_set": name, "sample_count": len(subset)}
    for key in metric_keys:
        values = [float(row[key]) for row in subset]
        out[key] = float(np.mean(values)) if values else 0.0
    return out


def comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, lower in LOWER_IS_BETTER.items():
        cand = float(candidate[key])
        base = float(baseline[key])
        delta = cand - base
        improved = delta < -1.0e-12 if lower else delta > 1.0e-12
        out[key] = {"candidate": cand, "baseline_20_85": base, "delta": delta, "improved": improved}
    return out


def threshold_check(metrics: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, spec in OLD_TEST_THRESHOLDS.items():
        value = float(metrics[key])
        target = float(spec["value"])
        passed = value <= target if spec["op"] == "<=" else value >= target
        rows[key] = {"value": value, "op": spec["op"], "threshold": target, "pass": passed}
    return {"pass": all(row["pass"] for row in rows.values()), "metrics": rows}


def eval_sets(dataset: ExpansionDataset) -> dict[str, np.ndarray]:
    source = dataset.source_dataset == SOURCE_DATASET_ID
    topup = dataset.source_dataset == TOPUP_DATASET_ID
    test = dataset.split == "test"
    sets = {
        "old_v3_240_test": np.where(source & test)[0],
        "topup_test": np.where(topup & test)[0],
        "assembled_v1_360_test": np.where(test)[0],
    }
    topup_test = topup & test
    for field, values in (
        ("targeted_role", dataset.targeted_role),
        ("edge_position_bin", dataset.edge_position_bin),
        ("curvature_template", dataset.curvature_template),
        ("depth_bin", dataset.depth_bin),
        ("aspect_bin", dataset.aspect_bin),
        ("edge_or_interior_category", edge_or_interior(dataset.edge_position_bin)),
    ):
        for value in sorted(set(values[topup_test].astype(str).tolist())):
            mask = topup_test & (values.astype(str) == value)
            if int(mask.sum()) > 0:
                sets[f"topup_test__{field}={value}"] = np.where(mask)[0]
    return sets


def load_baseline_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any], gate.RBCConvRegressor]:
    manifest = read_manifest(path)
    if manifest.get("dataset_id") != SOURCE_DATASET_ID:
        raise RuntimeError(f"unexpected baseline artifact dataset_id: {manifest.get('dataset_id')}")
    checkpoint_path = Path(str(manifest["checkpoint_path"]))
    prediction_path = Path(str(manifest["prediction_artifact_path"]))
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    if sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("baseline checkpoint sha256 mismatch")
    if sha256_file(prediction_path) != manifest["prediction_artifact_sha256"]:
        raise RuntimeError("baseline prediction artifact sha256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = gate.RBCConvRegressor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return manifest, checkpoint, model


def predict_with_checkpoint(dataset: ExpansionDataset, checkpoint: dict[str, Any], model: gate.RBCConvRegressor) -> np.ndarray:
    x_mean = np.asarray(checkpoint["normalization"]["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["normalization"]["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["normalization"]["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["normalization"]["y_std"], dtype=np.float32)
    x_norm = ((dataset.x_channels - x_mean) / x_std).astype(np.float32)
    pred_norm = gate.predict_norm(model, x_norm)
    return denormalize_y(pred_norm, {"y_mean": y_mean, "y_std": y_std})


def seed_summary_row(seed_out: dict[str, Any], selected_seed: int, dataset: ExpansionDataset, stats: dict[str, np.ndarray]) -> dict[str, Any]:
    pred_raw = denormalize_y(seed_out["pred_norm"], stats)
    rows = evaluate_predictions(dataset, pred_raw, stats, train_bounds(dataset), f"candidate_seed_{seed_out['seed']}")
    split_aggs = {split: aggregate_rows(rows, split_indices(dataset)[split], split) for split in ("train", "val", "test")}
    return {
        "seed": int(seed_out["seed"]),
        "selected_seed": int(seed_out["seed"]) == selected_seed,
        "best_epoch": int(seed_out["best_epoch"]),
        "best_val_selection_metric": float(seed_out["best_val_score"]),
        "min_train_epoch": int(seed_out["min_train_epoch"]),
        "min_train_normalized_param_mae": float(seed_out["min_train_normalized_param_mae"]),
        "train_normalized_param_mae": split_aggs["train"]["normalized_param_mae"],
        "val_normalized_param_mae": split_aggs["val"]["normalized_param_mae"],
        "test_normalized_param_mae": split_aggs["test"]["normalized_param_mae"],
        "test_profile_depth_rmse_m": split_aggs["test"]["profile_depth_rmse_m"],
        "test_er_like_profile_error": split_aggs["test"]["er_like_profile_error"],
        "test_L_mae_mm": split_aggs["test"]["L_mae_mm"],
        "test_W_mae_mm": split_aggs["test"]["W_mae_mm"],
        "test_D_mae_mm": split_aggs["test"]["D_mae_mm"],
        "test_projected_mask_dice": split_aggs["test"]["projected_mask_dice"],
    }


def decide_gate(old_thresholds: dict[str, Any], eval_results: dict[str, Any]) -> dict[str, Any]:
    topup_comp = eval_results["topup_test"]["comparison"]
    assembled_comp = eval_results["assembled_v1_360_test"]["comparison"]
    topup_improvements = sum(1 for key in PRIMARY_IMPROVEMENT_METRICS if bool(topup_comp[key]["improved"]))
    assembled_improvements = sum(1 for key in PRIMARY_IMPROVEMENT_METRICS if bool(assembled_comp[key]["improved"]))
    hard_bin_items = [
        item
        for name, item in eval_results.items()
        if name.startswith("topup_test__")
    ]
    hard_bin_improved_sets = sum(
        1
        for item in hard_bin_items
        if sum(1 for key in PRIMARY_IMPROVEMENT_METRICS if bool(item["comparison"][key]["improved"])) >= 3
    )
    base = {
        "topup_primary_improvements": int(topup_improvements),
        "assembled_primary_improvements": int(assembled_improvements),
        "hard_bin_improved_sets": int(hard_bin_improved_sets),
        "hard_bin_total_sets": int(len(hard_bin_items)),
        "primary_improvement_metrics": PRIMARY_IMPROVEMENT_METRICS,
    }
    if not old_thresholds["pass"]:
        return {
            **base,
            "outcome": "FAIL",
            "reason": "old_v3_240_test non-regression threshold failed",
            "old_test_non_regression_pass": False,
        }
    if topup_improvements >= 3 or assembled_improvements >= 3:
        outcome = "PASS"
        reason = "old-test non-regression passed and top-up or assembled test improves in most primary metrics"
    elif topup_improvements > 0 or assembled_improvements > 0 or hard_bin_improved_sets > 0:
        outcome = "PARTIAL"
        reason = "old-test non-regression passed but gains are mixed or hard-bin-local"
    else:
        outcome = "FAIL"
        reason = "old-test non-regression passed but top-up/assembled tests do not improve meaningfully"
    return {
        **base,
        "outcome": outcome,
        "reason": reason,
        "old_test_non_regression_pass": True,
    }


def build_gate_manifest(
    *,
    source: Pack,
    topup: Pack,
    dataset: ExpansionDataset,
    validation: dict[str, Any],
    gate_decision: dict[str, Any],
    metrics_path: Path,
    summary_path: Path,
    manifest_path: Path,
    assembled_npz: Path,
) -> dict[str, Any]:
    return {
        "manifest_type": "surface_rbc_expansion_v1_360_training_gate",
        "dataset_role": "assembled_training_gate_candidate",
        "route": MANIFEST_ROUTE,
        "schema_version": SCHEMA_VERSION,
        "source_dataset_id": SOURCE_DATASET_ID,
        "topup_dataset_id": str(topup.manifest.get("dataset_id", TOPUP_DATASET_ID)),
        "assembled_dataset_id": ASSEMBLED_DATASET_ID,
        "source_manifest_paths": [str(SOURCE_MANIFEST), str(TOPUP_MANIFEST)],
        "source_npz_paths": [str(source.npz_path), str(topup.npz_path)],
        "assembled_npz_path": str(assembled_npz),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "N": int(len(dataset.sample_ids)),
        "shape": list(dataset.delta_b.shape),
        "view_shape": list(dataset.x_channels.shape),
        "split_counts": {name: int(len(idx)) for name, idx in split_indices(dataset).items()},
        "source_split_counts": {"train": 162, "val": 39, "test": 39},
        "topup_split_counts": split_counts(dataset.split[dataset.source_dataset == TOPUP_DATASET_ID]),
        "topup_split_policy": "preserved_manifest_split_80_20_20",
        "baseline_ready": False,
        "train_ready_candidate": bool(validation["pass"]),
        "validation_pass": bool(validation["pass"]),
        "training_gate_outcome": gate_decision["outcome"],
        "status": f"training_gate_{str(gate_decision['outcome']).lower()}",
        "allowed_use": [
            "schema_validation",
            "explicit_surface_rbc_expansion_training_gate_review",
        ],
        "forbidden_use": [
            "automatic_mainline_training",
            "baseline_update",
            "current_baseline_replacement",
            "latest_newest_auto_discovery",
            "direct_training_without_manifest_gate",
        ],
        "axes": ["Bx", "By", "Bz"],
        "sensor_z_m": NOMINAL_SENSOR_Z_M,
        "scan_line_y": [-0.001, 0.0, 0.001],
        "sensor_x_count": 201,
        "labels": PARAM_NAMES,
        "current_baseline_update": False,
        "baseline_transition": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pinn_commit": git_value(["rev-parse", "HEAD"]),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "topup_manifest_sha256": sha256_file(TOPUP_MANIFEST),
        "assembled_npz_sha256": sha256_file(assembled_npz) if assembled_npz.exists() else "",
    }


def write_summary(
    path: Path,
    dataset: ExpansionDataset,
    validation: dict[str, Any],
    gate_decision: dict[str, Any],
    eval_results: dict[str, Any],
    selected_seed: int,
) -> None:
    old = eval_results["old_v3_240_test"]["candidate"]
    topup = eval_results["topup_test"]
    assembled = eval_results["assembled_v1_360_test"]
    lines = [
        "# Surface RBC Expansion v1 360 Training Gate",
        "",
        f"- assembled_dataset_id: `{ASSEMBLED_DATASET_ID}`",
        f"- N: {len(dataset.sample_ids)}",
        f"- split: { {name: int(len(idx)) for name, idx in split_indices(dataset).items()} }",
        f"- selected_seed: {selected_seed}",
        f"- validation_pass: {validation['pass']}",
        f"- gate_outcome: {gate_decision['outcome']}",
        f"- gate_reason: {gate_decision['reason']}",
        f"- baseline_ready: false",
        f"- CURRENT_BASELINE_update: false",
        "",
        "## Old v3_240 Test Non-Regression",
        "",
        f"- profile_depth_rmse_m: {old['profile_depth_rmse_m']:.12g} <= {OLD_TEST_THRESHOLDS['profile_depth_rmse_m']['value']}",
        f"- Er-like: {old['er_like_profile_error']:.12g} <= {OLD_TEST_THRESHOLDS['er_like_profile_error']['value']}",
        f"- L_MAE_mm: {old['L_mae_mm']:.12g} <= {OLD_TEST_THRESHOLDS['L_mae_mm']['value']}",
        f"- W_MAE_mm: {old['W_mae_mm']:.12g} <= {OLD_TEST_THRESHOLDS['W_mae_mm']['value']}",
        f"- D_MAE_mm: {old['D_mae_mm']:.12g} <= {OLD_TEST_THRESHOLDS['D_mae_mm']['value']}",
        f"- projected_mask_Dice: {old['projected_mask_dice']:.12g} >= {OLD_TEST_THRESHOLDS['projected_mask_dice']['value']}",
        "",
        "## Comparator Summary",
        "",
        f"- topup_test primary improvements vs 20.85: {gate_decision['topup_primary_improvements']}/4",
        f"- assembled_test primary improvements vs 20.85: {gate_decision['assembled_primary_improvements']}/4",
        f"- hard-bin improved sets: {gate_decision.get('hard_bin_improved_sets', 0)}/{gate_decision.get('hard_bin_total_sets', 0)}",
        f"- topup_test profile RMSE candidate/baseline: {topup['candidate']['profile_depth_rmse_m']:.12g} / {topup['baseline_20_85']['profile_depth_rmse_m']:.12g}",
        f"- topup_test Er-like candidate/baseline: {topup['candidate']['er_like_profile_error']:.12g} / {topup['baseline_20_85']['er_like_profile_error']:.12g}",
        f"- topup_test D MAE mm candidate/baseline: {topup['candidate']['D_mae_mm']:.12g} / {topup['baseline_20_85']['D_mae_mm']:.12g}",
        f"- topup_test Dice candidate/baseline: {topup['candidate']['projected_mask_dice']:.12g} / {topup['baseline_20_85']['projected_mask_dice']:.12g}",
        f"- assembled_test profile RMSE candidate/baseline: {assembled['candidate']['profile_depth_rmse_m']:.12g} / {assembled['baseline_20_85']['profile_depth_rmse_m']:.12g}",
        f"- assembled_test Er-like candidate/baseline: {assembled['candidate']['er_like_profile_error']:.12g} / {assembled['baseline_20_85']['er_like_profile_error']:.12g}",
        f"- assembled_test D MAE mm candidate/baseline: {assembled['candidate']['D_mae_mm']:.12g} / {assembled['baseline_20_85']['D_mae_mm']:.12g}",
        f"- assembled_test Dice candidate/baseline: {assembled['candidate']['projected_mask_dice']:.12g} / {assembled['baseline_20_85']['projected_mask_dice']:.12g}",
        "",
        "Boundary: this gate creates an assembled candidate dataset for explicit review only. It is not a baseline transition.",
    ]
    write_text(path, lines)


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit(f"Run from PINN_project root: {ROOT}")
    check_no_overwrite([args.metrics_json, args.summary_md, args.gate_manifest, args.assembled_npz], args.overwrite)

    source = load_pack(args.source_manifest, SOURCE_DATASET_ID)
    topup = load_pack(args.topup_manifest, TOPUP_DATASET_ID)
    preflight_checks, preflight_pass = validate_pack_preflight(source, topup)
    if not preflight_pass:
        raise RuntimeError("preflight failed: " + ", ".join(row["name"] for row in preflight_checks if not row["pass"]))

    assembled_arrays = concatenate_arrays(source, topup)
    save_assembled_npz(args.assembled_npz, assembled_arrays)
    assembled_manifest_stub = {
        "dataset_id": ASSEMBLED_DATASET_ID,
        "source_dataset_id": SOURCE_DATASET_ID,
        "topup_dataset_id": str(topup.manifest.get("dataset_id", TOPUP_DATASET_ID)),
    }
    dataset = make_dataset(ASSEMBLED_DATASET_ID, assembled_manifest_stub, args.assembled_npz, assembled_arrays)
    validation, validation_pass = validate_assembled(dataset, source, topup)
    if not validation_pass:
        metrics = {
            "assembled_dataset_id": ASSEMBLED_DATASET_ID,
            "preflight": {"pass": preflight_pass, "checks": preflight_checks},
            "validation": validation,
            "training": {"run": False},
            "gate_decision": {"outcome": "FAIL", "reason": "assembled validation failed"},
            "baseline_ready": False,
        }
        write_json(args.metrics_json, metrics)
        manifest = build_gate_manifest(
            source=source,
            topup=topup,
            dataset=dataset,
            validation=validation,
            gate_decision=metrics["gate_decision"],
            metrics_path=args.metrics_json,
            summary_path=args.summary_md,
            manifest_path=args.gate_manifest,
            assembled_npz=args.assembled_npz,
        )
        write_json(args.gate_manifest, manifest)
        write_text(args.summary_md, ["# Surface RBC Expansion v1 360 Training Gate", "", "validation_pass: false", "gate_outcome: FAIL"])
        raise RuntimeError("assembled validation failed")

    stats = train_normalization(dataset)
    x_norm = normalize_x(dataset, stats)
    y_norm = normalize_y(dataset, stats)
    splits = split_indices(dataset)
    seed_outputs = [gate.train_one_seed(seed, x_norm, y_norm, splits, args) for seed in args.seeds]
    selected = min(seed_outputs, key=lambda item: item["best_val_score"])
    selected_seed = int(selected["seed"])
    seed_rows = [seed_summary_row(seed_out, selected_seed, dataset, stats) for seed_out in seed_outputs]
    candidate_pred_raw = denormalize_y(selected["pred_norm"], stats)
    candidate_rows = evaluate_predictions(dataset, candidate_pred_raw, stats, train_bounds(dataset), "surface_expansion_candidate")

    baseline_manifest, baseline_checkpoint, baseline_model = load_baseline_artifact(args.baseline_artifact_manifest)
    source_dataset = make_dataset(SOURCE_DATASET_ID, source.manifest, source.npz_path, source.arrays)
    baseline_stats = {
        "y_mean": np.asarray(baseline_checkpoint["normalization"]["y_mean"], dtype=np.float32),
        "y_std": np.asarray(baseline_checkpoint["normalization"]["y_std"], dtype=np.float32),
    }
    baseline_pred_raw = predict_with_checkpoint(dataset, baseline_checkpoint, baseline_model)
    baseline_rows = evaluate_predictions(dataset, baseline_pred_raw, baseline_stats, train_bounds(source_dataset), "baseline_20_85")

    sets = eval_sets(dataset)
    eval_results: dict[str, Any] = {}
    for name, indices in sets.items():
        cand = aggregate_rows(candidate_rows, indices, name)
        base = aggregate_rows(baseline_rows, indices, name)
        eval_results[name] = {
            "sample_count": int(len(indices)),
            "candidate": cand,
            "baseline_20_85": base,
            "comparison": comparison(cand, base),
        }

    old_thresholds = threshold_check(eval_results["old_v3_240_test"]["candidate"])
    gate_decision = decide_gate(old_thresholds, eval_results)
    hard_bin_summary = {
        name: {
            "sample_count": item["sample_count"],
            "primary_improvements": sum(
                1 for key in PRIMARY_IMPROVEMENT_METRICS if bool(item["comparison"][key]["improved"])
            ),
            "candidate": {key: item["candidate"][key] for key in PRIMARY_IMPROVEMENT_METRICS},
            "baseline_20_85": {key: item["baseline_20_85"][key] for key in PRIMARY_IMPROVEMENT_METRICS},
        }
        for name, item in eval_results.items()
        if name.startswith("topup_test__")
    }

    metrics = {
        "assembled_dataset_id": ASSEMBLED_DATASET_ID,
        "source_dataset_id": SOURCE_DATASET_ID,
        "topup_dataset_id": str(topup.manifest.get("dataset_id", TOPUP_DATASET_ID)),
        "N": int(len(dataset.sample_ids)),
        "shape": list(dataset.delta_b.shape),
        "view_shape": list(dataset.x_channels.shape),
        "split_counts": {name: int(len(idx)) for name, idx in splits.items()},
        "topup_split_policy": "preserved_manifest_split_80_20_20",
        "preflight": {"pass": preflight_pass, "checks": preflight_checks},
        "validation": validation,
        "coverage_report": coverage_report(dataset),
        "training": {
            "run": True,
            "model_family": "20.77/20.85 small Conv1D encoder + MLP six-parameter head",
            "input": "delta_b Bx/By/Bz flattened from (N,3,3,201) to (N,9,201)",
            "output_labels": PARAM_NAMES,
            "seeds": [int(seed) for seed in args.seeds],
            "selected_seed": selected_seed,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "selection": "assembled validation split only",
            "seed_summary": seed_rows,
            "checkpoint_written": False,
        },
        "baseline_comparator": {
            "artifact_manifest": str(args.baseline_artifact_manifest),
            "artifact_id": baseline_manifest.get("artifact_id"),
            "checkpoint_path": baseline_manifest.get("checkpoint_path"),
            "checkpoint_sha256": baseline_manifest.get("checkpoint_sha256"),
            "normalization": "frozen 20.85 v3_240 train-only stats",
            "clip_bounds": "frozen 20.85 v3_240 train split bounds",
        },
        "evaluation": {
            "old_test_thresholds": OLD_TEST_THRESHOLDS,
            "old_test_non_regression": old_thresholds,
            "eval_sets": eval_results,
            "hard_bin_summary": hard_bin_summary,
        },
        "gate_decision": gate_decision,
        "baseline_ready": False,
        "train_ready_candidate": bool(validation["pass"]),
        "CURRENT_BASELINE_update": False,
    }
    write_json(args.metrics_json, metrics)
    manifest = build_gate_manifest(
        source=source,
        topup=topup,
        dataset=dataset,
        validation=validation,
        gate_decision=gate_decision,
        metrics_path=args.metrics_json,
        summary_path=args.summary_md,
        manifest_path=args.gate_manifest,
        assembled_npz=args.assembled_npz,
    )
    write_json(args.gate_manifest, manifest)
    write_summary(args.summary_md, dataset, validation, gate_decision, eval_results, selected_seed)

    staged = staged_files()
    forbidden = forbidden_staged(staged)
    if forbidden:
        raise RuntimeError("forbidden artifacts staged: " + ", ".join(forbidden))
    print(f"assembled_dataset_id={ASSEMBLED_DATASET_ID}")
    print(f"split_counts={metrics['split_counts']}")
    print(f"validation_pass={validation['pass']}")
    print(f"gate_outcome={gate_decision['outcome']}")
    print(f"selected_seed={selected_seed}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
