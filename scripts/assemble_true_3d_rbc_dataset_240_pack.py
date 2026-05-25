"""Assemble v2_120 source + 20.76 top-up into the v3_240 dataset candidate."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SOURCE_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v2_120"
TOPUP_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76"
ASSEMBLED_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v3_240"

DEFAULT_SOURCE = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.npz"
DEFAULT_TOPUP = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76.npz"
DEFAULT_ASSEMBLED = ROOT / "data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.npz"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_dataset_240_assembled_summary.txt"
DEFAULT_INDEX = ROOT / "results/metrics/true_3d_rbc_dataset_240_assembled_index.csv"
DEFAULT_GROUPS = ROOT / "results/metrics/true_3d_rbc_dataset_240_assembled_group_summary.csv"

INDEX_FIELDS = [
    "source_pack",
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "size_bin",
    "aspect_bin",
    "schema_pass",
    "delta_max_abs_error",
    "defect_signal_norm",
]
GROUP_FIELDS = ["group_key", "group_value", "sample_count", "schema_pass_count", "mean_delta_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble v2_120 + 20.76 top-up into v3_240.")
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--topup-npz", type=Path, default=DEFAULT_TOPUP)
    parser.add_argument("--assembled-npz", type=Path, default=DEFAULT_ASSEMBLED)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def scalar_str(array: np.ndarray) -> str:
    return str(np.asarray(array).reshape(-1)[0])


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def concat_pack(source: dict[str, np.ndarray], topup: dict[str, np.ndarray], status: str) -> dict[str, np.ndarray]:
    source_n = len(source["sample_ids"])
    topup_n = len(topup["sample_ids"])
    out: dict[str, np.ndarray] = {}
    for key, value in source.items():
        if key in {"dataset_id", "status", "metadata"}:
            continue
        if key in topup and value.shape[:1] == (source_n,) and topup[key].shape[:1] == (topup_n,):
            out[key] = np.concatenate([value, topup[key]], axis=0)
        else:
            out[key] = value
    out["dataset_id"] = np.asarray([ASSEMBLED_ID], dtype="<U96")
    out["status"] = np.asarray([status], dtype="<U64")
    out["metadata"] = np.asarray(
        {
            "dataset_id": ASSEMBLED_ID,
            "source_dataset_ids": [SOURCE_ID, TOPUP_ID],
            "status": status,
            "baseline_ready": False,
            "stage": "20.76",
        },
        dtype=object,
    )
    return out


def string_array(value: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(value).tolist()]


def int_scalar_array(value: np.ndarray, index: int) -> int:
    return int(np.asarray(value).reshape(-1)[index])


def float_scalar_array(value: np.ndarray, index: int) -> float:
    return float(np.asarray(value).reshape(-1)[index])


def bool_scalar_array(value: np.ndarray, index: int) -> bool:
    return bool(np.asarray(value).reshape(-1)[index])


def metrics(pack: dict[str, np.ndarray], source_labels: list[str]) -> tuple[list[dict[str, Any]], bool]:
    sample_ids = string_array(pack["sample_ids"])
    splits = string_array(pack["split"])
    curvatures = string_array(pack["curvature_template"])
    depths = string_array(pack["depth_bin"])
    sizes = string_array(pack["size_bin"])
    aspects = string_array(pack["aspect_bin"])
    methods = string_array(pack["geometry_method_used"])
    protocols = string_array(pack["selected_solver_protocol"])
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(sample_ids):
        delta = np.asarray(pack["delta_b"][idx], dtype=float)
        defect = np.asarray(pack["b_defect"][idx], dtype=float)
        no_defect = np.asarray(pack["b_no_defect"][idx], dtype=float)
        delta_error = float(np.max(np.abs(delta - (defect - no_defect))))
        norm = float(np.linalg.norm(delta))
        schema_pass = (
            delta.shape == (3, 3, 201)
            and bool(np.isfinite(delta).all() and np.isfinite(defect).all() and np.isfinite(no_defect).all())
            and delta_error <= 1.0e-12
            and norm > 0.0
            and methods[idx] == "imported_watertight_mesh_solid"
            and protocols[idx] == "default"
            and int_scalar_array(pack["mesh_auto_size"], idx) == 5
            and abs(float_scalar_array(pack["full_source_jscale"], idx) - 1.0) <= 1.0e-12
            and bool_scalar_array(pack["material_fix_applied"], idx)
            and bool_scalar_array(pack["domain_material_audit_pass"], idx)
            and bool_scalar_array(pack["solver_probe_pass"], idx)
            and not bool_scalar_array(pack["exact_piao_rbc"], idx)
            and bool_scalar_array(pack["rbc_style_approximation"], idx)
        )
        rows.append(
            {
                "source_pack": source_labels[idx],
                "sample_id": sample_id,
                "split": splits[idx],
                "curvature_template": curvatures[idx],
                "depth_bin": depths[idx],
                "size_bin": sizes[idx],
                "aspect_bin": aspects[idx],
                "schema_pass": schema_pass,
                "delta_max_abs_error": delta_error,
                "defect_signal_norm": norm,
            }
        )
    validation_pass = len(sample_ids) == len(set(sample_ids)) and all(bool(row["schema_pass"]) for row in rows)
    return rows, validation_pass


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for key in ("source_pack", "split", "curvature_template", "depth_bin", "size_bin", "aspect_bin"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row[key])].append(row)
        for value, bucket in sorted(buckets.items()):
            groups.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "sample_count": len(bucket),
                    "schema_pass_count": sum(1 for row in bucket if row["schema_pass"]),
                    "mean_delta_norm": float(np.mean([float(row["defect_signal_norm"]) for row in bucket])),
                }
            )
    return groups


def train_ready(rows: list[dict[str, Any]], validation_pass: bool) -> bool:
    split = Counter(row["split"] for row in rows)
    curv = Counter(row["curvature_template"] for row in rows)
    return (
        validation_pass
        and len(rows) >= 216
        and split.get("train", 0) >= 144
        and split.get("val", 0) >= 36
        and split.get("test", 0) >= 36
        and all(curv.get(name, 0) >= 43 for name in ["sharp", "round", "boxy", "LD_dominant", "WD_dominant"])
    )


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.assembled_npz, args.summary, args.index, args.groups], args.overwrite)
    source = load_npz(args.source_npz)
    topup = load_npz(args.topup_npz)
    if scalar_str(source["dataset_id"]) != SOURCE_ID:
        raise RuntimeError(f"source NPZ dataset_id mismatch: {scalar_str(source['dataset_id'])}")
    if scalar_str(topup["dataset_id"]) != TOPUP_ID:
        raise RuntimeError(f"top-up NPZ dataset_id mismatch: {scalar_str(topup['dataset_id'])}")

    source_labels = ["v2_120_source"] * len(source["sample_ids"]) + ["v3_topup_20_76"] * len(topup["sample_ids"])
    draft = concat_pack(source, topup, "assembled_pending")
    rows, validation_pass = metrics(draft, source_labels)
    status = "pilot_generated" if train_ready(rows, validation_pass) else "partial_pilot_generated"
    assembled = concat_pack(source, topup, status)
    args.assembled_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.assembled_npz, **assembled)

    write_csv(args.index, rows, INDEX_FIELDS)
    group_metrics = group_rows(rows)
    write_csv(args.groups, group_metrics, GROUP_FIELDS)
    split = Counter(row["split"] for row in rows)
    curv = Counter(row["curvature_template"] for row in rows)
    depth = Counter(row["depth_bin"] for row in rows)
    aspect = Counter(row["aspect_bin"] for row in rows)
    lines = [
        "20.76 true 3D RBC v3_240 assembled summary",
        "",
        f"dataset_id: {ASSEMBLED_ID}",
        f"source_dataset_id: {SOURCE_ID}",
        f"topup_dataset_id: {TOPUP_ID}",
        f"assembled_npz: {args.assembled_npz}",
        f"n_samples: {len(rows)}",
        f"split_counts: {dict(split)}",
        f"curvature_counts: {dict(curv)}",
        f"depth_counts: {dict(depth)}",
        f"aspect_counts: {dict(aspect)}",
        f"schema_validation_pass_precheck: {validation_pass}",
        f"status: {status}",
        f"train_ready_candidate_precheck: {train_ready(rows, validation_pass)}",
        "baseline_ready: False",
        "",
        "Boundary: assembled NPZ is generated data and must not be committed.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if len(rows) < 216:
        raise RuntimeError(f"assembled N below minimum acceptable threshold: {len(rows)}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
