#!/usr/bin/env python
"""组装 22.2b internal defect hard-case augmented pack。

输入为显式 v2_240 manifest 和 hard-case top-up NPZ；输出 ignored NPZ：
comsol_internal_defect_pilot_pack_v3_hardcase.npz。脚本不训练、不运行
COMSOL、不扫描 latest/newest、不更新 CURRENT_BASELINE。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_pilot_pack_v3_hardcase"
SOURCE_DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
TOPUP_DATASET_ID = "comsol_internal_defect_hard_case_topup_pack_v1"
SOURCE_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json"
PLAN_CSV = ROOT / "results/metrics/internal_defect_hard_case_topup_plan.csv"
TOPUP_NPZ = ROOT / "data/comsol_mfl/generated/internal_defect_hard_case_topup_pack/internal_defect_hard_case_topup_pack_v1.npz"
OUTPUT_NPZ = ROOT / "data/comsol_mfl/generated/internal_defect_pilot_pack_v3_hardcase/comsol_internal_defect_pilot_pack_v3_hardcase.npz"
SUMMARY = ROOT / "results/summaries/internal_defect_hard_case_augmented_pack_assembly_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_hard_case_augmented_pack_assembly_metrics.csv"
MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json"
REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
REGISTRY_SUMMARY = ROOT / "results/summaries/internal_defect_hard_case_topup_registry_summary.txt"

ROUTE = "internal_buried_defect_hardcase"
SCHEMA_VERSION = "internal_defect_feasibility_v3_hardcase"
ALLOWED_USE = ["schema_validation", "explicit_internal_training_gate"]
FORBIDDEN_USE = [
    "automatic_mainline_training",
    "baseline_update",
    "current_baseline_replacement",
    "latest_newest_auto_discovery",
    "direct_training_without_manifest_gate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="组装 internal defect hard-case augmented pack。")
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--topup-npz", type=Path, default=TOPUP_NPZ)
    parser.add_argument("--output-npz", type=Path, default=OUTPUT_NPZ)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--registry-summary", type=Path, default=REGISTRY_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def strings(arr: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(arr).reshape(-1).tolist()]


def source_npz_from_manifest(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != SOURCE_DATASET_ID:
        raise ValueError(f"source manifest dataset_id 不匹配: {payload.get('dataset_id')}")
    npz_path = Path(str(payload.get("npz_path", "")))
    if not npz_path.exists():
        raise FileNotFoundError(f"source NPZ 不存在: {npz_path}")
    return npz_path


def plan_by_id(plan_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in plan_rows:
        sample_id = row.get("planned_sample_id") or row.get("sample_id")
        if sample_id:
            out[sample_id] = row
    return out


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("拒绝覆盖已存在文件:\n" + "\n".join(str(path) for path in existing))


def row_count(arrays: dict[str, np.ndarray]) -> int:
    return int(len(arrays["sample_ids"]))


def can_concat(value: np.ndarray, expected_rows: int) -> bool:
    return value.ndim >= 1 and int(value.shape[0]) == expected_rows


def combine_arrays(source: dict[str, np.ndarray], topup: dict[str, np.ndarray], plan_rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    n_source = row_count(source)
    n_topup = row_count(topup)
    plan_lookup = plan_by_id(plan_rows)
    topup_ids = strings(topup["sample_ids"])
    source_ids = strings(source["sample_ids"])
    duplicate = sorted(set(source_ids).intersection(topup_ids))
    if duplicate:
        raise RuntimeError(f"sample_id 重复: {duplicate[:10]}")

    out: dict[str, np.ndarray] = {
        "dataset_id": np.asarray(DATASET_ID, dtype=object),
        "source_dataset_ids": np.asarray([SOURCE_DATASET_ID, TOPUP_DATASET_ID], dtype=object),
    }
    for key, value in source.items():
        if key in {"dataset_id", "source_dataset_ids"}:
            continue
        if key in topup and can_concat(value, n_source) and can_concat(topup[key], n_topup):
            out[key] = np.concatenate([value, topup[key]], axis=0)
        else:
            out[key] = value

    topup_split = [plan_lookup.get(sample_id, {}).get("split_hint") or str(topup["split"][idx]) for idx, sample_id in enumerate(topup_ids)]
    out["split"] = np.asarray(strings(source["split"]) + topup_split, dtype="<U16")
    out["row_origin"] = np.asarray(["source_v2_240"] * n_source + ["hardcase_topup_v1"] * n_topup, dtype="<U32")
    out["source_dataset_id_per_row"] = np.asarray([SOURCE_DATASET_ID] * n_source + [TOPUP_DATASET_ID] * n_topup, dtype="<U96")
    for field in ["target_id", "target_reason", "neighbor_strategy", "source_failure_sample_id", "center_region", "selection_priority"]:
        out[f"hardcase_{field}"] = np.asarray([""] * n_source + [plan_lookup.get(sample_id, {}).get(field, "") for sample_id in topup_ids], dtype=object)
    return out


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def counts(arrays: dict[str, np.ndarray], field: str) -> dict[str, int]:
    return dict(Counter(strings(arrays[field]))) if field in arrays else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_entry(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"## {manifest['dataset_id']}",
            "",
            f"- dataset_role: {manifest['dataset_role']}",
            f"- status: {manifest['status']}",
            f"- route: {manifest['route']}",
            f"- stage: {manifest['stage']}",
            f"- schema_version: {manifest['schema_version']}",
            f"- internal_surface_mixed: {str(manifest['internal_surface_mixed']).lower()}",
            f"- path: `{manifest['npz_path']}`",
            f"- manifest_path: `{manifest['manifest_path']}`",
            f"- n_samples: {manifest['n_samples']}",
            f"- source_rows: {manifest['source_rows']}",
            f"- topup_rows: {manifest['topup_rows']}",
            f"- split_counts: {manifest['split_counts']}",
            f"- shape_counts: {manifest['shape_counts']}",
            f"- burial_depth_counts: {manifest['burial_depth_counts']}",
            f"- size_counts: {manifest['size_counts']}",
            f"- aspect_counts: {manifest['aspect_counts']}",
            f"- train_ready_candidate: {str(manifest['train_ready_candidate']).lower()}",
            f"- baseline_ready: {str(manifest['baseline_ready']).lower()}",
            f"- auto_discovery_allowed: {str(manifest['auto_discovery_allowed']).lower()}",
            f"- latest_newest_discovery_allowed: {str(manifest['latest_newest_discovery_allowed']).lower()}",
            f"- allowed_use: {', '.join(manifest['allowed_use'])}",
            f"- forbidden_use: {', '.join(manifest['forbidden_use'])}",
            f"- source_dataset_ids: {', '.join(manifest['source_dataset_ids'])}",
            f"- npz_sha256: {manifest['npz_sha256']}",
            "- notes: Hard-case augmented internal branch pack; generated NPZ/data files are not committed; not a baseline.",
            "",
        ]
    )


def update_registry(path: Path, manifest: dict[str, Any]) -> None:
    entry = registry_entry(manifest)
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else "# COMSOL Data Registry\n\n"
    heading = f"## {manifest['dataset_id']}"
    if heading in text:
        start = text.index(heading)
        next_start = text.find("\n## ", start + 1)
        if next_start == -1:
            text = text[:start].rstrip() + "\n\n" + entry
        else:
            text = text[:start].rstrip() + "\n\n" + entry + text[next_start:].lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + entry
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.output_npz, args.summary, args.metrics, args.manifest], args.overwrite)
    source_npz = source_npz_from_manifest(args.source_manifest)
    source = load_npz(source_npz)
    topup = load_npz(args.topup_npz)
    plan_rows = read_csv(args.plan_csv)
    assembled = combine_arrays(source, topup, plan_rows)
    write_npz(args.output_npz, assembled)

    source_rows = row_count(source)
    topup_rows = row_count(topup)
    total_rows = row_count(assembled)
    train_ready = topup_rows >= 72 and total_rows >= source_rows + 72
    status = "hardcase_pack_generated" if topup_rows >= 120 else ("partial_hardcase_pack_generated" if topup_rows >= 72 else "blocked")
    metrics_rows = [
        {"metric": "source_rows", "value": source_rows, "expected": 240, "pass": source_rows == 240},
        {"metric": "topup_rows", "value": topup_rows, "expected": ">=72 target 120", "pass": topup_rows >= 72},
        {"metric": "assembled_rows", "value": total_rows, "expected": f"{source_rows}+{topup_rows}", "pass": total_rows == source_rows + topup_rows},
        {"metric": "duplicate_sample_ids", "value": total_rows - len(set(strings(assembled["sample_ids"]))), "expected": 0, "pass": total_rows == len(set(strings(assembled["sample_ids"])))},
        {"metric": "train_ready_candidate", "value": train_ready, "expected": True, "pass": train_ready},
        {"metric": "split_counts", "value": json.dumps(counts(assembled, "split"), ensure_ascii=False, sort_keys=True), "expected": "v2 split preserved plus top-up split_hint", "pass": True},
        {"metric": "shape_counts", "value": json.dumps(counts(assembled, "shape_type"), ensure_ascii=False, sort_keys=True), "expected": "source plus hard-case top-up", "pass": True},
        {"metric": "burial_counts", "value": json.dumps(counts(assembled, "burial_depth_level"), ensure_ascii=False, sort_keys=True), "expected": "source plus hard-case top-up", "pass": True},
    ]
    write_csv(args.metrics, metrics_rows, ["metric", "value", "expected", "pass"])

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_role": "internal_defect_hardcase_augmented_pack",
        "status": status,
        "route": ROUTE,
        "stage": "22.2b",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "npz_path": str(args.output_npz),
        "manifest_path": str(args.manifest),
        "n_samples": total_rows,
        "source_rows": source_rows,
        "topup_rows": topup_rows,
        "split_counts": counts(assembled, "split"),
        "shape_counts": counts(assembled, "shape_type"),
        "burial_depth_counts": counts(assembled, "burial_depth_level"),
        "size_counts": counts(assembled, "size_level"),
        "aspect_counts": counts(assembled, "aspect_bin"),
        "hardcase_target_counts": counts(assembled, "hardcase_target_id"),
        "train_ready_candidate": train_ready,
        "baseline_ready": False,
        "internal_surface_mixed": False,
        "surface_rbc_baseline_update": False,
        "auto_discovery_allowed": False,
        "latest_newest_discovery_allowed": False,
        "allowed_use": ALLOWED_USE,
        "forbidden_use": FORBIDDEN_USE,
        "source_dataset_ids": [SOURCE_DATASET_ID, TOPUP_DATASET_ID],
        "source_manifest": str(args.source_manifest),
        "topup_npz_path": str(args.topup_npz),
        "assembly_script": "scripts/assemble_internal_defect_hard_case_augmented_pack.py",
        "npz_sha256": sha256_file(args.output_npz),
    }
    write_json(args.manifest, manifest)
    update_registry(args.registry, manifest)
    args.registry_summary.parent.mkdir(parents=True, exist_ok=True)
    args.registry_summary.write_text(
        "\n".join(
            [
                "22.2b internal defect hard-case registry summary",
                "",
                f"dataset_id: {DATASET_ID}",
                f"status: {status}",
                f"train_ready_candidate: {str(train_ready).lower()}",
                f"baseline_ready: false",
                f"manifest_path: {args.manifest}",
                f"npz_path_ignored: {args.output_npz}",
                "说明：v3_hardcase 是 internal branch 显式 training gate 候选，不是 CURRENT_BASELINE。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "\n".join(
            [
                "22.2b internal defect hard-case augmented pack assembly summary",
                "",
                f"source_dataset_id: {SOURCE_DATASET_ID}",
                f"topup_dataset_id: {TOPUP_DATASET_ID}",
                f"assembled_dataset_id: {DATASET_ID}",
                f"source_rows: {source_rows}",
                f"topup_rows: {topup_rows}",
                f"assembled_rows: {total_rows}",
                f"split_counts: {counts(assembled, 'split')}",
                f"shape_counts: {counts(assembled, 'shape_type')}",
                f"burial_counts: {counts(assembled, 'burial_depth_level')}",
                f"size_counts: {counts(assembled, 'size_level')}",
                f"aspect_counts: {counts(assembled, 'aspect_bin')}",
                f"train_ready_candidate: {str(train_ready).lower()}",
                "",
                "说明：v2_240 原 split 已保留，top-up 使用 22.2 plan split_hint；未训练、未运行 COMSOL、未更新 CURRENT_BASELINE。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if train_ready else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
