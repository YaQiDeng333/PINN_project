#!/usr/bin/env python
"""组装 21.3b internal defect v2_240 数据包。

输入来自显式 manifest/计划文件：v1 source NPZ + top-up NPZ + 21.3 plan。
本脚本不扫描 latest/newest，不训练，不运行 COMSOL，不修改 CURRENT_BASELINE.md。
生成的 assembled NPZ 写入 data/ ignored 路径，不应提交。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_internal_defect_pilot_pack_v2_240"
SOURCE_DATASET_ID = "comsol_internal_defect_pilot_pack_v1"
TOPUP_DATASET_ID = "comsol_internal_defect_dataset_topup_pack_v1"
SOURCE_MANIFEST = ROOT / "results/manifests/comsol_internal_defect_pilot_pack_v1.manifest.json"
PLAN_CSV = ROOT / "results/metrics/internal_defect_dataset_expansion_plan.csv"
TOPUP_NPZ = ROOT / "data/comsol_mfl/generated/internal_defect_dataset_topup_pack/internal_defect_dataset_topup_pack_v1.npz"
OUTPUT_NPZ = ROOT / "data/comsol_mfl/generated/internal_defect_pilot_pack_v2_240/comsol_internal_defect_pilot_pack_v2_240.npz"
SUMMARY = ROOT / "results/summaries/internal_defect_dataset_v2_assembly_summary.txt"
METRICS = ROOT / "results/metrics/internal_defect_dataset_v2_assembly_metrics.csv"

FINAL_SPLIT_TARGET = {"train": 160, "val": 40, "test": 40}
FINAL_SHAPE_SPLIT_TARGET = {
    "internal_sphere": {"train": 54, "val": 13, "test": 13},
    "internal_ellipsoid": {"train": 53, "val": 14, "test": 13},
    "internal_cuboid": {"train": 53, "val": 13, "test": 14},
}
FINAL_BURIAL_SPLIT_TARGET = {
    level: {"train": 40, "val": 10, "test": 10}
    for level in ["shallow", "medium", "deep", "deep_plus"]
}
FINAL_SIZE_SPLIT_TARGET = {
    "small": {"train": 54, "val": 13, "test": 13},
    "medium": {"train": 53, "val": 14, "test": 13},
    "large": {"train": 53, "val": 13, "test": 14},
}
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="组装 internal defect v2_240 pack。")
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument("--topup-npz", type=Path, default=TOPUP_NPZ)
    parser.add_argument("--output-npz", type=Path, default=OUTPUT_NPZ)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
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


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("拒绝覆盖已存在文件:\n" + "\n".join(str(path) for path in existing))


def source_npz_from_manifest(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_id = payload.get("dataset_id")
    if dataset_id != SOURCE_DATASET_ID:
        raise ValueError(f"source manifest dataset_id 不匹配: {dataset_id}")
    npz_path = Path(str(payload.get("npz_path", "")))
    if not npz_path.exists():
        raise FileNotFoundError(f"source NPZ 不存在: {npz_path}")
    return npz_path


def row_meta(arrays: dict[str, np.ndarray], index: int) -> dict[str, str]:
    return {
        "sample_id": str(arrays["sample_ids"][index]),
        "shape_type": str(arrays["shape_type"][index]),
        "burial_depth_level": str(arrays["burial_depth_level"][index]),
        "size_level": str(arrays["size_level"][index]),
        "aspect_bin": str(arrays["aspect_bin"][index]),
    }


def plan_by_id(plan_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["topup_sample_id"]: row for row in plan_rows if row.get("topup_sample_id")}


def select_topup_indices(topup: dict[str, np.ndarray], plan_rows: list[dict[str, str]]) -> tuple[list[int], list[str], list[str]]:
    sample_ids = strings(topup["sample_ids"])
    index_by_id = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    selected_plan = [row for row in plan_rows if row.get("topup_role") == "selected_quota"]
    buffer_plan = [row for row in plan_rows if row.get("topup_role") in {"buffer", "buffer_only"}]
    selected_ids = [row["topup_sample_id"] for row in selected_plan if row["topup_sample_id"] in index_by_id]
    missing_selected = [row["topup_sample_id"] for row in selected_plan if row["topup_sample_id"] not in index_by_id]
    if len(selected_ids) < 144:
        needed = 144 - len(selected_ids)
        buffer_ids = [row["topup_sample_id"] for row in buffer_plan if row["topup_sample_id"] in index_by_id]
        selected_ids.extend(buffer_ids[:needed])
    selected_ids = selected_ids[:144]
    return [index_by_id[sample_id] for sample_id in selected_ids], selected_ids, missing_selected


def decrement(target: dict[str, dict[str, int]], field_value: str, split: str) -> None:
    if field_value in target and split in target[field_value]:
        target[field_value][split] -= 1


def remaining_targets(topup: dict[str, np.ndarray], topup_indices: list[int], plan: dict[str, dict[str, str]]) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    split_target = dict(FINAL_SPLIT_TARGET)
    shape_target = json.loads(json.dumps(FINAL_SHAPE_SPLIT_TARGET))
    burial_target = json.loads(json.dumps(FINAL_BURIAL_SPLIT_TARGET))
    size_target = json.loads(json.dumps(FINAL_SIZE_SPLIT_TARGET))
    sample_ids = strings(topup["sample_ids"])
    for idx in topup_indices:
        sample_id = sample_ids[idx]
        row = plan.get(sample_id, {})
        split = row.get("v2_split_hint") or str(topup["split"][idx])
        if split not in SPLITS:
            continue
        split_target[split] -= 1
        decrement(shape_target, str(topup["shape_type"][idx]), split)
        decrement(burial_target, str(topup["burial_depth_level"][idx]), split)
        decrement(size_target, str(topup["size_level"][idx]), split)
    return split_target, shape_target, burial_target, size_target


def assign_source_splits(source: dict[str, np.ndarray], split_target: dict[str, int], shape_target: dict[str, dict[str, int]], burial_target: dict[str, dict[str, int]], size_target: dict[str, dict[str, int]]) -> list[str]:
    rows = [row_meta(source, idx) | {"index": idx} for idx in range(len(source["sample_ids"]))]
    rows.sort(key=lambda row: (row["shape_type"], row["burial_depth_level"], row["size_level"], row["aspect_bin"], row["sample_id"]))
    assigned = [""] * len(rows)
    for row in rows:
        best_split = None
        best_score = -10**9
        for split in SPLITS:
            if split_target.get(split, 0) <= 0:
                continue
            if shape_target.get(row["shape_type"], {}).get(split, 0) <= 0:
                continue
            if burial_target.get(row["burial_depth_level"], {}).get(split, 0) <= 0:
                continue
            if size_target.get(row["size_level"], {}).get(split, 0) <= 0:
                continue
            score = (
                split_target[split] * 100
                + shape_target[row["shape_type"]][split] * 10
                + burial_target[row["burial_depth_level"]][split] * 5
                + size_target[row["size_level"]][split]
            )
            if score > best_score:
                best_score = score
                best_split = split
        if best_split is None:
            raise RuntimeError(f"无法为 source row 分配 split: {row}")
        original_index = int(row["index"])
        assigned[original_index] = best_split
        split_target[best_split] -= 1
        shape_target[row["shape_type"]][best_split] -= 1
        burial_target[row["burial_depth_level"]][best_split] -= 1
        size_target[row["size_level"]][best_split] -= 1
    if any(value != 0 for value in split_target.values()):
        raise RuntimeError(f"source split target 未满足: {split_target}")
    return assigned


def topup_splits(topup: dict[str, np.ndarray], topup_indices: list[int], plan: dict[str, dict[str, str]]) -> list[str]:
    ids = strings(topup["sample_ids"])
    out: list[str] = []
    for idx in topup_indices:
        sample_id = ids[idx]
        split = plan.get(sample_id, {}).get("v2_split_hint") or str(topup["split"][idx])
        if split not in SPLITS:
            split = "train"
        out.append(split)
    return out


def assign_all_splits_milp(source: dict[str, np.ndarray], topup: dict[str, np.ndarray], topup_indices: list[int]) -> tuple[list[str], list[str]]:
    """对 source+selected top-up 重新分配 split。

    v1 原 split 无效，top-up split_hint 也只是计划提示；v2 的硬约束是
    全量 240 行满足 160/40/40，并且 shape/burial/size 在 split 内达到
    21.3 指定配额。这里用 scipy MILP 做确定性可行分配，避免手写贪心
    在交叉 strata 上过早消耗 quota。
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    rows: list[dict[str, Any]] = []
    for idx in range(len(source["sample_ids"])):
        rows.append(row_meta(source, idx) | {"origin": "source", "index": idx})
    for idx in topup_indices:
        rows.append(row_meta(topup, idx) | {"origin": "topup", "index": idx})
    n_rows = len(rows)
    n_vars = n_rows * len(SPLITS)
    constraints: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    def add_eq(indices: list[int], value: int) -> None:
        row = np.zeros(n_vars, dtype=float)
        row[indices] = 1.0
        constraints.append(row)
        lb.append(float(value))
        ub.append(float(value))

    def add_min(indices: list[int], value: int) -> None:
        row = np.zeros(n_vars, dtype=float)
        row[indices] = 1.0
        constraints.append(row)
        lb.append(float(value))
        ub.append(np.inf)

    for row_idx in range(n_rows):
        add_eq([row_idx * len(SPLITS) + split_idx for split_idx in range(len(SPLITS))], 1)
    for split_idx, split in enumerate(SPLITS):
        add_eq([row_idx * len(SPLITS) + split_idx for row_idx in range(n_rows)], FINAL_SPLIT_TARGET[split])
    for value, target in FINAL_SHAPE_SPLIT_TARGET.items():
        for split_idx, split in enumerate(SPLITS):
            add_eq([row_idx * len(SPLITS) + split_idx for row_idx, row in enumerate(rows) if row["shape_type"] == value], target[split])
    for value, target in FINAL_BURIAL_SPLIT_TARGET.items():
        for split_idx, split in enumerate(SPLITS):
            add_eq([row_idx * len(SPLITS) + split_idx for row_idx, row in enumerate(rows) if row["burial_depth_level"] == value], target[split])
    for value, target in FINAL_SIZE_SPLIT_TARGET.items():
        for split_idx, split in enumerate(SPLITS):
            add_eq([row_idx * len(SPLITS) + split_idx for row_idx, row in enumerate(rows) if row["size_level"] == value], target[split])
    for shape in ["internal_ellipsoid", "internal_cuboid"]:
        for aspect in ["compact", "elongated_x", "elongated_y"]:
            for split_idx, _split in enumerate(SPLITS):
                add_min(
                    [
                        row_idx * len(SPLITS) + split_idx
                        for row_idx, row in enumerate(rows)
                        if row["shape_type"] == shape and row["aspect_bin"] == aspect
                    ],
                    1,
                )

    objective = np.zeros(n_vars, dtype=float)
    result = milp(
        c=objective,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.vstack(constraints), np.asarray(lb), np.asarray(ub)),
        options={"time_limit": 60},
    )
    if not result.success:
        raise RuntimeError(f"v2 split MILP 不可行: {result.message}")
    x = np.rint(result.x).astype(int)
    source_splits = [""] * len(source["sample_ids"])
    selected_topup_splits = [""] * len(topup_indices)
    topup_position = {idx: pos for pos, idx in enumerate(topup_indices)}
    for row_idx, row in enumerate(rows):
        split_idx = int(np.argmax(x[row_idx * len(SPLITS) : (row_idx + 1) * len(SPLITS)]))
        split = SPLITS[split_idx]
        if row["origin"] == "source":
            source_splits[int(row["index"])] = split
        else:
            selected_topup_splits[topup_position[int(row["index"])]] = split
    return source_splits, selected_topup_splits


def assemble_arrays(source: dict[str, np.ndarray], topup: dict[str, np.ndarray], topup_indices: list[int], source_splits: list[str], selected_topup_splits: list[str]) -> dict[str, np.ndarray]:
    n_source = len(source["sample_ids"])
    assembled: dict[str, np.ndarray] = {
        "dataset_id": np.asarray(DATASET_ID, dtype=object),
        "source_dataset_ids": np.asarray([SOURCE_DATASET_ID, TOPUP_DATASET_ID], dtype=object),
    }
    for key, value in source.items():
        if key == "dataset_id":
            continue
        if value.shape[:1] == (n_source,) and key in topup:
            assembled[key] = np.concatenate([value, topup[key][topup_indices]], axis=0)
        else:
            assembled[key] = value
    assembled["split"] = np.asarray(source_splits + selected_topup_splits, dtype="<U16")
    assembled["row_origin"] = np.asarray(["source_v1"] * n_source + ["topup_v1"] * len(topup_indices), dtype="<U32")
    assembled["source_dataset_id_per_row"] = np.asarray(
        [SOURCE_DATASET_ID] * n_source + [TOPUP_DATASET_ID] * len(topup_indices),
        dtype="<U80",
    )
    return assembled


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def counts_by(arrays: dict[str, np.ndarray], field: str) -> dict[str, int]:
    return dict(Counter(strings(arrays[field])))


def split_cross_counts(arrays: dict[str, np.ndarray], field: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for split, value in zip(strings(arrays["split"]), strings(arrays[field]), strict=False):
        result[value][split] = result[value].get(split, 0) + 1
    return dict(result)


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.output_npz, args.summary, args.metrics], args.overwrite)
    source_npz = source_npz_from_manifest(args.source_manifest)
    plan_rows = read_csv(args.plan_csv)
    plan_lookup = plan_by_id(plan_rows)
    source = load_npz(source_npz)
    topup = load_npz(args.topup_npz)
    topup_indices, topup_ids, missing_selected = select_topup_indices(topup, plan_rows)
    source_splits, selected_topup_splits = assign_all_splits_milp(source, topup, topup_indices)
    assembled = assemble_arrays(source, topup, topup_indices, source_splits, selected_topup_splits)
    write_npz(args.output_npz, assembled)

    metrics_rows = [
        {"metric": "source_rows", "value": len(source["sample_ids"]), "expected": 96, "pass": len(source["sample_ids"]) == 96},
        {"metric": "topup_available_rows", "value": len(topup["sample_ids"]), "expected": ">=144", "pass": len(topup["sample_ids"]) >= 144},
        {"metric": "selected_topup_rows", "value": len(topup_indices), "expected": 144, "pass": len(topup_indices) == 144},
        {"metric": "assembled_rows", "value": len(assembled["sample_ids"]), "expected": 240, "pass": len(assembled["sample_ids"]) == 240},
        {"metric": "missing_selected_topup_rows", "value": len(missing_selected), "expected": 0, "pass": len(missing_selected) == 0},
        {"metric": "split_counts", "value": json.dumps(counts_by(assembled, "split"), ensure_ascii=False, sort_keys=True), "expected": json.dumps(FINAL_SPLIT_TARGET, ensure_ascii=False, sort_keys=True), "pass": counts_by(assembled, "split") == FINAL_SPLIT_TARGET},
        {"metric": "shape_counts", "value": json.dumps(counts_by(assembled, "shape_type"), ensure_ascii=False, sort_keys=True), "expected": "80 each", "pass": set(counts_by(assembled, "shape_type").values()) == {80}},
        {"metric": "burial_counts", "value": json.dumps(counts_by(assembled, "burial_depth_level"), ensure_ascii=False, sort_keys=True), "expected": "60 each", "pass": set(counts_by(assembled, "burial_depth_level").values()) == {60}},
        {"metric": "size_counts", "value": json.dumps(counts_by(assembled, "size_level"), ensure_ascii=False, sort_keys=True), "expected": "80 each", "pass": set(counts_by(assembled, "size_level").values()) == {80}},
    ]
    write_csv(args.metrics, metrics_rows, ["metric", "value", "expected", "pass"])
    lines = [
        "21.3b internal defect v2_240 assembly summary",
        f"source_dataset_id: {SOURCE_DATASET_ID}",
        f"topup_dataset_id: {TOPUP_DATASET_ID}",
        f"assembled_dataset_id: {DATASET_ID}",
        f"source_npz: {source_npz}",
        f"topup_npz: {args.topup_npz}",
        f"assembled_npz: {args.output_npz}",
        f"source_rows: {len(source['sample_ids'])}",
        f"selected_topup_rows: {len(topup_indices)}",
        f"assembled_rows: {len(assembled['sample_ids'])}",
        f"split_counts: {counts_by(assembled, 'split')}",
        f"shape_counts: {counts_by(assembled, 'shape_type')}",
        f"burial_counts: {counts_by(assembled, 'burial_depth_level')}",
        f"size_counts: {counts_by(assembled, 'size_level')}",
        f"aspect_counts: {counts_by(assembled, 'aspect_bin')}",
        f"missing_selected_topup_rows: {len(missing_selected)}",
        "",
        "说明：v1 原 split 已废弃，v2 split 由 21.3 deterministic quota 重新分配；本脚本未训练、未运行 COMSOL、未更新 CURRENT_BASELINE。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
