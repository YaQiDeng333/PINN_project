"""Build a matched-coverage resplit for the COMSOL V3 polygon pack."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
TARGET_COUNTS = {
    "train": {
        "x_bin_wrong_like": 10,
        "both_bins_wrong_like": 5,
        "bins_correct_center_or_offset_bad": 7,
        "geometry_or_type_interaction": 5,
        "rare_y_bin_wrong": 3,
    },
    "val": {
        "x_bin_wrong_like": 3,
        "both_bins_wrong_like": 2,
        "bins_correct_center_or_offset_bad": 2,
        "geometry_or_type_interaction": 2,
        "rare_y_bin_wrong": 1,
    },
    "test": {
        "x_bin_wrong_like": 3,
        "both_bins_wrong_like": 2,
        "bins_correct_center_or_offset_bad": 2,
        "geometry_or_type_interaction": 2,
        "rare_y_bin_wrong": 1,
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: str | bool | int | float | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0", ""}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return False


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _sample_npz_name(split: str) -> str:
    return f"{split}_comsol_v3_polygon_hard_case.npz"


def _distance_to_train(bin_pair: tuple[int, int], train_pairs: set[tuple[int, int]]) -> int:
    if not train_pairs:
        return 999
    return min(abs(bin_pair[0] - x) + abs(bin_pair[1] - y) for x, y in train_pairs)


def _record_bin_pairs(targets: dict[str, np.ndarray], sample_index: int) -> list[tuple[int, int]]:
    presence = targets["presence_targets"][sample_index] > 0.5
    x_bins = targets["center_x_bin_targets"][sample_index]
    y_bins = targets["center_y_bin_targets"][sample_index]
    return [(int(x_bins[slot]), int(y_bins[slot])) for slot, present in enumerate(presence) if bool(present)]


def _collect_records(converted_dir: Path, target_root: Path, raw_root: Path) -> tuple[list[dict], dict, dict]:
    records: list[dict] = []
    npz_by_split: dict[str, dict[str, np.ndarray]] = {}
    targets_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        npz_by_split[split] = _load_npz(converted_dir / _sample_npz_name(split))
        targets = _load_npz(target_root / split / "center_anchored_polygon_targets.npz")
        targets_by_split[split] = targets
        defect_rows = {int(row["sample_index"]): row for row in _read_csv(raw_root / split / "defect_params.csv")}
        polygon_rows = _read_csv(raw_root / split / "polygon_params.csv")
        component_counts = Counter(int(row["sample_index"]) for row in polygon_rows if _as_bool(row.get("presence", "1")))
        sample_count = int(npz_by_split[split]["signals"].shape[0])
        for sample_index in range(sample_count):
            defect = defect_rows[sample_index]
            bin_pairs = _record_bin_pairs(targets, sample_index)
            records.append(
                {
                    "global_id": f"{split}_{sample_index:03d}",
                    "old_split": split,
                    "old_sample_index": sample_index,
                    "hard_case_type": defect["hard_case_type"],
                    "true_rotated": _as_bool(defect.get("true_rotated_geometry", "false")),
                    "true_multi_component": _as_bool(defect.get("true_multi_component_geometry", "false")),
                    "component_count": int(component_counts[sample_index]),
                    "bin_pairs": bin_pairs,
                    "center_bin_signature": ";".join(f"{x}:{y}" for x, y in bin_pairs),
                }
            )
    return records, npz_by_split, targets_by_split


def _score_assignment(assignments: dict[str, list[dict]]) -> tuple:
    train_pairs = {pair for row in assignments["train"] for pair in row["bin_pairs"]}
    heldout_rows = assignments["val"] + assignments["test"]
    uncovered_component_count = 0
    uncovered_sample_count = 0
    exact_uncovered_component_count = 0
    total_distance = 0
    max_sample_distance = 0
    for row in heldout_rows:
        distances = [_distance_to_train(pair, train_pairs) for pair in row["bin_pairs"]]
        exact_uncovered_component_count += sum(1 for value in distances if value > 0)
        uncovered_component_count += sum(1 for value in distances if value > 1)
        if any(value > 1 for value in distances):
            uncovered_sample_count += 1
        sample_distance = max(distances) if distances else 0
        total_distance += sum(distances)
        max_sample_distance = max(max_sample_distance, sample_distance)
    flag_penalty = 0
    for split in SPLITS:
        rows = assignments[split]
        if not any(row["true_rotated"] for row in rows):
            flag_penalty += 1
        if not any(row["true_multi_component"] for row in rows):
            flag_penalty += 1
    return (
        uncovered_component_count,
        uncovered_sample_count,
        total_distance,
        max_sample_distance,
        exact_uncovered_component_count,
        flag_penalty,
    )


def _random_assignment(records: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_type[row["hard_case_type"]].append(row)
    assignments = {split: [] for split in SPLITS}
    for hard_case, rows in by_type.items():
        shuffled = list(rows)
        rng.shuffle(shuffled)
        cursor = 0
        for split in SPLITS:
            count = TARGET_COUNTS[split][hard_case]
            assignments[split].extend(shuffled[cursor : cursor + count])
            cursor += count
        if cursor != len(shuffled):
            raise ValueError(f"Unused records for {hard_case}: {len(shuffled) - cursor}")
    for split in SPLITS:
        assignments[split].sort(key=lambda row: (row["hard_case_type"], row["global_id"]))
    return assignments


def choose_resplit(records: list[dict], seed: int, iterations: int) -> tuple[dict[str, list[dict]], tuple]:
    rng = random.Random(seed)
    best_assignment: dict[str, list[dict]] | None = None
    best_score: tuple | None = None
    for _ in range(iterations):
        candidate = _random_assignment(records, rng)
        score = _score_assignment(candidate)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = candidate
            if score[0] == 0 and score[1] == 0 and score[5] == 0:
                break
    if best_assignment is None or best_score is None:
        raise RuntimeError("Failed to build matched-coverage resplit.")
    return best_assignment, best_score


def _slice_first_dim(data_by_split: dict[str, dict[str, np.ndarray]], selected: list[dict], sample_key: str) -> dict[str, np.ndarray]:
    first_split = selected[0]["old_split"]
    first_data = data_by_split[first_split]
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    output: dict[str, np.ndarray] = {}
    for row in selected:
        data = data_by_split[row["old_split"]]
        sample_count = int(data[sample_key].shape[0])
        idx = int(row["old_sample_index"])
        for key, value in data.items():
            if getattr(value, "ndim", 0) > 0 and value.shape[0] == sample_count:
                chunks[key].append(value[idx : idx + 1])
            elif key not in output:
                output[key] = value
    for key, values in chunks.items():
        output[key] = np.concatenate(values, axis=0)
    return output


def _write_split_npz(
    split: str,
    selected: list[dict],
    npz_by_split: dict[str, dict[str, np.ndarray]],
    targets_by_split: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    npz_out = _slice_first_dim(npz_by_split, selected, "signals")
    target_out = _slice_first_dim(targets_by_split, selected, "presence_targets")
    source_ids = np.asarray([row["global_id"] for row in selected], dtype="U32")
    old_splits = np.asarray([row["old_split"] for row in selected], dtype="U16")
    old_indices = np.asarray([row["old_sample_index"] for row in selected], dtype=np.int64)
    new_indices = np.arange(len(selected), dtype=np.int64)
    for out in (npz_out, target_out):
        out["sample_indices"] = new_indices
        out["matched_coverage_source_ids"] = source_ids
        out["matched_coverage_old_splits"] = old_splits
        out["matched_coverage_old_sample_indices"] = old_indices
        out["matched_coverage_json"] = np.array(
            json.dumps({int(i): row["global_id"] for i, row in enumerate(selected)}, sort_keys=True),
            dtype="U2048",
        )
    np.savez_compressed(split_dir / "comsol_v3_polygon_matched_coverage.npz", **npz_out)
    np.savez_compressed(split_dir / "center_anchored_polygon_targets.npz", **target_out)


def _write_reindexed_raw(
    split: str,
    selected: list[dict],
    raw_root: Path,
    output_dir: Path,
) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    defect_by_key = {}
    polygon_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for old_split in SPLITS:
        for row in _read_csv(raw_root / old_split / "defect_params.csv"):
            defect_by_key[(old_split, int(row["sample_index"]))] = row
        for row in _read_csv(raw_root / old_split / "polygon_params.csv"):
            polygon_by_key[(old_split, int(row["sample_index"]))].append(row)
    defect_rows = []
    polygon_rows = []
    for new_index, row in enumerate(selected):
        key = (row["old_split"], int(row["old_sample_index"]))
        defect = dict(defect_by_key[key])
        defect["sample_index"] = new_index
        defect["split"] = split
        defect["matched_coverage_old_split"] = row["old_split"]
        defect["matched_coverage_old_sample_index"] = row["old_sample_index"]
        defect["matched_coverage_source_id"] = row["global_id"]
        defect_rows.append(defect)
        for poly in polygon_by_key[key]:
            item = dict(poly)
            item["sample_index"] = new_index
            item["split"] = split
            item["matched_coverage_old_split"] = row["old_split"]
            item["matched_coverage_old_sample_index"] = row["old_sample_index"]
            item["matched_coverage_source_id"] = row["global_id"]
            polygon_rows.append(item)
    _write_csv(split_dir / "defect_params.csv", defect_rows)
    _write_csv(split_dir / "polygon_params.csv", polygon_rows)


def _coverage_rows(assignments: dict[str, list[dict]]) -> list[dict]:
    train_pairs = {pair for row in assignments["train"] for pair in row["bin_pairs"]}
    rows = []
    for split in SPLITS:
        for new_index, row in enumerate(assignments[split]):
            distances = [_distance_to_train(pair, train_pairs) for pair in row["bin_pairs"]]
            rows.append(
                {
                    "new_split": split,
                    "new_sample_index": new_index,
                    "source_id": row["global_id"],
                    "old_split": row["old_split"],
                    "old_sample_index": row["old_sample_index"],
                    "hard_case_type": row["hard_case_type"],
                    "center_bin_signature": row["center_bin_signature"],
                    "exactly_covered_component_count": sum(1 for value in distances if value == 0),
                    "within_distance1_component_count": sum(1 for value in distances if value <= 1),
                    "uncovered_distance_gt1_component_count": sum(1 for value in distances if value > 1),
                    "max_center_bin_distance_to_train": max(distances) if distances else 0,
                    "mean_center_bin_distance_to_train": float(sum(distances) / len(distances)) if distances else 0.0,
                    "all_bins_exactly_covered": all(value == 0 for value in distances),
                    "all_bins_within_distance1": all(value <= 1 for value in distances),
                    "true_rotated": row["true_rotated"],
                    "true_multi_component": row["true_multi_component"],
                    "component_count": row["component_count"],
                }
            )
    return rows


def _manifest_rows(assignments: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for split in SPLITS:
        for new_index, row in enumerate(assignments[split]):
            rows.append(
                {
                    "new_split": split,
                    "new_sample_index": new_index,
                    "source_id": row["global_id"],
                    "old_split": row["old_split"],
                    "old_sample_index": row["old_sample_index"],
                    "hard_case_type": row["hard_case_type"],
                    "center_bin_signature": row["center_bin_signature"],
                    "true_rotated": row["true_rotated"],
                    "true_multi_component": row["true_multi_component"],
                    "component_count": row["component_count"],
                }
            )
    return rows


def _counts(rows: list[dict]) -> Counter:
    return Counter(row["hard_case_type"] for row in rows)


def _write_summary(output_dir: Path, assignments: dict[str, list[dict]], score: tuple, coverage: list[dict]) -> None:
    lines = [
        "# S299 COMSOL polygon matched-coverage resplit",
        "",
        "This resplit uses the existing S254-S258 polygon V3 pack only. It does not generate COMSOL data and does not overwrite the original train/val/test split.",
        "",
        "## Split Counts",
        "",
    ]
    for split in SPLITS:
        rows = assignments[split]
        counts = _counts(rows)
        lines.append(
            f"- {split}: samples `{len(rows)}`, hard_case counts "
            + ", ".join(f"{name}={counts.get(name, 0)}" for name in TARGET_COUNTS[split])
            + f", rotated `{sum(1 for row in rows if row['true_rotated'])}`, multi-component `{sum(1 for row in rows if row['true_multi_component'])}`."
        )
    heldout = [row for row in coverage if row["new_split"] in {"val", "test"}]
    exact = sum(1 for row in heldout if row["all_bins_exactly_covered"])
    within = sum(1 for row in heldout if row["all_bins_within_distance1"])
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Search score: `{score}`.",
            f"- Held-out samples with all component bins exactly covered by train: `{exact}` / `{len(heldout)}`.",
            f"- Held-out samples with all component bins within train distance <= 1: `{within}` / `{len(heldout)}`.",
            "- Coverage is prioritized over exact split provenance; hard-case counts remain at the requested targets.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_resplit(args: argparse.Namespace) -> None:
    converted_dir = Path(args.converted_dir)
    target_root = Path(args.target_root)
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, npz_by_split, targets_by_split = _collect_records(converted_dir, target_root, raw_root)
    assignments, score = choose_resplit(records, args.seed, args.iterations)
    selected_ids = [row["global_id"] for split in SPLITS for row in assignments[split]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Resplit selected duplicate source samples.")
    for split in SPLITS:
        _write_split_npz(split, assignments[split], npz_by_split, targets_by_split, output_dir)
        _write_reindexed_raw(split, assignments[split], raw_root, output_dir)
    manifest = _manifest_rows(assignments)
    coverage = _coverage_rows(assignments)
    _write_csv(output_dir / "split_manifest.csv", manifest)
    _write_csv(output_dir / "coverage_report.csv", coverage)
    _write_summary(output_dir, assignments, score, coverage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-dir", default="experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/converted")
    parser.add_argument("--target-root", default="experiments/dual_network/S290_comsol_v3_center_anchored_polygon_targets")
    parser.add_argument("--raw-root", default="experiments/dual_network/S254_comsol_v3_polygon_hard_case_ingest/raw")
    parser.add_argument("--output-dir", default="experiments/dual_network/S299_comsol_polygon_matched_coverage_resplit")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50000)
    args = parser.parse_args(argv)
    build_resplit(args)
    print(f"Saved matched-coverage resplit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
