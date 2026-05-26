"""Create tiny COMSOL parametric NPZ/target subset packages without changing the runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NPZ_SAMPLE_KEYS = {
    "signals",
    "masks",
    "mu_maps",
    "defect_params",
    "source_sample_ids",
    "source_global_indices",
    "csv_sample_indices",
}
TARGET_SAMPLE_KEYS = {
    "continuous_targets",
    "continuous_targets_raw",
    "continuous_targets_unscaled",
    "presence_targets",
    "type_targets",
    "component_counts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create COMSOL parametric tiny-overfit subset packages.")
    parser.add_argument("--npz-path", help="Input converted NPZ path.")
    parser.add_argument("--targets-path", help="Input parametric_targets.npz path.")
    parser.add_argument("--indices", help="Comma-separated split-local sample indices, e.g. 0,1,2.")
    parser.add_argument("--output-npz", help="Output subset NPZ path.")
    parser.add_argument("--output-targets", help="Output subset parametric targets NPZ path.")
    return parser.parse_args()


def _usage_and_exit() -> int:
    print(
        "Usage: python make_comsol_parametric_subset_package.py "
        "--npz-path INPUT.npz --targets-path parametric_targets.npz --indices 0,1 "
        "--output-npz subset.npz --output-targets subset_targets.npz"
    )
    return 0


def parse_indices(value: str, sample_count: int) -> np.ndarray:
    if not value:
        raise ValueError("indices must be non-empty.")
    indices = np.array([int(item.strip()) for item in value.split(",") if item.strip()], dtype=np.int64)
    if indices.size == 0:
        raise ValueError("indices must be non-empty.")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError(f"indices contain duplicates: {indices.tolist()}")
    if np.any(indices < 0) or np.any(indices >= sample_count):
        raise ValueError(f"indices {indices.tolist()} outside sample count {sample_count}")
    return indices


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _subset_npz(data: dict, indices: np.ndarray) -> dict:
    out = {}
    sample_count = int(data["signals"].shape[0])
    for key, value in data.items():
        if key in NPZ_SAMPLE_KEYS and getattr(value, "ndim", 0) > 0 and value.shape[0] == sample_count:
            out[key] = value[indices]
        else:
            out[key] = value
    out["csv_sample_indices"] = np.arange(len(indices), dtype=np.int64)
    original = {int(new): int(old) for new, old in enumerate(indices.tolist())}
    out["subset_source_indices_json"] = np.array(json.dumps(original, sort_keys=True))
    return out


def _subset_targets(data: dict, indices: np.ndarray) -> dict:
    out = {}
    sample_count = int(data["continuous_targets"].shape[0])
    for key, value in data.items():
        if key in TARGET_SAMPLE_KEYS and getattr(value, "ndim", 0) > 0 and value.shape[0] == sample_count:
            out[key] = value[indices]
        elif key == "sample_indices":
            out[key] = np.arange(len(indices), dtype=np.int64)
        else:
            out[key] = value
    out["subset_source_indices_json"] = np.array(
        json.dumps({int(new): int(old) for new, old in enumerate(indices.tolist())}, sort_keys=True)
    )
    return out


def create_subset(npz_path: Path, targets_path: Path, indices: np.ndarray, output_npz: Path, output_targets: Path) -> None:
    npz_data = _load_npz(npz_path)
    targets_data = _load_npz(targets_path)
    if npz_data["signals"].shape[0] != targets_data["continuous_targets"].shape[0]:
        raise ValueError("NPZ signals and targets sample counts do not match.")
    subset_npz = _subset_npz(npz_data, indices)
    subset_targets = _subset_targets(targets_data, indices)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_targets.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **subset_npz)
    np.savez_compressed(output_targets, **subset_targets)
    print(f"Saved subset NPZ to {output_npz}")
    print(f"Saved subset targets to {output_targets}")


def main() -> int:
    args = parse_args()
    if not args.npz_path or not args.targets_path or not args.indices or not args.output_npz or not args.output_targets:
        return _usage_and_exit()
    with np.load(args.npz_path, allow_pickle=True) as data:
        sample_count = int(data["signals"].shape[0])
    indices = parse_indices(args.indices, sample_count)
    create_subset(Path(args.npz_path), Path(args.targets_path), indices, Path(args.output_npz), Path(args.output_targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
