"""Create center-anchored polygon inverse subset packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _usage() -> str:
    return (
        "Usage: python make_comsol_center_anchored_polygon_subset_package.py "
        "--npz-path INPUT.npz --targets center_anchored_polygon_targets.npz "
        "--indices 0,1 --output-npz subset.npz --output-targets subset_targets.npz"
    )


def parse_indices(value: str, sample_count: int) -> np.ndarray:
    if not value:
        raise ValueError("indices must be non-empty.")
    indices = np.asarray([int(item.strip()) for item in value.split(",") if item.strip()], dtype=np.int64)
    if indices.size == 0:
        raise ValueError("indices must be non-empty.")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError(f"indices contain duplicates: {indices.tolist()}")
    if np.any(indices < 0) or np.any(indices >= sample_count):
        raise ValueError(f"indices {indices.tolist()} outside sample count {sample_count}.")
    return indices


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _subset_by_first_dim(data: dict, indices: np.ndarray, sample_count: int) -> dict:
    out = {}
    for key, value in data.items():
        if getattr(value, "ndim", 0) > 0 and value.shape[0] == sample_count:
            out[key] = value[indices]
        else:
            out[key] = value
    return out


def create_subset(
    npz_path: Path,
    targets_path: Path,
    indices: np.ndarray,
    output_npz: Path,
    output_targets: Path,
) -> None:
    npz_data = _load_npz(npz_path)
    target_data = _load_npz(targets_path)
    sample_count = int(npz_data["signals"].shape[0])
    if target_data["presence_targets"].shape[0] != sample_count:
        raise ValueError("NPZ signals and center-anchored targets sample counts do not match.")
    subset_npz = _subset_by_first_dim(npz_data, indices, sample_count)
    subset_targets = _subset_by_first_dim(target_data, indices, sample_count)
    mapping = {int(new): int(old) for new, old in enumerate(indices.tolist())}
    subset_targets["sample_indices"] = np.arange(len(indices), dtype=np.int64)
    subset_npz["subset_source_indices_json"] = np.array(json.dumps(mapping, sort_keys=True))
    subset_targets["subset_source_indices_json"] = subset_npz["subset_source_indices_json"]
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_targets.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **subset_npz)
    np.savez_compressed(output_targets, **subset_targets)
    print(f"Saved subset NPZ to {output_npz}")
    print(f"Saved subset center-anchored targets to {output_targets}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path")
    parser.add_argument("--targets")
    parser.add_argument("--indices")
    parser.add_argument("--output-npz")
    parser.add_argument("--output-targets")
    args = parser.parse_args(argv)
    if not args.npz_path or not args.targets or not args.indices or not args.output_npz or not args.output_targets:
        print(_usage())
        return 0
    with np.load(args.npz_path, allow_pickle=True) as data:
        sample_count = int(data["signals"].shape[0])
    indices = parse_indices(args.indices, sample_count)
    create_subset(Path(args.npz_path), Path(args.targets), indices, Path(args.output_npz), Path(args.output_targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
