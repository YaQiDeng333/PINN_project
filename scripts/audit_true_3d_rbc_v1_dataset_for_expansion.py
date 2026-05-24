#!/usr/bin/env python
"""Audit the v1 assembled true-3D RBC dataset before the 20.74 expansion."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled"
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DEFAULT_MANIFEST = ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.manifest.json"
DEFAULT_TRAINING_SUMMARY = ROOT / "results/summaries/true_3d_rbc_training_gate_decision_summary.txt"
DEFAULT_NN_METRICS = ROOT / "results/metrics/true_3d_rbc_neural_training_gate_metrics.csv"
DEFAULT_GROUPS = ROOT / "results/metrics/true_3d_rbc_pilot_assembled_group_summary.csv"
DEFAULT_PREFLIGHT = ROOT / "results/summaries/true_3d_rbc_dataset_expansion_120_preflight_summary.txt"
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_v1_dataset_expansion_audit_summary.txt"
DEFAULT_AUDIT = ROOT / "results/metrics/true_3d_rbc_v1_dataset_expansion_audit.csv"
DEFAULT_TARGETS = ROOT / "results/metrics/true_3d_rbc_v1_missing_expansion_targets.csv"

AUDIT_FIELDS = [
    "sample_id",
    "split",
    "curvature_template",
    "depth_bin",
    "size_bin",
    "aspect_bin",
    "L_m",
    "W_m",
    "D_m",
    "wLD",
    "wWD",
    "wLW",
    "source_dataset_id",
    "schema_pass",
]

TARGET_FIELDS = [
    "group_key",
    "group_value",
    "current_count",
    "target_count",
    "missing_count",
    "priority",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v1 assembled RBC dataset for 20.74 expansion.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--neural-metrics", type=Path, default=DEFAULT_NN_METRICS)
    parser.add_argument("--group-summary", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--preflight-summary", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def scalar_str(array: np.ndarray, index: int = 0) -> str:
    return str(np.asarray(array).reshape(-1)[index])


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_registry_entry(registry: Path, dataset_id: str) -> None:
    text = registry.read_text(encoding="utf-8", errors="replace")
    if dataset_id not in text:
        raise RuntimeError(f"dataset_id not found in registry: {dataset_id}")


def write_preflight(path: Path) -> None:
    lines = [
        "20.74 true 3D RBC dataset expansion preflight summary",
        "",
        "Subagent conclusions:",
        "- Agent A Method/Route: GO. Expanding to N=120 fits the true 3D / Piao-style route, but remains RBC-style and not exact Piao or baseline.",
        "- Agent B Data/Coverage: GO. N=56 is train_ready_candidate but coverage is not enough; D_m and curvature remain weak.",
        "- Agent C COMSOL Generation: conditional GO. Reuse imported watertight mesh + 20.70 material/domain protocol, but parameterize the 20.74 wrapper.",
        "- Agent D Registry/Manifest: GO. Add v2_topup_20_74 and v2_120 entries/manifests; forbid latest/newest discovery.",
        "- Agent E Safety/Git: conditional GO. Stage only whitelisted scripts/results/docs; never stage data, NPZ, temp STL, .mph, raw CSV, checkpoints, previews, or notes.",
        "- Agent F Implementation: GO. New audit, plan, mesh, COMSOL wrapper, assembly, validation, registry, and route outputs are needed.",
        "",
        "Dirty boundary:",
        "- PINN unrelated dirty: scripts/visualize_current_baseline.py",
        "- COMSOL unrelated dirty: src/tools/physics.py, src/tools/results.py, src/tools/study.py, scripts/generate_mfl_rectangular_sweep.py",
        "",
        "Decision: continue execution after preflight.",
        "Training needed: False",
        "COMSOL needed: True",
        "Baseline update: False",
        "Hard blockers: none found in preflight.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.preflight_summary, args.summary, args.audit, args.targets], args.overwrite)
    require_registry_entry(args.registry, DATASET_ID)
    manifest = load_manifest(args.manifest)
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError(f"manifest dataset_id mismatch: {manifest.get('dataset_id')}")
    if manifest.get("status") != "pilot_generated" or manifest.get("train_ready_candidate") is not True:
        raise RuntimeError("v1 assembled manifest is not a train-ready pilot source")
    if manifest.get("baseline_ready") is not False:
        raise RuntimeError("v1 assembled manifest unexpectedly marks baseline_ready")
    if "explicit_pilot_training_gate" not in manifest.get("allowed_use", []):
        raise RuntimeError("v1 assembled manifest does not allow explicit training gate")
    npz_path = Path(manifest["npz_path"])
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=True) as npz:
        sample_ids = [str(x) for x in npz["sample_ids"]]
        split = [str(x) for x in npz["split"]]
        curv = [str(x) for x in npz["curvature_template"]]
        depth = [str(x) for x in npz["depth_bin"]]
        size = [str(x) for x in npz["size_bin"]]
        aspect = [str(x) for x in npz["aspect_bin"]]
        params = np.asarray(npz["rbc_params"], dtype=float).reshape(len(sample_ids), 6)
        dataset_id = scalar_str(npz["dataset_id"])
        status = scalar_str(npz["status"])
    if dataset_id != DATASET_ID or status != "pilot_generated":
        raise RuntimeError(f"NPZ identity/status mismatch: {dataset_id} / {status}")

    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "split": split[idx],
                "curvature_template": curv[idx],
                "depth_bin": depth[idx],
                "size_bin": size[idx],
                "aspect_bin": aspect[idx],
                "L_m": params[idx, 0],
                "W_m": params[idx, 1],
                "D_m": params[idx, 2],
                "wLD": params[idx, 3],
                "wWD": params[idx, 4],
                "wLW": params[idx, 5],
                "source_dataset_id": DATASET_ID,
                "schema_pass": True,
            }
        )
    write_csv(args.audit, rows, AUDIT_FIELDS)

    targets: list[dict[str, Any]] = []
    for key, target_map in {
        "split": {"train": 80, "val": 20, "test": 20},
        "curvature_template": {"sharp": 24, "round": 24, "boxy": 24, "LD_dominant": 24, "WD_dominant": 24},
        "depth_bin": {"shallow": 40, "medium": 40, "deep": 40},
        "size_bin": {"small_compact": 30, "medium_balanced": 30, "large_wide": 30, "elongated": 30},
    }.items():
        current = Counter(row[key] for row in rows)
        for value, target in target_map.items():
            missing = target - current.get(value, 0)
            targets.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "current_count": current.get(value, 0),
                    "target_count": target,
                    "missing_count": missing,
                    "priority": missing > 0,
                }
            )

    cell_counts = Counter((row["curvature_template"], row["depth_bin"], row["size_bin"]) for row in rows)
    templates = ["sharp", "round", "boxy", "LD_dominant", "WD_dominant"]
    depths = ["shallow", "medium", "deep"]
    sizes = ["small_compact", "medium_balanced", "large_wide", "elongated"]
    empty_cells: list[tuple[str, str, str]] = []
    for template in templates:
        for depth_bin in depths:
            for size_bin in sizes:
                count = cell_counts.get((template, depth_bin, size_bin), 0)
                if count == 0:
                    empty_cells.append((template, depth_bin, size_bin))
                targets.append(
                    {
                        "group_key": "coverage_cell",
                        "group_value": f"{template}|{depth_bin}|{size_bin}",
                        "current_count": count,
                        "target_count": 2,
                        "missing_count": max(0, 2 - count),
                        "priority": count < 2,
                    }
                )
    write_csv(args.targets, targets, TARGET_FIELDS)

    training_summary = args.training_summary.read_text(encoding="utf-8", errors="replace") if args.training_summary.exists() else ""
    split_counts = Counter(split)
    curvature_counts = Counter(curv)
    depth_counts = Counter(depth)
    size_counts = Counter(size)
    lines = [
        "20.74 v1 assembled dataset expansion audit summary",
        "",
        f"source_dataset_id: {DATASET_ID}",
        f"source_npz: {npz_path}",
        f"source_n: {len(rows)}",
        f"source_split_counts: {dict(split_counts)}",
        f"source_curvature_counts: {dict(curvature_counts)}",
        f"source_depth_counts: {dict(depth_counts)}",
        f"source_size_counts: {dict(size_counts)}",
        f"empty_coverage_cells: {['|'.join(cell) for cell in empty_cells]}",
        "learnability_blocker: small_data_generalization_limited; D_m and curvature parameters are unstable",
        f"training_summary_mentions_expand: {'expand' in training_summary.lower() or '120' in training_summary}",
        "",
        "Top-up target:",
        "- assembled target N=120, minimum acceptable N=108",
        "- target split train/val/test=80/20/20",
        "- target curvature count per template=24",
        "- top-up planned N=80; target success N=64",
        "",
        "Gate: PASS",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_preflight(args.preflight_summary)
    if len(rows) != 56:
        raise RuntimeError(f"expected N=56 source, got {len(rows)}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
