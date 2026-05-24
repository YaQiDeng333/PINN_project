#!/usr/bin/env python
"""Validate COMSOL data registry entries for explicit dataset loading."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "COMSOL_DATA_REGISTRY.md"
DEFAULT_MANIFESTS = [
    ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1.manifest.json",
    ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1_topup.manifest.json",
    ROOT / "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.manifest.json",
]
DEFAULT_SUMMARY = ROOT / "results/summaries/true_3d_rbc_pilot_registry_validation_summary.txt"
DEFAULT_METRICS = ROOT / "results/metrics/true_3d_rbc_pilot_registry_validation.csv"

FIELDS = ["dataset_id", "check_name", "pass", "observed", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate COMSOL data registry / manifest governance.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_no_overwrite(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing files:\n" + "\n".join(str(path) for path in existing))


def run(args: argparse.Namespace) -> int:
    check_no_overwrite([args.summary, args.metrics], args.overwrite)
    manifest_paths = args.manifest or DEFAULT_MANIFESTS
    registry_text = args.registry.read_text(encoding="utf-8", errors="replace")
    manifests = []
    for path in manifest_paths:
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["_path"] = str(path)
            manifests.append(manifest)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        dataset_id = manifest["dataset_id"]
        duplicate = dataset_id in seen
        seen.add(dataset_id)
        rows.extend(
            [
                {
                    "dataset_id": dataset_id,
                    "check_name": "dataset_id_unique",
                    "pass": not duplicate,
                    "observed": dataset_id,
                    "notes": "",
                },
                {
                    "dataset_id": dataset_id,
                    "check_name": "manifest_referenced_in_registry",
                    "pass": dataset_id in registry_text,
                    "observed": manifest["_path"],
                    "notes": "",
                },
                {
                    "dataset_id": dataset_id,
                    "check_name": "auto_discovery_forbidden",
                    "pass": manifest.get("auto_discovery_allowed") is False
                    and manifest.get("latest_newest_discovery_allowed") is False,
                    "observed": f"auto={manifest.get('auto_discovery_allowed')}; latest={manifest.get('latest_newest_discovery_allowed')}",
                    "notes": "",
                },
                {
                    "dataset_id": dataset_id,
                    "check_name": "baseline_ready_false",
                    "pass": manifest.get("baseline_ready") is False,
                    "observed": f"baseline_ready={manifest.get('baseline_ready')}",
                    "notes": "",
                },
                {
                    "dataset_id": dataset_id,
                    "check_name": "use_guards_present",
                    "pass": bool(manifest.get("allowed_use")) and bool(manifest.get("forbidden_use")),
                    "observed": f"allowed={manifest.get('allowed_use')}; forbidden={manifest.get('forbidden_use')}",
                    "notes": "",
                },
            ]
        )
    write_csv(args.metrics, rows)
    pass_count = sum(1 for row in rows if bool(row["pass"]))
    lines = [
        "20.72 COMSOL data registry validation summary",
        "",
        f"manifest_count: {len(manifests)}",
        f"check_count: {len(rows)}",
        f"pass_count: {pass_count}",
        f"validation_pass: {pass_count == len(rows)}",
        "",
        "Boundary: registry validates metadata governance only; no data/NPZ files are committed.",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if pass_count != len(rows):
        raise RuntimeError("registry validation failed")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
