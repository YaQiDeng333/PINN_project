"""Audit v2_120 true 3D RBC pack before expanding to v3_240.

The dataset path is resolved only through COMSOL_DATA_REGISTRY.md and the
tracked manifest. This script intentionally does not scan data directories for
latest/newest NPZ files.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASET_ID = "comsol_true_3d_rbc_imported_watertight_pilot_v2_120"
MANIFEST_PATH = Path(
    "results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.manifest.json"
)
REGISTRY_PATH = Path("COMSOL_DATA_REGISTRY.md")

SUMMARY_PATH = Path("results/summaries/true_3d_rbc_v2_120_dataset_expansion_audit_summary.txt")
AUDIT_CSV_PATH = Path("results/metrics/true_3d_rbc_v2_120_dataset_expansion_audit.csv")
MISSING_CSV_PATH = Path("results/metrics/true_3d_rbc_v2_120_missing_expansion_targets.csv")

PREV_DECISION_SUMMARY = Path("results/summaries/true_3d_rbc_v2_120_training_gate_decision_summary.txt")
PREV_NEURAL_METRICS = Path("results/metrics/true_3d_rbc_v2_120_neural_training_gate_metrics.csv")

TARGET_N = 240
TARGET_SPLIT = {"train": 160, "val": 40, "test": 40}
TARGET_CURVATURE = {
    "sharp": 48,
    "round": 48,
    "boxy": 48,
    "LD_dominant": 48,
    "WD_dominant": 48,
}
TARGET_DEPTH = {"shallow": 80, "medium": 80, "deep": 80}
TARGET_ASPECT = {"narrow": 60, "compact": 60, "balanced": 60, "wide": 60}

TOPUP_SUCCESS_TARGET = 128
TOPUP_SPLIT_TARGET = {"train": 84, "val": 22, "test": 22}


@dataclass
class Row:
    sample_id: str
    split: str
    curvature_template: str
    depth_bin: str
    aspect_bin: str
    L_m: float
    W_m: float
    D_m: float
    wLD: float
    wWD: float
    wLW: float


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def registry_contains_dataset(path: Path, dataset_id: str) -> bool:
    if not path.exists():
        return False
    return dataset_id in path.read_text(encoding="utf-8")


def resolve_npz_path(manifest: dict[str, Any]) -> Path:
    raw = manifest.get("npz_path") or manifest.get("path")
    if not raw:
        raise ValueError("manifest missing npz_path/path")
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"dataset NPZ not found from manifest: {path}")
    return path


def as_text_array(value: Any) -> list[str]:
    arr = np.asarray(value)
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in arr.reshape(-1)]


def maybe_field_array(npz: np.lib.npyio.NpzFile, names: list[str]) -> list[str] | None:
    for name in names:
        if name in npz.files:
            return as_text_array(npz[name])
    return None


def load_rows(npz_path: Path) -> list[Row]:
    with np.load(npz_path, allow_pickle=True) as npz:
        if "sample_ids" in npz.files:
            sample_ids = as_text_array(npz["sample_ids"])
        elif "sample_id" in npz.files:
            sample_ids = as_text_array(npz["sample_id"])
        else:
            raise ValueError("NPZ missing sample_ids/sample_id")

        split = maybe_field_array(npz, ["split", "splits"])
        if split is None:
            raise ValueError("NPZ missing split")

        curvature = maybe_field_array(npz, ["curvature_template", "curvature_templates"])
        depth = maybe_field_array(npz, ["depth_bin", "depth_bins"])
        aspect = maybe_field_array(npz, ["aspect_bin", "aspect_bins"])
        if curvature is None or depth is None or aspect is None:
            raise ValueError("NPZ missing curvature_template/depth_bin/aspect_bin metadata")

        params = npz["rbc_params"] if "rbc_params" in npz.files else None
        if params is None:
            raise ValueError("NPZ missing rbc_params")

        names = params.dtype.names if hasattr(params.dtype, "names") else None
        params_plain = np.asarray(params, dtype=float) if not names else None
        rows: list[Row] = []
        for i, sample_id in enumerate(sample_ids):
            if names:
                get = lambda key: float(params[key][i])
            else:
                get = lambda key: float(params_plain.reshape(len(sample_ids), -1)[i, ["L_m", "W_m", "D_m", "wLD", "wWD", "wLW"].index(key)])
            rows.append(
                Row(
                    sample_id=sample_id,
                    split=split[i],
                    curvature_template=curvature[i],
                    depth_bin=depth[i],
                    aspect_bin=aspect[i],
                    L_m=get("L_m"),
                    W_m=get("W_m"),
                    D_m=get("D_m"),
                    wLD=get("wLD"),
                    wWD=get("wWD"),
                    wLW=get("wLW"),
                )
            )
        return rows


def counter(rows: list[Row], attr: str) -> Counter[str]:
    return Counter(getattr(row, attr) for row in rows)


def tuple_counter(rows: list[Row], attrs: tuple[str, ...]) -> Counter[tuple[str, ...]]:
    return Counter(tuple(getattr(row, attr) for attr in attrs) for row in rows)


def deficit_rows(rows: list[Row]) -> list[dict[str, Any]]:
    split_counts = counter(rows, "split")
    curv_counts = counter(rows, "curvature_template")
    depth_counts = counter(rows, "depth_bin")
    aspect_counts = counter(rows, "aspect_bin")
    split_curv_counts = tuple_counter(rows, ("split", "curvature_template"))
    cell_counts = tuple_counter(rows, ("curvature_template", "depth_bin", "aspect_bin"))

    output: list[dict[str, Any]] = []
    for split_name, target in TARGET_SPLIT.items():
        output.append(
            {
                "target_type": "split",
                "key": split_name,
                "current_count": split_counts.get(split_name, 0),
                "target_count": target,
                "needed_success": max(0, target - split_counts.get(split_name, 0)),
            }
        )
    for key, target in TARGET_CURVATURE.items():
        output.append(
            {
                "target_type": "curvature_template",
                "key": key,
                "current_count": curv_counts.get(key, 0),
                "target_count": target,
                "needed_success": max(0, target - curv_counts.get(key, 0)),
            }
        )
    for key, target in TARGET_DEPTH.items():
        output.append(
            {
                "target_type": "depth_bin",
                "key": key,
                "current_count": depth_counts.get(key, 0),
                "target_count": target,
                "needed_success": max(0, target - depth_counts.get(key, 0)),
            }
        )
    for key, target in TARGET_ASPECT.items():
        output.append(
            {
                "target_type": "aspect_bin",
                "key": key,
                "current_count": aspect_counts.get(key, 0),
                "target_count": target,
                "needed_success": max(0, target - aspect_counts.get(key, 0)),
            }
        )

    for split_name in ["train", "val", "test"]:
        for curv in TARGET_CURVATURE:
            current = split_curv_counts.get((split_name, curv), 0)
            target = {"train": 32, "val": 8, "test": 8}[split_name]
            output.append(
                {
                    "target_type": "split_curvature",
                    "key": f"{split_name}/{curv}",
                    "current_count": current,
                    "target_count": target,
                    "needed_success": max(0, target - current),
                }
            )

    for curv in TARGET_CURVATURE:
        for depth in TARGET_DEPTH:
            for aspect in TARGET_ASPECT:
                current = cell_counts.get((curv, depth, aspect), 0)
                if current <= 1:
                    output.append(
                        {
                            "target_type": "coverage_cell_low_count",
                            "key": f"{curv}/{depth}/{aspect}",
                            "current_count": current,
                            "target_count": 2,
                            "needed_success": max(0, 2 - current),
                        }
                    )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    manifest: dict[str, Any],
    npz_path: Path,
    rows: list[Row],
    missing: list[dict[str, Any]],
) -> None:
    split_counts = counter(rows, "split")
    curv_counts = counter(rows, "curvature_template")
    depth_counts = counter(rows, "depth_bin")
    aspect_counts = counter(rows, "aspect_bin")
    d_values = [row.D_m for row in rows]
    wld_values = [row.wLD for row in rows]
    wwd_values = [row.wWD for row in rows]
    wlw_values = [row.wLW for row in rows]

    high_priority = [
        item
        for item in missing
        if item["target_type"] == "coverage_cell_low_count"
        and item["key"]
        in {
            "sharp/deep/narrow",
            "WD_dominant/medium/narrow",
            "WD_dominant/deep/narrow",
            "WD_dominant/deep/compact",
            "sharp/medium/narrow",
            "round/deep/narrow",
            "boxy/medium/narrow",
        }
    ]
    previous_decision = (
        PREV_DECISION_SUMMARY.read_text(encoding="utf-8", errors="ignore").strip()
        if PREV_DECISION_SUMMARY.exists()
        else "missing previous 20.75 decision summary"
    )

    lines = [
        "20.76 v2_120 dataset expansion audit",
        f"dataset_id={DATASET_ID}",
        f"registry_contains_dataset={registry_contains_dataset(REGISTRY_PATH, DATASET_ID)}",
        f"manifest_path={MANIFEST_PATH}",
        f"npz_path={npz_path}",
        f"manifest_status={manifest.get('status')}",
        f"manifest_train_ready_candidate={manifest.get('train_ready_candidate')}",
        f"manifest_baseline_ready={manifest.get('baseline_ready')}",
        f"N={len(rows)}",
        f"split_counts={dict(split_counts)}",
        f"curvature_template_counts={dict(curv_counts)}",
        f"depth_bin_counts={dict(depth_counts)}",
        f"aspect_bin_counts={dict(aspect_counts)}",
        f"D_m_min_max=({min(d_values):.6g}, {max(d_values):.6g})",
        f"wLD_min_max=({min(wld_values):.6g}, {max(wld_values):.6g})",
        f"wWD_min_max=({min(wwd_values):.6g}, {max(wwd_values):.6g})",
        f"wLW_min_max=({min(wlw_values):.6g}, {max(wlw_values):.6g})",
        "",
        "Top-up targets:",
        f"target_N={TARGET_N}",
        f"target_split={TARGET_SPLIT}",
        f"target_curvature={TARGET_CURVATURE}",
        f"target_depth={TARGET_DEPTH}",
        f"target_aspect={TARGET_ASPECT}",
        f"topup_success_target={TOPUP_SUCCESS_TARGET}",
        f"topup_split_target={TOPUP_SPLIT_TARGET}",
        "",
        "Main gaps:",
        "- WD_dominant has the lowest current count and needs +28 successes.",
        "- Medium/deep D_m bins are thin and should be prioritized without using unsafe extremes.",
        "- Narrow high-risk cells remain underrepresented, especially sharp/deep/narrow and WD_dominant medium/deep/narrow.",
        "- Curvature values need more numerical diversity, not only more template labels.",
        "",
        "High-priority low-count cells:",
    ]
    lines.extend(
        f"- {item['key']}: current={item['current_count']} target={item['target_count']}"
        for item in high_priority
    )
    lines.extend(
        [
            "",
            "20.75 learnability blocker:",
            "L_m, W_m, and D_m are learnable on v2_120, but D_m remains imprecise and wLD/wWD/wLW remain unstable.",
            "The v3_240 top-up should improve validation/test support and curvature-depth-aspect coverage before the next training gate.",
            "",
            "Previous 20.75 decision summary excerpt:",
            previous_decision[:3000],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if Path.cwd().name != "PINN_project":
        raise SystemExit("Run from PINN_project root.")
    if SUMMARY_PATH.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {SUMMARY_PATH}; pass --overwrite")

    if not registry_contains_dataset(REGISTRY_PATH, DATASET_ID):
        raise SystemExit(f"registry does not contain dataset_id={DATASET_ID}")
    manifest = read_manifest(MANIFEST_PATH)
    if manifest.get("dataset_id") != DATASET_ID:
        raise SystemExit(f"manifest dataset_id mismatch: {manifest.get('dataset_id')}")
    if manifest.get("status") != "pilot_generated":
        raise SystemExit(f"manifest status is not pilot_generated: {manifest.get('status')}")
    if manifest.get("train_ready_candidate") is not True:
        raise SystemExit("manifest train_ready_candidate is not true")
    if manifest.get("baseline_ready") is not False:
        raise SystemExit("manifest baseline_ready is not false")

    npz_path = resolve_npz_path(manifest)
    rows = load_rows(npz_path)
    if len(rows) != 112:
        raise SystemExit(f"expected N=112 for v2_120, got {len(rows)}")

    audit_rows = [
        {
            "sample_id": row.sample_id,
            "split": row.split,
            "curvature_template": row.curvature_template,
            "depth_bin": row.depth_bin,
            "aspect_bin": row.aspect_bin,
            "L_m": row.L_m,
            "W_m": row.W_m,
            "D_m": row.D_m,
            "wLD": row.wLD,
            "wWD": row.wWD,
            "wLW": row.wLW,
        }
        for row in rows
    ]
    missing = deficit_rows(rows)
    write_csv(AUDIT_CSV_PATH, audit_rows)
    write_csv(MISSING_CSV_PATH, missing)
    write_summary(SUMMARY_PATH, manifest, npz_path, rows, missing)

    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {AUDIT_CSV_PATH}")
    print(f"wrote {MISSING_CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
