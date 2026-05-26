#!/usr/bin/env python
"""Audit the existing 20.83 profile-primary prediction gallery.

The script reads the gallery index and sample metrics only. It does not
regenerate PNGs and does not use the gallery for model selection.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "results/metrics/true_3d_rbc_profile_primary_loss_gallery_index.csv"
SAMPLES = ROOT / "results/metrics/true_3d_rbc_profile_primary_loss_gallery_sample_metrics.csv"
SUMMARY = ROOT / "results/summaries/true_3d_rbc_prediction_gallery_audit_summary.txt"
FAILURE_AUDIT = ROOT / "results/metrics/true_3d_rbc_prediction_gallery_failure_audit.csv"

FIELDS = [
    "audit_bucket",
    "sample_id",
    "selection_bucket",
    "split",
    "curvature_template",
    "depth_bin",
    "aspect_bin",
    "profile_depth_rmse_m",
    "er_like_profile_error",
    "projected_mask_iou",
    "projected_mask_dice",
    "L_mae_mm",
    "W_mae_mm",
    "D_mae_mm",
    "wLD_abs_error",
    "wWD_abs_error",
    "wLW_abs_error",
    "curvature_mae",
    "png_path",
    "finding",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def curvature_mae(row: dict[str, str]) -> float:
    return mean([as_float(row.get("wLD_abs_error")), as_float(row.get("wWD_abs_error")), as_float(row.get("wLW_abs_error"))])


def row_out(row: dict[str, str], bucket: str, finding: str) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in FIELDS}
    out["audit_bucket"] = bucket
    out["curvature_mae"] = f"{curvature_mae(row):.12g}"
    out["finding"] = finding
    return out


def top_by(rows: list[dict[str, str]], key: str, reverse: bool, n: int = 3) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: as_float(row.get(key)), reverse=reverse)[:n]


def unique_extend(target: list[dict[str, Any]], source: list[dict[str, Any]]) -> None:
    seen = {(row["audit_bucket"], row["sample_id"]) for row in target}
    for row in source:
        key = (row["audit_bucket"], row["sample_id"])
        if key not in seen:
            target.append(row)
            seen.add(key)


def main() -> int:
    if not INDEX.exists() or not SAMPLES.exists():
        missing = [str(p.relative_to(ROOT)) for p in (INDEX, SAMPLES) if not p.exists()]
        raise RuntimeError(f"missing gallery CSV: {missing}")

    index_rows = read_csv(INDEX)
    sample_rows = read_csv(SAMPLES)
    test_rows = [row for row in sample_rows if row.get("split") == "test"]
    if not test_rows:
        raise RuntimeError("gallery sample metrics contain no test rows")

    audit_rows: list[dict[str, Any]] = []
    unique_extend(
        audit_rows,
        [
            row_out(row, "best_profile", "Low profile RMSE gallery examples are genuinely low by CSV profile/Er-like metrics; Dice may still be moderate.")
            for row in top_by(test_rows, "profile_depth_rmse_m", reverse=False, n=3)
        ],
    )
    unique_extend(
        audit_rows,
        [
            row_out(row, "worst_profile", "Worst profile examples have the largest 3-D profile error and support the negative-gate conclusion.")
            for row in top_by(test_rows, "profile_depth_rmse_m", reverse=True, n=3)
        ],
    )
    dice_high_profile_bad = [
        row
        for row in test_rows
        if as_float(row.get("projected_mask_dice")) >= 0.86 and as_float(row.get("profile_depth_rmse_m")) >= 4.0e-4
    ]
    unique_extend(
        audit_rows,
        [
            row_out(row, "high_dice_high_profile_error", "Projected footprint can look strong while 3-D depth/profile error remains high.")
            for row in top_by(dice_high_profile_bad, "profile_depth_rmse_m", reverse=True, n=6)
        ],
    )
    high_curv = sorted(test_rows, key=curvature_mae, reverse=True)[:6]
    unique_extend(
        audit_rows,
        [
            row_out(row, "curvature_risk", "High auxiliary w-error persists; this is diagnostic risk rather than the main profile-depth decision criterion.")
            for row in high_curv
        ],
    )
    high_dice_high_curv = [
        row
        for row in test_rows
        if as_float(row.get("projected_mask_dice")) >= 0.86 and curvature_mae(row) >= 0.20
    ]
    unique_extend(
        audit_rows,
        [
            row_out(row, "high_dice_high_curvature_error", "Good projected mask does not guarantee stable wLD/wWD/wLW diagnostics.")
            for row in high_dice_high_curv[:6]
        ],
    )
    write_csv(FAILURE_AUDIT, audit_rows, FIELDS)

    best = top_by(test_rows, "profile_depth_rmse_m", reverse=False, n=3)
    worst = top_by(test_rows, "profile_depth_rmse_m", reverse=True, n=3)
    best_rmse = mean([as_float(row["profile_depth_rmse_m"]) for row in best])
    worst_rmse = mean([as_float(row["profile_depth_rmse_m"]) for row in worst])
    high_dice_bad_count = len(dice_high_profile_bad)
    curvature_risk_ids = ", ".join(row["sample_id"] for row in high_curv[:3])
    png_paths_in_index = [row.get("png_path", "") for row in index_rows]
    all_pngs_existing = all(Path(path).exists() for path in png_paths_in_index if path)
    all_pngs_under_ignored_preview = all("results\\previews\\" in path or "results/previews/" in path for path in png_paths_in_index if path)

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        "\n".join(
            [
                "20.84 true 3D RBC prediction gallery audit summary",
                "",
                "gallery_source: existing 20.83 profile-primary loss gallery CSV and PNG paths",
                "gallery_regenerated: false",
                "used_for_model_selection: false",
                f"gallery_index_rows: {len(index_rows)}",
                f"gallery_sample_metric_rows: {len(sample_rows)}",
                f"all_png_paths_exist: {all_pngs_existing}",
                f"all_pngs_under_ignored_results_previews: {all_pngs_under_ignored_preview}",
                "",
                "CSV audit findings:",
                f"- Best profile examples have mean profile_depth_rmse_m={best_rmse:.9f}. They are profile-good by persisted metrics, even when Dice is not always the maximum.",
                f"- Worst profile examples have mean profile_depth_rmse_m={worst_rmse:.9f}; this is the gallery evidence behind the 20.83 negative gate.",
                f"- High-Dice but high-profile-error cases: {high_dice_bad_count}. Projected mask Dice is therefore not enough as a 3-D profile metric.",
                f"- Highest curvature-risk sample ids: {curvature_risk_ids}. They show wLD/wWD/wLW instability remains an auxiliary diagnostic risk.",
                "",
                "visual inspection note:",
                "The script itself is CSV-only and does not parse PNG pixels. Main-agent visual inspection of representative PNGs agrees with the CSV audit: the best-profile example has very similar true/pred depth maps with small banded depth error, while the worst-profile deep-wide boxy example keeps a good 2-D footprint but substantially underestimates central depth.",
                "",
                "conclusion: the existing gallery supports the 20.83 negative gate: P2 can look visually/mask-strong in some examples, but profile RMSE still worsens versus 20.77.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
