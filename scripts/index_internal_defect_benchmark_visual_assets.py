#!/usr/bin/env python
"""21.8 internal defect visual asset index.

不生成 PNG；只记录当前是否存在可用 prediction/profile artifacts。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_benchmark_visual_assets_summary.txt"
INDEX_OUT = ROOT / "results/metrics/internal_defect_benchmark_visual_assets_index.csv"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    candidate_paths = [
        ROOT / "checkpoints/internal_defect_benchmark_b2_predictions.npz",
        ROOT / "checkpoints/internal_defect_benchmark_artifacts/internal_defect_b2_predictions.npz",
        ROOT / "results/metrics/internal_defect_benchmark_b2_per_sample_predictions.csv",
        ROOT / "results/previews/internal_defect_benchmark_gallery",
    ]
    rows = []
    for path in candidate_paths:
        rows.append(
            {
                "asset_type": "prediction_or_gallery_artifact",
                "path": str(path),
                "exists": path.exists(),
                "tracked_policy": "do_not_commit_binary_or_png",
                "usable_for_gallery": path.exists() and path.suffix.lower() in {".csv", ".npz"},
                "notes": "existing artifact only; this script does not generate images",
            }
        )

    write_csv(
        INDEX_OUT,
        rows,
        ["asset_type", "path", "exists", "tracked_policy", "usable_for_gallery", "notes"],
    )
    existing = [row for row in rows if row["exists"]]
    summary = [
        "21.8 internal defect benchmark visual assets index",
        "generation_policy: no PNG generated; no NPZ/checkpoint/data written.",
        f"existing_candidate_artifacts: {len(existing)}",
        "visual_assets_available: false" if not existing else "visual_assets_available: partial",
        "reason: 21.7 produced formal aggregate/group metrics but did not persist a per-sample B2 prediction/profile artifact.",
        "next_visual_step: recover/export internal B2 inference artifact first, then generate best/worst gallery without committing PNG.",
    ]
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
