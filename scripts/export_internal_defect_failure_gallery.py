#!/usr/bin/env python
"""Export internal B2 failure gallery previews and index.

PNG previews are written under ignored results/previews and must not be staged.
The tracked CSV index records best/median/worst/failure samples.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from load_internal_defect_pilot_dataset import ROOT, write_csv


CASES = ROOT / "results/metrics/internal_defect_b2_failure_cases.csv"
SUMMARY_OUT = ROOT / "results/summaries/internal_defect_b2_failure_gallery_summary.txt"
INDEX_OUT = ROOT / "results/metrics/internal_defect_b2_failure_gallery_index.csv"
PREVIEW_DIR = ROOT / "results/previews/internal_defect_b2_failure_gallery"


INDEX_FIELDS = [
    "selection_bucket",
    "sample_id",
    "split",
    "true_shape_type",
    "pred_shape_type",
    "true_L_mm",
    "pred_L_mm",
    "true_W_mm",
    "pred_W_mm",
    "true_D_mm",
    "pred_D_mm",
    "true_burial_depth_mm",
    "pred_burial_depth_mm",
    "true_center_xyz_mm",
    "pred_center_xyz_mm",
    "L_error_mm",
    "W_error_mm",
    "D_error_mm",
    "burial_depth_error_mm",
    "center_xyz_error_mm",
    "total_abs_normalized_error",
    "failure_tags",
    "preview_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export B2 internal defect failure gallery.")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    parser.add_argument("--no-png", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def shape_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            safe_float(row[f"{prefix}_L_mm"]),
            safe_float(row[f"{prefix}_W_mm"]),
            safe_float(row[f"{prefix}_D_mm"]),
            safe_float(row[f"{prefix}_burial_depth_mm"]),
            safe_float(row[f"{prefix}_center_x_mm"]),
            safe_float(row[f"{prefix}_center_y_mm"]),
            safe_float(row[f"{prefix}_center_z_mm"]),
        ],
        dtype=np.float64,
    )


def extents(params: np.ndarray) -> tuple[float, float, float, float]:
    l, w, _d, _burial, cx, cy, _cz = params
    return cx - l / 2, cx + l / 2, cy - w / 2, cy + w / 2


def surface_top_z(params: np.ndarray, shape_name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    l, w, d, _burial, cx, cy, cz = params
    l = max(abs(l), 1e-4)
    w = max(abs(w), 1e-4)
    d = max(abs(d), 1e-4)
    rx, ry, rz = l / 2, w / 2, d / 2
    z = np.zeros_like(x, dtype=np.float64)
    if shape_name == "internal_cuboid":
        mask = (np.abs(x - cx) <= rx) & (np.abs(y - cy) <= ry)
        z[mask] = min(cz + rz, 0.0)
        return z
    term = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    mask = term <= 1.0
    top = cz + rz * np.sqrt(np.clip(1.0 - term, 0.0, 1.0))
    z[mask] = np.minimum(top[mask], 0.0)
    return z


def local_grid(true_params: np.ndarray, pred_params: np.ndarray, n: int = 70) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    te = extents(true_params)
    pe = extents(pred_params)
    xmin, xmax = min(te[0], pe[0]), max(te[1], pe[1])
    ymin, ymax = min(te[2], pe[2]), max(te[3], pe[3])
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max((xmax - xmin) * 0.75, (ymax - ymin) * 0.75, 3.0)
    u = np.linspace(-1, 1, n)
    v = np.linspace(-1, 1, n)
    uu, vv = np.meshgrid(u, v)
    return uu, vv, cx + uu * half, cy + vv * half


def add_panel(ax: Any, u: np.ndarray, v: np.ndarray, z: np.ndarray, title: str, cmap: Any, vmin: float | None = None, vmax: float | None = None) -> Any:
    surf = ax.plot_surface(u, v, z, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True, alpha=0.96)
    offset = float(np.nanmin(z) - 0.08 * max(1.0, np.ptp(z)))
    ax.contour(u, v, z, zdir="z", offset=offset, levels=8, cmap=cmap, linewidths=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("normalized local length u")
    ax.set_ylabel("normalized local width v")
    ax.set_zlabel("top z (mm)")
    ax.view_init(elev=24, azim=-58)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return surf


def render_preview(row: dict[str, str], bucket: str, preview_dir: Path) -> Path:
    true_params = shape_row(row, "true")
    pred_params = shape_row(row, "pred")
    u, v, x, y = local_grid(true_params, pred_params)
    z_true = surface_top_z(true_params, row["true_shape_type"], x, y)
    z_pred = surface_top_z(pred_params, row["pred_shape_type"], x, y)
    z_err = z_pred - z_true
    min_z = min(float(z_true.min()), float(z_pred.min()))
    zlim = (min_z * 1.08, 0.05)
    err_abs = max(abs(float(z_err.min())), abs(float(z_err.max())), 0.05)
    fig = plt.figure(figsize=(18, 6.2), dpi=170)
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    s1 = add_panel(ax1, u, v, z_true, "True internal top-depth surface", cm.viridis, vmin=zlim[0], vmax=0)
    s2 = add_panel(ax2, u, v, z_pred, "Predicted internal top-depth surface", cm.viridis, vmin=zlim[0], vmax=0)
    s3 = add_panel(ax3, u, v, z_err, "Prediction top-depth error", cm.coolwarm, vmin=-err_abs, vmax=err_abs)
    ax1.set_zlim(*zlim)
    ax2.set_zlim(*zlim)
    ax3.set_zlim(-err_abs, err_abs)
    fig.colorbar(s1, ax=ax1, shrink=0.65, pad=0.05).set_label("z / depth (mm)")
    fig.colorbar(s2, ax=ax2, shrink=0.65, pad=0.05).set_label("z / depth (mm)")
    fig.colorbar(s3, ax=ax3, shrink=0.65, pad=0.05).set_label("pred - true top z (mm)")
    title = (
        f"{bucket}: {row['sample_id']} | true/pred={row['true_shape_type']}/{row['pred_shape_type']}\n"
        f"total={safe_float(row['total_abs_normalized_error']):.3f}, burial_err={safe_float(row['burial_depth_error_mm']):.3f} mm, "
        f"center_err={safe_float(row['center_xyz_error_mm']):.3f} mm, tags={row['failure_tags']}"
    )
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    preview_dir.mkdir(parents=True, exist_ok=True)
    safe_bucket = bucket.replace("/", "_").replace("|", "_")
    out = preview_dir / f"{safe_bucket}_{row['sample_id']}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def choose_rows(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    by_total = sorted(rows, key=lambda row: safe_float(row["total_abs_normalized_error"]))
    by_center = sorted(rows, key=lambda row: safe_float(row["center_xyz_error_mm"]), reverse=True)
    by_burial = sorted(rows, key=lambda row: safe_float(row["burial_depth_error_mm"]), reverse=True)
    median_start = max(0, len(by_total) // 2 - 1)
    selected: list[tuple[str, dict[str, str]]] = []
    selected.extend(("best_total", row) for row in by_total[:3])
    selected.extend(("median_total", row) for row in by_total[median_start : median_start + 3])
    selected.extend(("worst_center", row) for row in by_center[:5])
    selected.extend(("worst_burial", row) for row in by_burial[:5])
    selected.extend(("shape_misclassified", row) for row in rows if "shape_misclassified" in row["failure_tags"])
    selected.extend(("full_shift_failure", row) for row in rows if "full_shift_failure" in row["failure_tags"])
    selected.extend(("geometry_branch_failure", row) for row in rows if "geometry_branch_failure" in row["failure_tags"])
    return selected


def index_row(bucket: str, row: dict[str, str], preview_path: str) -> dict[str, Any]:
    return {
        "selection_bucket": bucket,
        "sample_id": row["sample_id"],
        "split": row["split"],
        "true_shape_type": row["true_shape_type"],
        "pred_shape_type": row["pred_shape_type"],
        "true_L_mm": row["true_L_mm"],
        "pred_L_mm": row["pred_L_mm"],
        "true_W_mm": row["true_W_mm"],
        "pred_W_mm": row["pred_W_mm"],
        "true_D_mm": row["true_D_mm"],
        "pred_D_mm": row["pred_D_mm"],
        "true_burial_depth_mm": row["true_burial_depth_mm"],
        "pred_burial_depth_mm": row["pred_burial_depth_mm"],
        "true_center_xyz_mm": f"{row['true_center_x_mm']},{row['true_center_y_mm']},{row['true_center_z_mm']}",
        "pred_center_xyz_mm": f"{row['pred_center_x_mm']},{row['pred_center_y_mm']},{row['pred_center_z_mm']}",
        "L_error_mm": row["L_error_mm"],
        "W_error_mm": row["W_error_mm"],
        "D_error_mm": row["D_error_mm"],
        "burial_depth_error_mm": row["burial_depth_error_mm"],
        "center_xyz_error_mm": row["center_xyz_error_mm"],
        "total_abs_normalized_error": row["total_abs_normalized_error"],
        "failure_tags": row["failure_tags"],
        "preview_path": preview_path,
    }


def main() -> int:
    args = parse_args()
    rows = [row for row in read_csv(args.cases) if row.get("split") == "test"]
    selected = choose_rows(rows)
    index_rows = []
    generated_paths: set[str] = set()
    for bucket, row in selected:
        path = ""
        if not args.no_png:
            out = render_preview(row, bucket, args.preview_dir)
            path = str(out)
            generated_paths.add(path)
        index_rows.append(index_row(bucket, row, path))
    write_csv(INDEX_OUT, index_rows, INDEX_FIELDS)
    summary = [
        "22.0 内部缺陷 B2 failure gallery",
        f"index_rows: {len(index_rows)}",
        f"unique_png_generated: {len(generated_paths)}",
        f"preview_dir: {args.preview_dir}",
        "png_committed: false",
        "selection_buckets: total error 最好 3 个、median 3 个、center 最差 5 个、burial 最差 5 个、全部 shape_misclassified、全部 full_shift_failure、全部 geometry_branch_failure。",
    ]
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
