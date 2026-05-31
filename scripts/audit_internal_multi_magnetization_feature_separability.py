#!/usr/bin/env python
"""23.5 multi-magnetization feature separability audit.

只使用 delta_b 派生特征做近邻可分性诊断；shape / burial / size / aspect
标签只用于 metrics，不作为输入。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from audit_internal_multi_magnetization_pairs import (
    ROOT,
    build_feature_matrix,
    load_dataset,
    train_standardize,
    write_csv,
)


SUMMARY = ROOT / "results/summaries/internal_multi_magnetization_feature_separability_summary.txt"
METRICS = ROOT / "results/metrics/internal_multi_magnetization_feature_separability_metrics.csv"
NN_AMBIGUITY = ROOT / "results/metrics/internal_multi_magnetization_nearest_neighbor_ambiguity.csv"

PRIMARY_CONFIGS = ["mag_x_5line_only", "dual_mag_xy_5line", "mag_x_9line_only", "dual_mag_xy_9line"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit feature separability for internal multi-magnetization observations.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--nearest-neighbor", type=Path, default=NN_AMBIGUITY)
    return parser.parse_args()


def nearest_indices(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.mean(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = np.argmin(dist, axis=1)
    return nn, dist[np.arange(x.shape[0]), nn]


def separability_metrics(config: str, x: np.ndarray, dataset: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_idx = np.arange(x.shape[0])
    x_std, _, _ = train_standardize(x, all_idx)
    nn, nn_dist = nearest_indices(x_std)

    shape_same = dataset.shape_type == dataset.shape_type[nn]
    burial_same = dataset.burial_depth_level == dataset.burial_depth_level[nn]
    size_same = dataset.size_level == dataset.size_level[nn]
    aspect_same = dataset.aspect_bin == dataset.aspect_bin[nn]
    center_dist = np.linalg.norm((dataset.y[:, 4:7] - dataset.y[nn, 4:7]) * 1000.0, axis=1)
    burial_dist = np.abs((dataset.y[:, 3] - dataset.y[nn, 3]) * 1000.0)

    cuboid_ellipsoid = np.isin(dataset.shape_type, ["internal_cuboid", "internal_ellipsoid"])
    cuboid_ellipsoid_cross = (
        cuboid_ellipsoid
        & np.isin(dataset.shape_type[nn], ["internal_cuboid", "internal_ellipsoid"])
        & (dataset.shape_type != dataset.shape_type[nn])
    )
    elongated = np.isin(dataset.aspect_bin, ["elongated_x", "elongated_y"])
    elongated_cross = elongated & np.isin(dataset.aspect_bin[nn], ["elongated_x", "elongated_y"]) & (dataset.aspect_bin != dataset.aspect_bin[nn])
    ambiguous = (~shape_same) | (burial_dist > 1.0) | (center_dist > 3.0)

    metrics = {
        "observation_config": config,
        "sample_count": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "shape_nn_consistency": float(np.mean(shape_same)),
        "burial_depth_bin_nn_consistency": float(np.mean(burial_same)),
        "size_bin_nn_consistency": float(np.mean(size_same)),
        "aspect_bin_nn_consistency": float(np.mean(aspect_same)),
        "cuboid_ellipsoid_cross_nn_rate": float(np.mean(cuboid_ellipsoid_cross[cuboid_ellipsoid])) if np.any(cuboid_ellipsoid) else 0.0,
        "elongated_x_y_cross_nn_rate": float(np.mean(elongated_cross[elongated])) if np.any(elongated) else 0.0,
        "ambiguous_neighbor_rate": float(np.mean(ambiguous)),
        "mean_nn_center_distance_mm": float(np.mean(center_dist)),
        "median_nn_center_distance_mm": float(np.median(center_dist)),
        "mean_nn_burial_distance_mm": float(np.mean(burial_dist)),
        "median_nn_burial_distance_mm": float(np.median(burial_dist)),
        "mean_nn_feature_distance": float(np.mean(nn_dist)),
    }

    rows: list[dict[str, Any]] = []
    for i, base in enumerate(dataset.base_ids):
        rows.append(
            {
                "observation_config": config,
                "base_group_id": str(base),
                "nearest_base_group_id": str(dataset.base_ids[nn[i]]),
                "shape_type": str(dataset.shape_type[i]),
                "nearest_shape_type": str(dataset.shape_type[nn[i]]),
                "burial_depth_level": str(dataset.burial_depth_level[i]),
                "nearest_burial_depth_level": str(dataset.burial_depth_level[nn[i]]),
                "size_level": str(dataset.size_level[i]),
                "nearest_size_level": str(dataset.size_level[nn[i]]),
                "aspect_bin": str(dataset.aspect_bin[i]),
                "nearest_aspect_bin": str(dataset.aspect_bin[nn[i]]),
                "nearest_feature_distance": float(nn_dist[i]),
                "center_label_distance_mm": float(center_dist[i]),
                "burial_label_distance_mm": float(burial_dist[i]),
                "shape_match": bool(shape_same[i]),
                "burial_bin_match": bool(burial_same[i]),
                "ambiguous_neighbor": bool(ambiguous[i]),
                "cuboid_ellipsoid_cross": bool(cuboid_ellipsoid_cross[i]),
                "elongated_x_y_cross": bool(elongated_cross[i]),
            }
        )
    return metrics, rows


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    metric_rows: list[dict[str, Any]] = []
    nn_rows: list[dict[str, Any]] = []
    for config in PRIMARY_CONFIGS:
        features, _names = build_feature_matrix(dataset, config)
        metrics, rows = separability_metrics(config, features, dataset)
        metric_rows.append(metrics)
        nn_rows.extend(rows)

    metric_fields = [
        "observation_config",
        "sample_count",
        "feature_dim",
        "shape_nn_consistency",
        "burial_depth_bin_nn_consistency",
        "size_bin_nn_consistency",
        "aspect_bin_nn_consistency",
        "cuboid_ellipsoid_cross_nn_rate",
        "elongated_x_y_cross_nn_rate",
        "ambiguous_neighbor_rate",
        "mean_nn_center_distance_mm",
        "median_nn_center_distance_mm",
        "mean_nn_burial_distance_mm",
        "median_nn_burial_distance_mm",
        "mean_nn_feature_distance",
    ]
    nn_fields = [
        "observation_config",
        "base_group_id",
        "nearest_base_group_id",
        "shape_type",
        "nearest_shape_type",
        "burial_depth_level",
        "nearest_burial_depth_level",
        "size_level",
        "nearest_size_level",
        "aspect_bin",
        "nearest_aspect_bin",
        "nearest_feature_distance",
        "center_label_distance_mm",
        "burial_label_distance_mm",
        "shape_match",
        "burial_bin_match",
        "ambiguous_neighbor",
        "cuboid_ellipsoid_cross",
        "elongated_x_y_cross",
    ]
    write_csv(args.metrics, metric_rows, metric_fields)
    write_csv(args.nearest_neighbor, nn_rows, nn_fields)

    by_config = {row["observation_config"]: row for row in metric_rows}
    lines = [
        "23.5 internal multi-magnetization feature separability summary",
        "",
        "输入边界：只使用 delta_b/BxByBz 派生特征；标签只用于可分性指标。",
        f"mag_x_5line_only shape_nn_consistency: {by_config['mag_x_5line_only']['shape_nn_consistency']:.6f}",
        f"dual_mag_xy_5line shape_nn_consistency: {by_config['dual_mag_xy_5line']['shape_nn_consistency']:.6f}",
        f"mag_x_9line_only shape_nn_consistency: {by_config['mag_x_9line_only']['shape_nn_consistency']:.6f}",
        f"dual_mag_xy_9line shape_nn_consistency: {by_config['dual_mag_xy_9line']['shape_nn_consistency']:.6f}",
        f"dual_mag_xy_5line ambiguous_neighbor_rate: {by_config['dual_mag_xy_5line']['ambiguous_neighbor_rate']:.6f}",
        f"dual_mag_xy_9line ambiguous_neighbor_rate: {by_config['dual_mag_xy_9line']['ambiguous_neighbor_rate']:.6f}",
        f"dual_mag_xy_5line cuboid_ellipsoid_cross_nn_rate: {by_config['dual_mag_xy_5line']['cuboid_ellipsoid_cross_nn_rate']:.6f}",
        f"dual_mag_xy_9line cuboid_ellipsoid_cross_nn_rate: {by_config['dual_mag_xy_9line']['cuboid_ellipsoid_cross_nn_rate']:.6f}",
        "",
        "结论：该阶段只评估双磁化是否改善 delta_b 特征空间的几何可分性；是否进入 23.6 训练由 diagnostic probe 与 route decision 共同判断。",
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
