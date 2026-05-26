# S336 Component-Query 1-Sample Raster Repair Summary

本阶段只修 component-query 1-sample hard-raster precision，不进入 5-sample、train30、multi-seed，也不生成新 COMSOL 数据。

S330 的失败样本是 train sample `0`。关键证据来自 S334/S335：hard polygon IoU `0.974227`，pred/target area `194 / 189`，false-positive / false-negative pixels `5 / 0`。`gt_center + pred_local_vertices` 和 centroid-aligned variants 都能到 IoU `1.000000`，说明主要误差是 decoded center / polygon centroid 的亚像素偏移被 hard rasterizer 放大。

本阶段修复方向是 default-off、grid-cell unit 的 tiny center / centroid auxiliary loss。旧 component-query 默认路径必须完全保持不变；S185/S181 center-bin candidate、absolute polygon runner、center-anchored runner都不替换。
