# S337 Component-Query Center/Centroid Auxiliary Support

已在 shared center-anchored loss path 中加入默认关闭的 precision auxiliary，并由 component-query runner 暴露 CLI：

- `--lambda-decoded-center-aux`，默认 `0.0`
- `--lambda-polygon-centroid-aux`，默认 `0.0`
- `--center-centroid-aux-smoothl1-beta`，默认 `0.01`

两个 loss 都使用 grid-cell units。`decoded_center_aux_loss` 约束 hard decoded center 到 target center；`polygon_centroid_aux_loss` 从 decoded polygon vertices 的 masked mean 得到 centroid，再对齐 GT polygon centroid。默认 lambda 为 `0.0`，因此旧实验语义不变。

runner metrics/history/config 新增 `decoded_center_aux_loss`、`polygon_centroid_aux_loss`、`weighted_decoded_center_aux_loss`、`weighted_polygon_centroid_aux_loss` 以及对应 lambda/beta 字段。smoke tests 覆盖默认关闭路径和 aux 开启路径。
