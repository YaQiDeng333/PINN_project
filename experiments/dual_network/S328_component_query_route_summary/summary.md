# S328 Component-Query Polygon Route Summary

本阶段进入 component-query center/shape joint head 路线，但不替换 S185/S181 center-bin candidate，也不替换 existing polygon / center-anchored runners。

## Motivation

S323-S327 已证明 loss-side decoded-center consistency 不足以修复 held-out center decode failure。`gt_center_bin_offset` 离线替换可以把 val/test IoU 提到 `0.450778 / 0.438502`，但 `soft_decoded_center_consistency` 让 held-out 变差。因此下一步需要结构性耦合：每个固定 component slot/query 使用同一个 query latent 同时预测 center bin、center offset 和 local polygon shape。

## Implemented Scope

- 新增独立模型：`comsol_component_query_polygon_inverse_models.py`。
- 新增独立 runner：`train_comsol_component_query_polygon_inverse.py`。
- 新增 model / runner smoke tests。
- 保持旧 center-anchored runner 默认行为不变。
- 不使用 attention、Hungarian matching、dynamic query、bbox-scale head、teacher forcing、soft decoded consistency、area/edge aux、y-loss、bounded output 或 local-conditioning sweep。

## Model Contract

- input signals: `[B, 3, 200]`
- encoder latent: `[B, D]`
- fixed learned component queries: `Q=3`
- per-query outputs:
  - `presence_logits [B, 3]`
  - `type_logits [B, 3, T]`
  - `center_x_bin_logits [B, 3, XB]`
  - `center_y_bin_logits [B, 3, YB]`
  - `center_offset [B, 3, 2]`
  - `local_vertices_grid [B, 3, 4, 2]`

## Boundary

本阶段仍是 diagnostic route，不是 main baseline replacement。若 1-sample gate 不通过，必须停止，不进入 5-sample、same-run reference 或 train30。
