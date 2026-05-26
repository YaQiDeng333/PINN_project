# S132 COMSOL differentiable parametric rasterizer

## 目的

S132 新增 PyTorch differentiable soft rasterizer，将 parametric inverse model 输出的 component geometry 转成 soft mask，使 mask-level loss 可以反传到 geometry parameters。

## soft rasterizer 公式

对每个 component：

1. 将 grid 点从全局坐标旋转到 component 局部坐标；
2. 使用 S117 hard rasterizer 的语义：`axis_x` / `axis_y` 表示 full width / full height，因此 half-axis 为 `abs(axis) * 0.5`；
3. soft rectangle 概率为：
   - `prob_x = sigmoid((half_axis_x - abs(local_x)) / softness)`
   - `prob_y = sigmoid((half_axis_y - abs(local_y)) / softness)`
   - `component_prob = presence_prob * prob_x * prob_y`
4. 多 component union 使用 probabilistic union：
   - `soft_mask = 1 - product(1 - component_prob)`

`softness = softness_cells * mean_grid_spacing`，默认 `softness_cells=1.0`。

## axis / rotation 处理

- axis 正值处理：使用 `abs(axis).clamp_min(eps)`，与 S117 hard rasterizer 的 `abs(axis)` 口径一致。
- 如果 schema 包含 `rotation_angle`，自动判断 degree / radian；大于 `2*pi` 视为 degree。
- 如果 schema 包含 `rotation_sin` / `rotation_cos`，使用 `atan2(sin, cos)` 得到 angle。

## 当前边界

- 这是近似 soft rasterizer，不是 COMSOL forward consistency。
- `type_logits` 第一版不参与几何选择；当前 V2 的 `rectangular_notch` 和 `rotated_rect` 都按 rotated rectangle approximation 处理。
- 目标是让 mask-level loss 可微地约束 center / axis / rotation / presence。

## 自评

- soft mask shape、数值范围和 backward 已由 smoke test 覆盖。
- 与 S117 hard rasterizer 的 axis / rotation 语义保持一致。
- 仍需通过 S133/S134 验证 mask supervision 是否改善 held-out mask IoU。
