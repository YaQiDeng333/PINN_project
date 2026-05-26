# COMSOL_PARAMETRIC_INVERSE_PLAN

## 1. 为什么从 dense mask 转向 parametric inverse

COMSOL V2 dense conditional mask 训练已经多次出现全背景或全前景塌缩。S87-S90 显示 target/mask 定义不是主因，V2 的主要变化是 label area 更小、目标包含 multi_defect / non-ellipsoid，任务分布明显比 V1 更难。

S91-S111 进一步说明，继续调 dense mask loss 的效率很低：weighted BCE、focal BCE、positive-balanced sampling、area loss、threshold margin 和 validation-aware endpoint selection 都没有稳定解决 V2 localization / shape 问题。

V2 数据同时提供 `defect_params`。这些参数提供了更低维、更结构化的监督信号，可以先学习 `multi-height Bz -> geometry parameters`，再由参数 rasterize 出 mask 做 IoU / Dice 评估。该路线避免直接在 200 x 100 dense mask 空间中学习 sparse foreground。

## 2. 输入输出

输入：
- `signals` shape `[B, 3, 200]`
- 或 flatten 后的 `[B, 600]`

输出：
- `max_components` 个 component 参数。

每个 component 包含：
- `presence`
- `component_type`
- `center_x`
- `center_y`
- `axis_x` / `width`
- `axis_y` / `height`
- `rotation_angle`
- `depth_or_shape_param`
- optional `mu` / magnetic parameters

第一版只使用 geometry 参数，不把 magnetic parameters 纳入训练 target。

## 3. component sorting

为避免 permutation ambiguity，第一版按 `center_x` 排序。如果 `center_x` 相同，再按 `center_y` 排序。

V2 的 `source_component_json` 是 component-level metadata，优先使用其中的 `component_type`、`center_x_m`、`center_y_m`、`length_m`、`width_m`、`depth_m` 和 `angle_deg`。

## 4. loss

第一版训练 loss：
- `presence` BCE；
- `component_type` CE；
- continuous parameter MSE / SmoothL1；
- mask IoU 只做评估，不先做可微 loss。

连续参数第一版包含：
- `center_x`
- `center_y`
- `axis_x`
- `axis_y`
- `depth_or_shape_param`
- `rotation_angle`

## 5. rasterization

第一版使用 numpy / torch 非可微 rasterization：
- 根据预测参数生成近似矩形 / rotated rectangle mask；
- 与 target mask 比较 IoU / Dice；
- 不反传 mask IoU。

如果参数路线有效，后续再考虑可微 rasterizer 或 dense mask refinement。

## 6. 当前阶段边界

- 只做 skeleton、smoke test 和 small train probe；
- 不替代主线；
- 不声称最终效果；
- 不保存模型权重或 checkpoint；
- 如果参数预测有非平凡信号，下一步再增强 rasterization / mask refinement；
- 如果参数预测也失败，说明 `Bz -> geometry` 仍需要更强 encoder、更多数据或重新设计 COMSOL data curriculum。

## 7. S117-S120 后的当前状态

S117 已用 ground-truth parametric targets 做 oracle rasterization。train / val / test oracle mask IoU 分别约为 `0.722997`、`0.723288`、`0.716584`，说明当前 target + rasterizer 有可用上限，但仍存在 rasterizer gap。

S118 已将 raw `rotation_angle` 扩展为 `rotation_sin` / `rotation_cos`，并加入 train-stat continuous normalization。S119 的 `refined_mlp` 没有改善 held-out mask IoU，val / test mask IoU 为 `0.325765` / `0.388509`，低于 S115。

当前结论是：parametric inverse route 继续成立，但下一步不应只延长普通 MLP。推荐 S121-S125 依次做 error decomposition、component-specific heads、CNN/attention signal encoder、quick gate architecture probe 和 route decision。forward consistency / differentiable rasterization 应在 rasterizer semantics 更稳定后再进入。
