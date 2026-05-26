# S157 COMSOL parametric oracle ablation implementation

## 目的

S157 新增 parameter-level oracle ablation diagnostic，用已有 S126 prediction export 和 S113 ground-truth parametric targets 做替换实验，不运行新训练。

## 新增脚本

- `comsol_parametric_oracle_ablation.py`
- `smoke_test_comsol_parametric_oracle_ablation.py`

## Variants

- `pred_all`: 全部使用预测的 presence / type / continuous。
- `gt_type`: 只替换 GT type。
- `gt_rotation`: 只替换 GT rotation。
- `gt_type_rotation`: 替换 GT type 和 GT rotation。
- `gt_center`: 替换 GT `center_x` / `center_y`。
- `gt_axis`: 替换 GT `axis_x` / `axis_y`。
- `gt_depth`: 替换 GT `depth_or_shape_param`。
- `gt_continuous_all`: 替换全部 continuous，保留预测 type / presence。
- `gt_type_continuous`: 替换 type + continuous，保留预测 presence。
- `gt_all`: 替换 presence + type + continuous，作为 hard-raster oracle sanity check。

## 对齐与安全检查

- 强制检查 prediction CSV 的必需字段。
- 强制检查 `(sample_index, component_slot)` 唯一且完整。
- 当前 S126 fixed-order export 要求 `matched_slot == component_slot`。
- 强制检查 CSV `target_schema`、`type_vocab` 与 S113 targets 一致。
- 强制检查 CSV true 值与 S113 targets 对齐。
- rotation 按 raw degree 语义处理，并在调用 hard rasterizer 前转成 `rotation_sin` / `rotation_cos`，避免旧 `rotation_angle` degree/radian heuristic。

## 边界

当前 hard rasterizer 将 `rectangular_notch` 和 `rotated_rect` 都近似为 rotated rectangle，因此单独替换 `type` 通常不会改变 rasterized mask。`gt_type` 仍保留为 diagnostic variant，用于确认 type 通过当前 mask rasterization 路径是否直接影响 IoU。

## 自评

- S157 不训练、不保存权重、不保存 checkpoint、不保存图片。
- smoke test 覆盖 rotation 替换提升、`gt_all` sanity、type mismatch 重建和缺字段错误。
- 后续 S158 用 train / val / test 运行该脚本，并用 `gt_all` 对齐 S117 oracle 判断实现可信度。
