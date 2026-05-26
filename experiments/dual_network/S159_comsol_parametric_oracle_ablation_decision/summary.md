# S159 COMSOL parametric oracle ablation decision

## S156 结论

S156 确认 simple type / rotation extra losses 没有稳定突破 S115 / S143 baseline。继续盲扫同类 loss 权重的信息密度不足，因此改用 oracle ablation 判断参数误差对 final mask IoU 的真实影响。

## S158 结果

S158 的 `gt_all` 精确复现 S117 oracle：

- train `7.229967e-01`
- val `7.232882e-01`
- test `7.165838e-01`

因此当前 ablation 的 prediction / target / rasterizer 对齐可信。

## 主要瓶颈排序

1. `center_x` / `center_y`
   - `gt_center` 是最大单项提升来源。
   - val IoU 从 `3.699078e-01` 提升到 `7.148715e-01`。
   - test IoU 从 `4.244624e-01` 提升到 `7.229199e-01`。
2. 多个 continuous 参数的累计误差
   - `gt_continuous_all` 达到 `gt_all` oracle。
   - 但其主要收益几乎都来自 center，axis 只有小幅贡献。
3. `axis_x` / `axis_y`
   - val / test 分别只提升 `1.112311e-02` / `8.216901e-03`。
4. `rotation`
   - `gt_rotation` 没有改善 val/test，说明 rotation MAE 虽显眼，但当前 final mask IoU 的主限制不是 rotation。
5. `type`
   - `gt_type` 对 mask IoU 无影响；当前 hard rasterizer 对 `rectangular_notch` 和 `rotated_rect` 使用同一种 rotated-rectangle approximation。
6. `depth_or_shape_param`
   - 当前 mask rasterizer 不使用 depth，因此 `gt_depth` 对 mask IoU 无影响。
7. presence
   - S115 / S154 presence accuracy 已为 `1.0`，不是当前瓶颈。

## 路线判断

Parametric route 继续，但下一步不应再优先调 type / rotation / forward consistency loss。当前最有效的诊断结果指向 localization：模型预测的 component center 偏差是 held-out mask IoU gap 的主要来源。

## 下一步建议

- 优先做 center / localization targeted diagnostic：
  - center-specific loss scaling；
  - center target normalization / coordinate reparameterization；
  - signal-to-center auxiliary head；
  - center-bin classification + continuous offset；
  - per-component Bz peak position alignment features。
- 不建议继续盲扫 simple type CE、rotation extra loss、forward consistency lambda 或 dense mask loss。
- 如果继续改 rasterizer，应先区分两个问题：
  - GT oracle rasterizer gap 约 `0.72` 是 target representation 上限；
  - 当前 prediction gap 的大头是 center localization，不是 type / rotation。

## 自评

- S159 给出了明确参数瓶颈排序。
- 没有把 parametric route 声称为最终 baseline。
- 下一阶段建议直接围绕 center localization，而不是继续泛化调 loss。
