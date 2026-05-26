# S145 COMSOL parametric physics feature stage summary

## 目的

S145 收束 S140-S144，明确 direct physics feature concat 没有超过 raw signal baseline，下一步转向 learned forward surrogate / forward consistency。

## S140-S144 结论

- S140 确认 raster fine-tune 未稳定改善，当前不继续盲扫 raster BCE / Dice。
- S141 已成功从 COMSOL V2 multi-height Bz 中提取 physics features，train / val / test shapes 为 `[100,58]` / `[20,58]` / `[20,58]`。
- S142 已支持 `feature_fusion_mode=features_only|concat_latent`，并保证 feature normalization 只使用 train stats。
- S143 显示 `physics_features_only` 可以强拟合 train，但 val/test mask IoU 明显低于 raw baseline。
- S143 的 `raw_plus_physics_features` 对 val type / rotation 有局部改善，但没有改善 held-out mask IoU，test type / rotation 也退化。
- 当前最佳仍是 S115 / S143 `raw_signal_reference` raw MLP / shared head / fixed-order baseline。

S115 / S143 raw baseline train / val / test mask IoU 为 `6.980716e-01` / `3.699078e-01` / `4.244624e-01`。

## 当前判断

S141 physics features 本身不是无用信号：`features_only` 的 train mask IoU 达到 `7.273660e-01`。但直接 concat 或 features-only 并没有稳定改善 held-out generalization，因此当前瓶颈不只是 raw MLP 缺少显式 peak / width / decay features。

下一步更合理的方向是学习 geometry -> Bz 的 lightweight forward surrogate，并把它作为 consistency referee：如果 inverse model 预测的 geometry 不能通过 surrogate 重建输入 Bz，则用 forward residual 约束 geometry。

## 下一步

- S146：实现 geometry parameters -> multi-height Bz learned forward surrogate。
- S147：训练并验证 surrogate quality。
- S148/S149：仅当 surrogate gate 通过时，在 inverse training 中加入 forward consistency。

## 自评

- 没有把 direct feature fusion 说成路线失败，只限定为当前 quick gate 未超过 raw baseline。
- 下一步明确转向 learned forward surrogate / consistency，而不是 dense runner、raster loss 或 feature concat 盲扫。
