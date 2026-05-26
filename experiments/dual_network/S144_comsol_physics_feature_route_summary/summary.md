# S144 COMSOL physics feature route summary

## S140 结论

S140 确认 S136-S139 的 two-stage raster fine-tune 没有稳定超过 S115 / S134 parameter-only baseline。Raster loss 仍是可用工具，但当前不继续盲扫 raster BCE / Dice 或 delayed fine-tune，而是转向 signal representation：显式提取 multi-height Bz 的 MFL physics features。

## S141 feature extraction

S141 新增 `comsol_mfl_physics_features.py`，从 V2 train / val / test signals 中提取 `[100,58]` / `[20,58]` / `[20,58]` features。主要 categories 包括 peak、peak position、peak width、energy、abs area、lift-off decay ratios 和 inter-channel correlations。所有 features 均为 finite。

## S142 feature fusion support

S142 扩展 `ParametricInverseNet` 和 `train_comsol_parametric_inverse.py`：

- `feature_fusion_mode=none`：默认旧行为。
- `feature_fusion_mode=features_only`：仅使用 `FeatureMLP` 编码 physics features。
- `feature_fusion_mode=concat_latent`：融合 raw signal latent 与 feature latent。

Feature normalization 只使用 train mean/std，val/test 使用 train stats。Claude Code review 未发现 must-fix。

## S143 结果

| run | val type_acc | val rotation_mae | val mask_iou | test type_acc | test rotation_mae | test mask_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_signal_reference` | 6.500000e-01 | 7.731843e+00 | 3.699078e-01 | 6.666667e-01 | 7.740397e+00 | 4.244624e-01 |
| `physics_features_only` | 6.500000e-01 | 8.987270e+00 | 2.362846e-01 | 5.500000e-01 | 1.098153e+01 | 2.327774e-01 |
| `raw_plus_physics_features` | 6.666667e-01 | 6.144658e+00 | 3.313752e-01 | 5.833333e-01 | 8.876973e+00 | 3.051455e-01 |

## 判断

- `physics_features_only` 能强拟合 train，train mask IoU 为 `7.273660e-01`，但 held-out val/test 明显退化。
- `raw_plus_physics_features` 对 val type / rotation 有局部改善，但没有改善 val/test mask IoU，test type / rotation 也低于 raw reference。
- 当前 physics features 没有证明 raw MLP 的主要瓶颈只是“缺少显式 peak/decay features”。
- 当前最佳 parametric 配置仍是 S115 / S143 `raw_signal_reference` raw MLP / shared head / fixed-order baseline。

## 下一步建议

Parametric route 继续，但不把 S141 features 直接作为默认。下一步更合理的是：

- forward consistency / learned forward surrogate：让 predicted geometry 解释 multi-height Bz signal。
- physics feature regularization 或 auxiliary feature prediction，而不是直接 concat features。
- 如果继续 feature route，先做 grouped feature error / feature importance，确认哪些 features 与 type / rotation / mask IoU 相关。
- 若 type 改善但 mask IoU 不改善，应继续检查 rasterizer gap、geometry loss 和 forward consistency。

## 自评

- 没有过度声称 physics features 失败；它们有 train-side signal，但直接 fusion 不稳定。
- 结论明确：当前默认仍是 raw MLP baseline，下一步优先 forward consistency / learned forward surrogate。
