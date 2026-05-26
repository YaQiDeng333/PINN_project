# S106 COMSOL V2 threshold-margin summary

## S103 诊断结论

S103 确认 S101 三组都存在 hard-threshold crossing 问题：soft foreground 面积非零，但 hard `defect_area_pred=0`。三组最后的 `min_mu` 都高于 `500`，因此没有形成 hard foreground。`area_ratio_mse` 降低了连续 `mu` 误差，但仍停在 threshold 上方。

## S104 新增功能

`train_conditional_dual.py` 新增：

- `--lambda-threshold-margin`
- `--threshold-margin-mode none|positive_hinge|bidirectional_hinge`
- `--positive-mu-margin`
- `--negative-mu-margin`

`training_history.csv` 新增：

- `threshold_margin_loss`
- `positive_margin_loss`
- `negative_margin_loss`
- `sampled_positive_count`
- `sampled_negative_count`
- `sampled_mu_positive_mean`
- `sampled_mu_negative_mean`

默认 `lambda_threshold_margin=0.0` 且 `threshold_margin_mode=none`，保持旧行为。

## S105 结果结论

- baseline 仍全背景。
- `positive_margin_lambda1` 和 `positive_margin_lambda10` 恢复了非零 hard mask，但退化为全前景，val/test IoU 约等于 label area ratio。
- `bidirectional_margin_lambda1` 恢复了非零 hard mask，且避免全前景，train / val / test IoU = `1.305192e-01` / `1.185913e-01` / `1.266814e-01`。
- threshold-margin objective 证明 hard-threshold crossing 是关键瓶颈之一。
- 但 S105 最佳仍低于 S85 `big_multichannel_v2`，说明 margin loss 只解决了 crossing，不解决定位 / shape。

## 当前瓶颈更新

1. hard threshold crossing 是真实瓶颈，已经被 S103/S105 验证。
2. positive-only crossing 会导致全前景，需要负样本 margin 或其他背景约束。
3. bidirectional margin 能恢复非零 IoU，但定位质量仍不足。
4. 下一步瓶颈是 localization / boundary / shape，而不是单纯 hard mask area。
5. model/conditioning 和 staged curriculum 仍可能需要调整。

## 下一步建议

- 继续基于 `bidirectional_hinge` 调参，并加入 validation-aware selection。
- 测试 `bidirectional_hinge + direct mask head`，判断是否比 hard `mu_threshold` 输出路径更稳定。
- 增加 localization / boundary loss，避免只恢复面积而不能定位多缺陷。
- 如果生成下一批数据，应加入 intermediate-area 和 V1-like bridge samples，降低 V2 multi_defect 任务跳变。
