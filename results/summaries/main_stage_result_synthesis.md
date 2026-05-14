# 主线阶段性结果整理

本文件只汇总已有实验结果，用于后续论文和汇报材料整理。不包含新实验、新评估或新路线建议。

## 1. CURRENT_BASELINE 当前状态

当前 v3_complex 推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

对应数据集：

* train: `data/training_data_v3_complex_train.npz`
* val: `data/training_data_v3_complex_val.npz`
* test: `data/training_data_v3_complex_test.npz`

推荐配置：

* `lambda_tv = 2e-6`
* physics_loss: off
* L-BFGS: off

test 指标：

| MSE | MAE | IoU | Dice | area_error | center_error |
|---:|---:|---:|---:|---:|---:|
| 2.07377174e+04 | 4.44655262e+01 | 2.95272047e-01 | 4.21885407e-01 | 3.94517442e-01 | 1.32594189e+00 |

该 baseline 保持不变的原因是：后续 v4、loss、threshold、mask、oversampling、encoder、multi-liftoff 等探索均没有在主要指标上形成稳定、综合优于该模型的结果。部分实验有局部改善，但同时带来 MSE / MAE / area_error、small defect 或稳定性方面的明显副作用。

## 2. v4 / calibrated_mu / enhanced decoder 线结论

v4_balanced_complex 数据集改善了复杂缺陷样本组织和平衡性，但 v4 baseline 没有明显超过当前 v3_complex 推荐 baseline。

calibrated_mu 输出参数化在 v4 上有局部正信号：缺陷区预测 μ_r 分布下降，IoU、Dice、center_error 和部分 polygon / small polygon 指标有改善。但它没有根本解决 area_error，且 `pred_area > true_area` 数量增加。

enhanced decoder 在多 seed 下稳定改善 MAE、小幅改善 IoU / Dice，并减少 small polygon IoU=0 数量；同时它也稳定加重面积高估，使 area_error 变差。threshold calibration 能缓解 enhanced decoder 的面积高估，但属于 evaluation-level 后处理，不是模型本身的稳定改进。

最终状态：v4 / calibrated_mu / enhanced decoder 线封存为消融记录，不替换 CURRENT_BASELINE。

## 3. threshold calibration 线结论

第 10.1 failure audit 显示 CURRENT_BASELINE 的主要失败模式是系统性面积低估：`pred_area < true_area` 为 158 / 200，`pred_area=0` 为 18。

第 10.2 test-set threshold sensitivity 显示，提高 mask threshold 能缓解面积低估。第 10.3 使用 validation set 选出 `threshold=600`，并在 test set 验证：

| threshold | IoU | Dice | area_error | center_error | pred_area=0 | pred_area<true | pred_area>true |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.2953 | 0.4219 | 0.3945 | 1.3259 | 18 | 158 | 42 |
| 600 | 0.3528 | 0.4952 | 0.4158 | 1.2031 | 9 | 80 | 120 |

threshold=600 改善 IoU、Dice、center_error，并减少空预测和面积低估；代价是 area_error 小幅变差，且面积高估显著增加。

因此 threshold calibration 只作为 evaluation-level calibration 记录，不更新 CURRENT_BASELINE，也不继续做更多 threshold trick。

## 4. aux mask head / aux mask loss 线结论

aux_mask_head 直接输出 mask 的固定阈值结果不够好，不适合作为最终输出主线。

aux mask loss 作为 regularizer 在 v4 standard decoder 上有正信号：相比 baseline_mu_threshold，aux_mu_threshold 的 MSE、MAE、IoU / Dice、area_error 和 small IoU=0 均有改善。但 enhanced decoder + aux mask loss 仍无法降低面积高估，area_error 继续恶化。

迁移到 v3 baseline 后，第 9.1 aux mask regularizer transfer gate 失败：除 center_error 改善外，MSE、MAE、IoU、Dice、area_error 都比 CURRENT_BASELINE 变差。

最终状态：aux mask head 不作为最终输出；aux mask loss 不继续迁移或调参；该方向停止。

## 5. v3 baseline transfer gates 结论

围绕 CURRENT_BASELINE 的 fast gates 结果如下：

| step | 方法 | 主要结果 | 结论 |
|---|---|---|---|
| 9.1 | aux mask regularizer transfer | center_error 改善，但 MSE、MAE、IoU、Dice、area_error 变差 | 失败 |
| 9.2 | shape-aware loss transfer | IoU / Dice / center_error 改善，但 MSE、MAE、area_error 明显恶化，area_error 升到 0.9167 | 失败 |
| 10.4 | defect-weighted MSE | IoU / Dice 提升、空预测减少，但 MSE、MAE、area_error 变差，并从面积低估推成面积高估 | 失败 |
| 10.5 | calibrated_mu | MSE、MAE、center_error 有改善，但 IoU / Dice 下降，pred_area=0 从 18 增到 32 | 失败 |
| 10.6 | threshold-margin loss | 只改善 center_error，IoU / Dice 下降，area_error 变差，pred_area=0 从 18 增到 27 | 失败 |

这些结果说明，v4 阶段的 loss / aux / calibration 技巧不能直接迁移到 v3_complex baseline；继续围绕这些局部技巧调参缺乏证据支撑。

## 6. failure audit / observability audit 结论

CURRENT_BASELINE 的主要失败模式是系统性面积低估，小缺陷漏检/欠检是重要次级问题：

* `pred_area < true_area`: 158 / 200
* `pred_area=0`: 18 / 200
* small / medium / large 三个面积分桶中，平均 `pred_area` 都小于 `true_area`
* 最差 IoU 样本主要集中在 polygon 和 rotated_rect 的 small 样本；最差 center_error 样本多为 multi_defect

prediction distribution audit 显示，真实缺陷像素中只有 40.85% 的 `pred_mu < 500`，16.20% 落在 500-600，42.95% 大于等于 600；small defect 更差，`pred_mu < 500` 仅 29.84%，`pred_mu >= 600` 达到 50.95%。

signal difficulty audit 显示 small / low-signal 样本确实更难：

* small mean `max_abs_bz = 6.96`，large mean `max_abs_bz = 14.66`
* `pred_area=0` 样本 mean `max_abs_bz = 4.06`，非空预测样本为 10.96
* `peak_to_peak_bz` 与 IoU 的相关系数约 0.593，与 Dice 约 0.599

结论是：失败与 Bz 信号弱、single-signal 可辨识性较低有关，但不是完全不可辨识；模型对弱信号的利用和输出校准也有限。

## 7. multi-liftoff 线结论

第 11.4 seed=42 gate 中，multi-liftoff 有明显正信号：overall IoU / Dice、area_error、center_error、small defect、low-signal 样本和 `pred_area=0` 均优于 fair single-liftoff。

但第 11.5 三 seed 配对实验没有稳定复现该正信号。3 seed mean 中，multi-liftoff 相比 fair single-liftoff：

| method | IoU | Dice | area_error | center_error | pred_area=0 |
|---|---:|---:|---:|---:|---:|
| fair single-liftoff | 0.2932 | 0.4233 | 0.4139 | 1.2823 | 18.67 |
| multi-liftoff | 0.2825 | 0.4099 | 0.4398 | 1.3336 | 17.33 |

small / low-signal 样本也没有稳定改善。提升主要来自 seed=42，seed=123 和 seed=2026 未复现。

最终状态：multi-liftoff 不进入正式主线候选，不继续扩展 seed，不继续调 multi-liftoff 结构。

## 8. 当前总判断

当前阶段的主线结论是：

* 不再继续 v4 作为替代 baseline；
* 不再继续小 loss trick、threshold trick、mask head、small oversampling、简单 encoder 替换、简单 Bz 输入增强或 multi-liftoff 修补；
* CURRENT_BASELINE 保持为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`；
* 当前后续工作更适合转向阶段性总结、baseline 结果整理和论文/汇报材料准备；
* 如果继续研究，应重新定义更大的实验包、接受条件和停止条件，而不是继续围绕局部副作用追加小修补。
