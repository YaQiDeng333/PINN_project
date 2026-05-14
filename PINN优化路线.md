# PINN 缺陷边界形状反演 / 漏磁反演优化路线

## 当前进度

* [x] 第一步：改 data_generator
  让它生成大量样本，并保存完整参数 metadata。
  已完成：生成 train / val / test 数据集，并保存 signals、mu_maps、defect_types、metadata、x、y。
  结论：数据生成流程已经跑通，train / val / test 三个 npz 文件可正常生成和读取；
  metadata 和 metadata_keys 已保存，后续训练与评估可以基于该数据结构继续推进。

* [x] 第二步：改 train_pinn
  让网络输入 Bz 信号，而不是只输入坐标。
  目标：从 “坐标 → μ” 升级为 “Bz + 坐标 → μ”。
  已完成：加载 train / val 数据集，使用 BzEncoder 将一维 Bz signal 编码成 latent vector；
  对空间坐标使用 Fourier feature，并拼接 [bz_latent, coord_features] 后通过 MLP 输出 μ(x,y)。
  当前训练流程支持 batch 训练，只使用 MSE Loss，每轮输出 train loss / val loss；
  已保存验证集 loss 最低模型 checkpoints/best_model.pt，并输出 results/loss_curve.png 和 results/val_prediction.png。
  结论：Bz + 坐标输入的 batch 训练流程已经跑通；20 个 epoch 内 train loss 和 val loss 正常下降，
  best_model.pt 可作为后续 TV Loss、L-BFGS 和评估阶段的基线模型。

* [x] 第三步：改 evaluate_pinn
  加入定量评价指标：IoU、Dice、面积误差、中心误差。
  目的：后续判断 TV Loss、L-BFGS、物理 Loss 是否真的有效。
  已完成：新增 evaluate_pinn.py，加载 data/training_data_test.npz 和 checkpoints/best_model.pt；
  复用 train_pinn.py 中一致的 BzEncoder、Fourier feature、PINN forward 逻辑，在整个 test 集上逐批预测 μ map。
  当前评估输出 MSE、MAE、IoU、Dice、area_error、center_error；
  已保存整体平均指标 results/evaluation_metrics.txt、逐样本指标 results/evaluation_metrics.csv，
  并输出 3 个测试样本的预测 μ map、真实 μ map、预测 mask、真实 mask 对比图。
  结论：评估流程已经覆盖整个 test 集，不再只看第 0 个样本；基线模型 test 指标为
  MSE=2.17269746e+04、MAE=4.61076419e+01、IoU=4.25961039e-01、Dice=5.76173878e-01、
  area_error=2.69236126e-01、center_error=1.02991446e+00。

* [x] 第四步：加 TV Loss
  先解决重建图毛刺和背景噪点。
  目标：减少孤立斑点，让 μ map 更平滑。
  已完成：在 train_pinn.py 中加入 Total Variation Loss，训练目标为 total_loss = mse_loss + lambda_tv * tv_loss。
  当前 lambda_tv = 1e-4，TV Loss 作用在 reshape 后的预测 μ map 上；
  训练日志输出 mse_loss、tv_loss、total_loss、val_mse_loss。
  已保存 TV 模型 checkpoints/best_model_tv.pt、TV loss 曲线 results/loss_curve_tv.png、
  验证样本对比图 results/reconstruction_preview_tv.png。
  已通过 evaluate_pinn.py 对比原模型和 TV 模型：TV 后 MAE 降低，但 MSE、IoU、Dice、面积误差、中心误差略变差。
  结论：TV Loss 训练流程已经跑通，lambda_tv=1e-4 时 MAE 从 4.61076419e+01 降到 4.29366580e+01；
  但 MSE、IoU、Dice、area_error、center_error 均未改善，说明当前 TV 权重或训练方式还不是稳定收益。

* [x] 第五步：加入 L-BFGS
  用于后期精修和降低 Loss 曲线毛刺。
  注意：作为 optional refine，不要影响基础训练流程。
  已完成：在 train_pinn.py 中新增 --mode lbfgs_refine，从 checkpoints/best_model_tv.pt 加载初始权重；
  使用固定小子集 refine_train_samples=8、refine_val_samples=16 进行 L-BFGS 后期精修。
  当前 L-BFGS 参数为 lr=0.5、max_iter=20、history_size=20、outer_steps=10，lambda_tv 保持 1e-4。
  已保存 checkpoints/best_model_tv_lbfgs.pt、results/loss_curve_tv_lbfgs.png、
  results/reconstruction_preview_tv_lbfgs.png。
  评估结果显示 refine 子集 loss 明显降低，但 test 集 MSE、MAE、IoU、Dice、面积误差、中心误差均比 TV 模型变差，存在小子集过拟合迹象。
  结论：L-BFGS refine 已能从 best_model_tv.pt 独立运行并保存新模型；
  但当前小子集配置下泛化变差，TV+L-BFGS 的 test 指标整体劣于 TV 模型，暂不适合作为默认训练输出。

* [x] 第 5.5 步：TV Loss 和 L-BFGS 参数扫描
  目标：不加入新模块，系统对比第四步和第五步参数，主要基于 val 集选择当前更合适的训练配置。
  已完成：新增 parameter_sweep.py，对 lambda_tv=0、1e-6、5e-6、1e-5、5e-5、1e-4 分别训练并评估；
  汇总结果已保存到 results/tv_lambda_sweep.csv。
  本轮综合 val_iou、val_dice、val_mae 的排序后，推荐 lambda_tv=5e-6，
  推荐模型为 checkpoints/best_model_tv_5e-6.pt。
  该模型 val 指标为 MSE=2.19721258e+04、MAE=4.40473880e+01、IoU=4.21560496e-01、Dice=5.70277275e-01、
  area_error=2.42747850e-01、center_error=1.03141948e+00。
  最终 test 指标为 MSE=2.16568206e+04、MAE=4.39399008e+01、IoU=4.32040206e-01、Dice=5.82132493e-01、
  area_error=2.42350201e-01、center_error=1.03291037e+00。
  已对最佳 TV 模型做 L-BFGS 小范围扫描：refine_train_samples=8/16/32、lr=0.1/0.5、outer_steps=5/10；
  汇总结果已保存到 results/lbfgs_sweep.csv。
  L-BFGS 最佳候选为 checkpoints/best_model_tv_5e-6_lbfgs_rs16_lr0p1_os5.pt，
  但其 val_mae、val_iou、val_dice 均未优于最佳 TV 模型，test 指标也整体变差。
  结论：当前默认推荐使用 checkpoints/best_model_tv_5e-6.pt 和 lambda_tv=5e-6；
  L-BFGS 暂时只保留为可选实验，不建议作为默认推荐方案；第六步物理一致性 Loss 暂不进入，等待确认。

* [x] 第六步：加入物理一致性 Loss
  让预测 μ 反推得到的 Bz 和输入 Bz 匹配。
  目标：让模型更像真正的 PINN，而不是普通监督学习。
  准备状态：第 5.5 步参数扫描已完成，进入第六步前的固定 baseline 为 checkpoints/best_model_tv_5e-6.pt；
  推荐 lambda_tv=5e-6；L-BFGS 当前不作为默认推荐方案，仅保留为 optional refine 实验。
  已完成：在 train_pinn.py 中加入简化 physics_loss，训练目标为
  total_loss = mse_loss + lambda_tv * tv_loss + lambda_phy * physics_loss。
  当前 lambda_phy=1e-4，模型从 checkpoints/best_model_tv_5e-6.pt 初始化，不启用 L-BFGS；
  生成 checkpoints/best_model_tv_phy.pt、results/loss_curve_tv_phy.png、
  results/reconstruction_preview_tv_phy.png、results/physics_loss_log.csv。
  评估结果：MSE 从 2.16568206e+04 降到 2.15898657e+04，但 MAE、IoU、Dice、area_error、center_error 均变差。
  结论：物理一致性 Loss 初版流程已跑通，但当前简化 forward model 和 lambda_phy=1e-4 下不能作为默认最佳模型；
  默认 baseline 仍保持 checkpoints/best_model_tv_5e-6.pt，lambda_tv=5e-6，默认不启用 L-BFGS。

* [x] 第 6.5 步：物理一致性 Loss 效果验证与对比总结
  目标：不新增复杂缺陷、不叠加新模块，验证第六步物理一致性 Loss 是否真正优于第 5.5 步 baseline。
  已完成：基于已有 test 评估结果对比 checkpoints/best_model_tv_5e-6.pt 和 checkpoints/best_model_tv_phy.pt；
  对比结果已保存到 results/summaries/physics_loss_comparison_summary.txt 和 results/metrics/physics_loss_comparison.csv。
  结论：物理 Loss 模型 MSE 从 2.16568206e+04 降到 2.15898657e+04，但 MAE、IoU、Dice、area_error、center_error 均变差；
  其中 IoU 从 4.32040206e-01 降到 4.15690292e-01，Dice 从 5.82132493e-01 降到 5.65850626e-01。
  因此 checkpoints/best_model_tv_phy.pt 不作为默认最佳模型；
  当前推荐模型仍为 checkpoints/best_model_tv_5e-6.pt，lambda_tv=5e-6，lambda_phy=0，默认不启用 L-BFGS。
  物理一致性 Loss 初版建议保留为实验模块，后续如需继续优化应优先扫描 lambda_phy 或改进 forward model。

* [x] 第七步：复杂缺陷扩展第一版
  加入不规则、多缺陷、旋转、不同深度、不同提离高度。
  已完成第一版：在 data_generator_v2.py 中新增 rotated_rect、polygon、multi_defect 三类复杂缺陷；
  metadata 在保留旧字段的基础上新增 num_defects、component_types、component_centers、component_sizes、
  component_angles、polygon_vertices、num_vertices、min_mu、complexity_level，并同步更新 metadata_keys。
  已新增 --complex 命令行参数，复杂缺陷数据默认保存为 data/training_data_v3_complex_train.npz、
  data/training_data_v3_complex_val.npz、data/training_data_v3_complex_test.npz，不覆盖旧数据集。
  已完成小样本生成验证：train=20、val=5、test=5；三个 split 均包含 rotated_rect、polygon、multi_defect；
  signals shape、mu_maps shape、metadata 字段、mask 非空和 Bz signal 有效性均检查通过。
  已保存 5 张可视化检查图到 results/previews/data_v3_complex_check_*.png。
  结论：复杂缺陷生成第一版已跑通；尚未重新训练模型，当前推荐 baseline 仍为 checkpoints/best_model_tv_5e-6.pt。

* [x] 第七步：生成正式规模 v3 complex 数据集
  目标：在第七步复杂缺陷生成逻辑验证通过后，生成正式规模复杂缺陷数据集。
  已完成：生成 train=1000、val=200、test=200 的 v3 complex 数据集；
  输出文件为 data/training_data_v3_complex_train.npz、data/training_data_v3_complex_val.npz、
  data/training_data_v3_complex_test.npz。
  三个 split 均包含 signals、mu_maps、defect_types、metadata、metadata_keys、x、y；
  样本数量、shape、metadata 新字段、metadata_keys、缺陷 mask 非空、signals / mu_maps 无 NaN / Inf 均检查通过。
  defect_types 分布：train 中 multi_defect=331、polygon=348、rotated_rect=321；
  val 中 multi_defect=71、polygon=72、rotated_rect=57；
  test 中 multi_defect=62、polygon=75、rotated_rect=63。
  检查摘要已保存到 results/summaries/v3_complex_dataset_summary.txt。
  旧 simple 数据集未覆盖，未重新训练模型，当前推荐 baseline 仍为 checkpoints/best_model_tv_5e-6.pt。

* [x] 第 7.5 步：v3_complex 复杂缺陷模型训练
  目标：使用 v3_complex train / val 数据集训练新的复杂缺陷 baseline，并与旧 simple baseline 分开记录。
  已完成：在 train_pinn.py 中新增 --dataset v3_complex 选择，默认读取
  data/training_data_v3_complex_train.npz 和 data/training_data_v3_complex_val.npz；
  在 evaluate_pinn.py 中新增可选输出路径参数，用于将 v3 complex 评估结果保存到 results/metrics 和 results/previews。
  训练配置为 mode=adam_tv、lambda_tv=5e-6、epochs=20、默认不启用 L-BFGS、默认不启用 physics_loss。
  已保存模型 checkpoints/best_model_v3_complex_tv.pt、
  loss 曲线 results/loss_curves/loss_curve_v3_complex_tv.png、
  预测对比图 results/previews/reconstruction_preview_v3_complex_tv.png。
  v3 complex test 指标：MSE=2.07475147e+04、MAE=4.36197426e+01、
  IoU=2.76481934e-01、Dice=3.97991681e-01、
  area_error=4.26162950e-01、center_error=1.34338298e+00。
  训练摘要已保存到 results/summaries/v3_complex_training_summary.txt。
  结论：复杂缺陷 baseline 已跑通；旧 simple baseline 仍保留为 checkpoints/best_model_tv_5e-6.pt，
  新复杂缺陷 baseline 为 checkpoints/best_model_v3_complex_tv.pt，两者对应不同数据集，不直接混用。

* [x] 第 7.6 步：v3_complex 复杂缺陷 baseline 诊断分析
  目标：分析 checkpoints/best_model_v3_complex_tv.pt 在不同复杂缺陷类型上的表现差异。
  已完成：按 defect_type 统计 rotated_rect、polygon、multi_defect 的 MSE、MAE、IoU、Dice、area_error、center_error 和样本数；
  同时按 metadata 中的 num_defects、complexity_level、num_vertices 做简单分组统计。
  诊断结果保存到 results/metrics/v3_complex_metrics_by_type.csv、
  results/metrics/v3_complex_worst_samples.csv、
  results/summaries/v3_complex_diagnosis_summary.txt，
  最差样本图保存到 results/previews/v3_complex_worst_samples/。
  结论：polygon 的 IoU 和 Dice 最低，IoU=2.20801408e-01，Dice=3.16226151e-01；
  multi_defect 的 MSE、MAE、center_error 较差，但不是唯一拖低整体结果的原因。
  下一步建议先增加 epoch 或对 v3_complex 单独扫描 lambda_tv，暂不进入 physics_loss、L-BFGS 或模型结构大改。

* [x] 第 7.7 步：v3_complex 延长训练与专用 lambda_tv 扫描
  目标：不改数据生成器、不改评价指标、不加 physics_loss / L-BFGS / 新模型结构，先验证复杂缺陷 baseline 是否只是训练不充分或 TV 权重不合适。
  已完成：使用 v3_complex 数据集训练 100 epoch 长训练模型 checkpoints/best_model_v3_complex_tv_long.pt；
  该模型 test 指标为 MSE=2.06158473e+04、MAE=4.73349950e+01、IoU=2.75949820e-01、Dice=3.95393491e-01、
  area_error=4.32650875e-01、center_error=1.30235745e+00。
  长训练只改善了 MSE 和 center_error，MAE、IoU、Dice、area_error 变差，因此不作为默认推荐。
  随后完成 v3_complex 专用 lambda_tv 扫描：0、1e-6、2e-6、5e-6、1e-5，每组 50 epoch，并主要基于 val 集选择参数。
  val IoU / Dice / MAE 综合排序推荐 lambda_tv=2e-6，对应模型 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt。
  该模型 test 指标为 MSE=2.07377174e+04、MAE=4.44655262e+01、IoU=2.95272047e-01、Dice=4.21885407e-01、
  area_error=3.94517442e-01、center_error=1.32594189e+00。
  结论：lambda_tv=2e-6 是当前 v3_complex 推荐配置；polygon IoU/Dice 有改善，multi_defect 仍是主要难点之一。

* [x] 第 7.8 步：polygon / multi_defect 细诊断
  目标：不重新训练、不改模型结构、不改评价指标，基于当前 v3_complex 推荐模型细分分析 polygon 和 multi_defect 的失败原因。
  已完成：基于 data/training_data_v3_complex_test.npz、checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt、
  results/metrics/evaluation_metrics_v3_complex_tv_sweep_2e-6_test.csv，按 polygon 的 num_vertices、multi_defect 的 num_defects、
  以及全体样本的 complexity_level 做分组统计。
  输出保存到 results/metrics/v3_complex_polygon_by_vertices.csv、
  results/metrics/v3_complex_multi_defect_by_count.csv、
  results/metrics/v3_complex_by_complexity_level.csv、
  results/metrics/v3_complex_polygon_worst10.csv、
  results/metrics/v3_complex_multi_defect_worst10.csv、
  results/summaries/v3_complex_fine_diagnosis_summary.txt，
  最差样本图保存到 results/previews/v3_complex_fine_diagnosis/。
  结论：polygon 不是顶点数越多越差，5 顶点样本 IoU/Dice 最低；polygon 最差 10 个样本全部 pred_area=0，说明主要是漏检。
  multi_defect 从 2 个缺陷到 3 个缺陷时 MSE、MAE、area_error、center_error 明显变差；complexity_level=3 的 MSE/MAE/area_error/center_error 也更差。

* [x] 第 7.9 步：v4 balanced complex 数据增强与样本平衡
  目标：暂不改模型结构和评价指标，先修正复杂缺陷数据分布，重点补 polygon 漏检和 multi_defect 三缺陷样本。
  已完成实现：data_generator_v2.py 新增 --dataset v4_balanced_complex；
  输出 data/training_data_v4_balanced_complex_train.npz、data/training_data_v4_balanced_complex_val.npz、
  data/training_data_v4_balanced_complex_test.npz。
  小样本阶段曾使用 train=50、val=10、test=10 做验证；正式阶段已生成 train=1000、val=200、test=200。
  v4 规则：complexity_level 按 level1/2/3 约 30%/40%/30% 分配；
  level2 中 polygon/rotated_rect 约 75%/25%；polygon 中 5 顶点权重最高；
  polygon 记录 mask_pixels、signal_peak_to_peak、signal_snr，并要求 mask_pixels >= 30、signal_snr >= 5；
  multi_defect 中 2/3 缺陷按 40%/60% 分配，并限制组件过度重叠。
  Claude Code review 后已修复 area_bin 阈值和 polygon 双层 retry 问题：
  area_bin 阈值改为 small < 120、120 <= medium < 500、large >= 500；
  polygon 生成改为样本级单层有限重试，仍同时检查 mask_pixels 和 signal_snr。
  修复后小样本 train 中 polygon area_bin 分布为 small=8、medium=4、large=3。
  Claude Code 复审通过后，已使用 seed=7904 生成正式规模 v4 balanced complex 数据集：
  train=1000、val=200、test=200。
  正式 train 中 defect_types 分布为 circle=75、ellipse=75、multi_defect=300、polygon=300、rect=75、rotated_rect=100、triangle=75；
  polygon area_bin 分布为 small=124、medium=103、large=73；
  multi_defect 的 2/3 缺陷分布为 120/180；complexity_level 分布为 1=300、2=400、3=300。
  检查通过：metadata_keys 与 metadata 字段一致，无 NaN / Inf，无空 mask，polygon 均满足 mask_pixels >= 30 和 signal_snr >= 5。
  结论：正式规模 v4 数据集已生成；当前未重新训练模型，当前 v3_complex 推荐模型仍为 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt。

## 推荐执行顺序

1. data_generator_v2.py：批量样本 + metadata + train/val/test
2. train_pinn.py：Bz + 坐标 → μ
3. evaluate_pinn.py：IoU、Dice、面积误差、中心误差
4. train_pinn.py：TV Loss
5. train_pinn.py：L-BFGS refine
6. parameter_sweep.py：TV Loss 和 L-BFGS 参数扫描
7. train_pinn.py：物理一致性 Loss
8. results/metrics + results/summaries：物理一致性 Loss 效果验证与对比总结
9. data_generator_v2.py：复杂缺陷扩展第一版
10. train_pinn.py + evaluate_pinn.py：v3 complex 复杂缺陷 baseline 训练与评估
11. results/metrics + results/summaries：v3 complex 复杂缺陷 baseline 诊断分析
12. train_pinn.py + evaluate_pinn.py：v3 complex 延长训练与 lambda_tv 专用扫描
13. results/metrics + results/summaries：polygon / multi_defect 细诊断
14. data_generator_v2.py：v4 balanced complex 数据增强与样本平衡

## 后续建议

1. 当前 v3_complex 推荐模型固定为 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt，推荐 lambda_tv=2e-6。
2. v4 balanced complex 正式规模数据集已生成并通过检查。
3. 下一步建议训练独立 v4 baseline，不覆盖 v3_complex 推荐模型。
4. 暂不进入新的 physics_loss、L-BFGS 或模型结构大改。

## 当前下一步

详见 `NEXT_STEP.md`。

---

## 第 7.10 步补充：v4_balanced_complex baseline 训练

已完成 v4_balanced_complex 正式数据集上的 baseline 训练：

* 模型：checkpoints/best_model_v4_balanced_complex_tv.pt
* lambda_tv = 2e-6
* epoch = 100
* physics_loss：未启用
* L-BFGS：未启用
* v4 test：MSE=2.39571663e+04，MAE=4.88803274e+01，IoU=2.67902294e-01，Dice=3.81393009e-01，area_error=4.79983772e-01，center_error=1.41093149e+00

分类型诊断显示 polygon 仍然最弱，尤其 small polygon 出现明显漏检；multi_defect 的 center_error 仍偏高。当前不更新 CURRENT_BASELINE.md，v3_complex 推荐模型仍保持为 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt。

建议下一步：进行 v4_balanced_complex 专属 lambda_tv 小范围扫描，优先关注 polygon/small area_bin 和 multi_defect center_error。

---

## 第 7.11 步补充：v4_balanced_complex 专属 lambda_tv 扫描

已完成 v4_balanced_complex 专属 lambda_tv 扫描：

* 候选值：0、5e-7、1e-6、2e-6、5e-6、1e-5
* 每组训练：50 epoch
* physics_loss：未启用
* L-BFGS：未启用
* 模型结构：未修改
* 评价指标定义：未修改

输出文件：

* results/metrics/v4_balanced_complex_lambda_tv_sweep.csv
* results/summaries/v4_balanced_complex_lambda_tv_sweep_summary.txt

按 val_iou、val_dice、val_mae、val_area_error、val_center_error 综合排序，本轮推荐候选为：

* lambda_tv = 0
* 模型：checkpoints/best_model_v4_balanced_complex_tv_sweep_0.pt

该候选 test 指标：

* MSE = 2.41644578e+04
* MAE = 5.09550103e+01
* IoU = 2.73743067e-01
* Dice = 3.87241381e-01
* area_error = 4.90251054e-01
* center_error = 1.38652205e+00

结论：v4 sweep 没有解决 small polygon 漏检问题，small polygon IoU/Dice 仍为 0；multi_defect center_error 比第 7.10 v4 baseline 略有改善，但仍不优于当前 v3_complex 推荐模型。v4 sweep 候选未明显超过 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt，因此不切换 CURRENT_BASELINE.md。

后续建议：不要继续只扩大 lambda_tv 扫描，下一步应进入模型结构或训练策略优化方案设计，重点解决 small polygon 漏检和 multi_defect 分离/定位问题。

---

## 第 7.12A 步补充：small polygon defect-weighted MSE Loss

已完成 small polygon 漏检专项第一轮实验：

* `train_pinn.py` 新增 `--loss-type mse / weighted_mse`，默认仍为 `mse`。
* 新增 `--defect-weight`，默认值为 `10.0`。
* 本轮配置：v4_balanced_complex、`loss_type=weighted_mse`、`defect_weight=10.0`、`lambda_tv=0`、100 epoch。
* 未修改 data_generator_v2.py。
* 未修改 evaluate_pinn.py 的评价指标定义。
* 未修改模型结构。
* 未启用 physics_loss、L-BFGS、soft Dice 或 oversampling。

输出文件：

* checkpoints/best_model_v4_balanced_complex_smallpoly_loss.pt
* results/loss_curves/loss_curve_v4_smallpoly_loss.png
* results/previews/reconstruction_preview_v4_smallpoly_loss.png
* results/metrics/evaluation_metrics_v4_smallpoly_loss.csv
* results/metrics/evaluation_metrics_v4_smallpoly_loss.txt
* results/summaries/v4_smallpoly_loss_summary.txt

test 整体指标：

* MSE = 4.10216735e+04
* MAE = 7.83255570e+01
* IoU = 3.22104979e-01
* Dice = 4.67866207e-01
* area_error = 1.34222578e+00
* center_error = 1.14444251e+00

关键结论：

* small polygon 不再全部漏检，pred_area = 0 的样本数为 0 / 25；
* small polygon IoU = 1.36334593e-01，Dice = 2.26148223e-01；
* overall IoU、Dice、center_error 相比第 7.11 v4 `lambda_tv=0` 候选改善；
* MSE、MAE、area_error 明显变差，说明 `defect_weight=10.0` 可能使预测缺陷区域偏大；
* 当前不切换 CURRENT_BASELINE.md，v3_complex 推荐模型仍为 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt。

后续建议：进入 defect_weight 小范围扫描，候选值 5 / 10 / 20 / 50；暂不加入 soft Dice、oversampling、physics_loss、L-BFGS 或模型结构改动。

---

## 第 7.12B 步补充：v4 small polygon defect_weight 扫描

已完成 defect-weighted MSE 的权重扫描：

* 候选值：2、3、5、7、10
* 每组训练：100 epoch
* loss_type：weighted_mse
* lambda_tv：0
* physics_loss：未启用
* L-BFGS：未启用
* Dice Loss：未启用
* oversampling：未启用
* 模型结构：未修改
* 评价指标定义：未修改

输出文件：

* results/metrics/v4_smallpoly_defect_weight_sweep.csv
* results/summaries/v4_smallpoly_defect_weight_sweep_summary.txt
* checkpoints/best_model_v4_smallpoly_w2.pt
* checkpoints/best_model_v4_smallpoly_w3.pt
* checkpoints/best_model_v4_smallpoly_w5.pt
* checkpoints/best_model_v4_smallpoly_w7.pt
* checkpoints/best_model_v4_smallpoly_w10.pt

本轮推荐候选：

* defect_weight = 5
* 模型：checkpoints/best_model_v4_smallpoly_w5.pt

该候选 test 指标：

* MSE = 3.12321945e+04
* MAE = 6.23678583e+01
* IoU = 3.39080635e-01
* Dice = 4.77603301e-01
* area_error = 8.38023859e-01
* center_error = 1.17307553e+00
* small polygon IoU = 6.54854895e-02
* small polygon Dice = 1.04442883e-01
* small polygon pred_area=0：12 / 25
* multi_defect center_error = 1.08730941e+00

结论：`defect_weight=5` 比 `defect_weight=10` 的面积误差明显更低，同时 small polygon 不再全部漏检；但该模型仍不切换为全项目当前 baseline，当前 v3_complex 推荐模型仍为 checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt。

后续建议：先让 Claude Code review 第 7.12A / 7.12B；暂不直接进入 Dice Loss。review 通过后，再以 `defect_weight=5` 为基础讨论 soft Dice / focal 类 loss。

---

## 第 7.13 步补充：weighted MSE + soft Dice Loss

已完成 soft Dice Loss 第一轮实验：

* `train_pinn.py` 新增 `--loss-type weighted_mse_dice`
* `train_pinn.py` 新增 `--lambda-dice`
* 默认 `--loss-type` 仍为 `mse`
* weighted MSE 缺陷阈值改为 `MASK_THRESHOLD / MU_SCALE`，含义仍为真实 `μ_r < 500`
* 训练配置：v4_balanced_complex、`defect_weight=5`、`lambda_dice=0.05`、`lambda_tv=0`、100 epoch
* 未修改 data_generator_v2.py
* 未修改 evaluate_pinn.py 的评价指标定义
* 未修改模型结构，未启用 physics_loss、L-BFGS 或 oversampling

输出文件：

* checkpoints/best_model_v4_smallpoly_w5_dice.pt
* results/loss_curves/loss_curve_v4_smallpoly_w5_dice.png
* results/previews/reconstruction_preview_v4_smallpoly_w5_dice.png
* results/metrics/evaluation_metrics_v4_smallpoly_w5_dice.csv
* results/metrics/evaluation_metrics_v4_smallpoly_w5_dice.txt
* results/summaries/v4_smallpoly_w5_dice_summary.txt

test 指标：

* MSE = 3.56734905e+04
* MAE = 6.02042826e+01
* IoU = 3.25826098e-01
* Dice = 4.64347405e-01
* area_error = 6.12110696e-01
* center_error = 1.24440727e+00

关键结论：

* small polygon pred_area=0 从 12 / 25 降到 0 / 25；
* small polygon IoU 从 6.54854895e-02 升到 1.26014768e-01；
* small polygon Dice 从 1.04442883e-01 升到 2.01116176e-01；
* overall area_error 从 8.38023859e-01 降到 6.12110696e-01；
* overall IoU / Dice 下降；
* multi_defect center_error 从 1.08730941e+00 升到 1.15517406e+00；
* 当前不切换 CURRENT_BASELINE.md。

后续建议：进入 `lambda_dice` 小范围扫描，暂不切换 baseline。

---

## 第 7.13B 步补充：v4 small polygon lambda_dice 扫描已完成

在 `defect_weight=5`、`lambda_tv=0`、`weighted_mse_dice` 配置下完成 `lambda_dice = 0.01 / 0.03 / 0.05 / 0.1` 扫描。

本轮推荐 v4 small polygon 专项候选：

`checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt`

对应 `lambda_dice = 0.03`。

关键结论：
* small polygon 完全漏检降为 0 / 25；
* overall IoU / Dice 相比第 7.12B weighted MSE w5 恢复并提升；
* multi_defect center_error 改善；
* area_error 仍偏大；
* 不切换全项目 CURRENT_BASELINE。

当前全项目推荐模型仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

## 第 7.14 步补充：area_error 诊断已完成

已对 `checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt` 完成 area_error 诊断。

关键结论：
* pred_area 系统性大于 true_area：190 / 200；
* mean area_ratio = 1.989118，median area_ratio = 1.806303；
* area_error 主要集中在 polygon；
* small polygon 的 IoU / Dice 最低，但 mean area_error 最高的是 medium polygon；
* worst10 中 9 个是 polygon；
* multi_defect 不是本轮 area_error 主因；
* `lambda_dice=0.03` 仍是当前 v4 small polygon 最平衡候选，但不切换全项目 baseline。

下一步建议先降低 `defect_weight` 到 3 或 4 做快速验证；如果仍存在明显过分割，再考虑可选 `area-aware loss`。
---

## 第 7.15 步补充：area-aware loss 面积约束实验已完成

已在 `train_pinn.py` 中加入可选 `weighted_mse_dice_area` loss 和 `--lambda-area` 参数。默认 `--loss-type` 仍为 `mse`，不影响旧训练流程。

本轮扫描 `lambda_area = 0.005 / 0.01 / 0.03 / 0.05`，固定 `defect_weight=5`、`lambda_dice=0.03`、`lambda_tv=0`。

推荐 v4 area-aware 专项候选：

`checkpoints/best_model_v4_smallpoly_w5_dice_area_0p05.pt`

对应 `lambda_area = 0.05`。

关键结论：
* overall area_error 明显下降；
* polygon area_error 和 medium polygon area_error 明显下降；
* small polygon 仍保持 0 / 25 漏检；
* overall IoU / Dice 轻微下降；
* multi_defect center_error 略有变差；
* area-aware loss 仍未完全消除 pred_area 系统性偏大；
* 当前不切换全项目 CURRENT_BASELINE。

当前全项目推荐模型仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
---

## 第 7.16 步补充：面积约束细化实验已完成

已完成 symmetric `lambda_area=0.04 / 0.05 / 0.07` 细扫，并新增 `over_only` 面积约束对比。

关键结论：
* symmetric `lambda_area=0.07` 的 area_error、polygon area_error 和 medium polygon area_error 最低，同时 small polygon pred_area=0 仍为 0 / 25；
* `over_only lambda_area=0.05` 将 pred_area > true_area 降到 166 / 200，但 small polygon pred_area=0 回升到 14 / 25；
* over_only 不适合作为当前推荐方案；
* 当前不切换全项目 baseline。

当前全项目推荐模型仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
---

## 第 7.17 步补充：symmetric area loss 组合验证

第 7.17 步已完成。验证范围限定在 v4_balanced_complex 数据集上，不修改数据生成器、评价指标定义或模型结构，不启用 physics_loss、L-BFGS、focal loss 或 oversampling。

固定配置：

* loss_type = weighted_mse_dice_area
* lambda_dice = 0.03
* lambda_tv = 0
* area_loss_type = symmetric
* epochs = 100

组合验证结果显示：

* `defect_weight=5, lambda_area=0.04` 是本轮综合表现最好的 v4 small polygon / area loss 候选；
* 模型路径：`checkpoints/best_model_v4_w5_dice003_area004.pt`；
* small polygon `pred_area=0` 保持为 0 / 25；
* overall IoU / Dice 和 multi_defect center_error 在本轮四组中最好；
* polygon area_error 没有继续下降，不如第 7.16 步的 symmetric `lambda_area=0.07`。

因此，本轮不切换全项目 baseline。当前推荐 baseline 仍以 `CURRENT_BASELINE.md` 为准：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

后续建议暂停继续扩大 loss 调参，转向模型结构或后处理方案讨论；如继续做 loss 对比，需要加入 seed / repeat 验证。
---

## 第 7.18 步补充：后处理与阈值分析

第 7.18 步已完成。分析对象为第 7.17 步推荐的 v4 small polygon / area-aware 候选：

`checkpoints/best_model_v4_w5_dice003_area004.pt`

本轮不重新训练，不修改 `data_generator_v2.py`，不修改模型结构，不修改 `evaluate_pinn.py` 中 MSE、MAE、IoU、Dice、area_error、center_error 的标准定义。

主要发现：

* 标准 threshold=500 时，overall area_error = 0.911511，pred_area > true_area = 191 / 200；
* threshold=300 时，overall area_error = 0.292975，pred_area > true_area = 114 / 200；
* threshold=300 时，polygon area_error = 0.390191，medium polygon area_error = 0.543884；
* threshold=300 时，small polygon `pred_area=0` 仍为 0 / 25；
* threshold=450 的 IoU / Dice 最高，分别为 0.354303 / 0.497498；
* 连通域过滤 remove < 5 / 10 / 20 pixels 基本没有额外收益。

结论：

降低 mask threshold 可以明显缓解预测面积系统性偏大，尤其改善 polygon / medium polygon 的 area_error。但 threshold 调整存在 trade-off：threshold=300 面积误差最佳，threshold=450 IoU / Dice 更好。连通域过滤不是主要改进方向。

后处理可作为可选评估方案，不替代标准评价流程，也不切换全项目 baseline。当前全项目推荐 baseline 仍为：

`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

后续建议进入第 7.19 步：模型结构优化方案设计，并考虑给 `train_pinn.py` 增加 `--seed` 参数。
---

## 第 7.18.5 步补充：训练随机种子 seed 支持

第 7.18.5 步已完成。为了支撑第 7.19 模型结构优化实验的可复现对比，`train_pinn.py` 已新增 `--seed` 参数。

实现内容：

* 默认 `--seed 42`；
* 新增 `set_seed(seed)`；
* 同步设置 Python random、NumPy、PyTorch 和 CUDA 随机种子；
* Adam 训练的 `DataLoader(shuffle=True)` 使用固定 `torch.Generator()`；
* 训练启动时打印当前 seed。

说明：

第 7.15–7.17 的实验表明，相同配置存在随机性波动。因此第 7.19 及之后的结构对比实验必须固定 seed，默认使用 `42`。

第 7.18 后处理阈值分析说明，当前模型预测 μ 值存在校准偏软问题：缺陷区常预测到 μ≈200–400，而不是接近真实 μ≈1。threshold=300 能显著降低 area_error，说明该问题是模型输出校准与边界表达问题，不是单纯评估阈值问题。

下一步进入第 7.19：模型结构优化方案设计。

## 第 7.19 步补充：模型结构优化方案设计

第 7.19 步已完成方案设计，未修改模型代码，未重新训练。方案文件：

* `MODEL_STRUCTURE_PLAN.md`

主要结论：

* 第 7.12-7.18 的 loss 与后处理实验说明，small polygon 漏检已经得到明显缓解；
* area-aware loss 和 threshold 调整能降低 area_error，但继续调 loss 收益递减；
* threshold=300 明显降低 area_error，说明当前模型输出 μ 值偏软，缺陷区域常停留在 `μ_r≈200-400`；
* 当前输出层实际是 `Linear + Softplus`，Softplus 有下界但无上界；缺陷端要逼近 `mu_norm≈0.001` 时需要很负的 pre-activation，可能导致输出偏软；
* 当前问题更像是输出校准和 decoder 表达能力问题，不是单纯评价阈值问题；
* 第 7.20 拆成 7.20A / 7.20B：7.20A 只做输出 μ 参数化校准，保持当前 decoder 不变；7.20B 只有在 7.20A 有效或部分有效后，再考虑轻量增强 decoder。

第 7.20A 状态：准备开始。要求固定 `seed=42`，使用 v4_balanced_complex 数据集，不启用 physics_loss，不启用 L-BFGS，不切换 CURRENT_BASELINE。

---

## 第 7.20A 步：calibrated_mu 输出 μ 参数化校准实验

状态：已完成。

目标：验证只改变输出 μ 参数化、保持当前 BzEncoder 和 decoder 主体不变时，是否能改善 μ 值偏软和 `pred_area` 偏大的问题。

本轮新增：

* `train_pinn.py` 支持 `--model-variant baseline / calibrated_mu`；
* 默认 `baseline`，保持 `Linear(64, 1) + Softplus` 输出行为；
* `calibrated_mu` 将 decoder logit 转成 defect probability，再映射到 `mu_norm ∈ [0.001, 1.0]`；
* `evaluate_pinn.py` 兼容加载 `model_variant`，并输出本轮所需的 μ 校准诊断字段；标准指标定义不变。

固定配置：

* dataset = `v4_balanced_complex`
* seed = 42
* loss_type = `weighted_mse_dice_area`
* defect_weight = 5
* lambda_dice = 0.03
* lambda_area = 0.04
* area_loss_type = `symmetric`
* lambda_tv = 0
* epochs = 100

结果摘要：

* baseline seed=42：IoU = 3.39044536e-01，Dice = 4.80770498e-01，area_error = 6.40443541e-01；
* calibrated_mu seed=42：IoU = 3.54232016e-01，Dice = 4.96098795e-01，area_error = 6.40109928e-01；
* 缺陷区预测 μ_r 均值从约 399 降到约 361，中位数从约 295 降到约 262；
* small polygon `pred_area=0` 仍为 0 / 25；
* `pred_area > true_area` 数量从 174 / 200 增加到 182 / 200。

结论：`calibrated_mu` 证明输出参数化校准方向有效，但单独改变输出映射不足以解决面积系统性偏大。当前不切换全项目 baseline。

当前下一步：第 7.20B，轻量 decoder 增强 A/B 实验。第 7.20B 应继续固定 seed=42，保持同一 loss 配置，不加入新 loss、physics_loss、L-BFGS 或数据增强。

---

## 第 7.20B 步：calibrated_mu 轻量 decoder 增强实验

状态：已完成。

目标：在第 7.20A `calibrated_mu` 输出参数化基础上，只增强 decoder 容量，验证 decoder 表达能力是否限制 μ 校准和 area_error 改善。

本轮新增：

* `train_pinn.py` 支持 `--decoder-variant standard / enhanced`；
* 默认 `standard`，保持旧 decoder：`128 / 128 / 64 + Tanh`；
* `enhanced` 使用 `256 / 256 / 128 / 64 + SiLU`；
* BzEncoder、Fourier feature、`calibrated_mu` 输出映射和 loss 配置均保持不变；
* `evaluate_pinn.py` 仅增加 `decoder_variant` 加载兼容和 summary 字段，标准指标定义不变。

固定配置：

* dataset = `v4_balanced_complex`
* seed = 42
* model_variant = `calibrated_mu`
* loss_type = `weighted_mse_dice_area`
* defect_weight = 5
* lambda_dice = 0.03
* lambda_area = 0.04
* area_loss_type = `symmetric`
* lambda_tv = 0
* epochs = 100

结果摘要：

* standard decoder：IoU = 3.54232016e-01，Dice = 4.96098795e-01，area_error = 6.40109928e-01；
* enhanced decoder：IoU = 3.53319625e-01，Dice = 4.99793934e-01，area_error = 9.58160563e-01；
* 缺陷区预测 μ_r 均值从约 361 降到约 333，中位数从约 262 降到约 238；
* small polygon `pred_area=0` 仍为 0 / 25，small polygon IoU=0 从 10 / 25 降到 7 / 25；
* `pred_area > true_area` 数量从 182 / 200 增加到 189 / 200。

结论：enhanced decoder 能进一步改善 μ 校准和部分 mask 指标，但显著恶化 area_error，特别是 polygon / medium polygon 面积误差。当前不切换全项目 baseline。

当前下一步：建议做 seed repeat / 稳定性验证，确认该 trade-off 是否稳定，再决定是否继续结构优化。

---

## 第 7.21 步：calibrated_mu decoder 多 seed 配对重复实验

状态：已完成。

目标：对 `calibrated_mu + standard decoder` 和 `calibrated_mu + enhanced decoder` 做多 seed 配对重复，验证第 7.20B 中的 decoder trade-off 是否稳定。本实验不用于更新 baseline。

实验设计：

* seeds = 42 / 123 / 2026
* decoder_variant = `standard` / `enhanced`
* standard seed=42 复用第 7.20A checkpoint
* enhanced seed=42 复用第 7.20B checkpoint
* seed=123 和 seed=2026 为本轮补跑
* dataset = `v4_balanced_complex`
* model_variant = `calibrated_mu`
* loss_type = `weighted_mse_dice_area`
* defect_weight = 5
* lambda_dice = 0.03
* lambda_area = 0.04
* lambda_tv = 0
* area_loss_type = `symmetric`

结果摘要：

* standard mean Dice = 0.491443，enhanced mean Dice = 0.500730；
* standard mean area_error = 0.829989，enhanced mean area_error = 0.953397；
* paired mean ΔDice = +0.009287；
* paired mean Δarea_error = +0.123408；
* enhanced decoder 在 3 个 seed 上均降低 MAE，并降低 defect_mu_mean / defect_mu_median；
* enhanced decoder 在多 seed 均值和 paired mean 上小幅稳定改善 IoU / Dice；
* enhanced decoder 让 small polygon IoU=0 数量下降，并改善 small `pred_area=0` 问题；
* enhanced decoder 在 3 个 seed 上均增加 area_error。
* enhanced decoder 的 `pred_area>true_area` 数量增加或保持更高水平，说明 enhanced decoder 存在更明显的面积高估问题。

结论：enhanced decoder 对 μ 校准、MAE 和 IoU / Dice 的改善具有一定稳定性，但面积高估恶化也稳定存在。因此 enhanced decoder 不更新 `CURRENT_BASELINE`。后续不应继续单纯加宽 decoder，应转向 area calibration / threshold calibration / post-processing 方向。

当前下一步：后续是否进入 post-processing / area calibration / threshold calibration，由主线对话决定。

---

## 第 7.22 步：calibrated_mu decoder threshold calibration

状态：已完成。

目标：在不重新训练的前提下，对第 7.21 的 calibrated_mu standard / enhanced decoder checkpoint 做 evaluation-level threshold calibration，判断面积高估是否能通过 mask threshold 调整缓解。

本轮设置：

* 不重新训练；
* 不修改 `train_pinn.py`；
* `evaluate_pinn.py` 已支持 `--mask-threshold`，默认保持 500.0，本轮未改代码；
* 不修改 `CURRENT_BASELINE`；
* validation set 用于选择 threshold；
* test set 只用于最终验证；
* threshold 使用 raw μ_r：300 / 350 / 400 / 450 / 500 / 550 / 600 / 650 / 700。

validation 推荐 threshold：

* standard decoder：400；
* enhanced decoder：350。

test set 结果摘要：

* standard threshold 500 -> 400：area_error 从 0.829989 降到 0.513358，IoU 从 0.348739 升到 0.350794，Dice 从 0.491443 小幅降到 0.488918；
* enhanced threshold 500 -> 350：area_error 从 0.953397 降到 0.416337，IoU 从 0.354685 升到 0.355723，Dice 从 0.500730 小幅降到 0.496510；
* enhanced `pred_area>true_area` 从 190.0 降到 146.67；
* small polygon IoU=0 和 small `pred_area=0` 略有恶化。

结论：threshold calibration 能明显缓解 area_error，且没有明显牺牲 overall IoU / Dice。enhanced decoder 在 calibrated threshold 下的面积高估大幅缓解，但本轮仍不切换 `CURRENT_BASELINE`。如果后续接受 evaluation-level calibration，可继续做 area calibration / threshold calibration / post-processing；如果不接受后处理阈值校准，enhanced decoder 仍只作为结构消融记录。

---

## 第 7.23 步：calibrated_mu adaptive threshold calibration

状态：已完成。

目标：缓解第 7.22 中 global threshold 对 small polygon 的副作用，测试是否可以只根据 default threshold=500 下的 predicted area 做 sample-wise adaptive threshold calibration。

本轮设置：

* 不重新训练；
* 不修改 `train_pinn.py`；
* 不修改 `CURRENT_BASELINE`；
* validation set 用于搜索 rule；
* test set 只用于最终验证；
* adaptive rule 只能使用 default `pred_area`，不能使用 true area。

validation 选出的 adaptive rule：

* standard：A=9.654345，B=12.387713，T_small=450，T_medium=350，T_large=350；
* enhanced：A=9.897988，B=15.232851，T_small=350，T_medium=350，T_large=300。

test set 结果摘要：

* standard default：area_error = 0.829989，IoU / Dice = 0.348739 / 0.491443；
* standard global：area_error = 0.513358，IoU / Dice = 0.350794 / 0.488918；
* standard adaptive：area_error = 0.474476，IoU / Dice = 0.347960 / 0.485609；
* enhanced default：area_error = 0.953397，IoU / Dice = 0.354685 / 0.500730；
* enhanced global：area_error = 0.416337，IoU / Dice = 0.355723 / 0.496510；
* enhanced adaptive：area_error = 0.360101，IoU / Dice = 0.350185 / 0.490040。

结论：adaptive threshold 能继续降低 area_error，但相比 global threshold 会进一步牺牲 IoU / Dice。standard adaptive 比 standard global 更少伤害 small polygon；enhanced adaptive 与 enhanced global 的 small polygon 指标基本相同。当前仍不切换 `CURRENT_BASELINE`。adaptive threshold 作为 evaluation-level calibration 记录。

---

## 第 7.24 步：calibrated_mu decoder + threshold calibration 阶段性 consolidation

状态：已完成。

目标：对第 7.20B 到第 7.23 的 calibrated_mu decoder 与 threshold calibration 结果做阶段性归纳。本阶段只做文档收尾，不训练、不评估、不修改 `CURRENT_BASELINE`。

阶段归类：

* `calibrated_mu + enhanced decoder`：有价值但不进入 baseline 的结构消融；
* global / adaptive threshold calibration：有价值但不进入 baseline 的 evaluation-level calibration；
* 当前没有结果足以更新 `CURRENT_BASELINE`。

enhanced decoder 的收益：

* 稳定改善 MAE；
* 在多 seed mean 上小幅改善 IoU / Dice；
* 减少 small polygon IoU=0 和 small `pred_area=0`；
* 降低 defect_mu_mean / defect_mu_median，说明缺陷区 μ 校准方向更低。

enhanced decoder 的副作用：

* MSE 更高；
* default threshold 下 area_error 更高；
* default threshold 下 `pred_area>true_area` 更多；
* 面积高估问题比 standard decoder 更明显。

threshold calibration 的收益：

* global threshold calibration 显著降低 area_error 和 `pred_area>true_area`；
* adaptive threshold calibration 进一步降低 area_error 和 `pred_area>true_area`；
* enhanced decoder 在 threshold calibration 后的面积高估大幅缓解。

threshold calibration 的代价：

* Dice 下降；
* small polygon IoU=0 和 small `pred_area=0` 风险上升；
* adaptive threshold 的 area_error 低于 global threshold，但 IoU / Dice 进一步下降，因此不全面优于 global threshold。

结论：calibrated_mu enhanced decoder 与 threshold calibration 这条线已经说明了主要 trade-off。后续不再沿着 threshold trick 继续修补；进入下一阶段前，应由主线对话重新定义实验包、接受条件和停止条件。
