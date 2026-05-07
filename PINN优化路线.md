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

## 后续建议

1. 基于 v3 complex train / val 重新训练模型，保存为新的 checkpoint，不覆盖当前 baseline。
2. 使用 v3 complex test 做阶段性最终评估，并和当前 simple baseline 分开记录。
3. 训练前确认 train_pinn.py 的输入数据路径和输出 checkpoint / results 路径均指向新文件名。

## 当前下一步

详见 `NEXT_STEP.md`。
