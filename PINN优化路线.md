# PINN 缺陷边界形状反演 / 漏磁反演优化路线

## 当前进度

* [x] 第一步：改 data_generator
  让它生成大量样本，并保存完整参数 metadata。
  已完成：生成 train / val / test 数据集，并保存 signals、mu_maps、defect_types、metadata、x、y。

* [x] 第二步：改 train_pinn
  让网络输入 Bz 信号，而不是只输入坐标。
  目标：从 “坐标 → μ” 升级为 “Bz + 坐标 → μ”。
  已完成：加载 train / val 数据集，使用 BzEncoder 将一维 Bz signal 编码成 latent vector；
  对空间坐标使用 Fourier feature，并拼接 [bz_latent, coord_features] 后通过 MLP 输出 μ(x,y)。
  当前训练流程支持 batch 训练，只使用 MSE Loss，每轮输出 train loss / val loss；
  已保存验证集 loss 最低模型 checkpoints/best_model.pt，并输出 results/loss_curve.png 和 results/val_prediction.png。

* [x] 第三步：改 evaluate_pinn
  加入定量评价指标：IoU、Dice、面积误差、中心误差。
  目的：后续判断 TV Loss、L-BFGS、物理 Loss 是否真的有效。
  已完成：新增 evaluate_pinn.py，加载 data/training_data_test.npz 和 checkpoints/best_model.pt；
  复用 train_pinn.py 中一致的 BzEncoder、Fourier feature、PINN forward 逻辑，在整个 test 集上逐批预测 μ map。
  当前评估输出 MSE、MAE、IoU、Dice、area_error、center_error；
  已保存整体平均指标 results/evaluation_metrics.txt、逐样本指标 results/evaluation_metrics.csv，
  并输出 3 个测试样本的预测 μ map、真实 μ map、预测 mask、真实 mask 对比图。

* [ ] 第四步：加 TV Loss
  先解决重建图毛刺和背景噪点。
  目标：减少孤立斑点，让 μ map 更平滑。

* [ ] 第五步：加入 L-BFGS
  用于后期精修和降低 Loss 曲线毛刺。
  注意：作为 optional refine，不要影响基础训练流程。

* [ ] 第六步：加入物理一致性 Loss
  让预测 μ 反推得到的 Bz 和输入 Bz 匹配。
  目标：让模型更像真正的 PINN，而不是普通监督学习。

* [ ] 第七步：扩展复杂缺陷
  加入不规则、多缺陷、旋转、不同深度、不同提离高度。

## 推荐执行顺序

1. data_generator_v2.py：批量样本 + metadata + train/val/test
2. train_pinn.py：Bz + 坐标 → μ
3. evaluate_pinn.py：IoU、Dice、面积误差、中心误差
4. train_pinn.py：TV Loss
5. train_pinn.py：L-BFGS refine
6. train_pinn.py：物理一致性 Loss
7. data_generator_v2.py：复杂缺陷扩展

## 当前下一步

第四步：在 train_pinn.py 中加入 TV Loss，先解决重建图毛刺和背景噪点。

具体要求：

1. 保留现有 Bz + 坐标 → μ 的模型输入结构；
2. 在基础 MSE Loss 上加入可配置的 TV Loss；
3. 先保持训练流程简单，不引入 L-BFGS 或物理一致性 Loss；
4. 使用 evaluate_pinn.py 对比加入 TV Loss 前后的 IoU、Dice、面积误差、中心误差。
