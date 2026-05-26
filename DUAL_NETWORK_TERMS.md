# DUAL_NETWORK_TERMS

## 双网络变分支线术语说明

本文档解释 `feature/dual-network-variational` 支线中反复使用的术语。当前支线不替代 `main`。当前最稳定的结果来自半监督 `BCE mask prior`，这些结果是 diagnostic upper bound，不是纯 weak-form 无监督反演成功。

## 1. phi-Net

`phi-Net` 输入坐标 `(x, y)`，输出磁标量势 `phi(x, y)`。它负责场重构。当前支线在 `phi-step` 中用能量项、边界项和探头数据项训练它。

## 2. mu-Net

`mu-Net` 输入坐标 `(x, y)`，输出磁导率分布 `mu(x, y)`。它负责材料分布更新。当前实现把 raw 输出通过 `sigmoid` 映射到 `[mu_min, mu_max]`。

## 3. weak-form loss

`weak-form loss` 来自静磁方程 `div(mu grad phi)=0` 的弱形式。固定 `phi` 后，用测试函数梯度 `grad(v_q)` 计算 `integral mu grad(phi) dot grad(v_q) dOmega` 的残差。它是本支线的核心物理项，但当前版本单独使用时不足以稳定定位缺陷。

## 4. compact-support test functions

局部紧支撑测试函数。当前第一版使用 bump 函数，只在给定 `center` 和 `radius` 附近非零，用来构造 weak-form residual。后续可以替换为 FEM-like basis functions 或更严格的积分权重。

## 5. test_grads

`test_grads` 是测试函数梯度 `grad(v_q)`，形状通常为 `[Q, N, 2]`，其中 `Q` 是测试函数数量，`N` 是积分点数量。它不是可训练参数。

## 6. center_mode

`center_mode` 定义 compact-support 测试函数中心的布点策略。当前包括固定中心 `three/five/nine`，以及诊断用的 `signal_*`、`label_*` 模式。

## 7. test_radius

`test_radius` 是 compact-support 测试函数的半径。它影响 weak-form residual 的覆盖区域和数值尺度。当前多个实验使用 `5.0`，但这不是理论最优值。

## 8. area prior

`area prior` 用 soft defect fraction 约束预测缺陷面积。它可以抑制全域低 `mu` 塌陷，但不能单独解决定位问题。它使用 `mu_label` 的面积信息，因此属于诊断 / 半监督项。

## 9. mask prior / soft Dice prior

`mask prior` 或 soft Dice prior 使用 `mu_label < 500` 得到 label mask，并对 soft predicted defect mask 计算 Dice loss。它比 `area prior` 更局部，但仍不能完全抑制 false positives。

## 10. BCE mask prior

`BCE mask prior` 对 soft predicted defect mask 和 label mask 计算 binary cross entropy。S14-S29 中它是最有效的 false-positive 抑制信号。因为它使用 `mu_label` mask，所以它是半监督 / 诊断上界，不是无监督反演方法。

## 11. baseline

S15 之后，`baseline` 通常指 runner 中的 `weak-form + area prior + soft Dice mask prior`，但不启用 `BCE mask prior`，即 `lambda_mask_bce_prior=0.0`。

## 12. bce

`bce` 通常指在 baseline 基础上启用 `lambda_mask_bce_prior` 的半监督 / 诊断运行。不同实验中权重可能为 `1.0`、`3.0` 或其他值。

## 13. defect_area_pred

`defect_area_pred` 是阈值 `mu_pred < 500` 的预测缺陷点数。它衡量预测低磁导率区域面积。

## 14. defect_area_label

`defect_area_label` 是阈值 `mu_label < 500` 的真实缺陷点数。它来自监督标签，只用于诊断或半监督 prior。

## 15. defect_iou

`defect_iou` 是预测缺陷 mask 与 label mask 的 intersection-over-union。当前阈值为 `mu < 500`。它是当前最直观的定位诊断指标。

## 16. mu_mse / mu_mae

`mu_mse` 和 `mu_mae` 分别是 `mu_pred` 与 `mu_label` 的均方误差和平均绝对误差。当前主要作为诊断指标，不代表核心无监督训练目标。

## 17. coords

`coords` 是定义域内部点坐标，形状通常为 `[N, 2]`。支线中计算 `phi`、`mu`、梯度和 weak-form residual 都依赖它。

## 18. probe_coords

`probe_coords` 是探头线坐标。第一版使用 `y_s=10.0`，与数据生成中的 `bz_signal[-1, :]` 对齐。

## 19. bz_meas

`bz_meas` 是探头线上的测量 / 生成信号，在当前 `.npz` 中来自 `signals`。第一版把它用于 `data_loss`，不自动外推到 `lift_off` 域外位置。

## 20. mu_label

`mu_label` 是数据集中 `mu_maps` 展平后的真实磁导率标签。它用于诊断、可视化、半监督 prior 和上界实验，不是纯 weak-form 无监督训练可用的信息。

## 21. semi-supervised / diagnostic upper bound

半监督 / 诊断上界指使用了 `mu_label` 或 label mask 信息的实验。它能判断当前结构在有局部监督信号时的潜力，但不能证明无监督 weak-form 方法本身成功。

## 22. S18 / S19 / S20

- S18：20 个 `20x10` 样本的 runner probe。`BCE mask prior` 明显优于 baseline，验证了半监督上界的稳定性。
- S19：50 个 `20x10` 样本的 runner validation。`BCE mask prior` 继续稳定优于 baseline，是低分辨率下的重要证据。
- S20：20 个 `40x20` 样本的 resolution probe。BCE 平均优于 baseline，但改善弱于低分辨率，说明需要分辨率适配。

## 23. S28 / S29

- S28：50 个 `80x40` 样本的 default validation。`temp25_lambda3` 在 50/50 个样本上 IoU 都优于 baseline，并成为当前 `80x40` 综合默认候选。
- S29：只读取 S28 结果做可视化与失败样本诊断。S29 显示弱样本主要与形状细节、边界 / 窄缺陷、centroid 偏移和局部几何误差有关。

## 24. signal-conditioned dual network

`signal-conditioned dual network` 指新的支线阶段模型：先用 `BzEncoder` 把 `signals` / `Bz_meas` 编码成 `latent`，再用 `coords + latent` 预测 `mu(x,y)` 和可选的 `phi(x,y)`。它的关键边界是推理时只依赖 `Bz signal + coords`，不使用 `mu_label` 或 `label_mask`。

## 25. BzEncoder

`BzEncoder` 是 signal-conditioned 模型中的信号编码器。输入 shape 为 `[B, signal_len]`，输出 shape 为 `[B, latent_dim]`。它负责把一维测量信号压缩成跨坐标点共享的样本级条件向量。

## 26. latent

`latent` 是 `BzEncoder` 输出的样本级条件向量。对每个样本，`latent` 会 broadcast 到所有 `coords` 点，并与 `(x,y)` 拼接后送入 conditional coordinate MLP。

## 27. ConditionalMuNet

`ConditionalMuNet` 输入 `coords + latent`，输出 bounded `mu_pred`，当前骨架使用 `sigmoid` 将 raw output 映射到 `[mu_min, mu_max]`，默认范围为 `[1.0, 1000.0]`。

## 28. ConditionalPhiNet

`ConditionalPhiNet` 输入 `coords + latent`，输出 `phi_pred`。它为后续 conditional weak-form / data-loss 训练保留与 per-sample `PhiNet` 对应的接口。

## 29. per-sample optimization vs conditional model

`per-sample optimization` 指当前 S47 之前的 runner：每个样本单独优化一套 `PhiNet` / `MuNet`，可用于诊断和上界实验，但不能直接泛化。`conditional model` 指共享参数模型：训练后对新样本只需输入 `signals + coords` 即可前向预测。支线若要与主线 baseline 形成可比关系，必须推进到 conditional model 并在推理时保持 label-free。

## 30. conditional batch

`conditional batch` 指送入 `ConditionalDualNet` 的 batch 输入。当前 S49 接口包含 `signals [B, signal_len]`、共享 `coords [N,2]`、监督用 `mu_label [B,N,1]` 和 `mask_label [B,N,1]`。推理时只需要 `signals + coords`。

## 31. mask_label

`mask_label` 是由 `mu_label < 500` 得到的二值 mask，shape 为 `[B,N,1]`。它可以用于 supervised / semi-supervised conditional training，但不能作为推理输入。

## 32. signal_len

`signal_len` 是每个 `Bz_meas` / `signals` 样本的一维长度，对应 `signals.shape[-1]`。`ConditionalDualNet(signal_len=...)` 必须与输入 batch 的信号长度一致。

## 33. conditional data utils

`conditional data utils` 指 `conditional_dual_data_utils.py` 中的最小数据接口。它负责从 `.npz` 读取 `signals`、`mu_maps`、`coords` 或 `x/y`，并构造 conditional model batch；它不训练模型，也不设置 `coords.requires_grad_(True)`。

## 34. conditional supervised runner

`conditional supervised runner` 指 `train_conditional_dual.py` 这一类共享参数训练入口。它使用 `signals + coords` 前向预测 `mu` / `phi`，训练时可以使用 `mu_label` / `mask_label` 监督，推理时不应使用 label 信息。

## 35. mask BCE loss

`mask BCE loss` 是对 soft predicted defect mask 和 `mask_label` 计算 binary cross entropy。S50 中 soft predicted defect mask 定义为 `sigmoid((500.0 - mu) / mask_temperature)`。

## 36. mask Dice loss

`mask Dice loss` 是对 soft predicted defect mask 和 `mask_label` 计算的 soft Dice penalty，用于鼓励预测缺陷区域与监督 mask 重叠。

## 37. conditional training vs per-sample optimization

`conditional training` 训练一套跨样本共享的模型参数，输入是 batch `signals + coords`。`per-sample optimization` 为每个样本单独优化一套网络参数。前者才是后续与主线 baseline 形成推理成本和泛化能力对比的必要路径。

## 38. eval_metrics.csv

`eval_metrics.csv` 是 `train_conditional_dual.py` 在提供 `--eval-npz-path` 时输出的 held-out eval 指标文件。它与 train `metrics.csv` 使用相同列，但只由 forward pass 产生，不参与反向传播。

## 39. train/val conditional generalization

`train/val conditional generalization` 指共享参数 conditional model 在 train samples 上学习后，对 held-out val samples 仍保持有效 `defect_iou` 和连续 `mu` 指标的能力。它是后续与主线 baseline 比较前的必要检查。

## 40. train-val gap

`train-val gap` 指 train 指标和 val 指标之间的差距。S53 中主要看 `train avg defect_iou - val avg defect_iou`。如果 train 高而 val 低，说明 conditional model 当前更像记忆训练集，而不是泛化到新信号。

## 41. test_metrics.csv

`test_metrics.csv` is the held-out test metrics file written by `train_conditional_dual.py` when `--test-npz-path` is provided. It uses the same columns as train `metrics.csv` and optional `eval_metrics.csv`, and it is produced by forward-only evaluation with no backpropagation.

## 42. train/val/test split

`train/val/test split` means the conditional model is optimized only on train samples, selected or diagnosed on val samples, and finally checked on test samples. S54 adds the first branch-local runner support for all three splits in the conditional supervised setting.

## 43. conditional generalization

`conditional generalization` means a shared `ConditionalDualNet` trained on some `signals + coords` samples can produce useful `mu` / mask predictions on held-out signals without using `mu_label` or `label_mask` at inference time. S54 shows this remains the key bottleneck for the conditional branch.

## 44. signal_normalization

`signal_normalization` is the `train_conditional_dual.py` option that transforms input `signals` before they enter `BzEncoder`. It only changes model inputs in memory and does not modify source `.npz` data.

## 45. train_zscore

`train_zscore` computes one global mean and standard deviation from train `signals`, then applies the same normalization to train, eval, and test signals. This tests whether raw signal scale is hurting conditional generalization.

## 46. per_sample_zscore

`per_sample_zscore` normalizes each sample's signal independently along `signal_len`. It removes absolute amplitude information but keeps per-sample signal shape, so it tests whether shape is more useful than scale for the current encoder.

## 47. conditioning_mode

`conditioning_mode` selects how the conditional coordinate network uses the signal latent vector. Current options are `concat` and `film`, with `concat` kept as the default for backward compatibility.

## 48. concat conditioning

`concat conditioning` broadcasts the per-sample latent vector to every coordinate point and concatenates `[x, y, latent]` before feeding the coordinate MLP. This is the original S48-S57 conditional model behavior.

## 49. FiLM conditioning

`FiLM conditioning` uses the latent vector to generate per-layer `gamma` and `beta` parameters for hidden coordinate features: `hidden = hidden * (1 + gamma) + beta`. S58 tests this as an alternative to direct coordinate-latent concatenation.

## 50. encoder_type

`encoder_type` selects the signal encoder used by `ConditionalDualNet`. Current options are `mlp` and `cnn`, with `mlp` kept as the default for backward compatibility.

## 51. MLP BzEncoder

`MLP BzEncoder` is the original conditional signal encoder. It maps each flattened `signals [B, signal_len]` vector to `latent [B, latent_dim]` through fully connected layers.

## 52. ConvBzEncoder / CNN encoder

`ConvBzEncoder` is the S59 1D CNN signal encoder. It reshapes `signals [B, signal_len]` to `[B, 1, signal_len]`, applies `Conv1d` layers and pooling, then projects the pooled feature to `latent [B, latent_dim]`. It is tested as an alternative to the MLP encoder for conditional generalization.

## 53. point_features

`point_features` are optional coordinate-aligned features passed to the conditional coordinate MLP with shape `[B,N,K]`. They are concatenated with coordinate features before the MLP when `extra_point_dim > 0`. In S60 they carry local Bz signal values aligned to each coordinate's x position.

## 54. point_signal_mode

`point_signal_mode` is the `train_conditional_dual.py` option that controls whether local signal features are generated for each coordinate point. Current options are `none`, `local_value`, and `local_value_abs`.

## 55. local_value

`local_value` maps each coordinate x position to the nearest signal index and passes the corresponding normalized Bz value as a one-dimensional point feature.

## 56. local_value_abs

`local_value_abs` passes two local point features: the normalized local Bz value and its absolute value. It tests whether signed and magnitude-only local signal information help conditional generalization.

## 57. direct mask head

`direct mask head` is an optional conditional output head that predicts defect-mask logits directly instead of deriving the mask from `mu_pred < 500`. It is enabled by `predict_mask=True` in `ConditionalDualNet`.

## 58. mask_logits

`mask_logits` are the raw direct mask-head outputs with shape `[B,N,1]`. Applying `sigmoid(mask_logits)` gives `mask_prob`.

## 59. mask_prob

`mask_prob` is the direct defect probability predicted by the mask head. In `mask_head_mode=direct`, BCE / Dice loss and mask IoU / area metrics use this output.

## 60. mask_head_mode

`mask_head_mode` is the `train_conditional_dual.py` option selecting how masks are produced. Current options are `mu_threshold` and `direct`.

## 61. mu_threshold mask head

`mu_threshold` is the default mask mode. It uses `soft_defect = sigmoid((500 - mu) / mask_temperature)` during training and `mu < 500` for hard mask metrics.

## 62. signal_feature_mode

`signal_feature_mode` is the `train_conditional_dual.py` option that controls derived encoder inputs from the normalized Bz signal. Current options are `raw` and `raw_abs_grad`.

## 63. raw_abs_grad

`raw_abs_grad` concatenates three one-dimensional signal channels before the encoder: normalized raw Bz, `abs(Bz)`, and finite-difference Bz gradient. For an original `signal_len=L`, the encoder receives length `3L`.

## 64. finite-difference Bz gradient

`finite-difference Bz gradient` is the signal-axis derivative used by `raw_abs_grad`. Interior points use central difference; the two endpoints use forward and backward difference so the gradient channel keeps the same length as the original signal.

## 65. derived Bz features

`derived Bz features` are in-memory transformations of an existing single Bz signal. S63 uses them to test whether richer signal representation helps conditional generalization without changing the source `.npz` files or introducing multi-height / COMSOL data.

## 66. multi-height Bz

`multi-height Bz` means a conditional input signal that contains Bz measurements at multiple lift-off heights. It is represented as multi-channel `signals [num_samples, num_channels, signal_len]` before flattening.

## 67. lift-off

`lift-off` is the sensor height or probe distance from the inspected plane. Different lift-off values can expose complementary magnetic-field information for the same defect.

## 68. multi-channel signals

`multi-channel signals` are Bz inputs with shape `[num_samples, num_channels, signal_len]`. S64 flattens them channels-first to `[B, num_channels * signal_len]` for the current encoder.

## 69. signal_channels

`signal_channels` records how many Bz channels are present before flattening. A channel can represent a lift-off height, field component, probe line, or another COMSOL-style signal source.

## 70. flattened signal length

`flattened signal length` is the encoder input length after data utilities normalize signal shape. For single-channel `[B,L]` it is `L`; for multi-channel `[B,C,L]` it is `C * L`.

## 71. synthetic multi-height proxy

`synthetic multi-height proxy` is a diagnostic multi-channel signal built from one existing Bz signal by smoothing and scaling it into several channels. It is not physical COMSOL multi-height data.

## 72. smooth_decay_proxy

`smooth_decay_proxy` is the S65 proxy-builder mode. It creates channels `[raw, smooth3 * 0.8, smooth7 * 0.6]` from each single-channel Bz signal.

## 73. signal_channel_names

`signal_channel_names` names the pre-flattening signal channels in a multi-channel `.npz`. In S65 they are `raw`, `smooth3_decay0.8`, and `smooth7_decay0.6`.

## 74. COMSOL multi-height Bz

`COMSOL multi-height Bz` refers to Bz signals exported from COMSOL at multiple lift-off heights or related measurement channels. In the S66 schema it is represented as `signals [num_samples, num_channels, signal_len]`.

## 75. lift_off_values

`lift_off_values` is recommended metadata listing the physical probe heights for each signal channel.

## 76. field_components

`field_components` is recommended metadata listing the magnetic-field component represented by each channel, such as `Bz` or `Bx`.

## 77. probe_line_y_values

`probe_line_y_values` is recommended metadata for multi-line probe exports. It records the y location of each probe line when signals come from multiple spatial lines.

## 78. comsol_multiheight schema

`comsol_multiheight schema` is the S66 `.npz` convention for real or mock COMSOL-style multi-channel data: required fields are `signals`, `mu_maps` or `masks`, and `x/y` or `coords`, with 3D `signals [N,C,L]` and recommended physical metadata.

## 79. COMSOL long CSV

`COMSOL long CSV` 是 S67 使用的 signal export table format。每一行保存一个 sample、一个 channel、一个 x 位置上的 signal value，核心列包括 `sample_index`、`channel_index`、`x_index`、`value`。

## 80. sample_index

`sample_index` 表示 COMSOL long CSV 中该行属于哪个 defect sample。converter 会按该字段分组，构造 `signals [N,C,L]` 的第一个维度。

## 81. channel_index

`channel_index` 表示 signal channel，例如不同 lift-off height 或 field component。S67 converter 按 `channel_index` 升序排列 channels。

## 82. x_index

`x_index` 表示 probe line 上的位置点。S67 converter 要求每个 sample / channel pair 都包含同一组完整的 `x_index` values。

## 83. CSV to NPZ converter

`CSV to NPZ converter` 指 `convert_comsol_multiheight_csv_to_npz.py`。它把 COMSOL-style long signal CSV 与包含 `mu_maps` 或 `masks` 的 target `.npz` 合并，输出兼容 S66 schema 的 multi-channel `.npz`。

## Boundary Notes

- 当前支线不替代 `main`。
- 当前不能声称纯 weak-form 无监督反演成功。
- `label-informed centers` 是 oracle diagnostic，只用于判断 center localization 是否是瓶颈。
- `BCE mask prior` 使用 `mu_label < 500`，因此属于半监督 / 诊断上界。
- 当前 COMSOL V2 推进重点已从 dense mask loss 小修转向 parametric inverse route。S121-S130 已完成 architecture、per-sample diagnostics 和 set-matching 诊断；下一步建议转向 forward consistency / differentiable rasterization、geometry-aware rotation/type objective 或更明确的 slot/query decoder，而不是继续盲扫 dense mask margin / area / focal / sampling。


## S78 之后新增术语

- `mask_source`: conditional batch / runner 中选择 `mask_label` 来源的参数。当前支持 `mu_threshold` 和 `masks`。
- `mu_threshold mask source`: 默认 mask 来源，使用 `mu_label < mu_threshold` 构造 `mask_label`。本支线默认 `mu_threshold=500.0`。
- `provided masks`: `.npz` 文件中直接提供的 `masks` 字段，训练时可通过 `mask_source=masks` 使用。
- `target/mask consistency diagnostics`: 比较 `mu_maps < threshold` 与 provided `masks` 是否一致的诊断步骤，用于排查训练 target 是否存在定义差异。
- `train-fit adaptation probe`: 当 train IoU 偏低时，测试训练步数、点采样数量或 loss 权重是否限制 train-set 拟合能力的诊断实验。
## S91-S94 small-label runner terms

- `mask_bce_mode`: conditional runner 中选择 mask BCE 形式的参数。当前支持 `bce`、`pos_weighted_bce`、`focal_bce`。默认 `bce` 保持旧行为。
- `pos_weighted_bce`: 对 `mask_label=1` 的点施加 `pos_weight` 的 BCE 诊断 loss，用于测试 small-label class imbalance 是否限制训练。
- `focal_bce`: 使用 `focal_gamma` 和 `focal_alpha` 调整 easy / hard examples 权重的 focal-style BCE。
- `point_sampling_mode`: conditional runner 中选择训练坐标采样策略的参数。当前支持 `random` 和 `positive_balanced`。
- `positive_balanced`: 训练点采样模式，按 batch 内任一样本为正的空间坐标聚合正类点，再按 `positive_fraction` 尽量采样正负点。
- `positive_fraction`: `positive_balanced` 采样时目标正类点比例。它只影响训练 loss，不影响 eval / test full-grid metrics。
## S95-S98 train dynamics and curriculum terms

- `training_history.csv`: `train_conditional_dual.py` 可选输出的训练动态记录文件，由 `--history-interval` 控制。它记录 phase、step、loss、batch IoU、batch area、`mu` 范围和当前 loss/sampling 配置。
- `history_interval`: 每隔多少 training step 写一行 `training_history.csv`。默认 `0` 表示不写 history，保持旧行为。
- `pretrain/fine-tune curriculum`: 先用 `--pretrain-npz-path` 指定的数据训练 `pretrain_steps`，再用主 `--npz-path` 数据训练 `steps`。两个阶段使用同一个模型和 optimizer，不保存中间权重。
- `pretrain_npz_path`: curriculum 中的预训练 `.npz` 路径。当前用于测试 V1 COMSOL data 是否能作为 V2 bridge。
- `V1-to-V2 curriculum bridge`: 使用 V1-like larger ellipsoid COMSOL 数据预训练，再在 V2 small-label multi_defect 数据上 finetune 的诊断实验。
- `full-background collapse`: mask prediction 最终变为全背景，表现为 `defect_area_pred=0` 和 IoU=0。S97 history 显示 V2 finetune 会出现该现象。

## S99-S102 area calibration terms

- `area_loss_mode`: `train_conditional_dual.py` 中控制 foreground area calibration 的参数。当前支持 `none`、`batch_ratio_mse` 和 `foreground_floor`。
- `lambda_area_loss`: area loss 的权重。默认 `0.0`，表示不改变旧训练行为。
- `batch_ratio_mse`: 对 sampled training step 中的 soft predicted foreground ratio 和 label foreground ratio 计算 MSE 的 area loss。
- `foreground_floor`: 只惩罚 predicted foreground ratio 低于 `foreground_floor_ratio * true_ratio` 的 area loss，用于测试是否能避免全背景塌缩。
- `foreground_floor_ratio`: `foreground_floor` 模式下的前景面积下限比例。
- `pred_area_soft_mean`: `training_history.csv` 中记录的 soft foreground 面积均值，来自 `soft_defect` 的求和。
- `true_area_mean`: `training_history.csv` 中记录的 label foreground 面积均值，来自当前 training step 的 `mask_label`。
- `hard foreground`: metrics 中由 hard threshold 得到的预测前景面积，即 `defect_area_pred`。S101 显示 soft foreground 非零并不必然恢复 hard foreground。

## S103-S106 threshold-margin terms

- `threshold_margin_mode`: `train_conditional_dual.py` 中控制 hard `mu_threshold` crossing objective 的参数。当前支持 `none`、`positive_hinge` 和 `bidirectional_hinge`。
- `lambda_threshold_margin`: threshold-margin loss 权重。默认 `0.0`，表示保持旧行为。
- `positive_hinge`: 只约束正样本点的 margin loss，推动 defect points 的 `mu_pred` 低于 `mu_threshold - positive_mu_margin`。
- `bidirectional_hinge`: 同时约束正样本和负样本的 margin loss，正样本低于 `mu_threshold - positive_mu_margin`，负样本高于 `mu_threshold + negative_mu_margin`。
- `positive_mu_margin`: 正样本相对 `mu_threshold` 的目标 margin。
- `negative_mu_margin`: 负样本相对 `mu_threshold` 的目标 margin。
- `sampled_mu_positive_mean`: training history 中当前 sampled positive points 的 `mu_pred` 均值，用于判断正样本是否跨过 threshold。
- `sampled_mu_negative_mean`: training history 中当前 sampled negative points 的 `mu_pred` 均值，用于判断背景是否被错误压到 threshold 以下。
- `soft-hard mismatch`: soft foreground 面积非零，但 hard `defect_area_pred=0` 的现象。S103 显示 S101 三组都存在该问题。

## S112-S120 parametric inverse terms

- `parametric inverse route`: COMSOL V2 当前主推的 geometry-aware 路线。模型先从 multi-height Bz signals 预测 component-level geometry parameters，再由参数 rasterize 成 mask 做 IoU / Dice 评估，用于避开 dense sparse-mask 训练中的全背景 / 全前景塌缩。
- `defect_params`: V2 数据中的缺陷几何 metadata。当前优先读取 split 目录中的 `defect_params.csv`，其中 `source_component_json` 提供 component-level type、center、axis、depth 和 rotation 信息。
- `max_components`: 每个 sample 最多建模的 component 数。S113-S120 使用 `max_components=3`，当前 V2 train / val / test 均无 component 截断。
- `presence_targets`: shape 为 `[N,max_components]` 的 binary targets，表示每个 component slot 是否存在真实 component。
- `type_targets`: shape 为 `[N,max_components]` 的 component type label。当前 `type_vocab=[rectangular_notch, rotated_rect]`。
- `continuous_targets`: shape 为 `[N,max_components,P]` 的连续几何参数 targets。S113 raw schema 为 `center_x, center_y, axis_x, axis_y, depth_or_shape_param, rotation_angle`；S118 refined schema 将 angle 改为 `rotation_sin, rotation_cos`。
- `type_vocab`: component type 到 integer label 的稳定映射。train / val / test 必须保持一致，否则 type CE 和 metrics 不可比。
- `component sorting`: 为降低 permutation ambiguity，每个 sample 内 component 按 `center_x`、再按 `center_y` 排序后填入 slots。
- `oracle rasterization`: 使用 ground-truth parametric targets 直接 rasterize mask，并与 target mask 比较。S117 用它检查 target+rasterizer 的理论上限。
- `oracle mask IoU`: oracle rasterization 得到的 mask IoU。S117 train / val / test avg oracle IoU 约为 `0.723` / `0.723` / `0.717`，通过 `0.70` gate，但仍显示 rasterizer 是近似表达。
- `rasterizer gap`: oracle mask IoU 低于 1 的差距，可能来自 axis 语义、rotation、notch 近似、边界离散化或 defect_params 不足以完全表达 target mask。
- `rotation_sin` / `rotation_cos`: S118 refined target 中的 angle encoding，用 `sin(angle)` 和 `cos(angle)` 替代 raw `rotation_angle`，用于减少角度周期边界问题。
- `component-specific heads`: planned S122 方向。不同 component slot 或 component type 使用更分离的 prediction heads，以减少 slot/type 之间的互相干扰。
- `CNN1D signal encoder`: planned S122 方向。用 1D convolution 处理 flatten 前后的 Bz signal 结构，目标是比普通 MLP 更好提取局部 probe pattern。
- `attention pooling`: planned S122 方向。对 signal sequence 或 channel features 做 attention-style 聚合，用于增强 multi-height Bz signal 的条件表示。
- `forward consistency`: planned 后续方向。用预测 geometry 参数再生成或约束与观测 Bz 一致的 forward-style objective；当前 S112-S120 尚未实现。
- `geometry-aware decoder`: 泛指输出显式几何参数、component slots 或可 rasterize shape 的 decoder。parametric inverse route 是当前 geometry-aware decoder 的第一版实现。

## S126-S130 parametric diagnostics and set matching terms

- `prediction export`: `train_comsol_parametric_inverse.py` 的 `--export-predictions` 输出模式，用于写出 per-sample / per-component prediction CSV 和 per-sample mask metrics，不保存模型权重或 checkpoint。
- `per-component prediction CSV`: S126 输出的 `train_predictions.csv` / `val_predictions.csv` / `test_predictions.csv`。每行对应一个 sample 的一个 component slot，包含 presence、type、center、axis、depth、rotation 和误差。
- `prediction mask metrics`: S126 输出的 per-sample mask metrics，包含 `pred_mask_iou`、`pred_dice`、`oracle_mask_iou`、`oracle_gap`、`target_area`、`pred_area` 和 type sequence。
- `grouped diagnostics`: S127 根据 prediction export 按 `type_true`、`component_slot`、rotation error bin、target area bin 和 oracle gap bin 汇总错误，用于定位 parametric route 的主要失败模式。
- `worst_samples.csv`: S127 输出的低 `pred_mask_iou` 样本列表，用于人工检查最差样本是否来自 type、rotation、area 或 oracle gap。
- `component_matching_mode`: `train_comsol_parametric_inverse.py` 中控制 component loss 对齐方式的参数。当前支持 `fixed` 和 `permutation_min`。
- `fixed component matching`: 默认模式，预测 slot 与 target slot 一一对应。当前 S115/S126/S129 fixed baseline 使用该模式。
- `permutation_min`: S128 新增的 set-matching loss，对 `max_components=3` 枚举所有 6 个 target slot permutation，并选择 component loss 最小的 permutation 反传。
- `matched_slot`: prediction export 中记录的 target slot id。`fixed` 模式下等于 `component_slot`，`permutation_min` 模式下记录最小匹配选择的 target slot。
- `slot/order ambiguity`: component target 排序和预测 slot 对齐不唯一导致的学习困难。S129 显示简单 loss-side `permutation_min` 没有改善当前 parametric route。
- `slot/query decoder`: planned 后续方向，指显式设计 component slot 或 query-based decoder，而不是仅在 loss 侧做 permutation matching。

## S131-S135 differentiable raster supervision terms

- `differentiable rasterizer`: 用 PyTorch tensor 运算把 predicted component geometry 转成 soft mask 的可微模块，使 mask loss 可以反传到 center、axis、rotation 和 presence。
- `soft rasterization`: 使用 sigmoid soft rectangle 近似 hard rectangle 边界。S132 中 `softness_cells` 控制边界软化宽度，单位是 grid cell。
- `raster mask loss`: 在 parametric inverse training 中对 soft rasterized mask 与 target mask 计算的 BCE / Dice loss。它是 mask supervision，不是 COMSOL forward consistency。
- `lambda_raster_bce`: `train_comsol_parametric_inverse.py` 中 raster BCE loss 权重，默认 `0.0`。
- `lambda_raster_dice`: `train_comsol_parametric_inverse.py` 中 raster Dice loss 权重，默认 `0.0`。
- `raster_softness_cells`: soft rasterizer 的边界 softness，按 x/y 平均 grid spacing 换算。
- `raster_target_source`: raster loss 的 target 来源。当前支持 `masks` 和 `mu_threshold`。
- `soft union`: 多 component soft mask 合成方式，S132 使用 `1 - product(1 - component_prob)`。
- `forward consistency`: 比 raster mask loss 更进一步的物理一致性目标，要求预测 geometry 通过 forward model 或 surrogate 解释 Bz signal；S132-S135 尚未实现。

## S136-S139 two-stage raster fine-tune terms

- `two-stage raster fine-tune`: 先用 parameter-only objective 训练 parametric inverse model，再在后期加入 differentiable raster mask loss 做 geometry / mask calibration 的训练策略。
- `raster_loss_start_step`: `train_comsol_parametric_inverse.py` 中控制 raster BCE / Dice loss 从第几个 step 开始启用的参数。默认 `0` 表示如果 raster loss 权重非零，则从训练开始启用；权重为零时该参数无作用。
- `raster_loss_active`: `training_history.csv` 中记录当前 step 是否实际启用了 raster loss 的字段。
- `validation-aware endpoint selection`: 训练中周期性评估 validation set，并把最佳模型状态保存在内存中，训练结束后恢复该 endpoint 再输出最终 metrics。当前不保存权重或 checkpoint。
- `val_selection_metric`: `train_comsol_parametric_inverse.py` 中控制 endpoint selection 指标的参数，当前支持 `none`、`val_mask_iou` 和 `val_loss`。
- `val_selection_interval`: 每隔多少 step 计算一次 validation selection 指标；如果启用 selection，必须大于 0。
- `best_step`: validation-aware endpoint selection 选中的训练 step。
- `best_val_mask_iou`: 使用 `val_selection_metric=val_mask_iou` 时记录的最佳 validation mask IoU。

## S141-S144 physics feature fusion terms

- `physics_features`: 从 multi-height Bz signal 中显式提取的 physics-inspired scalar features，用于辅助 parametric inverse model。
- `peak_abs`: 单个 signal channel 中 `abs(signal)` 的最大值，用于描述 MFL 响应强度。
- `peak_to_peak`: `max(signal) - min(signal)`，用于描述正负峰跨度。
- `half_abs_width`: `abs(signal) >= 0.5 * peak_abs` 的 x 范围宽度，用于近似 defect response 的空间宽度。
- `lift-off decay ratio`: 不同高度 / channel 的峰值或能量相对 ch0 的比例，例如 `peak_abs_ch1_over_ch0` 和 `energy_ch2_over_ch0`。
- `inter-channel correlation`: 不同 Bz channel 之间的 Pearson correlation，用于描述 multi-height waveform 是否保持形状一致。
- `feature_fusion_mode`: parametric inverse runner 中控制 raw signal 与 physics features 如何融合的参数，当前计划支持 `none`、`features_only` 和 `concat_latent`。

## S146-S150 learned forward consistency terms

- `learned forward surrogate`: 用 supervised learning 近似 `geometry parameters -> multi-height Bz signal` 的轻量模型。当前只用于诊断和 consistency referee，不替代 COMSOL solver。
- `geometry_vector`: 将 fixed-order component geometry 展平后的向量，包含每个 slot 的 `presence`、type one-hot / probabilities 和 continuous geometry parameters。
- `forward consistency`: 要求 inverse model 预测出的 geometry 经 learned forward surrogate 后能够重建输入 Bz signal 的约束。
- `signal_nrmse_raw`: 在反归一化 raw Bz signal 上计算的 normalized RMSE，用于衡量 surrogate signal reconstruction error。
- `signal_corr`: predicted signal 与 true signal 的整体 Pearson correlation，用于判断 waveform 是否被 surrogate 捕捉。
- `peak_abs_nrmse`: predicted signal 与 true signal 的 absolute peak amplitude 误差归一化指标，用于检查 MFL peak scale 是否被 surrogate 捕捉。

## S151-S155 residual and targeted supervision terms

- `forward residual sensitivity`: 比较 true geometry、predicted geometry 和人为扰动 geometry 经 learned forward surrogate 后的 signal residual，用于判断 residual 是否能区分几何错误。
- `type_swapped_geometry`: 将 present component 的 defect type 替换为另一个 type 的诊断 variant，用于测试 forward residual 对 type 错误的敏感性。
- `rotation_perturbed_geometry`: 将 `rotation_angle` 加固定角度扰动的诊断 variant，用于测试 forward residual 对 rotation 错误的敏感性。
- `axis_scaled_geometry`: 将 `axis_x/y` 乘固定比例的诊断 variant，用于测试 forward residual 对尺寸/面积相关几何错误的敏感性。
- `lambda_type_extra`: `train_comsol_parametric_inverse.py` 中额外 type CE loss 权重，默认 `0.0`。
- `lambda_rotation_extra`: `train_comsol_parametric_inverse.py` 中额外 rotation-specific loss 权重，默认 `0.0`。
- `rotation_loss_mode`: rotation targeted loss 模式，当前支持 `mse` 和 `circular`。

## S160-S165 center localization terms

- `center localization bottleneck`: S158 oracle ablation 显示 `gt_center` 几乎关闭 held-out baseline 到 oracle 的主要 mask IoU gap，因此当前 parametric route 的主要瓶颈是 component center prediction。
- `center_grid_mae`: 将 `center_x/center_y` 的 meter error 分别除以 x/y grid spacing 后计算的 center L2 error，用 grid-cell units 表示。
- `center_axis_relative_mae`: 将 center error 分别除以 GT `axis_x/axis_y` full-width/full-height 后计算的 axis-relative center L2 error。
- `lambda_center_grid`: `train_comsol_parametric_inverse.py` 中额外 center grid-cell MSE loss 的权重，默认 `0.0`，只对 present components 生效。
- `lambda_center_axis_relative`: `train_comsol_parametric_inverse.py` 中额外 axis-relative SmoothL1 center loss 的权重，默认 `0.0`，只对 present components 生效。
- `center_axis_relative_eps`: axis-relative center loss 的分母稳定项，默认 `1e-6`。
- `center-bin classification + offset`: planned center representation，将 center 先预测到离散 grid/bin，再回归局部 offset，用于替代单纯 global coordinate regression。
## S166-S170 center-grid stability terms

- `seed`: `train_comsol_parametric_inverse.py` 的 reproducibility 参数。默认 `0`，用于 `torch`、`numpy` 和 Python `random`。
- `existing_unrecorded`: S164 产生的 center-grid full run，因为当时没有 CLI seed 记录，只能作为 existing run 复用，不能假定为 seed0。
- `center-grid stability repeat`: 对 `lambda_center_grid=0.1` 做少量 seed repeat，验证改善是否不依赖单次随机初始化。
- `current COMSOL parametric route candidate`: 当前支线内最值得作为后续 reference 的 parametric 配置，不等同于 main baseline replacement。

## S171-S175 center-grid candidate consolidation terms

- `center-grid candidate`: the branch-local COMSOL parametric candidate defined as raw MLP / shared head / fixed-order + `lambda_center_grid=0.1`.
- `branch-local candidate`: a result accepted as the current reference for `feature/dual-network-variational`, not a main baseline replacement.
- `candidate reproduction command`: the explicit `train_comsol_parametric_inverse.py` command in `DUAL_NETWORK_REPRODUCE.md`; it keeps raster loss, forward consistency, and validation selection off.
- `center-bin classification + offset`: the next recommended center representation route after consolidating the center-grid candidate.

## S176-S180 center-bin offset terms

- `center_representation`: runner mode for center prediction. `continuous` preserves the previous coordinate regression path; `bin_offset` uses discrete center bins plus local offsets.
- `center_bin_size_cells`: number of x/y grid cells per coarse center bin. S176-S180 uses `8`.
- `center_x_bin_logits` / `center_y_bin_logits`: per-axis bin classification outputs for each component center.
- `center_offset`: per-component normalized offset from the selected bin center, decoded by multiplying by the x/y bin width.
- `center_bin_offset_plus_grid`: S178/S179 configuration combining bin CE, offset SmoothL1, and decoded-center `lambda_center_grid=0.1`.

## S181-S185 center-bin stability terms

- `center-bin offset stability repeat`: multi-seed validation of `center_bin_offset_plus_grid` against the S170 center-grid candidate range.
- `S170 center-grid range`: the historical comparison range from the three S170 candidate runs; S181-S185 uses it as the promotion baseline instead of rerunning the reference.
- `center-bin current candidate`: after S185, the branch-local COMSOL parametric candidate is raw MLP / shared head / fixed-order with `center_representation=bin_offset`, `center_bin_size_cells=8`, `lambda_center_bin=1.0`, `lambda_center_offset=1.0`, and `lambda_center_grid=0.1`.

## S189-S193 signal-to-center auxiliary terms

- `signal-to-center auxiliary head`: optional head attached to the inverse latent that directly predicts center x/y bins and bin-normalized offsets, without replacing the main center-bin output.
- `aux_center_head`: runner flag enabling the auxiliary center head. It defaults to off and requires `center_representation=bin_offset`.
- `lambda_aux_center_bin`: auxiliary x/y center-bin CE loss weight.
- `lambda_aux_center_offset`: auxiliary center-offset SmoothL1 loss weight.
- `aux_center_x_weight` / `aux_center_y_weight`: per-axis weights applied to auxiliary bin CE and offset components.
- `aux_center_x_bin_accuracy` / `aux_center_y_bin_accuracy`: auxiliary head bin accuracy metrics, kept separate from the main `center_x_bin_accuracy` / `center_y_bin_accuracy`.

## S198-S202 x-bin calibration terms

- `center_bin_x_weight`: optional weight for the main x-axis center-bin CE term. It defaults to `1.0` and does not affect auxiliary head losses.
- `center_bin_y_weight`: optional weight for the main y-axis center-bin CE term. It defaults to `1.0`; with both x/y weights at `1.0`, the main bin CE stays equivalent to the previous `0.5 * (x_loss + y_loss)`.
- `center_bin_slot_weights`: optional comma-separated per-component-slot weights for main center-bin CE, for example `1.5,1.0,1.5` when `max_components=3`. It is static, fixed-order, and only applies to present components.
- `x-bin calibration`: the S198-S202 probe that increases pressure on main `center_x_bin_logits` without changing model structure, offset loss, decoded center, rasterization, or prediction export.
- `slot-aware weighting`: static weighting by component slot. In S200, slot-aware x-bin weighting did not improve held-out IoU and should not be promoted without a later multi-seed validation.

## S247-S253 polygon geometry terms

- `polygon geometry route`: V3 true-geometry representation route that uses explicit component vertices instead of `center + axis + rotation` as the main oracle target.
- `polygon_params.csv`: COMSOL export table with one row per sample, component, and vertex. It records raw and normalized vertices, component slots, COMSOL geometry tags, selections, hard-case type, and true rotated / true multi-component audit flags.
- `polygon_vertices_raw`: `[N,max_components,max_vertices,2]` array of COMSOL raw-space vertices, kept for geometry audit.
- `polygon_vertices_norm`: `[N,max_components,max_vertices,2]` array of V2-compatible normalized vertices, used by polygon oracle rasterization and future polygon inverse training.
- `polygon_vertex_mask`: `[N,max_components,max_vertices]` binary mask indicating valid vertices for each present component.
- `clockwise_top_left`: vertex ordering convention: clockwise polygon order, starting from the normalized-space top-left-like corner, defined by minimum `y` and then minimum `x`.
- `hard polygon rasterizer`: point-in-polygon rasterizer that turns polygon vertices into binary masks and unions present components by boolean OR.
- `polygon oracle gate`: target/rasterizer gate requiring true COMSOL polygon smoke to reconstruct masks with per-sample IoU at least `0.95`.

## S254-S258 polygon ingest terms

- `embedded polygon targets`: polygon arrays stored directly in converted COMSOL V3 NPZ files, including `polygon_vertices_raw`, `polygon_vertices_norm`, `polygon_vertex_mask`, and `polygon_presence`.
- `wide polygon_params.csv`: one-row-per-present-component polygon audit table used by the S254 pack, with corner columns `raw_x0/raw_y0` through `raw_x3/raw_y3` and `norm_x0/norm_y0` through `norm_x3/norm_y3`.
- `comsol_polygon_target_utils.py`: bridge utility that validates embedded polygon arrays against the wide polygon audit table and writes rasterizer-ready `polygon_targets.npz`.
- `polygon V3 ingest gate`: S254-S258 gate requiring finite repaired Bz signals, exact mask threshold agreement, complete polygon component coverage, and train/val/test polygon oracle IoU at or above `0.95`.

## S259-S263 polygon inverse terms

- `polygon inverse runner`: independent supervised runner that maps multi-height Bz signals to fixed-slot polygon vertices, separate from the old `center + axis + rotation` parametric runner.
- `vertices_norm`: predicted normalized polygon vertices with shape `[B,3,4,2]`, matching `polygon_vertices_norm` targets.
- `present-vertex SmoothL1`: vertex regression loss applied only to present component slots and valid vertices.
- `polygon mask IoU`: hard-rasterized mask IoU computed from predicted polygon vertices; it is an evaluation metric in S259-S263, not a differentiable loss.
- `vertex-to-raster sensitivity`: failure mode where vertex MAE is small but hard polygon IoU remains below gate because small edge shifts change discrete raster area.

## S264-S268 polygon raster-sensitivity repair terms

- `vertex_loss_space`: `train_comsol_polygon_inverse.py` option controlling whether vertex SmoothL1 is computed in normalized coordinate units (`norm`) or grid-cell units (`grid`). Default is `norm`.
- `grid-space vertex loss`: vertex loss that scales x/y errors by raster grid spacing before SmoothL1, aligning continuous vertex regression with hard raster pixel sensitivity.
- `lambda_area_aux`: optional polygon area auxiliary loss weight. It defaults to `0.0`, uses torch shoelace area in grid-cell units, and requires four valid vertices for each present component.
- `lambda_edge_aux`: optional polygon edge-length auxiliary loss weight. It defaults to `0.0`, compares the four edge lengths in grid-cell units, and requires four valid vertices for each present component.
- `longer_overfit`: S267 one-sample repair run that keeps the old norm-space vertex loss but increases optimization steps enough to pass the hard polygon IoU gate.

## S289-S293 center-anchored polygon terms

- `center-anchored polygon representation`: polygon inverse representation that predicts component center first, then predicts polygon vertices as local offsets relative to that center.
- `center_x_bin_targets` / `center_y_bin_targets`: discrete x/y center-bin labels for each present polygon component, using `center_bin_size_cells=8`.
- `center_offset_targets`: bin-width-normalized x/y offset from the selected center-bin center to the component polygon center.
- `local_vertices_grid`: `[N,3,4,2]` grid-cell local vertex offsets relative to the component center, decoded with `vertices_norm = center + local_vertices_grid * [dx,dy]`.
- `decoded_vertex_mae`: normalized-coordinate MAE between decoded center-anchored predicted vertices and target `polygon_vertices_norm`.
- `center-anchored polygon runner`: independent runner in `train_comsol_center_anchored_polygon_inverse.py`; it does not replace the absolute-vertex polygon runner or the S185/S181 center-bin candidate.

## S298-S302 matched-coverage split terms

- `matched-coverage resplit`: diagnostic-only reassignment of the existing polygon V3 samples so held-out component center bins are covered by train exactly or within a small center-bin distance.
- `center-bin distance`: Manhattan distance between component `(center_x_bin, center_y_bin)` pairs.
- `distance-1 coverage`: held-out component bin has a train component bin with center-bin distance `<=1`.
- `exact same-bin coverage`: held-out component bin appears exactly in the train component-bin set.

## S303-S307 y-bin localization repair terms

- `center_y_bin_extra_loss_mode`: default-off center-anchored runner option for adding a y-bin soft-target loss on top of the unchanged hard center-bin CE.
- `neighbor_soft_ce`: y-bin extra loss mode that gives the true y-bin most probability mass and distributes smoothing mass to immediate neighbor bins.
- `distance_soft_ce`: y-bin extra loss mode that uses a Gaussian soft target over y-bin distance.
- `center_y_bin_within1_acc`: metric reporting whether predicted y-bin is exact or adjacent to the target y-bin for present components.
- `center_y_bin_abs_error`: mean absolute ordered y-bin distance for present components.

## S308-S312 bounded local output terms

- `local_shape_output_mode`: center-anchored runner option controlling whether local vertex head outputs are used directly (`raw`) or passed through bounded decode (`bounded_tanh`).
- `bounded_tanh`: local-shape output mode that computes `effective_local_vertices_grid = tanh(raw_local) * [bound_x,bound_y]`.
- `local_shape_bound_mode`: mode for choosing local-shape bounds. `fixed_grid` uses CLI constants; `train_stats` computes bounds from train targets only.
- `local_shape_saturation_frac`: fraction of valid local vertex coordinates whose effective absolute value is at least `98%` of its configured bound.
- `effective local vertices`: the local grid-cell offsets actually used by local vertex loss, center/local decode, hard polygon metrics, and prediction export.

## S313-S317 local-shape conditioning terms

- `local_shape_conditioning_mode`: center-anchored runner option controlling whether the local vertex head is conditioned on predicted context. `none` preserves the previous unconditioned path; `center_bin`, `center_bin_slot`, and `center_bin_slot_type` enable progressively richer conditioning.
- `local_shape_conditioning_dim`: embedding dimension used for center-bin, slot, and type contexts in conditioned local-shape modes.
- `center-bin soft context`: differentiable summary of predicted x/y center-bin logits, computed as `softmax(center_bin_logits) @ embedding`.
- `slot local-shape conditioning`: optional component-slot embedding added to local vertex prediction context.
- `type soft context`: optional summary of predicted component type logits, computed as `softmax(type_logits) @ embedding`.
- `detached conditioning context`: conditioning features are detached before feeding the local vertex head so the local vertex loss does not train center-bin or type heads through the conditioning path.
- `joint center-bin/local-shape repair`: next recommended route after S313-S317, because simple local-shape conditioning improved train/local fit but did not improve held-out mask IoU.

## S318-S322 joint center/local diagnosis terms

- `center-anchored oracle ablation`: offline reconstruction diagnostic that replaces selected predicted center/local fields with GT targets before hard polygon rasterization.
- `gt_center_bin`: ablation variant using GT x/y center bins with predicted offset and predicted local vertices.
- `gt_offset`: ablation variant using predicted center bins with GT center offset and predicted local vertices.
- `gt_center_bin_offset`: ablation variant using GT center bins and GT center offset with predicted local vertices.
- `gt_local`: ablation variant using predicted center bins and predicted offset with GT local vertices.
- `gt_center_bin_offset_local`: full center/local GT ablation variant; it should recover polygon oracle IoU when sample joins and rasterization are aligned.
- `joint_center_shape_mode`: center-anchored runner option for joint center/local-shape prediction. Default `none` preserves the previous path.
- `soft_center_scheduled`: first joint mode tested in S321; the local-shape head receives soft continuous center context with teacher-forced center context annealed toward predicted context.
- `teacher_center_context`: GT continuous component-center context used only during scheduled joint training, not during evaluation export.

## S323-S327 decoded-center coupling terms

- `center_consistency_mode`: default-off center-anchored runner option for differentiable center/vertex consistency. `none` preserves previous behavior.
- `soft_decoded_center`: center consistency mode that decodes center as `softmax(center_bin_logits) @ bin_centers + center_offset * bin_width` and compares it to GT center in grid-cell units.
- `soft_decoded_vertex`: center consistency mode that combines soft decoded center with local vertices and compares the resulting vertices to GT polygon vertices in grid-cell units.
- `hard decoded center`: official eval/export center obtained from argmax center bins plus predicted offset.
- `soft expected center`: differentiable diagnostic/training center obtained from center-bin probability expectation plus predicted offset.
- `center-bin probability margin`: top1 center-bin probability minus top2 probability for x or y; low margin means the bin classifier is uncertain even if argmax is correct.

## S328-S332 component-query center/shape terms

- `component-query polygon inverse`: independent polygon inverse route where each fixed component slot has a learned query latent that jointly predicts component presence, type, center bins, center offset, and local polygon vertices.
- `component query`: learned fixed-slot embedding, one per `max_components=3` component slot.
- `query latent`: per-sample, per-slot hidden representation formed by combining the shared signal encoder latent with the component query embedding.
- `shared component representation`: the intended coupling mechanism in which center decode and local shape are predicted from the same slot-specific latent rather than from weakly related separate heads.
- `component-query one-sample gate`: strict hard-raster precision gate requiring IoU `>=0.99` before running 5-sample, same-run reference, or train30.
