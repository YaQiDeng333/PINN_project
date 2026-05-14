# 双网络变分 / weak-form 半监督支线

本 worktree 对应 `feature/dual-network-variational` 支线，用于探索主线之外的双网络反演方案。当前支线不替代 `main`，也不证明纯无监督 weak-form 反演已经成功。

## 支线定位

本支线探索：

- `phi-Net / mu-Net` 双网络结构；
- 使用 `phi-Net` 做变分场重构；
- 使用 `mu-Net` 做 weak-form material update；
- 在 weak-form runner 上加入半监督 `BCE mask prior`，用于诊断上界验证；
- 从单样本 prototype 推进到小规模 runner 与多分辨率实验；
- 保持与 `main` 主线隔离，不直接改主线训练流程。

关键边界：

- `BCE mask prior` 使用 `mu_label < 500` 构造 mask，因此属于半监督 / 诊断上界；
- 当前结果不能写成“纯 weak-form 无监督反演成功”；
- `label-informed centers` 是 oracle diagnostic，不是可部署方案；
- 本支线当前不建议直接合并进 `main`。

## 当前阶段结论

- `weak-form + area / soft Dice` baseline 可以工程跑通，但缺陷定位能力不足，容易出现过大的低 `mu` 区域。
- `BCE mask prior` 在 `20x10`、`40x20`、`80x40` 小规模 runner 中稳定显著优于 baseline。
- `BCE mask prior` 的有效性说明半监督 / 诊断上界有效，不代表无监督 weak-form 本身已经解决定位问题。
- 当前 `40x20` IoU 优先候选为 `temp25_lambda1`。
- 当前 `80x40` 综合候选为 `temp25_lambda3`；`temp20_lambda3` 可作为 IoU 优先参考。
- S29 显示，`80x40` 下的弱样本主要与形状细节、边界 / 窄缺陷样本、centroid 偏移和局部几何误差有关。

## 主要文件说明

- `DUAL_NETWORK_EXPERIMENT_LOG.md`：支线实验日志，记录 S3 之后的实验过程和结论。
- `DUAL_NETWORK_STAGE_SUMMARY.md`：阶段总结，整理当前支线能力、边界和下一阶段建议。
- `DUAL_NETWORK_TERMS.md`：术语说明，解释支线中的模型、loss、prior、指标和实验编号。
- `train_dual_variational.py`：小规模 runner，对 `.npz` 中多个 sample 独立运行双网络 weak-form loop，并输出 `metrics.csv`。
- `minimal_dual_single_sample_loop.py`：单样本 prototype，用于快速验证单样本闭环和诊断输出。
- `dual_network_models.py`：定义 `PhiNet` 和 `MuNet`。
- `dual_network_losses.py`：定义 `energy_loss`、`data_loss`、`tv_loss`、`weak_form_loss`、`generate_compact_support_test_grads` 等。
- `dual_network_data_utils.py`：读取 `.npz` 数据、构造 `coords`、构造 probe 坐标和单样本输入。
- `experiments/dual_network/`：支线实验记录、summary、metrics 和代表性图像。

## 运行示例

以下命令只是 runner 用法示例，不要求直接运行：

```powershell
python train_dual_variational.py --npz-path path/to/train.npz --output-dir experiments/dual_network/example_run --sample-indices 0,1,2 --outer-steps 30 --phi-steps 30 --mu-steps 30 --test-radius 5.0 --center-mode three --lambda-area-prior 1.0 --lambda-mask-prior 1.0 --lambda-mask-bce-prior 3.0 --mask-prior-temperature 25.0
```

## 文档语言规范

- 顶层支线文档正文默认使用中文。
- 文件名、参数名、指标名、命令和代码标识保留英文。
- 历史 `experiments/*/summary.md` 是实验产物，不强制逐个翻译。
- 从 S29 之后新增 `summary.md` 尽量使用中文正文。

## 边界说明

- 不要把 `BCE mask prior` 结果写成无监督成功。
- 不要把 `label-informed centers` 写成可部署方法。
- 不要把本支线直接同步或合并进 `main`。
- 不建议继续盲目扫描 `test_radius`、`center_mode` 或 `area prior`。
- 若继续推进，应优先整理阶段性报告，或针对 S29 的弱样本做定向失败分析。
