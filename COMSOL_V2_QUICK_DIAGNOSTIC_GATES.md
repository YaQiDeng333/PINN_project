# COMSOL_V2_QUICK_DIAGNOSTIC_GATES

## 1. 动机

V2 long-run 搜索耗时高。S93、S97、S101 和 S109 多次显示全背景、全前景或 localization 不足。直接跑 full train / val / test 配置效率低，容易消耗大量时间却只得到“仍然不行”的结论。

后续所有 COMSOL V2 runner objective、output path 或 loss 组合，必须先通过小规模 gate，再进入更大规模实验。

## 2. Gate 1: 5-sample train-overfit gate

数据：

- V2 train samples: `0,1,2,3,4`

目的：

- 判断模型 / loss / output path 能否在极小训练集上学到非零且位置合理的 mask。

建议配置：

- `steps = 1000`
- `hidden_dim = 128`
- `num_layers = 4`
- `latent_dim = 64`
- `train_point_subsample = 4096`
- `history_interval = 100`
- 不保存权重。

通过条件：

- train IoU 明显大于 0。
- `area_pred` 非 0 且不是接近全图。
- `training_history.csv` 不出现持续全背景或全前景。
- 建议 train IoU >= 0.5 才进入 Gate 2。

## 3. Gate 2: 20-train / 5-val mini generalization gate

数据：

- train samples: `0..19`
- val samples: `0..4`

目的：

- 判断是否有初步泛化信号。

通过条件：

- train IoU 比 Gate 1 稳定。
- val IoU 非零。
- `area_pred` 不极端。
- val 不明显全背景或全前景。

## 4. Gate 3: full V2 train/val/test

只有 Gate 1 和 Gate 2 都通过后才允许运行：

- train samples = 100
- val samples = 20
- test samples = 20

## 5. 失败处理

如果 Gate 1 不通过：

- 不允许跑 full experiment。
- 转向模型结构 / output head / target 设计。
- 或先检查数据 / label / signal。

如果 Gate 2 不通过：

- 不允许 full run。
- 转向 regularization / simpler model / validation-aware selection。

## 6. 当前推荐下一步

S112-S120 已将当前主线方向从 dense mask objective 切换到 COMSOL parametric inverse route。S117 oracle rasterization 已通过，S119 refined target 没有整体改善，因此下一步不再建议继续 dense mask loss / margin / area / focal / sampling 的 quick gate 盲扫。

当前 planned S121-S125 应沿用 quick gate 思想，但对象切换为 parametric architecture：

- S121: 先做 parametric error decomposition，明确 type、rotation、component slot 和 geometry range 的误差来源。
- S122: 实现 component-specific heads、CNN1D signal encoder 或 attention pooling。
- S123: 对新 architecture 先运行 quick gate architecture probe。
- S124: 只有 quick gate 通过后，才运行 best architecture full probe。
- S125: 汇总 route decision，判断是否进入 forward consistency / differentiable rasterization。
