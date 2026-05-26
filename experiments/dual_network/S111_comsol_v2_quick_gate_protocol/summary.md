# S111 COMSOL V2 quick diagnostic gate protocol

## 目的

S111 建立 COMSOL V2 quick diagnostic gate 规则，避免继续用 full V2 train / val / test 长实验盲扫 objective、output path 或 loss 组合。

## 规则

- Gate 1: 5-sample train-overfit gate。
- Gate 2: 20-train / 5-val mini generalization gate。
- Gate 3: full V2 train / val / test。

只有前一个 gate 通过，才允许进入下一 gate。

## 为什么改用该规则

S93、S97、S101 和 S109 已经显示 V2 训练常见失败模式包括：

- 全背景塌缩；
- 全前景扩张；
- soft foreground 非零但 hard threshold 不跨越；
- area / margin / endpoint selection 不能稳定解决 localization。

因此后续需要先用低成本 gate 判断一个 objective 是否有基本训练能力，而不是直接启动 3000-step full split 实验。

## 下一步建议

S112 应先在 Gate 1 上比较：

- S85 big baseline；
- direct mask head；
- bidirectional margin；
- direct mask + area。

只有 Gate 1 通过的配置才进入 Gate 2。
