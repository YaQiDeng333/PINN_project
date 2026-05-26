# S76 COMSOL geometry pilot summary

## S74 ingest 结论

S74 已完成真实 COMSOL geometry-variation train / val / test 数据接入。S67 converter 将 long CSV + target NPZ 转成支线可读 multi-channel NPZ，S66 validator 通过，conditional loader 可将 `[B,3,200]` signals channels-first flatten 成 `[B,600]`，并完成 `ConditionalDualNet` forward。

数据质量检查显示 train / val / test signals 和 targets 均 finite，target label area 非零且范围合理。当前缺陷参数已变化 center、axis / width、depth / shape parameter 和磁性参数，但 defect type 仍固定为 ellipsoid。

## S75 主要结果

| run | train IoU | val IoU | test IoU | test mu_mse | test mu_mae |
| --- | ---: | ---: | ---: | ---: | ---: |
| medium_multichannel | 5.225618e-01 | 4.088045e-01 | 3.961416e-01 | 8.931295e+04 | 2.355511e+02 |
| big_multichannel | 5.391816e-01 | 4.067505e-01 | 3.997817e-01 | 8.749112e+04 | 2.218037e+02 |

`big_multichannel` 的 test IoU 和 continuous mu errors 略好；`medium_multichannel` 的 val IoU 略好且 train-held-out gap 更小。两者差异不大。

## 与 synthetic conditional 阶段对比

早期 synthetic single-Bz conditional 阶段的 held-out IoU 长期接近 0.08 到 0.13；S75 在真实 COMSOL multi-height geometry-variation 数据上取得约 0.40 的 val/test IoU。这个结果支持“真实 multi-height Bz 数据比 synthetic single-Bz 输入更有潜力”的判断。

但 S75 仍是 pilot 规模，且模型没有保存 checkpoint，也没有做正式 hyperparameter search。当前结果不能直接写成最终主线替代结论。

## 当前限制

- train samples 只有 50，val/test 各 10。
- defect type 固定为 ellipsoid。
- 未变化旋转角、边界不规则度或多缺陷形态。
- `mu_maps` 当前使用 projected footprint target，而不是完整 COMSOL material field。
- runner 使用 supervised BCE + Dice mask loss，未接入 weak-form / physics loss。

## 下一步建议

1. 扩大 COMSOL geometry-variation 数据量，优先增加 train samples 到 200+，并保持独立 val/test。
2. 增加几何多样性：旋转角、长短轴比例、边界不规则度、多缺陷或非椭圆 shape。
3. 检查 target/mask 定义：确认 projected footprint 是否符合反演目标，必要时输出更物理一致的 `mu_maps`。
4. 在 runner 侧测试 validation-aware selection、loss balance、轻量 `mu_mse` 和 direct mask head 的 COMSOL 数据版本。
5. 如果扩大数据后 val/test 仍停滞，再优先排查 signal sampling line / lift-off 设计和 model conditioning 架构。
