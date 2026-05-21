# NEXT_STEP

## 当前状态

`CURRENT_BASELINE` 仍以 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 为准：

- v3_complex mask-only grid decoder + forward consistency
- `lambda_forward = 0.10`
- validation-selected probability threshold = `0.80`
- forward surrogate = `checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`

该 baseline 是当前 v3_complex 上最强 boundary-oriented baseline，但 polygon / rotated_rect 精细边界圆斑化、small / low-signal、多缺陷仍未根本解决。

内部 decoder / loss / threshold / geometry / basis / refinement 小修补已经基本收口。当前第 20 阶段已经从单纯 COMSOL 数据包验证，进一步进入 geometry-aware / forward-consistent inverse reconstruction 方法验证。当前这些工作仍是 COMSOL data-domain POC / candidate，不更新 v3_complex `CURRENT_BASELINE`。

## 最近下一步

当前不要继续普通 dense decoder patch，也不要继续单独 geometry head 小修补。第 20.48-20.55 已证明：

1. geometry labels 和 differentiable rotated-rectangle rasterizer 没有 blocker；
2. direct delta_bz-only geometry head 的 type / angle 学习不足；
3. controlled architecture sweep 没有找到有效 head 结构；
4. feature-assisted geometry head + lightweight forward consistency 只带来边际 mask / angle 改善，type confusion 仍是主因；
5. Priewald-style low-dimensional refinement 能降低 forward residual，并可小幅改善 geometry-raster mask，但 initializer / proposal 质量决定上限；
6. dense/coarse mask initializer + PCA bbox extraction 在 20.53 中没有超过 20.51 geometry-head proposal，type / angle proposal 仍弱；
7. 第 20.54 的 strong dense initializer 和 improved proposal extraction 已把 rect/rot geometry proposal 提到 test IoU/Dice `0.6726 / 0.8017`，但 Priewald-style refinement 让 test IoU/Dice 回落到 `0.6646 / 0.7958`，同时 forward NRMSE 下降，说明当前主要 blocker 已从 proposal quality 转为 forward surrogate mismatch；
8. 第 20.55 的 calibrated surrogate sweep 没有找到可用 residual objective：S2 的 waveform NRMSE 最好，但 val residual-error correlation 为负，S3 的正相关也只有 `0.0215`，未过 gate，因此 calibrated refinement 被正确跳过。

因此最近下一步优先转向：

1. **生成 synthetic perturbation forward data / 局部扰动校准数据**：当前缺少同一 geometry 附近的已知扰动与 forward response，surrogate 学不到“几何越差 residual 越高”的局部单调关系。
2. 如果不能生成扰动 forward 数据，则转向 **mask/profile basis refinement**，减少对当前低维 rect/rot geometry residual objective 的依赖。
3. 暂停继续对现有 surrogate loss、peak weighting 或 refinement objective 做小调；20.55 已说明 waveform 拟合不等于 residual 可用于 geometry refinement。
4. 继续保持 train-only normalization、validation checkpoint / threshold selection、test-only final evaluation。

如果后续 refinement 不能在更强 proposal 上稳定改善 mask / geometry，再暂停 rect/rot geometry route，等待更丰富观测、更多通道或更强 forward surrogate；当前不建立新 baseline。

## 当前不要继续的方向

不要继续围绕现有 v3_complex grid decoder 做 selection metric、ensemble、threshold trick、loss trick、decoder head、SDF / boundary head、coordinate refinement、hand-crafted Bz features、U-Net-like decoder、shape-type conditional、star-convex、retrieval、box / quad / basis / profile 或 mask-logit refinement 小修补。

也不要继续单独调 rect/rot neural geometry head。新的实验必须回答：显式 geometry representation、differentiable rasterization 和 forward residual 是否能稳定提高边界反演可辨识性，而不是只带来局部指标波动。
