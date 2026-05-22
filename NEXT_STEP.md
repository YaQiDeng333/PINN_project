# NEXT_STEP

## 2026-05-22 更新：第 20.58 后的下一步

第 20.58 已完成 mask/profile basis refinement POC。profile extraction 从 predicted dense mask/probability 中提取 K=8 profile 表示，validation 选择 `P1_hardmask_profile`；profile-extracted test IoU/Dice/area_error 为 `0.6589 / 0.7921 / 0.2170`。no-forward profile refinement 只拟合 dense initial probability 并加 smoothness / area / bounds prior，test 提升到 `0.6697 / 0.8002 / 0.2196`，说明 profile basis 相比第 20.57 的 single rotated-box refinement 更稳，但没有稳定超过第 20.54 extracted rotated-box proposal `0.6726 / 0.8017 / 0.1945`。

forward profile refinement 已执行受控 sweep，但 validation 选择 `lambda_forward=0.0`，test 为 `0.6620 / 0.7938 / 0.2243`。这说明当前第 20.56/20.57 的 S1 surrogate 通过 lossy profile-to-rect summary 接入后，不能作为可靠的 profile-space forward consistency 约束。Claude Code review 通过且无必须修复；审查结论是不建议继续在当前 surrogate-dependent profile refinement 上小调。

当前下一步唯一优先级：**改进 profile-compatible forward surrogate**。如果继续 profile/basis 路线，应先让 forward model 直接接受 profile/basis 或 rasterized-profile derived features，而不是把 profile 压回单个 rect/rot summary；否则应暂停 geometry/refinement route，等待更丰富观测或更强 forward data。仍不更新 `CURRENT_BASELINE.md`，也不创建新的 COMSOL baseline 文档。

## 2026-05-22 更新：第 20.57 后的下一步

第 20.57 已完成 perturbation-calibrated surrogate 的受控 Priewald-style refinement retry。`S1_perturb_geom_mlp` 按第 20.56 protocol 重训于内存中，recovery 指标与 20.56 对齐：val/test waveform NRMSE 为 `0.3666 / 0.4289`，residual ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 为 `0.2679 / 0.1964`。

但是连续低维 refinement 没有通过 gate。validation 上 8 个 config 全部导致 mask 指标退化或 mismatch 过高，最终仅选最高分 config 做 diagnostic：`steps=50, lr=0.003, lambda_prior=0.10`。test geometry-raster IoU/Dice/area_error 从 `0.6726 / 0.8017 / 0.1945` 变为 `0.6492 / 0.7829 / 0.2417`；forward NRMSE 下降 `0.0713`，但 mismatch_rate 为 `0.6212`，residual reduction 与 IoU/Dice delta 相关性为 `-0.1824 / -0.2250`。

当前判断：20.56 的 pairwise residual ordering 改善没有转化为可用的连续 geometry optimization 梯度。不要继续在当前 rect/rot low-dimensional refinement objective 上小调 steps / lr / prior；也不要回到 direct geometry head 或 dense baseline patch。最近下一步优先转向 **mask/profile basis refinement**，降低对 single rect/rot parameter residual landscape 的依赖。若未来重新尝试 Priewald-style refinement，应先扩大 perturbation pack 或加入 richer observations，再重新验证 residual landscape。

## 2026-05-22 更新：第 20.56 后的下一步

第 20.56 已生成小规模 local geometry perturbation forward-calibration pack，并完成 surrogate residual ordering audit。实际 COMSOL pack 是 96 行 partial pack（12 个 base，train/val/test = 64/16/16，rect/rot = 48/48），84 行为真实 COMSOL forward，12 行 true reference 复用原始 NPZ；`delta_bz = bz_defect - bz_no_defect` 校验通过。

关键结论是：COMSOL oracle residual 的 val/test ordering accuracy 为 `0.6607 / 0.8393`，选中的 `S1_perturb_geom_mlp` surrogate 的 val/test ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 为 `0.2679 / 0.1964`，较 20.55 明显改善。这说明 perturbation forward data 对 surrogate mismatch 有帮助，下一步可以回到 **controlled Priewald-style refinement retry**，但必须继续把它作为 POC/candidate，不更新 baseline。

限制也很明确：当前 pack 只有 96/192 行，且 selected surrogate 的 test residual-error correlation 仍为负（`-0.0462`）。因此下一步不要直接扩大为正式路线，也不要继续训练新的 direct geometry head；应先用 perturbation-calibrated surrogate 做一次受控 refinement retry，观察 residual ordering 是否能转化为 mask / geometry 改善。如果 retry 仍出现 residual 下降但 mask 退化，则优先扩 perturbation data 或转向 mask/profile basis refinement。

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
