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

当前不要继续普通 dense decoder patch，也不要继续单独 geometry head 小修补。第 20.48-20.51 已证明：

1. geometry labels 和 differentiable rotated-rectangle rasterizer 没有 blocker；
2. direct delta_bz-only geometry head 的 type / angle 学习不足；
3. controlled architecture sweep 没有找到有效 head 结构；
4. feature-assisted geometry head + lightweight forward consistency 只带来边际 mask / angle 改善，type confusion 仍是主因。

因此最近下一步优先转向：

1. **Priewald-style coarse-to-fine / forward-consistent low-dimensional refinement**：先用现有 dense / geometry proposal 得到 coarse shape，再在低维几何参数空间用 frozen forward surrogate residual 精修。
2. 若要继续 neural geometry route，必须回答 forward residual 是否能在 refinement 阶段稳定降低 type / angle / size 错误，而不是再调 head / loss / threshold。
3. 如果 coarse-to-fine refinement 也失败，再考虑暂停 geometry route，回到数据多样性或更强 forward surrogate 设计。
4. 继续保持 train-only normalization、validation checkpoint / threshold selection、test-only final evaluation。

如果第 20.51 这类 feature-assisted + forward consistency 后续被人工确认成功，才考虑扩展到 polygon、multi-component 或更正式的 forward consistency candidate；当前不建立新 baseline。

## 当前不要继续的方向

不要继续围绕现有 v3_complex grid decoder 做 selection metric、ensemble、threshold trick、loss trick、decoder head、SDF / boundary head、coordinate refinement、hand-crafted Bz features、U-Net-like decoder、shape-type conditional、star-convex、retrieval、box / quad / basis / profile 或 mask-logit refinement 小修补。

也不要继续单独调 rect/rot neural geometry head。新的实验必须回答：显式 geometry representation、differentiable rasterization 和 forward residual 是否能稳定提高边界反演可辨识性，而不是只带来局部指标波动。
