# NEXT_STEP

## 当前状态

`CURRENT_BASELINE` 仍以 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 为准：

- v3_complex mask-only grid decoder + forward consistency
- `lambda_forward = 0.10`
- validation-selected probability threshold = `0.80`
- forward surrogate = `checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`

该 baseline 是当前 v3_complex 上最强 boundary-oriented baseline，但 polygon / rotated_rect 精细边界圆斑化、small / low-signal、多缺陷仍未根本解决。

内部 decoder / loss / threshold / geometry / basis / refinement 小修补已经基本收口。当前主线已经转向第 20 阶段：COMSOL / forward data augmentation，目标是提高 MFL 反演问题本身的可辨识性。

## 最近下一步

优先处理 COMSOL forward data pilot 的后续数据多样性：

1. 以第 20.15 rotated_rect pilot_v3 ingest + training gate 的结果为最近状态，确认 rotated_rect / angle variation 数据链路是否稳定。
2. 后续优先考虑合并 rectangular_notch pilot_v2 与 rotated_rect pilot_v3，形成 mixed defect_type pilot pack。
3. 如果合并链路稳定，再扩展 rotated_rect 样本数并评估是否加入 polygon。
4. 继续保持 train-only normalization、validation threshold selection、test-only final evaluation。

这些工作只用于验证 COMSOL forward 数据链路和下一阶段数据设计，不更新 v3_complex `CURRENT_BASELINE`，也不与 v3_complex baseline 做正式性能比较。

## 当前不要继续的方向

不要继续围绕现有 v3_complex grid decoder 做 selection metric、ensemble、threshold trick、loss trick、decoder head、SDF / boundary head、coordinate refinement、hand-crafted Bz features、U-Net-like decoder、shape-type conditional、star-convex、retrieval、box / quad / basis / profile、proposal refinement 或 mask-logit refinement 小修补。

新的实验必须回答：更可靠的 forward data / 多观测输入是否提高了边界反演可辨识性，而不是只带来局部指标波动。
