NEXT_STEP

## 当前状态

`CURRENT_BASELINE` 仍为 v3_complex mask-only grid decoder + forward consistency：

* `lambda_forward = 0.10`
* validation-selected probability threshold = `0.80`
* forward surrogate = `checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`

该 baseline 是当前 v3_complex 上最强的 boundary-oriented baseline。它相比上一版 mask-only grid decoder 同时改善了 IoU、Dice、area_error、center_error 与 Bz residual，并且 `pred_area=0` 没有恶化。

但当前核心问题仍未根本解决：

* polygon / rotated_rect 的直边、角点和旋转边界仍有圆斑化；
* multi-defect、small defect 和 low-signal 样本仍然困难；
* 第 18.x / 19.x 的 geometry、basis、profile、proposal refinement、mask-logit refinement 等内部小修补均未超过 CURRENT_BASELINE。

因此当前不再继续 decoder、loss、threshold、basis、geometry、refinement 或 post-processing 小变体。

## 当前下一步

下一阶段进入 forward model / COMSOL / 多观测数据 feasibility，而不是继续在现有单条 Bz + 当前模型上小修补。

最推荐的最小实验包名称：

`comsol_single_defect_multiline_forward_pack_v1`

目标：

* 先生成一个小规模、可审计的 COMSOL / physics-forward single-defect 数据包；
* 优先覆盖 polygon / rotated_rect 或其等价可实现形状族；
* 输入优先设计为 multi-line `delta_Bz`，仍输出 2D / quasi-2D defect mask；
* 先验证 Mask/Geometry -> Bz forward surrogate 是否可靠，再讨论 inverse boundary model；
* 不直接上完整 3D，不直接替换当前 v3_complex `CURRENT_BASELINE`。

下一步应先 review / 固化：

* COMSOL 几何、材料、源项、sensor line / multi-line schema；
* no-defect baseline 和 `delta_Bz` 定义；
* label 到 2D / quasi-2D mask 的映射；
* train / val / test split 与最小样本规模；
* forward surrogate 的接受条件，避免再次出现 Bz residual 下降但 mask 指标变差的 surrogate over-optimization。

## 不再继续的方向

不再继续 selection metric、ensemble、threshold trick、loss trick、decoder 小修补。

不再继续 SDF v2、boundary head v2、coordinate refinement v2、hand-crafted Bz features、普通 U-Net-like decoder、shape-type conditional、star-convex、retrieval、rotated box / deformable quad / oracle quad、profile-band、anisotropic basis、proposal refinement、mask-logit refinement 等已停止方向的小修补。

新的实验必须回答：更可靠 forward model 或更丰富观测是否提高了边界反演可辨识性；预测 mask 是否更能解释 Bz；是否在 IoU / Dice / area_error / small-low-signal / polygon-rotated_rect 视觉上优于当前 baseline。
