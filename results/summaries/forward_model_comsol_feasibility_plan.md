# 第 20.1 forward model / COMSOL feasibility plan

## 1. 当前内部模型路线为什么到边界

当前 `CURRENT_BASELINE` 是 v3_complex mask-only grid decoder + forward consistency，`lambda_forward=0.10`，validation-selected threshold=`0.80`。它已经同时改善了 IoU、Dice、area_error、center_error 与 Bz residual，是现有 v3_complex 单通道 Bz 条件下最强的 boundary-oriented baseline。

但第 18.x / 19.x 的连续 gate 说明，继续在同一条 Bz 输入和同一类 mask decoder 上做小修补已经接近边界：

* SDF、boundary head、coordinate refinement、hand-crafted Bz features、U-Net-like decoder、shape-type conditional 等方向没有稳定解决 polygon / rotated_rect 圆斑化；
* rotated box、deformable quad、oracle-quad supervised、profile-band、anisotropic basis 等结构化表示的 oracle capacity 有时很强，但 Bz -> 低维几何参数学习不稳定；
* proposal refinement 和 mask-logit refinement 可以降低 Bz residual 或微调面积，但经常牺牲 IoU / Dice，或者几乎不改变 coarse mask；
* 第 19.4 中 `proposal_forward` 是典型 surrogate over-optimization：Bz MSE 明显下降，但 IoU / Dice / area_error 明显变差。

因此当前问题不再只是模型结构、loss、threshold 或后处理不足，而是反演可辨识性不足：单条 / 单通道 Bz 对 polygon 角点、rotated_rect 旋转边界、多缺陷组件的约束不够强。

## 2. 当前已有 forward surrogate 和数据生成能力

### mask-to-Bz forward surrogate

当前已有 `scripts/train_mask_to_bz_forward_surrogate.py`。它以二值 defect mask 加固定 x/y coordinate channels 为输入，输出 normalized Bz signal，训练目标是 `MSE(pred_bz_norm, true_bz_norm)`。

已记录性能：

* val: R2=`0.8383`, corr=`0.9170`
* test: R2=`0.8520`, corr=`0.9231`
* test small: R2=`0.8089`, corr=`0.9031`
* test low_signal: R2=`0.5022`, corr=`0.9044`

它足以作为第 18.4 forward consistency 的 frozen module，并证明“预测 mask 是否解释 Bz”是有用约束。局限也很明确：它是从现有 synthetic v3_complex 数据学出来的经验 surrogate，不是独立物理 forward model；low-signal R2 边缘；直接 test-time 优化它可能出现 Bz residual 下降但 mask 变差。

### data_generator_v2.py

`data_generator_v2.py` 目前能生成 v3_complex 的 polygon、rotated_rect、multi_defect，forward signal 是简化 analytic Bz：每个 component 按 mask area、depth、center_x、lift_off 生成一条 x 方向 Bz 曲线。当前 v3_complex train/val/test 的 `signals` 形状分别为：

* train: `(1000, 200)`
* val: `(200, 200)`
* test: `(200, 200)`

这说明当前主线数据是单条、单通道 Bz，而不是 multi-line、three-axis 或 2D sensor field。

生成器已有一个 `v3_complex_multiliftoff` 模式，可输出 `(N, 2, W)` 的 two-liftoff Bz，但它仍是同一简化 analytic forward，不等价于更可靠物理观测。生成器当前没有 Bx / By / Bz 三轴输出，也没有多条 y 扫描线的 multi-line 输出。

### forward consistency 的成功和失败说明

成功之处：第 18.4 说明，在当前数据上加入 frozen mask-to-Bz surrogate consistency 后，整体 shape metrics 和 Bz MSE 可以同时改善。

失败之处：第 19.4 说明，如果只在 coarse probability 上做 test-time optimization，强 forward objective 会破坏 mask；更结构化的 box / quad / basis / profile 表示也没有超过 CURRENT_BASELINE。这说明当前 surrogate 是有用的判别约束，但不能替代更可靠的 forward model 或更丰富观测数据。

结论：没有足够证据继续做内部 decoder/loss/threshold/refinement 小修补。下一阶段应提高观测和 forward model 的信息量。

## 3. 当前仓库是否已有 COMSOL / forward 数据管线

当前仓库中已有轻量 COMSOL 数据接入材料，但还没有正式训练级 COMSOL 数据管线。

已存在文件：

* `data/comsol_mfl/README.md`：说明当前 COMSOL MFL 数据来源和限制；
* `data/comsol_mfl/FORMAT_BRIDGE.md`：说明 COMSOL 中间格式和 PINN 主线格式差异；
* `data/comsol_mfl/rectangular_sweep_small/metadata.csv` 与 `sample_001.csv` 到 `sample_005.csv`：5 个 rectangular notch smoke-test 样本；
* `data/comsol_mfl/rectangular_sweep_small/processed/comsol_rectangular_sweep_small.npz`：由 CSV 转换得到的中间 NPZ；
* `scripts/inspect_comsol_mfl_dataset.py`：检查 raw CSV / processed NPZ 的字段、shape 和一致性；
* `scripts/prepare_comsol_mfl_npz.py`：将 COMSOL small sweep CSV 转成统一中间 NPZ。

当前 COMSOL 数据特点：

* 只有 5 个 rectangular notch 样本；
* sensor line 为 201 点，包含 `Bz_no_defect`、`Bz_defect`、`delta_Bz`、`normB_no_defect`、`normB_defect`、`delta_normB`；
* 标签是 rectangular notch 参数 `width_mm, depth_mm, length_mm, center_x_mm, center_y_mm, center_z_mm`，不是当前主线需要的 2D `mu_maps`；
* README 明确说明该数据只用于 COMSOL -> MCP -> PINN_project 的 smoke test，不是正式训练数据；
* 项目内没有 `.mph` 模型，也没有可直接运行 COMSOL 生成正式 train/val/test 的管线。

因此结论是：项目已经具备 COMSOL 数据接入的雏形和格式桥接经验，但还不具备可直接替换 v3_complex 的 COMSOL 训练集。

## 4. 候选多观测方案比较

### 1. multi-line Bz

定义：在多个 y_scan 位置采集 Bz(x, y_scan)，形成 `(C_line, W)` 或 `(H_scan, W)` 输入，仍输出 2D defect mask。

优点：直接提高 y 方向可辨识性，理论上最贴近当前 2D mask 目标；对 polygon / rotated_rect 的横向位置、宽度、旋转和多缺陷组件应更有约束。

风险：需要修改数据 schema、loader 和 encoder；如果 forward 仍是简化 analytic generator，收益可能有限。

判断：作为观测设计很有价值，但最好和 COMSOL single-defect 包一起做，而不是继续在现有 generator 上临时扩展。

### 2. multi-liftoff Bz

定义：同一条扫描线，在多个 lift-off 高度采集 Bz。

优点：对深度、面积和信号衰减有帮助；改动比 multi-line / three-axis 小。

风险：当前项目已有 two-liftoff 方向曾不稳定；多 lift-off 主要增强深度/尺度约束，不一定能解决 polygon 角点和 rotated edge。

判断：优先级低于 multi-line 和 COMSOL focused dataset。只有在 COMSOL 或真实数据生成成本限制很强时才作为补充。

### 3. three-axis MFL

定义：输入 Bx / By / Bz 或 delta_Bx / delta_By / delta_Bz。

优点：三轴场对边界方向、旋转和局部磁场扰动更有信息，理论上最有利于边界细节。

风险：现有 data_generator_v2.py 不支持；COMSOL 数据生成、单位校准、输入归一化和模型接口改动更大。

判断：物理信息价值高，但不是最小下一步。可作为第二阶段扩展。

### 4. COMSOL-generated single-defect focused dataset

定义：先只生成 polygon / rotated_rect single-defect，固定任务为 2D / quasi-2D mask 或参数化 mask；使用 COMSOL 或已固化 forward model 输出 delta_Bz / delta_normB，可逐步扩展到 multi-line。

优点：直接检验当前圆斑化是否来自简化 synthetic forward 过弱；可控制缺陷形状、旋转、尺寸、深度和观测通道；比直接上完整 3D 更可控。

风险：需要固化 COMSOL 几何、材料、源项、sensor schema、label 到 mask 的转换；生成成本和数据管理成本较高。

判断：这是最推荐的最小下一阶段路线。

### 5. forward surrogate improvement

定义：先用更可靠的 forward 数据训练 Mask/Geometry -> Bz surrogate，再把它作为 consistency module 或 inverse validation tool。

优点：延续第 18.4 的正信号；能在不直接训练 inverse model 的情况下先验证 forward 可学习性。

风险：如果 forward 数据本身仍是当前 weak synthetic，surrogate 只会复现旧问题；如果直接用于 optimization，仍需防止 surrogate over-optimization。

判断：应作为 COMSOL focused dataset 的配套子任务，而不是独立取代数据阶段。

## 5. 最推荐的下一阶段最小实验包

实验名称：`comsol_single_defect_multiline_forward_pack_v1`

数据目标：生成一个小规模、可审计的 COMSOL / physics-forward single-defect 数据包，专注 polygon / rotated_rect 或其最小可实现近似。目标不是立即追求完整 3D，而是验证更可靠 forward model 与更丰富观测是否能减少当前 CURRENT_BASELINE 的圆斑化。

最小样本数建议：

* smoke split：约 60-90 个样本，用于检查 schema、单位、信号质量和 loader；
* gate split：约 300-600 个样本，按 train / val / test 划分，例如 400 / 100 / 100；
* 每类至少覆盖 rotated_rect 与 polygon-like notch，先不做 multi_defect。

输入通道：

* 首选 `delta_Bz`；
* 观测形式优先使用 multi-line Bz，例如 3-5 条 y_scan，每条 201 点；
* 同时保留 no-defect baseline 与 raw `Bz_defect`，但 inverse model 输入先固定为 delta_Bz；
* three-axis 暂不作为 v1 必需项。

输出目标：

* 继续 2D / quasi-2D defect mask，不直接做完整 3D；
* 同时保存几何参数 metadata，用于后续辅助审计，而不是直接替代 mask 目标。

是否需要 COMSOL：需要。当前 data_generator_v2.py 的 analytic Bz 已不足以判断真实可辨识性。

是否需要先生成 forward 数据：需要。第一阶段只做 forward data package 和 forward surrogate feasibility，不直接训练新的 inverse baseline。

需要新增脚本：

* `scripts/inspect_comsol_forward_pack.py`：检查 COMSOL pack 的字段、shape、单位、split、NaN/Inf、信号强度；
* `scripts/prepare_comsol_forward_pack_npz.py`：将 COMSOL CSV/JSON 转成项目统一中间 NPZ；
* `scripts/train_comsol_mask_to_bz_surrogate.py`：先验证 mask/geometry -> multi-line delta_Bz 是否可学；
* 后续再考虑 `scripts/train_comsol_multiline_boundary_candidate.py`，但不应在数据包未通过 forward gate 前启动。

与 CURRENT_BASELINE 的公平比较：

* 不直接用 COMSOL 数据替换 v3_complex baseline；
* 先在 COMSOL pack 内建立自己的 train/val/test；
* 若要比较 CURRENT_BASELINE 思路，应实现同等 mask-only grid decoder + forward consistency 结构，输入改为 multi-line delta_Bz；
* metric 保持 IoU、Dice、area_error、center_error、pred_area=0、small/low-signal、polygon/rotated_rect 视觉审查；
* 报告中必须明确：这是新数据/新观测上的 feasibility，不是直接刷新 v3_complex CURRENT_BASELINE。

## 6. 接受条件与停止条件

接受条件：

* COMSOL pack schema 固定，所有样本有统一 sensor grid、单位、delta_Bz 定义和 defect mask/geometry label；
* forward surrogate 在 validation/test 上达到高 correlation / R2，并且 small / low-signal 不崩；
* multi-line delta_Bz 相比单线 Bz 能明显提高 shape distinguishability；
* 在同一 COMSOL pack 上，mask-only + forward consistency 至少能复现当前 v3_complex baseline 的定位能力，并在 polygon / rotated_rect 视觉上减少圆斑化；
* Bz residual 改善不能以显著牺牲 IoU / Dice / area_error 为代价。

停止条件：

* COMSOL pack 只能生成少量 rectangular smoke-test，无法覆盖 polygon / rotated_rect 或等价 shape family；
* 信号强度、单位或 no-defect baseline 定义不稳定，导致 forward surrogate 学不动；
* multi-line / richer observation 仍无法改善 polygon / rotated_rect，或只降低 Bz residual 但破坏 mask；
* 数据生成成本过高，无法形成最小 train/val/test；
* 出现新一轮 decoder/loss/threshold 小修补倾向，应停止并回到数据/forward model 问题。

## 7. 对后续 Codex 实验的建议边界

后续 Codex 任务应先围绕数据和 forward model，而不是围绕新 decoder 变体：

* 不继续 SDF、boundary head、coordinate refinement、shape-type conditioning、U-Net-like decoder、basis / geometry / mask-logit refinement v2；
* 不继续 threshold / post-processing trick；
* 不直接上完整 3D defect reconstruction；
* 先把 COMSOL / multi-line forward 数据包做成可审计、可复现、可 split 的最小版本；
* 先验证 Mask/Geometry -> Bz forward surrogate，再训练 inverse model；
* 若 forward surrogate 可靠，再以当前 `CURRENT_BASELINE` 的评价口径设计新的 multi-observation boundary candidate。

推荐优先路线：`COMSOL-generated single-defect focused dataset`，其中 v1 应尽量包含 multi-line `delta_Bz`。`forward surrogate improvement` 是该路线的第一道 gate，而不是独立的小修补方向。
