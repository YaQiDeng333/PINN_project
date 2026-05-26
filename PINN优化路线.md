# PINN 优化路线

## 2026-05-26 路线同步：20.86 true 3D RBC baseline transition

20.86 完成主线 baseline transition：项目当前 baseline 从 v3_complex 2D mask / boundary prediction 切换为 true 3D RBC-style profile-depth reconstruction。旧 2D baseline 没有删除，而是降级为 archived comparator；它不再是当前 `CURRENT_BASELINE`。

新的当前 baseline 以 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 为数据身份，输入是 Bx/By/Bz `delta_b=(N,3,3,201)`，模型是 20.77 small Conv1D encoder + MLP six-parameter head，输出 `L_m/W_m/D_m/wLD/wWD/wLW`，再生成 RBC-style 3D profile/depth 和 projected mask。20.85 formal rerun 复现了 20.77：selected seed `42`，test normalized MAE `0.678014`，profile depth RMSE `0.000387737 m`，Er-like profile error `0.340544`，L/W/D MAE `1.892/2.186/0.800 mm`，projected mask Dice `0.847727`。

路线边界同步更新：20.81 只作为 projected-mask / visual comparator，20.83 作为 profile-primary negative gate；`wLD/wWD/wLW` 仍是 auxiliary diagnostics，不作为 headline metric。当前仍是 `exact_piao_rbc=False`、`rbc_style_approximation=True`，尚未在真实实验数据上验证，也不是 arbitrary free-form / multi-defect 部署级模型。后续工作应围绕 benchmark/report package、real-data alignment、exact Piao feature/representation refinement 展开，而不是继续把旧 2D baseline 当主线。

## 2026-05-26 路线同步：20.85 formal true 3D RBC benchmark rerun

20.85 的路线意义是把 20.77 从一次 training gate 结果收口为可复核的 formal benchmark candidate。该 rerun 固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240`，通过 registry / manifest 显式加载，不运行 COMSOL，不生成新数据，不修改 NPZ，也不更新 `CURRENT_BASELINE.md`。

结果稳定复现 20.77：selected seed `42`，test normalized MAE `0.678014`，L/W/D MAE `1.892/2.186/0.800 mm`，profile depth RMSE `0.000387737 m`，projected mask Dice `0.847727`。profile/depth 主指标仍优于 20.81 feature-fusion 和 20.83 profile-primary negative gate；20.81 只保留为 projected-mask / visual comparator，20.83 只保留为 negative evidence。

路线判断更新为：true 3D / Piao-style branch 可以围绕 20.77/formal rerun 做 benchmark candidate 展示和报告整理，但不能写成 baseline replacement。`wLD/wWD/wLW` 继续作为 auxiliary diagnostics，profile/depth metric 继续作为当前 true 3D RBC branch 的主评价轴；dense mask baseline 仍只作为 comparator。

## 2026-05-25 路线同步：20.82 true 3D RBC curvature output representation audit

20.82 的路线意义是把 true 3D / Piao-style 分支的评价口径从“逐项硬拧 `wLD/wWD/wLW`”调整为“六参数生成的 3D profile 是否准确”。这不是 baseline replacement，也不改变 v3_complex `CURRENT_BASELINE.md`；它只改变 true 3D RBC branch 后续训练和评价的主次关系。

审计证据显示，raw curvature MAE 与 profile quality 不同步：20.77 test 的 curvature-vs-profile depth RMSE correlation 只有 `0.358243`；20.81 feature-fusion 的 projected mask Dice 更好，但 profile depth RMSE 更差；20.80 feature-only curvature 更好，但只有 aggregate/group artifacts，不能替代 per-sample profile evidence。因此 projected mask IoU/Dice 只能继续作为 2D footprint QA，`wLD/wWD/wLW` 只能作为 auxiliary diagnostics，不能单独决定 true 3D profile route 是否成功。

路线判断更新为：下一步优先做 `R1_six_params_profile_primary_loss`，即仍保持 Piao-style six-parameter output，但把 profile-depth reconstruction loss / metric 升为主目标；`R2 template + residual`、`R3 depth/profile basis`、`R5 hybrid multitask` 作为后续备选。`exact_piao_rbc=False` 和 `rbc_style_approximation=True` 继续成立，dense mask baseline 继续只作为 comparator。
## 2026-05-25 路线同步：20.80 Piao/NLS-inspired feature diagnostic

20.80 的路线意义是：curvature 问题不是简单 neural head/loss 能修好，但 Bx/By/Bz 信号里确实存在一部分 physical curvature cue。固定 v3_240 dataset 上，validation 选中的不是 F4 NLS proxy，而是 `F0_F1_F2_basic_physical + svr_rbf_C10_eps0.03`；它把 test curvature MAE 从 20.77 neural 的 `0.201076` 降到 `0.190304`，并优于 20.77 feature baseline 的 `0.195046`。但 total MAE `0.695724`、L/W/D `2.595/2.361/0.966 mm` 和 projected mask Dice `0.826272` 都不如 20.77 neural，因此不能替代 20.77 benchmark candidate，更不能写成 baseline。

本轮也明确了 claim 边界：`exact_piao_rbc=False`、`rbc_style_approximation=True` 继续成立；F4 bounded gaussian / derivative-of-gaussian NLS proxy 可以稳定提取，fit_success_rate=1.0，但它没有被 validation 选择为最优 feature set，所以不能声称 exact Piao/NLS/LS-SVM route 已通过。真正可复用的信号来自 F1/F2 的 peak / width / lobe / gradient / asymmetry 类特征，说明下一步应做 feature-fusion / hybrid，而不是继续拆 neural head 或盲目扩样。

路线判断更新为：保留 20.77 v3_240 neural benchmark candidate 作为当前 true 3D / Piao-style 主线候选；下一步优先把 F1/F2 physical features 融入 neural model 的 curvature branch，同时保持 L/W/D 与 mask/profile 的 20.77 neural 表现。dense mask baseline 继续只作为 comparator，`CURRENT_BASELINE.md` 不更新。

## 2026-05-25 路线同步：20.78 formal true 3D RBC benchmark candidate audit

20.78 的路线意义是：true 3D / Piao-style 主线正式进入 benchmark candidate 阶段，但仍不是 baseline replacement。v3_240 的证据链已经足够说明 Bx/By/Bz 输入能学习 RBC-style 主几何参数：neural test normalized MAE `0.678014`，优于 feature comparator `0.715395`；L/W/D MAE 为 `1.892/2.186/0.800 mm`，D_m、projected mask Dice 和 profile depth RMSE 都较 N=112 改善。

真正的风险边界也更清楚了：curvature 不是随 N=240 自然解决的问题。`wLD/wWD/wLW` 仍不稳定，boxy / sharp 模板明显最差；且存在 projected mask 很好但 curvature 很差的样本，说明 2D footprint / mask 指标不足以代表 true 3D RBC profile quality。当前仍是 `exact_piao_rbc=False`、`rbc_style_approximation=True`，因此不能写成完整 Piao 2019，也不能更新 `CURRENT_BASELINE.md`。

路线判断更新为：下一步优先 model refinement，而不是盲目扩到 480。具体方向是 curvature-aware model/head/loss、stronger Bx/By/Bz sequence encoder，以及 exact Piao / NLS-inspired feature diagnostic；curvature-targeted sampling 可作为第二阶段补强。dense mask baseline 继续只作为 comparator。

## 2026-05-25 路线同步：20.77 true 3D RBC v3_240 training gate

20.77 的路线意义是：true 3D / Piao-style 主线从 small pilot training signal 推进到 formal benchmark candidate 前的正向证据。v3_240 仍是 RBC-style / Piao-inspired engineering approximation，`exact_piao_rbc=False`，不是完整 Piao 2019 复现；本轮只验证 Bx/By/Bz 是否能学习 RBC-style 3D profile parameters、projected mask 和 depth/profile metrics，不建立 baseline。

技术链路固定为：registry/manifest explicit dataset_id gate → `delta_b` Bx/By/Bz input `(240,3,3,201)` → Conv1D channels `(240,9,201)` → train-only normalization → validation-only selection → test-final metrics。N=240 相对 N=112 的核心改善是 neural test normalized MAE `0.7039 → 0.6780`，D_m MAE `1.106 mm → 0.800 mm`，projected mask Dice `0.8364 → 0.8477`，profile depth RMSE `0.000548 m → 0.000388 m`。神经模型也继续优于 Piao-inspired feature comparator：`0.6780` vs `0.7154`。

路线判断更新为：`L_m/W_m/D_m` 可学习，`wLD/wWD/wLW` 仍是主要不稳定项；N=240 足以进入 formal true 3D RBC benchmark candidate / model refinement，但不足以自动替换 baseline。dense mask baseline 继续只作为 comparator，`CURRENT_BASELINE.md` 不更新；后续重点应从“继续盲目扩样”转向 curvature learnability：更强 sequence encoder、curvature-aware objective、targeted curvature sampling 或更接近 Piao 的 feature / NLS 方案。

## 2026-05-24 路线同步：20.72 true 3D RBC assembled pilot pack

20.72 的路线意义是：true 3D / Piao-style 主线已经从 partial pilot pack 推进到 assembled pilot pack candidate。当前 pack 仍是 RBC-style / Piao-inspired engineering approximation，`exact_piao_rbc=False`，不声称完整复现 Piao 2019，也不建立或替换 baseline。

assembled pack 的技术链路保持为：RBC-style params → depth grid → watertight mesh → imported watertight COMSOL solid → 20.70 material/domain fix → full-source Bx/By/Bz @ 0.008 → `delta_b` check → assembled NPZ/schema validation。最终 assembled N=56，split 为 36/10/10，五类 curvature template 全部覆盖：sharp=11、round=11、boxy=12、LD_dominant=11、WD_dominant=11。`train_ready_candidate=True` 只表示可以进入 explicit training gate，`baseline_ready=False` 仍然固定。

路线判断更新为：下一步可以做 true 3D training gate，但必须通过 registry / manifest 显式加载 `comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled`，禁止 latest/newest 自动扫描，禁止自动替换 `CURRENT_BASELINE.md`。dense mask baseline 继续只作为 comparator；后续训练若不能稳定超过当前 baseline，也不能把该 pilot pack 或模型写成新 baseline。

## 2026-05-24 路线同步：20.71 true 3D RBC pilot pack metadata

20.71 的路线意义是：true 3D / Piao-style 主线已经从 one-sample imported-solid smoke 推进到可注册、可审计的 small pilot pack，但还没有达到训练入口。当前 pack 是 RBC-style / Piao-inspired engineering approximation，`exact_piao_rbc=False`，不声称完整复现 Piao 2019，也不建立 baseline。

技术链路已跑通到 30 个样本：RBC-style params → depth grid → watertight mesh → imported watertight COMSOL solid → 20.70 material/domain fix → full-source Bx/By/Bz @ 0.008 → `delta_b` check → NPZ/schema validation。30 个成功样本 split 为 20/5/5，registry/manifest 已禁止 latest/newest 自动扫描和 baseline replacement；dense mask baseline 继续只作为 comparator。

路线判断：当前必须先 top-up，而不是训练。inventory 已完整记录 60 行，但状态为 30 pass、2 fail、28 not_attempted，且 `LD_dominant` / `WD_dominant` 两个 curvature family 缺失。下一步应补齐 missing families 和 deep-elongated timeout 样本；只有 top-up 后 curvature coverage、split 和 manifest 全部重新验证通过，才进入 explicit true 3D training gate。

## 2026-05-24 路线同步：20.70 imported watertight solid solver robustness

20.70 的路线意义是：true 3D / Piao-style 主线第一次跑通了 imported watertight mesh solid 的 full-source COMSOL forward smoke。20.69 已经证明 Python watertight mesh、COMSOL import / repair / form solid / Boolean subtract / mesh precheck 可行；20.70 进一步证明，失败点可由 domain/material selection 最小修复解决，而不是必须回退 high-layer approximation。

关键修复边界必须保留：没有重做 mesh builder，没有改 defect geometry，没有换回 high-layer，也没有简化为 constant-depth；只是动态查询 post-Boolean selections，并把 air material selection 中与 `steel_notched` 重叠的 domain 去掉。`material_domain_fixed` 在 default solver、`mesh_auto_size=5`、`Jscale=1.0` 下通过，随后 Bx/By/Bz forward、`delta_b` check 和 NPZ/schema validation 全部通过。

路线判断更新为：imported watertight solid route 已从 “geometry feasible but solver blocked” 变为 “one-sample forward-ready”。下一步可以进入 smooth/mesh-based true 3D RBC pilot 的设计/小规模生成，但 pilot 必须继续记录 material selection protocol、mesh sensitivity 和 generated data boundary；dense mask baseline 仍只是 comparator，2D profile-forward 小修继续暂停，`CURRENT_BASELINE.md` 不更新。

## 2026-05-24 路线同步：20.69 watertight imported solid builder hardening

20.69 继续 true 3D / Piao-style 主线，但只攻 imported watertight mesh solid builder，不进入 pilot、不训练、不更新 baseline。相对 20.68，真正推进点是把 imported mesh route 从 “Boolean 后 empty steel domain” 推进到 “RBC watertight STL 可 import / repair / form solid / Boolean subtract / mesh precheck”。这一步不再依赖 high-layer control，也没有把 high-layer stepped control 写成 smooth / near-smooth。

当前分级状态是：Python watertight mesh pass，RBC imported solid geometry gate pass，forward smoke fail。`medium_round` mesh 明确记录了 `mesh_units=m`、top cap `z=0`、bottom surface `z=-depth`、defect void 嵌入 steel 并与 surface 相交；COMSOL 侧 `import_success=True`、`form_solid_success=True`、`boolean_subtract_success=True`、`mesh_precheck_success=True`。但 imported watertight defect model 的 stationary solve 不收敛，因此没有 `b_defect`、没有 `delta_b`、没有 NPZ/schema validation。

路线判断：imported watertight solid route 技术上可行到 geometry gate，但还不是 smooth/mesh-based pilot-ready。下一步应修 imported solid 的 COMSOL solve robustness、mesh quality 或 solver setting；在 forward smoke 通过前，不扩样、不训练、不进入 60-sample pilot。dense mask baseline 继续只作 comparator，2D profile-forward 小修继续暂停。

## 2026-05-24 路线同步：20.68 smooth / near-smooth true 3D builder completion

第 20.68 继续沿 true 3D / Piao-style 主线推进，但没有进入 pilot，也没有训练任何模型。本轮专门验证 smooth / near-smooth variable-depth defect builder：从 `medium_round` RBC-style depth map 出发，有限尝试 lofted contour、stacked workplane contour loft、interpolated surface、imported closed mesh，再以 24-layer high-layer control 作对照。

路线判断是：smooth / near-smooth builder 仍未完成。Loft、workplane loft、ParametricSurface 和 imported closed mesh 都没有同时满足 `closed_body_success=True`、`boolean_subtract_success=True`、`mesh_precheck_success=True`、`spatial_depth_variation=True`、`is_constant_depth=False` 的 Stage C gate；imported mesh route 的失败点是 Boolean subtract 产生 empty steel domain selection。唯一通过的是 `high_layer_control_24`，它比 20.66 的 5-layer 和 20.67 的 12-layer 更细，但仍是 high-layer stepped control，不能当作 smooth / near-smooth 或 exact Piao RBC。

因此 true 3D 路线继续保留，但后续扩样需要先做口径选择：接受 high-layer approximation 作为 pilot label / geometry definition，或继续修 smooth closed-surface builder。当前不允许自动进入 60-sample pilot，不回退到 2D profile-forward 小修，也不更新 `CURRENT_BASELINE.md`；dense mask baseline 仍只作 comparator。

## 2026-05-23 路线同步：20.63 multi-axis profile oracle ordering feasibility

第 20.63 将第 20.62 的 richer observation 判断继续拆开验证：如果 multi-liftoff Bz residual 仍不能稳定排序 profile quality，那么先不训练 surrogate，而是直接用真实 COMSOL oracle residual 比较 same-liftoff Bx/By/Bz vector observation 是否更有辨识力。本轮使用 24 base / 192 profile rows，在 `sensor_z_m=0.008` 下导出 `[mf.Bx, mf.By, mf.Bz]`；所有 profile row 都是真实 COMSOL forward，包括 `true_reference`，不复用旧 Bz-only 数组。

结果显示 same-liftoff multi-axis alone 仍没有解决 profile residual non-identifiability。test ordering accuracy 为：Bx-only `0.4505`，By-only `0.4955`，Bz-only `0.4505`，Bx+By+Bz train-std normalized `0.4505`；all-axis mismatch_rate 为 `0.5495`，residual-error correlation 为 `0.0242`。它没有超过同 pack Bz-only，也没有超过 20.61 single-height Bz oracle test reference `0.5030`，更没有达到 `>0.65` 或 `+0.10` improvement gate。

路线判断因此更新为：当前瓶颈不再是 profile-compatible surrogate 训练、profile perturbation data 规模、lift-off 选择或单纯 field component 数量，而是当前 scan geometry / excitation 下 observation 对 profile boundary quality 的排序信息不足。下一步若继续 forward-guided profile route，应优先转向 **multi-direction excitation / richer scan geometry**，而不是训练 multi-axis profile surrogate、继续扩同 liftoff 三轴数据，或回到 profile-forward refinement。第 20.63 仍是 POC，不更新任何 baseline。

## 2026-05-23 路线同步：20.62 multi-height profile oracle ordering feasibility

第 20.62 将第 20.61 的 observation-identifiability 判断进一步拆开验证：如果 single-height Bz residual 不能稳定排序 profile quality，那么先不训练 surrogate，而是直接用真实 COMSOL oracle residual 比较 multi-liftoff observation 是否更有辨识力。本轮使用 12 base / 96 profile rows，在 `sensor_z_m = [0.004, 0.008, 0.012]` 下生成 multi-height Bz；0.008m 复用第 20.61 exact observation，0.004m 和 0.012m 共 192 个 observation 使用真实 COMSOL forward。

结果显示 multi-height / multi-liftoff alone 没有解决 profile residual non-identifiability。test ordering accuracy 为：0.008m single-height `0.4909`，0.004m `0.4364`，0.012m `0.4545`，multi-height normalized `0.4545`；multi-height mismatch_rate 为 `0.5455`，residual-error correlation 为 `-0.5920`。它没有超过 20.61 single-height oracle test reference `0.5030`，也没有达到 `>0.65` 或 `+0.10` improvement gate。

路线判断因此更新为：当前问题不是 profile-compatible surrogate 训练不足，也不只是 profile perturbation data 规模不足，而是单 Bz、少量 scan line、仅改变 lift-off 的 observation 对 profile boundary quality 的排序信息仍不够。下一步若继续 forward-guided profile route，应优先转向 **multi-axis / multi-direction observation** 或更丰富 scan pattern / component，而不是训练 multi-height profile surrogate 或继续扩大同类 lift-off pack。第 20.62 仍是 POC，不更新任何 baseline。

## 2026-05-23 路线同步：20.61 expanded profile perturbation forward calibration

第 20.61 将第 20.60 的 profile-native perturbation 数据覆盖从 12 base / 96 rows 扩大到 36 base / 288 rows，其中 252 行是真实 profile polygon COMSOL forward，36 行 `true_reference` 复用 pilot_v9 原始数组作为 residual anchor。该实验仍是 forward surrogate calibration POC：不做 profile refinement，不训练 inverse model，不更新 baseline。

结果显示扩大数据确实缓解了 20.60 的极端 test collapse：selected `EPPF1_profile_station_mlp` 的 test surrogate ordering 从 20.60 的 `0.2143` 提升到 `0.5569`，test mismatch_rate 从 `0.7857` 降到 `0.4431`，waveform val/test correlation 也保持在 `0.9435 / 0.9299`。但是 strict gate 未通过，validation ordering 只有 `0.5361`，test ordering 也未达到 `>0.65`；更关键的是 COMSOL oracle residual ordering 在 train/val/test 只有 `0.4471 / 0.5120 / 0.5030`，说明当前观测配置下真实 residual 本身无法稳定排序 profile quality。

路线判断因此更新为：profile-compatible surrogate 的 waveform fit 不是主要 blocker，当前瓶颈转为 **profile residual objective / observation identifiability**。在 3 条 scan line、单 Bz、constant-depth top-view profile polygon 设置下，继续扩同类 profile perturbation data 或小调 surrogate architecture 的收益有限；下一步如果继续 forward-guided route，应优先转向 richer observations（multi-height / multi-axis / more scan lines）或专门做 non-identifiability audit。第 20.61 不支持直接回到 profile-forward refinement retry，也不改变现有 baseline。

## 2026-05-22 路线同步：20.60 profile perturbation forward calibration

第 20.60 将第 20.59 的结论进一步落地：如果要让 forward residual 支撑 profile-basis refinement，校准数据必须围绕 profile representation 本身，而不是继续复用 rect/rot geometry perturbation。为此本轮设计了 24 base / 192 rows 的 profile perturbation plan，并在 COMSOL 侧按 minimum partial 生成 12 base / 96 rows forward pack，其中 84 行是真实 profile polygon COMSOL forward，12 行 `true_reference` 复用原始 pilot_v9 作为 residual anchor。

该实验确认了两点。第一，profile polygon generation 和 forward-pack schema 本身可行：split/type/variant coverage 达到 minimum partial，`delta_bz = bz_defect - bz_no_defect` 校验通过，且没有把 profile perturbation 退回 single rotated box。第二，当前数据规模仍不足以支撑 profile-forward refinement：`PPF1_profile_station_mlp` 的 waveform val/test NRMSE 为 `0.4396 / 0.3758`，但 residual ordering test collapse 到 `0.2143`，mismatch_rate 达到 `0.7857`；同时 COMSOL oracle residual test ordering 也只有 `0.5357`，说明在当前 2 个 test base 的 partial pack 上，真实 residual 对 profile quality 的排序信号也不稳定。

路线判断因此更新为：profile-compatible surrogate 方向没有被实现错误否定，但当前 partial pack 不能支撑 profile-forward refinement retry。下一步若继续 forward-guided profile route，必须先扩大 profile perturbation data，特别是增加 val/test base 覆盖并重新验证 oracle ordering；若扩展后 oracle residual 仍弱，则主要瓶颈可能是观测 non-identifiability，需要 richer observations / multi-axis / multi-height，而不是继续调 surrogate architecture 或 refinement loss。第 20.60 仍是 POC，不更新 baseline。

## 2026-05-22 路线同步：20.58 mask/profile basis refinement

第 20.58 将第 20.57 否定的 single rect/rot low-dimensional refinement，替换为从 dense/coarse initializer 提取的 K=8 mask/profile basis 表示。该路线的目标是避免单个 rotated rectangle 参数空间过窄，同时不回到完全自由的 dense mask decoder。

结果显示 profile basis 有边际价值：profile extraction 基本保留 dense proposal，no-forward profile refinement 在 test 上达到 `0.6697 / 0.8002 / 0.2196`，好于第 20.57 calibrated rect/rot refinement `0.6492 / 0.7829 / 0.2417`，也接近第 20.54 strong dense initializer。它说明 profile/basis 表示能作为更柔性的低维形状空间，但尚未稳定超过第 20.54 extracted rotated-box proposal `0.6726 / 0.8017 / 0.1945`。

更关键的是 forward profile refinement 的 validation sweep 选择了 `lambda_forward=0.0`，说明当前 S1 perturbation-calibrated surrogate 通过 lossy profile-to-rect summary 接入后，不能可靠驱动 profile-space optimization。路线判断因此更新为：当前瓶颈不是再换一个更复杂的 shape basis，而是需要 profile-compatible forward surrogate 或 richer observations。继续在 current surrogate + current profile space 上调 steps、lr、lambda_forward 意义有限；下一步若继续 geometry/refinement route，应优先改进 forward surrogate 的输入表示和 residual calibration，而不是继续增加 direct geometry head 或单 box refinement 变体。

## 2026-05-22 路线同步：20.57 calibrated refinement retry

第 20.57 验证了一个关键负面结果：第 20.56 的 perturbation-calibrated `S1_perturb_geom_mlp` 虽然能复现较好的 pairwise residual ordering（val/test `0.7321 / 0.8036`），但把它放进连续低维 Priewald-style refinement 后，mask / geometry 指标没有同步改善。以第 20.54 的 strong dense/extracted proposal 为初值，test geometry-raster IoU/Dice 从 `0.6726 / 0.8017` 下降到 `0.6492 / 0.7829`；forward residual 继续下降，但 mismatch_rate 达到 `0.6212`，residual reduction 与 IoU/Dice delta 呈负相关。

路线判断因此更新为：当前瓶颈不只是 surrogate waveform fit，也不只是 pairwise ordering，而是 residual objective 在连续 rect/rot parameter space 中的可优化性。继续对 steps、lr、prior、surrogate loss 做小调意义不大；direct neural geometry head 也已经在 20.48-20.51 收口。下一步若继续 geometry-aware route，应优先转向 **mask/profile basis refinement** 或更高维形状表示；如果未来回到 Priewald-style refinement，需要先扩大 perturbation pack 或引入 richer observations，再重新证明 residual landscape 能提供稳定梯度。

## 2026-05-22 路线同步：20.56 perturbation forward calibration

第 20.56 把 20.55 的 forward surrogate mismatch 问题拆成两个问题验证：真实 COMSOL residual 是否能排序局部几何质量，以及 surrogate 是否能学到这种排序。结果显示，在 rect/rot local perturbation partial pack（96 行，84 行真实 COMSOL forward）上，COMSOL oracle residual 的 val/test ordering accuracy 为 `0.6607 / 0.8393`，选中的 `S1_perturb_geom_mlp` 的 val/test ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 相比 20.55 明显降低。

这说明当前 Priewald-style 路线的瓶颈不是“forward residual 完全无信息”，而是原先 surrogate 缺少围绕同一几何样本的局部扰动校准数据。后续如果继续 refinement，应使用 perturbation-calibrated surrogate 做受控 retry，并继续记录 residual-ordering、mask/geometry improvement 和 surrogate mismatch。由于当前 pack 仍是 96/192 partial pack，且 test residual-error correlation 仍为负，不能把 20.56 写成 baseline，也不能直接扩大为正式方法结论。

## 当前路线同步：第 20 阶段 forward data / COMSOL pilot

当前 `CURRENT_BASELINE` 仍是 v3_complex mask-only grid decoder + forward consistency `lambda_forward=0.10` + validation-selected threshold `0.80`。第 18.x / 19.x 已经说明继续做 decoder、loss、threshold、geometry、basis 或 refinement 小修补收益不足。

第 20 阶段的路线目标是提高反演问题本身的可辨识性：用 COMSOL / physics-forward 数据构建 multi-line `delta_Bz` -> 2D / quasi-2D mask 的可审计训练包。当前已经完成 rectangular_notch small / pilot / pilot_v2 数据链路，以及 rotated_rect / angle variation pilot_v3 数据链路。后续优先级是合并 `rectangular_notch` + `rotated_rect`，再扩展样本数和 defect_type 多样性；不要回到当前 grid decoder 的小 head / 小 loss / 小 threshold 调参。

本文件记录当前主线判断，不再按早期实验流水账追加。历史细节以 `EXPERIMENT_LOG.md`、`CURRENT_BASELINE.md` 和 `results/summaries/` 为准；本文件只保留路线层面的结论、停止条件和下一阶段原则。

## 下一阶段：forward model / 多观测数据 / COMSOL feasibility

第 18.x / 19.x 后，内部结构和 test-time refinement 路线已经基本到达边界。当前 `CURRENT_BASELINE` 仍保留为 mask-only grid decoder + forward consistency `lambda_forward=0.10` + validation-selected threshold `0.80`，它是现有 v3_complex 单通道 Bz 数据上的最强 boundary-oriented baseline。

下一阶段目标不再是继续调 decoder、loss、threshold、basis、geometry、proposal refinement 或 post-processing，而是提高反演问题本身的可辨识性。核心判断是：现有单条 / 单通道 Bz 对 polygon / rotated_rect 的直边、角点、rotation 和 multi-defect 组件约束不足，因此继续在当前输入上做小修补很难根本解决圆斑化。

推荐优先路线是 `comsol_single_defect_multiline_forward_pack_v1`：

* 先做小规模、可审计的 COMSOL / physics-forward single-defect 数据包；
* 优先覆盖 polygon / rotated_rect 或可实现的等价形状族；
* 输入优先使用 multi-line `delta_Bz`，仍输出 2D / quasi-2D defect mask；
* 先验证 Mask/Geometry -> Bz forward surrogate 是否可靠，再进入 inverse boundary model；
* 不直接上完整 3D，不直接替换当前 v3_complex `CURRENT_BASELINE`。

候选方向优先级：

1. COMSOL-generated single-defect focused dataset，最好包含 multi-line `delta_Bz`；
2. 基于该数据包训练更可靠的 forward surrogate，作为 inverse model 前置 gate；
3. three-axis MFL 作为后续扩展；
4. 单独 multi-liftoff 只作为低优先级补充，因为它更偏深度 / 尺度约束，不一定解决边界角点；
5. 不再围绕当前 grid decoder 做小 head / 小 loss / 小 threshold / 小 refinement。

该阶段的接受条件不是某个局部指标波动，而是更可靠 forward model 或更丰富观测能否稳定改善边界可辨识性：预测 mask 是否更能解释 Bz，同时 IoU / Dice / area_error / small-low-signal / polygon-rotated_rect 视觉质量是否优于当前 baseline。

## 当前主线状态

当前 `CURRENT_BASELINE` 已从早期的 μ-field threshold / composite-selection 路线，以及上一版 mask-only grid decoder boundary baseline，更新为：

* 模型族：mask-only grid decoder + forward consistency
* 数据集：`v3_complex`
* forward surrogate：`checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`
* `lambda_forward = 0.10`
* 选定 probability threshold：`0.80`
* threshold 选择来源：仅 validation set

本项目目标是缺陷边界形状反演，因此 baseline 选择优先看 IoU、Dice、area_error、`pred_area=0`、small / low-signal 表现，而不是只看 μ-field 的 MSE / MAE。

保留参考线：

* `v3_complex_tv_sweep_2e-6` 保留为 MSE-oriented reference，不再是主线 `CURRENT_BASELINE`。
* composite-selection 保留为 μ-threshold shape-oriented reference，用于对照完整 μ-field threshold 方案。
* 前一版 mask-only MLP boundary model 保留为 boundary reference。

当前 forward consistency baseline 的核心结论是：在 mask-only grid decoder 基础上加入 frozen mask-to-Bz surrogate consistency 后，整体 IoU / Dice / area_error / center_error 与 Bz MSE 同时优于上一版 grid decoder baseline，说明预测 mask 能更好解释观测到的 Bz 信号。但它仍没有根本解决边界精细形状，polygon area_error 轻微恶化，polygon / rotated_rect 仍可能被预测成偏圆的平滑斑块，multi-defect 和 small / low-signal 样本仍是难点。

## 已停止路线

以下方向已经完成阶段性验证或被明确停止，不再继续做小修补：

* v4 / calibrated_mu / enhanced decoder
* threshold trick / adaptive threshold
* aux mask head 作为最终输出
* aux mask loss / shape-aware loss 迁移到 v3
* defect-weighted MSE / threshold-margin loss
* small oversampling
* CNN1D BzEncoder
* hand-crafted Bz features
* multi-liftoff
* geometry auxiliary supervision
* warm-start / curriculum
* selection metric 细调
* ensemble
* SDF / boundary loss
* coordinate refinement
* shape-prior latent
* exemplar retrieval
* star-convex radial model
* U-Net-like decoder
* shape-type conditional decoder
* 继续围绕当前 grid decoder 做小 head / 小 loss / 小 threshold 修补

停止这些方向的共同原因是：它们要么只带来局部指标波动，要么改善 IoU / Dice 的同时恶化 area_error / small / low-signal，要么视觉上仍然没有解决 polygon / rotated_rect 的圆斑化。

## Mask-Only 路线结论

mask-only boundary model 是第一次真正比 μ-threshold 路线更贴近项目目标的方向。它把任务从“先预测 μ，再用 `mu < 500` 间接取 mask”改成“直接从 Bz 预测 defect probability / mask”，因此更符合缺陷边界形状反演。

第 15.2 的 mask-only MLP + validation-selected threshold=0.90 已经明显优于 composite-selection。第 15.4 的 mask-only grid decoder 进一步提升，并成为当时的 boundary-oriented `CURRENT_BASELINE`。第 18.4 之后，mask-only grid decoder + forward consistency `lambda_forward=0.10` 进一步成为新的 `CURRENT_BASELINE`。

当前问题已经不是“完全找不到缺陷”，而是：

* polygon / rotated_rect 的直边、角点、旋转结构仍被圆斑化；
* multi-defect 有时漏掉组件；
* small / low-signal 样本仍不稳定；
* 继续在同一个 grid decoder 上加小 loss、小 head、小 threshold 或小 feature，收益不足以支持新主线。

## 外部研究与路线判断

Deep Research / NotebookLM 相关整理给出的共同判断是：下一阶段不应继续把问题当作普通 segmentation 小修补，而应转向更接近 MFL 反演本质的路线：

* geometry-aware inversion
* differentiable rasterization
* forward consistency
* hybrid forward-model + neural network inversion
* geometry / shape-parameter reconstruction

这些方向的共同点是：预测结果不只要像真实 mask，还要能解释观测到的 Bz 信号。也就是说，后续主线应从“mask 分割模型调参”转向“可解释 Bz 的边界 / 几何反演”。

## 当前最有价值的正方向

第 18.2 已验证 `mask -> Bz` forward surrogate 具有可用性：

* test R2 = 0.8520
* test correlation = 0.9231

这说明用预测 mask 解释 Bz 是可行的。随后 forward consistency 在第 18.2 / 18.3 / 18.4 中出现明确正信号：

* IoU / Dice 提升；
* center_error 改善；
* Bz residual / Bz MSE 明显下降；
* `lambda_forward=0.10` 在 bounded bracket check 和 3 seed validation 中表现最好；
* `pred_area=0` 没有明显恶化；
* small / low-signal 的 IoU / Dice 有改善，但 area_error 仍需谨慎看待；
* polygon 有小幅 IoU / Dice 改善但 area_error 仍是风险，rotated_rect 和 multi_defect 的改善更明确。

因此，`lambda_forward=0.10` forward consistency 已从 candidate 提升为当前 `CURRENT_BASELINE`。它是目前最有价值的正方向，但它的意义是把主线推进到 physics-consistent / hybrid inversion，而不是继续围绕同一 grid decoder 做 loss、head、threshold 或 feature 小修补。

## 下一阶段原则

后续不再做局部小修补。新实验必须围绕以下问题判断：

* 预测 mask 是否更贴合真实边界；
* 预测 mask 是否更能解释观测 Bz；
* IoU / Dice / area_error / `pred_area=0` 是否优于当前 mask-only grid decoder baseline；
* small / low-signal 是否不变差，最好改善；
* polygon / rotated_rect 的圆斑化是否真的减轻，而不是只出现局部指标波动。

forward consistency 已通过 review 和 baseline 决策，后续应进入 physics-consistent / hybrid inversion 主线。

如果后续发现 forward consistency 仍无法解决 polygon / rotated_rect 精细边界，则下一步应转向更严格的 geometry parameterization + forward consistency，而不是继续调 decoder、threshold、loss、head 或手工 Bz feature。

## 第 20.42-20.55 方法路线阶段判断

第 20.42-20.55 的结论进一步确认：外部 deep research 报告的核心路线不是普通 segmentation，也不是继续做 dense decoder patch，而是把 MFL 缺陷边界问题视为 inverse reconstruction。当前算法主线已经从 mask-only decoder / combined baseline 评估，切换到：

```text
geometry-aware representation
-> differentiable rasterization
-> forward consistency / forward-model residual
-> low-dimensional refinement
```

几个关键判断如下：

* combined COMSOL_DATA_BASELINE_V3 lightweight decoder 失败主要来自 `component_count=2` 与其他拓扑任务的冲突，不应通过继续加宽普通 decoder 解决。
* topology-gated decoder v1/v2 只是 weak topology-aware decoder patch。它没有显式 geometry 参数、没有 differentiable rasterization、没有 predicted geometry 到 Bz 的 forward residual，因此不属于外部报告的核心 geometry-aware / forward-consistent 方法。
* Piao 2019 当前只适合作为弱适配探索：本项目只有 multi-line Bz / quasi-2D geometry，而论文核心是三轴 MFL、RBC 3D profile、NLS 物理特征和 LS-SVM。20.47-revised 的 Bz-only NLS-style features + SVR/KRR/Ridge 没有通过 acceptance，后续若进入 3D / 三轴数据阶段再考虑更深入的 Piao-style 方法。
* 20.48 证明 differentiable rotated-rectangle rasterizer 与 geometry labels 可用，说明 geometry-aware route 本身有可行性；但 20.49 / 20.50 说明继续修 direct neural geometry head 难以解决 type / angle 学习不足。
* 20.51 的 feature-assisted geometry head + lightweight forward consistency 只带来边际 mask / angle 改善，type confusion 仍未解决。因此 direct Bz -> geometry head 不应继续小修补。
* 20.52 证明 Priewald-style low-dimensional refinement 有正信号：frozen forward surrogate residual 能显著降低 forward NRMSE，并带来小幅 geometry-raster mask 改善，但 geometry-head initializer 偏弱。
* 20.53 的 dense/coarse initializer + refinement 进一步说明，refinement 上限主要受 proposal 质量限制。当前 binary dense mask + PCA rotated bbox extraction 没有超过 20.51 geometry-head proposal，type / angle 初始化仍弱，因此不能作为 candidate 或 baseline。
* 20.54 用 strong dense initializer + improved proposal extraction 显著修复了 proposal 质量：geometry-raster test IoU/Dice 达到 `0.6726 / 0.8017`。但从该强 proposal 做 Priewald-style refinement 后，forward NRMSE 下降而 mask IoU/Dice 回落到 `0.6646 / 0.7958`，说明当前主要 bottleneck 已从 initializer/proposal extraction 转为 forward surrogate mismatch / non-identifiability。
* 20.55 进一步确认 bottleneck 不是简单 waveform fit，而是 residual objective calibration：S1/S2/S3 三个 calibrated surrogate candidate 均未让 residual 与 geometry/mask error 建立非平凡正相关，Stage C refinement 因 gate 未过被跳过。这个阴性结果说明继续调当前 surrogate loss 或 refinement objective 意义有限。

Priewald 2013 对当前阶段更重要的启发不是复现完整 FEM、解析 Jacobian 或 Gauss-Newton 工程，而是 forward-model-based inversion / refinement：用 predicted geometry 通过 forward surrogate 生成 MFL，再用 observed MFL residual 约束低维几何参数。当前应停止 direct neural geometry head 小修补；20.55 之后若继续 Priewald-style refinement，前置条件应是 synthetic perturbation forward data 或等价局部扰动数据，让 surrogate 学到 geometry perturbation 与 signal residual 的局部排序关系。若无法补足这个 calibration 证据，则转向 mask/profile basis refinement，避免被低保真 residual objective 牵引到错误几何。
## 第 20.59 方法路线判断：profile-compatible forward surrogate

第 20.59 将第 20.58 的结论进一步拆开验证：profile basis 本身仍有价值，但 forward consistency 必须换成 profile-compatible forward surrogate，不能把 K=8 profile stations 再压缩成 single rotated-box summary。外部文献路线，尤其 Priewald-style forward-model-based inversion，支持这种判断：关键不是复现完整 FEM Jacobian，而是让 forward residual 对待优化的 shape/profile representation 有一致、可校准的响应。

本轮使用已有 pilot_v9 original samples 和 20.56 perturbation pack 构建 profile-forward dataset，没有运行 COMSOL，也没有生成新数据。`PFS3_profile_station_sequence` 的 waveform fit 可接受（val/test NRMSE `0.3841 / 0.3995`），说明 profile-native 表示可被 forward surrogate 消化；但 validation residual ordering accuracy 只有 `0.6607`，mismatch_rate 为 `0.3393`，未达到 refinement gate。因此第 20.59 不执行 profile-forward refinement retry。

路线判断：profile-compatible surrogate 相比旧 rect-like bridge 有边际价值，但当前 perturbation coverage 太小，不足以支撑连续优化。下一步若继续 forward-guided profile refinement，应先扩展 profile perturbation data；否则保留 20.58 的 no-forward profile basis 作为更稳的 representation 证据，并暂停对当前 forward residual objective 的小调。该阶段仍是 POC，不更新任何 baseline。
## 2026-05-23 路线同步：20.66 true 3D RBC-style smoke pack

第 20.66 是 true 3D / Piao-style 主线的第一个执行 smoke，不是 baseline，也不是正式训练数据集。本轮只验证 `RBC params -> depth/profile grid -> COMSOL 3D/stepped-depth defect -> Bx/By/Bz @ sensor_z_m=0.008 -> delta_b check -> schema validation`，没有训练 surrogate / inverse model，没有做 refinement，也没有更新 `CURRENT_BASELINE.md` 或 COMSOL baseline 文档。

结果分级为 `stepped_depth_smoke_pass`。6 个 RBC-style single-defect samples 的 pure-Python depth/profile validation 全部通过，真实 COMSOL forward 6/6 通过，NPZ/schema validation 6/6 通过。输出 schema 已包含 `rbc_params`、`profile_pose`、`profile_depth_grid_m`、`profile_depth_map_xy_m`、`projected_mask_2d`、`depth_levels_m`、`stepped_depth_approximation` 和 `geometry_params_json`；其中 `projected_mask_2d` 只作为 2D comparator，不能替代 3D profile label。

本轮必须保持诚实边界：`exact_piao_rbc=False`，当前 generator 是 RBC-style / RBC-inspired engineering approximation，不是完整 Piao 2019 RBC 公式复现；`smooth_variable_depth_solid_verified=False`，COMSOL 几何是 5 层 `stepped_depth_layered_approximation`，不是 smooth true variable-depth solid；`constant_depth_extrusion_used_as_success=False`，没有把恒深 extrusion 伪装成 true 3D。

路线判断：true 3D / Piao-style 技术链路已经在 stepped-depth smoke 层级跑通，说明该方向值得继续；但 smooth variable-depth COMSOL geometry 仍是主 blocker。下一步不应直接训练模型，也不应回到 2D profile-forward 小修，而应先决策：继续实现 smooth variable-depth RBC solid，还是接受 stepped-depth approximation 作为 20.67 pilot 的明确标签。

## 2026-05-23 路线同步：20.65 true 3D / Piao-style feasibility design

第 20.65 完成的是 feasibility design，不是数据生成或模型训练。本轮没有运行 COMSOL、没有生成 NPZ / raw CSV / `.mph` / preview PNG、没有训练 surrogate 或 inverse model、没有做 refinement，也没有修改 `CURRENT_BASELINE.md` 或 COMSOL baseline 文档。Claude Code review 通过且无 must-fix。

路线判断更新为：2D top-view profile-forward 小修正式暂停。20.61 expanded profile perturbation、20.62 multi-height Bz、20.63 same-direction Bx/By/Bz、20.64 multi-direction excitation 都没有让真实 COMSOL residual 稳定排序 profile quality；问题已经不是再调一个 2D profile surrogate、再换一个 residual weight 或再扩一点同类 observation 能稳定解决的。下一条主线切到 **true 3D profile / Piao-style geometry profile**，dense mask baseline 只作为 comparator。

Piao-style 的迁移边界必须保持诚实：本轮基于既有 fullpaper alignment summary 和已上传 PDF 的标题、摘要、章节级上下文，不声称重新阅读全文，也不声称完整复现 Piao 2019。可迁移的是 three-axis MFL、RBC six-parameter 3D profile、geometry parameter regression、projection metrics 和 forward consistency；不可直接迁移的是当前 Bz-only、2D top-view mask、2D profile perturbation residual 和完整 PIG experimental setup。

COMSOL 能力边界也同步更新：当前链路支持真实 3D volume solve，并已有 Bx/By/Bz 输出和 `Je` 方向控制证据；但现有 rect/rot/polygon/profile geometry 主要仍是 constant-depth prism / top-view extrusion。RBC depth-varying defect solid、variable-depth surface、loft/sweep/slice-union geometry 仍是第 20.66 必须验证的 blocker，不能写成当前已支持 true 3D profile generation 或 train-ready pack。

第一个 3D pilot 的 representation 选择为 `Piao RBC six params + derived depth/profile grid`：`L, W, D, wLD, wWD, wLW` 是主标签，depth map / projected 2D mask 是派生监督和 QA。第一版只做 single-defect，不做 polygon、multi_defect 或 arbitrary free-form 3D volume。20.66 smoke 只验证 `Bx/By/Bz @ sensor_z_m=0.008`，目标链路是 `RBC params -> depth map -> COMSOL variable-depth defect solid -> same-source projected mask -> delta_B check`；`0.012m` 只作为 20.67 或后续 ablation 的 schema 选项。

20.67 中 projected mask IoU `>=0.65`、Dice `>=0.78`、profile error `<=0.25` 等阈值只作为 preliminary acceptance guidance，不是已验证硬标准。如果 20.66 不能构建 variable-depth true 3D solid，就暂停 geometry-forward route，先解决 COMSOL geometry blocker。

## 2026-05-23 路线同步：20.64 multi-direction excitation profile oracle ordering feasibility

第 20.64 在 20.63 same-direction multi-axis 失败后，进一步验证真实改变 COMSOL excitation / magnetization direction 是否能让 profile perturbation oracle residual 更稳定排序 profile quality。本轮仍是 feasibility POC：不训练 forward surrogate，不训练 inverse model，不做 profile refinement，不更新 baseline。实验固定 `sensor_z_m=0.008`、scan lines 为 `[-0.001, 0.0, 0.001]`，使用 12 base / 96 profile rows，并导出 `[mf.Bx, mf.By, mf.Bz]` 三轴响应。

COMSOL direction probe 证明方向改变是真实的：`direction_0` 使用默认 +Y `Je=["0","1e6[A/m^2]","0"]`，`direction_45` 使用 equal XY，`direction_90` 使用 +X `Je=["1e6[A/m^2]","0","0"]`；`direction_90` 相对 `direction_0` 的 no-defect / defect field response NRMSE 为 `1.6479 / 1.7981`，并且 dominant axis 从 `Bx` 转到 `By`。因此 20.64 没有通过旋转数组、信号或 mask 伪造 multi-direction。

路线判断仍然是负面的。same-pack test baseline `direction_0` Bz-only ordering 为 `0.4545`，`direction_90` Bz-only 提升到 `0.5273`，multi-direction Bz train-std normalized 为 `0.5636`，说明改变 excitation direction 带来边际正信号；但 multi-direction all-axis normalized ordering 只有 `0.3455`，mismatch_rate 为 `0.6545`，residual-error correlation 为 `-0.8028`，未达到 `>0.65` 和 `+0.10` 的主 gate。Claude Code review 通过且无 must-fix，结论是不建议训练 multi-direction profile surrogate。

因此 profile-forward route 的当前瓶颈不是再加 same-direction field components、multi-height Bz 或直接训练 surrogate，而是当前 2D top-view profile + 现有 observation configuration 对 profile quality 的可辨识性仍不足。下一步若继续 geometry/forward 路线，应转向 **true 3D profile / Piao-style route**；20.64 不改变 `CURRENT_BASELINE`，也不创建 COMSOL baseline 文档。

---

## 第 20.67 路线状态：smooth / near-smooth variable-depth geometry feasibility

20.67 继续 true 3D / Piao-style 主线，但只做 COMSOL geometry feasibility。结果分级为 `high_layer_pass`：12-layer high-layer nested contour approximation 比 20.66 的 5-layer stepped-depth smoke 更进一步，并且完成了 `medium_round` 的 Bx/By/Bz @ `sensor_z_m=0.008` forward 与 schema validation。

这不是 `variable_depth_pass`。当前 smooth / loft / imported closed-surface route 只完成有限 probe，尚未形成 verified closed smooth defect body，也不能写成 exact Piao RBC geometry。`exact_piao_rbc=False` 仍然保留，projected mask 仍只作 2D comparator，3D label 仍是 RBC params + depth/profile grid/map。

路线判断：
- 2D profile-forward 小修继续暂停。
- true 3D / Piao-style route 仍是主线，但下一步必须先由人工确认是否接受 high-layer approximation 作为 pilot approximation。
- 若不接受 approximation，下一步应继续修 smooth / closed-surface COMSOL geometry builder，而不是扩样或训练。
- dense mask baseline 仍只作 comparator，不回到主线。
## 第 20.73 路线状态：true 3D RBC pilot training gate

20.73 是 true 3D / Piao-style 主线的第一个 training gate，不是 baseline gate。数据包固定为 `comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled`，通过 registry + manifest 显式加载；`allowed_use` 只允许 `schema_validation` 和 `explicit_pilot_training_gate`，`forbidden_use` 继续禁止 automatic mainline training、baseline update 和 current baseline replacement。

结果说明 true 3D RBC-style 参数反演有信号，但当前 N=56 只够做 learnability 诊断。Feature sanity 的 selected test normalized MAE 为 `0.7564`；small Conv1D 的 selected test normalized MAE 为 `0.7601`，projected mask Dice 为 `0.8347`，说明 raw Bx/By/Bz delta_B 输入能恢复一部分投影形状，但还没有超过手工特征 sanity。完整训练轨迹能把 train normalized MAE 拟合到 `0.0012`，因此当前瓶颈不是 train fit，而是小样本泛化和 curvature 可辨识性。

参数层面的分界点是：`L_m`、`W_m` 有初步可学习信号，`D_m` 仍偏弱，`wLD/wWD/wLW` 三个 curvature 参数在 test 上没有稳定学习。下一阶段主线应扩展 imported watertight mesh solid 的 true 3D RBC 数据量到 120/240，并保持 dataset registry / manifest gate；dense mask baseline 继续只作为 comparator，不回到主线，也不更新 `CURRENT_BASELINE.md`。
## 2026-05-25：第 20.74 true 3D RBC v2_120 数据扩展状态

20.74 继续 true 3D / Piao-style 主线，但只做 imported-watertight 数据扩展，不训练、不更新 baseline、不修改 `CURRENT_BASELINE.md`。数据身份为 `comsol_true_3d_rbc_imported_watertight_pilot_v2_120`，仍明确标记 `exact_piao_rbc=False`、`rbc_style_approximation=True`、`geometry_method=imported_watertight_mesh_solid`。

执行结果：v1 assembled N=56 作为 source，20.74 top-up plan 为 80 rows，Python watertight mesh 80/80 通过；COMSOL full-source top-up 成功 56/80，7 fail，17 not attempted，使用 20.70 material/domain fix、`mesh_auto_size=5`、`Jscale=1.0`、`[mf.Bx,mf.By,mf.Bz] @ sensor_z_m=0.008`，没有 high-layer fallback。assembled v2_120 实际 N=112，split=76/18/18，curvature coverage 为 sharp=22、round=23、boxy=23、LD_dominant=24、WD_dominant=20；`delta_b=b_defect-b_no_defect` 校验为 0.0，schema/registry/manifest validation 全部通过。

路线判断：v2_120 达到 `pilot_generated` 和 `train_ready_candidate=True`，但 `baseline_ready=False`。下一步应对 v2_120 做显式 dataset_id + manifest gated 的 true 3D training gate；dense mask baseline 继续只作为 comparator，不能把 v2_120 自动接入 mainline training 或替换 current baseline。

## 2026-05-25：第 20.75 true 3D RBC v2_120 training gate 状态

20.75 是 true 3D / Piao-style 主线在 v2_120 上的第二轮 training gate，不是 baseline gate。数据包固定为 `comsol_true_3d_rbc_imported_watertight_pilot_v2_120`，通过 `COMSOL_DATA_REGISTRY.md` 和 manifest 显式加载；`allowed_use` 只允许 `schema_validation` 和 `explicit_pilot_training_gate`，`forbidden_use` 继续禁止 automatic mainline training、baseline update 和 current baseline replacement。本轮没有运行 COMSOL，没有生成或修改 NPZ，也没有更新 `CURRENT_BASELINE.md`。

结果相比 20.73 有明确改善。Feature sanity validation 选择 `svr_rbf_C10`，test normalized MAE 为 `0.7677`；small Conv1D validation 选择 seed `42`，test normalized MAE 为 `0.7039`，优于 mean baseline `0.8803` 和 feature baseline `0.7677`。相对 N=56，neural test MAE 从 `0.7601` 降到 `0.7039`，L/W/D MAE 改善到 `2.51/2.59/1.11 mm`，curvature MAE 从 `0.2095` 降到 `0.1905`，projected mask Dice 稳定在 `0.8364`，profile depth RMSE 改善到 `0.000548 m`。

路线判断：v2_120 证明扩样有效，`L_m`、`W_m`、`D_m` 已有可学习信号；但 `wLD/wWD/wLW` 仍不是稳定可辨识参数，N=112 和 val/test 各 18 个样本仍不足以建立 baseline。下一阶段应优先扩展 true 3D RBC imported-watertight 数据到 240，并保持 dataset registry / manifest gate；dense mask baseline 继续只作为 comparator，不回到主线。
## 2026-05-25 路线同步：20.76 true 3D RBC v3_240 数据扩展

20.76 继续 true 3D / Piao-style 主线，但仍是 RBC-style / Piao-inspired engineering approximation：`exact_piao_rbc=False`、`rbc_style_approximation=True`、`geometry_method=imported_watertight_mesh_solid`。本轮只扩展数据，不训练、不建立 baseline、不替换 `CURRENT_BASELINE.md`；dense mask baseline 继续只作为 comparator。

v3_240 的核心结果是：source v2_120 N=112 保持不变，20.76 top-up plan 160 rows，Python watertight mesh 160/160 pass，COMSOL full-source top-up 128/160 success；成功样本使用 20.70 imported-solid material/domain/solver protocol，`Jscale=1.0`，三轴 `[mf.Bx,mf.By,mf.Bz]` 真实导出，`delta_b=b_defect-b_no_defect` 校验通过。Assembled dataset `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 达到 N=240、split=162/39/39、curvature=48/49/47/46/50，schema/registry/manifest validation 通过，status=`pilot_generated`，train_ready_candidate=True，baseline_ready=False。

路线判断：20.76 已把 true 3D RBC route 推进到可进行下一轮显式 training gate 的数据规模，但仍不能写成 baseline。下一步应在 v3_240 上复用 20.73/20.75 的 registry/manifest-gated training gate，验证扩样是否继续改善 `D_m` 与 `wLD/wWD/wLW` 的可辨识性；如果 curvature 仍不稳定，再决定是 targeted curvature/depth top-up、模型改进，还是引入更接近 Piao NLS/LS-SVM 的特征管线。
# 2026-05-25 路线同步：第 20.79 curvature-aware refinement

第 20.79 在 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上测试了 curvature-aware head / loss / encoder refinement，仍然是 model refinement，不是 baseline。数据入口继续通过 registry + manifest 显式 dataset_id 加载；没有运行 COMSOL，没有生成或修改 NPZ，没有更新 `CURRENT_BASELINE.md`。

结果没有支持升级模型：validation-only selection 选中 `C1_split_heads`，但 selected test normalized MAE 从 20.77 的 `0.678014` 退化到 `0.753387`，curvature MAE 从 `0.201076` 退化到 `0.211584`，projected mask Dice 从 `0.847727` 降到 `0.834597`，且 L_m / D_m 也退化。`wLD` 仍是最弱 curvature 参数。`C2` 的 test 指标较好但没有被 validation 选中，不能作为模型选择依据。

路线判断：v3_240 仍保留为第 20.77 的 formal benchmark candidate；第 20.79 refined model 不进入 candidate upgrade，不写 baseline。下一步优先 exact Piao / NLS-inspired feature pipeline，用更贴近三轴 MFL 物理特征的 comparator 诊断 curvature identifiability；curvature-targeted data top-up 是第二选择。dense mask baseline 继续只作为 comparator。
## 2026-05-25 路线同步：第 20.81 feature-fusion neural model

20.81 继续 true 3D / Piao-style RBC 主线，但仍是 model refinement，不是 baseline。输入固定为 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`，通过 registry + manifest 显式加载；没有运行 COMSOL，没有生成或修改 NPZ，没有更新 `CURRENT_BASELINE.md`。`exact_piao_rbc=False` 和 `rbc_style_approximation=True` 继续保留，dense mask baseline 仍只作为 comparator。

feature-fusion 的设计边界是：20.77 neural encoder 继续负责 raw `delta_b` 中的 L/W/D 和 mask/profile 表达；20.80 的 Piao/NLS-inspired physical features 只作为 curvature 辅助输入。候选筛选后选中 `H3_curv_fusion_F0F1F2_w1p0`，multi-seed validation selection 选中 seed `2026`。结果显示 total MAE 从 20.77 的 `0.678014` 改善到 `0.667888`，projected mask Dice 从 `0.847727` 改善到 `0.866573`，说明 physical feature fusion 对整体拟合和 mask 有帮助。

但 curvature 风险没有被解决：curvature MAE 只从 `0.201076` 到 `0.194483`，未达到 `>=0.01` 实质改善门槛；`wLD` 从 `0.209439` 退化到 `0.217079`；同时 L/D 和 profile depth RMSE 存在 trade-off，且 curvature 仍弱于 20.80 feature-only 的 `0.190304`。因此路线判断更新为 `feature_fusion_total_not_curvature`：feature-fusion 可以作为 auxiliary representation evidence 保留，但不能升级 benchmark candidate，也不能替代 20.77 reference，更不能写入 baseline。

下一步主线应转向 curvature label / output representation redefinition：先判断 `wLD/wWD/wLW` 这组 RBC-style curvature 参数在当前 Bx/By/Bz、projected mask 和 depth/profile grid 口径下是否是合适的监督目标；必要时改成更直接的 profile/depth basis、curvature field、depth-grid auxiliary target 或重新定义的 shape factors。curvature-targeted data top-up 和 exact Piao feature reproduction 应排在这个审计之后，而不是先继续 head/loss 小修。

# 2026-05-26 Stage 20.83 route note

20.83 tested the R1 route: keep the six RBC-style parameters as outputs, but train with profile-primary loss. The experiment did not pass the upgrade gate. It improved projected mask Dice but worsened the primary 3D profile reconstruction metric relative to 20.77.

Current route state:
- `wLD / wWD / wLW` remain auxiliary diagnostics, not headline pass/fail metrics.
- `profile_depth_rmse_m` / Er-like profile reconstruction are the right main metrics for the true 3D RBC-style branch.
- 20.77 remains the stronger profile reconstruction reference.
- 20.81 remains useful as a visual/mask comparator.
- 20.83 is a negative result and must not be written as a baseline or `CURRENT_BASELINE` replacement.

Next route direction should avoid another narrow weight tweak on the same six-param loss. If the route continues, prefer a profile-native output representation experiment, while retaining `exact_piao_rbc=False` and `rbc_style_approximation=True`.

# 2026-05-26 Stage 20.84 route consolidation

Stage 20.84 closed the current true 3D RBC candidate ambiguity without new training or data generation. The branch should now treat 20.77 as the profile/depth benchmark candidate, 20.81 as the non-negative projected-mask / visual comparator, and 20.83 as negative evidence for the current R1 profile-primary loss design.

The important route correction is that projected mask Dice and 3D profile quality must remain separate. 20.83 has the numerically highest Dice among the three compared candidates, but its profile RMSE is worse than 20.77; therefore it is not a replacement. 20.81 is retained as the visual comparator because it is not a negative profile-depth gate and has strong projected-mask behavior.

Next route direction: use 20.77 for a formal benchmark rerun if consolidating the current candidate, or move to a more profile-native output representation if improving the method. Do not update `CURRENT_BASELINE.md`, do not write v3_240 as a baseline, and keep `exact_piao_rbc=False` / `rbc_style_approximation=True`.
