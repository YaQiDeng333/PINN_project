# PINN 优化路线

## 2026-05-29 路线同步：21.3 internal defect dataset expansion plan

21.3 不是训练阶段，而是把 internal / buried defect 分支的数据覆盖问题收口成可执行的扩展方案。`comsol_internal_defect_pilot_pack_v1` 的 N=96 可以复用，但旧 split 不能复用：val/test 都只有 `internal_cuboid`，test burial 只覆盖 `deep/deep_plus`，所以 21.2 的 shape accuracy 只能说明 cuboid-only 局部信号，不能说明三类 shape 泛化。

下一阶段的分界点是 split 质量而不是模型结构。21.3 规划 `comsol_internal_defect_pilot_pack_v2_240`，目标 assembled N=240，source N=96，selected top-up N=144，planned top-up N=168；v2 split 重新固定为 `160/40/40`，并要求每个 split 都覆盖三类 shape、四档 burial depth、三档 size，以及 ellipsoid/cuboid 的三种 aspect。

路线判断：先执行 21.3b 生成 top-up COMSOL pack，并完成 v2 assembly/validation；21.4 才重新做 internal training gate。internal branch 仍然独立于 surface / near-surface RBC baseline，`CURRENT_BASELINE.md` 保持不变。

## 2026-05-28 路线同步：21.2 internal defect training gate

21.2 把 internal / buried defect 分支推进到第一个 training gate，但结论仍是“可学习性成立，泛化证据不足”。`comsol_internal_defect_pilot_pack_v1` 通过 registry/manifest 显式加载，模型输入只使用三轴 `delta_b/BxByBz`，labels 和 metadata 只用于 supervision/metrics；本轮没有运行 COMSOL，没有修改 NPZ，没有更新 `CURRENT_BASELINE.md`。

结果分界点在这里：neural 的 shape classification 与 center_xyz 有信号，但 split 设计暴露出结构性 blocker。validation/test 都只包含 `internal_cuboid`，burial depth 也覆盖不完整，所以 shape accuracy `0.812500` 不能解释为三类 shape 泛化能力；同时 pure regression 上 selected feature baseline `svr_rbf_C10` 的 test total MAE `0.878883` 和 burial_depth MAE `0.922 mm` 仍优于 neural 的 `1.004271` 和 `1.947 mm`。

路线判断：internal branch 不能升级 baseline，也不能和 surface RBC baseline 合并。下一步应扩展 internal dataset，并重做分层 split，让 train/val/test 都覆盖三类 shape、四档 burial depth、三档 size 和主要 aspect；之后再判断是否进入 internal formal training gate。当前 20.85 surface / near-surface true 3D RBC baseline 与 A2 liftoff companion 仍保持原角色。

## 2026-05-28 路线同步：21.1 internal defect pilot pack

21.1 把 internal / buried defect 分支从 feasibility smoke 推进到 pilot pack。`comsol_internal_defect_pilot_pack_v1` 已生成并验证为 `pilot_generated`：96/96 COMSOL rows 成功，覆盖三种 shape、四档 burial depth、三档 size，并保留 `train/val/test=64/16/16`。它是 internal branch 的显式 training-gate 数据候选，不是 current baseline，也不替换 20.85 surface / near-surface true 3D RBC profile-depth baseline。

路线边界继续保持清楚：surface RBC branch 学的是 surface profile/depth；internal branch 学的是内部空腔几何、中心位置和 `burial_depth_m/depth_to_surface_m`。如果把 internal defect 强行映射成 surface RBC 六参数，会把埋深变化误解释为表面 profile/curvature 变化。因此 21.2 应单独做 internal defect training gate，并继续使用 `COMSOL_DATA_REGISTRY.md + manifest + dataset_id` 显式加载；真实实验数据仍暂缓。

## 2026-05-27 路线同步：20.89 gain/amplitude calibration and augmentation gate

20.89 把 20.88 暴露出的 gain / Bx-amplitude sensitivity 单独拆出来验证。路线边界保持不变：不运行 COMSOL，不生成或修改 data / NPZ，不更新 `CURRENT_BASELINE.md`；本轮只在固定 v3_240 和 20.88a frozen artifact 上测试输入校准与 in-memory augmentation gate。

校准和增强都证明“幅值归一化方向有用”，但都不能替代当前 baseline。Validation-selected `per_axis_rms_train_stats` 把 gain 0.8 / 1.2 的 test profile degradation 压到 `21.194% / 21.194%`，Bx 50% attenuation 压到 `12.331%`，但 clean profile 同样退化 `21.194%`，超过 `<=10%` clean gate。Validation-selected augmentation candidate `A2_axis_gain_aug` seed `123` 把 gain 0.8 degradation 压到 `24.614%`、Bx 50% attenuation 压到 `59.279%`，但 clean profile RMSE 退化 `35.464%`，gain 1.2 仍为 `38.768%`，所以只能作为 non-baseline robustness diagnostic。

路线判断更新为：20.85 true 3D RBC profile-depth baseline 继续作为 `CURRENT_BASELINE`；gain/amplitude calibration 是真实数据接入前的明确 blocker；不要继续用小型 augmentation 直接替换 baseline。下一步应进入 20.90 liftoff / sensor-offset COMSOL diagnostic pack，并在该 pack 中显式记录 source/gain control、Bx channel dependence 和 amplitude normalization 假设。

## 2026-05-26 路线同步：20.88 observation perturbation robustness audit

20.88 用 20.88a 恢复的 frozen baseline artifact 完成了 observation-space robustness audit。路线边界保持不变：不运行 COMSOL，不训练，不生成或修改 data / NPZ，不改 `CURRENT_BASELINE.md`；本轮只回答“当前 true 3D RBC baseline 对已有 `delta_b` 观测扰动是否稳”。

结果把下一阶段风险切得更清楚：随机噪声不是首要瓶颈，noise 10% 仍是 green，profile RMSE degradation `4.095415%`，Dice drop `-0.000252`；reference subtraction error 和 sensor_x jitter 也没有触发 fail。真正敏感的是幅值标定和通道依赖，global gain 0.8x 造成 `123.845240%` profile RMSE degradation，Bx 50% attenuation 造成 `141.577253%` degradation，Bx missing 造成 Dice drop `0.163825`。这说明当前 baseline 更依赖绝对幅值和 Bx 通道完整性，不能写成 broad robust。

路线更新为：20.89 仍应做 liftoff / sensor-offset COMSOL diagnostic pack，因为 observation-space gain/jitter 不能替代真实传感器几何变化；同时 20.92 之前需要考虑 gain normalization、amplitude calibration 或 augmentation gate。真实实验数据继续后置；如果不先处理幅值标定和 COMSOL liftoff/sensor-offset，直接 real-data alignment 风险过高。

## 2026-05-26 路线同步：20.88a baseline inference artifact recovery

20.88a 解决了 20.88 preflight 暴露的 frozen artifact blocker。当前 true 3D RBC baseline 仍保持为 20.86 transition 后的 profile-depth baseline；本轮没有改 `CURRENT_BASELINE.md`，也没有生成新数据、修改 NPZ、运行 COMSOL 或改变模型路线。

路线状态更新为：`comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 的 20.77/20.85 seed=42 small Conv1D + MLP six-parameter head 已导出可复用 inference artifact。checkpoint 和 raw prediction artifact 放在 ignored 的 `checkpoints/true_3d_rbc_baseline_artifacts/`，提交的只是 `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json`。clean verification 精确复现 20.85 的 profile-depth 主指标和 projected-mask QA 指标。

因此下一步可以回到 20.88 observation perturbation robustness audit：只在内存中扰动现有 v3_240 `delta_b`，用 frozen artifact 前向评估 noise、gain、zero drift、reference subtraction error、jitter 和 channel dropout。20.88 仍然不是训练阶段，也不是 COMSOL 数据生成阶段；如果 observation perturbation 暴露敏感项，再根据 20.87 路线进入 augmentation 或 20.89 liftoff/sensor-offset COMSOL diagnostic pack。

## 2026-05-26 路线同步：20.88 observation robustness preflight blocker

20.88 没有进入 observation perturbation robustness 评估，而是在 preflight 停止。原因不是 dataset/schema 问题：v3_240 registry / manifest gate 通过，输入 shape、split、Bx/By/Bz 轴和 profile/mask label 均可用；真正 blocker 是 frozen model artifact 缺失。当前仓库只有 20.77/20.85 的 clean metrics 和 per-sample profile error rows，没有能对扰动 `delta_b` 重新前向的 seed=42 checkpoint 或 raw prediction artifact。

路线判断因此暂时前移一格：在评估 noise、gain、zero drift、reference subtraction error、channel dropout、jitter 之前，必须先做 artifact recovery/export。优先恢复 20.77/20.85 selected checkpoint；无法恢复时，再单独批准固定 20.85 protocol 的 artifact-export rerun。该 rerun 不能被写成 20.88 的一部分，因为 20.88 明确禁止训练。恢复 artifact 后，再回到 20.88 做 frozen-model observation perturbation audit。

## 2026-05-26 路线同步：20.87 true 3D RBC robustness expansion design

20.87 将第 20.86 的 true 3D RBC profile-depth baseline 后续路线拆成“先仿真鲁棒性、再缺陷类型扩展、最后再考虑真实实验对齐”的阶段计划。本轮不运行 COMSOL、不生成新数据、不训练、不修改 `CURRENT_BASELINE.md`；当前 baseline 仍是 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上的 Bx/By/Bz `delta_b -> six RBC-style params -> 3D profile/depth/projected mask`。

路线分层已经固定：Layer 1 是 observation-space perturbation，可在 20.88 用现有 v3_240 评估 frozen baseline，包括 noise、gain scaling、zero drift、no-defect reference error、channel dropout，以及只作诊断的 `sensor_x_resampling_jitter`。Layer 2 是 sensor/physics setting variation，可近似模拟但正式结论需要 20.89 COMSOL diagnostic pack，包括 liftoff、scan_line_y offset、Bx/By/Bz spatial misalignment、source strength 和 material/B-H proxy。Layer 3 是新几何或新标签问题，必须新 COMSOL 和 schema：surface shape extension、internal/buried defect、multi-defect、free-form profile。

路线边界也同步固定：internal/buried defect 不是当前 surface-breaking RBC baseline 的 robustness 子问题，而是 20.91 的独立 feasibility/schema 分支，至少要先定义 `burial_depth`、`depth_to_surface` 和新的 profile/mask 语义。真实实验数据继续后置；若 20.88/20.89 的 clean simulation robustness 不通过，不应直接进入 real-data alignment。下一步唯一主线是 20.88 observation perturbation robustness audit。

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
# 2026-05-28 Stage 21.0 route note

21.0 把 internal / buried defect 从 schema 设计推进到 COMSOL feasibility smoke。`comsol_internal_defect_smoke_pack_v1` 已完成 12/12 rows，三类内部缺陷 `internal_sphere`、`internal_ellipsoid`、`internal_cuboid` 均通过 Boolean subtract、mesh/solve、Bx/By/Bz export 和 `delta_b=b_defect-b_no_defect` validation。

路线边界保持不变：internal defect 不是 surface RBC baseline 的 top-up。当前 `CURRENT_BASELINE.md` 仍是 20.85 surface / near-surface true 3D RBC profile-depth baseline，A2 仍只是 liftoff companion module；internal branch 的核心标签是 `shape_type + L/W/D + burial_depth_m / depth_to_surface_m + defect_center_xyz_m`，不能强行套 surface RBC 的 profile/depth 语义。

下一步路线：进入 21.1 internal pilot pack。21.1 应扩大样本数并继续固定三轴 `Bx/By/Bz`、no-defect reference、`sensor_z_m`、坐标系、材料/试件几何和 ground truth method；Bz-only 只能作为低能力诊断分支。真实实验 internal block 继续暂缓，直到 internal pilot pack 和 validator 更稳定。

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

# 2026-05-29 Stage 21.5 route note

21.5 把 internal / buried defect 分支的当前角色收口为 benchmark candidate，而不是 baseline。`internal_v2_conv1d_multitask` seed `42` 是当前 neural candidate：total normalized MAE `0.406366`，优于 selected feature baseline `0.416406`；shape_type 和 center_xyz 也更强。但 burial_depth 是最清楚的风险项，feature baseline `0.472 mm` 优于 neural `0.595 mm`，group-level audit 也显示 feature 在 burial_depth 上系统性更强。

这说明 internal branch 已经不是“能不能学”的问题，而是“埋深如何更稳地学”的问题。下一步应聚焦 burial-depth head/model：可以做 feature-fusion burial diagnostic、shape-conditioned burial head、或调整 burial-depth loss/selection；shape-conditioned model 是次优 ablation。不要直接 baseline transition，不要把 internal defect 写进 surface / near-surface `CURRENT_BASELINE`。

# 2026-05-29 Stage 21.4 route note

21.4 在 `comsol_internal_defect_pilot_pack_v2_240` 上完成 internal/buried defect training gate。v2_240 修复了 21.2 的 split blocker，因此这次结果可以作为 internal branch 的正向 training-gate 证据：三轴 `Bx/By/Bz delta_b` 能学习尺寸、中心位置和 shape_type，burial_depth 也有信号。

关键结果是 neural selected seed `42` 的 test total normalized MAE 为 `0.406366`，优于 selected feature baseline `0.416406`；shape accuracy/F1 为 `1.000000 / 1.000000`，L/W/D MAE 为 `0.761 / 0.947 / 0.093 mm`，center_xyz MAE 为 `1.380 mm`。同时必须保留一个风险判断：burial_depth 单项上 feature baseline 更强，`0.472 mm` 优于 neural 的 `0.595 mm`，说明下一步不能只看 total score。

路线状态：internal branch 已从 feasibility smoke / pilot pack 进入 positive training gate，但仍不是 baseline。下一步应做 formal benchmark/report，复核 seed stability、feature-vs-neural trade-off 和 burial_depth 风险；不要更新 `CURRENT_BASELINE.md`，不要把 internal defect 混入 surface / near-surface RBC baseline。

# 2026-05-29 Stage 21.3b route note

21.3b 把 internal defect 数据分支从 21.2 的 split blocker 推进到可训练的数据包状态。`comsol_internal_defect_pilot_pack_v2_240` 由 v1 source N=96 和 top-up selected N=144 组装而成；COMSOL top-up 168/168 成功，assembled N=240，split=`160/40/40`。

路线上的关键变化是：internal/buried defect 现在有了更可信的 train/val/test 覆盖。每个 split 都覆盖三类 shape、四档 burial depth、三档 size，ellipsoid/cuboid 的三类 aspect 也在每个 split 内出现。这直接修复了 21.2 中 val/test 只有 `internal_cuboid`、burial 覆盖不完整的问题。

下一步只能是 21.4 internal v2_240 training gate。该数据包 `train_ready_candidate=true`，但仍是 internal branch 的显式训练 gate 数据，不是 baseline，`baseline_ready=false`，也不改变 surface / near-surface true 3D RBC `CURRENT_BASELINE`。

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

# 2026-05-27 Stage 20.90 route note

Stage 20.90 adds the first small COMSOL diagnostic for physical acquisition variation around the current true 3D RBC baseline. The diagnostic uses 12 base geometries, 96 COMSOL variant rows, and 36 nominal-derived postprocess axis-misalignment rows. It is not training, not a baseline replacement, and does not change `CURRENT_BASELINE.md`.

The route implication is narrow but important: liftoff is now the main robustness blocker. Raw source/amplitude variation can be reduced by the fixed 20.89 `per_axis_rms_train_stats` calibration, but calibration is only a diagnostic/acquisition caveat because it changes clean-scale behavior and is not promoted into the baseline. Scan-line offset and small sensor_x axis misalignment are lower risk in this pack.

Next route direction: build a dedicated COMSOL liftoff robustness / augmentation data design before internal defect feasibility. Internal/buried defects still require a separate label schema and should not be mixed into the current surface RBC profile-depth baseline.
# 2026-05-27 Stage 20.91 route note

Stage 20.91 is a plan-only follow-up to the 20.90 liftoff diagnostic. It defines a dedicated liftoff augmentation pack with 48 base geometries and four paired liftoff levels per base (`0.006 / 0.008 / 0.010 / 0.012 m`), for 192 planned COMSOL rows. It does not run COMSOL, generate data, train, or update `CURRENT_BASELINE.md`.

The route implication is that liftoff robustness should be addressed before internal/buried defects and before real-data claims. Calibration remains an acquisition diagnostic caveat only. The next method gate should compare the current unconditioned baseline family against a scalar `sensor_z_m` conditioned liftoff-aware variant in 20.92, after the pack is generated and validated.

# 2026-05-27 Stage 20.91b route note

Stage 20.91b executed the dedicated liftoff COMSOL pack and produced a full paired dataset: 48 base geometries, four liftoff levels per base (`0.006 / 0.008 / 0.010 / 0.012 m`), 192/192 successful rows, and 48/48 complete paired bases. The pack is registered as `comsol_true_3d_rbc_liftoff_aug_pack_v1` under `true_3d_piao_style_liftoff_robustness`.

This does not change `CURRENT_BASELINE.md`. The route now moves to 20.92 liftoff-aware training gate: compare the current unconditioned model family with a scalar `sensor_z_m` conditioned variant. Internal/buried defects remain deferred until surface-defect liftoff robustness is understood.
# 2026-05-27 Stage 20.92 route note

Stage 20.92 tested whether the full 20.91b liftoff pack can produce a liftoff-robust true 3D RBC model. The data gate passed: `comsol_true_3d_rbc_liftoff_aug_pack_v1` has 48 paired base geometries, four liftoff levels per base, and grouped base-level train/val/test splits with no geometry leakage.

The route result is mixed. Liftoff augmentation clearly helps non-nominal rows: selected `C1_unconditioned_liftoff_aug` seed `123` reduced test non-nominal profile RMSE from the fixed C0 baseline `0.000874310 m` to `0.000659761 m` and improved projected Dice from `0.683351` to `0.833129`. But it does not preserve the nominal operating point: nominal `0.008 m` profile RMSE regressed from `0.000333059 m` to `0.000809011 m`. C2 sensor_z conditioning did not win by validation selection, and its C1/C2 comparison remains a post-hoc diagnostic rather than a selection criterion.

Therefore 20.92 is not a baseline transition and not yet a formal robustness candidate. `CURRENT_BASELINE.md` remains the 20.85 nominal true 3D RBC profile-depth baseline. The next route should inspect liftoff failure cases and design a nominal-preserving liftoff objective or paired-liftoff consistency protocol before internal defects, real-data claims, or more broad COMSOL expansion.

# 2026-05-27 Stage 20.93 route note

Stage 20.93 closes the first liftoff-aware training gate with a route correction. The useful signal from 20.92 is real but incomplete: unconditioned liftoff augmentation improved non-nominal rows, yet it severely damaged nominal `0.008 m` behavior. The failure mechanism is not lack of data volume; it is an unconditioned mixed-liftoff inverse problem plus validation that did not explicitly protect the 20.85 nominal baseline path.

The next route should not continue C1-style unconditional augmentation. Use a nominal-preserving baseline+liftoff adapter as the primary next experiment: keep the 20.85 nominal path anchored and learn a small `sensor_z_m`-conditioned correction for non-nominal liftoff. A revised full `sensor_z_m`-conditioned model remains the secondary ablation, and paired liftoff consistency can be added as a regularizer if the adapter objective is stable.

No new COMSOL is needed before that training gate. `CURRENT_BASELINE.md` remains unchanged, and internal/buried defects plus real-data alignment stay deferred until liftoff robustness can preserve nominal behavior while improving non-nominal rows.

# 2026-05-27 Stage 20.94 route note

Stage 20.94 validates the 20.93 route correction. The selected `A2_latent_residual_adapter` uses the frozen 20.85/20.77 baseline latent and baseline six-parameter prediction plus `sensor_z_m` to predict a residual correction. It keeps the nominal operating point effectively intact while cutting non-nominal liftoff profile RMSE by about half.

This forms a liftoff robustness candidate, not a baseline replacement. The current baseline remains the 20.85 nominal true 3D RBC profile-depth baseline. The next route should be a formal liftoff benchmark for A2, with validation-only selection and grouped base splits retained. Internal/buried defects and real-data alignment remain deferred until the liftoff benchmark confirms stability.

# 2026-05-28 Stage 20.95 route note

Stage 20.95 formalizes the 20.94 A2 result as a companion robustness path. `A2_latent_residual_adapter` seed `2026` preserves the nominal `0.008 m` operating point (`0.000333059 m -> 0.000335821 m`, `+0.829%`) and improves non-nominal liftoff profile RMSE (`0.000874310 m -> 0.000437214 m`, `-49.993%`) with non-nominal Dice rising from `0.683351` to `0.842378`.

The route status is now split deliberately: `CURRENT_BASELINE.md` remains the 20.85 nominal true 3D RBC profile-depth baseline, while A2 is accepted only as its liftoff companion module. `sensor_z_m` is a required metadata field for multi-liftoff or real-experimental inference. Internal/buried defect feasibility remains deferred; the next route step is a liftoff-conditioned inference smoke stage that verifies baseline + A2 loading and the `sensor_z_m` metadata contract.

# 2026-05-28 Stage 20.96a route note

Stage 20.96a resolves the 20.96 blocker: A2 now has a loadable inference artifact. The recovered artifact uses the fixed 20.94 `A2_latent_residual_adapter` protocol and seed `2026`, with the 20.85 baseline frozen. The checkpoint and prediction artifact remain in ignored `checkpoints/` paths, and the tracked manifest records the model config, baseline manifest, normalization, input contract, and routing contract.

Verification reproduced the formal A2 metrics: nominal RMSE `0.000335821 m`, non-nominal RMSE `0.000437214 m`, and non-nominal Dice `0.842378`. This does not update `CURRENT_BASELINE.md`; it only enables the next liftoff-conditioned inference smoke. The next route step is to exercise live routing: nominal `sensor_z_m≈0.008` uses the 20.85 baseline, non-nominal liftoff uses baseline + A2, and missing `sensor_z_m` must fail rather than being guessed.

# 2026-05-28 Stage 20.96 route note

Stage 20.96 turns the baseline plus A2 companion result into a usable inference path. The route is now explicit: `delta_b` plus mandatory `sensor_z_m` enters the runner; nominal `0.008 m` uses the frozen 20.85 baseline, and non-nominal liftoff uses the frozen baseline plus the A2 latent residual adapter. Override modes exist only for audit: `force_baseline` and `force_adapter`.

The smoke result preserves the split baseline/companion role. Auto routing keeps nominal profile RMSE at `0.000333059 m` and reproduces A2 non-nominal RMSE `0.000437214 m`; force-baseline non-nominal RMSE remains `0.000874310 m`. `sensor_z_m` is now a required metadata field for multi-liftoff or real-data inference. Values outside `[0.006, 0.012]` are flagged, and missing liftoff is a hard error.

This is not a baseline replacement. `CURRENT_BASELINE.md` remains the 20.85 nominal true 3D RBC profile-depth baseline, while A2 remains a liftoff robustness companion module. The next route should move to real-data schema intake and acquisition metadata definition before real-data claims; internal/buried defects remain deferred.
# 2026-05-28 Stage 20.97 route note

Stage 20.97 moves the true 3D RBC route from inference smoke into real-data intake definition. The current model path is unchanged: `CURRENT_BASELINE.md` remains the 20.85 nominal true 3D RBC profile-depth baseline, and A2 remains a liftoff robustness companion module selected only when valid non-nominal `sensor_z_m` metadata is present.

The real-data boundary is now explicit. The current route requires tri-axis `Bx/By/Bz`, a trusted no-defect reference or prepared `delta_b`, Tesla units, three scan lines, 201 x-samples, known axis order, known coordinate system, `sensor_z_m` in meters, sensor alignment status, gain calibration status, specimen/material metadata, and magnetization setup. Missing `sensor_z_m`, missing reference subtraction, Bz-only input, unknown units/axis order, or internal/buried defects are blockers for this branch.

The next route is a manifest-only dry run before any real signal array is accepted. This keeps real-data alignment grounded in acquisition metadata instead of silently forcing incomplete observations through the COMSOL-trained baseline.

# 2026-05-28 Stage 20.98 route note

20.98 用 20.97 的真实数据接入模板做了一次 manifest-only dry run。它没有读取真实信号数组，没有生成 data/NPZ，没有训练，没有运行 COMSOL，也没有修改 `CURRENT_BASELINE.md`。

这次 dry run 的关键判断是：用户当前的“内部开缺陷铁块”不属于当前 true 3D RBC surface / near-surface baseline 的适用范围。manifest 已标记 `defect_location_type=internal_or_buried`，validator 因此给出 hard blocker；同时 `sensor_z_m`、三轴 `Bx/By/Bz`、no-defect reference、轴顺序、三条扫描线、201 点 `sensor_x_m`、单位、坐标系、传感器对齐状态、gain 状态和励磁设置都还未知，所以 `ready_for_inference=false`。

路线结论：不要把 internal/buried defect 强行混入当前 surface RBC schema。下一步应先创建 internal defect feasibility schema，单独定义 burial depth / depth-to-surface、内部缺陷几何标签、no-defect reference、采集几何和真实数据 validator；当前 20.85 baseline 与 A2 liftoff companion 继续只服务 surface / near-surface RBC-style 分支。

# 2026-05-28 Stage 20.99 route note

20.99 建立了 internal / buried defect 的独立 feasibility schema，但没有执行 COMSOL、没有生成 data/NPZ、没有训练，也没有更新 `CURRENT_BASELINE.md`。

真正的分界点是 `burial_depth_m`。surface RBC baseline 的输出语义是表面 profile/depth，而 internal defect 的核心标签是缺陷体尺寸、中心位置和到扫描表面的埋深；如果把内部缺陷强行套到 `L_m/W_m/D_m/wLD/wWD/wLW`，模型会把埋深变化误解释成 surface profile 或 curvature 变化。

路线更新为：internal branch 先做 `shape_type + L/W/D + burial_depth + center_xyz` 的可行性 smoke。主输入仍应是三轴 `Bx/By/Bz`；Bz-only 只能作为低能力诊断分支。下一步只有在确认 no-defect reference、坐标系、`sensor_z_m`、试件几何、ground truth method 和埋深标签可采之后，才进入 6-12 sample internal COMSOL smoke pack。当前 surface RBC baseline 与 A2 liftoff companion 不扩展为 internal baseline。
## 2026-05-29 路线同步：21.6 internal defect burial-depth refinement

21.6 把 internal branch 的主要短板从“能否学习内部缺陷”推进到“如何让 burial_depth head 吃到更合适的物理信号”。21.5 的判断是 neural candidate 在 total、center_xyz、shape_type 上更强，但 burial_depth 弱于 feature baseline；21.6 证明这个短板不是 label/schema blocker，而是模型头和输入表征问题。

路线上的关键变化是：B2_feature_fusion_burial_head 可以作为 internal refinement candidate。它只在 Bx/By/Bz delta_b 上额外计算 peak、energy、gradient、width、cross-axis ratio、line-to-line shift 等派生特征，没有把 true shape_type、burial bin、size/aspect、split 或 sample_id 作为输入。multi-seed validation 选择 seed `2026` 后，test burial_depth MAE 为 `0.413 mm`，优于 21.4 neural `0.595 mm` 和 feature baseline `0.472 mm`，同时 total normalized MAE 改善到 `0.395256`。

下一步路线应进入 internal benchmark rerun / candidate upgrade，而不是 baseline transition。`CURRENT_BASELINE.md` 继续是 surface / near-surface true 3D RBC baseline；internal defect 仍是独立分支。后续 benchmark 需要重点复核 B2 的 center_xyz / shape 轻微代价、分组失败样本和 feature-fusion 泛化风险。

## 2026-05-29 路线同步：21.7 internal defect benchmark candidate

21.7 把 21.6 的 B2_feature_fusion_burial_head 从 refinement candidate 固定为 internal benchmark candidate。固定协议是 `delta_b/BxByBz` 加 delta_b-derived features，禁止 true `shape_type`、burial bin、size/aspect、split、sample_id 进入模型输入；selection 仍是 train-only normalization、validation-only selection、test final only。

路线上的判断是：B2 已稳定优于 21.4 neural 和 feature baseline。selected seed `2026` 的 test total normalized MAE 为 `0.395256`，burial_depth MAE 为 `0.413 mm`，优于 21.4 neural `0.595 mm` 和 feature baseline `0.472 mm`；三 seed burial_depth MAE 也都低于两个 reference。因此 burial_depth 不再是 primary shortfall。

但这仍不是 baseline transition。internal defect 数据仍是 COMSOL 仿真，shape 只覆盖 internal_sphere / internal_ellipsoid / internal_cuboid，真实 internal sample 还未验证；`CURRENT_BASELINE.md` 继续是 surface / near-surface true 3D RBC baseline。下一步路线是 internal report / visualization package，先固化报告、分组失败样本和效果图，再决定真实 internal schema alignment 或更大 shape 扩展。

## 2026-05-29 路线同步：21.8 internal defect report package

21.8 收口了 internal defect benchmark candidate 的报告层。B2_feature_fusion_burial_head 的角色从“需要复核的候选”变成“可汇报的 internal benchmark candidate”，但没有进入 `CURRENT_BASELINE.md`，也不替代 surface / near-surface true 3D RBC baseline。

路线上的关键判断是：继续盲目扩 internal COMSOL 数据不是当前优先项。21.7/21.8 已经说明 B2 在 v2_240 上稳定优于 21.4 neural 和 feature baseline，主要风险转为真实样本对齐和可解释的分组失败：`elongated_y`、`internal_ellipsoid`、`large`、`internal_cuboid`、`deep_plus` 需要在报告和后续样本设计中重点观察。

下一步路线应进入 internal real-data schema alignment：先确认真实内部缺陷样本是否能提供 no-defect reference、三轴 Bx/By/Bz、sensor_z_m、坐标系、单位、扫描线、sensor_x 对齐、gain 状态，以及 L/W/D、center_xyz、burial_depth 的 ground truth。若要做 internal inference smoke 或效果图，需要先恢复 B2 inference artifact；缺 artifact 时不得伪造 per-sample gallery。

## 2026-05-29 路线同步：21.9 internal B2 inference artifact

21.9 解决了 21.8 的展示和推理阻塞：B2_feature_fusion_burial_head 现在有可加载 checkpoint、prediction artifact 和 tracked manifest。artifact 按 21.7 固定协议恢复，seed `2026`、best epoch `277`、train-only normalization、validation-only selection、test final only，输入仍只来自 `delta_b/BxByBz` 和 delta_b-derived features。

路线上的变化是：internal branch 已具备做 per-sample inference smoke / gallery 的条件。此前效果图需要临时复现模型，现在可以通过 `results/manifests/internal_defect_b2_inference_artifact_manifest.json` 定位 ignored checkpoint 和 prediction artifact，直接生成 true vs pred 内部空腔图、best/worst 样本索引和 failure gallery。

这仍不是 baseline transition。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline，internal defect 仍是独立 benchmark branch；checkpoint 和 prediction artifact 只留在 ignored `checkpoints/`，不提交。后续真实 internal 样本接入仍需要 schema/metadata alignment。

## 2026-05-29 路线同步：22.0 internal B2 failure-driven audit

22.0 把 internal branch 从平均指标汇报推进到 failure-driven 推理审计。B2_feature_fusion_burial_head 仍复现 21.7/21.9 的平均能力，但 tail error 暴露出不能直接进入真实样本 inference smoke：test 40 个样本中有 `5` 个 full-shift catastrophic failure，`1` 个 geometry_branch_failure。

真正的机制问题是 shape branch 与 center/burial 回归耦合。最严重的 geometry branch case 是 `internal_pilot_091`：true `internal_cuboid` 被预测成 `internal_ellipsoid`，burial_depth error `1.674 mm`，center_xyz error `4.306 mm`。最差 center case `internal_topup_045` 虽然 shape 仍是 cuboid，但 center error 达到 `8.785 mm`，说明问题不只是分类错一类，而是内部几何分支和空间定位会一起漂移。

路线结论：B2 保留为 internal benchmark candidate，但不能称为 stable inference model，更不能写入 `CURRENT_BASELINE.md`。下一步优先做 shape-conditioned / two-stage internal model：先让模型稳定判定 shape branch，再在对应分支内回归 L/W/D、burial_depth 和 center_xyz；center/burial focused refinement 只作为次级 ablation。真实 internal sample inference smoke 暂缓。

## 2026-05-30 路线同步：22.1 shape-conditioned internal model

22.1 验证了一个关键判断：22.0 的 tail failure 不能只靠把 regression head 条件化到 predicted shape 上解决。T3_shape_specific_heads 的平均指标确实更好，test total normalized MAE 从 B2 的 `0.395256` 降到 `0.357371`，center p95 也从 `8.309 mm` 降到 `5.999 mm`；但 hard failure 没有被真正消除，catastrophic failure 仍是 `5/40`，geometry_branch_failure 仍是 `1/40`，且 center max 升到 `10.468 mm`。

路线上的分界点是：shape branch conditioning 可以改善均值和部分 tail，但现有 v2_240 对 hard cases 的覆盖不足以把模型训练成 stable inference candidate。继续只调 head 结构会有 test tail 偶然波动风险，也容易把 validation 上的 tail 改善误当成稳定泛化。

下一步路线转为 targeted internal hard-case top-up。补样应围绕 compact、large/medium、shallow/deep_plus、cuboid/ellipsoid 易混，以及 center/burial 同时大偏移的组合；之后再训练更强 two-stage branch。internal defect 仍是独立 benchmark branch，不进入 `CURRENT_BASELINE.md`，真实 internal inference smoke 继续暂缓。

## 2026-05-30 路线同步：22.2 targeted internal hard-case top-up plan

22.2 把 22.0/22.1 暴露的 tail failure 固化成数据补充路线，而不是继续无数据地调模型。B2 与 T3 都不是 stable inference model：catastrophic failure 仍为 `5/40`，geometry_branch_failure 仍为 `1/40`，说明当前 internal v2_240 对 hard-case geometry/center/burial 组合覆盖不足。

本阶段只做 plan：没有运行 COMSOL，没有训练，没有生成或修改 data/NPZ，也没有更新 `CURRENT_BASELINE.md`。hard-case top-up 设计目标 N=`120`，minimum usable N=`72`，围绕 cuboid/ellipsoid confusion、compact、medium/large、shallow/deep_plus 和 center-region neighbor cases 生成 matched neighbor samples。9 个 target 的配额已经逐项对齐为 `24/20/18/16/14/10/10/4/4`。

下一步路线是 22.2b targeted COMSOL hard-case top-up pack generation。22.2b 只应生成 hard-case COMSOL pack、inventory、validation 和 registry/manifest；在 top-up pack 验证前，继续暂缓 further model refinement 和真实 internal inference smoke。internal defect 仍是独立 benchmark branch，不进入 surface / near-surface `CURRENT_BASELINE.md`。
