# 缺陷预测复现文件清单

这份清单面向“另一台电脑已经配置好环境，只需要复现当前缺陷预测链”的场景。当前可复现主线不是旧 2D mask baseline，而是第 20.86 后的 true 3D RBC profile/depth baseline。

核心链路：

`delta_b(Bx/By/Bz)` -> `20.77/20.85 Conv1D + MLP 六参数模型` -> `L_m/W_m/D_m/wLD/wWD/wLW` -> `RBC-style 3D profile/depth` -> `projected mask QA`

## 迁移原则

最稳妥的迁移方式是：

1. 用 Git 带走代码、脚本、Markdown、`results/manifests/`、精简后的代表性 `results/summaries/` 和已跟踪的 `results/metrics/` 小表；新的 summary / metrics 默认仍被忽略，需要记录时按项目规则显式 `git add -f`。
2. 手动复制被 `.gitignore` 忽略的数据和模型 artifact：`data/`、`checkpoints/`、代表性 `results/previews/`、代表性 `results/loss_curves/`。
3. 如果新电脑上的项目根目录不是 `C:\Users\19166\Desktop\PINN_project`，需要先修正 registry / manifest 里的绝对路径。

当前脚本禁止 latest/newest NPZ 自动扫描；数据必须通过 `dataset_id` + `COMSOL_DATA_REGISTRY.md` + manifest 显式定位。

## 当前 baseline 必须带走

这些文件是复现当前 true 3D RBC baseline 的最小集合。

### 代码和文档

- `README.md`
- `CURRENT_BASELINE.md`
- `EXPERIMENT_LOG.md`
- `COMSOL_DATA_REGISTRY.md`
- `ARTIFACT_RETENTION_NOTES.md`
- `CLEANUP_DELETE_CANDIDATES.md`
- `REPRODUCTION_FILE_MANIFEST.md`
- `scripts/`

根目录旧入口 `train_pinn.py`、`evaluate_pinn.py`、`data_generator_v2.py` 已移除。它们主要服务早期 synthetic / v3_complex / v4 2D mask 或 μ-field 路线，不是当前 true 3D RBC baseline 的最小运行依赖。当前主线复现入口在 `scripts/` 下，尤其是 manifest loader、artifact loader 和 liftoff-conditioned inference 相关脚本。

### 当前主线数据

- `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.npz`

### 当前主线 checkpoint / prediction artifact

- `checkpoints/true_3d_rbc_baseline_artifacts/true_3d_rbc_v3_240_seed42_20_77_baseline.pt`
- `checkpoints/true_3d_rbc_baseline_artifacts/true_3d_rbc_v3_240_seed42_20_77_predictions.npz`

### 当前主线 manifest

- `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json`
- `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json`

## liftoff 伴随模块建议带走

liftoff adapter 不是 `CURRENT_BASELINE` 替代品，但它是当前保留下来的伴随鲁棒性模块；如果要跑 liftoff-conditioned inference smoke，需要一起迁移。

- `data/comsol_mfl/generated/true_3d_rbc_liftoff_aug_pack/true_3d_rbc_liftoff_aug_pack.npz`
- `checkpoints/true_3d_rbc_liftoff_adapter_artifacts/true_3d_rbc_liftoff_a2_adapter_seed2026.pt`
- `checkpoints/true_3d_rbc_liftoff_adapter_artifacts/true_3d_rbc_liftoff_a2_adapter_seed2026_predictions.npz`
- `results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json`
- `results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json`
- `results/summaries/true_3d_rbc_liftoff_inference_metadata_contract.md`
- `scripts/run_true_3d_rbc_liftoff_conditioned_inference.py`

## 代表性历史记录

这些不是当前 baseline 的运行依赖，但用于复盘早期路线和旧 2D comparator。

- `data/comsol_mfl/prepared/experimental/true_3d_rbc_pilot/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.npz`
- `data/training_data_v3_complex_train.npz`
- `data/training_data_v3_complex_val.npz`
- `data/training_data_v3_complex_test.npz`
- `results/previews/`
- `results/loss_curves/`
- `results/summaries/` 中保留下来的代表性文件

代表性 preview、loss curve 和 summary 的具体保留理由见 `ARTIFACT_RETENTION_NOTES.md`。旧 review / preflight / route-decision 文档和大量逐样本日志已删除，不作为复现迁移资产。

## 路径迁移注意

当前 registry 和 manifest 保存了绝对路径，例如：

`C:\Users\19166\Desktop\PINN_project`

如果另一台电脑也使用这个根目录，直接复制即可。如果新根目录不同，例如 `D:\PINN_project`，需要把下列文件中的旧根路径替换为新根路径：

- `COMSOL_DATA_REGISTRY.md`
- `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json`
- `results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json`
- `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json`
- `results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json`

替换后再运行验证；不要绕过 manifest，也不要改成 latest/newest 自动扫描。

## 快速检查命令

在项目根目录运行：

```powershell
cd C:\Users\19166\Desktop\PINN_project
& "C:\Users\19166\anaconda3\envs\pinn_mfl\python.exe" scripts\run_true_3d_rbc_liftoff_conditioned_inference.py --skip-preflight
```

这条命令会加载 frozen 20.85 baseline artifact 和 A2 liftoff adapter artifact，执行 liftoff-conditioned inference smoke。它不训练、不写 checkpoint、不写 NPZ。

如果只想确认当前 baseline 的定位关系，先检查这些文件是否存在：

```powershell
Test-Path .\data\comsol_mfl\prepared\experimental\true_3d_rbc_pilot\comsol_true_3d_rbc_imported_watertight_pilot_v3_240.npz
Test-Path .\checkpoints\true_3d_rbc_baseline_artifacts\true_3d_rbc_v3_240_seed42_20_77_baseline.pt
Test-Path .\results\manifests\true_3d_rbc_baseline_inference_artifact_manifest.json
```

## 不需要带走

- `.claude/`
- `.claudian/`
- `.obsidian/`
- `notes/`
- `.git/` 以外的本地工具状态
- 旧 checkpoint、旧全量 preview、旧全量 loss curve、旧大计划 CSV、旧 review / preflight / route-decision 文档、未入选的历史 `.txt` summary

这些内容不是当前项目运行依赖。`.claude/`、`.claudian/`、`.obsidian/`、`notes/` 是本地工具或个人笔记状态，可以按个人需要保留，但不应作为复现依赖。
