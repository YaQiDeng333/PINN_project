# Dual-network / COMSOL V3 branch

本分支对应 `feature/dual-network-variational`，用于探索主线之外的 dual-network、weak-form 和 COMSOL multi-height Bz 反演方案。当前结论仍是支线研究结论，不能写成 `main` baseline 替代结果，也不能把半监督或 diagnostic upper bound 结果表述为纯无监督成功。

## 当前定位

- V2-style 当前冻结候选仍是 S185/S181 `center_bin_offset_plus_grid`。
- COMSOL V3 polygon / component-query 路线还在 branch-local 诊断阶段，尚未晋升为候选。
- V3 repaired Bz 信号路线已经验证可用，后续问题主要转向几何表示、中心定位和硬栅格边界精度。
- raw COMSOL exports、训练输出、图片、权重和临时文件不进入 Git。

## 当前技术状态

S341-S345 的 component-query 1-sample boundary precision repair 仍卡在硬门槛前：`center_aux_half` 达到 polygon IoU `0.989528796`，pred / target area 为 `191 / 189`，FP / FN 为 `2 / 0`，presence/type/x-bin/y-bin accuracy 均为 `1.0`，但仍低于显式 `>=0.99` gate。

因此 5-sample 和 train30 仍不应继续跑。下一步只有两个合理方向：要么做真正 boundary-aware 的 1-sample repair，要么明确审视 `>=0.99` 对 189-pixel mask 是否过严。

## 数据与提交边界

本分支只应提交代码、Markdown、summary、manifest、轻量 CSV/JSON/YAML/TOML 配置或索引文件。以下内容默认不提交：

- `comsol_*_exports/` raw export 根目录；
- `experiments/dual_network/*/raw/` 和 `raw_normalized/`；
- `signals_multiheight.csv`、大型 CSV/NPZ/NPY；
- `*.png`、checkpoint、`.pt`、`.pth`、`.ckpt`；
- `*.stdout.txt`、`*.stderr.txt`、`*.pid`；
- `copy-output*.md`、`*.class`、`*.pyc`、`__pycache__/`；
- partial / blocked 实验目录，例如 S35。

raw COMSOL 数据应放在外部 archive 或 route-scoped local export root；Git 中只保留足以复现判断的轻量文档和索引。

## 主要入口

- [DUAL_NETWORK_STAGE_SUMMARY.md](DUAL_NETWORK_STAGE_SUMMARY.md)：当前阶段性结论和下一步判断。
- [DUAL_NETWORK_EXPERIMENT_LOG.md](DUAL_NETWORK_EXPERIMENT_LOG.md)：完整实验日志。
- [DUAL_NETWORK_ARTIFACT_INDEX.md](DUAL_NETWORK_ARTIFACT_INDEX.md)：代码、文档和实验产物索引。
- [DUAL_NETWORK_RESULTS_REPORT.md](DUAL_NETWORK_RESULTS_REPORT.md)：跨阶段结果报告。
- [DUAL_NETWORK_TERMS.md](DUAL_NETWORK_TERMS.md)：术语、指标和实验边界说明。
- [COMSOL_V3_HARD_CASE_DATA_REQUEST.md](COMSOL_V3_HARD_CASE_DATA_REQUEST.md)：V3 hard-case 数据请求边界。

## 运行边界

不要默认运行训练、COMSOL solve/export、批量清理或 push。涉及 baseline 替换、数据包扩展、模型结构变更、训练目标变更或主线合并前，需要先明确当前分支边界和验收门槛。
