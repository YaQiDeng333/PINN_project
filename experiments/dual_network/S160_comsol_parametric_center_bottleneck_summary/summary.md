# S160 COMSOL parametric center bottleneck summary

## 结论

S156-S159 的 parameter-level oracle ablation 已把当前 parametric route 的主要瓶颈定位到 `center_x` / `center_y` localization。

- `pred_all` 复现 S115 / S126 raw MLP baseline：train / val / test mask IoU = `0.698072` / `0.369908` / `0.424462`。
- `gt_center` 是单项替换中最大、最稳定的提升：train / val / test mask IoU = `0.723396` / `0.714872` / `0.722920`。
- `gt_all` 与 S117 oracle 对齐到约 `1e-6`：train / val / test = `0.722997` / `0.723288` / `0.716584`。
- `gt_type` 和 `gt_depth` 在当前 hard rasterizer 下不直接改变 mask；`gt_rotation` 对 val/test 没有形成改善；`gt_axis` 只有小幅影响。

## 路线判断

Parametric route 继续，但下一步不再盲扫 type / rotation / forward consistency / raster loss。当前最有信息密度的方向是专门诊断和修复 component center localization。

## 下一步

- S161：量化 center error 的 grid-cell / axis-relative 尺度，并检查它与 mask IoU 的相关性。
- S162：为 parametric inverse runner 增加 center-specific grid loss 和 axis-relative loss。
- S163：用同一轮 1500-step reference 做 quick gate，判断 center-aware loss 是否同时改善 val/test。
- S164：只有 val 和 test 同时通过 gate 时才做 3000-step winner confirm。

## 自评

本阶段准确承接 S158/S159，没有把 type、rotation 或 forward residual 误判为当前 final mask IoU 的主瓶颈。下一步聚焦 center localization 是由 oracle ablation 直接支持的。
