# S73 COMSOL geometry-variation data request

## 目的

S73 基于 S71/S72 的真实 COMSOL pilot 接入结果，准备下一批更有训练价值的 COMSOL 数据请求。第一批 pilot 固定仿体，只改动磁性参数，因此主要验证接口链路；下一批必须变化缺陷几何，才能让 conditional model 学习 Bz -> shape 的映射。

## 为什么第一批 pilot 不足以判断泛化

- samples 只有 5。
- 仿体几何固定。
- 主要变化来自磁性参数，而不是 defect center、size、depth 或 shape。
- S72 只证明 converted NPZ 可以进入 conditional runner，不代表 held-out shape generalization。

## 下一批建议规模

优先建议：
- train samples = 50 到 100；
- val samples = 10 到 20；
- test samples = 10 到 20；
- grid_x = 200；
- grid_y = 100；
- probe x points = 200；
- channels = 3；
- lift_off_values = [0.5, 1.0, 2.0]；
- field_components = ["Bz", "Bz", "Bz"]。

如果 COMSOL 运行成本过高，可以先做：
- train = 20；
- val = 5；
- test = 5。

## 必须变化的缺陷参数

建议至少变化：
- defect center x；
- defect center y；
- defect radius / width；
- defect depth 或等效形状参数；
- defect permeability / mu；
- 可选：椭圆长短轴、旋转角、边界不规则度。

## 后续接入命令

分别转换 train / val / test：

```powershell
python convert_comsol_multiheight_csv_to_npz.py --signals-csv train/signals_multiheight.csv --target-npz train/targets.npz --output-npz train_comsol_multiheight.npz
python convert_comsol_multiheight_csv_to_npz.py --signals-csv val/signals_multiheight.csv --target-npz val/targets.npz --output-npz val_comsol_multiheight.npz
python convert_comsol_multiheight_csv_to_npz.py --signals-csv test/signals_multiheight.csv --target-npz test/targets.npz --output-npz test_comsol_multiheight.npz
```

再分别运行 validator：

```powershell
python comsol_multiheight_npz_utils.py --npz-path train_comsol_multiheight.npz
python comsol_multiheight_npz_utils.py --npz-path val_comsol_multiheight.npz
python comsol_multiheight_npz_utils.py --npz-path test_comsol_multiheight.npz
```

## 当前边界

- S73 只创建数据请求文档。
- S73 不调用 COMSOL，不生成数据，不训练模型。
- 后续需要在 COMSOL 侧生成真实 geometry-variation 数据，再回到本支线转换、验证和训练。
