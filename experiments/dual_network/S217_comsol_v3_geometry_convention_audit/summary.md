# S217 COMSOL V3 geometry convention audit

S217 audits the V2/V3 coordinate convention mismatch that blocked S213 zero-shot evaluation. No training was run and no source export files were modified.

## Coordinate Ranges

| dataset | x range | y range | dx | dy | geometry_units | field_units |
|---|---:|---:|---:|---:|---|---|
| V2 train | `[-0.040000, 0.040000]` | `[-0.010000, 0.010000]` | `4.020100e-04` | `2.020202e-04` | `m` | `T` |
| V3 train raw | `[0.000000, 4500.000000]` | `[0.000000, 3000.000000]` | `22.613066` | `30.303030` | `m` | `T` |
| V3 val raw | `[0.000000, 4500.000000]` | `[0.000000, 3000.000000]` | `22.613066` | `30.303030` | `m` | `T` |
| V3 test raw | `[0.000000, 4500.000000]` | `[0.000000, 3000.000000]` | `22.613066` | `30.303030` | `m` | `T` |

## Defect Parameter Ranges

| split | center_x raw | center_y raw | axis_x raw | axis_y raw | depth raw | rotation |
|---|---:|---:|---:|---:|---:|---:|
| V3 train | `[704.070352, 3770.402010]` | `[810.000000, 2342.424242]` | `[420.000000, 880.000000]` | `[110.000000, 264.000000]` | `[70.000000, 85.000000]` | `0.0` |
| V3 val | `[781.407035, 3418.090452]` | `[810.000000, 1856.060606]` | `[450.000000, 880.000000]` | `[120.000000, 260.000000]` | `[70.000000, 85.000000]` | `0.0` |
| V3 test | `[1066.331658, 3770.402010]` | `[1200.000000, 2340.909091]` | `[420.000000, 850.000000]` | `[110.000000, 264.000000]` | `[70.000000, 85.000000]` | `0.0` |

## Mask / Defect Alignment

- V3 raw `masks == (mu_maps < 500)` is true for train/val/test.
- V3 raw mask bbox center and `defect_center_x/y` agree within raw grid discretization: max absolute bbox-center delta is train `10.050254 / 11.060608`, val `10.050271 / 10.727234`, test `7.789028 / 9.090894` for x/y.
- This confirms S208 raw is internally self-consistent.

## Conclusion

S208 V3 raw data is self-consistent but not V2-compatible. The most likely convention is COMSOL application raw model coordinates `[0,4500] x [0,3000]`, despite `geometry_units=m` in metadata. The fix belongs in a normalized V3 data copy, not in the training runner.
