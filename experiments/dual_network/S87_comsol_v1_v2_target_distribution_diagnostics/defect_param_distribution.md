# S87 defect parameter distribution

## v1 train

- samples: 50
- defect_type 分布:
  - `ellipsoid`: 50
- component type 分布:
  - 不可用
- boundary_irregularity_proxy 分布:
  - `不可用`: 50
- `rotation_angle` 范围: 不可用
- `defect_center_x` 范围: `1651.21` 到 `2885.41`
- `defect_center_y` 范围: `1260.68` 到 `1781.26`
- `defect_center_z` 范围: `154.737` 到 `275.479`
- `defect_axis_x` 范围: `525.582` 到 `1043.74`
- `defect_axis_y` 范围: `110.702` 到 `319.791`
- `defect_axis_z` 范围: `35.4551` 到 `89.3547`
- `defect_depth_or_shape_param` 范围: `35.4551` 到 `89.3547`
- `defect_mu`: 变化，范围 `1.41618` 到 `2.411`
- `c_magn`: 变化，范围 `0.165682` 到 `0.376068`
- `mur_magn`: 变化，范围 `2.64317` 到 `4.97225`
- `Mr_magn_A_per_m`: 变化，范围 `40.3449` 到 `88.389`

## v1 val

- samples: 10
- defect_type 分布:
  - `ellipsoid`: 10
- component type 分布:
  - 不可用
- boundary_irregularity_proxy 分布:
  - `不可用`: 10
- `rotation_angle` 范围: 不可用
- `defect_center_x` 范围: `1876.15` 到 `2674.23`
- `defect_center_y` 范围: `1233.15` 到 `1621.71`
- `defect_center_z` 范围: `177.229` 到 `265.311`
- `defect_axis_x` 范围: `546.931` 到 `1043.49`
- `defect_axis_y` 范围: `111.401` 到 `249.773`
- `defect_axis_z` 范围: `37.5204` 到 `88.497`
- `defect_depth_or_shape_param` 范围: `37.5204` 到 `88.497`
- `defect_mu`: 变化，范围 `1.38214` 到 `2.38817`
- `c_magn`: 变化，范围 `0.188725` 到 `0.372553`
- `mur_magn`: 变化，范围 `2.60567` 到 `4.95329`
- `Mr_magn_A_per_m`: 变化，范围 `47.443` 到 `86.6055`

## v1 test

- samples: 10
- defect_type 分布:
  - `ellipsoid`: 10
- component type 分布:
  - 不可用
- boundary_irregularity_proxy 分布:
  - `不可用`: 10
- `rotation_angle` 范围: 不可用
- `defect_center_x` 范围: `1619.98` 到 `2696.56`
- `defect_center_y` 范围: `1293.98` 到 `1550.87`
- `defect_center_z` 范围: `159.637` 到 `267.756`
- `defect_axis_x` 范围: `537.75` 到 `1005.31`
- `defect_axis_y` 范围: `134.017` 到 `306.217`
- `defect_axis_z` 范围: `53.3856` 到 `89.9676`
- `defect_depth_or_shape_param` 范围: `53.3856` 到 `89.9676`
- `defect_mu`: 变化，范围 `1.43576` 到 `2.33866`
- `c_magn`: 变化，范围 `0.173709` 到 `0.375937`
- `mur_magn`: 变化，范围 `2.76664` 到 `4.96701`
- `Mr_magn_A_per_m`: 变化，范围 `47.0249` 到 `81.858`

## v2 train

- samples: 100
- defect_type 分布:
  - `multi_defect_rectangular_notch_rectangular_notch_rectangular_notch`: 25
  - `multi_defect_rectangular_notch_rectangular_notch_rotated_rect`: 25
  - `multi_defect_rectangular_notch_rotated_rect_rotated_rect`: 25
  - `multi_defect_rotated_rect_rotated_rect_rotated_rect`: 25
- component type 分布:
  - `["rectangular_notch", "rectangular_notch", "rectangular_notch"]`: 25
  - `["rectangular_notch", "rectangular_notch", "rotated_rect"]`: 25
  - `["rectangular_notch", "rotated_rect", "rotated_rect"]`: 25
  - `["rotated_rect", "rotated_rect", "rotated_rect"]`: 25
- boundary_irregularity_proxy 分布:
  - `far`: 32
  - `medium`: 32
  - `near`: 36
- `rotation_angle` 范围: `0` 到 `30`
- `defect_center_x` 范围: `-0.0012` 到 `0.0012`
- `defect_center_y` 范围: `-0.0018` 到 `0.00156667`
- `defect_center_z` 范围: `-0.001125` 到 `-0.00075`
- `defect_axis_x` 范围: `0.0264` 到 `0.0446`
- `defect_axis_y` 范围: `0.0121` 到 `0.0157`
- `defect_axis_z` 范围: `0.002` 到 `0.0025`
- `defect_depth_or_shape_param` 范围: `0.002` 到 `0.0025`
- `defect_mu`: 固定 `1.0`
- `c_magn`: 固定 `0.0`
- `mur_magn`: 固定 `1.0`
- `Mr_magn_A_per_m`: 固定 `0.0`

## v2 val

- samples: 20
- defect_type 分布:
  - `multi_defect_rectangular_notch_rectangular_notch_rectangular_notch`: 5
  - `multi_defect_rectangular_notch_rectangular_notch_rotated_rect`: 5
  - `multi_defect_rectangular_notch_rotated_rect_rotated_rect`: 5
  - `multi_defect_rotated_rect_rotated_rect_rotated_rect`: 5
- component type 分布:
  - `["rectangular_notch", "rectangular_notch", "rectangular_notch"]`: 5
  - `["rectangular_notch", "rectangular_notch", "rotated_rect"]`: 5
  - `["rectangular_notch", "rotated_rect", "rotated_rect"]`: 5
  - `["rotated_rect", "rotated_rect", "rotated_rect"]`: 5
- boundary_irregularity_proxy 分布:
  - `far`: 8
  - `medium`: 8
  - `near`: 4
- `rotation_angle` 范围: `0` 到 `30`
- `defect_center_x` 范围: `-0.0012` 到 `0.0012`
- `defect_center_y` 范围: `-0.0018` 到 `0.00156667`
- `defect_center_z` 范围: `-0.001125` 到 `-0.00075`
- `defect_axis_x` 范围: `0.0264` 到 `0.0446`
- `defect_axis_y` 范围: `0.0125` 到 `0.0157`
- `defect_axis_z` 范围: `0.002` 到 `0.0025`
- `defect_depth_or_shape_param` 范围: `0.002` 到 `0.0025`
- `defect_mu`: 固定 `1.0`
- `c_magn`: 固定 `0.0`
- `mur_magn`: 固定 `1.0`
- `Mr_magn_A_per_m`: 固定 `0.0`

## v2 test

- samples: 20
- defect_type 分布:
  - `multi_defect_rectangular_notch_rectangular_notch_rectangular_notch`: 5
  - `multi_defect_rectangular_notch_rectangular_notch_rotated_rect`: 5
  - `multi_defect_rectangular_notch_rotated_rect_rotated_rect`: 5
  - `multi_defect_rotated_rect_rotated_rect_rotated_rect`: 5
- component type 分布:
  - `["rectangular_notch", "rectangular_notch", "rectangular_notch"]`: 5
  - `["rectangular_notch", "rectangular_notch", "rotated_rect"]`: 5
  - `["rectangular_notch", "rotated_rect", "rotated_rect"]`: 5
  - `["rotated_rect", "rotated_rect", "rotated_rect"]`: 5
- boundary_irregularity_proxy 分布:
  - `far`: 8
  - `medium`: 4
  - `near`: 8
- `rotation_angle` 范围: `0` 到 `30`
- `defect_center_x` 范围: `-0.0012` 到 `0.0012`
- `defect_center_y` 范围: `-0.0018` 到 `0.00156667`
- `defect_center_z` 范围: `-0.001125` 到 `-0.00075`
- `defect_axis_x` 范围: `0.0264` 到 `0.0446`
- `defect_axis_y` 范围: `0.0121` 到 `0.0157`
- `defect_axis_z` 范围: `0.002` 到 `0.0025`
- `defect_depth_or_shape_param` 范围: `0.002` 到 `0.0025`
- `defect_mu`: 固定 `1.0`
- `c_magn`: 固定 `0.0`
- `mur_magn`: 固定 `1.0`
- `Mr_magn_A_per_m`: 固定 `0.0`

