# S84 defect parameter summary

S84 ?? COMSOL geometry V2 fallback ???train=100, val=20, test=20?????? `rectangular_notch` / `rotated_rect` multi_defect ?????? `ellipsoid`?`boundary_irregularity` ?? component distance bin ??? proxy?magnetic parameters ?????

## train

- samples: 100
- defect_type ??:
  - `multi_defect_rectangular_notch_rectangular_notch_rectangular_notch`: 25
  - `multi_defect_rectangular_notch_rectangular_notch_rotated_rect`: 25
  - `multi_defect_rectangular_notch_rotated_rect_rotated_rect`: 25
  - `multi_defect_rotated_rect_rotated_rect_rotated_rect`: 25
- component_types ??:
  - `["rectangular_notch", "rectangular_notch", "rectangular_notch"]`: 25
  - `["rectangular_notch", "rectangular_notch", "rotated_rect"]`: 25
  - `["rectangular_notch", "rotated_rect", "rotated_rect"]`: 25
  - `["rotated_rect", "rotated_rect", "rotated_rect"]`: 25
- boundary_irregularity_proxy ??:
  - `far`: 32
  - `medium`: 32
  - `near`: 36
- `defect_center_x` ??: -0.0012 ? 0.0012
- `defect_center_y` ??: -0.0018 ? 0.00156667
- `defect_center_z` ??: -0.001125 ? -0.00075
- `defect_axis_x` ??: 0.0264 ? 0.0446
- `defect_axis_y` ??: 0.0121 ? 0.0157
- `defect_axis_z` ??: 0.002 ? 0.0025
- `defect_depth_or_shape_param` ??: 0.002 ? 0.0025
- `rotation_angle` ??: 0 ? 30
- `boundary_irregularity` ??: 0 ? 1
- `defect_mu` ??: [1.0]
- `c_magn` ??: [0.0]
- `mur_magn` ??: [1.0]
- `Mr_magn_A_per_m` ??: [0.0]
- magnetic parameters ?????? V2 fallback source pack ????????????

## val

- samples: 20
- defect_type ??:
  - `multi_defect_rectangular_notch_rectangular_notch_rectangular_notch`: 5
  - `multi_defect_rectangular_notch_rectangular_notch_rotated_rect`: 5
  - `multi_defect_rectangular_notch_rotated_rect_rotated_rect`: 5
  - `multi_defect_rotated_rect_rotated_rect_rotated_rect`: 5
- component_types ??:
  - `["rectangular_notch", "rectangular_notch", "rectangular_notch"]`: 5
  - `["rectangular_notch", "rectangular_notch", "rotated_rect"]`: 5
  - `["rectangular_notch", "rotated_rect", "rotated_rect"]`: 5
  - `["rotated_rect", "rotated_rect", "rotated_rect"]`: 5
- boundary_irregularity_proxy ??:
  - `far`: 8
  - `medium`: 8
  - `near`: 4
- `defect_center_x` ??: -0.0012 ? 0.0012
- `defect_center_y` ??: -0.0018 ? 0.00156667
- `defect_center_z` ??: -0.001125 ? -0.00075
- `defect_axis_x` ??: 0.0264 ? 0.0446
- `defect_axis_y` ??: 0.0125 ? 0.0157
- `defect_axis_z` ??: 0.002 ? 0.0025
- `defect_depth_or_shape_param` ??: 0.002 ? 0.0025
- `rotation_angle` ??: 0 ? 30
- `boundary_irregularity` ??: 0 ? 1
- `defect_mu` ??: [1.0]
- `c_magn` ??: [0.0]
- `mur_magn` ??: [1.0]
- `Mr_magn_A_per_m` ??: [0.0]
- magnetic parameters ?????? V2 fallback source pack ????????????

## test

- samples: 20
- defect_type ??:
  - `multi_defect_rectangular_notch_rectangular_notch_rectangular_notch`: 5
  - `multi_defect_rectangular_notch_rectangular_notch_rotated_rect`: 5
  - `multi_defect_rectangular_notch_rotated_rect_rotated_rect`: 5
  - `multi_defect_rotated_rect_rotated_rect_rotated_rect`: 5
- component_types ??:
  - `["rectangular_notch", "rectangular_notch", "rectangular_notch"]`: 5
  - `["rectangular_notch", "rectangular_notch", "rotated_rect"]`: 5
  - `["rectangular_notch", "rotated_rect", "rotated_rect"]`: 5
  - `["rotated_rect", "rotated_rect", "rotated_rect"]`: 5
- boundary_irregularity_proxy ??:
  - `far`: 8
  - `medium`: 4
  - `near`: 8
- `defect_center_x` ??: -0.0012 ? 0.0012
- `defect_center_y` ??: -0.0018 ? 0.00156667
- `defect_center_z` ??: -0.001125 ? -0.00075
- `defect_axis_x` ??: 0.0264 ? 0.0446
- `defect_axis_y` ??: 0.0121 ? 0.0157
- `defect_axis_z` ??: 0.002 ? 0.0025
- `defect_depth_or_shape_param` ??: 0.002 ? 0.0025
- `rotation_angle` ??: 0 ? 30
- `boundary_irregularity` ??: 0 ? 1
- `defect_mu` ??: [1.0]
- `c_magn` ??: [0.0]
- `mur_magn` ??: [1.0]
- `Mr_magn_A_per_m` ??: [0.0]
- magnetic parameters ?????? V2 fallback source pack ????????????

