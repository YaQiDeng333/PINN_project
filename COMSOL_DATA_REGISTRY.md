# COMSOL Data Registry

This registry records generated COMSOL dataset identities and allowed usage. It is not a baseline document.

## comsol_true_3d_rbc_imported_watertight_pilot_v1_partial_20_71

- dataset_role: partial_source
- status: partial_pilot_generated
- route: true_3d_piao_style
- stage: 20.72
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\prepared\experimental\true_3d_rbc_pilot\comsol_true_3d_rbc_imported_watertight_pilot_v1.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_true_3d_rbc_imported_watertight_pilot_v1.manifest.json`
- n_samples: 30
- split_counts: {'val': 5, 'test': 5, 'train': 20}
- train_ready_candidate: false
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, assembly_input
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: none
- npz_sha256: 28a1343910a953000aa7105c5d1b78bf6cf9168acae21c342325883da293e303
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_true_3d_rbc_imported_watertight_pilot_v1_topup_20_72

- dataset_role: topup_source
- status: topup_generated
- route: true_3d_piao_style
- stage: 20.72
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\prepared\experimental\true_3d_rbc_pilot\comsol_true_3d_rbc_imported_watertight_pilot_v1_topup.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_true_3d_rbc_imported_watertight_pilot_v1_topup.manifest.json`
- n_samples: 26
- split_counts: {'val': 5, 'test': 5, 'train': 16}
- train_ready_candidate: false
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, assembly_input
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_true_3d_rbc_imported_watertight_pilot_v1_partial_20_71
- npz_sha256: 153e94abedb14d2aa9697414711d6e152fd683cea5ae77a2627c18575ececd26
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled

- dataset_role: assembled
- status: pilot_generated
- route: true_3d_piao_style
- stage: 20.72
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\prepared\experimental\true_3d_rbc_pilot\comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.manifest.json`
- n_samples: 56
- split_counts: {'val': 10, 'test': 10, 'train': 36}
- train_ready_candidate: true
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, explicit_pilot_training_gate
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_true_3d_rbc_imported_watertight_pilot_v1_partial_20_71, comsol_true_3d_rbc_imported_watertight_pilot_v1_topup_20_72
- npz_sha256: d9b84ca3086dccb484abbae32b7b4f84da50c342f955775144a7e1cd7fc4b784
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_true_3d_rbc_imported_watertight_pilot_v2_topup_20_74

- dataset_role: topup_source
- status: topup_generated
- route: true_3d_piao_style
- stage: 20.74
- schema_version: true3d_profile_v1_piao_rbc
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\prepared\experimental\true_3d_rbc_pilot\comsol_true_3d_rbc_imported_watertight_pilot_v2_topup_20_74.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_true_3d_rbc_imported_watertight_pilot_v2_topup_20_74.manifest.json`
- n_samples: 56
- split_counts: {'train': 40, 'val': 8, 'test': 8}
- curvature_counts: {'sharp': 11, 'round': 12, 'boxy': 11, 'LD_dominant': 13, 'WD_dominant': 9}
- train_ready_candidate: false
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, assembly_input
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled
- generator_script: `scripts/generate_mfl_true_3d_rbc_dataset_120_topup_pack.py`
- validation_script: `scripts/validate_true_3d_rbc_dataset_120_pack.py`
- npz_sha256: 29436c4aec8f7ae6968d02ce6a125de5ee02fcd0a75314b9f2879dd4d4bbdf5d
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_true_3d_rbc_imported_watertight_pilot_v2_120

- dataset_role: assembled
- status: pilot_generated
- route: true_3d_piao_style
- stage: 20.74
- schema_version: true3d_profile_v1_piao_rbc
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\prepared\experimental\true_3d_rbc_pilot\comsol_true_3d_rbc_imported_watertight_pilot_v2_120.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_true_3d_rbc_imported_watertight_pilot_v2_120.manifest.json`
- n_samples: 112
- split_counts: {'val': 18, 'test': 18, 'train': 76}
- curvature_counts: {'sharp': 22, 'round': 23, 'boxy': 23, 'LD_dominant': 24, 'WD_dominant': 20}
- train_ready_candidate: true
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, explicit_pilot_training_gate
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled, comsol_true_3d_rbc_imported_watertight_pilot_v2_topup_20_74
- generator_script: `scripts/generate_mfl_true_3d_rbc_dataset_120_topup_pack.py`
- validation_script: `scripts/validate_true_3d_rbc_dataset_120_pack.py`
- npz_sha256: a79b56955ac4df4ed3f36388f4019f08d7ef92671ef330c6c3a6cb85fe5a4a49
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76

- dataset_role: topup_source
- status: topup_generated
- route: true_3d_piao_style
- stage: 20.76
- schema_version: true3d_profile_v1_piao_rbc
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\prepared\experimental\true_3d_rbc_pilot\comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76.manifest.json`
- n_samples: 128
- split_counts: {'train': 86, 'val': 21, 'test': 21}
- curvature_counts: {'WD_dominant': 30, 'sharp': 26, 'round': 26, 'boxy': 24, 'LD_dominant': 22}
- train_ready_candidate: false
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, assembly_input
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_true_3d_rbc_imported_watertight_pilot_v2_120
- generator_script: `scripts/generate_mfl_true_3d_rbc_dataset_240_topup_pack.py`
- validation_script: `scripts/validate_true_3d_rbc_dataset_240_pack.py`
- npz_sha256: 2f854e559b4996d71c2bf5b6ce274c4cf497686c47d2ae761139188d02ddeb91
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_true_3d_rbc_imported_watertight_pilot_v3_240

- dataset_role: assembled
- status: pilot_generated
- route: true_3d_piao_style
- stage: 20.76
- schema_version: true3d_profile_v1_piao_rbc
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\prepared\experimental\true_3d_rbc_pilot\comsol_true_3d_rbc_imported_watertight_pilot_v3_240.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json`
- n_samples: 240
- split_counts: {'val': 39, 'test': 39, 'train': 162}
- curvature_counts: {'sharp': 48, 'round': 49, 'boxy': 47, 'LD_dominant': 46, 'WD_dominant': 50}
- train_ready_candidate: true
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, explicit_pilot_training_gate
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_true_3d_rbc_imported_watertight_pilot_v2_120, comsol_true_3d_rbc_imported_watertight_pilot_v3_topup_20_76
- generator_script: `scripts/generate_mfl_true_3d_rbc_dataset_240_topup_pack.py`
- validation_script: `scripts/validate_true_3d_rbc_dataset_240_pack.py`
- npz_sha256: 6cfd6c4dd32c474dbe4843ac520f541123a54a8f0082a94d3e28d0c8cd196817
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_true_3d_rbc_liftoff_aug_pack_v1

- dataset_role: liftoff_augmentation_diagnostic_pack
- status: diagnostic_pack_generated
- route: true_3d_piao_style_liftoff_robustness
- stage: 20.91b
- schema_version: true3d_profile_v1_piao_rbc_liftoff_aug
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\generated\true_3d_rbc_liftoff_aug_pack\true_3d_rbc_liftoff_aug_pack.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json`
- n_samples: 192
- base_count: 48
- paired_liftoff_complete: true
- liftoff_levels_m: [0.006, 0.008, 0.01, 0.012]
- split_counts: {'test': 32, 'train': 128, 'val': 32}
- curvature_counts: {'LD_dominant': 36, 'round': 40, 'sharp': 40, 'WD_dominant': 36, 'boxy': 40}
- train_ready_candidate: true
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, explicit_liftoff_training_gate
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_true_3d_rbc_imported_watertight_pilot_v3_240
- generator_script: `scripts/generate_mfl_true_3d_rbc_liftoff_aug_pack.py`
- validation_script: `scripts/validate_true_3d_rbc_liftoff_aug_pack.py`
- npz_sha256: 71022720e500d9a2be0ca09837bdbe705958cba879fc58d5f8ffff7404372a3c
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_internal_defect_smoke_pack_v1

- dataset_role: internal_defect_feasibility_smoke_pack
- status: smoke_generated
- route: internal_buried_defect_feasibility
- stage: 21.0
- schema_version: internal_defect_feasibility_v1
- geometry_method: internal_cavity_comsol_solid
- exact_piao_rbc: false
- rbc_style_approximation: false
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\generated\internal_defect_smoke_pack\internal_defect_smoke_pack_v1.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_internal_defect_smoke_pack_v1.manifest.json`
- n_samples: 12
- shape_counts: {'internal_sphere': 4, 'internal_ellipsoid': 4, 'internal_cuboid': 4}
- burial_depth_counts: {'shallow': 6, 'medium': 3, 'deep': 3}
- train_ready_candidate: false
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, explicit_internal_training_gate
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: none
- generator_script: `C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\scripts\generate_mfl_internal_defect_smoke_pack.py`
- validation_script: `scripts/validate_internal_defect_smoke_pack.py`
- npz_sha256: 29bdc339b8fc659d9cc888889a7d3f175630e28cc2e77cfa565c4599b0badb79
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_internal_defect_pilot_pack_v1

- dataset_role: internal_defect_feasibility_pilot_pack
- status: pilot_generated
- route: internal_buried_defect_feasibility
- stage: 21.1
- schema_version: internal_defect_feasibility_v1
- geometry_method: internal_cavity_comsol_solid
- exact_piao_rbc: false
- rbc_style_approximation: false
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\generated\internal_defect_pilot_pack\internal_defect_pilot_pack_v1.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_internal_defect_pilot_pack_v1.manifest.json`
- n_samples: 96
- planned_samples: 96
- split_counts: {'train': 64, 'val': 16, 'test': 16}
- shape_counts: {'internal_sphere': 24, 'internal_ellipsoid': 36, 'internal_cuboid': 36}
- burial_depth_counts: {'shallow': 24, 'medium': 24, 'deep': 24, 'deep_plus': 24}
- train_ready_candidate: true
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, explicit_internal_training_gate
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_internal_defect_smoke_pack_v1
- generator_script: `C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP\scripts\generate_mfl_internal_defect_pilot_pack.py`
- validation_script: `scripts/validate_internal_defect_pilot_pack.py`
- npz_sha256: ac48ae5bfe1f33d848bb3c07b96d8be822f8d79a4b50cbace915575025e65f0d
- notes: Metadata only. Generated NPZ/data files are not committed and must be loaded only by explicit dataset_id + manifest.

## comsol_internal_defect_pilot_pack_v2_240

- dataset_role: internal_defect_feasibility_pilot_pack_v2_240
- status: pilot_generated
- route: internal_buried_defect_feasibility
- stage: 21.3b
- schema_version: internal_defect_feasibility_v2
- geometry_method: internal_cavity_comsol_solid
- internal_surface_mixed: false
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\generated\internal_defect_pilot_pack_v2_240\comsol_internal_defect_pilot_pack_v2_240.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_internal_defect_pilot_pack_v2_240.manifest.json`
- n_samples: 240
- split_counts: {'train': 160, 'test': 40, 'val': 40}
- shape_counts: {'internal_sphere': 80, 'internal_ellipsoid': 80, 'internal_cuboid': 80}
- burial_depth_counts: {'shallow': 60, 'medium': 60, 'deep': 60, 'deep_plus': 60}
- size_counts: {'small': 80, 'medium': 80, 'large': 80}
- aspect_counts: {'compact': 134, 'elongated_x': 54, 'elongated_y': 52}
- train_ready_candidate: true
- baseline_ready: false
- auto_discovery_allowed: false
- latest_newest_discovery_allowed: false
- allowed_use: schema_validation, explicit_internal_training_gate
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement, latest_newest_auto_discovery, direct_training_without_manifest_gate
- source_dataset_ids: comsol_internal_defect_pilot_pack_v1, comsol_internal_defect_dataset_topup_pack_v1
- validation_script: `scripts/validate_internal_defect_dataset_v2_pack.py`
- npz_sha256: 73118397dc810a4e11eb67acb0b31f8155e0c6150d472d9e7a567a5b77329404
- notes: Generated NPZ/data files are not committed. Use only explicit dataset_id + manifest; not a baseline.
