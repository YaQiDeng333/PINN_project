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
