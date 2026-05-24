# COMSOL Data Registry

This registry records generated COMSOL dataset identities and allowed usage. It is not a baseline document.

## comsol_true_3d_rbc_imported_watertight_pilot_v1

- status: partial_pilot_generated
- route: true_3d_piao_style
- stage: 20.71
- schema_version: true3d_profile_v1_piao_rbc
- geometry_method: imported_watertight_mesh_solid
- exact_piao_rbc: false
- rbc_style_approximation: true
- path: `C:\Users\19166\Desktop\PINN_project\data\comsol_mfl\prepared\experimental\true_3d_rbc_pilot\comsol_true_3d_rbc_imported_watertight_pilot_v1.npz`
- manifest_path: `C:\Users\19166\Desktop\PINN_project\results\manifests\comsol_true_3d_rbc_imported_watertight_pilot_v1.manifest.json`
- n_samples: 30
- split_counts: {'val': 5, 'test': 5, 'train': 20}
- allowed_use: schema_validation, explicit_pilot_training_gate
- forbidden_use: automatic_mainline_training, baseline_update, current_baseline_replacement
- generator_script: `COMSOL_Multiphysics_MCP/scripts/generate_mfl_true_3d_rbc_pilot_pack.py`
- validation_script: `PINN_project/scripts/validate_true_3d_rbc_pilot_pack.py`
- pinn_commit: de9e1080751d4efd01e75d4eb6f7fdacb60d877f
- comsol_commit: 928c82e875c5cee766cbd5e834ed8e30e7e0e514
- inventory_status_counts: {'pass': 30, 'fail': 2, 'not_attempted': 28}
- missing_curvature_templates: LD_dominant, WD_dominant
- comsol_worktree_status_entries: 5
- npz_sha256: 28a1343910a953000aa7105c5d1b78bf6cf9168acae21c342325883da293e303
- notes: Pilot pack metadata only. This dataset is not a baseline and must not be loaded by latest/newest auto-discovery. See manifest for worktree status and partial-pack blockers.
