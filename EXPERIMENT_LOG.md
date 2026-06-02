# 实验工作日志

## 2026-06-03 Stage 25.14 surface multi-pit label-v3 derivation + validator

- Scope: derived and validated label schema v3 targets in memory from `comsol_surface_multipit_component_set_pilot_v1`. This stage read 25.12b/25.13/25.13b evidence and the explicit dataset manifest/NPZ; it did not train, tune losses, run COMSOL, mutate data/NPZ files, expand model capacity, export checkpoints/previews/notes, or update `CURRENT_BASELINE.md`.
- Main finding: label v3 can be derived inside `PINN_project` from existing raw component masks/depths. Hard ownership remains unique, union/depth invariants remain intact, and soft/local valid-region support substantially increases the component-mask learning signal that collapsed under target v2.
- V3 support audit: active components `236`; v2 hard foreground pixels mean/min `99.851695/47`; v3 soft positive pixels mean/min/p05 `210.110169/126/147.750000`; v3/v2 positive support ratio mean/min `2.203184/1.697183`; existing v3 tiny/empty count `0`; existing depth-valid empty count `0`; inactive slot violations `0`.
- Group slices: component_count=3 v3/v2 support ratio mean/min `2.445849/1.972789`; partially_overlapping `2.210708/1.795775`; touching_boundary `2.279145/1.842105`.
- Invariants: duplicate hard ownership remains resolved (`297 -> 0`), overlap-depth-conflict remains resolved under hard ownership (`271 -> 0`), raw OR and v2 OR both reproduce union masks with mismatch `0`, and raw max(component depth) reproduces union depth with max RMSE `0.000000000000 m`.
- Label v3 schema: preserve `raw_component_mask_raw` and `component_ownership_map`; add `component_mask_target_v3_soft`, `component_sdf_target_v3`, `component_valid_region_mask`, `component_depth_target_v3` with explicit depth-valid region, plus `overlap_region_mask` and `contact_boundary_mask`.
- Acceptance decision: `READY_FOR_25.15_TRAINING`.
- Route decision: unique next step is `A. enter 25.15 label-v3 training gate using 25.10 loss mainline + label-v3 supervision; do not use the 25.11/25.12 rebalance stack`. This is not a baseline transition.

## 2026-06-03 Stage 25.13b surface multi-pit generator/label schema audit after target-v2 collapse

- Scope: audited generator/label schema after the 25.13 target-v2 `FAIL`. This stage read the 25.12b redesign metrics, 25.13 gate metrics/manifest, and the explicit `comsol_surface_multipit_component_set_pilot_v1` manifest/NPZ. It did not train, tune losses, run COMSOL, mutate data/NPZ files, expand model capacity, export checkpoints/previews/notes, or update `CURRENT_BASELINE.md`.
- Main finding: near-empty mask collapse is not caused by empty v2 components or broad generator corruption. Target v2 clears duplicate ownership (`297 -> 0`) and overlap-depth-conflict (`271 -> 0`) while keeping existing component masks non-empty.
- Positive-support audit: active components `236`; v2 foreground pixels mean/min/p05 are `99.851695/47/58.750000`; v2 positive fraction mean is only `0.012188928` of the `64x128` grid; v2/v1 shrink ratio mean/min is `0.985320/0.746032`; empty existing v2 masks `0`, tiny existing v2 masks `<20 px` `0`, and shrink ratio `<0.80` occurs in only `3` components.
- Collapse evidence from 25.13: recall `0.674419`, missed `0.325581`, extra `0.292683`, merged `0.000000`, component Dice `0.005536`, union Dice `0.002829`, depth RMSE `0.000242891 m`. The zero merged rate is a near-empty mask artifact, not valid component separation.
- Diagnosis: hard binary ownership-resolved labels remain too sparse and unsupported for stable component-local mask learning. The current schema lacks component-local soft mask/SDF, valid-region masks, overlap-region masks, contact-boundary masks, and ownership/boundary confidence.
- Acceptance decision: `NEEDS_PINN_LABEL_DERIVATION_V3`; existing raw masks/depths and geometry labels are sufficient to derive v3 labels inside `PINN_project`, so no COMSOL generator/export fix is required yet.
- Route decision: unique next step is `A. enter 25.14 label-v3 derivation + validator, no training`. Do not continue loss tuning and do not discuss baseline transition.

## 2026-06-03 Stage 25.13 surface multi-pit component-set target-v2 training gate

- Scope: executed the target-v2 training gate on `comsol_surface_multipit_component_set_pilot_v1`. The architecture, fixed `K=3` component-set representation, fixed split `72/20/20`, Hungarian matching, and `component_set_gate_v1` 25.10 loss mainline were kept unchanged. The 25.11/25.12 rebalance stacks were not used.
- Boundary: no COMSOL, no data/NPZ mutation, no checkpoint/preview/notes artifact commit, no model-capacity expansion, no formal inference artifact export, no baseline transition, and no `CURRENT_BASELINE.md` update.
- Target v2 usage: `component_mask_target_v2=true`, `component_depth_target_v2=true`, and `component_ownership_map=true`; loaded `112` samples, resolved `23565` foreground ownership pixels and `290` raw overlap pixels; duplicate ownership moved `297 -> 0`, and overlap-depth-conflict moved `271 -> 0`.
- Gate decision: `FAIL`. Target v2 removed ownership conflicts, but the 25.10 loss mainline learned near-empty component/union masks rather than useful component-level separation.
- Test metrics: recall `0.674419`, missed `0.325581`, extra `0.292683`, merged `0.000000`, component Dice `0.005536`, union Dice `0.002829`, depth RMSE `0.000242891 m`.
- Comparison: versus 25.10, recall worsened by `-0.162791`, missed worsened by `+0.162791`, extra worsened by `+0.149826`, component Dice collapsed by `-0.104026`, union Dice collapsed by `-0.127651`, while depth RMSE was effectively unchanged (`-0.000000424 m`). Versus 25.11/25.12, merged rate improves numerically, but this is not success because mask/union Dice collapsed.
- Subsets: component_count=3 test rows have merged `0.000000` but component Dice `0.000000` and union Dice `0.000000`; partially_overlapping rows have recall `0.555556`, missed `0.444444`, component Dice `0.004950`, and union Dice `0.002463`.
- Route decision: unique next step is `C. return to generator/label schema; do not continue loss tuning`. Treat the v2 ownership transform as diagnostically useful but insufficient as a training target under the current component-level supervision.

## 2026-06-03 Stage 25.12b surface multi-pit component raster/depth target redesign

- Scope: audited and redesigned component raster/depth targets after the 25.12 `FAIL`. This stage read the 25.10/25.10b/25.11/25.11b/25.12 metrics and the explicit `comsol_surface_multipit_component_set_pilot_v1` manifest only; it did not train, run COMSOL, mutate data/NPZ files, expand model capacity, export checkpoints/previews/notes, or update `CURRENT_BASELINE.md`.
- Main finding: v1 generator labels are globally consistent, but component-level raster/depth supervision lacks explicit ownership in overlapping/touching pixels. Component OR reconstructs the union mask exactly, max(component depth) reconstructs union depth exactly, empty slots are zero, and center-to-mask-centroid error is small; therefore this is not a generator corruption finding.
- V1 target audit: component OR to union Dice mean/min is `1.000000/1.000000`; max(component depth) to union depth RMSE mean/max is `0.000000000/0.000000000 m`; empty-slot mask/depth violations are `0`; center-to-mask-centroid error mean/p95/max is `0.000059604/0.000108224/0.001045136 m`.
- Ownership issue: `25/112` samples have duplicated component foreground ownership, with `297` duplicated component target pixels and `271` overlap-depth-conflict pixels. The problem is concentrated in `partially_overlapping` (`18/24`) and component_count=3 (`10/12`), while union OR/max hides the conflict from sample-level targets.
- Target v2 design: introduce `component_mask_target_v2`, `component_depth_target_v2`, and `component_ownership_map`; separated/close rows must be mutually exclusive, touching rows may share continuous boundaries but not raster ownership, partially-overlapping rows keep raw overlap diagnostics but train on ownership-resolved component targets, and union mask/depth remain OR/max from components.
- Acceptance decision: `READY_FOR_25.13_TRAINING`.
- Route decision: unique next step is `A. enter 25.13 target-v2 training gate using the 25.10 loss mainline; do not use the 25.11/25.12 rebalance stack`. This is not a baseline transition.

## 2026-06-02 Stage 25.12 surface multi-pit component-separation-aware rebalance training

- Scope: executed the component-separation-aware rebalance gate on `comsol_surface_multipit_component_set_pilot_v1`. The architecture, fixed `K=3` component-set representation, fixed split `72/20/20`, and Hungarian matching were unchanged.
- Boundary: no COMSOL, no data/NPZ mutation, no checkpoint/preview/notes artifact commit, no model-capacity expansion, no formal inference artifact export, no baseline transition, and no `CURRENT_BASELINE.md` update.
- Loss config: added `component_separation_rebalance_v1`, with delayed/low union mask loss, higher component-mask emphasis, component-normalized foreground depth, logged background depth diagnostics, pairwise separation consistency, and anti-merge penalty.
- Gate decision: `FAIL`. Merged collapse was only partially reduced versus 25.11 (`0.900000 -> 0.700000`) but remained far worse than 25.10 (`0.200000`), while recall/missed/extra collapsed relative to 25.11.
- Test metrics: recall `0.744186`, missed `0.255814`, extra `0.200000`, merged `0.700000`, component Dice `0.108790`, union Dice `0.138075`, depth RMSE `0.000501023 m`.
- Comparison: versus 25.10, recall worsened by `-0.093023`, component Dice stayed slightly below (`-0.000772`), and depth RMSE remained worse by `+0.000257708 m`. Versus 25.11, merged improved by `-0.200000` and depth improved by `-0.000172604 m`, but recall worsened by `-0.116279`, missed worsened by `+0.116279`, extra worsened by `+0.102439`, and union Dice fell by `-0.028157`.
- Three-component slice: still failed, with merged rate `1.0`, recall `0.555556`, missed `0.444444`, and predicted component count mean `2.0`.
- Route decision: unique next step is `C. rollback to 25.10 loss mainline and redesign component raster/depth targets before further training`. Do not continue with more capacity or stacked training.

## 2026-06-02 Stage 25.11b surface multi-pit component-set merge-collapse audit

- Scope: audited the 25.11 `PARTIAL` mask/depth rebalance result only. No new training, no COMSOL, no model-capacity expansion, no data/NPZ mutation, no checkpoint/preview/notes artifact commit, no baseline transition, and no `CURRENT_BASELINE.md` update.
- Inputs: compared `results/metrics/25_10_component_set_training_gate_metrics.json`, `results/metrics/25_10b_component_set_failure_audit.json`, `results/metrics/25_11_mask_depth_loss_rebalance_training_metrics.json`, and `results/manifests/25_11_mask_depth_loss_rebalance_training_manifest.json`.
- Main finding: 25.11 induced union-over-component merge collapse. Union Dice improved (`0.130480 -> 0.166233`) and recall/extra/missed improved, but merged rate worsened (`0.200000 -> 0.900000`), component Dice stayed flat/slightly worse (`0.109562 -> 0.108737`), and depth RMSE worsened (`0.000243315 -> 0.000673627 m`).
- Loss-scale audit: final train/val mask-depth weighted ratios were `0.973279` and `0.929449`; final validation union terms were `0.339797` of mask/depth weighted mass. Component mask weighted loss was also much larger than geometry terms, showing mask/depth supervision dominated the objective after rebalance.
- Collapse split: test newly merged rate was `0.700000`; union-over-component collapse rate was `0.500000`. The collapse is not only topology-driven: separated rows had newly merged rate `0.750000` and disconnected topology had newly merged rate `0.750000`.
- Secondary issues: depth-supervision dilution rate was `0.600000`; touching_boundary and partially_overlapping rows still had merged rate `1.0`; three-component rows remain a required audit slice.
- Threshold audit: selected threshold shifted `0.25 -> 0.35`, but 25.11 validation merged rate stayed `0.9` across candidate thresholds, so this is not just a threshold-selection bug.
- Targeted design: delay/cap union mask loss, add explicit component-separation regularization, add topology-aware merge penalties, redesign/stage depth loss with component-normalized foreground masks, and keep three-component rows separately reported.
- Route decision: unique next step is `A. enter 25.12 component-separation-aware rebalance training`. Do not simply increase model capacity and do not discuss baseline replacement.

## 2026-06-02 Stage 25.11 surface multi-pit component-set mask/depth loss rebalance training

- Scope: executed a bounded mask/depth loss rebalance training gate on `comsol_surface_multipit_component_set_pilot_v1`. The model architecture, fixed `K=3` component-set representation, fixed split `72/20/20`, and min-over-slot-permutations Hungarian matching stayed unchanged.
- Boundary: no COMSOL, no data/NPZ mutation, no checkpoint/preview/notes artifact commit, no model-capacity expansion, no formal inference artifact export, no baseline transition, and no `CURRENT_BASELINE.md` update.
- Loss config: added `mask_depth_rebalance_v1`, keeping existence/center/LWD/rotation/shape losses while adding foreground/background-balanced component mask BCE/Dice, union mask BCE/Dice, and valid-target-mask-only component/union depth losses. JSON outputs use strict serialization with non-finite values mapped to null.
- Training: fixed seed `42`, `180` epochs, selected existence threshold `0.35`, best validation loss at epoch `3`. Train loss decreased from `6.458076` to `0.454230`; final train/val mask-depth weighted ratios were `0.973279` and `0.929449`.
- Gate decision: `PARTIAL`. Versus 25.10, test component recall improved `0.837209 -> 0.860465`, missed rate improved `0.162791 -> 0.139535`, extra rate improved `0.142857 -> 0.097561`, and union mask Dice improved `0.130480 -> 0.166233`.
- Remaining blockers: component mask Dice did not improve (`0.109562 -> 0.108737`), depth-grid RMSE worsened (`0.000243315 -> 0.000673627 m`), merged rate worsened (`0.200000 -> 0.900000`), and the three-component test subset remains merged (`merged_rate=1.0`).
- Route decision: unique next step is `B. run 25.11b targeted rebalance or topology-focused failure audit`. Do not continue by simply increasing model capacity and do not discuss baseline replacement.

## 2026-06-02 Stage 25.10b surface multi-pit component-set failure audit

- Scope: audited the 25.10 `PARTIAL` component-set training gate only. No new training, no COMSOL, no data/NPZ mutation, no checkpoint/preview/notes artifact commit, no baseline transition, and no `CURRENT_BASELINE.md` update.
- Inputs: read `results/metrics/25_10_component_set_training_gate_metrics.json`, `results/manifests/25_10_component_set_training_gate_manifest.json`, `results/summaries/25_10_component_set_training_gate_summary.md`, and the explicit dataset manifest for `comsol_surface_multipit_component_set_pilot_v1`.
- Main finding: the primary failure is mask/depth loss imbalance, not slot matching or target-coordinate corruption. Component existence and coarse geometry learned (`test component_recall=0.837209`, `center_error_mean=0.004284 m`, `L/W/D relative error=0.154552`), while component mask Dice stayed `0.109562`, union mask Dice `0.130480`, and depth RMSE barely beat degenerate baselines.
- Target integrity: component-union target masks are aligned (`target_union_mask_iou_mean=1.000000`), center-to-mask centroid error is `0.000059604 m`, and empty-slot mask/depth sums are `0.0`, so a raster/coordinate bug is not the leading explanation.
- Slot/matching audit: the 25.10 training script uses min-over-slot-permutations and masks parameter, shape, mask, and depth losses by existing slots. Empty slots are included for existence BCE only, which keeps empty-slot treatment structurally correct.
- Three-component audit: three-component coverage is small (`train=7`, `val=2`, `test=3`), and all three test rows predicted two components, giving `merged_rate=1.0`. This is a secondary data-scarcity/topology issue rather than the primary global failure.
- Route decision: unique next step is `B. enter 25.11 mask/depth loss rebalance training`. Do not simply increase model size, and do not discuss baseline replacement before a later formal benchmark and review.

## 2026-06-02 Stage 25.10 surface multi-pit component-set training gate

- Scope: executed a minimal component-set training gate on `comsol_surface_multipit_component_set_pilot_v1`. This trained and evaluated a lightweight C1 fixed-K component-set model only; it did not update `CURRENT_BASELINE.md`, did not export an inference artifact, and did not commit checkpoint/data/NPZ artifacts.
- Data gate: consumed the explicit manifest `results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json`, verified `N=112`, split `72/20/20`, `K=3`, `train_ready_candidate=true`, `baseline_ready=false`, disabled auto/latest discovery, and sha256-matched the ignored assembled NPZ.
- Model gate: implemented `scripts/train_surface_multipit_component_set_gate.py` with Bx/By/Bz `delta_b` input plus `sensor_z_m`, K=3 existence/geometry/rotation/shape-family/mask/depth outputs, and a min-over-slot-permutations loss for existence BCE, active component parameter regression, shape CE, mask BCE/Dice, and depth loss.
- Training result: fixed seed `42`, `180` epochs, selected existence threshold `0.25` on validation. Train loss decreased from `3.014077` to `0.090511`; best validation loss was `1.992049` at epoch `6`.
- Gate decision: `PARTIAL`. Test component recall was `0.837209`, missed rate `0.162791`, extra rate `0.142857`, merged rate `0.200000`, mean center error `0.004284 m`, mean L/W/D relative error `0.154556`, and mean rotation error `0.575410 rad`.
- Signal and failure split: the gate is clearly better than empty / one-slot prior on component recall, and it avoids empty or single-component collapse on average (`pred_component_count_mean=2.1`). However, component mask Dice is only `0.109562`, union mask Dice is `0.130480`, validation union mask Dice is weak (`0.071413`), depth RMSE barely beats empty/prior, and test three-component rows show merged rate `1.0`.
- Route decision: unique next step is `B. run 25.10b failure audit for merged/missed, overlap/touching, slot permutation, and three-component rows`. Do not enter stronger training or baseline discussion until the failure audit explains the raster/merge weaknesses.
- Review: independent read-only review is required before commit and will check implementation, metrics, forbidden artifacts, and baseline boundaries.

## 2026-06-02 Stage 25.9b surface multi-pit component-set top-up pack

- Scope: executed the approved surface multi-pit component-set COMSOL top-up generation and validation. No training, no model gate, no checkpoint/preview/notes artifact commit, and no `CURRENT_BASELINE.md` update.
- COMSOL result: generated `comsol_surface_multipit_topup_pack_v1` with planned/success `96/96`, failed `0`, split `63/17/16`, finite `Bx/By/Bz`, no-defect reference reuse, and `delta_b=b_defect-b_no_defect`.
- Coverage: top-up component counts are `2=84` and `3=12`; separation is balanced across `separated/close/touching/partially_overlapping=24` each; topology covers `disconnected=48`, `touching_boundary=24`, and `partially_overlapping=24`; orientation covers `aligned_x/aligned_y/diagonal=32` each.
- Assembly: created explicit dataset identity `comsol_surface_multipit_component_set_pilot_v1` from the old 25.2 multi-pit seed `16` plus top-up `96`, for assembled `N=112`, split `72/20/20`, fixed `K=3`, and source counts `16/96`.
- Label completeness: validation found `component_label_missing_count=0`; per-component centers, L/W/D, rotation, shape family, component masks, component depth grids, union masks, and union depth grids passed.
- Registry/manifest: `COMSOL_DATA_REGISTRY.md` and `results/manifests/comsol_surface_multipit_component_set_pilot_v1.manifest.json` mark `train_ready_candidate=true`, `baseline_ready=false`, `auto_discovery_allowed=false`, and forbid baseline replacement, latest/newest discovery, and six-parameter RBC success credit.
- Data policy: generated NPZ/data remain ignored and uncommitted. The committed record is limited to generator, inventory/summary, validation/assembly/decision scripts, registry/manifest, and this run's summaries/metrics.
- Review: independent read-only review passed after one scope-control fix. The COMSOL repo had pre-existing unrelated dirty files; they were left untouched and excluded from the staged 25.9b commit boundary.
- Route decision: unique next step is `A. enter 25.10 component-set training gate`. This next step is an explicit future gate, not an automatic training run or baseline transition.

## 2026-06-02 Stage 25.9 surface multi-pit component-set branch plan

- Scope: completed a plan-only component-set branch design for `multi_pit_two_component_surface_defect`. No COMSOL, no training, no data/NPZ generation or mutation, no checkpoint/preview/notes artifact, and no `CURRENT_BASELINE.md` update.
- Preflight: used explicit 25.1 taxonomy/schema/COMSOL feasibility, the 25.2 `comsol_surface_shape_extension_pilot_v1` manifest, the 25.3 oracle-vs-baseline diagnosis, and the 25.8 report route decision. `COMSOL_Multiphysics_MCP` was used only as read-only generator context.
- Label audit: existing multi-pit seed rows are `16` samples with split `9/3/4`; all have `component_count=2`, two component JSON entries, component centers, L/W/depth, union `projected_mask_2d`, and union `depth_grid_m`. They are usable as `C1` fixed-K seed rows but not sufficient for training.
- Schema gaps: top-up must add per-component `rotation_angle`, component-level projected masks, component-level depth grids, and explicit separation/touching/overlap/topology labels. Current seed coverage is all separated/aligned_x/disconnected, so close/touching/overlap coverage is missing.
- Representation: selected `C1 fixed_K_component_set` with `K=3` as first route. Per slot output is `existence_prob`, `center_x_m`, `center_y_m`, `L_m`, `W_m`, `D_m`, `rotation_angle`, `shape_family`, and local profile parameters. Training should use Hungarian matching and permutation-invariant component losses in a later stage.
- Top-up plan: target top-up `N=96`, fallback `N=60`, assembled target `N=112`, default split `72/20/20`, top-up split `63/17/16`. Coverage must balance component count, separation, relative depth, size pair, orientation, topology, and primitive mix.
- COMSOL feasibility: preferred future route is two/three-component Boolean subtract with Bx/By/Bz export, no-defect reference reuse when solver/domain settings match, and explicit failure ledger for Boolean, mesh, solver, component-count, merged-domain, and label/union mismatch.
- Acceptance gates: component recall, missed/merged/extra rates, matched center and L/W/D errors, projected mask Dice/IoU, depth-grid RMSE, single-component control non-collapse, and no six-parameter RBC success credit for multi-pit. Baseline transition remains blocked.
- Route decision: unique next step is `A. execute multi-pit COMSOL top-up generation`. This is a later approved generation stage, not part of 25.9.
- Review: independent read-only review passed with no must-fix and confirmed no COMSOL/training/data mutation, no `CURRENT_BASELINE.md` diff, clear component-set boundary, route uniqueness, and no forbidden artifacts.

## 2026-06-02 Stage 25.8 surface forward-refinement report / visualization package

- Scope: prepared the surface forward-refinement report / visualization package from the locked 25.7 inference runner outputs. No COMSOL, no training, no data/NPZ mutation, no checkpoint artifact commit, no preview PNG commit, no notes artifact, and no `CURRENT_BASELINE.md` update.
- Report package: `scripts/build_surface_forward_refinement_report_package.py` wrote the preflight, report summary, report metrics, and candidate comparison matrix. The report keeps the roles separate: frozen 20.85 is the current baseline, the 25.7 runner is a post-hoc companion, and RBC oracle is an evaluation ceiling only.
- Improvement audit: `scripts/audit_surface_forward_refinement_improvement_cases.py` recorded best improvements, worst remaining target failures, degraded rows, RBC-like controls, multi-pit negative controls, and group audits by `shape_type`, `representation_target`, and failure reason.
- Gallery: `scripts/export_surface_forward_refinement_gallery.py` wrote `results/metrics/surface_forward_refinement_gallery_index.csv` with `42` gallery rows and generated `42` preview PNGs under ignored `results/previews/surface_forward_refinement_gallery/`. PNG previews are not committed.
- Reported results: target subset remained `82` rows. Profile RMSE was `0.000509518351056 -> 0.000220386413188 m`, Er-like was `2.80015739379 -> 0.909941363416`, IoU/Dice moved from `0.32360140234/0.480524080842` to `0.578523465369/0.709451842351`, and forward residual moved from `70.5944261489` to `0.564105036956`.
- Case audit: best profile RMSE improvements were led by `surf_shape_003_rbc_like_smooth_pit`, `surf_shape_005_rbc_like_smooth_pit`, and `surf_shape_065_asymmetric_corrosion`. Worst remaining target failures were led by `surf_shape_073_elongated_crack_like_surface_defect`, `surf_shape_031_flat_bottom_pit`, and `surf_shape_083_elongated_crack_like_surface_defect`.
- Controls and failures: RBC-like control stayed stable (`RMSE 0.000501181023155 -> 0.000165198934316`, `Dice 0.493556208833 -> 0.689754743215`). The audit recorded `17` degraded target rows and `37` degraded-any rows. Multi-pit/component-set rows remain `not_suitable_for_rbc_refinement` and receive no RBC success credit.
- Route decision: unique next step is `A. component-set branch for multi-pit`. Baseline transition remains forbidden unless separately requested and reviewed; `CURRENT_BASELINE.md` remains the 20.85 surface RBC baseline.
- Review: independent read-only review passed with no must-fix. It confirmed no COMSOL/training/data mutation, no `CURRENT_BASELINE.md` diff, no preview PNG staging, clear baseline/companion/oracle roles, and no multi-pit RBC success claim.

## 2026-06-02 Stage 25.7 surface forward-refinement inference artifact / runner

- Scope: exported the fixed 25.6 surface forward-refinement inference artifact and implemented the runtime runner. No COMSOL, no main neural training, no data/NPZ mutation, no checkpoint artifact commit, no preview/notes artifact, and no `CURRENT_BASELINE.md` update.
- Artifact: wrote the runtime body to ignored `checkpoints/surface_forward_refinement_artifacts/surface_forward_refinement_inference_artifact_v1.json` and committed only `results/manifests/surface_forward_refinement_inference_artifact_manifest.json`. The manifest records `allowed_use=explicit_surface_forward_refinement_inference` and forbids `current_baseline_replacement` / `automatic_baseline_update`.
- Fixed protocol: preserved `ridge_param_only_linear_alpha_10`, `alpha=10.0`, `lambda_profile=1.0`, `lambda_param=1.0`, `R1_low_dim_param_refinement`, frozen 20.85 predicted six-parameter initialization, and optimization over `L_m/W_m/D_m/wLD/wWD/wLW`.
- Runner boundary: runtime refinement inputs are observed `delta_b`-derived features, frozen 20.85 predicted six params, and the exported artifact. Labels, oracle params, true masks, and true depth are used only after refinement for metrics on the labeled pilot.
- Verification: runner reproduced 25.6 per-sample checked fields with `max_abs_diff=0`. Target subset remained `82` rows, with profile RMSE `0.000509518351056 -> 0.000220386413188 m`, Er-like `2.80015739379 -> 0.909941363416`, IoU/Dice `0.32360140234/0.480524080842 -> 0.578523465369/0.709451842351`, and forward residual `70.5944261489 -> 0.564105036956`.
- Controls: RBC-like control did not degrade (`RMSE 0.000501181023155 -> 0.000165198934316`, `Dice 0.493556208833 -> 0.689754743215`). Multi-pit/component-set rows are marked `not_suitable_for_rbc_refinement`, skip six-parameter optimization, and receive no RBC success credit.
- Contract: `results/summaries/surface_forward_refinement_inference_contract.md` states the runner is a companion/post-hoc refinement layer, not a baseline replacement. Unknown real samples may report `refinement_applied`, not representable success, without oracle/label or human confirmation.
- Route decision: unique next step is `A. surface refinement visualization/report package`. Multi-pit remains a future `component_set` branch, and any baseline transition still requires a separate explicit request.
- Review: independent read-only review passed with no must-fix. It confirmed no forbidden artifact staging, clean runtime input boundary, sufficient manifest identity/protocol, and no `CURRENT_BASELINE.md` diff.

## 2026-06-02 Stage 25.6 surface forward-refinement formal benchmark

- Scope: formalized the 25.5 `F0_feature_space_consistency + R1_low_dim_param_refinement` candidate as a no-baseline-transition benchmark. No COMSOL, no main neural training, no data/NPZ mutation, no checkpoint/preview/notes artifact, and no `CURRENT_BASELINE.md` update.
- Preflight: confirmed `comsol_surface_shape_extension_pilot_v1` registry/manifest loading, 25.3 oracle/baseline/diagnosis metrics, 25.5 refinement metrics and gates, 20.85 baseline artifact manifest, RBC profile generator helpers, and reusable 25.5 scripts. The loader still verifies the ignored NPZ by sha256 and forbids latest/newest discovery.
- Fixed protocol: replayed `ridge_param_only_linear_alpha_10`, `alpha=10.0`, `lambda_param=1.0`, `R1_low_dim_param_refinement`, frozen 20.85 six-parameter initialization, and `L_m/W_m/D_m/wLD/wWD/wLW` optimization. No surrogate-family, alpha, loss-weight, or optimizer hyperparameter search was performed.
- Formal replay: per-sample formal benchmark metrics exactly matched the 25.5 refinement metrics on checked fields (`120` rows, max absolute difference `0.0`). Target subset remained `82` `rbc_representable_but_model_fail` rows.
- Target results: baseline/refined/oracle profile RMSE was `0.000509518351056 / 0.000220386413188 / 0.0000784896954944 m`; Er-like was `2.80015739379 / 0.909941363416 / 0.28925522333`; IoU/Dice moved from `0.32360140234/0.480524080842` to `0.578523465369/0.709451842351`; forward residual moved from `70.5944261489` to `0.564105036956`.
- Formal gates: all `10/10` gates passed. RBC-like control did not degrade (`RMSE 0.000501181023155 -> 0.000165198934316`, `Dice 0.493556208833 -> 0.689754743215`). Multi-pit / `rbc_not_representable` rows stayed negative controls with no RBC success credit.
- Candidate report: the candidate applies to `rbc_representable_but_model_fail` only. It is frozen 20.85 baseline plus post-hoc parameter refinement, not a baseline replacement. Multi-pit still needs a future component-set branch.
- Route decision: unique next step is `A. export surface forward-refinement inference artifact / runner`. `CURRENT_BASELINE.md` remains the 20.85 surface RBC baseline and is unchanged.
- Review: independent read-only review passed with no must-fix. Residual risks are 17 target rows with local non-improvement modes and already-pass references worsening on mean RMSE; a future runner should guard against degrading already-good cases.

## 2026-06-02 Stage 25.5 surface feature-space forward-consistency refinement diagnostic

- Scope: executed the bounded surface feature-space forward-consistency refinement diagnostic. No COMSOL, no main neural training, no data/NPZ mutation, no checkpoint/preview/notes artifact, and no `CURRENT_BASELINE.md` update. The only fitting was the allowed lightweight train-split feature-space ridge surrogate/scaler.
- Preflight: loaded `comsol_surface_shape_extension_pilot_v1` only through `COMSOL_DATA_REGISTRY.md` plus `results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json`; the loader sha256-verified the ignored NPZ and kept latest/newest discovery forbidden.
- Target materialization: carried forward the 25.4 split as `82` refinement targets, `22` already-pass references, and `16` excluded negative controls. Multi-pit / component-set rows were not refined and received no RBC-refinement success credit.
- Surrogate: validation-only selection chose `ridge_param_only_linear_alpha_10`, trained on `63` train-split RBC-representable rows and selected on `21` validation rows. Test rows were final-reporting only.
- Refinement: selected `lambda_param=1.0`. The test-time objective used only observed `delta_b`-derived features and frozen 20.85 predicted `L_m/W_m/D_m/wLD/wWD/wLW`; labels/oracle params were used only for validation selection metrics and final reports.
- Target results: profile RMSE improved from `0.000509518351056 m` to `0.000220386413188 m`; Er-like improved from `2.80015739379` to `0.909941363416`; IoU/Dice improved from `0.32360140234/0.480524080842` to `0.578523465369/0.709451842351`; forward residual improved from `70.5944261489` to `0.564105036956`.
- Gates: all `10/10` acceptance gates passed. RBC-like control did not degrade: RMSE improved from `0.000501181023155 m` to `0.000165198934316 m`, Dice improved from `0.493556208833` to `0.689754743215`.
- Route decision: `refinement_candidate_formed=true`; unique next step is `A. lock 25.5 F0/R1 candidate for a formal no-baseline-transition benchmark`. Already-pass references remain a monitoring bucket, and this result still does not authorize a baseline transition.
- Review: independent read-only review passed with no must-fix. It confirmed explicit manifest loading, split boundaries, test-time input boundaries, multi-pit exclusion, acceptance-gate consistency, and no forbidden artifact or `CURRENT_BASELINE.md` diff.

## 2026-06-02 Stage 25.4 surface forward-consistency refinement plan

- Scope: completed a plan-only forward-consistency refinement design for the surface shape-extension branch. No training, no COMSOL, no data/NPZ generation or mutation, no checkpoint/preview/notes artifact, and no `CURRENT_BASELINE.md` update.
- Preflight: fixed the evidence chain to 25.3 oracle fit, frozen 20.85 baseline inference, oracle-vs-baseline diagnosis, `comsol_surface_shape_extension_pilot_v1` manifest, frozen baseline artifact manifest, current RBC profile generator helpers, NLS-lite / feature baseline context, and `COMSOL_DATA_REGISTRY.md`.
- Target set: selected `82` `rbc_representable_but_model_fail` rows as refinement targets, kept `22` `rbc_representable_and_model_pass` rows as pass references, and marked the `16` `multi_pit_two_component_surface_defect` / `rbc_not_representable` rows as excluded negative controls.
- Surrogate plan: selected `F0_feature_space_consistency` as the first route for 25.5. `F1_neural_forward_surrogate`, `F2_cached_COMSOL_local_refinement`, and `F3_direct_COMSOL_refinement` remain later or non-recommended options.
- Refinement plan: selected `R1_low_dim_param_refinement`, initialized from frozen 20.85 predicted six parameters, optimizing `L_m/W_m/D_m/wLD/wWD/wLW` with forward-feature residual, profile regularity, and parameter-bound terms. Model weights remain unchanged.
- Acceptance gates: require target-subset profile RMSE and Er-like error reduction, IoU/Dice improvement, no RBC-like control collapse, forward residual alignment with profile metrics, no nonphysical parameters, no multi-pit success credit, and no baseline transition.
- Route decision: unique next step is `A. execute 25.5 feature-space forward-consistency refinement diagnostic`. Multi-pit remains a future `component_set` branch and does not block the RBC-representable refinement diagnostic.
- Review: independent read-only review passed with no must-fix and confirmed target counts, multi-pit exclusion, F0/R1 selection, executable gates, output whitelist, and no forbidden artifacts or `CURRENT_BASELINE.md` diff.

## 2026-06-02 Stage 20.99 internal / buried defect feasibility schema

- Scope: defined the internal / buried defect feasibility schema and data-generation design only. No COMSOL, no training, no data/NPZ generation or mutation, and no `CURRENT_BASELINE.md` update.
- Preflight: carried forward the 20.98 dry-run blocker state: `ready_for_inference=false`; the user specimen is `internal_or_buried`; required Bx/By/Bz, no-defect reference, `sensor_z_m`, axis order, scan geometry, units, coordinate system, alignment/gain metadata, and internal ground truth are missing.
- Schema: `INTERNAL_DEFECT_SCHEMA.md` now separates surface / near-surface RBC defects from buried cavity defects. Internal labels require `L_m`, `W_m`, `D_m` or cavity size, `burial_depth_m` / `depth_to_surface_m`, `defect_center_xyz_m`, `shape_type`, profile descriptor or cavity mask, and `ground_truth_method`.
- Smoke-pack design: `scripts/design_internal_defect_comsol_smoke_pack_plan.py` writes a 12-row plan covering `internal_ellipsoid`, `internal_cuboid`, and `sphere_like`, with shallow/medium/deep burial levels, required Bx/By/Bz, no-defect reference, and `delta_b=b_defect-b_no_defect`. It is design-only and does not run COMSOL.
- Route decision: `scripts/decide_internal_defect_feasibility_route.py` forbids direct migration from the surface RBC baseline, requires an independent internal COMSOL generator, keeps Bx/By/Bz as the mainline, marks Bz-only as a limited diagnostic branch, and selects the unique next step `A. execute internal COMSOL smoke pack after metadata confirmation`.
- Review: independent read-only review passed with no must-fix and confirmed the internal/surface boundary, burial-depth labels, script compilation, output whitelist, no `CURRENT_BASELINE.md` diff, and no COMSOL/training/data mutation.

## 2026-06-02 Stage 25.3 current surface RBC baseline generalization audit

- Scope: audited frozen 20.85/20.86 surface RBC baseline on `comsol_surface_shape_extension_pilot_v1`. No training, no COMSOL, no data/NPZ mutation, no checkpoint/preview/notes artifact, and no `CURRENT_BASELINE.md` update.
- Preflight: loaded the pilot only through `COMSOL_DATA_REGISTRY.md` plus `results/manifests/comsol_surface_shape_extension_pilot_v1.manifest.json`; baseline artifact manifest and ignored checkpoint/prediction artifacts were present and sha256-verified.
- RBC oracle fit: all 120 samples fit without silent skip. Oracle representability was `104/120`; non-RBC representability was `80/96`. `multi_pit_two_component_surface_defect` was the only full representation failure family, with representable rate `0.000000` and component merge proxy `1.000000`.
- Frozen baseline inference: model pass was `22/120`; non-RBC pass was `19/96`; RBC-like control pass was only `3/24`. Labels were used only for post-inference metrics; model input was `delta_b/BxByBz` only.
- Diagnosis: `82/120` samples were `rbc_representable_but_model_fail`, `22/120` were `rbc_representable_and_model_pass`, and `16/120` were `rbc_not_representable`. The representation failure bucket was entirely multi-pit; most other failures are 20.85 inversion/generalization failures under the new pilot distribution.
- Required answers: RBC-like smooth pit did not pass as a frozen-model control; flat-bottom and sharp-wall did not fail at the oracle level; asymmetric, crack-like, flat-bottom, sharp-wall, and irregular were mainly model failures; multi-pit is a component representation failure; current 20.85 cannot serve as a non-RBC-like baseline.
- Route decision: unique next step is `D. forward-consistency refinement plan`, with component-set decoder retained as an explicit multi-pit sub-branch. Direct 20.85-style non-RBC baseline transition remains forbidden.
- Review: independent read-only review passed with no must-fix and confirmed no COMSOL/training/data mutation, no label leakage in model inference, no `CURRENT_BASELINE.md` diff, and no forbidden artifacts staged.

## 2026-06-02 Stage 25.2 surface shape-extension COMSOL pilot pack

- Scope: executed the approved surface shape-extension COMSOL pilot generation only. No training, no checkpoint/preview/notes artifact, no `CURRENT_BASELINE.md` update, and no baseline transition.
- COMSOL result: generated `comsol_surface_shape_extension_pilot_v1` with planned/success `120/120`, failed `0`, split `72/24/24`, and axes `Bx/By/Bz`.
- Shape coverage: `rbc_like_smooth_pit=24`; `flat_bottom_pit=16`; `sharp_wall_boxy_corrosion=16`; `asymmetric_corrosion=16`; `elongated_crack_like_surface_defect=16`; `multi_pit_two_component_surface_defect=16`; `irregular_corrosion_non_rbc=16`.
- Geometry / solve gates: Boolean subtract, mesh precheck, and COMSOL solve all passed for successful rows. The generated pack includes `b_defect`, `b_no_defect`, `delta_b`, depth/profile labels, projected masks, component labels, topology labels, and explicit `representation_target`.
- Data policy: generated data and NPZ live under ignored `data/comsol_mfl/...` paths and are not committed. The committed record is limited to the generator, inventory/summary, validation/decision scripts, manifest, registry, and this run's summaries/metrics.
- Validation: PINN validation passed with `n_samples=120`, `delta_b` shape `(120,3,3,201)`, finite `Bx/By/Bz`, `delta_max_abs_error=0.0`, full split/shape/topology coverage, `train_ready_candidate=false`, and `baseline_ready=false`.
- Route decision: enter `25.3 current baseline generalization audit` using the frozen 20.85/20.86 surface RBC baseline. Training and baseline updates remain forbidden until a later explicit 25.4 gate.
- Review: independent read-only review passed with no must-fix. It confirmed `CURRENT_BASELINE.md` unchanged, non-RBC labels are not forced into six RBC parameters, and forbidden generated artifacts are not staged.

## 2026-06-01 Stage 25.1 surface shape-extension dataset plan

- Scope: completed the plan-only surface / near-surface shape-extension dataset design. No COMSOL, no training, no data/NPZ generation or modification, no checkpoint/preview artifact, and no `CURRENT_BASELINE.md` update.
- Preflight: fixed the anchor to the current 20.85/20.86 true 3D RBC-style profile-depth baseline on `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240`; 25.0 Piao-NLS closeout remains diagnostic/QC/classical-comparator only.
- Taxonomy: defined seven families: `rbc_like_smooth_pit`, `flat_bottom_pit`, `sharp_wall_boxy_corrosion`, `asymmetric_corrosion`, `elongated_crack_like_surface_defect`, `multi_pit_two_component_surface_defect`, and `irregular_non_rbc_corrosion`.
- Label schema: kept RBC-like controls on `six_param_rbc`, but non-RBC-like cases use `profile_basis`, `depth_grid`, `component_set`, or `polygon_or_contour`. Multi-pit requires `component_count` and `component_params_json`; crack-like requires `aspect_ratio` and `rotation_angle`; irregular corrosion keeps `depth_grid_m` and `profile_descriptor`.
- Dataset plan: target pilot `N=120`, split `72/24/24`, with RBC-like control `24` and each non-RBC-like family `16`. `N=84` is marked reduced feasibility only because it cannot satisfy full coverage; minimum full-coverage fallback is `N=96`.
- COMSOL feasibility: planned per-shape generation routes across COMSOL-native geometry, imported watertight mesh, height/depth-map solid, stacked layer control, and multi-component Boolean subtract. The 20.70 dynamic material/domain/solver fix remains a required safety pattern for imported or Boolean-subtracted defect solids.
- Model route: fixed the later sequence as 25.2 pilot generation only, 25.3 audit current 20.85 baseline on the pilot, and 25.4 consider training. R0 audit comes before R1/R2/R3/R4/R5 model routes.
- Acceptance gates: generation, dataset, baseline audit, and model gates now require closed body, Boolean subtract, mesh/solver, finite Bx/By/Bz, delta_b check, label validity, split/topology coverage, no label leakage, component/edge/crack metrics, non-RBC improvement, RBC control stability, and forward residual non-worsening.
- Review: independent read-only review passed with no must-fix. It confirmed no forbidden tracked modifications and noted that allowed ignored result files must be staged with explicit `git add -f`.
- Route decision: unique next step is `A_execute_surface_shape_extension_COMSOL_pilot_generation`, after review, still without training or baseline transition.

## 2026-06-01 Stage 25.0 surface Piao-NLS diagnostic branch closeout

- Scope: closed the 24.0A / 24.0B / 24.1 / 24.2 surface Piao-NLS branch as a diagnostic/QC/classical-comparator route. No COMSOL, no training, no data/NPZ generation or modification, and no `CURRENT_BASELINE.md` update.
- Closeout: NLS-lite remains useful as stable delta_b-derived diagnostic/QC features and as a classical comparator input, but it is not an exact Piao 18-feature reproduction. NLS-full-compatible remains a future richer-observation interface because current v3_240 has only `scan_line_count=3`, below the `M>=5` minimum.
- Baseline decision: 24.1 LS-SVM-like NLS-lite improves total MAE/wMAE/Dice but worsens L/D/profile RMSE/Er-like versus 20.85, so it cannot replace the current profile-depth baseline. 24.2 feature fusion improved the neural metrics, but it remains a diagnostic candidate inside the RBC six-parameter bottleneck and is not the next mainline.
- Route change: stop NLS feature-fusion small fixes as the main surface route. The blocker is representation coverage for non-RBC-like surface defects, not another handcrafted feature tweak.
- Next capability target: asymmetric, flat-bottom, crack-like, multi-pit / multi-component surface defects with profile-level reconstruction and forward-consistency gates.
- 25.1 unique recommendation: `A_surface_shape_extension_dataset_plan`, before any geometry-aware decoder, profile-basis decoder, or forward-surrogate consistency implementation.

## 2026-06-01 Stage 24.2 surface RBC NLS-lite feature-fusion diagnostic

- Scope: fused `delta_b/BxByBz` with the existing 24.0A `nlslite_*` feature matrix on fixed `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240`. No COMSOL, no data/NPZ generation or modification, no checkpoint artifact, and no `CURRENT_BASELINE.md` update.
- Input gate: `scripts/load_surface_rbc_nls_feature_fusion_dataset.py` loads v3_240 only through `COMSOL_DATA_REGISTRY.md + manifest`, joins the 291 `nlslite_*` columns by `sample_id`, keeps `sample_id`/`split` out of model input, and uses train-only standardization for both delta_b and features. Preflight checks passed with zero failed checks.
- Candidate screen: seed `42`, epochs `220`, batch size `8`; validation-only selection chose `F1_late_fusion`.
- Multi-seed: seeds `42/123/2026`; validation-only seed selection chose seed `123`. Test final metrics: total normalized MAE `0.598309`, L/W/D MAE `1.816667/1.657295/0.654960 mm`, wMAE `0.183249`, wLD/wWD/wLW `0.203378/0.189823/0.156546`, profile RMSE `0.000317238 m`, Er-like `0.267248`, IoU/Dice `0.793564/0.877942`.
- Comparison: relative to 20.85/20.77, total `-0.079706`, wMAE `-0.017827`, profile RMSE `-0.000070499 m`, Dice `+0.030215`. Relative to 24.1, total `-0.055738`, wMAE `-0.002475`, profile RMSE `-0.000127943 m`, Dice `+0.014953`.
- Route decision: 24.2 forms a surface feature-fusion candidate and should go to a formal rerun if continued, but it remains diagnostic and does not replace `CURRENT_BASELINE.md`. 24.0A remains the NLS-lite feature source, 24.0B remains the future NLS-full-compatible interface, and 24.1 remains the feature-only comparator.

## 2026-06-01 Stage 24.1 surface RBC Piao-style NLS-lite feature baseline

- 范围：使用 24.0A 已生成的 `nlslite_*` 特征训练 classical feature-to-geometry comparator；输入只允许 `nlslite_*`，`sample_id` 只用于 join/reporting，`split` 只用于 train/val/test 划分。没有运行 COMSOL，没有生成或修改 data/NPZ，没有写 checkpoint，没有更新 `CURRENT_BASELINE.md`。
- 数据入口：固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240`，通过 `COMSOL_DATA_REGISTRY.md + manifest` 显式加载；preflight 确认 24.0A feature manifest/CSV/quality 存在，feature CSV 只有 `sample_id`、`split` 和 291 个 `nlslite_*` 列。
- 模型选择：新增 `scripts/train_surface_rbc_piao_style_feature_baseline.py`，候选包括 mean、Ridge、closed-form LS-SVM-like RBF、KernelRidge RBF、SVR RBF；使用 train-only feature scaler、train-only target scaler、validation-only selection，test 只对 validation-selected 模型和固定 mean comparator 报告。
- 结果：validation 选中 `lssvm_rbf_alpha_0p1_gamma_0p00171821`（`LS-SVM-like-RBF`）。test total normalized MAE=`0.654046`，L/W/D MAE=`1.913667/1.991751/0.967491 mm`，wMAE auxiliary=`0.185724`，wLD/wWD/wLW=`0.197556/0.191340/0.168276`，profile RMSE=`0.000445182 m`，Er-like=`0.431187`，IoU/Dice=`0.769353/0.862988`。
- 对比：相对 20.85/20.77，total MAE、W、wMAE、wLD/wWD/wLW、IoU/Dice 改善，但 L、D、profile RMSE 和 Er-like 退化；相对 20.81，total MAE、L、wMAE、wLD/wWD、profile RMSE 轻微改善，但 W、D、wLW、IoU/Dice 较弱。
- route decision：新增 `scripts/decide_surface_rbc_piao_style_feature_baseline_route.py`，结论是 `enter_24_2_feature_fusion_candidate`。24.1 有补充价值，可进入 24.2 feature-fusion diagnostic，但不能替代 `CURRENT_BASELINE.md`，也不能声称 exact Piao NLS。

## 2026-06-01 Stage 24.0B surface RBC NLS full-compatible feature framework

- Scope: implemented the Piao NLS full-compatible schema, extractor, input adequacy validator, synthetic tests, route decision, and review record. No COMSOL, no training, no data/NPZ generation or modification, and no `CURRENT_BASELINE.md` update.
- Schema: `NLS_FULL_COMPATIBLE_FEATURE_SCHEMA.md` fixes required input as aligned Bx/By/Bz ROI matrices over axial x and tangential y, with `M>=5` as the full-compatible minimum and `M>=9` as the recommended full-candidate count.
- Extractor: `scripts/extract_surface_rbc_nls_full_compatible_features.py` emits 82 stable feature columns plus 82 `valid__*` flags, records `feature_schema_version=nls_full_compatible_v1`, and writes degraded/full mode metadata per row.
- Current v3_240 result: `delta_b` shape is `[240,3,3,201]`, axis order is `[Bx, By, Bz]`, `scan_line_count=3`, `sensor_x_count=201`; therefore `full_feature_ready=false`, `degraded_mode=true`, `degraded_mode_reason=scan_line_count_lt_5`.
- Route decision: NLS-full-compatible is a future full ROI / richer-observation interface for surface RBC, not exact Piao full NLS and not a replacement for NLS-lite. Existing internal richer-observation 5/9-line pack is only an interface reference, not surface RBC full data.
- Tests: synthetic tests passed for 3-line degraded mode, 5-line compatible mode, 9-line full-candidate mode, explicit fit failure handling, and feature name stability.
- Claim boundary: `piao_full_compatible=true`; `exact_piao_full=false` until full ROI data and exact equations are validated.

## 2026-06-01 Stage 24.0A surface RBC NLS-lite feature extractor

- 范围：在当前 surface / near-surface true 3D RBC baseline 的固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上实现 NLS-lite feature extractor；没有运行 COMSOL、没有训练、没有生成或修改 data/NPZ、没有更新 `CURRENT_BASELINE.md`。
- 数据入口：通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json` 显式加载；禁止 latest/newest NPZ scan。预检确认 `delta_b=(240,3,3,201)`、axis order `[Bx, By, Bz]`、三条 `scan_line_y=[-0.001,0,0.001]`、split=`162/39/39`。
- extractor：新增 `scripts/extract_surface_rbc_nls_lite_features.py`，输出 `results/metrics/surface_rbc_nls_lite_features.csv`、`results/metrics/surface_rbc_nls_lite_feature_quality.csv`、`results/summaries/surface_rbc_nls_lite_feature_summary.txt` 和 `results/manifests/surface_rbc_nls_lite_feature_manifest.json`。
- 特征边界：正式 feature CSV 只有 `sample_id`、`split` 元数据和 `nlslite_*` 特征列；数值特征只来自 `delta_b/BxByBz`，不使用 `rbc_params`、profile、mask、split、sample_id 或分组 metadata；target labels 只在 correlation audit 中作为 target。
- 质量结果：feature_count=`291`，overall finite fraction=`1.0`，fit_success_rate=`1.0`，fallback_rate=`0.0`，fit_residual_mean=`0.164980`，fit_residual_max=`0.302589`。
- correlation audit：新增 `scripts/audit_surface_rbc_nls_lite_feature_correlations.py`，输出 `results/metrics/surface_rbc_nls_lite_feature_correlations.csv` 和 summary。主要相关特征包括 `nlslite_Bx_yneg_half_peak_width_m` 对 `L_m`/`W_m`，`nlslite_Bx_yneg_abs_peak` 对 `D_m`/profile depth，`nlslite_Bx_yneg_fit_width_m` 对 projected mask area。
- route decision：新增 `scripts/decide_surface_rbc_nls_lite_feature_route.py`，判断 NLS-lite 稳定、fit failure 可接受、无 label leakage，可进入 24.1 feature baseline；真实实验预处理适配为 `yes_with_calibration_caveat`。
- claim 边界：`exact_piao_nls=false`、`piao_nls_lite=true`；当前数据只有三条 scan line，不声称 exact Piao 18-feature reproduction。

## 2026-05-31 Stage 23.4 internal multi-magnetization diagnostic pack

- 范围：执行 internal multi-magnetization COMSOL diagnostic pack generation、schema validation、assembly、registry/manifest 和 route decision；没有训练，没有更新 `CURRENT_BASELINE.md`，没有把 internal defect 写成 baseline。
- COMSOL 结果：计划/成功 `60/60`，30 个 base 全部具备 `M1_mag_y_5line_z0p008` 和 `M2_mag_y_9line_z0p008`，失败 `0`。
- 真实 source 改向：23.4 不是 metadata-only。COMSOL generator 将 source `Je` 从 nominal `["0","1e6[A/m^2]","0"]` 显式改为 orthogonal `["1e6[A/m^2]","0","0"]` 后分别求解 no-defect 和 defect。
- 数据组装：`comsol_internal_defect_multi_magnetization_pack_v1` 已通过 registry/manifest 显式注册；assembled `delta_b` shape 为 `[60,3,2,9,201]`，方向维为 `magnetization_direction_names=["mag_x","mag_y"]`，并记录精确 `nominal_source_je` / `orthogonal_source_je`。
- validation：`validation_passed=true`，`source_je_changed_all_true=true`，`delta_b=b_defect-b_no_defect` 最大误差 `1.1641532182693481e-10`，M1/M2 paired completeness `30/30`。
- registry：`route=internal_buried_defect_multi_magnetization`，`status=diagnostic_pack_generated`，`train_ready_candidate=false`，`baseline_ready=false`，allowed use 仅为 `schema_validation, explicit_multi_magnetization_diagnostic`。
- route decision：可进入 23.5 internal multi-magnetization diagnostic evaluation；训练、真实 internal sample inference 和 `CURRENT_BASELINE.md` 更新继续暂缓。
- review：独立只读 review 通过；初始流程 follow-up 是补写 review summary 和同步 Markdown，已处理。COMSOL repo 既有无关 dirty items 未 staging、未回滚、未提交。

## 2026-05-29 Stage 21.3 internal defect dataset expansion plan

- 范围：只做 internal / buried defect dataset expansion plan。未运行 COMSOL，未训练，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`，未创建 baseline。
- preflight：确认 `comsol_internal_defect_pilot_pack_v1` manifest、21.1 validation、21.2 metrics/decision、`INTERNAL_DEFECT_SCHEMA.md` 和 COMSOL pilot generator 可用；COMSOL repo 既有 dirty items 只记录，不纳入本轮。
- coverage audit：当前 N=96 的 split blocker 明确成立。train 有 `internal_sphere=24/internal_ellipsoid=36/internal_cuboid=4`，但 val/test 都只有 `internal_cuboid=16`；test burial 只覆盖 `deep=7/deep_plus=9`。
- metric trust：21.2 的 center_xyz 可学习信号和 cuboid-only shape signal 可参考；三类 shape 泛化、完整 burial depth 泛化和 baseline 级结论不可信。
- expansion target：v2 assembled dataset 固定为 `comsol_internal_defect_pilot_pack_v2_240`，source N=96，selected top-up N=144，planned top-up N=168，assembled N=240。
- split target：v2 split 固定为 `train/val/test=160/40/40`，重新分配 source + selected top-up rows，不沿用 v1 split。每个 split 都覆盖三类 shape、四档 burial depth、三档 size；ellipsoid/cuboid 在每个 split 都覆盖 compact / elongated_x / elongated_y。
- route decision：下一步进入 21.3b top-up COMSOL pack generation；21.4 才做 internal training gate。internal branch 继续独立，不混入 surface RBC baseline。
- review：独立只读 review 通过。初次 review 要求确认中文 UTF-8 和 aspect gate；已补 18 行 `shape_type_x_aspect_bin` coverage gate 并复审通过。

## 2026-05-28 Stage 21.2 internal defect training gate

- 范围：只在 `comsol_internal_defect_pilot_pack_v1` 上执行 internal defect training gate；未运行 COMSOL，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`，未创建 baseline。
- 数据入口：通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_internal_defect_pilot_pack_v1.manifest.json` 显式加载，禁止 latest/newest NPZ scan。数据为 N=96，split=`64/16/16`，输入 `delta_b` 形状 `(96,3,3,201)`，Conv1D 输入 `(96,9,201)`。
- 输入边界：模型输入只使用 `delta_b/BxByBz`；`shape_type`、`burial_depth_level`、`split`、`sample_id` 和所有 labels 只用于 supervision、metrics 和分组审计。
- feature baseline：validation 选中 `svr_rbf_C10`，test total normalized MAE `0.878883`，L/W/D MAE `1.540 / 1.959 / 0.159 mm`，burial_depth MAE `0.922 mm`，center_xyz MAE `3.828 mm`，shape accuracy `0.000000`。
- neural gate：small Conv1D multitask 模型运行 seeds `42/123/2026`，validation 选中 seed `2026`，test total normalized MAE `1.004271`，L/W/D MAE `1.629 / 1.597 / 0.159 mm`，burial_depth MAE `1.947 mm`，center_xyz MAE `2.666 mm`，shape accuracy `0.812500`。
- 对比结论：neural 综合 score 优于 mean baseline 和 selected feature baseline，主要来自 shape classification；但纯回归 total MAE 与 burial_depth 上，`svr_rbf_C10` feature baseline 更强。
- 关键 blocker：现有 split 覆盖不完整，val/test 都只有 `internal_cuboid`，且 burial_depth 覆盖不全；因此 N=96 只足够证明 internal 分支存在可学习信号，不足以支撑稳健 shape/depth 泛化结论。
- route decision：下一步唯一建议是扩展 internal defect dataset，并重做分层 split；不要把 21.2 写成 baseline，也不要更新 `CURRENT_BASELINE.md`。
- review：独立只读 review 通过，无 must-fix；建议修复项已落实为更精确的 feature baseline 对比口径。

## 2026-05-28 Stage 21.1 internal / buried defect pilot pack generation

- scope: internal / buried defect pilot pack generation, schema validation, registry/manifest, and route decision only. No training, no `CURRENT_BASELINE.md` update, and no surface RBC baseline mixing.
- source: 21.0 `comsol_internal_defect_smoke_pack_v1` had `smoke_generated` status with 12/12 successful COMSOL rows.
- generated dataset: `comsol_internal_defect_pilot_pack_v1`, route `internal_buried_defect_feasibility`, status `pilot_generated`.
- COMSOL result: planned/success `96/96`; shape coverage `internal_sphere=24`, `internal_ellipsoid=36`, `internal_cuboid=36`; burial coverage `shallow=24`, `medium=24`, `deep=24`, `deep_plus=24`; split `train/val/test=64/16/16`.
- signal/schema: `Bx/By/Bz` exported with shape `(96,3,3,201)`; `delta_b=b_defect-b_no_defect` validation passed; labels include `L_m/W_m/D_m`, `burial_depth_m`, `depth_to_surface_m`, `defect_center_xyz_m`, `shape_type`, `aspect_bin`, and `ground_truth_method`.
- registry/manifest: `results/manifests/comsol_internal_defect_pilot_pack_v1.manifest.json` created; `train_ready_candidate=true`, `baseline_ready=false`; latest/newest discovery remains forbidden.
- decision: may enter 21.2 internal defect training gate. This is not a baseline transition; real experimental data remains deferred.
- review: independent read-only review passed with no must-fix.

## 2026-05-27 更新：第 20.89 true 3D RBC gain/amplitude calibration and augmentation gate

第 20.89 在固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 和 20.88a baseline inference artifact 上完成 gain/amplitude robustness gate。本轮没有运行 COMSOL，没有生成或修改 data / NPZ，没有更新 `CURRENT_BASELINE.md`，也没有提交 checkpoint / preview / notes；所有扰动、校准和增强都只作用在内存中的 `delta_b` / BxByBz 输入上。

Calibration-only 阶段复用 20.77/20.85 frozen checkpoint，validation-only 选择 `per_axis_rms_train_stats`。它在 test 上显著降低 gain/channel 退化：gain 0.8 profile degradation 从 no-calibration `123.845%` 降到 `21.194%`，gain 1.2 从 `69.657%` 降到 `21.194%`，Bx 50% attenuation 从 `141.577%` 降到 `12.331%`；但 clean profile RMSE 同时退化 `21.194%`，超过 `<=10%` clean gate，因此不能作为直接校准升级。

Augmentation gate 按 20.77 small Conv1D + MLP six-parameter head 训练 in-memory augmentation 候选，seeds=`42/123/2026`，不保存 checkpoint。第一次 review 指出候选选择误用了 test 诊断排名；已修复为 calibration 用 validation 选策略、augmentation 用训练记录的 validation-only `best_val_score` 选 candidate/seed，test 只做最终报告。复审通过后，validation-selected candidate 为 `A2_axis_gain_aug` seed `123`。其 test clean profile RMSE 为 `0.000525245 m`，相对 20.85 clean 退化 `35.464%`；gain 0.8 degradation 为 `24.614%`，gain 1.2 degradation 为 `38.768%`，Bx 50% attenuation degradation 为 `59.279%`，clean Dice 为 `0.857961`，L/W/D MAE 为 `1.924/1.937/1.044 mm`，wMAE auxiliary 为 `0.213495`。增强显著缓解 gain / Bx attenuation，但 clean profile 代价过大，且 gain 1.2 reduction 未达到 50% gate，因此不升级 baseline。

结论：`CURRENT_BASELINE` 继续保留 20.85 true 3D RBC profile-depth baseline。20.89 只证明当前模型的主要鲁棒性瓶颈是 amplitude/gain calibration 与 Bx 幅值依赖；`A2_axis_gain_aug` 只能作为 non-baseline robustness diagnostic。下一步唯一建议是进入 20.90 liftoff / sensor-offset COMSOL diagnostic pack，并带上 gain/amplitude control caveat；真实数据接入前仍需要明确 amplitude calibration protocol。

## 2026-05-26 更新：第 20.88 true 3D RBC observation perturbation robustness audit

第 20.88 已在固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上完成 observation perturbation robustness audit。本轮复用第 20.88a 的 artifact manifest 和 ignored checkpoint，只对内存中的 `delta_b` / BxByBz 做扰动后前向推理；没有运行 COMSOL，没有训练，没有生成或修改 data / NPZ，没有更新 `CURRENT_BASELINE.md`。

clean replay 与 baseline 对齐：profile RMSE `0.000387737259305327 m`，projected mask Dice `0.8477271366767738`。扰动测试共 31 组，test status 为 green=17、warning=3、fail=11。小噪声表现稳定：noise 10% 的 profile RMSE degradation 为 `4.095415%`，Dice drop 为 `-0.000252`；noise 20% 为 warning，profile degradation `18.302278%`，Dice drop `0.005418`。no-defect reference error 和 sensor_x jitter 也不是当前首要风险，最差 reference error 反而为 `-1.127847%` profile degradation，最差 jitter 只有 `0.731484%`。

主要风险集中在幅值和通道依赖：global gain 0.8x 的 profile degradation 为 `123.845240%`；combined_light / combined_hard 分别为 `37.202858%` / `60.816996%`；Bx 通道最敏感，`channel_attenuation_Bx_50pct` profile degradation 为 `141.577253%`，`channel_dropout_Bx_missing` profile degradation 为 `82.667222%` 且 Dice drop `0.163825`。因此本轮结论不是“整体 robust”，而是“小噪声、reference error、jitter 相对稳定；gain scaling、combined perturbation 和 Bx channel failure 敏感”。wMAE 继续只作为 auxiliary diagnostic。

只读 review agent 通过，唯一 must-fix 是旧 review summary 仍写 preflight blocker；已替换为当前 20.88 review 结论。review 建议也已处理：脚本强制使用 `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json`，并从 artifact manifest 的 clean test metrics 读取 baseline 值。下一步建议是先记录 gain/amplitude calibration 或 augmentation 方案，同时继续准备第 20.89 liftoff / sensor-offset COMSOL diagnostic pack；不要把当前结果写成真实物理鲁棒性或真实实验验证。

## 2026-05-26 更新：第 20.88a true 3D RBC baseline inference artifact recovery

第 20.88a 已完成 frozen baseline artifact recovery/export。本轮只按固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 和 seed `42` 复现 20.77/20.85 small Conv1D encoder + MLP six-parameter head；没有运行 COMSOL，没有生成或修改 data / NPZ，没有更新 `CURRENT_BASELINE.md`，也没有调参、换模型或用 test 反选。

导出的 checkpoint 和 raw prediction artifact 位于 ignored 路径 `checkpoints/true_3d_rbc_baseline_artifacts/`，不会提交；可提交的定位文件是 `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json`。verification 重新加载 checkpoint 并与 prediction artifact 对齐，clean test 指标精确复现 20.85：normalized MAE `0.6780143536818333`，profile_depth_rmse `0.0003877372636895579 m`，Er-like `0.3405436946031375`，L/W/D MAE `1.8918915996566796 / 2.1857599088778863 / 0.8002313476246901 mm`，projected mask IoU/Dice `0.7506502455785019 / 0.8477271366767738`，wMAE auxiliary `0.20107580616306037`。

只读 review agent 通过，无 must-fix。review 建议将 dataset/seed 固定为硬约束，已处理：导出脚本现在拒绝非 v3_240 dataset 和非 seed=42。下一步可以回到第 20.88 observation perturbation robustness audit，用该 manifest 定位 ignored artifact，对扰动后的 `delta_b` 做 frozen-model inference。

## 2026-05-26 更新：第 20.88 observation perturbation robustness audit preflight blocker

第 20.88 按规则进入 true 3D RBC observation perturbation robustness audit，但在 Stage A preflight 停止，没有执行扰动评估。本轮没有运行 COMSOL、没有训练、没有生成或修改 data / NPZ、没有修改 `CURRENT_BASELINE.md`，也没有实现 Stage B/C 扰动脚本。数据 gate 通过：`dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 通过 `COMSOL_DATA_REGISTRY.md` + manifest 显式解析，`delta_b` shape 为 `(240,3,3,201)`，Conv1D view 为 `(240,9,201)`，split 为 `162/39/39`，Bx/By/Bz 轴和标签字段完整。

停止原因是 baseline artifact blocker。20.88 需要 frozen 20.77/20.85 selected seed=42 checkpoint 或足够的 prediction artifact 来对扰动后的 `delta_b` 重新前向；本地只找到 clean seed/profile metrics 和 per-sample profile error rows，没有 true 3D RBC seed=42 checkpoint，也没有 raw `pred_params` / 可重放预测 artifact。Clean metrics 只能说明干净输入上的结果，不能评估 perturbed `delta_b`，因此不能用它们伪造 robustness audit；根据用户硬规则，本轮不允许用重训补 artifact。

独立只读 review agent 同意停止，无 must-fix。Preflight 已补充说明：20.88 没有发起 COMSOL；本机存在 2026-05-20 启动的旧 `comsolmphserver` 进程，但不属于本阶段证据、也未被本阶段使用。下一步唯一建议改为单独的 artifact recovery/export stage：优先恢复 20.77/20.85 seed=42 checkpoint；若无法恢复，再在另一个明确批准的 artifact-export 阶段按固定 20.85 protocol 重新导出 checkpoint / prediction artifact，然后再回到 20.88 做 observation perturbation audit。

## 2026-05-26 更新：第 20.87 true 3D RBC robustness and defect-type expansion design

第 20.87 只完成 robustness / defect-type expansion 方案设计，没有运行 COMSOL，没有训练，没有生成或修改 data / NPZ，也没有修改 `CURRENT_BASELINE.md`、`COMSOL_DATA_REGISTRY.md` 或 manifest。当前 baseline 继续保持第 20.86 的 true 3D RBC profile-depth baseline：`dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240`，输入为 Bx/By/Bz `delta_b`，输出为六个 RBC-style 参数并生成 3D profile/depth 与 projected mask。

Subagent preflight 结论为 GO。Method agent 建议先围绕 profile-depth 主指标做仿真鲁棒性审计，不把 `wLD/wWD/wLW` 重新升为主指标；Data/schema agent 确认 v3_240 的 `(N,3,3,201)` 三轴输入、162/39/39 split 和六参数标签保持不变；COMSOL feasibility agent 将因素分为三层：Layer 1 可做 observation-space 后处理诊断，Layer 2 可近似但正式结论需要 COMSOL，Layer 3 必须新 COMSOL 和/或新标签；Experiment design agent 给出 20.88-20.92 阶段路线；Safety agent 确认本轮只允许提交设计脚本、summary/metrics 和三份路线文档。

本轮新增 `scripts/design_true_3d_rbc_robustness_expansion_plan.py`，生成 factor matrix、stage table 和 acceptance matrix。Layer 1 的优先因素是 additive noise、amplitude scaling / sensor gain error、zero drift、no-defect reference error、channel dropout，以及仅作 20.88 诊断的 `sensor_x_resampling_jitter`。Layer 2 包括 liftoff、scan_line_y offset、Bx/By/Bz spatial misalignment、source strength variation、material/B-H proxy variation；这些因素需要 20.89 小规模 COMSOL diagnostic pack 才能做正式鲁棒性结论。Layer 3 包括 cuboid / ellipsoid / flat-bottom / RBC-like surface shape extension、internal/buried defect、multi-defect 和 arbitrary/free-form profile，必须新 COMSOL 数据和新标签定义。

Acceptance gate 固定为 profile RMSE 主导：green 为 `<= +10%` degradation，warning 为 `+10% to +25%`，fail 为 `> +25%`；L/W/D MAE 的 green/warning/fail 分别为 `<= +15%`、`+15% to +30%`、`> +30%`；projected Dice drop 的 green/warning/fail 为 `<=0.02`、`0.02-0.05`、`>0.05`。Internal/buried defect 被明确放入 20.91 feasibility design，需先定义 `burial_depth` / `depth_to_surface` 与新的 profile/mask 语义，不能混入当前 surface RBC baseline。独立只读 review agent 通过，无 must-fix；三条建议已处理：`sensor_x` formal claim 改入 20.89，20.92 区分 observation robustness 和 shape augmentation，安全清单补充 `.mph`、raw CSV、`*.pt/*.pth` 和 preview 路径。下一步唯一建议是第 20.88 observation perturbation robustness audit。

## 2026-05-26 更新：第 20.86 true 3D RBC benchmark report + baseline transition

第 20.86 完成 true 3D RBC benchmark report package，并将第 20.77 / 20.85 formal rerun 的 profile-depth candidate 升级为新的 `CURRENT_BASELINE`。本轮没有训练、没有运行 COMSOL、没有生成新数据、没有修改 NPZ，也没有提交 data / checkpoint / preview PNG / notes。此次是明确的 baseline transition：旧 v3_complex 2D mask-only / forward-consistency baseline 被降级为 archived comparator，不删除历史记录。

新的 baseline 固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240`，输入为 Bx/By/Bz `delta_b`，shape `(N,3,3,201)`，Conv1D 视图为 `(N,9,201)`；模型为 20.77 small Conv1D encoder + MLP six-parameter head，输出 `L_m/W_m/D_m/wLD/wWD/wLW`，再生成 RBC-style 3D profile/depth 和 projected mask。Formal rerun selected seed 为 `42`，train/val/test normalized MAE 为 `0.646111/0.748694/0.678014`，profile depth RMSE 为 `0.000387737 m`，Er-like profile error 为 `0.340544`，L/W/D MAE 为 `1.892/2.186/0.800 mm`，projected mask IoU/Dice 为 `0.750650/0.847727`，wMAE `0.201076` 只作为 auxiliary diagnostic。

Benchmark report 明确了 comparator roles：20.81 只作为 projected-mask / visual comparator，20.83 是 profile-primary negative gate，旧 2D baseline 是 archived comparator。限制也同步写入：`exact_piao_rbc=False`，当前是 RBC-style / Piao-inspired approximation；尚未在真实实验数据上验证，也不是 arbitrary free-form / multi-defect 部署级 baseline。Review agent 只读复核通过，无 must-fix。

## 2026-05-26 更新：第 20.85 formal true 3D RBC benchmark rerun based on 20.77 candidate

第 20.85 在固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上完成 formal benchmark rerun。本轮没有运行 COMSOL，没有生成或修改 data / NPZ，没有建立 baseline，也没有更新 `CURRENT_BASELINE.md`。数据加载继续通过 `COMSOL_DATA_REGISTRY.md` + manifest 的显式 dataset_id gate，禁止 latest/newest NPZ 自动扫描；模型输入仅为 `delta_b` / BxByBz，labels 只用于 supervision 和 metrics。

Formal rerun 复用第 20.77 的 small Conv1D encoder + MLP six-parameter head、weighted SmoothL1 loss 和 validation-only selection protocol，运行 seeds `42/123/2026`，validation 选择 seed `42`。selected train/val/test normalized MAE 为 `0.646111/0.748694/0.678014`；test L/W/D MAE 为 `1.892/2.186/0.800 mm`；wMAE 仅作为 auxiliary diagnostic，test 为 `0.201076`；profile depth RMSE 为 `0.000387737 m`，Er-like profile error 为 `0.340544`，projected mask IoU/Dice 为 `0.750650/0.847727`。

Audit 结论是 formal rerun 稳定复现 20.77 profile/depth 优势：profile RMSE 与 original 20.77 一致，优于 20.81 feature-fusion 的 `0.000445297 m` 和 20.83 profile-primary negative gate 的 `0.000409718 m`。20.81 仍只作为 projected-mask / visual comparator；20.83 仍是 profile-primary loss path 的 negative evidence。Review agent 只读复核通过，无 must-fix；本轮结果可称 true 3D RBC benchmark candidate，但仍不能称 baseline。

## 2026-05-25 更新：第 20.82 curvature label / output representation audit for true 3D RBC profile

第 20.82 只做 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 的 label / output representation audit；没有运行 COMSOL，没有生成或修改 data / NPZ，没有重新训练模型，没有建立 baseline，也没有修改 `CURRENT_BASELINE.md` 或任何 COMSOL baseline 文档。所有数据口径仍通过 registry / manifest / dataset_id 显式引用，禁止 latest/newest 自动扫描。

Subagent preflight 和 review agent 均为只读。preflight 结论是：Piao-style `wLD/wWD/wLW` 更合理地理解为 RBC profile 生成参数，而不是最终逐项 headline metric；20.77 和 20.81 有逐样本 profile/error metrics，可做参数误差与 profile 指标关系审计；20.80 只有 aggregate/group/failure-case artifacts，不能和 20.77/20.81 的 per-sample artifacts 等价比较；现有 artifacts 没有 raw `pred_params` 或 predicted profile arrays，因此本轮不做 raw prediction reconstruction。

Stage A-D 生成了 Piao evaluation alignment、profile-vs-parameter audit、alternative representation design 和 route decision。关键结果：20.77 test 的 curvature-vs-profile RMSE correlation 只有 `0.358243`；20.81 fusion 的 projected mask Dice 从 `0.847727` 提高到 `0.866573`，但 profile depth RMSE 从 `0.000387737 m` 退到 `0.000445297 m`，说明 projected mask 好并不等于 true 3D profile 好。20.80 feature-only curvature 更好，但 aggregate Dice/profile RMSE 不如 20.77 neural，也不能作为逐样本 profile 证据。

结论是：true 3D / Piao-style branch 应把 `profile_depth_rmse_m` / Er-like profile reconstruction error 升为主评价；`wLD/wWD/wLW` 不删除，但降级为 auxiliary diagnostics；projected mask IoU/Dice 只作为 2D footprint QA。推荐下一步是 `R1_six_params_profile_primary_loss`：仍输出六参数，但训练和 validation 以 profile-level loss / metric 为主，wMAE 只做诊断。review agent 通过，无 must-fix；两条建议已处理：修正 `all` 聚合行，并补充 20.80 aggregate-only caveat。
## 2026-05-25 更新：第 20.80 Piao / NLS-inspired feature diagnostic for true 3D RBC curvature

第 20.80 只在固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上做 Piao/NLS-inspired feature diagnostic；没有运行 COMSOL，没有生成或修改 NPZ，没有训练 neural model，没有建立 baseline，也没有更新 `CURRENT_BASELINE.md`。数据仍通过 `COMSOL_DATA_REGISTRY.md` + manifest 的显式 dataset_id gate 加载，禁止 latest/newest NPZ 自动扫描。本轮不是完整 Piao 2019 复现，也不是 LS-SVM 复现；preflight 中 Piao PDF 正文抽取不稳定，因此只使用已有 fullpaper alignment summary 和当前项目结果定义可执行特征。

Stage A-B 生成并提取了 F0-F5 特征：F0 复用 20.77 的 135 维 hand-crafted control，F1 增加 peak / width / lobe / flatness / sharpness，F2 增加 gradient / zero-crossing / left-right asymmetry，F3 增加 Bx/By/Bz cross-axis 和 vector magnitude，F4 做 bounded gaussian / derivative-of-gaussian NLS proxy，F5 做 curvature-focused ratio 派生量。最终特征数为 642，全部 finite；F4 NLS proxy fit_success_rate=1.0，但它只是 proxy，不是 exact Piao two-stage 18-feature extraction。

Stage C 的 feature regression 使用 train-only imputation/scaling，validation-only feature/model selection，test final only。validation 选中 `F0_F1_F2_basic_physical + svr_rbf_C10_eps0.03`，不是包含 F4 的 NLS feature set。test normalized MAE 为 `0.695724`，L/W/D MAE 为 `2.595/2.361/0.966 mm`，curvature MAE 为 `0.190304`，wLD/wWD/wLW 为 `0.209649/0.194797/0.166465`，projected mask IoU/Dice 为 `0.714534/0.826272`，profile depth RMSE 为 `0.000449640 m`。

与 20.77 neural reference 相比，本轮 curvature 从 `0.201076` 改善到 `0.190304`，wWD / wLW 改善，但 wLD 基本没有改善；同时 total MAE、L/W/D 和 Dice 均弱于 20.77 neural。与 20.77 feature baseline 相比，total MAE、curvature、wLD、wLW、Dice 和 depth RMSE 有改善，wWD 基本持平略差。结论是：F0+F1+F2 physical features 对 curvature 有实质但有限的帮助，F4 NLS proxy 可稳定提取但不是本轮收益来源。Review agent 只读复核通过，无 must-fix；建议已采纳，收紧归因并保证候选阶段不计算 test。下一步唯一建议是 feature-fusion / hybrid：保留 20.77 neural path 负责 L/W/D 与 mask/profile，引入 F1/F2 这类稳定物理特征辅助 curvature；仍不做 baseline replacement。

## 2026-05-25 更新：第 20.79 curvature-aware true 3D RBC model refinement on v3_240

第 20.79 只在固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上做 curvature-aware model refinement；没有运行 COMSOL，没有生成新数据，没有修改 NPZ，没有创建或更新 baseline，也没有修改 `CURRENT_BASELINE.md`。所有训练和评估继续通过 `COMSOL_DATA_REGISTRY.md` + manifest 显式加载 dataset_id，禁止 latest/newest NPZ 自动扫描。

本轮先固定 20.77 / 20.78 reference metrics，然后做 seed=42 candidate screen。Validation-only selection 选中 `C1_split_heads`，随后对该 candidate 运行 seeds `42/123/2026`。Selected seed 仍为 `42`，test normalized MAE 为 `0.753387`，L/W/D MAE 为 `2.660/2.135/1.112 mm`，curvature MAE 为 `0.211584`，wLD/wWD/wLW 为 `0.232094/0.217639/0.185019`，projected mask IoU/Dice 为 `0.728240/0.834597`，profile depth RMSE 为 `0.000555089 m`。

与第 20.77 reference 相比，curvature MAE 从 `0.201076` 退化到 `0.211584`，total normalized MAE 从 `0.678014` 退化到 `0.753387`，L_m、D_m、wLD、wWD、mask Dice 和 depth RMSE 均退化；只有 W_m 和 wLW 有轻微改善。`C2_split_heads_curv_weight_1p5` 在 test 上看起来更好，但 validation score 未选中，因此不能用 test 反选。Review agent 只读复核通过，无 must-fix，同意本轮不升级 refined model。

路线结论：第 20.79 的价值是诊断性负结果，说明简单 split-head / curvature-weighted 小改没有解决 curvature identifiability。当前应保留第 20.77 v3_240 benchmark candidate，不把 20.79 refined model 写成 baseline 或 candidate upgrade；下一步优先 exact Piao / NLS-inspired feature pipeline，其次 curvature-targeted data top-up。

## 2026-05-25 更新：第 20.78 formal true 3D RBC benchmark candidate audit on v3_240

第 20.78 在第 20.77 training gate 通过后，只做 formal benchmark candidate audit：不运行 COMSOL、不生成或修改 NPZ、不重新训练模型、不做 architecture search、不建立 baseline，也不更新 `CURRENT_BASELINE.md`。Subagent preflight 全部为 GO：registry / manifest gate 通过，20.77 / 20.75 / 20.73 指标完整且一致，现有材料足够支撑固定口径审计。

Benchmark candidate summary 固定为 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`：`status=pilot_generated`、`train_ready_candidate=True`、`baseline_ready=False`、`geometry_method=imported_watertight_mesh_solid`、`exact_piao_rbc=False`、`rbc_style_approximation=True`。核心证据仍是 20.77 的 test 指标：neural selected seed `42`，test normalized MAE `0.678014`，优于 mean baseline `0.912677` 和 feature comparator `0.715395`；L/W/D MAE 为 `1.892/2.186/0.800 mm`，projected mask IoU/Dice 为 `0.750650/0.847727`，profile depth RMSE 为 `0.000387737 m`。

Curvature / failure audit 显示风险不是全局平均失败，而是集中在 boxy / sharp 形状和曲率表示：test split 中 `wLD/wWD/wLW` absolute error 为 `0.209439/0.204469/0.189319`，其中 `wLD` 最差；`boxy` curvature MAE `0.296838`、`sharp` `0.2817`，而 `round` 只有 `0.094515`。D_m 已随 N=240 明显改善，但 curvature 没有随之改善；典型样本 `rbc_v3topup_123_test_boxy_medium_wide` 的 projected mask Dice 达到 `0.956750`、D error 只有 `0.131 mm`，curvature error 仍为 `0.364948`，说明 2D projected mask 指标不足以评价 true 3D RBC curvature。

Route decision 是 `promising_but_curvature_risk`：v3_240 可以称为 formal true 3D RBC benchmark candidate，但不能称为 baseline，也不能替换 `CURRENT_BASELINE.md`。下一步唯一建议从“继续盲目扩样到 480”改为 **model refinement**，重点做 curvature-aware head / loss、stronger sequence encoder，以及 exact Piao / NLS-inspired feature diagnostic；curvature-targeted data top-up 作为后续第二选择。

## 2026-05-25 更新：第 20.77 true 3D RBC training gate on v3_240

第 20.77 在 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上完成第三轮 true 3D RBC training gate。本轮只在 PINN_project 执行，不运行 COMSOL、不生成或修改 NPZ、不建立 baseline、不更新 `CURRENT_BASELINE.md`。数据加载严格通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v3_240.manifest.json` 的显式 `dataset_id` gate，禁止 latest/newest NPZ 自动扫描；input gate 确认 `delta_b=(240,3,3,201)`，Conv1D 输入为 `(240,9,201)`，split 为 train/val/test = 162/39/39。

Piao-inspired feature sanity comparator 只使用 Bx/By/Bz-derived features，不声称完整复现 Piao 2019。validation 选择 `svr_rbf_C10`，test normalized MAE 为 `0.715395`，L/W/D MAE 为 `2.703/2.486/0.980 mm`，curvature MAE 为 `0.195046`，projected mask IoU/Dice 为 `0.702335/0.815450`。该 comparator 只作为 sanity baseline，不作为正式 baseline。

Neural gate 使用 small Conv1D + MLP，输入只含 `delta_b`，labels / split / template / depth_bin / aspect_bin 仅用于 supervision、selection 或 grouping metrics。三个 seeds `42/123/2026` 全部完成，validation 选择 seed `42`；best train fit 为 seed `123` 的 normalized MAE `0.010562`，说明模型可拟合训练集。selected test normalized MAE 为 `0.678014`，优于 mean baseline `0.912677` 和 feature baseline `0.715395`；L/W/D MAE 为 `1.892/2.186/0.800 mm`，projected mask IoU/Dice 为 `0.750650/0.847727`，profile depth RMSE 为 `0.000388 m`。

相对第 20.75 N=112，N=240 的 neural test normalized MAE 从 `0.703907` 改善到 `0.678014`，D_m MAE 从 `1.106 mm` 改善到 `0.800 mm`，mask Dice 从 `0.8364` 改善到 `0.8477`；相对第 20.73 N=56 也整体改善。当前可学习参数仍是 `L_m/W_m/D_m`，`wLD/wWD/wLW` 仍不稳定，curvature MAE 相对 N=112 从 `0.190509` 退到 `0.201076`。route decision 为 `v3_240_promising_benchmark_candidate`：下一步可以进入 formal true 3D RBC benchmark candidate / model refinement，但不能自动替换 baseline。

## 2026-05-24 更新：第 20.72 true 3D RBC pilot top-up generation and assembled pack validation

第 20.72 在不覆盖 20.71 partial NPZ 的前提下完成 top-up generation 和 assembled pack validation；本轮不训练 surrogate / inverse model、不做 refinement、不建立 baseline、不更新 `CURRENT_BASELINE.md`，也不提交 data / NPZ / temp STL / `.mph` / raw CSV。Subagent preflight 已完成，Agent F 因平台 agent 上限未能 spawn，主控用只读检查补齐 feasibility 结论并记录在 preflight summary 中。

Stage A-C 先审计 20.71 partial pack：原始 source pack 为 30 pass、2 fail、28 not_attempted，split 为 20/5/5，缺失 `LD_dominant` 和 `WD_dominant`。本轮设计 33-row top-up plan，重点补齐 LD/WD curvature families，并为 deep-elongated timeout 样本设置 bounded replacement。Top-up watertight mesh validation 为 33/33 pass，全部仍是 `imported_watertight_mesh_solid` 路线，不使用 high-layer fallback。

Stage D 的 COMSOL top-up 使用 20.70 protocol：`selected_solver_protocol=default`、`mesh_auto_size=5`、`material_fix_applied=True`、`full_source_jscale=1.0`、`no_defect_reused=True`。Top-up 结果为 26 pass、5 documented failures、2 not_attempted，成功 split 为 16/5/5；失败集中在 mesh/domain 边界和 deep-elongated bounded retry，没有把失败样本静默写入 assembled pack。所有成功样本真实导出 `[mf.Bx, mf.By, mf.Bz] @ sensor_z_m=0.008`，`delta_b = b_defect - b_no_defect` 校验通过。

Stage E-F 将 20.71 partial source 与 20.72 top-up source 组装为 `comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled`：assembled N=56，split 为 train/val/test = 36/10/10，curvature coverage 为 sharp=11、round=11、boxy=12、LD_dominant=11、WD_dominant=11。NPZ/schema validation 通过，`train_ready_candidate=True`，`baseline_ready=False`；registry / manifest 已更新为 partial_source、topup_source、assembled 三层身份，并显式禁止 latest/newest auto-discovery、baseline update 和 current baseline replacement。Claude Code review 通过，无 must-fix；下一步唯一建议是进入 true 3D training gate，但仍必须通过 explicit dataset_id + manifest 读取，不能把该 pack 写成 baseline。

## 2026-05-24 更新：第 20.71 smooth/mesh-based true 3D RBC pilot pack generation

第 20.71 生成了第一个 smooth/mesh-based true 3D RBC-style imported-watertight pilot pack，但结果必须写成 `partial_pilot_generated`，不是 train-ready，也不是 baseline。本轮不训练 surrogate / inverse model、不做 refinement、不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档，也不提交 data / NPZ / temp STL / `.mph` / raw CSV。

Stage A-B 生成 60-sample RBC-style plan 和 watertight mesh：split 计划为 train/val/test = 40/10/10，参数覆盖 `L_m=0.010-0.030`、`W_m=0.006-0.020`、`D_m=0.001-0.006`、`wLD/wWD/wLW=0.55-1.20`，`angle_rad=0`；60/60 profile validation 通过，60/60 watertight mesh validation 通过。全部样本均标记 `exact_piao_rbc=False`、`rbc_style_approximation=True`，不声称完整复现 Piao 2019。

Stage C-D 使用 20.70 imported watertight mesh solid protocol 生成 partial pack：`geometry_method=imported_watertight_mesh_solid`、`selected_solver_protocol=default`、`mesh_auto_size=5`、`material_fix_applied=True`、`full_source_jscale=1.0`，没有 high-layer fallback。COMSOL 成功样本数为 30，split 为 20/5/5；inventory 完整覆盖 60 行，其中 30 pass、2 fail、28 not_attempted。成功样本真实导出 `[mf.Bx, mf.By, mf.Bz] @ sensor_z_m=0.008`，`delta_b = b_defect - b_no_defect` 校验通过，NPZ/schema validation 30/30 通过。

Registry / manifest 已建立：`dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v1`，`status=partial_pilot_generated`，`allowed_use=schema_validation, explicit_pilot_training_gate`，`forbidden_use=automatic_mainline_training, baseline_update, current_baseline_replacement`。route decision 明确 `train_ready=False`、`baseline_ready=False`；缺失 curvature family 为 `LD_dominant` 和 `WD_dominant`，下一步唯一建议是 top-up generation 后再评估 explicit training gate。Claude Code review 经一轮 must-fix 修复后通过，无剩余 must-fix。

## 2026-05-24 更新：第 20.70 imported watertight solid solver robustness diagnostic

第 20.70 只诊断 20.69 之后的 imported watertight mesh solid defect model stationary solve blocker；本轮不重新生成 Python watertight mesh、不修改 Boolean subtract、不回退 high-layer、不扩样、不训练 surrogate / inverse model、不做 refinement，也不更新 `CURRENT_BASELINE.md` 或任何 COMSOL baseline 文档。

Stage A 的 domain/material audit 复现了关键问题：no-defect reference 的动态 material / source selection 通过，但 imported solid 原始 post-Boolean selection 中 `steel_domains=[2]`、`external_air_domains=[1,2]`、`cavity_domains=[]`，导致 air/steel overlap 且 COMSOL API 未暴露稳定的 cavity domain selection。修复边界被限制为最小 selection/material fix：保持 imported watertight mesh geometry 不变，只把 air selection 动态过滤为 `external_air - steel_notched + exposed cavity if any`，source `Je` 仍只作用于动态 steel selection，没有使用固定 domain ID。

Stage B 的 bounded solver probes 先用 `baseline_reproduce_failure` 复现原始失败，然后 `material_domain_fixed` 在 `mesh_auto_size=5`、`selected_solver_protocol=default`、`source_scale=1.0` 下通过 full-source defect solve；`Jscale=0/0.1/0.3` 仍只作为 diagnostic，不计入 forward-ready。由于已经获得 full-source success，后续 mesh bracket / direct solver / ramp probes 按 bounded rule 标记为 `not_run_after_full_source_success`，没有无界调参，也没有切回 `high_layer_approx_12`。

Stage C-D 完成 one-sample `medium_round` imported-solid forward smoke 和 PINN schema validation：`axis_names=[Bx, By, Bz]`、`axis_expressions=[mf.Bx, mf.By, mf.Bz]`、`sensor_z_m=0.008`，`delta_b / b_defect / b_no_defect` shape 均为 `(1, 3, 3, 201)`；`delta_b = b_defect - b_no_defect` 的 `delta_max_abs_error=0.0`，`defect_signal_norm=0.0205825717`，`Bx/By/Bz` norms 分别为 `0.0153035969 / 0.0023070441 / 0.0135690725`，所有值 finite 且非零。NPZ 只作为 generated data 留在 data 路径，不提交。

Route decision 更新为 `A_imported_solid_solve_forward_validation_pass`：imported watertight solid solve / forward 已在 one-sample smoke 层级可用，且无需 direct solver；下一步可以设计 smooth/mesh-based true 3D RBC pilot，但 pilot 阶段仍应保留 mesh sensitivity / material selection QA。Claude Code review 经 must-fix loop 后最终通过，无剩余 must-fix；本轮不建立 baseline。

## 2026-05-24 更新：第 20.68 smooth / near-smooth true 3D variable-depth builder completion

第 20.68 只做 smooth / near-smooth true 3D defect builder feasibility，不进入 pilot、不训练 surrogate / inverse model、不做 refinement，也不更新 baseline。Stage A 从 20.66/20.67 的 RBC-style plan 中生成 `medium_round` 主样本和 `deep_round`、`medium_boxy` geometry-only 备选样本；3 个样本的 depth map、projected mask、max-depth tolerance、candidate method metadata 均通过 plan validation。

Stage B 在 COMSOL 仓库按 bounded rule 测试 `lofted_contour_solid`、`stacked_workplane_contour_loft`、`interpolated_surface_solid`、`imported_closed_mesh_solid` 和 `high_layer_control_24_or_32`。四个 smooth / near-smooth candidate 均未通过 Stage C hard gate：Loft 操作无法在当前 COMSOL API context 中创建，workplane contour-to-solid loft closure 未验证，ParametricSurface 没有 verified closed-solid / Boolean subtract route，imported closed mesh 虽记录 `mesh_source=triangulated_depth_grid`、`surface_continuity_assumption=piecewise-linear continuous depth surface from RBC-style depth grid; not stepped_layers`、`depth_rmse_vs_target=0.0`，但 Boolean subtract 产生 empty steel domain selection。唯一通过的是 `high_layer_control_24`，其 `closed_body_success=True`、`boolean_subtract_success=True`、`mesh_precheck_success=True`、`spatial_depth_variation=True`、`is_constant_depth=False`，但它被严格标记为 `high_layer_control_pass`，不是 smooth / near-smooth，也不是 exact Piao RBC。

因此 Stage C forward smoke 未执行：没有生成 20.68 smooth NPZ，没有导出 20.68 Bx/By/Bz field，没有计算 20.68 `delta_b`，也没有运行 20.68 NPZ/schema validation。route decision 明确写为 smooth builder remains incomplete；除非人工确认接受 high-layer approximation 作为 pilot 口径，否则不能自动进入 60-sample true 3D RBC pilot。Claude Code review 已完成；其对 `MODEL_STRUCTURE_PLAN.md` 删除项的提示属于用户已有无关 dirty item，本轮按白名单 staging 排除且不回滚用户改动。第 20.68 不修改 `CURRENT_BASELINE.md` 或任何 COMSOL baseline 文档。

## 2026-05-23 更新：第 20.63 multi-axis MFL profile perturbation oracle ordering feasibility

第 20.63 在第 20.62 证明 multi-height Bz 仍不能稳定排序 profile quality 后，转向 same-liftoff multi-axis MFL observation feasibility。本轮只做真实 COMSOL oracle residual ordering audit，不训练 surrogate、不训练 inverse model、不运行 profile refinement，也不更新 baseline。

Stage 0 已完成 multi-agent preflight。Method / literature agent 判断 Bx/By/Bz 三轴观测符合 richer observation / forward consistency 路线，且 20.60-20.62 只否定了 Bz-only 与 multi-liftoff Bz，并未否定 multi-axis；Codebase / Artifact agent 确认 20.60/20.61/20.62 profile plan、profile polygon generator 和 audit 代码可复用；Experiment Design agent 强调若 `mf.Bx/mf.By/mf.Bz` 不能稳定同步导出则必须 block；Safety agent 明确 `MODEL_STRUCTURE_PLAN.md` 删除和 `scripts/visualize_current_baseline.py` 均为无关 dirty item，不提交；Implementation Feasibility agent 确认 `_evaluate_at_points` 支持 expression list，但 `mf.Bx/mf.By` 必须先实测。

Stage A 生成 multi-axis profile perturbation plan：24 base samples、192 profile rows，split = train/val/test 128/32/32 rows，rect/rot = 96/96，8 类 variant 各 24 行。每行固定 `sensor_z_m=0.008`、axis = `[Bx, By, Bz]`、expressions = `[mf.Bx, mf.By, mf.Bz]`；所有 profile polygon valid、mask non-empty，且不使用 polygon base samples。

Stage B 在 COMSOL 仓库生成 multi-axis forward pack。Expression probe 通过，`mf.Bx/mf.By/mf.Bz` 在同一次 expression-list export 中稳定导出；pack 覆盖 192 profile rows、3 axes、576 axis observations，所有行（包括 `true_reference`）均为真实 COMSOL forward，不复用旧 Bz-only 数组。`delta_B = B_defect - B_no_defect` 对三轴全部校验通过，profile polygon valid、mask non-empty、split/type/variant coverage 达到 target。

Stage C 只做 oracle residual ordering audit。结果显示 same-liftoff multi-axis 并没有缓解 profile residual non-identifiability：test Bx-only ordering = `0.4505`，By-only = `0.4955`，Bz-only = `0.4505`，Bx+By+Bz train-std normalized = `0.4505`，mismatch_rate = `0.5495`，residual-error correlation = `0.0242`。三轴 normalized 与 Bz-only 没有 improvement，且低于 20.61 single-height oracle test reference `0.5030`。

结论：same-liftoff Bx/By/Bz multi-axis observation 没有提供可用的 profile quality residual ordering signal，不建议训练 multi-axis profile surrogate，也不应回到 profile-forward refinement。Claude Code review 复审通过，无 method/data must-fix；其提醒 `MODEL_STRUCTURE_PLAN.md` 删除与 `scripts/visualize_current_baseline.py` 为未提交的无关 dirty item，提交时按白名单显式 staging 排除。第 20.63 不更新 baseline。下一步唯一建议转向 **multi-direction excitation / richer scan geometry**，而不是继续扩同 liftoff multi-axis 数据或训练 surrogate。

## 2026-05-23 更新：第 20.62 multi-height Bz profile perturbation oracle ordering feasibility

第 20.62 在第 20.61 证明 single-height Bz oracle residual 接近随机后，只做 richer observation feasibility：生成同一 profile perturbation 的 multi-height Bz observation，并审计真实 COMSOL oracle residual 是否更能排序 profile quality。本轮没有训练 surrogate、没有训练 inverse model、没有运行 profile refinement，也没有更新 baseline。

Stage 0 已完成 multi-agent preflight。Method / literature agent 判断 multi-height / multi-liftoff Bz 符合 richer observation / forward consistency 路线，但只应作为 oracle feasibility；Codebase / Artifact agent 确认 20.61 expanded profile plan / pack 可复用，COMSOL 侧 profile polygon generator 可扩展；Experiment Design agent 建议 target 12 base / 96 profile rows / 3 heights；Safety agent 明确禁止提交 data、NPZ、checkpoint、preview PNG、.mph、raw CSV、notes 和 baseline docs；Implementation Feasibility agent 确认使用 `sensor_z_m = [0.004, 0.008, 0.012]` 可行。

Stage A 生成 multi-height profile perturbation plan：12 base samples、96 profile rows，split = train/val/test 64/16/16 rows，rect/rot = 48/48，8 类 variant 各 12 行。每行固定 3 个 sensor_z heights，因此 total height observations = 288；0.008m observation 标记为可从 20.61 expanded pack 复用，0.004m 和 0.012m 需要真实 COMSOL forward。

Stage B 在 COMSOL 仓库生成 multi-height forward pack：profile_rows=96，height_count=3，total_height_observations=288，reused_observations=96，real_comsol_forward_observations=192。0.008m observation 复用 20.61 exact row；0.004m 和 0.012m observation 使用 profile polygon geometry 做真实 COMSOL forward。`delta_bz = bz_defect - bz_no_defect` 校验通过，profile polygon valid、mask non-empty、split/type/variant coverage 达到 target。

Stage C 只做 oracle residual ordering audit。结果显示 multi-height lift-off 没有改善 profile quality ordering：test single-height 0.008m ordering = `0.4909`，0.004m = `0.4364`，0.012m = `0.4545`，multi-height train-std normalized = `0.4545`；test mismatch_rate = `0.5455`，residual-error correlation = `-0.5920`。相比 20.61 single-height oracle test `0.5030`，multi-height 没有达到 `>0.65` gate，也没有 `+0.10` improvement。

结论：multi-height Bz / multi-liftoff alone 没有缓解 profile residual non-identifiability。Claude Code review 通过，无 must-fix；review 结论是数据生成、row accounting、delta check 和 split discipline 可接受，但 oracle residual 仍接近随机且 test correlation 为负。第 20.62 不支持训练 multi-height profile surrogate，下一步优先转向 **multi-axis / multi-direction observation**，而不是继续扩大同类 multi-liftoff 或回到 profile-forward refinement。

## 2026-05-23 更新：第 20.61 expanded profile perturbation forward pack + surrogate ordering audit

第 20.61 在第 20.60 阴性结果基础上扩大 profile-native perturbation forward data 覆盖，只做 forward pack、profile-compatible surrogate calibration 和 residual ordering audit；不做 refinement，不训练 inverse model，不更新 baseline。

Stage 0 已完成 multi-agent preflight。Method / literature agent 判断 expanded profile perturbation data 符合 profile/basis + forward consistency 路线，且 20.60 只否定了小 partial pack 的稳定性，没有否定 profile-native route；Codebase / Artifact agent 确认可复用 20.58 profile basis、20.60 profile polygon generator 和 20.60 surrogate training code；Experiment Design agent 建议 target 36 base / 288 rows，minimum 24 base / 192 rows；Safety agent 明确禁止提交 data、NPZ、checkpoint、preview PNG、.mph、raw CSV、notes 和 baseline docs；Implementation Feasibility agent 确认需要 COMSOL 和 surrogate training，但不需要 refinement。

Stage A 生成 expanded profile perturbation plan：36 base samples、288 rows，split = train/val/test 192/48/48，rect/rot = 144/144，8 类 variant 各 36 行。row accounting 为 reused_original_rows=36、reused_from_20_60_rows=0、planned_real_comsol_forward_rows=252；所有 profile polygon valid、mask non-empty、variant coverage complete。

Stage B 在 COMSOL 仓库生成 expanded forward pack：total_rows=288，reused_original_rows=36，reused_from_20_60_rows=0，real_comsol_forward_rows=252。`true_reference` 行复用 pilot_v9 原始数组，不计入 real COMSOL forward；其余 profile perturbation 行使用 profile polygon geometry 做真实 COMSOL forward，`delta_bz = bz_defect - bz_no_defect` 校验通过，split/type/variant coverage 达到 target。

Stage C 训练两个 profile-compatible surrogate：`EPPF1_profile_station_mlp` 和 `EPPF2_profile_raster_sequence`。validation 选择 `EPPF1_profile_station_mlp`。selected waveform train/val/test NRMSE = `0.2956 / 0.3314 / 0.3735`，correlation = `0.9554 / 0.9435 / 0.9299`。但 residual ordering gate 未通过：oracle ordering train/val/test = `0.4471 / 0.5120 / 0.5030`，surrogate ordering = `0.5291 / 0.5361 / 0.5569`，mismatch_rate = `0.4709 / 0.4639 / 0.4431`，residual-error correlation = `0.0313 / -0.3945 / 0.2336`。

结论：expanded data 相比 20.60 的 test collapse 有方向性改善（test surrogate ordering 从 `0.2143` 到 `0.5569`，mismatch_rate 从 `0.7857` 到 `0.4431`），但没有达到可用于 refinement 的 gate；更关键的是 COMSOL oracle residual 本身接近随机排序，说明当前 3 条 scan line、单 Bz、constant-depth profile polygon 设置下，profile residual 对 profile quality 的可辨识性不足。Claude Code review 通过，无 must-fix。第 20.61 不更新 baseline；下一步不应直接回到 profile-forward refinement，优先转向 richer observations / multi-height / multi-axis 或 non-identifiability audit。

## 2026-05-22 更新：第 20.60 profile perturbation forward pack + profile surrogate calibration

第 20.60 按修正版 gate 执行 profile-native perturbation forward calibration POC，不训练 inverse model，不运行 refinement，不更新 baseline。Stage 0 先完成 multi-agent preflight：结论是 profile perturbation forward data 符合 Priewald-style forward-model / profile-basis 路线，20.59 只否定了现有 profile surrogate 的 validation gate，没有否定 profile-native perturbation 数据本身；但必须严格区分 true reference 复用行和真实 COMSOL forward 行。

Stage A 在 rect/rot 子集上生成 24 个 base sample、192 行 profile perturbation plan，split 为 train/val/test = 128/32/32 rows，rect/rot = 96/96，每个 base 覆盖 8 类 variant：`true_reference`、`profile_extracted_reference`、`half_width_shrink_local`、`half_width_expand_local`、`smooth_global_width_scale`、`centerline_offset_small`、`roughness_noise`、`mixed_profile_perturbation`。K=8 profile stations 被转换为 16-vertex top-view polygon，polygon validity、non-empty mask、variant completeness 均通过。

Stage B 在 COMSOL 仓库按 minimum acceptable partial 生成 forward pack：`total_rows=96`，`reused_original_rows=12`，`real_comsol_forward_rows=84`，represented base samples = 12，split = 64/16/16，rect/rot = 48/48，8 类 variant 各 12 行。`true_reference` 行明确 `reused_original=True`，其 `delta_bz / bz_defect / bz_no_defect` 来自原始 pilot_v9 NPZ，不计入真实 COMSOL forward rows；真实生成行使用 profile polygon geometry 做 COMSOL forward，`delta_bz = bz_defect - bz_no_defect` 校验通过。

Stage C 只训练两个 profile-compatible surrogate：`PPF1_profile_station_mlp` 和 `PPF2_profile_raster_sequence`，只用 train rows 训练、val 选择、test final，不写 checkpoint。validation score 选中 `PPF1_profile_station_mlp`，其 waveform val/test NRMSE/correlation 为 `0.4396 / 0.8990` 和 `0.3758 / 0.9274`。但是 residual ordering gate 未通过：oracle residual ordering val/test = `0.6786 / 0.5357`，selected surrogate residual ordering val/test = `0.6607 / 0.2143`，mismatch_rate = `0.3393 / 0.7857`，residual-error correlation = `0.5703 / -0.7167`。这说明 profile perturbation data 在 validation 上有局部正信号，但 test split 发生明显 collapse，且 oracle residual 本身在 test 上也偏弱。

因此 Stage D 只写 residual objective audit，不运行 profile refinement。Claude Code review 通过且无 must-fix；review 结论是 pipeline / data boundary / split discipline 可接受，但当前 96-row partial pack 不足以支撑 profile-forward refinement。第 20.60 不作为 baseline。下一步若继续 profile-forward route，应先扩大 profile perturbation data，特别是增加 base sample 覆盖和 test split 稳定性；如果 oracle residual 继续弱，则应转向 richer observations / multi-axis 或保留 no-forward profile basis，而不是继续小调 forward-guided refinement。

## 2026-05-22 更新：第 20.58 mask/profile basis refinement POC

第 20.58 在第 20.57 否定当前 rect/rot low-dimensional Priewald 小调路线后，改为从 strong dense/coarse initializer 的预测 mask 中提取 K=8 profile/basis 表示。该轮不运行 COMSOL、不生成新数据、不训练 direct geometry head，也不更新任何 baseline。

输入检查通过：rect+rot subset 仍为 400，split 为 train/val/test = 268/66/66；dense initializer 只按 20.54 protocol 在内存中复现作为 proposal generator，不写 checkpoint。profile extraction 只比较 P1/P2/P3 三个方法，并由 validation 选择 `P1_hardmask_profile`。最新一次输出中，dense mask test IoU/Dice/area_error 为 `0.6625 / 0.7947 / 0.2225`，profile-extracted mask 为 `0.6589 / 0.7921 / 0.2170`，说明 profile 投影基本保留了 dense proposal 的主要形状，但没有超过第 20.54 的 extracted rotated-box geometry `0.6726 / 0.8017 / 0.1945`。

no-forward profile refinement 只使用 dense initial probability、smoothness、area/bounds prior，不使用 true mask / true geometry 做 optimization。validation 选择 `lambda_smooth=0.01`；test 从 `0.6589 / 0.7921 / 0.2170` 改为 `0.6697 / 0.8002 / 0.2196`，说明 profile basis 本身有边际价值，且明显好于第 20.57 的 rect/rot calibrated refinement `0.6492 / 0.7829 / 0.2417`。但它仍只是接近、没有稳定超过第 20.54 extracted rotated-box proposal。

forward profile refinement 使用第 20.56/20.57 的 S1 surrogate 时只做谨慎 sweep，并保留 `lambda_forward=0` 对照。validation 最终选择 `lambda_forward=0.0, steps=50, lr=0.003`；test post-refine 为 `0.6620 / 0.7938 / 0.2243`，forward residual 未被选择为有效约束。Claude Code review 通过且无必须修复；结论是 mask/profile basis 有边际表示价值，但当前 profile-compatible forward surrogate 不足，继续增加 profile 表示复杂度前应优先解决 forward surrogate mismatch，或暂停 surrogate-dependent refinement。

## 主线同步补充：第 20.x forward data augmentation / COMSOL 数据阶段

第 18.x / 19.x 的内部模型、几何参数化、basis、profile、proposal refinement 和 mask-logit refinement 已基本收口，均未替代当前 `CURRENT_BASELINE`。第 20 阶段的主线已经转向 forward 数据增强和 COMSOL multi-line forward data，目标是提高 MFL 反演问题本身的可辨识性，而不是继续调 decoder / loss / threshold。

近期关键结果：

* 第 20.8：外部 `COMSOL_Multiphysics_MCP` 工程生成 8 个真实 `rectangular_notch` multi-line smoke samples，schema 可读，tiny training smoke 跑通，但只能作为链路验证。
* 第 20.10 / 20.11：生成并验证 36-sample `rectangular_notch` pilot pack，PINN_project 中 ingest、loader、normalization、training gate 和 preview 链路可用。
* 第 20.12 / 20.13：生成并验证 120-sample `rectangular_notch` pilot_v2 pack，split 为 80 / 20 / 20；train-only normalization 和 pilot_v2 training gate 跑通，但 defect_type 仍单一。
* 第 20.14 / 20.15：生成并验证 48-sample `rotated_rect` / angle variation pilot_v3 pack，split 为 32 / 8 / 8；angle 覆盖 `-30, -20, -10, 10, 20, 30`，mask 真实体现旋转，training gate 跑通，per-angle 没有明显 schema 或 loader 问题。

这些结果只说明 COMSOL forward data 链路和 pilot training gate 可用，不更新 v3_complex `CURRENT_BASELINE`，也不作为正式泛化性能结论。下一阶段应优先合并 `rectangular_notch` + `rotated_rect`，再扩展样本数和 defect_type 多样性。

## 当前主线摘要（第 18.4 后）

当前 `CURRENT_BASELINE` 已更新为 v3_complex mask-only grid decoder + forward consistency，`lambda_forward=0.10`，validation-selected probability threshold=`0.80`。第 18.4 review 确认 mask-to-Bz surrogate 独立训练并冻结使用，checkpoint selection 和 threshold selection 均只使用 validation set，test set 只用于最终评估。

相比上一版 mask-only grid decoder + threshold `0.90`，新的 forward consistency baseline 改善 overall IoU、Dice、area_error、center_error 和 Bz MSE，且 `pred_area=0` 保持不变。原 mask-only grid decoder baseline 保留为 boundary reference，composite-selection 保留为 μ-threshold shape-oriented reference，`v3_complex_tv_sweep_2e-6` 保留为 MSE-oriented reference。

需要明确的是：polygon area_error 轻微恶化，polygon / rotated_rect 精细边界圆斑化问题仍未根本解决。后续实验应以新的 forward consistency baseline 为对照，不再继续 loss、threshold、decoder head 或 feature 的小修补。

本文件按实验推进顺序记录项目过程、参数、结果和结论。当前推荐模型和 baseline 以 `CURRENT_BASELINE.md` 为准。

## 目录

1. 第 1 步：data_generator 批量数据生成
2. 第 2 步：Bz signal + 坐标 -> μ map 训练结构
3. 第 3 步：evaluate_pinn 定量评价
4. 第 4 步：TV Loss
5. 第 5 步：L-BFGS refine
6. 第 5.5 步：TV / L-BFGS 参数扫描
7. 第 6 步：物理一致性 Loss
8. 第 6.5 步：物理 Loss 效果对比
9. 第 7 步：v3_complex 复杂缺陷数据生成
10. 第 7.5 步：v3_complex baseline 训练
11. 第 7.6 步：v3_complex 按 defect_type 诊断
12. 第 7.7 步：v3_complex 长训练与 lambda_tv 扫描
13. 第 7.8 步：polygon / multi_defect 细诊断
14. 第 7.9 步：v4_balanced_complex 数据增强与小样本验证
15. 第 7.10 步：v4_balanced_complex 正式数据集生成与 baseline 训练

---

## 第 1 步：data_generator 批量数据生成

### 目标

让 `data_generator_v2.py` 支持批量生成样本，并保存 train / val / test 三个数据集及完整 metadata。

### 修改内容

* 增加批量样本生成流程。
* 保存 `signals`、`mu_maps`、`defect_types`、`metadata`、`metadata_keys`、`x`、`y`。
* metadata 包含 `defect_type`、`center_x`、`center_y`、`width`、`height`、`radius`、`ellipse_a`、`ellipse_b`、`angle`、`triangle_vertices`、`area`、`depth`、`lift_off`、`noise_level` 等字段。
* 增加随机样本可视化检查函数，用于查看 Bz 信号和 μ map。

### 输出文件

* `data/training_data_train.npz`
* `data/training_data_val.npz`
* `data/training_data_test.npz`

### 关键指标 / 结果

* train = 1000 个样本。
* val = 200 个样本。
* test = 200 个样本。
* 三个 npz 均包含 `signals`、`mu_maps`、`defect_types`、`metadata`、`metadata_keys`、`x`、`y`。
* 当时未单独记录模型评价指标。

### 结论

第一步完成，后续训练不再只依赖单个样本，可以基于 train / val / test 数据集推进。

### 下一步

修改 `train_pinn.py`，让模型从只输入坐标升级为输入 Bz signal 和坐标。

---

## 第 2 步：Bz signal + 坐标 -> μ map 训练结构

### 目标

将模型输入从“只输入坐标”升级为 `Bz signal + 坐标 -> μ map`。

### 修改内容

* 在 `train_pinn.py` 中加入 `BzEncoder`，把一维 Bz signal 编码为 latent vector。
* 对空间坐标 `(x, y)` 使用 Fourier feature。
* 拼接 `[bz_latent, coord_features]` 后通过 MLP 输出 `μ(x, y)`。
* 支持 train / val 数据集和 batch 训练。
* 训练阶段先只使用 MSE Loss。

### 输出文件

* `checkpoints/best_model.pt`
* `results/loss_curve.png`
* `results/val_prediction.png` 或同阶段预测对比图

### 关键指标 / 结果

* 训练流程可以读取 `data/training_data_train.npz` 和 `data/training_data_val.npz`。
* 每轮输出 train loss 和 val loss。
* 20 epoch 内 train loss 和 val loss 正常下降。
* 当时未单独记录具体 loss 数值。

### 结论

第二步完成，模型开始具备从漏磁 Bz 信号反演二维 μ map 的基本结构。

### 下一步

新增 `evaluate_pinn.py`，建立统一 test 集评价流程。

---

## 第 3 步：evaluate_pinn 定量评价

### 目标

建立测试集评价流程，加入定量指标，避免只看单个样本可视化结果。

### 修改内容

* 新增或修改 `evaluate_pinn.py`。
* 加载 `data/training_data_test.npz` 和 `checkpoints/best_model.pt`。
* 复用 `train_pinn.py` 中一致的模型结构、BzEncoder、Fourier feature 和 forward 逻辑。
* 在整个 test 集上逐批预测 μ map。
* 将 μ map 按 `mu < 500` 转为缺陷 mask 后计算 mask 类指标。

### 输出文件

* `results/metrics/evaluation_metrics.csv` 或整理前的 `results/evaluation_metrics.csv`
* `results/metrics/evaluation_metrics.txt` 或整理前的 `results/evaluation_metrics.txt`
* test 样本预测图、真实图、预测 mask、真实 mask 对比图

### 关键指标 / 结果

`checkpoints/best_model.pt` 在 simple test 集上的指标：

* MSE = 2.17269746e+04
* MAE = 4.61076419e+01
* IoU = 4.25961039e-01
* Dice = 5.76173878e-01
* area_error = 2.69236126e-01
* center_error = 1.02991446e+00

### 结论

第三步完成，后续不同模型可以用统一指标比较。

### 下一步

在训练中加入 TV Loss，减少 μ map 毛刺和孤立噪点。

---

## 第 4 步：TV Loss

### 目标

在 MSE Loss 基础上加入 Total Variation Loss，减少预测 μ map 的毛刺、孤立噪点和背景碎斑。

### 修改内容

* 在 `train_pinn.py` 中实现 `tv_loss`。
* 将预测结果 reshape 为 μ map 后计算空间相邻差分。
* 损失形式：`total_loss = mse_loss + lambda_tv * tv_loss`。
* 初始 `lambda_tv = 1e-4`。
* 新模型和结果使用 TV 专用文件名，避免覆盖原始模型。

### 输出文件

* `checkpoints/best_model_tv.pt`
* `results/loss_curves/loss_curve_tv.png` 或整理前的 `results/loss_curve_tv.png`
* `results/previews/reconstruction_preview_tv.png` 或整理前的 `results/reconstruction_preview_tv.png`

### 关键指标 / 结果

`lambda_tv = 1e-4` 的 TV 模型 test 指标：

* MSE = 2.17338678e+04
* MAE = 4.29366580e+01
* IoU = 4.11922591e-01
* Dice = 5.60657765e-01
* area_error = 2.69431885e-01
* center_error = 1.04590862e+00

### 结论

TV Loss 流程跑通。初始 `lambda_tv = 1e-4` 下 MAE 有改善，但 IoU、Dice、area_error、center_error 等 mask 类指标没有明显改善。后续已在第 5.5 步中重新评估 TV 权重。

### 下一步

加入 L-BFGS refine，并在后续系统扫描 `lambda_tv`。

---

## 第 5 步：L-BFGS refine

### 目标

在 TV Loss 模型基础上加入 L-BFGS 后期精修，用于降低 refine 子集上的 loss，并验证是否改善泛化指标。

### 修改内容

* 在 `train_pinn.py` 中增加 `--mode lbfgs_refine`。
* 从 `checkpoints/best_model_tv.pt` 加载初始权重。
* 使用固定小子集进行 L-BFGS refine，避免 full-batch 占用过大内存。
* 不覆盖 `checkpoints/best_model.pt` 或 `checkpoints/best_model_tv.pt`。

### 输出文件

* `checkpoints/best_model_tv_lbfgs.pt`
* `results/loss_curves/loss_curve_tv_lbfgs.png`
* `results/previews/reconstruction_preview_tv_lbfgs.png`

### 关键指标 / 结果

当前小子集 refine 设置下的 test 指标：

* MSE = 2.62752306e+04
* MAE = 4.48119284e+01
* IoU = 3.49731439e-01
* Dice = 4.86349610e-01
* area_error = 4.15055531e-01
* center_error = 1.15353433e+00

### 结论

L-BFGS refine 流程已跑通，但当前小子集设置下 test 泛化指标整体变差，因此不作为默认推荐方案，只保留为 optional refine。后续已在第 5.5 步中重新评估 L-BFGS 参数。

### 下一步

对 TV Loss 和 L-BFGS 进行系统参数扫描，固定进入物理 Loss 前的 baseline。

---

## 第 5.5 步：TV / L-BFGS 参数扫描

### 目标

在进入物理一致性 Loss 前，系统比较 TV Loss 和 L-BFGS 参数，选择当前最合适的 baseline。

### 修改内容

* 新增 `parameter_sweep.py`。
* 扫描 `lambda_tv = 0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4`。
* 对最佳 TV 模型做小范围 L-BFGS refine 对比。
* 主要基于 val 集选择参数，test 集用于阶段性最终对比。

### 输出文件

* `results/sweeps/tv_lambda_sweep.csv` 或整理前的 `results/tv_lambda_sweep.csv`
* `results/sweeps/lbfgs_sweep.csv` 或整理前的 `results/lbfgs_sweep.csv`
* `results/summaries/parameter_sweep_summary.txt`
* `checkpoints/best_model_tv_5e-6.pt`

### 关键指标 / 结果

推荐 TV 模型：`checkpoints/best_model_tv_5e-6.pt`

推荐配置：

* `lambda_tv = 5e-6`
* L-BFGS 默认不启用，只保留为 optional refine

最佳 TV 模型 test 指标：

* MSE = 2.16568206e+04
* MAE = 4.39399008e+01
* IoU = 4.32040206e-01
* Dice = 5.82132493e-01
* area_error = 2.42350201e-01
* center_error = 1.03291037e+00

最佳 L-BFGS 候选 test 指标：

* MSE = 2.22554668e+04
* MAE = 5.02426600e+01
* IoU = 4.22638392e-01
* Dice = 5.73455660e-01
* area_error = 2.69747073e-01
* center_error = 1.04688911e+00

### 结论

`lambda_tv = 5e-6` 是 simple 数据集阶段的推荐配置。L-BFGS refine 流程可用，但当前参数下 val/test 指标整体不如最佳 TV 模型，因此不作为默认推荐方案。

### 下一步

以 `checkpoints/best_model_tv_5e-6.pt` 作为第六步物理一致性 Loss 的 baseline。

---

## 第 6 步：物理一致性 Loss

### 目标

在当前最佳 TV baseline 基础上加入物理一致性 Loss，使预测 μ map 经过简化 forward model 后得到的 `Bz_pred` 与输入 Bz signal 匹配。

### 修改内容

* 在 `train_pinn.py` 中加入简化版 `physics_loss`。
* 总损失形式：`total_loss = mse_loss + lambda_tv * tv_loss + lambda_phy * physics_loss`。
* `lambda_tv = 5e-6`。
* `lambda_phy = 1e-4`。
* 从 `checkpoints/best_model_tv_5e-6.pt` 初始化。
* 不启用 L-BFGS。

### 输出文件

* `checkpoints/best_model_tv_phy.pt`
* `results/loss_curves/loss_curve_tv_phy.png`
* `results/previews/reconstruction_preview_tv_phy.png`
* `results/archive/physics_loss_log.csv` 或整理前的 `results/physics_loss_log.csv`

### 关键指标 / 结果

第六步物理 Loss 模型 test 指标：

* MSE = 2.15898657e+04
* MAE = 4.45462792e+01
* IoU = 4.15690292e-01
* Dice = 5.65850626e-01
* area_error = 2.77560911e-01
* center_error = 1.03311558e+00
* physics_loss / Bz reconstruction error = 8.15256860e-02

### 结论

物理一致性 Loss 初版流程跑通，MSE 略有改善，但 MAE、IoU、Dice、area_error、center_error 均变差。该模型不作为默认最佳模型。后续已在第 6.5 步中正式对比确认。

### 下一步

对物理 Loss 模型和第 5.5 步 baseline 做正式效果对比。

---

## 第 6.5 步：物理 Loss 效果对比

### 目标

判断第六步物理一致性 Loss 是否真正优于第 5.5 步 TV baseline。

### 修改内容

* 不重新训练。
* 不修改评价指标定义。
* 基于已有 evaluation 结果对比 `checkpoints/best_model_tv_5e-6.pt` 和 `checkpoints/best_model_tv_phy.pt`。

### 输出文件

* `results/summaries/physics_loss_comparison_summary.txt`
* `results/metrics/physics_loss_comparison.csv`

### 关键指标 / 结果

对比结果：

| 模型 | MSE | MAE | IoU | Dice | area_error | center_error |
|---|---:|---:|---:|---:|---:|---:|
| `checkpoints/best_model_tv_5e-6.pt` | 2.16568206e+04 | 4.39399008e+01 | 4.32040206e-01 | 5.82132493e-01 | 2.42350201e-01 | 1.03291037e+00 |
| `checkpoints/best_model_tv_phy.pt` | 2.15898657e+04 | 4.45462792e+01 | 4.15690292e-01 | 5.65850626e-01 | 2.77560911e-01 | 1.03311558e+00 |

### 结论

物理 Loss 模型只改善 MSE，mask 类指标和 MAE 变差，因此不更新默认最佳模型。当前 simple baseline 仍为 `checkpoints/best_model_tv_5e-6.pt`。

### 下一步

进入复杂缺陷数据生成扩展，不继续叠加物理 Loss 模块。

---

## 第 7 步：v3_complex 复杂缺陷数据生成

### 目标

扩展更复杂、更接近真实情况的缺陷类型，形成第一版复杂缺陷数据集。

### 修改内容

* 在 `data_generator_v2.py` 中新增缺陷类型：`rotated_rect`、`polygon`、`multi_defect`。
* 新增复杂缺陷 metadata 字段：`num_defects`、`component_types`、`component_centers`、`component_sizes`、`component_angles`、`polygon_vertices`、`num_vertices`、`min_mu`、`complexity_level`。
* 新增 `--complex` 命令行参数。
* 保留旧 simple 数据集，不覆盖旧 npz。

### 输出文件

* `data/training_data_v3_complex_train.npz`
* `data/training_data_v3_complex_val.npz`
* `data/training_data_v3_complex_test.npz`
* `results/summaries/v3_complex_dataset_summary.txt`
* `results/previews/data_v3_complex_check_*.png`

### 关键指标 / 结果

正式 v3_complex 数据集规模：

* train = 1000
* val = 200
* test = 200

正式 defect_types 分布：

* train：`multi_defect=331`，`polygon=348`，`rotated_rect=321`
* val：`multi_defect=71`，`polygon=72`，`rotated_rect=57`
* test：`multi_defect=62`，`polygon=75`，`rotated_rect=63`

检查通过：npz 字段完整，signals / mu_maps 无 NaN / Inf，缺陷 mask 非空。

### 结论

v3_complex 数据集生成完成。此阶段只扩展数据，不重新训练模型。

### 下一步

使用 v3_complex 数据集训练独立复杂缺陷 baseline。

---

## 第 7.5 步：v3_complex baseline 训练

### 目标

使用 v3_complex 数据集训练新的复杂缺陷 baseline，并与 simple baseline 区分记录。

### 修改内容

* 在 `train_pinn.py` 中支持 `--dataset v3_complex`。
* 在 `evaluate_pinn.py` 中支持 v3_complex test 数据和自定义输出路径。
* 使用 Adam + TV Loss 训练，不启用 physics_loss，不启用 L-BFGS。

### 输出文件

* `checkpoints/best_model_v3_complex_tv.pt`
* `results/loss_curves/loss_curve_v3_complex_tv.png`
* `results/previews/reconstruction_preview_v3_complex_tv.png`
* `results/metrics/evaluation_metrics_v3_complex_tv.csv`
* `results/metrics/evaluation_metrics_v3_complex_tv.txt`
* `results/summaries/v3_complex_training_summary.txt`

### 关键指标 / 结果

训练配置：

* `lambda_tv = 5e-6`
* `epochs = 20`
* physics_loss：未启用
* L-BFGS：未启用

v3_complex test 指标：

* MSE = 2.07475147e+04
* MAE = 4.36197426e+01
* IoU = 2.76481934e-01
* Dice = 3.97991681e-01
* area_error = 4.26162950e-01
* center_error = 1.34338298e+00

### 结论

v3_complex baseline 跑通，但复杂缺陷 mask 类指标明显低于 simple 数据集结果。`checkpoints/best_model_v3_complex_tv.pt` 与 simple baseline 对应不同数据集，不应混用。

### 下一步

按缺陷类型诊断 v3_complex 模型的薄弱点。

---

## 第 7.6 步：v3_complex 按 defect_type 诊断

### 目标

分析 `checkpoints/best_model_v3_complex_tv.pt` 在不同复杂缺陷类型上的表现差异。

### 修改内容

* 不重新训练。
* 不修改评价指标定义。
* 基于 v3_complex test 结果按 `defect_type`、`num_defects`、`complexity_level`、`num_vertices` 做分组统计。
* 保存最差样本编号和预测对比图。

### 输出文件

* `results/metrics/v3_complex_metrics_by_type.csv`
* `results/metrics/v3_complex_worst_samples.csv`
* `results/summaries/v3_complex_diagnosis_summary.txt`
* `results/previews/v3_complex_worst_samples/`

### 关键指标 / 结果

按 defect_type 诊断：

* polygon 的 IoU 和 Dice 最低：IoU = 2.20801408e-01，Dice = 3.16226151e-01。
* multi_defect 的 MSE、MAE、center_error 较差。
* rotated_rect 相对表现最好。

### 结论

polygon 是主要薄弱类型之一，multi_defect 的中心定位误差也较明显。当前不建议直接进入物理 Loss 或模型结构大改。

### 下一步

先增加训练轮数，并对 v3_complex 单独扫描 `lambda_tv`。

---

## 第 7.7 步：v3_complex 长训练与 lambda_tv 扫描

### 目标

验证 v3_complex baseline 是否只是训练不足或 TV 权重不合适，而不是立即修改模型结构。

### 修改内容

* 使用 v3_complex 数据集训练 100 epoch 长训练模型。
* 进行 v3_complex 专用 `lambda_tv` 扫描：`0`、`1e-6`、`2e-6`、`5e-6`、`1e-5`。
* 每组扫描训练 50 epoch，主要根据 val 集选择参数。
* 不启用 physics_loss，不启用 L-BFGS，不修改模型结构。

### 输出文件

* `checkpoints/best_model_v3_complex_tv_long.pt`
* `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`
* `results/loss_curves/loss_curve_v3_complex_tv_long.png`
* `results/loss_curves/loss_curve_v3_complex_tv_sweep_2e-6.png`
* `results/previews/reconstruction_preview_v3_complex_tv_long.png`
* `results/previews/reconstruction_preview_v3_complex_tv_sweep_2e-6.png`
* `results/metrics/evaluation_metrics_v3_complex_tv_long.csv`
* `results/metrics/v3_complex_long_metrics_by_type.csv`
* `results/metrics/v3_complex_lambda_tv_sweep.csv`
* `results/metrics/evaluation_metrics_v3_complex_tv_sweep_2e-6_test.csv`
* `results/metrics/v3_complex_sweep_2e-6_test_metrics_by_type.csv`
* `results/summaries/v3_complex_long_training_summary.txt`
* `results/summaries/v3_complex_lambda_tv_sweep_summary.txt`

### 关键指标 / 结果

100 epoch 长训练模型 test 指标：

* MSE = 2.06158473e+04
* MAE = 4.73349950e+01
* IoU = 2.75949820e-01
* Dice = 3.95393491e-01
* area_error = 4.32650875e-01
* center_error = 1.30235745e+00

v3_complex 推荐扫描模型：`checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`

推荐 `lambda_tv = 2e-6`，test 指标：

* MSE = 2.07377174e+04
* MAE = 4.44655262e+01
* IoU = 2.95272047e-01
* Dice = 4.21885407e-01
* area_error = 3.94517442e-01
* center_error = 1.32594189e+00

按类型指标：

* rotated_rect：IoU = 3.55649595e-01，Dice = 4.94667257e-01
* polygon：IoU = 2.47369939e-01，Dice = 3.51968972e-01
* multi_defect：IoU = 2.91866765e-01，Dice = 4.32505989e-01

### 结论

长训练只改善部分指标，不作为默认推荐。`lambda_tv = 2e-6` 是当前 v3_complex 推荐配置，模型为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

### 下一步

细分 polygon 和 multi_defect 的失败原因。

---

## 第 7.8 步：polygon / multi_defect 细诊断

### 目标

进一步分析 polygon 和 multi_defect 为什么效果差，判断问题来自形状、数量还是复杂度。

### 修改内容

* 不重新训练。
* 不修改模型结构。
* 不修改评价指标定义。
* 对 polygon 按 `num_vertices` 分组。
* 对 multi_defect 按 `num_defects` 分组。
* 对所有样本按 `complexity_level` 分组。
* 找出 polygon IoU 最低的 10 个样本和 multi_defect center_error 最高的 10 个样本。

### 输出文件

* `results/metrics/v3_complex_polygon_by_vertices.csv`
* `results/metrics/v3_complex_multi_defect_by_count.csv`
* `results/metrics/v3_complex_by_complexity_level.csv`
* `results/metrics/v3_complex_polygon_worst10.csv`
* `results/metrics/v3_complex_multi_defect_worst10.csv`
* `results/summaries/v3_complex_fine_diagnosis_summary.txt`
* `results/previews/v3_complex_fine_diagnosis/`

### 关键指标 / 结果

* polygon 不是顶点数越多越差，5 顶点 polygon 反而最差。
* 最差 polygon 样本多数 IoU = 0、Dice = 0、pred_area = 0，说明主要问题是漏检。
* multi_defect 中缺陷数量越多，center_error 越大。
* complexity_level 越高，MSE、MAE、area_error、center_error 越差。

### 结论

当前不建议直接进入模型结构优化，应先从数据增强和样本平衡处理 small polygon 漏检以及 3 缺陷 multi_defect 泛化问题。

### 下一步

设计并实现 v4_balanced_complex 数据集。

---

## 第 7.9 步：v4_balanced_complex 数据增强与小样本验证

### 目标

基于第 7.8 步诊断结论改进复杂缺陷数据分布，重点缓解 small polygon 漏检和 3 缺陷 multi_defect 样本不足问题。

### 修改内容

* 在 `data_generator_v2.py` 中新增 `--dataset v4_balanced_complex`。
* 针对 polygon 增加 `mask_pixels`、`signal_peak_to_peak`、`signal_snr`、`area_bin`、`balance_group` metadata。
* polygon 要求 `mask_pixels >= 30`、`signal_snr >= 5`。
* 提高 5 顶点 polygon 权重。
* multi_defect 按 2 缺陷 / 3 缺陷约 40% / 60% 分配。
* 限制 multi_defect 组件过度重叠。
* 平衡 `complexity_level` 1 / 2 / 3。
* Claude Code review 后修复 area_bin 阈值与 polygon 半径范围不匹配问题。
* Claude Code review 后移除 polygon 双层 retry，改为样本级单层有限重试，同时保留 `mask_pixels` 和 `signal_snr` 检查。

### 输出文件

小样本阶段使用与正式数据相同的 v4 路径，后续第 7.10 步已用正式规模数据覆盖：

* `data/training_data_v4_balanced_complex_train.npz`
* `data/training_data_v4_balanced_complex_val.npz`
* `data/training_data_v4_balanced_complex_test.npz`
* `results/summaries/v4_balanced_complex_dataset_summary.txt`
* `results/previews/data_v4_balanced_complex_check_*.png`

### 关键指标 / 结果

小样本验证规模：

* train = 50
* val = 10
* test = 10
* seed = 7904

修复后的 area_bin 阈值：

* small：`mask_pixels < 120`
* medium：`120 <= mask_pixels < 500`
* large：`mask_pixels >= 500`

修复后小样本 train 中 polygon area_bin 分布：

* small = 8
* medium = 4
* large = 3

检查通过：npz 字段完整，无 NaN / Inf，无空 mask，polygon 满足 `mask_pixels >= 30` 和 `signal_snr >= 5`，multi_defect 2/3 缺陷比例接近目标。

### 结论

v4_balanced_complex 小样本生成和 review 必修复项均已完成。小样本阶段确认生成逻辑可用，后续已在第 7.10 步生成正式规模数据集。

### 下一步

生成正式规模 v4_balanced_complex 数据集，并训练独立 v4 baseline。

---

## 第 7.10 步：v4_balanced_complex 正式数据集生成与 baseline 训练

### 目标

生成正式规模 v4_balanced_complex 数据集，并使用该数据集训练独立复杂缺陷 baseline，验证数据增强和平衡后是否优于当前 v3_complex 推荐模型。

### 修改内容

* 使用 `data_generator_v2.py --dataset v4_balanced_complex` 生成正式规模数据集。
* 在 `train_pinn.py` 中增加 `v4_balanced_complex` 数据集配置。
* 在 `evaluate_pinn.py` 中增加 `v4_balanced_complex` test 数据配置。
* 使用 Adam + TV Loss 训练 v4 baseline。
* 不启用 physics_loss。
* 不启用 L-BFGS。
* 不覆盖 v3 推荐模型 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

### 输出文件

正式 v4 数据集：

* `data/training_data_v4_balanced_complex_train.npz`
* `data/training_data_v4_balanced_complex_val.npz`
* `data/training_data_v4_balanced_complex_test.npz`

训练与评估输出：

* `checkpoints/best_model_v4_balanced_complex_tv.pt`
* `results/loss_curves/loss_curve_v4_balanced_complex_tv.png`
* `results/previews/reconstruction_preview_v4_balanced_complex_tv.png`
* `results/metrics/evaluation_metrics_v4_balanced_complex_tv.csv`
* `results/metrics/evaluation_metrics_v4_balanced_complex_tv.txt`
* `results/metrics/v4_balanced_complex_metrics_by_type.csv`
* `results/metrics/v4_balanced_complex_polygon_by_area_bin.csv`
* `results/metrics/v4_balanced_complex_polygon_by_vertices.csv`
* `results/metrics/v4_balanced_complex_multi_defect_by_count.csv`
* `results/metrics/v4_balanced_complex_by_complexity_level.csv`
* `results/summaries/v4_balanced_complex_dataset_summary.txt`
* `results/summaries/v4_balanced_complex_training_summary.txt`
* `results/summaries/v4_balanced_complex_diagnosis_summary.txt`

### 关键指标 / 结果

正式 v4 数据集规模：

* train = 1000
* val = 200
* test = 200
* seed = 7904

正式 train 分布：

* defect_types：`circle=75`，`ellipse=75`，`multi_defect=300`，`polygon=300`，`rect=75`，`rotated_rect=100`，`triangle=75`
* polygon area_bin：`small=124`，`medium=103`，`large=73`
* multi_defect num_defects：`2 defects=120`，`3 defects=180`
* complexity_level：`1=300`，`2=400`，`3=300`

v4 baseline 训练配置：

* `lambda_tv = 2e-6`
* `epochs = 100`
* physics_loss：未启用
* L-BFGS：未启用
* 最佳 val_mse_loss = 2.331650e-02

v4 test 整体指标：

* MSE = 2.39571663e+04
* MAE = 4.88803274e+01
* IoU = 2.67902294e-01
* Dice = 3.81393009e-01
* area_error = 4.79983772e-01
* center_error = 1.41093149e+00

v4 分类型指标：

* rotated_rect：IoU = 3.47260192e-01，Dice = 4.77808664e-01
* polygon：IoU = 1.16971656e-01，Dice = 1.69279274e-01
* multi_defect：IoU = 2.85015178e-01，Dice = 4.25033551e-01，center_error = 1.85770430e+00

polygon 按 area_bin：

* small：IoU = 0，Dice = 0，area_error = 1
* medium：IoU = 1.04425846e-01，Dice = 1.51751818e-01
* large：IoU = 3.44668330e-01，Dice = 4.97854875e-01

### 结论

v4 数据集更平衡，但当前 100 epoch / `lambda_tv = 2e-6` 的 v4 baseline 没有明显超过当前 v3_complex 推荐模型 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。polygon 尤其 small area_bin 仍存在明显漏检，multi_defect 的 center_error 仍偏高。因此当前不更新 `CURRENT_BASELINE.md`，v3_complex 推荐模型保持不变。

### 下一步

建议进行 v4_balanced_complex 专属 `lambda_tv` 小范围扫描，优先关注 small polygon 漏检和 multi_defect center_error；暂时不进入 physics_loss、L-BFGS 或模型结构改动。

---

## 第 7.11 步：v4_balanced_complex lambda_tv 扫描

### 目标

在不修改数据生成器、不修改模型结构、不修改评价指标定义、不加入 physics_loss 和 L-BFGS 的前提下，扫描 v4_balanced_complex 专属 `lambda_tv`，判断 TV 权重是否能改善 small polygon 漏检和 multi_defect 定位问题。

### 修改内容

* 使用 v4_balanced_complex train / val / test 数据集。
* 候选 `lambda_tv`：0、5e-7、1e-6、2e-6、5e-6、1e-5。
* 每组训练 50 epoch。
* 每组保存独立 checkpoint，不覆盖 v3_complex 推荐模型。
* 评估 val 和 test，并额外统计 polygon、small polygon、multi_defect、complexity_level 分组表现。

### 输出文件

* `checkpoints/best_model_v4_balanced_complex_tv_sweep_0.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_5e-7.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_1e-6.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_2e-6.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_5e-6.pt`
* `checkpoints/best_model_v4_balanced_complex_tv_sweep_1e-5.pt`
* `results/metrics/v4_balanced_complex_lambda_tv_sweep.csv`
* `results/summaries/v4_balanced_complex_lambda_tv_sweep_summary.txt`
* `results/loss_curves/loss_curve_v4_balanced_complex_tv_sweep_*.png`
* `results/previews/reconstruction_preview_v4_balanced_complex_tv_sweep_*.png`
* `results/metrics/evaluation_metrics_v4_balanced_complex_tv_sweep_*_val.csv`
* `results/metrics/evaluation_metrics_v4_balanced_complex_tv_sweep_*_test.csv`

### 关键指标 / 结果

按 val_iou、val_dice、val_mae、val_area_error、val_center_error 综合排序，本轮推荐候选为：

* `lambda_tv = 0`
* 模型：`checkpoints/best_model_v4_balanced_complex_tv_sweep_0.pt`

推荐候选 val 指标：

* MSE = 2.36408017e+04
* MAE = 5.00588014e+01
* IoU = 2.81450625e-01
* Dice = 3.98008756e-01
* area_error = 4.75910469e-01
* center_error = 1.32469751e+00

推荐候选 test 指标：

* MSE = 2.41644578e+04
* MAE = 5.09550103e+01
* IoU = 2.73743067e-01
* Dice = 3.87241381e-01
* area_error = 4.90251054e-01
* center_error = 1.38652205e+00

重点分组结果：

* test polygon：IoU = 1.16992269e-01，Dice = 1.69966771e-01
* test small polygon：IoU = 0，Dice = 0
* test multi_defect：IoU = 2.79066639e-01，Dice = 4.17910142e-01，center_error = 1.81483585e+00
* complexity_level=1：test IoU = 3.93557208e-01，Dice = 5.34432500e-01
* complexity_level=2：test IoU = 1.79889781e-01，Dice = 2.53846471e-01
* complexity_level=3：test IoU = 2.79066639e-01，Dice = 4.17910142e-01

### 结论

v4 lambda_tv 扫描没有解决 small polygon 漏检问题，small polygon IoU/Dice 仍为 0。`lambda_tv=0` 在 val IoU/Dice 上最好，但 test 指标仍未明显超过当前 v3_complex 推荐模型 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。因此不切换 `CURRENT_BASELINE.md`。

### 下一步

不建议继续只扩大 `lambda_tv` 扫描。下一步应进入模型结构或训练策略优化方案设计，重点考虑 small polygon 的目标权重、mask-aware loss / focal 类 loss，以及 BzEncoder 或空间解码结构增强。

---

## 第 7.12A 步：small polygon defect-weighted MSE Loss

### 目标

在不修改数据生成器、不修改模型结构、不修改评价指标定义、不加入 physics_loss / L-BFGS / soft Dice / oversampling 的前提下，先验证缺陷像素加权 MSE 是否能缓解 v4_balanced_complex 中 small polygon 漏检。

### 修改内容

* 在 `train_pinn.py` 中新增 `--loss-type mse / weighted_mse` 开关，默认值保持 `mse`。
* 新增 `--defect-weight` 参数，默认值为 `10.0`。
* `weighted_mse` 使用归一化标签阈值 `mu_targets < 0.5` 得到缺陷像素，缺陷像素权重为 `defect_weight`，背景权重为 `1.0`。
* 训练日志中在启用 `weighted_mse` 时输出 `weighted_mse_loss`、`unweighted_mse_loss`、`tv_loss`、`physics_loss`、`total_loss` 和 `val_unweighted_mse_loss`。
* 本轮训练配置为 v4_balanced_complex、`loss_type=weighted_mse`、`defect_weight=10.0`、`lambda_tv=0`、100 epoch。

### 输出文件

* `checkpoints/best_model_v4_balanced_complex_smallpoly_loss.pt`
* `results/loss_curves/loss_curve_v4_smallpoly_loss.png`
* `results/previews/reconstruction_preview_v4_smallpoly_loss.png`
* `results/metrics/evaluation_metrics_v4_smallpoly_loss.csv`
* `results/metrics/evaluation_metrics_v4_smallpoly_loss.txt`
* `results/summaries/v4_smallpoly_loss_summary.txt`

### 关键指标 / 结果

训练阶段最佳 `val_unweighted_mse_loss = 3.956198e-02`。

v4 test 整体指标：

* MSE = 4.10216735e+04
* MAE = 7.83255570e+01
* IoU = 3.22104979e-01
* Dice = 4.67866207e-01
* area_error = 1.34222578e+00
* center_error = 1.14444251e+00

重点分组结果：

* polygon：IoU = 2.13777746e-01，Dice = 3.30787212e-01
* small polygon：IoU = 1.36334593e-01，Dice = 2.26148223e-01
* small polygon `pred_area = 0`：0 / 25
* multi_defect：center_error = 9.98214696e-01

与第 7.11 步 v4 `lambda_tv=0` 相比，overall IoU / Dice 和 center_error 改善，small polygon 不再全部漏检；但 MSE、MAE 和 area_error 明显变差。

### 结论

defect-weighted MSE 能缓解 small polygon 漏检，但 `defect_weight=10.0` 会使模型更倾向预测更大的缺陷区域，导致 MSE、MAE、area_error 变差。当前不切换 `CURRENT_BASELINE.md`，v3_complex 推荐模型仍保持 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

### 下一步

建议进入 defect_weight 小范围扫描，候选值可先使用 `5 / 10 / 20 / 50`，主要基于 val 集选择，再用 test 集做阶段性确认。

---

## 第 7.12B 步：v4 small polygon defect_weight 扫描

### 目标

在第 7.12A 证明 weighted MSE 能缓解 small polygon 漏检后，扫描更小的 `defect_weight`，寻找 small polygon 检出率和面积误差之间的平衡点。

### 修改内容

* 不修改 `data_generator_v2.py`。
* 不修改 `evaluate_pinn.py` 的评价指标定义。
* 不加入 Dice Loss、oversampling、physics_loss 或 L-BFGS。
* 不修改模型结构。
* 使用 `train_pinn.py` 已有的 `--loss-type weighted_mse` 和 `--defect-weight` 参数训练。
* 候选 `defect_weight`：2、3、5、7、10。
* 每组训练 100 epoch，`lambda_tv=0`。
* `defect_weight=10` 复用第 7.12A 的同配置模型，并复制为本轮独立命名 checkpoint，未重复训练。

### 输出文件

* `checkpoints/best_model_v4_smallpoly_w2.pt`
* `checkpoints/best_model_v4_smallpoly_w3.pt`
* `checkpoints/best_model_v4_smallpoly_w5.pt`
* `checkpoints/best_model_v4_smallpoly_w7.pt`
* `checkpoints/best_model_v4_smallpoly_w10.pt`
* `results/metrics/v4_smallpoly_defect_weight_sweep.csv`
* `results/summaries/v4_smallpoly_defect_weight_sweep_summary.txt`
* `results/metrics/evaluation_metrics_v4_smallpoly_w*_val.csv`
* `results/metrics/evaluation_metrics_v4_smallpoly_w*.csv`
* `results/loss_curves/loss_curve_v4_smallpoly_w*.png`
* `results/previews/reconstruction_preview_v4_smallpoly_w*.png`

### 关键指标 / 结果

推荐候选：`defect_weight = 5`。

`defect_weight=5` 的 v4 test 指标：

* MSE = 3.12321945e+04
* MAE = 6.23678583e+01
* IoU = 3.39080635e-01
* Dice = 4.77603301e-01
* area_error = 8.38023859e-01
* center_error = 1.17307553e+00
* small polygon IoU = 6.54854895e-02
* small polygon Dice = 1.04442883e-01
* small polygon pred_area=0：12 / 25
* multi_defect center_error = 1.08730941e+00

与 `defect_weight=10` 对比：

* area_error 从 1.34222578e+00 降到 8.38023859e-01；
* overall IoU 从 3.22104979e-01 升到 3.39080635e-01；
* overall Dice 从 4.67866207e-01 升到 4.77603301e-01；
* small polygon IoU/Dice 低于 w10，但不再像 w2 那样全部漏检。

### 结论

`defect_weight=5` 是当前 weighted MSE 的较平衡候选。较小权重 2 / 3 面积误差更低，但 small polygon 仍接近漏检；较大权重 7 / 10 会继续扩大预测缺陷区域，导致 area_error 明显变大。当前不切换 `CURRENT_BASELINE.md`，v3_complex 推荐模型仍为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

### 下一步

建议先让 Claude Code review 第 7.12A / 7.12B 的实现和结果记录。暂不直接进入 Dice Loss；如继续优化，建议以 `defect_weight=5` 为基础讨论 soft Dice / focal 类 loss。

---

## 第 7.13 步：weighted MSE + soft Dice Loss

### 目标

在第 7.12B 推荐的 `defect_weight=5` 基础上加入 soft Dice Loss，继续降低 small polygon 完全漏检数量，同时观察 overall 指标、面积误差和 multi_defect 定位是否变差。

### 修改内容

* 在 `train_pinn.py` 中将 weighted MSE 的缺陷阈值改为 `MASK_THRESHOLD / MU_SCALE`，语义仍对应真实 `μ_r < 500`。
* 新增 `compute_soft_dice_loss()`，只用于训练阶段，不修改 `evaluate_pinn.py` 的 Dice 指标定义。
* 新增 `--loss-type weighted_mse_dice`，默认仍为 `--loss-type mse`。
* 新增 `--lambda-dice`，默认值为 `0.05`。
* 本轮训练配置：v4_balanced_complex、`defect_weight=5`、`lambda_dice=0.05`、`lambda_tv=0`、100 epoch。
* 未修改 data_generator_v2.py。
* 未修改 evaluate_pinn.py 的评价指标定义。
* 未修改模型结构，未启用 physics_loss / L-BFGS / oversampling。

### 输出文件

* `checkpoints/best_model_v4_smallpoly_w5_dice.pt`
* `results/loss_curves/loss_curve_v4_smallpoly_w5_dice.png`
* `results/previews/reconstruction_preview_v4_smallpoly_w5_dice.png`
* `results/metrics/evaluation_metrics_v4_smallpoly_w5_dice.csv`
* `results/metrics/evaluation_metrics_v4_smallpoly_w5_dice.txt`
* `results/summaries/v4_smallpoly_w5_dice_summary.txt`

### 关键指标 / 结果

v4 test 整体指标：

* MSE = 3.56734905e+04
* MAE = 6.02042826e+01
* IoU = 3.25826098e-01
* Dice = 4.64347405e-01
* area_error = 6.12110696e-01
* center_error = 1.24440727e+00

与第 7.12B 的 w5 weighted MSE 相比：

* small polygon pred_area=0：从 12 / 25 降到 0 / 25
* small polygon IoU：从 6.54854895e-02 升到 1.26014768e-01
* small polygon Dice：从 1.04442883e-01 升到 2.01116176e-01
* small polygon area_error：从 9.29900114e-01 降到 6.83718661e-01
* polygon IoU / Dice 均提升
* overall area_error 从 8.38023859e-01 降到 6.12110696e-01
* overall IoU / Dice 下降
* multi_defect center_error 从 1.08730941e+00 升到 1.15517406e+00

### 结论

soft Dice Loss 对 small polygon 漏检有明确改善，也降低了 area_error；但 overall IoU / Dice 和 multi_defect center_error 变差。当前不切换 `CURRENT_BASELINE.md`，也不把该模型设为 v4 small polygon 默认候选。

### 下一步

建议进入 `lambda_dice` 小范围扫描，而不是直接切换 baseline。候选可围绕 `0.01 / 0.03 / 0.05 / 0.1` 设计，并继续以 `defect_weight=5` 为基础。

---

## 第 7.13B 步：v4 small polygon lambda_dice 扫描

### 目标

在 `defect_weight=5`、`lambda_tv=0`、`weighted_mse_dice` 训练配置下扫描 `lambda_dice`，判断 soft Dice Loss 是否能在保持 small polygon 检出的同时恢复 overall IoU / Dice，并避免 multi_defect center_error 明显恶化。

### 修改内容

* 未修改 `data_generator_v2.py`。
* 未修改 `evaluate_pinn.py` 的评价指标定义。
* 未修改模型结构，未启用 physics_loss、L-BFGS 或 oversampling。
* 复用第 7.13A 已实现的 `weighted_mse_dice` 训练逻辑。
* 扫描 `lambda_dice = 0.01 / 0.03 / 0.05 / 0.1`。
* `lambda_dice=0.05` 复用第 7.13A 同配置模型，并复制为本轮独立命名 checkpoint。

### 输出文件

* `results/metrics/v4_smallpoly_lambda_dice_sweep.csv`
* `results/summaries/v4_smallpoly_lambda_dice_sweep_summary.txt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_0p01.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_0p05.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_0p1.pt`

### 关键指标 / 结果

推荐候选：`lambda_dice = 0.03`。

该候选 v4 test 指标：
* MSE = 3.36594865e+04
* MAE = 6.24013944e+01
* IoU = 3.52031766e-01
* Dice = 4.97474011e-01
* area_error = 1.00202667e+00
* center_error = 1.11096996e+00
* small polygon IoU = 1.09439347e-01
* small polygon Dice = 1.83384834e-01
* small polygon pred_area=0 = 0 / 25
* multi_defect center_error = 8.77950897e-01

对比第 7.12B weighted MSE w5：
* small polygon pred_area=0 从 12 / 25 降到 0 / 25。
* overall IoU 从 3.39080635e-01 升到 3.52031766e-01。
* overall Dice 从 4.77603301e-01 升到 4.97474011e-01。
* multi_defect center_error 从 1.08730941e+00 降到 8.77950897e-01。
* area_error 从 8.38023859e-01 升到 1.00202667e+00，面积误差变差。

### 结论

`lambda_dice=0.03` 是本轮最平衡的 v4 small polygon 专项候选：small polygon 完全漏检降为 0/25，overall IoU / Dice 相比 weighted MSE w5 恢复并提升，multi_defect center_error 也改善；代价是 overall area_error 和 small polygon area_error 偏大。该模型仍不切换全项目 `CURRENT_BASELINE.md`，当前全项目推荐 baseline 仍为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

### 下一步

暂不建议立刻进入 focal loss 或 oversampling。建议先围绕 `lambda_dice=0.03` 做预测面积偏大原因分析，或设计更温和的面积约束 / 后处理验证，再决定是否加入新 loss 或采样策略。
---

## 第 7.14 步：area_error 诊断与面积约束方案

### 目标

分析 `checkpoints/best_model_v4_smallpoly_w5_dice_0p03.pt` 为什么 area_error 偏大，判断面积高估是全局问题，还是集中在特定缺陷类型、尺寸或复杂度样本上。

### 修改内容

* 未重新训练。
* 未修改 `data_generator_v2.py`。
* 未修改 `evaluate_pinn.py` 的评价指标定义。
* 未修改模型结构，未启用 physics_loss、L-BFGS、focal loss 或 oversampling。
* 在 `train_pinn.py` 中仅做轻量整理：新增 `normalized_defect_threshold()`，让 weighted MSE 和 soft Dice 共用 `MASK_THRESHOLD / MU_SCALE`，阈值含义仍为真实 `mu_r < 500`。
* 基于第 7.13B 的 per-sample evaluation CSV 和 v4 test metadata 生成面积诊断表。

### 输出文件

* `results/metrics/v4_smallpoly_area_error_per_sample.csv`
* `results/metrics/v4_smallpoly_area_error_by_type.csv`
* `results/metrics/v4_smallpoly_area_error_by_area_bin.csv`
* `results/metrics/v4_smallpoly_area_error_by_vertices.csv`
* `results/metrics/v4_smallpoly_area_error_by_num_defects.csv`
* `results/metrics/v4_smallpoly_area_error_by_complexity.csv`
* `results/metrics/v4_smallpoly_area_error_worst10.csv`
* `results/summaries/v4_smallpoly_area_error_diagnosis_summary.txt`
* `results/previews/v4_smallpoly_area_error_worst10/`

### 关键指标 / 结果

整体面积行为：
* mean area_ratio = 1.989118
* median area_ratio = 1.806303
* pred_area > true_area = 190 / 200
* pred_area < true_area = 10 / 200

按 defect_type：
* polygon mean area_error = 1.478094，为所有类型中最高。
* multi_defect mean area_error = 0.667843，不是本轮面积误差主因。

polygon 按 area_bin：
* small mean area_error = 1.595121，IoU = 0.109439，Dice = 0.183385。
* medium mean area_error = 1.891204，为 polygon area_bin 中最高。
* large mean area_error = 0.649455。

worst10：
* 9 个 polygon，1 个 ellipse。
* polygon worst10 中 small 6 个、medium 3 个。
* complexity_level=2 占 9 个。

### 结论

`lambda_dice=0.03` 的 area_error 偏大主要是系统性过分割：绝大多数样本 pred_area 大于 true_area。问题集中在 polygon，尤其 small / medium polygon；small polygon 的 IoU / Dice 最低，但 mean area_error 最高的是 medium polygon。multi_defect 不是本轮 area_error 主因。

`lambda_dice=0.03` 仍是第 7.13B 的最平衡 v4 small polygon 候选，因为它让 small polygon 完全漏检降为 0 / 25，同时 overall IoU / Dice 和 multi_defect center_error 相比 weighted MSE w5 改善。但该模型不切换全项目 baseline，当前全项目推荐 baseline 仍为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

### 下一步

建议先做一个轻量验证：将 `defect_weight` 降到 3 或 4，观察是否能降低过分割和 area_error。如果仍不能平衡，再考虑新增可选 `area-aware loss`，例如基于 soft mask 面积的相对面积约束，并通过新参数 `lambda_area` 控制，不改评价指标定义。
---

## 第 7.15 步：area-aware loss 面积约束实验

### 目标

在第 7.14 步发现 `lambda_dice=0.03` 模型存在系统性面积高估后，加入可选 area-aware loss，尝试降低 overall area_error、polygon area_error 和 medium polygon area_error，同时观察 small polygon 是否仍保持检出。

### 修改内容

* `train_pinn.py` 新增 `compute_area_loss()`。
* `train_pinn.py` 新增 `--loss-type weighted_mse_dice_area`。
* `train_pinn.py` 新增 `--lambda-area`，默认值为 `0.0`。
* 默认 `--loss-type` 仍为 `mse`，旧训练流程默认不受影响。
* 未修改 `data_generator_v2.py`。
* 未修改 `evaluate_pinn.py` 的评价指标定义。
* 未修改模型结构，未启用 physics_loss、L-BFGS 或 oversampling。

### 输出文件

* `results/metrics/v4_smallpoly_area_loss_sweep.csv`
* `results/summaries/v4_smallpoly_area_loss_sweep_summary.txt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_0p005.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_0p01.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_0p03.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_0p05.pt`

### 关键指标 / 结果

本轮推荐候选：`lambda_area = 0.05`。

该候选 v4 test 指标：
* MSE = 3.20781063e+04
* MAE = 6.21932516e+01
* IoU = 3.50964003e-01
* Dice = 4.94989266e-01
* area_error = 7.94284719e-01
* center_error = 1.24532434e+00
* area_ratio = 1.774067
* pred_area > true_area = 186 / 200
* polygon area_error = 1.197988
* small polygon pred_area=0 = 0 / 25
* small polygon IoU = 1.24021762e-01
* small polygon Dice = 2.04457790e-01
* medium polygon area_error = 1.429367
* multi_defect center_error = 9.67055063e-01

与第 7.13B 无 area loss 候选相比：
* overall area_error 从 1.002027 降到 0.794285；
* polygon area_error 从 1.478094 降到 1.197988；
* medium polygon area_error 从 1.891204 降到 1.429367；
* small polygon pred_area=0 仍为 0 / 25；
* small polygon IoU / Dice 有提升；
* overall IoU / Dice 轻微下降；
* multi_defect center_error 略有变差。

### 结论

area-aware loss 能明显缓解面积高估，但不能完全消除系统性 pred_area 偏大。`lambda_area=0.05` 是本轮最合适的 v4 small polygon / polygon area_error 专项候选，但不切换全项目 baseline。当前全项目推荐 baseline 仍为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

### 下一步

建议先让 Claude Code review 第 7.15 步实现与结果。若继续优化，优先考虑围绕 `lambda_area=0.05` 做小范围复核，或验证 `defect_weight=3 / 4` 是否能进一步减少过分割；暂不建议直接进入 focal loss 或 oversampling。
---

## 第 7.16 步：面积约束细化实验

### 目标

在第 7.15 步 `lambda_area=0.05` 的基础上继续细化面积约束，比较 symmetric area loss 的 `lambda_area=0.04 / 0.05 / 0.07`，并新增 `over_only` 面积约束对比，判断是否能进一步减少 `pred_area > true_area`。

### 修改内容

* `train_pinn.py` 新增 `--area-loss-type symmetric / over_only`。
* 默认 `area_loss_type = symmetric`，避免影响旧实验。
* `over_only` 使用 `relu(pred_soft_area - true_area)`，只惩罚预测面积过大。
* loss curve 标题改为 `Training Loss Curve`。
* 未修改 `data_generator_v2.py`。
* 未修改 `evaluate_pinn.py` 的评价指标定义。
* 未修改模型结构，未启用 physics_loss、L-BFGS、focal loss 或 oversampling。

### 输出文件

* `results/metrics/v4_smallpoly_area_loss_refine.csv`
* `results/summaries/v4_smallpoly_area_loss_refine_summary.txt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_refine_0p04.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_refine_0p05.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_refine_0p07.pt`
* `checkpoints/best_model_v4_smallpoly_w5_dice_area_over_0p05.pt`

### 关键指标 / 结果

symmetric 细扫中，面积指标最好的候选是 `lambda_area=0.07`：
* test IoU = 3.45279115e-01
* test Dice = 4.86960577e-01
* test area_error = 6.91137229e-01
* polygon area_error = 9.26742e-01
* small polygon pred_area=0 = 0 / 25
* medium polygon area_error = 1.203071
* multi_defect center_error = 9.92382e-01

`over_only lambda_area=0.05`：
* test area_error = 6.72180866e-01，为本轮最低；
* pred_area > true_area 从 symmetric 约 185-189 / 200 降到 166 / 200；
* 但 small polygon pred_area=0 回升到 14 / 25；
* small polygon IoU / Dice 明显下降。

### 结论

`over_only` 更能减少面积高估数量，但破坏了 small polygon 检出目标，不适合作为当前推荐方案。symmetric `lambda_area=0.07` 是面积约束最强且仍保持 small polygon 0 漏检的候选，但 IoU / Dice 有下降。继续单纯调 area loss 的收益已经变小，并出现明显 trade-off。

当前不切换全项目 baseline，仍保持 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

### 下一步

建议先让 Claude Code review 第 7.16 步实现和结果。若继续实验，优先考虑 `defect_weight=3 / 4` 与 symmetric `lambda_area=0.04 / 0.07` 的组合；暂不建议进入 focal loss 或 oversampling。
## 第 7.17 步：symmetric area loss 组合验证

### 目标

验证 `defect_weight` 与 `lambda_area` 的组合是否能比第 7.16 步的单独 `lambda_area` 细扫更稳定，尤其关注 small polygon 检出、polygon area_error、overall IoU / Dice 和 multi_defect center_error。

### 修改内容

本轮没有修改数据生成器、评价指标定义或模型结构。训练仍使用 `weighted_mse_dice_area`，固定：

* dataset = v4_balanced_complex
* lambda_dice = 0.03
* lambda_tv = 0
* area_loss_type = symmetric
* epochs = 100

组合验证：

* defect_weight = 5, lambda_area = 0.04
* defect_weight = 5, lambda_area = 0.07
* defect_weight = 7, lambda_area = 0.04
* defect_weight = 7, lambda_area = 0.07

当前 `train_pinn.py` 未提供 seed 参数，本轮每组为单次训练，结果可能受随机性影响。

### 输出文件

* `results/metrics/v4_smallpoly_weight_area_combo.csv`
* `results/summaries/v4_smallpoly_weight_area_combo_summary.txt`
* `checkpoints/best_model_v4_w5_dice003_area004.pt`
* `checkpoints/best_model_v4_w5_dice003_area007.pt`
* `checkpoints/best_model_v4_w7_dice003_area004.pt`
* `checkpoints/best_model_v4_w7_dice003_area007.pt`

### 关键指标 / 结果

本轮综合表现最好的组合是：

* defect_weight = 5
* lambda_area = 0.04
* model = `checkpoints/best_model_v4_w5_dice003_area004.pt`
* test IoU = 0.351333
* test Dice = 0.496197
* test area_error = 0.911511
* polygon area_error = 1.428304
* small polygon pred_area=0 = 0 / 25
* small polygon IoU = 0.123676
* small polygon Dice = 0.202804
* multi_defect center_error = 0.864541

### 结论

`defect_weight=5, lambda_area=0.04` 在本轮中恢复了 overall IoU / Dice，并保持 small polygon 不漏检，multi_defect center_error 也最好。但 polygon area_error 没有继续下降，且不如第 7.16 步的 symmetric `lambda_area=0.07`。本轮组合没有稳定优于当前全项目推荐 baseline，因此不切换 `CURRENT_BASELINE.md`。

第 7.16 中 symmetric `lambda_area=0.04` 与 `0.07` 的 trade-off 仍然成立：0.04 偏向 overall IoU / Dice，0.07 偏向 area_error / polygon area_error。`over_only` 不作为主方案，因为 `pred_area < true_area` 时没有面积修正梯度，会导致 small polygon 漏检回升。

### 下一步

建议暂停继续扩大 loss 调参。若继续验证，需要增加 seed / repeat；更建议转向模型结构或后处理方案讨论。
## 第 7.18 步：后处理与阈值分析

### 目标

在不重新训练、不修改模型结构和不修改 `evaluate_pinn.py` 标准指标定义的前提下，分析第 7.17 步 v4 small polygon / area loss 候选模型的预测面积偏大问题，判断是否可以通过 mask threshold 或连通域后处理改善 area_error。

### 修改内容

本轮未修改训练代码、评价指标定义或数据生成器。使用离线分析流程加载第 7.17 推荐候选：

* checkpoint = `checkpoints/best_model_v4_w5_dice003_area004.pt`
* dataset = `data/training_data_v4_balanced_complex_test.npz`
* 原始训练配置 = defect_weight=5, lambda_dice=0.03, lambda_area=0.04, area_loss_type=symmetric

扫描 mask threshold：

* 300
* 350
* 400
* 450
* 500
* 550
* 600

并在最佳 threshold 上测试连通域过滤：

* 不处理
* remove components < 5 pixels
* remove components < 10 pixels
* remove components < 20 pixels

### 输出文件

* `results/metrics/v4_postprocess_threshold_sweep.csv`
* `results/metrics/v4_postprocess_component_filter.csv`
* `results/summaries/v4_postprocess_analysis_summary.txt`
* `results/previews/v4_postprocess_examples/`

### 关键指标 / 结果

标准 threshold=500：

* IoU = 0.351333
* Dice = 0.496197
* area_error = 0.911511
* pred_area > true_area = 191 / 200
* polygon area_error = 1.428304
* small polygon pred_area=0 = 0 / 25

area_error 最优 threshold=300：

* IoU = 0.337845
* Dice = 0.475548
* area_error = 0.292975
* pred_area > true_area = 114 / 200
* polygon area_error = 0.390191
* small polygon pred_area=0 = 0 / 25
* medium polygon area_error = 0.543884

IoU / Dice 最优 threshold 在本轮为 450：

* IoU = 0.354303
* Dice = 0.497498
* area_error = 0.685771

连通域过滤在 threshold=300 下基本没有额外收益：remove < 5 pixels 与不处理结果相同，remove < 10 / 20 pixels 只带来极小变化。

### 结论

降低 mask threshold 可以明显缓解 pred_area 系统性偏大，并显著降低 overall area_error、polygon area_error 和 medium polygon area_error。若本阶段以面积误差为主，threshold=300 是最佳后处理候选；若更重视 IoU / Dice，threshold=450 更接近最优。

small polygon 在 threshold=300 下仍保持 `pred_area=0 = 0 / 25`，没有重新漏检。连通域过滤不值得作为当前主要方案，因为预测区域主要不是由极小孤立连通域造成的。

本轮后处理可作为可选评估方案，但不应替换标准 `evaluate_pinn.py` 指标定义，也不足以替代模型结构改进。当前全项目 baseline 不切换，仍为 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt`。

### 下一步

建议进入第 7.19 步：模型结构优化方案设计。优先讨论多尺度坐标特征、边界表达能力、Bz encoder 表达能力，以及是否给 `train_pinn.py` 增加 `--seed` 参数和 repeat 实验机制。
---

## 第 7.18.5 步：训练随机种子 seed 支持

### 目标

在进入第 7.19 模型结构优化前，先给 `train_pinn.py` 增加 `--seed` 参数，降低后续结构对比实验中的训练随机性影响。

### 修改内容

* `train_pinn.py` 新增 `--seed` 参数，默认值为 `42`。
* 新增 `set_seed(seed)` 函数，设置：
  * `random.seed(seed)`
  * `numpy.random.seed(seed)`
  * `torch.manual_seed(seed)`
  * CUDA 可用时设置 `torch.cuda.manual_seed_all(seed)`
* Adam 训练的 `DataLoader(shuffle=True)` 使用固定 `torch.Generator()`，并执行 `generator.manual_seed(seed)`。
* Adam 训练和 L-BFGS refine 启动时打印 `Using random seed: 42`。

### 输出文件

本步不重新训练，不生成新 checkpoint，不生成新 metrics。

### 关键指标 / 结果

当时未单独记录具体指标。本步是可复现性基础设施修改。

### 结论

第 7.15 到第 7.17 的实验显示，相同配置可能出现训练随机性波动。因此从第 7.18.5 开始，训练默认固定 `seed=42`。后续第 7.19 模型结构优化实验必须固定 seed，并尽量在关键对比中做 repeat 验证。

第 7.12–7.17 的历史模型是在 `--seed` 参数加入之前训练的，因此即使后续使用相同超参数重新训练，指标也可能因 seed 固定方式不同而不完全一致。

第 7.18 后处理阈值分析还说明：模型预测的 μ 值存在校准偏软问题，缺陷区域常被预测为 μ≈200–400，而不是接近真实 μ≈1。因此 threshold=300 能显著降低 area_error。这是模型输出校准问题，不是单纯评估阈值问题。

### 下一步

进入第 7.19 步：模型结构优化方案设计。先制定方案，不直接大改代码。

## 第 7.19 步：模型结构优化方案设计

### 目标

在不修改模型代码、不重新训练的前提下，分析当前模型为什么存在 μ 值偏软、polygon area_error 偏大和 multi_defect 定位不稳定的问题，并制定第 7.20 的最小可行结构优化方案。

### 修改内容

* 新增 `MODEL_STRUCTURE_PLAN.md`；
* 更新 `NEXT_STEP.md`，把当前下一步改为第 7.20A 输出 μ 参数化校准实验；
* 更新 `PINN优化路线.md` 和 `README.md`，记录第 7.19 的结论和第 7.20A / 7.20B 拆分方向；
* 未修改 `train_pinn.py` 的模型结构；
* 未修改 `evaluate_pinn.py` 的评价指标定义；
* 未重新训练模型。

### 输出文件

* `MODEL_STRUCTURE_PLAN.md`

### 关键指标 / 结果

本步为方案设计，不产生新的训练指标。核心依据来自第 7.12-7.18 的已有结果：

* weighted MSE 和 soft Dice Loss 已能缓解 small polygon 漏检；
* area-aware loss 可以降低部分面积误差，但收益开始递减；
* 第 7.18 阈值分析显示 threshold=300 能显著降低 area_error，说明当前模型输出 μ 值偏软；
* 缺陷区域预测 μ 往往停留在 `μ_r≈200-400`，而不是接近真实缺陷 `μ_r≈1`。

### 结论

当前问题不应继续优先扩大 loss 调参。第一个结构优化主方案应拆成两步：第 7.20A 只做输出 μ 参数化校准，保持当前 decoder 结构不变；第 7.20B 只有在 7.20A 有效或部分有效后，再考虑轻量增强 decoder。

需要修正的结构事实已记录到 `MODEL_STRUCTURE_PLAN.md`：当前输出层不是无约束线性层，而是 `Linear + Softplus`。Softplus 有下界但无上界；缺陷端要逼近 `mu_norm≈0.001` 时需要很负的 pre-activation，可能导致输出偏软。

当前全项目 baseline 不变，仍以 `CURRENT_BASELINE.md` 为准。

### 下一步

进入第 7.20A 步：输出 μ 参数化校准实验。建议固定 `seed=42`，使用 v4_balanced_complex 数据集和当前 v4 small polygon 候选 loss 配置，做旧结构与 calibrated_mu 输出参数化的 A/B 对比；暂不增强 decoder，避免混淆变量。

---

## 第 7.20A 步：calibrated_mu 输出 μ 参数化校准实验

### 目标

验证只改变输出 μ 参数化、保持 BzEncoder 和 decoder 主体不变时，是否能缓解当前 v4 模型的 μ 值偏软和 `pred_area` 偏大问题。

### 修改内容

* `train_pinn.py` 新增 `--model-variant baseline / calibrated_mu`，默认仍为 `baseline`；
* `baseline` 保持旧输出行为：`Linear(64, 1) + Softplus`；
* `calibrated_mu` 使用同一 decoder 主体，只将最后 logit 经 `sigmoid` 得到 defect probability，再映射到 `mu_norm ∈ [0.001, 1.0]`；
* `evaluate_pinn.py` 增加对 checkpoint 中 `model_variant` 的加载兼容，并额外输出第 7.20A 所需的汇总诊断字段；未改变 MSE、MAE、IoU、Dice、area_error、center_error 的标准定义。

### 输出文件

* `checkpoints/best_model_v4_baseline_seed42_w5_dice003_area004.pt`
* `checkpoints/best_model_v4_calibrated_mu_seed42_w5_dice003_area004.pt`
* `results/metrics/v4_calibrated_mu_ablation.csv`
* `results/metrics/evaluation_metrics_v4_baseline_seed42_w5_dice003_area004.csv`
* `results/metrics/evaluation_metrics_v4_calibrated_mu.csv`
* `results/summaries/v4_calibrated_mu_ablation_summary.txt`
* `results/loss_curves/loss_curve_v4_baseline_seed42_w5_dice003_area004.png`
* `results/loss_curves/loss_curve_v4_calibrated_mu.png`
* `results/previews/reconstruction_preview_v4_baseline_seed42_w5_dice003_area004.png`
* `results/previews/reconstruction_preview_v4_calibrated_mu.png`

### 关键指标 / 结果

固定配置：`dataset=v4_balanced_complex`、`seed=42`、`loss_type=weighted_mse_dice_area`、`defect_weight=5`、`lambda_dice=0.03`、`lambda_area=0.04`、`area_loss_type=symmetric`、`lambda_tv=0`、`epochs=100`。

| model | MSE | MAE | IoU | Dice | area_error | center_error |
|---|---:|---:|---:|---:|---:|---:|
| baseline_seed42 | 3.18715908e+04 | 5.64035633e+01 | 3.39044536e-01 | 4.80770498e-01 | 6.40443541e-01 | 1.19076125e+00 |
| calibrated_mu_seed42 | 3.06620221e+04 | 5.87495580e+01 | 3.54232016e-01 | 4.96098795e-01 | 6.40109928e-01 | 1.13673565e+00 |

calibrated_mu 让缺陷区预测 μ_r 分布有所降低：`defect_mu_mean` 从约 399 降到约 361，`defect_mu_median` 从约 295 降到约 262；small polygon `pred_area=0` 保持 0 / 25，small polygon IoU / Dice 也有小幅提升。需要注意：small polygon `pred_area=0` 为 0 / 25 只说明没有完全空预测，仍需要关注 small polygon IoU=0 的样本；后续评估应同时记录 small polygon IoU=0 数量。

### 结论

`calibrated_mu` 方向有效但改善幅度有限。它改善了 IoU、Dice、center_error、polygon area_error、small polygon 指标和部分 μ 校准统计，但 area_error 几乎不变，`pred_area > true_area` 数量从 174 / 200 增加到 182 / 200，说明单独改变输出参数化还不能解决预测面积系统性偏大的问题。

当前不切换全项目 baseline，仍以 `CURRENT_BASELINE.md` 为准。

### 下一步

建议进入第 7.20B：在保留 `calibrated_mu` 可选变体和固定 `seed=42` 的前提下，做轻量 decoder 增强 A/B 实验，验证 decoder 表达能力是否能进一步改善 μ 校准和面积误差。

---

## 第 7.20B 步：calibrated_mu 轻量 decoder 增强实验

### 目标

在第 7.20A `calibrated_mu` 输出参数化基础上，只增强 decoder 容量，验证 decoder 表达能力是否限制 μ 校准和 area_error 改善。

### 修改内容

* `train_pinn.py` 新增 `--decoder-variant standard / enhanced`，默认仍为 `standard`；
* `standard` 保持旧 decoder：`128 / 128 / 64 + Tanh`；
* `enhanced` 使用 `256 / 256 / 128 / 64 + SiLU`；
* BzEncoder、Fourier feature、`calibrated_mu` 输出映射和 loss 配置均保持不变；
* `evaluate_pinn.py` 仅增加 checkpoint 中 `decoder_variant` 的加载兼容和 summary 字段，未改变标准指标定义。

### 输出文件

* `checkpoints/best_model_v4_calibrated_mu_enhanced_decoder_seed42_w5_dice003_area004.pt`
* `results/metrics/v4_calibrated_mu_decoder_ablation.csv`
* `results/metrics/evaluation_metrics_v4_calibrated_mu_standard_decoder.csv`
* `results/metrics/evaluation_metrics_v4_calibrated_mu_enhanced_decoder.csv`
* `results/summaries/v4_calibrated_mu_decoder_ablation_summary.txt`
* `results/loss_curves/loss_curve_v4_calibrated_mu_enhanced_decoder.png`
* `results/previews/reconstruction_preview_v4_calibrated_mu_enhanced_decoder.png`

说明：第 7.20B 对比中的 standard decoder 结果复用第 7.20A 的 `calibrated_mu + standard decoder + seed=42` checkpoint，不是本轮重新训练结果。

### 关键指标 / 结果

固定配置：`dataset=v4_balanced_complex`、`seed=42`、`model_variant=calibrated_mu`、`loss_type=weighted_mse_dice_area`、`defect_weight=5`、`lambda_dice=0.03`、`lambda_area=0.04`、`area_loss_type=symmetric`、`lambda_tv=0`、`epochs=100`。

| model | MSE | MAE | IoU | Dice | area_error | center_error |
|---|---:|---:|---:|---:|---:|---:|
| calibrated_mu_standard_seed42 | 3.06620221e+04 | 5.87495580e+01 | 3.54232016e-01 | 4.96098795e-01 | 6.40109928e-01 | 1.13673565e+00 |
| calibrated_mu_enhanced_seed42 | 3.26245823e+04 | 5.79398843e+01 | 3.53319625e-01 | 4.99793934e-01 | 9.58160563e-01 | 1.12645624e+00 |

enhanced decoder 的 μ 校准继续改善：`defect_mu_mean` 从约 361 降到约 333，`defect_mu_median` 从约 262 降到约 238。small polygon `pred_area=0` 保持 0 / 25，small polygon IoU=0 从 10 / 25 降到 7 / 25。

### 结论

enhanced decoder 说明 decoder 容量确实会影响 μ 校准，并能改善 Dice、small polygon 指标和 multi_defect center_error；但 area_error 明显恶化，polygon area_error 从 0.7938 升到 1.4199，pred_area > true_area 从 182 / 200 增加到 189 / 200。

因此本轮不切换全项目 baseline，也不把 enhanced decoder 作为默认 v4 方案。当前全项目推荐 baseline 仍以 `CURRENT_BASELINE.md` 为准。

### 下一步

建议先做 seed repeat 验证该 trade-off 是否稳定。若 repeat 后仍确认 enhanced decoder 会稳定加重面积高估，则应考虑更温和的 decoder 或专门的输出校准 / 面积校准策略，而不是继续单纯增大 decoder。

---

## 第 7.21 步：calibrated_mu decoder 多 seed 配对重复实验

### 目标

对 `calibrated_mu + standard decoder` 和 `calibrated_mu + enhanced decoder` 做 3 个 seed 的配对重复实验，验证第 7.20B 中 “μ 校准 / Dice 略改善但 area_error 恶化” 的 trade-off 是否稳定。本实验不用于更新 baseline。

### 修改内容

* 复用第 7.20A 的 standard seed=42 checkpoint；
* 复用第 7.20B 的 enhanced seed=42 checkpoint；
* 补跑 standard seed=123、standard seed=2026、enhanced seed=123、enhanced seed=2026；
* 使用同一 v4 test set 统一评估 6 组模型；
* 汇总单 seed 指标、mean ± std 和同 seed paired difference；
* 未修改 `CURRENT_BASELINE.md`。

### 输出文件

* `checkpoints/best_model_v4_calibrated_mu_standard_decoder_seed123_w5_dice003_area004.pt`
* `checkpoints/best_model_v4_calibrated_mu_standard_decoder_seed2026_w5_dice003_area004.pt`
* `checkpoints/best_model_v4_calibrated_mu_enhanced_decoder_seed123_w5_dice003_area004.pt`
* `checkpoints/best_model_v4_calibrated_mu_enhanced_decoder_seed2026_w5_dice003_area004.pt`
* `results/metrics/v4_calibrated_mu_decoder_seed_repeat.csv`
* `results/summaries/v4_calibrated_mu_decoder_seed_repeat_summary.txt`

### 关键指标 / 结果

固定配置：`dataset=v4_balanced_complex`、`model_variant=calibrated_mu`、`loss_type=weighted_mse_dice_area`、`defect_weight=5`、`lambda_dice=0.03`、`lambda_area=0.04`、`area_loss_type=symmetric`、`lambda_tv=0`、`epochs=100`。

| decoder_variant | MSE | MAE | IoU | Dice | area_error | center_error |
|---|---:|---:|---:|---:|---:|---:|
| standard mean ± std | 3.177953e+04 ± 9.684922e+02 | 64.071944 ± 5.078808 | 0.348739 ± 0.004885 | 0.491443 ± 0.004296 | 0.829989 ± 0.164446 | 1.129777 ± 0.009350 |
| enhanced mean ± std | 3.242061e+04 ± 7.778385e+02 | 57.576270 ± 1.424945 | 0.354685 ± 0.002091 | 0.500730 ± 0.002992 | 0.953397 ± 0.007471 | 1.125450 ± 0.004811 |

paired difference mean（enhanced - standard）：

* ΔMSE = +641.074535
* ΔMAE = -6.495674
* ΔIoU = +0.005946
* ΔDice = +0.009287
* Δarea_error = +0.123408
* Δcenter_error = -0.004326

### 结论

enhanced decoder 的 trade-off 在多 seed 下稳定存在：MAE 稳定改善，IoU / Dice 在多 seed 均值和 paired mean 上小幅稳定改善，small polygon IoU=0 数量下降，small `pred_area=0` 问题也改善。

但 enhanced decoder 的 area_error 稳定恶化，`pred_area>true_area` 数量增加或保持更高水平，说明 enhanced decoder 存在更明显的面积高估问题。

因此第 7.21 不切换 `CURRENT_BASELINE`。如果 enhanced 继续表现为 area_error 恶化，后续不应继续单纯加宽 decoder。

### 下一步

后续是否进入 post-processing / area calibration / threshold calibration，由主线对话决定。

---

## 第 7.22 步：calibrated_mu decoder threshold calibration

### 目标

对第 7.21 的 6 个 calibrated_mu standard / enhanced decoder checkpoint 做 evaluation-level mask threshold sweep，判断 enhanced decoder 的面积高估问题是否能通过后处理阈值校准缓解。本实验不重新训练、不修改模型结构、不修改 `CURRENT_BASELINE`。

### 修改内容

* 检查 `evaluate_pinn.py`，确认已支持 `--mask-threshold`，默认仍为 500.0；
* 使用 validation set 扫描 raw μ_r threshold：300 / 350 / 400 / 450 / 500 / 550 / 600 / 650 / 700；
* 只根据 validation set 选择 recommended threshold；
* 使用推荐 threshold 在 test set 上验证；
* 未修改 `train_pinn.py`、`evaluate_pinn.py` 或 `CURRENT_BASELINE.md`。

### 输出文件

* `results/metrics/v4_calibrated_mu_threshold_validation_sweep.csv`
* `results/metrics/v4_calibrated_mu_threshold_test_comparison.csv`
* `results/summaries/v4_calibrated_mu_threshold_calibration_summary.txt`

### 关键指标 / 结果

validation set 推荐 threshold：

* standard decoder：400
* enhanced decoder：350

test set mean 对比：

| decoder | threshold | MSE | MAE | IoU | Dice | area_error | center_error | small IoU=0 | small pred_area=0 | pred_area>true_area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | 500 | 3.177953e+04 | 64.071944 | 0.348739 | 0.491443 | 0.829989 | 1.129777 | 11.000000 | 1.333333 | 182.333333 |
| standard | 400 | 3.177953e+04 | 64.071944 | 0.350794 | 0.488918 | 0.513358 | 1.139972 | 14.000000 | 5.333333 | 151.000000 |
| enhanced | 500 | 3.242061e+04 | 57.576270 | 0.354685 | 0.500730 | 0.953397 | 1.125450 | 7.000000 | 0.000000 | 190.000000 |
| enhanced | 350 | 3.242061e+04 | 57.576270 | 0.355723 | 0.496510 | 0.416337 | 1.146558 | 9.666667 | 0.333333 | 146.666667 |

### 结论

threshold calibration 明显缓解面积高估。standard decoder 的 area_error 从 0.829989 降到 0.513358；enhanced decoder 的 area_error 从 0.953397 降到 0.416337，且 `pred_area>true_area` 从 190.0 降到 146.67。

overall IoU / Dice 没有明显牺牲：standard IoU 略升、Dice 小幅下降；enhanced IoU 略升、Dice 小幅下降。但 small polygon IoU=0 和 small `pred_area=0` 有轻微恶化，需要后续继续监控。

本轮不切换 `CURRENT_BASELINE`。如果后续接受 evaluation-level calibration，enhanced decoder 可以作为继续做 area calibration / threshold calibration / post-processing 的候选；否则 enhanced decoder 仍只保留为结构消融记录。

### 下一步

后续是否进入 fixed evaluation threshold、area calibration 或 post-processing，由主线对话决定。

---

## 第 7.23 步：calibrated_mu adaptive threshold calibration

### 目标

在第 7.22 global threshold calibration 基础上，测试 evaluation-level adaptive threshold rule，判断是否能在降低 area_error 的同时减少 global threshold 对 small polygon 的副作用。本实验不重新训练、不修改模型结构、不修改 `CURRENT_BASELINE`。

### 修改内容

* 使用第 7.21 / 7.22 已有的 6 个 calibrated_mu checkpoint；
* 不修改 `train_pinn.py` 和 `evaluate_pinn.py`；
* 在 validation set 上搜索 adaptive threshold rule；
* adaptive rule 只使用 default threshold=500 下的 predicted area，不使用 true area；
* 使用 test set 对 default、global calibrated、adaptive 三种方法做最终验证。

### 输出文件

* `results/metrics/v4_calibrated_mu_adaptive_threshold_validation.csv`
* `results/metrics/v4_calibrated_mu_adaptive_threshold_test.csv`
* `results/summaries/v4_calibrated_mu_adaptive_threshold_summary.txt`

### 关键指标 / 结果

validation set 选出的 adaptive rule：

| decoder | A | B | T_small | T_medium | T_large |
|---|---:|---:|---:|---:|---:|
| standard | 9.654345 | 12.387713 | 450 | 350 | 350 |
| enhanced | 9.897988 | 15.232851 | 350 | 350 | 300 |

test set mean 对比：

| decoder | method | MSE | MAE | IoU | Dice | area_error | center_error | small IoU=0 | small pred_area=0 | pred_area>true_area |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | default | 3.177953e+04 | 64.071944 | 0.348739 | 0.491443 | 0.829989 | 1.129777 | 11.000000 | 1.333333 | 182.333333 |
| standard | global | 3.177953e+04 | 64.071944 | 0.350794 | 0.488918 | 0.513358 | 1.139972 | 14.000000 | 5.333333 | 151.000000 |
| standard | adaptive | 3.177953e+04 | 64.071944 | 0.347960 | 0.485609 | 0.474476 | 1.157978 | 12.666667 | 3.000000 | 143.000000 |
| enhanced | default | 3.242061e+04 | 57.576270 | 0.354685 | 0.500730 | 0.953397 | 1.125450 | 7.000000 | 0.000000 | 190.000000 |
| enhanced | global | 3.242061e+04 | 57.576270 | 0.355723 | 0.496510 | 0.416337 | 1.146558 | 9.666667 | 0.333333 | 146.666667 |
| enhanced | adaptive | 3.242061e+04 | 57.576270 | 0.350185 | 0.490040 | 0.360101 | 1.155400 | 9.666667 | 0.333333 | 127.333333 |

### 结论

adaptive threshold 能继续降低 area_error：standard 从 default 0.829989 降到 0.474476，enhanced 从 default 0.953397 降到 0.360101。相比 global threshold，adaptive 的 area_error 也更低。

代价是 overall IoU / Dice 相比 global threshold 更低。small polygon 保护方面，standard adaptive 比 standard global 更少伤害 small polygon，但 enhanced adaptive 与 enhanced global 基本相同，没有额外改善 small polygon。

本轮不更新 `CURRENT_BASELINE`。adaptive threshold 作为 evaluation-level calibration 记录；是否作为后续候选需要继续由主线对话决定。

### 下一步

后续是否将 adaptive threshold 作为评估层 calibration 候选，由主线对话决定。

---

## 第 7.24 步：calibrated_mu decoder + threshold calibration 阶段性 consolidation

### 目标

对第 7.20B、7.21、7.22、7.23 的 calibrated_mu decoder 与 threshold calibration 结果做阶段性归纳，明确这条线当前得到的事实结论。本阶段只做文档收尾，不训练、不评估、不修改 `CURRENT_BASELINE`。

### 修改内容

* 汇总第 7.20B enhanced decoder 单 seed 消融结论；
* 汇总第 7.21 standard vs enhanced 多 seed 配对结论；
* 汇总第 7.22 global threshold calibration 结论；
* 汇总第 7.23 adaptive threshold calibration 结论；
* 明确本阶段不更新 `CURRENT_BASELINE`；
* 明确停止继续做更多 threshold trick。

### 输出文件

* `EXPERIMENT_LOG.md`
* `PINN优化路线.md`
* `NEXT_STEP.md`

### 关键指标 / 结果

第 7.20B 单 seed 消融显示，enhanced decoder 能降低缺陷区预测 μ_r，改善 MAE、Dice、center_error、small polygon IoU/Dice 和 multi_defect center_error，但会明显放大面积高估：area_error 从 0.6401 升到 0.9582，`pred_area>true_area` 从 182 / 200 增加到 189 / 200。

第 7.21 多 seed 配对确认该 trade-off 稳定存在：enhanced mean MAE = 57.576270，standard mean MAE = 64.071944；enhanced mean Dice = 0.500730，standard mean Dice = 0.491443；但 enhanced mean area_error = 0.953397，高于 standard 的 0.829989。

第 7.22 global threshold calibration 显著降低 area_error：standard 从 0.829989 降到 0.513358，enhanced 从 0.953397 降到 0.416337；IoU 基本持平或略升，Dice 略降，同时 small polygon IoU=0 和 small `pred_area=0` 风险上升。

第 7.23 adaptive threshold calibration 继续降低 area_error：standard adaptive = 0.474476，enhanced adaptive = 0.360101；但相比 global threshold，adaptive 的 IoU / Dice 进一步下降。standard adaptive 对 small polygon 的副作用小于 standard global，enhanced adaptive 对 small polygon 没有额外改善。

### 结论

enhanced decoder 是有价值的结构消融：它稳定改善 MAE、IoU/Dice、小 polygon 漏检和缺陷区 μ 校准。但 enhanced decoder 在 default threshold 下稳定带来更高 MSE、更严重 area_error 和面积高估，因此不进入 baseline。

threshold calibration 是有价值的 evaluation-level calibration：它能显著降低 area_error 和 `pred_area>true_area`。代价是 Dice 下降，并增加 small polygon IoU=0 / small `pred_area=0` 风险。adaptive threshold 的 area_error 更低，但不全面优于 global threshold，因为 IoU / Dice 进一步下降。

当前没有结果足以更新 `CURRENT_BASELINE`。本阶段停止继续做更多 threshold trick。

### 下一步

进入下一阶段前，先由主线对话重新定义实验包、接受条件和停止条件，不再从当前副作用继续追加 threshold trick。

---

## 第 8.4 步：auxiliary mask head 阶段收口

### 目标

对第 8.2 standard decoder + aux mask loss 和第 8.3 enhanced decoder + aux mask loss 做阶段性收口，明确 auxiliary mask head 方向当前得到的结论。本阶段只做文档更新，不训练、不评估、不修改 `CURRENT_BASELINE`。

### 修改内容

* 汇总第 8.2 standard decoder + auxiliary mask head 正式小实验结果；
* 汇总第 8.3 enhanced decoder + auxiliary mask loss 实验结果；
* 明确 aux_mask_head 直接作为最终 mask 输出不作为主线方向；
* 明确 aux mask loss 作为 regularizer 对 standard decoder 有正信号；
* 明确 enhanced + aux mask loss 不满足接受条件，停止该方向；
* 明确不继续调 mask_pred threshold、不继续调 lambda_mask、不继续 enhanced aux；
* 未修改 `CURRENT_BASELINE.md`。

### 输出文件

* `EXPERIMENT_LOG.md`
* `PINN优化路线.md`
* `NEXT_STEP.md`

### 关键指标 / 结果

第 8.2 standard decoder + aux mask loss 中，`aux_mu_threshold` 相比 `baseline_mu_threshold` 的 mean 指标整体更好：MSE 从 3.177953e+04 降到 3.089576e+04，MAE 从 64.071944 降到 60.114897，IoU 从 0.348739 升到 0.353377，Dice 从 0.491443 升到 0.494331，area_error 从 0.829989 降到 0.719793，small IoU=0 从 11.000000 降到 9.666667。

但第 8.2 的 `aux_mask_head` 直接输出 mask 在固定 threshold=0.5 下不够好：mean IoU/Dice 低于 baseline，且 small pred_area=0 没有改善。因此 aux_mask_head 不作为最终 mask 输出主线。

第 8.3 enhanced decoder + aux mask loss 中，`enhanced_aux_mu_threshold` 相比 `enhanced_baseline_mu_threshold` 改善了 MSE、IoU/Dice 和 small IoU=0：MSE 从 3.242061e+04 降到 3.114191e+04，IoU 从 0.354685 升到 0.358267，Dice 从 0.500730 升到 0.503920，small IoU=0 从 7.000000 降到 5.333333。

但第 8.3 没有降低关键失败模式：area_error 从 0.953397 升到 0.982450，enhanced decoder 仍然严重面积高估。因此 enhanced + aux mask loss 不满足接受条件，触发停止条件。

### 结论

aux mask loss 作为训练 regularizer 对 standard decoder 有正信号，值得作为 v4 内部候选记录；但 aux_mask_head 直接作为最终 mask 输出不够好，不作为主线方向。

enhanced + aux mask loss 未能解决 enhanced decoder 的面积高估问题，因此停止该方向。当前不继续调 mask_pred threshold，不继续调 lambda_mask，不继续 enhanced aux，也不更新 `CURRENT_BASELINE`。

### 下一步

进入下一阶段前，由主线对话重新定义新的实验包、接受条件和停止条件，不从第 8.3 的副作用继续修补。

---

## 第 9.1 / 9.2 步：baseline transfer gates

### 目标

快速判断 v4 阶段中有效或有信号的 aux mask regularizer 与 shape-aware loss，是否能迁移到当前 v3_complex baseline。本轮只做 seed=42 fast gate，不扩展 3 seed，不更新 `CURRENT_BASELINE`。

### 关键结果 / 结论

第 9.1 aux mask regularizer transfer gate 失败：除 center_error 改善外，MSE、MAE、IoU、Dice、area_error 均比 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt` 变差。

第 9.2 shape-aware loss transfer gate 失败：IoU、Dice、center_error 改善，但 MSE、MAE、area_error 明显变差，area_error 从 0.3945 升到 0.9167。

两者都不能挑战 `CURRENT_BASELINE`。当前 baseline 不变，不扩展 3 seed，不继续调 `lambda_mask`、`lambda_dice`、`lambda_area`，也不继续把 v4 的 loss / aux / threshold 技巧硬迁移到 v3_complex baseline。

v4 / aux / threshold / shape-aware transfer 线到此停止。下一阶段转向 CURRENT_BASELINE failure-driven analysis。

### 输出文件

* `results/summaries/v3_baseline_transfer_gates_summary.txt`

---

## 第 10.1 / 10.2 / 10.3 步：CURRENT_BASELINE failure audit 与 threshold calibration

### 目标

分析当前 `CURRENT_BASELINE` 在 v3_complex test set 上的主要失败模式，并用 validation set 选择 evaluation-level mask threshold，判断面积低估是否可以被稳定缓解。本阶段不训练、不修改模型、不更新 `CURRENT_BASELINE`。

### 修改内容

* 第 10.1 对 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt` 做 per-sample failure audit；
* 第 10.2 在 test set 上做 threshold sensitivity 诊断；
* 第 10.3 在 validation set 上选择 threshold，并只在最后用 test set 验证；
* 明确 threshold calibration 只作为 evaluation-level calibration 记录，不更新 baseline；
* 明确不继续做更多 threshold trick。

### 输出文件

* `results/metrics/v3_current_baseline_failure_audit.csv`
* `results/summaries/v3_current_baseline_failure_audit_summary.txt`
* `results/metrics/v3_current_baseline_threshold_sensitivity.csv`
* `results/summaries/v3_current_baseline_threshold_sensitivity_summary.txt`
* `results/metrics/v3_current_baseline_val_selected_threshold.csv`
* `results/summaries/v3_current_baseline_val_selected_threshold_summary.txt`

### 关键指标 / 结果

第 10.1 failure audit 显示 `CURRENT_BASELINE` 的主要失败模式是系统性面积低估：`pred_area < true_area` 为 158 / 200，`pred_area=0` 为 18。small / medium / large 三个面积分桶中，平均 `pred_area` 都小于 `true_area`。最差样本主要集中在 polygon、rotated_rect、multi_defect。

第 10.2 threshold sensitivity 显示，提高 mask threshold 可以明显缓解面积低估。`threshold=550` 在 test set 上改善 IoU、Dice、area_error、center_error，并减少 `pred_area=0`。但第 10.2 是 test-set 诊断，不能作为正式选阈值依据。

第 10.3 validation-selected threshold calibration 使用 validation set 选出 `threshold=600`。在 test set 上，`threshold=600` 相比默认 `threshold=500`：

* IoU 从 0.2953 提升到 0.3528；
* Dice 从 0.4219 提升到 0.4952；
* center_error 从 1.3259 降到 1.2031；
* `pred_area=0` 从 18 降到 9；
* `pred_area < true_area` 从 158 降到 80；
* area_error 从 0.3945 小幅升到 0.4158；
* `pred_area > true_area` 从 42 增加到 120。

### 结论

`threshold=600` 能稳定缓解 CURRENT_BASELINE 的面积低估和空预测问题，并改善 IoU、Dice、center_error；但它也引入更多面积高估，使 test set area_error 小幅变差。

因此 `threshold=600` 只作为 validation-selected evaluation-level calibration 记录，不更新 `CURRENT_BASELINE`。本阶段不继续做更多 threshold trick。

---

## 第 10.7 / 10.8 / 10.9 步：CURRENT_BASELINE prediction / oversampling / signal difficulty audit

### 目标

继续围绕 `CURRENT_BASELINE` 的失败模式做轻量诊断，不训练新主线模型，不修改模型结构，不更新 `CURRENT_BASELINE`。本阶段重点确认面积低估是否来自预测 μ 分布、small defect 数据分布或 Bz 输入信号可辨识性。

### 修改内容

* 第 10.7：分析 CURRENT_BASELINE 在 test set 上的 `pred_mu` 分布；
* 第 10.8：使用 v3_complex train set 构造 small-defect oversampling gate，只做 seed=42 快速验证；
* 第 10.9：分析 Bz signal 强度与 IoU、Dice、`pred_area=0` 的关系；
* 更新 `EXPERIMENT_LOG.md`；
* 补充 `术语说明.md` 中与 gate / audit / threshold / area 相关的术语解释。

### 输出文件

* `results/metrics/v3_current_baseline_prediction_distribution_audit.csv`
* `results/summaries/v3_current_baseline_prediction_distribution_audit_summary.txt`
* `results/metrics/evaluation_metrics_v3_complex_small_os3_seed42.csv`
* `results/metrics/evaluation_metrics_v3_complex_small_os3_seed42.txt`
* `results/metrics/evaluation_metrics_v3_complex_small_os3_seed42_summary.csv`
* `results/summaries/v3_complex_small_os3_gate_summary.txt`
* `results/metrics/v3_current_baseline_signal_difficulty_audit.csv`
* `results/summaries/v3_current_baseline_signal_difficulty_audit_summary.txt`

### 关键指标 / 结果

第 10.7 prediction distribution audit 显示，真实缺陷像素中只有 40.85% 的 `pred_mu < 500`，16.20% 落在 500-600，42.95% 大于等于 600。small defect 更差，`pred_mu < 500` 比例只有 29.84%，`pred_mu >= 600` 比例达到 50.95%。在 `pred_area=0` 样本中，真实缺陷像素的 median `pred_mu` 为 746.56，90.36% 大于等于 600。

第 10.8 small-defect oversampling gate 失败。临时训练集将 train set 中 true_area 最低 1/3 的 small 样本复制到约 3x 出现次数，训练集从 1000 扩展到 1666。seed=42 结果显示 small pred_area=0 从 15 增加到 21，small IoU 从 0.2022 降到 0.1425，small Dice 从 0.2930 降到 0.2112；overall IoU 从 0.2953 降到 0.2496，Dice 从 0.4219 降到 0.3638，area_error 从 0.3945 升到 0.5114，`pred_area < true_area` 从 158 增加到 192。因此不扩展 3 seed，不继续调 oversampling ratio。

第 10.9 signal difficulty audit 显示，small 样本的平均 Bz 信号强度明显低于 large 样本：small 的 mean `max_abs_bz` 为 6.96，large 为 14.66；small 的 mean `l2_energy_bz` 为 1615.70，large 为 8007.20。`pred_area=0` 样本的 Bz 信号也明显弱于非空预测样本：mean `max_abs_bz` 为 4.06 vs 10.96，mean `l2_energy_bz` 为 697.92 vs 4554.65。

信号强度与性能存在可见但不决定性的相关性。`peak_to_peak_bz` 与 IoU 的相关系数约为 0.593，与 Dice 约为 0.599；`max_abs_bz` 与 `pred_area=0` 的相关系数约为 -0.399。最差 IoU 样本多数是 small defect，且通常具有弱到中等强度的 Bz 信号。

### 结论

CURRENT_BASELINE 的 small defect 失败既与 Bz 信号较弱、可辨识性较低有关，也反映出模型对弱信号的利用和输出校准不足；它不是单纯由阈值选择造成，也不是简单 small oversampling 可以解决。

第 10.8 证明 small-defect oversampling gate 没有改善 small defect，反而加重面积低估。第 10.7 / 10.9 说明继续做 loss trick、threshold trick 或 small oversampling ratio trick 都缺乏直接证据支撑。

`CURRENT_BASELINE` 保持不变。

### 下一步

不继续 loss trick、threshold trick、small oversampling ratio trick。后续实验应围绕 CURRENT_BASELINE 的真实失败样本和弱 Bz 信号可辨识性重新定义实验包、接受条件和停止条件。

---

## 第 11.2 / 11.3 步：v3_complex data observability audit 与 Bz input feature augmentation gate

### 目标

第 11.2 只分析 v3_complex 数据本身的可观测性，判断 small / low-signal 样本是否天然更难；第 11.3 在不改 loss、decoder、threshold、oversampling 和 evaluate_pinn.py 的前提下，快速测试 `raw_plus_norm_stats` Bz 输入增强是否能改善 weak Bz signal 的利用。

### 修改内容

* 第 11.2：读取 v3_complex train / val / test，统计 true_area、Bz signal 强度、Bz peak x、defect centroid x 和 peak-centroid offset；
* 第 11.2：复用 CURRENT_BASELINE failure audit，把 test 失败样本与信号强度关联；
* 第 11.3：临时在 `train_pinn.py` 中加入 `--bz-feature-mode raw_plus_norm_stats`，训练 seed=42 gate；
* 第 11.3 gate 失败后已恢复 `train_pinn.py`，不保留失败代码分支；
* 更新 `EXPERIMENT_LOG.md` 和 `术语说明.md`。

### 输出文件

* `results/metrics/v3_complex_data_observability_audit.csv`
* `results/summaries/v3_complex_data_observability_audit_summary.txt`
* `results/summaries/v3_complex_bz_feature_aug_gate_summary.txt`
* `results/metrics/evaluation_metrics_v3_complex_bz_feature_aug_seed42.csv`
* `results/metrics/evaluation_metrics_v3_complex_bz_feature_aug_seed42.txt`
* `results/metrics/evaluation_metrics_v3_complex_bz_feature_aug_seed42_summary.csv`

### 关键指标 / 结果

第 11.2 data observability audit 显示，v3_complex 的 train / val / test small 比例接近，分别为 33.40%、34.00%、33.50%，说明 small 失败不是由 split 分布异常造成。

按 train-based true_area 分桶统计，small 样本的 Bz 信号明显弱于 large：small mean `max_abs_bz = 6.8745`、`l2_energy_bz = 1605.81`，large mean `max_abs_bz = 14.6291`、`l2_energy_bz = 8452.48`。low-signal 样本主要集中在 polygon 和 rotated_rect：polygon 占 low-signal 的 53.02%，rotated_rect 占 24.14%，multi_defect 占 22.84%。

Bz peak x 与 defect centroid x 不是可靠一一对应。small 样本 mean `peak_centroid_dx_abs = 1.5186 mm`，其中 `>1 mm` 为 409 / 469，`>2 mm` 为 53 / 469。multi_defect 中最强 Bz peak 也可能与几何中心分离。

CURRENT_BASELINE 的失败样本与低信号相关：`pred_area=0` 样本 mean `max_abs_bz = 4.0629`，非空预测样本为 10.9604；`IoU < 0.1` 样本 mean `max_abs_bz = 5.7437`，`IoU >= 0.1` 样本为 11.4886。test set 中 `max_abs_bz` 与 IoU 的相关系数约 0.5866，`peak_to_peak_bz` 与 IoU 约 0.5935；peak-centroid offset 与 IoU 只有弱负相关，约 -0.1364。

第 11.3 `raw_plus_norm_stats` gate 失败。相比 CURRENT_BASELINE：

* overall IoU 从 0.2953 降到 0.2787；
* overall Dice 从 0.4219 降到 0.4009；
* area_error 从 0.3945 升到 0.4269；
* pred_area=0 从 18 增加到 21；
* small IoU 从 0.2022 降到 0.1687，small Dice 从 0.2930 降到 0.2487；
* low-signal IoU 从 0.1607 降到 0.1477，low-signal Dice 从 0.2461 降到 0.2277；
* 仅 MSE 和 center_error 有轻微改善，不构成综合改善。

### 结论

v3_complex 的 small defect 失败具有明显数据可观测性因素：small / low-signal 样本确实 Bz 信号更弱，且 Bz peak x 不稳定等价于 defect centroid x。但该问题不是纯粹不可辨识，模型训练与表示仍有影响。

第 11.3 表明，简单地把 per-sample normalized Bz 和少量 signal stats 拼入输入并不能改善 small / low-signal 样本，反而降低 IoU / Dice 并增加空预测。因此不扩展 3 seed，不继续调输入特征，不保留本次 `raw_plus_norm_stats` 代码分支。

`CURRENT_BASELINE` 保持不变。

### 下一步

不继续 loss trick、threshold trick、small oversampling、CNN1D encoder 或简单 Bz input feature augmentation。后续需要重新定义新的实验包、接受条件和停止条件。

---

## 第 11.4 / 11.5 步：multi-liftoff Bz observation gate 与 3 seed expansion

### 目标

验证在 v3_complex 数据上增加 multi-liftoff Bz 观测是否能稳定改善 weak Bz / small defect 样本的可辨识性。本阶段不更新 `CURRENT_BASELINE`。

### 修改内容

* 第 11.4：在 `data_generator_v2.py` 中新增 `v3_complex_multiliftoff` 数据生成路径，同一样本输出两个 lift-off Bz 通道；
* 第 11.4：在 `train_pinn.py` 中加入多通道 Bz 输入兼容，默认单通道路径保持不变；
* 第 11.4：在 `evaluate_pinn.py` 中加入 multi-channel checkpoint 的 `signal_channels` 推断；
* 第 11.4：完成 seed=42 fair single-liftoff vs multi-liftoff gate；
* 第 11.5：复用第 11.4 的 seed=42 checkpoint，补跑 seed=123 和 seed=2026，完成 3 seed 配对实验；
* 第 11.5 不修改代码，不更新 `CURRENT_BASELINE`。

### 输出文件

* `results/summaries/v3_complex_multiliftoff_gate_summary.txt`
* `results/metrics/evaluation_metrics_v3_complex_multiliftoff_seed42.csv`
* `results/metrics/evaluation_metrics_v3_complex_multiliftoff_seed42.txt`
* `results/metrics/evaluation_metrics_v3_complex_multiliftoff_seed42_summary.csv`
* `results/summaries/v3_complex_multiliftoff_3seed_summary.txt`
* `results/metrics/v3_complex_multiliftoff_3seed_metrics.csv`

### 关键指标 / 结果

第 11.4 seed=42 gate 中，multi-liftoff 相比 fair single-liftoff 有明显正信号：

* overall IoU 从 0.2883 提升到 0.3179；
* overall Dice 从 0.4183 提升到 0.4530；
* area_error 从 0.4243 降到 0.3538；
* center_error 从 1.3390 降到 1.2772；
* `pred_area=0` 从 19 降到 10；
* small IoU 从 0.1831 提升到 0.2430；
* low-signal IoU 从 0.1426 提升到 0.2066。

第 11.5 三 seed 配对实验显示该正信号不稳定。3 seed mean 对比中，multi-liftoff 相比 fair single-liftoff：

* mean IoU 从 0.2932 降到 0.2825；
* mean Dice 从 0.4233 降到 0.4099；
* mean area_error 从 0.4139 升到 0.4398；
* mean center_error 从 1.2823 升到 1.3336；
* mean `pred_area=0` 从 18.67 小幅降到 17.33。

small / low-signal 样本也没有稳定改善：

* small mean IoU 基本持平，0.2000 vs 0.2000；
* small mean Dice 仅轻微变化，0.2926 vs 0.2939；
* low-signal mean IoU 从 0.1566 降到 0.1528；
* low-signal mean Dice 从 0.2383 降到 0.2313。

paired difference 显示，multi-liftoff 的提升主要来自 seed=42，seed=123 和 seed=2026 未复现。

### 结论

第 11.4 的 seed=42 正信号没有在三 seed 配对实验中稳定复现。multi-liftoff 相比 fair single-liftoff 的 3 seed mean 在 IoU、Dice、area_error、center_error 上均变差，`pred_area=0` 仅小幅减少，不足以抵消整体指标下降。

因此 multi-liftoff 不进入正式主线候选，不继续扩展 seed，不继续调 multi-liftoff 结构。`CURRENT_BASELINE` 保持不变。

### 下一步

不继续小 gate 或 multi-liftoff 修补。下一阶段转向阶段性总结、当前 baseline 结果整理和论文材料准备，除非重新定义更大的实验包、接受条件和停止条件。
## 第 12.6 / 12.7 步：overfit30 capacity diagnostic 与 longer-training gate

### 目标

判断 CURRENT_BASELINE 架构是否具备小样本拟合能力，并检查 v3_complex 全量训练是否主要受 epoch 不足限制。

### 修改内容

第 12.6 从 v3_complex train 中按 true_area 三分位选取 10 small、10 medium、10 large，共 30 个样本，构造 overfit30 临时数据集；第 12.7 在不改结构、不改 loss、不改数据的前提下，将 v3_complex 训练预算扩展到 200 epoch。

### 输出文件

* `results/summaries/v3_complex_overfit30_capacity_diagnostic_summary.txt`
* `results/metrics/v3_complex_overfit30_capacity_diagnostic_metrics.csv`
* `results/summaries/v3_complex_longer_training_gate_summary.txt`
* `results/metrics/v3_complex_longer_training_gate_metrics.csv`

### 关键指标 / 结果

第 12.6 中，CURRENT_BASELINE 在 overfit30 同一批样本上 IoU=0.3250、Dice=0.4608、area_error=0.3606；overfit30 模型达到 IoU=0.8821、Dice=0.9364、area_error=0.0394，且 pred_area=0 从 2 降到 0。small / medium / large 三个分桶均显著改善，说明当前架构具备小样本过拟合能力。

第 12.7 中，200 epoch 训练的 best checkpoint 实际来自 epoch 17，best_val_loss=2.22840904e-02。相比 CURRENT_BASELINE，200 epoch best checkpoint 在 train / val / test 上 IoU、Dice 均下降，MAE 与 area_error 变差，pred_area=0 和 pred_area<true_area 增加，面积低估加重。

### 结论

CURRENT_BASELINE 架构不是完全没有表达能力；问题更像全量训练、样本复杂度、优化目标或模型选择准则导致的系统性困难。单纯增加 epoch 没有改善，反而在 best-val checkpoint 下选到更保守的早期模型，因此不扩展 3 seed，也不继续盲目加 epoch。

### 下一步

不在本记录中提出新实验方案；CURRENT_BASELINE 不变。

---
## 第 12.8 / 12.9 步：overfit100 capacity diagnostic 与 warm-start gate

### 目标

判断 CURRENT_BASELINE 架构的过拟合能力是否能从 30 个样本扩展到 100 个样本，并测试 overfit100 学到的形状表达是否能通过 warm-start 迁移到 full v3_complex 训练。

### 修改内容

第 12.8 从 `data/training_data_v3_complex_train.npz` 中按 true_area 三分位选取 34 small、33 medium、33 large，共 100 个样本，构造临时 `v3_complex_overfit100` 数据集，train / val / test 均使用同一批样本。

第 12.9 使用 `checkpoints/best_model_v3_complex_overfit100_seed42.pt` 作为初始化，在 full v3_complex 上继续训练 50 epoch。未修改 `train_pinn.py`、`evaluate_pinn.py`、`data_generator_v2.py`，未更新 `CURRENT_BASELINE`。

### 输出文件

* `results/summaries/v3_complex_overfit100_capacity_diagnostic_summary.txt`
* `results/metrics/v3_complex_overfit100_capacity_diagnostic_metrics.csv`
* `results/summaries/v3_complex_overfit100_warmstart_gate_summary.txt`
* `results/metrics/v3_complex_overfit100_warmstart_gate_metrics.csv`

### 关键指标 / 结果

第 12.8 overfit100 diagnostic 显示当前模型具备 100 样本过拟合能力。CURRENT_BASELINE 在这 100 个样本上 IoU=0.3104、Dice=0.4347、area_error=0.4108、pred_area=0 为 9；overfit100 模型达到 IoU=0.8426、Dice=0.9080、area_error=0.0958、pred_area=0 为 0。small / medium / large 三个分桶均明显优于 baseline。

第 12.9 中，overfit100 checkpoint 直接在 full val/test 上泛化较差。warm-start full training 改善了 train MSE/MAE，但 val/test IoU、Dice、area_error、center_error 均未超过 CURRENT_BASELINE；small / medium 分桶没有稳定改善。

### 结论

当前架构不是完全没有表达能力，至少可以在 100 个混合面积样本上强力过拟合。但 overfit100 学到的表示不能直接泛化到 full v3_complex，也不能通过 50 epoch warm-start 转化为优于 CURRENT_BASELINE 的 full-data 模型。

因此 CURRENT_BASELINE 不变；不扩展 3 seed；不继续 curriculum / warm-start 方向。

### 下一步

不在本记录中提出新实验方案；CURRENT_BASELINE 不变。

---

## 第 13.4 / 13.5 步：v3_complex composite-selection candidate 与逐样本审计

### 目标

验证 `train_pinn.py` 正式集成的 `--selection-metric composite` 是否能作为 shape-oriented baseline 的模型选择标准，并通过逐样本审计确认改善是否来自多数样本，而不是少数 outlier。

### 修改内容

第 13.4 使用正式训练流程训练 v3_complex composite-selection 3 seed candidate。模型结构和 loss 均不变，仍使用默认 MSE + `lambda_tv=2e-6`；仅将 checkpoint selection 从默认 `mse` 改为 `composite`。

第 13.5 对旧 v3_complex MSE-oriented baseline 和 composite-selection candidate 做 test set 逐样本对比，统计 IoU / Dice / area_error 改善样本数量、`pred_area=0` 修复情况、area bin 和 defect_type 分组改善情况。

### 输出文件

* `results/summaries/v3_complex_composite_selection_candidate_summary.txt`
* `results/metrics/v3_complex_composite_selection_candidate_metrics.csv`
* `results/summaries/v3_complex_composite_candidate_per_sample_audit_summary.txt`
* `results/metrics/v3_complex_composite_candidate_per_sample_audit.csv`

### 关键指标 / 结果

composite-selection 的定义为：

`composite = val IoU + val Dice - val area_error`

threshold=500 下，composite-selection candidate 的 3 seed mean test 指标为：

* MSE = 2.1444e+04 +/- 2.72e+02
* MAE = 4.9181e+01 +/- 1.39e+00
* IoU = 3.2170e-01 +/- 7.30e-03
* Dice = 4.5460e-01 +/- 8.70e-03
* area_error = 3.3740e-01 +/- 1.95e-02
* center_error = 1.2257e+00 +/- 1.23e-02
* `pred_area=0` = 10.33 +/- 5.13

旧 v3_complex MSE-oriented reference baseline `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt` 的 test 指标为：

* MSE = 2.07377174e+04
* MAE = 4.44655262e+01
* IoU = 2.95272047e-01
* Dice = 4.21885407e-01
* area_error = 3.94517442e-01
* center_error = 1.32594189e+00
* `pred_area=0` = 18

第 13.5 逐样本审计显示：

* IoU 改善样本数 = 125 / 200
* Dice 改善样本数 = 121 / 200
* area_error 改善样本数 = 121 / 200
* `pred_area=0` 被修复 = 14 / 18
* small / medium / large 分桶均有改善信号
* multi_defect、polygon、rotated_rect 三类中均存在多数样本改善
* MSE / MAE 代价主要来自背景区误差上升；缺陷区 MAE 反而下降

### 结论

第 13.4 / 13.5 证明，composite-selection 更符合本项目“缺陷边界形状反演”的主目标。它牺牲部分背景区域数值精度，但改善了 defect shape / mask 指标，并且逐样本改善不是少数 outlier 驱动。

因此，CURRENT_BASELINE 已更新为 v3_complex composite-selection shape-oriented baseline；旧模型 `checkpoints/best_model_v3_complex_tv_sweep_2e-6.pt` 降级为 MSE-oriented reference baseline。

### 下一步

后续主线应以新的 shape-oriented CURRENT_BASELINE 为锚点。旧 MSE-oriented baseline 仍保留为数值误差参考，但不再作为唯一主线最佳模型。

---

## 第 14.2 步：area-bin-balanced composite selection audit

### 目标

诊断 `macro_area_composite` 是否能在不修改模型、loss、数据和训练主流程的前提下，比当前 `composite` selection 更好地照顾 small / low-signal 样本。

### 修改内容

新增独立诊断脚本 `scripts/macro_area_selection_audit.py`，使用 v3_complex baseline 配置训练 3 个 seed，并在 validation set 上同时审计：

* `composite = IoU + Dice - area_error`
* `macro_area_composite = mean over small / medium / large bins of (IoU_bin + Dice_bin - area_error_bin)`

small / medium / large 仅按 validation set true_area 三分位定义；test set 只用于最终报告。未修改 `train_pinn.py`、`evaluate_pinn.py`、`data_generator_v2.py` 或 `CURRENT_BASELINE.md`。

### 输出文件

* `scripts/macro_area_selection_audit.py`
* `results/summaries/v3_complex_macro_area_selection_audit_summary.txt`
* `results/metrics/v3_complex_macro_area_selection_audit_metrics.csv`
* `results/metrics/v3_complex_macro_area_selection_epoch_log.csv`

### 关键指标 / 结果

`best_macro_area_composite` 与当前 `best_composite` 在 3 个 seed 上选中了完全相同的 epoch：

| seed | best_composite epoch | best_macro_area_composite epoch |
|---:|---:|---:|
| 42 | 13 | 13 |
| 123 | 25 | 25 |
| 2026 | 11 | 11 |

因此 overall 指标、small / medium / large 分桶指标、low-signal 指标都与当前 composite-selection CURRENT_BASELINE 完全一致。macro-area selection 没有额外减少 small `pred_area=0`，也没有改善 low-signal 样本。

### 结论

`macro_area_composite` 未带来新增收益，不满足接受条件。不建议将 `macro_area_composite` 集成进 `train_pinn.py`，也不继续设计更多 small-weighted / low-signal-weighted / polygon-weighted selection metric。当前 composite-selection shape-oriented `CURRENT_BASELINE` 保持不变。

### 下一步

不继续 selection metric 细调；`CURRENT_BASELINE` 不变。

---

## 第 15.1 / 15.2 步：mask-only boundary model 与 probability threshold calibration

### 目标

直接测试以缺陷边界形状反演为主任务的 mask-only boundary model：从 Bz 输入直接预测 defect probability / mask，不再通过 `pred_mu < 500` 间接得到 mask。第 15.2 进一步只做一次 validation-selected probability threshold calibration，判断第 15.1 的面积高估是否可以被合理概率阈值缓解。

### 修改内容

新增独立训练脚本 `scripts/train_mask_boundary_candidate.py`，未修改 `train_pinn.py`、`evaluate_pinn.py` 或 `data_generator_v2.py`。模型输出 mask logit / probability，target mask 定义为 `target_mu_norm < 0.5`，loss 使用 `BCEWithLogits + soft Dice`，checkpoint selection 仅使用 validation set。

新增独立校准脚本 `scripts/mask_boundary_threshold_calibration.py`。第 15.2 在 validation set 上从 `0.50 / 0.60 / 0.70 / 0.80 / 0.85 / 0.90 / 0.95` 中选择统一 probability threshold，最终选中 `0.90`；test set 只用于最终验证。

### 输出文件

* `scripts/train_mask_boundary_candidate.py`
* `scripts/mask_boundary_threshold_calibration.py`
* `results/summaries/v3_complex_mask_boundary_candidate_summary.txt`
* `results/metrics/v3_complex_mask_boundary_candidate_metrics.csv`
* `results/summaries/v3_complex_mask_boundary_threshold_calibration_summary.txt`
* `results/metrics/v3_complex_mask_boundary_threshold_calibration_metrics.csv`

### 关键指标 / 结果

第 15.1 fixed threshold=0.5 下，mask-only 明显提升 IoU / Dice，但面积高估严重，因此不作为 as-is baseline：

* IoU = 3.761e-01 +/- 3.10e-03
* Dice = 5.294e-01 +/- 3.20e-03
* area_error = 7.700e-01 +/- 3.44e-02
* `pred_area=0` = 1.67 +/- 1.15

第 15.2 validation-selected threshold=0.90 后，mask-only 在 test set 上相对 composite-selection CURRENT_BASELINE 的 3 seed mean 指标为：

| metric | composite-selection reference | mask-only threshold=0.90 |
|---|---:|---:|
| IoU | 3.217e-01 +/- 7.30e-03 | 3.319e-01 +/- 1.69e-02 |
| Dice | 4.546e-01 +/- 8.70e-03 | 4.729e-01 +/- 1.89e-02 |
| area_error | 3.374e-01 +/- 1.95e-02 | 3.220e-01 +/- 8.71e-03 |
| `pred_area=0` | 10.33 +/- 5.13 | 3.67 +/- 0.58 |

small 分桶从 IoU / Dice = 0.2219 / 0.3224 提升到 0.2743 / 0.3981，area_error 从 0.4560 降到 0.3706，`pred_area=0` 从 8.67 降到 2.33。low-signal 样本从 IoU / Dice = 0.1976 / 0.2973 提升到 0.2454 / 0.3673，area_error 从 0.5202 降到 0.4222，`pred_area=0` 从 10.33 降到 3.67。

### 结论

第 15.1 证明 mask-only boundary model 对边界 mask 任务有强正信号，但 fixed threshold=0.5 存在严重面积高估。第 15.2 使用 validation set 选择 probability threshold=0.90 后，IoU / Dice / area_error / `pred_area=0` / small / low-signal 均优于 composite-selection reference，满足接受条件。

因此，mask-only boundary model + validation-selected threshold=0.90 已提升为新的 boundary-oriented `CURRENT_BASELINE`。原 composite-selection baseline 保留为 μ-threshold shape-oriented reference；原 `v3_complex_tv_sweep_2e-6` 保留为 MSE-oriented reference。

### 下一步

后续实验应以新的 mask-only boundary baseline 为对照，不再围绕 composite-selection、selection metric、ensemble、threshold trick 做小修补。
# 主线同步：第 13.x-18.x 阶段

本节只同步近期主线路线判断。各阶段的完整结果仍以 `results/summaries/` 和 `results/metrics/` 中的文件为准。

## 13.x Composite Selection

第 13.4 / 13.5 证明，v3_complex μ-threshold 路线的主要瓶颈之一是 checkpoint selection。将 selection 从 MSE 改为 `composite = IoU + Dice - area_error` 后，shape-oriented mask 指标明显改善，并曾一度替换旧的 MSE-oriented baseline。

在后续 mask-only 路线完成后，composite-selection 不再是 active `CURRENT_BASELINE`。它保留为 μ-threshold shape-oriented reference；旧 `v3_complex_tv_sweep_2e-6` 保留为 MSE-oriented reference。

## 15.x Mask-Only Boundary Models

第 15.1 说明，直接做 Bz -> defect mask 是第一条真正比 μ-threshold reconstruction 更贴近项目目标的方向。固定 threshold `0.50` 时，IoU / Dice 明显提升，但面积高估严重。第 15.2 只使用 validation set 做 probability threshold calibration，并选出 threshold `0.90`；该设置相对 composite-selection 改善了 IoU / Dice / area_error / `pred_area=0` / small / low-signal。

第 15.4 进一步将 active baseline 更新为 mask-only grid decoder boundary model + validation-selected threshold `0.90`。这仍是当前 `CURRENT_BASELINE`。

SDF supervision、boundary head、coordinate refinement 都出现过局部指标正信号，但没有解决 polygon / rotated_rect 圆斑化，或引入了 area_error trade-off。因此这些方向停止，不再扩展 v2。

## 16.x Shape Prior、Signal Features 与 Retrieval

shape-prior latent 证明 mask autoencoder 本身可学，但 Bz -> latent 后不满足接受条件，主要问题是 area_error 和 small 样本表现不足。

hand-crafted Bz signal feature augmentation 没有通过 seed=42 screening gate。exemplar retrieval 在 validation 上选出 `stats_plus_shape + cosine + top1`，但 test 表现低于当前 baseline，polygon / rotated_rect 指标下降。这说明单纯 shape prior、retrieval 或手工 Bz 特征不能解决当前圆斑化问题。

## 17.x 几何模型与条件解码器尝试

star-convex radial model 的 oracle shape capacity 有一定效果，尤其 K=32，但训练出的 Bz -> radial 模型 area_error 明显差于 `CURRENT_BASELINE`，因此没有进入 3 seed。

U-Net-like decoder 和 shape-type conditional decoder 都没有通过 validation threshold rescue gate。主要问题是 area_error、small / low-signal trade-off，或没有真正改善 polygon / rotated_rect 圆斑化。因此普通 decoder 扩容和 type conditioning 都不是当前主线 candidate。

## 18.x Geometry 与 Forward Consistency

第 18.1 single-defect geometry decoder 使用 differentiable rotated-box rasterizer。它因为输出被限制为矩形，视觉上减少了圆斑化，并改善部分 IoU / Dice；但 area_error 明显变差，polygon 细节仍不能贴合，因此没有成为 candidate。

第 18.2 训练了 mask-to-Bz forward surrogate，并证明其足以进入 feasibility gate：test R2 `0.8520`，correlation `0.9231`。`lambda_forward=0.05` 的 forward consistency 在 IoU / Dice / center_error / Bz residual 上有明确正信号，但因为 area_error 和 small / low-signal trade-off，尚不足以替换 baseline。

第 18.3 bounded lambda bracket 只测试 `0.02`、`0.05`、`0.10`。其中 `lambda_forward=0.10` 是最佳固定值，因此进入 controlled 3 seed validation。

第 18.4 的 `lambda_forward=0.10` forward consistency 通过 review 和 3 seed validation，已提升为新的 `CURRENT_BASELINE`。Claude Code review 确认 mask-to-Bz surrogate 独立训练并冻结使用，checkpoint selection 只使用 validation score，probability threshold `0.80` 只由 validation set 选择，test set 只用于最终评估。

相比上一版 mask-only grid decoder baseline，第 18.4 的 overall IoU 从 `0.3391` 提升到 `0.3563`，Dice 从 `0.4812` 提升到 `0.5017`，area_error 从 `0.2885` 降到 `0.2734`，center_error 从 `1.2489` 降到 `1.1464`，Bz MSE 从 `0.3323` 降到 `0.1649`，`pred_area=0` 保持 `1.33`。这说明 forward consistency 同时改善 shape metrics 和 Bz consistency。

新的 `CURRENT_BASELINE` 为 mask-only grid decoder + forward consistency `lambda_forward=0.10` + validation-selected threshold `0.80`。原 mask-only grid decoder + threshold `0.90` 保留为 boundary reference，composite-selection 保留为 μ-threshold shape-oriented reference，`v3_complex_tv_sweep_2e-6` 保留为 MSE-oriented reference。

需要保留的限制是：polygon area_error 从 `0.3563` 轻微恶化到 `0.3743`，polygon / rotated_rect 精细边界圆斑化问题仍未根本解决。第 18.4 的主要收益是整体 shape metrics 与 Bz consistency 同时改善，而不是彻底解决 polygon 细边界问题。

---
## 第 19.3 步：CURRENT_BASELINE proposal + anisotropic basis refinement gate

第 19.3 测试了“CURRENT_BASELINE coarse probability / mask + K=4 anisotropic basis test-time refinement + forward consistency”的最后一条 basis refinement 路线。实验只使用当前 forward-consistency CURRENT_BASELINE 的 3 seed mean probability 生成 coarse proposal，不训练新网络，不更新 baseline checkpoint。

validation 在 `proposal_only` 和 `proposal_forward` 中选择了 `proposal_forward`。该目标显著降低 Bz MSE，并使 area_error 略低于 CURRENT_BASELINE，但 test set 上 IoU / Dice 低于 CURRENT_BASELINE；polygon / rotated_rect 预览中也没有稳定呈现更贴合直边、角点或旋转边界的效果，主要收益更像是 Bz residual 与面积控制，而不是边界细节真正改善。

因此第 19.3 不满足完整接受条件，不作为正式 candidate，也不更新 CURRENT_BASELINE。后续不继续 K / temperature / combine function / optimization steps / lambda / basis refinement v2 等小修补；当前 CURRENT_BASELINE 仍保持 mask-only grid decoder + forward consistency `lambda_forward=0.10` + validation-selected threshold `0.80`。
---

## 第 20.1 步：forward model / COMSOL feasibility planning

19.x 后内部 geometry / basis / proposal / mask-logit refinement 路线已基本到达边界，继续围绕现有单条 Bz 和当前 decoder 做小修补不再是主线。下一阶段转向 forward model / COMSOL / 多观测数据可行性，目标是提高缺陷边界反演问题本身的可辨识性。

---

## 第 20.42-20.55 步：combined COMSOL V3 与 geometry-aware / forward-consistent 方法审计

本节只记录阶段性路线结论，完整数值以对应 `results/summaries/` 和 `results/metrics/` 为准。

第 20.42 将 single-defect、`component_count=2`、`component_count=3` 合并为 COMSOL_DATA_BASELINE_V3 candidate，并训练统一 lightweight mask-only decoder。数据包和 schema 通过 review，但 baseline acceptance 未通过：single 和 cc3 尚可，cc2 相比独立 `COMSOL_MULTI_DEFECT_DATA_BASELINE` 明显退化，说明 combined benchmark 的主要瓶颈不是数据读取，而是不同拓扑任务在单一 dense decoder 中互相干扰。

第 20.43 单纯提升模型 capacity 后整体、cc2、cc3 都退化，证明“加宽普通 decoder”不能解决 combined V3 的 cc2 退化。第 20.44 / 20.45 的 topology-gated decoder 有一定方向信号，但本质仍是 weak topology-aware decoder patch：没有显式 geometry 参数、没有 differentiable rasterization、没有 predicted geometry -> forward Bz residual，也不是外部报告建议的 faithful implementation。20.46 方法审计后，停止继续 topology-gated decoder 小修补。

第 20.47-revised 按 Piao 2019 思路做 Bz-only weak adaptation：从 multi-line `delta_bz` 提取 hand-crafted / NLS-style features，再用 SVR / KRR / Ridge 预测 2D/quasi-2D geometry 参数。该路线没有完整实现 RBC、三轴 NLS 或 LS-SVM，且 test all-3 type accuracy 与 rect/rot rasterized mask IoU/Dice 均不足，结论是 Bz-only handcrafted/NLS-style features + classical regressor 不足以稳定完成 geometry inversion。

第 20.48 进入外部报告更推荐的路线：neural geometry head + PyTorch differentiable rotated-rectangle rasterization。它证明 geometry labels 与 rasterizer 没有 blocker，true geometry raster IoU 可达 1.0000，且显著优于 Piao weak adaptation；但 type accuracy 和 rotated angle prediction 仍不足。第 20.49 separate rect/rot heads 没有改善，说明简单拆分 head 可能带来分支样本不足。第 20.50 受控比较 5 个 geometry head 结构，best candidate 仍未超过 20.48，未找到能稳定减少 type confusion / angle error 的结构。

第 20.51 将 delta_bz-derived physics features 与 lightweight forward surrogate consistency 接入 neural geometry head。结果显示 mask IoU/Dice 相比 20.48 / 20.50 有小幅提升，angle MAE 也有改善，但 type accuracy 仍低于门槛，wrong_type 仍是主要失败模式；Claude Code review 通过且认为没有实现 blocker，但不建议继续 direct neural geometry head 小修补。

第 20.52 转向 Priewald-style coarse-to-fine / forward-consistent low-dimensional refinement：从 20.51 geometry-head proposal 出发，在低维几何参数空间用 frozen forward surrogate residual 精修。结果显示 forward NRMSE 明显下降，test IoU/Dice 也从 `0.6138 / 0.7577` 小幅提升到 `0.6194 / 0.7619`，说明 forward residual 对 refinement 有价值；但 area_error 变差，且 initializer 本身偏弱，因此不应把 20.52 写成 baseline。

第 20.53 改用 dense/coarse mask initializer 提取 rotated bbox / geometry proposal，再做同样的 Priewald-style refinement。refinement 相对 extracted geometry proposal 有稳定局部收益：test geometry-raster IoU/Dice 从 `0.5652 / 0.7169` 提升到 `0.5810 / 0.7300`，forward NRMSE 从 `0.4869` 降到 `0.3641`；但 dense/coarse proposal 低于 20.51 initializer 和 dense single-defect baseline，type / angle 提取仍弱，Claude review 认为方法协议可接受但当前结果无 acceptance 价值。下一步不继续 direct geometry head，也不把 20.53 提升为 candidate；优先改进 dense-to-geometry proposal extraction，或转向更少依赖 hard bbox 的 mask/profile basis refinement。

第 20.54 先查找可复用 COMSOL single-defect dense baseline artifact；只找到 pilot_v9 summary / metrics / script，没有可复用 checkpoint 或 prediction artifact，因此训练了一个只作为 proposal generator 的 strong rect/rot dense initializer。该 initializer 明显强于 20.53：test dense mask IoU/Dice 为 `0.6689 / 0.7994`，improved proposal extraction 的 geometry-raster test IoU/Dice 为 `0.6726 / 0.8017`，说明 initializer / proposal quality 瓶颈已被大幅缓解。但 Priewald-style refinement 从该强 proposal 出发后，test geometry-raster IoU/Dice 反而为 `0.6646 / 0.7958`，forward NRMSE 从 `0.4632` 降到 `0.4049` 的同时 mask 指标下降，属于 surrogate mismatch。Claude Code review 通过且无必须修复；结论是不把 20.54 写成 baseline，下一步优先改进 forward surrogate，若短期内不能降低空间峰位误差，则转向 mask/profile basis refinement。

第 20.55 专门审计并校准 forward surrogate / residual objective。20.54 mismatch audit 显示 test forward NRMSE 平均下降 `0.0584`，但 IoU/Dice 平均变化为 `-0.0079 / -0.0059`，forward reduction 与 IoU/Dice delta 的相关性为负或接近零。随后只训练 3 个受控 surrogate candidate：S1 geometry MLP waveform、S2 geometry + rasterized mask feature MLP、S3 peak-aware geometry MLP。S2 的 waveform NRMSE 最好，val/test NRMSE 为 `0.4489 / 0.4895`，但所有候选的 residual-error correlation 都没有通过 `> 0.05` gate；S2 的 val residual-error correlation 为 `-0.0292`，S3 虽为正也只有 `0.0215`。因此 Stage C calibrated refinement 被正确跳过，Claude Code review 确认无必须修复且同意停止。结论是：当前 forward surrogate 能拟合波形趋势，但 residual 不能可靠排序 geometry quality；下一步若继续 Priewald-style refinement，应先生成 synthetic perturbation forward data 或等价的局部扰动校准数据，而不是继续对当前 residual objective 调参。
## 第 20.56 步：local geometry perturbation forward-calibration pack + surrogate residual ordering audit

本轮只做 forward-calibration POC，不训练 inverse geometry head，不做 Priewald refinement，也不更新任何 baseline 文档。先在 `COMSOL single-defect pilot_v9` 的 rect/rot 子集上设计了 24 个 base sample、192 行 local perturbation plan；实际 COMSOL 侧按最小可接受 partial pack 生成 12 个 base、96 行 forward pack，其中 84 行为真实 COMSOL forward，12 行 `true_geometry_reference` 复用原始 NPZ 的 Bz 数组。生成结果满足 train/val/test = 64/16/16，rect/rot = 48/48，8 类 perturbation variant 均覆盖；`delta_bz = bz_defect - bz_no_defect` 校验通过。

随后只训练两个 perturbation-calibrated forward surrogate 候选并做 residual ordering audit。COMSOL oracle residual 对 geometry quality 的 val/test 排序准确率为 `0.6607 / 0.8393`，说明真实 forward residual 在该局部扰动包上具备排序信号。选中的 `S1_perturb_geom_mlp` 的 val/test waveform NRMSE 为 `0.3666 / 0.4289`，surrogate residual ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 为 `0.2679 / 0.1964`，较 20.55 的 S2 mismatch `0.3030 / 0.3939` 明显改善。

结论：local perturbation forward data 缓解了 20.55 的 surrogate mismatch，至少证明“好几何 residual 较小、坏几何 residual 较大”的 pairwise ordering 可以被学习。但 test residual-error correlation 仍为负值（`-0.0462`），因此只能建议下一步做受控 Priewald refinement retry，不能把它写成 baseline 或 production-ready forward objective。
## 第 20.57 步：perturbation-calibrated surrogate controlled refinement retry

第 20.57 用第 20.56 选中的 `S1_perturb_geom_mlp` 做一次受控 Priewald-style refinement retry。S1 没有 checkpoint 复用，因此按 20.56 protocol 在 perturbation pack 上重训于内存中，recovery 指标完全对齐：val/test waveform NRMSE 为 `0.3666 / 0.4289`，residual ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 为 `0.2679 / 0.1964`。初始化仍使用第 20.54 的 improved dense/extracted geometry proposal，true mask / true geometry 只用于 validation selection 和 final metrics，不参与 optimization。

validation 上 8 个 refinement config 全部使 mask 指标退化或 mismatch 过高，因此只选出最高分配置作为 diagnostic：`steps=50, lr=0.003, lambda_prior=0.10`。最终 test geometry-raster IoU/Dice/area_error 从 `0.6726 / 0.8017 / 0.1945` 变为 `0.6492 / 0.7829 / 0.2417`，forward NRMSE 虽下降 `0.0713`，但 mismatch_rate 达 `0.6212`，residual reduction 与 IoU/Dice delta 的相关性为 `-0.1824 / -0.2250`。这说明 20.56 的 pairwise ordering 改善没有转化为连续低维 geometry refinement 的可靠梯度。

结论：第 20.57 不通过 promising gate，不更新任何 baseline。当前 evidence 指向 residual objective / low-dimensional rect-rot parameterization 的连续优化瓶颈；不建议继续在该设置上小调 config。下一步若继续 geometry-aware route，应优先转向 mask/profile basis refinement，或在更大 perturbation pack / richer observations 上重新验证 forward residual landscape。
## 第 20.59 步：profile-compatible forward surrogate + controlled profile refinement retry

第 20.59 先按要求完成 multi-agent preflight。Method agent 结合 `PINN_literature` 中 Priewald 2013 等资料判断：profile-compatible forward surrogate 符合 forward-model-based inversion / refinement 路线；已有结果只否定了把 profile 压缩回 rect-like summary 的旧桥接方式，没有否定 profile-native surrogate。Codebase / safety / design / feasibility agents 均确认：本轮不需要 COMSOL，不生成新数据，复用 20.54 / 20.56 / 20.58 产物即可。

随后构建两个 profile-forward dataset：original profile dataset 为 400 个 rect/rot pilot_v9 样本，split = 268 / 66 / 66；perturb profile dataset 为 20.56 的 96 行 partial perturbation pack，split = 64 / 16 / 16。输入 profile representation 保留 K=8 station、half_width、occupancy、global center/angle/length/area/roughness 等 profile/basis 信息，不再压缩成 single rotated box。

训练的 3 个 profile-compatible surrogate candidate 中，validation score 选中 `PFS3_profile_station_sequence`。其 val/test waveform NRMSE/correlation 为 `0.3841 / 0.9233` 和 `0.3995 / 0.9177`，说明 profile 表示能拟合 waveform；但 val ordering accuracy 只有 `0.6607`，mismatch_rate 为 `0.3393`，未通过可用于 refinement 的 gate。test ordering 虽较高（`0.8929`），但只作为 final diagnostic，不能反向用于选择。

因此 Stage C controlled profile-forward refinement 被正确跳过，没有继续优化 profile 参数。Claude Code review 复审通过，无必须修复。结论是：profile-compatible forward surrogate 有边际价值，但当前 small perturbation coverage 还不足以支撑 forward-guided profile refinement。下一步优先扩展 profile perturbation data；若扩展后 validation ordering / mismatch 仍不通过，则应回到 no-forward profile basis 或 richer observations，而不是继续小调当前 forward refinement objective。
## 第 20.64 步：multi-direction excitation profile perturbation oracle ordering feasibility

本轮只恢复并完成 Stage E-G 之前的 20.64 POC 收口，不进入 20.65，也不开始 true 3D / Piao-style route。实验目标是验证真实改变 COMSOL excitation / magnetization direction 是否能改善 profile perturbation oracle residual ordering；不训练 surrogate、不训练 inverse model、不做 profile refinement、不更新 baseline。

Stage 0 preflight 确认该方向未被 20.61-20.63 否定，COMSOL 可通过 `ExternalCurrentDensity.Je` 控制源项方向，但必须用 direction probe 验证真实 field response。Stage A 生成 12 base / 96 profile rows 的 multi-direction plan，split 为 64/16/16，rect/rot = 48/48，8 类 variants 各 12。Stage B 使用真实 COMSOL forward 生成 direction_45 / direction_90，direction_0 复用同 geometry 的 20.63 default-direction Bx/By/Bz rows；pack 共有 96 rows、3 directions、3 axes、864 direction-axis observations，其中 576 为 real COMSOL forward。

Direction convention 固定为：`direction_0` default +Y `Je=["0","1e6[A/m^2]","0"]`，`direction_45` equal XY，`direction_90` +X `Je=["1e6[A/m^2]","0","0"]`。direction probe 显示 `direction_90` 相对 `direction_0` 的 no-defect / defect NRMSE 为 `1.6479 / 1.7981`，dominant axis 也从 `Bx` 转到 `By`，因此通过真实方向改变 gate；没有使用数组旋转、signal summation 或 fake fields。`delta_B = B_defect - B_no_defect` 对所有 direction-axis observations 校验通过，profile polygon valid，mask non-empty。

Stage C same-pack oracle audit 显示：test `direction_0` Bz-only ordering = `0.4545`，`direction_0` all-axis normalized = `0.4182`，`direction_90` Bz-only = `0.5273`，`direction_45` Bz-only = `0.4364`，multi-direction Bz train-std normalized = `0.5636`，multi-direction all-axis normalized = `0.3455`。multi-direction all-axis mismatch_rate = `0.6545`，residual-error correlation = `-0.8028`。因此 Bz-only multi-direction 有边际提升，但 all-axis normalized 明显更差，20.64 promising gate 未通过。

Claude Code review 最终通过且无 must-fix。review 同意真实 excitation direction change 有部分信号价值，但不足以支撑 multi-direction profile surrogate training；建议不训练 surrogate，不回到 profile-forward refinement。路线结论是：20.64 未证明 richer direction observation 稳定缓解 profile residual non-identifiability；下一步建议 true 3D profile / Piao-style route，但本轮不进入下一阶段，不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。
## 第 20.65 步：true 3D / Piao-style geometry profile feasibility design

本轮只完成 feasibility design，没有运行 COMSOL、没有生成数据、没有训练 surrogate / inverse model、没有执行 refinement，也没有更新任何 baseline 文档。Claude Code review 已通过且无 must-fix；review 的唯一建议是同步更新 `EXPERIMENT_LOG.md`、`NEXT_STEP.md` 和 `PINN优化路线.md`。

方法判断是：20.61-20.64 已连续证明当前 2D top-view profile-forward 小修不足以让真实 COMSOL residual 稳定排序 profile quality。single-height Bz、multi-height Bz、same-direction Bx/By/Bz、multi-direction excitation 都没有通过 same-pack oracle ordering gate。因此 2D profile-forward 小修正式暂停，下一条研究主线切换到 true 3D / Piao-style geometry profile。

Piao 2019 的迁移口径保持保守：本轮基于既有 fullpaper alignment summary 和已上传 PDF 的标题、摘要、章节级上下文，不声称本轮重新抽取并阅读全文，也不声称完整复现 Piao 2019。可迁移部分是 three-axis MFL observation、RBC six-parameter 3D profile label、geometry parameter regression、profile projection metrics 和 forward consistency；不可直接迁移的是当前 Bz-only、2D top-view mask、2D profile perturbation residual，以及 full PIG experimental setup。

COMSOL 能力边界也保持明确：当前 COMSOL 链路支持真实 3D volume solve，并已验证 Bx/By/Bz 输出和 source `Je` 方向控制；但现有 rect/rot/polygon/profile geometry 主要仍是 constant-depth prism / top-view extrusion。RBC depth-varying defect solid、variable-depth surface、loft/sweep/slice-union geometry 仍是 20.66 smoke 必须验证的 blocker，不能写成已支持或 train-ready。

第一个 3D pilot 推荐采用 Piao-style RBC six parameters：`L, W, D, wLD, wWD, wLW`，并派生 depth grid / projected 2D mask 用于 QA 和兼容对照。第一版只做 single-defect，不纳入 polygon、multi_defect 或 arbitrary free-form 3D volume。20.66 smoke 的第一目标固定为 `Bx/By/Bz @ sensor_z_m=0.008`，验证 `RBC params -> depth map -> COMSOL variable-depth defect solid -> same-source projected mask -> delta_B check`；`0.012m` 只保留为 20.67 或后续 ablation 的 schema 选项。

20.67 pilot 的 projected mask IoU `>=0.65`、Dice `>=0.78`、profile error `<=0.25` 等阈值只作为 preliminary acceptance guidance，不是已验证硬标准。dense mask baseline 只保留为 comparator，不再作为当前 geometry-forward 主线。
## 第 20.66 步：true 3D RBC-style smoke pack generation

本轮执行 true 3D / Piao-style 路线的第一个 smoke，只验证 `RBC params -> depth/profile grid -> COMSOL 3D/stepped-depth defect -> Bx/By/Bz @ sensor_z_m=0.008 -> delta_b check -> schema validation`。没有训练 forward surrogate 或 inverse model，没有做 refinement，没有更新 `CURRENT_BASELINE.md`，也没有创建或修改 COMSOL baseline 文档。

Stage 0 preflight 结论是：20.66 值得执行，但当前 COMSOL 能力边界必须诚实记录。现有链路支持真实 3D volume solve、Boolean Difference、no-defect/defect pair 和 `[mf.Bx, mf.By, mf.Bz]` 三轴导出；但 smooth RBC variable-depth surface / solid 仍未验证。RBC generator 本轮标记为 `exact_piao_rbc=False`，属于 RBC-style / RBC-inspired engineering approximation，不声称完整复现 Piao 2019。

Stage A 生成 6 个 single-defect RBC-style smoke samples，范围为 `L_m=0.010-0.030`、`W_m=0.006-0.020`、`D_m=0.001-0.006`，覆盖 shallow / medium / deep、narrow / wide、round / boxy / sharper profile。pure-Python validation 6/6 通过，`profile_depth_grid_m`、`profile_depth_map_xy_m`、`projected_mask_2d`、`profile_pose`、`rbc_params`、`geometry_params_json` 均可序列化且 mask 非空。

Stage B 使用真实 COMSOL forward 生成 6/6 smoke rows，几何实现为 `stepped_depth_layered_approximation`：每个样本使用 5 层 nested depth-level polygon prisms 做 stepped-depth approximation。`smooth_variable_depth_solid_verified=False`，`stepped_depth_approximation=True`，`constant_depth_extrusion_used_as_success=False`。本轮通过状态因此是 `stepped_depth_smoke_pass`，不是 `variable_depth_pass`。

Stage C schema validation 6/6 通过。NPZ 中保存了 `delta_b` / `b_defect` / `b_no_defect`，shape 为 `(N, 3, 3, 201)`；同时保存 `rbc_params`、`profile_pose`、`profile_depth_grid_m`、`profile_depth_map_xy_m`、`projected_mask_2d`、`depth_levels_m`、`stepped_depth_approximation` 和 `geometry_params_json`。`projected_mask_2d` 只作为 2D comparator，不替代 3D profile label。`delta_b = b_defect - b_no_defect` 的保存数组校验通过，Bx / By / Bz 均 finite 且非零。

Claude Code review 完成且无 must-fix。review 认可本轮没有把 constant-depth extrusion 伪装成 true 3D，也认可 Bx/By/Bz、depth map、projected mask 和 RBC params schema 一致；建议项是 COMSOL inventory 中的 delta error 计算偏定义式，后续可增强为更独立的求解漂移检查，但不构成本轮 blocker。

路线结论：20.66 技术链路在 stepped-depth smoke 层级通过，说明 true 3D / Piao-style 路线具备继续验证价值；但 smooth true variable-depth RBC solid 仍未通过。下一步唯一建议不是直接声称 smooth 3D pilot ready，而是先决策：继续攻 smooth variable-depth COMSOL geometry，还是明确接受 stepped-depth 作为 20.67 pilot approximation。dense mask baseline 仍只作为 comparator。

---

## 第 20.67 步：smooth / near-smooth variable-depth true 3D geometry feasibility

本轮只验证 COMSOL geometry feasibility，没有训练模型、没有做 refinement、没有更新 `CURRENT_BASELINE.md`，也没有提交 data / NPZ / `.mph` / raw CSV / checkpoint / preview PNG / notes。

Stage A 从 20.66 的 RBC-style smoke plan 选择 `medium_round`、`deep_round`、`medium_boxy` 作为 target geometry test samples，其中 `medium_round` 是唯一必跑 forward sample。三者的 pure-Python depth/profile validation 均通过，生成了 12-layer / 16-layer high-layer fallback contours；该计划明确保留 `exact_piao_rbc=False`，仍是 RBC-style engineering approximation。

Stage B 的 smooth / loft / imported closed-surface probe 只做有限检查。当前 COMSOL/MCP 路径没有形成 verified closed smooth defect body；`Loft` 创建失败，`ParametricSurface` / `Import` 只能说明 feature 可创建，不能形成已验证可 Boolean subtract 的 closed defect body。因此按计划回退到 high-layer fallback。`medium_round` 的 `high_layer_approx_12` geometry-only gate 通过：12 个 depth levels，区别于 20.66 的 5-layer stepped-depth smoke；Boolean subtract 和 mesh precheck 成功；没有把 constant-depth extrusion 算作成功。

Stage C 仅对 `medium_round` 执行 one-sample no-defect / defect forward solve，导出 `[mf.Bx, mf.By, mf.Bz]` @ `sensor_z_m=0.008`，`delta_b = b_defect - b_no_defect` 校验通过，NPZ/schema validation 通过。最终状态是 `high_layer_pass`，不是 `variable_depth_pass`，也不是 `near_smooth_pass`。

Claude Code review 已完成，无明确 must-fix。review 接受当前结论：20.67 证明 12-layer high-layer approximation 可以跑通 geometry + Bx/By/Bz forward + schema validation，但没有证明 smooth variable-depth geometry 技术可行。下一步不能直接声称 smooth true 3D RBC pilot ready；若要扩样，需要人工确认是否接受 high-layer approximation 作为 pilot approximation，否则应继续修 smooth/closed-surface geometry builder。
## 第 20.69 步：watertight imported solid builder hardening

本轮只验证 imported watertight mesh solid route，不训练模型、不做 refinement、不进入 pilot、不更新 `CURRENT_BASELINE.md`，也不修改任何 COMSOL baseline 文档。先补跑了正确指向 `C:\Users\19166\Desktop\PINN_project` 和 `C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP` 的 Safety/Git subagent；结论是本轮只能逐文件白名单 staging，必须排除 `MODEL_STRUCTURE_PLAN.md` 删除项、`scripts/visualize_current_baseline.py`、COMSOL 既有 dirty items、data/NPZ/.mph/raw CSV/checkpoint/preview PNG/notes/temp STL。

Stage A/B 在 PINN 侧只选择 `medium_round`，由 RBC-style depth grid 生成 positive-depth watertight STL。坐标和单位已显式记录：`mesh_units=m`，top cap 位于 `z=0`，bottom surface 为 `z=-depth`，steel surface 为 `z=0`，steel z 范围为 `[-0.006, 0.0] m`，profile pose 通过 `center_x_m / center_y_m / angle_rad` 映射到 COMSOL 坐标。mesh validation 通过：`is_watertight=True`，edge incidence 全为 2，`nonmanifold_edges_count=0`，`zero_area_triangles_count=0`，`volume_m3=1.2918e-07`，`max_depth_m=0.0025`，`depth_rmse_vs_target=2.5767e-05`，defect void 嵌入 steel 且与 surface 相交。temp STL 只作为 generated artifact，未提交。

Stage C 在 COMSOL 侧先跑 known prism sanity probe，再跑 RBC mesh import probe，二者分开记录。known sanity probe 通过；RBC watertight STL 也通过 geometry gate：`import_success=True`、`repair_success=True`、`form_solid_success=True`、`imported_domain_count=1`、`boolean_subtract_success=True`、`steel_notched_domain_count=1`、`mesh_precheck_success=True`。这说明 20.68 中 imported mesh route 的 Boolean empty steel domain blocker 已被推进到 imported watertight solid geometry gate 通过。

Stage D 只在 Stage C 通过后执行 one-sample forward smoke。no-defect model 求解成功，但 defect model 的 stationary solver 在 imported watertight geometry 上不收敛，错误为线性迭代发散 / no solution returned。由于 `b_defect` 未生成，本轮没有 `delta_b`，没有生成 `true_3d_imported_watertight_forward_smoke_v1.npz`，也没有运行 Stage E NPZ/schema validator。

Route decision 为 `C_import_boolean_pass_forward_not_run_or_failed`。20.69 证明 imported watertight mesh solid 的 geometry path 技术上前进了一步，明显优于 20.68 的 imported mesh failure 和 high-layer-only 状态；但全链路尚未 forward-ready，更不能 pilot-ready。Claude Code review 通过，无 must-fix；review 同意当前 route decision：下一步应修 COMSOL imported solid 的 solve / mesh-quality / solver robustness，再考虑 smooth/mesh-based pilot generation。
## 第 20.73 步：true 3D RBC pilot training gate

本轮只执行 training gate，不运行 COMSOL、不生成或修改 NPZ、不训练 baseline、不更新 `CURRENT_BASELINE.md`。数据入口固定为 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled`，通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.manifest.json` 显式解析，禁止 latest/newest NPZ 自动扫描。

输入 gate 通过：assembled pack 为 `pilot_generated`、`train_ready_candidate=True`、`baseline_ready=False`，`delta_b` shape 为 `(56, 3, 3, 201)`，Conv1D 输入为 `(56, 9, 201)`，split 为 train/val/test = `36/10/10`，`delta_b = b_defect - b_no_defect` 原始精度校验误差为 `0.0`。feature sanity 使用 Piao-inspired 手工信号特征，不声称复现 Piao 2019 NLS + LS-SVM；validation 选择 `svr_rbf_C10`，test normalized MAE 为 `0.7564`，L/W/D MAE 为 `3.10/3.36/0.95 mm`，projected mask IoU/Dice 为 `0.6785/0.8003`。

neural gate 使用小型 Conv1D，只输入 Bx/By/Bz 的 `delta_b`，输出 6 个 train-normalized RBC-style 参数。3 个 seed 均完成，validation 选择 seed `2026`、best epoch `3`；完整训练轨迹可把 train normalized MAE 拟合到 `0.0012`，但 selected checkpoint 的 val/test normalized MAE 为 `0.6886/0.7601`，只明显优于 mean baseline `0.8598`，没有超过 feature baseline `0.7564`。test L/W/D MAE 为 `2.55/2.80/1.22 mm`，curvature 平均 MAE 为 `0.2095`，projected mask IoU/Dice 为 `0.7285/0.8347`，profile depth RMSE 为 `0.000606 m`。

路线判断：20.73 是 `small_data_generalization_limited`，不是 baseline。当前证据说明 `L_m`、`W_m` 有可学习信号，`D_m` 边缘不足，`wLD/wWD/wLW` 三个 curvature 参数尚不可辨识；N=56 不足以支撑 baseline 或模型结构结论。下一步应优先扩展 true 3D RBC 数据到 120/240 量级，并增加 validation 样本，再判断 curvature 是否需要更强输入观测或 exact Piao/NLS-style feature pipeline。
## 第 20.74 步：true 3D RBC imported-watertight dataset expansion to v2_120

本轮只做数据扩展、top-up、assembled pack validation 和 registry/manifest 更新；没有训练模型、没有运行 baseline gate、没有更新 `CURRENT_BASELINE.md`，也没有创建或修改 COMSOL baseline 文档。路线仍标记为 `true_3d_piao_style` 下的 RBC-style / Piao-inspired engineering approximation，`exact_piao_rbc=False`，`rbc_style_approximation=True`。

v1 source pack 为 `comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled`，N=56，split=36/10/10。20.73 training gate 已证明 train 可拟合但泛化受小样本限制，尤其 `D_m` 和 `wLD/wWD/wLW` 曲率参数不稳，因此 20.74 设计了 80-row top-up plan，目标补足 D_m、curvature、LD/WD 和 deep/elongated coverage。Python watertight mesh generation 80/80 通过；COMSOL top-up 使用 `imported_watertight_mesh_solid`、20.70 material/domain fix、`selected_solver_protocol=default`、`mesh_auto_size=5`、`Jscale=1.0`，最终 56/80 top-up samples 成功，7 fail，17 not attempted；no-defect field 复用，`[mf.Bx, mf.By, mf.Bz] @ sensor_z_m=0.008` 真实导出，`delta_b = b_defect - b_no_defect` 最大误差为 0.0。

assembled dataset `comsol_true_3d_rbc_imported_watertight_pilot_v2_120` 已生成并验证通过，实际 N=112，split=train/val/test 76/18/18，curvature coverage 为 sharp=22、round=23、boxy=23、LD_dominant=24、WD_dominant=20，depth coverage 为 shallow=41、medium=36、deep=35。虽然未达到目标 N=120，但超过最低 gate：N>=108、split>=72/18/18、每个 curvature_template>=20，因此状态记录为 `pilot_generated`，`train_ready_candidate=True`，`baseline_ready=False`。`COMSOL_DATA_REGISTRY.md` 已新增 v2 top-up 和 v2_120 assembled entries，tracked manifests 已生成；allowed_use/forbidden_use 继续禁止 latest/newest 自动扫描、baseline update 和 current baseline replacement。

Claude Code review 复审通过，无 must-fix。下一步唯一建议是对 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v2_120` 执行显式 registry/manifest-gated true 3D training gate；如果 WD_dominant 或 deep/elongated 在训练中仍不稳，再做第二波 targeted top-up，而不是更新 baseline。

## 第 20.75 步：true 3D RBC training gate on v2_120

本轮只执行 training gate，不运行 COMSOL、不生成新数据、不修改 NPZ、不训练或创建正式 baseline，也不更新 `CURRENT_BASELINE.md`。数据入口固定为 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v2_120`，通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.manifest.json` 显式解析，禁止 latest/newest NPZ 自动扫描。

输入 gate 通过：v2_120 pack 为 `pilot_generated`、`train_ready_candidate=True`、`baseline_ready=False`，`delta_b` shape 为 `(112, 3, 3, 201)`，Conv1D 输入为 `(112, 9, 201)`，split 为 train/val/test = `76/18/18`，`delta_b = b_defect - b_no_defect` 原始精度校验误差为 `0.0`。feature sanity 使用 Piao-inspired 手工信号特征，不声称复现 Piao 2019 NLS + LS-SVM；validation 选择 `svr_rbf_C10`，test normalized MAE 为 `0.7677`，L/W/D MAE 为 `3.30/3.37/0.99 mm`，projected mask IoU/Dice 为 `0.6417/0.7717`。

neural gate 使用小型 Conv1D，只输入 Bx/By/Bz 的 `delta_b`，输出 6 个 train-normalized RBC-style 参数。3 个 seed 均完成，validation 选择 seed `42`、best epoch `2`；完整训练轨迹可把 train normalized MAE 拟合到 `0.0044`，selected checkpoint 的 train/val/test normalized MAE 为 `0.7323/0.6949/0.7039`，优于 mean baseline `0.8803`，也优于 feature baseline `0.7677`。test L/W/D MAE 为 `2.51/2.59/1.11 mm`，curvature 平均 MAE 为 `0.1905`，projected mask IoU/Dice 为 `0.7297/0.8364`，profile depth RMSE 为 `0.000548 m`。

路线判断：20.75 是 `v2_120_promising_but_not_baseline`。相对 20.73 N=56，neural test normalized MAE、L/W/D MAE、D_m、curvature MAE、projected mask Dice 和 profile depth RMSE 都有改善；`L_m`、`W_m`、`D_m` 可作为 provisional learnable params。但 `wLD/wWD/wLW` 仍不稳定，val/test 各 18 个样本仍偏小，N=112 不足以支撑 baseline。下一步唯一建议是扩展 true 3D RBC dataset 到 240，再重跑同一套 registry/manifest-gated training gate。
## 第 20.76 步：true 3D RBC imported-watertight dataset expansion to v3_240

本轮只做 data expansion / top-up / assembled pack validation；没有训练模型、没有建立 baseline、没有更新 `CURRENT_BASELINE.md`，也没有修改 COMSOL baseline 文档。`comsol_true_3d_rbc_imported_watertight_pilot_v2_120` 作为 source pack 保持不覆盖，v3 top-up 使用 `imported_watertight_mesh_solid`、20.70 material/domain fix、`selected_solver_protocol=default`、`mesh_auto_size=5`、`Jscale=1.0`，无 high-layer fallback。

Stage A/B/C 完成 v2_120 audit、160-row top-up plan 和 watertight mesh generation。Plan 的 first 128 rows 精确补齐 split、curvature、depth、aspect 目标；mesh validation 为 160/160 pass，temp STL 只写入 ignored `data/comsol_mfl/generated/temp_true_3d_rbc_dataset_240_topup_meshes/`。COMSOL top-up 最终 128/160 success，18 fail，14 not_attempted；失败主要集中在 deep/narrow 或部分 imported-solid mesh/geometry cases，已记录在 `results/inventory_true_3d_rbc_dataset_240_topup_pack.csv`。成功样本均导出 `[mf.Bx,mf.By,mf.Bz] @ sensor_z_m=0.008`，`delta_b=b_defect-b_no_defect` validation 通过。

Assembled dataset `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 已生成并验证：N=240，split=train/val/test `162/39/39`，curvature coverage 为 sharp=48、round=49、boxy=47、LD_dominant=46、WD_dominant=50，depth coverage 为 shallow=86、medium=79、deep=75。Schema validation、registry validation 和 manifest validation 均通过；status=`pilot_generated`，train_ready_candidate=True，baseline_ready=False。独立 review agent 通过，无 must-fix；唯一建议是避免 top-up summary 将 top-up source pack 误写成 standalone pilot status，该措辞已修正为 `topup_status=topup_generated`。下一步唯一建议：对 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 执行显式 registry/manifest-gated training gate。
## 第 20.81 步：feature-fusion neural model for true 3D RBC curvature

本轮只在 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上执行 feature-fusion model refinement，没有运行 COMSOL，没有生成或修改 NPZ/data，没有创建 baseline，也没有更新 `CURRENT_BASELINE.md`。数据入口继续通过 `COMSOL_DATA_REGISTRY.md` 和 tracked manifest 显式解析，禁止 latest/newest NPZ 自动扫描。

Stage 0 subagent preflight 结论为 GO：registry/manifest gate 通过；20.80 的 Piao/NLS-inspired feature CSV 可用，包含 240 行和 642 个 delta_b/BxByBz 派生特征；泄漏边界明确为模型输入只能使用 `F0__..F5__` 特征列，`sample_id`、`split`、`curvature_template`、`depth_bin`、`aspect_bin`、`rbc_params`、`projected_mask` 等只可用于 join、split、supervision 或 metrics；20.77 neural、20.80 feature-only、20.79 failed refinement 三组 reference metrics 完整；安全边界要求继续排除 `scripts/visualize_current_baseline.py` 和所有 data/NPZ/checkpoint/preview/baseline artifacts。

Stage A/B 固定 reference 与 feature-fusion input gate。20.77 neural reference 为 test total MAE `0.678014`、L/W/D `1.892/2.186/0.800 mm`、curvature `0.201076`、wLD/wWD/wLW `0.209439/0.204469/0.189319`、IoU/Dice `0.750650/0.847727`、profile depth RMSE `0.000387737 m`。20.80 feature-only reference 为 total `0.695724`、L/W/D `2.595/2.361/0.966 mm`、curvature `0.190304`、wLD/wWD/wLW `0.209649/0.194797/0.166465`、Dice `0.826272`。20.79 failed refinement reference 为 total `0.753387`、curvature `0.211584`、Dice `0.834597`。feature matrix validation 通过，`FS_basic_physical`、`FS_basic_cross_axis`、`FS_nls_optional`、`FS_curvature_focused` 和 `FS_F1F2_curvature_only` 均可 train-only impute/scale 为 finite matrix。

Stage C candidate screen 使用 seed=42，只测 3 个受控候选：`H1_curv_fusion_F1F2_w0p5`、`H2_curv_fusion_F0F1F2_w0p5`、`H3_curv_fusion_F0F1F2_w1p0`。validation-only selection 选中 `H3_curv_fusion_F0F1F2_w1p0`，使用 `FS_basic_physical` 441 个特征，validation total MAE `0.734925`、curvature MAE `0.194861`、Dice `0.821305`，满足进入 multi-seed 的诊断 gate。screen 的 selected test 指标为 total `0.701155`、curvature `0.188367`、wLD/wWD/wLW `0.211349/0.193620/0.160133`、Dice `0.826273`，说明 curvature 有单 seed 信号但 total/Dice 有 trade-off。

Stage D multi-seed 只复跑 validation-selected `H3_curv_fusion_F0F1F2_w1p0`，seeds=`42/123/2026`，validation-only 选择 seed `2026`。selected test 指标为 total MAE `0.667888`，L/W/D MAE `2.030/1.807/0.957 mm`，curvature MAE `0.194483`，wLD/wWD/wLW `0.217079/0.202304/0.164068`，projected mask IoU/Dice `0.774541/0.866573`，profile depth RMSE `0.000445297 m`。相对 20.77 neural，total 改善 `-0.010126`，Dice 改善 `+0.018845`，但 curvature 只改善 `-0.006592`，未达到本轮 `>=0.01` 实质改善门槛，wLD 反而退化 `+0.007639`。相对 20.80 feature-only，total 更好但 curvature 更差。

Route decision 为 `feature_fusion_total_not_curvature`。feature-fusion 对整体参数 MAE 和 projected mask 有价值，但没有实质解决 curvature 风险；不能升级 formal benchmark candidate，也不能写成 baseline。下一步唯一建议是 `D_redefine_curvature_labels_output_representation`，优先复查 `wLD/wWD/wLW` 的输出定义、可辨识性和 label representation，再决定是否做 curvature-targeted data top-up 或 exact Piao feature reproduction。

Review agent 已完成只读复核，无 must-fix。review 建议把 audit/decision 中的 “Did fusion improve curvature?” 改成 “Did fusion reach >=0.01 substantive curvature improvement?”，该建议已采纳并重跑 audit。所有新增结果均为 scripts/results/Markdown；未提交 data、NPZ、checkpoint、preview PNG、notes、COMSOL artifacts 或 baseline docs。

# 2026-05-26 Stage 20.83 R1 six-params profile-primary loss training gate

- dataset_id: `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`
- scope: fixed v3_240 dataset only; no COMSOL, no new data, no NPZ modification, no baseline update.
- profile generator gate: passed. Torch differentiable RBC-style generator matches the existing NumPy generator and stored `profile_depth_grid_m` / `projected_mask_2d` labels.
- candidate screen: completed with seed=42. Validation-selected candidate was `P2_profile_primary_dim_guard_w1p0`.
- Stage C multi-seed: skipped by gate because `eligible_for_multiseed=False`.
- test result for selected candidate: total MAE `0.689411`, L/W/D MAE `2.183 / 1.860 / 0.848 mm`, auxiliary wLD/wWD/wLW MAE `0.207 / 0.216 / 0.193`, profile_depth_rmse_m `0.000409718`, Er-like profile error `0.283775`, projected mask IoU/Dice `0.778090 / 0.868042`.
- comparison: 20.83 improved projected mask Dice over 20.77 (`0.868042` vs `0.847727`) but did not improve the primary profile metric (`0.000409718` vs 20.77 `0.000387737`).
- route decision: negative gate. Keep 20.77 as profile reference and 20.81 as visual/mask comparator; do not upgrade R1 and do not update `CURRENT_BASELINE.md`.
- review: independent review agent passed after one must-fix loop. The must-fix was validation selection purity; candidate selection now uses validation reference Dice, not test Dice.

# 2026-05-26 Stage 20.84 true 3D RBC candidate consolidation / visual audit

- scope: existing-results audit only. No training, no COMSOL, no new data, no NPZ modification, no preview PNG regeneration, and no baseline update.
- inputs: existing 20.77 neural metrics, 20.81 feature-fusion metrics, 20.83 profile-primary metrics, and the existing 20.83 prediction gallery CSV/PNG paths under ignored `results/previews/`.
- candidate roles:
  - 20.77 neural reference is the profile/depth main candidate with `profile_depth_rmse_m=0.000387737` and projected mask Dice `0.847727`.
  - 20.81 feature-fusion is the non-negative projected-mask / visual reference with Dice `0.866573`, but its profile RMSE is `0.000445297`, worse than 20.77.
  - 20.83 profile-primary loss is negative evidence for the current R1 setup: Dice `0.868042` is numerically highest, but profile RMSE `0.000409718` is worse than 20.77, so it cannot replace 20.77/20.81.
- gallery audit: best-profile examples are low profile-error by persisted CSV metrics and visual inspection; worst-profile examples, especially the deep/wide boxy case, retain a plausible 2D footprint but substantially underestimate 3D depth. High-Dice/high-profile-error cases confirm that projected mask quality cannot replace 3D profile metrics.
- route decision: keep 20.77 as the profile/depth benchmark candidate for formal rerun; keep 20.81 as visual/mask comparator; do not continue small loss-weight tweaks on the current 20.83 profile-primary path.
- review: independent read-only review agent passed. It suggested tightening the wording around Dice because 20.83 has the numerically highest Dice but is a negative profile-depth gate; the summaries were updated accordingly.

# 2026-05-27 Stage 20.90 true 3D RBC liftoff / sensor-offset COMSOL diagnostic

- scope: small diagnostic pack only. COMSOL was used to generate diagnostic rows; no training, no baseline update, no `CURRENT_BASELINE.md` change, and generated data/NPZ artifacts remain uncommitted.
- dataset/baseline: fixed current true 3D RBC profile-depth baseline using `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` and the 20.88a inference artifact manifest.
- pack: 12 base geometries, 96 COMSOL solve rows, and 36 nominal-derived postprocess spatial-misalignment rows; total evaluation rows = 132. COMSOL success count was 96/96.
- validation: diagnostic NPZ validation passed; validation explicitly allows float32 delta recompute tolerance for the generated diagnostic pack.
- raw-input finding: liftoff is the dominant risk. The worst raw factor was `liftoff_z_0p012`, with profile RMSE degradation about `627.747%` and projected Dice drop about `0.211738` versus nominal.
- calibrated-input finding: the fixed 20.89 `per_axis_rms_train_stats` calibration strongly reduced source/amplitude variation and partially reduced liftoff damage, but it is diagnostic only and is not a baseline replacement. Source/amplitude mean degradation improved from about `261.622%` raw to `0.000%` calibrated; liftoff remained failing at about `42.762%` mean calibrated degradation.
- scan-line offset and postprocess axis misalignment were low risk in this pack: raw mean degradation was about `1.766%` for scan-line offset and `1.700%` for axis misalignment.
- route decision: keep the 20.85 baseline unchanged. The next technical step should be dedicated COMSOL liftoff robustness / augmentation data design before internal-defect feasibility or real-data claims.
- review: independent read-only review passed after one must-fix loop. The must-fix corrected nominal replay from circular nominal-vs-nominal comparison to regenerated COMSOL nominal rows versus the 20.88a clean prediction artifact for the same 12 base samples.

# 2026-05-27 Stage 20.91 true 3D RBC liftoff augmentation pack plan

- scope: plan-only. No COMSOL run, no generated data/NPZ, no training, no checkpoint, no preview, and no `CURRENT_BASELINE.md` update.
- dataset gate: explicit `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` through `COMSOL_DATA_REGISTRY.md` and the tracked v3_240 manifest.
- baseline artifact: fixed 20.88a artifact manifest `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json`.
- motivation: 20.90 identified liftoff as the main unsolved physical-acquisition blocker. Source/amplitude calibration remains a diagnostic caveat only.
- plan result: target pack size met, with 48 base geometries and 4 paired liftoff levels per base, for 192 planned COMSOL rows.
- liftoff levels: `sensor_z_m=0.006 / 0.008 / 0.010 / 0.012`; `0.008m` remains nominal.
- coverage: split `train/val/test=32/8/8`; curvature template counts `sharp=10, round=10, boxy=10, LD_dominant=9, WD_dominant=9`; depth counts `shallow=16, medium=16, deep=16`; aspect counts `compact=12, balanced=12, wide=12, narrow=12`.
- route decision: the next execution stage should generate this dedicated liftoff COMSOL pack, then 20.92 should compare unconditioned vs scalar `sensor_z_m` conditioned liftoff-aware training. Internal defect feasibility remains deferred.
- review: independent read-only review passed with no must-fix. A suggestion to record exact registry/manifest paths in human-facing summaries was adopted.

# 2026-05-28 Stage 21.0 internal / buried defect COMSOL smoke pack

- scope: internal / buried defect feasibility smoke only. COMSOL was run for the smoke pack, but no training was performed, no `CURRENT_BASELINE.md` update was made, and internal defect was not mixed into the surface / near-surface RBC baseline.
- dataset_id: `comsol_internal_defect_smoke_pack_v1`.
- route: `internal_buried_defect_feasibility`.
- generation result: full smoke pack generated, `planned_samples=12`, `successful_samples=12`, `status=smoke_generated`.
- shape coverage: `internal_sphere=4`, `internal_ellipsoid=4`, `internal_cuboid=4`.
- burial-depth coverage: `shallow=6`, `medium=3`, `deep=3`.
- COMSOL result: Boolean subtract, mesh/solve, Bx/By/Bz export, and `delta_b=b_defect-b_no_defect` all passed for the successful 12/12 rows.
- validation result: `delta_b`, `b_defect`, and `b_no_defect` have shape `(12,3,3,201)`; axis names are `Bx/By/Bz`; required labels include `L_m`, `W_m`, `D_m`, `burial_depth_m`, `depth_to_surface_m`, `defect_center_xyz_m`, `shape_type`, and `ground_truth_method`.
- registry/manifest: `COMSOL_DATA_REGISTRY.md` and `results/manifests/comsol_internal_defect_smoke_pack_v1.manifest.json` were created/updated with `baseline_ready=false`, `train_ready_candidate=false`, `allowed_use=schema_validation, explicit_internal_training_gate`, and forbidden baseline/latest-newest usage.
- route decision: full smoke feasibility passed and can enter 21.1 internal pilot pack. Real-data alignment remains deferred until internal pilot data and label semantics are stronger.
- review: independent read-only review passed with no must-fix. The review confirmed that no forbidden artifacts were staged and that COMSOL repo existing dirty items were not mixed into this task.

# 2026-05-27 Stage 20.91b true 3D RBC liftoff augmentation pack generation

- scope: executed the approved 20.91 liftoff pack only. COMSOL was run for the dedicated liftoff pack; no training, no 20.92 execution, no `CURRENT_BASELINE.md` update, and no generated data/NPZ/.mph/raw CSV/checkpoint/preview/notes/temp STL artifacts were committed.
- dataset_id: `comsol_true_3d_rbc_liftoff_aug_pack_v1`.
- source: `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`, through explicit registry/manifest lineage.
- generation result: full pack generated, `planned_comsol_rows=192`, `successful_comsol_rows=192`.
- base/liftoff structure: 48 base geometries, each with paired `sensor_z_m=0.006 / 0.008 / 0.010 / 0.012 m`; complete paired base count = 48 and incomplete pair count = 0.
- validation result: `validation_pass=True`, `train_ready_candidate=True`, `status=diagnostic_pack_generated`, `baseline_ready=False`.
- registry/manifest: `COMSOL_DATA_REGISTRY.md` and `results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json` were updated/created with `allowed_use=schema_validation, explicit_liftoff_training_gate` and forbidden baseline/latest-newest usage.
- route decision: full pack can enter 20.92 liftoff-aware training gate; compare unconditioned vs scalar `sensor_z_m` conditioned model. Internal defect feasibility remains deferred.
- review: independent read-only review passed with no must-fix.
# 2026-05-27 Stage 20.92 liftoff-aware true 3D RBC training gate

- dataset_id: `comsol_true_3d_rbc_liftoff_aug_pack_v1`, loaded only through `COMSOL_DATA_REGISTRY.md` and `results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json`.
- scope: training gate on the existing 20.91b liftoff pack only. No COMSOL run, no new data generation, no NPZ modification, no checkpoint/preview artifact committed, and no `CURRENT_BASELINE.md` update.
- input/split gate: passed. Pack has 192 rows, 48 base geometries, liftoff levels `0.006 / 0.008 / 0.010 / 0.012 m`, complete paired liftoff rows for every base, and grouped split by `base_sample_id`: train/val/test bases `32/8/8`, rows `128/32/32`.
- fixed C0 baseline on the liftoff pack: test all-liftoff profile RMSE `0.000738997 m`, nominal `0.000333059 m`, non-nominal `0.000874310 m`, non-nominal Dice `0.683351`.
- trained candidates: `C1_unconditioned_liftoff_aug` and `C2_sensor_z_conditioned`, seeds `42/123/2026`, validation-only selection and test-final reporting. `C3_calibrated_input_conditioned` was skipped because 20.89 calibration remains a diagnostic caveat, not a training/baseline protocol.
- selected by validation: `C1_unconditioned_liftoff_aug`, seed `123`.
- selected test metrics: all-liftoff profile RMSE `0.000697073 m`; nominal `0.000809011 m`; non-nominal `0.000659761 m`; non-nominal Er-like `0.899135`; non-nominal L/W/D MAE `2.224 / 1.741 / 1.477 mm`; non-nominal projected mask IoU/Dice `0.734606 / 0.833129`; non-nominal auxiliary wMAE `0.257595`.
- comparison vs C0: non-nominal profile RMSE improved by `24.539%` and Dice improved by `0.149778`, but nominal `0.008 m` profile RMSE regressed by `142.903%`.
- decision: partial liftoff signal, not a passed robustness candidate. Keep `CURRENT_BASELINE.md` unchanged; inspect liftoff pack failure cases and nominal/non-nominal trade-off before more COMSOL, real-data alignment, or internal defect feasibility.
- review: independent read-only review passed with no must-fix. One wording suggestion was adopted: sensor_z usefulness is marked as a post-hoc test diagnostic, not model selection.

# 2026-05-27 Stage 20.93 liftoff trade-off audit and nominal-preserving strategy design

- scope: read-only audit/design. No COMSOL run, no training, no data/NPZ/checkpoint/preview/notes mutation, and no `CURRENT_BASELINE.md` update.
- sources: existing 20.90/20.91/20.92 summaries and metrics only.
- audit result: the 20.92 selected `C1_unconditioned_liftoff_aug` seed `123` improved non-nominal profile RMSE from C0 `0.000874310 m` to `0.000659761 m` and Dice from `0.683351` to `0.833129`, but nominal `0.008 m` profile RMSE regressed from `0.000333059 m` to `0.000809011 m`.
- diagnosis: C1 is an unconditioned mixed-liftoff model. It sees liftoff-dependent amplitude/shape changes without `sensor_z_m`, while nominal rows are only one quarter of the paired pack and the validation score did not include an explicit nominal-preservation penalty. The result is nominal forgetting on held-out base geometries.
- C2 status: `C2_sensor_z_conditioned` was not selected by the predeclared validation protocol. Its post-hoc test signals remain diagnostic only and were not used for model selection.
- strategy design: primary next strategy is `S3_baseline_plus_liftoff_adapter`; secondary ablation is `S2_sensor_z_conditioned_revised_selection`; `S4_paired_consistency_loss` is reserved as a regularizer after the base objective is stable.
- route decision: next unique step is a nominal-preserving baseline+liftoff adapter training gate. It needs training but does not need new COMSOL data before that gate. `CURRENT_BASELINE.md` remains the 20.85 nominal true 3D RBC profile-depth baseline.
- review: independent read-only review passed with no must-fix. A suggested CSV clarity fix was adopted by splitting profile RMSE relative change and wMAE relative change into explicit fields.

# 2026-05-27 Stage 20.94 nominal-preserving liftoff adapter training gate

- dataset_id: `comsol_true_3d_rbc_liftoff_aug_pack_v1`, loaded through `COMSOL_DATA_REGISTRY.md` and the tracked manifest only.
- scope: liftoff robustness candidate training gate. No COMSOL run, no new data generation, no NPZ modification, no checkpoint/preview artifact committed, and no `CURRENT_BASELINE.md` update.
- input/split gate: passed. Pack has 192 rows, 48 base geometries, four complete liftoff levels, and grouped split by `base_sample_id`: train/val/test bases `32/8/8`. `base_sample_id` is used only for split and paired consistency, not as model input.
- baseline replay: frozen 20.85/20.77 baseline reproduced C0 test nominal profile RMSE `0.000333059 m`, non-nominal profile RMSE `0.000874310 m`, and non-nominal Dice `0.683351`.
- candidate screen: seed `42` selected `A2_latent_residual_adapter`; it beat A1 output-residual and A3 full sensor_z model under validation-only nominal-preserving selection.
- multi-seed: selected `A2_latent_residual_adapter`, seed `2026`.
- selected test metrics: nominal profile RMSE `0.000335821 m`; non-nominal profile RMSE `0.000437214 m`; non-nominal projected mask IoU/Dice `0.741925 / 0.842378`; non-nominal L/W/D MAE `1.939 / 1.715 / 0.871 mm`; non-nominal auxiliary wMAE `0.253896`.
- comparison vs C0: nominal profile RMSE degraded only `0.829%`, within the <=10% guard; non-nominal profile RMSE improved by `49.993%`; non-nominal Dice improved by `0.159027`.
- decision: A2 is a liftoff robustness candidate and should enter a formal liftoff benchmark. This is not a baseline replacement; `CURRENT_BASELINE.md` remains the nominal 20.85 baseline.
- review: independent read-only review passed with no must-fix. Two provenance/input-boundary suggestions were adopted.

# 2026-05-29 Stage 21.5 internal defect benchmark report / candidate consolidation

- scope: benchmark report, group-level failure/risk audit, candidate comparison, and route decision only. No training, no COMSOL, no data/NPZ mutation, no checkpoint/preview artifact, no `CURRENT_BASELINE.md` update, and no baseline transition.
- dataset_id: `comsol_internal_defect_pilot_pack_v2_240`, explicitly referenced through `COMSOL_DATA_REGISTRY.md` and manifest.
- candidate roles: mean baseline remains the trivial comparator; `svr_rbf_C10` remains the selected delta_b feature baseline; `internal_v2_conv1d_multitask` seed `42` remains the selected neural benchmark candidate from 21.4.
- feature baseline: test total normalized MAE `0.416406`; L/W/D MAE `0.757 / 1.151 / 0.108 mm`; burial_depth MAE `0.472 mm`; center_xyz MAE `1.560 mm`; shape accuracy/F1 `0.950000 / 0.950549`.
- neural candidate: test total normalized MAE `0.406366`; L/W/D MAE `0.761 / 0.947 / 0.093 mm`; burial_depth MAE `0.595 mm`; center_xyz MAE `1.380 mm`; shape accuracy/F1 `1.000000 / 1.000000`.
- learnability: L/W/D, center_xyz, and shape_type are positive; burial_depth has signal but remains the main risk because feature baseline is stronger.
- failure audit: group-level only because 21.4 did not persist per-sample predictions/checkpoints. Feature wins burial_depth in all compared test groups; neural wins center_xyz in most groups and total/shape overall.
- decision: no baseline transition. Next step is `B_improve_burial_depth_head_model`, with shape-conditioned model as secondary ablation.
- review: independent read-only review passed with no must-fix.

# 2026-05-29 Stage 21.4 internal defect v2_240 training gate

- dataset_id: `comsol_internal_defect_pilot_pack_v2_240`, explicitly loaded through `COMSOL_DATA_REGISTRY.md` and `results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json`.
- scope: internal/buried defect training gate only. No COMSOL run, no data/NPZ mutation, no checkpoint/preview artifact committed, no `CURRENT_BASELINE.md` update, and no baseline creation.
- input gate: passed. N=`240`, split=`160/40/40`, `delta_b` shape `(240,3,3,201)`, Conv1D input `(240,9,201)`, axis names `Bx/By/Bz`.
- coverage: each split covers `internal_sphere/internal_ellipsoid/internal_cuboid`, `shallow/medium/deep/deep_plus`, and `small/medium/large`; aspect coverage is available for ellipsoid/cuboid.
- feature baseline: validation selected `svr_rbf_C10`; test total normalized MAE `0.416406`; L/W/D MAE `0.757 / 1.151 / 0.108 mm`; burial_depth MAE `0.472 mm`; center_xyz MAE `1.560 mm`; shape accuracy/F1 `0.950000 / 0.950549`.
- neural gate: small Conv1D multitask, seeds `42/123/2026`; validation selected seed `42`, best epoch `293`.
- selected neural metrics: train/val/test total normalized MAE `0.068995 / 0.236902 / 0.406366`; L/W/D MAE `0.761 / 0.947 / 0.093 mm`; burial_depth MAE `0.595 mm`; center_xyz MAE `1.380 mm`; shape accuracy/F1 `1.000000 / 1.000000`.
- comparison: neural beat mean baseline and selected feature baseline on composite score and total normalized MAE; feature baseline remains stronger on burial_depth single metric (`0.472 mm` vs neural `0.595 mm`).
- decision: v2_240 is sufficient for a positive internal training gate, but not baseline-ready. Next step is `E_prepare_internal_benchmark_report`, not baseline transition.
- review: independent read-only review passed with no must-fix.

# 2026-05-29 Stage 21.3b internal defect dataset v2_240 pack generation

- scope: internal/buried defect dataset generation, assembly, validation, registry/manifest, and route decision only. No training, no `CURRENT_BASELINE.md` update, and no internal defect baseline creation.
- source dataset: `comsol_internal_defect_pilot_pack_v1`, source rows `96`.
- top-up COMSOL pack: planned/success `168/168`; selected top-up rows `144`; buffer rows `24`; no failed rows.
- assembled dataset: `comsol_internal_defect_pilot_pack_v2_240`, assembled rows `240`.
- split: train/val/test `160/40/40`; v1 original split was discarded and recomputed for v2.
- coverage: `internal_sphere/internal_ellipsoid/internal_cuboid = 80/80/80`; `shallow/medium/deep/deep_plus = 60/60/60/60`; `small/medium/large = 80/80/80`; aspect counts `compact=134`, `elongated_x=54`, `elongated_y=52`.
- validation: Bx/By/Bz finite, `delta_b=b_defect-b_no_defect` passed, every split covers all three shape types, all four burial levels, all three size levels, and ellipsoid/cuboid aspect coverage.
- registry/manifest: created `results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json`; `train_ready_candidate=true`, `baseline_ready=false`, and latest/newest auto-discovery remains forbidden.
- route decision: enter 21.4 internal v2_240 training gate. Internal branch remains separate and is not mixed into the surface/near-surface RBC baseline.
- review: independent read-only review passed with no must-fix.

# 2026-05-28 Stage 20.95 formal liftoff benchmark for A2 residual adapter

- dataset_id: `comsol_true_3d_rbc_liftoff_aug_pack_v1`, explicitly loaded through `COMSOL_DATA_REGISTRY.md` and `results/manifests/comsol_true_3d_rbc_liftoff_aug_pack_v1.manifest.json`.
- scope: formal report/audit only. No training, no COMSOL, no data/NPZ mutation, no checkpoint/preview/notes artifact committed, and no `CURRENT_BASELINE.md` update.
- selected module: `A2_latent_residual_adapter`, seed `2026`, carried forward from 20.94 validation-only selection.
- nominal result: C0 frozen 20.85 profile RMSE `0.000333059 m` -> A2 `0.000335821 m`, only `+0.829%`; nominal Dice improved from `0.843957` to `0.855910`.
- non-nominal result: C0 profile RMSE `0.000874310 m` -> A2 `0.000437214 m`, `-49.993%`; non-nominal Dice improved from `0.683351` to `0.842378`; non-nominal L/W/D MAE was `1.939 / 1.715 / 0.871 mm`.
- per-liftoff note: A2 strongly improves `0.010 m` and `0.012 m`; `0.006 m` remains a watch case with profile RMSE `0.000390434 m`, `+9.110%` vs C0, while Dice improves to `0.882553`.
- decision: accept A2 as a `CURRENT_BASELINE` companion robustness module, not as `CURRENT_BASELINE` itself. The baseline remains the 20.85 nominal true 3D RBC profile-depth baseline.
- metadata contract: `sensor_z_m` is required for multi-liftoff / real-experimental inference using the companion module.
- artifact boundary: 20.95 uses persisted 20.94 aggregate metrics; per-sample A2 failure ranking was not recomputed because no per-sample A2 prediction artifact was available.
- review: independent read-only review passed after the review file was saved; no experimental or route must-fix remained.

# 2026-05-28 Stage 20.96a A2 liftoff adapter inference artifact recovery

- dataset_id: `comsol_true_3d_rbc_liftoff_aug_pack_v1`, explicitly loaded through `COMSOL_DATA_REGISTRY.md` and the tracked manifest.
- scope: artifact recovery for 20.96 inference smoke. No COMSOL, no new data generation, no NPZ modification, no `CURRENT_BASELINE.md` update, and no checkpoint/prediction artifact committed.
- fixed protocol: `A2_latent_residual_adapter`, seed `2026`, frozen 20.85/20.77 baseline, base-grouped split, train-only normalization, validation-only selection, and test-final verification. No hyperparameter tuning or model change.
- exported ignored artifacts:
  - checkpoint: `checkpoints/true_3d_rbc_liftoff_adapter_artifacts/true_3d_rbc_liftoff_a2_adapter_seed2026.pt`
  - prediction artifact: `checkpoints/true_3d_rbc_liftoff_adapter_artifacts/true_3d_rbc_liftoff_a2_adapter_seed2026_predictions.npz`
- tracked manifest: `results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json`.
- verification: checkpoint reload passed with zero prediction/residual diff. Metrics reproduced 20.94/20.95: nominal profile RMSE `0.000335821 m`, non-nominal profile RMSE `0.000437214 m`, and non-nominal Dice `0.842378`.
- review: independent read-only review passed with no must-fix; checkpoint and prediction artifact are ignored and uncommitted.

# 2026-05-28 Stage 20.96 liftoff-conditioned true 3D RBC inference smoke

- dataset_id: `comsol_true_3d_rbc_liftoff_aug_pack_v1`, explicitly loaded through `COMSOL_DATA_REGISTRY.md` and the tracked manifest.
- scope: inference smoke only. No training, no COMSOL, no data/NPZ mutation, no checkpoint/preview artifact committed, and no `CURRENT_BASELINE.md` update.
- artifacts loaded: the frozen 20.85/20.77 baseline via `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json` and the A2 companion adapter via `results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json`.
- runner: `scripts/run_true_3d_rbc_liftoff_conditioned_inference.py` supports `auto`, `force_baseline`, and `force_adapter`. In `auto`, nominal `sensor_z_m=0.008` uses the frozen baseline and non-nominal liftoff uses baseline plus A2 adapter.
- smoke result: test auto all-liftoff profile RMSE `0.000411175 m`, Dice `0.842773`; nominal RMSE `0.000333059 m`, Dice `0.843957`; non-nominal RMSE `0.000437214 m`, Dice `0.842378`.
- comparison: force-baseline non-nominal RMSE was `0.000874310 m`, while auto/A2 non-nominal RMSE was `0.000437214 m`, reproducing the 20.95 A2 companion behavior. Auto route accuracy on test was `1.0`.
- metadata contract: `sensor_z_m` is mandatory in meters; supported range is `[0.006, 0.012]`; missing `sensor_z_m` raises an error; out-of-range values are flagged and not treated as validated.
- review: independent read-only review passed after fixing the `0.012 m` boundary out-of-range flag. A2 remains a companion robustness module, not `CURRENT_BASELINE`.

# 2026-05-28 Stage 20.97 real-data schema intake contract

- scope: schema, metadata contract, templates, validator, preprocessing plan, and route decision only. No training, no COMSOL, no data/NPZ/checkpoint/preview/notes mutation, and no `CURRENT_BASELINE.md` update.
- baseline context: `CURRENT_BASELINE` remains the 20.85 nominal true 3D RBC profile-depth baseline; A2 remains the liftoff robustness companion module.
- schema document: `REAL_DATA_INTAKE_SCHEMA.md`.
- supported intake formats: recommended prepared `delta_b` with shape `(N,3,3,201)` or single sample `(3,3,201)`, and raw `b_defect + b_no_defect` with `delta_b=b_defect-b_no_defect`.
- required metadata: `sensor_z_m`, `axis_order=[Bx,By,Bz]`, `scan_line_y_m`, `sensor_x_m` length 201, Tesla units, `no_defect_reference_id`, no-defect reference method, coordinate system, sensor alignment status, gain calibration status, material/specimen information, and magnetization setup.
- blockers: missing `sensor_z_m`, missing no-defect reference, Bz-only data, unknown axis order or unit, inability to resample `sensor_x` to 201, inability to map three scan lines, out-of-range liftoff without retraining/validation, and internal/buried defect mixing.
- validator: `scripts/validate_true_3d_rbc_real_data_intake_schema.py` supports manifest-only validation and does not require real data files. The template intentionally reports `ready_for_inference=False` until placeholder fields are replaced.
- route decision: next step is a real-data manifest dry run, initially without a data file. Internal/buried defect remains a separate schema branch.
- review: independent read-only review passed with no must-fix; two validator hardening suggestions were adopted.

# 2026-05-28 Stage 20.98 true 3D RBC real-data manifest dry run

- scope: manifest dry run only. No real signal array was read, no data/NPZ was generated, no COMSOL run, no training, no checkpoint/preview/notes mutation, and no `CURRENT_BASELINE.md` update.
- dry-run manifest: `results/manifests/real_data_internal_block_dry_run_manifest.json`, based on the 20.97 real-data intake template.
- specimen status: the user-owned iron block is marked as `specimen_type=internal_defect_iron_block` and `defect_location_type=internal_or_buried`.
- validation result: `ready_for_inference=false`; 16 hard blockers were reported.
- main blockers: internal/buried defect branch mismatch, no real signal array, unknown tri-axis `Bx/By/Bz`, unknown no-defect reference, missing/unknown `sensor_z_m`, unknown `axis_order`, unknown three-line `scan_line_y_m`, unknown 201-point `sensor_x_m`, unknown unit, unknown coordinate system, unknown sensor alignment / gain calibration status, and unknown magnetization setup.
- route decision: the current surface / near-surface RBC baseline must not be used for this internal defect block. The next step is `C. create internal defect feasibility schema`.
- review: independent read-only review passed with no must-fix. One suggestion was adopted by unifying `ready_for_inference=false` casing and writing dry-run blocker text in Chinese.

# 2026-05-28 Stage 20.99 internal / buried defect feasibility schema

- scope: schema / label / data-generation design only. No COMSOL run, no training, no data/NPZ generation or mutation, no checkpoint/preview/notes artifact, and no `CURRENT_BASELINE.md` update.
- schema document: `INTERNAL_DEFECT_SCHEMA.md`.
- boundary: internal / buried defects are not surface RBC defects. Current 20.85 surface / near-surface RBC baseline and A2 liftoff companion remain limited to the surface RBC-style branch.
- required labels: `L_m`, `W_m`, `D_m_or_cavity_size_m`, `burial_depth_m` / `depth_to_surface_m`, `defect_center_xyz_m`, `shape_type`, profile descriptor or cavity mask, and `ground_truth_method`.
- hard blockers: missing burial depth, missing no-defect reference, Bz-only without a dedicated low-capability branch, unknown defect coordinates relative to the scan surface, missing ground truth, unknown `sensor_z_m`, unknown material/specimen geometry, unknown magnetization setup, or unknown alignment/gain status.
- smoke pack plan: 12 planned COMSOL smoke samples, covering `internal_ellipsoid`, `internal_cuboid`, and `sphere_like`, each at shallow / medium / deep / deep_plus burial levels. This is a plan only; COMSOL was not run.
- route decision: do not transfer directly from the surface RBC baseline. The recommended first output is `shape_type + L/W/D + burial_depth + center_xyz`; Bx/By/Bz is the main branch, while Bz-only is only a limited diagnostic branch.
- review: independent read-only review passed with no must-fix.
## 2026-05-29 Stage 21.6 internal defect burial-depth focused refinement

- 范围：只在 `comsol_internal_defect_pilot_pack_v2_240` 上做 burial-depth model diagnostic / controlled training；未运行 COMSOL，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`，未创建 baseline。
- 数据入口：通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json` 显式加载，禁止 latest/newest NPZ scan。
- 信号审计：feature baseline 的 burial_depth 优势主要来自 delta_b-derived 的幅值、能量、线宽、跨轴比例等统计量；21.4 neural 在 total/center/shape 上更强，但 burial head 对这些物理统计量利用不足。风险组包括 `shallow`、`elongated_x`、`internal_cuboid` 和 large/medium size。
- candidate screen：B0 是 21.4 neural reference；B1 提高 burial loss weight；B2 使用 Conv1D latent + delta_b-derived feature MLP 进入 burial head；B3 使用 predicted shape logits 条件化 burial head。true shape_type 没有作为模型输入。
- selection：seed=42 screen 按 validation burial_depth MAE 和 total/center/shape guard 选择 B2；非 selected candidate 的 test 指标只作诊断，不参与选择。
- multi-seed：B2 在 seeds `42/123/2026` 上复跑，validation-only 选择 seed `2026`，best epoch `277`。
- selected result：test total normalized MAE `0.395256`；L/W/D MAE `0.849 / 0.985 / 0.090 mm`；burial_depth MAE `0.413 mm`；center_xyz MAE `1.466 mm`；shape accuracy/F1 `0.975000 / 0.975309`。
- 对比：burial_depth MAE 优于 21.4 neural `0.595 mm`，也优于 selected feature baseline `0.472 mm`；total MAE 也优于 21.4 neural `0.406366` 和 feature baseline `0.416406`。center_xyz 和 shape 相比 21.4 neural 有轻微代价，但未触发 secondary metric collapse。
- decision：B2_feature_fusion_burial_head 形成 internal refinement candidate。下一步唯一建议是 internal benchmark rerun / candidate upgrade；仍不更新 `CURRENT_BASELINE.md`，internal defect 继续作为独立分支。
- review：独立只读 review 通过，无 must-fix；已补充说明非 selected candidate test 指标只作诊断。

## 2026-05-29 Stage 21.7 internal defect benchmark rerun / candidate upgrade

- 范围：固定 `comsol_internal_defect_pilot_pack_v2_240` 和 `B2_feature_fusion_burial_head` 做 formal rerun / candidate upgrade；未运行 COMSOL，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`，未创建 baseline。
- 数据入口：通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json` 显式加载，禁止 latest/newest NPZ scan。
- 输入边界：模型只使用 `delta_b/BxByBz` 和 train-normalized delta_b-derived features；true `shape_type`、burial bin、size/aspect、split、sample_id 均未作为输入。
- rerun 协议：seeds `42/123/2026`，train-only normalization，validation-only epoch/seed selection，test final only；不调新超参，不做 test 反选。
- selected result：validation 选择 seed `2026`，best epoch `277`；train/val/test total normalized MAE `0.055742 / 0.234877 / 0.395256`。
- test metrics：L/W/D MAE `0.849 / 0.985 / 0.090 mm`；burial_depth MAE `0.413 mm`；center_xyz MAE `1.466 mm`；shape accuracy/F1 `0.975000 / 0.975309`。
- 稳定性：三 seed test burial_depth MAE 为 `0.399 / 0.428 / 0.413 mm`，均优于 21.4 neural `0.595 mm` 和 feature baseline `0.472 mm`。
- 对比结论：B2 test total normalized MAE `0.395256` 优于 21.4 neural `0.406366` 和 feature baseline `0.416406`；burial_depth 短板已不再是 primary blocker，但 center_xyz 和 shape 仍需在报告/可视化中持续审计。
- decision：B2 形成 internal benchmark candidate，但不是 baseline；`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline，internal defect 仍是独立分支。
- review：独立只读 review 通过，无 must-fix；已按建议补充 metrics 的 `selected` 兼容列。

## 2026-05-29 Stage 21.8 internal defect benchmark report / visualization package

- 范围：基于 21.7 B2 formal rerun 生成 internal defect benchmark report package、group-level failure audit、visual asset index 和 route decision；未训练，未运行 COMSOL，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`。
- 报告包：`results/summaries/internal_defect_benchmark_report_package_summary.txt`、`results/metrics/internal_defect_benchmark_report_package_metrics.csv`、`results/metrics/internal_defect_benchmark_candidate_comparison_matrix.csv`。
- candidate 口径：`B2_feature_fusion_burial_head` 是 internal benchmark candidate；输入边界仍是 `delta_b/BxByBz` + delta_b-derived features，不使用 true `shape_type`、burial bin、size/aspect、split 或 sample_id。
- 核心指标：selected seed `2026`，best epoch `277`；train/val/test total normalized MAE `0.055742 / 0.234877 / 0.395256`；L/W/D MAE `0.849 / 0.985 / 0.090 mm`；burial_depth MAE `0.413 mm`；center_xyz MAE `1.466 mm`；shape accuracy/F1 `0.975000 / 0.975309`。
- group audit：弱势分组包括 `aspect_bin=elongated_y`、`shape_type=internal_ellipsoid`、`size_level=large` 和 `shape_type=internal_cuboid`；worst burial group 是 `size_level=medium`，burial_depth MAE `0.617 mm`；worst center group 是 `burial_depth_level=deep_plus`，center_xyz MAE `1.966 mm`；shape risk group 是 `deep_plus`，shape accuracy `0.900`。
- visual asset 状态：21.7 没有 per-sample B2 prediction/profile artifact，本轮只生成 visual assets index 和 missing 说明；没有生成或提交 PNG，也没有伪造 best/worst 单样本图。
- 限制：仍是 COMSOL 仿真域；shape 只覆盖 internal_sphere / internal_ellipsoid / internal_cuboid；真实 internal 数据未验证；不是 `CURRENT_BASELINE`。
- route decision：下一步唯一建议是 `A_internal_real_data_schema_alignment`；如果要做 gallery 或 inference smoke，需要先恢复 B2 inference artifact。
- review：独立只读 review 通过，无 must-fix；已按建议在 failure cases CSV 中增加 `artifact_scope=group_level_only` / `aggregate_only`。

## 2026-05-29 Stage 21.9 internal defect B2 inference artifact recovery

- 范围：按 21.7 固定协议恢复 `B2_feature_fusion_burial_head` inference artifact；未运行 COMSOL，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`。
- 数据入口：通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json` 显式加载 `comsol_internal_defect_pilot_pack_v2_240`，禁止 latest/newest scan。
- 固定协议：seed `2026`，best epoch `277`，split `160/40/40`，train-only normalization，validation-only checkpoint selection，test final only；输入只使用 `delta_b/BxByBz` 和 delta_b-derived features。
- ignored checkpoint: `checkpoints/internal_defect_b2_artifacts/internal_defect_b2_feature_fusion_seed2026.pt`。
- ignored prediction artifact: `checkpoints/internal_defect_b2_artifacts/internal_defect_b2_feature_fusion_seed2026_predictions.npz`。
- tracked manifest: `results/manifests/internal_defect_b2_inference_artifact_manifest.json`，记录 checkpoint/prediction path、sha256、model_config、feature_config、normalization 和 selected metrics。
- verification：checkpoint reload max prediction diff `0`，shape diff count `0`；test total normalized MAE `0.395256`，L/W/D MAE `0.849 / 0.985 / 0.090 mm`，burial_depth MAE `0.413 mm`，center_xyz MAE `1.466 mm`，shape accuracy/F1 `0.975000 / 0.975309`，逐项复现 21.7/21.8。
- artifact policy：checkpoint 和 prediction artifact 均未提交；只提交 manifest、summary/metrics 和脚本。
- review：独立只读 review 通过，无 must-fix；已将 preflight summary 重写为 UTF-8 可读中文。

## 2026-05-29 Stage 22.0 internal defect B2 failure-driven inference audit

- 范围：基于 21.9 `B2_feature_fusion_burial_head` artifact 做 test split replay、failure-driven audit、failure gallery index 和 route decision；未训练，未运行 COMSOL，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`。
- 数据入口：显式引用 `dataset_id=comsol_internal_defect_pilot_pack_v2_240`、`COMSOL_DATA_REGISTRY.md`、`results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json` 和 `results/manifests/internal_defect_b2_inference_artifact_manifest.json`；禁止 latest/newest scan。
- B2 replay：test total normalized error 的 mean/median/p95/max 为 `0.395 / 0.429 / 0.881 / 0.913`；burial_depth error 为 `0.413 / 0.260 / 1.266 / 1.674 mm`；center_xyz error 为 `3.096 / 3.033 / 8.309 / 8.785 mm`。
- failure tags：`catastrophic_failure=5/40`，`geometry_branch_failure=1/40`，`shape_misclassified=1/40`；geometry branch failure 定义为 shape 错分且 center error > `3.0 mm` 且 burial error > `1.0 mm`。
- worst cases：worst center 是 `internal_topup_045`，true/pred 都是 `internal_cuboid`，center error `8.785 mm`，burial error `1.533 mm`，属于 full_shift_failure；worst burial / geometry branch 是 `internal_pilot_091`，true `internal_cuboid`、pred `internal_ellipsoid`，burial error `1.674 mm`，center error `4.306 mm`。
- 分组判断：cuboid -> ellipsoid 不是系统性大面积错分，目前只有 1 例，但它是严重 geometry branch failure；catastrophic full-shift 更集中在 compact、large/medium、shallow/deep_plus tail cases，不是单一 shape 独有。
- gallery/index：生成 `results/metrics/internal_defect_b2_failure_gallery_index.csv`，PNG 只生成到 ignored `results/previews/internal_defect_b2_failure_gallery/`，未提交。
- decision：B2 仍可称 internal benchmark candidate，但不能称 stable inference model，也不是 baseline。下一步唯一建议是 `B_shape_conditioned_two_stage_internal_model`，真实 internal inference smoke 暂缓。
- review：独立只读 review 通过，无 must-fix；确认 checkpoint/prediction artifact 和 preview PNG 未提交，`CURRENT_BASELINE.md` 未修改。

## 2026-05-30 Stage 22.1 shape-conditioned / two-stage internal defect model

- 范围：在 `comsol_internal_defect_pilot_pack_v2_240` 上训练 shape-conditioned / two-stage internal candidates，目标是降低 22.0 暴露的 center/burial tail failure；未运行 COMSOL，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`，未创建 baseline。
- 数据入口：仍通过 `COMSOL_DATA_REGISTRY.md` 与 `results/manifests/comsol_internal_defect_pilot_pack_v2_240.manifest.json` 显式加载；禁止 latest/newest scan。
- 输入边界：正式候选只使用 `delta_b/BxByBz` 和 delta_b-derived features；true `shape_type`、burial bin、size/aspect、split、sample_id 均未作为正式模型输入。`T4_oracle_true_shape_diagnostic` 仅作 oracle 诊断，未参与正式选择。
- B2 reference：test total error mean/median/p95/max `0.395 / 0.429 / 0.881 / 0.913`；burial p95/max `1.266 / 1.674 mm`；center p95/max `8.309 / 8.785 mm`；catastrophic failure `5/40`；geometry_branch_failure `1/40`。
- candidate screen：seed `42` 下 T3_shape_specific_heads 是唯一 validation-eligible 正式候选；T1/T2 未过 validation tail gate，T4 是 oracle diagnostic。
- multi-seed：T3 在 seeds `42/123/2026` 上复跑，validation-only 选择 seed `123`，best epoch `283`；test total normalized MAE `0.357371`，L/W/D MAE `0.735 / 0.992 / 0.086 mm`，burial_depth MAE `0.457 mm`，center component MAE `1.143 mm`，shape accuracy/F1 `0.950000 / 0.950549`。
- tail result：T3 test center p95 从 B2 `8.309 mm` 降到 `5.999 mm`，但 center max 升到 `10.468 mm`；burial p95/max 退化到 `1.690 / 1.848 mm`；catastrophic failure 仍是 `5/40`，geometry_branch_failure 仍是 `1/40`。
- decision：22.1 不是 stable inference candidate upgrade。T3 是有用的 shape-specific heads 诊断，但没有通过 tail gate；下一步唯一建议是 targeted internal hard-case top-up，或在 top-up 后重新设计更强 two-stage branch。internal defect 仍不是 baseline。
- review：独立只读 review 通过，无 must-fix；确认没有 true shape label leakage、没有 test 反选、没有 forbidden artifacts staged。

## 2026-05-30 Stage 22.2 targeted internal hard-case top-up plan

- 范围：只做 hard-case top-up 设计；未运行 COMSOL，未训练，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`，internal defect 仍是独立 benchmark branch。
- 输入证据：显式引用 `comsol_internal_defect_pilot_pack_v2_240` manifest、22.0 B2 failure audit、22.1 shape-conditioned tail metrics 和 `INTERNAL_DEFECT_SCHEMA.md`。
- hard-case audit：B2 与 T3 都不是 stable inference model；catastrophic failure 仍为 `5/40`，geometry_branch_failure 仍为 `1/40`。重点 strata 固定为 cuboid/ellipsoid confusion、compact、medium/large、shallow/deep_plus，以及 x/y center-region neighbor cases。
- top-up plan：生成 `results/metrics/internal_defect_hard_case_topup_plan.csv`，目标 N=`120`，minimum usable N=`72`，`sensor_z_m=0.008`，不引入 liftoff variation，不混入 surface RBC baseline。
- target quota：9 个 target 的实际计划行数与推荐配额完全一致：`24/20/18/16/14/10/10/4/4`，总计 120；coverage gate 通过。
- route decision：唯一下一步是 `22.2b targeted COMSOL hard-case top-up pack generation`。继续模型 refinement 和真实 internal inference smoke 都暂缓；当前不改 schema。
- review：独立只读复审通过；确认 plan-only、安全边界、target quota、hard-case evidence 和 route decision 均满足要求。
## 2026-05-30 Stage 22.2b internal defect hard-case top-up pack generation

- 范围：执行 22.2 hard-case top-up 计划，只生成 internal/buried defect hard-case COMSOL top-up、schema validation、v3_hardcase assembly、registry/manifest 和 route decision；未训练，未更新 `CURRENT_BASELINE.md`，未把 internal defect 写成 baseline。
- COMSOL 结果：计划 `120` 行，成功 `120/120`，失败 `0`。成功覆盖 `internal_cuboid=55`、`internal_ellipsoid=50`、`internal_sphere=15`；覆盖 shallow/deep_plus、compact、medium/large、center-region neighbor、cuboid/ellipsoid confusion 和 full-shift catastrophic target strata。
- 数据边界：top-up NPZ 写入 ignored `data/comsol_mfl/generated/internal_defect_hard_case_topup_pack/internal_defect_hard_case_topup_pack_v1.npz`，v3_hardcase assembled NPZ 写入 ignored `data/comsol_mfl/generated/internal_defect_pilot_pack_v3_hardcase/comsol_internal_defect_pilot_pack_v3_hardcase.npz`，均未提交。
- assembled dataset：`dataset_id=comsol_internal_defect_pilot_pack_v3_hardcase`，source rows `240`，top-up rows `120`，assembled rows `360`，split=`train 240 / val 60 / test 60`，`train_ready_candidate=true`，`baseline_ready=false`。
- registry/manifest：已新增 `results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json` 和 `COMSOL_DATA_REGISTRY.md` 条目；禁止 latest/newest discovery、automatic mainline training、baseline update、current baseline replacement。
- route decision：可以进入 `22.3 hard-case augmented internal training gate`；真实 internal inference smoke 和 baseline transition 继续暂缓。

## 2026-05-30 Stage 22.3 internal defect hard-case augmented training gate

- 范围：在 `comsol_internal_defect_pilot_pack_v3_hardcase` 上训练和评估 hard-case augmented internal model；未运行 COMSOL，未生成或修改 data/NPZ，未提交 checkpoint/preview/notes，未修改 `CURRENT_BASELINE.md`。
- 数据入口：通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_internal_defect_pilot_pack_v3_hardcase.manifest.json` 显式加载，禁止 latest/newest scan。数据为 `N=360`，split=`train 240 / val 60 / test 60`，source v2=`240`，hard-case top-up=`120`。
- old B2 reference：21.9 B2 artifact 在 v3_hardcase test 上 total normalized MAE `0.515047`，L/W/D MAE `0.915 / 1.308 / 0.111 mm`，burial_depth MAE `0.558 mm`，center component MAE `2.174 mm`，shape accuracy/F1 `0.833333 / 0.841143`；catastrophic failure `12/60`，geometry_branch_failure `3/60`，center p95/max `12.077 / 22.544 mm`，burial p95/max `1.693 / 2.096 mm`。
- training gate：固定 21.7 B2 feature-fusion architecture；H1 为标准 hard-case augmented training，H2 只在 train split 使用 `row_origin=hardcase_topup_v1` 做轻量 sample weighting。正式模型输入仍只包含 `delta_b/BxByBz` 和 delta_b-derived features；shape/burial/size/aspect/split/sample_id/hard-case target 均未作为模型输入。
- selected result：validation-only 选择 `H2_B2_hardcase_tail_weighted` seed `42`，best epoch `294`。test total normalized MAE `0.421782`，L/W/D MAE `0.792 / 1.038 / 0.080 mm`，burial_depth MAE `0.531 mm`，center component MAE `1.709 mm`，shape accuracy/F1 `0.766667 / 0.778163`。
- tail result：catastrophic failure 从 old B2 `12/60` 降到 `9/60`，geometry_branch_failure 从 `3/60` 降到 `2/60`，center p95/max 从 `12.077 / 22.544 mm` 降到 `8.886 / 14.608 mm`；burial p95 从 `1.693 mm` 降到 `1.599 mm`，但 burial max 从 `2.096 mm` 升到 `2.861 mm`，shape F1 明显下降。
- decision：hard-case top-up 有效降低部分 mean/tail risk，但没有达到 stable inference gate；catastrophic rate `15%` 仍高于目标 `<=5%`，geometry branch 仍非零。internal branch 仍只能称 benchmark candidate，不是 stable inference model，更不是 baseline。下一步唯一建议是第二轮 hard-case top-up 或 tail-specific refinement，真实 internal inference smoke 继续暂缓。
- review：独立只读 review 通过，0 个 must-fix；2 个 should-fix 已修复：summary 明确 H2 的 train-only sample weighting，input validator 增加了 `delta_b.reshape` 和 delta_b-derived feature source 检查。
## 2026-05-30 Stage 22.4 shape-preserving internal defect tail strategy

- 范围：只做 analysis artifact generation、trade-off audit、strategy design 和 route decision；没有训练，没有运行 COMSOL，没有生成或修改 data/NPZ，没有提交 checkpoint/preview/notes，也没有修改 `CURRENT_BASELINE.md`。
- 输入证据：显式使用 `comsol_internal_defect_pilot_pack_v3_hardcase` manifest、22.3 hard-case selected predictions、old B2 reference comparison、22.0/22.1 reference metrics 和 22.3 failure cases。
- 22.3 H2 复核：`H2_B2_hardcase_tail_weighted` seed `42` 的 test total normalized MAE 是 `0.421782`，shape acc/F1 是 `0.766667 / 0.778163`，catastrophic failure 是 `9/60`，geometry_branch_failure 是 `2/60`，center p95/max 是 `8.886 / 14.608 mm`，burial p95/max 是 `1.599 / 2.861 mm`。
- 机制判断：H2 的 hard-case weighting 确实把旧 B2 的 center p95/max 从 `12.077 / 22.544 mm` 降到 `8.886 / 14.608 mm`，但 shape branch 被破坏，shape F1 从旧 B2 的 `0.841143` 降到 `0.778163`，并且 burial max 从 `2.096 mm` 退化到 `2.861 mm`。
- failure 分布：catastrophic failure 有 deep_plus、compact、medium/large 倾向，但不是单一 strata；继续盲目 second top-up 不是首选。
- 策略设计：S1-S5 全部列入策略矩阵，唯一主推荐是 `S3_two_stage_freeze_shape_then_tail_regress`，即先保护 shape classifier / shared encoder，再训练 center/burial tail heads。
- route decision：唯一下一步是 `A_train_freeze_shape_then_tail_regression_model`；shape-confidence router 作为后续推理安全层，第二轮 hard-case top-up 只在 freeze-shape 后仍有集中 strata failure 时考虑。
- baseline 状态：H2 不是 stable inference model，internal defect 仍是独立 benchmark branch，不是 `CURRENT_BASELINE`，不混入 surface / near-surface RBC baseline。
- review：独立只读 review 通过，无 must-fix；已按 should-fix 将三个脚本明确标注为 analysis artifact generator。
## 2026-05-30 Stage 22.5 freeze-shape internal tail-regression model

- 范围：在 `comsol_internal_defect_pilot_pack_v3_hardcase` 上训练 freeze-shape + tail-regression 候选；没有运行 COMSOL，没有生成或修改 data/NPZ，没有保存 checkpoint，没有提交 preview/notes，也没有修改 `CURRENT_BASELINE.md`。
- 输入边界：正式 F1/F2/F3 只使用 frozen B2 latent/logits/predictions 和 delta_b-derived feature latent；true `shape_type`、burial/size/aspect、split、sample_id 不作为正式模型输入。F4 使用 true shape 但只作为 oracle diagnostic，不能被选中。
- reference：B2 artifact 在 v3_hardcase test 上 total MAE `0.515047`，shape F1 `0.841143`，catastrophic `12/60`，geometry `3/60`；22.3 H2 total MAE `0.421782`，shape F1 `0.778163`，catastrophic `9/60`，geometry `2/60`。
- candidate screen：seed `42` 下 validation-only 选择 `F2_freeze_shape_train_center_burial_with_residual` 进入 multi-seed；F1/F3 未过 validation guard，F4 是 oracle only。
- multi-seed：F2 在 seeds `42/123/2026` 上复跑，validation-only 选择 seed `42`，best epoch `4`。
- selected result：test total normalized MAE `0.510008`；L/W/D MAE `1.014 / 1.327 / 0.109 mm`；burial_depth MAE `0.587 mm`；center_xyz component MAE `1.970 mm`；shape accuracy/F1 `0.816667 / 0.824172`。
- tail result：center p90/p95/max `7.162 / 8.940 / 22.017 mm`；burial p90/p95/max `1.517 / 1.841 / 2.490 mm`；catastrophic failure `11/60`；geometry_branch_failure `4/60`。
- 结论：freeze-shape 确实避免了 H2 式 shape collapse，shape F1 从 H2 `0.778163` 回升到 `0.824172`，但没有解决 tail failure；center max 接近 B2，burial p95/max 仍退化，catastrophic 和 geometry branch 都劣于 H2。
- route decision：不是 stable internal inference candidate，也不是 stronger benchmark candidate；下一步唯一建议是 tail-specific refinement plus uncertainty/output gate，不进入真实 internal inference smoke，不更新 baseline。
- review：独立只读 review 通过，无 must-fix；已补注释说明 group metadata 只用于 reporting，不进入模型输入。

## 2026-05-30 Stage 22.6 internal defect tail-risk / uncertainty gate

- 范围：基于 `comsol_internal_defect_pilot_pack_v3_hardcase`、B2/H2/F2 既有 prediction CSV 和 delta_b-derived features 做 tail-risk / abstention gate；没有训练主模型，没有运行 COMSOL，没有生成或修改 data/NPZ，没有提交 checkpoint/preview/notes，也没有修改 `CURRENT_BASELINE.md`。
- 输入边界：risk gate 只使用推理时可得信号，包括跨模型 shape disagreement、center/burial/LWD disagreement、F2 预测参数范围异常、delta_b amplitude/energy、feature z-score anomaly；true shape、burial/size/aspect、split、sample_id 只用于 target/分组审计，不进入 gate 输入。
- selected gate：validation-only 选择 `random_forest_small`，threshold=`0.07046389`；train-only scaler 已记录到 `results/metrics/internal_defect_tail_risk_gate_model_contract.json`，并补齐 `f2_pred_train_range_violation_count` 所需 train label range 常量。
- test 结果：catastrophic failure recall `1.000`，geometry_branch_failure recall `1.000`，false alarm rate `0.417`，coverage retained `0.283`。accepted subset 的 center p95/max 从 `8.940 / 22.017 mm` 降到 `4.569 / 5.290 mm`，burial p95/max 从 `1.841 / 2.490 mm` 降到 `0.590 / 0.911 mm`。
- 合同：新增 `results/summaries/internal_defect_uncertainty_inference_contract.md`，明确 high-risk 样本只能输出 `abstain_need_review`，不得给出确定 center/burial 结论。
- route decision：可以进入 `A_internal_inference_smoke_with_abstention`，但只能作为带风险门控的 smoke；internal branch 仍不是 stable inference model，更不是 baseline。
- review：独立只读 review 通过，无 must-fix；确认无 label leakage、validation-only threshold、test final only、forbidden artifacts 干净。

## 2026-05-30 Stage 22.7 internal defect inference smoke with abstention

- 范围：打通 internal defect 的带拒判推理流程；没有训练 internal 主模型，没有运行 COMSOL，没有生成或修改 data/NPZ，没有更新 `CURRENT_BASELINE.md`，也没有提交 checkpoint/preview/notes。
- 数据入口：显式使用 `dataset_id=comsol_internal_defect_pilot_pack_v3_hardcase`、registry/manifest、`results/manifests/internal_defect_b2_inference_artifact_manifest.json` 和 22.6 risk gate contract；禁止 latest/newest scan。
- risk gate artifact：22.6 只有 contract，没有可加载 sklearn 模型。本轮按固定协议确定性恢复 `random_forest_small` risk gate，pickle 写入 ignored `checkpoints/internal_defect_tail_risk_gate_artifacts/internal_defect_tail_risk_gate_random_forest_v22_6.pkl`，只提交 manifest `results/manifests/internal_defect_tail_risk_gate_artifact_manifest.json`。
- runner：新增 `scripts/run_internal_defect_inference_with_abstention.py`。推理输出包含 raw B2 prediction、`risk_score`、`inference_status`；`risk_score >= 0.07046389` 时 status=`abstain_need_review`，不得给稳定 center/burial 结论。
- smoke：v3_hardcase test split `60` 行，accepted `17`，abstained `43`，coverage retained `0.283`。catastrophic recall `1.000`，geometry_branch recall `1.000`，false alarm `0.381`。
- no-abstention B2 对比：full-set center p95/max `12.077 / 22.544 mm`，accepted subset 降到 `4.832 / 4.962 mm`；full-set burial p95/max `1.693 / 2.096 mm`，accepted subset 降到 `0.605 / 1.106 mm`；accepted shape accuracy/F1 `0.941 / 0.915`。
- contract：新增 `results/summaries/internal_defect_inference_abstention_contract.md`，明确 internal inference 不是 stable all-sample predictor，高风险样本只能输出 raw prediction + warning。
- route decision：唯一下一步是 `A_internal_real_sample_metadata_alignment_with_abstention`；可以做真实样品 metadata alignment，但仍暂缓直接真实样品推理。
- review：独立只读 review 通过，无 must-fix；确认 risk gate 无 label leakage、threshold 未用 test 重选、checkpoint artifact 未 staged。

## 2026-05-30 Stage 22.8 internal richer-observation feasibility plan

- 范围：只做 richer-observation 仿真诊断规划；没有运行 COMSOL，没有训练，没有生成或修改 data/NPZ，没有更新 `CURRENT_BASELINE.md`，internal defect 仍是独立 benchmark branch。
- 输入证据：显式引用 22.0–22.7 failure / abstention 结果和 `comsol_internal_defect_pilot_pack_v3_hardcase` manifest。22.7 coverage retained 只有 `0.283`，high-risk `43/60`，catastrophic 与 geometry-branch recall 均为 `1.000`。
- failure-to-observation mapping：abstained `43`，failure cases `39`；failure 分布为 `internal_cuboid=17`、`internal_ellipsoid=14`、`internal_sphere=8`，`deep_plus=18`，`large=20`，`compact=22`，`elongated_y=15`。
- 机制判断：center/lateral tail 优先对应 `R1_more_y_lines`；burial/size tail 对应 `R2_multi_liftoff`；cuboid/ellipsoid 与 elongated aspect confusion 先用 R1 检查，R3 multi-scan-direction 作为第二优先级；R4 multi-magnetization 暂缓。
- COMSOL 可行性：现有 internal generator 已有 `scan_line_y_m` 和 `sensor_z_m` 入口，R1/R2 可低风险进入 22.9；R3/R4 需要新增 direction/source 协议，不进第一轮主包。
- 第一轮 diagnostic pack plan：target `30` base geometries / `180` rows，fallback `24` / `144`。每个 base 都有 6 个 paired variants：`R0_3line_z0p008`、`R1_5line_z0p008`、`R1_9line_z0p008`、`R2_5line_z0p006`、`R2_5line_z0p010`、`R2_5line_z0p012`。
- route decision：唯一下一步是 `22.9_richer_observation_COMSOL_diagnostic_pack_generation`；仍不进入训练或真实样品推理。
- review：独立只读 review 通过，无 must-fix；确认 plan-only、安全边界和 180-row plan gate 均满足要求。

## 2026-05-30 Stage 22.9 internal richer-observation COMSOL diagnostic pack

- 范围：执行 22.8 设计的 internal richer-observation COMSOL diagnostic pack；没有训练，没有更新 `CURRENT_BASELINE.md`，没有把 internal defect 写成 baseline。
- 数据包：新增 `dataset_id=comsol_internal_defect_richer_observation_pack_v1`，通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_internal_defect_richer_observation_pack_v1.manifest.json` 显式引用，禁止 latest/newest discovery。
- COMSOL 结果：planned/success `180/180`，base count `30`，complete paired base count `30`，失败 `0`。
- paired variants：每个 base 都有 `R0_3line_z0p008`、`R1_5line_z0p008`、`R1_9line_z0p008`、`R2_5line_z0p006`、`R2_5line_z0p010`、`R2_5line_z0p012` 六个观测版本。
- 观测覆盖：scan line count 覆盖 `3/5/9`，liftoff 覆盖 `0.006/0.008/0.010/0.012 m`；NPZ 使用 padded `(N,3,9,201)` 并记录 `scan_line_count` 和 `scan_line_mask`。
- validation：`validation_passed=true`，`failed_blockers=[]`，`delta_b=b_defect-b_no_defect` 校验通过，Bx/By/Bz finite，shape/burial/size/aspect 覆盖满足 diagnostic 目标。
- registry：`status=diagnostic_pack_generated`，`train_ready_candidate=false`，`baseline_ready=false`，allowed use 仅为 `schema_validation, explicit_richer_observation_diagnostic`。
- route decision：可进入 23.0 richer-observation evaluation gate；训练、真实样品推理和 `CURRENT_BASELINE.md` 更新继续暂缓。
- review：独立只读复审通过，无 must-fix；COMSOL 既有无关 dirty items 保持未 staged、未回滚、未提交。

## 2026-05-31 Stage 23.1 internal richer-observation training gate

- 范围：在 `comsol_internal_defect_richer_observation_pack_v1` 上执行 richer-observation training gate；没有运行 COMSOL，没有生成或修改 data/NPZ，没有提交 checkpoint/preview/notes，也没有修改 `CURRENT_BASELINE.md`。
- 数据入口：通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_internal_defect_richer_observation_pack_v1.manifest.json` 显式加载，禁止 latest/newest scan。22.9 richer-observation pack 仍是 diagnostic pack，不是 baseline，也不是自动训练集。
- 23.0 前置选择：补齐 reference evaluation / route decision 后，validation-only 选择 `R1_plus_R2_combined`，包含 `R1_9line_z0p008`、`R1_5line_z0p008` 和 `R2_5line_z0p006/0p010/0p012`；test 指标只报告，不参与配置选择。
- 输入边界：正式模型只使用 `delta_b/BxByBz`、`scan_line_mask`、`sensor_z_m` 和 delta_b-derived features。true `shape_type`、burial/size/aspect、split、sample_id 只用于 supervision / metrics，不作为模型输入。
- split / normalization：30 个 base 按 `base_group_id` 聚合后做确定性分组，split=`train 20 / val 5 / test 5`；raw、feature、target normalization 均只使用 train split 统计量。
- selected result：validation-only 选择 `O3_richer_observation_tail_aware` seed `2026`，best epoch `5`。Test total normalized MAE `0.629543`，L/W/D MAE `1.112 / 1.168 / 0.126 mm`，burial_depth MAE `0.879 mm`，center_xyz component MAE `2.095 mm`，shape accuracy/F1 `0.800000 / 0.600000`。
- tail result：test center p95/max `7.314 / 7.531 mm`，burial p95/max `1.966 / 2.180 mm`，catastrophic failure `4/5`，geometry_branch_failure `1/5`。R1/R2 richer observation 在该 30-base diagnostic scope 内没有形成 stable inference candidate。
- decision：23.1 不是 internal benchmark candidate upgrade，也不是 stable internal inference model。下一步唯一建议是 `23.2_internal_multi_scan_direction_plan`，优先检查 R3 multi-scan-direction 是否能补足 cuboid/ellipsoid 和 elongated aspect 的几何分支信息。
- review：独立只读 review 首轮发现 23.0 配置选择存在 test 指标 tie-breaker，已修为只按 validation score 和固定配置顺序选择；复审通过，无 must-fix。
## 2026-05-31 Stage 23.2 internal multi-scan-direction observation plan

- 范围：只做 multi-scan-direction diagnostic plan；未运行 COMSOL，未训练，未生成或修改 data/NPZ/checkpoint/preview/notes，未更新 `CURRENT_BASELINE.md`。
- 输入证据：23.1 在 `comsol_internal_defect_richer_observation_pack_v1` 上的 `R1_plus_R2_combined` / `O3_richer_observation_tail_aware` 没有通过 stable inference gate，test total normalized MAE `0.629543`，shape F1 `0.600000`，catastrophic failure `4/5`，geometry_branch_failure `1/5`。
- 机制判断：22.9 的 R1/R2 只扩展 y-line 和 liftoff，没有提供正交扫描方向；现有 COMSOL richer generator 虽有 `scan_direction` metadata，但 sensor points 仍固定为 x-path，23.2b 必须新增真正的 direction-aware sensor point builder。
- 候选设计：D0 复用 single-direction reference；D1 为 dual-direction 5-line；D2 为 dual-direction 9-line；D3 dual-direction + multi-liftoff 留作二阶段；D4 multi-magnetization 暂缓。
- 第一轮 diagnostic pack：复用 22.9 的 30 个 base，只补生成 `D1_y_scan_5line_z0p008` 和 `D2_y_scan_9line_z0p008`，planned rows `60`，fallback `24 base / 48 rows`；组装时与既有 `R1_5line_z0p008` / `R1_9line_z0p008` x_scan 配对。
- 数据契约：23.2b assembled tensor 规划为 `delta_b=(N,3,2,9,201)`，包含 `direction_names=[x_scan,y_scan]`、`direction_mask`、`scan_line_mask`、path/line coordinate axis 和 padded scan-line metadata。
- route decision：唯一下一步是 `23.2b_internal_multi_scan_direction_generation`；训练暂缓到 23.3，真实 internal sample inference 继续暂缓，internal branch 仍不是 baseline。
- review：独立只读 review 通过，无 must-fix；已按建议明确 forbidden artifact check 只覆盖非 ignored tracked/untracked 路径。

## 2026-05-31 Stage 23.2b internal multi-scan-direction diagnostic pack

- 范围：执行 23.2b y_scan COMSOL diagnostic pack generation、x/y dual-direction assembly、schema validation、registry/manifest 和 route decision；没有训练，没有更新 `CURRENT_BASELINE.md`，没有把 internal defect 写成 baseline。
- COMSOL 结果：计划 `60` 行，成功 `60/60`；30 个 base 全部具备 `D1_y_scan_5line_z0p008` 与 `D2_y_scan_9line_z0p008`，失败 `0`。
- 方向化坐标：23.2b 新增真正的 `direction-aware` y_scan 采样，`path_coordinate_axis=y`、`line_coordinate_axis=x`，采样点为 `(x_line, y_path, sensor_z_m)`，不是只写 `scan_direction=y_scan` metadata。
- 组装结果：新数据集 `comsol_internal_defect_multi_scan_direction_pack_v1` 已注册；与 22.9 既有 x_scan `R1_5line_z0p008` / `R1_9line_z0p008` 配对后，assembled `delta_b` shape 为 `(60,3,2,9,201)`，`direction_names=[x_scan,y_scan]`。
- validation：`validation_passed=true`，`delta_b=b_defect-b_no_defect` 最大误差 `1.1641532182693481e-10`，`direction_aware_y_scan_coordinate_check=true`，D1/D2 paired completeness `30/30`。
- registry：`status=diagnostic_pack_generated`，`train_ready_candidate=false`，`baseline_ready=false`，allowed use 仅为 `schema_validation, explicit_multi_scan_direction_diagnostic`。
- route decision：可进入 23.3 internal multi-scan-direction diagnostic evaluation；训练、真实 internal sample inference 和 `CURRENT_BASELINE.md` 更新继续暂缓。

## 2026-05-31 Stage 23.3 internal multi-scan-direction diagnostic evaluation

- 范围：只做 `comsol_internal_defect_multi_scan_direction_pack_v1` 的 diagnostic evaluation；未运行 COMSOL，未训练 formal model，未生成或修改 data/NPZ，未更新 `CURRENT_BASELINE.md`，internal defect 仍是独立 diagnostic / benchmark branch。
- 数据入口：显式使用 `COMSOL_DATA_REGISTRY.md` 与 `results/manifests/comsol_internal_defect_multi_scan_direction_pack_v1.manifest.json`；assembled `delta_b` shape 为 `(60,3,2,9,201)`，方向维为 `[x_scan,y_scan]`，30 个 base 均 D1/D2 paired complete。
- paired audit：y_scan 对每个 base 都提供非冗余正交观测；D1 5-line 的 mean y/x RMS 为 `1.077851`、mean corr 为 `0.225959`，D2 9-line 的 mean y/x RMS 为 `0.996417`、mean corr 为 `0.197233`。
- feature separability：dual_xy_9line 的 shape nearest-neighbor consistency 从 single_x_9line 的 `0.600000` 升到 `0.700000`，ambiguous neighbor rate 从 `0.866667` 降到 `0.833333`；dual_xy_5line 相对 single_x_5line 没有改善 shape NN consistency。
- lightweight probe：本轮 probe 只用于路线判断，不保存 checkpoint，不形成 formal candidate。validation-selected best test config 是 `single_x_9line`，test total normalized MAE `0.499454`，shape accuracy/F1 `0.800000 / 0.600000`，center p95/max `5.678 / 6.341 mm`，burial p95/max `1.522 / 1.652 mm`，catastrophic/geometry count `2 / 0`。
- dual-direction 对比：dual_xy_5line 的 center p95 比 x-only 5line 改善 `0.923 mm`，但 burial p95 退化 `0.595 mm` 且 shape F1 下降；dual_xy_9line 的 burial p95 比 x-only 9line 改善 `0.115 mm`，但 center p95 退化 `4.627 mm` 且 shape F1 下降。
- route decision：dual-direction 没有清晰优于 x-only，23.3 不进入 23.4 training gate。下一步唯一建议是 `D_generate_multi_magnetization_diagnostic_pack`，用更高信息量的 source / magnetization observation 继续诊断。
- review：独立只读 review 通过，无 must-fix；一个 should-fix 已处理，把 cuboid/ellipsoid confusion 和 center/burial tail 的中间判断从 yes 改为 mixed，避免误读。

## 2026-05-31 Stage 23.5 internal multi-magnetization diagnostic evaluation

- 范围：只评估 `comsol_internal_defect_multi_magnetization_pack_v1` 的 dual-magnetization 诊断价值；未运行 COMSOL，未进行正式模型训练，未生成或修改 data/NPZ，未提交 checkpoint/preview/notes，也未修改 `CURRENT_BASELINE.md`。
- 数据入口：通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_internal_defect_multi_magnetization_pack_v1.manifest.json` 显式加载；assembled `delta_b` shape 为 `(60,3,2,9,201)`，30 个 base 全部具备 `mag_x/mag_y` paired completeness。
- paired audit：`mag_y` 是真实 COMSOL source `Je` 改向后的 paired observation，不是 metadata-only；M1/M2 都是 `30/30` nonredundant。M1 mean mag_y/mag_x RMS 为 `0.235954`，mean corr 为 `0.024762`；M2 mean ratio 为 `0.242644`，mean corr 为 `0.002597`。
- feature separability：dual 5line 的 shape NN consistency 与 mag_x 5line 持平为 `0.700000`；dual 9line 从 mag_x 9line 的 `0.600000` 升到 `0.766667`，cuboid/ellipsoid cross NN rate 从 `0.521739` 降到 `0.304348`，但 ambiguous neighbor rate 升到 `0.900000`，center NN distance 也变大。
- lightweight diagnostic probe：本轮 probe 只用于 observation 诊断，不保存 checkpoint，不形成 formal candidate。标签只用于 supervision / validation selection / metrics，不作为输入；输入仅为 `delta_b/BxByBz` 派生特征。
- single vs dual：`mag_x_5line_only` test total MAE `0.504394`、shape F1 `0.500000`、catastrophic `3/5`；`dual_mag_xy_5line` 退化到 total MAE `0.623999`、catastrophic `3/5`。`mag_x_9line_only` test total MAE `0.499454`、shape F1 `0.600000`、catastrophic `2/5`；`dual_mag_xy_9line` 退化到 total MAE `0.558467`、shape F1 `0.500000`、catastrophic `3/5`。
- route decision：dual magnetization 提供非冗余信号，但没有稳定优于 single-mag reference；23.5 不推荐进入 23.6 multi-magnetization formal training gate。唯一下一步是保留 abstention-only route 并暂停 internal refinement，除非后续重新定义 observation / label 或提出新的诊断假设。
- review：独立只读 review 通过，无 must-fix；确认无 COMSOL、无 data/NPZ mutation、无 label-as-input leakage、无 forbidden staged artifacts。`CURRENT_BASELINE.md` 保持 surface / near-surface true 3D RBC baseline 不变。
## 2026-06-03 Stage 25.14 surface RBC targeted +120 top-up calibration

- scope: top-up pack generation/validation branch only. No training, no `CURRENT_BASELINE.md` update, no assembled dataset creation, and no `v3_240 + topup_v1_120` assembly in this stage.
- plan: generated `comsol_true_3d_rbc_surface_targeted_topup_v1_120` plan with 120 rows, split `80/20/20`, role quotas `balanced_interior=60`, `hard_depth_aspect=24`, `edge_position=24`, `old_distribution_anchor=12`, `sensor_z_m=0.008`, axis order `Bx,By,Bz`, scan lines `[-0.001,0.0,0.001]`, and 201 x points.
- implementation: added targeted plan, validation, and route-decision scripts in PINN; added chunked COMSOL top-up orchestrator with per-worker temp/log/inventory/partial-NPZ isolation in COMSOL_Multiphysics_MCP.
- dry run: `workers=4`, 24-row calibration dry-run split to `chunk_00..chunk_03` as expected.
- calibration: real `workers=1`, 24 rows, mesh pre-stage `24/24` pass, COMSOL forward `22/24` pass. Two sample-level geometry blockers were classified as `sample_geometry_failure`: `surface_rbc_targeted_008_balanced_interior_sharp_medium_narrow` and `surface_rbc_targeted_022_balanced_interior_round_deep_balanced`.
- blocker decision: calibration is not zero-failure, so full generation was not run. `workers=2/4` calibration and full 120 are blocked until same-signature replacements are proposed and rerun.
- replacement signatures: 008 requires `balanced_interior|medium|narrow|sharp|interior`; 022 requires `balanced_interior|deep|balanced|round|interior`. Replacement must preserve role, depth_bin, aspect_bin, curvature_template, and edge_position_bin.
- route decision: `can_enter_plus120_training_gate=false`; `creates_assembled_dataset=false`; `CURRENT_BASELINE_update=false`.
