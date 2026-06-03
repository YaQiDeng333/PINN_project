# NEXT_STEP

## 2026-06-03 after Stage 25.15b surface multi-pit label-v3 target failure audit

Next step: run **A. 25.16 label-v3b derivation + validator, no training**. The audit shows the fix is still inside `PINN_project`: raw component masks/depths and ownership diagnostics are available, but v3 soft/valid support must be tightened before another training gate.

The root mechanism is component-identity leakage. Label v3 expanded positive support enough to avoid the 25.13 near-empty failure, but the soft OR/raw union ratio averages `2.010499`, all `112/112` samples are union-like under soft support, and test merged rate is `1.000000` across separated, close, touching, partially_overlapping, component_count=2, and component_count=3 slices. That means this is not a topology-only hard case and not a threshold artifact.

25.16 should derive v3b targets only: keep an exclusive hard core, split valid region into hard core / boundary halo / ignore overlap, restrict soft halo to non-overlapping boundary context, keep partially-overlapping shared pixels as ignore or diagnostic unless ownership confidence is explicit, and supervise depth only on hard core plus narrow owned boundary. `CURRENT_BASELINE.md` remains unchanged.

## 2026-06-03 after Stage 25.15 surface multi-pit label-v3 training gate

Next step: run **C. return to label-v3 derivation or generator/export schema; do not continue loss tuning**. The 25.15 gate used the 25.10 loss mainline plus label-v3 supervision, so this failure should not be answered by reusing the 25.11/25.12 rebalance stack or by increasing model capacity.

The decisive mechanism is a trade from near-empty collapse to merge collapse. Label v3 improved raster support enough that component Dice rose from `0.005536` to `0.034245` and union Dice from `0.002829` to `0.061694` versus 25.13, but merged rate became `1.000000`, depth RMSE worsened to `0.001106223 m`, and recall stayed at `0.674419`. Component_count=3, partially_overlapping, and touching_boundary test slices all have merged rate `1.000000`.

The next audit should check whether v3 soft bands/SDF/valid regions are too permissive, whether overlap/contact targets need explicit non-merge ownership confidence, and whether generator/export schema must provide component-local separation fields beyond soft support. `CURRENT_BASELINE.md` remains unchanged.

## 2026-06-03 after Stage 25.14 surface multi-pit label-v3 derivation + validator

Next step: run **A. 25.15 label-v3 training gate using the 25.10 loss mainline + label-v3 supervision**. Do not use the 25.11/25.12 rebalance stack, do not update `CURRENT_BASELINE.md`, and do not frame this as a baseline transition.

The key result is that v3 directly addresses the v2 sparsity mechanism without changing data files. V2 hard component support averaged `99.851695` pixels with minimum `47`; v3 soft/local support averages `210.110169` pixels with minimum `126`, so the mean support ratio is `2.203184` and the minimum ratio is `1.697183`. Empty slots remain clean, existing slots keep depth-valid support, duplicate hard ownership stays resolved (`297 -> 0`), and raw OR/max still reproduces union mask/depth exactly.

25.15 should train only the controlled next gate: same dataset, same split, same component-set representation, same architecture family, and 25.10 loss mainline adapted to `component_mask_target_v3_soft`, `component_sdf_target_v3`, `component_valid_region_mask`, and `component_depth_target_v3`. Three-component, partially-overlapping, and touching-boundary rows must stay separately reported.

## 2026-06-03 after Stage 25.13b surface multi-pit generator/label schema audit after target-v2 collapse

Next step: run **A. 25.14 label-v3 derivation + validator, no training**. The 25.13b audit shows the raw labels are sufficient to derive better supervision inside `PINN_project`; do not return to COMSOL yet and do not continue loss tuning.

The core mechanism is sparse hard-target collapse. Target v2 does exactly what it promised (`duplicate ownership 297 -> 0`, overlap-depth-conflict `271 -> 0`), but each component still averages only about `99.85` positive pixels on a `64x128` grid, or `0.012188928` positive fraction. Existing slots are not empty, so the failure is not missing masks; the problem is that hard binary component-local labels lack soft boundary/context support.

25.14 should derive and validate label schema v3 from existing raw labels: preserve raw component masks and ownership maps, add soft mask or SDF targets, component valid-region masks, overlap-region masks, contact-boundary masks, and component depth targets with valid regions. Keep union mask/depth as OR/max evaluation targets. `CURRENT_BASELINE.md` remains unchanged.

## 2026-06-03 after Stage 25.13 surface multi-pit component-set target-v2 training gate

Next step: run **C. return to generator/label schema; do not continue loss tuning**. Target v2 successfully resolves duplicated ownership in the labels, but the current component-level supervision still does not produce usable raster masks under the 25.10 loss mainline.

The decisive failure is mask collapse, not merge recovery. Duplicate component ownership is cleared (`297 -> 0`) and overlap-depth-conflict is cleared (`271 -> 0`), but test component Dice falls to `0.005536` and union Dice to `0.002829`; merged rate `0.000000` is therefore a degenerate near-empty-mask effect, not healthy component separation. Recall also drops to `0.674419`, missed rises to `0.325581`, and extra rises to `0.292683` versus the 25.10 mainline.

The next design stage should revisit generator/label schema semantics before more training: component-level mask/depth targets likely need stronger positive support or alternate supervision beyond ownership-resolved hard masks. Do not keep tuning loss weights, and do not update `CURRENT_BASELINE.md`.

## 2026-06-03 after Stage 25.12b surface multi-pit component raster/depth target redesign

Next step: run **A. 25.13 target-v2 training gate using the 25.10 loss mainline**. Do not continue the 25.11/25.12 rebalance stack: target ownership, not more loss pressure, is the next controlled variable.

25.12b found that v1 labels are globally consistent but component-local targets are ambiguous in overlap pixels. Component OR exactly reconstructs the union mask, max(component depth) exactly reconstructs union depth, empty slots are zero, and center-to-mask-centroid error is small. The blocking issue is duplicated component ownership: `25/112` samples have overlap, including `18/24` partially-overlapping rows and `10/12` three-component rows.

25.13 should transform targets at loader/training time without writing a new NPZ: build `component_ownership_map`, train `component_mask_target_v2` and `component_depth_target_v2` as ownership-resolved foreground targets, keep union mask/depth as OR/max diagnostics, and report three-component rows separately. `CURRENT_BASELINE.md` remains unchanged.

## 2026-06-02 after Stage 25.12 surface multi-pit component-separation-aware rebalance training

Next step: run **C. rollback to 25.10 loss mainline and redesign component raster/depth targets before further training**. The component-separation-aware loss reduced merged rate from `0.900000` to `0.700000`, but it bought that by worsening recall, missed rate, extra rate, and union Dice, so it is not a viable training route.

The key failure is that loss-only separation pressure does not recover stable component masks on the current target formulation. 25.12 test recall is `0.744186` versus `0.860465` in 25.11 and `0.837209` in 25.10; missed rate is `0.255814`, extra rate `0.200000`, component Dice `0.108790`, union Dice `0.138075`, and depth RMSE `0.000501023 m`. Three-component test rows still merge completely (`merged_rate=1.0`).

The next design stage should return to the 25.10 loss mainline as the stable comparator, then redesign component raster/depth targets before more training: inspect component-mask target separability, overlap ownership, per-component depth semantics, and whether component masks need explicit non-overlap/ownership labels rather than only loss penalties. `CURRENT_BASELINE.md` remains unchanged.

## 2026-06-02 after Stage 25.11b surface multi-pit component-set merge-collapse audit

Next step: run **A. 25.12 component-separation-aware rebalance training**. The 25.11b audit shows the blocker is not generic capacity and not just touching/overlap topology: the rebalance made union-level agreement easier while component-level separation stayed underconstrained.

The key evidence is direct: 25.11 improved union Dice (`0.130480 -> 0.166233`), recall (`0.837209 -> 0.860465`), missed rate, and extra rate, but merged rate collapsed from `0.200000` to `0.900000`, component Dice stayed flat/slightly worse, and depth RMSE worsened. Test newly merged rate is `0.700000`; separated rows also newly merge at `0.750000`, so the issue is not confined to touching or overlapping cases.

25.12 should keep the same representation and model size unless separately justified, but change the objective schedule: delay or cap union mask loss, add component-separation regularization, penalize predicted component-mask overlap, keep topology-aware merge penalties for touching/overlap, and stage foreground-normalized depth after masks separate. `CURRENT_BASELINE.md` remains unchanged.

## 2026-06-02 after Stage 25.11 surface multi-pit component-set mask/depth loss rebalance training

Next step: run **B. 25.11b targeted rebalance or topology-focused failure audit**. The first mask/depth rebalance produced a real but incomplete signal: union mask Dice improved, component recall improved, and extra/missed rates improved, but component masks, depth, and merge behavior are still not stable.

25.11 kept the component-set route fixed: same architecture, same `K=3`, same `72/20/20` split, same manifest-driven dataset loading, and same Hungarian matching. The `mask_depth_rebalance_v1` loss added foreground-balanced component mask supervision, union mask supervision, and valid-mask-only depth losses without exporting checkpoints or touching `CURRENT_BASELINE.md`.

The gate remains `PARTIAL`: test recall `0.837209 -> 0.860465`, missed rate `0.162791 -> 0.139535`, extra rate `0.142857 -> 0.097561`, and union mask Dice `0.130480 -> 0.166233`; however component mask Dice is flat/slightly worse (`0.109562 -> 0.108737`), depth RMSE worsened (`0.000243315 -> 0.000673627 m`), merged rate worsened (`0.200000 -> 0.900000`), and three-component test rows still merge (`merged_rate=1.0`). 25.11b should isolate whether the union-mask term is encouraging merged blobs, whether depth should be staged/renormalized, and whether topology-aware penalties are needed before another training route.

## 2026-06-02 after Stage 25.10b surface multi-pit component-set failure audit

Next step: run **B. enter 25.11 mask/depth loss rebalance training**. The audit found that component existence and coarse geometry have learning signal, while projected-mask/depth supervision is not being converted into aligned component masks; do not jump to a larger model or baseline transition.

25.10b ruled out the most likely implementation bugs first. Target component masks reconstruct the union mask with IoU `1.000000`, center-to-mask centroid error is only `0.000059604 m`, and empty-slot mask/depth sums are `0.0`. The training script also uses min-over-slot-permutations and masks parameter, shape, mask, and depth losses by existing slots, so slot permutation and empty-slot punishment are not the leading failure mode.

The failure ranking is: primary `loss imbalance`, secondary `data scarcity failure` for three-component rows, tertiary `genuine hard topology failure` on touching/overlap. Three-component test coverage is only `3` rows and all predicted two components (`merged_rate=1.0`), so 25.11 should keep a three-component/topology audit slice, but the first route should rebalance mask/depth loss and diagnostics before changing representation or expanding training scale. `CURRENT_BASELINE.md` remains unchanged.

## 2026-06-02 after Stage 25.10 surface multi-pit component-set training gate

Next step: run **B. 25.10b failure audit for merged/missed, overlap/touching, slot permutation, and three-component rows**. Do not continue directly to stronger training until the audit explains why component existence/geometry learned while mask/depth and merged-component behavior stayed weak.

25.10 trained a lightweight `C1_fixed_K_component_set` gate on `comsol_surface_multipit_component_set_pilot_v1` with fixed split `72/20/20` and `K=3`. The result is `PARTIAL`: test component recall reached `0.837209`, missed rate `0.162791`, extra rate `0.142857`, and mean predicted component count `2.1`, so the model is not empty or single-component collapsed. The blocker is raster/topology quality: test component mask Dice is only `0.109562`, union mask Dice `0.130480`, validation union mask Dice `0.071413`, depth RMSE barely beats degenerate baselines, and three-component test rows have merged rate `1.0`.

25.10b should inspect matched/unmatched samples by separation, topology, component_count, and slot assignment. The key questions are whether the mask/depth head is underpowered, whether the loss is over-weighting component existence/geometry, whether overlap/touching labels need separate supervision, and whether K=3 slot permutation is unstable. `CURRENT_BASELINE.md` remains unchanged.

## 2026-06-02 after Stage 25.9b surface multi-pit component-set top-up pack

Next step: run **A. enter 25.10 component-set training gate**. This is the only recommended next step after the 25.9b pack validation, but it must be a separately gated training stage; do not auto-train and do not update `CURRENT_BASELINE.md`.

25.9b generated the approved top-up `comsol_surface_multipit_topup_pack_v1` with `96/96` successful COMSOL rows and split `63/17/16`. The assembled component-set pilot is `comsol_surface_multipit_component_set_pilot_v1`, with `N=112`, split `72/20/20`, fixed `K=3`, old seed `16`, top-up `96`, `train_ready_candidate=true`, and `baseline_ready=false`.

The training gate should use Hungarian matching and permutation-invariant component losses over existence, center, `L/W/D`, rotation, shape family/local profile, projected mask, and depth-grid targets. Multi-pit still receives no six-parameter RBC success credit; frozen 20.85 and the forward-refinement runner remain comparators only.

## 2026-06-02 after Stage 25.9 surface multi-pit component-set branch plan

Next step: run **A. execute multi-pit COMSOL top-up generation**. This is the only recommended route after the 25.9 plan, and it should remain a future generation stage with explicit approval, not training and not a baseline transition.

25.9 kept the branch plan-only: no COMSOL, no training, no data/NPZ mutation, no checkpoint/preview/notes artifact, and no `CURRENT_BASELINE.md` update. The label audit found `16` existing multi-pit seed rows, all `component_count=2`, split `9/3/4`, with enough component JSON, union mask, and union depth labels for a C1 seed audit. The gaps are per-component rotation, component-level masks/depth grids, and explicit separation/touching/overlap/topology labels.

The selected representation is `C1 fixed_K_component_set` with `K=3` and Hungarian matching. The planned top-up is `N=96` with assembled `N=112` and split `72/20/20`; coverage must add close, touching, partially overlapping, mixed-depth, mixed-size, x/y/diagonal, and disconnected/merged/touching topology cases. Multi-pit remains outside six-parameter RBC success credit, and frozen 20.85 plus forward-refinement runner are comparator-only for this branch.

## 2026-06-02 after Stage 25.8 surface forward-refinement report package

Next step: run **A. component-set branch for multi-pit**. The report package confirms the 25.7 forward-refinement runner is stable as a companion for RBC-representable surface model failures, while multi-pit remains a representation problem that needs component-set output.

25.8 did not train, did not run COMSOL, did not mutate data/NPZ, did not commit preview PNGs, and did not update `CURRENT_BASELINE.md`. The report keeps the boundaries fixed: frozen 20.85 is still the current baseline, the forward-refinement runner is post-hoc companion refinement, and the oracle is an evaluation ceiling only.

The report target subset remains `82` rows, with profile RMSE `0.000509518351056 -> 0.000220386413188 m`, Er-like `2.80015739379 -> 0.909941363416`, IoU/Dice `0.32360140234/0.480524080842 -> 0.578523465369/0.709451842351`, and forward residual `70.5944261489 -> 0.564105036956`. Multi-pit/component-set rows stay `not_suitable_for_rbc_refinement` and receive no RBC success credit.

## 2026-06-02 after Stage 25.7 surface forward-refinement inference runner

Next step: run **A. surface refinement visualization/report package**. Keep it as a reporting package around the verified companion runner: no COMSOL, no main neural training, no data/NPZ mutation, no checkpoint artifact commit, and no `CURRENT_BASELINE.md` update.

25.7 exported `surface_forward_refinement_inference_artifact_v1` and implemented the runtime chain `frozen 20.85 baseline -> predicted RBC six params -> observed delta_b feature-space forward-consistency refinement -> refined profile/projected mask`. The runner reproduced 25.6 checked per-sample fields with `max_abs_diff=0`. On the 82 target rows, profile RMSE is `0.000509518351056 -> 0.000220386413188 m`, Er-like is `2.80015739379 -> 0.909941363416`, IoU/Dice improves from `0.32360140234/0.480524080842` to `0.578523465369/0.709451842351`, and forward residual moves from `70.5944261489` to `0.564105036956`.

The boundary remains fixed: this is a companion/post-hoc surface refinement runner, not a baseline replacement. Multi-pit/component-set rows are marked `not_suitable_for_rbc_refinement` and remain a future component-set branch. Unknown real samples can only report `refinement_applied`, not representable success, unless oracle/label or human confirmation exists.

## 2026-06-02 after Stage 25.6 surface forward-refinement formal benchmark

Next step: run **A. export surface forward-refinement inference artifact / runner**. Keep it as a no-baseline-transition runtime artifact for the fixed 25.6 candidate: no COMSOL, no main neural training, no data/NPZ mutation, and no `CURRENT_BASELINE.md` update unless a separate baseline-transition request is explicitly approved later.

25.6 replayed the fixed 25.5 protocol exactly: `ridge_param_only_linear_alpha_10`, `alpha=10.0`, `lambda_param=1.0`, frozen 20.85 six-parameter initialization, and post-hoc optimization over `L_m/W_m/D_m/wLD/wWD/wLW`. The formal target subset is still the 82 `rbc_representable_but_model_fail` rows. Baseline/refined/oracle profile RMSE is `0.000509518351056 / 0.000220386413188 / 0.0000784896954944 m`, Er-like is `2.80015739379 / 0.909941363416 / 0.28925522333`, and IoU/Dice improves from `0.32360140234/0.480524080842` to `0.578523465369/0.709451842351`.

The runner export must preserve the same boundaries: refinement inputs are observed `delta_b` features plus frozen 20.85 predicted six params; labels are not runtime inputs; multi-pit remains a future `component_set` branch; already-pass references need a guard because they are monitoring-only and can degrade under unconditional refinement.

## 2026-06-02 after Stage 25.5 surface feature-space forward-consistency refinement diagnostic

Next step: run **A. formal no-baseline-transition benchmark for the 25.5 F0/R1 refinement candidate**. Keep it as a benchmark candidate audit: no COMSOL, no main neural training unless separately approved, no data/NPZ mutation, and no `CURRENT_BASELINE.md` update.

25.5 selected `ridge_param_only_linear_alpha_10` plus `R1_low_dim_param_refinement` with `lambda_param=1.0`. On the 82 `rbc_representable_but_model_fail` targets, profile RMSE improved from `0.000509518351056 m` to `0.000220386413188 m`, Er-like from `2.80015739379` to `0.909941363416`, IoU/Dice from `0.32360140234/0.480524080842` to `0.578523465369/0.709451842351`, and forward residual from `70.5944261489` to `0.564105036956`. All `10/10` gates passed and the RBC-like control did not degrade.

Multi-pit stays outside the RBC-refinement success gate: the 16 `multi_pit_two_component_surface_defect` rows are excluded negative controls and remain a future `component_set` branch. The 22 already-pass references are monitoring rows, not success-credit rows, so the next audit should keep them visible while deciding whether the F0/R1 candidate is stable enough for any later benchmark discussion.

## 2026-06-02 after Stage 25.4 surface forward-consistency refinement plan

Next step: run **A. execute 25.5 feature-space forward-consistency refinement diagnostic**. Keep it diagnostic-only: no training, no COMSOL, no data/NPZ mutation, and no `CURRENT_BASELINE.md` update.

25.4 narrowed the target to the 82 `rbc_representable_but_model_fail` samples from 25.3. The 22 already-pass rows are references, while the 16 multi-pit/component-set representation failures are negative controls and cannot count as RBC-refinement success. The selected route is `F0_feature_space_consistency + R1_low_dim_param_refinement`: start from frozen 20.85 six parameters, adjust only `L_m/W_m/D_m/wLD/wWD/wLW`, and judge success by profile/mask metrics plus forward-feature residual alignment.

25.5 must report metrics by target role, shape type, split, and representation target. It must prove target-subset profile RMSE / Er-like / IoU / Dice improvements, protect RBC-like control, and keep multi-pit outside the success gate. Component-set decoding for multi-pit remains a later branch.

## 2026-06-02 after Stage 20.99 internal / buried defect feasibility schema

Next step: run **A. execute internal COMSOL smoke pack after metadata confirmation**. First confirm the required metadata and labels, then in a later approved stage generate only the 6-12 sample internal smoke pack; do not train and do not update `CURRENT_BASELINE.md`.

20.99 fixed the boundary: internal / buried defects are not surface RBC defects. The required internal labels are `L_m`, `W_m`, `D_m` or cavity size, `burial_depth_m` / `depth_to_surface_m`, `defect_center_xyz_m`, `shape_type`, profile descriptor or cavity mask, and `ground_truth_method`. Hard blockers remain missing burial depth, no no-defect reference, Bz-only without a degraded-branch declaration, unknown coordinates relative to the scan surface, missing `sensor_z_m`, and missing ground truth.

The recommended first representation is `shape_type + L/W/D + burial_depth + center_xyz`. The smoke plan is 12 rows across `internal_ellipsoid`, `internal_cuboid`, and `sphere_like`, with shallow/medium/deep burial levels and required Bx/By/Bz plus `delta_b=b_defect-b_no_defect`. Bz-only is allowed only as a low-capability diagnostic branch.

## 2026-06-02 after Stage 25.3 current baseline generalization audit

Next step: run **D. forward-consistency refinement plan** for the surface shape-extension branch. Keep `component_set` as a required sub-branch for multi-pit, but do not jump straight into training a 20.85-style six-parameter model.

25.3 showed the real split: the RBC oracle can represent `104/120` samples and `80/96` non-RBC samples, so most single-component non-RBC failures are not caused by the six-parameter shape family alone. The frozen 20.85 baseline fails broadly, with pass `22/120`, non-RBC pass `19/96`, and RBC-like control pass only `3/24`; therefore the immediate blocker is model/generalization and forward consistency under the new pilot distribution. The exception is multi-pit: `16/16` are `rbc_not_representable`, with component merge proxy `1.000000`.

The next work should be plan-only unless separately approved: define how forward-consistency refinement would use the pilot without label leakage, how RBC-like control is protected, and how multi-pit component-set handling is separated. Do not train, do not update `CURRENT_BASELINE.md`, and do not treat the 25.2 pilot as an automatic training dataset.

## 2026-06-02 after Stage 25.2 surface shape-extension COMSOL pilot pack

Next step: run **25.3 current baseline generalization audit** on `comsol_surface_shape_extension_pilot_v1`. Use the frozen 20.85/20.86 surface RBC baseline as the audited model, report where it fails on non-RBC-like surface defects, and keep the result as an audit, not a baseline transition.

25.2 completed the COMSOL pilot pack with `N=120`, train/val/test=`72/24/24`, `rbc_like_smooth_pit=24`, and six non-RBC-like families at `16` each. Boolean subtract, mesh precheck, solve, finite `Bx/By/Bz`, `delta_b=b_defect-b_no_defect`, profile/depth labels, projected masks, topology labels, and explicit `representation_target` passed validation. The generated NPZ/data remain ignored and uncommitted.

25.3 should measure 20.85 failure modes: profile RMSE, component recall, edge/corner metrics, multi-pit merge rate, crack-like miss rate, RBC-like control stability, and forward residual behavior. Do not train, do not update `CURRENT_BASELINE.md`, and do not treat `comsol_surface_shape_extension_pilot_v1` as an automatic training dataset.

## 2026-06-01 after Stage 25.1 surface shape-extension dataset plan

Next step: run **25.2 surface shape-extension COMSOL pilot generation**. This is the only recommended next step, and it is allowed only as pilot generation after the 25.1 taxonomy, schema, feasibility, route, and acceptance gates passed review.

25.1 completed plan-only surface shape-extension design. The pilot target is `N=120` with train/val/test=`72/24/24`, `rbc_like_smooth_pit=24`, and six non-RBC-like families at `16` each: flat-bottom, sharp-wall/boxy, asymmetric, elongated/crack-like, multi-pit/two-component, and irregular corrosion. `N=84` is only reduced feasibility because it cannot satisfy RBC-like >=24 plus seven shape families >=12; the minimum full-coverage fallback is `N=96`.

The label contract now separates RBC-compatible six-parameter controls from non-RBC targets: `profile_basis`, `depth_grid`, `component_set`, and `polygon_or_contour`. Multi-pit requires component-level labels, crack-like cases require aspect/rotation, irregular corrosion keeps depth-grid/profile targets, and forward-consistency remains a later gate. Do not train, do not update `CURRENT_BASELINE.md`, and do not treat the 25.2 pilot as a baseline transition.

## 2026-06-01 after Stage 25.0 surface Piao-NLS branch closeout

Next step: run **25.1 surface shape-extension dataset plan**. This is the only recommended next step: define the taxonomy, labels, split coverage, and acceptance gates for non-RBC-like surface defects before any decoder or forward-consistency implementation.

25.0 closes the Piao-NLS branch as diagnostic/QC/classical-comparator work. Keep `nlslite_*` for QC and classical comparison, keep NLS-full-compatible as a future richer-observation interface, and stop small NLS feature-fusion tweaks as the mainline. 24.1 cannot replace 20.85 because it worsens profile RMSE and Er-like error, and 24.2 remains a diagnostic candidate rather than a baseline transition.

25.1 should cover asymmetric pits, flat-bottom defects, crack-like slots, multi-pit / multi-component surface damage, profile-depth labels, projected-mask QA, topology labels, forward residual gates, and train/val/test coverage. Do not train, do not run COMSOL, do not generate data/NPZ, and do not update `CURRENT_BASELINE.md` until a later explicitly approved generation or baseline-transition stage.

## 2026-06-01 after Stage 24.2 surface RBC NLS-lite feature fusion diagnostic

Superseded by Stage 25.0: the earlier formal `F1_late_fusion` rerun suggestion is closed as a diagnostic branch note, not the current next step. The current next step is the Stage 25.1 surface shape-extension dataset plan at the top of this file.

24.2 used fixed v3_240 registry/manifest loading and fused `delta_b/BxByBz + nlslite_*` with train-only scalers, validation-only selection, and test-final-only reporting. Validation selected `F1_late_fusion`; multi-seed selected seed `123`. Test total normalized MAE was `0.598309`, L/W/D MAE `1.816667/1.657295/0.654960 mm`, wMAE `0.183249`, profile RMSE `0.000317238 m`, Er-like `0.267248`, and IoU/Dice `0.793564/0.877942`. This improves over 20.85/20.77 and 24.1 on the main reported metrics.

Keep the role split clear: 24.0A is the three-line NLS-lite feature source, 24.0B is the future NLS-full-compatible interface for richer y-line ROI data, 24.1 is the feature-only comparator, and 24.2 is a diagnostic surface feature-fusion candidate. Do not call it exact Piao full NLS, do not write a baseline, and do not update `CURRENT_BASELINE.md`.

## 2026-06-01 after Stage 24.1 surface RBC Piao-style NLS-lite feature baseline

下一步唯一建议：进入 **24.2 NLS-lite feature fusion diagnostic**，把 `nlslite_*` 作为神经模型的辅助输入做 bounded fusion gate；不要替换 `CURRENT_BASELINE.md`，也不要把 24.1 classical feature baseline 写成 exact Piao NLS。

24.1 validation 选中 `lssvm_rbf_alpha_0p1_gamma_0p00171821`（`LS-SVM-like-RBF`）。test total normalized MAE=`0.654046`，优于 20.85/20.77 的 `0.678014` 和 20.81 的 `0.667888`；Dice=`0.862988`，优于 20.85/20.77 但略弱于 20.81；profile RMSE=`0.000445182 m`，仍弱于 20.85/20.77 的 `0.000387737 m`。真正的价值是 classical comparator 与 curvature/w 参数补充信号，而不是 profile-depth baseline replacement。

24.2 如果启动，应继续固定 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240`，只通过 registry + manifest 加载；feature scaler 和 target scaler 仍必须 train-only，candidate/seed/epoch selection 仍必须 validation-only，test 只做最终报告。24.0B full-compatible framework 继续作为未来 richer y-line ROI 接口，不参与 24.2 输入。

## 2026-06-01 after Stage 24.0B surface RBC NLS full-compatible framework

Next step: keep NLS-full-compatible as a gated interface and do not claim full Piao mode until surface RBC has a richer y-line ROI pack.

The current v3_240 pack is three-axis but only `scan_line_count=3`, so the new framework correctly reports `full_feature_ready=false`, `degraded_mode=true`, and `degraded_mode_reason=scan_line_count_lt_5`. The full-compatible minimum is `M>=5`, and the recommended full-candidate count is `M>=9`, with aligned Bx/By/Bz ROI matrices, validated `sensor_x_m`, validated `scan_line_y_m`, no missing values, and validated equations.

This branch should run in parallel with the existing NLS-lite / Piao-inspired 3-line path. Do not replace NLS-lite, do not update `CURRENT_BASELINE.md`, and do not describe current 3-line features as exact Piao full NLS. A future surface richer y-line pack can reuse this schema/validator/extractor interface when full ROI data is available.

## 2026-06-01 after Stage 24.0A surface RBC NLS-lite feature extractor

下一步唯一建议：进入 **24.1 surface RBC NLS-lite feature baseline**，但仍然保持不训练神经模型、不替换 `CURRENT_BASELINE.md`，先把 24.0A 的 `nlslite_*` 稳定物理特征作为 feature-only / hybrid diagnostic baseline 输入做正式 gate。

24.0A 已确认 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 可以通过 registry + manifest 显式加载，`delta_b=(240,3,3,201)`，feature_count=`291`，finite fraction=`1.0`，fit_success_rate=`1.0`，fallback_rate=`0.0`。最强信号集中在 `Bx` 的 line width / amplitude / energy：`nlslite_Bx_yneg_half_peak_width_m` 对 `L_m`/`W_m` 最强，`nlslite_Bx_yneg_abs_peak` 对 `D_m`/profile depth 最强，curvature 相关性较弱但有诊断价值。正式 feature CSV 没有 target labels，labels 只用于 correlation audit。

不要把这一步写成 exact Piao 18-feature reproduction；当前边界是 `exact_piao_nls=false`、`piao_nls_lite=true`，因为 v3_240 只有三条 `scan_line_y`。真实实验预处理可以沿用这些 delta_b-only 特征，但前提是 Bx/By/Bz 轴序、三条扫描线、sensor_x、no-defect reference 和 gain/calibration 条件先对齐。

## 2026-05-31 after Stage 23.5 internal multi-magnetization diagnostic evaluation

下一步唯一建议：**暂停 internal refinement，保留 abstention-only route，不进入 23.6 multi-magnetization training gate**。

23.5 证明 `mag_y` 是真实 COMSOL source `Je` 改向后的非冗余 paired observation，但它没有在 diagnostic probe 中稳定优于 single-mag reference。5line dual 从 `mag_x_5line_only` 的 test total MAE `0.504394` 退化到 `0.623999`；9line dual 从 `mag_x_9line_only` 的 `0.499454` 退化到 `0.558467`，catastrophic failure 也从 `2/5` 升到 `3/5`。虽然 dual 9line 的 feature separability 有改善，shape NN consistency 从 `0.600000` 升到 `0.766667`，但 center/burial tail 和 probe 指标没有同步改善。

因此不要把 23.5 写成正式模型候选，也不要启动 23.6 训练或真实 internal sample inference。当前 internal branch 仍只能作为 diagnostic / benchmark branch，`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline。

## 2026-05-31 after Stage 23.4 internal multi-magnetization diagnostic pack

下一步唯一建议：进入 **23.5 internal multi-magnetization diagnostic evaluation**。

23.4 已完成 multi-magnetization COMSOL diagnostic pack generation：planned/success `60/60`，30 个 base 均有 `M1_mag_y_5line_z0p008` 与 `M2_mag_y_9line_z0p008`，并与既有 R1 reference 形成 paired pack。关键不是只写了 `mag_y` metadata，而是 COMSOL source `Je` 从 nominal `["0","1e6[A/m^2]","0"]` 显式改为 orthogonal `["1e6[A/m^2]","0","0"]` 后重新求解。assembled `delta_b` shape 为 `[60,3,2,9,201]`，`validation_passed=true`，`can_enter_23_5=true`。

23.5 只应做 diagnostic evaluation：比较 multi-magnetization 是否相对 single-source / multi-scan-direction 进一步改善 shape confusion、center/burial tail 和 geometry_branch failure。不要在 23.5 直接训练，不要接真实 internal sample，不要更新 `CURRENT_BASELINE.md`；internal defect 仍是独立 diagnostic / benchmark branch。

## 2026-05-30 after Stage 22.2 targeted internal hard-case top-up plan

下一步唯一建议：**执行 22.2b targeted COMSOL hard-case top-up pack generation**。

22.2 已把 22.0/22.1 的 tail failure 转成可执行数据矩阵：目标 top-up N=`120`，minimum usable N=`72`，重点覆盖 `internal_cuboid/internal_ellipsoid` confusion、`compact`、`medium/large`、`shallow/deep_plus` 和 center-region neighbor cases。9 个 target 的计划配额已经逐项对齐为 `24/20/18/16/14/10/10/4/4`，route decision 也明确检查逐 target quota 和 minimum。

不要在 22.2b 之前继续模型 refinement，也不要进入真实 internal inference smoke。当前问题更像 hard-case coverage 不足，而不是 schema 定义错误；22.2b 应只生成 targeted COMSOL top-up pack、inventory、validation 和 registry/manifest，仍不更新 `CURRENT_BASELINE.md`，internal defect 仍是独立 benchmark branch。

## 2026-05-30 after Stage 22.1 shape-conditioned internal model

下一步唯一建议：**targeted internal hard-case top-up**。

22.1 说明单纯把 B2 改成 shape-conditioned / shape-specific heads 还不够稳定。T3_shape_specific_heads 由 validation-only 选中 seed `123`，test total normalized MAE 从 B2 的 `0.395256` 降到 `0.357371`，center p95 从 `8.309 mm` 降到 `5.999 mm`；但 hard gate 没过：catastrophic failure 仍是 `5/40`，geometry_branch_failure 仍是 `1/40`，center max 还升到 `10.468 mm`，burial p95/max 也退化到 `1.690 / 1.848 mm`。

因此不要把 T3 作为 stable inference model，也不要进入真实 internal inference smoke。下一步应围绕 22.0/22.1 的 hard cases 做 targeted COMSOL top-up：优先覆盖 compact、large/medium、shallow/deep_plus、cuboid/ellipsoid 易混和 center 远偏样本，然后再回到更强 two-stage branch。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline。

## 2026-05-29 after Stage 22.0 internal defect B2 failure audit

下一步唯一建议：**B. shape-conditioned / two-stage internal model**。

22.0 证明 B2 不能直接当作 stable inference model 使用。它仍是 internal benchmark candidate，但 test split 的 tail failure 很重：`catastrophic_failure=5/40`，`geometry_branch_failure=1/40`，center_xyz error 的 mean/median/p95/max 为 `3.096 / 3.033 / 8.309 / 8.785 mm`，burial_depth error 为 `0.413 / 0.260 / 1.266 / 1.674 mm`。最关键的坏样本是 `internal_pilot_091`：true `internal_cuboid` 被预测成 `internal_ellipsoid`，同时 burial 和 center 都明显偏移。

因此下一步不要直接进入真实 internal inference smoke，也不要把 B2 写成 baseline。先做 shape-conditioned / two-stage internal model：先稳定 shape branch，再让 center/burial head 在对应 shape 分支内回归；center/burial focused refinement 可以作为 secondary ablation。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline，internal defect 仍是独立 benchmark branch。

## 2026-05-29 after Stage 21.5 internal defect benchmark report

下一步唯一建议：**B. improve burial-depth head/model**。

21.5 已把 internal v2_240 的候选角色收口：neural candidate 是正向 benchmark candidate，test total normalized MAE `0.406366`，略优于 selected feature baseline `0.416406`，并且 shape accuracy/F1 为 `1.000000 / 1.000000`，center_xyz MAE `1.380 mm` 也优于 feature baseline `1.560 mm`。但真正的下一步 blocker 是 burial_depth：feature baseline `0.472 mm` 明确优于 neural `0.595 mm`，group-level audit 中 burial_depth 也是 feature baseline 系统性更强。

因此不要直接 baseline transition，也不要先扩数据或接真实实验。下一步应做 burial-depth focused diagnostic：例如 burial-depth head/loss、feature-fusion、shape-conditioned burial head 或把 delta_b-derived physical features 作为辅助分支；shape-conditioned model 作为第二优先级。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline，internal defect 仍是独立 branch。

## 2026-05-29 after Stage 21.4 internal defect v2_240 training gate

下一步唯一建议：**进入 internal defect v2_240 formal benchmark/report，不要直接升 baseline**。

21.4 已证明 v2_240 上三轴 `Bx/By/Bz delta_b` 对 internal/buried defect 的主要标签有可学习信号：selected neural seed `42` 的 test total normalized MAE 为 `0.406366`，略优于 selected feature baseline `0.416406`；shape accuracy/F1 达到 `1.000000 / 1.000000`；L/W/D MAE 为 `0.761 / 0.947 / 0.093 mm`，center_xyz MAE 为 `1.380 mm`。但 burial_depth 单项仍是 feature baseline 更强：`0.472 mm` vs neural `0.595 mm`。

因此下一步不是继续扩数据，也不是把 internal defect 写成 `CURRENT_BASELINE`。更稳的路线是 formal benchmark/report：复核 seed 稳定性、分组误差、feature-vs-neural trade-off 和 burial_depth 单项风险，再决定是否做 shape-conditioned / feature-fusion internal model。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline。

## 2026-05-29 after Stage 21.3b internal defect dataset v2_240 pack generation

下一步唯一建议：**进入 21.4 internal defect v2_240 training gate**。

21.3b 已生成并验证 `comsol_internal_defect_pilot_pack_v2_240`：source N=96，top-up COMSOL planned/success `168/168`，按 quota 选 144 行进入 assembled pack，最终 N=240，split=`160/40/40`。新的 split 已解决 21.2 的 blocker：train/val/test 都覆盖 `internal_sphere/internal_ellipsoid/internal_cuboid`、`shallow/medium/deep/deep_plus`、`small/medium/large`，ellipsoid/cuboid 在每个 split 内也覆盖 compact / elongated_x / elongated_y。

21.4 才做训练。当前 v2_240 只是 internal/buried defect 独立分支的 `train_ready_candidate=true` 数据包，`baseline_ready=false`；不要更新 `CURRENT_BASELINE.md`，不要把 internal defect 混入 surface / near-surface RBC baseline。

## 2026-05-29 after Stage 21.3 internal defect dataset expansion plan

下一步唯一建议：**执行 21.3b internal defect top-up COMSOL pack generation，然后 assemble/validate `comsol_internal_defect_pilot_pack_v2_240`**。

21.3 已把 21.2 的 split blocker 固化成扩展方案：复用 source N=96，但丢弃旧 split；计划生成 top-up N=168，其中按 quota 选 144 行进入 v2_240，最终 assembled N=240，split=`160/40/40`。新的 split 必须让 train/val/test 都覆盖 `internal_sphere/internal_ellipsoid/internal_cuboid`、四档 burial depth、三档 size，并让 ellipsoid/cuboid 的三种 aspect 在每个 split 都出现。

21.3b 只负责 COMSOL top-up generation、v2 assembly、schema validation、registry/manifest 和 route decision；仍不训练、不更新 `CURRENT_BASELINE.md`。21.4 才能在 v2_240 上重新跑 internal defect training gate。

## 2026-05-28 after Stage 21.2 internal defect training gate

下一步唯一建议：**扩展 internal defect dataset，并重做真正分层的 train/val/test split**。

21.2 已证明 internal / buried defect 分支存在可学习信号：neural gate 在综合 score 上优于 mean baseline 和 selected feature baseline，center_xyz 与 shape_type 有明显信号。但现有 N=96 的 split 不能支撑稳健结论：val/test 都只有 `internal_cuboid`，burial_depth 也没有完整覆盖；同时纯回归 total MAE 和 burial_depth 上，`svr_rbf_C10` feature baseline 仍强于 neural。

因此不要把 21.2 升级为 baseline，也不要继续只小修当前 Conv1D。下一阶段应生成更大的 internal pilot/formal pack，保证每个 split 同时覆盖 `internal_sphere/internal_ellipsoid/internal_cuboid`、四档 burial depth、三档 size 和主要 aspect，再重新跑 training gate。`CURRENT_BASELINE.md` 仍保持 surface / near-surface true 3D RBC baseline。

## 2026-05-28 after Stage 21.1 internal / buried defect pilot pack

下一步唯一建议：**进入 21.2 internal defect training gate**。

21.1 已把 21.0 smoke 扩展为 `comsol_internal_defect_pilot_pack_v1`：COMSOL planned/success `96/96`，覆盖 `internal_sphere/internal_ellipsoid/internal_cuboid`，四档 `burial_depth_level=shallow/medium/deep/deep_plus`，并固定 split `train/val/test=64/16/16`。Bx/By/Bz、`delta_b=b_defect-b_no_defect`、internal labels、registry/manifest validation 均通过，`train_ready_candidate=true`，但 `baseline_ready=false`。

21.2 应只做显式 internal training gate：输入仍是三轴 `delta_b`，标签是 `shape_type + L/W/D + burial_depth + center_xyz` 等 internal schema 字段；不要更新 `CURRENT_BASELINE.md`，不要把 internal pilot 写成 surface RBC baseline，也不要接入真实实验数据。

## 2026-05-28 after Stage 21.0 internal / buried defect COMSOL smoke pack

下一步唯一建议：**进入 21.1 internal defect pilot pack 设计与生成**。

21.0 已经跑通 internal / buried defect feasibility smoke：`comsol_internal_defect_smoke_pack_v1` 完成 12/12 COMSOL rows，覆盖 `internal_sphere`、`internal_ellipsoid`、`internal_cuboid`，并通过 Bx/By/Bz、`delta_b=b_defect-b_no_defect`、internal labels、registry/manifest validation。它仍不是训练集，也不是 baseline；`CURRENT_BASELINE.md` 继续保持 20.85 surface / near-surface true 3D RBC profile-depth baseline。

21.1 应把 smoke 放大为 internal pilot pack：继续要求三轴 `Bx/By/Bz`、no-defect reference、`sensor_z_m`、`burial_depth_m / depth_to_surface_m`、`defect_center_xyz_m`、`shape_type` 和 ground truth method。真实实验数据继续暂缓；Bz-only 只能作为低能力诊断分支，不能替代三轴 internal 主线。

## 2026-05-28 after Stage 20.99 internal / buried defect feasibility schema

下一步唯一建议：**执行 internal COMSOL smoke pack 前，先确认可采 metadata 和 ground truth 定义**。

20.99 已把 internal / buried defect 从当前 surface RBC baseline 中拆出来。当前 baseline 仍只适用于 surface / near-surface RBC-style 缺陷；内部缺陷必须单独定义 `burial_depth_m` / `depth_to_surface_m`、`defect_center_xyz_m`、`shape_type`、空腔尺寸或体素 mask、以及 ground truth method。

推荐的下一步不是训练，也不是把现有铁块直接送进 20.96 runner，而是确认实验端能提供：三轴 `Bx/By/Bz`、匹配 no-defect reference、`sensor_z_m`、扫描坐标系、试件几何和埋深/缺陷中心标签。确认这些字段后，再执行 12-sample internal COMSOL smoke pack；如果只能提供 Bz，则只能走低能力诊断分支，不能宣称进入 true 3D internal baseline。

## 2026-05-28 after Stage 20.98 real-data manifest dry run

下一步唯一建议：**C. create internal defect feasibility schema**。

20.98 只做真实数据 manifest dry run，没有读取真实信号数组、没有生成 data/NPZ、没有训练、没有运行 COMSOL，也没有更新 `CURRENT_BASELINE.md`。用户现有铁块被明确标记为 `internal_defect_iron_block` / `internal_or_buried`，因此不适合直接进入当前 surface / near-surface RBC-style true 3D baseline。

当前 dry-run 的 `ready_for_inference=false`。硬 blocker 包括：缺真实 `Bx/By/Bz` 数组、缺匹配 no-defect reference、缺实测 `sensor_z_m`、缺轴顺序、缺三条 `scan_line_y_m`、缺 201 点 `sensor_x_m`、缺 Tesla 单位和坐标系、缺传感器对齐与 gain 状态、缺励磁设置；更关键的是缺陷位置属于 internal/buried 分支。不要把它强行送入 20.96 surface RBC 推理 runner。下一步应先定义 internal defect feasibility schema，包括 burial depth / depth-to-surface、内部缺陷标签、对应采集几何和 no-defect reference 规则。

## 2026-05-27 after Stage 20.92 liftoff-aware training gate

Next step: **inspect liftoff pack failure cases and the nominal/non-nominal trade-off before more COMSOL or real-data alignment**.

Stage 20.92 trained the liftoff-aware gate on `comsol_true_3d_rbc_liftoff_aug_pack_v1` with grouped `base_sample_id` splits. `C1_unconditioned_liftoff_aug` seed `123` was selected by validation. It improved non-nominal liftoff profile RMSE versus the fixed 20.85 baseline (`0.000874310 m -> 0.000659761 m`) and improved non-nominal Dice (`0.683351 -> 0.833129`), but it badly regressed nominal `0.008 m` profile RMSE (`0.000333059 m -> 0.000809011 m`).

This means 20.92 is useful evidence but not a robustness candidate upgrade. Keep `CURRENT_BASELINE.md` unchanged at the 20.85 true 3D RBC profile-depth baseline. Do not move to internal/buried defect feasibility yet, and do not claim real-data readiness. The next controlled step should audit failure cases by liftoff level/base geometry and decide whether the model needs a nominal-preserving loss, paired liftoff consistency, or a better sensor_z-conditioned protocol before generating more COMSOL data.

## 2026-05-27 after Stage 20.89 gain/amplitude calibration and augmentation gate

Next step: **20.90 liftoff/sensor-offset COMSOL diagnostic pack, with explicit gain/amplitude control caveat**.

Stage 20.89 showed that the current baseline is not noise-limited; it is amplitude/gain and Bx-amplitude sensitive. Calibration-only helped the stressed cases but cost too much clean profile accuracy: validation-selected `per_axis_rms_train_stats` reduced test gain 0.8 degradation from `123.845%` to `21.194%` and Bx 50% attenuation degradation from `141.577%` to `12.331%`, but clean profile RMSE degraded `21.194%`, above the `<=10%` gate.

In-memory augmentation also helped robustness but is not a baseline upgrade. Validation-selected `A2_axis_gain_aug` seed `123` reduced gain 0.8 degradation to `24.614%` and Bx 50% attenuation degradation to `59.279%`, but clean profile RMSE degraded `35.464%` and gain 1.2 degradation remained `38.768%`. Keep `CURRENT_BASELINE` unchanged at the 20.85 true 3D RBC profile-depth baseline. Treat `A2_axis_gain_aug` only as a non-baseline robustness diagnostic, not as a replacement model.

The next stage should measure physics-side variation instead of continuing small augmentation tweaks: run a controlled 20.90 liftoff / sensor-offset COMSOL diagnostic pack, while explicitly tracking amplitude normalization and Bx dependence. Real-data alignment remains blocked until there is a concrete amplitude calibration protocol.

## 2026-05-26 after Stage 20.88 observation perturbation robustness audit

Next step: **gain/amplitude calibration or augmentation planning, while preparing the 20.89 liftoff/sensor-offset COMSOL diagnostic pack**.

Stage 20.88 reused the recovered 20.88a frozen artifact and perturbed only in-memory v3_240 `delta_b`; no COMSOL, no training, no data/NPZ changes, and no `CURRENT_BASELINE.md` update. Clean replay matched the baseline, and noise <=10% was stable: noise 10% profile degradation `4.095415%`, Dice drop `-0.000252`. no-defect reference error and sensor_x jitter were also low-risk in this observation-space diagnostic.

The blocker is amplitude/channel sensitivity, not random noise. Global gain 0.8x degraded profile RMSE by `123.845240%`; `channel_attenuation_Bx_50pct` degraded profile RMSE by `141.577253%`; `channel_dropout_Bx_missing` caused Dice drop `0.163825`. Do not claim broad robustness yet. The next implementation step should either design gain normalization / amplitude calibration / augmentation for 20.92, or proceed with 20.89 small COMSOL liftoff/sensor-offset pack as the next physics diagnostic. Real-data alignment should stay behind these two gates.

## 2026-05-26 after Stage 20.88a

Next step: **return to 20.88 observation perturbation robustness audit using the recovered frozen baseline artifact**.

Stage 20.88a recovered the fixed 20.77/20.85 seed=42 inference artifact for `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`. The checkpoint and raw prediction artifact are intentionally stored only under the ignored path `checkpoints/true_3d_rbc_baseline_artifacts/`; the committable locator is `results/manifests/true_3d_rbc_baseline_inference_artifact_manifest.json`.

The artifact reload verification exactly reproduces the clean 20.85 metrics: test normalized MAE `0.6780143536818333`, profile_depth_rmse `0.0003877372636895579 m`, Er-like `0.3405436946031375`, L/W/D MAE `1.8918915996566796 / 2.1857599088778863 / 0.8002313476246901 mm`, projected mask IoU/Dice `0.7506502455785019 / 0.8477271366767738`. 20.88 can now perturb `delta_b` in memory and run frozen-model inference without retraining. Do not retrain inside 20.88, do not modify NPZ/data, and do not commit the ignored checkpoint or prediction artifact.

## 2026-05-26 after Stage 20.88 preflight blocker

Next step: **recover or export the frozen 20.77/20.85 baseline model artifact before robustness evaluation**.

20.88 stopped at preflight. Registry/manifest/schema checks passed for `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`, but the required frozen-model artifact was not available locally: no selected seed=42 true 3D RBC checkpoint and no sufficient raw prediction artifact that can rerun the model on perturbed `delta_b`.

Do not treat the clean per-sample metrics as a robustness result. The next step is a separate artifact recovery/export stage: first try to recover the 20.77/20.85 selected checkpoint; if that is impossible, explicitly approve a fixed 20.85 artifact-export rerun that saves a checkpoint/prediction artifact. Only after that should 20.88 observation perturbation robustness be rerun. Do not train inside 20.88.

## 2026-05-26 after Stage 20.87

Next step: **20.88 observation perturbation robustness audit on the current true 3D RBC baseline**.

Stage 20.87 was design-only: no COMSOL, no training, no data/NPZ generation, and no `CURRENT_BASELINE.md` change. The next actionable step is to perturb the existing v3_240 `delta_b` observations through explicit dataset_id / manifest loading and evaluate the frozen 20.86 baseline under observation-space stress only.

20.88 should start with additive noise `0/5/10/15/20%`, amplitude scaling / sensor gain error, baseline zero drift, no-defect reference subtraction error, channel dropout, and `sensor_x_resampling_jitter` as a diagnostic-only interpolation perturbation. Formal spatial-sampling, liftoff, scan-line offset, Bx/By/Bz misalignment, source-strength, and material/B-H claims require the later 20.89 COMSOL diagnostic pack. Internal/buried defects stay out of the current surface RBC baseline and should wait for 20.91 label/schema design.

## 2026-05-26 after Stage 20.86

Next step: **benchmark documentation and real-data alignment planning around the new true 3D RBC profile-depth baseline**.

Stage 20.86 promoted the 20.77/20.85 formal rerun candidate to `CURRENT_BASELINE`. The project baseline has transitioned from old v3_complex 2D mask/boundary prediction to true 3D RBC-style profile-depth reconstruction. The old 2D baseline remains an archived comparator; it was not deleted.

The new current baseline is fixed to `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240`, with Bx/By/Bz `delta_b` input `(N,3,3,201)`, Conv1D input `(N,9,201)`, and six RBC-style outputs `L_m/W_m/D_m/wLD/wWD/wLW`. Its headline metrics are profile/depth metrics: `profile_depth_rmse_m=0.000387737`, Er-like profile error `0.340544`, and L/W/D MAE `1.892/2.186/0.800 mm`. Projected mask Dice `0.847727` remains QA; wMAE `0.201076` remains auxiliary diagnostic.

Immediate next work should not be another training tweak. The useful next step is to prepare a concise benchmark/report narrative and plan real-data alignment or exact-Piao/representation follow-up under the new baseline scope. Keep `exact_piao_rbc=False` and do not claim real experimental deployment readiness.

## 2026-05-26 after Stage 20.85

Next step: **prepare paper/report display around the formal 20.77-profile benchmark candidate**.

Stage 20.85 reran the 20.77 neural candidate on `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` with the original architecture, loss, and validation-only selection protocol. Seeds `42/123/2026` completed; seed `42` was selected. The rerun exactly reproduced the 20.77 profile/depth result: test normalized MAE `0.678014`, L/W/D MAE `1.892/2.186/0.800 mm`, profile depth RMSE `0.000387737 m`, Er-like profile error `0.340544`, projected mask IoU/Dice `0.750650/0.847727`, and auxiliary wMAE `0.201076`.

The role split remains fixed: 20.77/formal rerun is the profile/depth benchmark candidate; 20.81 remains the projected-mask / visual comparator because its Dice is higher but profile RMSE is worse; 20.83 remains negative evidence for the tested profile-primary loss. This is still not a baseline replacement, and `CURRENT_BASELINE.md` must remain unchanged.

## 2026-05-26 after Stage 20.84

Next step: **A. keep 20.77 as profile/depth benchmark candidate for formal rerun**.

Stage 20.84 consolidated the existing 20.77 / 20.81 / 20.83 candidates without retraining, COMSOL, new data, NPZ changes, or baseline updates. The role split is now fixed:

- 20.77 remains the profile/depth main candidate: `profile_depth_rmse_m=0.000387737`, projected mask Dice `0.847727`.
- 20.81 remains the non-negative projected-mask / visual reference: Dice `0.866573`, but profile RMSE `0.000445297` is worse than 20.77.
- 20.83 remains negative evidence for the current R1 profile-primary loss path: Dice `0.868042` is numerically high, but profile RMSE `0.000409718` is worse than 20.77, so it cannot replace the profile/depth candidate.

The prediction gallery audit supports the same split: best-profile samples are genuinely low profile-error cases, but worst-profile and high-Dice/high-profile-error samples show that 2D projected mask quality is not enough to judge the 3D profile. Do not continue small tweaks to the current 20.83 profile-primary loss. If the route continues, run a formal benchmark rerun around 20.77 as the profile/depth candidate and keep 20.81 only as the visual/mask comparator. Do not update `CURRENT_BASELINE.md`.

## 2026-05-26 after Stage 20.83

Next step: **B. keep 20.77/20.81 candidate**.

Stage 20.83 tested `R1_six_params_profile_primary_loss` on `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`. The result is a negative training-gate result: the selected profile-primary candidate improved projected mask Dice (`0.868042`) but did not improve the primary 3D profile metric (`profile_depth_rmse_m=0.000409718` vs 20.77 `0.000387737`). Multi-seed was correctly skipped because the candidate screen gate failed.

Keep 20.77 as the profile reconstruction reference and 20.81 as the visual/mask comparator. Do not update baseline docs or `CURRENT_BASELINE.md`.

Immediate boundary for the next stage: no COMSOL or data expansion is justified by 20.83 alone. If continuing the true 3D route, the next useful work is a cleaner profile-native representation experiment, not another small loss-weight tweak.

## 2026-05-25 更新：第 20.82 后的下一步

第 20.82 已完成 true 3D RBC curvature label / output representation audit。本轮没有运行 COMSOL，没有生成或修改 data / NPZ，没有重新训练模型，没有建立 baseline，也没有修改 `CURRENT_BASELINE.md`。审计边界很明确：20.77 / 20.81 有逐样本 profile/error artifacts；20.80 只有 aggregate/group/failure-case artifacts；当前没有 raw `pred_params` 或 predicted profile arrays，因此不做 prediction reconstruction。

核心判断是：不要继续把 `wLD/wWD/wLW` 逐项 MAE 当作 true 3D branch 的主评价。它们仍是有用的 curvature diagnostic，但 Piao-style 路线真正要评价的是六参数生成的 3D profile 是否准确。20.77 test 的 curvature-vs-profile RMSE correlation 只有 `0.358243`；20.81 虽然 Dice 更高，但 profile depth RMSE 更差，说明 projected mask 也不能替代 3D profile 指标。

下一步唯一建议：**20.83 做 `R1_six_params_profile_primary_loss`**。继续输出 `L/W/D/wLD/wWD/wLW`，但把 validation / loss 主目标改成 profile-level reconstruction，例如 `profile_depth_rmse_m` 或 Er-like depth/profile error；`wLD/wWD/wLW` 降为 auxiliary diagnostics；不需要新 COMSOL 数据，不需要扩到 480，不做 baseline replacement。
## 2026-05-25 更新：第 20.81 后的下一步

第 20.81 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上的 feature-fusion neural model diagnostic。本轮没有运行 COMSOL，没有生成或修改 NPZ/data，没有创建 baseline，也没有更新 `CURRENT_BASELINE.md`；所有输入仍通过 registry/manifest 显式 dataset_id 加载，禁止 latest/newest 自动扫描。

核心判断是：feature-fusion 改善了整体拟合和 projected mask，但没有解决 curvature 风险。validation-only selection 选中 `H3_curv_fusion_F0F1F2_w1p0`，multi-seed 后 selected seed `2026` 的 test total MAE 为 `0.667888`，优于 20.77 neural 的 `0.678014`；projected mask Dice 为 `0.866573`，也优于 20.77 的 `0.847727`。但是 curvature MAE 只是从 `0.201076` 降到 `0.194483`，改善 `0.006592`，未达到本轮 `>=0.01` 实质改善门槛；wLD 从 `0.209439` 退到 `0.217079`，且 curvature 仍弱于 20.80 feature-only 的 `0.190304`。

下一步唯一建议：**redefine curvature labels / output representation**。真正的分界点不是再往 neural head 里塞更多 feature，而是 `wLD/wWD/wLW` 这组三维 profile curvature label 是否以当前形式可辨识、可学习、可评价。下一轮应先审计 curvature 参数定义、替代输出表示、profile/depth loss 口径和与 projected mask 的脱钩关系；不要直接 formal benchmark rerun，不要 baseline replacement，也不要先扩到 480。

## 2026-05-25 更新：第 20.80 后的下一步

第 20.80 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上的 Piao/NLS-inspired feature diagnostic。本轮没有运行 COMSOL，没有生成或修改 NPZ，没有训练 neural model，没有创建 baseline，也没有更新 `CURRENT_BASELINE.md`；所有输入仍通过 registry/manifest 显式 dataset_id 加载，禁止 latest/newest NPZ 自动扫描。

核心判断是：F0+F1+F2 physical features 对 curvature 有真实但有限的帮助。validation 选中 `F0_F1_F2_basic_physical + svr_rbf_C10_eps0.03`，test total MAE 为 `0.695724`，curvature MAE 为 `0.190304`，wLD/wWD/wLW 为 `0.209649/0.194797/0.166465`，projected mask Dice 为 `0.826272`。它优于 20.77 feature baseline 的 total `0.715395` / curvature `0.195046`，也比 20.79 refined model 更好；但仍弱于 20.77 neural 的 total `0.678014` 和 Dice `0.847727`，且 wLD 没有改善。F4 NLS proxy 提取稳定，但没有被 validation 选中，不能把本轮写成 exact Piao/NLS 成功。

下一步唯一建议：做 **feature-fusion / hybrid neural model**，保留 20.77 neural path 负责 L/W/D 与 mask/profile，把 F1/F2 这类稳定物理特征作为 curvature 辅助输入或辅助 head；不要做 baseline replacement，不要声称完整 Piao 复现，也不要直接扩到 480。

## 2026-05-25 更新：第 20.79 后的下一步

第 20.79 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 上的 curvature-aware model refinement。本轮没有运行 COMSOL，没有生成或修改 NPZ，没有更新 `CURRENT_BASELINE.md`，也没有把 refined model 写成 baseline。

核心判断是：`C1_split_heads` 被 validation-only selection 选中，但 test 指标相对 20.77 reference 退化。20.77 reference test normalized MAE 为 `0.678014`、curvature MAE 为 `0.201076`、projected mask Dice 为 `0.847727`；20.79 selected refined model test normalized MAE 为 `0.753387`、curvature MAE 为 `0.211584`、projected mask Dice 为 `0.834597`。`wLW` 和 `W_m` 有轻微改善，但 `L_m`、`D_m`、`wLD`、`wWD` 和 profile depth RMSE 退化，因此不能升级 benchmark candidate。

下一步唯一建议：保留第 20.77 的 v3_240 benchmark candidate，不采用 20.79 refined model；优先做 **exact Piao / NLS-inspired feature pipeline** 作为 curvature 诊断和 comparator，其次再考虑 curvature-targeted data top-up。不要把本轮结果写成 baseline replacement。

## 2026-05-25 更新：第 20.78 后的下一步

第 20.78 已完成 formal true 3D RBC benchmark candidate audit。本轮没有运行 COMSOL、没有生成或修改 NPZ、没有重新训练模型，也没有更新 `CURRENT_BASELINE.md`。审计结论是：`comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 可以进入 **formal benchmark candidate**，但必须带 curvature risk，且明确不是 baseline。

核心分界点是 curvature：v3_240 的 neural test normalized MAE `0.678014` 优于 feature comparator `0.715395`，L/W/D MAE 为 `1.892/2.186/0.800 mm`，D_m、projected mask Dice 和 profile depth RMSE 都较 N=112 改善；但 `wLD/wWD/wLW` 仍不稳定，boxy / sharp 最差，且出现 Dice `0.956750` 但 curvature error `0.364948` 的样本，说明 2D projected mask 已不足以评价 true 3D profile curvature。

下一步唯一建议：进入 **model refinement for formal benchmark candidate**，先做 curvature-aware model/head/loss、stronger Bx/By/Bz sequence encoder，以及 exact Piao / NLS-inspired feature diagnostic。curvature-targeted data top-up 是第二选择；不要把第一步直接设成扩到 480，也不要做 baseline replacement。

## 2026-05-25 更新：第 20.77 后的下一步

第 20.77 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240` 的 true 3D RBC training gate。本轮通过 `dataset_id + COMSOL_DATA_REGISTRY.md + manifest` 显式加载 v3_240，没有 latest/newest NPZ 自动扫描，没有运行 COMSOL，没有生成或修改数据，也没有更新 `CURRENT_BASELINE.md`。输入为 `delta_b=(240,3,3,201)`，Conv1D 输入为 `(240,9,201)`，split=train/val/test 162/39/39。

结果是 **promising benchmark candidate, not baseline**：feature sanity comparator selected `svr_rbf_C10`，test normalized MAE 为 `0.7154`；neural gate 三个 seed 完成，validation 选择 seed `42`，test normalized MAE 为 `0.6780`，优于 mean baseline `0.9127` 和 feature comparator `0.7154`。相对第 20.75 N=112，L/W/D MAE 改善为 `1.89/2.19/0.80 mm`，D_m 从 `1.106 mm` 改善到 `0.800 mm`，projected mask Dice 从 `0.8364` 到 `0.8477`，profile depth RMSE 从 `0.000548 m` 到 `0.000388 m`。但 curvature 参数 `wLD/wWD/wLW` 仍不可稳定学习，curvature MAE 相对 N=112 从 `0.1905` 退到 `0.2011`。

下一步唯一建议：进入 **formal true 3D RBC benchmark candidate / model refinement**，但继续把 dense mask baseline 只作为 comparator，不做 baseline replacement。下一阶段应固定 registry/manifest gate，优先处理 curvature learnability：可以比较更强的 Conv1D/Transformer-style sequence encoder、curvature-aware loss 或 Piao-inspired NLS/LS-SVM 特征；不要直接把 v3_240 模型写入 `CURRENT_BASELINE.md`。

## 2026-05-25 更新：第 20.75 后的下一步

第 20.75 已完成 `comsol_true_3d_rbc_imported_watertight_pilot_v2_120` 的 true 3D RBC training gate。本轮只用 registry + manifest 显式加载数据，不运行 COMSOL、不生成或修改 NPZ、不训练正式 baseline、不更新 `CURRENT_BASELINE.md`。v2_120 的输入为 `delta_b=(112,3,3,201)`，Conv1D 输入为 `(112,9,201)`，split=train/val/test 76/18/18。

结果是 promising but not baseline：feature sanity validation 选择 `svr_rbf_C10`，test normalized MAE 为 `0.7677`；neural gate 三个 seed 全部完成，validation 选择 seed `42`，test normalized MAE 为 `0.7039`，优于 mean baseline `0.8803` 和 feature baseline `0.7677`。相对 20.73 N=56，neural test MAE 从 `0.7601` 降到 `0.7039`，L/W/D MAE 改善到 `2.51/2.59/1.11 mm`，curvature MAE 从 `0.2095` 降到 `0.1905`，projected mask Dice 从 `0.8347` 到 `0.8364`。但 `wLD/wWD/wLW` 仍不稳定，N=112 仍不足以写成 baseline。

下一步唯一建议：扩展 true 3D RBC dataset 到 240，再用同一套 `dataset_id + manifest + registry` gate 重跑 training gate。不要先更新 baseline，也不要把 v2_120 自动接入 mainline training；dense mask baseline 继续只作为 comparator。

## 2026-05-25 更新：第 20.74 后的下一步

第 20.74 已把 true 3D RBC imported-watertight 数据集从 v1 assembled N=56 扩展到 `comsol_true_3d_rbc_imported_watertight_pilot_v2_120`。实际 assembled N=112，split=train/val/test 76/18/18，curvature coverage 为 sharp=22、round=23、boxy=23、LD_dominant=24、WD_dominant=20；NPZ/schema validation、registry validation、manifest 和 Claude Code review 均通过。状态是 `pilot_generated`、`train_ready_candidate=True`、`baseline_ready=False`，不是 baseline。

下一步唯一建议：进入 **true 3D training gate on v2_120**。训练/评估必须通过 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v2_120`、`COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v2_120.manifest.json` 显式加载，禁止 latest/newest NPZ 自动扫描；不更新 `CURRENT_BASELINE.md`，dense mask baseline 继续只作为 comparator。如果 v2_120 训练后 WD_dominant、deep/elongated 或 curvature 参数仍不稳，再做第二波 targeted top-up。

## 2026-05-24 更新：第 20.72 后的下一步

第 20.72 已把 20.71 的 partial pilot 补齐为 assembled true 3D RBC-style pilot pack candidate：assembled dataset_id 为 `comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled`，N=56，split = train/val/test 36/10/10，curvature coverage 包含 sharp、round、boxy、`LD_dominant`、`WD_dominant`，NPZ/schema validation、registry validation 和 Claude Code review 均通过。

下一步唯一建议：进入 **true 3D training gate**，但这仍是 pilot training，不是 baseline replacement。训练/评估脚本必须通过 `COMSOL_DATA_REGISTRY.md` 和 `results/manifests/comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled.manifest.json` 显式读取 dataset_id，禁止 latest/newest 自动扫描；`CURRENT_BASELINE.md` 不更新，dense mask baseline 继续只作为 comparator。

## 2026-05-24 更新：第 20.71 后的下一步

第 20.71 已生成 true 3D RBC-style imported-watertight pilot pack，但状态是 `partial_pilot_generated`。当前有效样本为 30，split = train/val/test 20/5/5；所有成功样本使用 `imported_watertight_mesh_solid`、20.70 material/domain fix、default solver、`mesh_auto_size=5`、`Jscale=1.0`，真实导出 `[mf.Bx, mf.By, mf.Bz] @ sensor_z_m=0.008`，`delta_b` 和 NPZ/schema validation 均通过。

当前不能进入 training gate：inventory 虽已完整覆盖 60 行，但只有 30 pass、2 fail、28 not_attempted；`LD_dominant` 和 `WD_dominant` 两个 curvature family 尚无成功样本。Registry / manifest 已建立，`allowed_use=schema_validation, explicit_pilot_training_gate`，`forbidden_use=automatic_mainline_training, baseline_update, current_baseline_replacement`；该 pack 不是 baseline，`CURRENT_BASELINE.md` 不更新。

下一步唯一建议：做 20.71 top-up generation，优先补齐 `LD_dominant` / `WD_dominant` not_attempted 样本，并复查 `deep_elongated` timeout 样本。top-up 通过后再重新验证 manifest / registry / split / curvature coverage，才讨论 explicit true 3D training gate。

## 2026-05-24 更新：第 20.70 后的下一步

第 20.70 已把 20.69 的 imported watertight solid 路线从 geometry gate pass 推进到 full-source forward smoke pass。原始 blocker 不是 Python mesh 或 Boolean subtract，而是 imported solid 后的 domain/material selection：COMSOL 未暴露稳定 cavity domain selection，且原始 air selection 与 steel selection 重叠。采用最小 selection/material fix 后，`material_domain_fixed` 在 `mesh_auto_size=5`、default solver、`Jscale=1.0` 下通过 defect stationary solve。

当前 forward-ready 证据是：`medium_round` one-sample imported watertight solid 已真实导出 `[mf.Bx, mf.By, mf.Bz] @ sensor_z_m=0.008`，`delta_b = b_defect - b_no_defect` 校验误差为 `0.0`，NPZ/schema validation 通过，`selected_solver_protocol=default`，没有 direct solver 依赖，也没有使用 high-layer fallback。

下一步唯一建议：进入 **smooth/mesh-based true 3D RBC pilot generation design**，先做小规模 pilot 计划与 mesh/material selection QA，而不是回到 high-layer approximation、2D profile-forward 小修或训练 surrogate。`CURRENT_BASELINE.md` 仍不更新，dense mask baseline 继续只作为 comparator；generated NPZ、temp STL、raw CSV、`.mph` 等仍不能提交。

## 2026-05-24 更新：第 20.69 后的下一步

第 20.69 已完成 watertight imported solid builder hardening。本轮不训练模型、不进入 pilot、不更新 `CURRENT_BASELINE.md`，也不创建或修改 COMSOL baseline 文档。`medium_round` 的 RBC-style depth map 已由 pure NumPy 生成 watertight closed STL：`mesh_units=m`，top cap 位于 `z=0`，bottom surface 使用 `z=-depth`，defect void 嵌入 steel 且与 steel surface 相交；mesh validation 通过，`is_watertight=True`，`nonmanifold_edges_count=0`，`zero_area_triangles_count=0`，`volume_m3=1.2918e-07`，`max_depth_m=0.0025`。

COMSOL 侧结果比 20.68 更进一步：known prism sanity probe 通过；RBC watertight STL 的 `import_success=True`、`repair_success=True`、`form_solid_success=True`、`imported_domain_count=1`、`boolean_subtract_success=True`、`steel_notched_domain_count=1`、`mesh_precheck_success=True`。这说明 20.68 的 imported mesh Boolean empty steel domain blocker 已被推进到 geometry gate 通过。

但 one-sample forward smoke 只完成到 no-defect solve；defect model 的 stationary solver 不收敛，未生成 `true_3d_imported_watertight_forward_smoke_v1.npz`，因此没有运行 NPZ/schema validator，也没有 `delta_b`。当前 route decision 是 `C_import_boolean_pass_forward_not_run_or_failed`：imported watertight solid geometry route 技术上可行，但尚不是 pilot-ready。下一步唯一建议是先修 COMSOL imported solid 的 solve / mesh-quality / solver robustness，再考虑 smooth/mesh-based true 3D RBC pilot；不要扩样、不要训练、不要回退到 2D profile-forward 小修。dense mask baseline 仍只作 comparator。

## 2026-05-24 更新：第 20.68 后的下一步

第 20.68 已完成 smooth / near-smooth true 3D variable-depth builder completion feasibility。本轮没有训练模型、没有进入 pilot、没有生成 20.68 forward NPZ、没有更新 `CURRENT_BASELINE.md`，也没有创建或修改 COMSOL baseline 文档。bounded geometry probe 的结论是：`lofted_contour_solid`、`stacked_workplane_contour_loft`、`interpolated_surface_solid`、`imported_closed_mesh_solid` 均未形成可进入 forward 的 smooth / near-smooth candidate；唯一通过的是 `high_layer_control_24`。

当前状态必须写成 `high_layer_control_pass`，不能写成 `variable_depth_pass` 或 `near_smooth_pass`。`high_layer_control_24` 比 20.67 的 12-layer control 更进一步，记录了 24 个 depth levels，且 `closed_body_success=True`、`boolean_subtract_success=True`、`mesh_precheck_success=True`、`spatial_depth_variation=True`、`is_constant_depth=False`；但它仍是 stepped/high-layer control，不是 smooth RBC surface，也不是 exact Piao RBC geometry。

下一步唯一需要人工确认：是否接受 high-layer approximation 作为后续 pilot 口径。如果接受，可以设计 true 3D RBC pilot，但所有文件必须显式标注 `high_layer_approximation`，不能写成 smooth / near-smooth；如果不接受，则继续修 COMSOL smooth / closed-surface builder，优先诊断 imported mesh Boolean empty steel domain 或寻找可用的 closed-surface / convert-to-solid 路径。dense mask baseline 仍只作 comparator。

## 2026-05-23 更新：第 20.66 后的下一步

第 20.66 已完成 true 3D RBC-style smoke pack generation。本轮只做 smoke pack generation 和 schema validation：没有训练 forward surrogate 或 inverse model，没有做 refinement，没有更新 `CURRENT_BASELINE.md`，也没有创建或修改 COMSOL baseline 文档。Claude Code review 完成且无 must-fix。

当前通过状态是 `stepped_depth_smoke_pass`。Stage A 生成 6 个 RBC-style single-defect samples，`L_m=0.010-0.030`、`W_m=0.006-0.020`、`D_m=0.001-0.006`，pure-Python depth/profile/mask validation 6/6 通过；Stage B 真实 COMSOL forward 6/6 通过，输出 `[mf.Bx, mf.By, mf.Bz] @ sensor_z_m=0.008`，`delta_b = b_defect - b_no_defect` 校验通过；Stage C NPZ schema validation 6/6 通过。

边界必须写清：本轮没有通过 smooth true variable-depth RBC solid。COMSOL 几何实现是 5 层 `stepped_depth_layered_approximation`，`smooth_variable_depth_solid_verified=False`，`stepped_depth_approximation=True`，`constant_depth_extrusion_used_as_success=False`。RBC generator 也标记为 `exact_piao_rbc=False`，属于 RBC-style / RBC-inspired engineering approximation，不是完整复现 Piao 2019。

下一步唯一建议：先做路线决策，不要直接把 20.66 写成 smooth 3D pilot ready。需要在两个选项中选择：继续改 COMSOL smooth variable-depth geometry，或明确接受 stepped-depth 作为 20.67 pilot approximation 后再设计 60-sample pilot。dense mask baseline 仍只作为 comparator。

## 2026-05-23 更新：第 20.65 后的下一步

第 20.65 已完成 true 3D / Piao-style geometry profile feasibility design。本轮是 design-only：没有运行 COMSOL、没有生成数据、没有训练 surrogate / inverse model、没有做 refinement，也没有更新 `CURRENT_BASELINE.md` 或任何 COMSOL baseline 文档。Claude Code review 通过且无 must-fix。

当前判断是：20.61-20.64 已足以暂停 2D profile-forward 小修。single-height Bz、multi-height Bz、same-direction Bx/By/Bz、multi-direction excitation 都没有让真实 COMSOL oracle residual 稳定排序 profile quality；继续训练 2D profile surrogate 或继续 refinement config 微调没有主线价值。下一步主线切换为 **true 3D / Piao-style geometry profile**，dense mask baseline 只作为 comparator。

第 20.66 的唯一推荐任务是 small smoke，不是正式数据集：验证 `RBC params -> depth map -> COMSOL variable-depth defect solid -> same-source projected mask -> Bx/By/Bz @ sensor_z_m=0.008 -> delta_B = B_defect - B_no_defect`。当前只能写成 COMSOL 支持真实 3D volume solve；RBC / variable-depth true 3D profile generation 尚未验证，是 20.66 的核心 blocker。不要把 20.66 smoke 扩成 multi-height，`0.012m` 只作为后续 pilot / ablation 设计保留。

如果 20.66 无法构建 variable-depth true 3D solid，应暂停 geometry-forward route，先解决 COMSOL geometry blocker；如果 smoke 通过，再进入小规模 3D pilot。未来 20.67 的 IoU/Dice/profile-error 阈值目前只是 preliminary acceptance guidance，不是已验证硬标准。

## 2026-05-23 更新：第 20.64 后的下一步

第 20.64 已完成 multi-direction excitation profile perturbation oracle ordering feasibility POC。本轮只做 oracle residual audit，不训练 surrogate、不做 refinement、不更新 baseline。COMSOL pack 覆盖 12 base samples / 96 profile rows / 3 directions / 3 axes；`direction_0` 复用同 geometry 的 20.63 default-direction rows，`direction_45` 和 `direction_90` 使用真实 COMSOL forward，并通过 `ExternalCurrentDensity.Je` 设置真实改变 excitation / magnetization direction。direction probe 显示 `direction_90` 相对 `direction_0` 的 no-defect / defect response NRMSE 为 `1.6479 / 1.7981`，因此不是数组旋转或信号伪造。

结果没有通过 gate。same-pack test 中，`direction_0` Bz-only ordering = `0.4545`，`direction_90` Bz-only = `0.5273`，multi-direction Bz train-std normalized = `0.5636`，说明 Bz-only multi-direction 有边际正信号；但 all-axis normalized ordering 只有 `0.3455`，mismatch_rate = `0.6545`，residual-error correlation = `-0.8028`，明显劣于 same-pack default-direction Bz。Claude Code review 通过且无 must-fix，同时指出 Bz-only 正信号仍不稳定，test base 太少，不能支撑 surrogate 训练。

当前下一步唯一优先级：**true 3D profile / Piao-style route**。不要进入 20.64 的 multi-direction surrogate training，也不要回到 profile-forward refinement；第 20.64 只说明改变 excitation direction 比同方向 multi-axis / multi-height 更有一点信号，但仍未证明 richer direction observation 可以稳定缓解 profile residual non-identifiability。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-23 更新：第 20.63 后的下一步

第 20.63 已完成 multi-axis MFL profile perturbation oracle ordering feasibility POC。本轮只做 oracle residual audit，不训练 surrogate、不做 refinement、不更新 baseline。multi-axis pack 覆盖 24 base samples、192 profile rows、3 个 field axes `[Bx, By, Bz]`，共 576 个 axis observations；所有行包括 `true_reference` 均由真实 COMSOL forward 生成，未复用旧 Bz-only 数组。实际 COMSOL expressions 为 `[mf.Bx, mf.By, mf.Bz]`，expression probe 通过，`delta_B = B_defect - B_no_defect` 三轴校验通过。

结果是否定性的：test Bx-only oracle ordering = `0.4505`，By-only = `0.4955`，Bz-only = `0.4505`，Bx+By+Bz train-std normalized ordering = `0.4505`，mismatch_rate = `0.5495`，residual-error correlation = `0.0242`。multi-axis 没有超过同 pack 的 Bz-only，也没有超过 20.61 single-height Bz oracle test reference `0.5030`，未达到 `>0.65` 或 `+0.10` improvement gate。

当前下一步唯一优先级：**multi-direction excitation / richer scan geometry**。不要训练 multi-axis profile surrogate，也不要回到 profile-forward refinement retry；因为真实 COMSOL oracle residual 在 same-liftoff Bx/By/Bz 下仍不能稳定排序 profile quality。若继续 forward-guided route，应改变激励方向、扫描方向或 scan geometry；若不扩展观测物理，则应暂停 profile-forward residual route。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-23 更新：第 20.62 后的下一步

第 20.62 已完成 multi-height Bz profile perturbation oracle ordering feasibility POC。本轮只做 oracle residual audit，不训练 surrogate、不做 refinement、不更新 baseline。multi-height pack 覆盖 12 base samples、96 profile rows、3 个 sensor_z heights `[0.004, 0.008, 0.012]`，共 288 个 height observations；其中 0.008m 的 96 个 observation 复用第 20.61 exact rows，0.004m 和 0.012m 共 192 个 observation 由真实 COMSOL forward 生成。profile polygon geometry 有效，`delta_bz = bz_defect - bz_no_defect` 校验通过。

结果是否定性的：test single-height 0.008m oracle ordering = `0.4909`，0.004m = `0.4364`，0.012m = `0.4545`，multi-height train-std normalized ordering = `0.4545`，mismatch_rate = `0.5455`，residual-error correlation = `-0.5920`。multi-height 没有超过 20.61 single-height oracle test reference `0.5030`，也没有达到 `>0.65` 或 `+0.10` improvement gate。

当前下一步唯一优先级：**multi-axis / multi-direction observation**。不要训练 multi-height profile surrogate，也不要回到 profile-forward refinement retry；因为真实 COMSOL oracle residual 在 multi-liftoff Bz 下仍不能稳定排序 profile quality。若继续 forward-guided route，应改变观测信息维度，例如不同扫描方向、横向 scan lines / components 或更丰富观测，而不是继续小调当前 profile surrogate、refinement loss 或单纯扩大同类 lift-off 数据。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-23 更新：第 20.61 后的下一步

第 20.61 已完成 expanded profile perturbation forward pack + profile-compatible surrogate calibration POC。expanded pack 达到 target：36 base samples、288 rows，其中 `reused_original_rows=36`、`reused_from_20_60_rows=0`、`real_comsol_forward_rows=252`；split = 192/48/48，rect/rot = 144/144，8 类 profile variant 各 36 行。COMSOL 侧 profile polygon geometry 有效，`delta_bz = bz_defect - bz_no_defect` 校验通过。

selected surrogate 为 `EPPF1_profile_station_mlp`，waveform val/test NRMSE/correlation = `0.3314 / 0.9435` 和 `0.3735 / 0.9299`。相比 20.60，test surrogate ordering 从 `0.2143` 提升到 `0.5569`，mismatch_rate 从 `0.7857` 降到 `0.4431`，说明扩大 profile perturbation data 缓解了最严重的 test collapse；但 val/test surrogate ordering 仍低于可用 gate，且 COMSOL oracle residual ordering train/val/test 仅 `0.4471 / 0.5120 / 0.5030`，接近随机。

当前下一步唯一优先级：**richer observations / multi-height / multi-axis 或 non-identifiability audit**。不要直接回到 profile-forward refinement retry，也不要继续小调当前 profile surrogate architecture / loss；因为 oracle residual 本身不能可靠排序 profile quality，继续优化 surrogate 很难把不可辨识的 residual objective 变成可用 refinement 梯度。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-22 更新：第 20.60 后的下一步

第 20.60 已完成 profile perturbation forward pack + profile-compatible surrogate calibration POC。按修正版 row-count gate，COMSOL 侧生成 minimum partial pack：`total_rows=96`，`reused_original_rows=12`，`real_comsol_forward_rows=84`，represented base samples = 12，split = 64/16/16，rect/rot = 48/48，8 类 profile variant 各 12 行。`true_reference` 行只作为 residual ordering anchor 复用 pilot_v9 原始数组，不计入真实 COMSOL forward rows；真实生成行使用 profile polygon geometry，delta check 通过。

profile-native surrogate 的 waveform fit 可以接受但 residual ordering 不足。validation 选中 `PPF1_profile_station_mlp`，val/test NRMSE/correlation 为 `0.4396 / 0.8990` 和 `0.3758 / 0.9274`；但 oracle residual ordering val/test 只有 `0.6786 / 0.5357`，selected surrogate residual ordering 为 `0.6607 / 0.2143`，mismatch_rate 为 `0.3393 / 0.7857`。因此第 20.60 没有进入 profile-forward refinement，也不更新任何 baseline。

当前下一步唯一优先级：**扩 profile perturbation data**。需要优先增加 base sample 覆盖，尤其是 val/test base 数，重新检查 oracle residual 是否能稳定排序 profile quality；若 oracle ordering 仍弱，则应转向 richer observations / multi-axis / multi-height 或保留 no-forward profile basis，不应继续对当前 profile-forward surrogate 做小幅架构或 loss 微调。仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

## 2026-05-22 更新：第 20.58 后的下一步

第 20.58 已完成 mask/profile basis refinement POC。profile extraction 从 predicted dense mask/probability 中提取 K=8 profile 表示，validation 选择 `P1_hardmask_profile`；profile-extracted test IoU/Dice/area_error 为 `0.6589 / 0.7921 / 0.2170`。no-forward profile refinement 只拟合 dense initial probability 并加 smoothness / area / bounds prior，test 提升到 `0.6697 / 0.8002 / 0.2196`，说明 profile basis 相比第 20.57 的 single rotated-box refinement 更稳，但没有稳定超过第 20.54 extracted rotated-box proposal `0.6726 / 0.8017 / 0.1945`。

forward profile refinement 已执行受控 sweep，但 validation 选择 `lambda_forward=0.0`，test 为 `0.6620 / 0.7938 / 0.2243`。这说明当前第 20.56/20.57 的 S1 surrogate 通过 lossy profile-to-rect summary 接入后，不能作为可靠的 profile-space forward consistency 约束。Claude Code review 通过且无必须修复；审查结论是不建议继续在当前 surrogate-dependent profile refinement 上小调。

当前下一步唯一优先级：**改进 profile-compatible forward surrogate**。如果继续 profile/basis 路线，应先让 forward model 直接接受 profile/basis 或 rasterized-profile derived features，而不是把 profile 压回单个 rect/rot summary；否则应暂停 geometry/refinement route，等待更丰富观测或更强 forward data。仍不更新 `CURRENT_BASELINE.md`，也不创建新的 COMSOL baseline 文档。

## 2026-05-22 更新：第 20.57 后的下一步

第 20.57 已完成 perturbation-calibrated surrogate 的受控 Priewald-style refinement retry。`S1_perturb_geom_mlp` 按第 20.56 protocol 重训于内存中，recovery 指标与 20.56 对齐：val/test waveform NRMSE 为 `0.3666 / 0.4289`，residual ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 为 `0.2679 / 0.1964`。

但是连续低维 refinement 没有通过 gate。validation 上 8 个 config 全部导致 mask 指标退化或 mismatch 过高，最终仅选最高分 config 做 diagnostic：`steps=50, lr=0.003, lambda_prior=0.10`。test geometry-raster IoU/Dice/area_error 从 `0.6726 / 0.8017 / 0.1945` 变为 `0.6492 / 0.7829 / 0.2417`；forward NRMSE 下降 `0.0713`，但 mismatch_rate 为 `0.6212`，residual reduction 与 IoU/Dice delta 相关性为 `-0.1824 / -0.2250`。

当前判断：20.56 的 pairwise residual ordering 改善没有转化为可用的连续 geometry optimization 梯度。不要继续在当前 rect/rot low-dimensional refinement objective 上小调 steps / lr / prior；也不要回到 direct geometry head 或 dense baseline patch。最近下一步优先转向 **mask/profile basis refinement**，降低对 single rect/rot parameter residual landscape 的依赖。若未来重新尝试 Priewald-style refinement，应先扩大 perturbation pack 或加入 richer observations，再重新验证 residual landscape。

## 2026-05-22 更新：第 20.56 后的下一步

第 20.56 已生成小规模 local geometry perturbation forward-calibration pack，并完成 surrogate residual ordering audit。实际 COMSOL pack 是 96 行 partial pack（12 个 base，train/val/test = 64/16/16，rect/rot = 48/48），84 行为真实 COMSOL forward，12 行 true reference 复用原始 NPZ；`delta_bz = bz_defect - bz_no_defect` 校验通过。

关键结论是：COMSOL oracle residual 的 val/test ordering accuracy 为 `0.6607 / 0.8393`，选中的 `S1_perturb_geom_mlp` surrogate 的 val/test ordering accuracy 为 `0.7321 / 0.8036`，mismatch_rate 为 `0.2679 / 0.1964`，较 20.55 明显改善。这说明 perturbation forward data 对 surrogate mismatch 有帮助，下一步可以回到 **controlled Priewald-style refinement retry**，但必须继续把它作为 POC/candidate，不更新 baseline。

限制也很明确：当前 pack 只有 96/192 行，且 selected surrogate 的 test residual-error correlation 仍为负（`-0.0462`）。因此下一步不要直接扩大为正式路线，也不要继续训练新的 direct geometry head；应先用 perturbation-calibrated surrogate 做一次受控 refinement retry，观察 residual ordering 是否能转化为 mask / geometry 改善。如果 retry 仍出现 residual 下降但 mask 退化，则优先扩 perturbation data 或转向 mask/profile basis refinement。

## 当前状态

`CURRENT_BASELINE` 仍以 [CURRENT_BASELINE.md](CURRENT_BASELINE.md) 为准：

- v3_complex mask-only grid decoder + forward consistency
- `lambda_forward = 0.10`
- validation-selected probability threshold = `0.80`
- forward surrogate = `checkpoints/mask_to_bz_forward_surrogate/best_mask_to_bz_forward_surrogate.pt`

该 baseline 是当前 v3_complex 上最强 boundary-oriented baseline，但 polygon / rotated_rect 精细边界圆斑化、small / low-signal、多缺陷仍未根本解决。

内部 decoder / loss / threshold / geometry / basis / refinement 小修补已经基本收口。当前第 20 阶段已经从单纯 COMSOL 数据包验证，进一步进入 geometry-aware / forward-consistent inverse reconstruction 方法验证。当前这些工作仍是 COMSOL data-domain POC / candidate，不更新 v3_complex `CURRENT_BASELINE`。

## 最近下一步

当前不要继续普通 dense decoder patch，也不要继续单独 geometry head 小修补。第 20.48-20.55 已证明：

1. geometry labels 和 differentiable rotated-rectangle rasterizer 没有 blocker；
2. direct delta_bz-only geometry head 的 type / angle 学习不足；
3. controlled architecture sweep 没有找到有效 head 结构；
4. feature-assisted geometry head + lightweight forward consistency 只带来边际 mask / angle 改善，type confusion 仍是主因；
5. Priewald-style low-dimensional refinement 能降低 forward residual，并可小幅改善 geometry-raster mask，但 initializer / proposal 质量决定上限；
6. dense/coarse mask initializer + PCA bbox extraction 在 20.53 中没有超过 20.51 geometry-head proposal，type / angle proposal 仍弱；
7. 第 20.54 的 strong dense initializer 和 improved proposal extraction 已把 rect/rot geometry proposal 提到 test IoU/Dice `0.6726 / 0.8017`，但 Priewald-style refinement 让 test IoU/Dice 回落到 `0.6646 / 0.7958`，同时 forward NRMSE 下降，说明当前主要 blocker 已从 proposal quality 转为 forward surrogate mismatch；
8. 第 20.55 的 calibrated surrogate sweep 没有找到可用 residual objective：S2 的 waveform NRMSE 最好，但 val residual-error correlation 为负，S3 的正相关也只有 `0.0215`，未过 gate，因此 calibrated refinement 被正确跳过。

因此最近下一步优先转向：

1. **生成 synthetic perturbation forward data / 局部扰动校准数据**：当前缺少同一 geometry 附近的已知扰动与 forward response，surrogate 学不到“几何越差 residual 越高”的局部单调关系。
2. 如果不能生成扰动 forward 数据，则转向 **mask/profile basis refinement**，减少对当前低维 rect/rot geometry residual objective 的依赖。
3. 暂停继续对现有 surrogate loss、peak weighting 或 refinement objective 做小调；20.55 已说明 waveform 拟合不等于 residual 可用于 geometry refinement。
4. 继续保持 train-only normalization、validation checkpoint / threshold selection、test-only final evaluation。

如果后续 refinement 不能在更强 proposal 上稳定改善 mask / geometry，再暂停 rect/rot geometry route，等待更丰富观测、更多通道或更强 forward surrogate；当前不建立新 baseline。

## 当前不要继续的方向

不要继续围绕现有 v3_complex grid decoder 做 selection metric、ensemble、threshold trick、loss trick、decoder head、SDF / boundary head、coordinate refinement、hand-crafted Bz features、U-Net-like decoder、shape-type conditional、star-convex、retrieval、box / quad / basis / profile 或 mask-logit refinement 小修补。

也不要继续单独调 rect/rot neural geometry head。新的实验必须回答：显式 geometry representation、differentiable rasterization 和 forward residual 是否能稳定提高边界反演可辨识性，而不是只带来局部指标波动。
## 2026-05-22 更新：第 20.59 后的下一步

第 20.59 已完成 profile-compatible forward surrogate POC。preflight 结论是该方向符合 Priewald-style forward-model-based inversion / refinement 路线，但必须先证明 profile-native residual 能在 validation 上稳定排序 geometry/profile quality，不能再使用把 profile 压缩为单个 rotated box 的旧 surrogate bridge。

本轮构建 original profile-forward dataset（rect/rot N=400，split=268/66/66）和 perturb profile-forward dataset（20.56 partial pack N=96，split=64/16/16），并训练 3 个 profile-compatible surrogate。validation 选中 `PFS3_profile_station_sequence`，其 waveform val/test NRMSE 为 `0.3841 / 0.3995`，但 validation ordering accuracy 仅 `0.6607`，mismatch_rate 为 `0.3393`，未通过 usable-surrogate gate。因此 profile-forward refinement retry 被跳过。

当前下一步唯一优先级：**扩展 profile perturbation data**。如果扩展后的 validation ordering / mismatch gate 仍不通过，则不再继续 forward-guided profile refinement 小调，改回 no-forward profile basis 或等待 richer observations / multi-axis data。当前仍不更新 `CURRENT_BASELINE.md`，不创建或修改 COMSOL baseline 文档。

---

## 第 20.67 后下一步

20.67 的结论是 `high_layer_pass`：`medium_round` 的 12-layer high-layer approximation 通过了 geometry-only gate、Bx/By/Bz one-sample forward 和 NPZ/schema validation，并且明确区别于 20.66 的 5-layer stepped-depth smoke。

但本轮没有证明 smooth variable-depth true 3D geometry 可行：limited smooth / loft / imported closed-surface probe 没有形成 verified closed defect body，最终不是 `variable_depth_pass`，也不是 `near_smooth_pass`。因此下一步不能直接进入 60-sample true 3D RBC pilot，除非人工确认接受 high-layer approximation 作为 pilot approximation。

唯一推荐下一步：
- 如果接受 high-layer approximation：进入小规模 true 3D RBC pilot plan/generation，但所有文件和结论必须标注 `high_layer_approximation`，不得写成 smooth RBC。
- 如果不接受 high-layer approximation：继续修 smooth / closed-surface COMSOL geometry builder，不扩样、不训练。

仍然不更新 `CURRENT_BASELINE.md`；dense mask baseline 只作为 comparator。
## 2026-05-24 更新：第 20.73 后的下一步

第 20.73 已完成 true 3D RBC pilot training gate。它不是 baseline，也不更新 `CURRENT_BASELINE.md`；所有训练/评估都通过 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v1_assembled`、`COMSOL_DATA_REGISTRY.md` 和 manifest 显式加载，没有 latest/newest NPZ 自动扫描，也没有运行 COMSOL 或生成新数据。

训练 gate 的核心结论是：模型能拟合 train，但 N=56 的泛化证据不足。小型 Conv1D 在完整训练轨迹中可把 train normalized MAE 降到 `0.0012`，说明链路能学习训练样本；但 validation 选择的 checkpoint 在 test 上 normalized MAE 为 `0.7601`，只优于 mean baseline `0.8598`，没有超过 Piao-inspired feature baseline `0.7564`。当前可学习信号主要在 `L_m`、`W_m`，`D_m` 边缘，`wLD/wWD/wLW` 三个 curvature 参数仍不可稳定辨识。

唯一下一步建议：扩展 true 3D RBC dataset 到 120/240 量级，并把 validation set 扩到至少 20-30 个样本，再重新跑同一套 registry/manifest-gated training gate。不要先调大模型、不要更新 baseline、不要回到 dense mask 主线；dense mask baseline 继续只作 comparator。
## 2026-05-25 更新：第 20.76 后的下一步

第 20.76 已把 true 3D RBC imported-watertight dataset 从 v2_120 扩展到 `comsol_true_3d_rbc_imported_watertight_pilot_v3_240`。本轮没有训练、没有 baseline、没有更新 `CURRENT_BASELINE.md`；v2_120 source pack 未覆盖，v3 top-up 和 assembled NPZ 仍是 generated data，不提交。

当前 v3_240 状态是 `pilot_generated` 且 `train_ready_candidate=True`：N=240，split=162/39/39，curvature coverage=sharp 48 / round 49 / boxy 47 / LD_dominant 46 / WD_dominant 50，schema/registry/manifest validation 全部通过，baseline_ready=False。下一步唯一建议是执行 true 3D training gate on v3_240，必须通过 `dataset_id=comsol_true_3d_rbc_imported_watertight_pilot_v3_240` + manifest + `COMSOL_DATA_REGISTRY.md` 显式加载，禁止 latest/newest NPZ 自动扫描；重点检查 `D_m` 和 `wLD/wWD/wLW` 是否相对 v2_120 进一步稳定。


## 2026-05-27 update after Stage 20.90

Stage 20.90 completed a small true 3D RBC liftoff / sensor-offset COMSOL diagnostic pack using the fixed 20.85 baseline and the 20.88a inference artifact. It did not train a model, did not update `CURRENT_BASELINE.md`, and did not commit generated diagnostic data.

The key result is that the baseline is not mainly noise-limited here; it is liftoff-sensitive. Source/amplitude variation is largely corrected by the fixed 20.89 `per_axis_rms_train_stats` diagnostic calibration, but liftoff still fails even after calibration. Scan-line offset and postprocess Bx/By/Bz sample misalignment were low-risk in this 12-base diagnostic pack.

唯一下一步: design a dedicated COMSOL liftoff robustness / augmentation data pack before internal-defect feasibility or real-data claims. Calibration remains an acquisition diagnostic caveat, not a baseline replacement.
## 2026-05-27 update after Stage 20.91

Stage 20.91 completed the dedicated liftoff augmentation pack plan without running COMSOL, generating data/NPZ, training, or changing `CURRENT_BASELINE.md`. The plan selects 48 base geometries and four paired liftoff levels per base: `sensor_z_m=0.006 / 0.008 / 0.010 / 0.012`, for 192 planned COMSOL rows.

唯一下一步：execute the 20.91 COMSOL liftoff augmentation pack generation as a separate confirmed stage, keeping generated NPZ/data ignored and uncommitted. After that, 20.92 should compare the current unconditioned baseline family against a scalar `sensor_z_m` conditioned liftoff-aware variant under multi-liftoff evaluation.

## 2026-05-27 update after Stage 20.91b

Stage 20.91b generated and validated the dedicated liftoff pack: 48 base geometries × 4 paired liftoff levels = 192/192 successful COMSOL rows. The generated NPZ remains in the ignored data path and is registered as `comsol_true_3d_rbc_liftoff_aug_pack_v1`; no training and no baseline update were performed.

唯一下一步：enter 20.92 liftoff-aware training gate, comparing the current unconditioned baseline family against a scalar `sensor_z_m` conditioned model. Keep calibration as a diagnostic/acquisition caveat and keep internal/buried defect feasibility deferred.

## 2026-05-27 update after Stage 20.93

Stage 20.93 audited the 20.92 nominal/non-nominal liftoff trade-off without COMSOL, training, data/NPZ mutation, or `CURRENT_BASELINE.md` changes. The key finding is that `C1_unconditioned_liftoff_aug` is not a robustness candidate: it improves non-nominal profile RMSE and Dice, but nominal `0.008 m` profile RMSE regresses from `0.000333059 m` to `0.000809011 m`.

Only next step: train a nominal-preserving `S3_baseline_plus_liftoff_adapter` candidate. Keep the 20.85 nominal baseline path anchored, add a small `sensor_z_m`-conditioned correction for non-nominal liftoff, and evaluate with explicit nominal and non-nominal validation gates. No new COMSOL data is needed before this training gate; do not continue unconditional C1 augmentation, and keep internal/buried defects and real-data claims deferred.

## 2026-05-27 update after Stage 20.94

Stage 20.94 trained the nominal-preserving baseline+liftoff adapter on `comsol_true_3d_rbc_liftoff_aug_pack_v1` without COMSOL, new data, NPZ mutation, or `CURRENT_BASELINE.md` changes. Validation selected `A2_latent_residual_adapter`, seed `2026`. It preserved nominal behavior (`0.000333059 m -> 0.000335821 m`, `+0.829%`) and improved non-nominal profile RMSE (`0.000874310 m -> 0.000437214 m`, `-49.993%`) while raising non-nominal Dice from `0.683351` to `0.842378`.

Only next step: run a formal liftoff benchmark for the A2 robustness candidate. Keep `CURRENT_BASELINE.md` unchanged until a separate benchmark/baseline transition explicitly approves a replacement or an auxiliary robustness baseline.

## 2026-05-28 update after Stage 20.95

Stage 20.95 completed the formal liftoff benchmark for `A2_latent_residual_adapter` using persisted 20.94 metrics and explicit `comsol_true_3d_rbc_liftoff_aug_pack_v1` registry/manifest loading. A2 is accepted as a `CURRENT_BASELINE` companion robustness module: nominal RMSE is preserved within `+0.829%`, while non-nominal RMSE improves by `49.993%` and non-nominal Dice rises to `0.842378`.

Only next step: run a liftoff-conditioned inference smoke stage. Verify the frozen 20.85 baseline + A2 companion loading path, require `sensor_z_m` metadata, and keep `CURRENT_BASELINE.md` unchanged before real-data alignment or internal-defect feasibility.

## 2026-05-28 update after Stage 20.96a

Stage 20.96a recovered the missing A2 inference artifact needed by the 20.96 smoke test. The ignored checkpoint and prediction artifact were exported under `checkpoints/true_3d_rbc_liftoff_adapter_artifacts/`, while the tracked manifest is `results/manifests/true_3d_rbc_a2_liftoff_adapter_inference_artifact_manifest.json`. Verification exactly reproduced the 20.94/20.95 A2 reference metrics: nominal RMSE `0.000335821 m`, non-nominal RMSE `0.000437214 m`, and non-nominal Dice `0.842378`.

Only next step: return to 20.96 liftoff-conditioned inference smoke. Load the frozen 20.85 baseline plus A2 manifest, enforce required `sensor_z_m`, test auto / force-baseline / force-adapter routing, and keep `CURRENT_BASELINE.md` unchanged.

## 2026-05-28 update after Stage 20.96

Stage 20.96 completed the liftoff-conditioned inference smoke. The runner loads the frozen 20.85/20.77 baseline and the A2 liftoff companion adapter from tracked manifests, requires `sensor_z_m`, routes nominal `0.008 m` rows to the baseline, and routes non-nominal rows to baseline plus A2. It does not train, run COMSOL, write NPZ/data/checkpoints, or modify `CURRENT_BASELINE.md`.

Smoke test result: auto test all-liftoff profile RMSE is `0.000411175 m` with Dice `0.842773`; nominal RMSE remains `0.000333059 m`; non-nominal RMSE is `0.000437214 m`, matching the 20.95 A2 companion result and improving over force-baseline non-nominal RMSE `0.000874310 m`. The `sensor_z_m` contract is now explicit: unit meters, validated range `[0.006, 0.012]`, missing value is an error, and out-of-range values are flagged.

Only next step: move to real-data schema intake / acquisition metadata contract. Require `delta_b`, matched no-defect reference metadata, axis order, and `sensor_z_m` before any real-data inference claim. Internal/buried defect feasibility remains deferred.

## 2026-05-28 update after Stage 20.97

Stage 20.97 defined the real-data intake schema without training, COMSOL, data/NPZ mutation, or `CURRENT_BASELINE.md` changes. The intake contract now supports prepared `delta_b` and raw `b_defect + b_no_defect`, requires tri-axis `Bx/By/Bz`, `sensor_z_m`, no-defect reference provenance, Tesla units, coordinate system, sensor alignment status, gain status, and 201-sample `sensor_x`.

The validator can run without real data files and checks the manifest/schema first. The included template is intentionally not inference-ready until placeholders such as specimen material and magnetization setup are replaced. Bz-only data is a blocker for this route, and internal/buried defects remain a separate schema.

Only next step: perform a real-data manifest dry run. Start with metadata only: fill `results/templates/real_data_intake_manifest_template.json` or an equivalent manifest with actual `sensor_z_m`, no-defect reference, axis order, units, alignment, gain, specimen, and magnetization fields before attaching real signal arrays.
## 2026-05-29 after Stage 21.6 internal defect burial-depth refinement

下一步唯一建议：**A. internal benchmark rerun / candidate upgrade**。

21.6 证明 burial_depth 短板可以通过合法的 delta_b-derived feature fusion 改善。B2_feature_fusion_burial_head 在 multi-seed 中由 validation-only 选择 seed `2026`，test burial_depth MAE 从 21.4 neural 的 `0.595 mm` 降到 `0.413 mm`，并且优于 selected feature baseline 的 `0.472 mm`；test total normalized MAE 也从 `0.406366` 改善到 `0.395256`。代价是 center_xyz 从 `1.380 mm` 到 `1.466 mm`，shape F1 从 `1.000000` 到 `0.975309`，但没有触发 secondary metric collapse。

因此不要继续盲目加权 burial loss，也不要扩数据或改 schema。下一步应做 internal benchmark rerun / candidate upgrade：固定 B2 feature-fusion burial head，复核 seed stability、分组失败样本、feature-fusion 风险和与 21.4 neural / feature baseline 的正式比较。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline；internal defect 仍是独立 branch，不写成 baseline。

## 2026-05-29 after Stage 21.7 internal defect benchmark candidate

下一步唯一建议：**A. internal report / visualization package**。

21.7 固定 B2_feature_fusion_burial_head 做 formal rerun 后，validation-only 仍选择 seed `2026`，test total normalized MAE 为 `0.395256`，burial_depth MAE 为 `0.413 mm`，shape accuracy/F1 为 `0.975000 / 0.975309`。三 seed burial_depth MAE `0.399 / 0.428 / 0.413 mm` 均优于 21.4 neural `0.595 mm` 和 feature baseline `0.472 mm`，说明 B2 的 burial_depth 改善不是单 seed 偶然。

因此下一步不应直接 baseline transition，也不应立刻扩数据。应先做 internal report / visualization package：整理 by-shape、by-burial、by-size/aspect、failure cases 和可视化，解释 B2 在 center_xyz / shape 上的轻微代价，并明确 internal defect 仍是独立 benchmark candidate。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline。

## 2026-05-29 after Stage 21.8 internal defect benchmark report package

下一步唯一建议：**A. internal real-data schema alignment**。

21.8 已把 B2 internal benchmark candidate 的指标、分组风险和限制整理成报告包。B2 的 test total normalized MAE 为 `0.395256`，burial_depth MAE 为 `0.413 mm`，shape accuracy/F1 为 `0.975000 / 0.975309`；弱势分组集中在 `elongated_y`、`internal_ellipsoid`、`large`、`internal_cuboid`，以及 `deep_plus` 的 center/shape 风险。

当前真正缺口不再是继续扩仿真数据，而是真实 internal 样本能否满足 schema/metadata：缺陷中心和埋深 ground truth、no-defect reference、Bx/By/Bz、sensor_z_m、坐标系、单位、扫描线、sensor_x 对齐和 gain 状态。下一步先做 internal real-data schema alignment；如果要做 gallery 或 inference smoke，则先恢复 B2 inference artifact。`CURRENT_BASELINE.md` 不变，internal defect 仍是独立 benchmark branch。

## 2026-05-29 after Stage 21.9 internal B2 artifact recovery

下一步唯一建议：**internal inference smoke / visualization gallery**。

21.9 已恢复可加载的 B2 inference artifact。checkpoint 位于 ignored `checkpoints/internal_defect_b2_artifacts/internal_defect_b2_feature_fusion_seed2026.pt`，prediction artifact 位于同目录的 `_predictions.npz`，tracked manifest 是 `results/manifests/internal_defect_b2_inference_artifact_manifest.json`。verification 逐项复现 21.7/21.8：test total normalized MAE `0.395256`，burial_depth MAE `0.413 mm`，shape accuracy/F1 `0.975000 / 0.975309`，checkpoint reload diff 为 `0`。

因此现在可以做真正的 per-sample gallery / inference smoke，而不必再临时复训。下一步应读取 artifact manifest，生成 good/bad/failure/risk 样本图和逐样本索引；checkpoint/prediction artifact 继续不提交，`CURRENT_BASELINE.md` 继续不变。
## 2026-05-30 after Stage 22.2b internal hard-case top-up generation

下一步唯一建议：进入 **22.3 hard-case augmented internal training gate**。

22.2b 已按 22.2 hard-case plan 执行 COMSOL top-up：计划 `120` 行，成功 `120/120`，覆盖 cuboid/ellipsoid confusion、full-shift catastrophic、worst center、worst burial、compact medium/large、shallow/deep_plus 和 center-region neighbor targets。生成的 top-up NPZ 与 v3_hardcase assembled NPZ 都留在 ignored `data/` 路径，未提交；`CURRENT_BASELINE.md` 不变，internal defect 仍是独立分支。

新 assembled dataset 为 `comsol_internal_defect_pilot_pack_v3_hardcase`：source rows `240`，top-up rows `120`，assembled rows `360`，split=`240/60/60`，`train_ready_candidate=true`，`baseline_ready=false`。22.3 只应在该显式 manifest 上做 hard-case augmented training gate，不应进入真实 internal inference smoke 或 baseline transition。
## 2026-05-30 after Stage 22.3 internal hard-case augmented training

下一步唯一建议：先做第二轮 hard-case top-up 或 tail-specific refinement，不进入真实 internal inference smoke。

22.3 在 `comsol_internal_defect_pilot_pack_v3_hardcase` 上完成 hard-case augmented training gate。旧 B2 在 v3_hardcase test 上的 catastrophic failure 是 `12/60`，geometry_branch_failure 是 `3/60`；validation-only 选择的 `H2_B2_hardcase_tail_weighted` seed `42` 将 catastrophic failure 降到 `9/60`，geometry_branch_failure 降到 `2/60`，center p95/max 从 `12.077 / 22.544 mm` 降到 `8.886 / 14.608 mm`。

真正的分界点是 stable inference gate 仍未通过：catastrophic rate 仍为 `15%`，高于目标 `<=5%`，geometry branch 仍非零，burial max 从 `2.096 mm` 升到 `2.861 mm`，shape F1 从旧 B2 的 `0.841143` 降到 `0.778163`。internal defect 仍只能称为 benchmark branch，不是 stable inference model，也不是 `CURRENT_BASELINE.md`。
## 2026-05-30 after Stage 22.4 shape-preserving internal tail strategy

下一步唯一建议：训练 `A_train_freeze_shape_then_tail_regression_model`，不要继续直接做 H2 tail weighting。

22.4 的关键判断是，H2 不是单纯“还不够强”，而是优化方向把 shape branch 拉坏了。它把 center p95/max 从旧 B2 的 `12.077 / 22.544 mm` 降到 `8.886 / 14.608 mm`，但 shape F1 从 `0.841143` 降到 `0.778163`，burial max 从 `2.096 mm` 退化到 `2.861 mm`，所以继续加 hard-case 权重会继续在 shape 与 tail 之间拉扯。

下一阶段应先保护 shape classifier / shared encoder，再单独训练 center/burial tail heads；shape-confidence router 可以作为后续安全层，第二轮 hard-case top-up 只在 freeze-shape 后仍发现集中 strata failure 时再考虑。internal defect 仍是独立 benchmark branch，不是 stable inference model，也不是 `CURRENT_BASELINE.md`。
## 2026-05-30 after Stage 22.5 freeze-shape internal tail regression

下一步唯一建议：做 tail-specific refinement plus uncertainty/output gate，不进入真实 internal inference smoke。

22.5 验证了一个关键点：freeze-shape 能保住 shape branch，但不能自动解决 center/burial tail。F2 selected seed `42` 的 shape F1 是 `0.824172`，比 H2 的 `0.778163` 明显恢复，说明冻结 B2 shape/encoder 方向是对的；但 catastrophic failure 是 `11/60`，geometry_branch_failure 是 `4/60`，center p95/max 是 `8.940 / 22.017 mm`，burial p95/max 是 `1.841 / 2.490 mm`，都没有过 stable gate。

路线分界点是：问题已经不是“shape 被训练破坏”，而是 tail correction head 本身对最坏 center/burial case 不够可靠。下一步应改成更明确的 tail-specific objective 和 uncertainty/output gate：对高风险样本输出 unstable/abstain 或风险分数，同时重新设计 tail loss；不要把 F2 称为 stable inference model，也不要更新 `CURRENT_BASELINE.md`。

## 2026-05-30 after Stage 22.6 internal tail-risk gate

下一步唯一建议：进入 **internal inference smoke with abstention**，但不能声称 stable inference。

22.6 已验证一个可用的安全门控：`random_forest_small` risk gate 在 test split 上捕获了 `100%` catastrophic failure 和 `100%` geometry_branch_failure，false alarm rate 为 `0.417`，coverage retained 为 `0.283`。accept 后的 tail 明显收缩：center p95/max 从 `8.940 / 22.017 mm` 降到 `4.569 / 5.290 mm`，burial p95/max 从 `1.841 / 2.490 mm` 降到 `0.590 / 0.911 mm`。

真正的口径是：internal model 仍不能盲目输出稳定 center/burial；下一步只允许做带 `risk_score` 和 `abstain_need_review` 的 inference smoke。高风险样本不给确定几何结论，真实 internal sample 仍需先满足 metadata/schema。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline。

## 2026-05-30 after Stage 22.7 internal inference smoke with abstention

下一步唯一建议：做 **internal real-sample metadata alignment with abstention**，不是直接真实样品推理。

22.7 已打通带拒判的 internal inference smoke。B2 full-set test center p95/max 是 `12.077 / 22.544 mm`，burial p95/max 是 `1.693 / 2.096 mm`；使用 22.6 risk gate 后，accepted subset 的 center p95/max 降到 `4.832 / 4.962 mm`，burial p95/max 降到 `0.605 / 1.106 mm`。catastrophic 和 geometry_branch failure 都被捕获，false alarm 为 `0.381`，coverage retained 为 `0.283`。

真正的取舍是 coverage 很低：60 个 test 样本只有 17 个可 accept，所以这不是 stable all-sample predictor。下一步只能对真实 internal 样品先做 metadata/schema alignment，并且保留 `risk_score` / `abstain_need_review` 机制；缺 no-defect reference、Bx/By/Bz、sensor_z_m、坐标系、单位或 ground truth 时仍停止。

## 2026-05-30 after Stage 22.8 internal richer-observation feasibility plan

下一步唯一建议：执行 **22.9 richer-observation COMSOL diagnostic pack generation**。

22.8 的判断是，22.7 的高拒判率不是单纯模型阈值问题，而是当前观测配置可能信息不足。risk gate 能抓住 catastrophic / geometry branch，但 coverage retained 只有 `0.283`；failure cases 集中在 deep_plus、large、compact/elongated_y，以及 cuboid/ellipsoid hard cases。

22.9 第一轮只做 R0/R1/R2 diagnostic pack：30 个 base geometry，每个 base 6 个 paired variants，总计 180 rows；fallback 是 24 base / 144 rows。R1_more_y_lines 用来验证 center/lateral tail，R2_multi_liftoff 用来验证 burial/size 混淆。R3 multi-scan-direction 暂作第二优先级，R4 multi-magnetization 暂缓。不要在 22.9 里训练或接真实样品。

## 2026-05-30 after Stage 22.9 internal richer-observation pack generation

下一步唯一建议：进入 **23.0 richer-observation evaluation gate**。

22.9 已完成 COMSOL diagnostic pack generation：planned/success `180/180`，30 个 base 全部具备 6 个 paired variants，scan line 覆盖 `3/5/9`，liftoff 覆盖 `0.006/0.008/0.010/0.012 m`。新数据集 `comsol_internal_defect_richer_observation_pack_v1` 已通过 registry/manifest 显式注册，`validation_passed=true`，`status=diagnostic_pack_generated`，但 `train_ready_candidate=false`、`baseline_ready=false`。

23.0 不应直接训练，也不应直接接真实样品；应先比较 R0/R1/R2 是否真的降低 center/burial tail、geometry branch risk 和 abstention rate。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline，internal defect 仍是独立 diagnostic branch。

## 2026-05-31 after Stage 23.1 internal richer-observation training gate

下一步唯一建议：进入 **23.2 internal multi-scan-direction plan**。

23.1 已按 23.0 的 validation-only 选择训练 `R1_plus_R2_combined` 输入，但结果没有过 stable inference gate：selected `O3_richer_observation_tail_aware` seed `2026` 的 test total normalized MAE 为 `0.629543`，shape F1 为 `0.600000`，catastrophic failure 为 `4/5`，geometry_branch_failure 为 `1/5`，center p95/max 为 `7.314 / 7.531 mm`，burial p95/max 为 `1.966 / 2.180 mm`。

真正的分界点是：更多 y-lines 和 multi-liftoff 没有在 30-base diagnostic scope 内解决几何分支错位，下一步不应继续直接调 O3，也不应进入真实 internal sample inference。应先规划 R3 multi-scan-direction diagnostic，验证双扫描方向是否能补足 cuboid/ellipsoid 和 elongated aspect 的形状判别信息。`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline，internal branch 仍是独立 diagnostic / benchmark 分支。
## 2026-05-31 after Stage 23.2 internal multi-scan-direction plan

下一步唯一建议：执行 **23.2b internal multi-scan-direction generation**。

23.2 的核心判断是：23.1 失败不是继续调 O3 能解决的问题，而是当前 internal observation 缺少正交扫描方向。R1/R2 已经补了更多 y-lines 和 multi-liftoff，但 shape F1 仍只有 `0.600000`，catastrophic failure `4/5`，geometry branch `1/5`；因此下一步应补生成 y_scan 的 `5-line` 和 `9-line` 观测，与既有 x_scan 数据配对，验证 cuboid/ellipsoid 和 elongated aspect 的方向性信息是否真的不足。

23.2b 只应运行 COMSOL diagnostic pack generation，不训练、不更新 `CURRENT_BASELINE.md`、不提交 data/NPZ/.mph/raw CSV/checkpoint/preview/notes。生成目标是复用 22.9 的 30 个 base，补 `60` 行 y_scan；fallback 是 24 个 base / 48 行。COMSOL 侧必须实现真正的 direction-aware sensor point builder，不能只写 `scan_direction=y_scan` metadata。

## 2026-05-31 after Stage 23.2b internal multi-scan-direction pack

下一步唯一建议：进入 **23.3 internal multi-scan-direction diagnostic evaluation**。

23.2b 已经完成 y_scan 方向化 COMSOL 生成和 dual-direction assembly：planned/success `60/60`，30 个 base 全部有 `D1_y_scan_5line_z0p008` 与 `D2_y_scan_9line_z0p008`，并且与既有 x_scan `R1_5line_z0p008` / `R1_9line_z0p008` 成对。真正的分界点是这次不只是写了 `scan_direction=y_scan` metadata，而是把传感器点改成 `(x_line, y_path, sensor_z_m)`，也就是路径沿 y 方向、line offset 沿 x 方向。

assembled dataset 为 `comsol_internal_defect_multi_scan_direction_pack_v1`，`delta_b` shape 是 `(60,3,2,9,201)`，`validation_passed=true`，`train_ready_candidate=false`，`baseline_ready=false`。23.3 只应该先评估双方向观测是否缓解 cuboid/ellipsoid 与 elongated aspect 的几何分支错误；不要直接训练或接真实 internal sample，`CURRENT_BASELINE.md` 继续保持 surface / near-surface true 3D RBC baseline。

## 2026-05-31 after Stage 23.3 internal multi-scan-direction diagnostic evaluation

下一步唯一建议：执行 **multi-magnetization diagnostic pack plan / generation**，不要进入 23.4 dual-direction training gate。

23.3 证明 y_scan 不是冗余观测：D1/D2 paired completeness 为 `30/30`，assembled `delta_b=(60,3,2,9,201)`，D1/D2 的 y/x RMS 接近 1 且方向相关性低。但轻量 probe 没有证明 dual-direction 比 x-only 更稳：best validation-selected test config 是 `single_x_9line`，而 dual_xy_5line 只改善 center tail、牺牲 burial 与 shape，dual_xy_9line 只改善 burial tail、牺牲 center 与 shape。

因此当前瓶颈不应再归因于“缺一个正交扫描方向”本身。下一步应检查更高信息量的源/磁化观测轴；internal branch 继续是 diagnostic / benchmark branch，不是 stable inference model，也不进入 `CURRENT_BASELINE.md`。
## 2026-06-03 after Stage 25.14 surface RBC targeted top-up calibration

Only next step: create deterministic replacements for the two calibration blockers, preserving the exact coverage signatures, then rerun calibration from zero.

Stage 25.14 generated the 120-row top-up plan and verified the COMSOL orchestrator dry-run/chunk isolation, but real `workers=1` calibration stopped the route: `22/24` samples passed and two sample-level geometry failures were classified as `sample_geometry_failure`. The required replacement signatures are `balanced_interior|medium|narrow|sharp|interior` for `surface_rbc_targeted_008_balanced_interior_sharp_medium_narrow` and `balanced_interior|deep|balanced|round|interior` for `surface_rbc_targeted_022_balanced_interior_round_deep_balanced`.

Do not run full 120, do not run `+120 training gate`, and do not assemble `v3_240 + topup_v1_120` until calibration is zero-failure and validation passes. `CURRENT_BASELINE.md` remains unchanged.

## 2026-06-03 after Stage 25.16 surface multi-pit label-v3b derivation

Only next step: **enter 25.17 label-v3b training gate using the 25.10 loss mainline plus label-v3b supervision; do not use the 25.11/25.12 rebalance stack**.

25.16 resolved the immediate label-target blocker without touching data/NPZ: v3b keeps the v2 exclusive hard identity, adds a capped one-pixel soft halo for anti-sparsity, and prevents v3-style union-like support leakage. The validator passed with v3b soft OR/raw union mean/max `1.247726 / 1.250000`, v3b/v3 shrink ratio mean `0.627196`, duplicate hard ownership `0`, and empty slot violations `0`.

This authorizes only an explicit 25.17 training gate. It is not a baseline transition, does not update `CURRENT_BASELINE.md`, and should not continue loss tuning from the failed 25.11/25.12 rebalance stack.

## 2026-06-03 after Stage 25.17 label-v3b training gate

Only next step: **enter 25.17b label-v3b failure audit focused on hard-core/halo/SDF/depth-valid-region usage**.

25.17 used the 25.10 component-set architecture, split, Hungarian matching, and `component_set_gate_v1` loss mainline with `target_version=v3b`; it did not use the failed 25.11/25.12 rebalance stack and did not update `CURRENT_BASELINE.md`. The gate decision is `PARTIAL`: near-empty collapse improved versus 25.13, but merged rate stayed `1.000000` as in 25.15, so v3b still behaves like a union-like merged branch under this gate.

Do not continue by tuning loss weights or expanding model capacity. The next audit must check whether hard-core identity supervision, halo/SDF support, depth valid regions, and overlap ignore masks are actually used in the loader/loss/metrics as intended.

## 2026-06-03 after surface RBC +120 assembly/training gate

Only next step for the surface RBC expansion line: do not promote `comsol_true_3d_rbc_surface_expansion_v1_360`; analyze why the +120 train split improves top-up test but regresses old v3_240 test before any further assembly/training attempt.

The gate assembled v3_240 plus the validated top-up into `comsol_true_3d_rbc_surface_expansion_v1_360`, N=`360`, split=`242/59/59`, and validation passed. Training used the same 20.85-style Conv1D six-parameter model with validation-only seed selection and selected seed `2026`.

Gate outcome is `FAIL`: old v3_240 test profile RMSE `0.000429130307356 m`, Er-like `0.432564256474`, and D MAE `0.921162613500 mm` exceed the required non-regression thresholds. `CURRENT_BASELINE.md` remains the 20.85 nominal true 3D RBC baseline.
