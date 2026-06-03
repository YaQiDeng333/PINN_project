#!/usr/bin/env python
"""Synthetic contract tests for the surface RBC targeted top-up branch."""

from __future__ import annotations

from collections import Counter


def test_plan_contract() -> None:
    import design_surface_rbc_targeted_expansion_plan as plan_mod

    rows = plan_mod.build_plan()
    assert len(rows) == 120
    assert Counter(row["split"] for row in rows) == {"train": 80, "val": 20, "test": 20}
    assert Counter(row["targeted_role"] for row in rows) == {
        "balanced_interior": 60,
        "hard_depth_aspect": 24,
        "edge_position": 24,
        "old_distribution_anchor": 12,
    }
    assert len({row["sample_id"] for row in rows}) == 120
    assert all(row["exact_piao_rbc"] == "False" for row in rows)
    assert all(row["rbc_style_approximation"] == "True" for row in rows)
    assert all(float(row["sensor_z_m"]) == 0.008 for row in rows)
    assert {row["axis_order"] for row in rows} == {"Bx,By,Bz"}
    assert {row["scan_line_y_m"] for row in rows} == {"-0.001,0.0,0.001"}
    assert {row["sensor_x_count"] for row in rows} == {"201"}
    assert all(row["edge_position_bin"] for row in rows)


def test_replacement_signature_preserves_coverage() -> None:
    import design_surface_rbc_targeted_expansion_plan as plan_mod

    rows = plan_mod.build_plan()
    row = next(item for item in rows if item["targeted_role"] == "edge_position")
    replacement = plan_mod.make_replacement_row(row, replacement_index=1)
    assert replacement["sample_id"] != row["sample_id"]
    assert plan_mod.coverage_signature(replacement) == plan_mod.coverage_signature(row)


def test_known_comsol_blockers_use_safe_replacements() -> None:
    import design_surface_rbc_targeted_expansion_plan as plan_mod

    rows = plan_mod.build_plan()
    by_original = {row["replacement_of_sample_id"]: row for row in rows if row["replacement_of_sample_id"]}
    original_rows = []
    for original_id in plan_mod.COMSOL_SAFE_REPLACEMENTS:
        table = plan_mod.COMSOL_SAFE_REPLACEMENTS[original_id]
        replacement = by_original[original_id]
        original = plan_mod.make_row(
            sample_id=original_id,
            split=replacement["split"],
            spec=next(spec for spec in plan_mod.selected_specs(replacement["targeted_role"]) if spec.depth_bin == replacement["depth_bin"] and spec.aspect_bin == replacement["aspect_bin"] and spec.curvature_template == replacement["curvature_template"]),
            role=replacement["targeted_role"],
            edge_position_bin=replacement["edge_position_bin"],
            variant=1,
            temp_mesh_dir=plan_mod.DEFAULT_TEMP_MESH_DIR,
        )
        original_rows.append(original)
        assert replacement["sample_id"] == table["sample_id"]
        assert replacement["replacement_of_sample_id"] == original_id
        assert replacement["coverage_signature"] == original["coverage_signature"]
        assert "deterministic COMSOL-safe replacement" in replacement["notes"]
        assert "multi" not in replacement["sample_id"].lower()
        assert "internal" not in replacement["sample_id"].lower()
        assert "buried" not in replacement["sample_id"].lower()

    assert len(original_rows) == len(plan_mod.COMSOL_SAFE_REPLACEMENTS)

    sharp = by_original["surface_rbc_targeted_008_balanced_interior_sharp_medium_narrow"]
    assert sharp["sample_id"].endswith("_repl03")
    assert sharp["coverage_signature"] == "balanced_interior|medium|narrow|sharp|interior"
    assert (float(sharp["L_m"]), float(sharp["W_m"]), float(sharp["D_m"])) == (0.024, 0.0095, 0.0031)
    assert (float(sharp["wLD"]), float(sharp["wWD"]), float(sharp["wLW"])) == (0.55, 0.6105, 0.55)

    round_deep = by_original["surface_rbc_targeted_022_balanced_interior_round_deep_balanced"]
    assert round_deep["sample_id"].endswith("_repl01")
    assert round_deep["coverage_signature"] == "balanced_interior|deep|balanced|round|interior"
    assert (float(round_deep["wLD"]), float(round_deep["wWD"]), float(round_deep["wLW"])) == (0.722, 0.6965, 0.713)


def test_manifest_and_route_are_topup_only() -> None:
    import decide_surface_rbc_targeted_expansion_route as route_mod
    import validate_surface_rbc_targeted_expansion_pack as validate_mod

    manifest = validate_mod.build_manifest(
        n_success=120,
        validation_pass=True,
        npz_sha256="0" * 64,
        pinn_commit="pinn",
        comsol_commit="comsol",
    )
    assert manifest["dataset_id"] == "comsol_true_3d_rbc_surface_targeted_topup_v1_120"
    assert manifest["dataset_role"] == "topup_source"
    assert manifest["train_ready_candidate"] is True
    assert manifest["baseline_ready"] is False
    assert "assembled" not in manifest["dataset_role"]
    assert "explicit_surface_rbc_expansion_training_gate" in manifest["allowed_use"]
    partial_manifest = validate_mod.build_manifest(
        n_success=119,
        validation_pass=True,
        npz_sha256="0" * 64,
        pinn_commit="pinn",
        comsol_commit="comsol",
    )
    assert partial_manifest["train_ready_candidate"] is False
    assert partial_manifest["baseline_ready"] is False

    decision = route_mod.decide_route(
        calibration_pass=True,
        full_success=True,
        validation_pass=True,
        n_success=120,
        systemic_blocker=False,
    )
    assert decision["can_enter_training_gate"] is True
    assert decision["creates_assembled_dataset"] is False
    assert "assemble v3_240 + topup_v1_120" in decision["next_step"]


def test_mesh_wrapper_supports_calibration_batches() -> None:
    import argparse

    import generate_surface_rbc_targeted_expansion_meshes as mesh_wrapper

    args = argparse.Namespace(max_samples=24, min_success=1)
    mesh_wrapper.mesh_stage.MIN_SUCCESS = 30
    mesh_wrapper.mesh_stage.FULL_SUCCESS = 54
    try:
        original_run = mesh_wrapper.mesh_stage.run
        mesh_wrapper.mesh_stage.run = lambda ns: 0
        assert mesh_wrapper.run(args) == 0
        assert mesh_wrapper.mesh_stage.MIN_SUCCESS == 1
        assert mesh_wrapper.mesh_stage.FULL_SUCCESS == 24
    finally:
        mesh_wrapper.mesh_stage.run = original_run


def run() -> int:
    tests = [
        test_plan_contract,
        test_replacement_signature_preserves_coverage,
        test_known_comsol_blockers_use_safe_replacements,
        test_manifest_and_route_are_topup_only,
        test_mesh_wrapper_supports_calibration_batches,
    ]
    for test in tests:
        test()
    print(f"surface RBC targeted expansion synthetic tests passed: {len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
