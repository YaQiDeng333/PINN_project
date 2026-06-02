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
    assert manifest["train_ready_candidate"] is False
    assert manifest["baseline_ready"] is False
    assert "assembled" not in manifest["dataset_role"]
    assert "explicit_surface_rbc_expansion_training_gate" in manifest["allowed_use"]

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
        test_manifest_and_route_are_topup_only,
        test_mesh_wrapper_supports_calibration_batches,
    ]
    for test in tests:
        test()
    print(f"surface RBC targeted expansion synthetic tests passed: {len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
