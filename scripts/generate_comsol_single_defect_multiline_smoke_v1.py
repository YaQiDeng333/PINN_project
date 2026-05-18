from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "comsol_mfl"
    / "generated"
    / "comsol_single_defect_multiline_forward_pack_v1"
)
DEFAULT_PREPARED_NPZ = (
    PROJECT_ROOT
    / "data"
    / "comsol_mfl"
    / "prepared"
    / "comsol_single_defect_multiline_forward_pack_v1_smoke.npz"
)
DEFAULT_COMSOL_MCP_ROOT = Path(
    r"C:\Users\19166\Desktop\COMSOL_Multiphysics_MCP"
)
MAX_ALLOWED_SAMPLES = 3
ENV_KEYS = (
    "COMSOL_HOME",
    "COMSOLPATH",
    "JAVA_HOME",
    "JRE_HOME",
    "MPH_SERVER",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run-first scaffold for a minimal COMSOL single-defect "
            "multi-line delta_Bz forward smoke pack. This script does not "
            "fabricate signals or masks."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_false",
        dest="dry_run",
        help=(
            "Attempt real generation. Currently this exits with blockers unless "
            "a supported COMSOL multi-line generation backend is available."
        ),
    )
    parser.set_defaults(dry_run=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Number of requested smoke samples. Must be between 1 and 3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Target directory for generated smoke files if real generation is enabled.",
    )
    parser.add_argument(
        "--prepared-npz",
        type=Path,
        default=DEFAULT_PREPARED_NPZ,
        help="Target NPZ path if all required real fields are generated.",
    )
    parser.add_argument(
        "--comsol-mcp-root",
        type=Path,
        default=DEFAULT_COMSOL_MCP_ROOT,
        help="Path to the local COMSOL_Multiphysics_MCP checkout.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def smoke_samples(max_samples: int) -> list[dict[str, Any]]:
    base_samples = [
        {
            "sample_id": "smoke_rect_001",
            "defect_type": "rectangular_notch",
            "geometry_params": {
                "center_x_m": 0.0,
                "center_y_m": 0.0,
                "width_m": 0.008,
                "length_m": 0.008,
                "depth_m": 0.001,
                "angle_rad": 0.0,
            },
        },
        {
            "sample_id": "smoke_rect_002",
            "defect_type": "rectangular_notch",
            "geometry_params": {
                "center_x_m": 0.006,
                "center_y_m": 0.0,
                "width_m": 0.012,
                "length_m": 0.008,
                "depth_m": 0.002,
                "angle_rad": 0.0,
            },
        },
        {
            "sample_id": "smoke_rect_003",
            "defect_type": "rectangular_notch",
            "geometry_params": {
                "center_x_m": -0.006,
                "center_y_m": 0.0,
                "width_m": 0.016,
                "length_m": 0.008,
                "depth_m": 0.003,
                "angle_rad": 0.0,
            },
        },
    ]
    return base_samples[:max_samples]


def check_environment(comsol_mcp_root: Path) -> dict[str, Any]:
    external_generator = comsol_mcp_root / "scripts" / "generate_mfl_rectangular_sweep.py"
    external_config = comsol_mcp_root / "configs" / "mfl_rectangular_sweep_small.json"
    opencode_config = comsol_mcp_root / "opencode.json"
    path_matches = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if any(token in entry.lower() for token in ("comsol", "java", "jdk", "jre", "mph"))
    ]
    return {
        "comsol_mcp_root": str(comsol_mcp_root),
        "comsol_mcp_root_exists": comsol_mcp_root.exists(),
        "external_rectangular_generator": str(external_generator),
        "external_rectangular_generator_exists": external_generator.exists(),
        "external_rectangular_config": str(external_config),
        "external_rectangular_config_exists": external_config.exists(),
        "opencode_mcp_config": str(opencode_config),
        "opencode_mcp_config_exists": opencode_config.exists(),
        "python_executable": sys.executable,
        "mph_import_available": importlib.util.find_spec("mph") is not None,
        "environment_variables": {key: os.environ.get(key) for key in ENV_KEYS},
        "path_entries_matching_comsol_or_java": path_matches,
        "mcp_server_command_from_opencode": ["python", "-m", "src.server"]
        if opencode_config.exists()
        else None,
        "known_generator_limitations": [
            "external generator is rectangular_notch only",
            "external generator uses one sensor_line.y_m, not multi-line scan_line_y",
            "external generator does not emit 2D/quasi-2D masks",
            "external generator does not emit mask_x/mask_y coordinate mapping",
        ],
    }


def build_generation_plan(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = normalize_path(args.output_dir)
    prepared_npz = normalize_path(args.prepared_npz)
    comsol_mcp_root = normalize_path(args.comsol_mcp_root)
    environment = check_environment(comsol_mcp_root)
    blockers = []
    if not environment["comsol_mcp_root_exists"]:
        blockers.append("COMSOL_Multiphysics_MCP root is missing.")
    if not environment["external_rectangular_generator_exists"]:
        blockers.append("No existing rectangular COMSOL generation script was found.")
    if not environment["mph_import_available"]:
        blockers.append("The current Python environment cannot import the mph package.")
    blockers.extend(
        [
            "No implemented PINN_project generation backend creates multi-line delta_Bz.",
            "No implemented PINN_project generation backend creates 2D/quasi-2D masks.",
            "No implemented PINN_project generation backend records mask_x/mask_y mapping.",
        ]
    )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "max_samples": int(args.max_samples),
        "output_dir": str(output_dir),
        "prepared_npz": str(prepared_npz),
        "environment": environment,
        "requested_pack": "comsol_single_defect_multiline_forward_pack_v1",
        "sensor_config": {
            "sensor_x_start_m": -0.04,
            "sensor_x_stop_m": 0.04,
            "sensor_points": 201,
            "scan_line_y_m": [-0.004, 0.0, 0.004],
            "sensor_z_m": 0.002,
            "signal_kind": "delta_Bz",
        },
        "mask_grid": {
            "mask_x_start_m": -0.04,
            "mask_x_stop_m": 0.04,
            "mask_y_start_m": -0.01,
            "mask_y_stop_m": 0.01,
            "height": 64,
            "width": 128,
            "value_convention": "defect=1, background=0",
        },
        "samples": smoke_samples(int(args.max_samples)),
        "required_output_fields": [
            "sample_id",
            "defect_type",
            "geometry_params",
            "mask_x",
            "mask_y",
            "masks",
            "sensor_x",
            "scan_line_y",
            "bz_defect",
            "bz_reference",
            "delta_bz",
            "metadata",
        ],
        "minimum_external_csv_schema": {
            "manifest": [
                "sample_id",
                "defect_type",
                "geometry_params_json",
                "signal_kind",
                "n_lines",
                "signal_length",
                "mask_shape",
                "sensor_x_path",
                "scan_line_y_path",
                "bz_defect_path",
                "bz_reference_path",
                "delta_bz_path",
                "mask_path",
                "mask_x_path",
                "mask_y_path",
                "units",
                "comsol_model_path",
            ],
            "per_sample_signal_arrays": {
                "sensor_x": "(L,)",
                "scan_line_y": "(n_lines,)",
                "bz_defect": "(n_lines, L)",
                "bz_reference": "(n_lines, L)",
                "delta_bz": "(n_lines, L)",
                "mask": "(H, W)",
                "mask_x": "(W,)",
                "mask_y": "(H,)",
            },
        },
        "minimum_npz_schema": {
            "delta_bz_or_signals": "(N, n_lines, L)",
            "bz_defect": "(N, n_lines, L)",
            "bz_reference": "(N, n_lines, L)",
            "masks": "(N, H, W)",
            "sensor_x": "(L,)",
            "scan_line_y": "(n_lines,)",
            "mask_x": "(W,)",
            "mask_y": "(H,)",
            "defect_types": "(N,)",
            "sample_ids": "(N,)",
            "geometry_params": "json strings or structured table",
            "metadata": "units, COMSOL settings, generation date, signal_kind",
        },
        "reference_defect_flow": [
            "Build or load a no-defect COMSOL model on the fixed geometry.",
            "Evaluate Bz at all sensor_x and scan_line_y points.",
            "Build or load the matching defect COMSOL model for each sample.",
            "Evaluate Bz on the same fixed sensor coordinates.",
            "Compute delta_Bz = Bz_defect - Bz_reference.",
            "Rasterize mask from the same geometry_params on mask_x/mask_y.",
            "Write per-sample manifest and optional NPZ only when all fields are real.",
        ],
        "scenario_a_if_pinn_project_calls_comsol": {
            "required_first_fix": "Provide a stable COMSOL backend callable in this Python environment or via MCP under 120 seconds.",
            "required_generation_change": "Implement multi-line point evaluation for scan_line_y and mask rasterization from geometry_params.",
            "safe_sample_limit": MAX_ALLOWED_SAMPLES,
        },
        "scenario_b_if_external_comsol_project_generates_data": {
            "recommended": True,
            "external_project": str(comsol_mcp_root),
            "pinn_project_role": "ingest/prepare/inspect only",
            "required_deliverable": "real CSV/array bundle matching minimum_external_csv_schema",
        },
        "recommended_path": (
            "Generate real COMSOL multi-line samples in the external "
            "COMSOL_Multiphysics_MCP project, then ingest them in PINN_project."
        ),
        "can_execute_real_generation": False,
        "blockers": blockers,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 1 or args.max_samples > MAX_ALLOWED_SAMPLES:
        raise ValueError("--max-samples must be between 1 and 3 for this smoke entry.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    plan = build_generation_plan(args)

    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("DRY_RUN: no COMSOL model, signal, mask, CSV, or NPZ was generated.")
        return 0

    print(
        "REAL_GENERATION_BLOCKED: this scaffold has no supported multi-line COMSOL "
        "generation backend yet; refusing to fabricate data.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
