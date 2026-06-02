#!/usr/bin/env python
"""Write the 25.7 surface forward-refinement inference contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from build_surface_forward_refinement_target_set import ROOT
from run_surface_forward_refinement_inference import DEFAULT_MANIFEST, METRICS as RUNNER_METRICS
from verify_surface_forward_refinement_inference_runner import MATRIX as VERIFICATION_MATRIX


CONTRACT = ROOT / "results/summaries/surface_forward_refinement_inference_contract.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if not DEFAULT_MANIFEST.exists():
        raise FileNotFoundError(DEFAULT_MANIFEST)
    manifest = read_json(DEFAULT_MANIFEST)
    lines = [
        "# 25.7 Surface Forward-Refinement Inference Contract",
        "",
        "This runner is a post-hoc surface refinement companion. It is not a baseline replacement and it does not authorize a `CURRENT_BASELINE.md` transition.",
        "",
        "## Applicable Scope",
        "",
        "- Use only for RBC-representable surface defects where a six-parameter RBC profile is an acceptable representation and the frozen 20.85 model produced a poor initial estimate.",
        "- The fixed runtime chain is frozen 20.85 baseline prediction -> observed `delta_b` feature extraction -> exported F0/R1 artifact -> refined `L_m/W_m/D_m/wLD/wWD/wLW`.",
        "- Runtime refinement inputs are observed `delta_b`-derived features, frozen 20.85 predicted six params, parameter bounds, and the exported artifact.",
        "",
        "## Not Applicable Scope",
        "",
        "- Multi-pit, component-set, multi-component, or clearly non-RBC-representable defects are not suitable for six-parameter RBC refinement.",
        "- If metadata shows `multi_component`, `component_set`, or `multi_pit_two_component_surface_defect`, the runner must mark the sample `not_suitable_for_rbc_refinement` and must not count it as RBC-refinement success.",
        "- Unknown real samples can only report `refinement_applied`; they cannot claim representable success unless an oracle label, validated ground truth, or human review confirms representability.",
        "",
        "## Required Outputs",
        "",
        "- Output both baseline six params and refined six params.",
        "- Generate the refined RBC profile and projected mask in memory for metrics/reporting; do not write NPZ, checkpoint, or preview PNG artifacts.",
        "- When labels exist, metrics may include profile RMSE, Er-like error, projected-mask IoU/Dice, area error, component count, and failure-case rows.",
        "- Labels, oracle params, true masks, and true depth maps are allowed only for evaluation metrics, never as refinement inputs.",
        "",
        "## Artifact Lock",
        "",
        f"- Artifact manifest: `{DEFAULT_MANIFEST}`",
        f"- Ignored artifact path: `{manifest.get('artifact_path')}`",
        f"- Artifact id: `{manifest.get('artifact_id')}`",
        f"- Selected surrogate: `{manifest.get('selected_surrogate')}`",
        f"- Alpha: `{manifest.get('alpha')}`",
        f"- Lambda profile / parameter: `{manifest.get('lambda_profile')}` / `{manifest.get('lambda_param')}`",
        "- Allowed use: `explicit_surface_forward_refinement_inference`",
        "- Forbidden use: `current_baseline_replacement`, `automatic_baseline_update`",
        "",
        "## Baseline Boundary",
        "",
        "- `CURRENT_BASELINE.md` remains the 20.85 surface RBC baseline.",
        "- This runner is a companion/post-hoc refinement layer over the frozen baseline output.",
        "- A baseline transition would require a separate user request, formal benchmark, review, and baseline-transition gate.",
        "",
        "## Evidence Files",
        "",
        f"- Runner metrics: `{RUNNER_METRICS}`",
        f"- Verification matrix: `{VERIFICATION_MATRIX}`",
    ]
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
