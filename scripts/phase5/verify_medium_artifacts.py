#!/usr/bin/env python3
"""Verify Phase 5 repository-external artifacts without data or reference truth."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from rtkfree_equivariant_gnss_ins.phase4_config import load_phase4_config
from rtkfree_equivariant_gnss_ins.phase5_config import load_phase5_config
from rtkfree_equivariant_gnss_ins.phase5_validate import (
    compare_phase5_development_runs,
    compare_phase5_smoke_runs,
    validate_phase5_development_run,
    validate_phase5_smoke_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Phase 5 smoke/development artifacts and optional repeats."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--compare-smoke-dir", type=Path)
    parser.add_argument("--diagnostics-name", default="diagnostics_dtype_audit.json")
    parser.add_argument("--failed-diagnostics-name", default="diagnostics.json")
    args = parser.parse_args()
    if "RTKFREE_SEALED_REFERENCE_ROOT" in os.environ:
        raise ValueError("sealed-reference environment must be absent during Phase 5 verification")
    if torch_runtime_is_not_locked():
        raise ValueError("Phase 5 verification requires PyTorch 2.13.0+cpu with no CUDA runtime")
    if args.compare_smoke_dir is not None and args.smoke_dir is None:
        raise ValueError("--compare-smoke-dir requires --smoke-dir")
    for path in (args.run_dir, args.compare_dir, args.smoke_dir, args.compare_smoke_dir):
        if path is not None and path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Phase 5 run artifacts must remain outside the repository")

    phase4_config = load_phase4_config(ROOT / "config" / "phase4" / "ordinary_student_v1.json")
    phase5_config = load_phase5_config(
        ROOT / "config" / "phase5" / "gravity_aware_so2_v1.json",
        ROOT / "config" / "phase4" / "ordinary_student_v1.json",
    )
    run_report = validate_phase5_development_run(
        phase4_config,
        phase5_config,
        args.run_dir,
        diagnostics_name=args.diagnostics_name,
        failed_diagnostics_name=args.failed_diagnostics_name,
    )
    report: dict[str, object] = {"development": run_report}
    if args.compare_dir is not None:
        comparison_report = validate_phase5_development_run(
            phase4_config,
            phase5_config,
            args.compare_dir,
        )
        report["development_repeat"] = comparison_report
        report["development_determinism"] = compare_phase5_development_runs(
            run_report, comparison_report
        )
    if args.smoke_dir is not None:
        smoke_report = validate_phase5_smoke_run(
            phase4_config, phase5_config, args.smoke_dir
        )
        report["smoke"] = smoke_report
        if args.compare_smoke_dir is not None:
            smoke_repeat = validate_phase5_smoke_run(
                phase4_config, phase5_config, args.compare_smoke_dir
            )
            report["smoke_repeat"] = smoke_repeat
            report["smoke_determinism"] = compare_phase5_smoke_runs(
                smoke_report, smoke_repeat
            )

    print("COMPLETE Phase 5 repository-external artifact audit")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def torch_runtime_is_not_locked() -> bool:
    import torch

    return torch.__version__ != "2.13.0+cpu" or torch.version.cuda is not None


if __name__ == "__main__":
    raise SystemExit(main())
