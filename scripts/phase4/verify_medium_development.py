#!/usr/bin/env python3
"""Verify Phase 4 development artifacts without deployable data or reference truth."""

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
from rtkfree_equivariant_gnss_ins.phase4_validate import (
    compare_phase4_development_runs,
    validate_phase4_development_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Phase 4 development run and optional independent repeat."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument(
        "--diagnostics-name",
        default="diagnostics_with_outage_detail.json",
    )
    args = parser.parse_args()
    if "RTKFREE_SEALED_REFERENCE_ROOT" in os.environ:
        raise ValueError("sealed-reference environment must be absent during Phase 4 verification")
    if torch_runtime_is_not_locked():
        raise ValueError("Phase 4 verification requires PyTorch 2.13.0+cpu with no CUDA runtime")

    config = load_phase4_config(ROOT / "config" / "phase4" / "ordinary_student_v1.json")
    run_report = validate_phase4_development_run(
        config,
        args.run_dir,
        args.diagnostics_name,
    )
    report: dict[str, object] = {"run": run_report}
    if args.compare_dir is not None:
        comparison_report = validate_phase4_development_run(
            config,
            args.compare_dir,
            args.diagnostics_name,
        )
        report["compare"] = comparison_report
        report["determinism"] = compare_phase4_development_runs(
            run_report,
            comparison_report,
        )
    print("PASS Phase 4 reference-free development artifact validation")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def torch_runtime_is_not_locked() -> bool:
    import torch

    return torch.__version__ != "2.13.0+cpu" or torch.version.cuda is not None


if __name__ == "__main__":
    raise SystemExit(main())
