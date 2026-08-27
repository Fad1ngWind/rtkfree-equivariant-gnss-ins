#!/usr/bin/env python3
"""Verify fixed Phase 3 Medium outputs without any reference trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from rtkfree_equivariant_gnss_ins.phase3_config import load_fixed_eskf_config
from rtkfree_equivariant_gnss_ins.phase3_validate import validate_baseline_outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Phase 3 run and an optional independent repeat."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "phase3" / "fixed_eskf_v1.json",
    )
    args = parser.parse_args()
    report = validate_baseline_outputs(
        load_fixed_eskf_config(args.config),
        args.run_dir,
        args.compare_dir,
    )
    print("PASS Phase 3 fixed-baseline structural validation")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
