#!/usr/bin/env python3
"""Read-only verifier for the complete Phase 6 external run directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from rtkfree_equivariant_gnss_ins.phase4_config import load_phase4_config
from rtkfree_equivariant_gnss_ins.phase5_config import load_phase5_config
from rtkfree_equivariant_gnss_ins.phase6_config import load_phase6_config
from rtkfree_equivariant_gnss_ins.phase6_validate import validate_phase6_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify one complete Phase 6 run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    phase4_path = ROOT / "config" / "phase4" / "ordinary_student_v1.json"
    phase5_path = ROOT / "config" / "phase5" / "gravity_aware_so2_v1.json"
    phase6_path = ROOT / "config" / "phase6" / "final_freeze_v1.json"
    phase4 = load_phase4_config(phase4_path)
    phase5 = load_phase5_config(phase5_path, phase4_path)
    phase6 = load_phase6_config(phase6_path, phase4_path, phase5_path)
    report = validate_phase6_run(
        phase4, phase5, phase6, phase6_path, args.run_dir
    )
    print("PASS Phase 6 prospective run validation")
    print(f"selected_structure={report['selected_structure']}")
    print(f"selected_physics_weight={report['selected_physics_weight']}")
    print(f"validated_artifact_count={report['validated_artifact_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
