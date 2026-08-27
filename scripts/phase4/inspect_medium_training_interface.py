#!/usr/bin/env python3
"""Inspect the reference-isolated Medium Phase 4 interface without training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from rtkfree_equivariant_gnss_ins.phase3_config import load_fixed_eskf_config
from rtkfree_equivariant_gnss_ins.phase4_config import load_phase4_config
from rtkfree_equivariant_gnss_ins.phase4_data import assemble_phase4_sequence


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the frozen Medium Phase 4 training interface.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-run-dir", type=Path, required=True)
    args = parser.parse_args()
    if "RTKFREE_SEALED_REFERENCE_ROOT" in os.environ:
        raise ValueError("sealed-reference environment must be absent during Phase 4 development")

    commit = "075f96b6a6d9252b37486ecb175b4ae690c56f54"
    session = "UrbanNav-HK-Medium-Urban-1"
    phase4_config = load_phase4_config(ROOT / "config" / "phase4" / "ordinary_student_v1.json")
    phase3_config = load_fixed_eskf_config(ROOT / "config" / "phase3" / "fixed_eskf_v1.json")
    sequence = assemble_phase4_sequence(
        phase4_config,
        phase3_config,
        args.data_root / "standardized" / "UrbanNav" / commit / session / "gps_l1ca_broadcast_spp_v1" / "pvt.jsonl",
        args.data_root / "standardized" / "UrbanNav" / commit / session / "imu_contract_v1" / "imu.csv",
        args.data_root / "deployable" / "UrbanNav" / commit / session / "calibration" / "gnss_imu_extrinsic.json",
        args.teacher_run_dir / "fixed_eskf.jsonl",
    )
    segment_counts = [step.dt_s.size for step in sequence.steps]
    print("PASS Phase 4 reference-isolated interface")
    print(f"profile_id={phase4_config.profile_id}")
    print(f"record_count={len(sequence.steps)}")
    print(f"valid_pvt_count={sum(step.pvt_solution_valid for step in sequence.steps)}")
    print(f"imu_segments_min={min(segment_counts)} max={max(segment_counts)}")
    print(f"timing_gap_step_count={sum(step.imu_timing_gap for step in sequence.steps)}")
    print("teacher_covariance_used=false")
    print("high_precision_reference_used=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
