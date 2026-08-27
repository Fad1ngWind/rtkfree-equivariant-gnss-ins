#!/usr/bin/env python3
"""Run deterministic Phase 3 baselines on frozen standardized Medium inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from rtkfree_equivariant_gnss_ins.phase3_config import load_fixed_eskf_config
from rtkfree_equivariant_gnss_ins.phase3_run import run_deployable_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SPP, INS, fixed ESKF, and 20/30-second outage baselines."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "phase3" / "fixed_eskf_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_fixed_eskf_config(args.config)
    commit = "075f96b6a6d9252b37486ecb175b4ae690c56f54"
    session = "UrbanNav-HK-Medium-Urban-1"
    pvt_path = (
        args.data_root
        / "standardized"
        / "UrbanNav"
        / commit
        / session
        / "gps_l1ca_broadcast_spp_v1"
        / "pvt.jsonl"
    )
    imu_path = (
        args.data_root
        / "standardized"
        / "UrbanNav"
        / commit
        / session
        / "imu_contract_v1"
        / "imu.csv"
    )
    extrinsic_path = (
        args.data_root
        / "deployable"
        / "UrbanNav"
        / commit
        / session
        / "calibration"
        / "gnss_imu_extrinsic.json"
    )
    imu_noise_parameter_path = (
        args.data_root
        / "deployable"
        / "UrbanNav"
        / commit
        / session
        / "calibration"
        / "xsens_imu_param.yaml"
    )
    summary_path = run_deployable_baselines(
        config,
        pvt_path,
        imu_path,
        extrinsic_path,
        imu_noise_parameter_path,
        args.output_dir,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"summary={summary_path}")
    print(f"profile_id={summary['profile_id']}")
    for name, digest in summary["output_sha256"].items():
        print(f"sha256 {name}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
