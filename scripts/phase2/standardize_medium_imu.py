from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rtkfree_equivariant_gnss_ins.imu_standardize import standardize_medium_imu


def main() -> int:
    configured_root = os.environ.get("RTKFREE_DATA_ROOT")
    parser = argparse.ArgumentParser(
        description="Write the deterministic Phase 2 standardized IMU record."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(configured_root) if configured_root else None,
        required=configured_root is None,
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=Path("config/phase2/urbannav_medium_source_lock.json"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("config/phase2/gps_l1ca_spp_profile.json"),
    )
    args = parser.parse_args()
    output, summary_path = standardize_medium_imu(
        args.source_lock, args.profile, args.data_root
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"imu={output}")
    print(f"summary={summary_path}")
    print(f"rows={summary['row_count']}")
    print(f"valid={summary['valid_count']}")
    print(f"timing_gaps={summary['timing_gap_count']}")
    print(f"within_common_interval={summary['within_common_interval_count']}")
    print(f"sha256={summary['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
