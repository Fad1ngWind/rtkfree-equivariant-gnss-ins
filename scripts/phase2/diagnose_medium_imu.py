from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rtkfree_equivariant_gnss_ins.imu_diagnostics import (
    align_imu_and_gnss_time,
    inspect_imu_csv,
)
from rtkfree_equivariant_gnss_ins.rinex_diagnostics import (
    inspect_navigation_gzip,
    inspect_observation,
)


def main() -> int:
    configured_root = os.environ.get("RTKFREE_DATA_ROOT")
    parser = argparse.ArgumentParser(
        description="Scan the selected IMU CSV and verify its GNSS time overlap."
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
    args = parser.parse_args()
    source = json.loads(args.source_lock.read_text(encoding="utf-8"))
    artifacts = {item["id"]: item for item in source["artifacts"]}
    session_root = (
        args.data_root
        / "deployable"
        / source["dataset"]
        / source["repository_commit"]
        / source["session"]
    )
    observation_name = next(
        name
        for name in artifacts["gnss_observation_archive"]["selected_member_basenames"]
        if name.casefold().endswith(".obs")
    )
    observation = inspect_observation(session_root / "gnss" / observation_name)
    navigation = inspect_navigation_gzip(
        session_root / "gnss" / artifacts["broadcast_navigation"]["local_name"]
    )
    imu = inspect_imu_csv(
        session_root / "imu" / artifacts["low_cost_imu"]["local_name"]
    )
    report = {
        "schema_version": 1,
        "dataset": source["dataset"],
        "session": source["session"],
        "imu": imu,
        "time_alignment": align_imu_and_gnss_time(imu, observation, navigation),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
