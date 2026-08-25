from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from rtkfree_equivariant_gnss_ins.gps_spp import _write_atomic
from rtkfree_equivariant_gnss_ins.pvt_crosscheck import crosscheck_pvt_against_nmea


def main() -> int:
    configured_root = os.environ.get("RTKFREE_DATA_ROOT")
    parser = argparse.ArgumentParser(
        description="Validate standardized PVT and cross-check receiver-native GGA."
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
    source = json.loads(args.source_lock.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    artifacts = {item["id"]: item for item in source["artifacts"]}
    session_root = (
        args.data_root
        / "deployable"
        / source["dataset"]
        / source["repository_commit"]
        / source["session"]
    )
    nmea_name = next(
        name
        for name in artifacts["gnss_observation_archive"]["selected_member_basenames"]
        if name.endswith(".nmea")
    )
    output_root = (
        args.data_root
        / "standardized"
        / source["dataset"]
        / source["repository_commit"]
        / source["session"]
        / profile["profile_id"]
    )
    report = crosscheck_pvt_against_nmea(
        output_root / "pvt.jsonl",
        session_root / "gnss" / nmea_name,
        date(2021, 5, 17),
    )
    report_path = output_root / "receiver_native_crosscheck.json"
    _write_atomic(
        report_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={report_path}")
    return 0 if (
        report["pvt_contract"]["contract_status"] == "PASS"
        and report["gross_check_status"] == "PASS"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
