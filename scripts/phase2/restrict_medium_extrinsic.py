from __future__ import annotations

import argparse
import os
from pathlib import Path

from rtkfree_equivariant_gnss_ins.phase2_ingest import (
    refresh_selected_medium_extrinsic,
    restrict_existing_medium_extrinsic,
)


def main() -> int:
    configured_root = os.environ.get("RTKFREE_DATA_ROOT")
    parser = argparse.ArgumentParser(
        description="Keep only the selected GNSS-IMU key in the deployable calibration."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(configured_root) if configured_root else None,
        required=configured_root is None,
    )
    parser.add_argument("--preserved-staging-source", type=Path, required=True)
    parser.add_argument(
        "--refresh-selected",
        action="store_true",
        help="Refresh an already selected record after an evidence interpretation update.",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=Path("config/phase2/urbannav_medium_source_lock.json"),
    )
    args = parser.parse_args()
    action = (
        refresh_selected_medium_extrinsic
        if args.refresh_selected
        else restrict_existing_medium_extrinsic
    )
    selected, size, digest, manifest = action(
        args.source_lock, args.data_root, args.preserved_staging_source
    )
    print(f"selected_extrinsic={selected}")
    print(f"bytes={size}")
    print(f"sha256={digest}")
    print(f"provenance_manifest={manifest}")
    print("mixed_deployable_copy=ABSENT")
    print("preserved_staging_source=UNCHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
