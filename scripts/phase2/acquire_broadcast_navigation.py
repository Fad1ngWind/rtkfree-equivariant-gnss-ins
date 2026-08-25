from __future__ import annotations

import argparse
import os
from pathlib import Path

from rtkfree_equivariant_gnss_ins.phase2_ingest import (
    acquire_broadcast_navigation,
)


def main() -> int:
    configured_root = os.environ.get("RTKFREE_DATA_ROOT")
    parser = argparse.ArgumentParser(
        description="Acquire the recorded daily broadcast-navigation file once."
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
    path, size, digest, manifest = acquire_broadcast_navigation(
        args.source_lock,
        args.data_root,
    )
    print(f"broadcast_navigation={path}")
    print(f"bytes={size}")
    print(f"sha256={digest}")
    print(f"provenance_manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
