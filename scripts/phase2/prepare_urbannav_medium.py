from __future__ import annotations

import argparse
import os
from pathlib import Path

from rtkfree_equivariant_gnss_ins.phase2_ingest import prepare_medium


def main() -> int:
    configured_root = os.environ.get("RTKFREE_DATA_ROOT")
    parser = argparse.ArgumentParser(
        description="Reuse a verified UrbanNav GNSS ZIP and prepare only selected deployable files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(configured_root) if configured_root else None,
        required=configured_root is None,
    )
    parser.add_argument("--existing-gnss-zip", type=Path, required=True)
    parser.add_argument("--existing-direct-dir", type=Path)
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=Path("config/phase2/urbannav_medium_source_lock.json"),
    )
    args = parser.parse_args()
    manifest = prepare_medium(
        args.source_lock,
        args.data_root,
        args.existing_gnss_zip,
        args.existing_direct_dir,
    )
    print(f"provenance_manifest={manifest}")
    print("selected_outputs=5")
    print("selection_status=provisional_pending_rinex_deployable_diagnostics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
