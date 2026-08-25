from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rtkfree_equivariant_gnss_ins.gps_spp import run_medium_spp


def main() -> int:
    configured_root = os.environ.get("RTKFREE_DATA_ROOT")
    parser = argparse.ArgumentParser(
        description="Generate the frozen GPS L1 C/A conventional SPP record."
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
    pvt_path, summary_path = run_medium_spp(
        args.source_lock, args.profile, args.data_root
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"pvt={pvt_path}")
    print(f"summary={summary_path}")
    print(f"epochs={summary['epoch_count']}")
    print(f"valid={summary['valid_solution_count']}")
    print(f"invalid={summary['invalid_solution_count']}")
    print(f"pvt_sha256={summary['pvt_sha256']}")
    print(f"status_counts={json.dumps(summary['status_counts'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
