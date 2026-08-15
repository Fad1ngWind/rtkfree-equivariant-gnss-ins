#!/usr/bin/env python3
"""Fail when staged or candidate repository content is unsafe to publish."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "src"))

from rtkfree_equivariant_gnss_ins.release_guard import format_findings, scan_repository


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--staged",
        action="store_true",
        help="scan the exact staged blobs instead of tracked/untracked publish candidates",
    )
    args = parser.parse_args()
    findings = scan_repository(REPO_ROOT, staged=args.staged)
    if findings:
        print(format_findings(findings), file=sys.stderr)
        return 1
    scope = "staged content" if args.staged else "tracked and unignored candidates"
    print(f"PASS public-release guard: {scope} contains no recognized high-risk files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
