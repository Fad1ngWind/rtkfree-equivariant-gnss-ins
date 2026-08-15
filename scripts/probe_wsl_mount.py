#!/usr/bin/env python3
"""Non-destructive, self-cleaning probe for the canonical DrvFS workspace."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP_PARENT = ROOT / ".phase0_tmp"


def main() -> int:
    TMP_PARENT.mkdir(exist_ok=True)
    report: dict[str, object] = {"root": ROOT.as_posix()}
    try:
        with tempfile.TemporaryDirectory(prefix="mount-probe-", dir=TMP_PARENT) as raw:
            probe = Path(raw)
            mixed = probe / "CaseProbe.txt"
            mixed.write_text("probe\n", encoding="utf-8")
            report["case_alias_exists"] = (probe / "caseprobe.txt").exists()

            before_mode = mixed.stat().st_mode & 0o777
            os.chmod(mixed, 0o600)
            after_mode = mixed.stat().st_mode & 0o777
            report["chmod_before"] = oct(before_mode)
            report["chmod_after_requested_0600"] = oct(after_mode)

            link = probe / "probe-link"
            try:
                link.symlink_to(mixed.name)
                report["symlink_created"] = link.is_symlink()
                report["symlink_resolves"] = link.resolve() == mixed.resolve()
            except OSError as exc:
                report["symlink_created"] = False
                report["symlink_error"] = type(exc).__name__

            block = b"0" * (1024 * 1024)
            target = probe / "write-probe.bin"
            started = time.perf_counter()
            with target.open("wb") as handle:
                for _ in range(8):
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            elapsed = time.perf_counter() - started
            report["sequential_write_mib"] = 8
            report["sequential_write_seconds"] = round(elapsed, 4)
            report["sequential_write_mib_per_second"] = round(8 / elapsed, 2)
    finally:
        try:
            TMP_PARENT.rmdir()
        except OSError:
            pass

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
