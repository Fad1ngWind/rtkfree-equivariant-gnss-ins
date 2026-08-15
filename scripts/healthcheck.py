#!/usr/bin/env python3
"""Standard-library-only Phase 0 health check with explicit safety gates."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        "Non-normative runtime: Ubuntu-24.04 Python 3.12 is required; "
        f"got Python {platform.python_version()}"
    )

import tomllib


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC))

EXPECTED_SOURCES = [
    "reproducible_conventional_wls_spp_pvt",
    "low_cost_imu",
]
ALLOWED_PHASE0_STATUSES = {
    "awaiting_controller_review",
    "returned_for_rework",
    "ready_for_controller_review",
    "accepted",
}


class HealthCheckError(RuntimeError):
    """Raised when a frozen Phase 0 invariant is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HealthCheckError(message)


def main() -> int:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        document = tomllib.load(handle)
    project = document.get("project")
    require(isinstance(project, dict), "pyproject [project] table is required")

    with (ROOT / "config" / "phase0.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    require(isinstance(config, dict), "Phase 0 config must be a JSON object")

    import rtkfree_equivariant_gnss_ins as package
    from rtkfree_equivariant_gnss_ins.policy import validate_deployable_config

    validate_deployable_config(config)
    require(config.get("schema_version") == 1, "schema_version must equal integer 1")
    require(type(config.get("schema_version")) is int, "schema_version must be an integer")
    require(config.get("phase") == 0, "phase must equal integer 0")
    require(type(config.get("phase")) is int, "phase must be an integer")
    require(config.get("status") in ALLOWED_PHASE0_STATUSES, "status must be an allowed Phase 0 state")
    require(config.get("formal_data_enabled") is False, "formal_data_enabled must be false")
    require(config.get("training_enabled") is False, "training_enabled must be false")
    require(config.get("deployable_sources") == EXPECTED_SOURCES, "deployable_sources must match the frozen ordered list")
    require(project.get("dependencies") == [], "Phase 0 third-party dependency closure must be empty")
    require(project.get("requires-python") == ">=3.12,<3.13", "requires-python must remain >=3.12,<3.13")
    require(package.__version__ == project.get("version"), "package and project versions must match")

    print(f"PASS python={platform.python_version()} optimized={not __debug__}")
    print(f"PASS package={project['name']} version={package.__version__}")
    print(f"PASS phase0_schema=1 phase=0 status={config['status']} allowed=true")
    print("PASS formal_data_enabled=false training_enabled=false")
    print("PASS deployable_sources=frozen_exact_match")
    print("PASS third_party_runtime_dependencies=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
