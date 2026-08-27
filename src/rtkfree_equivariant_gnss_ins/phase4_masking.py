"""Causal GNSS visibility for Phase 4 using the frozen Phase 3 time basis."""

from __future__ import annotations

from collections.abc import Sequence

from .phase3_forward import GnssOutage, gnss_is_masked


def student_gnss_available(
    timestamp_ns_utc: int,
    pvt_solution_valid: bool,
    outages: Sequence[GnssOutage],
) -> bool:
    """Expose current GNSS only when valid and outside every half-open outage."""

    return pvt_solution_valid and not gnss_is_masked(timestamp_ns_utc, outages)
