"""Causal forward ESKF composition and controlled GNSS masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .phase3_eskf import (
    EskfNoiseDensities,
    PositionUpdateResult,
    propagate_error_covariance,
    update_position,
)
from .phase3_ins import InsState, propagate_ins


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GnssOutage:
    """Half-open GNSS masking interval ``[start, start + duration)``."""

    start_ns_utc: int
    duration_ns: int

    def __post_init__(self) -> None:
        if self.start_ns_utc < 0 or self.duration_ns <= 0:
            raise ValueError("GNSS outage start must be nonnegative and duration positive")

    @property
    def end_ns_utc(self) -> int:
        return self.start_ns_utc + self.duration_ns

    def contains(self, timestamp_ns_utc: int) -> bool:
        return self.start_ns_utc <= timestamp_ns_utc < self.end_ns_utc


@dataclass(frozen=True)
class ForwardEskfState:
    """Nominal and covariance state at one causal replay time."""

    timestamp_ns_utc: int
    nominal: InsState
    covariance: FloatArray


@dataclass(frozen=True)
class ForwardPositionEvent:
    """Result of presenting one PVT event to the forward filter."""

    filter_state: ForwardEskfState
    update_applied: bool
    masked: bool
    diagnostics: PositionUpdateResult | None


def gnss_is_masked(timestamp_ns_utc: int, outages: Sequence[GnssOutage]) -> bool:
    return any(outage.contains(timestamp_ns_utc) for outage in outages)


def propagate_forward_eskf(
    filter_state: ForwardEskfState,
    target_timestamp_ns_utc: int,
    linear_acceleration_b_mps2: Sequence[float],
    angular_velocity_b_radps: Sequence[float],
    gravity_n_mps2: Sequence[float],
    noise: EskfNoiseDensities,
    earth_rotation_n_radps: Sequence[float] | None = None,
) -> ForwardEskfState:
    """Propagate nominal state and covariance to one strictly later IMU time."""

    interval_ns = target_timestamp_ns_utc - filter_state.timestamp_ns_utc
    if interval_ns <= 0:
        raise ValueError("forward ESKF target time must be strictly later")
    dt_s = interval_ns * 1e-9
    acceleration = np.asarray(linear_acceleration_b_mps2, dtype=np.float64)
    angular_velocity = np.asarray(angular_velocity_b_radps, dtype=np.float64)
    earth_rate_n = (
        np.zeros(3)
        if earth_rotation_n_radps is None
        else np.asarray(earth_rotation_n_radps, dtype=np.float64)
    )
    corrected_acceleration = acceleration - filter_state.nominal.accelerometer_bias_b_mps2
    bias_corrected_inertial_angular_velocity = (
        angular_velocity - filter_state.nominal.gyroscope_bias_b_radps
    )
    propagated_covariance = propagate_error_covariance(
        filter_state.covariance,
        filter_state.nominal.rotation_n_from_b,
        corrected_acceleration,
        bias_corrected_inertial_angular_velocity,
        noise,
        dt_s,
        earth_rate_n,
    )
    propagated_nominal = propagate_ins(
        filter_state.nominal,
        acceleration,
        angular_velocity,
        np.asarray(gravity_n_mps2, dtype=np.float64),
        dt_s,
        earth_rate_n,
    )
    return ForwardEskfState(
        timestamp_ns_utc=target_timestamp_ns_utc,
        nominal=propagated_nominal,
        covariance=propagated_covariance,
    )


def apply_forward_position_event(
    filter_state: ForwardEskfState,
    measured_antenna_position_n_m: Sequence[float],
    measurement_covariance_n_m2: Sequence[Sequence[float]],
    antenna_lever_imu_m: Sequence[float],
    outages: Sequence[GnssOutage] = (),
) -> ForwardPositionEvent:
    """Apply or causally mask a PVT event at the current filter time."""

    if gnss_is_masked(filter_state.timestamp_ns_utc, outages):
        return ForwardPositionEvent(
            filter_state=filter_state,
            update_applied=False,
            masked=True,
            diagnostics=None,
        )
    update = update_position(
        filter_state.nominal,
        filter_state.covariance,
        measured_antenna_position_n_m,
        measurement_covariance_n_m2,
        antenna_lever_imu_m,
    )
    return ForwardPositionEvent(
        filter_state=ForwardEskfState(
            timestamp_ns_utc=filter_state.timestamp_ns_utc,
            nominal=update.state,
            covariance=update.covariance,
        ),
        update_applied=True,
        masked=False,
        diagnostics=update,
    )
