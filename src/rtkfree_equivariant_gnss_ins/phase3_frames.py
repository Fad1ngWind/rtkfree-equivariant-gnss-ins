"""Minimal Phase 3 frame and lever-arm operations."""

from __future__ import annotations

from typing import Sequence


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(
        tuple(matrix[row][column] for row in range(3))
        for column in range(3)
    )  # type: ignore[return-value]


def _vector3(values: Sequence[float]) -> Vector3:
    if len(values) != 3:
        raise ValueError("a three-component vector is required")
    return (float(values[0]), float(values[1]), float(values[2]))


def antenna_lever_in_imu(
    rotation_antenna_from_imu: Matrix3,
    translation_antenna_from_imu_m: Sequence[float],
) -> Vector3:
    """Return the IMU-origin-to-antenna-origin lever expressed in IMU axes.

    The selected source matrix is interpreted as
    ``p_antenna = R_antenna_from_imu @ p_imu + translation``.  Therefore its
    inverse translation is the antenna origin expressed in the IMU frame.
    """

    translation = _vector3(translation_antenna_from_imu_m)
    inverse_translation = _matvec(_transpose(rotation_antenna_from_imu), translation)
    return tuple(-value for value in inverse_translation)  # type: ignore[return-value]


def antenna_position_from_imu(
    imu_position_navigation_m: Sequence[float],
    rotation_navigation_from_imu: Matrix3,
    antenna_lever_imu_m: Sequence[float],
) -> Vector3:
    """Compose IMU position and body-frame lever arm in a navigation frame."""

    imu_position = _vector3(imu_position_navigation_m)
    lever_navigation = _matvec(
        rotation_navigation_from_imu,
        _vector3(antenna_lever_imu_m),
    )
    return tuple(
        imu_position[index] + lever_navigation[index] for index in range(3)
    )  # type: ignore[return-value]


def imu_position_from_antenna(
    antenna_position_navigation_m: Sequence[float],
    rotation_navigation_from_imu: Matrix3,
    antenna_lever_imu_m: Sequence[float],
) -> Vector3:
    """Recover IMU position from an antenna measurement in a navigation frame."""

    antenna_position = _vector3(antenna_position_navigation_m)
    lever_navigation = _matvec(
        rotation_navigation_from_imu,
        _vector3(antenna_lever_imu_m),
    )
    return tuple(
        antenna_position[index] - lever_navigation[index] for index in range(3)
    )  # type: ignore[return-value]
