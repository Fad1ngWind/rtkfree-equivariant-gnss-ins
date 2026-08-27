"""Readers for only the Phase 2 frozen PVT, IMU, and extrinsic interfaces."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from .phase3_frames import antenna_lever_in_imu


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PvtRecord:
    timestamp_ns_utc: int
    within_common_interval: bool
    solution_valid: bool
    solution_status: str
    position_ecef_m: FloatArray | None
    covariance_ecef_m2: FloatArray | None


@dataclass(frozen=True)
class ImuRecord:
    timestamp_ns_utc: int
    angular_velocity_b_radps: FloatArray
    linear_acceleration_b_mps2: FloatArray
    timing_gap: bool
    within_common_interval: bool


@dataclass(frozen=True)
class SelectedExtrinsic:
    rotation_antenna_from_imu: FloatArray
    translation_antenna_from_imu_m: FloatArray
    antenna_lever_imu_m: FloatArray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_interface_path(path: Path, basename: str) -> None:
    if path.name != basename or path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular frozen interface file: {basename}")


def _vector3(values: object, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-component vector")
    return array


def read_standardized_pvt(path: Path) -> tuple[PvtRecord, ...]:
    _require_interface_path(path, "pvt.jsonl")
    records: list[PvtRecord] = []
    previous_timestamp: int | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                timestamp = int(raw["timestamp_ns_utc"])
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    raise ValueError("PVT timestamps must be strictly increasing")
                if raw["coordinate_frame"] != "WGS84_ECEF_broadcast":
                    raise ValueError("PVT coordinate frame differs from the frozen contract")
                solution_valid = raw["solution_valid"] is True
                position: FloatArray | None = None
                covariance: FloatArray | None = None
                if solution_valid:
                    position = _vector3(raw["ecef_position_m"], "PVT ECEF position")
                    covariance = np.asarray(
                        raw["position_covariance_ecef_m2"],
                        dtype=np.float64,
                    ).reshape(3, 3)
                    if not raw["covariance_valid"] or not np.all(np.isfinite(covariance)):
                        raise ValueError("valid PVT must have a finite valid covariance")
                    if not np.allclose(covariance, covariance.T, atol=1e-8, rtol=0.0):
                        raise ValueError("PVT covariance must be symmetric")
                    covariance = 0.5 * (covariance + covariance.T)
                    if float(np.min(np.linalg.eigvalsh(covariance))) <= 0.0:
                        raise ValueError("PVT covariance must be positive definite")
                records.append(
                    PvtRecord(
                        timestamp_ns_utc=timestamp,
                        within_common_interval=raw["within_common_interval"] is True,
                        solution_valid=solution_valid,
                        solution_status=str(raw["solution_status"]),
                        position_ecef_m=position,
                        covariance_ecef_m2=covariance,
                    )
                )
                previous_timestamp = timestamp
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("cannot read frozen PVT interface") from exc
    if not records:
        raise ValueError("frozen PVT interface is empty")
    return tuple(records)


def iter_standardized_imu(path: Path) -> Iterator[ImuRecord]:
    _require_interface_path(path, "imu.csv")
    previous_timestamp: int | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                timestamp = int(raw["timestamp_ns_utc"])
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    raise ValueError("IMU timestamps must be strictly increasing")
                if raw["frame_id"] != "/imu" or raw["valid"] != "true":
                    raise ValueError("IMU record differs from the frozen valid /imu contract")
                angular_velocity = _vector3(
                    tuple(raw[f"angular_velocity_radps_{axis}"] for axis in "xyz"),
                    "IMU angular velocity",
                )
                linear_acceleration = _vector3(
                    tuple(raw[f"linear_acceleration_mps2_{axis}"] for axis in "xyz"),
                    "IMU linear acceleration",
                )
                yield ImuRecord(
                    timestamp_ns_utc=timestamp,
                    angular_velocity_b_radps=angular_velocity,
                    linear_acceleration_b_mps2=linear_acceleration,
                    timing_gap=raw["timing_gap"] == "true",
                    within_common_interval=raw["within_common_interval"] == "true",
                )
                previous_timestamp = timestamp
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError) as exc:
        raise ValueError("cannot read frozen IMU interface") from exc


def read_selected_extrinsic(path: Path, expected_sha256: str) -> SelectedExtrinsic:
    _require_interface_path(path, "gnss_imu_extrinsic.json")
    if sha256_file(path) != expected_sha256:
        raise ValueError("selected extrinsic differs from the Phase 2 frozen hash")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["selected_key"] != "ANTENNA_T_IMU" or raw["translation_unit"] != "m":
            raise ValueError("selected extrinsic metadata differs from the frozen contract")
        matrix = np.asarray(raw["matrix_4x4_row_major"], dtype=np.float64)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("cannot read selected frozen extrinsic") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("selected extrinsic must be a finite 4 by 4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=0.0, rtol=0.0):
        raise ValueError("selected extrinsic has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError("selected extrinsic rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("selected extrinsic rotation must be proper")
    translation = matrix[:3, 3]
    lever = np.asarray(
        antenna_lever_in_imu(
            tuple(tuple(float(value) for value in row) for row in rotation),
            translation,
        ),
        dtype=np.float64,
    )
    return SelectedExtrinsic(
        rotation_antenna_from_imu=rotation,
        translation_antenna_from_imu_m=translation,
        antenna_lever_imu_m=lever,
    )
