from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from rtkfree_equivariant_gnss_ins.pvt_crosscheck import (
    crosscheck_pvt_against_nmea,
    geodetic_to_ecef,
    read_nmea_gga,
    validate_pvt_records,
)


def nmea_sentence(payload: str) -> bytes:
    checksum = 0
    for byte in payload.encode("ascii"):
        checksum ^= byte
    return f"${payload}*{checksum:02X}\r\n".encode("ascii")


class PvtCrosscheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rtkfree-pvt-crosscheck-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_mixed_binary_nmea_and_standardized_contract_crosscheck(self) -> None:
        nmea_path = self.root / "mixed.nmea"
        pvt_path = self.root / "pvt.jsonl"
        start = datetime(2021, 5, 17, 2, 32, 55, tzinfo=timezone.utc)
        latitude_deg = 22.0 + 18.0 / 60.0
        longitude_deg = 114.0 + 10.0 / 60.0
        receiver = geodetic_to_ecef(
            math.radians(latitude_deg), math.radians(longitude_deg), 8.0
        )
        nmea_bytes = bytearray(b"\x00\x01binary-prefix\xff\n")
        pvt_lines: list[str] = []
        for index in range(100):
            timestamp = start + timedelta(seconds=index)
            time_field = timestamp.strftime("%H%M%S") + ".00"
            payload = (
                f"GNGGA,{time_field},2218.00000,N,11410.00000,E,1,10,0.8,"
                "10.0,M,-2.0,M,,"
            )
            nmea_bytes.extend(nmea_sentence(payload))
            nmea_bytes.extend(b"\x01\xfe")
            pvt_lines.append(
                json.dumps(
                    {
                        "timestamp_ns_utc": int(timestamp.timestamp() * 1e9) + 6_000_000,
                        "solution_valid": True,
                        "ecef_position_m": list(receiver),
                        "receiver_clock_bias_m": 100.0,
                        "position_covariance_ecef_m2": [
                            9.0,
                            0.0,
                            0.0,
                            0.0,
                            9.0,
                            0.0,
                            0.0,
                            0.0,
                            16.0,
                        ],
                        "covariance_valid": True,
                        "coordinate_frame": "WGS84_ECEF_broadcast",
                        "velocity_valid": False,
                        "ecef_velocity_mps": None,
                    }
                )
                + "\n"
            )
        pvt_lines.append(
            json.dumps(
                {
                    "timestamp_ns_utc": int((start + timedelta(seconds=100)).timestamp() * 1e9),
                    "solution_valid": False,
                    "ecef_position_m": None,
                    "receiver_clock_bias_m": None,
                    "covariance_valid": False,
                }
            )
            + "\n"
        )
        nmea_path.write_bytes(bytes(nmea_bytes))
        pvt_path.write_text("".join(pvt_lines), encoding="utf-8")

        gga = read_nmea_gga(nmea_path, date(2021, 5, 17))
        _, contract = validate_pvt_records(pvt_path)
        report = crosscheck_pvt_against_nmea(
            pvt_path, nmea_path, date(2021, 5, 17)
        )

        self.assertEqual(len(gga), 100)
        self.assertEqual(contract["contract_status"], "PASS")
        self.assertEqual(report["matched_valid_epoch_count"], 100)
        self.assertEqual(report["gross_check_status"], "PASS")
        self.assertLess(report["difference_m"]["distance_3d_max"], 1e-6)
        self.assertFalse(report["configuration_changed_from_crosscheck"])


if __name__ == "__main__":
    unittest.main()
