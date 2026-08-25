from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from rtkfree_equivariant_gnss_ins.gps_spp import (
    _gps_seconds,
    read_gps_c1c_observations,
    read_gps_navigation,
    satellite_position_clock,
    select_ephemeris,
    solve_static_weighted_position,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config" / "phase2" / "gps_l1ca_spp_profile.json"


def header_line(content: str, label: str) -> str:
    return f"{content:<60}{label:<20}\n"


def identity_line(file_type: str, satellite_system: str) -> str:
    content = list(" " * 60)
    content[:9] = f"{3.04:9.2f}"
    content[20] = file_type
    content[40] = satellite_system
    return header_line("".join(content), "RINEX VERSION" + " / TYPE")


def nav_field(value: float) -> str:
    return f"{value:19.12E}".replace("E", "D")


class GpsSppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rtkfree-gps-spp-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_profile_is_minimal_reference_free_and_frozen(self) -> None:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(profile["constellations"], ["G"])
        self.assertEqual(profile["observation_code"], "C1C")
        self.assertEqual(profile["coordinate_frame"], "WGS84_ECEF_broadcast")
        self.assertFalse(profile["corrections"]["precise_orbit_or_clock"])
        self.assertFalse(profile["corrections"]["lever_arm"])
        self.assertFalse(profile["validity"]["residual_or_truth_tuned_rejection"])
        self.assertRegex(profile["evidence"]["rtklib_commit"], r"^[0-9a-f]{40}$")

    def test_rinex_reader_selects_only_gps_c1c_fields(self) -> None:
        observation = self.root / "fixture.obs"
        lines = [
            identity_line("O", "M"),
            header_line("G    4 C1C L1C D1C S1C", "SYS / # / OBS TYPES"),
            header_line("  2021     5    17     0     0    0.0000000     GPS", "TIME OF FIRST OBS"),
            header_line("", "END OF HEADER"),
            "> 2021 05 17 00 00 00.0000000  0  1\n",
            (
                "G 1"
                f"{20_000_001.0:14.3f}  "
                f"{123.0:14.3f}  "
                f"{-1200.0:14.3f}  "
                f"{42.0:14.3f}  \n"
            ),
        ]
        observation.write_text("".join(lines), encoding="ascii")

        header, epochs = read_gps_c1c_observations(observation)

        self.assertEqual(header["time_system"], "GPS")
        self.assertEqual(len(epochs), 1)
        self.assertEqual(epochs[0].observations[0].satellite, "G01")
        self.assertEqual(epochs[0].observations[0].pseudorange_m, 20_000_001.0)
        self.assertEqual(epochs[0].observations[0].signal_strength_dbhz, 42.0)
        self.assertEqual(epochs[0].observations[0].doppler_hz, -1200.0)

    def test_navigation_parser_and_broadcast_state_are_finite(self) -> None:
        navigation = self.root / "fixture.rnx.gz"
        clock = [2.0e-4, -4.0e-12, 0.0]
        continuation = [
            [1.0, 80.0, 4.0e-9, 0.5],
            [1.0e-6, 0.01, 2.0e-6, 5153.7955],
            [86400.0, 1.0e-7, 1.0, -1.0e-7],
            [0.94, 250.0, 0.4, -8.0e-9],
            [1.0e-10, 0.0, 2158.0, 0.0],
            [2.0, 0.0, -2.0e-8, 1.0],
            [86100.0, 4.0, 0.0, 0.0],
        ]
        with gzip.open(navigation, "wt", encoding="ascii") as handle:
            handle.write(identity_line("N", "M"))
            handle.write(
                header_line("GPSA  1.0D-08 0.0D+00 0.0D+00 0.0D+00", "IONOSPHERIC CORR")
            )
            handle.write(
                header_line("GPSB  9.0D+04 0.0D+00 0.0D+00 0.0D+00", "IONOSPHERIC CORR")
            )
            handle.write(header_line("    18", "LEAP SECONDS"))
            handle.write(header_line("", "END OF HEADER"))
            handle.write(
                "G01 2021 05 17 00 00 00"
                + "".join(nav_field(item) for item in clock)
                + "\n"
            )
            for values in continuation:
                handle.write("    " + "".join(nav_field(item) for item in values) + "\n")

        parsed = read_gps_navigation(navigation)
        receive = _gps_seconds(datetime(2021, 5, 17, 0, 10, 0))
        ephemeris = select_ephemeris(parsed, "G01", receive, 7200.0)

        self.assertIsNotNone(ephemeris)
        assert ephemeris is not None
        state = satellite_position_clock(ephemeris, receive - 0.07)
        self.assertTrue(all(math.isfinite(value) for value in state.position_ecef_m))
        self.assertGreater(math.sqrt(sum(value * value for value in state.position_ecef_m)), 2e7)
        self.assertTrue(math.isfinite(state.clock_bias_s))

    def test_weighted_position_recovers_synthetic_receiver(self) -> None:
        receiver = (1_100_000.0, -4_200_000.0, 4_000_000.0)
        clock_m = 125.0
        satellites = [
            (26_000_000.0, 0.0, 0.0),
            (0.0, 26_000_000.0, 0.0),
            (0.0, 0.0, 26_000_000.0),
            (-20_000_000.0, -15_000_000.0, 10_000_000.0),
            (15_000_000.0, -20_000_000.0, -10_000_000.0),
            (-10_000_000.0, 10_000_000.0, -22_000_000.0),
        ]
        pseudoranges = [
            math.dist(receiver, satellite) + clock_m for satellite in satellites
        ]
        state, covariance = solve_static_weighted_position(
            satellites,
            pseudoranges,
            [9.0] * len(satellites),
            (receiver[0] + 100.0, receiver[1] - 80.0, receiver[2] + 50.0),
        )

        self.assertLess(math.dist(state[:3], receiver), 1e-5)
        self.assertAlmostEqual(state[3], clock_m, places=5)
        self.assertTrue(all(covariance[index][index] > 0.0 for index in range(4)))


if __name__ == "__main__":
    unittest.main()
