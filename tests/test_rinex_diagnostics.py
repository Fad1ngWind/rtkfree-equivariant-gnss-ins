from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from rtkfree_equivariant_gnss_ins.rinex_diagnostics import (
    inspect_navigation_gzip,
    inspect_observation,
)


def header_line(content: str, label: str) -> str:
    return f"{content:<60}{label:<20}\n"


def identity_line(file_type: str, satellite_system: str) -> str:
    content = list(" " * 60)
    content[:9] = f"{3.03:9.2f}"
    content[20] = file_type
    content[40] = satellite_system
    return header_line("".join(content), "RINEX VERSION" + " / TYPE")


def observation_record(satellite: str, pseudorange: float) -> str:
    return f"{satellite}{pseudorange:14.3f}  {123.0:14.3f}  \n"


class RinexDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rtkfree-rinex-diagnostics-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_observation_reports_types_intervals_gaps_and_pseudoranges(self) -> None:
        observation = self.root / "fixture.obs"
        lines = [
            identity_line("O", "M"),
            header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("     1.000", "INTERVAL"),
            header_line("", "END OF HEADER"),
        ]
        for second in (0, 1, 2, 5):
            lines.append(f"> 2021 05 17 00 00 {second:02d}.0000000  0  4\n")
            for prn in range(1, 5):
                lines.append(observation_record(f"G{prn:2d}", 20_000_000.0 + prn))
        observation.write_text("".join(lines), encoding="ascii")

        report = inspect_observation(observation)

        self.assertEqual(report["header"]["version"], 3.03)
        self.assertEqual(report["header"]["observation_types"], {"G": ["C1C", "L1C"]})
        self.assertEqual(report["observation_epoch_count"], 4)
        self.assertEqual(report["nominal_interval_s"], 1.0)
        self.assertEqual(report["obvious_gap_count"], 1)
        self.assertEqual(report["largest_interval_s"], 3.0)
        self.assertEqual(report["pseudorange_satellites_per_epoch"]["min"], 4)
        self.assertEqual(
            report["pseudorange_satellites_per_epoch_by_system"]["G"]["min"], 4
        )
        self.assertEqual(
            report["pseudorange_satellites_per_epoch_by_signal"]["G:C1C"]["min"],
            4,
        )
        self.assertEqual(report["epochs_with_at_least_4_pseudorange_satellites"], 4)
        self.assertEqual(
            report["epochs_with_at_least_4_pseudorange_satellites_by_signal"]["G:C1C"],
            4,
        )
        self.assertEqual(report["declared_vs_parsed_satellite_mismatch_epochs"], 0)

    def test_navigation_gzip_reports_constellations_and_message_range(self) -> None:
        navigation = self.root / "fixture.rnx.gz"
        with gzip.open(navigation, "wt", encoding="ascii") as handle:
            handle.write(identity_line("N", "M"))
            handle.write(header_line("    18", "LEAP SECONDS"))
            handle.write(header_line("", "END OF HEADER"))
            handle.write("G 1 2021 05 17 00 00 00 0.0 0.0 0.0\n")
            handle.write("   0.0 0.0 0.0 0.0\n")
            handle.write("E11 2021 05 17 23 00 00 0.0 0.0 0.0\n")
            handle.write("   0.0 0.0 0.0 0.0\n")

        report = inspect_navigation_gzip(navigation)

        self.assertEqual(report["header"]["version"], 3.03)
        self.assertEqual(report["header"]["leap_seconds"], 18)
        self.assertEqual(report["navigation_record_count"], 2)
        self.assertEqual(report["records_by_constellation"], {"E": 1, "G": 1})
        self.assertEqual(report["first_message_epoch"], "2021-05-17T00:00:00.000000")
        self.assertEqual(report["last_message_epoch"], "2021-05-17T23:00:00.000000")


if __name__ == "__main__":
    unittest.main()
