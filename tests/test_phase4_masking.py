from __future__ import annotations

import inspect
import unittest

from rtkfree_equivariant_gnss_ins.phase3_forward import GnssOutage
from rtkfree_equivariant_gnss_ins.phase4_masking import student_gnss_available


class Phase4MaskingTests(unittest.TestCase):
    def test_twenty_and_thirty_second_masks_are_left_closed_right_open(self) -> None:
        start_ns = 1_000_000_000_000
        for duration_s in (20, 30):
            outage = GnssOutage(start_ns, duration_s * 1_000_000_000)
            self.assertTrue(student_gnss_available(start_ns - 1, True, (outage,)))
            self.assertFalse(student_gnss_available(start_ns, True, (outage,)))
            self.assertFalse(student_gnss_available(outage.end_ns_utc - 1, True, (outage,)))
            self.assertTrue(student_gnss_available(outage.end_ns_utc, True, (outage,)))

    def test_one_hertz_masks_hide_exactly_twenty_and_thirty_valid_events(self) -> None:
        start_ns = 10_000_000_000
        timestamps = range(start_ns, start_ns + 40_000_000_000, 1_000_000_000)
        for duration_s in (20, 30):
            outage = GnssOutage(start_ns, duration_s * 1_000_000_000)
            hidden_count = sum(
                not student_gnss_available(timestamp, True, (outage,))
                for timestamp in timestamps
            )
            self.assertEqual(hidden_count, duration_s)

    def test_visibility_api_does_not_expose_planned_duration_or_recovery(self) -> None:
        argument_names = set(inspect.signature(student_gnss_available).parameters)
        self.assertNotIn("duration", argument_names)
        self.assertNotIn("recovery", argument_names)
        self.assertFalse(student_gnss_available(100, False, ()))


if __name__ == "__main__":
    unittest.main()
