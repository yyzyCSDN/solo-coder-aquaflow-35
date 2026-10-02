import unittest
from datetime import datetime, timedelta, timezone

from aquaflow.regulatory.rolling import Reading, average


class RollingAverageTests(unittest.TestCase):
    WINDOW = timedelta(hours=1)
    AT = datetime(2026, 10, 2, 12, 0, 0)

    def test_reading_exactly_at_left_boundary_is_excluded(self):
        rows = [Reading(self.AT - self.WINDOW, 100.0), Reading(self.AT, 4.0)]
        self.assertEqual(average(rows, self.WINDOW, self.AT), 4.0)

    def test_reading_one_second_into_window_is_included(self):
        rows = [
            Reading(self.AT - self.WINDOW, 100.0),
            Reading(self.AT - self.WINDOW + timedelta(seconds=1), 8.0),
            Reading(self.AT, 4.0),
        ]
        self.assertEqual(average(rows, self.WINDOW, self.AT), 6.0)

    def test_reading_at_observation_time_is_included(self):
        rows = [Reading(self.AT, 5.0)]
        self.assertEqual(average(rows, self.WINDOW, self.AT), 5.0)

    def test_empty_window_returns_none(self):
        rows = [
            Reading(self.AT - self.WINDOW, 100.0),
            Reading(self.AT - self.WINDOW - timedelta(seconds=1), 200.0),
            Reading(self.AT + timedelta(seconds=1), 300.0),
        ]
        self.assertIsNone(average(rows, self.WINDOW, self.AT))

    def test_no_rows_returns_none(self):
        self.assertIsNone(average([], self.WINDOW, self.AT))

    def test_timezone_aware_boundaries(self):
        tz = timezone(timedelta(hours=5, minutes=30))
        at = self.AT.replace(tzinfo=tz)
        rows = [
            Reading(at - self.WINDOW, 100.0),
            Reading(at - self.WINDOW + timedelta(seconds=1), 8.0),
            Reading(at, 4.0),
        ]
        self.assertEqual(average(rows, self.WINDOW, at), 6.0)

    def test_same_instant_across_timezones_uses_instant_not_wall_clock(self):
        at_utc = self.AT.replace(tzinfo=timezone.utc)
        # Same instant as at_utc - WINDOW expressed in another offset.
        boundary = (at_utc - self.WINDOW).astimezone(
            timezone(timedelta(hours=9))
        )
        rows = [Reading(boundary, 100.0), Reading(at_utc, 4.0)]
        self.assertEqual(average(rows, self.WINDOW, at_utc), 4.0)


if __name__ == "__main__":
    unittest.main()
