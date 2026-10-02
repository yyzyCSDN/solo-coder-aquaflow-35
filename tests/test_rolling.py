import unittest
from datetime import datetime, timedelta, timezone

from aquaflow.regulatory.rolling import Reading, average
from aquaflow.workflow_service import WorkflowService

HOUR = timedelta(hours=1)


class RollingAverageTest(unittest.TestCase):
    def test_reading_exactly_on_left_boundary_is_excluded(self):
        at = datetime(2026, 10, 2, 12, 0, 0)
        boundary = at - HOUR
        rows = [Reading(boundary, 100.0), Reading(at, 10.0)]
        # The boundary reading (100) must not enter the half-open (start, at] window.
        self.assertEqual(average(rows, HOUR, at), 10.0)

    def test_reading_exactly_at_observation_time_is_included(self):
        at = datetime(2026, 10, 2, 12, 0, 0)
        rows = [Reading(at, 42.0)]
        self.assertEqual(average(rows, HOUR, at), 42.0)

    def test_reading_one_second_inside_window_is_included(self):
        at = datetime(2026, 10, 2, 12, 0, 0)
        start = at - HOUR
        rows = [Reading(start + timedelta(seconds=1), 8.0), Reading(at, 12.0)]
        self.assertEqual(average(rows, HOUR, at), 10.0)

    def test_reading_one_second_outside_window_is_excluded(self):
        at = datetime(2026, 10, 2, 12, 0, 0)
        start = at - HOUR
        rows = [
            Reading(start - timedelta(seconds=1), 100.0),
            Reading(start, 100.0),
            Reading(at, 4.0),
        ]
        self.assertEqual(average(rows, HOUR, at), 4.0)

    def test_boundary_reading_is_not_double_counted_across_adjacent_windows(self):
        # A reading at 11:00 is the right edge of the [10:00,11:00] regulatory
        # window and must not also be the left edge of (11:00,12:00].
        rows = [
            Reading(datetime(2026, 10, 2, 11, 0, 0), 100.0),
            Reading(datetime(2026, 10, 2, 12, 0, 0), 20.0),
        ]
        self.assertEqual(average(rows, HOUR, datetime(2026, 10, 2, 11, 0, 0)), 100.0)
        self.assertEqual(average(rows, HOUR, datetime(2026, 10, 2, 12, 0, 0)), 20.0)

    def test_observation_time_with_no_valid_readings_returns_none(self):
        at = datetime(2026, 10, 2, 12, 0, 0)
        start = at - HOUR
        rows = [
            Reading(start, 100.0),          # exactly on excluded left boundary
            Reading(start - timedelta(seconds=1), 100.0),
            Reading(at + timedelta(seconds=1), 100.0),
        ]
        self.assertIsNone(average(rows, HOUR, at))
        self.assertIsNone(average([], HOUR, at))

    def test_timezone_aware_boundaries_are_respected(self):
        at = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
        start = at - HOUR
        rows = [
            Reading(start, 100.0),                          # excluded boundary
            Reading(start + timedelta(seconds=1), 8.0),    # one second inside
            Reading(at, 12.0),                              # observation instant
        ]
        self.assertEqual(average(rows, HOUR, at), 10.0)

    def test_mixed_equivalent_timezones_share_one_boundary(self):
        # +09:00 noon is 03:00 UTC; the 03:00 UTC reading sits exactly on the
        # left boundary expressed in the other timezone and must be excluded.
        at = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone(timedelta(hours=9)))
        start_utc = (at - HOUR).astimezone(timezone.utc)
        rows = [Reading(start_utc, 100.0), Reading(at, 30.0)]
        self.assertEqual(average(rows, HOUR, at), 30.0)


class RegulatoryWindowWorkflowTest(unittest.TestCase):
    def _payload(self, at, readings):
        return {
            "readings": [{"at": r.at.isoformat(), "value": r.value} for r in readings],
            "window_hours": 1,
            "at": at.isoformat(),
        }

    def test_workflow_uses_half_open_window(self):
        at = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
        start = at - HOUR
        readings = [
            Reading(start, 100.0),                       # excluded boundary
            Reading(start + timedelta(seconds=1), 9.0), # one second inside
            Reading(at, 11.0),                           # observation instant
        ]
        result = WorkflowService().run("regulatory_window", self._payload(at, readings))
        self.assertEqual(result["decision"], "report")
        self.assertEqual(result["value"], 10.0)

    def test_workflow_window_is_not_inflated_beyond_left_boundary(self):
        # Regression: the workflow used to add an extra hour to the window.
        # A reading one second before the true left boundary must stay out.
        at = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
        start = at - HOUR
        readings = [
            Reading(start - timedelta(seconds=1), 100.0),
            Reading(start, 100.0),
            Reading(at, 7.0),
        ]
        result = WorkflowService().run("regulatory_window", self._payload(at, readings))
        self.assertEqual(result["value"], 7.0)

    def test_workflow_defers_when_no_valid_readings(self):
        at = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
        start = at - HOUR
        readings = [Reading(start, 100.0), Reading(at + HOUR, 100.0)]
        result = WorkflowService().run("regulatory_window", self._payload(at, readings))
        self.assertIsNone(result["value"])
        self.assertEqual(result["decision"], "defer")


if __name__ == "__main__":
    unittest.main()
