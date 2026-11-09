"""Chronological replay: feed readings one at a time, group consecutive flags into alert events."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .detectors import Flag


@dataclass
class AlertEvent:
    """A run of consecutive flagged scheduled readings. The first flag is the primary trigger."""

    method: str
    flags: list[Flag] = field(default_factory=list)

    @property
    def first_flagged_utc(self):
        return self.flags[0].timestamp

    @property
    def last_flagged_utc(self):
        return self.flags[-1].timestamp

    @property
    def triggering_voltage_v(self) -> float:
        return self.flags[0].voltage_v

    @property
    def breached_threshold_v(self) -> float:
        return self.flags[0].threshold_v

    @property
    def reason(self) -> str:
        return self.flags[0].reason

    @property
    def contributing_utc(self) -> list:
        return [f.timestamp for f in self.flags]

    def interval(self, interval_min: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Half-open span [first, last + one sampling interval) covered by the alert."""
        return self.first_flagged_utc, self.last_flagged_utc + pd.Timedelta(minutes=interval_min)


def replay(readings: pd.DataFrame, detector, interval_min: int = 5):
    """Return (flags, alert_events). Readings are processed in timestamp order, one at a time.

    A flagged reading extends the open event only if it is exactly one sampling interval after
    the previous contributing flag. Unflagged or missing readings close the open event.
    """
    step = pd.Timedelta(minutes=interval_min)
    flags: list[Flag] = []
    events: list[AlertEvent] = []
    open_event: AlertEvent | None = None

    for row in readings.sort_values("timestamp_utc").itertuples(index=False):
        flag = detector.update(row.timestamp_utc, row.voltage_v)
        if flag is None:
            open_event = None
            continue
        flags.append(flag)
        if open_event is not None and flag.timestamp - open_event.last_flagged_utc == step:
            open_event.flags.append(flag)
        else:
            open_event = AlertEvent(method=detector.method, flags=[flag])
            events.append(open_event)
    return flags, events
