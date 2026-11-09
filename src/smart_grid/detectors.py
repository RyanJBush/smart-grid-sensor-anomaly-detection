"""Causal detectors. update() sees one reading at a time, so it cannot use future data.

All parameters are provisional SYNTHETIC-test settings, not operational grid thresholds.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass


def _is_missing(voltage_v) -> bool:
    return voltage_v is None or math.isnan(voltage_v)


@dataclass(frozen=True)
class Flag:
    """One flagged reading, carrying its own thresholds and direction."""

    timestamp: object  # pandas Timestamp
    voltage_v: float
    threshold_v: float  # the threshold that was breached
    reason: str
    direction: str  # "high" or "low"
    lower_v: float  # lower threshold in force for this reading
    upper_v: float  # upper threshold in force for this reading


@dataclass(frozen=True)
class Score:
    """Per-reading rolling-detector record. Fields not computed for a reading are None."""

    timestamp: object
    voltage_v: float | None
    status: str  # "missing" | "warmup" | "scored"
    history_count: int  # valid prior readings held when this reading arrived
    median_v: float | None
    mad_v: float | None
    spread_v: float | None
    lower_v: float | None
    upper_v: float | None
    flagged: bool
    direction: str | None


class FixedLimitDetector:
    """Flags readings strictly outside [lower_v, upper_v]. Stateless."""

    method = "fixed_limit"

    def __init__(self, lower_v: float = 114.0, upper_v: float = 126.0):
        self.lower_v = lower_v
        self.upper_v = upper_v

    def update(self, timestamp, voltage_v: float) -> Flag | None:
        if _is_missing(voltage_v):
            return None  # missing readings never alert
        if voltage_v > self.upper_v:
            return Flag(timestamp, voltage_v, self.upper_v,
                        f"{voltage_v:g} V above upper limit {self.upper_v:g} V",
                        "high", self.lower_v, self.upper_v)
        if voltage_v < self.lower_v:
            return Flag(timestamp, voltage_v, self.lower_v,
                        f"{voltage_v:g} V below lower limit {self.lower_v:g} V",
                        "low", self.lower_v, self.upper_v)
        return None


class RollingMADDetector:
    """Causal rolling median/MAD detector.

    For each valid reading, thresholds come from the `window` most recent valid readings that
    arrived BEFORE it:  median +/- k * max(mad_scale * MAD, floor_v).  The reading is flagged only
    if strictly outside. Only after scoring is it appended to the history (flagged or not).
    Missing readings are not scored and never enter history; a gap does not reset or age history.
    Until `window` prior valid readings exist (warm-up), readings update history but cannot alert.

    k=4.0 and floor_v=0.5 are PROVISIONAL and not tuned.
    """

    method = "rolling_median_mad"

    def __init__(self, window: int = 12, k: float = 4.0, floor_v: float = 0.5, mad_scale: float = 1.4826):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window, self.k, self.floor_v, self.mad_scale = window, k, floor_v, mad_scale
        self._history: deque = deque(maxlen=window)
        self.scores: list[Score] = []

    @property
    def history(self) -> tuple:
        return tuple(self._history)

    def update(self, timestamp, voltage_v: float) -> Flag | None:
        n = len(self._history)
        if _is_missing(voltage_v):
            self.scores.append(Score(timestamp, None, "missing", n, None, None, None, None, None, False, None))
            return None
        if n < self.window:
            self.scores.append(Score(timestamp, voltage_v, "warmup", n, None, None, None, None, None, False, None))
            self._history.append(voltage_v)
            return None

        # Thresholds use history only; the current reading is not yet in it.
        hist = list(self._history)
        median = statistics.median(hist)
        mad = statistics.median(abs(x - median) for x in hist)
        spread = max(self.mad_scale * mad, self.floor_v)
        lower = median - self.k * spread
        upper = median + self.k * spread

        flag, direction = None, None
        if voltage_v > upper:
            direction = "high"
            flag = Flag(timestamp, voltage_v, upper,
                        f"{voltage_v:g} V above rolling upper threshold {upper:g} V "
                        f"(median {median:g} V, k={self.k:g})",
                        direction, lower, upper)
        elif voltage_v < lower:
            direction = "low"
            flag = Flag(timestamp, voltage_v, lower,
                        f"{voltage_v:g} V below rolling lower threshold {lower:g} V "
                        f"(median {median:g} V, k={self.k:g})",
                        direction, lower, upper)

        self.scores.append(Score(timestamp, voltage_v, "scored", n, median, mad, spread, lower, upper,
                                 flag is not None, direction))
        self._history.append(voltage_v)  # appended only after scoring, flagged or not
        return flag
