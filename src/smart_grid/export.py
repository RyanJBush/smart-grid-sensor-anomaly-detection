"""Serialization of alert events and per-reading scores. Every export is labeled SYNTHETIC."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fixtures import SYNTHETIC_BANNER

FMT = "%Y-%m-%dT%H:%M:%SZ"


ALERT_COLUMNS = [
    "data_label", "method", "first_flagged_utc", "last_flagged_utc", "triggering_voltage_v",
    "breached_threshold_v", "triggering_direction", "lower_threshold_v", "upper_threshold_v", "reason",
    "contributing_flagged_utc", "contributing_voltages_v", "contributing_directions",
    "contributing_breached_thresholds_v", "contributing_lower_thresholds_v", "contributing_upper_thresholds_v",
]


def _num(x) -> str:
    return "" if x is None else repr(float(x))  # repr keeps full (unrounded) precision


def alerts_to_frame(alerts) -> pd.DataFrame:
    """One row per alert event. `method` distinguishes fixed_limit from rolling_median_mad."""
    rows = []
    for a in alerts:
        first = a.flags[0]
        rows.append({
            "data_label": "SYNTHETIC",
            "method": a.method,
            "first_flagged_utc": a.first_flagged_utc.strftime(FMT),
            "last_flagged_utc": a.last_flagged_utc.strftime(FMT),
            "triggering_voltage_v": first.voltage_v,
            "breached_threshold_v": first.threshold_v,
            "triggering_direction": first.direction,
            "lower_threshold_v": first.lower_v,
            "upper_threshold_v": first.upper_v,
            "reason": first.reason,
            "contributing_flagged_utc": ";".join(f.timestamp.strftime(FMT) for f in a.flags),
            "contributing_voltages_v": ";".join(_num(f.voltage_v) for f in a.flags),
            "contributing_directions": ";".join(f.direction for f in a.flags),
            "contributing_breached_thresholds_v": ";".join(_num(f.threshold_v) for f in a.flags),
            "contributing_lower_thresholds_v": ";".join(_num(f.lower_v) for f in a.flags),
            "contributing_upper_thresholds_v": ";".join(_num(f.upper_v) for f in a.flags),
        })
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


def scores_to_frame(scores, method: str) -> pd.DataFrame:
    """One row per reading (including warm-up and missing), thresholds unrounded."""
    return pd.DataFrame([
        {
            "data_label": "SYNTHETIC",
            "method": method,
            "index": i,
            "timestamp_utc": s.timestamp.strftime(FMT),
            "voltage_v": s.voltage_v,
            "status": s.status,
            "history_count": s.history_count,
            "median_v": s.median_v,
            "mad_v": s.mad_v,
            "spread_v": s.spread_v,
            "lower_threshold_v": s.lower_v,
            "upper_threshold_v": s.upper_v,
            "flagged": s.flagged,
            "direction": s.direction,
        }
        for i, s in enumerate(scores)
    ])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {SYNTHETIC_BANNER}\n")
        df.to_csv(fh, index=False, lineterminator="\n")
