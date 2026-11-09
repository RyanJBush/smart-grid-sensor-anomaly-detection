"""Match alert events to ground-truth faults and compute summary metrics (SYNTHETIC data only).

Matching contract (Handoff 3): chronological, one-to-one, and an alert matches a fault only if at
least one of its contributing flags lies inside the fault's half-open interval [start, end).
A flag that merely abuts the fault through its trailing sampling slot does NOT match.
"""
from __future__ import annotations

import pandas as pd


def first_in_fault_utc(fault, alert):
    """Earliest contributing flag timestamp inside [fault.start, fault.end), or None."""
    inside = [f.timestamp for f in alert.flags if fault.start_utc <= f.timestamp < fault.end_utc_exclusive]
    return min(inside) if inside else None


def detection_latency_min(fault, alert):
    """First in-fault flag minus fault onset, in minutes (None if the alert has no in-fault flag)."""
    ts = first_in_fault_utc(fault, alert)
    return None if ts is None else (ts - fault.start_utc).total_seconds() / 60


def classify_unmatched(alert, faults: pd.DataFrame):
    """Break down an unmatched alert: ('in_fault_duplicate', fault_id) if it has a flag inside a fault
    interval (that fault was matched to another alert), else ('no_in_fault_flag', None)."""
    for f in faults.sort_values("start_utc").itertuples(index=False):
        if first_in_fault_utc(f, alert) is not None:
            return "in_fault_duplicate", f.fault_id
    return "no_in_fault_flag", None


def match_alerts(alerts, faults: pd.DataFrame, interval_min: int = 5):
    """One-to-one chronological matching.

    Faults are visited in start order; each takes the earliest not-yet-used alert (by first flag)
    that has at least one contributing flag inside the fault interval. `interval_min` is retained
    only for call compatibility and is unused. Returns ([(fault_row, alert|None)], unused_alerts);
    unused alerts are false alert events.
    """
    used: set[int] = set()
    out = []
    order = sorted(range(len(alerts)), key=lambda i: alerts[i].first_flagged_utc)
    for fault in faults.sort_values("start_utc").itertuples(index=False):
        hit = next((i for i in order if i not in used and first_in_fault_utc(fault, alerts[i]) is not None), None)
        if hit is not None:
            used.add(hit)
        out.append((fault, None if hit is None else alerts[hit]))
    unmatched = [a for i, a in enumerate(alerts) if i not in used]
    return out, unmatched


def score(alerts, faults: pd.DataFrame, readings: pd.DataFrame, interval_min: int = 5) -> dict:
    """Legacy reference-trace scoring (denominator = valid samples; no warm-up exclusion).

    The Handoff 3 contract (eligible, warmed-up slots) lives in smart_grid.evaluation.
    """
    matches, unmatched = match_alerts(alerts, faults, interval_min)
    detected = [(f, a) for f, a in matches if a is not None]
    valid = int(readings["voltage_v"].notna().sum())
    sensor_hours = valid * interval_min / 60
    false_alerts = len(unmatched)
    return {
        "alert_events": len(alerts),
        "faults_total": len(faults),
        "faults_detected": len(detected),
        "faults_missed": len(faults) - len(detected),
        "false_alert_events": false_alerts,
        "latency_min": {f.fault_id: detection_latency_min(f, a) for f, a in detected},
        "first_in_fault_utc": {f.fault_id: first_in_fault_utc(f, a) for f, a in detected},
        "alert_start_utc": {f.fault_id: a.first_flagged_utc for f, a in detected},
        "valid_samples": valid,
        "total_samples": len(readings),
        "monitored_sensor_hours": sensor_hours,
        "false_alerts_per_24_sensor_hours": false_alerts / sensor_hours * 24 if sensor_hours else float("nan"),
        "matches": {f.fault_id: (None if a is None else a.first_flagged_utc) for f, a in matches},
    }
