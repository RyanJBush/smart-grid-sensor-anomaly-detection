"""Handoff 3 scoring contract (SYNTHETIC data only).

Both methods are scored on the SAME slots:
  eligible slot = a valid (non-missing) reading that has at least `window` valid readings before it.
Only eligible slots inside the partition [start, end) can raise flags that count, form alert
events, or contribute to the monitored-time denominator. Earlier readings (including those from
earlier partitions) may initialise rolling history. A missing or ineligible slot breaks grouping.
Matching: chronological, one-to-one, requires a contributing flag inside the half-open fault
interval; unused alert events are false alerts.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .replay import AlertEvent
from .scoring import classify_unmatched, first_in_fault_utc, match_alerts

WINDOW = 12  # rolling warm-up length; also defines slot eligibility for BOTH methods


@dataclass(frozen=True)
class Partition:
    name: str
    start: pd.Timestamp  # inclusive, UTC
    end: pd.Timestamp  # exclusive, UTC


def eligible_slots(readings: pd.DataFrame, window: int = WINDOW) -> pd.Series:
    """Boolean per row (readings sorted by time): valid and >= `window` valid readings before it.

    Depends only on the data and `window`, never on a detector or its parameters.
    """
    valid = readings["voltage_v"].notna()
    prior_valid = valid.cumsum() - valid.astype(int)
    return valid & (prior_valid >= window)


def collect_flags(readings: pd.DataFrame, detector) -> dict:
    """Feed readings chronologically, one at a time; return {timestamp: Flag} for every raw flag."""
    flags = {}
    for row in readings.sort_values("timestamp_utc").itertuples(index=False):
        flag = detector.update(row.timestamp_utc, row.voltage_v)
        if flag is not None:
            flags[row.timestamp_utc] = flag
    return flags


def group_eligible_flags(method: str, flags_by_ts: dict, slots, interval_min: int = 5) -> list[AlertEvent]:
    """Group consecutive flagged ELIGIBLE slots. `slots` is [(timestamp, eligible)] in time order."""
    step = pd.Timedelta(minutes=interval_min)
    events: list[AlertEvent] = []
    open_event = None
    for ts, ok in slots:
        flag = flags_by_ts.get(ts) if ok else None
        if flag is None:
            open_event = None
            continue
        if open_event is not None and ts - open_event.last_flagged_utc == step:
            open_event.flags.append(flag)
        else:
            open_event = AlertEvent(method=method, flags=[flag])
            events.append(open_event)
    return events


def validate_faults_within_partitions(faults: pd.DataFrame, parts: dict) -> None:
    """Every fault must lie wholly inside ONE partition. Call this before any label filtering, so a
    fault crossing a cutoff is rejected loudly instead of being silently dropped."""
    bad = [f.fault_id for f in faults.itertuples(index=False)
           if not any(p.start <= f.start_utc and f.end_utc_exclusive <= p.end for p in parts.values())]
    if bad:
        raise ValueError(f"faults not wholly inside a single partition: {bad}")


def evaluate_partition(readings: pd.DataFrame, faults: pd.DataFrame, flags_by_ts: dict, partition: Partition,
                       method: str, window: int = WINDOW, interval_min: int = 5) -> dict:
    """Score one method on one partition under the contract. `readings` must include earlier data."""
    r = readings.sort_values("timestamp_utc").reset_index(drop=True)
    eligible = eligible_slots(r, window)
    in_part = (r["timestamp_utc"] >= partition.start) & (r["timestamp_utc"] < partition.end)
    slots = list(zip(r.loc[in_part, "timestamp_utc"], eligible[in_part]))
    n_eligible = int(eligible[in_part].sum())

    inside, straddling = [], []
    for f in faults.itertuples(index=False):
        if f.start_utc >= partition.start and f.end_utc_exclusive <= partition.end:
            inside.append(f.fault_id)
        elif f.start_utc < partition.end and f.end_utc_exclusive > partition.start:
            straddling.append(f.fault_id)
    if straddling:
        raise ValueError(f"faults straddle partition {partition.name!r} boundary: {straddling}")
    part_faults = faults[faults["fault_id"].isin(inside)]

    alerts = group_eligible_flags(method, flags_by_ts, slots, interval_min)
    matches, unused = match_alerts(alerts, part_faults)
    unmatched = [{"alert": a, "kind": k, "fault_id": fid}
                 for a in unused for k, fid in [classify_unmatched(a, part_faults)]]

    elig_ts = set(ts for ts, ok in slots if ok)
    per_fault, latencies = [], []
    for f, a in matches:
        first_ts = None if a is None else first_in_fault_utc(f, a)
        lat = None if first_ts is None else (first_ts - f.start_utc).total_seconds() / 60
        if lat is not None:
            latencies.append(lat)
        per_fault.append({
            "fault_id": f.fault_id,
            "detected": a is not None,
            "alert_start_utc": None if a is None else a.first_flagged_utc,
            "first_in_fault_utc": first_ts,
            "latency_min": lat,
            "eligible_slots_in_fault": sum(1 for t in elig_ts if f.start_utc <= t < f.end_utc_exclusive),
        })

    hours = n_eligible * interval_min / 60
    return {
        "method": method,
        "partition": partition.name,
        "scheduled_slots": int(in_part.sum()),
        "valid_slots": int(r.loc[in_part, "voltage_v"].notna().sum()),
        "eligible_slots": n_eligible,
        "monitored_sensor_hours": hours,
        "alert_events": len(alerts),
        "faults_total": len(part_faults),
        "faults_detected": len(latencies),
        "faults_missed": len(part_faults) - len(latencies),
        "false_alert_events": len(unused),
        "false_alerts_per_24_sensor_hours": (len(unused) / hours * 24) if hours else None,
        "mean_latency_min": (sum(latencies) / len(latencies)) if latencies else None,
        "unmatched_in_fault_duplicates": sum(1 for u in unmatched if u["kind"] == "in_fault_duplicate"),
        "unmatched_no_in_fault_flag": sum(1 for u in unmatched if u["kind"] == "no_in_fault_flag"),
        "per_fault": per_fault,
        "alerts": alerts,
        "matches": [(f.fault_id, a) for f, a in matches],
        "unmatched": unmatched,
    }


def evaluate_method(readings: pd.DataFrame, faults: pd.DataFrame, detector, partition: Partition,
                    window: int = WINDOW, interval_min: int = 5) -> dict:
    """Convenience: replay `detector` over all readings before partition.end, then score the partition."""
    cut = readings[readings["timestamp_utc"] < partition.end]
    return evaluate_partition(cut, faults, collect_flags(cut, detector), partition, detector.method,
                              window, interval_min)
