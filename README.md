# Smart Grid Sensor Anomaly Detection - synthetic replay MVP

**Everything here is SYNTHETIC.** The data are invented by a seeded generator, every threshold is an arbitrary
test setting, and no result says anything about real grid equipment or operational limits. Findings are specific
to this one seeded fixture (seed 20260101, 5-minute cadence, 15 days, one sensor).

Two causal detectors are compared: a **fixed-limit** band and a **rolling median/MAD** detector. Each reading is
scored using only earlier readings.

## Setup

Python 3.13. Dependencies are pinned in `requirements-dev.txt` (pandas, pytest, matplotlib).

```bash
pip install -r requirements-dev.txt
```

Fixtures and results use LF line endings (`.gitattributes` enforces this). The frozen config records SHA-256 hashes
of the fixture files, so a checkout that rewrites line endings fails verification instead of giving silent
differences.

## Demo walkthrough

### 1. Run it

```bash
python run_heldout.py
```

This verifies the shipped freeze (config hash pin plus fixture hashes), builds both detectors from
`config/frozen_config.json` (read-only), replays the series, and scores only the held-out window
2026-01-11 to 2026-01-16 UTC (end excluded). It overwrites `results/heldout/` with identical content.
It exits with code 2 and writes nothing if the freeze does not verify.

### 2. Where to look

| What | File |
|---|---|
| Chart: voltage, gaps, fault intervals, alerts for both methods | `results/heldout/heldout_chart.png` |
| Report: headline metrics, per-fault latency, failure case | `results/heldout/heldout_report.md` |
| Every alert event with its match status | `results/heldout/heldout_alerts.csv` |
| Per-fault detection and latency | `results/heldout/heldout_per_fault.csv` |
| All numbers in one file | `results/heldout/heldout_metrics.json` |

### 3. Trace an alert to its reading and fault

Take alert `FIX-011` in `heldout_alerts.csv` (the fixed-limit alert that covers the last shift):

1. Its row gives `first_flagged_utc` 2026-01-15T15:00:00Z, `triggering_voltage_v` 124.31, `breached_threshold_v` 123.0,
   `triggering_direction` high, and `reason` "124.31 V above upper limit 123 V". `contributing_flagged_utc` lists every flagged
   slot in the event; the `contributing_*` columns hold each flag's own voltage, direction, and thresholds.
2. The triggering reading is the row with that timestamp in `data/synthetic_voltage_readings.csv`
   (`2026-01-15T15:00:00Z,SYNTHETIC-SENSOR-SEEDED-01,124.31`).
3. `matched_fault_id` is `SF16`. Its ground truth is in `data/synthetic_voltage_faults.csv`
   (`SF16,sustained_shift,2026-01-15T15:00:00Z,2026-01-15T19:00:00Z,5.2,heldout`).
   `first_in_fault_utc` and `latency_min` (0.0) show the first flag inside that interval.

Unmatched alerts have `match_status` `unmatched_in_fault_duplicate` (see `duplicate_of_fault_id`) or
`unmatched_no_in_fault_flag`. Readings and labels live in separate files; detectors never see the labels.

```python
import pandas as pd
alerts = pd.read_csv("results/heldout/heldout_alerts.csv", comment="#")
readings = pd.read_csv("data/synthetic_voltage_readings.csv", comment="#")
row = alerts[alerts.alert_id == "FIX-011"].iloc[0]
print(row.reason, "->", readings[readings.timestamp_utc == row.first_flagged_utc].voltage_v.item(), "V ->", row.matched_fault_id)
```

### 4. What happened on sustained fault SF16

SF16 is a +5.2 V shift lasting four hours (15:00 to 19:00 on 2026-01-15, 48 eligible slots).

- **Fixed limit [117, 123] V:** flagged it at 15:00 (latency 0 min). The shifted voltage (about 124 to 125.6 V) sat above 123 V the whole time.
- **Rolling median/MAD:** missed it. No in-fault slot was flagged. Just before onset the band half-width was 5.00 V
  (k 10 x floor 0.5 V); the first shifted readings (124.31 and 124.61 V) stayed under the 124.89 to 124.90 V upper threshold.
  As the 12-slot window filled with shifted values its median followed the shift (119.86 V before onset, 124.94 V by 18:55) and
  the band widened to 69.2 V total at 15:30. When the shift ended at 19:00 the detector flagged the return to normal
  (19:00 to 19:20), which counts as an unmatched alert.

This is one fault in one seeded replay. It illustrates how a window-based detector can adapt to a sustained level change
that a fixed band still sees; it does not show that either detector is better in general.

### Observed headline (held-out, frozen scoring rules)

| | Fixed limit | Rolling median/MAD |
|---|---|---|
| Faults detected / total | 6 / 6 | 5 / 6 |
| Alert events (unmatched) | 11 (5) | 9 (4) |
| Unmatched per 24 monitored sensor-hours | 1.01 | 0.81 |
| Eligible slots / monitored sensor-hours | 1423 / 118.58 | 1423 / 118.58 |

## Reference settings vs frozen settings

| | Where used | Fixed limit | Rolling median/MAD |
|---|---|---|---|
| **Reference** (Handoffs 1-2) | tiny hand-checkable unit-test traces (`data/reference_*`, `data/rolling_reference_*`) | 114 to 126 V | window 12, k 4.0, floor 0.5 V; provisional, never tuned |
| **Frozen** (Handoff 3) | the seeded 15-day fixture and the held-out replay | 117 to 123 V (centre 120.0, half-width 3.0) | window 12, k 10.0, floor 0.5 V |

The frozen settings were chosen from small grids using only the calibration and tuning partitions (2026-01-01 to
2026-01-11), under a provisional budget of at most 1.0 unmatched alert per 24 monitored sensor-hours. The held-out
partition was not used for selection and settings were not changed afterwards. The selection rule is recorded in
`config/frozen_config.json`; all 53 candidates are in `config/tuning_candidates.csv`.

## Scoring contract (shared by both methods)

- **Eligible slot:** a valid reading with at least 12 valid readings before it. Only eligible slots in the partition form
  alert groups and count toward monitored time; earlier data may warm up rolling history.
- **Alert event:** consecutive flagged eligible slots; a missing, ineligible, or unflagged slot ends it.
- **Match:** chronological, one-to-one, and the alert needs at least one flag inside the half-open fault interval [start, end).
  Latency is the first in-fault flag minus fault onset.
- **Unmatched alerts:** all alerts without a match, split into in-fault duplicates and events with no in-fault flag.
  A sustained shift split into several alerts, or an alert at the return to normal, adds unmatched events.

## Tests and CI

```bash
python verify_freeze.py     # checks the shipped freeze; writes nothing
python -m pytest -q         # 103 tests; temporary directories only
python run_heldout.py       # the demo command
```

`.github/workflows/ci.yml` runs, in order: install, `verify_freeze.py`, the full test suite, `run_heldout.py`,
`verify_freeze.py` again, and a `git diff` check that regeneration left `config/`, `data/` and `results/` unchanged (the chart
PNG is excluded from that diff because image bytes can vary with the matplotlib build).

Regenerating the fixture and tuning is guarded. `python run_tuning.py` verifies the shipped freeze and refuses to
overwrite it. To regenerate, write elsewhere and compare:

```bash
python run_tuning.py --out-dir /tmp/regen
```

`python run_reference.py` replays the small reference traces, rewrites the (identical) `data/reference_*` fixtures, and writes alert exports to `output/` (git-ignored).

## Layout

- `src/smart_grid/` - `detectors.py`, `replay.py`, `scoring.py`, `evaluation.py` (scoring contract), `synthetic_series.py`
  (seeded generator, partitions), `tuning.py`, `heldout.py` and `heldout_report.py` (replay, exports, chart), `export.py`, `fixtures.py`
- `data/` - synthetic readings (detector input) and separate fault / context-event label files
- `config/` - `frozen_config.json` (+ `frozen_config.sha256` pin), `fixture_manifest.json`, `tuning_candidates.csv`
- `results/heldout/` - published held-out artifacts
- `tests/` - 103 tests; `run_heldout.py`, `run_tuning.py`, `verify_freeze.py`, `run_reference.py`

## Limits

One seeded series, six held-out faults, no repeated seeds or confidence intervals. The 1.0 per 24 h budget, the candidate
grids, and the event sizes are arbitrary choices. Whether recovery alerts after a fault ends should be excluded from unmatched
counts is an open scoring question.
