"""Generate the seeded SYNTHETIC fixture, run calibration + tuning, write a frozen config.

Does NOT evaluate the held-out partition.

Safety: the repository ships a frozen config (config/frozen_config.json) that held-out results depend on.
Run with no arguments from a checkout that already contains it, this script first VERIFIES the shipped
freeze and then REFUSES to overwrite it. To regenerate, write somewhere else and compare:

    python run_tuning.py --out-dir /tmp/regen

The comparison reports whether every regenerated file is byte-identical to the shipped one.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from smart_grid import heldout  # noqa: E402
from smart_grid.export import write_csv  # noqa: E402
from smart_grid.fixtures import load_faults, load_readings  # noqa: E402
from smart_grid.synthetic_series import (  # noqa: E402
    CONTEXT_FILE, FAULTS_FILE, MANIFEST_FILE, READINGS_FILE, SEED, partitions, write_synthetic_fixture,
)
from smart_grid.tuning import file_sha256, tune, write_frozen_config  # noqa: E402

SHIPPED = {  # relative path -> what it is
    "data/" + READINGS_FILE: "readings", "data/" + FAULTS_FILE: "faults", "data/" + CONTEXT_FILE: "context events",
    "config/" + MANIFEST_FILE: "manifest", "config/frozen_config.json": "frozen config",
    "config/tuning_candidates.csv": "candidate table",
}


def regenerate(out: Path):
    data, config = out / "data", out / "config"
    write_synthetic_fixture(data, config)
    readings = load_readings(data / READINGS_FILE)
    faults = load_faults(data / FAULTS_FILE)
    body, rows = tune(readings, faults, partitions())
    frozen = write_frozen_config(
        body, {READINGS_FILE: data / READINGS_FILE, FAULTS_FILE: data / FAULTS_FILE,
               CONTEXT_FILE: data / CONTEXT_FILE, MANIFEST_FILE: config / MANIFEST_FILE},
        config / "frozen_config.json")
    write_csv(pd.DataFrame(rows), config / "tuning_candidates.csv")
    return frozen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=None, help="write the regenerated tree here (never the shipped tree)")
    args = ap.parse_args(argv)
    out = (args.out_dir or ROOT).resolve()

    if out == ROOT and (ROOT / "config" / "frozen_config.json").exists():
        try:
            sha = heldout.verify_shipped_freeze(ROOT)
        except heldout.FreezeError as exc:
            print(f"ABORT: shipped freeze is INVALID ({exc}); not touching it.", file=sys.stderr)
            return 2
        print(f"Shipped freeze verified (frozen_config.json SHA-256 {sha}).")
        print("Refusing to overwrite it. Use --out-dir DIR to regenerate elsewhere and compare.")
        return 3

    frozen = regenerate(out)
    print("*** SYNTHETIC DATA - tuning on calibration/tuning partitions only; held-out NOT evaluated ***")
    print(f"seed={SEED}  cadence=5 min  wrote tree to {out}")
    for p in partitions().values():
        print(f"  {p.name:12s} [{p.start:%Y-%m-%dT%H:%M:%SZ}, {p.end:%Y-%m-%dT%H:%M:%SZ})")
    for m in ("fixed_limit", "rolling_median_mad"):
        print(f"\n{m}: {frozen[m]}")
    if out != ROOT:
        print("\nComparison with the shipped tree:")
        same = True
        for rel, what in SHIPPED.items():
            a, b = ROOT / rel, out / rel
            ok = a.exists() and file_sha256(a) == file_sha256(b)
            same &= ok
            print(f"  {'IDENTICAL' if ok else 'DIFFERS  '}  {rel}  ({what})")
        return 0 if same else 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
