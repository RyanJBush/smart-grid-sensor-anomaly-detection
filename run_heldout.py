"""Frozen held-out replay. Reads config/frozen_config.json (never modifies it) and writes results/heldout/.

Aborts with a non-zero exit code if the recorded fixture hashes do not match the files on disk.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from smart_grid import heldout  # noqa: E402
from smart_grid.heldout_report import BANNER, write_outputs  # noqa: E402


def main() -> int:
    try:
        heldout.verify_shipped_freeze(ROOT)  # config hash pin + fixture hashes, before anything runs
        res = heldout.run(ROOT)
    except heldout.FreezeError as exc:
        print(f"ABORT: freeze verification failed: {exc}", file=sys.stderr)
        return 2
    out = write_outputs(res, ROOT / "results" / "heldout")
    met = out["metrics"]
    print(f"*** {BANNER} ***")
    print(f"frozen_config sha256: {met['frozen_config_sha256']}  (fixture hashes verified)")
    print(f"held-out window: {met['heldout_window_utc']['start']} .. {met['heldout_window_utc']['end_exclusive']} (end excluded)")
    for m, v in met["methods"].items():
        print(f"\n{m}")
        for k, x in v.items():
            print(f"  {k}: {x}")
    print(f"\nfailure case: {met['failure_case']['kind']} / {met['failure_case']['method']} / {met['failure_case']['ref']}")
    for p in out["case"]["paragraphs"]:
        print("  " + p)
    print(f"\nwrote {ROOT / 'results' / 'heldout'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
