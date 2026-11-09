"""Verify the SHIPPED freeze without regenerating or writing anything. Exit 0 = valid, 2 = invalid."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from smart_grid import heldout  # noqa: E402


def main() -> int:
    try:
        sha = heldout.verify_shipped_freeze(ROOT)
    except heldout.FreezeError as exc:
        print(f"FREEZE INVALID: {exc}", file=sys.stderr)
        return 2
    print(f"FREEZE OK: config/frozen_config.json SHA-256 {sha} matches config/{heldout.SIDECAR_FILE}; "
          "fixture hashes, status and held-out boundaries verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
