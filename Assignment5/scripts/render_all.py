"""Run every script that the README cites. Fail loud."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/ledger.py",
    "scripts/agentic.py",
    "scripts/indic.py",
    "scripts/indic_fertility.py",
    "scripts/indic_gated.py",
    "scripts/selector_sim.py",
    "scripts/build_escrow.py",
    "scripts/data_gate.py",
]


def main():
    for s in SCRIPTS:
        print(f"\n\n########## {s} ##########\n")
        r = subprocess.run([sys.executable, str(ROOT / s)], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(r.returncode)
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=ROOT)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
