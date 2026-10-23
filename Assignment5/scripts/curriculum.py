"""Stage mixtures under the budget-integral constraint. Wraps ledger.py IPF."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import build, print_tables

if __name__ == "__main__":
    b = build()
    print_tables(b["M"], b["stage_tok"], b["named"], b["overlay"], b["effective"],
                 b["indic_per"], b["indic_tot"], b["loss"], b["notes"])
