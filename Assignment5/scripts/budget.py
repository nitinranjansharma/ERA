"""Lane budgets, epoch arithmetic, composer vs proposed. Wraps ledger.py."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import main
if __name__ == "__main__":
    main()
