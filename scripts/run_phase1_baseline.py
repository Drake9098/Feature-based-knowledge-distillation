"""Wrapper: esegui dalla root del repo.

Esempio:
  python scripts/run_phase1_baseline.py --config configs/phase1_baseline.yaml

Equivalente (senza questo script):
  python -m src.training.train_baseline --config configs/phase1_baseline.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train_baseline import main

if __name__ == "__main__":
    main()
