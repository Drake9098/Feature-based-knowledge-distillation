"""Wrapper: esegui dalla root del repo.

Esempio:
  python scripts/run_phase2_kd.py --config configs/phase2_kd.yaml

Equivalente (senza questo script):
  python -m src.training.train_kd --config configs/phase2_kd.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train_kd import main

if __name__ == "__main__":
    main()
