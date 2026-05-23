"""Wrapper: esegui dalla root del repo.

Esempio:
  python scripts/run_at.py --config configs/at_kd.yaml

Equivalente (senza questo script):
  python -m src.training.train_at --config configs/at_kd.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train_at import main

if __name__ == "__main__":
    main()
