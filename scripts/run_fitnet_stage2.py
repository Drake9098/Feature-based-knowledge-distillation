"""Wrapper: esegui dalla root del repo.

Esempio:
  python scripts/run_fitnet_stage2.py --config configs/fitnet_middle_s2.yaml
  python scripts/run_fitnet_stage2.py --config configs/fitnet_deep_s2.yaml
  python scripts/run_fitnet_stage2.py --config configs/fitnet_full_s2.yaml

Equivalente (senza questo script):
  python -m src.training.train_fitnet_stage2 --config configs/fitnet_middle_s2.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train_fitnet_stage2 import main

if __name__ == "__main__":
    main()
