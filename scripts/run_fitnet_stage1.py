"""Wrapper: esegui dalla root del repo.

Esempio:
  python scripts/run_fitnet_stage1.py --config configs/fitnet_middle_s1.yaml
  python scripts/run_fitnet_stage1.py --config configs/fitnet_deep_s1.yaml
  python scripts/run_fitnet_stage1.py --config configs/fitnet_full_s1.yaml

Equivalente (senza questo script):
  python -m src.training.train_fitnet_stage1 --config configs/fitnet_middle_s1.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train_fitnet_stage1 import main

if __name__ == "__main__":
    main()
