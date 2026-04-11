"""Wrapper: eseguire dalla root del repo.

  python scripts/run_teacher_finetune.py --config configs/teacher_finetune.yaml

Equivalente:
  python -m src.training.train_teacher_finetune --config configs/teacher_finetune.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train_teacher_finetune import main

if __name__ == "__main__":
    main()
