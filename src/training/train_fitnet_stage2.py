"""Fase 3 — FitNets Stage 2: Full KD con warm-start da Stage 1.

Riusa train_kd.py: lo student viene inizializzato dai pesi dello Stage 1
(model.student_checkpoint nel YAML), poi viene addestrato con la standard KD loss.

Esempio config: configs/fitnet_middle_s2.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.train_kd import main

if __name__ == "__main__":
    main()
