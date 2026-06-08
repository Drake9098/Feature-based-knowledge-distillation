"""Genera grafici da training_summary.json / metrics.jsonl.

Esempi:
  python scripts/plot_training_metrics.py --run-dir experiments/checkpoints/phase1_baseline/2026-05-10_12-00-00
  python scripts/plot_training_metrics.py --all
  python scripts/plot_training_metrics.py --all --output-dir figures
  python scripts/plot_training_metrics.py --run-dir run1 --run-dir run2 --compare
  python scripts/plot_training_metrics.py --run-dir PATH --step-plot --show
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.plot_training_metrics import main

if __name__ == "__main__":
    main()
