from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class MetricsWriter:
    path: Path
    experiment_name: str
    training_type: str
    run_dir: Path
    config_path: Optional[str] = None

    def write(self, event: dict[str, Any]) -> None:
        """
        Appende un evento JSONL e forza flush.

        L'evento viene arricchito con metadati standard (timestamp, exp, type, run_dir).
        """
        base: dict[str, Any] = {
            "ts": time.time(),
            "experiment": self.experiment_name,
            "training_type": self.training_type,
            "run_dir": str(self.run_dir),
        }
        if self.config_path is not None:
            base["config"] = self.config_path

        payload = {**base, **event}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered + flush esplicito per vedere aggiornamenti in real-time.
        with self.path.open("a", encoding="utf-8", buffering=1) as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()


def safe_parse_json(line: str) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

