from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_training_summary(run_dir: Path, summary: dict[str, Any]) -> Path:
    """
    Scrive `training_summary.json` nella cartella della run.

    Contenuto tipico: metadati run, `per_epoch` per curve loss/acc, `best`, `final`, path checkpoint.
    Utile per script di plotting senza parsare `metrics.jsonl`.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = summary.copy()
    out.setdefault("schema_version", SCHEMA_VERSION)
    path = run_dir / "training_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path
