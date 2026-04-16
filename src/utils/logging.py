from __future__ import annotations

import logging
from pathlib import Path


def setup_run_logger(
    *,
    run_dir: Path,
    training_type: str,
    name: str = "train",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configura un logger che scrive sia su stdout sia su file dentro run_dir.

    Nome file: train_<training_type>.log
    Esempi: train_teacher.log, train_baseline.log, train_student.log
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    t = str(training_type).lower().strip()
    if not t:
        t = "unknown"

    log_path = run_dir / f"train_{t}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Evita handler duplicati se lo script viene richiamato più volte nello stesso processo.
    if getattr(logger, "_run_logger_configured", False):
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    setattr(logger, "_run_logger_configured", True)
    logger.info("Log file: %s", log_path.resolve())
    logger.info("Training type: %s", t)
    return logger

