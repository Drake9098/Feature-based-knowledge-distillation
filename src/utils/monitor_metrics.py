from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.utils.metrics_jsonl import safe_parse_json


@dataclass
class State:
    last_offset: int = 0
    # Latest values
    experiment: str = ""
    training_type: str = ""
    run_dir: str = ""
    config: str = ""

    epoch: Optional[int] = None
    epochs_total: Optional[int] = None
    step: Optional[int] = None
    steps_per_epoch: Optional[int] = None

    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_acc: Optional[float] = None
    lr: Optional[float] = None

    # For ETA / throughput
    step_times_s: deque[float] = deque(maxlen=50)
    last_event_ts: Optional[float] = None


def _fmt_float(x: Optional[float], fmt: str = ".4f") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return format(float(x), fmt)


def _fmt_int(x: Optional[int]) -> str:
    return "-" if x is None else str(int(x))


def _fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "-"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


def _compute_eta(state: State) -> Optional[float]:
    if state.epoch is None or state.epochs_total is None:
        return None
    if not state.step_times_s:
        return None
    steps_per_epoch = state.steps_per_epoch
    if steps_per_epoch is None:
        return None
    # Remaining steps in current epoch + full epochs ahead
    step_in_epoch = 0 if state.step is None else int(state.step)
    remaining_in_epoch = max(0, steps_per_epoch - step_in_epoch)
    remaining_epochs = max(0, int(state.epochs_total) - int(state.epoch))
    remaining_steps = remaining_in_epoch + remaining_epochs * steps_per_epoch
    avg_step = sum(state.step_times_s) / len(state.step_times_s)
    return remaining_steps * avg_step


def _build_view(state: State) -> Panel:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("k", style="bold")
    table.add_column("v")

    title = f"{state.experiment or '-'} | {state.training_type or '-'}"
    subtitle = state.run_dir or "-"

    table.add_row("run_dir", subtitle)
    if state.config:
        table.add_row("config", state.config)
    table.add_row(
        "epoch",
        f"{_fmt_int(state.epoch)}/{_fmt_int(state.epochs_total)}   step {_fmt_int(state.step)}/{_fmt_int(state.steps_per_epoch)}",
    )
    table.add_row("lr", _fmt_float(state.lr, ".6g"))
    table.add_row("train_loss", _fmt_float(state.train_loss, ".4f"))
    table.add_row("eval_loss", _fmt_float(state.eval_loss, ".4f"))
    table.add_row("eval_acc", "-" if state.eval_acc is None else f"{state.eval_acc:.2f}%")

    it_s = None
    if state.step_times_s:
        it_s = sum(state.step_times_s) / len(state.step_times_s)
    it_ms = None if it_s is None else it_s * 1000.0
    itps = None if it_s is None or it_s <= 0 else 1.0 / it_s

    eta_s = _compute_eta(state)

    perf = Text()
    perf.append("iter: ")
    perf.append("-" if it_ms is None else f"{it_ms:.1f} ms", style="bold")
    perf.append("   it/s: ")
    perf.append("-" if itps is None else f"{itps:.2f}", style="bold")
    perf.append("   ETA: ")
    perf.append(_fmt_eta(eta_s), style="bold")

    outer = Table.grid(padding=(0, 1))
    outer.add_row(Panel(table, title=title, border_style="cyan"))
    outer.add_row(Panel(perf, title="performance", border_style="magenta"))
    return Panel(outer, border_style="white")


def _apply_event(state: State, ev: dict[str, Any]) -> None:
    state.experiment = str(ev.get("experiment") or state.experiment)
    state.training_type = str(ev.get("training_type") or state.training_type)
    state.run_dir = str(ev.get("run_dir") or state.run_dir)
    cfg = ev.get("config")
    if cfg:
        state.config = str(cfg)

    # Counters
    if ev.get("epoch") is not None:
        state.epoch = int(ev["epoch"])
    if ev.get("epochs_total") is not None:
        state.epochs_total = int(ev["epochs_total"])
    if ev.get("step") is not None:
        state.step = int(ev["step"])
    if ev.get("steps_per_epoch") is not None:
        state.steps_per_epoch = int(ev["steps_per_epoch"])

    if ev.get("lr") is not None:
        state.lr = float(ev["lr"])

    kind = str(ev.get("kind") or "")
    if kind == "train":
        if ev.get("loss") is not None:
            state.train_loss = float(ev["loss"])
    elif kind == "eval":
        if ev.get("loss") is not None:
            state.eval_loss = float(ev["loss"])
        if ev.get("acc") is not None:
            state.eval_acc = float(ev["acc"])

    # perf
    step_time = ev.get("step_time_s")
    if step_time is not None:
        try:
            state.step_times_s.append(float(step_time))
        except Exception:
            pass

    ts = ev.get("ts")
    if ts is not None:
        try:
            state.last_event_ts = float(ts)
        except Exception:
            pass


def _follow_jsonl(path: Path, state: State, max_lines_per_tick: int = 200) -> int:
    """
    Legge nuove righe da path a partire da state.last_offset.
    Ritorna numero eventi applicati.
    """
    if not path.exists():
        return 0

    try:
        size = path.stat().st_size
    except OSError:
        return 0

    # File ruotato / truncato
    if state.last_offset > size:
        state.last_offset = 0

    n_applied = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(state.last_offset)
        for _ in range(max_lines_per_tick):
            line = f.readline()
            if not line:
                break
            ev = safe_parse_json(line)
            if ev is not None:
                _apply_event(state, ev)
                n_applied += 1
        state.last_offset = f.tell()
    return n_applied


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor real-time (TUI) per metrics.jsonl")
    p.add_argument(
        "run_dir",
        type=Path,
        help="Directory della run (contiene metrics.jsonl).",
    )
    p.add_argument(
        "--file",
        type=str,
        default="metrics.jsonl",
        help="Nome file JSONL (default: metrics.jsonl).",
    )
    p.add_argument(
        "--poll",
        type=float,
        default=0.5,
        help="Polling interval in secondi (default: 0.5).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    path = run_dir / args.file

    console = Console()
    state = State()

    console.print(f"[bold]Monitoring[/bold] {path} (poll={args.poll}s)")
    console.print("Premi Ctrl+C per uscire.")

    with Live(_build_view(state), console=console, refresh_per_second=10, screen=False) as live:
        while True:
            _follow_jsonl(path, state)
            live.update(_build_view(state))
            time.sleep(float(args.poll))


if __name__ == "__main__":
    main()

