from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt

from src.utils.metrics_jsonl import safe_parse_json


TRAINING_SUMMARY_NAME = "training_summary.json"
METRICS_JSONL_NAME = "metrics.jsonl"

# Chiavi per-epoch per tipo di training (train loss, eval loss, accuracy).
_EPOCH_SPECS: dict[str, dict[str, Any]] = {
    "baseline": {
        "train_loss": ("train_loss", "Train loss"),
        "eval_loss": ("test_loss", "Test loss"),
        "accuracy": ("test_accuracy_percent", "Test accuracy (%)"),
        "extra_train": [],
    },
    "teacher": {
        "train_loss": ("train_loss", "Train loss"),
        "eval_loss": ("test_loss", "Test loss"),
        "accuracy": ("test_accuracy_percent", "Test accuracy (%)"),
        "extra_train": [],
    },
    "kd": {
        "train_loss": ("train_loss", "Train loss"),
        "eval_loss": ("test_loss", "Test loss"),
        "accuracy": ("test_accuracy_percent", "Test accuracy (%)"),
        "extra_train": [
            ("train_hard_loss", "Hard CE"),
            ("train_kd_loss", "KD loss"),
        ],
    },
    "at": {
        "train_loss": ("train_loss", "Train loss"),
        "eval_loss": ("test_loss", "Test loss"),
        "accuracy": ("test_accuracy_percent", "Test accuracy (%)"),
        "extra_train": [
            ("train_kd_loss", "KD loss"),
            ("train_at_loss", "AT loss"),
        ],
    },
    "fitnet_s1": {
        "train_loss": ("train_mse", "Train MSE"),
        "eval_loss": ("eval_mse", "Eval MSE"),
        "accuracy": ("student_acc_percent", "Student accuracy (%)"),
        "extra_train": [],
    },
}


def load_training_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Formato non valido in {path}")
    return data


def read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ev = safe_parse_json(line)
            if ev is not None:
                events.append(ev)
    return events


def _pick_loss_key(training_type: str, event: dict[str, Any]) -> Optional[str]:
    if training_type == "fitnet_s1":
        return "mse_loss" if "mse_loss" in event else None
    if "loss" in event:
        return "loss"
    return None


def per_epoch_from_jsonl(events: list[dict[str, Any]], training_type: str) -> list[dict[str, Any]]:
    """Ricostruisce curve per-epoch da metrics.jsonl (utile se manca training_summary.json)."""
    spec = _EPOCH_SPECS.get(training_type, _EPOCH_SPECS["baseline"])
    train_key = spec["train_loss"][0]
    eval_key = spec["eval_loss"][0]
    acc_key = spec["accuracy"][0]
    extra_keys = [k for k, _ in spec["extra_train"]]

    by_epoch: dict[int, dict[str, Any]] = {}

    for ev in events:
        kind = str(ev.get("kind") or "")
        epoch = ev.get("epoch")
        if epoch is None:
            continue
        ep = int(epoch)
        row = by_epoch.setdefault(ep, {"epoch": ep})

        if kind == "train":
            loss_key = _pick_loss_key(training_type, ev)
            if loss_key is not None:
                row[train_key] = float(ev[loss_key])
            for key in extra_keys:
                if key.replace("train_", "") in ev:
                    row[key] = float(ev[key.replace("train_", "")])
                elif key in ev:
                    row[key] = float(ev[key])
            if ev.get("lr") is not None:
                row["lr"] = float(ev["lr"])
        elif kind == "eval":
            loss_key = _pick_loss_key(training_type, ev)
            if loss_key is not None:
                row[eval_key] = float(ev[loss_key])
            if ev.get("acc") is not None:
                row[acc_key] = float(ev["acc"])
            if ev.get("lr") is not None:
                row["lr"] = float(ev["lr"])
            if ev.get("beta") is not None:
                row["beta"] = float(ev["beta"])

    return [by_epoch[ep] for ep in sorted(by_epoch) if ep > 0]


def load_run_data(run_dir: Path, *, prefer_jsonl: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_path = run_dir / TRAINING_SUMMARY_NAME
    metrics_path = run_dir / METRICS_JSONL_NAME

    summary: dict[str, Any] = {}
    if summary_path.is_file() and not prefer_jsonl:
        summary = load_training_summary(summary_path)
    elif metrics_path.is_file():
        events = read_metrics_jsonl(metrics_path)
        if not events:
            raise FileNotFoundError(f"Nessun evento in {metrics_path}")
        training_type = str(events[0].get("training_type") or "baseline")
        experiment = str(events[0].get("experiment") or run_dir.parent.name)
        summary = {
            "experiment": experiment,
            "training_type": training_type,
            "run_dir": str(run_dir),
            "per_epoch": per_epoch_from_jsonl(events, training_type),
            "source": "metrics.jsonl",
        }
    else:
        raise FileNotFoundError(
            f"Né {TRAINING_SUMMARY_NAME} né {METRICS_JSONL_NAME} trovati in {run_dir}"
        )

    summary.setdefault("run_dir", str(run_dir))
    summary.setdefault("experiment", run_dir.parent.name)
    summary.setdefault("training_type", "baseline")
    summary.setdefault("per_epoch", [])
    if metrics_path.is_file():
        summary["metrics_jsonl"] = str(metrics_path)
    if summary_path.is_file():
        summary["training_summary"] = str(summary_path)
    return summary


def discover_runs(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Directory non trovata: {root}")

    runs: list[Path] = []
    for summary in root.rglob(TRAINING_SUMMARY_NAME):
        runs.append(summary.parent)
    for metrics in root.rglob(METRICS_JSONL_NAME):
        if metrics.parent not in runs:
            runs.append(metrics.parent)
    return sorted(runs)


def _series(per_epoch: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in per_epoch:
        if key in row and row[key] is not None:
            epochs.append(int(row["epoch"]))
            values.append(float(row[key]))
    return epochs, values


def _lr_series_specs(per_epoch: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Restituisce le serie LR presenti nei dati (chiave, etichetta legenda)."""
    keys_present: set[str] = set()
    for row in per_epoch:
        keys_present.update(row.keys())

    specs: list[tuple[str, str]] = []
    if "lr_new" in keys_present or "lr_backbone" in keys_present:
        if "lr_new" in keys_present:
            specs.append(("lr_new", "LR new layers (conv1, fc)"))
        if "lr_backbone" in keys_present:
            specs.append(("lr_backbone", "LR backbone"))
    elif "lr" in keys_present:
        specs.append(("lr", "Learning rate"))
    return specs


def _run_label(summary: dict[str, Any]) -> str:
    exp = summary.get("experiment", "?")
    ttype = summary.get("training_type", "?")
    run_name = Path(str(summary.get("run_dir", ""))).name
    return f"{exp} ({ttype}) — {run_name}"


def _extract_temperature(experiment: str) -> Optional[int]:
    match = re.search(r"_t(\d+)$", experiment.lower())
    return int(match.group(1)) if match else None


def _fitnet_variant(experiment: str) -> Optional[str]:
    exp = experiment.lower()
    for variant in ("middle", "deep", "full"):
        if variant in exp:
            return variant.capitalize()
    return None


def _macro_category_title(summary: dict[str, Any]) -> str:
    """Titolo leggibile per la macro-categoria dell'esperimento."""
    experiment = str(summary.get("experiment") or "").lower()
    training_type = str(summary.get("training_type") or "").lower()

    if "teacher_finetune" in experiment or training_type == "teacher":
        return "Teacher Finetune"
    if experiment.startswith("phase1_baseline") or training_type == "baseline":
        return "Phase 1 Baseline"
    if experiment.startswith("at_kd") or training_type == "at":
        return "Attention Transfer"

    fitnet_variant = _fitnet_variant(experiment)
    if "fitnet" in experiment and "_s1" in experiment:
        if fitnet_variant:
            return f"FitNet {fitnet_variant} Stage 1"
        return "FitNet Stage 1"
    if "fitnet" in experiment and "_s2" in experiment:
        if fitnet_variant:
            return f"FitNet {fitnet_variant} Stage 2"
        return "FitNet Stage 2"

    if experiment.startswith("phase2_kd") or (
        training_type == "kd" and "fitnet" not in experiment
    ):
        return "Phase 2 Knowledge Distillation"
    if training_type == "fitnet_s1":
        return "FitNet Stage 1"
    if training_type == "kd":
        return "Knowledge Distillation"

    return str(summary.get("experiment") or "Training run").replace("_", " ").title()


def _plot_label(summary: dict[str, Any]) -> str:
    """Etichetta per grafici: macro-categoria, con T se presente nel nome run."""
    title = _macro_category_title(summary)
    temperature = _extract_temperature(str(summary.get("experiment") or ""))
    if temperature is not None:
        return f"{title} (T={temperature})"
    return title


def plot_single_run(
    summary: dict[str, Any],
    out_path: Path,
    *,
    show: bool = False,
    dpi: int = 150,
) -> Path:
    per_epoch = summary.get("per_epoch") or []
    if not per_epoch:
        raise ValueError(f"Nessun dato per-epoch per {_run_label(summary)}")

    training_type = str(summary.get("training_type") or "baseline")
    spec = _EPOCH_SPECS.get(training_type, _EPOCH_SPECS["baseline"])
    train_key, train_label = spec["train_loss"]
    eval_key, eval_label = spec["eval_loss"]
    acc_key, acc_label = spec["accuracy"]
    extra_train = spec["extra_train"]

    has_lr = bool(_lr_series_specs(per_epoch))
    has_extra = bool(extra_train) and any(
        key in row for row in per_epoch for key, _ in extra_train
    )
    n_rows = 2 + int(has_lr) + int(has_extra and training_type in {"kd", "at"})

    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    ax_loss = axes[0]
    ep_train, v_train = _series(per_epoch, train_key)
    ep_eval, v_eval = _series(per_epoch, eval_key)
    if ep_train:
        ax_loss.plot(ep_train, v_train, marker="o", markersize=3, label=train_label)
    if ep_eval:
        ax_loss.plot(ep_eval, v_eval, marker="s", markersize=3, label=eval_label)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best")

    ax_acc = axes[1]
    ep_acc, v_acc = _series(per_epoch, acc_key)
    if ep_acc:
        ax_acc.plot(ep_acc, v_acc, color="tab:green", marker="o", markersize=3, label=acc_label)
        best_idx = max(range(len(v_acc)), key=lambda i: v_acc[i])
        ax_acc.axvline(ep_acc[best_idx], color="tab:green", linestyle="--", alpha=0.35, linewidth=1)
        ax_acc.scatter([ep_acc[best_idx]], [v_acc[best_idx]], color="tab:green", s=40, zorder=5)
    ax_acc.set_ylabel("Accuracy (%)" if "acc" in acc_key else acc_label)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="best")

    row_idx = 2
    if has_extra and training_type in {"kd", "at"}:
        ax_comp = axes[row_idx]
        for key, label in extra_train:
            ep, vals = _series(per_epoch, key)
            if ep:
                ax_comp.plot(ep, vals, marker=".", markersize=3, label=label)
        ax_comp.set_ylabel("Component loss")
        ax_comp.grid(True, alpha=0.3)
        ax_comp.legend(loc="best")
        row_idx += 1

    if has_lr:
        ax_lr = axes[row_idx]
        for lr_key, lr_label in _lr_series_specs(per_epoch):
            ep_lr, v_lr = _series(per_epoch, lr_key)
            if ep_lr:
                ax_lr.plot(ep_lr, v_lr, marker=".", markersize=3, label=lr_label)
        ax_lr.set_ylabel("LR")
        ax_lr.set_yscale("log")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.legend(loc="best")
        row_idx += 1

    axes[-1].set_xlabel("Epoch")

    fig.suptitle(_macro_category_title(summary), fontsize=11)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_compare_runs(
    summaries: list[dict[str, Any]],
    out_path: Path,
    *,
    metric: str = "test_accuracy_percent",
    show: bool = False,
    dpi: int = 150,
) -> Path:
    if not summaries:
        raise ValueError("Nessuna run da confrontare.")

    fig, ax = plt.subplots(figsize=(10, 5))
    for summary in summaries:
        per_epoch = summary.get("per_epoch") or []
        ep, vals = _series(per_epoch, metric)
        if not ep:
            # fitnet_s1 usa student_acc_percent
            if metric == "test_accuracy_percent":
                ep, vals = _series(per_epoch, "student_acc_percent")
        if ep:
            ax.plot(ep, vals, marker="o", markersize=3, label=_plot_label(summary))

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Confronto curve di training")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_step_loss_from_jsonl(
    run_dir: Path,
    out_path: Path,
    *,
    show: bool = False,
    dpi: int = 150,
) -> Optional[Path]:
    """Loss per step (kind=train) da metrics.jsonl — opzionale, più granulare."""
    metrics_path = Path(run_dir) / METRICS_JSONL_NAME
    if not metrics_path.is_file():
        return None

    events = read_metrics_jsonl(metrics_path)
    if not events:
        return None

    training_type = str(events[0].get("training_type") or "baseline")
    steps: list[int] = []
    losses: list[float] = []
    global_step = 0
    steps_per_epoch = int(events[0].get("steps_per_epoch") or 0)

    for ev in events:
        if str(ev.get("kind") or "") != "train":
            continue
        loss_key = _pick_loss_key(training_type, ev)
        if loss_key is None:
            continue
        epoch = int(ev.get("epoch") or 0)
        step = int(ev.get("step") or 0)
        if steps_per_epoch > 0:
            global_step = (epoch - 1) * steps_per_epoch + step
        else:
            global_step += 1
        steps.append(global_step)
        losses.append(float(ev[loss_key]))

    if not steps:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, linewidth=1, alpha=0.85)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Train loss")
    ax.set_title(f"Train loss per step — {_macro_category_title({'experiment': events[0].get('experiment'), 'training_type': training_type, 'run_dir': run_dir})}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def default_output_dir(run_dir: Path, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path(run_dir) / "figures"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(
        description="Genera grafici da training_summary.json / metrics.jsonl.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory di una run (contiene training_summary.json o metrics.jsonl). Ripetibile.",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=root / "experiments" / "checkpoints",
        help="Radice per --all (default: experiments/checkpoints).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Plotta tutte le run trovate sotto --root.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Con --run-dir (2+ path): un unico grafico comparativo sull'accuracy.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="test_accuracy_percent",
        help="Metrica per --compare (default: test_accuracy_percent).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Cartella output PNG (default: <run-dir>/figures o figures/compare).",
    )
    p.add_argument(
        "--from-jsonl",
        action="store_true",
        help="Ignora training_summary.json e ricostruisce da metrics.jsonl.",
    )
    p.add_argument(
        "--step-plot",
        action="store_true",
        help="Genera anche il grafico loss-per-step da metrics.jsonl.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Mostra i grafici a schermo (oltre al salvataggio).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Risoluzione PNG (default: 150).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_dirs: list[Path] = [Path(p) for p in args.run_dir]

    if args.all:
        discovered = discover_runs(args.root)
        if not discovered:
            print(f"Nessuna run trovata sotto {args.root}")
            return
        run_dirs.extend(discovered)

    if not run_dirs:
        print("Specifica --run-dir <path> oppure --all.")
        return

    # Deduplica preservando ordine
    seen: set[Path] = set()
    unique_dirs: list[Path] = []
    for d in run_dirs:
        resolved = d.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(d)

    summaries = [load_run_data(d, prefer_jsonl=args.from_jsonl) for d in unique_dirs]

    if args.compare:
        if len(summaries) < 2:
            print("--compare richiede almeno 2 run.")
            return
        out = args.output_dir or (Path(args.root).parent / "figures" / "compare_accuracy.png")
        if out.suffix.lower() != ".png":
            out = out / "compare_accuracy.png"
        path = plot_compare_runs(
            summaries,
            out,
            metric=args.metric,
            show=args.show,
            dpi=args.dpi,
        )
        print(f"Confronto salvato: {path.resolve()}")
        return

    for run_dir, summary in zip(unique_dirs, summaries):
        out_dir = default_output_dir(run_dir, args.output_dir)
        exp = summary.get("experiment", run_dir.parent.name)
        ttype = summary.get("training_type", "run")
        curves_path = out_dir / f"{exp}_{ttype}_curves.png"
        saved = plot_single_run(summary, curves_path, show=args.show, dpi=args.dpi)
        print(f"Curve salvate: {saved.resolve()}")

        if args.step_plot:
            step_path = out_dir / f"{exp}_{ttype}_train_steps.png"
            step_saved = plot_step_loss_from_jsonl(run_dir, step_path, show=args.show, dpi=args.dpi)
            if step_saved:
                print(f"Step plot salvato: {step_saved.resolve()}")


if __name__ == "__main__":
    main()
