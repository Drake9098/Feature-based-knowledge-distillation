"""Fase 1 — addestramento baseline con sola cross-entropy (nessuna distillazione)."""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from src.data.cifar100 import build_cifar100_loaders
from src.models import baseline
from src.models.baseline import build_baseline
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_yaml_config
from src.utils.logging import setup_run_logger
from src.utils.metrics_jsonl import MetricsWriter
from src.utils.seed import set_seed
from src.utils.training_summary import save_training_summary, utc_now_iso
from src.training.metrics import accuracy_percent, inference_latency_ms, model_size_mb


def _resolve_device(name: str) -> torch.device:
    n = str(name).lower().strip()
    if n == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config richiede device=cuda ma CUDA non è disponibile. "
                "Usa device: cpu o auto nel YAML."
            )
        return torch.device("cuda")
    return torch.device(n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 1 — baseline baseline (CE only)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1_baseline.yaml"),
        help="Percorso al file YAML di configurazione.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    device = _resolve_device(cfg["experiment"]["device"])
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")

    set_seed(int(cfg["experiment"]["seed"]))

    data_cfg = cfg["data"]
    train_loader, eval_loader = build_cifar100_loaders(
        root=data_cfg["root"],
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        data_config=data_cfg,
    )

    baseline = build_baseline(cfg["model"])
    baseline = baseline.to(device)

    criterion = nn.CrossEntropyLoss()
    t_cfg = cfg["training"]
    optimizer = torch.optim.SGD(
        baseline.parameters(),
        lr=float(t_cfg["learning_rate"]),
        momentum=float(t_cfg["momentum"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )

    scheduler = None
    scheduler_cfg = t_cfg.get("scheduler")
    if isinstance(scheduler_cfg, dict):
        sched_name = str(scheduler_cfg.get("name", "cosine")).lower().strip()
        if sched_name in {"cosine", "cosineannealinglr"}:
            t_max = int(scheduler_cfg.get("t_max", int(t_cfg["epochs"])))
            eta_min = float(scheduler_cfg.get("eta_min", 0.0))
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
            print(f"Scheduler: CosineAnnealingLR T_max={t_max} eta_min={eta_min}")
        elif sched_name in {"multistep", "multi_step", "multisteplr"}:
            milestones = scheduler_cfg.get("milestones", [60, 120, 160])
            gamma = float(scheduler_cfg.get("gamma", 0.2))
            scheduler = MultiStepLR(optimizer, milestones=[int(m) for m in milestones], gamma=gamma)
            print(f"Scheduler: MultiStepLR milestones={scheduler.milestones} gamma={scheduler.gamma}")
        else:
            raise ValueError(f"Scheduler non supportato: {sched_name!r}. Usa 'cosine' o 'multistep'.")

    ckpt_root = Path(cfg["checkpoint"]["dir"])
    exp_name = str(cfg["experiment"]["name"])
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_run_dir = ckpt_root / exp_name / ts
    run_dir = base_run_dir
    if run_dir.exists():
        i = 2
        while True:
            candidate = ckpt_root / exp_name / f"{ts}_{i}"
            if not candidate.exists():
                run_dir = candidate
                break
            i += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_run_logger(run_dir=run_dir, training_type="baseline", name="train.baseline")
    metrics = MetricsWriter(
        path=run_dir / "metrics.jsonl",
        experiment_name=exp_name,
        training_type="baseline",
        run_dir=run_dir,
        config_path=str(args.config),
    )
    logger.info("Run dir: %s", run_dir.resolve())
    logger.info("Metrics:  %s", (run_dir / "metrics.jsonl").resolve())

    ckpt_stem = f"{exp_name}_baseline"
    ckpt_last_path = run_dir / f"{ckpt_stem}_last.pt"
    ckpt_best_path = run_dir / f"{ckpt_stem}_best.pt"
    ckpt_final_path = run_dir / f"{ckpt_stem}_final.pt"
    best_acc = float("-inf")
    best_epoch: int | None = None
    best_train_loss: float | None = None
    best_test_loss: float | None = None
    per_epoch: list[dict[str, float | int]] = []

    run_t0 = time.perf_counter()
    started_at = utc_now_iso()

    epochs = int(t_cfg["epochs"])
    steps_per_epoch = len(train_loader)
    log_every_steps = int(t_cfg.get("log_every_steps", 50))
    global_step = 0
    metrics.write(
        {
            "kind": "meta",
            "epoch": 0,
            "epochs_total": epochs,
            "step": 0,
            "steps_per_epoch": steps_per_epoch,
            "msg": "run_start",
        }
    )
    for epoch in range(epochs):
        baseline.train()
        running_loss = 0.0
        n_train = 0
        epoch_start = time.perf_counter()
        last_step_t = time.perf_counter()
        for inputs, labels in train_loader:
            step_start = time.perf_counter()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = baseline(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_train += labels.size(0)
            global_step += 1

            if log_every_steps > 0 and (global_step % log_every_steps == 0):
                step_time_s = step_start - last_step_t
                last_step_t = step_start
                metrics.write(
                    {
                        "kind": "train",
                        "epoch": epoch + 1,
                        "epochs_total": epochs,
                        "step": (global_step % steps_per_epoch) or steps_per_epoch,
                        "steps_per_epoch": steps_per_epoch,
                        "loss": float(running_loss / max(n_train, 1)),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "step_time_s": float(step_time_s),
                    }
                )
        train_loss = running_loss / max(n_train, 1)

        baseline.eval()
        eval_loss_sum = 0.0
        n_eval = 0
        with torch.inference_mode():
            for inputs, labels in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = baseline(inputs)
                batch_loss = criterion(outputs, labels)
                eval_loss_sum += batch_loss.item() * labels.size(0)
                n_eval += labels.size(0)
        eval_loss = eval_loss_sum / max(n_eval, 1)

        acc = accuracy_percent(baseline, eval_loader, device)

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        epoch_time_s = time.perf_counter() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {eval_loss:.4f} | "
            f"test_acc: {acc:.2f}% | "
            f"lr: {lr:.6g}"
        )
        metrics.write(
            {
                "kind": "eval",
                "epoch": epoch + 1,
                "epochs_total": epochs,
                "step": steps_per_epoch,
                "steps_per_epoch": steps_per_epoch,
                "loss": float(eval_loss),
                "acc": float(acc),
                "lr": float(lr),
                "epoch_time_s": float(epoch_time_s),
            }
        )
        per_epoch.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "test_loss": float(eval_loss),
                "test_accuracy_percent": float(acc),
                "lr": float(lr),
                "epoch_time_s": float(epoch_time_s),
            }
        )

        # Checkpoint "last" a ogni epoca (progressivo).
        save_checkpoint(
            ckpt_last_path,
            {
                "model_state_dict": baseline.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                "epoch": epoch + 1,
                "test_acc": acc,
                "train_loss": train_loss,
                "test_loss": eval_loss,
                "is_best": False,
                "best_acc_so_far": best_acc,
            },
        )

        # Checkpoint "best" (sovrascritto) quando migliora la metrica.
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_train_loss = float(train_loss)
            best_test_loss = float(eval_loss)
            save_checkpoint(
                ckpt_best_path,
                {
                    "model_state_dict": baseline.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "test_acc": acc,
                    "train_loss": train_loss,
                    "test_loss": eval_loss,
                    "is_best": True,
                    "best_acc_so_far": best_acc,
                },
            )
            logger.info("[ckpt] nuovo best: acc=%.2f%% -> %s", best_acc, ckpt_best_path.resolve())

    # Eval finale usando il best model (non l'ultima epoca).
    if ckpt_best_path.is_file():
        best_state = load_checkpoint(ckpt_best_path, map_location="cpu")
        baseline.load_state_dict(best_state["model_state_dict"], strict=True)
        baseline = baseline.to(device)
        baseline.eval()

    # Ricalcola loss/acc finali sul best model
    baseline.eval()
    eval_loss_sum = 0.0
    n_eval = 0
    with torch.inference_mode():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = baseline(inputs)
            batch_loss = criterion(outputs, labels)
            eval_loss_sum += batch_loss.item() * labels.size(0)
            n_eval += labels.size(0)
    final_eval_loss = eval_loss_sum / max(n_eval, 1)
    final_acc = accuracy_percent(baseline, eval_loader, device)

    size_mib = model_size_mb(baseline)
    lat_ms = inference_latency_ms(baseline, eval_loader, device, cfg["metrics"])
    logger.info(
        f"Final(best) | test_loss: {final_eval_loss:.4f} | test_acc: {final_acc:.2f}% | "
        f"model_size: {size_mib:.2f} MiB | inference: {lat_ms:.4f} ms/image "
        f"(incl. host→device transfer)"
    )

    ckpt_path = ckpt_final_path
    save_checkpoint(
        ckpt_path,
        {
            "model_state_dict": baseline.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "epoch": epochs,
            "test_acc": final_acc,
            "test_loss": final_eval_loss,
            "model_size_mib": size_mib,
            "inference_ms_per_image": lat_ms,
            "best_checkpoint_path": str(ckpt_best_path),
        },
    )
    logger.info("Checkpoint baseline (final): %s", ckpt_path.resolve())
    logger.info("Checkpoint baseline (last):  %s", ckpt_last_path.resolve())
    logger.info("Checkpoint baseline (best):  %s", ckpt_best_path.resolve())
    metrics.write(
        {
            "kind": "meta",
            "epoch": epochs,
            "epochs_total": epochs,
            "step": steps_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "msg": "run_end",
            "final_acc": float(final_acc),
            "final_eval_loss": float(final_eval_loss),
        }
    )

    wall_s = time.perf_counter() - run_t0
    summary_path = save_training_summary(
        run_dir,
        {
            "experiment": exp_name,
            "training_type": "baseline",
            "config_path": str(args.config.resolve()),
            "run_dir": str(run_dir.resolve()),
            "device": str(device),
            "seed": int(cfg["experiment"]["seed"]),
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "wall_time_seconds": float(wall_s),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "per_epoch": per_epoch,
            "best": (
                None
                if best_epoch is None
                else {
                    "epoch": best_epoch,
                    "train_loss": best_train_loss,
                    "test_loss": best_test_loss,
                    "test_accuracy_percent": float(best_acc),
                    "checkpoint_path": str(ckpt_best_path.resolve()),
                }
            ),
            "final": {
                "test_accuracy_percent": float(final_acc),
                "test_loss": float(final_eval_loss),
                "model_size_mib": float(size_mib),
                "inference_ms_per_image": float(lat_ms),
            },
            "checkpoints": {
                "last": str(ckpt_last_path.resolve()),
                "best": str(ckpt_best_path.resolve()),
                "final": str(ckpt_final_path.resolve()),
            },
            "artifacts": {
                "metrics_jsonl": "metrics.jsonl",
                "training_summary": "training_summary.json",
            },
        },
    )
    logger.info("Training summary: %s", summary_path.resolve())


if __name__ == "__main__":
    main()
