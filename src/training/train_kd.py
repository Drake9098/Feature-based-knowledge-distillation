"""Fase 2 — Knowledge Distillation standard (Hinton) tra teacher ResNet-50 e student ResNet-18.

Il teacher è caricato da un checkpoint fine-tuned su CIFAR-100 e tenuto frozen.
Lo student viene addestrato con KDLoss = α·CE(hard) + (1-α)·T²·KL(soft).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from src.data.cifar100 import build_cifar100_loaders
from src.models.baseline import build_baseline
from src.models.teacher import build_teacher
from src.training.loss import KDLoss
from src.training.metrics import accuracy_percent, inference_latency_ms, model_size_mb
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_yaml_config
from src.utils.logging import setup_run_logger
from src.utils.metrics_jsonl import MetricsWriter
from src.utils.seed import set_seed
from src.utils.training_summary import save_training_summary, utc_now_iso


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
    p = argparse.ArgumentParser(description="Fase 2 — Knowledge Distillation standard (Hinton)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_kd.yaml"),
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

    # --- Teacher (frozen) ---
    # build_teacher con pretrained_teacher: false costruisce ResNet-50 CIFAR con pesi random;
    # poi carichiamo il checkpoint fine-tuned su CIFAR-100 con strict=True per avere tutti i layer.
    teacher = build_teacher(cfg["model"])
    t_ckpt_path_raw = cfg["model"].get("teacher_checkpoint")
    if not t_ckpt_path_raw:
        raise ValueError(
            "model.teacher_checkpoint non impostato nel YAML.\n"
            "Esegui prima train_teacher_finetune.py e specifica il percorso del _best.pt."
        )
    t_ckpt_path = Path(t_ckpt_path_raw)
    if not t_ckpt_path.is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint non trovato: {t_ckpt_path}\n"
            "Esegui prima train_teacher_finetune.py e aggiorna model.teacher_checkpoint nel YAML."
        )
    ckpt = load_checkpoint(t_ckpt_path, map_location="cpu")
    teacher.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Teacher: caricato da {t_ckpt_path}")

    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    # --- Student ---
    student = build_baseline(cfg["model"])
    s_ckpt_path_raw = cfg["model"].get("student_checkpoint")
    if s_ckpt_path_raw:
        s_ckpt_path = Path(s_ckpt_path_raw)
        if not s_ckpt_path.is_file():
            raise FileNotFoundError(
                f"Student checkpoint non trovato: {s_ckpt_path}\n"
                "Esegui prima train_fitnet_stage1.py e aggiorna model.student_checkpoint nel YAML."
            )
        s_ckpt = load_checkpoint(s_ckpt_path, map_location="cpu")
        student.load_state_dict(s_ckpt["model_state_dict"], strict=True)
        print(f"Student: warm-start da {s_ckpt_path}")
    student = student.to(device)

    # --- Distillation hyperparams ---
    kd_cfg = cfg["distillation"]
    temperature = float(kd_cfg["temperature"])
    alpha = float(kd_cfg["alpha"])
    kd_criterion = KDLoss(temperature=temperature, alpha=alpha)
    print(f"KD: temperature={temperature}, alpha={alpha} (CE={alpha:.2f}, KL={1 - alpha:.2f})")

    # --- Optimizer & scheduler ---
    t_cfg = cfg["training"]
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=float(t_cfg["learning_rate"]),
        momentum=float(t_cfg["momentum"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )

    scheduler = None
    scheduler_cfg = t_cfg.get("scheduler")
    if isinstance(scheduler_cfg, dict):
        name = str(scheduler_cfg.get("name", "multistep")).lower().strip()
        if name in {"multistep", "multi_step", "multisteplr"}:
            milestones = scheduler_cfg.get("milestones")
            if milestones is None:
                milestones = [60, 120, 160]
            gamma = float(scheduler_cfg.get("gamma", 0.2))
            scheduler = MultiStepLR(
                optimizer, milestones=[int(m) for m in milestones], gamma=gamma
            )
        else:
            raise ValueError(
                f"Scheduler non supportato: {name!r}. "
                "Usa name: multistep oppure rimuovi training.scheduler."
            )
    if scheduler is not None:
        print(f"Scheduler: MultiStepLR milestones={scheduler.milestones} gamma={scheduler.gamma}")

    # --- Run dir & logging ---
    ckpt_root = Path(cfg["checkpoint"]["dir"])
    exp_name = str(cfg["experiment"]["name"])
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ckpt_root / exp_name / ts
    if run_dir.exists():
        i = 2
        while True:
            candidate = ckpt_root / exp_name / f"{ts}_{i}"
            if not candidate.exists():
                run_dir = candidate
                break
            i += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_run_logger(run_dir=run_dir, training_type="kd", name="train.kd")
    metrics = MetricsWriter(
        path=run_dir / "metrics.jsonl",
        experiment_name=exp_name,
        training_type="kd",
        run_dir=run_dir,
        config_path=str(args.config),
    )
    logger.info("Run dir: %s", run_dir.resolve())
    logger.info("Metrics:  %s", (run_dir / "metrics.jsonl").resolve())
    logger.info("Teacher ckpt: %s", t_ckpt_path.resolve())
    logger.info("KD: temperature=%.1f  alpha=%.3f  (CE=%.2f  KL=%.2f)", temperature, alpha, alpha, 1 - alpha)

    # --- Checkpoint paths ---
    ckpt_stem = f"{exp_name}_kd"
    ckpt_last_path = run_dir / f"{ckpt_stem}_last.pt"
    ckpt_best_path = run_dir / f"{ckpt_stem}_best.pt"
    ckpt_final_path = run_dir / f"{ckpt_stem}_final.pt"

    best_acc = float("-inf")
    best_epoch: int | None = None
    best_train_loss: float | None = None
    best_test_loss: float | None = None
    per_epoch: list[dict] = []

    run_t0 = time.perf_counter()
    started_at = utc_now_iso()

    epochs = int(t_cfg["epochs"])
    steps_per_epoch = len(train_loader)
    log_every_steps = int(t_cfg.get("log_every_steps", 50))
    global_step = 0

    metrics.write({
        "kind": "meta",
        "epoch": 0,
        "epochs_total": epochs,
        "step": 0,
        "steps_per_epoch": steps_per_epoch,
        "msg": "run_start",
        "temperature": temperature,
        "alpha": alpha,
    })

    # --- Training loop ---
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        running_hard = 0.0
        running_kd = 0.0
        n_train = 0
        epoch_start = time.perf_counter()
        last_step_t = time.perf_counter()

        for inputs, labels in train_loader:
            step_start = time.perf_counter()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            optimizer.zero_grad(set_to_none=True)
            student_logits = student(inputs)
            loss = kd_criterion(student_logits, teacher_logits, labels)

            # Calcolo componenti per il logging (senza costruire nuovo grafo)
            with torch.no_grad():
                hard_l = F.cross_entropy(student_logits, labels).item()
                soft_s = F.log_softmax(student_logits / temperature, dim=1)
                soft_t = F.softmax(teacher_logits / temperature, dim=1)
                kd_l = float(temperature ** 2) * F.kl_div(soft_s, soft_t, reduction="batchmean").item()

            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_hard += hard_l * bs
            running_kd += kd_l * bs
            n_train += bs
            global_step += 1

            if log_every_steps > 0 and (global_step % log_every_steps == 0):
                step_time_s = step_start - last_step_t
                last_step_t = step_start
                metrics.write({
                    "kind": "train",
                    "epoch": epoch + 1,
                    "epochs_total": epochs,
                    "step": (global_step % steps_per_epoch) or steps_per_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "loss": float(running_loss / max(n_train, 1)),
                    "hard_loss": float(running_hard / max(n_train, 1)),
                    "kd_loss": float(running_kd / max(n_train, 1)),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "step_time_s": float(step_time_s),
                })

        train_loss = running_loss / max(n_train, 1)
        train_hard = running_hard / max(n_train, 1)
        train_kd = running_kd / max(n_train, 1)

        # --- Eval: solo student con CE standard ---
        student.eval()
        eval_loss_sum = 0.0
        n_eval = 0
        with torch.inference_mode():
            for inputs, labels in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = student(inputs)
                eval_loss_sum += F.cross_entropy(logits, labels).item() * labels.size(0)
                n_eval += labels.size(0)
        eval_loss = eval_loss_sum / max(n_eval, 1)

        acc = accuracy_percent(student, eval_loader, device)

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        epoch_time_s = time.perf_counter() - epoch_start

        logger.info(
            "Epoch %d/%d | train_loss: %.4f (hard=%.4f kd=%.4f) | "
            "test_loss: %.4f | test_acc: %.2f%% | lr: %.6g",
            epoch + 1, epochs,
            train_loss, train_hard, train_kd,
            eval_loss, acc, lr,
        )
        metrics.write({
            "kind": "eval",
            "epoch": epoch + 1,
            "epochs_total": epochs,
            "step": steps_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "loss": float(eval_loss),
            "acc": float(acc),
            "lr": float(lr),
            "epoch_time_s": float(epoch_time_s),
        })
        per_epoch.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_hard_loss": float(train_hard),
            "train_kd_loss": float(train_kd),
            "test_loss": float(eval_loss),
            "test_accuracy_percent": float(acc),
            "lr": float(lr),
            "epoch_time_s": float(epoch_time_s),
        })

        # Checkpoint "last" a ogni epoca
        save_checkpoint(ckpt_last_path, {
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch + 1,
            "test_acc": acc,
            "train_loss": train_loss,
            "test_loss": eval_loss,
            "is_best": False,
            "best_acc_so_far": best_acc,
        })

        # Checkpoint "best" quando migliora
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_train_loss = float(train_loss)
            best_test_loss = float(eval_loss)
            save_checkpoint(ckpt_best_path, {
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                "epoch": epoch + 1,
                "test_acc": acc,
                "train_loss": train_loss,
                "test_loss": eval_loss,
                "is_best": True,
                "best_acc_so_far": best_acc,
            })
            logger.info("[ckpt] nuovo best: acc=%.2f%% -> %s", best_acc, ckpt_best_path.resolve())

    # --- Eval finale sul best model ---
    if ckpt_best_path.is_file():
        best_state = load_checkpoint(ckpt_best_path, map_location="cpu")
        student.load_state_dict(best_state["model_state_dict"], strict=True)
        student = student.to(device)
    student.eval()

    eval_loss_sum = 0.0
    n_eval = 0
    with torch.inference_mode():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = student(inputs)
            eval_loss_sum += F.cross_entropy(logits, labels).item() * labels.size(0)
            n_eval += labels.size(0)
    final_eval_loss = eval_loss_sum / max(n_eval, 1)
    final_acc = accuracy_percent(student, eval_loader, device)

    size_mib = model_size_mb(student)
    lat_ms = inference_latency_ms(student, eval_loader, device, cfg["metrics"])
    logger.info(
        "Final(best) | test_loss: %.4f | test_acc: %.2f%% | "
        "model_size: %.2f MiB | inference: %.4f ms/image (incl. host→device transfer)",
        final_eval_loss, final_acc, size_mib, lat_ms,
    )

    save_checkpoint(ckpt_final_path, {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "epoch": epochs,
        "test_acc": final_acc,
        "test_loss": final_eval_loss,
        "model_size_mib": size_mib,
        "inference_ms_per_image": lat_ms,
        "best_checkpoint_path": str(ckpt_best_path),
        "temperature": temperature,
        "alpha": alpha,
    })
    logger.info("Checkpoint KD (final): %s", ckpt_final_path.resolve())
    logger.info("Checkpoint KD (last):  %s", ckpt_last_path.resolve())
    logger.info("Checkpoint KD (best):  %s", ckpt_best_path.resolve())

    metrics.write({
        "kind": "meta",
        "epoch": epochs,
        "epochs_total": epochs,
        "step": steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "msg": "run_end",
        "final_acc": float(final_acc),
        "final_eval_loss": float(final_eval_loss),
    })

    wall_s = time.perf_counter() - run_t0
    summary_path = save_training_summary(
        run_dir,
        {
            "experiment": exp_name,
            "training_type": "kd",
            "config_path": str(args.config.resolve()),
            "run_dir": str(run_dir.resolve()),
            "device": str(device),
            "seed": int(cfg["experiment"]["seed"]),
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "wall_time_seconds": float(wall_s),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "distillation": {
                "temperature": temperature,
                "alpha": alpha,
                "teacher_checkpoint": str(t_ckpt_path.resolve()),
            },
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
