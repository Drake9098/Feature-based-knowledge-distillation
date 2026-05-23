"""Attention Transfer (AT) — Zagoruyko & Komodakis, ICLR 2017.

Loss totale: L_CE + L_KD + L_AT

    L_KD  = alpha * CE(hard) + (1 - alpha) * T² * KL(soft_student || soft_teacher)
    L_AT  = (beta/2) * Σ_j || Q_S^j/‖Q_S^j‖₂ − Q_T^j/‖Q_T^j‖₂ ‖₂²

Nessun regressore: le mappe di attenzione sono confrontate direttamente dopo
normalizzazione L2, senza proiettare canali. Un unico stadio di training.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from src.data.cifar100 import build_cifar100_loaders
from src.models.baseline import build_baseline
from src.models.teacher import build_teacher
from src.models.distillation_utils import FeatureExtractor
from src.training.attention_utils import at_loss
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
    p = argparse.ArgumentParser(description="Attention Transfer (AT) — Zagoruyko & Komodakis 2017")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/at_kd.yaml"),
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
            "Aggiorna model.teacher_checkpoint nel YAML."
        )
    ckpt = load_checkpoint(t_ckpt_path, map_location="cpu")
    teacher.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Teacher: caricato da {t_ckpt_path}")
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- Student ---
    student = build_baseline(cfg["model"])
    s_ckpt_path_raw = cfg["model"].get("student_checkpoint")
    if s_ckpt_path_raw:
        s_ckpt_path = Path(s_ckpt_path_raw)
        if not s_ckpt_path.is_file():
            raise FileNotFoundError(f"Student checkpoint non trovato: {s_ckpt_path}")
        s_ckpt = load_checkpoint(s_ckpt_path, map_location="cpu")
        student.load_state_dict(s_ckpt["model_state_dict"], strict=True)
        print(f"Student: warm-start da {s_ckpt_path}")
    student = student.to(device)

    # --- AT config ---
    at_cfg = cfg["attention_transfer"]
    beta_0: float = float(at_cfg["beta"])
    beta_decay: bool = bool(at_cfg.get("beta_decay", False))
    at_layers: list[str] = list(at_cfg["layers"])
    beta: float = beta_0  # valore corrente, aggiornato ad ogni milestone se beta_decay=True
    initial_lr: float = float(cfg["training"]["learning_rate"])
    print(f"AT: beta_0={beta_0}, beta_decay={beta_decay}, layers={at_layers}")

    # --- Feature extractors (hooks su layer1..4 per entrambi i modelli) ---
    t_extractor = FeatureExtractor(teacher, at_layers)
    s_extractor = FeatureExtractor(student, at_layers)

    # --- KD loss ---
    kd_cfg = cfg["distillation"]
    temperature = float(kd_cfg["temperature"])
    alpha = float(kd_cfg["alpha"])
    kd_criterion = KDLoss(temperature=temperature, alpha=alpha)
    print(f"KD: temperature={temperature}, alpha={alpha}")

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

    logger = setup_run_logger(run_dir=run_dir, training_type="at", name="train.at")
    metrics = MetricsWriter(
        path=run_dir / "metrics.jsonl",
        experiment_name=exp_name,
        training_type="at",
        run_dir=run_dir,
        config_path=str(args.config),
    )
    logger.info("Run dir: %s", run_dir.resolve())
    logger.info("Teacher ckpt: %s", t_ckpt_path.resolve())
    logger.info("AT layers: %s  beta=%.1f", at_layers, beta)
    logger.info("KD: temperature=%.1f  alpha=%.3f", temperature, alpha)

    # --- Checkpoint paths ---
    ckpt_stem = f"{exp_name}_at"
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
        "beta": beta,
        "at_layers": at_layers,
    })

    # --- Training loop ---
    for epoch in range(epochs):
        s_extractor.train()
        running_loss = 0.0
        running_kd = 0.0
        running_at = 0.0
        n_train = 0
        epoch_start = time.perf_counter()
        last_step_t = time.perf_counter()

        for inputs, labels in train_loader:
            step_start = time.perf_counter()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Teacher: forward senza gradienti — cattura logits + mappe di attenzione
            with torch.no_grad():
                teacher_logits, teacher_feats = t_extractor(inputs)

            # Student: forward con gradienti — cattura logits + mappe di attenzione
            optimizer.zero_grad(set_to_none=True)
            student_logits, student_feats = s_extractor(inputs)

            # KD loss: alpha*CE(hard) + (1-alpha)*T²*KL(soft)
            kd_l = kd_criterion(student_logits, teacher_logits, labels)

            # AT loss: (beta/2) * Σ_j ||Q_S^j/‖Q_S^j‖₂ − Q_T^j/‖Q_T^j‖₂||₂²
            at_l = at_loss(student_feats, teacher_feats, at_layers, beta)

            loss = kd_l + at_l

            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_kd += kd_l.item() * bs
            running_at += at_l.item() * bs
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
                    "kd_loss": float(running_kd / max(n_train, 1)),
                    "at_loss": float(running_at / max(n_train, 1)),
                    "beta": float(beta),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "step_time_s": float(step_time_s),
                })

        train_loss = running_loss / max(n_train, 1)
        train_kd = running_kd / max(n_train, 1)
        train_at = running_at / max(n_train, 1)

        # --- Eval: CE sul test set + accuracy student ---
        s_extractor.eval()
        eval_loss_sum = 0.0
        n_eval = 0
        with torch.inference_mode():
            for inputs, labels in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = s_extractor.model(inputs)
                eval_loss_sum += F.cross_entropy(logits, labels).item() * labels.size(0)
                n_eval += labels.size(0)
        eval_loss = eval_loss_sum / max(n_eval, 1)

        acc = accuracy_percent(s_extractor.model, eval_loader, device)

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # Beta decay proporzionale all'LR: beta(t) = beta_0 * lr(t) / lr_0
        if beta_decay:
            beta = beta_0 * (lr / initial_lr)

        epoch_time_s = time.perf_counter() - epoch_start

        logger.info(
            "Epoch %d/%d | train_loss: %.4f (kd=%.4f at=%.4f) | "
            "test_loss: %.4f | test_acc: %.2f%% | lr: %.6g | beta: %.4g",
            epoch + 1, epochs,
            train_loss, train_kd, train_at,
            eval_loss, acc, lr, beta,
        )
        metrics.write({
            "kind": "eval",
            "epoch": epoch + 1,
            "epochs_total": epochs,
            "step": steps_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "loss": float(eval_loss),
            "acc": float(acc),
            "beta": float(beta),
            "lr": float(lr),
            "epoch_time_s": float(epoch_time_s),
        })
        per_epoch.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_kd_loss": float(train_kd),
            "train_at_loss": float(train_at),
            "test_loss": float(eval_loss),
            "test_accuracy_percent": float(acc),
            "beta": float(beta),
            "lr": float(lr),
            "epoch_time_s": float(epoch_time_s),
        })

        # Checkpoint "last" ad ogni epoca
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

        # Checkpoint "best" quando migliora la accuracy sul test set
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
        "model_size: %.2f MiB | inference: %.4f ms/image",
        final_eval_loss, final_acc, size_mib, lat_ms,
    )

    save_checkpoint(ckpt_final_path, {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "test_acc": final_acc,
        "test_loss": final_eval_loss,
        "model_size_mib": size_mib,
        "inference_ms_per_image": lat_ms,
        "best_checkpoint_path": str(ckpt_best_path),
        "temperature": temperature,
        "alpha": alpha,
        "beta": beta,
        "at_layers": at_layers,
    })
    logger.info("Checkpoint AT (final): %s", ckpt_final_path.resolve())
    logger.info("Checkpoint AT (last):  %s", ckpt_last_path.resolve())
    logger.info("Checkpoint AT (best):  %s", ckpt_best_path.resolve())

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
            "training_type": "at",
            "config_path": str(args.config.resolve()),
            "run_dir": str(run_dir.resolve()),
            "device": str(device),
            "seed": int(cfg["experiment"]["seed"]),
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "wall_time_seconds": float(wall_s),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "attention_transfer": {
                "beta_0": beta_0,
                "beta_final": beta,
                "beta_decay": beta_decay,
                "at_layers": at_layers,
            },
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
