"""Fine-tuning del teacher ResNet-50 su CIFAR-100 (CE) con LR separati per conv1/fc e backbone."""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from src.data.cifar100 import build_cifar100_loaders
from src.models.teacher import build_teacher
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


def _param_groups_for_resnet_teacher(
    model: nn.Module,
    lr_new: float,
    lr_backbone: float,
    weight_decay: float,
) -> list[dict]:
    """conv1.* e fc.* → lr_new; tutto il resto (bn1, layer1–4, …) → lr_backbone."""
    new_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("conv1.") or name.startswith("fc."):
            new_params.append(p)
        else:
            backbone_params.append(p)
    if not new_params:
        raise RuntimeError("Nessun parametro in conv1/fc: verifica l'architettura.")
    if not backbone_params:
        raise RuntimeError("Nessun parametro backbone: verifica l'architettura.")
    return [
        {"params": new_params, "lr": lr_new, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tuning teacher ResNet-50 su CIFAR-100 (LR conv1+fc vs backbone)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teacher_finetune.yaml"),
        help="Percorso al file YAML.",
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

    teacher = build_teacher(cfg["model"])
    teacher = teacher.to(device)

    t_cfg = cfg["training"]
    lr_new = float(t_cfg["learning_rate_new"])
    lr_backbone = lr_new * float(t_cfg["backbone_lr_mult"])
    wd = float(t_cfg["weight_decay"])
    momentum = float(t_cfg["momentum"])

    param_groups = _param_groups_for_resnet_teacher(teacher, lr_new, lr_backbone, wd)
    optimizer = torch.optim.SGD(param_groups, momentum=momentum)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Scheduler (MultiStepLR) opzionale via YAML:
    # training:
    #   scheduler:
    #     name: multistep
    #     milestones: [30, 40]
    #     gamma: 0.1
    scheduler = None
    scheduler_cfg = t_cfg.get("scheduler")
    if isinstance(scheduler_cfg, dict):
        name = str(scheduler_cfg.get("name", "multistep")).lower().strip()
        if name in {"multistep", "multi_step", "multisteplr"}:
            milestones = scheduler_cfg.get("milestones")
            if milestones is None:
                # Default: drop a ~60% e ~80% delle epoche
                # TODO: se vuoi replicare una specifica ricetta paper, imposta milestones esplicite nel YAML.
                milestones = [max(1, int(0.6 * int(t_cfg["epochs"]))), max(1, int(0.8 * int(t_cfg["epochs"])))]
            gamma = float(scheduler_cfg.get("gamma", 0.1))
            scheduler = MultiStepLR(optimizer, milestones=[int(m) for m in milestones], gamma=gamma)
        else:
            raise ValueError(f"Scheduler non supportato: {name}. Usa name: multistep oppure rimuovi training.scheduler.")

    print(
        f"Optimizer: conv1+fc lr={lr_new}, backbone lr={lr_backbone} "
        f"(mult={t_cfg['backbone_lr_mult']}), wd={wd}, momentum={momentum}",
    )
    if scheduler is not None:
        print(f"Scheduler: MultiStepLR milestones={scheduler.milestones} gamma={scheduler.gamma}")

    ckpt_root = Path(cfg["checkpoint"]["dir"])
    # Salviamo checkpoint progressivi e best model in una sottocartella per-run basata su timestamp,
    # così run diverse non si sovrascrivono tra loro.
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

    logger = setup_run_logger(run_dir=run_dir, training_type="teacher", name="train.teacher")
    metrics = MetricsWriter(
        path=run_dir / "metrics.jsonl",
        experiment_name=exp_name,
        training_type="teacher",
        run_dir=run_dir,
        config_path=str(args.config),
    )
    logger.info("Run dir: %s", run_dir.resolve())
    logger.info("Metrics:  %s", (run_dir / "metrics.jsonl").resolve())

    # Salviamo checkpoint progressivi e best model (sovrascritto quando migliora).
    # Esempio: <exp>_teacher_last.pt e <exp>_teacher_best.pt
    ckpt_stem = f"{exp_name}_teacher"
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
        teacher.train()
        running_loss = 0.0
        n_train = 0
        epoch_start = time.perf_counter()
        last_step_t = time.perf_counter()
        for inputs, labels in train_loader:
            step_start = time.perf_counter()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = teacher(inputs)
            loss = criterion(logits, labels)
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

        teacher.eval()
        eval_loss_sum = 0.0
        n_eval = 0
        with torch.inference_mode():
            for inputs, labels in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = teacher(inputs)
                batch_loss = criterion(logits, labels)
                eval_loss_sum += batch_loss.item() * labels.size(0)
                n_eval += labels.size(0)
        eval_loss = eval_loss_sum / max(n_eval, 1)

        acc = accuracy_percent(teacher, eval_loader, device)

        if scheduler is not None:
            scheduler.step()
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        epoch_time_s = time.perf_counter() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {eval_loss:.4f} | "
            f"test_acc: {acc:.2f}% | "
            f"lr(new/backbone): {lrs[0]:.6g}/{lrs[1]:.6g}",
        )
        # Usiamo lr del primo group per display/monitor (conv1+fc)
        metrics.write(
            {
                "kind": "eval",
                "epoch": epoch + 1,
                "epochs_total": epochs,
                "step": steps_per_epoch,
                "steps_per_epoch": steps_per_epoch,
                "loss": float(eval_loss),
                "acc": float(acc),
                "lr": float(lrs[0]),
                "epoch_time_s": float(epoch_time_s),
            }
        )
        per_epoch.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "test_loss": float(eval_loss),
                "test_accuracy_percent": float(acc),
                "lr_new": float(lrs[0]),
                "lr_backbone": float(lrs[1]),
                "epoch_time_s": float(epoch_time_s),
            }
        )

        # Checkpoint "last" a ogni epoca (progressivo).
        save_checkpoint(
            ckpt_last_path,
            {
                "model_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                "epoch": epoch + 1,
                "test_acc": acc,
                "train_loss": train_loss,
                "test_loss": eval_loss,
                "lr_new": lr_new,
                "lr_backbone": lr_backbone,
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
                    "model_state_dict": teacher.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "test_acc": acc,
                    "train_loss": train_loss,
                    "test_loss": eval_loss,
                    "lr_new": lr_new,
                    "lr_backbone": lr_backbone,
                    "is_best": True,
                    "best_acc_so_far": best_acc,
                },
            )
            logger.info("[ckpt] nuovo best: acc=%.2f%% -> %s", best_acc, ckpt_best_path.resolve())

    # Eval finale usando il best model (non l'ultima epoca).
    if ckpt_best_path.is_file():
        best_state = load_checkpoint(ckpt_best_path, map_location="cpu")
        teacher.load_state_dict(best_state["model_state_dict"], strict=True)
        teacher = teacher.to(device)
        teacher.eval()

    # Ricalcola loss/acc finali sul best model
    teacher.eval()
    eval_loss_sum = 0.0
    n_eval = 0
    with torch.inference_mode():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = teacher(inputs)
            batch_loss = criterion(logits, labels)
            eval_loss_sum += batch_loss.item() * labels.size(0)
            n_eval += labels.size(0)
    final_eval_loss = eval_loss_sum / max(n_eval, 1)
    final_acc = accuracy_percent(teacher, eval_loader, device)

    size_mib = model_size_mb(teacher)
    lat_ms = inference_latency_ms(teacher, eval_loader, device, cfg["metrics"])
    logger.info(
        f"Final(best) | test_loss: {final_eval_loss:.4f} | test_acc: {final_acc:.2f}% | "
        f"model_size: {size_mib:.2f} MiB | inference: {lat_ms:.4f} ms/image "
        f"(incl. host→device transfer)",
    )

    # Salviamo anche un "final" (con metriche aggiornate sul best) per comodità.
    ckpt_path = ckpt_final_path
    save_checkpoint(
        ckpt_path,
        {
            "model_state_dict": teacher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "epoch": epochs,
            "test_acc": final_acc,
            "test_loss": final_eval_loss,
            "model_size_mib": size_mib,
            "inference_ms_per_image": lat_ms,
            "lr_new": lr_new,
            "lr_backbone": lr_backbone,
            "best_checkpoint_path": str(ckpt_best_path),
        },
    )
    logger.info("Checkpoint teacher (final): %s", ckpt_path.resolve())
    logger.info("Checkpoint teacher (last):  %s", ckpt_last_path.resolve())
    logger.info("Checkpoint teacher (best):  %s", ckpt_best_path.resolve())
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
            "training_type": "teacher",
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
            "optimizer": {
                "lr_new": float(lr_new),
                "lr_backbone": float(lr_backbone),
                "weight_decay": float(wd),
                "momentum": float(momentum),
            },
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
