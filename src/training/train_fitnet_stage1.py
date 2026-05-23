"""Fase 3 — FitNets Stage 1: Hint Training.

Lo student impara a imitare le feature intermedie del teacher tramite regressori 1×1.
Loss: MSE(regressor(student_feat[guided_layer]), teacher_feat[hint_layer])

I layer dello student *dopo* l'ultimo guided layer vengono congelati;
solo la parte "FitNet" (fino al layer guidato) + i regressori vengono aggiornati.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from src.data.cifar100 import build_cifar100_loaders
from src.models.baseline import build_baseline
from src.models.teacher import build_teacher
from src.models.distillation_utils import FeatureExtractor, FitNetRegressor
from src.training.metrics import accuracy_percent, model_size_mb, inference_latency_ms
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_yaml_config
from src.utils.logging import setup_run_logger
from src.utils.metrics_jsonl import MetricsWriter
from src.utils.seed import set_seed
from src.utils.training_summary import save_training_summary, utc_now_iso

# Ordine canonico dei layer in ResNet-18 (esclude maxpool=Identity e avgpool senza parametri)
_STUDENT_LAYER_ORDER = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


def _freeze_after_guided(student: nn.Module, student_layers: list[str]) -> None:
    """Congela i parametri dello student nei layer DOPO l'ultimo layer supervisionato.

    Congela sempre anche `fc` e `avgpool` (non usati in Stage 1).
    """
    guided_indices = [
        _STUDENT_LAYER_ORDER.index(l)
        for l in student_layers
        if l in _STUDENT_LAYER_ORDER
    ]
    if not guided_indices:
        return

    last_idx = max(guided_indices)
    to_freeze = _STUDENT_LAYER_ORDER[last_idx + 1:] + ["fc", "avgpool"]

    for freeze_name in to_freeze:
        for mod_name, module in student.named_modules():
            if mod_name == freeze_name or mod_name.startswith(freeze_name + "."):
                for p in module.parameters():
                    p.requires_grad_(False)


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
    p = argparse.ArgumentParser(description="FitNets Stage 1 — Hint Training")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fitnet_middle_s1.yaml"),
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
            "Esegui prima train_teacher_finetune.py e aggiorna model.teacher_checkpoint nel YAML."
        )
    ckpt = load_checkpoint(t_ckpt_path, map_location="cpu")
    teacher.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Teacher: caricato da {t_ckpt_path}")
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- Student (pesi random) ---
    student = build_baseline(cfg["model"])

    # --- Hint layers dal config ---
    fitnet_cfg = cfg["fitnet"]
    hint_layers_cfg: list[dict] = fitnet_cfg["hint_layers"]

    teacher_layer_names = [h["teacher_layer"] for h in hint_layers_cfg]
    student_layer_names = [h["student_layer"] for h in hint_layers_cfg]
    # Coppie: [(teacher_layer, student_layer, t_channels, s_channels), ...]
    layer_pairs = [
        (h["teacher_layer"], h["student_layer"], int(h["teacher_channels"]), int(h["student_channels"]))
        for h in hint_layers_cfg
    ]

    # Congela layer dello student dopo l'ultimo layer supervisionato
    _freeze_after_guided(student, student_layer_names)

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(
        f"Student: {n_trainable:,}/{n_total:,} parametri addestrabili "
        f"({100 * n_trainable / max(n_total, 1):.1f}%)"
    )

    student = student.to(device)

    # --- Regressori FitNet (uno per coppia di layer) ---
    regressors = nn.ModuleDict({
        s_layer: FitNetRegressor(s_ch, t_ch).to(device)
        for _, s_layer, t_ch, s_ch in layer_pairs
    })
    print(f"Regressori: {list(regressors.keys())}")

    # --- Feature extractors ---
    t_extractor = FeatureExtractor(teacher, teacher_layer_names)
    s_extractor = FeatureExtractor(student, student_layer_names)

    # --- Loss ---
    mse_criterion = nn.MSELoss()

    # --- Optimizer: solo parametri con requires_grad=True (student parziale + regressori) ---
    t_cfg = cfg["training"]
    trainable_params = (
        [p for p in student.parameters() if p.requires_grad]
        + list(regressors.parameters())
    )
    optimizer = torch.optim.SGD(
        trainable_params,
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
            milestones = scheduler_cfg.get("milestones", [60, 80])
            gamma = float(scheduler_cfg.get("gamma", 0.1))
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

    logger = setup_run_logger(run_dir=run_dir, training_type="fitnet_s1", name="train.fitnet_s1")
    metrics_writer = MetricsWriter(
        path=run_dir / "metrics.jsonl",
        experiment_name=exp_name,
        training_type="fitnet_s1",
        run_dir=run_dir,
        config_path=str(args.config),
    )
    logger.info("Run dir: %s", run_dir.resolve())
    logger.info("Teacher ckpt: %s", t_ckpt_path.resolve())
    logger.info("Hint layers: %s", [(t, s) for t, s, _, _ in layer_pairs])

    # --- Checkpoint paths ---
    ckpt_stem = f"{exp_name}_s1"
    ckpt_last_path = run_dir / f"{ckpt_stem}_last.pt"
    ckpt_best_path = run_dir / f"{ckpt_stem}_best.pt"
    ckpt_final_path = run_dir / f"{ckpt_stem}_final.pt"

    best_mse = float("inf")
    best_epoch: int | None = None
    per_epoch: list[dict] = []

    run_t0 = time.perf_counter()
    started_at = utc_now_iso()

    epochs = int(t_cfg["epochs"])
    steps_per_epoch = len(train_loader)
    log_every_steps = int(t_cfg.get("log_every_steps", 50))
    global_step = 0

    metrics_writer.write({
        "kind": "meta",
        "epoch": 0,
        "epochs_total": epochs,
        "step": 0,
        "steps_per_epoch": steps_per_epoch,
        "msg": "run_start",
        "hint_layers": [(t, s) for t, s, _, _ in layer_pairs],
    })

    # --- Training loop ---
    for epoch in range(epochs):
        s_extractor.train()
        regressors.train()
        running_mse = 0.0
        n_train = 0
        epoch_start = time.perf_counter()
        last_step_t = time.perf_counter()

        for inputs, _labels in train_loader:
            step_start = time.perf_counter()
            inputs = inputs.to(device, non_blocking=True)

            # Teacher: forward senza gradienti
            with torch.no_grad():
                _, teacher_feats = t_extractor(inputs)

            # Student: forward con gradienti (solo layer non congelati ricevono grad)
            optimizer.zero_grad(set_to_none=True)
            _, student_feats = s_extractor(inputs)

            # MSE per ogni coppia di layer
            loss = torch.tensor(0.0, device=device)
            for t_layer, s_layer, _, _ in layer_pairs:
                t_feat = teacher_feats[t_layer]
                s_feat = student_feats[s_layer]
                projected = regressors[s_layer](s_feat)
                loss = loss + mse_criterion(projected, t_feat)

            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            running_mse += loss.item() * bs
            n_train += bs
            global_step += 1

            if log_every_steps > 0 and (global_step % log_every_steps == 0):
                step_time_s = step_start - last_step_t
                last_step_t = step_start
                metrics_writer.write({
                    "kind": "train",
                    "epoch": epoch + 1,
                    "epochs_total": epochs,
                    "step": (global_step % steps_per_epoch) or steps_per_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "mse_loss": float(running_mse / max(n_train, 1)),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "step_time_s": float(step_time_s),
                })

        train_mse = running_mse / max(n_train, 1)

        # --- Eval: MSE sul test set + accuracy dello student ---
        s_extractor.eval()
        regressors.eval()
        eval_mse_sum = 0.0
        n_eval = 0
        with torch.inference_mode():
            for inputs, _labels in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                _, t_feats = t_extractor(inputs)
                _, s_feats = s_extractor(inputs)
                batch_mse = 0.0
                for t_layer, s_layer, _, _ in layer_pairs:
                    projected = regressors[s_layer](s_feats[s_layer])
                    batch_mse += mse_criterion(projected, t_feats[t_layer]).item()
                eval_mse_sum += batch_mse * inputs.size(0)
                n_eval += inputs.size(0)
        eval_mse = eval_mse_sum / max(n_eval, 1)

        # Accuracy dello student (frozen layers producono logits non ottimizzati — solo indicativa)
        acc = accuracy_percent(s_extractor.model, eval_loader, device)

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        epoch_time_s = time.perf_counter() - epoch_start

        logger.info(
            "Epoch %d/%d | train_mse: %.6f | eval_mse: %.6f | student_acc: %.2f%% | lr: %.6g",
            epoch + 1, epochs, train_mse, eval_mse, acc, lr,
        )
        metrics_writer.write({
            "kind": "eval",
            "epoch": epoch + 1,
            "epochs_total": epochs,
            "step": steps_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "mse_loss": float(eval_mse),
            "acc": float(acc),
            "lr": float(lr),
            "epoch_time_s": float(epoch_time_s),
        })
        per_epoch.append({
            "epoch": epoch + 1,
            "train_mse": float(train_mse),
            "eval_mse": float(eval_mse),
            "student_acc_percent": float(acc),
            "lr": float(lr),
            "epoch_time_s": float(epoch_time_s),
        })

        # Checkpoint last
        save_checkpoint(ckpt_last_path, {
            "model_state_dict": student.state_dict(),
            "regressors_state_dict": regressors.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch + 1,
            "eval_mse": eval_mse,
            "is_best": False,
            "best_mse_so_far": best_mse,
        })

        # Checkpoint best (minima MSE eval)
        if eval_mse < best_mse:
            best_mse = eval_mse
            best_epoch = epoch + 1
            save_checkpoint(ckpt_best_path, {
                "model_state_dict": student.state_dict(),
                "regressors_state_dict": regressors.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                "epoch": epoch + 1,
                "eval_mse": eval_mse,
                "is_best": True,
                "best_mse_so_far": best_mse,
            })
            logger.info("[ckpt] nuovo best: eval_mse=%.6f -> %s", best_mse, ckpt_best_path.resolve())

    # --- Checkpoint final (best model ricaricato) ---
    if ckpt_best_path.is_file():
        best_state = load_checkpoint(ckpt_best_path, map_location="cpu")
        student.load_state_dict(best_state["model_state_dict"], strict=True)
        regressors.load_state_dict(best_state["regressors_state_dict"])
        student = student.to(device)

    student.eval()
    final_acc = accuracy_percent(student, eval_loader, device)
    size_mib = model_size_mb(student)
    lat_ms = inference_latency_ms(student, eval_loader, device, cfg["metrics"])

    logger.info(
        "Final(best) | eval_mse: %.6f | student_acc: %.2f%% | "
        "model_size: %.2f MiB | inference: %.4f ms/image",
        best_mse, final_acc, size_mib, lat_ms,
    )

    save_checkpoint(ckpt_final_path, {
        "model_state_dict": student.state_dict(),
        "regressors_state_dict": regressors.state_dict(),
        "epoch": epochs,
        "eval_mse": best_mse,
        "student_acc": final_acc,
        "model_size_mib": size_mib,
        "inference_ms_per_image": lat_ms,
        "best_checkpoint_path": str(ckpt_best_path),
    })
    logger.info("Checkpoint S1 (final): %s", ckpt_final_path.resolve())
    logger.info("Checkpoint S1 (best):  %s", ckpt_best_path.resolve())
    logger.info("Checkpoint S1 (last):  %s", ckpt_last_path.resolve())
    logger.info("")
    logger.info(">>> Per Stage 2, imposta nel config:")
    logger.info("    model.student_checkpoint: %s", str(ckpt_best_path.resolve()))

    metrics_writer.write({
        "kind": "meta",
        "epoch": epochs,
        "epochs_total": epochs,
        "step": steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "msg": "run_end",
        "best_eval_mse": float(best_mse),
        "final_acc": float(final_acc),
    })

    wall_s = time.perf_counter() - run_t0
    summary_path = save_training_summary(
        run_dir,
        {
            "experiment": exp_name,
            "training_type": "fitnet_s1",
            "config_path": str(args.config.resolve()),
            "run_dir": str(run_dir.resolve()),
            "device": str(device),
            "seed": int(cfg["experiment"]["seed"]),
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
            "wall_time_seconds": float(wall_s),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "hint_layers": [
                {
                    "teacher_layer": t_l,
                    "student_layer": s_l,
                    "teacher_channels": t_ch,
                    "student_channels": s_ch,
                }
                for t_l, s_l, t_ch, s_ch in layer_pairs
            ],
            "per_epoch": per_epoch,
            "best": (
                None
                if best_epoch is None
                else {
                    "epoch": best_epoch,
                    "eval_mse": float(best_mse),
                    "checkpoint_path": str(ckpt_best_path.resolve()),
                }
            ),
            "final": {
                "eval_mse": float(best_mse),
                "student_acc_percent": float(final_acc),
                "model_size_mib": float(size_mib),
                "inference_ms_per_image": float(lat_ms),
                "student_checkpoint_for_stage2": str(ckpt_best_path.resolve()),
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
