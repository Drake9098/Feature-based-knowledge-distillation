"""Fase 1 — addestramento student con sola cross-entropy (nessuna distillazione)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data.cifar100 import build_cifar100_loaders
from src.models.student import build_student
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_yaml_config
from src.utils.seed import set_seed
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
    p = argparse.ArgumentParser(description="Fase 1 — baseline student (CE only)")
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

    student = build_student(cfg["model"])
    student = student.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student.parameters(), lr=cfg["training"]["learning_rate"], momentum=cfg["training"]["momentum"], weight_decay=cfg["training"]["weight_decay"])
    
    epochs = int(cfg["training"]["epochs"])
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        n_train = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = student(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_train += labels.size(0)
        train_loss = running_loss / max(n_train, 1)

        student.eval()
        eval_loss_sum = 0.0
        n_eval = 0
        with torch.inference_mode():
            for inputs, labels in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = student(inputs)
                batch_loss = criterion(outputs, labels)
                eval_loss_sum += batch_loss.item() * labels.size(0)
                n_eval += labels.size(0)
        eval_loss = eval_loss_sum / max(n_eval, 1)

        acc = accuracy_percent(student, eval_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {eval_loss:.4f} | "
            f"test_acc: {acc:.2f}%"
        )

    size_mib = model_size_mb(student)
    lat_ms = inference_latency_ms(student, eval_loader, device, cfg["metrics"])
    print(
        f"Final | model_size: {size_mib:.2f} MiB | "
        f"inference: {lat_ms:.4f} ms/image (incl. host→device transfer)"
    )

    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_name = f"{cfg['experiment']['name']}_student_baseline.pt"
    ckpt_path = ckpt_dir / ckpt_name
    save_checkpoint(
        ckpt_path,
        {
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs,
            "test_acc": acc,
            "train_loss": train_loss,
            "test_loss": eval_loss,
            "model_size_mib": size_mib,
            "inference_ms_per_image": lat_ms,
        },
    )
    print(f"Checkpoint salvato: {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
