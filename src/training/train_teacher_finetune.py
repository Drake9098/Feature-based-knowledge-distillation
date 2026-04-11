"""Fine-tuning del teacher ResNet-50 su CIFAR-100 (CE) con LR separati per conv1/fc e backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data.cifar100 import build_cifar100_loaders
from src.models.teacher import build_teacher
from src.training.metrics import accuracy_percent, inference_latency_ms, model_size_mb
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_yaml_config
from src.utils.seed import set_seed


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
    criterion = nn.CrossEntropyLoss()

    print(
        f"Optimizer: conv1+fc lr={lr_new}, backbone lr={lr_backbone} "
        f"(mult={t_cfg['backbone_lr_mult']}), wd={wd}, momentum={momentum}",
    )

    epochs = int(t_cfg["epochs"])
    for epoch in range(epochs):
        teacher.train()
        running_loss = 0.0
        n_train = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = teacher(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_train += labels.size(0)
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

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {eval_loss:.4f} | "
            f"test_acc: {acc:.2f}%",
        )

    size_mib = model_size_mb(teacher)
    lat_ms = inference_latency_ms(teacher, eval_loader, device, cfg["metrics"])
    print(
        f"Final | model_size: {size_mib:.2f} MiB | "
        f"inference: {lat_ms:.4f} ms/image (incl. host→device transfer)",
    )

    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_name = f"{cfg['experiment']['name']}_teacher.pt"
    ckpt_path = ckpt_dir / ckpt_name
    save_checkpoint(
        ckpt_path,
        {
            "model_state_dict": teacher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs,
            "test_acc": acc,
            "train_loss": train_loss,
            "test_loss": eval_loss,
            "model_size_mib": size_mib,
            "inference_ms_per_image": lat_ms,
            "lr_new": lr_new,
            "lr_backbone": lr_backbone,
        },
    )
    print(f"Checkpoint teacher: {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
