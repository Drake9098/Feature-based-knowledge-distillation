"""Student: rete più leggera per baseline e distillazione."""

from typing import Any, Dict

import torch.nn as nn
import torchvision.models as models


def build_baseline(model_config: Dict[str, Any]) -> nn.Module:
    """ResNet-18 adattato a CIFAR (32×32): conv1 3×3 stride 1, senza primo max pooling.

    ``model_config`` deve contenere almeno ``num_classes``. Opzionale: ``student_name``.

    Returns:
        Modulo che mappa un batch ``[B, 3, 32, 32]`` in logits ``[B, num_classes]``.
    """
    name = model_config.get("student_name", "resnet18")
    if name != "resnet18":
        raise ValueError(f"Student non supportato: {name!r}.")

    num_classes = int(model_config["num_classes"])

    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model
