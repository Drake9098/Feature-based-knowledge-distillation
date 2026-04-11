"""Teacher: ResNet-50 con backbone ImageNet e testa CIFAR-100"""

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.models as models


def build_teacher(model_config: Dict[str, Any]) -> nn.Module:
    """ResNet-50 per input 32×32, ``num_classes`` dal config.

    - Stem tipo CIFAR: ``conv1`` 3×3 stride 1, ``maxpool`` disattivato (come lo student).
    - Se ``pretrained_teacher`` è true, carica ``teacher_weights_path``:
      state_dict puro (es. ImageNet da ``download_offline_assets``) oppure checkpoint con chiave
      ``model_state_dict`` (es. output di ``train_teacher_finetune``). Con ImageNet si usa
      ``strict=False`` così il backbone pretrain si allinea e conv1/fc restano nuovi fino al fine-tuning.
    """
    name = model_config.get("teacher_name", "resnet50")
    if name != "resnet50":
        raise ValueError(f"Teacher non supportato: {name!r}.")

    num_classes = int(model_config["num_classes"])

    model = models.resnet50(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    use_pretrained = bool(model_config.get("pretrained_teacher", True))
    if use_pretrained:
        path = model_config.get("teacher_weights_path")
        if not path:
            raise ValueError("pretrained_teacher=true richiede teacher_weights_path nel config.")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint teacher non trovato: {path.resolve()}")

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except (TypeError, RuntimeError):
            state = torch.load(path, map_location="cpu")

        if not isinstance(state, dict):
            raise TypeError("Il file teacher deve essere un dict (state_dict o checkpoint).")

        if "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state, strict=False)

    return model
