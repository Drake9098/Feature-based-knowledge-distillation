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

        state = state.get("state_dict", state.get("model_state_dict", state))

        keys_to_delete = ["conv1.weight", "fc.weight", "fc.bias"]
        common_prefixes = ["", "module.", "model.", "teacher."]
        for base_key in keys_to_delete:
            for prefix in common_prefixes:
                state.pop(f"{prefix}{base_key}", None)

        # Carica solo i pesi compatibili (shape match). Serve perché:
        # - conv1 è 3x3 (CIFAR) invece di 7x7 (ImageNet)
        # - fc è num_classes (100) invece di 1000 (ImageNet)
        model_sd = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for k, v in state.items():
            if k not in model_sd:
                continue
            if isinstance(v, torch.Tensor) and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(
                f"[teacher] checkpoint parziale: caricate {len(filtered)}/{len(state)} chiavi; "
                f"saltate {len(skipped)} per mismatch di shape (es. conv1/fc)."
            )
        if unexpected:
            print(f"[teacher] chiavi inattese nel checkpoint (ignorate): {len(unexpected)}")
        if missing:
            # Con stem CIFAR è normale che conv1/fc risultino missing.
            print(f"[teacher] chiavi mancanti (ok se conv1/fc): {len(missing)}")

    return model
