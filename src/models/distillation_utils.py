from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class FitNetRegressor(nn.Module):
    """
    FitNets regressor: proietta una feature map dello Student nello spazio del Teacher.

    Architettura: Conv2d 1x1 + BatchNorm2d
    """

    def __init__(self, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        if student_channels <= 0:
            raise ValueError("student_channels deve essere > 0")
        if teacher_channels <= 0:
            raise ValueError("teacher_channels deve essere > 0")

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=student_channels,
                out_channels=teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(teacher_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeatureExtractor(nn.Module):
    """
    Wrapper per estrarre feature intermedie tramite forward hooks, senza modificare il modello base.

    - target_layers: lista di nomi come appaiono in model.named_modules() (es. "layer1.0.relu")
    - forward(x) -> (logits, features_dict)
    """

    def __init__(self, model: nn.Module, target_layers: list[str]) -> None:
        super().__init__()
        self.model = model
        self.target_layers = list(target_layers)

        self.features: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        wanted = set(self.target_layers)
        for name, module in self.model.named_modules():
            if name in wanted:
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

        missing = wanted.difference({n for n, _ in self.model.named_modules()})
        if missing:
            # Fail fast: evita training "silenzioso" senza feature attese.
            raise ValueError(f"Layer non trovati in model.named_modules(): {sorted(missing)}")

    def _make_hook(self, layer_name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            # Importante: NON fare detach/clone; preserviamo il graph per gradient flow nello student.
            self.features[layer_name] = output

        return hook

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.features.clear()
        logits = self.model(x)
        return logits, self.features

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

