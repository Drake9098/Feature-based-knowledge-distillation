from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    r"""
    Funzione di costo combinata per la Knowledge Distillation classica (Hinton et al.).

    Implementa la loss:

    L = alpha * CE(y, z_s) + (1 - alpha) * T^2 * KL(softmax(z_s/T) || softmax(z_t/T))

    dove:
    - z_s: logits dello student
    - z_t: logits del teacher
    - y: etichette hard (ground-truth)
    - T: temperature
    - alpha: bilanciamento hard vs soft target
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature deve essere > 0")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha deve essere in [0, 1]")

        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Loss standard sulle etichette reali (Hard Target)
        hard_loss = self.cross_entropy(student_logits, labels)

        # 2. Loss di distillazione sulle probabilità ammorbidite (Soft Target)
        # PyTorch richiede che l'input della KLDiv sia in spazio log-prob (log_softmax)
        # e che il target sia in spazio di probabilità lineare (softmax)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

        kl_loss = self.kl_div(soft_student, soft_teacher)

        # 3. Combinazione
        # Hinton impone di moltiplicare la KL per T^2 per bilanciare i gradienti
        loss = (self.alpha * hard_loss) + (
            (1.0 - self.alpha) * (self.temperature**2) * kl_loss
        )

        return loss

