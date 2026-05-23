"""Attention Transfer utilities — Zagoruyko & Komodakis, ICLR 2017.

Implementa la funzione di attenzione F_sum^2 e la AT loss:

    L_AT = (β/2) * Σ_j || Q_S^j / ||Q_S^j||_2  −  Q_T^j / ||Q_T^j||_2 ||_2²

dove Q^j è la mappa di attenzione piatta (H·W,) per il layer j, ottenuta
sommando i quadrati delle attivazioni lungo la dimensione dei canali.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_map(feat: torch.Tensor) -> torch.Tensor:
    """Calcola la mappa F_sum^2 da un tensore di feature.

    Args:
        feat: Tensore (B, C, H, W).

    Returns:
        Mappa piatta (B, H·W) — **non normalizzata**.
    """
    # Somma dei quadrati lungo la dimensione canali → (B, H, W)
    a = feat.pow(2).sum(dim=1)
    return a.view(a.shape[0], -1)  # (B, H·W)


def at_loss(
    student_feats: dict[str, torch.Tensor],
    teacher_feats: dict[str, torch.Tensor],
    at_layers: list[str],
    beta: float,
) -> torch.Tensor:
    """Loss AT: (β/2) · Σ_j || Q_S^j/‖Q_S^j‖₂ − Q_T^j/‖Q_T^j‖₂ ‖₂².

    Se student e teacher hanno risoluzioni spaziali diverse su un layer, la mappa
    dello student viene interpolata bilinearmente alla risoluzione del teacher
    *prima* della normalizzazione, come prescritto dal paper.

    Args:
        student_feats: Dizionario layer_name → tensore (B, C, H, W) dello student.
        teacher_feats: Dizionario layer_name → tensore (B, C, H, W) del teacher.
        at_layers:     Nomi dei layer da includere nella loss.
        beta:          Peso del termine AT (paper: 1000).

    Returns:
        Tensore scalare — la AT loss (sullo stesso device delle feature student).
    """
    device = next(iter(student_feats.values())).device
    total = torch.zeros(1, device=device, dtype=torch.float32)

    for layer in at_layers:
        s_feat = student_feats[layer]
        t_feat = teacher_feats[layer]

        # Mappa teacher: somma quadrati canali → piatta
        t_map = attention_map(t_feat)

        # Mappa student: eventuale interpolazione spaziale prima di piattare
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            s_a_2d = s_feat.pow(2).sum(dim=1, keepdim=True)  # (B, 1, H_s, W_s)
            s_a_2d = F.interpolate(
                s_a_2d,
                size=t_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            s_map = s_a_2d.squeeze(1).view(s_a_2d.shape[0], -1)  # (B, H_t·W_t)
        else:
            s_map = attention_map(s_feat)

        # Normalizzazione L2 per campione
        s_norm = F.normalize(s_map, p=2, dim=1)
        t_norm = F.normalize(t_map, p=2, dim=1)

        # Distanza L2 quadratica per campione, media sul batch
        layer_loss = (s_norm - t_norm).pow(2).sum(dim=1).mean()
        total = total + layer_loss

    return (beta / 2.0) * total
