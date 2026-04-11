"""Metriche richieste dalla Fase 1: accuracy, dimensione modello, latenza."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def accuracy_percent(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Accuracy top-1 in percentuale [0, 100] sul dataloader indicato."""
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += labels.size(0)
    if total == 0:
        return 0.0
    return 100.0 * correct / total


def model_size_mb(state_dict_or_model: Any) -> float:
    """Somma dei byte dei tensori nello state_dict, espressa in MiB (2^20).

    Accetta ``nn.Module`` o un ``dict`` compatibile con ``state_dict`` (valori tensori).
    """
    if isinstance(state_dict_or_model, nn.Module):
        sd = state_dict_or_model.state_dict()
    elif isinstance(state_dict_or_model, dict):
        sd = state_dict_or_model
    else:
        raise TypeError("Atteso nn.Module o dict (state_dict).")

    total_bytes = 0
    for v in sd.values():
        if isinstance(v, torch.Tensor):
            total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024**2)


def _next_batch(loader: DataLoader, iterator_holder: list[Iterator[Tuple[torch.Tensor, ...]]]) -> torch.Tensor:
    """Restituisce il tensore di input del batch successivo, ricreando l'iteratore se finisce."""
    it = iterator_holder[0]
    try:
        batch = next(it)
    except StopIteration:
        iterator_holder[0] = iter(loader)
        batch = next(iterator_holder[0])
    inputs, _ = batch
    return inputs


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inference_latency_ms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg_metrics: Dict[str, Any],
) -> float:
    """Millisecondi medi per immagine (forward), sul device indicato.

    - Include trasferimento host→device del batch nel loop (``to(device)``).
    - Su CUDA usa ``synchronize`` per tempi di GPU affidabili.
    - Warmup: primi ``latency_warmup_batches`` batch non misurati.
    - Misura: ``latency_num_batches`` batch consecutivi; tempo totale / numero di immagini.
    """
    warmup = int(cfg_metrics["latency_warmup_batches"])
    num_batches = int(cfg_metrics["latency_num_batches"])

    model.eval()
    iterator_holder: list[Iterator[Tuple[torch.Tensor, ...]]] = [iter(loader)]

    with torch.inference_mode():
        for _ in range(warmup):
            x = _next_batch(loader, iterator_holder).to(device, non_blocking=True)
            model(x)
            _maybe_sync(device)

        _maybe_sync(device)
        t0 = time.perf_counter()
        n_images = 0
        for _ in range(num_batches):
            x = _next_batch(loader, iterator_holder).to(device, non_blocking=True)
            model(x)
            n_images += x.size(0)
        _maybe_sync(device)
        t1 = time.perf_counter()

    if n_images == 0:
        return 0.0
    elapsed_ms = (t1 - t0) * 1000.0
    return elapsed_ms / n_images
