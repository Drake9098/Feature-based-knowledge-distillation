"""Salvataggio e caricamento checkpoint (pesi, ottimizzatore, metadati)."""

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: Path | str, state: Dict[str, Any]) -> None:
    """Salva ``state`` su disco con ``torch.save``.

    Crea le directory parent se mancanti. Convenzione tipica per ``state``:
    ``model_state_dict``, ``optimizer_state_dict``, ``epoch``, metriche (es. ``test_acc``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path | str,
    map_location: str | torch.device | None = None,
) -> Dict[str, Any]:
    """Carica un checkpoint salvato con :func:`save_checkpoint`.

    ``map_location`` consente di caricare su CPU (es. ``\"cpu\"``) da file creati su GPU.
    """
    kwargs: Dict[str, Any] = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
