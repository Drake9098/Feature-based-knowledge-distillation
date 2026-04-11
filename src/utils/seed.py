"""Reproducibilità."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fissa seed per random, NumPy e PyTorch (CPU/GPU).

    # TODO (opzionale): per massima riproducibilità su GPU, valuta
    #   torch.backends.cudnn.deterministic = True
    #   torch.backends.cudnn.benchmark = False
    #   Nota: può rallentare il training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
