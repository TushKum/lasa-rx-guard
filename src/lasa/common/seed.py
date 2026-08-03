"""Global determinism helpers.

Every entrypoint in this project calls :func:`set_seed` before touching data or
models. The goal is that two runs of the same script with the same config
produce byte-identical checkpoints on the same hardware.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 1337


def set_seed(seed: int = DEFAULT_SEED, deterministic_torch: bool = True) -> int:
    """Seed python, numpy and (if installed) torch.

    ``deterministic_torch`` trades a little speed for exact reproducibility of
    cuDNN kernels. Keep it on for the reported results; turn it off only if a
    run is too slow to fit the 2h T4 budget.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # data-prep only environments
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding — without this, augmentation RNG is shared."""
    import torch

    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
