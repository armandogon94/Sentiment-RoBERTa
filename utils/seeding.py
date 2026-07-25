"""One seed, threaded everywhere. Fixes the source notebook's biggest reproducibility hole.

The notebook passed ``random_state=42`` to two ``pandas.DataFrame.sample`` calls and stopped
there. ``DataLoader(shuffle=True)``, dropout masks, and the classification head's weight
initialisation were all unseeded, so two runs of the same code gave two different numbers
and neither could be reproduced.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every source of randomness this repo touches.

    Includes ``PYTHONHASHSEED`` (affects set iteration order, which reaches
    ``TfidfVectorizer`` vocabulary ordering) and ``torch.mps``, which has its own generator
    separate from the CPU one.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on this hardware
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=...)`` — derives each worker's seed from torch's."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int) -> torch.Generator:
    """A ``DataLoader(generator=...)`` so shuffle order is a function of the seed alone."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
