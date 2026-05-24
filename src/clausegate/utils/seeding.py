"""Deterministic seeding across numpy + python random.

We never touch torch or cuda from this module to avoid pulling in
heavy deps unless the caller is in the training path. The training
script seeds torch and pycuda explicitly.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_all(seed: int) -> None:
    """Seed numpy + python random + PYTHONHASHSEED.

    Idempotent. Callers that need torch/cuda should additionally
    call torch.manual_seed and torch.cuda.manual_seed_all.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
