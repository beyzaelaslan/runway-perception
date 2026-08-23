"""Tekrar üretilebilirlik: rastgelelik kaynaklarını sabitler.

Case study "eğitim süreci tekrar üretilebilir olmalı" diyor. Tek çağrıda tüm
kütüphanelerin seed'ini sabitliyoruz.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """random, numpy ve torch için seed sabitler (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinizm için cudnn ayarları (küçük hız kaybı, tekrar üretilebilirlik kazancı)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
