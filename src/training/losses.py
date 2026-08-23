"""Kayıp fonksiyonu — Dice + BCE kombinasyonu.

Neden kombinasyon:
- **BCE** piksel bazlı, stabil gradyan verir ama aşırı sınıf dengesizliğinde (pist
  görüntünün ~%0.17'si, bkz. Faz 1 bulgusu) arka plana yığılır.
- **Dice** doğrudan örtüşmeyi (IoU'ya yakın) optimize eder, dengesizliğe dayanıklıdır ama
  tek başına gradyanları gürültülü olabilir.
- İkisini birleştirmek: BCE stabilite + Dice dengesizlik direnci.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Ağırlıklı Dice + BCE. Girdi logits (aktivasyonsuz), hedef {0,1} float."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + \
            self.bce_weight * self.bce(logits, target)
