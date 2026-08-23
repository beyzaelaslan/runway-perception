"""Segmentasyon metrikleri — IoU, Dice, precision, recall, F1.

Neden bu metrikler (case study "neden bu metrikler" diye soruyor):
- **IoU / Dice:** örtüşme temelli, segmentasyonun standart ölçütü. Accuracy yanıltıcı
  olurdu çünkü pist görüntünün ~%0.17'si → her şeyi "arka plan" diyen model %99+ accuracy
  alır ama işe yaramaz. IoU/Dice bu tuzağa düşmez.
- **Precision / Recall:** hata tipini ayırır — precision düşük = yanlış pozitif (taxiway'i
  pist sanma), recall düşük = yanlış negatif (pisti kaçırma). Failure analizinin temeli.
- **F1:** precision/recall dengesi (binary'de Dice'a eşittir; ikisini de raporluyoruz).

Metrikler piksel bazında TP/FP/FN/TN sayımıyla, tüm veri seti üzerinde biriktirilerek
hesaplanır (görüntü başına ortalamadan daha stabil, küçük maskelerde yanlılık yaratmaz).
"""

from __future__ import annotations

import torch

EPS = 1e-7


class SegmentationMetrics:
    """Piksel bazlı TP/FP/FN/TN biriktirici; sonunda metrikleri hesaplar."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Bir batch'in tahminlerini biriktirir.

        Args:
            logits: (B,1,H,W) model çıktısı (aktivasyonsuz).
            target: (B,1,H,W) {0,1} float ground truth.
        """
        pred = (torch.sigmoid(logits) > self.threshold).float()
        t = (target > 0.5).float()
        self.tp += float((pred * t).sum())
        self.fp += float((pred * (1 - t)).sum())
        self.fn += float(((1 - pred) * t).sum())
        self.tn += float(((1 - pred) * (1 - t)).sum())

    def compute(self) -> dict[str, float]:
        """Biriken sayımlardan metrikleri hesaplar."""
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = tp / (tp + fp + fn + EPS)
        dice = 2 * tp / (2 * tp + fp + fn + EPS)
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)
        return {"iou": iou, "dice": dice, "precision": precision,
                "recall": recall, "f1": f1}
