"""Maske post-processing — morfolojik temizleme + küçük bileşen eleme.

Ham model çıktısı genelde küçük gürültü pikselleri ve delikler içerir. Geometri çıkarımı
tek, temiz bir bileşen beklediğinden burada:
  1. Morfolojik açma (opening) → küçük gürültüyü siler,
  2. Morfolojik kapama (closing) → maske içindeki küçük delikleri doldurur,
  3. Sadece en büyük bağlı bileşeni tutar → taxiway/yansıma gibi ikincil parçaları eler.
"""

from __future__ import annotations

import cv2
import numpy as np


def clean_mask(mask: np.ndarray, kernel_size: int = 5, keep_largest: bool = True) -> np.ndarray:
    """Binary maskeyi morfolojik olarak temizler ve en büyük bileşeni tutar.

    Args:
        mask: (H,W) binary maske (nonzero = ön plan).
        kernel_size: morfoloji çekirdek boyutu (tek sayı).
        keep_largest: True ise sadece en büyük bağlı bileşen tutulur.

    Returns:
        (H,W) uint8 temizlenmiş maske (0/255).
    """
    binary = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if keep_largest:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8), connectivity=8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = (labels == largest).astype(np.uint8) * 255

    return binary
