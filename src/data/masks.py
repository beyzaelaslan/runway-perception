"""Köşe koordinatlarından binary segmentasyon maskesi üretimi.

LARD hazır maske vermez; pistin 4 köşesini piksel koordinatı olarak verir. Segmentasyon
eğitimi için ground-truth maskeye ihtiyacımız var, o yüzden 4 köşeyi bir dörtgen (polygon)
kabul edip içini dolduruyoruz.

Bu modül modelden ve veri setinden bağımsızdır: girdi = 4 köşe, çıktı = binary maske.
"""

from __future__ import annotations

import numpy as np
import cv2

# labels.csv'deki köşe sütunları. Sıra ÖNEMLİ: dörtgeni saat yönünde dolaşmalı
# (TL -> TR -> BR -> BL), aksi halde polygon kendini keser ve maske bozulur.
CORNER_ORDER = ["TL", "TR", "BR", "BL"]


def corners_to_polygon(row: dict) -> np.ndarray | None:
    """labels.csv satırındaki köşe sütunlarını (N,2) polygon dizisine çevirir.

    Args:
        row: `x_TL, y_TL, ...` anahtarlarını içeren sözlük (CSV satırı).

    Returns:
        (4, 2) int32 dizi [ [x,y], ... ] saat yönünde; köşeler eksik/geçersizse None.
    """
    pts = []
    for c in CORNER_ORDER:
        x, y = row.get(f"x_{c}"), row.get(f"y_{c}")
        if x is None or y is None or x == "" or y == "":
            return None
        try:
            pts.append([int(float(x)), int(float(y))])
        except (ValueError, TypeError):
            return None
    return np.array(pts, dtype=np.int32)


def polygon_to_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    """Dörtgeni doldurarak binary maske üretir.

    Args:
        polygon: (4, 2) köşe dizisi.
        height: Maske yüksekliği (görüntüyle aynı olmalı).
        width: Maske genişliği.

    Returns:
        (H, W) uint8 maske; pist=255, arka plan=0.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=255)
    return mask


def row_to_mask(row: dict, height: int, width: int) -> np.ndarray:
    """CSV satırından doğrudan binary maske üretir (köşe yoksa boş maske döner).

    Boş maske dönmesi bilinçli: pist görünmüyorsa (köşe etiketi yoksa) crash etmek
    yerine tamamen arka plan maske veriyoruz; pipeline kesintisiz akıyor.
    """
    polygon = corners_to_polygon(row)
    if polygon is None:
        return np.zeros((height, width), dtype=np.uint8)
    return polygon_to_mask(polygon, height, width)
