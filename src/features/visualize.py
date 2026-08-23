"""Çıkarılan geometrik özellikleri görüntü üzerine çizer.

geometry.py'den bağımsız bir sunum katmanı: RunwayFeatures alır, RGB görüntü üzerine
pist sınırları, merkez hattı, threshold kenarı ve açı bilgisini çizer.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.features.geometry import RunwayFeatures

# BGR değil, RGB varsayıyoruz (matplotlib/PIL ile uyumlu). Renkler (R,G,B).
COLOR_BOX = (255, 235, 59)      # sarı — pist sınırları
COLOR_CENTER = (0, 229, 255)    # camgöbeği — merkez hattı
COLOR_THRESH = (255, 61, 61)    # kırmızı — threshold (en yakın kenar)
COLOR_TEXT = (255, 255, 255)


def draw_features(
    image: np.ndarray,
    feat: RunwayFeatures,
    thickness: int = 2,
) -> np.ndarray:
    """Özellikleri görüntü kopyası üzerine çizer ve döndürür.

    Args:
        image: (H,W,3) RGB görüntü.
        feat: extract_features çıktısı.
        thickness: çizgi kalınlığı.

    Returns:
        Üzerine çizim yapılmış (H,W,3) RGB görüntü. Özellik geçersizse görüntüye
        "pist bulunamadı" notu yazılır (crash yok).
    """
    canvas = image.copy()

    if not feat.valid:
        cv2.putText(canvas, f"pist bulunamadi ({feat.reason})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_THRESH, 2)
        return canvas

    # Pist sınırları (dörtgen)
    box = feat.corners.astype(np.int32)
    cv2.polylines(canvas, [box], isClosed=True, color=COLOR_BOX, thickness=thickness)

    # Merkez hattı
    far, near = feat.center_line
    cv2.line(canvas, _pt(far), _pt(near), COLOR_CENTER, thickness)

    # Threshold kenarı (en yakın kenar) — kalın çiz
    t1, t2 = feat.threshold_edge
    cv2.line(canvas, _pt(t1), _pt(t2), COLOR_THRESH, thickness + 1)

    # Açı metni
    cv2.putText(canvas, f"aci: {feat.approach_angle_deg:.1f} derece", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

    return canvas


def _pt(p: tuple) -> tuple[int, int]:
    """Float nokta -> int piksel koordinatı."""
    return (int(round(p[0])), int(round(p[1])))
