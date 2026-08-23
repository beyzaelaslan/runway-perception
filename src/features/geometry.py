"""Pist geometrik özellik çıkarımı (saf OpenCV, modelden bağımsız).

Girdi: binary maske (pist=nonzero). Çıktı: geometrik özellikler.
Bu modülün model veya veri setiyle hiçbir bağı yoktur — dummy maske ile test edilebilir.
Bu ayrım bilinçli: model zayıf çıksa bile bu katman doğrulanabilir ve savunulabilir kalır.

Çıkarılan özellikler:
  1. Pist sınırları  — cv2.minAreaRect ile 4 köşe (döndürülmüş dikdörtgen).
  2. Merkez hattı    — kısa kenarların orta noktalarını birleştiren doğru (pistin uzun ekseni).
  3. Yaklaşma açısı  — merkez hattının görüntü dikeyiyle yaptığı işaretli açı (hizalama göstergesi).
  4. Threshold kenarı (opsiyonel) — pistin görüntüye (kameraya) en yakın kısa kenarı.

Tüm çıktılar boş/parçalı maskede graceful (crash yok): valid=False döner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

# En büyük bileşen bu piksel sayısının altındaysa güvenilir geometri üretmeyiz.
MIN_COMPONENT_AREA = 20


@dataclass
class RunwayFeatures:
    """Çıkarılan geometrik özellikler. Geçersizse valid=False ve alanlar None."""
    valid: bool
    corners: np.ndarray | None = None          # (4,2) float — minAreaRect köşeleri
    center_line: tuple | None = None           # ((xf,yf),(xn,yn)) — (uzak uç, yakın uç)
    approach_angle_deg: float | None = None     # dikeyle işaretli açı (0 = hizalı)
    threshold_edge: tuple | None = None         # ((x1,y1),(x2,y2)) — en yakın kenar
    area_px: int = 0                            # en büyük bileşenin alanı
    reason: str = ""                            # geçersizse sebep


def largest_component(mask: np.ndarray) -> np.ndarray | None:
    """Maskenin en büyük bağlı bileşenini döndürür (gürültü/küçük parçaları eler).

    Args:
        mask: (H,W) binary maske (nonzero = ön plan).

    Returns:
        Sadece en büyük bileşeni içeren (H,W) uint8 maske; bileşen yoksa None.
    """
    binary = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:  # sadece arka plan
        return None
    # 0. etiket arka plan; en büyük alanlı ön plan bileşenini seç
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def _row_centroid_line(comp: np.ndarray) -> tuple[tuple, tuple] | None:
    """Her görüntü satırının maske orta noktalarına doğru fit eder → merkez hattı.

    minAreaRect'in "uzun eksen = pist boyu" varsayımı çok yakın mesafede kırılıyordu
    (pist enine boyundan geniş olunca eksen yatay seçiliyordu). Bunun yerine, pistin
    down-range ekseni görüntüde ~dikey olduğundan her satırın (y) maske x-ortalamasını
    alıp bunlara bir doğru (x = a·y + b) fit ediyoruz. En-boy oranından bağımsız.

    Returns:
        (far, near) uç noktaları — far=üstteki (uzak), near=alttaki (yakın); yetersizse None.
    """
    ys, xs = np.nonzero(comp)
    if ys.size == 0:
        return None
    y_min, y_max = int(ys.min()), int(ys.max())
    if y_max - y_min < 2:  # dikey yayılım yok, doğru fit edilemez
        return None

    # Satır başına x-ortalaması (vektörel): bincount ile
    sum_x = np.bincount(ys, weights=xs)
    cnt = np.bincount(ys)
    valid = cnt > 0
    row_y = np.nonzero(valid)[0].astype(np.float64)
    row_x = sum_x[valid] / cnt[valid]

    a, b = np.polyfit(row_y, row_x, 1)  # x = a·y + b
    far = (float(a * y_min + b), float(y_min))
    near = (float(a * y_max + b), float(y_max))
    return far, near


def _near_edge(comp: np.ndarray, band_frac: float = 0.05) -> tuple[tuple, tuple] | None:
    """Pistin kameraya en yakın kenarı (threshold): alt banttaki yatay yayılım.

    Alt %band_frac satırlardaki ön plan piksellerinin en sol/en sağ x'i.
    """
    ys, xs = np.nonzero(comp)
    if ys.size == 0:
        return None
    y_max = int(ys.max())
    y_min = int(ys.min())
    band = max(1, int((y_max - y_min) * band_frac))
    sel = ys >= (y_max - band)
    if not sel.any():
        return None
    x_left, x_right = int(xs[sel].min()), int(xs[sel].max())
    return (float(x_left), float(y_max)), (float(x_right), float(y_max))


def _angle_from_vertical(far: tuple, near: tuple) -> float:
    """Merkez hattının görüntü dikeyiyle yaptığı işaretli açıyı (derece) verir.

    0° = pist tam dikey (kameraya hizalı). Pozitif = yakın uç sağa kayık.
    """
    dx = near[0] - far[0]
    dy = near[1] - far[1]  # görüntüde y aşağı; yakın uç aşağıda (dy>0)
    return math.degrees(math.atan2(dx, dy))


def extract_features(mask: np.ndarray) -> RunwayFeatures:
    """Binary maskeden pist geometrik özelliklerini çıkarır (graceful).

    Args:
        mask: (H,W) binary maske (nonzero = pist).

    Returns:
        RunwayFeatures — maske boş/parçalıysa valid=False.
    """
    if mask is None or mask.size == 0 or int((mask > 0).sum()) == 0:
        return RunwayFeatures(valid=False, reason="bos_maske")

    comp = largest_component(mask)
    if comp is None:
        return RunwayFeatures(valid=False, reason="bilesen_yok")

    area = int((comp > 0).sum())
    if area < MIN_COMPONENT_AREA:
        return RunwayFeatures(valid=False, area_px=area, reason="bilesen_cok_kucuk")

    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return RunwayFeatures(valid=False, area_px=area, reason="kontur_yok")
    contour = max(contours, key=cv2.contourArea)

    # --- Özellik 1: pist sınırları (döndürülmüş dikdörtgen 4 köşe) ---
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)  # (4,2) float

    # --- Özellik 2: merkez hattı (satır-bazlı orta nokta fit) ---
    line = _row_centroid_line(comp)
    if line is None:
        return RunwayFeatures(valid=False, corners=box, area_px=area,
                              reason="merkez_hatti_fit_edilemedi")
    far, near = line

    # --- Özellik 3: yaklaşma ekseni açısı (dikeyden sapma) ---
    angle = _angle_from_vertical(far, near)

    # --- Özellik 4: threshold kenarı (kameraya en yakın kenar, alt bant) ---
    threshold_edge = _near_edge(comp)

    return RunwayFeatures(
        valid=True,
        corners=box,
        center_line=(far, near),
        approach_angle_deg=angle,
        threshold_edge=threshold_edge,
        area_px=area,
    )
