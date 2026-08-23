"""Feature extraction birim testleri (bilinen sentetik geometri ile).

Model/veri bağımsız test edilebilirlik bu modülün tasarım amacı: dummy maske ile
geometrinin doğruluğunu doğruluyoruz.
"""

import numpy as np
import cv2

from src.features.geometry import extract_features, largest_component


def _vertical_runway_mask(size: int = 400) -> np.ndarray:
    """Merkezde dikey dikdörtgen maske (uzun eksen dikey)."""
    m = np.zeros((size, size), np.uint8)
    cv2.rectangle(m, (170, 80), (230, 320), 255, -1)  # genişlik 60, boy 240
    return m


def test_vertical_runway_angle_near_zero():
    feat = extract_features(_vertical_runway_mask())
    assert feat.valid
    assert feat.corners.shape == (4, 2)
    # Dikey pist -> yaklaşma açısı ~0
    assert abs(feat.approach_angle_deg) < 3.0


def test_center_line_endpoints_are_vertical():
    feat = extract_features(_vertical_runway_mask())
    far, near = feat.center_line
    # near uç görüntüde daha aşağıda (y büyük) olmalı
    assert near[1] > far[1]
    # x'ler birbirine yakın (dikey hat)
    assert abs(near[0] - far[0]) < 5.0


def test_tilted_runway_has_signed_angle():
    m = np.zeros((400, 400), np.uint8)
    box = cv2.boxPoints(((200, 200), (60, 220), 20)).astype(np.int32)
    cv2.fillPoly(m, [box], 255)
    feat = extract_features(m)
    assert feat.valid
    # Eğik pist -> açı sıfırdan belirgin şekilde farklı
    assert abs(feat.approach_angle_deg) > 5.0


def test_threshold_edge_at_bottom():
    feat = extract_features(_vertical_runway_mask())
    (x1, y1), (x2, y2) = feat.threshold_edge
    # Threshold en yakın (alt) kenar: y'ler görüntünün altına yakın
    assert y1 > 250 and y2 > 250
    # Yatay bir kenar (sol-sağ uçlar)
    assert x2 > x1


def test_empty_mask_graceful():
    feat = extract_features(np.zeros((100, 100), np.uint8))
    assert feat.valid is False
    assert feat.reason == "bos_maske"


def test_tiny_noise_rejected():
    m = np.zeros((100, 100), np.uint8)
    m[10:13, 10:13] = 255  # 9 piksel, eşiğin altında
    feat = extract_features(m)
    assert feat.valid is False


def test_largest_component_selected():
    """İki bileşen varsa büyük olan seçilmeli (küçük gürültü elenmeli)."""
    m = np.zeros((400, 400), np.uint8)
    cv2.rectangle(m, (170, 80), (230, 320), 255, -1)  # büyük pist
    cv2.rectangle(m, (10, 10), (25, 25), 255, -1)      # küçük gürültü
    comp = largest_component(m)
    # Seçilen bileşen büyük dikdörtgenin alanına yakın olmalı (~60*240)
    assert 12000 < int((comp > 0).sum()) < 16000


def test_none_mask_graceful():
    feat = extract_features(None)
    assert feat.valid is False
