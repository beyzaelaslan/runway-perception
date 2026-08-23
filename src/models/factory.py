"""Model factory — segmentation_models_pytorch U-Net.

Model seçimi config'ten okunur. Varsayılan: U-Net + ResNet34 (ImageNet pretrained).

Gerekçe (DEVLOG'da detaylı):
- Küçük veri setinde (800 görüntü) pretrained encoder şart → sıfırdan eğitim yakınsamaz.
- U-Net binary segmentasyonda hızlı yakınsar, skip-connection ince yapıları (pist kenarı)
  korur.
- ResNet34 hafif → Colab ücretsiz GPU'da makul sürede eğitilir.
"""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch.nn as nn


def create_model(model_cfg: dict[str, Any]) -> nn.Module:
    """Config'ten smp segmentasyon modeli üretir.

    Args:
        model_cfg: config'in `model` bölümü (arch, encoder_name, ... anahtarları).

    Returns:
        torch.nn.Module — logits üreten segmentasyon modeli (aktivasyon yok).
    """
    arch = model_cfg.get("arch", "Unet")
    model_fn = getattr(smp, arch)  # smp.Unet, smp.DeepLabV3Plus, ...
    return model_fn(
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 3),
        classes=model_cfg.get("classes", 1),
    )
