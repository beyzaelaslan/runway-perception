"""Augmentation ve normalizasyon pipeline'ları (albumentations).

Augmentation seçimi pist geometrisini koruyacak şekilde:
- horizontal flip: pist simetrik, güvenli.
- brightness/contrast: farklı aydınlatma/hava koşullarına dayanıklılık.
- HAFİF rotation (±birkaç derece): kameranın küçük roll açısını taklit eder;
  büyük rotation yaklaşma geometrisini bozardı, o yüzden dar tutuldu.

Normalizasyon ImageNet istatistikleriyle: encoder (ResNet34) ImageNet pretrained.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet ortalama/std — pretrained encoder ile uyum için.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int,
    hflip: float = 0.5,
    brightness_contrast: float = 0.3,
    rotate_limit: int = 7,
) -> A.Compose:
    """Eğitim augmentation'ı (görüntü + maske birlikte dönüştürülür)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=hflip),
        A.RandomBrightnessContrast(p=brightness_contrast),
        A.Rotate(limit=rotate_limit, border_mode=0, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int) -> A.Compose:
    """Validation/test transform — augmentation yok, sadece resize + normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
