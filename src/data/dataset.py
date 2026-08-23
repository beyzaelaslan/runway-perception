"""LARD PyTorch Dataset.

Sorumluluğu: labels.csv'yi split'e göre filtrele, her örnek için görüntüyü yükle,
köşe etiketinden binary maskeyi üret (masks.py) ve transform uygula.

Maske burada, model bağımsız `masks.py` ile üretilir; böylece veri katmanı ile
feature/model katmanları birbirinden ayrık kalır.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.masks import row_to_mask


class LardDataset(Dataset):
    """LARD subset segmentasyon veri seti (görüntü + binary pist maskesi)."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        transform=None,
    ) -> None:
        """
        Args:
            data_dir: `images/` ve `labels.csv` içeren kök dizin (ör. data/lard).
            split: "train" | "val" | "test" (labels.csv'deki split sütununa göre filtre).
            transform: albumentations Compose (image+mask alır). None ise ham numpy döner.
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.transform = transform

        df = pd.read_csv(self.data_dir / "labels.csv")
        if "split" not in df.columns:
            raise ValueError("labels.csv'de 'split' sütunu yok. Önce src.data.split çalıştır.")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"'{split}' split'i boş.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx].to_dict()
        image = np.array(Image.open(self.image_dir / row["filename"]).convert("RGB"))
        mask = row_to_mask(row, int(row["height"]), int(row["width"]))  # 0/255 uint8

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
            # mask: (H,W) 0/255 -> (1,H,W) float {0,1}
            mask = (mask > 127).float().unsqueeze(0)
        else:
            mask = (mask > 127).astype(np.float32)

        return image, mask
