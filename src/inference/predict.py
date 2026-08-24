"""Inference pipeline — görüntü → maske → geometri → görselleştirme.

Katman sorumluluğu: eğitilmiş modeli yükle, yeni görüntü(ler) için pist maskesi üret,
maskeyi temizle (post-process), geometrik özellikleri çıkar ve görseli üret.

Tekil ve toplu (klasör) çalışmayı destekler. Desteklenmeyen formatta anlamlı hata verir.

Çalıştırma:
    python -m src.inference.predict --checkpoint outputs/best.pt --image path/img.jpg
    python -m src.inference.predict --checkpoint outputs/best.pt --input-dir imgs/ --output outputs/preds
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.transforms import get_eval_transforms
from src.features.geometry import RunwayFeatures, extract_features
from src.features.visualize import draw_features
from src.inference.postprocess import clean_mask
from src.models.factory import create_model

SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}


@dataclass
class PredictionResult:
    """Tek bir görüntünün inference çıktısı."""
    mask: np.ndarray            # (H,W) uint8 0/255 — orijinal çözünürlükte
    features: RunwayFeatures
    overlay: np.ndarray         # (H,W,3) RGB — maske + geometri çizili


class RunwayPredictor:
    """Eğitilmiş checkpoint'i yükleyip uçtan uca tahmin yapar."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.config = ckpt["config"]
        self.image_size = self.config["data"]["image_size"]
        self.model = create_model(self.config["model"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.transform = get_eval_transforms(self.image_size)

    @torch.no_grad()
    def _predict_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Görüntüden ham binary maske (orijinal çözünürlükte) üretir."""
        h, w = image_rgb.shape[:2]
        tensor = self.transform(image=image_rgb)["image"].unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask_small = (prob > 0.5).astype(np.uint8) * 255
        # Model çıktısını (image_size) orijinal çözünürlüğe geri ölçekle (nearest)
        return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    def predict(self, image_rgb: np.ndarray) -> PredictionResult:
        """Tek görüntü için tam pipeline: maske → temizle → geometri → overlay."""
        raw_mask = self._predict_mask(image_rgb)
        mask = clean_mask(raw_mask)
        feats = extract_features(mask)
        overlay = draw_features(image_rgb, feats)
        return PredictionResult(mask=mask, features=feats, overlay=overlay)

    def predict_path(self, path: str | Path) -> PredictionResult:
        """Dosya yolundan tahmin (format doğrulamalı)."""
        path = Path(path)
        if path.suffix.lower() not in SUPPORTED_EXT:
            raise ValueError(f"Desteklenmeyen format: {path.suffix}. "
                             f"Desteklenen: {sorted(SUPPORTED_EXT)}")
        image_rgb = np.array(Image.open(path).convert("RGB"))
        return self.predict(image_rgb)


def _iter_images(input_dir: Path):
    """Klasördeki desteklenen görüntüleri sırayla verir."""
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXT:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Pist segmentasyonu + geometri inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, help="tekil görüntü yolu")
    parser.add_argument("--input-dir", type=str, help="toplu: klasör yolu")
    parser.add_argument("--output", type=str, default="outputs/preds",
                        help="çıktı klasörü (overlay + maske)")
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("--image veya --input-dir vermelisin")

    predictor = RunwayPredictor(args.checkpoint)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(args.image)] if args.image else list(_iter_images(Path(args.input_dir)))
    if not paths:
        print("İşlenecek görüntü bulunamadı.")
        return

    for p in paths:
        try:
            res = predictor.predict_path(p)
        except ValueError as e:
            print(f"ATLANDI {p.name}: {e}")
            continue
        stem = p.stem
        Image.fromarray(res.overlay).save(out_dir / f"{stem}_overlay.png")
        Image.fromarray(res.mask).save(out_dir / f"{stem}_mask.png")
        f = res.features
        aci = f"{f.approach_angle_deg:.1f}°" if f.valid else "yok"
        print(f"{p.name}: valid={f.valid} açı={aci} -> {stem}_overlay.png")


if __name__ == "__main__":
    main()
