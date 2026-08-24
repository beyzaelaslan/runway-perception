"""Test setinde değerlendirme + feature doğrulama + failure analizi.

Üç iş yapar (case study'nin TESTING sorularına doğrudan cevap):
  1. Segmentasyon metrikleri (IoU/Dice/precision/recall/F1) — test setinde toplu.
  2. Feature doğrulama: GT maskeden vs tahmin maskeden çıkarılan geometriyi karşılaştırır
     (yaklaşma açısı hatası, derece cinsinden). "Özellikleri nasıl doğruladın" sorusunun cevabı.
  3. Failure analizi: en kötü N tahmini görsel olarak outputs/failures/ altına kaydeder,
     ayrıca yanlış pozitif/negatif eğilimini raporlar.

Çalıştırma:
    python -m src.training.evaluate --checkpoint outputs/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.masks import row_to_mask
from src.features.geometry import extract_features
from src.features.visualize import draw_features
from src.inference.predict import RunwayPredictor
from src.training.metrics import SegmentationMetrics


def _sample_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Tek görüntü için IoU (failure sıralaması için)."""
    p, g = pred > 0, gt > 0
    inter = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    return inter / union if union > 0 else 1.0  # ikisi de boşsa mükemmel


def evaluate(checkpoint: str, data_dir: str, worst_n: int) -> None:
    predictor = RunwayPredictor(checkpoint)
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "labels.csv")
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test görüntü sayısı: {len(test)}")

    metrics = SegmentationMetrics()
    per_image = []      # (filename, iou, angle_err, pred_valid, gt_valid, fp, fn)
    angle_errors = []

    for _, row in test.iterrows():
        r = row.to_dict()
        image = np.array(Image.open(data_dir / "images" / r["filename"]).convert("RGB"))
        h, w = int(r["height"]), int(r["width"])
        gt_mask = row_to_mask(r, h, w)

        res = predictor.predict(image)
        pred_mask = res.mask

        # Metrik biriktir (tensör bekliyor: (1,1,H,W))
        pt = torch.from_numpy((pred_mask > 0).astype("float32"))[None, None]
        gtt = torch.from_numpy((gt_mask > 0).astype("float32"))[None, None]
        # logits yerine doğrudan olasılık gibi davranması için büyük/küçük değer:
        metrics.update((pt * 10 - 5), gtt)

        iou = _sample_iou(pred_mask, gt_mask)

        # Feature doğrulama: GT vs tahmin açı hatası
        gt_feat = extract_features(gt_mask)
        angle_err = None
        if gt_feat.valid and res.features.valid:
            angle_err = abs(gt_feat.approach_angle_deg - res.features.approach_angle_deg)
            angle_errors.append(angle_err)

        # FP/FN eğilimi (piksel)
        fp = int(np.logical_and(pred_mask > 0, gt_mask == 0).sum())
        fn = int(np.logical_and(pred_mask == 0, gt_mask > 0).sum())
        per_image.append({"filename": r["filename"], "iou": iou,
                          "angle_err": angle_err, "pred_valid": res.features.valid,
                          "gt_valid": gt_feat.valid, "fp": fp, "fn": fn,
                          "slant_distance": r.get("slant_distance")})

    seg = metrics.compute()
    print("\n=== Segmentasyon metrikleri (test) ===")
    for k, v in seg.items():
        print(f"  {k:10s}: {v:.4f}")

    if angle_errors:
        ae = np.array(angle_errors)
        print("\n=== Feature doğrulama (yaklaşma açısı, GT vs tahmin) ===")
        print(f"  ortalama açı hatası: {ae.mean():.2f}°  medyan: {np.median(ae):.2f}°  "
              f"(n={len(ae)})")

    # Failure analizi: en düşük IoU'lu N görüntüyü kaydet
    pi = pd.DataFrame(per_image).sort_values("iou")
    fail_dir = Path("outputs/failures")
    fail_dir.mkdir(parents=True, exist_ok=True)
    for _, row in pi.head(worst_n).iterrows():
        r = test[test["filename"] == row["filename"]].iloc[0].to_dict()
        image = np.array(Image.open(data_dir / "images" / r["filename"]).convert("RGB"))
        res = predictor.predict(image)
        Image.fromarray(draw_features(image, res.features)).save(
            fail_dir / f"iou{row['iou']:.2f}_{row['filename']}.png")

    # Özet FP/FN eğilimi
    tot_fp = int(pi["fp"].sum())
    tot_fn = int(pi["fn"].sum())
    print(f"\n=== Hata eğilimi ===")
    print(f"  toplam yanlış pozitif piksel: {tot_fp:,}")
    print(f"  toplam yanlış negatif piksel: {tot_fn:,}")
    print(f"  baskın hata: {'yanlış negatif (pist kaçırma)' if tot_fn > tot_fp else 'yanlış pozitif (fazla pist)'}")
    print(f"  en kötü {worst_n} tahmin -> {fail_dir}/")

    # Rapor JSON (TESTING.md'ye taşımak için)
    report = {"segmentation": seg,
              "angle_error_mean": float(np.mean(angle_errors)) if angle_errors else None,
              "angle_error_median": float(np.median(angle_errors)) if angle_errors else None,
              "total_fp": tot_fp, "total_fn": tot_fn, "n_test": len(test)}
    with open("outputs/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nRapor: outputs/eval_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test değerlendirme + failure analizi")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/lard")
    parser.add_argument("--worst-n", type=int, default=12)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.worst_n)


if __name__ == "__main__":
    main()
