# ARCHITECTURE — Runway Perception System

Sistem, hava aracının yaklaşma/iniş fazında ön görüş kamerasından gelen görüntülerde
pisti segmente eder ve pistin geometrik özelliklerini çıkarır. Tasarımın merkezinde
**katman ayrımı** vardır: her katman bağımsız test edilebilir ve bir katman zayıf olsa
bile diğerleri savunulabilir kalır.

## Veri akışı

```
              ┌──────────────┐
   LARD V2 ─► │  DATA        │  download_lard.py → images + labels.csv (4 köşe)
              │  (src/data)  │  masks.py         → köşe → binary maske
              │              │  split.py         → scenario bazlı train/val/test
              │              │  dataset.py       → PyTorch Dataset (+transforms)
              └──────┬───────┘
                     │ (görüntü, maske)
              ┌──────▼───────┐
              │  TRAINING    │  factory.py  → U-Net + ResNet34
              │ (src/training)│ losses.py   → Dice + BCE
              │              │  metrics.py  → IoU/Dice/precision/recall/F1
              │              │  train.py    → döngü, checkpoint (Colab GPU)
              └──────┬───────┘
                     │ best.pt
              ┌──────▼───────┐
              │  INFERENCE   │  predict.py     → görüntü → maske
              │ (src/inference)│ postprocess.py → morfoloji + en büyük bileşen
              └──────┬───────┘
                     │ temiz binary maske
              ┌──────▼───────┐
              │  FEATURES    │  geometry.py  → 4 köşe, merkez hattı, açı, threshold
              │ (src/features)│ visualize.py → görüntü üzerine çizim
              └──────┬───────┘
                     │ geometri + overlay
              ┌──────▼───────┐
              │ PRESENTATION │  app/main.py  → Streamlit (yükle → sonuç)
              │              │  predict.py   → CLI (tekil/toplu)
              └──────────────┘

  Değerlendirme (dikey kesit): src/training/evaluate.py
  → test metrikleri + feature doğrulama (GT vs tahmin) + failure analizi
```

## Katmanlar ve sorumlulukları

| Katman | Konum | Girdi → Çıktı | Sorumluluk |
|--------|-------|---------------|------------|
| **Veri hazırlama** | `src/data/` | LARD → (görüntü, maske) | İndirme, köşe→maske, split, Dataset, augmentation |
| **Eğitim** | `src/training/` | (görüntü, maske) → `best.pt` | Model, loss, metrik, tekrar üretilebilir döngü |
| **Inference** | `src/inference/` | görüntü → temiz maske | Model servisi + post-processing |
| **Feature extraction** | `src/features/` | binary maske → geometri | Saf OpenCV; **modelden bağımsız** |
| **Görselleştirme/Arayüz** | `src/features/visualize.py`, `app/` | geometri → görsel | Overlay çizimi, Streamlit, CLI |
| **Değerlendirme** | `src/training/evaluate.py` | `best.pt` → rapor | Metrik + feature doğrulama + failure analizi |

## Neden bu ayrım?

- **`features/` modelden tamamen bağımsız** (girdi maske, çıktı geometri). Böylece model
  zayıf çıksa bile geometri katmanı dummy maskeyle test edilip savunulabilir. Bu, case
  study'nin "katmanlar net ayrıştırılmalı" ve "sistemin sınırlarını dürüst ortaya koyma"
  isteklerinin somut karşılığı.
- **Hiperparametreler koddan ayrı** (`configs/unet_r34.yaml`) → tekrar üretilebilirlik.
- **Eğitim ile inference ayrı** → eğitim Colab GPU'da, inference/arayüz lokalde (CPU) koşar.
- **Post-processing ayrı bir adım** → ham model çıktısı ile geometri girdisi arasında net sınır.

## Tekrar üretilebilirlik

`src/utils/reproducibility.py` random/numpy/torch seed'lerini sabitler; split ve indirme de
seed'lidir (`seed=42`). Aynı config aynı sonucu verir.
```
