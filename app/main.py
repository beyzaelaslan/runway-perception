"""Streamlit arayüzü — pist segmentasyonu + geometri görselleştirme.

Kullanıcı tekil veya toplu görüntü yükler; sistem her görüntü için maske overlay'i,
çizili geometrik özellikleri ve sayısal değerleri (açı, köşeler, threshold) gösterir.

Çalıştırma:
    streamlit run app/main.py

Not: eğitilmiş checkpoint (outputs/best.pt) gerekir. Yoksa arayüz anlamlı uyarı verir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# src'yi import edebilmek için proje kökünü path'e ekle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predict import RunwayPredictor, SUPPORTED_EXT  # noqa: E402

DEFAULT_CKPT = "outputs/best.pt"


@st.cache_resource
def load_predictor(checkpoint_path: str) -> RunwayPredictor:
    """Predictor'ı bir kez yükleyip cache'ler (her etkileşimde yeniden yüklenmesin)."""
    return RunwayPredictor(checkpoint_path)


def main() -> None:
    st.set_page_config(page_title="Runway Perception", layout="wide")
    st.title("🛬 Pist Algı Sistemi — Segmentasyon + Geometri")

    ckpt = st.sidebar.text_input("Checkpoint yolu", DEFAULT_CKPT)
    st.sidebar.caption("Eğitim Colab'da koşar; çıkan best.pt'i outputs/ altına koy.")

    if not Path(ckpt).exists():
        st.warning(f"Checkpoint bulunamadı: `{ckpt}`. "
                   "Modeli Colab'da eğitip `best.pt`'i bu yola koy.")
        st.stop()

    try:
        predictor = load_predictor(ckpt)
    except Exception as e:  # bozuk/uyumsuz checkpoint
        st.error(f"Checkpoint yüklenemedi: {e}")
        st.stop()

    st.sidebar.success(f"Model yüklendi (cihaz: {predictor.device})")

    files = st.file_uploader(
        "Görüntü yükle (tekil veya toplu)",
        type=[e.lstrip(".") for e in SUPPORTED_EXT],
        accept_multiple_files=True,
    )
    if not files:
        st.info("Başlamak için bir veya birden fazla pist görüntüsü yükle.")
        return

    for f in files:
        st.divider()
        st.subheader(f.name)
        image = np.array(Image.open(f).convert("RGB"))
        res = predictor.predict(image)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Girdi", use_container_width=True)
        col2.image(res.overlay, caption="Maske + geometri", use_container_width=True)

        feat = res.features
        if not feat.valid:
            st.warning(f"Pist bulunamadı ({feat.reason}). "
                       "Görüntüde pist görünmüyor olabilir veya model kaçırdı.")
            continue

        # Sayısal özellikler
        c1, c2, c3 = st.columns(3)
        c1.metric("Yaklaşma açısı", f"{feat.approach_angle_deg:.1f}°",
                  help="0° = piste hizalı (dikey). İşaret: sapma yönü.")
        c2.metric("Pist alanı (px)", f"{feat.area_px:,}")
        c3.metric("Durum", "geçerli ✓")

        with st.expander("Detaylı geometri (köşe koordinatları)"):
            st.write("**Pist sınır köşeleri (x, y):**")
            st.write([[round(x, 1), round(y, 1)] for x, y in feat.corners.tolist()])
            far, near = feat.center_line
            st.write(f"**Merkez hattı:** uzak uç {tuple(round(c, 1) for c in far)} → "
                     f"yakın uç {tuple(round(c, 1) for c in near)}")
            if feat.threshold_edge:
                t1, t2 = feat.threshold_edge
                st.write(f"**Threshold (en yakın) kenar:** "
                         f"{tuple(round(c, 1) for c in t1)} — {tuple(round(c, 1) for c in t2)}")


if __name__ == "__main__":
    main()
