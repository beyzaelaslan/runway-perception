# DEVLOG — Runway Perception

Yolculuk günlüğü. Kronolojik, kısa, dürüst. Kararlar Beyza'nın.

---

## 2026-08-23 · Problemi nasıl parçaladım  ⚠️ TASLAK (Beyza gözden geçirecek)

Görevi bağımsız test edilebilir katmanlara böldüm:
**veri → feature extraction → eğitim → inference → değerlendirme.**

Feature extraction'ı eğitimden **önce** yapacağım: modelden bağımsız (girdi = maske,
çıktı = geometri), dummy maske ile test edilebilir. Model zayıf çıksa bile bu modül çalışır.

**Kısıtlar:**
- Donanım: Intel Mac, GPU yok. Geliştirme lokalde (VSCode); **sadece eğitim** Colab'ın
  ücretsiz GPU'sunda koşacak, çıkan checkpoint lokale indirilecek.
- Süre kısıtlı → metrik kovalamak yok; hedef "çalışan MVP + dürüst sınır raporu".

---

## 2026-08-23 · Veri seti: LARD (V1, ~800 subset)  `[Claude önerisi — onaylandı]`

**Neden LARD:**
- Problemin tam senaryosu: yaklaşma fazı, ön görüş, pist.
- ~1800 gerçek görüntüsü var → domain gap dürüstçe tartışılabilir.
- Akademik referans + MIT lisans → savunulabilir, tekrar üretilebilir.

**Önemli bulgu:** LARD hazır maske vermiyor, pistin **4 köşesini (CSV)** veriyor.
→ GT maskeyi köşeleri doldurarak biz üreteceğiz. Bonus: feature extraction'ı bu
ground-truth köşelerle doğrulayabiliriz.

**Elenenler:** FS2020 (%100 sentetik, domain gap tartışılamaz) · LARD+FS2020 birlikte
(iki ayrı pipeline, süreye değmez).

---

## 2026-08-23 · Faz 0: kurulum  ⚠️ TASLAK (Beyza gözden geçirecek)

Katmanlı iskelet + `.gitignore` + `requirements.txt` + `configs/unet_r34.yaml` kuruldu.
Hiperparametreler config'te (koda gömülü değil), seed sabit.

**Tuzak:** `.gitignore`'daki `data/` deseni `src/data` paketini de eziyordu →
`/data/` diye sabitledim. `docs/` (case study PDF) public repoya girmesin diye ignore'landı.

**Takılma → çözüm:** venv kurulumunda torch (Intel Mac'in son x86 wheel'i, 2.2.2) NumPy 2
ile kırıldı; opencv 5.x ise NumPy 2 istiyor. Zincir: `numpy<2` + `opencv-python<5` pinledim.
Çalışan set: torch 2.2.2 (CPU) · numpy 1.26.4 · opencv 4.11 · smp 0.5. Tüm importlar ✓.

---

## 2026-08-23 · Karar revizyonu: LARD V1 → V2 (indirme lojistiği)  `[Claude önerisi — onaylandı]`

İndirme yolunu araştırınca **"V1 daha küçük" varsayımım yanlış çıktı:** V1 (data.gouv.fr)
sentetik train'i 3.5–7.2 GB'lık zip'lere bölmüş (toplam ~35 GB), gerçek görüntüler ayrı 5 GB.
~800 görüntü için bile min 3.5 GB indirmek gerekiyor.

**Revize karar:** V2 (HuggingFace, `DEEL-AI/LARD_V2`) — Parquet + streaming ile ~800 örneği
tüm seti indirmeden çekebiliyoruz (yüz MB'lar). Köşe koordinatı + zengin metadata
(slant_distance, height_above_runway, lateral_path_angle) var → failure analizinde
"uzak mesafe / alçak açı" örüntülerini nesnel ayırmak için avantaj.

**Kabul edilen ödün:** V2'nin kaynakları sentetik (arcgis/bingmaps/ges/xplane/flsim) →
muhtemelen gerçek görüntü yok. "Gerçek görüntüyle domain gap testi" hedefini,
"domain gap'i kavramsal + metadata üzerinden dürüstçe tartışma"ya çeviriyoruz. Gerekirse
sonradan V1'in 5 GB'lık gerçek test setini ekleyebiliriz.
