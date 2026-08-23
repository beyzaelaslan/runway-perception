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

---

## 2026-08-23 · Faz 1: veri katmanı tamam  ⚠️ TASLAK (Beyza gözden geçirecek)

**Ne yaptım:** 800 görüntü indirildi (5 kaynaktan 160'ar), köşe→maske dönüşümü, PyTorch
Dataset, augmentation ve scenario bazlı split yazıldı. Overlay ile gözle doğrulandı.

**Takılma → çözüm:** İlk `download_lard.py` HF **streaming** kullanıyordu; bağlantı büyük
parquet shard'larını tamponlarken sürekli düşüyordu (`peer closed connection`), çok yavaştı.
`hf_hub_download` ile shard'ı doğrudan indirmeye geçtim (resumable + cache'li). Bir shard
~1733 satır → config başına tek shard yetti, 5 shard toplam.

**İki dürüst bulgu (TESTING'e taşınacak):**
1. **Sadece ~20 benzersiz scenario/airport** → rastgele split sızıntı yapardı. Bu yüzden
   scenario bazlı (gruplu) split (`split.py`, GroupShuffleSplit, seed=42). Sonuç
   train 459 / val 226 / test 115, sızıntı 0. Ödün: oran tam 70/15/15 değil, test az
   havaalanı kapsıyor.
2. **Pist piksel oranı ~%0.17** (uzak yaklaşmada pist minik) → aşırı sınıf dengesizliği.
   Bu, **Dice+BCE combo loss** kararımızı doğruluyor (Dice dengesizliğe dayanıklı).

**Sonuç:** Veri katmanı hazır. Sıradaki: Faz 2 (feature extraction) — eğitimden önce,
dummy maske ile test edilebilir.

---

## 2026-08-23 · Faz 2: feature extraction  ⚠️ TASLAK (Beyza gözden geçirecek)

**Ne yaptım:** Modelden bağımsız geometri modülü (`geometry.py`) + görselleştirme
(`visualize.py`) + 8 birim test. Girdi binary maske, çıktı 4 özellik.

**Neden bu özellikler (iniş kararı bağlamı):**
- **Pist sınırları (4 köşe):** pistin görüntüdeki konumu/kapsamı.
- **Merkez hattı + yaklaşma açısı:** uçağın piste **hizalanması** — karar destekli inişin
  en kritik göstergesi (dikeyden sapma = yanlış hizalama).
- **Threshold kenarı:** pistin kameraya en yakın ucu → **mesafe/iniş noktası** sezgisi.

**Kritik karar — merkez hattı yöntemi değişti:**
İlk yöntem minAreaRect'in kısa kenar orta noktalarıydı. Test edince gördüm ki **çok yakın
mesafede kırılıyor**: pist enine boyundan geniş olunca "uzun eksen" yatay seçiliyordu
(yakın rejimde medyan |açı| 86.5°, olması gereken ~0). Bunun yerine **her görüntü satırının
maske orta noktalarına doğru fit** ettim (down-range eksen ~dikey varsayımı). Sonuç:
yakın medyan |açı| 86.5° → **12.2°**, yatay hata 92 → **0**. Orta/uzak rejim de iyileşti.

**Değerlendirdiğim alternatifler:** minAreaRect kısa-kenar (elendi, yakında kırılıyor);
PCA principal axis (aynı en-boy sorununa açık); satır-bazlı fit (seçildi).

**Ne çalışmadı / dürüst sınır:** Uzak rejimde (slant>3) pist minik (birkaç piksel) →
geometri hâlâ gürültülü (medyan |açı| 34°). Bu, %0.17 pist oranı bulgusuyla tutarlı,
bir model sınırı değil veri/çözünürlük sınırı. TESTING'e taşınacak.

**Doğrulama:** Sentetik (dikey→0°, boş→graceful) + gerçek GT maske üzerinde gözle overlay
(outputs/data_check/features_montage.png) + 8/8 pytest.

**Sonuç:** Feature katmanı hazır ve model olmadan çalışıyor. Sıradaki: Faz 3 eğitim altyapısı.

---

## 2026-08-23 · Faz 3: eğitim altyapısı  ⚠️ TASLAK (Beyza gözden geçirecek)

**Ne yaptım:** Model factory, Dice+BCE loss, metrikler (IoU/Dice/precision/recall/F1),
config'ten okuyan tekrar üretilebilir train loop, Colab notebook. CPU'da smoke test
(1 epoch/2 batch) uçtan uca geçti: loss iniyor, checkpoint + history kaydediliyor.

**Model seçim gerekçesi — U-Net + ResNet34 (ImageNet pretrained):**
- **Pretrained encoder şart:** 800 görüntü az; sıfırdan eğitim yakınsamaz. ImageNet
  ön-eğitimli ResNet34 düşük seviyeli özellikleri (kenar/doku) hazır getiriyor.
- **U-Net:** skip-connection'lar ince yapıları (pist kenarı, uzak/ince pist) korur;
  binary segmentasyonda hızlı yakınsar.
- **ResNet34:** hafif → Colab ücretsiz GPU'da makul sürede eğitilir (ResNet50+ gereksiz ağır).

**Değerlendirdiğim alternatifler:**
- **DeepLabv3+:** güçlü ama daha ağır; bu veri boyutunda ekstra kapasite overfit riski,
  Colab'da yavaş. smp ile arch değiştirmek tek satır (`factory.py`), gerekirse denenebilir.
- **SegFormer / transformer:** daha çok veri ister; 800 görüntüde avantajını gösteremez.
- **Klasik CV eşikleme (baseline):** pist rengi/kontrastı sahneye göre çok değişken
  (çim, asfalt, alacakaranlık) → kırılgan. Yine de feature extraction'ı zaten model-bağımsız
  yazdık; kötü durumda geometri katmanı klasik CV maskesiyle de beslenebilir.

**Metrik gerekçesi (neden accuracy değil):** Pist görüntünün ~%0.17'si; her şeyi arka plan
diyen model %99+ accuracy alır ama işe yaramaz. IoU/Dice örtüşmeyi ölçer, bu tuzağa düşmez.
Precision/recall hata tipini ayırır (taxiway'i pist sanma vs pisti kaçırma) → failure analizi.

**Ortam ayrımı:** Kod lokalde (CPU) yazılıp doğrulandı; asıl eğitim `notebooks/train_colab.ipynb`
ile Colab GPU'da koşacak. Lokalde tam epoch ~10+ dk (CPU) → GPU şart olduğunu doğruladı.

**Sonuç:** Altyapı hazır. **Manuel adım (Beyza):** notebook'u Colab'da koş, `best.pt` indir.
