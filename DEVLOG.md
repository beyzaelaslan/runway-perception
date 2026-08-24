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

---

## 2026-08-25 · Faz 5: inference pipeline  ⚠️ TASLAK (Beyza gözden geçirecek)

**Ne yaptım:** `predict.py` (RunwayPredictor) + `postprocess.py`. Uçtan uca:
görüntü → model maskesi → morfolojik temizleme + en büyük bileşen → geometri → overlay.
Tekil + toplu (klasör) çalışıyor; JPG/PNG doğrulaması, desteklenmeyen formatta anlamlı hata.

**Zaman kısıtı kararı:** Eğitim Colab'da koşarken (4-5 saat kaldı) beklemek yerine inference
katmanını paralel kurdum. Checkpoint henüz yokken pipeline mekaniğini **rastgele ağırlıklı
dummy checkpoint** ile doğruladım (maske anlamsız ama akış hatasız çalıştı) → best.pt gelince
sadece dosyayı koyup gerçek çıktı alacağız.

**Post-processing gerekçesi:** Ham model maskesi gürültü/delik içerir; geometri tek temiz
bileşen bekliyor. Açma (gürültü) + kapama (delik) + en büyük bileşen (taxiway/yansıma eleme).

**Sonuç:** Inference hazır. best.pt gelince test setinde 10-15 örnek koşulacak.

---

## 2026-08-25 · Faz 7/6/8: eğitim koşarken paralel iş  ⚠️ TASLAK (Beyza gözden geçirecek)

Eğitim Colab'da dönerken (süre kısıtı) checkpoint gerektirmeyen parçaları paralel kurdum:

- **Faz 7 — Streamlit (`app/main.py`):** tekil+toplu yükleme, maske overlay, geometri
  çizimi, sayısal değerler (açı/köşe/threshold). checkpoint yoksa / pist bulunamazsa
  graceful uyarı.
- **Faz 6 — `evaluate.py`:** test metrikleri + **feature doğrulama** (GT maskeden vs tahmin
  maskeden çıkarılan yaklaşma açısı hatası, derece) + en kötü N tahmini
  `outputs/failures/`'a kaydetme + yanlış pozitif/negatif eğilimi. best.pt gelince koşulacak.
- **Faz 8 — ARCHITECTURE.md + README.md:** katman diyagramı, çalıştırma komutları, veri
  seti erişimi.

**Neden feature doğrulamayı GT köşeleriyle yapıyoruz:** LARD zaten pistin 4 köşesini veriyor.
Bu yüzden geometrimizi ground-truth ile nesnel karşılaştırabiliyoruz (çoğu adayın atladığı
"özellikleri nasıl doğruladın" sorusunun cevabı).

**Kalan (best.pt'e bağlı):** Faz 4 eğitim eğrileri/gözlemler, Faz 6 gerçek metrikler +
failure örüntüleri, Faz 5/7 canlı test, TESTING.md sayıları.

---

## 2026-08-25 · Faz 4: eğitim sonucu (Colab, 25 epoch)  ⚠️ TASLAK (Beyza gözden geçirecek)

**Gözlem (eğitim eğrilerinden):**
- Loss: train ~0.08, val ~0.13'te oturdu; büyük düşüş epoch ~8-13, sonra plato.
  train/val farkı küçük → **ciddi overfitting yok**, sağlıklı yakınsama.
- Val metrikleri: **IoU ~0.66, Dice ~0.80** (tepe ~epoch 14). İlk epoch'larda zikzak
  (küçük val seti + %0.17 sınıf dengesizliği + LR ısınması), 14'ten sonra stabil.

**Yorum / dürüst sınır:**
- Bu **val** sayısı ve **sentetik + az sahne** üzerinde → gerçek uçuşa genelleme daha düşük
  olur (domain gap). Rapor edilecek asıl sayı test seti (evaluate.py).
- Agregat metrik uzak-pist zayıflığını gizliyor; mesafeye göre kırılım evaluate.py'de.

**Karar:** Sonuç MVP için tatmin edici → CLAUDE.md'deki "en fazla 1 kez yeniden eğit"
hakkını kullanmıyoruz, metrik kovalamıyoruz. Kalan fazlara geçiyoruz.

**Manuel (Beyza):** best.pt + history.json indirildi → outputs/ altına kondu.

---

## 2026-08-25 · Faz 6: değerlendirme + failure analizi  ⚠️ TASLAK (Beyza gözden geçirecek)

**Test metrikleri (115 görüntü, görülmemiş sahneler):**
IoU 0.631 · Dice 0.774 · precision 0.815 · recall 0.736. Val (0.68) > test (0.63) →
scenario split çalışıyor, sızıntı yok, fark genelleme sınırını dürüst gösteriyor.

**Feature doğrulama (GT köşe vs tahmin):** yaklaşma açısı medyan hatası **9.2°**
(ort 16.5°; uç örnekler ortalamayı çekiyor). LARD'ın hazır köşe etiketi sayesinde nesnel.

**Failure örüntüsü (net bulgu):** En kötü 12 tahminin **hepsi minik pist** (gt_px 71–370,
görüntünün %0.007–0.035'i) ve **9'unda model hiçbir şey üretmedi**. Birleştirici faktör
mesafe değil doğrudan **hedef boyutu** — pist birkaç piksele düşünce arka plandan
ayrılamıyor. Baskın hata **yanlış negatif** (recall<precision): model ihtiyatlı, kaçırıyor.
Bu, %0.17 dengesizlik ve Dice+BCE tercihiyle tam tutarlı.

**Sonuç:** TESTING.md gerçek sayılarla yazıldı; sistem sınırları (uzak/minik pist, sentetik
domain gap, az sahne) dürüstçe belgelendi. Teslim görselleri assets/ altında.
