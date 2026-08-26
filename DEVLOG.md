# DEVLOG — Runway Perception

Bu bir yolculuk günlüğü: projede izlediğim yolu, aldığım kararları, denediğim ve
vazgeçtiğim yaklaşımları, takıldığım yerleri ve nasıl çözdüğümü kronolojik olarak yazdım.
Final rapor değil; düşünce sürecimin kaydı.

---

## 2026-08-23 · Problemi nasıl parçaladım

Görevi tek bir "model eğit" işi olarak değil, bağımsız test edilebilir katmanlar olarak
gördüm: **veri → feature extraction → eğitim → inference → değerlendirme.**

Feature extraction'ı bilinçli olarak eğitimden **önce** yapmaya karar verdim: bu katman
modelden bağımsız (girdi = maske, çıktı = geometri), dummy maske ile test edilebilir. Böylece
model zayıf çıksa bile bu modül çalışır ve savunulabilir kalır.

**Kısıtlarım:**
- Donanım: Intel Mac, GPU yok. Geliştirmeyi lokalde yaptım; **sadece eğitimi** Colab'ın
  ücretsiz GPU'sunda koşturup çıkan checkpoint'i lokale indirdim.
- Süre kısıtlı → metrik kovalamak yerine hedefim "çalışan MVP + dürüst sınır raporu" oldu.

---

## 2026-08-23 · Veri seti kararı: LARD

**Neden LARD:**
- Problemin tam senaryosu: yaklaşma fazı, ön görüş, pist.
- Gerçek görüntü seçeneği olması domain gap'i dürüstçe tartışmayı mümkün kılıyor.
- Akademik referans + MIT lisans → savunulabilir ve tekrar üretilebilir.

**Önemli bulgu:** LARD hazır maske vermiyor, pistin **4 köşesini (CSV)** veriyor. GT maskeyi
köşeleri doldurarak üretmem gerekti. Bunun güzel bir yan etkisi oldu: köşeler zaten
ground-truth olduğundan, çıkardığım geometriyi bu köşelerle nesnel karşılaştırabildim.

**Elediklerim:** FS2020 (%100 sentetik, domain gap tartışılamaz) ve LARD+FS2020 birlikte
(iki ayrı format/pipeline, süreye değmez).

---

## 2026-08-23 · Faz 0: kurulum

Katmanlı iskelet + `.gitignore` + `requirements.txt` + `configs/unet_r34.yaml` kurdum.
Hiperparametreleri koda gömmedim, config'te tuttum; seed'i sabitledim.

**Tuzak:** `.gitignore`'daki `data/` deseni `src/data` paketini de eziyordu (git desende
başında `/` olmayınca her yerdeki `data`'yı eşliyor). `/data/` diye köke sabitleyerek çözdüm.

**Takıldığım yer → çözüm:** venv kurarken torch (Intel Mac'in son x86 wheel'i, 2.2.2) NumPy 2
ile kırıldı; opencv 5.x ise NumPy 2 istiyor. Zinciri `numpy<2` + `opencv-python<5` pinleyerek
çözdüm. Çalışan set: torch 2.2.2 (CPU) · numpy 1.26.4 · opencv 4.11 · smp 0.5.

---

## 2026-08-23 · Karar revizyonu: LARD V1 → V2 (indirme lojistiği)

İndirme yolunu araştırınca **"V1 daha küçük" varsayımımın yanlış** olduğunu gördüm: V1
(data.gouv.fr) sentetik train'i 3.5–7.2 GB'lık zip'lere bölmüş (toplam ~35 GB), gerçek
görüntüler ayrı 5 GB. ~800 görüntü için bile en az 3.5 GB indirmek gerekiyordu.

**Revize kararım:** V2 (HuggingFace, `DEEL-AI/LARD_V2`). Zengin metadata'sı da var
(slant_distance, height_above_runway, lateral_path_angle) — bunları failure analizinde
"uzak mesafe" örüntüsünü nesnel ayırmak için kullandım.

**Kabul ettiğim ödün:** V2'nin kaynakları tamamen sentetik → gerçek görüntü yok. "Gerçek
görüntüyle domain gap testi" hedefini, "domain gap'i kavramsal + metadata üzerinden dürüstçe
tartışma"ya çevirdim.

---

## 2026-08-23 · Faz 1: veri katmanı

800 görüntü indirdim (5 kaynaktan 160'ar), köşe→maske dönüşümünü, PyTorch Dataset'i,
augmentation'ı ve split'i yazdım. Overlay ile gözle doğruladım.

**Takıldığım yer → çözüm:** İlk indiricim HF **streaming** kullanıyordu; bağlantı büyük
parquet shard'larını tamponlarken sürekli düşüyordu (`peer closed connection`), çok yavaştı
ve arkada zombi süreç bırakıyordu. `hf_hub_download` ile shard'ı doğrudan indirmeye geçtim
(resumable + cache'li). Bir shard ~1733 satır içerdiğinden config başına tek shard yetti.

**İki dürüst bulgu:**
1. Subset'te yalnızca **~20 benzersiz sahne/havaalanı** var → rastgele split aynı havaalanını
   hem train hem test'e koyup **sızıntı** yaratırdı. Bu yüzden scenario bazlı (gruplu) split
   yaptım (GroupShuffleSplit, seed=42): train 459 / val 226 / test 115, sızıntı 0. Ödün:
   oranlar tam 70/15/15 değil ve test az havaalanı kapsıyor.
2. Pist, görüntünün ortalama **~%0.17'si** (uzak yaklaşmada minik) → aşırı sınıf dengesizliği.
   Bu bulgu, ileride Dice+BCE loss tercihimi doğrudan şekillendirdi.

---

## 2026-08-23 · Faz 2: feature extraction

Modelden bağımsız geometri modülü (`geometry.py`) + görselleştirme (`visualize.py`) + 8 birim
test yazdım. Girdi binary maske, çıktı 4 özellik.

**Bu özellikleri neden seçtim (iniş kararı bağlamı):**
- **Pist sınırları (4 köşe):** pistin görüntüdeki konumu/kapsamı.
- **Merkez hattı + yaklaşma açısı:** uçağın piste hizalanması — karar destekli inişin en
  kritik göstergesi (dikeyden sapma = yanlış hizalama).
- **Threshold kenarı:** pistin kameraya en yakın ucu → mesafe/iniş noktası sezgisi.

**Kritik karar — merkez hattı yöntemini değiştirdim:** İlk yöntemim minAreaRect'in kısa kenar
orta noktalarıydı. Test edince **çok yakın mesafede kırıldığını** gördüm: pist enine boyundan
geniş olunca "uzun eksen" yatay seçiliyordu (yakın rejimde medyan açı 86.5°, olması gereken
~0). Bunun yerine her görüntü satırının maske orta noktalarına doğru fit ettim (down-range
eksen ~dikey varsayımı). Sonuç: yakın rejim medyan açı **86.5° → 12.2°**, yatay hata 92 → 0.
Değerlendirdiğim alternatifler: minAreaRect kısa-kenar (elendi), PCA principal axis (aynı
en-boy sorununa açık, elendi), satır-bazlı fit (seçtim).

**Dürüst sınır:** Uzak rejimde (slant>3) pist minik olunca geometri hâlâ gürültülü
(medyan açı ~34°). Bu bir model sınırı değil, veri/çözünürlük sınırı.

**Doğrulama:** Sentetik maske (dikey→0°, boş→graceful) + gerçek GT maske üzerinde gözle
overlay + 8/8 pytest.

---

## 2026-08-23 · Faz 3: eğitim altyapısı

Model factory, Dice+BCE loss, metrikler (IoU/Dice/precision/recall/F1), config'ten okuyan
tekrar üretilebilir train loop ve Colab notebook'u yazdım. CPU'da smoke test (1 epoch/2 batch)
uçtan uca geçti.

**Model seçimim — U-Net + ResNet34 (ImageNet pretrained):**
- **Pretrained encoder şart:** 800 görüntü az; sıfırdan eğitim yakınsamaz.
- **U-Net:** skip-connection'lar ince yapıları (pist kenarı, uzak/ince pist) korur.
- **ResNet34:** hafif → Colab'da makul sürede eğitilir (ResNet50+ gereksiz ağır).

**Değerlendirdiğim alternatifler:**
- **DeepLabv3+:** daha güçlü ama ağır; bu veri boyutunda overfit riski + Colab'da yavaş.
- **SegFormer:** daha çok veri ister; 800 görüntüde avantajını gösteremez.
- **Klasik CV eşikleme (baseline):** pist rengi/kontrastı sahneye göre değişken → kırılgan.
  Yine de feature katmanı model-bağımsız olduğu için gerekirse klasik CV maskesiyle de beslenebilir.

**Neden accuracy değil:** Pist görüntünün ~%0.17'si; her şeyi arka plan diyen model %99+
accuracy alır ama işe yaramaz. IoU/Dice örtüşmeyi ölçer, bu tuzağa düşmez. Precision/recall
hata tipini ayırır (taxiway'i pist sanma vs pisti kaçırma).

**Ortam ayrımı:** Kodu lokalde (CPU) yazıp doğruladım; asıl eğitimi Colab GPU'da koştum.
Lokalde tam epoch ~10+ dk sürüyordu, bu da GPU'nun şart olduğunu gösterdi.

---

## 2026-08-25 · Faz 4: eğitim sonucu (Colab, 25 epoch)

**Gözlemim (eğitim eğrilerinden):**
- Loss: train ~0.08, val ~0.13'te oturdu; büyük düşüş epoch ~8-13 arası, sonra plato.
  train/val farkı küçük → ciddi overfitting yok, sağlıklı yakınsama.
- Val metrikleri: IoU ~0.68, Dice ~0.80 (tepe ~epoch 14). İlk epoch'larda zikzak (küçük val
  seti + %0.17 dengesizlik + LR ısınması), 14'ten sonra stabil.

**Yorumum:** Bu val sayısı ve sentetik + az sahne üzerinde; gerçek uçuşa genelleme daha düşük
olur. Rapor edeceğim asıl sayı test seti. Ayrıca agregat metrik uzak-pist zayıflığını gizliyor.

**Kararım:** Sonuç MVP için tatmin edici olduğundan yeniden eğitim hakkımı kullanmadım;
metrik kovalamak yerine kalan fazlara geçtim. Colab'daki `best.pt`'i lokale indirdim.

---

## 2026-08-25 · Faz 5: inference pipeline

`predict.py` (RunwayPredictor) + `postprocess.py` yazdım. Uçtan uca: görüntü → model maskesi
→ morfolojik temizleme + en büyük bileşen → geometri → overlay. Tekil + toplu çalışıyor;
JPG/PNG doğrulaması, desteklenmeyen formatta anlamlı hata veriyor.

**Zaman yönetimi kararım:** Eğitim Colab'da dönerken beklemek yerine inference katmanını
paralel kurdum. Checkpoint henüz yokken pipeline mekaniğini rastgele ağırlıklı bir dummy
checkpoint ile doğruladım (maske anlamsız ama akış hatasız çalıştı).

**Post-processing gerekçem:** Ham model maskesi gürültü/delik içerir; geometri tek temiz
bileşen bekliyor. Açma (gürültü sil) + kapama (delik doldur) + en büyük bileşen (taxiway/
yansıma ele).

---

## 2026-08-25 · Faz 6/7/8: eğitim koşarken paralel iş

Eğitim Colab'da dönerken checkpoint gerektirmeyen parçaları paralel kurdum:
- **Streamlit arayüzü** (`app/main.py`): tekil+toplu yükleme, maske overlay, geometri çizimi,
  sayısal değerler; checkpoint yoksa / pist bulunamazsa graceful uyarı.
- **`evaluate.py`:** test metrikleri + feature doğrulama (GT vs tahmin açı hatası) + en kötü
  N tahmini kaydetme + FP/FN eğilimi.
- **ARCHITECTURE.md + README.md.**

Feature doğrulamayı GT köşeleriyle yapabildim çünkü LARD zaten pistin 4 köşesini veriyor —
"özellikleri nasıl doğruladın" sorusuna nesnel cevap.

---

## 2026-08-25 · Faz 6 sonuç: değerlendirme + failure analizi

**Test metrikleri (115 görüntü, görülmemiş sahneler):**
IoU 0.631 · Dice 0.774 · precision 0.815 · recall 0.736. Val (0.68) > test (0.63) olması
beklenen ve sağlıklı — scenario split sayesinde test görülmemiş havaalanlarından oluşuyor,
sızıntı yok; aradaki fark genelleme sınırını dürüst gösteriyor.

**Feature doğrulama:** GT köşelerden vs tahmin maskesinden çıkardığım yaklaşma açısı — medyan
hata 9.2° (ort 16.5°; uç örnekler ortalamayı çekiyor).

**Failure örüntüsü (net bulgu):** En kötü 12 tahminin **hepsi minik pist** (gt_px 71–370,
görüntünün %0.007–0.035'i) ve **9'unda model hiçbir şey üretmedi**. Birleştirici faktör
mesafe değil doğrudan **hedef boyutu** — pist birkaç piksele düşünce arka plandan
ayrılamıyor. Baskın hata yanlış negatif (recall<precision): model ihtiyatlı, kaçırıyor.
Bu, %0.17 dengesizlik ve Dice+BCE tercihimle tutarlı.

---

## 2026-08-26 · Kapanış: zamanı nasıl harcadım & baştan başlasam

**Zamanı nasıl harcadım:**
En çok zaman **veri katmanı kararlarına** gitti; bunların çoğu araştırma/keşifti: LARD'ın
maske değil köşe verdiğini fark etmek, V1'in devasa boyutunu görüp V2'ye dönmek, streaming
çökünce shard indirmeye geçmek, 20 sahne bulunca scenario bazlı split'e karar vermek. İkinci
büyük dilim **feature extraction'ı sağlamlaştırmaktı** (minAreaRect'in yakında kırılması →
satır-bazlı fit). Eğitim altyapısını yazdıktan sonra asıl eğitimi Colab'a atıp, o arka planda
dönerken inference, arayüz, değerlendirme ve dokümanları paralel yürüttüm — GPU beklerken boş
durmadım. En sonda gerçek `best.pt` ile değerlendirme + failure analizini tamamladım.
Donanımım (GPU yok) yüzünden eğitimi Colab'a taşımak ve kalan her şeyi ona paralel kurmak
bilinçli bir zaman yönetimi kararıydı.

**Baştan başlasam neyi farklı yapardım:**
- **Daha çok sahne çeşitliliği:** config başına tek shard yerine birkaç shard indirip 20'den
  fazla havaalanı kapsardım; test seti daha temsili olurdu.
- **Uzak/minik pisti baştan ciddiye alırdım:** en büyük zayıflık bu. Daha yüksek çözünürlük,
  tiling veya minik-nesne odaklı loss (Tversky/Focal) ile başlardım.
- **Domain gap'i sayısal ölçerdim:** V2 sentetik-only olduğundan, LARD'ın gerçek test
  görüntülerini erken ekleyip sentetik→gerçek düşüşü rakamla gösterirdim.
- **Değerlendirmeyi daha erken kurardım:** evaluate.py'yi eğitimden önce iskelet olarak
  hazırlasaydım, checkpoint gelir gelmez failure analizine daha çok zaman kalırdı.
