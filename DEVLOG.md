# DEVLOG — Runway Perception System

> Bu dosya bir "final rapor" değil, bir **yolculuk günlüğü**dür.
> Kronolojik, dürüst, gerçekten olan şeyler. Kararlar Beyza'nındır.

---

## [2026-08-23] — Problemi nasıl parçaladım, kısıtlarım neler  ⚠️ TASLAK — BEYZA GÖZDEN GEÇİRECEK

<!--
Bu giriş Claude tarafından taslak olarak yazıldı (kullanıcının açık isteğiyle).
Beyza en sonda okuyup kendi cümleleriyle onaylayacak/düzeltecek. Bu bir "yolculuk
günlüğü" olduğu için nihai halinin SENİN sesinle olması gerekiyor.
-->

**Problemi nasıl parçaladım:**
Görevi tek bir "model eğit" işi olarak değil, birbirinden bağımsız test edilebilir
katmanlar olarak gördüm:
1. **Veri katmanı** — LARD'ı indir, köşe etiketinden binary maske üret, split'le.
2. **Feature extraction (geometri)** — girdisi binary maske, çıktısı geometri. Modelden
   *tamamen bağımsız*. Bu yüzden bunu eğitimden ÖNCE yapıp dummy maske ile test edeceğim;
   model kötü çıksa bile bu modül çalışır ve savunulabilir olur.
3. **Eğitim altyapısı** — U-Net+ResNet34, Dice+BCE, tekrar üretilebilir.
4. **Inference + görselleştirme** — görüntü → maske → geometri → çizim.
5. **Değerlendirme + failure analizi** — sadece metrik değil, nerede kırıldığını dürüstçe
   yazmak.

**Kısıtlarım:**
- **Donanım:** Intel Mac, GPU yok → eğitim Google Colab'ın ücretsiz GPU'sunda koşacak.
  Bu, model/çözünürlük/epoch seçimlerimi (U-Net+ResNet34, 384px, ~25 epoch) doğrudan
  belirledi: makul sürede yakınsayan, hafif bir kurulum.
- **Süre:** Kısıtlı. O yüzden metrik kovalamak yerine "çalışan MVP + dürüst sınır raporu"
  hedefliyorum. Kapsamı bilinçli daraltıyorum (LARD subset, en fazla 1 kez yeniden eğitim).

**Zamanı nasıl harcamayı planlıyorum:**
Feature extraction'ı eğitimden önce bitirip, uzun eğitim koşusunu arka planda döndürürken
diğer fazlara devam etmek → boş bekleme yok.

**Şu an bildiğimle en büyük erken içgörüm:**
LARD'ın hazır maske değil 4 köşe verdiğini araştırırken öğrendim; bu hem maske üretme işi
ekledi hem de feature extraction'ı ground-truth köşelerle doğrulama fırsatı sağladı.

---

## [2026-08-23] — Veri seti seçimi: LARD  `[Claude önerisi — kullanıcı onayladı]`

**Ne yaptım:**
LARD ve FS2020 veri setlerinin gerçek durumunu (etiket formatı, erişim, lisans,
gerçek/sentetik oranı) araştırıp doğruladım. LARD seçildi.

Doğrulanan gerçekler:
- **LARD etiketi = pistin 4 köşesinin piksel koordinatı (CSV).** Hazır piksel-seviye
  segmentasyon maskesi *vermiyor*. (Kaynak: arXiv 2304.09938, GitHub deel-ai/LARD)
- LARD: ~17K sentetik (V1) + ~1800 elle etiketli **gerçek** iniş görüntüsü. V2'de 100K+
  sentetik. Görüntüler 1024×1024 JPEG. **MIT lisans.**
- FS2020: Microsoft Flight Simulator tabanlı, **%100 sentetik**, gerçek görüntü yok
  (Kaggle: relufrank/fs2020-runway-dataset).

**Neden bu yolu seçtim (teknik gerekçe):**
1. LARD problemin tam tanımına oturuyor: *yaklaşma fazı, ön görüş, pist*. Case study'nin
   birebir senaryosu.
2. LARD'ın ~1800 gerçek görüntüsü, sentetik→gerçek **domain gap'i dürüstçe** tartışmayı
   (ve istenirse test etmeyi) mümkün kılıyor. FS2020 %100 sentetik olduğu için bu imkânsız.
3. Akademik referansı var → mülakatta savunması kolay.
4. **Bonus doğrulama fırsatı:** LARD'ın ham etiketi zaten 4 köşe. Faz 2'de maskeden
   çıkaracağımız köşe/merkez hattı/açı özelliklerini doğrudan LARD'ın ground-truth
   köşeleriyle karşılaştırabiliriz → TESTING'in "özellikleri nasıl doğruladın" sorusuna
   sağlam zemin.

**Değerlendirdiğim alternatifler:**
- **FS2020:** Erişimi Kaggle'dan kolay ama %100 sentetik; gerçek uçuşa genelleme sorusunu
  dürüstçe ele alamazdık. Elendi.
- **LARD + FS2020 birlikte:** Genelleme için cazip ama iki ayrı format/pipeline demek;
  süre kısıtı ve "aşırı mühendislik yok" ilkesiyle çelişiyor. Elendi.

**Ne çalışmadı / nerede tıkandım:**
- Kaggle FS2020 sayfası scriptli/oturum gerektirdiği için otomatik içerik çekilemedi;
  format detayı akademik kaynaklardan dolaylı doğrulandı.
- Önemli sürpriz: Başta "LARD hazır maske veriyor" varsayımıyla gelmiştik; araştırma
  bunu çürüttü. GT maskeyi **4 köşeyi polygon-fill ederek biz üreteceğiz**. (Bu iş her iki
  set için de gerekli olduğundan seçimi etkilemedi, ama Faz 1'in kapsamını değiştiriyor.)

**Sonuç:**
Veri seti = **LARD**. Sıradaki iş: küçük bir subset (500–1000 görüntü) indirmek, CSV köşe
etiketlerini gözle doğrulamak ve köşe→binary maske dönüştürme scriptini yazmak.

> NOT (Beyza): Yukarıdaki gerekçe Claude'un doğruladığı gerçeklere dayanıyor. Mülakatta
> kendi ağzından savunacağın için bu girişi oku; katılmadığın veya kendi cümlenle
> yazmak istediğin yerleri değiştir.

---

## [2026-08-23] — LARD subset stratejisi: V1, ~800 görüntü  `[Claude önerisi — kullanıcı onayladı]`

**Ne yaptım / karar:** Tam LARD yerine **V1'den ~800 görüntülük subset** kullanılacak.

**Neden:**
- V1 daha küçük ve yönetilebilir; V2 (100K+) küçük bir case study için gereğinden ağır
  (indirme/depolama/subset seçimi zahmeti).
- ~800 görüntü, CLAUDE.md'deki 500–1000 hedefiyle ve Colab ücretsiz GPU + süre kısıtıyla
  uyumlu. Ağırlıklı sentetik train + az sayıda gerçek görüntü test için ayrılabilir.

**Sonuç:** Faz 1'in ilk somut işi: V1 subset'i indir, CSV köşe etiketini gözle doğrula.

---

## [2026-08-23] — Faz 0: Repo iskeleti ve ortam kurulumu  ⚠️ TASLAK — BEYZA GÖZDEN GEÇİRECEK

**Ne yaptım:**
- Katmanlı klasör iskeleti: `src/{data,models,training,inference,features}`, `app`,
  `notebooks`, `configs`, `tests`. Her katman ayrı → case study'nin "katmanlar net
  ayrıştırılmalı" isteğini mimariye baştan gömdüm.
- `.gitignore` — CLAUDE.md, TASKS.md, `data/`, `outputs/`, checkpoint'ler, venv, .DS_Store
  repoya girmiyor (teslimat temizliği).
- `requirements.txt` ve `configs/unet_r34.yaml` — tüm hiperparametreler config'te, koda
  gömülü değil.

**Neden bu yolu seçtim:**
- İskeleti erken kurmak, sonraki fazlarda dosyaların "doğru yere" düşmesini garanti eder.
- Config'i baştan ayırmak tekrar üretilebilirliği (seed dahil) kolaylaştırır.

**Ne çalışmadı / açık kalan:**
- Lokal venv + `pip install` henüz koşulmadı (torch/smp büyük indirme). Bir sonraki adımda.

**Sonuç:** Repo Faz 1'e hazır. İlk temiz commit atılıyor.
