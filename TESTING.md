# TESTING — Test, Doğrulama ve Sistem Sınırları

Bu dosya sistemi nasıl test ettiğimizi, hangi metrikleri neden seçtiğimizi, çıkarılan
özellikleri nasıl doğruladığımızı ve **sistemin sınırlarını** dürüstçe belgeler.

Sonuçlar **test seti** (115 görüntü, scenario bazlı ayrılmış — eğitimde görülmemiş
havaalanları) üzerindedir. Model: U-Net + ResNet34, 25 epoch (en iyi epoch 14).

---

## 1. Hangi metrikler ve neden?

| Metrik | Test değeri | Neden |
|--------|-------------|-------|
| **IoU** | **0.631** | Segmentasyonun standart örtüşme ölçütü |
| **Dice** | 0.774 | IoU'ya benzer, küçük nesnelerde daha az cezalandırıcı |
| **Precision** | 0.815 | Yanlış pozitif eğilimi (taxiway/zemin pist sanma) |
| **Recall** | 0.736 | Yanlış negatif eğilimi (pisti kaçırma) |
| **F1** | 0.774 | Precision/recall dengesi (binary'de Dice'a eşit) |

**Neden accuracy değil:** Pist, görüntünün ortalama **~%0.17'si**. Her şeyi "arka plan"
diyen bir model %99+ accuracy alır ama tamamen işe yaramaz. IoU/Dice bu tuzağa düşmez;
precision/recall ise hata **tipini** ayırır — bu, aşağıdaki failure analizinin temeli.

> Loss ve val metrik eğrileri: `assets/training_curves.png`.
> train/val farkı küçük → ciddi overfitting yok; val IoU ~0.68'de oturdu.

**Val vs Test:** val IoU ~0.68, test IoU 0.63. Test'in bir miktar düşük olması beklenen ve
**sağlıklı** — scenario bazlı split sayesinde test seti eğitimde görülmemiş havaalanlarından
oluşuyor (veri sızıntısı yok). Aradaki fark modelin genelleme sınırını dürüstçe gösteriyor.

---

## 2. Segmentasyon maskelerinin nicel değerlendirmesi

Metrikler `src/training/evaluate.py` ile test setinde **piksel bazında** (TP/FP/FN/TN
biriktirilerek) hesaplandı — görüntü başına ortalamadan daha stabil, küçük maskelerde
yanlılık yaratmıyor. Rapor: `outputs/eval_report.json`.

---

## 3. Çıkarılan pist özelliklerini nasıl doğruladık?

Feature extraction **modelden bağımsız** (girdi maske, çıktı geometri), iki katmanda
doğrulandı:

**(a) Birim testler** (`tests/test_features.py`, 8/8 geçti): bilinen sentetik geometriyle —
dikey pist → açı ~0°, eğik pist → beklenen işaretli açı, boş/gürültülü maske → graceful.

**(b) Ground-truth ile karşılaştırma:** LARD zaten pistin 4 köşesini veriyor. Bu sayede
GT maskeden çıkarılan yaklaşma açısı ile **tahmin maskeden** çıkarılan açıyı test setinde
karşılaştırdık:
- **Medyan açı hatası: 9.2°**, ortalama: 16.5° (n=91).
- Medyanın ortalamadan düşük olması: birkaç uç örnek (minik/uzak pist) ortalamayı yukarı
  çekiyor; tipik durumda hizalama açısı ~9° hata ile doğru.

> Örnek geometri çıktısı: `assets/features_example.png` (sınırlar + merkez hattı + açı).

---

## 4. Yanlış pozitif / negatif örüntüleri (sistem sınırları)

**Baskın hata tipi: yanlış negatif (pist kaçırma).**
- Toplam FN pikseli (21.363) > FP pikseli (13.516); recall (0.736) < precision (0.815).
- Yani model **ihtiyatlı**: yanlış yere pist boyamaktan çok, var olan pisti (özellikle
  ince/uzak kısımları) **eksik** segmentliyor.

**En kötü 12 tahminin ortak örüntüsü — minik/uzak pist:**
- Hepsinde GT pist alanı **71–370 piksel** (görüntünün %0.007–0.035'i), yani uzak yaklaşma.
- **12'den 9'unda model hiçbir şey üretmedi** (tamamen kaçırdı).
- Birleştirici faktör mesafe değil doğrudan **hedef boyutu**: pist birkaç piksele
  düştüğünde model onu arka plandan ayıramıyor. Bu, %0.17 sınıf dengesizliği bulgusuyla
  ve Dice+BCE loss tercihiyle tutarlı.

> Failure örnekleri (GT yeşil, tahmin kırmızı): `assets/failure_examples.png`.

---

## 5. Sistemin sınırları (dürüst özet)

1. **Uzak/minik pist:** En büyük zayıflık. Birkaç piksellik pistler kaçırılıyor. Gerçek
   inişte kritik olan "uzaktan erken tespit" bu modelin en zayıf noktası. İyileştirme:
   daha yüksek çözünürlük, minik-nesne odaklı loss (ör. Tversky/focal), tiling.
2. **%100 sentetik veri (LARD V2):** Gerçek uçuş görüntüsü içermiyor. Işık, sensör gürültüsü,
   hava koşulları gerçek dünyada farklı → **domain gap** var; metrikler gerçek uçuşta düşer.
3. **Az sahne çeşitliliği:** Subset ~20 havaalanı/scenario. Test seti birkaç havaalanı
   kapsıyor; metrikler bu dar dağılıma göre okunmalı.
4. **Feature açısı yakın olmayan rejimlerde gürültülü:** minik maskede geometri kararsız
   (bkz. §3 ortalama vs medyan).
5. **Tek sınıf, tek pist varsayımı:** En büyük bileşen seçiliyor; kadrajda birden fazla
   pist olursa yalnızca biri işlenir.

Bu sınırlar gizlenmedi; aksine sistemin nerede kırılacağını bilmek, karar destekli iniş
bağlamında güvenlik açısından modelin kendisi kadar önemlidir.
