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

**Açık kalan:** lokal venv + `pip install` henüz koşulmadı.
