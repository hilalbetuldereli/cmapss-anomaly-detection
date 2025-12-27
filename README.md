# CMAPSS Uçak Motoru Verilerinde Anomali Tespiti

Bu proje, NASA tarafından yayınlanan C-MAPSS (Commercial Modular Aero-Propulsion
System Simulation) veri seti kullanılarak uçak motorlarına ait çok değişkenli
sensör verileri üzerinde **denetimsiz (unsupervised) anomali tespiti**
yapılmasını amaçlamaktadır.

Projenin temel hedefi, motorlarda oluşabilecek bozulmaları ve anormal
davranışları erken aşamada tespit ederek öngörücü bakım (predictive maintenance)
yaklaşımına katkı sağlamaktır.

---

## 📌 Proje Motivasyonu

Gerçek hayattaki endüstriyel sistemlerde arıza etiketleri çoğu zaman bulunmaz
veya eksiktir. Bu durum denetimli öğrenme yöntemlerinin kullanımını
zorlaştırmaktadır.

Bu nedenle, **etiketsiz veri ile çalışabilen anomali tespiti yöntemleri**
daha gerçekçi ve yaygın bir çözüm sunmaktadır. Autoencoder tabanlı yaklaşımlar,
karmaşık ve doğrusal olmayan sensör ilişkilerini öğrenebilme yetenekleri
nedeniyle bu problem için tercih edilmiştir.

---

## 📊 Veri Seti

- **Kaynak:** NASA C-MAPSS
- **Veri tipi:** Çok değişkenli zaman serisi
- **Dosyalar:**
  - `train_FD001.txt`
  - `test_FD001.txt`
- **Her satır:**
  - 1 motor ID
  - 1 çalışma döngüsü (cycle)
  - 3 operasyonel ayar
  - 21 sensör ölçümü

Bu proje kapsamında yalnızca **21 sensör verisi** kullanılmıştır.

---

## ⚙️ Yöntem

### Yaşam Evreleri

Her motorun kendi toplam çalışma süresine göre veriler üç yaşam evresine
ayrılmıştır:

- **Erken evre:** %0 – %30 (sağlıklı çalışma)
- **Orta evre:** %30 – %70
- **Geç evre:** %70 – %100 (bozulma riski yüksek)

---

### Model Eğitimi

- Model: Dense Autoencoder
- Girdi boyutu: 21 sensör
- Kayıp fonksiyonu: Mean Squared Error (MSE)
- Optimizasyon: Adam
- Normalizasyon: Min-Max Scaler

⚠️ **Autoencoder yalnızca erken yaşam evresi verileri ile eğitilmiştir.**  
Bu evre, motorların henüz arıza göstermediği normal çalışma koşulları olarak
kabul edilmiştir.

---

### Anomali Tespiti

- Eğitim sonrası her örnek için **Reconstruction Error (MSE)** hesaplanmıştır.
- Eğitim verisi üzerindeki MSE dağılımının **%95 persentili**, anomali eşiği
  (threshold) olarak belirlenmiştir.

| Durum | Açıklama |
|------|---------|
| MSE ≤ threshold | Normal |
| MSE > threshold | Anomali |

Bu eşik kullanılarak erken, orta ve geç yaşam evrelerinde anomali yoğunlukları
analiz edilmiştir.

---

## 📈 Model Değerlendirme

Bu problem **denetimsiz** olduğu için accuracy, precision ve recall gibi klasik
metrikler kullanılmamıştır.

Değerlendirme aşağıdaki yöntemlerle yapılmıştır:
- Reconstruction Error dağılımı
- Yaşam evrelerine göre anomali oranları
- Gradio arayüzü üzerinden görsel analiz

Geç yaşam evresinde artan anomali oranları, motorun zamanla bozulmaya
başlaması ile tutarlı sonuçlar üretmiştir.

---

## 🖥️ Gradio Arayüzü

Proje, kullanıcıların farklı yaşam evrelerini seçerek anomali analizini
gözlemleyebilmesi için Gradio tabanlı bir arayüz sunmaktadır.

**Özellikler:**
- Erken / Orta / Geç yaşam evresi seçimi
- Reconstruction Error grafiği
- Anomali sayısı ve oranı

---

## 📁 Proje Dosya Yapısı

```text
.
├── src/
│   ├── train.py            # Autoencoder eğitimi (sadece erken evre)
│   ├── anomaly.py          # Offline anomali analizi ve grafik
│   ├── app.py              # Gradio arayüzü
│   └── download_data.py    # C-MAPSS veri setini indirme
├── data/
│   ├── train_FD001.txt
│   └── test_FD001.txt
├── models/
│   ├── autoencoder.h5
│   ├── scaler.pkl
│   └── threshold.npy
├── requirements.txt
└── README.md

## 🛠️ Kurulum 

Projeyi çalıştırmadan önce Python ortamının hazırlanması ve gerekli
bağımlılıkların yüklenmesi gerekmektedir.

1️⃣ Depoyu Klonlayın
git clone https://github.com/hilalbetuldereli/cmapss-anomaly-detection.git
cd cmapss-anomaly-detection

2️⃣ Sanal Ortam Oluşturun (Önerilir)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux

3️⃣ Gerekli Bağımlılıkları Yükleyin
pip install -r requirements.txt

4️⃣ Veri Setini İndirin

C-MAPSS veri setini indirmek ve uygun klasör yapısına yerleştirmek için:

python src/download_data.py


Bu adım sonunda data/ klasörü altında train_FD001.txt ve
test_FD001.txt dosyaları bulunmalıdır.

▶️ Çalıştırma
1️⃣ Modeli Eğitin

Autoencoder modeli yalnızca erken yaşam evresi verileri kullanılarak
eğitilir:

python src/train.py


Bu adım sonunda aşağıdaki dosyalar oluşturulur:

models/autoencoder.h5

models/scaler.pkl

models/threshold.npy

2️⃣ Offline Anomali Analizi

Eğitilen model ile test verisi üzerinde anomali analizi yapmak ve grafik
üretmek için:

python src/anomaly.py

3️⃣ Gradio Arayüzünü Başlatın

Etkileşimli arayüz üzerinden yaşam evrelerine göre anomali analizi yapmak
için:

python src/app.py


Terminalde verilen bağlantıyı tarayıcıda açarak uygulamayı kullanabilirsiniz.