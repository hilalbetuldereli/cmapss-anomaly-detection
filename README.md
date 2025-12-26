🛩️ CMAPSS Uçak Motoru Verilerinde Anomali Tespiti

Autoencoder + Gradio

📌 Proje Tanımı

Bu projede, NASA tarafından yayınlanan C-MAPSS uçak motoru sensör verileri kullanılarak
denetimsiz (unsupervised) anomali tespiti yapılmıştır.

Amaç, uçak motorlarında oluşabilecek arızaları erken aşamada fark edebilmek ve bakım planlamasına katkı sağlamaktır.
Model çıktıları Gradio tabanlı web arayüzü ile görselleştirilmiştir.

🎯 Neden Bu Proje?

Gerçek sistemlerde arıza etiketleri çoğu zaman yoktur

Bu nedenle denetimsiz anomali tespiti yaygın kullanılır

NASA C-MAPSS, literatürde bu problem için en çok kullanılan veri setlerinden biridir

📊 Veri Seti

Kaynak: NASA C-MAPSS (Kaggle)

Dosya: train_FD001.txt

Yapı:

3 operasyonel ayar

21 sensör ölçümü

Her satır: bir motorun bir çalışma döngüsü

🧠 Kullanılan Yöntem

Autoencoder (Denetimsiz Öğrenme)

Model normal çalışma davranışını öğrenir

Girdi ile çıktı arasındaki fark (Reconstruction Error) hesaplanır

Anomali Kriteri

Hata metriği: Mean Squared Error (MSE)

Anomali eşiği: %95 persentil

Düşük hata → Normal

Yüksek hata → Anomali

📈 Model Değerlendirmesi

Bu çalışma etiketsiz olduğu için:

Accuracy, Precision, Recall gibi metrikler kullanılmamıştır

Değerlendirme şu şekilde yapılmıştır:

Reconstruction Error dağılımı

Anomali yoğunluğu

Grafiksel analiz

Bu yaklaşım, denetimsiz anomali tespitinde standarttır.

🖥️ Gradio Arayüzü

Proje, Gradio ile etkileşimli hale getirilmiştir.

Arayüzde:

Kullanıcı .txt dosyası yükleyebilir

Anomali grafiği görüntülenir

Toplam anomali sayısı gösterilir

Hazır örnek dosyalarla hızlı demo yapılabilir

▶️ Çalıştırma
pip install -r requirements.txt
python src/app.py