import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Dataseti Yükle

DATA_PATH = "data/train_FD001.txt"

columns = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(
    DATA_PATH,
    sep=" ",
    header=None
)

df = df.iloc[:, :26]
df.columns = columns

sensor_cols = [c for c in df.columns if "sensor" in c]
data = df[sensor_cols]

print("Dataset yüklendi:", data.shape)

# Normalizasyon

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Modeli Yükle

model = load_model("models/autoencoder.h5", compile=False)

print("Model yüklendi.")

# Reconstruction Error

reconstructions = model.predict(data_scaled)
reconstruction_errors = np.mean(
    np.square(data_scaled - reconstructions),
    axis=1
)

# Threshold Belirleme
# İstatistiksel yöntem: mean + 3*std

threshold = np.mean(reconstruction_errors) + 3 * np.std(reconstruction_errors)

print(f"Anomali Threshold: {threshold:.6f}")

# Anomali Etiketi

anomalies = reconstruction_errors > threshold
num_anomalies = np.sum(anomalies)

print(f"Tespit edilen anomali sayısı: {num_anomalies}")

# Görselleştirme

plt.figure(figsize=(10, 4))
plt.plot(reconstruction_errors, label="Reconstruction Error")
plt.axhline(threshold, color="r", linestyle="--", label="Threshold")
plt.xlabel("Örnek İndeksi")
plt.ylabel("Hata")
plt.title("Anomali Tespiti (Autoencoder)")
plt.legend()
plt.tight_layout()
plt.show()
