# anomaly.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model

DATA_PATH = "data/train_FD001.txt"

columns = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(DATA_PATH, sep=" ", header=None)
df = df.iloc[:, :26]
df.columns = columns

sensor_cols = [c for c in df.columns if "sensor" in c]
data = df[sensor_cols]

scaler = joblib.load("models/scaler.pkl")
data_scaled = scaler.transform(data)

model = load_model("models/autoencoder.h5", compile=False)
threshold = np.load("models/threshold.npy")

recon = model.predict(data_scaled)
mse = np.mean(np.square(data_scaled - recon), axis=1)

anomalies = mse > threshold

print("Toplam örnek:", len(mse))
print("Anomali sayısı:", anomalies.sum())

plt.figure(figsize=(10, 4))
plt.plot(mse, label="Reconstruction Error")
plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
plt.legend()
plt.title("CMAPSS – Anomali Tespiti")
plt.tight_layout()
plt.show()
