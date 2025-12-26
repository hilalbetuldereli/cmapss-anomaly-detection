import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# CONFIG
MODEL_PATH = "models/autoencoder.h5"
THRESHOLD_PERCENTILE = 95

# LOAD MODEL
model = load_model(MODEL_PATH, compile=False)

# CORE FUNCTION
def detect_anomaly(file, mode):
    df = pd.read_csv(file.name, sep=" ", header=None)
    df = df.dropna(axis=1)

    # MODE'A GÖRE FARKLI BÖLÜMLER
    if mode == "Erken":
        df = df.iloc[:3000]
    elif mode == "Orta":
        df = df.iloc[3000:9000]
    elif mode == "Geç":
        df = df.iloc[9000:]

    X = df.iloc[:, 5:26].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_pred = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

    threshold = np.percentile(mse, THRESHOLD_PERCENTILE)
    anomalies = mse > threshold
    anomaly_count = np.sum(anomalies)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mse, label="Reconstruction Error")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax.scatter(
        np.where(anomalies),
        mse[anomalies],
        color="red",
        s=10,
        label="Anomaly"
    )
    ax.set_title(f"Anomali Tespiti - {mode} Aşama")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("MSE")
    ax.legend()

    return fig, f"{mode} aşama – Anomali sayısı: {anomaly_count}"


# GRADIO UI
interface = gr.Interface(
    fn=detect_anomaly,
    inputs=[
        gr.File(label="CMAPSS txt dosyasını yükle"),
        gr.Dropdown(
            ["Erken", "Orta", "Geç"],
            value="Erken",
            label="Motor Yaşam Aşaması"
        )
    ],
    outputs=[
        gr.Plot(label="Anomali Grafiği"),
        gr.Textbox(label="Sonuç")
    ],
    title="CMAPSS Anomali Tespiti",
    description="Autoencoder ile uçak motoru sensör verilerinde anomali tespiti",
    examples=[
        ["data/train_FD001.txt", "Erken"],
        ["data/train_FD001.txt", "Orta"],
        ["data/train_FD001.txt", "Geç"]
    ]
)


if __name__ == "__main__":
    interface.launch(share=True)
