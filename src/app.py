# app.py
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model

MODEL_PATH = "models/autoencoder.h5"
THRESHOLD_PATH = "models/threshold.npy"
SCALER_PATH = "models/scaler.pkl"

model = load_model(MODEL_PATH, compile=False)
THRESHOLD = np.load(THRESHOLD_PATH)
scaler = joblib.load(SCALER_PATH)

# MOTOR-BAZLI YAŞAM EVRESİ
def split_by_life_stage(df, mode):
    parts = []
    for unit_id, g in df.groupby("unit"):
        max_cycle = g["cycle"].max()
        early_end = int(0.3 * max_cycle)
        late_start = int(0.7 * max_cycle)

        if mode == "Erken":
            part = g[g["cycle"] <= early_end]
        elif mode == "Orta":
            part = g[(g["cycle"] > early_end) & (g["cycle"] <= late_start)]
        else:
            part = g[g["cycle"] > late_start]

        parts.append(part)
    return pd.concat(parts)


def detect_anomaly(file, mode):
    columns = (
        ["unit", "cycle", "op1", "op2", "op3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv(file.name, sep=" ", header=None)
    df = df.iloc[:, :26]
    df.columns = columns

    df_stage = split_by_life_stage(df, mode)

    X = df_stage.iloc[:, 5:26].values
    X_scaled = scaler.transform(X)

    recon = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - recon), axis=1)

    anomalies = mse > THRESHOLD

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mse, label="Reconstruction Error")
    ax.axhline(THRESHOLD, color="red", linestyle="--", label="Threshold")
    ax.scatter(np.where(anomalies), mse[anomalies], s=10, color="red")
    ax.set_title(f"{mode} Yaşam Evresi – Anomali Analizi")
    ax.set_xlabel("Zaman Adımı (Cycle / Sample Index)")
    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.legend()

    return fig, (
        f"{mode} evre\n"
        f"Motor sayısı: {df_stage['unit'].nunique()}\n"
        f"Toplam örnek: {len(mse)}\n"
        f"Anomali sayısı: {anomalies.sum()}\n"
        f"Anomali oranı: %{(anomalies.mean()*100):.2f}"
    )

interface = gr.Interface(
    fn=detect_anomaly,
    inputs=[
        gr.File(label="CMAPSS txt dosyası"),
        gr.Dropdown(["Erken", "Orta", "Geç"], value="Erken")
    ],
    outputs=[gr.Plot(), gr.Textbox()],
    title="CMAPSS Motor Anomali Tespiti",
    description="Autoencoder yalnızca erken evre ile eğitilmiştir (sağlıklı motor).",
    examples=[
    ["data/test_FD001.txt", "Erken"],
    ["data/test_FD001.txt", "Orta"],
    ["data/test_FD001.txt", "Geç"]
]

)

if __name__ == "__main__":
    interface.launch(share=True)
