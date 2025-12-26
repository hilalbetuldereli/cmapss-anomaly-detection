# train.py
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "data/train_FD001.txt"

columns = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(DATA_PATH, sep=" ", header=None)
df = df.iloc[:, :26]
df.columns = columns

print("Dataset yüklendi:", df.shape)

# SADECE ERKEN YAŞAM EVRESİ
early_parts = []
for unit_id, g in df.groupby("unit"):
    max_cycle = g["cycle"].max()
    early_end = int(0.3 * max_cycle)
    early_parts.append(g[g["cycle"] <= early_end])

df_early = pd.concat(early_parts)
print("Erken evre veri şekli:", df_early.shape)

sensor_cols = [c for c in df.columns if "sensor" in c]
data = df_early[sensor_cols]

# Normalizasyon
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Scaler kaydet
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")


# AUTOENCODER

input_dim = data_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(14, activation="relu")(input_layer)
encoded = Dense(7, activation="relu")(encoded)
decoded = Dense(14, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

autoencoder.fit(
    data_scaled,
    data_scaled,
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

autoencoder.save("models/autoencoder.h5")

# THRESHOLD (TRAIN)
recon = autoencoder.predict(data_scaled)
mse_train = np.mean(np.square(data_scaled - recon), axis=1)

threshold = np.percentile(mse_train, 95)
np.save("models/threshold.npy", threshold)

print(f"Model, scaler ve threshold kaydedildi | Threshold: {threshold:.6f}")
