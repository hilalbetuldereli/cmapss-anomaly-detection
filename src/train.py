import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

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

# Sondaki boş kolonları at
df = df.iloc[:, :26]
df.columns = columns

print("Dataset yüklendi:", df.shape)

# Sadece Sensör Verileri

sensor_cols = [c for c in df.columns if "sensor" in c]
data = df[sensor_cols]

print("Sensör verisi şekli:", data.shape)

#  Normalizasyon

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


#  Autoencoder Modeli
input_dim = data_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(14, activation="relu")(input_layer)
encoded = Dense(7, activation="relu")(encoded)

decoded = Dense(14, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()


#  Model Eğitimi


early_stop = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

autoencoder.fit(
    data_scaled,
    data_scaled,
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)


# Modeli Kaydedelim


os.makedirs("models", exist_ok=True)
autoencoder.save("models/autoencoder.h5")

print("Model kaydedildi: models/autoencoder.h5")
