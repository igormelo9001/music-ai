import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(256, activation='relu'),
        layers.Dense(input_shape[0] * input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
