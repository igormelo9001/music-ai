import os
import numpy as np
import tensorflow as tf
from models.model import create_model

PROCESSED_DATA_PATH = 'data/processed'

def load_data():
    X = []
    for file_name in os.listdir(PROCESSED_DATA_PATH):
        if file_name.endswith('.npy'):
            data = np.load(os.path.join(PROCESSED_DATA_PATH, file_name))
            X.append(data)
    return np.array(X)

def train():
    X = load_data()
    input_shape = X.shape[1:]
    model = create_model(input_shape)
    model.fit(X, X, epochs=100, batch_size=16)
    model.save('models/music_gen_model.h5')

if __name__ == "__main__":
    train()
