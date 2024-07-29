import os
import librosa
import numpy as np

PROCESSED_DATA_PATH = 'data/processed'

def preprocess_file(file_path):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    file_name = os.path.basename(file_path).replace('.wav', '.npy')
    np.save(os.path.join(PROCESSED_DATA_PATH, file_name), mfccs)

def preprocess():
    RAW_DATA_PATH = 'data/raw'
    for file_name in os.listdir(RAW_DATA_PATH):
        if file_name.endswith('.wav'):
            file_path = os.path.join(RAW_DATA_PATH, file_name)
            preprocess_file(file_path)

if __name__ == "__main__":
    preprocess()
