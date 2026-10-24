import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

def load_fer2013(csv_path="data/fer2013.csv"):
    """
    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test)
    X shape: (N, 48, 48, 1), normalized to [0,1]
    y: one-hot vectors (7 classes)
    """
    df = pd.read_csv(csv_path)
    # pixels -> arrays
    pixels = df['pixels'].tolist()
    X = np.array([np.fromstring(p, dtype=np.uint8, sep=' ') for p in pixels])
    X = X.reshape(-1, 48, 48, 1).astype('float32') / 255.0
    y = to_categorical(df['emotion'], num_classes=7)

    usage = df['Usage'].values
    X_train = X[usage == 'Training']; y_train = y[usage == 'Training']
    X_val   = X[usage == 'PublicTest']; y_val   = y[usage == 'PublicTest']
    X_test  = X[usage == 'PrivateTest']; y_test  = y[usage == 'PrivateTest']

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
