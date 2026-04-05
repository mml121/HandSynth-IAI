# model.py — Lightweight CNN for hand gesture classification (0–5 fingers)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 64
NUM_CLASSES = 6  # 0, 1, 2, 3, 4, 5 fingers


def build_model():
    """Small CNN — 64x64 grayscale input, 6-class output."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model
