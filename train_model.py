# train_model.py — Train the CNN on collected hand gesture data
#
# Usage:
#   python train_model.py
#
# Expects dataset/ folder with subfolders 0–5 (from collect_data.py).
# Saves trained model to gesture_model.keras

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import build_model, IMG_SIZE, NUM_CLASSES

DATASET_DIR = "dataset"
MODEL_PATH = "gesture_model.keras"
EPOCHS = 10
BATCH_SIZE = 32


def load_dataset():
    """Load images and labels from dataset/ folder."""
    images = []
    labels = []

    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(DATASET_DIR, str(class_id))
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping class {class_id}")
            continue

        files = os.listdir(class_dir)
        print(f"  Class {class_id} ({class_id} fingers): {len(files)} images")

        for fname in files:
            path = os.path.join(class_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_id)

    images = np.array(images, dtype=np.float32) / 255.0  # normalize to 0–1
    images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)    # add channel dim
    labels = to_categorical(np.array(labels), NUM_CLASSES)

    return images, labels


def main():
    print("=== Loading dataset ===")
    X, y = load_dataset()
    print(f"\nTotal: {len(X)} images\n")

    if len(X) < 30:
        print("Not enough data! Collect at least 50 images per class.")
        print("Run: python collect_data.py")
        return

    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # Build and compile
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    print()

    # Train
    print("=== Training ===")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
    )

    # Evaluate
    print("\n=== Evaluation ===")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.1%}")
    print(f"Test loss:     {loss:.4f}")

    # Save
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
