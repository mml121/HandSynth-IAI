# train_model.py — Train the CNN on collected hand gesture data
#
# Usage:
#   python train_model.py
#
# Expects dataset/ folder with subfolders 0–5 (from collect_data.py).
# Saves trained model to gesture_model.keras + gesture_model.tflite
#
# Improvements over v1:
#   - Data augmentation (rotation, shift, zoom, brightness)
#   - Learning rate reduction on plateau
#   - Early stopping to prevent overfitting
#   - Auto-converts to TFLite for fast inference

import os
import sys
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

# Allow running directly: python training/train_model.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, IMG_SIZE, NUM_CLASSES

DATASET_DIR = os.path.join(_PROJECT_ROOT, "dataset")
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "gesture_model.keras")
TFLITE_PATH = os.path.join(_PROJECT_ROOT, "models", "gesture_model.tflite")
EPOCHS = 15
BATCH_SIZE = 32


def load_dataset():
    """Load images and labels from dataset/<hand>/<class>/ folders."""
    images = []
    labels = []

    hands_found = []
    for hand in ("left", "right"):
        hand_dir = os.path.join(DATASET_DIR, hand)
        if os.path.exists(hand_dir):
            hands_found.append(hand)

    # Support both old flat layout (dataset/0/) and new layout (dataset/left/0/)
    if not hands_found:
        search_dirs = [("", DATASET_DIR)]
    else:
        search_dirs = [(h, os.path.join(DATASET_DIR, h)) for h in hands_found]

    for hand_label, base_dir in search_dirs:
        prefix = f"  [{hand_label.upper()}] " if hand_label else "  "
        for class_id in range(NUM_CLASSES):
            class_dir = os.path.join(base_dir, str(class_id))
            if not os.path.exists(class_dir):
                continue

            files = os.listdir(class_dir)
            print(f"{prefix}Class {class_id} ({class_id} fingers): {len(files)} images")

            for fname in files:
                path = os.path.join(class_dir, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(class_id)

    images = np.array(images, dtype=np.float32) / 255.0
    images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = to_categorical(np.array(labels), NUM_CLASSES)

    return images, labels


def convert_to_tflite(model):
    """Convert Keras model to TFLite for faster inference."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"TFLite model saved to {TFLITE_PATH} ({size_kb:.0f} KB)")


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

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ]

    # Train
    print("=== Training ===")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

    # Evaluate
    print("\n=== Evaluation ===")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.1%}")
    print(f"Test loss:     {loss:.4f}")

    # Save Keras model
    model.save(MODEL_PATH)
    print(f"\nKeras model saved to {MODEL_PATH}")

    # Convert to TFLite for fast inference
    print("\n=== Converting to TFLite ===")
    convert_to_tflite(model)

    print("\nDone! The app will automatically use the TFLite model for faster inference.")


if __name__ == "__main__":
    main()
