# gesture.py — Two-hand detection using MediaPipe + CNN gesture classification
#
# Optimizations:
#   - Uses TFLite for ~10x faster inference than model.predict()
#   - Smooths predictions over a sliding window to eliminate flicker
#   - Confidence threshold rejects low-quality predictions

import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from src.config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_DIR, "models", "gesture_model.keras")
TFLITE_PATH = os.path.join(_DIR, "models", "gesture_model.tflite")
IMG_SIZE = 64
SMOOTHING_WINDOW = 5       # majority vote over last N frames per hand
CONFIDENCE_THRESHOLD = 0.6  # reject predictions below this

# ── Load the trained model: prefer TFLite, fall back to Keras, then rules ──
_tflite_interpreter = None
_tflite_input_details = None
_tflite_output_details = None
_cnn_model = None

if os.path.exists(TFLITE_PATH):
    import tensorflow as tf
    _tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    _tflite_interpreter.allocate_tensors()
    _tflite_input_details = _tflite_interpreter.get_input_details()
    _tflite_output_details = _tflite_interpreter.get_output_details()
    print(f"[gesture] Loaded TFLite model from {TFLITE_PATH} (fast inference)")
elif os.path.exists(MODEL_PATH):
    import tensorflow as tf
    _cnn_model = tf.keras.models.load_model(MODEL_PATH)
    # Warm up the model with a dummy input to avoid first-call lag
    _dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    _cnn_model(_dummy, training=False)
    print(f"[gesture] Loaded Keras model from {MODEL_PATH}")
    print(f"          Run convert_to_tflite.py for faster inference")
else:
    print(f"[gesture] No model found — using rule-based finger counting")
    print(f"          Run collect_data.py then train_model.py to train the CNN")

# Landmark indices (used by rule-based fallback)
FINGER_TIPS = [
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]
FINGER_PIPS = [
    mp_hands.HandLandmark.INDEX_FINGER_PIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    mp_hands.HandLandmark.RING_FINGER_PIP,
    mp_hands.HandLandmark.PINKY_PIP,
]
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
THUMB_IP = mp_hands.HandLandmark.THUMB_IP


def _get_hand_bbox(hand_landmarks, frame_shape):
    """Get bounding box from hand landmarks with padding."""
    h, w = frame_shape[:2]
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]

    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    pad_x = int((x_max - x_min) * 0.2)
    pad_y = int((y_max - y_min) * 0.2)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    return x_min, y_min, x_max, y_max


def _majority_vote(history):
    """Return the most common value in the deque."""
    if not history:
        return -1
    counts = {}
    for v in history:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


class HandDetector:
    """Detects up to 2 hands. Uses TFLite/CNN for classification if available,
    otherwise falls back to rule-based finger counting.
    Smooths results over a sliding window to reduce flicker."""

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        # Smoothing buffers per hand
        self._left_history = deque(maxlen=SMOOTHING_WINDOW)
        self._right_history = deque(maxlen=SMOOTHING_WINDOW)

    def process_frame(self, frame):
        """Process a BGR frame (already flipped/mirrored).

        Returns:
            frame: annotated BGR frame
            left_fingers: int (-1 if no left hand detected)
            right_fingers: int (-1 if no right hand detected)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        raw_left = -1
        raw_right = -1

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                mp_label = handedness_info.classification[0].label

                # Classify BEFORE drawing landmarks (clean image for CNN)
                if _tflite_interpreter is not None:
                    count = self._classify_tflite(hand_landmarks, frame)
                elif _cnn_model is not None:
                    count = self._classify_cnn(hand_landmarks, frame)
                else:
                    count = self._count_fingers_rules(hand_landmarks, mp_label)

                # Draw landmarks AFTER classification
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # MediaPipe already assumes mirrored input, so its labels
                # match the user's actual hands — no swap needed.
                if mp_label == "Right":
                    raw_right = count
                else:
                    raw_left = count

        # Update smoothing buffers
        if raw_left >= 0:
            self._left_history.append(raw_left)
        else:
            self._left_history.clear()

        if raw_right >= 0:
            self._right_history.append(raw_right)
        else:
            self._right_history.clear()

        # Return smoothed results
        left_fingers = _majority_vote(self._left_history) if raw_left >= 0 else -1
        right_fingers = _majority_vote(self._right_history) if raw_right >= 0 else -1

        return frame, left_fingers, right_fingers

    def _classify_tflite(self, hand_landmarks, frame):
        """Crop hand region, run TFLite inference (fastest path)."""
        input_tensor = self._prepare_input(hand_landmarks, frame)
        if input_tensor is None:
            return 0

        _tflite_interpreter.set_tensor(_tflite_input_details[0]['index'], input_tensor)
        _tflite_interpreter.invoke()
        prediction = _tflite_interpreter.get_tensor(_tflite_output_details[0]['index'])[0]

        confidence = float(np.max(prediction))
        if confidence < CONFIDENCE_THRESHOLD:
            return 0
        return int(np.argmax(prediction))

    def _classify_cnn(self, hand_landmarks, frame):
        """Crop hand region, feed through Keras model using direct call (not .predict())."""
        input_tensor = self._prepare_input(hand_landmarks, frame)
        if input_tensor is None:
            return 0

        # Direct __call__ is much faster than .predict() for single images
        prediction = _cnn_model(input_tensor, training=False).numpy()[0]

        confidence = float(np.max(prediction))
        if confidence < CONFIDENCE_THRESHOLD:
            return 0
        return int(np.argmax(prediction))

    def _prepare_input(self, hand_landmarks, frame):
        """Crop, grayscale, resize, normalize — shared by TFLite and Keras paths."""
        x1, y1, x2, y2 = _get_hand_bbox(hand_landmarks, frame.shape)
        hand_roi = frame[y1:y2, x1:x2]

        if hand_roi.size == 0:
            return None

        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    def _count_fingers_rules(self, hand, mp_label):
        """Fallback: count extended fingers using landmark positions."""
        lm = hand.landmark
        count = 0

        if mp_label == "Right":
            if lm[THUMB_TIP].x < lm[THUMB_IP].x:
                count += 1
        else:
            if lm[THUMB_TIP].x > lm[THUMB_IP].x:
                count += 1

        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
            if lm[tip].y < lm[pip].y:
                count += 1

        return count

    def release(self):
        self.hands.close()
