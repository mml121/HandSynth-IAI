# gesture.py — Two-hand detection using MediaPipe + CNN gesture classification

import os
import cv2
import numpy as np
import mediapipe as mp
from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MODEL_PATH = "gesture_model.keras"
IMG_SIZE = 64

# ── Load the trained CNN if available, otherwise fall back to rule-based ──
_cnn_model = None
if os.path.exists(MODEL_PATH):
    from tensorflow.keras.models import load_model
    _cnn_model = load_model(MODEL_PATH)
    print(f"[gesture] Loaded CNN model from {MODEL_PATH}")
else:
    print(f"[gesture] No model found at {MODEL_PATH} — using rule-based finger counting")
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


class HandDetector:
    """Detects up to 2 hands. Uses CNN for classification if available,
    otherwise falls back to rule-based finger counting."""

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    def process_frame(self, frame):
        """Process a BGR frame (already flipped/mirrored).

        Returns:
            frame: annotated BGR frame
            left_fingers: int (-1 if no left hand detected)
            right_fingers: int (-1 if no right hand detected)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        left_fingers = -1
        right_fingers = -1

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                mp_label = handedness_info.classification[0].label

                # Classify gesture: CNN or rule-based
                if _cnn_model is not None:
                    count = self._classify_cnn(hand_landmarks, frame)
                else:
                    count = self._count_fingers_rules(hand_landmarks, mp_label)

                # Swap labels: mirrored frame means MP "Right" = user's Left
                if mp_label == "Right":
                    left_fingers = count
                else:
                    right_fingers = count

        return frame, left_fingers, right_fingers

    def _classify_cnn(self, hand_landmarks, frame):
        """Crop hand region, feed through CNN, return predicted class."""
        x1, y1, x2, y2 = _get_hand_bbox(hand_landmarks, frame.shape)
        hand_roi = frame[y1:y2, x1:x2]

        if hand_roi.size == 0:
            return 0

        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        prediction = _cnn_model.predict(input_tensor, verbose=0)
        return int(np.argmax(prediction))

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
