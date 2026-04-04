# collect_data.py — Auto-collect labeled hand gesture images using MediaPipe
#
# How it works:
#   1. MediaPipe detects your hand and counts fingers (the old way)
#   2. It crops the hand region, resizes to 64x64 grayscale
#   3. Saves images into dataset/<finger_count>/ folders
#
# Usage:
#   python collect_data.py
#
# Controls:
#   SPACE  — start/stop collecting for the current gesture
#   0–5    — switch target class (which finger count you're showing)
#   Q      — quit
#
# Aim for ~200–400 images per class. Move your hand around, vary angles.

import os
import cv2
import mediapipe as mp
import numpy as np
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

DATASET_DIR = "dataset"
IMG_SIZE = 64
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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


def count_fingers(hand, mp_label):
    """Count extended fingers on a hand."""
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


def get_hand_bbox(hand_landmarks, frame_shape):
    """Get bounding box of hand from landmarks, with padding."""
    h, w = frame_shape[:2]
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]

    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # Add 20% padding
    pad_x = int((x_max - x_min) * 0.2)
    pad_y = int((y_max - y_min) * 0.2)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    return x_min, y_min, x_max, y_max


def main():
    # Create dataset folders
    for i in range(6):
        os.makedirs(os.path.join(DATASET_DIR, str(i)), exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    target_class = 0    # which class we're collecting
    collecting = False   # whether we're saving images
    img_count = {i: len(os.listdir(os.path.join(DATASET_DIR, str(i)))) for i in range(6)}

    print("=== Hand Gesture Data Collector ===")
    print("Keys: 0-5 = select class | SPACE = start/stop collecting | Q = quit")
    print(f"Images will be saved to {DATASET_DIR}/")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected_fingers = -1
        hand_crop = None

        if results.multi_hand_landmarks and results.multi_handedness:
            hand = results.multi_hand_landmarks[0]
            mp_label = results.multi_handedness[0].classification[0].label

            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            detected_fingers = count_fingers(hand, mp_label)

            # Crop hand region
            x1, y1, x2, y2 = get_hand_bbox(hand, frame.shape)
            hand_roi = frame[y1:y2, x1:x2]

            if hand_roi.size > 0:
                gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                hand_crop = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

                # Draw bounding box
                color = (0, 255, 0) if collecting else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Save image if collecting
            if collecting and hand_crop is not None:
                save_path = os.path.join(DATASET_DIR, str(target_class), f"{img_count[target_class]:04d}.jpg")
                cv2.imwrite(save_path, hand_crop)
                img_count[target_class] += 1

        # Display info on frame
        status = "COLLECTING" if collecting else "PAUSED"
        color = (0, 0, 255) if collecting else (200, 200, 200)
        cv2.putText(frame, f"Class: {target_class} fingers | {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Detected: {detected_fingers} fingers", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Show counts for all classes
        y_pos = 100
        for i in range(6):
            marker = " <--" if i == target_class else ""
            cv2.putText(frame, f"  {i} fingers: {img_count[i]} imgs{marker}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_pos += 25

        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            collecting = not collecting
            state = "STARTED" if collecting else "STOPPED"
            print(f"[{state}] Collecting class {target_class} — {img_count[target_class]} images so far")
        elif ord("0") <= key <= ord("5"):
            target_class = key - ord("0")
            collecting = False
            print(f"Switched to class {target_class} ({target_class} fingers)")

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("\n=== Collection complete ===")
    for i in range(6):
        print(f"  {i} fingers: {img_count[i]} images")


if __name__ == "__main__":
    main()
