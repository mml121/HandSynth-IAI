# collect_data.py — Collect labeled hand gesture images for LEFT and RIGHT hands
#
# How it works:
#   1. MediaPipe detects up to 2 hands and identifies left/right
#   2. It crops the hand region, resizes to 64x64 grayscale
#   3. Saves images into dataset/<hand>/<finger_count>/ folders
#
# Usage:
#   python -m training.collect_data
#
# Controls:
#   SPACE  — start/stop collecting for the current gesture
#   0–5    — switch target class (which finger count you're showing)
#   L / R  — switch which hand you're recording
#   Q      — quit
#
# Aim for ~200–400 images per class per hand. Move your hand around, vary angles.

import os
import sys
import cv2
import mediapipe as mp
# Allow running directly: python training/collect_data.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from src.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

DATASET_DIR = os.path.join(_PROJECT_ROOT, "dataset")
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


def get_hand_dir(hand_name):
    """Return dataset path for a hand (left or right)."""
    return os.path.join(DATASET_DIR, hand_name.lower())


def count_existing(hand_name):
    """Count existing images per class for a hand."""
    counts = {}
    for i in range(6):
        class_dir = os.path.join(get_hand_dir(hand_name), str(i))
        if os.path.exists(class_dir):
            counts[i] = len(os.listdir(class_dir))
        else:
            counts[i] = 0
    return counts


def main():
    # Create dataset folders for both hands
    for hand in ("left", "right"):
        for i in range(6):
            os.makedirs(os.path.join(get_hand_dir(hand), str(i)), exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    target_class = 0
    target_hand = "right"  # which hand we're recording
    collecting = False
    img_count = {
        "left": count_existing("left"),
        "right": count_existing("right"),
    }

    print("=== Hand Gesture Data Collector ===")
    print("Keys: 0-5 = select class | L/R = select hand | SPACE = start/stop | Q = quit")
    print(f"Images will be saved to {DATASET_DIR}/<hand>/<class>/")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Find the target hand in detections
        detected_hands = {}  # label -> (index, landmarks, handedness)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_lm, hand_info) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # MediaPipe mirrors labels in selfie mode (flipped frame),
                # so "Right" from MP = user's right hand in a flipped image
                mp_label = hand_info.classification[0].label
                detected_hands[mp_label.lower()] = (idx, hand_lm, mp_label)

        # Process and draw all detected hands
        for label, (idx, hand_lm, mp_label) in detected_hands.items():
            fingers = count_fingers(hand_lm, mp_label)
            x1, y1, x2, y2 = get_hand_bbox(hand_lm, frame.shape)
            hand_roi = frame[y1:y2, x1:x2]

            is_target = (label == target_hand)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Draw bounding box — green if target & collecting, cyan if target, grey otherwise
            if is_target and collecting:
                box_color = (0, 255, 0)
            elif is_target:
                box_color = (255, 255, 0)
            else:
                box_color = (128, 128, 128)

            if hand_roi.size > 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                # Label which hand
                cv2.putText(frame, label.upper(), (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Save image if this is the target hand and we're collecting
            if is_target and collecting and hand_roi.size > 0:
                gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                hand_crop = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                save_dir = os.path.join(get_hand_dir(target_hand), str(target_class))
                save_path = os.path.join(save_dir, f"{img_count[target_hand][target_class]:04d}.jpg")
                cv2.imwrite(save_path, hand_crop)
                img_count[target_hand][target_class] += 1

        # Display info on frame
        status = "COLLECTING" if collecting else "PAUSED"
        color = (0, 0, 255) if collecting else (200, 200, 200)
        cv2.putText(frame, f"Hand: {target_hand.upper()} | Class: {target_class} | {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show counts for target hand
        y_pos = 65
        for i in range(6):
            marker = " <--" if i == target_class else ""
            cv2.putText(frame, f"  {i} fingers: {img_count[target_hand][i]} imgs{marker}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_pos += 25

        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            collecting = not collecting
            state = "STARTED" if collecting else "STOPPED"
            print(f"[{state}] {target_hand.upper()} hand, class {target_class} — {img_count[target_hand][target_class]} images so far")
        elif ord("0") <= key <= ord("5"):
            target_class = key - ord("0")
            collecting = False
            print(f"Switched to class {target_class} ({target_class} fingers)")
        elif key == ord("l"):
            target_hand = "left"
            collecting = False
            print(f"Switched to LEFT hand")
        elif key == ord("r"):
            target_hand = "right"
            collecting = False
            print(f"Switched to RIGHT hand")

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("\n=== Collection complete ===")
    for hand in ("left", "right"):
        print(f"\n  {hand.upper()} hand:")
        for i in range(6):
            print(f"    {i} fingers: {img_count[hand][i]} images")


if __name__ == "__main__":
    main()
