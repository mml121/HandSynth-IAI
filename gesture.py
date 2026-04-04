# gesture.py — Two-hand detection and finger counting using MediaPipe

import cv2
import mediapipe as mp
from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for finger tips and PIPs
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


class HandDetector:
    """Detects up to 2 hands and counts fingers on each."""

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
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # MediaPipe label (on a mirrored image, labels are swapped)
                mp_label = handedness_info.classification[0].label
                count = self._count_fingers(hand_landmarks, mp_label)

                # Swap labels: mirrored frame means MP "Right" = user's Left
                if mp_label == "Right":
                    left_fingers = count
                else:
                    right_fingers = count

        return frame, left_fingers, right_fingers

    def _count_fingers(self, hand, mp_label):
        """Count extended fingers."""
        lm = hand.landmark
        count = 0

        # Thumb — direction depends on which hand MediaPipe sees
        if mp_label == "Right":
            if lm[THUMB_TIP].x < lm[THUMB_IP].x:
                count += 1
        else:
            if lm[THUMB_TIP].x > lm[THUMB_IP].x:
                count += 1

        # Four fingers — tip above PIP means extended
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
            if lm[tip].y < lm[pip].y:
                count += 1

        return count

    def release(self):
        self.hands.close()
