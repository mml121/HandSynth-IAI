# app.py — PyQt6 UI · minimal black aesthetic

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QHBoxLayout, QVBoxLayout, QSpacerItem, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QFontDatabase

from src.gesture import HandDetector, has_cnn_model
from src.audio import LiveSynth
from src.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, UI_FPS,
    RIGHT_HAND_MAP, LEFT_HAND_MAP, DEFAULT_VOLUME,
    get_frequency,
)

# ────────────────────────────────────────────────────────────────
#  STYLESHEET
# ────────────────────────────────────────────────────────────────
NEON = "#00FF41"
NEON_DIM = "#00AA2A"
NEON_DARK = "#005F15"
NEON_FAINT = "#003D0F"

QSS = f"""
QMainWindow, QWidget {{
    background-color: #000;
    color: {NEON};
}}
QLabel {{
    color: {NEON};
    background: transparent;
}}
QLabel#dim {{
    color: {NEON_DARK};
}}
QLabel#accent {{
    color: {NEON_DIM};
}}
QPushButton {{
    background-color: transparent;
    color: {NEON_DIM};
    border: 1px solid {NEON_DARK};
    padding: 10px 32px;
    font-size: 11px;
    letter-spacing: 3px;
}}
QPushButton:hover {{
    color: {NEON};
    border-color: {NEON};
}}
QPushButton:pressed {{
    color: #fff;
    border-color: {NEON};
    background-color: {NEON_FAINT};
}}
QPushButton:disabled {{
    color: {NEON_FAINT};
    border-color: #0a1f0a;
}}
QPushButton#active {{
    color: {NEON};
    border-color: {NEON};
}}
QSlider::groove:horizontal {{
    background: {NEON_FAINT};
    height: 2px;
}}
QSlider::handle:horizontal {{
    background: {NEON};
    width: 10px;
    height: 10px;
    margin: -5px 0;
    border-radius: 5px;
}}
QSlider::sub-page:horizontal {{
    background: {NEON_DARK};
    height: 2px;
}}
"""


# ────────────────────────────────────────────────────────────────
#  CAMERA THREAD
# ────────────────────────────────────────────────────────────────
class CameraWorker(QThread):
    """Captures frames and detects gestures in a background thread.

    Uses a frame-skip strategy: if processing is slow, it grabs (discards)
    buffered frames so the UI always shows the most recent camera image.
    """
    frame_ready = pyqtSignal(object, int, int)

    def __init__(self, use_cnn=False):
        super().__init__()
        self._running = False
        self._use_cnn = use_cnn
        self._detector = None

    def set_use_cnn(self, value):
        self._use_cnn = value
        if self._detector is not None:
            self._detector.use_cnn = value

    def run(self):
        import time
        self._running = True
        detector = HandDetector(use_cnn=self._use_cnn)
        self._detector = detector
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize internal buffer lag

        target_interval = 1.0 / UI_FPS

        while self._running and cap.isOpened():
            t0 = time.perf_counter()

            # Grab the latest frame (skip buffered stale frames)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, left_f, right_f = detector.process_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(rgb, left_f, right_f)

            # Throttle to target FPS to avoid overloading the UI
            elapsed = time.perf_counter() - t0
            remaining = target_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        cap.release()
        detector.release()

    def stop(self):
        self._running = False
        self.wait()


# ────────────────────────────────────────────────────────────────
#  CIRCULAR NOTE INDICATOR (custom painted widget)
# ────────────────────────────────────────────────────────────────
class NoteCircle(QWidget):
    """A thin circle with the current note rendered inside — the centrepiece."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.note = ""
        self.sub = ""          # small text below note (waveform)
        self.active = False
        self.setFixedSize(160, 160)

    def set_state(self, note: str, sub: str, active: bool):
        self.note = note
        self.sub = sub
        self.active = active
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width() // 2, self.height() // 2
        radius = 65

        # ── Outer glow ring (faint) when active ──────────────
        if self.active:
            glow_pen = QPen(QColor(0, 255, 65, 30))
            glow_pen.setWidthF(6)
            p.setPen(glow_pen)
            p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # ── Circle ────────────────────────────────────────────
        pen = QPen(QColor(NEON) if self.active else QColor(NEON_DARK))
        pen.setWidthF(1.5)
        p.setPen(pen)
        p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # ── Note name ─────────────────────────────────────────
        if self.note:
            p.setPen(QColor(NEON))
            font = QFont("Consolas", 28)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
            p.setFont(font)
            p.drawText(self.rect().adjusted(0, -14, 0, 0), Qt.AlignmentFlag.AlignCenter, self.note)

            # ── Sub-label (waveform) ──────────────────────────
            p.setPen(QColor(NEON_DIM))
            sub_font = QFont("Consolas", 9)
            sub_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
            p.setFont(sub_font)
            p.drawText(self.rect().adjusted(0, 30, 0, 0), Qt.AlignmentFlag.AlignCenter, self.sub.upper())
        else:
            p.setPen(QColor(NEON_DARK))
            font = QFont("Consolas", 9)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "SILENT")

        p.end()


# ────────────────────────────────────────────────────────────────
#  HELPER: create a styled label
# ────────────────────────────────────────────────────────────────
def _label(text="", size=11, color="#fff", spacing=3, bold=False, align=Qt.AlignmentFlag.AlignCenter):
    lbl = QLabel(text)
    font = QFont("Consolas", size)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    if bold:
        font.setBold(True)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    lbl.setAlignment(align)
    return lbl


# ────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ────────────────────────────────────────────────────────────────
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HAND  SYNTH")
        self.setMinimumSize(1100, 650)
        self.resize(1200, 700)
        self.setStyleSheet(QSS)

        # State
        self._synth = LiveSynth()
        self._synth.start()
        self._cam_worker = None
        self._use_cnn = False
        self._cur_note = None
        self._cur_freq = 0.0
        self._cur_waveform = "sine"
        self._cur_octave = 4

        self._build_ui()

    # ────────────────────────────────────────────────────────
    #  BUILD LAYOUT
    # ────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(30, 30, 30, 30)
        root.setSpacing(30)

        # ── LEFT: Camera feed ─────────────────────────────────
        cam_container = QVBoxLayout()
        cam_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._cam_label = QLabel("INSERT\nCOIN")
        self._cam_label.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setStyleSheet(
            f"border: 1px solid {NEON_DARK}; color: {NEON_DARK}; font: 11px Consolas; letter-spacing: 3px;"
        )
        cam_container.addWidget(self._cam_label)
        root.addLayout(cam_container, stretch=3)

        # ── RIGHT: Controls panel ─────────────────────────────
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.setSpacing(8)

        # ── Title ─────────────────────────────────────────────
        title = _label("HAND", size=26, color=NEON, spacing=12, bold=True)
        right.addWidget(title)
        subtitle = _label("SYNTH", size=26, color=NEON, spacing=12, bold=True)
        right.addWidget(subtitle)

        right.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Right hand info
        self._rh_label = _label("RIGHT HAND", size=9, color=NEON_DARK, spacing=4)
        right.addWidget(self._rh_label)
        self._rh_status = _label("—", size=11, color=NEON_DIM)
        right.addWidget(self._rh_status)

        right.addSpacing(20)

        # Note circle (centrepiece)
        circle_row = QHBoxLayout()
        circle_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._note_circle = NoteCircle()
        circle_row.addWidget(self._note_circle)
        right.addLayout(circle_row)

        # Frequency display
        self._freq_label = _label("", size=10, color=NEON_DIM)
        right.addWidget(self._freq_label)

        right.addSpacing(20)

        # Left hand info
        self._lh_label = _label("LEFT HAND", size=9, color=NEON_DARK, spacing=4)
        right.addWidget(self._lh_label)
        self._lh_status = _label("—", size=11, color=NEON_DIM)
        right.addWidget(self._lh_status)

        right.addSpacing(30)

        # ── Volume slider ─────────────────────────────────────
        vol_row = QVBoxLayout()
        vol_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(int(DEFAULT_VOLUME * 100))
        self._vol_slider.setFixedWidth(200)
        self._vol_slider.valueChanged.connect(self._on_volume)
        vol_row.addWidget(self._vol_slider, alignment=Qt.AlignmentFlag.AlignCenter)

        vol_text = _label("VOL", size=8, color=NEON_DARK, spacing=5)
        vol_row.addWidget(vol_text)

        right.addLayout(vol_row)

        right.addSpacing(30)

        # ── Buttons ───────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_row.setSpacing(20)

        self._start_btn = QPushButton("START")
        self._start_btn.setFont(self._btn_font())
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.clicked.connect(self._start_camera)
        btn_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("STOP")
        self._stop_btn.setFont(self._btn_font())
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_camera)
        btn_row.addWidget(self._stop_btn)

        right.addLayout(btn_row)

        right.addSpacing(15)

        # ── Mode toggle (CNN / Rules) ─────────────────────────
        self._mode_btn = QPushButton("MODE: RULES")
        self._mode_btn.setFont(self._btn_font())
        self._mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mode_btn.clicked.connect(self._toggle_mode)
        if not has_cnn_model():
            self._mode_btn.setEnabled(False)
            self._mode_btn.setToolTip("No CNN model found — train one first")
        mode_row = QHBoxLayout()
        mode_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_row.addWidget(self._mode_btn)
        right.addLayout(mode_row)

        right.addSpacerItem(QSpacerItem(0, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # ── Guide (very subtle, bottom) ───────────────────────
        guide = _label(
            "R: 1-C  2-D  3-E  4-G  5-A   ·   L: 1-LOW  2-MID  3-HIGH  4-SQR  5-SAW",
            size=8, color=NEON_FAINT, spacing=1
        )
        right.addWidget(guide)

        root.addLayout(right, stretch=2)

    @staticmethod
    def _btn_font():
        f = QFont("Consolas", 10)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        return f

    # ────────────────────────────────────────────────────────
    #  CAMERA
    # ────────────────────────────────────────────────────────
    def _start_camera(self):
        if self._cam_worker and self._cam_worker.isRunning():
            return

        self._cam_worker = CameraWorker(use_cnn=self._use_cnn)
        self._cam_worker.frame_ready.connect(self._on_frame)
        self._cam_worker.start()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._stop_btn.setObjectName("active")
        self._stop_btn.setStyleSheet(self._stop_btn.styleSheet())  # force re-style
        self._cam_label.setStyleSheet(
            f"border: 1px solid {NEON}; color: {NEON_DARK}; font: 11px Consolas; letter-spacing: 3px;"
        )

    def _stop_camera(self):
        if self._cam_worker:
            self._cam_worker.stop()
            self._cam_worker = None

        self._synth.mute()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setObjectName("")

        self._cam_label.setPixmap(QPixmap())
        self._cam_label.setText("INSERT\nCOIN")
        self._cam_label.setStyleSheet(
            f"border: 1px solid {NEON_DARK}; color: {NEON_DARK}; font: 11px Consolas; letter-spacing: 3px;"
        )
        self._note_circle.set_state("", "", False)
        self._freq_label.setText("")
        self._rh_status.setText("—")
        self._lh_status.setText("—")

    # ────────────────────────────────────────────────────────
    #  FRAME + GESTURE PROCESSING
    # ────────────────────────────────────────────────────────
    def _on_frame(self, rgb_frame, left_f, right_f):
        """Slot called from camera thread with each new frame."""
        # Update camera image
        h, w, ch = rgb_frame.shape
        qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._cam_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._cam_label.setPixmap(pixmap)

        # Process gestures
        self._process_gestures(left_f, right_f)

    def _process_gestures(self, left_f, right_f):
        # ── Left hand → waveform + octave ─────────────────
        if left_f >= 0:
            info = LEFT_HAND_MAP.get(left_f, LEFT_HAND_MAP[0])
            self._cur_waveform = info["waveform"]
            self._cur_octave = info["octave"]
            self._lh_status.setText(f"{info['label'].upper()}  ·  {info['waveform'].upper()}  OCT {info['octave']}")
            self._lh_status.setStyleSheet("color: #00FF41; background: transparent;")
        else:
            self._lh_status.setText("—")
            self._lh_status.setStyleSheet("color: #005F15; background: transparent;")

        # ── Right hand → note ─────────────────────────────
        if right_f >= 0:
            info = RIGHT_HAND_MAP.get(right_f, RIGHT_HAND_MAP[0])
            note = info["note"]
            self._cur_note = note
            self._rh_status.setText(f"{info['label'].upper()}")
            self._rh_status.setStyleSheet("color: #00FF41; background: transparent;")

            if note:
                freq = get_frequency(note, self._cur_octave)
                self._cur_freq = freq
                self._synth.set_note(freq, self._cur_waveform)
                self._note_circle.set_state(f"{note}{self._cur_octave}", self._cur_waveform, True)
                self._freq_label.setText(f"{freq:.1f} HZ")
            else:
                self._synth.mute()
                self._cur_freq = 0
                self._note_circle.set_state("", "", False)
                self._freq_label.setText("")
        else:
            self._rh_status.setText("—")
            self._rh_status.setStyleSheet("color: #005F15; background: transparent;")
            self._synth.mute()
            self._cur_note = None
            self._cur_freq = 0
            self._note_circle.set_state("", "", False)
            self._freq_label.setText("")

    def _toggle_mode(self):
        self._use_cnn = not self._use_cnn
        label = "CNN" if self._use_cnn else "RULES"
        self._mode_btn.setText(f"MODE: {label}")
        if self._use_cnn:
            self._mode_btn.setObjectName("active")
        else:
            self._mode_btn.setObjectName("")
        self._mode_btn.setStyleSheet(self._mode_btn.styleSheet())

        # Update running camera worker if active
        if self._cam_worker and self._cam_worker.isRunning():
            self._cam_worker.set_use_cnn(self._use_cnn)

    # ────────────────────────────────────────────────────────
    #  CALLBACKS
    # ────────────────────────────────────────────────────────
    def _on_volume(self, val):
        self._synth.set_volume(val / 100.0)

    def closeEvent(self, event):
        self._stop_camera()
        self._synth.shutdown()
        event.accept()


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
