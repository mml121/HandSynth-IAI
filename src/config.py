# config.py — Two-hand gesture mappings and audio parameters

# ── Musical Scale (C pentatonic — always sounds good) ──────────────
# Base frequencies for octave 4
NOTE_FREQS = {
    "C": 261.63,
    "D": 293.66,
    "E": 329.63,
    "G": 392.00,
    "A": 440.00,
}

# ── Right Hand → Melody (finger count selects note) ───────────────
RIGHT_HAND_MAP = {
    0: {"label": "Fist",       "note": None},       # silence
    1: {"label": "1 Finger",   "note": "C"},
    2: {"label": "2 Fingers",  "note": "D"},
    3: {"label": "3 Fingers",  "note": "E"},
    4: {"label": "4 Fingers",  "note": "G"},
    5: {"label": "Open Palm",  "note": "A"},
}

# ── Left Hand → Control (waveform + octave) ───────────────────────
LEFT_HAND_MAP = {
    0: {"label": "Fist",       "waveform": "sine",     "octave": 4},  # default
    1: {"label": "1 Finger",   "waveform": "sine",     "octave": 3},  # low sine
    2: {"label": "2 Fingers",  "waveform": "sine",     "octave": 4},  # mid sine
    3: {"label": "3 Fingers",  "waveform": "sine",     "octave": 5},  # high sine
    4: {"label": "4 Fingers",  "waveform": "square",   "octave": 4},  # square
    5: {"label": "Open Palm",  "waveform": "sawtooth", "octave": 4},  # sawtooth
}

WAVEFORM_TYPES = ["sine", "square", "sawtooth"]

# ── Audio settings ────────────────────────────────────────────────
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
DEFAULT_VOLUME = 0.3

# ── Camera settings ──────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
UI_FPS = 30  # target UI refresh rate

# ── Detection confidence thresholds (MediaPipe) ─────────────────
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6


def get_frequency(note, octave):
    """Get the frequency of a note at a given octave."""
    if note is None:
        return 0
    base = NOTE_FREQS.get(note, 0)
    # Shift octave relative to octave 4
    return base * (2 ** (octave - 4))
