# audio.py — Continuous audio engine using sounddevice OutputStream

import threading
import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, BLOCK_SIZE
from synth import get_samples


class LiveSynth:
    """Real-time synthesizer that plays continuously and responds to note changes instantly."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frequency = 0.0
        self._waveform = "sine"
        self._volume = 0.3
        self._phase = 0.0       # tracks position in the wave for click-free transitions
        self._active = False
        self._stream = None

    def start(self):
        """Open the audio stream."""
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        """Called by sounddevice to fill the output buffer."""
        with self._lock:
            freq = self._frequency
            waveform = self._waveform
            volume = self._volume
            active = self._active
            phase = self._phase

        if not active or freq <= 0:
            outdata[:] = 0.0
            return

        # Generate time array continuing from current phase
        t = (np.arange(frames) / SAMPLE_RATE) + phase
        samples = get_samples(t, freq, waveform)
        outdata[:, 0] = (volume * samples).astype(np.float32)

        # Advance phase, wrap to avoid float overflow
        new_phase = phase + frames / SAMPLE_RATE
        period = 1.0 / freq
        new_phase = new_phase % period

        with self._lock:
            self._phase = new_phase

    def set_note(self, frequency, waveform="sine"):
        """Change the currently playing note (instantly, click-free)."""
        with self._lock:
            if frequency != self._frequency or waveform != self._waveform:
                self._frequency = frequency
                self._waveform = waveform
                self._phase = 0.0  # reset phase on note change
            self._active = frequency > 0

    def set_volume(self, volume):
        """Set volume (0.0 to 1.0)."""
        with self._lock:
            self._volume = max(0.0, min(1.0, volume))

    def mute(self):
        """Stop playing."""
        with self._lock:
            self._active = False
            self._frequency = 0.0

    def shutdown(self):
        """Close the audio stream."""
        self.mute()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def is_active(self):
        with self._lock:
            return self._active
