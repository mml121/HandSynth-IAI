# synth.py — Waveform generation (pure math, no audio I/O)

import numpy as np


def generate_sine(t, frequency):
    """Sine wave samples for time array t."""
    return np.sin(2 * np.pi * frequency * t)


def generate_square(t, frequency):
    """Square wave samples for time array t."""
    return np.sign(np.sin(2 * np.pi * frequency * t))


def generate_sawtooth(t, frequency):
    """Sawtooth wave samples for time array t."""
    return 2.0 * (t * frequency - np.floor(0.5 + t * frequency))


GENERATORS = {
    "sine": generate_sine,
    "square": generate_square,
    "sawtooth": generate_sawtooth,
}


def get_samples(t, frequency, waveform="sine"):
    """Generate waveform samples for the given time array."""
    gen = GENERATORS.get(waveform, generate_sine)
    return gen(t, frequency)
