"""DSP building blocks. Pure numpy/scipy: no file I/O, no printing, no network."""

import numpy as np
from scipy import signal

EPS = 1e-12  # avoids log(0) on digital silence


def rms_db(x: np.ndarray) -> float:
    """RMS level in dBFS (full-scale sine = -3 dB)."""
    return float(10 * np.log10(np.mean(np.square(x)) + EPS))


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    """SNR of `test` against the `clean` reference; everything that differs counts as noise."""
    noise = test - clean
    return float(10 * np.log10((np.sum(clean**2) + EPS) / (np.sum(noise**2) + EPS)))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr: float) -> np.ndarray:
    """Add `noise` (looped or trimmed to length) scaled so the mix has the given SNR in dB."""
    noise = np.resize(noise, len(clean))  # np.resize repeats the array when it is too short
    gain = np.sqrt(np.sum(clean**2) / (np.sum(noise**2) * 10 ** (snr / 10)))
    return clean + gain * noise


def add_noise(x: np.ndarray, snr: float, seed: int | None = None) -> np.ndarray:
    """Add white Gaussian noise at the given SNR in dB."""
    noise = np.random.default_rng(seed).standard_normal(len(x))
    return mix_at_snr(x, noise, snr)


class BandPass:
    """Butterworth band-pass, causal, keeps filter state across blocks."""

    # Speech energy sits roughly in 100-4000 Hz; cutting outside removes hum and hiss for free.
    def __init__(self, sr: int, low_hz: float, high_hz: float, order: int = 4):
        if not 0 < low_hz < high_hz < sr / 2:
            raise ValueError(f"need 0 < low < high < {sr / 2} Hz, got {low_hz}, {high_hz}")
        self.sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, block: np.ndarray) -> np.ndarray:
        # sosfilt with carried-over zi = no clicks at block edges (sosfiltfilt would need the future)
        out, self.zi = signal.sosfilt(self.sos, block, zi=self.zi)
        return out
