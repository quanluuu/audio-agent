"""Filter chain shared by the realtime stream, the evaluation script and the agent."""

import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from voiceagent.dsp import BandPass, NoiseGate, SpectralSubtractor, rms_db


@dataclass(frozen=True)
class Params:
    bypass: bool = False
    bandpass_enabled: bool = True
    low_hz: float = 100.0
    high_hz: float = 3800.0
    gate_enabled: bool = False
    gate_threshold_db: float = -45.0
    ss_enabled: bool = False
    alpha: float = 2.0
    beta: float = 0.02
    smoothing: float = 0.0
    noise_mode: str = "calibrate"

@dataclass
class Meters:
    input_db: float = -120.0
    output_db: float = -120.0
    process_ms: float = 0.0  # moving average
    underruns: int = 0
    noise_floor_db: float = -120.0
    calibrating: bool = False


class Pipeline:
    """Runs one audio block through the enabled filters. Setters are safe to call from other threads."""

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.meters = Meters()
        self._lock = threading.Lock()
        self._params = Params()
        self._bandpass = BandPass(sr, self._params.low_hz, self._params.high_hz)
        self._gate = NoiseGate(threshold_db=self._params.gate_threshold_db)
        self._ss = SpectralSubtractor(alpha=self._params.alpha, beta=self._params.beta)
        self._calibrating = False
        self._calib_frames_needed = 0
        self._calib_mag_acc = np.zeros(257, dtype=np.float32)
        self._calib_count = 0

    def params(self) -> Params:
        with self._lock:
            return self._params

    def set_bypass(self, enabled: bool) -> None:
        with self._lock:
            self._params = replace(self._params, bypass=enabled)

    def set_bandpass(self, enabled: bool, low_hz: float, high_hz: float) -> None:
        bandpass = BandPass(self.sr, low_hz, high_hz)  # raises ValueError before anything changes
        with self._lock:
            self._bandpass = bandpass
            self._params = replace(self._params, bandpass_enabled=enabled, low_hz=low_hz, high_hz=high_hz)

    def set_noise_gate(self, enabled: bool, threshold_db: float) -> None:
        gate = NoiseGate(threshold_db=threshold_db)
        with self._lock:
            self._gate = gate
            self._params = replace(self._params, gate_enabled=enabled, gate_threshold_db=threshold_db)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        with self._lock:  # hold only long enough to grab references
            params, bandpass, ss, gate = self._params, self._bandpass, self._ss, self._gate

        out = block
        if not params.bypass and params.bandpass_enabled:
            out = bandpass.process(out).astype(block.dtype)

        if self._calibrating:
            frame = np.concatenate([self._ss.prev_input, out])
            spec = np.fft.rfft(frame * self._ss.window)
            self._calib_mag_acc += np.abs(spec)
            self._calib_count += 1
            if self._calib_count >= self._calib_frames_needed:
                avg_noise = self._calib_mag_acc / max(1, self._calib_count)
                self._ss.set_noise_spectrum(avg_noise)
                self.meters.noise_floor_db = rms_db(out)
                self._calibrating = False
                self.meters.calibrating = False
        elif not params.bypass and params.ss_enabled:
            out = ss.process(out).astype(block.dtype)

        if not params.bypass and params.gate_enabled:
            out = gate.process(out).astype(block.dtype)

        m = self.meters
        m.input_db, m.output_db = rms_db(block), rms_db(out)
        m.process_ms = 0.95 * m.process_ms + 0.05 * (time.perf_counter() - start) * 1000
        return out

    def set_spectral_subtraction(self, enabled: bool, alpha: float, beta: float, smoothing: float) -> None:
        ss = SpectralSubtractor(
            alpha=alpha,
            beta=beta,
            smoothing=smoothing,
            noise_mode=self._params.noise_mode,
        )
        ss.set_noise_spectrum(self._ss.noise_mag)
        with self._lock:
            self._ss = ss
            self._params = replace(
                self._params,
                ss_enabled=enabled,
                alpha=alpha,
                beta=beta,
                smoothing=smoothing,
            )

    def set_noise_estimator(self, mode: str) -> None:
        if mode not in ("calibrate", "minstat"):
            raise ValueError(f"noise_mode must be 'calibrate' or 'minstat', got {mode}")
        with self._lock:
            self._ss.set_noise_mode(mode)
            self._params = replace(self._params, noise_mode=mode)

    def calibrate_noise(self, seconds: float = 1.0) -> None:
        if seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {seconds}")
        with self._lock:
            self._calibrating = True
            self.meters.calibrating = True
            self._calib_frames_needed = int(seconds * self.sr / 256)
            self._calib_mag_acc.fill(0)
            self._calib_count = 0
