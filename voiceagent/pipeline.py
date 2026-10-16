"""Filter chain shared by the realtime stream, the evaluation script and the agent."""

import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from voiceagent.dsp import BandPass, rms_db


@dataclass(frozen=True)
class Params:
    bypass: bool = False
    bandpass_enabled: bool = True
    low_hz: float = 100.0
    high_hz: float = 3800.0


@dataclass
class Meters:
    input_db: float = -120.0
    output_db: float = -120.0
    process_ms: float = 0.0  # moving average
    underruns: int = 0


class Pipeline:
    """Runs one audio block through the enabled filters. Setters are safe to call from other threads."""

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.meters = Meters()
        self._lock = threading.Lock()
        self._params = Params()
        self._bandpass = BandPass(sr, self._params.low_hz, self._params.high_hz)

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

    def process_block(self, block: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        with self._lock:  # hold only long enough to grab references
            params, bandpass = self._params, self._bandpass

        out = block
        if not params.bypass and params.bandpass_enabled:
            out = bandpass.process(out).astype(block.dtype)

        m = self.meters
        m.input_db, m.output_db = rms_db(block), rms_db(out)
        m.process_ms = 0.95 * m.process_ms + 0.05 * (time.perf_counter() - start) * 1000
        return out
