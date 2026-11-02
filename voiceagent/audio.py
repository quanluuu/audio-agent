"""Realtime glue: microphone -> Pipeline -> headphones, plus WAV recording."""

import io
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from voiceagent.pipeline import Pipeline

BLOCK = 256  # 16 ms at 16 kHz
MIN_UTTERANCE_S = 0.3


class UtteranceRecorder:
    """Collects cleaned blocks between start() and stop() and packs them as an in-memory WAV."""

    def __init__(self, sr: int):
        self.sr = sr
        self._blocks = None  # None = not holding a turn

    def start(self) -> None:
        self._blocks = []

    def feed(self, block: np.ndarray) -> None:
        blocks = self._blocks  # single read: stop() may swap it from the main thread
        if blocks is not None:
            blocks.append(block)

    def stop(self) -> bytes | None:
        """End the turn. Returns 16 kHz mono float32 WAV bytes, or None if shorter than 0.3 s."""
        blocks, self._blocks = self._blocks, None
        if not blocks or sum(len(b) for b in blocks) < MIN_UTTERANCE_S * self.sr:
            return None
        buf = io.BytesIO()
        wavfile.write(buf, self.sr, np.concatenate(blocks).astype(np.float32))
        return buf.getvalue()


def make_callback(pipeline: Pipeline, raw: list, clean: list, utterance: UtteranceRecorder | None = None):
    """Build the sounddevice callback. Runs on the audio thread: no printing, no blocking."""

    def callback(indata, outdata, frames, time_info, status):
        if status:  # any input overflow / output underflow reported by PortAudio
            pipeline.meters.underruns += 1
        block = indata[:, 0].copy()
        out = pipeline.process_block(block)
        # monitor off by default: hearing your own voice late is distracting and loops back on speakers
        outdata[:, 0] = out if pipeline.params().monitor else 0.0
        raw.append(block)
        clean.append(out)
        if utterance is not None:
            utterance.feed(out)

    return callback


def open_stream(pipeline: Pipeline, raw: list, clean: list, input_device=None, output_device=None,
                utterance: UtteranceRecorder | None = None):
    """Open (not start) a duplex mono float32 stream. Raises sd.PortAudioError/ValueError on bad devices."""
    # shortcut: whole session kept in RAM (~128 KB/s for both tracks), upgrade to streaming writes for hour-long runs
    return sd.Stream(
        samplerate=pipeline.sr, blocksize=BLOCK, channels=1, dtype="float32",
        device=(input_device, output_device), callback=make_callback(pipeline, raw, clean, utterance),
    )


def save_recording(raw: list, clean: list, out_dir, sr: int) -> tuple[Path, Path]:
    """Write <timestamp>_raw.wav and <timestamp>_clean.wav as float32 (no clipping, no resampling)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths = out_dir / f"{stamp}_raw.wav", out_dir / f"{stamp}_clean.wav"
    for path, blocks in zip(paths, (raw, clean)):
        data = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
        wavfile.write(path, sr, data.astype(np.float32))
    return paths
