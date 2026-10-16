import subprocess
import sys

import numpy as np
from scipy.io import wavfile

from voiceagent.audio import make_callback, save_recording
from voiceagent.pipeline import Pipeline

SR = 16000


def test_callback_writes_processed_audio_and_records_both_tracks():
    pipeline = Pipeline(SR)
    pipeline.set_bypass(True)
    raw, clean = [], []
    callback = make_callback(pipeline, raw, clean)
    indata = np.random.default_rng(0).uniform(-0.5, 0.5, (256, 1)).astype(np.float32)
    outdata = np.zeros_like(indata)

    callback(indata, outdata, 256, None, None)

    assert np.array_equal(outdata, indata)
    assert np.array_equal(raw[0], indata[:, 0])
    assert np.array_equal(clean[0], indata[:, 0])


def test_callback_counts_stream_problems():
    pipeline = Pipeline(SR)
    callback = make_callback(pipeline, [], [])
    block = np.zeros((256, 1), np.float32)
    callback(block, np.zeros_like(block), 256, None, "input overflow")
    assert pipeline.meters.underruns == 1


def test_save_recording_writes_two_wavs(tmp_path):
    blocks = [np.full(256, 0.1, np.float32), np.full(256, -0.1, np.float32)]
    raw_path, clean_path = save_recording(blocks, blocks, tmp_path, SR)
    for path in (raw_path, clean_path):
        sr, data = wavfile.read(path)
        assert sr == SR
        assert len(data) == 512


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "voiceagent", *args],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
    )


def test_cli_lists_devices():
    result = run_cli("--list-devices")
    assert result.returncode == 0


def test_cli_bad_device_gives_clear_error_without_traceback():
    result = run_cli("--no-agent", "--input", "999")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "999" in result.stderr
