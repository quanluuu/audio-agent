import io
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

from voiceagent.__main__ import enable_all_filters
from voiceagent.audio import UtteranceRecorder, make_callback, save_recording
from voiceagent.dsp import rms_db
from voiceagent.pipeline import Pipeline

SR = 16000


def noise_block(seed=0):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, (256, 1)).astype(np.float32)


def test_callback_writes_processed_audio_and_records_both_tracks():
    pipeline = Pipeline(SR)
    pipeline.set_bypass(True)
    pipeline.set_monitor(True)
    raw, clean = [], []
    callback = make_callback(pipeline, raw, clean)
    indata = noise_block()
    outdata = np.zeros_like(indata)

    callback(indata, outdata, 256, None, None)

    assert np.array_equal(outdata, indata)
    assert np.array_equal(raw[0], indata[:, 0])
    assert np.array_equal(clean[0], indata[:, 0])


def test_monitor_off_sends_silence_but_still_records():
    pipeline = Pipeline(SR)
    pipeline.set_bypass(True)
    clean = []
    callback = make_callback(pipeline, [], clean)
    indata = noise_block()
    outdata = np.ones_like(indata)

    callback(indata, outdata, 256, None, None)

    assert not outdata.any()
    assert np.array_equal(clean[0], indata[:, 0])


def test_utterance_keeps_only_cleaned_blocks_between_start_and_stop():
    pipeline = Pipeline(SR)  # band-pass on, so cleaned != raw
    clean, rec = [], UtteranceRecorder(SR)
    callback = make_callback(pipeline, [], clean, rec)
    blocks = [noise_block(i) for i in range(40)]

    callback(blocks[0], np.zeros_like(blocks[0]), 256, None, None)
    rec.start()
    for b in blocks[1:-1]:
        callback(b, np.zeros_like(b), 256, None, None)
    wav = rec.stop()
    callback(blocks[-1], np.zeros_like(blocks[-1]), 256, None, None)

    sr, data = wavfile.read(io.BytesIO(wav))
    assert sr == SR and data.ndim == 1
    assert np.array_equal(data, np.concatenate(clean[1:-1]))
    assert not np.array_equal(data, np.concatenate([b[:, 0] for b in blocks[1:-1]]))


def test_short_or_missing_utterance_returns_none():
    rec = UtteranceRecorder(SR)
    assert rec.stop() is None
    rec.start()
    for _ in range(int(0.3 * SR / 256)):  # just under 0.3 s
        rec.feed(np.zeros(256, np.float32))
    assert rec.stop() is None


def test_enable_all_filters_uses_minstat_without_calibration():
    p = Pipeline(SR)
    enable_all_filters(p)
    params = p.params()
    assert params.bandpass_enabled and params.ss_enabled and params.gate_enabled
    assert (params.noise_mode, params.smoothing) == ("minstat", 0.5)
    assert p.meters.calibrating is False


def test_all_filters_suppress_steady_noise_without_silent_start():
    p = Pipeline(SR)
    enable_all_filters(p)
    rng = np.random.default_rng(1)
    noise = [(0.01 * rng.standard_normal(256)).astype(np.float32) for _ in range(int(3 * SR / 256))]
    out = [p.process_block(b) for b in noise]
    last = slice(-62, None)  # final ~1 s, after the 1.5 s minstat window has filled
    reduction = rms_db(np.concatenate(noise[last])) - rms_db(np.concatenate(out[last]))
    assert reduction >= 10.0


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
