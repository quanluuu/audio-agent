import time

import numpy as np
import pytest

from voiceagent.dsp import rms_db
from voiceagent.pipeline import Pipeline

SR = 16000
BLOCK = 256


def sine_block(freq, n=BLOCK, amp=0.5, start=0):
    t = (start + np.arange(n)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_bypass_returns_input_unchanged():
    p = Pipeline(SR)
    p.set_bypass(True)
    block = sine_block(6000)
    assert np.array_equal(p.process_block(block), block)


def test_default_pipeline_applies_bandpass():
    p = Pipeline(SR)
    blocks = [p.process_block(sine_block(6000, start=i * BLOCK)) for i in range(20)]
    assert rms_db(blocks[-1]) - rms_db(sine_block(6000)) <= -20.0


def test_param_change_applies_from_next_block():
    p = Pipeline(SR)
    p.process_block(sine_block(6000))
    p.set_bypass(True)
    block = sine_block(6000, start=BLOCK)
    assert np.array_equal(p.process_block(block), block)


def test_disabling_bandpass_passes_signal_through():
    p = Pipeline(SR)
    p.set_bandpass(False, 100, 3800)
    block = sine_block(6000)
    assert np.allclose(p.process_block(block), block)


def test_set_bandpass_changes_params():
    p = Pipeline(SR)
    p.set_bandpass(True, 300, 3400)
    assert (p.params().low_hz, p.params().high_hz) == (300, 3400)


def test_invalid_bandpass_raises_and_keeps_old_params():
    p = Pipeline(SR)
    before = p.params()
    with pytest.raises(ValueError):
        p.set_bandpass(True, 3000, 1000)
    assert p.params() == before


def test_meters_track_levels():
    p = Pipeline(SR)
    p.set_bypass(True)
    block = sine_block(1000)
    p.process_block(block)
    assert p.meters.input_db == pytest.approx(rms_db(block), abs=0.01)
    assert p.meters.output_db == pytest.approx(rms_db(block), abs=0.01)


def test_block_processing_is_fast():
    p = Pipeline(SR)
    blocks = [sine_block(1000, start=i * BLOCK) for i in range(500)]
    start = time.perf_counter()
    for b in blocks:
        p.process_block(b)
    avg_ms = (time.perf_counter() - start) / len(blocks) * 1000
    assert avg_ms < 2.0
    assert p.meters.process_ms > 0
