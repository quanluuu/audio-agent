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

def test_set_noise_gate_changes_params():
    p = Pipeline(SR)
    p.set_noise_gate(True, -40.0)
    assert p.params().gate_enabled is True
    assert p.params().gate_threshold_db == -40.0


def test_invalid_noise_gate_raises_and_keeps_old_params():
    p = Pipeline(SR)
    before = p.params()
    with pytest.raises(ValueError):
        p.set_noise_gate(True, 10.0)
    assert p.params() == before


def test_set_spectral_subtraction_changes_params():
    p = Pipeline(SR)
    p.set_spectral_subtraction(True, 2.5, 0.05, 0.3)
    params = p.params()
    assert (params.ss_enabled, params.alpha, params.beta, params.smoothing) == (True, 2.5, 0.05, 0.3)


def test_invalid_spectral_subtraction_raises_and_keeps_old_params():
    p = Pipeline(SR)
    before = p.params()
    with pytest.raises(ValueError):
        p.set_spectral_subtraction(True, -1.0, 0.02, 0.0)
    assert p.params() == before


def test_pipeline_calibrate_noise():
    p = Pipeline(SR)
    p.calibrate_noise(seconds=0.1)
    assert p.meters.calibrating is True
    block = sine_block(1000, amp=0.1)
    while p.meters.calibrating:
        p.process_block(block)
    assert p.meters.calibrating is False
    assert p.meters.noise_floor_db > -120.0


def test_invalid_calibrate_noise_raises():
    p = Pipeline(SR)
    with pytest.raises(ValueError):
        p.calibrate_noise(seconds=0.0)


def test_set_noise_estimator_changes_params():
    p = Pipeline(SR)
    p.set_noise_estimator("minstat")
    assert p.params().noise_mode == "minstat"


def test_invalid_noise_estimator_raises():
    p = Pipeline(SR)
    with pytest.raises(ValueError):
        p.set_noise_estimator("invalid_mode")
