import numpy as np
import pytest

from voiceagent.dsp import BandPass, add_noise, mix_at_snr, rms_db, snr_db

SR = 16000


def sine(freq, seconds=1.0, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_rms_db_of_full_scale_sine_is_minus_3db():
    assert rms_db(sine(440, amp=1.0)) == pytest.approx(-3.01, abs=0.01)


def test_rms_db_of_silence_is_finite():
    assert np.isfinite(rms_db(np.zeros(256)))


def test_snr_db_matches_known_noise_level():
    clean = sine(440)
    noise = np.random.default_rng(0).standard_normal(len(clean))
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2)) / 10  # 20 dB below signal
    assert snr_db(clean, clean + noise) == pytest.approx(20.0, abs=0.1)


@pytest.mark.parametrize("target", [0.0, 5.0, 10.0])
def test_add_noise_hits_target_snr(target):
    clean = sine(440)
    noisy = add_noise(clean, target, seed=0)
    assert snr_db(clean, noisy) == pytest.approx(target, abs=0.1)


def test_add_noise_is_reproducible_with_seed():
    clean = sine(440)
    assert np.array_equal(add_noise(clean, 5, seed=1), add_noise(clean, 5, seed=1))


@pytest.mark.parametrize("noise_len", [8000, 16000, 40000])
def test_mix_at_snr_loops_or_trims_noise_to_clean_length(noise_len):
    clean = sine(440)
    noise = np.random.default_rng(0).standard_normal(noise_len)
    noisy = mix_at_snr(clean, noise, 5.0)
    assert len(noisy) == len(clean)
    assert snr_db(clean, noisy) == pytest.approx(5.0, abs=0.1)


def run_in_blocks(filt, x, block=256):
    return np.concatenate([filt.process(x[i : i + block]) for i in range(0, len(x), block)])


def gain_db(freq, low=100.0, high=3800.0):
    x = sine(freq, seconds=1.0)
    y = run_in_blocks(BandPass(SR, low, high), x)
    settle = SR // 5  # skip the filter's start-up transient
    return rms_db(y[settle:]) - rms_db(x[settle:])


def test_bandpass_keeps_speech_band():
    assert abs(gain_db(1000)) < 1.0


@pytest.mark.parametrize("freq", [50, 6000])
def test_bandpass_attenuates_outside_band(freq):
    assert gain_db(freq) <= -20.0


def test_bandpass_in_blocks_equals_whole_signal():
    x = add_noise(sine(1000), 5, seed=0)
    whole = BandPass(SR, 100, 3800).process(x)
    assert np.max(np.abs(run_in_blocks(BandPass(SR, 100, 3800), x) - whole)) < 1e-6


@pytest.mark.parametrize("low, high", [(0, 3000), (3000, 1000), (100, 8000)])
def test_bandpass_rejects_invalid_band(low, high):
    with pytest.raises(ValueError):
        BandPass(SR, low, high)
