import numpy as np
import pytest
from voiceagent.dsp import (
    BandPass,
    NoiseGate,
    SpectralSubtractor,
    add_noise,
    mix_at_snr,
    rms_db,
    snr_db,
)

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


def test_noise_gate_attenuates_low_level_noise():
    gate = NoiseGate(threshold_db=-45.0, floor_gain=0.1)
    noise_block = np.random.default_rng(0).normal(0, 1e-3, 256).astype(np.float32)

    for _ in range(20):
        out = gate.process(noise_block)
    attenuation = rms_db(noise_block) - rms_db(out)
    assert attenuation >= 15.0


def test_noise_gate_passes_speech_level():
    gate = NoiseGate(threshold_db=-45.0, floor_gain=0.1)
    sine_block = sine(1000, seconds=256/SR, amp=0.14)  # ~ -20 dBFS
    for _ in range(10):
        out = gate.process(sine_block)
    assert abs(rms_db(out) - rms_db(sine_block)) < 1.0


@pytest.mark.parametrize("thresh", [1.0, -120.0])
def test_noise_gate_rejects_invalid_threshold(thresh):
    with pytest.raises(ValueError):
        NoiseGate(threshold_db=thresh)


def test_spectral_subtraction_alpha_zero_is_identity_with_delay():
    ss = SpectralSubtractor(alpha=0.0)
    # 64 blocks of 256 = 16384 samples (~1.024s)
    x = sine(440, seconds=16384 / SR).astype(np.float32)
    blocks = [x[i : i + 256] for i in range(0, len(x), 256)]
    outs = [ss.process(b) for b in blocks]
    y = np.concatenate(outs)
    # y bằng x trễ 256 mẫu
    assert np.allclose(y[256:len(x)], x[:len(x) - 256], atol=1e-5)


def test_spectral_subtraction_improves_snr_on_sine_plus_noise():
    rng = np.random.default_rng(0)
    noise_samples = 64 * 256
    noise_1s = rng.normal(0, 0.1, noise_samples).astype(np.float32)

    noise_blocks = [noise_1s[i : i + 256] for i in range(0, len(noise_1s), 256)]
    ss_temp = SpectralSubtractor()
    mags = []
    for b in noise_blocks:
        frame = np.concatenate([ss_temp.prev_input, b])
        ss_temp.prev_input = b.copy()
        spec = np.fft.rfft(frame * ss_temp.window)
        mags.append(np.abs(spec))
    avg_noise_mag = np.mean(mags, axis=0)

    ss = SpectralSubtractor(alpha=2.0, beta=0.02)
    ss.set_noise_spectrum(avg_noise_mag)

    clean = sine(440, seconds=noise_samples / SR, amp=0.5).astype(np.float32)
    test_noise = rng.normal(0, 0.1, len(clean)).astype(np.float32)
    noisy = mix_at_snr(clean, test_noise, 5.0)

    blocks = [noisy[i : i + 256] for i in range(0, len(noisy), 256)]
    out_y = np.concatenate([ss.process(b) for b in blocks])

    snr_before = snr_db(clean[:-256], noisy[:-256])
    snr_after = snr_db(clean[:-256], out_y[256:])
    assert snr_after - snr_before >= 5.0


@pytest.mark.parametrize("alpha, beta", [(-1.0, 0.02), (2.0, -0.1), (2.0, 1.5)])
def test_spectral_subtractor_rejects_invalid_params(alpha, beta):
    with pytest.raises(ValueError):
        SpectralSubtractor(alpha=alpha, beta=beta)


def test_spectral_subtraction_gain_smoothing_reduces_variance():
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, 150 * 256).astype(np.float32)
    blocks = [noise[i : i + 256] for i in range(0, len(noise), 256)]

    temp = SpectralSubtractor()
    mags = []
    for b in blocks[:20]:
        frame = np.concatenate([temp.prev_input, b])
        temp.prev_input = b.copy()
        mags.append(np.abs(np.fft.rfft(frame * temp.window)))
    avg_noise = np.mean(mags, axis=0)

    ss_raw = SpectralSubtractor(alpha=2.0, beta=0.02, smoothing=0.0)
    ss_raw.set_noise_spectrum(avg_noise)
    gains_raw = []
    for b in blocks[20:]:
        ss_raw.process(b)
        gains_raw.append(ss_raw.prev_gain.copy())
    # Bỏ qua 30 khối khởi động để đo ở trạng thái ổn định
    std_raw = np.mean(np.std(gains_raw[30:], axis=0))

    ss_smooth = SpectralSubtractor(alpha=2.0, beta=0.02, smoothing=0.5)
    ss_smooth.set_noise_spectrum(avg_noise)
    gains_smooth = []
    for b in blocks[20:]:
        ss_smooth.process(b)
        gains_smooth.append(ss_smooth.prev_gain.copy())
    std_smooth = np.mean(np.std(gains_smooth[30:], axis=0))

    assert std_smooth < std_raw


def test_minstat_tracks_step_increase_in_noise():
    rng = np.random.default_rng(123)
    ss = SpectralSubtractor(noise_mode="minstat")

    blocks_3s = [rng.normal(0, 0.05, 256).astype(np.float32) for _ in range(3 * SR // 256)]
    blocks_2s = [rng.normal(0, 0.10, 256).astype(np.float32) for _ in range(2 * SR // 256)]

    for b in blocks_3s:
        ss.process(b)
    est_before = np.mean(ss.noise_mag)

    for b in blocks_2s:
        ss.process(b)
    est_after = np.mean(ss.noise_mag)

    ratio = est_after / est_before
    ratio_db = 20 * np.log10(ratio)
    assert abs(ratio_db - 6.0) <= 3.0


def test_minstat_ignores_intermittent_tone():
    # Sine 1000 Hz bật 0.5 s, tắt 0.5 s, lặp lại
    rng = np.random.default_rng(0)
    ss = SpectralSubtractor(noise_mode="minstat")
    bin_1k = int(round(1000 / (SR / 512)))

    for _ in range(4):
        for _ in range(31):
            ss.process((sine(1000, 256 / SR, amp=0.01) + rng.normal(0, 0.02, 256)).astype(np.float32))
        for _ in range(31):
            ss.process(rng.normal(0, 0.02, 256).astype(np.float32))

    # So sánh với ước lượng ồn của minstat khi chỉ có nhiễu nền thuần túy
    rng_ref = np.random.default_rng(0)
    ss_ref = SpectralSubtractor(noise_mode="minstat")
    for _ in range(4 * 62):
        ss_ref.process(rng_ref.normal(0, 0.02, 256).astype(np.float32))

    diff_db = 20 * np.log10(ss.noise_mag[bin_1k] / ss_ref.noise_mag[bin_1k])
    assert abs(diff_db) <= 3.0


def test_minstat_continuous_tone_rises():
    ss = SpectralSubtractor(noise_mode="minstat")
    bin_1k = int(round(1000 / (SR / 512)))

    blocks_tone = [sine(1000, 256 / SR, amp=0.5).astype(np.float32) for _ in range(3 * SR // 256)]
    est_initial = None
    for i, b in enumerate(blocks_tone):
        ss.process(b)
        if i == 10:
            est_initial = ss.noise_mag[bin_1k]

    est_final = ss.noise_mag[bin_1k]
    assert est_final > est_initial
