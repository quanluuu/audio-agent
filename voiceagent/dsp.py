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

class NoiseGate:
    """Noise gate with exponential attack/release smoothing."""

    def __init__(
        self,
        threshold_db: float = -45.0,
        floor_gain: float = 0.1,
        attack_coeff: float = 0.6,
        release_coeff: float = 0.1,
    ):
        if not -100.0 <= threshold_db <= 0.0:
            raise ValueError(f"threshold_db must be in [-100, 0], got {threshold_db}")
        self.threshold_db = threshold_db
        self.floor_gain = floor_gain
        self.attack_coeff = attack_coeff
        self.release_coeff = release_coeff
        self.gain = floor_gain  # Khởi đầu đóng cổng

    def process(self, block: np.ndarray) -> np.ndarray:
        level = rms_db(block)
        target = 1.0 if level >= self.threshold_db else self.floor_gain

        coeff = self.attack_coeff if target > self.gain else self.release_coeff
        self.gain += coeff * (target - self.gain)

        return (block * self.gain).astype(block.dtype)


class SpectralSubtractor:
    """Single-channel spectral subtraction using sqrt-Hann windows and OLA."""

    def __init__(
        self,
        n_fft: int = 512,
        hop: int = 256,
        alpha: float = 2.0,
        beta: float = 0.02,
        smoothing: float = 0.0,
        noise_mode: str = "calibrate",
    ):
        if not (alpha >= 0.0 and 0.0 <= beta <= 1.0):
            raise ValueError(f"need alpha >= 0 and 0 <= beta <= 1, got alpha={alpha}, beta={beta}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"need 0 <= smoothing < 1, got {smoothing}")
        if noise_mode not in ("calibrate", "minstat"):
            raise ValueError(f"noise_mode must be 'calibrate' or 'minstat', got {noise_mode}")

        self.n_fft = n_fft
        self.hop = hop
        self.alpha = alpha
        self.beta = beta
        self.smoothing = smoothing
        self.noise_mode = noise_mode

        n_bins = n_fft // 2 + 1
        # Cửa sổ phân tích và tổng hợp sqrt-Hann (periodic để tổng OLA chuẩn 1.0)
        self.window = np.sqrt(signal.windows.hann(n_fft, sym=False)).astype(np.float32)

        # Bộ đệm nhớ khối trước (256 mẫu) để ghép thành khung 512 mẫu
        self.prev_input = np.zeros(hop, dtype=np.float32)

        # Bộ đệm chồng chập OLA (Overlap-Add)
        self.ola_buffer = np.zeros(hop, dtype=np.float32)

        # Phổ ồn nền (257 bins tần số từ 0 Hz đến 8000 Hz)
        self.noise_mag = np.zeros(n_bins, dtype=np.float32)

        # Gain làm mượt theo thời gian (Task 9)
        self.prev_gain = np.ones(n_bins, dtype=np.float32)

        # Trạng thái Minstat (Task 10 - Martin, 2001)
        self.p_smoothed = np.zeros(n_bins, dtype=np.float32)
        self.ring_buffer = np.zeros((96, n_bins), dtype=np.float32)
        self.ring_idx = 0
        self.frames_seen = 0

    def set_noise_spectrum(self, noise_mag: np.ndarray) -> None:
        """Cập nhật phổ biên độ ồn nền đã đo được."""
        self.noise_mag = noise_mag.astype(np.float32).copy()

    def set_noise_mode(self, mode: str) -> None:
        """Đổi chế độ ước lượng ồn ('calibrate' | 'minstat')."""
        if mode not in ("calibrate", "minstat"):
            raise ValueError(f"noise_mode must be 'calibrate' or 'minstat', got {mode}")
        self.noise_mode = mode

    def process(self, block: np.ndarray) -> np.ndarray:
        # 1. Ghép 256 mẫu cũ + 256 mẫu mới -> Khung 512 mẫu
        frame = np.concatenate([self.prev_input, block])
        self.prev_input = block.copy()

        # 2. Nhân cửa sổ phân tích
        windowed = frame * self.window

        # 3. Biến đổi Fourier sang miền tần số (FFT)
        spec = np.fft.rfft(windowed)
        mag = np.abs(spec)  # Biên độ (độ to từng tần số)

        # Ước lượng ồn liên tục nếu ở chế độ minstat
        if self.noise_mode == "minstat":
            power = mag**2
            if self.frames_seen == 0:
                self.p_smoothed = power.copy()
            else:
                self.p_smoothed = 0.85 * self.p_smoothed + 0.15 * power
            self.ring_buffer[self.ring_idx] = self.p_smoothed
            self.ring_idx = (self.ring_idx + 1) % 96
            self.frames_seen += 1
            valid_frames = min(self.frames_seen, 96)
            p_min = np.min(self.ring_buffer[:valid_frames], axis=0)
            self.noise_mag = np.sqrt(1.5 * p_min)

        # 4. Tính hệ số lọc Gain G[k]
        if self.alpha == 0.0 or (self.noise_mode == "calibrate" and np.all(self.noise_mag == 0)):
            raw_gain = np.ones_like(mag)
        else:
            snr_sub = 1.0 - self.alpha * (self.noise_mag / (mag + EPS))
            raw_gain = np.maximum(snr_sub, self.beta)

        # Làm mượt Gain theo thời gian (Task 9)
        if self.smoothing > 0.0:
            gain = self.smoothing * self.prev_gain + (1.0 - self.smoothing) * raw_gain
        else:
            gain = raw_gain
        self.prev_gain = gain.copy()

        # 5. Nhân Gain vào phổ (giữ nguyên pha phase)
        clean_spec = spec * gain

        # 6. Biến đổi ngược về sóng âm thời gian (IFFT)
        clean_frame = np.fft.irfft(clean_spec) * self.window

        # 7. Ghép chồng chập (Overlap-Add)
        out = self.ola_buffer + clean_frame[:self.hop]
        self.ola_buffer = clean_frame[self.hop:].copy()

        return out.astype(block.dtype)

