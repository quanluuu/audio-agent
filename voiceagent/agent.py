"""Agent: DSP tools now; LangGraph graph (llm -> confirm -> tools) comes in Task 15-16."""

from dataclasses import asdict

from voiceagent.pipeline import Pipeline


def make_dsp_tools(pipeline: Pipeline) -> list:
    """Build the 8 DSP tools bound to pipeline. Bad arguments return {"error": ...} and change nothing."""

    def run(setter, *args) -> dict:
        try:
            setter(*args)
        except (ValueError, TypeError) as e:  # Pipeline validates before touching Params
            return {"error": str(e)}
        return {"status": "ok"}

    def get_status() -> dict:
        """Xem tham số bộ lọc hiện tại và chỉ số đo (mức vào/ra dB, ồn nền, ms/khối, underrun). Gọi trước khi chỉnh bộ lọc."""
        return {"params": asdict(pipeline.params()), "meters": asdict(pipeline.meters)}

    def calibrate_noise(seconds: float = 1.0) -> dict:
        """Đo lại phổ ồn nền ở chế độ calibrate. Người dùng phải im lặng trong lúc đo.

        Args:
            seconds: Thời gian đo, giây, > 0. Thường 1.0.
        """
        return run(pipeline.calibrate_noise, seconds)

    def set_bandpass(enabled: bool, low_hz: float, high_hz: float) -> dict:
        """Bật/tắt và chỉnh bộ lọc thông dải, bỏ tần số ngoài dải giọng nói.

        Args:
            enabled: Bật (true) hay tắt (false).
            low_hz: Tần số cắt dưới, Hz. An toàn 80-300.
            high_hz: Tần số cắt trên, Hz, phải < 8000. An toàn 3400-7000; thấp quá làm mất phụ âm.
        """
        return run(pipeline.set_bandpass, enabled, low_hz, high_hz)

    def set_spectral_subtraction(enabled: bool, alpha: float, beta: float, smoothing: float) -> dict:
        """Bật/tắt và chỉnh trừ phổ (khử ồn chính).

        Args:
            enabled: Bật (true) hay tắt (false).
            alpha: Mức trừ ồn, >= 0. An toàn 1-3; cao thì sạch ồn hơn nhưng giọng rè, mất phụ âm.
            beta: Sàn phổ giữ lại, 0-1. An toàn 0.01-0.1; cao thì bớt rè nhưng còn ồn.
            smoothing: Độ mượt gain theo thời gian, 0 <= smoothing < 1. An toàn 0-0.7; tăng khi nghe lách tách.
        """
        return run(pipeline.set_spectral_subtraction, enabled, alpha, beta, smoothing)

    def set_noise_estimator(mode: str) -> dict:
        """Chọn cách ước lượng ồn nền cho trừ phổ.

        Args:
            mode: "calibrate" (đo lúc im lặng, hợp ồn đều) hoặc "minstat" (tự bám ồn liên tục, hợp ồn thay đổi).
        """
        return run(pipeline.set_noise_estimator, mode)

    def set_noise_gate(enabled: bool, threshold_db: float) -> dict:
        """Bật/tắt và chỉnh cổng ồn: tắt tiếng khi tín hiệu dưới ngưỡng.

        Args:
            enabled: Bật (true) hay tắt (false).
            threshold_db: Ngưỡng, dBFS, từ -100 đến 0. An toàn -55 đến -35; cao quá thì cắt mất đầu/cuối câu.
        """
        return run(pipeline.set_noise_gate, enabled, threshold_db)

    def set_bypass(enabled: bool) -> dict:
        """Bỏ qua toàn bộ bộ lọc để so sánh với tín hiệu gốc.

        Args:
            enabled: true = bỏ qua bộ lọc, false = lọc bình thường.
        """
        return run(pipeline.set_bypass, enabled)

    def set_monitor(enabled: bool) -> dict:
        """Bật/tắt nghe lại giọng mình (đã khử ồn) qua tai nghe.

        Args:
            enabled: Bật (true) hay tắt (false).
        """
        return run(pipeline.set_monitor, enabled)

    return [get_status, calibrate_noise, set_bandpass, set_spectral_subtraction,
            set_noise_estimator, set_noise_gate, set_bypass, set_monitor]
