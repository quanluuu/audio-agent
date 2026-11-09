import io
from types import SimpleNamespace

import numpy as np
from scipy.io import wavfile

from voiceagent.voice import append_transcript, listen_turn, whisper_transcribe


def wav(samples):
    buf = io.BytesIO()
    wavfile.write(buf, 16000, np.asarray(samples, np.float32))
    return buf.getvalue()


WAV = wav([0.1, -0.1, 0.2])


class FakeModel:
    def __init__(self, texts):
        self.texts, self.calls = texts, []

    def transcribe(self, audio, language):
        self.calls.append((audio, language))
        return iter(SimpleNamespace(text=t) for t in self.texts), None


def test_whisper_joins_segments_and_strips():
    model = FakeModel([" xin", " chào"])
    assert whisper_transcribe(model)(WAV) == "xin chào"


def test_whisper_gets_float32_samples_in_vietnamese():
    model = FakeModel([])
    whisper_transcribe(model)(WAV)
    audio, language = model.calls[0]
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0.1, -0.1, 0.2], atol=1e-7)
    assert language == "vi"


def test_whisper_no_segments_gives_empty_string():
    assert whisper_transcribe(FakeModel([]))(WAV) == ""


def test_append_transcript_writes_utf8_lines(tmp_path):
    path = tmp_path / "t_transcript.md"
    append_transcript(path, "Bạn", "xin chào")
    append_transcript(path, "Bạn (gõ)", "tạm biệt")
    assert path.read_text(encoding="utf-8") == "Bạn: xin chào\nBạn (gõ): tạm biệt\n"


def test_listen_turn_prints_and_logs_text(tmp_path):
    path, lines = tmp_path / "t.md", []
    assert listen_turn(b"RIFF", lambda wav: "xin chào", path, out=lines.append) == "xin chào"
    assert "Bạn: xin chào" in lines
    assert path.read_text(encoding="utf-8") == "Bạn: xin chào\n"


def test_listen_turn_stt_error_is_reported_not_raised(tmp_path):
    def broken(wav):
        raise RuntimeError("model gone")

    path, lines = tmp_path / "t.md", []
    assert listen_turn(b"RIFF", broken, path, out=lines.append) == ""
    assert any("model gone" in line for line in lines)
    assert not path.exists()


def test_listen_turn_without_stt_asks_to_type(tmp_path):
    lines = []
    assert listen_turn(b"RIFF", None, tmp_path / "t.md", out=lines.append) == ""
    assert any("gõ chữ" in line for line in lines)


def test_listen_turn_skips_short_or_silent_utterance(tmp_path):
    path, lines = tmp_path / "t.md", []
    assert listen_turn(None, lambda wav: "x", path, out=lines.append) == ""
    assert listen_turn(b"RIFF", lambda wav: "", path, out=lines.append) == ""
    assert not path.exists()
