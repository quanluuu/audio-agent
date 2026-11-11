"""Listening (STT) and the session transcript."""

import io
import os
from pathlib import Path
from typing import Callable

from scipy.io import wavfile

DEFAULT_GEMINI_STT_MODEL = "gemini-3.5-flash-lite"  # cheapest stable model with audio input (2026-11)
GEMINI_PROMPT = "Chép lại nguyên văn lời nói tiếng Việt trong đoạn audio. Không thêm gì khác."


def load_whisper():
    """Load the faster-whisper model once (downloads on first run). Name from WHISPER_MODEL, default small."""
    from faster_whisper import WhisperModel  # heavy import: keep it out of tests
    print("đang tải model Whisper…")
    return WhisperModel(os.environ.get("WHISPER_MODEL") or "small", device="cpu", compute_type="int8")


def whisper_transcribe(model) -> Callable[[bytes], str]:
    """Return transcribe(wav_bytes) -> text, using an already loaded faster-whisper model."""

    def transcribe(wav_bytes: bytes) -> str:
        # pass samples, not the file: faster-whisper 1.2.1 decodes files via av, and av 19 dropped an arg it uses
        _, audio = wavfile.read(io.BytesIO(wav_bytes))  # our turns are already 16 kHz mono float32
        segments, _ = model.transcribe(audio, language="vi")
        return "".join(s.text for s in segments).strip()

    return transcribe


def gemini_transcribe(client, model: str) -> Callable[[bytes], str]:
    """Return transcribe(wav_bytes) -> text, sending the WAV inline to Gemini."""

    def transcribe(wav_bytes: bytes) -> str:
        part = {"inline_data": {"data": wav_bytes, "mime_type": "audio/wav"}}
        response = client.models.generate_content(model=model, contents=[GEMINI_PROMPT, part])
        return (response.text or "").strip()

    return transcribe


def _make_gemini(key: str):
    from google import genai  # optional backend: a missing package falls back to Whisper
    return genai.Client(api_key=key)


def make_transcriber(backend: str, env, load_whisper=load_whisper, make_gemini=_make_gemini,
                     out=print) -> Callable[[bytes], str]:
    """Pick the STT backend for --stt. Gemini without a key or package falls back to Whisper."""
    if backend == "gemini":
        try:
            key = env.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("thiếu GEMINI_API_KEY")
            return gemini_transcribe(make_gemini(key), env.get("GEMINI_STT_MODEL") or DEFAULT_GEMINI_STT_MODEL)
        except Exception as e:
            out(f"Không dùng được Gemini STT ({e}), chuyển sang Whisper.")
    return whisper_transcribe(load_whisper())


def append_transcript(path, speaker: str, text: str) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(f"{speaker}: {text}\n")


def listen_turn(wav: bytes | None, transcribe, transcript, out=print) -> str:
    """Transcribe one utterance, print it and log it. Returns the text ("" if nothing). Never raises."""
    if wav is None:
        out("Câu quá ngắn, bỏ qua.")
        return ""
    if transcribe is None:
        out("STT không dùng được, hãy gõ chữ.")
        return ""
    out("đang nghe chữ…")
    try:
        text = transcribe(wav)
    except Exception as e:  # network/model failure: drop this turn, keep the session alive
        out(f"Lỗi STT: {e}")
        return ""
    if not text:
        out("(không nghe được lời nào)")
        return ""
    out(f"Bạn: {text}")
    append_transcript(transcript, "Bạn", text)
    return text
