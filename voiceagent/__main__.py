"""CLI entry: python -m voiceagent [--list-devices] [--input N] [--output N] [--no-agent] [--monitor] [--all-filters] [--stt whisper|gemini]"""

import argparse
import os
import sys
import time
from pathlib import Path

import sounddevice as sd

from voiceagent.audio import UtteranceRecorder, open_stream, save_recording
from voiceagent.pipeline import Pipeline
from voiceagent.voice import append_transcript, listen_turn, make_transcriber


def format_status(pipeline: Pipeline) -> str:
    m = pipeline.meters
    return (f"in {m.input_db:6.1f} dB | out {m.output_db:6.1f} dB | "
            f"xử lý {m.process_ms:.2f} ms/khối | underrun {m.underruns}")


def enable_all_filters(pipeline: Pipeline) -> None:
    """Turn on band-pass + spectral subtraction (minstat, smoothing 0.5) + gate: spec evaluation config 5."""
    p = pipeline.params()
    pipeline.set_bandpass(True, p.low_hz, p.high_hz)
    # minstat tracks the noise floor continuously, so no silent first second is needed
    pipeline.set_noise_estimator("minstat")
    pipeline.set_spectral_subtraction(True, p.alpha, p.beta, 0.5)
    pipeline.set_noise_gate(True, p.gate_threshold_db)


def load_env(path=".env", env=os.environ) -> None:
    """Read KEY=value lines from .env into env. Variables already set in the shell win."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            env.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    parser = argparse.ArgumentParser(prog="voiceagent", description="Khử ồn giọng nói thời gian thực.")
    parser.add_argument("--list-devices", action="store_true", help="in danh sách thiết bị âm thanh rồi thoát")
    parser.add_argument("--input", type=int, help="số thiết bị micro (xem --list-devices)")
    parser.add_argument("--output", type=int, help="số thiết bị loa/tai nghe")
    parser.add_argument("--no-agent", action="store_true", help="chỉ chạy DSP, không STT, không agent (monitor bật sẵn)")
    parser.add_argument("--monitor", action="store_true", help="nghe lại giọng mình qua tai nghe")
    parser.add_argument("--all-filters", action="store_true",
                        help="bật cả band-pass + spectral subtraction (minstat) + noise gate")
    parser.add_argument("--stt", choices=["whisper", "gemini"], default="whisper",
                        help="nhận dạng giọng: whisper (local, mặc định) hoặc gemini (cần GEMINI_API_KEY)")
    args = parser.parse_args()
    load_env()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    pipeline = Pipeline()
    pipeline.set_monitor(args.monitor or args.no_agent)
    if args.all_filters:
        enable_all_filters(pipeline)
    raw, clean = [], []
    utterance = UtteranceRecorder(pipeline.sr)
    try:
        stream = open_stream(pipeline, raw, clean, args.input, args.output, utterance)
    except (sd.PortAudioError, ValueError) as e:
        print(f"Lỗi thiết bị âm thanh (input={args.input}, output={args.output}): {e}", file=sys.stderr)
        print("Chạy `python -m voiceagent --list-devices` để xem số thiết bị.", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    transcript = Path("out") / f"{stamp}_transcript.md"
    transcript.parent.mkdir(exist_ok=True)
    transcribe = None
    if not args.no_agent:
        try:
            transcribe = make_transcriber(args.stt, os.environ)
        except Exception as e:  # no wheel, no network on first download: still usable by typing
            print(f"Không tải được model Whisper ({e}). Vẫn gõ chữ được.")

    if args.no_agent:
        print("Đang chạy. Đeo tai nghe để tránh hú. Enter: xem chỉ số, quit: thoát.")
    else:
        print("Đang chạy. Enter: bắt đầu/dừng nói, gõ chữ rồi Enter: gửi chữ, status: chỉ số, quit: thoát.")
    recording = False
    try:
        with stream:
            while True:
                try:
                    line = input("> ").strip()
                except EOFError:
                    break
                if line.lower() == "quit":
                    break
                if args.no_agent or line.lower() == "status":
                    print(format_status(pipeline))
                elif line:
                    append_transcript(transcript, "Bạn (gõ)", line)
                elif not recording:
                    utterance.start()
                    recording = True
                    print("🎙 đang nghe… (Enter để dừng)")
                else:
                    recording = False
                    listen_turn(utterance.stop(), transcribe, transcript)
    except KeyboardInterrupt:
        print()
    finally:
        raw_path, clean_path = save_recording(raw, clean, "out", pipeline.sr, stamp)
        print(f"Đã ghi {raw_path} và {clean_path}")
        print(format_status(pipeline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
