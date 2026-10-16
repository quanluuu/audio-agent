"""CLI entry: python -m voiceagent [--list-devices] [--input N] [--output N] [--no-agent]"""

import argparse
import sys

import sounddevice as sd

from voiceagent.audio import open_stream, save_recording
from voiceagent.pipeline import Pipeline


def format_status(pipeline: Pipeline) -> str:
    m = pipeline.meters
    return (f"in {m.input_db:6.1f} dB | out {m.output_db:6.1f} dB | "
            f"xử lý {m.process_ms:.2f} ms/khối | underrun {m.underruns}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="voiceagent", description="Khử ồn giọng nói thời gian thực.")
    parser.add_argument("--list-devices", action="store_true", help="in danh sách thiết bị âm thanh rồi thoát")
    parser.add_argument("--input", type=int, help="số thiết bị micro (xem --list-devices)")
    parser.add_argument("--output", type=int, help="số thiết bị loa/tai nghe")
    parser.add_argument("--no-agent", action="store_true", help="chỉ chạy DSP, không dùng Gemini")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    pipeline = Pipeline()
    raw, clean = [], []
    try:
        stream = open_stream(pipeline, raw, clean, args.input, args.output)
    except (sd.PortAudioError, ValueError) as e:
        print(f"Lỗi thiết bị âm thanh (input={args.input}, output={args.output}): {e}", file=sys.stderr)
        print("Chạy `python -m voiceagent --list-devices` để xem số thiết bị.", file=sys.stderr)
        return 2

    print("Đang chạy. Đeo tai nghe để tránh hú. Enter: xem chỉ số, quit: thoát.")
    try:
        with stream:
            while True:
                try:
                    line = input("> ").strip().lower()
                except EOFError:
                    break
                if line == "quit":
                    break
                print(format_status(pipeline))
    except KeyboardInterrupt:
        print()
    finally:
        raw_path, clean_path = save_recording(raw, clean, "out", pipeline.sr)
        print(f"Đã ghi {raw_path} và {clean_path}")
        print(format_status(pipeline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
