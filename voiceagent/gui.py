"""Optional Tkinter window (--gui). Same session as the CLI; only the input loop differs."""

import math
import queue
import threading

import numpy as np

from voiceagent.voice import append_transcript, listen_turn

N_FFT = 512
# dark -> violet -> pink -> yellow; red rises by 1 per step, so every entry is distinct
PALETTE = [f"#{i:02x}{max(0, 2 * i - 255):02x}{int(255 * math.sin(math.pi * i / 255)):02x}" for i in range(256)]
_WINDOW = np.hanning(N_FFT + 1)[:-1]  # periodic Hann
_SCALE = 2 / _WINDOW.sum()  # a full-scale sine reads 0 dB


def spectrum_column(samples: np.ndarray, rows: int = 128, floor_db: float = -100.0, ceil_db: float = -20.0) -> list[str]:
    """Colour one spectrogram column from the last 512 samples, bottom (0 Hz) to top (8 kHz at 16 kHz)."""
    frame = np.zeros(N_FFT, np.float32)
    tail = samples[-N_FFT:]
    frame[N_FFT - len(tail):] = tail
    mag = np.abs(np.fft.rfft(frame * _WINDOW))[:N_FFT // 2] * _SCALE  # drop the Nyquist bin: 256 bins
    db = 20 * np.log10(mag.reshape(rows, -1).max(axis=1) + 1e-12)  # max keeps a pure tone at full height
    level = np.clip((db - floor_db) / (ceil_db - floor_db), 0, 1)
    return [PALETTE[int(v * 255)] for v in level]


class TurnBridge:
    """Carries a turn's output from its worker thread to the Tk main thread, the only thread that touches widgets."""

    def __init__(self, timeout: float = 120.0):
        self._messages = queue.Queue()
        self.timeout = timeout

    def out(self, text: str) -> None:
        self._messages.put(("out", text))

    def done(self) -> None:
        self._messages.put(("done", None))

    def ask_yes_no(self) -> bool:
        """Block the worker until the main thread replies; no reply within timeout counts as no."""
        answer = queue.Queue(maxsize=1)  # one per question, so a late reply can never answer the next one
        self._messages.put(("ask", answer.put))
        try:
            return bool(answer.get(timeout=self.timeout))
        except queue.Empty:
            return False

    def drain(self) -> list:
        items = []
        while True:
            try:
                items.append(self._messages.get_nowait())
            except queue.Empty:
                return items


def run_gui(pipeline, utterance, raw: list, clean: list, transcribe, graph, config: dict, transcript) -> None:
    """Show the chat window and block until it is closed. Raises ImportError when tkinter is missing."""
    import tkinter as tk  # optional: Ubuntu needs python3.13-tk, tests never open a window
    from tkinter import messagebox, scrolledtext

    bridge = TurnBridge()
    state = {"busy": False, "recording": False, "last": "", "x": 0, "tick": 0}

    root = tk.Tk()
    root.title("Voice Agent")

    side = tk.Frame(root)
    side.pack(side="right", fill="y", padx=(0, 8), pady=8)
    width, rows = 300, 128  # 300 columns x 50 ms = 15 s of history
    spectrograms = []
    for title, blocks in (("Trước lọc", raw), ("Sau lọc", clean)):
        tk.Label(side, text=title, anchor="w").pack(fill="x")
        canvas = tk.Canvas(side, width=width, height=rows, highlightthickness=0, bg=PALETTE[0])
        canvas.pack(pady=(0, 6))
        image = tk.PhotoImage(width=width, height=rows)
        canvas.create_image(0, 0, image=image, anchor="nw")
        cursor = canvas.create_line(0, 0, 0, rows, fill="white")
        spectrograms.append((blocks, image, canvas, cursor))
    meters = {}
    for name in ("in", "out"):
        row = tk.Frame(side)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=name, width=4, anchor="w").pack(side="left")
        bar = tk.Canvas(row, width=200, height=12, highlightthickness=0, bg="#303030")
        bar.pack(side="left")
        label = tk.Label(row, width=8, anchor="e")
        label.pack(side="left")
        meters[name] = (bar, bar.create_rectangle(0, 0, 0, 12, fill="#4caf50", width=0), label)
    timing = tk.Label(side, anchor="w")
    timing.pack(fill="x")

    chat = scrolledtext.ScrolledText(root, state="disabled", wrap="word", width=80, height=24)
    chat.pack(fill="both", expand=True, padx=8, pady=(8, 4))
    bar = tk.Frame(root)
    bar.pack(fill="x", padx=8, pady=(0, 8))
    entry = tk.Entry(bar)
    entry.pack(side="left", fill="x", expand=True)
    send = tk.Button(bar, text="Gửi")
    send.pack(side="left", padx=4)
    talk = tk.Button(bar, text="Giữ để nói")
    talk.pack(side="left")
    status = tk.Label(bar, width=16, anchor="w")
    status.pack(side="left", padx=8)

    def show(text: str) -> None:
        state["last"] = text
        chat.configure(state="normal")
        chat.insert("end", text + "\n")
        chat.configure(state="disabled")
        chat.see("end")

    def set_busy(busy: bool, label: str = "") -> None:
        state["busy"] = busy
        for button in (send, talk):
            button.configure(state="disabled" if busy else "normal")
        status.configure(text=label)

    def turn(wav=None, text=None) -> None:  # worker thread: never touch widgets here
        try:
            if wav is not None:
                text = listen_turn(wav, transcribe, transcript, out=bridge.out)
            if text and graph is not None:
                from voiceagent.agent import handle_turn
                handle_turn(text, graph, config, out=bridge.out, ask_yes_no=bridge.ask_yes_no,
                            log=lambda who, t: append_transcript(transcript, who, t))
        finally:
            bridge.done()

    def start_turn(**kw) -> None:
        set_busy(True, "đang xử lý…")
        threading.Thread(target=turn, kwargs=kw, daemon=True).start()

    def on_send(_event=None) -> None:
        text = entry.get().strip()
        if state["busy"] or not text:
            return
        entry.delete(0, "end")
        show(f"Bạn (gõ): {text}")
        append_transcript(transcript, "Bạn (gõ)", text)
        start_turn(text=text)

    def on_press(_event) -> None:
        if state["busy"]:
            return
        utterance.start()
        state["recording"] = True
        status.configure(text="đang nghe…")

    def on_release(_event) -> None:
        if not state["recording"]:
            return
        state["recording"] = False
        start_turn(wav=utterance.stop())

    def poll() -> None:
        for kind, value in bridge.drain():
            if kind == "out":
                show(value)
            elif kind == "ask":
                value(messagebox.askyesno("Xác nhận", state["last"], parent=root))
            else:
                set_busy(False)
        draw()
        root.after(50, poll)  # rescheduled after the loop, so a modal askyesno never re-enters poll

    def draw() -> None:
        # read-only tail of lists the audio thread appends to; no work is added to the audio thread
        x = state["x"]
        for blocks, image, canvas, cursor in spectrograms:
            tail = blocks[-2:]
            if len(tail) == 2:
                image.put(" ".join("{%s}" % c for c in reversed(spectrum_column(np.concatenate(tail)))), to=(x, 0))
            canvas.coords(cursor, x + 1, 0, x + 1, rows)
        state["x"] = (x + 1) % width
        state["tick"] += 1
        if state["tick"] % 2:  # meters at 10 Hz
            return
        m = pipeline.meters
        for name, db in (("in", m.input_db), ("out", m.output_db)):
            bar, fill, label = meters[name]
            bar.coords(fill, 0, 0, 200 * min(max((db + 80) / 80, 0), 1), 12)
            label.configure(text=f"{db:6.1f} dB")
        timing.configure(text=f"{m.process_ms:.2f} ms/khối · {m.underruns} underrun")

    send.configure(command=on_send)
    entry.bind("<Return>", on_send)
    talk.bind("<ButtonPress-1>", on_press)
    talk.bind("<ButtonRelease-1>", on_release)
    if graph is None:
        show("Agent chưa chạy (thiếu DEEPSEEK_API_KEY hoặc --no-agent): chỉ có STT + transcript.")
    entry.focus_set()
    root.after(50, poll)
    root.mainloop()
