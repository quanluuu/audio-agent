"""Optional Tkinter window (--gui). Same session as the CLI; only the input loop differs."""

import queue
import threading

from voiceagent.voice import append_transcript, listen_turn


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


def run_gui(utterance, transcribe, graph, config: dict, transcript) -> None:
    """Show the chat window and block until it is closed. Raises ImportError when tkinter is missing."""
    import tkinter as tk  # optional: Ubuntu needs python3.13-tk, tests never open a window
    from tkinter import messagebox, scrolledtext

    bridge = TurnBridge()
    state = {"busy": False, "recording": False, "last": ""}

    root = tk.Tk()
    root.title("Voice Agent")
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
        root.after(50, poll)  # rescheduled after the loop, so a modal askyesno never re-enters poll

    send.configure(command=on_send)
    entry.bind("<Return>", on_send)
    talk.bind("<ButtonPress-1>", on_press)
    talk.bind("<ButtonRelease-1>", on_release)
    if graph is None:
        show("Agent chưa chạy (thiếu DEEPSEEK_API_KEY hoặc --no-agent): chỉ có STT + transcript.")
    entry.focus_set()
    root.after(50, poll)
    root.mainloop()
