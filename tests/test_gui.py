import re
import subprocess
import sys
import threading

import numpy as np

from voiceagent.gui import PALETTE, TurnBridge, spectrum_column


def test_out_and_done_drain_in_order():
    bridge = TurnBridge()
    bridge.out("Bạn: xin chào")
    bridge.out("Agent: Chào bạn.")
    bridge.done()
    assert bridge.drain() == [("out", "Bạn: xin chào"), ("out", "Agent: Chào bạn."), ("done", None)]
    assert bridge.drain() == []


def test_ask_waits_for_reply_from_main_thread():
    bridge, result = TurnBridge(), []
    worker = threading.Thread(target=lambda: result.append(bridge.ask_yes_no()))
    worker.start()
    while not (msgs := bridge.drain()):
        pass
    [(kind, reply)] = msgs
    assert kind == "ask"
    reply(True)
    worker.join(timeout=5)
    assert result == [True]


def test_ask_times_out_as_no_and_late_reply_is_ignored():
    bridge = TurnBridge(timeout=0.05)
    assert bridge.ask_yes_no() is False
    [(_, late_reply)] = bridge.drain()
    late_reply(True)  # user answered after the turn gave up
    worker_result = []
    worker = threading.Thread(target=lambda: worker_result.append(bridge.ask_yes_no()))
    worker.start()
    worker.join(timeout=5)
    assert worker_result == [False]  # the late True must not answer the next question


def test_importing_gui_does_not_import_tkinter():
    code = "import sys, voiceagent.gui; sys.exit('tkinter' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def sine(freq, amp=0.5, n=512, sr=16000):
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)


def test_spectrum_column_shape_and_colors():
    col = spectrum_column(sine(1000))
    assert len(col) == 128
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in col)


def test_spectrum_column_puts_1khz_at_row_16_from_bottom():
    levels = [PALETTE.index(c) for c in spectrum_column(sine(1000, amp=0.001))]  # -60 dB: below the ceiling
    assert levels.index(max(levels)) == 16


def test_spectrum_column_silence_is_background_and_loud_is_clamped():
    assert set(spectrum_column(np.zeros(512, np.float32))) == {PALETTE[0]}
    loud = spectrum_column(sine(1000, amp=100.0))
    assert loud[16] == PALETTE[-1]


def test_spectrum_column_pads_short_input():
    assert len(spectrum_column(np.zeros(100, np.float32))) == 128
