import subprocess
import sys
import threading

from voiceagent.gui import TurnBridge


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
