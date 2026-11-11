import inspect
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver

from voiceagent.agent import build_graph, make_dsp_tools
from voiceagent.pipeline import Pipeline

NAMES = ["get_status", "calibrate_noise", "set_bandpass", "set_spectral_subtraction",
         "set_noise_estimator", "set_noise_gate", "set_bypass", "set_monitor"]


@pytest.fixture
def setup():
    pipeline = Pipeline()
    return pipeline, {t.__name__: t for t in make_dsp_tools(pipeline)}


def test_tools_have_names_hints_and_arg_docs(setup):
    _, tools = setup
    assert list(tools) == NAMES
    for t in tools.values():
        sig = inspect.signature(t)
        assert all(p.annotation is not inspect.Parameter.empty for p in sig.parameters.values()), t.__name__
        if sig.parameters:
            assert "Args:" in t.__doc__ and all(f"{p}:" in t.__doc__ for p in sig.parameters), t.__name__


def test_valid_args_change_params(setup):
    pipeline, tools = setup
    assert tools["set_bandpass"](True, 200.0, 7000.0) == {"status": "ok"}
    assert tools["set_spectral_subtraction"](True, 1.0, 0.05, 0.5) == {"status": "ok"}
    assert tools["set_noise_estimator"]("minstat") == {"status": "ok"}
    assert tools["set_noise_gate"](True, -50.0) == {"status": "ok"}
    assert tools["set_bypass"](True) == {"status": "ok"}
    assert tools["set_monitor"](True) == {"status": "ok"}
    p = pipeline.params()
    assert (p.low_hz, p.high_hz, p.alpha, p.beta, p.smoothing) == (200.0, 7000.0, 1.0, 0.05, 0.5)
    assert p.ss_enabled and p.gate_enabled and p.gate_threshold_db == -50.0
    assert p.noise_mode == "minstat" and p.bypass and p.monitor


def test_calibrate_noise_starts_calibration(setup):
    pipeline, tools = setup
    assert tools["calibrate_noise"](0.5) == {"status": "ok"}
    assert pipeline.meters.calibrating


def test_get_status_returns_params_and_meters(setup):
    _, tools = setup
    status = tools["get_status"]()
    assert status["params"]["high_hz"] == 3800.0
    assert "underruns" in status["meters"]


@pytest.mark.parametrize("name,args", [
    ("set_bandpass", (True, 3000.0, 300.0)),
    ("set_bandpass", (True, 100.0, 9000.0)),
    ("set_spectral_subtraction", (True, -1.0, 0.02, 0.0)),
    ("set_spectral_subtraction", (True, 2.0, 0.02, 1.0)),
    ("set_spectral_subtraction", (True, "nhiều", 0.02, 0.0)),
    ("set_noise_estimator", ("ai",)),
    ("set_noise_gate", (True, 10.0)),
    ("calibrate_noise", (0.0,)),
])
def test_bad_args_return_error_and_keep_params(setup, name, args):
    pipeline, tools = setup
    before = pipeline.params()
    result = tools[name](*args)
    assert set(result) == {"error"}
    assert pipeline.params() == before


class FakeModel:
    """Stands in for ChatDeepSeek: replays scripted AIMessages and records what it was sent."""

    def __init__(self, *replies):
        self.replies, self.calls, self.bound = list(replies), [], None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def call(name, args, id="c1"):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def graph_with(*replies):
    pipeline, model, lines = Pipeline(), FakeModel(*replies), []
    graph = build_graph(model, make_dsp_tools(pipeline), [], InMemorySaver(), out=lines.append)
    return graph, model, pipeline, lines


def ask(graph, text, thread="t1"):
    return graph.invoke({"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": thread}})


def test_reply_without_tool_call_ends():
    graph, model, _, _ = graph_with(AIMessage("Paris."))
    result = ask(graph, "Thủ đô của Pháp là gì?")
    assert result["messages"][-1].content == "Paris."
    assert len(model.calls) == 1
    assert isinstance(model.calls[0][0], SystemMessage)
    assert {t.__name__ for t in model.bound} >= {"set_bypass", "get_status"}


def test_dsp_tool_call_runs_and_loops_back():
    graph, model, pipeline, lines = graph_with(
        AIMessage("", tool_calls=[call("set_bypass", {"enabled": True}, "abc")]), AIMessage("Đã tắt khử ồn."))
    result = ask(graph, "tắt khử ồn đi")
    assert pipeline.params().bypass
    assert len(model.calls) == 2
    tool_msg = model.calls[1][-1]
    assert isinstance(tool_msg, ToolMessage) and tool_msg.tool_call_id == "abc"
    assert json.loads(tool_msg.content) == {"status": "ok"}
    assert result["messages"][-1].content == "Đã tắt khử ồn."
    assert any(line.startswith("[tool] set_bypass") for line in lines)


def test_unknown_tool_and_missing_args_give_error_messages():
    graph, model, _, _ = graph_with(
        AIMessage("", tool_calls=[call("format_disk", {}, "a"), call("set_bandpass", {"enabled": True}, "b")]),
        AIMessage("Không làm được."))
    ask(graph, "làm gì đó")
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["a", "b"]
    assert all("error" in json.loads(m.content) for m in tool_msgs)


def test_tool_schema_has_required_described_params():
    set_bandpass = {t.__name__: t for t in make_dsp_tools(Pipeline())}["set_bandpass"]
    params = convert_to_openai_tool(set_bandpass)["function"]["parameters"]
    assert params["required"] == ["enabled", "low_hz", "high_hz"]
    assert all(p.get("description") for p in params["properties"].values())


def test_same_thread_remembers_previous_turn():
    graph, model, _, _ = graph_with(AIMessage("Chào Quân."), AIMessage("Bạn tên Quân."))
    ask(graph, "mình tên Quân")
    ask(graph, "mình tên gì?")
    texts = [m.content for m in model.calls[1]]
    assert "mình tên Quân" in texts and "Chào Quân." in texts
