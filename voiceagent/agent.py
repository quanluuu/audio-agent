"""Agent: DSP tools, plus a hand-built LangGraph StateGraph (llm <-> tools; confirm node comes in Task 16)."""

import json
from dataclasses import asdict

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from voiceagent.pipeline import Pipeline

DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"  # per api-docs.deepseek.com 2026-11: deepseek-v4-flash is a retired alias

SYSTEM_PROMPT = """Bạn là trợ lý giọng nói tiếng Việt. Câu trả lời hiện trên terminal, nên giữ ngắn (1-3 câu), không dùng markdown hay danh sách.
Lời người dùng đến từ nhận dạng giọng nói nên có thể sai chính tả; đoán ý theo ngữ cảnh.
Bạn điều khiển chuỗi khử ồn: band-pass -> trừ phổ (spectral subtraction) -> cổng ồn (noise gate). Gọi get_status trước khi chỉnh bộ lọc, và giữ tham số trong miền an toàn ghi ở mô tả tool.
Người dùng than giọng rè, kim loại, lách tách: tăng smoothing hoặc giảm alpha. Than ồn thay đổi hoặc không muốn im lặng để hiệu chuẩn: chuyển set_noise_estimator sang minstat.
Câu nghe không rõ hoặc mơ hồ thì hỏi lại, không đoán rồi hành động."""


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


def make_model(env):
    """ChatDeepSeek with thinking off (DeepSeek enables it by default; the spec keeps it out of scope)."""
    from langchain_deepseek import ChatDeepSeek  # network client: keep it out of tests
    return ChatDeepSeek(model=env.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL, api_key=env["DEEPSEEK_API_KEY"],
                        timeout=30, max_retries=1, extra_body={"thinking": {"type": "disabled"}})


class AgentState(MessagesState):
    preset: dict  # tool_call_id -> result decided by the confirm node (cancelled / invalid), Task 16


def _run_tool(tools_by_name: dict, name: str, args: dict) -> dict:
    fn = tools_by_name.get(name)
    if fn is None:
        return {"error": f"không có tool '{name}'"}
    try:
        return fn(**args)
    except (TypeError, ValueError) as e:  # missing/extra args from the model, or an action that rejects input
        return {"error": str(e)}


def build_graph(model, dsp_tools: list, action_tools: list, checkpointer, out=print):
    """Compile llm -> (tools -> llm)* -> END. model is a chat model with bind_tools (ChatDeepSeek or a fake)."""
    all_tools = dsp_tools + action_tools
    tools_by_name = {t.__name__: t for t in all_tools}
    bound = model.bind_tools(all_tools)

    def llm(state: AgentState) -> dict:
        # system prompt joined per call, never stored in the checkpointed history
        return {"messages": [bound.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])]}

    def route_after_llm(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    def tools(state: AgentState) -> dict:
        preset = state.get("preset") or {}
        messages = []
        # sequential, and one ToolMessage per call: the API rejects a turn with an unanswered tool call
        for c in state["messages"][-1].tool_calls:
            result = preset[c["id"]] if c["id"] in preset else _run_tool(tools_by_name, c["name"], c["args"])
            args = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
            out(f"[tool] {c['name']}({args}) -> {result}")
            messages.append(ToolMessage(json.dumps(result, ensure_ascii=False), tool_call_id=c["id"]))
        return {"messages": messages, "preset": {}}

    graph = StateGraph(AgentState)
    graph.add_node("llm", llm)
    graph.add_node("tools", tools)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route_after_llm, ["tools", END])
    graph.add_edge("tools", "llm")
    return graph.compile(checkpointer=checkpointer)
