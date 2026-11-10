import inspect

import pytest

from voiceagent.agent import make_dsp_tools
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
