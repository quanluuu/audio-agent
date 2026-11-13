import pytest

from voiceagent import actions
from voiceagent.actions import (ACTION_TOOLS, ALLOWED_APPS, describe_action, open_app, open_url, save_note,
                                search_web)


@pytest.fixture
def popen(monkeypatch):
    calls = []
    monkeypatch.setattr(actions.subprocess, "Popen", lambda args, **kw: calls.append((args, kw)))
    return calls


@pytest.fixture
def browser(monkeypatch):
    urls = []
    monkeypatch.setattr(actions.webbrowser, "open", lambda url: urls.append(url) or True)
    return urls


@pytest.fixture
def notes(tmp_path, monkeypatch):
    d = tmp_path / "notes"
    monkeypatch.setattr(actions, "NOTES_DIR", d)
    return d


def test_open_app_outside_allowlist_returns_error_without_popen(popen):
    assert "error" in open_app("cmd")
    assert "error" in open_app("notepad.exe & calc")
    assert popen == []


def test_open_app_runs_allowed_exe_without_shell(popen):
    assert open_app(" Notepad ") == {"status": "ok"}
    assert popen == [([ALLOWED_APPS["notepad"]], {})]


def test_open_url_rejects_unsafe_or_bad_urls(browser):
    for url in ["file:///C:/Windows", "javascript:alert(1)", "không phải url", "https://", "ftp://x.com"]:
        assert "error" in open_url(url), url
    assert browser == []


def test_open_url_opens_http(browser):
    assert open_url("https://vnexpress.net") == {"status": "ok"}
    assert browser == ["https://vnexpress.net"]


def test_search_web_encodes_vietnamese_and_spaces(browser):
    assert search_web("thời tiết Hà Nội")["status"] == "ok"
    assert browser == ["https://www.google.com/search?q=th%E1%BB%9Di+ti%E1%BA%BFt+H%C3%A0+N%E1%BB%99i"]


def test_search_web_rejects_empty_query(browser):
    assert "error" in search_web("  ")
    assert browser == []


def test_save_note_stays_inside_notes_dir(notes):
    result = save_note("../../evil", "x")
    path = notes / result["path"].split("/")[-1].split("\\")[-1]
    assert path.exists() and path.parent == notes
    assert path.read_text(encoding="utf-8") == "x"
    assert not (notes.parent.parent / "evil.md").exists()


def test_save_note_does_not_overwrite(notes):
    a = save_note("Tóm tắt buổi học", "một")
    b = save_note("Tóm tắt buổi học", "hai")
    assert a["path"] != b["path"]
    assert sorted(p.read_text(encoding="utf-8") for p in notes.iterdir()) == ["hai", "một"]


def test_describe_action_rejects_invalid():
    with pytest.raises(ValueError):
        describe_action("open_app", {"name": "cmd"})
    with pytest.raises(ValueError):
        describe_action("open_url", {"url": "file:///C:/"})
    with pytest.raises(ValueError):
        describe_action("open_app", {})
    with pytest.raises(ValueError):
        describe_action("format_disk", {})


def test_describe_action_asks_readable_questions():
    assert "Notepad" in describe_action("open_app", {"name": "notepad"})
    assert "https://vnexpress.net" in describe_action("open_url", {"url": "https://vnexpress.net"})
    assert "thời tiết" in describe_action("search_web", {"query": "thời tiết"})
    assert "Tóm tắt" in describe_action("save_note", {"title": "Tóm tắt", "content": "x"})


def test_action_tools_have_arg_docs():
    assert [t.__name__ for t in ACTION_TOOLS] == ["open_app", "open_url", "search_web", "save_note"]
    assert all("Args:" in t.__doc__ for t in ACTION_TOOLS)


def test_browser_failure_returns_error(monkeypatch):
    monkeypatch.setattr(actions.webbrowser, "open", lambda url: False)
    assert "error" in open_url("https://vnexpress.net")
    assert "error" in search_web("thời tiết")


def test_app_tables_use_absolute_paths_and_match_platform():
    import sys
    from pathlib import PurePosixPath, PureWindowsPath
    assert all(PureWindowsPath(p).is_absolute() for p in actions.WINDOWS_APPS.values())
    assert all(PurePosixPath(p).is_absolute() for p in actions.LINUX_APPS.values())
    assert {"notepad", "calculator"} <= set(actions.WINDOWS_APPS) & set(actions.LINUX_APPS)
    assert "paint" not in actions.LINUX_APPS
    assert ALLOWED_APPS is (actions.WINDOWS_APPS if sys.platform == "win32" else actions.LINUX_APPS)
