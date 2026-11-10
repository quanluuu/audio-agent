"""External actions the agent may take after the user confirms (Task 16 node `confirm`).

No LangGraph here and no shell: every tool re-checks its own input, so a call that skipped
confirmation still cannot open an unlisted exe, a file:// URL, or write outside notes/.
"""

import os
import re
import subprocess
import time
import urllib.parse
import webbrowser
from pathlib import Path

_SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
_STORE_APPS = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps"
# full paths so a planted notepad.exe in the working dir or PATH is never picked up
ALLOWED_APPS = {
    "notepad": str(_SYSTEM32 / "notepad.exe"),
    "calculator": str(_SYSTEM32 / "calc.exe"),
    "paint": str(_STORE_APPS / "mspaint.exe"),  # Windows 11 Paint is a Store app with an execution alias
}
NOTES_DIR = Path("notes")


def _check_app(name: str) -> str:
    key = name.strip().lower()
    if key not in ALLOWED_APPS:
        raise ValueError(f"'{name}' không có trong danh sách app được mở: {', '.join(ALLOWED_APPS)}")
    return key


def _check_url(url: str) -> str:
    parts = urllib.parse.urlparse(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"chỉ mở được URL http:// hoặc https://, không mở '{url}'")
    return url.strip()


def _check_query(query: str) -> str:
    if not query.strip():
        raise ValueError("từ khóa tìm kiếm rỗng")
    return query.strip()


def _slug(title: str) -> str:
    # keep letters (incl. Vietnamese), digits and '-'; everything else, '/', '.', '_' included, becomes '-'
    return re.sub(r"[\W_]+", "-", title).strip("-").lower() or "ghi-chu"


def open_app(name: str) -> dict:
    """Mở một ứng dụng Windows có trong danh sách cho phép.

    Args:
        name: Tên app: notepad, calculator hoặc paint.
    """
    try:
        key = _check_app(name)
    except ValueError as e:
        return {"error": str(e)}
    subprocess.Popen([ALLOWED_APPS[key]])
    return {"status": "ok"}


def open_url(url: str) -> dict:
    """Mở một trang web trong trình duyệt mặc định.

    Args:
        url: Địa chỉ đầy đủ, bắt đầu bằng http:// hoặc https://.
    """
    try:
        url = _check_url(url)
    except ValueError as e:
        return {"error": str(e)}
    webbrowser.open(url)
    return {"status": "ok"}


def search_web(query: str) -> dict:
    """Mở trang kết quả tìm kiếm Google trong trình duyệt để người dùng tự xem.

    Args:
        query: Từ khóa tìm kiếm, có thể là tiếng Việt.
    """
    try:
        query = _check_query(query)
    except ValueError as e:
        return {"error": str(e)}
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    webbrowser.open(url)
    return {"status": "ok", "url": url}


def save_note(title: str, content: str) -> dict:
    """Lưu một ghi chú hoặc bản tóm tắt thành file Markdown trong thư mục notes/.

    Args:
        title: Tiêu đề ngắn, dùng để đặt tên file.
        content: Nội dung ghi chú.
    """
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d')}_{_slug(title)}"
    i = 1
    while True:
        path = NOTES_DIR / (f"{stem}.md" if i == 1 else f"{stem}-{i}.md")
        try:
            with path.open("x", encoding="utf-8") as f:  # "x" fails instead of overwriting
                f.write(content)
            return {"status": "ok", "path": str(path)}
        except FileExistsError:
            i += 1


def describe_action(name: str, args: dict) -> str:
    """Validate an action call and return the yes/no question to ask. Raises ValueError if invalid."""
    try:
        if name == "open_app":
            return f"Mình sẽ mở {_check_app(args['name']).capitalize()}, đồng ý không?"
        if name == "open_url":
            return f"Mình sẽ mở {_check_url(args['url'])} trong trình duyệt, đồng ý không?"
        if name == "search_web":
            return f"Mình sẽ tìm Google \"{_check_query(args['query'])}\", đồng ý không?"
        if name == "save_note":
            return f"Mình sẽ lưu ghi chú \"{args['title']}\" vào {NOTES_DIR}/, đồng ý không?"
    except (KeyError, AttributeError) as e:  # missing arg or non-string value from the model
        raise ValueError(f"tham số sai cho {name}: {e}") from e
    raise ValueError(f"không có hành động '{name}'")


ACTION_TOOLS = [open_app, open_url, search_web, save_note]
