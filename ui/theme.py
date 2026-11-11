"""终端界面的配色、样式与控制台实例。

配色只用于强调（标题、边框、状态），正文沿用终端默认前景色，
这样在浅色和深色终端里都不会出现"白字白底"的问题。
"""
from typing import Optional

from rich.console import Console
from rich.theme import Theme

ACCENT = "#0EA5E9"        # 主色：天空蓝
ACCENT_ALT = "#14B8A6"    # 次色：青绿
MUTED = "#64748B"         # 次要文字
OK = "#059669"            # 成功
WARN = "#B45309"          # 警告
ERROR = "#DC2626"         # 错误

THEME = Theme(
    {
        "app.title": "bold " + ACCENT,
        "app.accent": ACCENT,
        "app.accent2": ACCENT_ALT,
        "app.muted": MUTED,
        "app.label": MUTED,
        "app.value": "bold",
        "app.user": "bold " + ACCENT,
        "app.ok": "bold " + OK,
        "app.warn": WARN,
        "app.error": "bold " + ERROR,
        "app.tool": ACCENT_ALT,
        "app.prompt": "bold " + ACCENT,
        "app.rule": MUTED,
    }
)


def make_console(**kwargs) -> Console:
    """创建统一风格的控制台，宽度上限 110 列，避免超宽屏上排版松散。"""
    import shutil

    kwargs.setdefault("theme", THEME)
    kwargs.setdefault("highlight", False)
    kwargs.setdefault("soft_wrap", True)
    kwargs.setdefault("width", max(60, min(110, shutil.get_terminal_size((100, 24)).columns)))
    return Console(**kwargs)


_CONSOLE: Optional[Console] = None


def get_console() -> Console:
    """进程内共享的控制台，保证样式一致。"""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = make_console()
    return _CONSOLE


def content_width(console: Console, margin: int = 0) -> int:
    """返回可用的正文宽度，避免宽屏终端上出现超长的一条横线。"""
    return max(40, console.width - margin)
