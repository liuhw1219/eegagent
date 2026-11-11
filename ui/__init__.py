"""终端界面层：横幅、信息卡、对话面板、工具卡片、交互会话。"""
from .session import InteractiveSession


def run_interactive(**kwargs) -> int:
    """启动交互式界面，返回进程退出码。"""
    session = InteractiveSession(**kwargs)
    return session.start()


__all__ = ["InteractiveSession", "run_interactive"]
