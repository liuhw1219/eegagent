"""会话记录：把每轮问答、工具调用和运行指标写成 Markdown，便于事后复核。"""
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional


class Transcript:
    def __init__(self, directory: str, data_name: str, meta: Optional[Dict[str, str]] = None):
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = os.path.splitext(os.path.basename(data_name))[0] or "session"
        self.path = os.path.join(directory, f"{stamp}_{stem}.md")
        self._lines: List[str] = []
        self._index = 0
        self._tools: List[str] = []
        self._meta = meta or {}
        self._write_header()

    # ------------------------------------------------------------------ 写入
    def _write_header(self) -> None:
        self._lines.append("# EEGAgent 会话记录\n")
        self._lines.append(f"- 开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for key, value in self._meta.items():
            self._lines.append(f"- {key}：{value}\n")
        self._lines.append("\n")

    def add_user(self, text: str) -> None:
        self._index += 1
        self._tools = []
        self._lines.append(f"## 问题 {self._index}\n\n")
        self._lines.append("**你**：\n\n")
        self._lines.append(text.strip() + "\n\n")

    def add_tool(self, name: str, args: Dict, result, ok: bool, seconds: Optional[float]) -> None:
        from .render import fmt_kwargs, summarize_result

        status = "成功" if ok else "失败"
        time_text = f"，{seconds:.2f}s" if seconds is not None else ""
        entry = f"- `{name}({fmt_kwargs(args or {})})` → {status}{time_text}：{summarize_result(result, limit=300)}"
        self._tools.append(entry)

    def add_assistant(self, text: str) -> None:
        if self._tools:
            self._lines.append("**工具调用**：\n\n")
            self._lines.extend(entry + "\n" for entry in self._tools)
            self._lines.append("\n")
        self._lines.append("**EEGAgent**：\n\n")
        self._lines.append(text.strip() + "\n\n")

    def add_metrics(self, metrics: Dict) -> None:
        from .render import fmt_seconds

        parts = [
            f"{metrics.get('rounds', '-')} 轮",
            f"模型 {fmt_seconds(metrics.get('model_time'))}",
            f"工具 {fmt_seconds(metrics.get('local_tool_time'))}",
            f"总计 {fmt_seconds(metrics.get('total_time'))}",
        ]
        self._lines.append("**耗时**：" + " · ".join(parts) + "\n\n")

    def add_note(self, text: str) -> None:
        self._lines.append(f"> {text}\n\n")

    # ------------------------------------------------------------------ 落盘
    def flush(self) -> str:
        """原子写入，避免中途崩溃留下半截文件。"""
        content = "".join(self._lines)
        directory = os.path.dirname(self.path)
        fd, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return self.path
