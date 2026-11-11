"""交互式终端会话：输入界面、运行状态、工具卡片、输出界面和命令系统。

界面只依赖一个 agent 工厂函数（由 main.py 传入），因此可以复用同一套 UI
去驱动其他后端，不必和 EEGAgent 的实现耦合。
"""
from __future__ import annotations

import contextlib
import copy
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from . import render
from .info import LOADER_LABELS, build_data_info
from .theme import ACCENT, ACCENT_ALT, get_console
from .transcript import Transcript

COMMANDS: Sequence[Tuple[str, str]] = (
    ("/help", "查看命令说明"),
    ("/examples", "查看示例问题，可直接照着提问"),
    ("/info", "重新查看当前数据概况"),
    ("/file <路径或文件名>", "切换脑电文件（相对 dataPath 或绝对路径）"),
    ("/loader <sleep|mdd|bipolar>", "切换读取方式并重新加载当前文件"),
    ("/clear", "清空对话上下文，开始新一轮分析"),
    ("/report", "按报告模板生成一份完整的 EEG 报告"),
    ("/overview", "快速概览：一句话总结这段数据"),
    ("/model <名称>", "切换模型，例如 /model qwen-plus"),
    ("/save", "立即保存会话记录"),
    ("/exit", "保存并退出"),
)

SHORTCUTS: Sequence[Tuple[str, str]] = (
    ("Ctrl+C", "中断当前分析；在输入界面再次按下则退出"),
    ("↑ / ↓", "终端自带的输入历史"),
)

REPORT_PROMPT = "请根据这份脑电数据生成一份完整的 EEG 报告，按报告模板填写各字段，并给出结论。"
OVERVIEW_PROMPT = "请用一段话概括这段脑电数据的整体情况：基本信息、背景活动是否正常、有无明显异常。"


class StatusLine:
    """运行状态行：阶段 + 计时 + 已生成字数，随 Live 自动刷新。"""

    def __init__(self) -> None:
        self.phase = "准备"
        self.detail = ""
        self.round_index = 0
        self.round_total = 0
        self.chars = 0
        self.stage_start = time.time()
        self.spinner = Spinner("dots", style=ACCENT_ALT)

    def reset_stage(self) -> None:
        self.stage_start = time.time()

    def __rich_console__(self, console: Console, options):
        line = Text()
        if self.round_index and self.round_total:
            line.append(f"第 {self.round_index}/{self.round_total} 轮   ", style="app.muted")
        line.append(self.phase, style="app.accent")
        line.append(f"   {time.time() - self.stage_start:.1f}s", style="app.value")
        if self.chars:
            line.append(f"   · 已生成 {self.chars} 字", style="app.muted")
        if self.detail:
            line.append(f"   · {self.detail}", style="app.muted")

        grid = Table.grid(padding=(0, 1))
        grid.add_row(self.spinner, line)
        yield Padding(grid, (0, 0, 0, 2))


class InteractiveSession:
    def __init__(
        self,
        agent_factory: Callable[..., object],
        config_path: str,
        file_name: str,
        model: str,
        loader: str = "sleep",
        max_rounds: int = 8,
        transcript_dir: str = "runs/ui_sessions",
        console: Optional[Console] = None,
        data_path: str = "",
        stream: bool = True,
        show_tools: bool = False,
    ):
        self.console = console or get_console()
        self.agent_factory = agent_factory
        self.config_path = config_path
        self.file_name = file_name
        self.model = model
        self.loader = loader
        self.max_rounds = max_rounds
        self.transcript_dir = transcript_dir
        self.data_path = data_path
        self.stream = stream
        self.show_tools = show_tools

        self.agent = None
        self.transcript: Optional[Transcript] = None
        self._live: Optional[Live] = None
        self._status: Optional[StatusLine] = None
        self._last_metrics: Dict = {}
        self._errors = 0

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> int:
        self.console.print()
        render.print_header(self.console)
        try:
            self._load_agent(self.file_name, self.loader)
        except Exception as exc:  # 数据读不出来时给出可以照做的下一步
            render.print_error(
                self.console,
                f"加载数据失败：{self.file_name}",
                hint=f"{exc}\n请用 /file 指定一个可读取的 EDF 文件，或检查 config/config.json 里的 dataPath。",
            )
            return 1

        render.print_hint(self.console)
        self._loop()
        return 0

    def _loop(self) -> None:
        while True:
            try:
                raw = self.console.input("\n[app.prompt]你 ❯[/] ")
            except KeyboardInterrupt:
                render.print_bye(self.console, self._save())
                return
            except EOFError:
                render.print_bye(self.console, self._save())
                return

            text = raw.strip()
            if not text:
                continue
            if text in {"exit", "quit", "q", "/exit", "/quit"}:
                render.print_bye(self.console, self._save())
                return
            if text.startswith("/"):
                if self._run_command(text):
                    return
                continue
            self._ask(text)

    # ------------------------------------------------------------------ 命令
    def _run_command(self, text: str) -> bool:
        """返回 True 表示要退出会话。"""
        command, _, argument = text.partition(" ")
        command = command.lower()
        argument = argument.strip().strip('"').strip("'")

        if command == "/help":
            render.print_help(self.console, COMMANDS, SHORTCUTS)
        elif command == "/examples":
            render.print_examples(self.console)
        elif command == "/info":
            render.print_data_card(self.console, self._data_info())
        elif command == "/file":
            if not argument:
                render.print_warning(self.console, "用法：/file <文件名或绝对路径>")
            else:
                self._switch_file(argument)
        elif command == "/loader":
            if argument not in LOADER_LABELS:
                render.print_warning(self.console, "用法：/loader <sleep|mdd|bipolar>")
            else:
                self._switch_loader(argument)
        elif command == "/clear":
            self.agent.reset_conversation()
            if self.transcript:
                self.transcript.add_note("上下文已清空，后续提问不再参考上面的内容。")
                self.transcript.flush()
            render.print_warning(self.console, "对话上下文已清空，系统提示词与数据保持不变。")
        elif command == "/model":
            if argument:
                self.model = argument
                self.agent.model = argument
                render.print_warning(self.console, f"已切换模型：{argument}")
            else:
                render.print_warning(self.console, f"当前模型：{self.model}；用法：/model <名称>")
        elif command == "/report":
            self._ask(REPORT_PROMPT)
        elif command == "/overview":
            self._ask(OVERVIEW_PROMPT)
        elif command == "/save":
            path = self._save()
            if path:
                render.print_warning(self.console, f"会话记录已保存：{path}")
        elif command == "/exit" or command == "/quit":
            render.print_bye(self.console, self._save())
            return True
        else:
            render.print_warning(self.console, f"未知命令 {command}，输入 /help 查看可用命令。")
        return False

    def _switch_file(self, target: str) -> None:
        resolved = self._resolve_path(target)
        if not os.path.exists(resolved):
            render.print_error(
                self.console,
                f"找不到文件：{resolved}",
                hint="可以填写 data/ 下的文件名，也可以直接粘贴绝对路径。",
            )
            return
        try:
            self._load_agent(resolved, self.loader)
        except Exception as exc:
            render.print_error(self.console, "切换文件失败，已保留原来的数据。", hint=str(exc))

    def _switch_loader(self, loader: str) -> None:
        try:
            self._load_agent(self.file_name, loader)
        except Exception as exc:
            render.print_error(self.console, f"用 {loader} 方式加载失败，已保留原来的数据。", hint=str(exc))

    def _resolve_path(self, target: str) -> str:
        if os.path.isabs(target):
            return os.path.normpath(target)
        base = self.data_path or os.getcwd()
        return os.path.normpath(os.path.join(base, target))

    # ------------------------------------------------------------------ 加载
    def _load_agent(self, file_path: str, loader: str) -> None:
        name = os.path.basename(file_path)
        with self._status_block(f"加载 {name}"):
            agent = self.agent_factory(file_path=file_path, loader=loader, model=self.model)
        self.agent = agent
        self.file_name = file_path
        self.loader = loader

        if self.transcript:
            self._save()
        self.transcript = Transcript(
            self.transcript_dir,
            name,
            {
                "数据文件": name,
                "读取方式": LOADER_LABELS.get(loader, loader),
                "模型": self.model,
            },
        )
        render.print_data_card(self.console, self._data_info())
        correction = getattr(agent, "fs_corrected", None)
        if correction:
            render.print_warning(
                self.console,
                f"配置采样率 {correction[0]:g} Hz 与文件实际 {correction[1]:g} Hz 不一致，"
                f"已按文件校正，所有按秒索引的工具都按 {correction[1]:g} Hz 计算。",
            )

    def _data_info(self) -> Dict[str, str]:
        return build_data_info(self.agent, self.model, self.loader)

    # ------------------------------------------------------------------ 状态
    @contextlib.contextmanager
    def _status_block(self, phase: str):
        """阻塞操作（加载数据、检索模型）的转圈提示。"""
        self._start_status(phase)
        try:
            yield
        finally:
            self._stop_status()

    def _start_status(self, phase: str) -> None:
        self._stop_status()
        status = StatusLine()
        status.phase = phase
        self._status = status
        if not self.console.is_terminal:
            # 输出被重定向时不画转圈，只留一行阶段提示，保持内容干净可检索
            self.console.print(Text("  " + phase + " …", style="app.muted"))
            return
        self._live = Live(
            status,
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()

    def _stop_status(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                self._live = None
        self._status = None

    def _ensure_status(self, phase: Optional[str] = None) -> None:
        if self._status is None:
            self._start_status(phase or "模型推理")
        elif phase:
            self._status.phase = phase

    # ------------------------------------------------------------------ 问答
    def _ask(self, question: str) -> None:
        render.print_user(self.console, question)
        if self.transcript:
            self.transcript.add_user(question)

        saved_messages = copy.deepcopy(self.agent.messages)
        self._last_metrics = {}
        self._start_status("准备检索")
        started = time.time()

        try:
            result = self.agent.run(
                question,
                max_rounds=self.max_rounds,
                on_event=self._on_event,
                stream=self.stream,
                on_token=self._on_token,
            )
        except KeyboardInterrupt:
            self._stop_status()
            self.agent.messages = saved_messages
            render.print_warning(self.console, "已中断本次分析，对话上下文已回退到提问前。")
            return
        except Exception as exc:
            self._stop_status()
            self.agent.messages = saved_messages
            render.print_error(
                self.console,
                f"分析失败：{type(exc).__name__}: {exc}",
                hint="常见原因：API Key 无效或额度不足、网络不通、模型名不存在。可先用 /model 换成 qwen-plus 再试。",
            )
            if self.transcript:
                self.transcript.add_note(f"分析失败：{type(exc).__name__}: {exc}")
                self.transcript.flush()
            return
        finally:
            self._stop_status()

        self._last_metrics = {
            "rounds": result.get("rounds"),
            "model_time": result.get("model_time"),
            "local_tool_time": result.get("local_tool_time"),
            "total_time": result.get("total_time", time.time() - started),
            "stopped_by_max_rounds": result.get("stopped_by_max_rounds", False),
        }
        render.print_metrics(self.console, self._last_metrics)
        if self.transcript:
            self.transcript.add_metrics(self._last_metrics)
            self.transcript.flush()

    def _on_token(self, delta: str) -> None:
        if self._status is None:
            return
        self._status.chars += len(delta)
        if self._live is not None:
            self._live.refresh()

    def _on_event(self, event: Dict) -> None:
        kind = event.get("type")
        if kind == "rag_start":
            self._ensure_status("检索知识库")
        elif kind == "rag_end":
            if self._status:
                self._status.phase = "整理上下文"
        elif kind == "round_start":
            self._ensure_status("模型推理")
            if self._status:
                self._status.round_index = event.get("round", 0)
                self._status.round_total = event.get("max_rounds", 0)
                self._status.chars = 0
                self._status.detail = ""
            else:
                self._start_status("模型推理")
                if self._status:
                    self._status.round_index = event.get("round", 0)
                    self._status.round_total = event.get("max_rounds", 0)
        elif kind == "round_end":
            if self._status:
                self._status.phase = "生成完成"
        elif kind == "tool_start":
            # 工具执行期间保留状态行，长耗时工具（如整段统计）也能看到进度
            self._ensure_status("执行 " + event.get("name", "工具"))
        elif kind == "tool_end":
            self._stop_status()
            if self.show_tools:
                render.print_tool(
                    self.console,
                    event.get("name", "?"),
                    event.get("args", {}),
                    event.get("result"),
                    ok=event.get("ok", True),
                    seconds=event.get("seconds"),
                )
            if self.transcript:
                self.transcript.add_tool(
                    event.get("name", "?"),
                    event.get("args", {}),
                    event.get("result"),
                    event.get("ok", True),
                    event.get("seconds"),
                )
        elif kind == "final":
            self._stop_status()
            if event.get("truncated"):
                render.print_warning(self.console, "已达到最大轮次，下面是最后一轮的输出。")
            render.print_assistant(self.console, event.get("content", ""))
            if self.transcript:
                self.transcript.add_assistant(event.get("content", ""))
        elif kind == "warn":
            render.print_warning(self.console, str(event.get("message", "")))

    # ------------------------------------------------------------------ 记录
    def _save(self) -> Optional[str]:
        if not self.transcript:
            return None
        try:
            return self.transcript.flush()
        except Exception:
            return None
