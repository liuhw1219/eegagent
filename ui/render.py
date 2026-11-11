"""终端界面的全部渲染组件：横幅、信息卡、对话、工具卡片、指标、帮助。

设计原则：
1. 正文只用一个视觉焦点（边框/左侧竖线），不堆叠花哨的装饰；
2. 数字（耗时、通道、时长）统一格式化，方便横向比较；
3. 每一个"输入 → 思考 → 工具 → 结论"的环节都有明确的位置和颜色。
"""
from typing import Dict, List, Optional, Sequence, Tuple

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .theme import ACCENT, ACCENT_ALT, ERROR, MUTED, OK, WARN

PANEL_MAX_WIDTH = 100


# --------------------------------------------------------------------------- 格式化
def fmt_seconds(value: Optional[float]) -> str:
    """把秒数格式化为易读的耗时，例如 12.4s / 2m05.0s。"""
    if value is None:
        return "-"
    if value < 1:
        return f"{value:.2f}s"
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes)}m{seconds:04.1f}s"


def fmt_hms(seconds: Optional[float]) -> str:
    """把秒数格式化为 时:分:秒。"""
    if seconds is None:
        return "-"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def fmt_height(seconds: Optional[float]) -> str:
    """记录时长的人类可读写法，例如 22.08 小时 / 3.5 分钟。"""
    if seconds is None:
        return "-"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} 小时"
    if seconds >= 60:
        return f"{seconds / 60:.1f} 分钟"
    return f"{seconds:.1f} 秒"


def fmt_number(value) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-3:
            return f"{value:.2e}"
        return f"{value:.4g}"
    return str(value)


def fmt_kwargs(args: Dict) -> str:
    """把工具参数压成一行：start=300, end=360, name=[Fp1, Fp2]。"""
    if not args:
        return ""

    def one(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            items = [str(item) for item in value[:5]]
            body = ", ".join(items)
            if len(value) > 5:
                body += f", …共{len(value)}项"
            return f"[{body}]"
        if isinstance(value, dict):
            return "{…}"
        return fmt_number(value)

    parts = [f"{key}={one(value)}" for key, value in args.items()]
    text = ", ".join(parts)
    return text if len(text) <= 78 else text[:77] + "…"


def _join_limited(parts: List[str], limit: int) -> str:
    """按字段边界截断，避免把 "max_val=1.46e-04" 这样的数值切一半。"""
    kept: List[str] = []
    for part in parts:
        candidate = " · ".join(kept + [part])
        if len(candidate) > limit:
            if kept:
                return " · ".join(kept) + " · …"
            return part[: limit - 1] + "…"
        kept.append(part)
    return " · ".join(kept)


def summarize_result(result, limit: int = 150) -> str:
    """把工具返回值压成一行摘要，优先展示数值字段。"""
    if isinstance(result, dict):
        if "error" in result:
            return "调用失败：" + str(result.get("error"))[:limit]
        parts = []
        for key, value in result.items():
            if isinstance(value, dict):
                inner = list(value.items())[:3]
                detail = ", ".join(f"{name}={fmt_number(item)}" for name, item in inner)
                if len(value) > 3:
                    detail += "…"
                parts.append(f"{key}: {detail}")
            elif hasattr(value, "shape") and not isinstance(value, (str, bytes)):
                parts.append(f"{key}=<{value.shape}>")
            elif isinstance(value, (list, tuple)):
                if value and isinstance(value[0], (int, float)):
                    parts.append(f"{key}=[{fmt_number(value[0])} … {len(value)} 项]")
                else:
                    parts.append(f"{key}=[{len(value)} 项]")
            else:
                parts.append(f"{key}={fmt_number(value)}")
        text = _join_limited(parts, limit)
    elif hasattr(result, "shape") and not isinstance(result, (str, bytes)):
        text = f"<array {result.shape}>"
    elif isinstance(result, (list, tuple)):
        head = fmt_number(result[0]) if result else ""
        text = f"[{len(result)} 项] {head}"
    else:
        text = fmt_number(result)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _field(value, placeholder: str = "未提供") -> str:
    if value is None or value == "":
        return placeholder
    return str(value)


# --------------------------------------------------------------------------- 横幅与信息卡
def print_header(console: Console, subtitle: str = "大语言模型驱动的自动化脑电分析框架") -> None:
    """应用横幅：名称、定位、能力标签。"""
    title = Text()
    title.append("EEGAgent", style="app.title")
    title.append("   脑电分析智能体", style="app.muted")
    title.append("\n")
    title.append(subtitle, style="app.accent")
    title.append("\n")
    title.append("LLM 调度  ·  工具调用  ·  知识检索  ·  报告生成", style="app.muted")

    console.print(
        Panel(
            title,
            box=box.ROUNDED,
            border_style=ACCENT,
            padding=(0, 2),
            expand=True,
        )
    )


def print_data_card(console: Console, info: Dict[str, str]) -> None:
    """数据概况：一眼看清"这段数据是什么"。"""
    label_width = 8
    table = Table.grid(padding=(0, 2))
    table.add_column(style="app.label", no_wrap=True, min_width=label_width)
    table.add_column(style="app.value", overflow="fold")

    rows = [
        ("文件", _field(info.get("file"))),
        ("采样率", _field(info.get("fs"))),
        ("时长", _field(info.get("duration"))),
        ("导联", _field(info.get("channels"))),
        ("受试者", _field(info.get("subject"))),
        ("记录于", _field(info.get("recorded"))),
        ("分析模式", _field(info.get("loader"), "默认")),
        ("模型", _field(info.get("model"))),
        ("知识库", _field(info.get("knowledge"))),
    ]
    for label, value in rows:
        table.add_row(Text(label, style="app.label"), Text(value))

    console.print(
        Panel(
            table,
            title=Text("数据概况", style="app.accent"),
            title_align="left",
            border_style=MUTED,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        )
    )


def print_hint(console: Console) -> None:
    hint = Text()
    hint.append("  直接输入问题开始分析", style="app.muted")
    hint.append("    ·    ", style="app.muted")
    hint.append("/help", style="app.accent")
    hint.append(" 命令", style="app.muted")
    hint.append("    ·    ", style="app.muted")
    hint.append("/examples", style="app.accent")
    hint.append(" 示例问题", style="app.muted")
    console.print(hint)


# --------------------------------------------------------------------------- 对话
def print_user(console: Console, text: str) -> None:
    """用户输入：左侧竖线 + 中文标签，和助手输出形成对话感。"""
    body = Text()
    body.append("你", style="app.user")
    body.append("   ")
    body.append(text)
    console.print()
    console.print(Text("▍", style="app.accent"), end="")
    console.print(body, style=None)


def print_assistant(console: Console, markdown_text: str, title: str = "EEGAgent") -> None:
    """助手最终回答：以 Markdown 渲染，表格/列表都能正常显示。"""
    console.print(
        Panel(
            Markdown(markdown_text.strip(), code_theme="monokai"),
            title=Text(title, style="app.accent2"),
            title_align="left",
            border_style=ACCENT_ALT,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        )
    )


def print_tool(
    console: Console,
    name: str,
    args: Optional[Dict] = None,
    result=None,
    ok: bool = True,
    seconds: Optional[float] = None,
    note: str = "",
) -> None:
    """工具调用卡片：一行调用 + 一行结果，保证过程可追溯但不喧宾夺主。"""
    head = Text()
    head.append("    ▸ ", style="app.accent2")
    head.append(name, style="bold")
    args_text = fmt_kwargs(args or {})
    if args_text:
        head.append("  " + args_text, style="app.muted")
    if seconds is not None:
        head.append("   " + fmt_seconds(seconds), style="app.muted")
    head.append("  ✓" if ok else "  ✗", style="app.ok" if ok else "app.error")
    if note:
        head.append("  " + note, style="app.muted")
    console.print(head)

    if result is not None:
        body = Text("      ← ", style="app.muted")
        body.append(summarize_result(result), style="app.muted" if ok else "app.error")
        console.print(body)


def print_metrics(console: Console, metrics: Dict, label: str = "完成") -> None:
    """运行指标：轮次 / 模型耗时 / 工具耗时 / 总耗时。"""
    line = Text()
    line.append("  ✓ " + label, style="app.ok")
    line.append("    ", style="app.muted")
    items = [
        ("轮次", str(metrics.get("rounds", "-"))),
        ("模型", fmt_seconds(metrics.get("model_time"))),
        ("工具", fmt_seconds(metrics.get("local_tool_time"))),
        ("总计", fmt_seconds(metrics.get("total_time"))),
    ]
    for index, (key, value) in enumerate(items):
        if index:
            line.append("  ·  ", style="app.rule")
        line.append(f"{key} ", style="app.label")
        line.append(value, style="app.value")
    if metrics.get("stopped_by_max_rounds"):
        line.append("  ·  ", style="app.rule")
        line.append("达到最大轮次，回答可能不完整", style="app.warn")
    console.print(line)


# --------------------------------------------------------------------------- 提示与错误
def print_warning(console: Console, message: str) -> None:
    console.print(Text("  ! " + message, style="app.warn"))


def print_error(
    console: Console,
    message: str,
    title: str = "出错了",
    hint: Optional[str] = None,
) -> None:
    body = Text(message)
    if hint:
        body.append("\n")
        body.append(hint, style="app.muted")
    console.print(
        Panel(
            body,
            title=Text(title, style="app.error"),
            title_align="left",
            border_style=ERROR,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        )
    )


def print_help(
    console: Console,
    commands: Sequence[Tuple[str, str]],
    shortcuts: Sequence[Tuple[str, str]] = (),
) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2), expand=True)
    table.add_column("命令", style="app.accent", no_wrap=True, ratio=1)
    table.add_column("作用", ratio=2, overflow="fold")
    for command, description in commands:
        table.add_row(command, description)

    console.print(
        Panel(
            table,
            title=Text("可用命令", style="app.accent"),
            title_align="left",
            border_style=MUTED,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        )
    )
    if shortcuts:
        line = Text("  ")
        for index, (key, description) in enumerate(shortcuts):
            if index:
                line.append("    ", style="app.muted")
            line.append(key, style="app.accent")
            line.append(" " + description, style="app.muted")
        console.print(line)


EXAMPLES: List[Tuple[str, str]] = [
    ("这段数据的采样率、导联和记录时长是多少？", "基础信息（baseInfo）"),
    ("第 5 到第 10 分钟有没有明显的慢波？", "分段探查（slowSeizBckg + compute_amplitude）"),
    ("FP1-F7 和 F7-T3 在第 300 到 360 秒有没有癫痫样活动？", "事件检测（seizNormal / slowSeizBckg）"),
    ("Fpz-Cz 在第 100 秒附近是眼动还是肌电伪迹？", "伪迹判别（eyeMuscle）"),
    ("分析第 1 小时里的睡眠分期，并给出各期占比。", "睡眠分期（sleepStage）"),
    ("根据这份数据生成一份完整的 EEG 报告。", "报告生成（/report）"),
]


def print_examples(console: Console) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2), expand=True)
    table.add_column("示例问题", ratio=3, overflow="fold")
    table.add_column("会用到", ratio=2, style="app.muted", overflow="fold")
    for question, capability in EXAMPLES:
        table.add_row(question, capability)
    console.print(
        Panel(
            table,
            title=Text("示例问题", style="app.accent"),
            title_align="left",
            border_style=MUTED,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        )
    )


def print_bye(console: Console, transcript_path: Optional[str] = None) -> None:
    line = Text()
    line.append("  会话结束", style="app.accent")
    if transcript_path:
        line.append("  ·  记录已保存到 ", style="app.muted")
        line.append(transcript_path, style="app.value")
    console.print()
    console.print(line)


# --------------------------------------------------------------------------- 设计预览
def demo(console: Console, info: Optional[Dict[str, str]] = None) -> None:
    """渲染一屏示例，用于在没有 API Key 的情况下检查界面排版。"""
    sample_info = info or {
        "file": "test.edf（46.1 MB）",
        "fs": "100 Hz（配置 256 Hz，已按文件校正）",
        "duration": "22.08 小时（79,500 s）",
        "channels": "7 个 · Fpz-Cz、Pz-Oz、EOG horizontal、Resp oro-nasal、EMG submental 等 7 个",
        "subject": "女（F）· 年龄未提供 · 编号 X",
        "recorded": "1989-04-24 16:13",
        "loader": "睡眠导联（Fpz-Cz / Pz-Oz）",
        "model": "qwen-plus",
        "knowledge": "3 篇文档 · 相似度阈值 0.6",
    }
    print_header(console)
    print_data_card(console, sample_info)
    print_hint(console)
    print_user(console, "第 5 到第 10 分钟有没有明显的慢波？")
    print_tool(
        console,
        "slowSeizBckg",
        {"start": 300, "end": 360},
        {"slow_wave": 0.91, "seizure": 0.03, "background": 0.06},
        ok=True,
        seconds=0.6,
    )
    print_tool(
        console,
        "compute_amplitude",
        {"name": ["Fpz-Cz", "Pz-Oz"], "start": 300, "end": 360},
        {"Fpz-Cz": {"mean_abs": 1.22e-05, "rms": 1.71e-05, "peak": 8.03e-05}},
        ok=True,
        seconds=0.4,
    )
    print_tool(
        console,
        "sleepStage",
        {"start": 600, "end": 610},
        {"error": "segment shorter than 30 seconds"},
        ok=False,
        seconds=0.0,
    )
    print_assistant(
        console,
        "第 5–10 分钟（300–360 s）以 **弥漫性慢波** 为主，慢波概率 0.91，未见癫痫样活动。\n\n"
        "| 通道 | 平均绝对幅值 | RMS | 峰值 |\n"
        "| --- | --- | --- | --- |\n"
        "| Fpz-Cz | 1.22e-05 | 1.71e-05 | 8.03e-05 |\n"
        "| Pz-Oz  | 1.10e-05 | 1.58e-05 | 7.44e-05 |\n\n"
        "结论：该时段背景活动偏慢，建议结合年龄（20–60 岁）与临床背景进一步判断。",
    )
    print_metrics(
        console,
        {"rounds": 3, "model_time": 8.4, "local_tool_time": 1.0, "total_time": 9.7},
    )
    print_error(
        console,
        "无法读取 EDF 文件头：文件被其他程序占用。",
        hint="请关闭正在写入该文件的采集软件后重试，或用 /file 切换到其他文件。",
    )
    print_bye(console, "runs/ui_sessions/20261109_150000_test.edf.md")
