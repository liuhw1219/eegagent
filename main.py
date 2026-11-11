"""EEGAgent 入口：独立窗口界面（默认）/ 交互式终端界面 / 单次问答。

用法：
    python main.py                     # 打开独立窗口（输入问题，只显示结论）
    python main.py --cli               # 在终端里交互（会显示工具调用等过程）
    python main.py --file xxx.edf      # 指定数据文件
    python main.py -q "有没有异常？"    # 只跑一个问题，适合批处理/脚本
    python main.py --ui-demo           # 只看终端界面样式（不需要 API Key）

API Key 解析顺序：--api-key > 环境变量 DASHSCOPE_API_KEY > config/secrets.json
"""
import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

import yaml
from openai import OpenAI

from prompt import getSystemPrompt
from tools import function_register
from tools.baseInfo import baseInfo, get_age_factor, probe_edf
from tools.dataLoad import dataLoad, load_MDD_edf, load_Sleep_edf
from tools.registerData import registerData
from utils.messageMerge import messageMerge
from utils.parseCalling import extract_tool_calls, has_config_parameter

DEFAULT_CONFIG = "config/config.json"
DEFAULT_FILE = "test.edf"
DEFAULT_MODEL = "qwen-plus"       # 也可用 qwen3-14b / qwen3-32b / qwen3-235b-a22b
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TIMEOUT = 120

DATA_LOADERS: Dict[str, Callable] = {
    "sleep": load_Sleep_edf,      # 睡眠导联：Fpz-Cz / Pz-Oz
    "mdd": load_MDD_edf,          # 单极 19 导
    "bipolar": dataLoad,          # 双极 22 导
}

# RAG 的向量模型和索引只加载一次，交互模式下每次提问不必重建
_RAG_CACHE: Dict[str, object] = {}


def get_rag_modules():
    _quiet_runtime()
    if not _RAG_CACHE:
        from RAG.embedder import BGEEmbedder
        from RAG.searcher import FaissSearcher

        _RAG_CACHE["embedder"] = BGEEmbedder()
        _RAG_CACHE["searcher"] = FaissSearcher("RAG/faiss.index", "RAG/chunks.pkl")
    return _RAG_CACHE["embedder"], _RAG_CACHE["searcher"]


def _quiet_runtime() -> None:
    """关掉 transformers 的权重加载进度条，避免打断界面上的状态行。"""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """按 参数 > 环境变量 > config/secrets.json 的顺序取 API Key。"""
    if explicit:
        return explicit
    env_key = os.getenv("DASHSCOPE_API_KEY")
    if env_key:
        return env_key
    secret_path = os.path.join("config", "secrets.json")
    if os.path.exists(secret_path):
        try:
            with open(secret_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            key = (data or {}).get("api_key")
            if key:
                return key
        except Exception:
            return None
    return None


class EEGAgent:
    def __init__(
        self,
        config_path: str,
        file_name: str,
        api_key: str,
        base_url: str,
        model: str = DEFAULT_MODEL,
        loader: str = "sleep",
        timeout: int = DEFAULT_TIMEOUT,
        stream: bool = False,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(os.path.join(self.config["report_template_path"], "report_template.yaml"), "r", encoding="utf-8") as file:
            report_template = yaml.safe_load(file)
        self.file_path = os.path.join(self.config["dataPath"], file_name)

        if loader not in DATA_LOADERS:
            raise ValueError(f"未知的读取方式 {loader}，可选：{', '.join(DATA_LOADERS)}")
        self.loader = loader
        from tools.polar import set_montage

        set_montage(loader)
        self.model = model
        self.timeout = timeout
        self.stream = stream

        ### read header first: 采样率以文件为准 ###
        self.record = probe_edf(self.file_path)
        configured_fs = self.config.get("fs")
        detected_fs = self.record.get("fs")
        self.fs_corrected: Optional[tuple] = None
        if detected_fs and configured_fs and abs(float(detected_fs) - float(configured_fs)) > 1e-6:
            # 工具里的所有时间索引都假设 config['fs'] 就是数据采样率，
            # 两者不一致会让"第几秒"整体偏移，这里按文件头校正。
            self.fs_corrected = (configured_fs, float(detected_fs))
            real_fs = float(detected_fs)
            # 采样率要保持 int（tools/registerData 直接用它做切片索引）
            self.config["fs"] = int(real_fs) if real_fs.is_integer() else real_fs

        ### Load Data... ###
        data = DATA_LOADERS[loader](self.file_path, self.config)
        self.data = data
        registerData(data)

        ### generate baseInfo ###
        self.info = baseInfo(self.file_path)
        self.prompt_info = dict(self.info)
        if isinstance(self.record.get("fs"), (int, float)):
            self.prompt_info["sampling_rate"] = f"{self.record['fs']:g} Hz"
        channels = self._analysis_channels(loader)
        if channels:
            self.prompt_info["analysis_channels"] = channels
            self.prompt_info["channel_note"] = "Tool channel names must come from analysis_channels."
        self.tool_schemas = function_register.export_tool_schemas()

        ### init ###
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        ### construct system prompt ###
        prior_knowledge = self.config["prior knowledge"].copy()
        if "age" in self.info.keys():
            age_factor = get_age_factor(self.info["age"], self.config["prior knowledge"]["Age factor"])
            prior_knowledge["Age factor"] = age_factor
        self.system_prompt = getSystemPrompt(self.prompt_info, self.tool_schemas, prior_knowledge, report_template)
        # 知识库检索结果每轮拼在 base 之后，不污染原始系统提示词
        self.base_system_prompt = self.system_prompt
        self.messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]

    @staticmethod
    def _analysis_channels(loader: str) -> List[str]:
        """当前读取方式真正交给工具分析的是哪些导联。"""
        from tools.polar import index2name, single_polar, sleep_polar

        if loader == "sleep":
            return list(sleep_polar)
        if loader == "mdd":
            return list(single_polar)
        return [index2name[index] for index in sorted(index2name)]

    # ------------------------------------------------------------------ 对话
    def prepare_user_message(self, user_query: str):
        self.messages.append({"role": "user", "content": user_query})

    def reset_conversation(self):
        """清空多轮上下文，保留数据、系统提示词和已加载的模型。"""
        self.messages = [{"role": "system", "content": self.base_system_prompt}]

    def call_model(self, model: Optional[str] = None, stream: Optional[bool] = None, on_token=None):
        model = model or self.model
        stream = self.stream if stream is None else stream

        if stream:
            completion = self.client.chat.completions.create(
                model=model,
                messages=self.messages,
                timeout=self.timeout,
                stream=True,
            )
            pieces: List[str] = []
            for chunk in completion:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = (getattr(delta, "content", None) or "") if delta else ""
                if not piece:
                    continue
                pieces.append(piece)
                if on_token is not None:
                    on_token(piece)
            response = "".join(pieces)
            if not response:
                # 少数兼容接口在流式下不返回正文，退回到非流式再取一次
                return self.call_model(model=model, stream=False, on_token=None)
        else:
            completion = self.client.chat.completions.create(
                model=model,
                messages=self.messages,
                timeout=self.timeout,
            )
            response = completion.choices[0].message.content or ""

        self.messages.append({"role": "assistant", "content": response})
        return response

    def handle_tool_calls(self, response, on_event=None):
        emit = on_event or (lambda event: None)
        calls = extract_tool_calls(
            response,
            on_error=(None if on_event else lambda exc: print(f"Parameter parsing failed: {exc}")),
        )
        if not calls:
            return False

        function_return = []
        for call in calls:
            function_name = call["name"].strip()
            args = call["args"].copy()
            function = function_register.get_function(function_name)

            if not function:
                emit({"type": "tool_end", "name": function_name, "args": args, "result": {"error": "no such tool"},
                      "ok": False, "seconds": 0.0})
                if on_event is None:
                    print(f"No function named {function_name}")
                continue

            ### automated import config parameters ###
            if has_config_parameter(function) and "config" not in args:
                args["config"] = self.config

            # 界面只展示模型传的参数，不把整份 config 打印出来
            display_args = {key: value for key, value in args.items() if key != "config"}

            emit({"type": "tool_start", "name": function_name, "args": display_args})
            started = time.time()
            try:
                output = function(**args)
                ok = True
            except Exception as e:
                output = {
                    "error": repr(e),
                    "instruction": "The tool call failed. Revise the arguments to satisfy the tool constraints before calling again.",
                }
                ok = False
                if on_event is None:
                    print(f"{function_name} execution unsuccessful: {e}")
            seconds = time.time() - started

            emit({"type": "tool_end", "name": function_name, "args": display_args, "result": output,
                  "ok": ok, "seconds": seconds})
            new_call = call.copy()
            new_call["return"] = output
            function_return.append(new_call)

        messageMerge(function_return, self.messages)
        return True

    def run(
        self,
        user_query: str,
        max_rounds: int = 8,
        on_event: Optional[Callable[[Dict], None]] = None,
        stream: Optional[bool] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ):
        emit = on_event or (lambda event: None)

        # RAG：检索到的知识拼在系统提示词后面（只覆盖本节，不累积）
        emit({"type": "rag_start"})
        embedder, searcher = get_rag_modules()
        query_vector = embedder.encode([user_query])[0]
        threshold = self.config.get("similarity_threshold") or 0.0
        hits = [text for text, score in searcher.search(query_vector, top_k=3) if score >= threshold]
        retrieval = "".join(f"{rank}. {text}\n" for rank, text in enumerate(hits, 1))
        self.messages[0]["content"] = self.base_system_prompt + retrieval
        emit({"type": "rag_end", "hits": len(hits)})

        self.prepare_user_message(user_query)

        total_rounds = 0
        total_time = 0.0
        local_tool_time = 0.0
        response = ""
        stopped_by_max_rounds = False

        start_total = time.time()
        while total_rounds < max_rounds:
            round_index = total_rounds + 1
            emit({"type": "round_start", "round": round_index, "max_rounds": max_rounds})
            round_start = time.time()
            response = self.call_model(stream=stream, on_token=on_token)
            round_end = time.time()
            total_rounds = round_index
            total_time += round_end - round_start
            emit({"type": "round_end", "round": round_index, "chars": len(response)})

            tool_start = time.time()
            has_more_tools = self.handle_tool_calls(response, on_event=on_event)
            local_tool_time += time.time() - tool_start

            if not has_more_tools:
                emit({"type": "final", "content": response, "truncated": False})
                break
        else:
            stopped_by_max_rounds = True
            emit({"type": "final", "content": response, "truncated": True})

        total_duration = time.time() - start_total
        metrics = {
            "response": response,
            "rounds": total_rounds,
            "model_time": total_time,
            "local_tool_time": local_tool_time,
            "total_time": total_duration,
            "stopped_by_max_rounds": stopped_by_max_rounds,
        }
        emit({"type": "metrics", **metrics})
        return metrics


# ---------------------------------------------------------------------- 运行入口
def _force_utf8_output() -> None:
    """Windows 终端默认可能是 GBK，这里统一成 UTF-8，避免界面字符报编码错误。"""
    if os.name == "nt":
        try:
            if sys.stdout.isatty():
                import ctypes

                ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def build_agent_factory(config_path: str, api_key: str, base_url: str, timeout: int):
    def factory(file_path: str, loader: str, model: str):
        return EEGAgent(
            config_path=config_path,
            file_name=file_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            loader=loader,
            timeout=timeout,
        )

    return factory


def run_once(agent: EEGAgent, question: str, max_rounds: int, show_tools: bool = False) -> int:
    """单次问答：只打印关键过程与结论，方便脚本调用。"""
    from rich.text import Text

    from ui import render
    from ui.theme import get_console

    console = get_console()
    console.print(Text("▍", style="app.accent"), end="")
    console.print(Text("你   " + question, style=""))

    def on_event(event: Dict) -> None:
        if event["type"] == "tool_end" and show_tools:
            render.print_tool(
                console,
                event.get("name", "?"),
                event.get("args", {}),
                event.get("result"),
                ok=event.get("ok", True),
                seconds=event.get("seconds"),
            )

    result = agent.run(question, max_rounds=max_rounds, on_event=on_event, stream=False)
    render.print_assistant(console, result.get("response", ""))
    render.print_metrics(console, result)
    return 0 if result.get("response") else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EEGAgent：大模型驱动的自动化脑电分析")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径")
    parser.add_argument("--file", default=DEFAULT_FILE, help="EDF 数据文件（相对 dataPath 或绝对路径）")
    parser.add_argument("--loader", default="sleep", choices=list(DATA_LOADERS), help="数据读取方式")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI 兼容接口地址")
    parser.add_argument("--api-key", default=None, help="API Key（也可用环境变量 DASHSCOPE_API_KEY）")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="单次请求超时（秒）")
    parser.add_argument("--max-rounds", type=int, default=8, help="工具调用最多轮次")
    parser.add_argument("-q", "--question", default=None, help="只跑一个问题，不进入交互界面")
    parser.add_argument("--ui-demo", action="store_true", help="仅预览界面样式，不调用模型")
    parser.add_argument("--no-stream", action="store_true", help="交互模式不使用流式输出")
    parser.add_argument("--cli", action="store_true", help="改用终端交互界面（默认打开独立窗口）")
    parser.add_argument("--no-window", action="store_true", help="只启动本地服务，不开浏览器窗口")
    parser.add_argument("--port", type=int, default=0, help="本地服务端口，默认自动选择")
    parser.add_argument("--show-tools", action="store_true", help="在终端里显示工具调用过程（窗口界面不受影响）")
    args = parser.parse_args(argv)

    _force_utf8_output()
    _quiet_runtime()

    if args.ui_demo:
        if not args.cli:
            from ui.webapp import launch

            return launch(
                agent_factory=None,
                file_name=args.file,
                loader=args.loader,
                model=args.model,
                port=args.port,
                open_window=not args.no_window,
                demo=True,
            )

        import shutil

        from ui import render
        from ui.theme import make_console

        console = make_console(width=min(110, shutil.get_terminal_size((100, 24)).columns))
        render.demo(console)
        return 0

    api_key = resolve_api_key(args.api_key)
    if not api_key or api_key in {"***"}:
        from ui import render
        from ui.theme import get_console

        render.print_error(
            get_console(),
            "没有找到可用的 API Key。",
            hint="三种设置方式任选其一：\n"
                 "1. 设置环境变量 DASHSCOPE_API_KEY\n"
                 "2. 在 config/secrets.json 里写入 {\"api_key\": \"sk-...\"}\n"
                 "3. 运行时加参数 --api-key sk-...",
        )
        return 2

    if args.question:
        agent = EEGAgent(
            config_path=args.config,
            file_name=args.file,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            loader=args.loader,
            timeout=args.timeout,
        )
        return run_once(agent, args.question, args.max_rounds, show_tools=args.show_tools)

    try:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        print(f"读取配置失败：{exc}")
        return 2

    factory = build_agent_factory(args.config, api_key, args.base_url, args.timeout)

    if not args.cli:
        from ui.webapp import launch

        return launch(
            agent_factory=factory,
            file_name=args.file,
            loader=args.loader,
            model=args.model,
            max_rounds=args.max_rounds,
            transcript_dir=os.path.join("runs", "ui_sessions"),
            data_path=config.get("dataPath", "") or "",
            port=args.port,
            open_window=not args.no_window,
            warmup=get_rag_modules,
        )

    from ui import run_interactive

    session = run_interactive(
        agent_factory=factory,
        config_path=args.config,
        file_name=args.file,
        model=args.model,
        loader=args.loader,
        max_rounds=args.max_rounds,
        transcript_dir=os.path.join("runs", "ui_sessions"),
        data_path=config.get("dataPath", "") or "",
        stream=not args.no_stream,
        show_tools=args.show_tools,
    )
    return session


if __name__ == "__main__":
    raise SystemExit(main())
