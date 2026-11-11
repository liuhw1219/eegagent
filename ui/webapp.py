"""独立窗口界面：本地 HTTP 服务 + 单页前端，由 Chrome/Edge 以应用窗口方式打开。

窗口只展示"问题和结论"：模型推理、工具调用、耗时状态行都不下发到前端。
服务端仍然是同一个 EEGAgent 实例，进程内复用，避免每问一次都重新加载数据和 RAG。
"""
from __future__ import annotations

import glob
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional

from .info import LOADER_LABELS, build_data_info
from .render import EXAMPLES
from .transcript import Transcript

MODEL_CHOICES = ["qwen-plus", "qwen3-14b", "qwen3-32b", "qwen3-235b-a22b"]


# ---------------------------------------------------------------------- 工具


def list_edf_files(data_path: str = "") -> List[str]:
    """列出当前目录、dataPath 和 data/ 下的脑电文件。"""
    roots = [os.getcwd()]
    if data_path:
        roots.append(data_path if os.path.isabs(data_path) else os.path.join(os.getcwd(), data_path))
    roots.append(os.path.join(os.getcwd(), "data"))

    seen, files = set(), []
    for root in roots:
        for pattern in ("*.edf", "*.EDF"):
            for path in glob.glob(os.path.join(root, pattern)):
                real = os.path.normpath(os.path.abspath(path))
                if real in seen:
                    continue
                seen.add(real)
                files.append(real)
    return sorted(files)


def clean_answer(text: str) -> str:
    """兜底：万一模型在结论里漏出工具调用标记，只保留标记之前的自然语言。"""
    text = (text or "").strip()
    for marker in ("<FUNCTION>", "<ARGS>", "<RESULT>"):
        if marker in text:
            text = text.split(marker)[0].rstrip()
    return text or "（模型没有返回内容）"


# ---------------------------------------------------------------------- 服务


class ChatService:
    """持有唯一的 EEGAgent 实例，串行处理提问与切换数据。"""

    def __init__(
        self,
        agent_factory: Callable[..., object],
        file_name: str,
        loader: str,
        model: str,
        max_rounds: int,
        transcript_dir: str,
        data_path: str = "",
        warmup: Optional[Callable[[], None]] = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.file_name = file_name
        self.loader = loader
        self.model = model
        self.max_rounds = max_rounds
        self.transcript_dir = transcript_dir
        self.data_path = data_path
        self.warmup = warmup

        self.agent = None
        self.info: Dict[str, str] = {}
        self.status = "正在准备…"
        self.error: Optional[str] = None
        self.ready = False

        self._lock = threading.Lock()
        self._transcript: Optional[Transcript] = None
        self.on_shutdown: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------ 生命周期
    def bootstrap(self) -> None:
        threading.Thread(target=self._load_initial, daemon=True).start()

    def _load_initial(self) -> None:
        try:
            self._build(self.file_name, self.loader)
            if self.warmup is not None:
                # 把 RAG 向量模型的加载也算作"准备中"，避免第一个问题白等一分钟
                self.ready = False
                self.status = "正在加载知识库…"
                try:
                    self.warmup()
                except Exception as warm_exc:  # noqa: BLE001 - 预热失败不影响主流程
                    self.status = f"就绪（知识库预热失败：{type(warm_exc).__name__}）"
                    self.ready = True
                    return
                self.status = "就绪"
                self.ready = True
        except Exception as exc:  # noqa: BLE001 - 错误要显示在窗口里
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "加载失败"
            self.ready = False

    def _build(self, file_path: str, loader: str) -> None:
        self.ready = False
        self.status = f"正在加载 {os.path.basename(file_path)}…"
        agent = self.agent_factory(file_path=file_path, loader=loader, model=self.model)
        self.agent = agent
        resolved = getattr(agent, "file_path", "") or file_path
        self.file_name = os.path.abspath(resolved)
        self.loader = loader
        self.info = build_data_info(agent, self.model, loader)
        self.error = None
        self.status = "就绪"
        self.ready = True
        self._new_transcript()

    def _new_transcript(self) -> None:
        self._transcript = Transcript(
            self.transcript_dir,
            os.path.basename(self.file_name),
            {
                "文件": os.path.basename(self.file_name),
                "分析模式": LOADER_LABELS.get(self.loader, self.loader),
                "模型": self.model,
            },
        )
        self._transcript.flush()

    # ------------------------------------------------------------ 状态查询
    def snapshot(self) -> Dict:
        files = list_edf_files(self.data_path)
        names = [{"path": path, "name": os.path.basename(path)} for path in files]
        return {
            "ready": self.ready,
            "status": self.status,
            "error": self.error,
            "info": self.info if self.ready else {},
            "files": names,
            "current": {"path": self.file_name, "loader": self.loader},
            "loaderOptions": [{"value": key, "label": label} for key, label in LOADER_LABELS.items()],
            "model": self.model,
            "models": MODEL_CHOICES,
            "examples": [list(item) for item in EXAMPLES],
        }

    # ------------------------------------------------------------ 操作
    def switch(self, file_path: str, loader: str) -> Dict:
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
        if not os.path.exists(file_path):
            return {"ok": False, "error": f"文件不存在：{file_path}"}
        if not self._lock.acquire(blocking=False):
            return {"ok": False, "error": "当前有分析正在进行，请稍后再切换。"}
        try:
            self._build(file_path, loader)
            return {"ok": True, "state": self.snapshot()}
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "加载失败"
            self.ready = False
            return {"ok": False, "error": self.error}
        finally:
            self._lock.release()

    def set_model(self, model: str) -> str:
        self.model = model
        if self.agent is not None:
            self.agent.model = model
        if self.info:
            self.info["model"] = model
        return model

    def new_chat(self) -> None:
        if self.agent is not None and hasattr(self.agent, "reset_conversation"):
            self.agent.reset_conversation()
        if self._transcript is not None:
            self._transcript.add_note("清空上下文，开始新对话。")
            self._transcript.flush()

    def request_shutdown(self) -> None:
        if self.on_shutdown is not None:
            threading.Thread(target=self.on_shutdown, daemon=True).start()

    # ------------------------------------------------------------ 提问
    def ask_stream(self, question: str):
        """把 agent 跑在后台线程里，这里只产出"进度"和"最终结论"。"""
        if not self.ready or self.agent is None:
            yield {"type": "error", "message": self.error or "数据尚未加载完成，请稍候。"}
            return
        if not self._lock.acquire(blocking=False):
            yield {"type": "error", "message": "上一条分析还没结束，请稍等。"}
            return

        try:
            events: "queue.Queue[Dict]" = queue.Queue()

            def on_event(event: Dict) -> None:
                events.put(event)

            def worker() -> None:
                try:
                    result = self.agent.run(
                        question,
                        max_rounds=self.max_rounds,
                        on_event=on_event,
                        stream=True,
                    )
                    events.put({"type": "__done__", "result": result})
                except Exception as exc:  # noqa: BLE001
                    events.put({"type": "__error__", "message": f"{type(exc).__name__}: {exc}"})

            if self._transcript is not None:
                self._transcript.add_user(question)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            started = time.time()
            while True:
                try:
                    event = events.get(timeout=1.0)
                except queue.Empty:
                    yield {"type": "progress", "elapsed": round(time.time() - started, 1)}
                    continue

                kind = event.get("type")
                if kind == "__done__":
                    result = event["result"]
                    content = clean_answer(result.get("response", ""))
                    if self._transcript is not None:
                        self._transcript.add_assistant(content)
                        self._transcript.add_metrics(result)
                        self._transcript.flush()
                    yield {
                        "type": "answer",
                        "content": content,
                        "metrics": {
                            "rounds": result.get("rounds"),
                            "model_time": result.get("model_time"),
                            "total_time": result.get("total_time"),
                            "stopped_by_max_rounds": result.get("stopped_by_max_rounds"),
                        },
                    }
                    return

                if kind == "__error__":
                    if self._transcript is not None:
                        self._transcript.add_note(f"运行失败：{event['message']}")
                        self._transcript.flush()
                    yield {"type": "error", "message": event["message"]}
                    return

                if kind == "tool_end" and self._transcript is not None:
                    # 过程只留在会话记录里，不下发到窗口
                    self._transcript.add_tool(
                        event.get("name", "?"),
                        event.get("args", {}),
                        event.get("result"),
                        bool(event.get("ok", True)),
                        event.get("seconds"),
                    )
                # 其他事件（rag / round / tool_start）一律忽略
        finally:
            self._lock.release()


# ---------------------------------------------------------------------- HTTP


class _Handler(BaseHTTPRequestHandler):
    service: ChatService = None  # type: ignore[assignment]
    html_path: str = ""

    protocol_version = "HTTP/1.0"

    def log_message(self, fmt: str, *args) -> None:  # 静默：窗口版不刷控制台
        pass

    # ------------------------------------------------------------ 响应
    def _json(self, obj: Dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # ------------------------------------------------------------ 路由
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                with open(self.html_path, "r", encoding="utf-8") as handle:
                    html = handle.read()
            except OSError as exc:
                self._json({"ok": False, "error": str(exc)}, 500)
                return
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/state":
            self._json(self.service.snapshot())
            return

        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/ask":
            self._handle_ask()
        elif path == "/api/switch":
            data = self._read_json()
            result = self.service.switch(str(data.get("file") or ""), str(data.get("loader") or "sleep"))
            self._json(result)
        elif path == "/api/model":
            data = self._read_json()
            model = str(data.get("model") or "").strip()
            if not model:
                self._json({"ok": False, "error": "模型名为空"}, 400)
                return
            self.service.set_model(model)
            self._json({"ok": True, "state": self.service.snapshot()})
        elif path == "/api/newchat":
            self.service.new_chat()
            self._json({"ok": True})
        elif path == "/api/shutdown":
            self._json({"ok": True})
            self.service.request_shutdown()
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    # ------------------------------------------------------------ 流式回答
    def _handle_ask(self) -> None:
        data = self._read_json()
        question = str(data.get("question") or "").strip()
        if not question:
            self._json({"ok": False, "error": "问题为空"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            for event in self.service.ask_stream(question):
                payload = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b'data: {"type":"done"}\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
        finally:
            self.close_connection = True


# ---------------------------------------------------------------------- 启动


_BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


def _open_window(url: str) -> (Optional[subprocess.Popen], str):
    """优先用浏览器的应用模式开一个独立窗口；否则退回默认浏览器。"""
    for exe in _BROWSERS:
        if not os.path.exists(exe):
            continue
        profile = tempfile.mkdtemp(prefix="eegagent-window-")
        args = [
            exe,
            f"--app={url}",
            f"--user-data-dir={profile}",
            "--window-size=1240,900",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-features=Translate,MediaRouter",
        ]
        try:
            return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), profile
        except OSError:
            shutil.rmtree(profile, ignore_errors=True)
            continue
    webbrowser.open(url)
    return None, ""


def launch(
    agent_factory: Optional[Callable[..., object]],
    file_name: str,
    loader: str = "sleep",
    model: str = "qwen-plus",
    max_rounds: int = 8,
    transcript_dir: str = "runs/ui_sessions",
    data_path: str = "",
    port: int = 0,
    open_window: bool = True,
    demo: bool = False,
    warmup: Optional[Callable[[], None]] = None,
) -> int:
    """启动本地服务并打开窗口，窗口关闭后返回。"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
    if not os.path.exists(html_path):
        print(f"找不到界面文件：{html_path}", file=sys.stderr)
        return 2

    service = ChatService(
        agent_factory=agent_factory or (lambda **_: None),
        file_name=file_name,
        loader=loader,
        model=model,
        max_rounds=max_rounds,
        transcript_dir=transcript_dir,
        data_path=data_path,
        warmup=warmup,
    )
    if demo:
        service.status = "预览模式"
        service.ready = False

    handler = type("EEGAgentHandler", (_Handler,), {"service": service, "html_path": html_path})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/" + ("?demo=1" if demo else "")

    service.on_shutdown = server.shutdown

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if not demo:
        service.bootstrap()

    print(f"EEGAgent 窗口：{url}")
    print("关闭窗口即退出；也可以在窗口里点“退出”。")

    proc, profile = _open_window(url) if open_window else (None, "")
    try:
        if proc is not None:
            proc.wait()
        else:
            while thread.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            server.server_close()
        except OSError:
            pass
        if profile:
            shutil.rmtree(profile, ignore_errors=True)
    return 0
