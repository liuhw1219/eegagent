"""数据概况：终端界面和新窗口共用的一份信息整理逻辑。"""
import json
import os
from typing import Dict

LOADER_LABELS: Dict[str, str] = {
    "sleep": "睡眠导联（Fpz-Cz / Pz-Oz）",
    "mdd": "单极 19 导（MDD）",
    "bipolar": "双极 22 导（TUEV）",
}


def _fmt_height(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} 小时"
    if seconds >= 60:
        return f"{seconds / 60:.1f} 分钟"
    return f"{seconds:.1f} 秒"


def build_data_info(agent, model: str, loader: str) -> Dict[str, str]:
    """把 agent 上的文件信息和 baseInfo 拼成界面要展示的字段。"""
    record = getattr(agent, "record", {}) or {}
    info = getattr(agent, "info", {}) or {}

    file_path = getattr(agent, "file_path", "") or ""
    name = os.path.basename(file_path)
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        file_text = f"{name}（{size_mb:.1f} MB）"
    except OSError:
        file_text = name or "未选择"

    fs = record.get("fs")
    fs_text = f"{fs:g} Hz" if isinstance(fs, (int, float)) else "未识别"
    correction = getattr(agent, "fs_corrected", None)
    if correction:
        fs_text += f"（配置 {correction[0]:g} Hz，已按文件校正）"

    duration = record.get("duration_s")
    duration_text = (
        f"{_fmt_height(duration)}（{int(duration):,} s）"
        if isinstance(duration, (int, float))
        else info.get("data_duration", "未识别")
    )

    channels = record.get("ch_names") or []
    channel_names = [str(ch).replace("EEG ", "") for ch in channels]
    if channel_names:
        shown = "、".join(channel_names[:4])
        if len(channel_names) > 4:
            shown += " …"
        channel_text = f"{len(channel_names)} 个 · {shown}"
    else:
        channel_text = "未识别"

    subject_parts = []
    if info.get("sex"):
        subject_parts.append({"M": "男（M）", "F": "女（F）"}.get(info["sex"], str(info["sex"])))
    else:
        subject_parts.append("性别未提供")
    subject_parts.append(f"{info['age']} 岁" if info.get("age") else "年龄未提供")
    if info.get("patient_id"):
        subject_parts.append(f"编号 {info['patient_id']}")
    subject_text = " · ".join(subject_parts)

    recorded = " ".join(part for part in [info.get("start_date"), info.get("start_time")] if part)
    meas_date = record.get("meas_date")
    if hasattr(meas_date, "strftime"):
        recorded = meas_date.strftime("%Y-%m-%d %H:%M")

    return {
        "file": file_text,
        "fs": fs_text,
        "duration": duration_text,
        "channels": channel_text,
        "subject": subject_text,
        "recorded": recorded or "未提供",
        "loader": LOADER_LABELS.get(loader, loader),
        "model": model,
        "knowledge": knowledge_text(agent),
        "channel_count": len(channel_names),
    }


def knowledge_text(agent) -> str:
    try:
        path = os.path.join("RAG", "registered_files.json")
        with open(path, "r", encoding="utf-8") as handle:
            docs = json.load(handle)
        count = len(docs)
    except Exception:
        count = 0
    threshold = getattr(agent, "config", {}).get("similarity_threshold")
    threshold_text = f"相似度阈值 {threshold}" if threshold is not None else "未设置阈值"
    return f"{count} 篇文档 · {threshold_text}" if count else threshold_text
