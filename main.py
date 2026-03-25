"""
教会同声翻译系统 — 粤语 → 普通话
OpenAI STT + 文本翻译 + WebSocket 字幕
"""

import asyncio
import base64
import csv
from collections import deque
from pathlib import Path
import io
import json
import os
import queue
import re
import socket
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager

import numpy as np
import qrcode
import requests
import sounddevice as sd
from fastapi import Body, FastAPI, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import edge_tts
import uvicorn

# ─── 配置 ──────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
BLOCK_MS = 100
SILENCE_RMS = float(os.getenv("SILENCE_RMS", "0.008"))
SILENCE_GAP_SEC = float(os.getenv("SILENCE_GAP_SEC", "1.0"))
MAX_SEGMENT_SEC = int(os.getenv("MAX_SEGMENT_SEC", "14"))
MIN_SEGMENT_SEC = float(os.getenv("MIN_SEGMENT_SEC", "1.0"))

OPENAI_STT_LANGUAGE = os.getenv("OPENAI_STT_LANGUAGE", "yue")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
OPENAI_STT_PROMPT = os.getenv(
    "OPENAI_STT_PROMPT",
    "粤语教会讲道。"
    "摩西、以赛亚、耶利米、大卫、歌利亚、保罗、彼得、马利亚、亚伯拉罕、约书亚、以利亚、但以理、约拿、"
    "哥林多前书、出埃及记、以赛亚书、逾越节、五旬节、撒拉弗、非利士人、"
    "上帝、耶稣、耶和华、圣灵、天鹅、丑小鸭、拐杖、"
    "奇异恩典、崇基学院、威尔斯亲王医院、港大、中大、中文大学",
)
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
TTS_ENABLE = os.getenv("TTS_ENABLE", "1").strip().lower() not in ("0", "false", "no")
TTS_LLM_SPLIT = os.getenv("TTS_LLM_SPLIT", "1").strip().lower() not in ("0", "false", "no")
TTS_PENDING_MAX = int(os.getenv("TTS_PENDING_MAX", "150"))
TTS_FLUSH_LEN = int(os.getenv("TTS_FLUSH_LEN", "30"))
TTS_MAX_LEN = int(os.getenv("TTS_MAX_LEN", "90"))
TTS_PUNCT_WAIT_SEC = float(os.getenv("TTS_PUNCT_WAIT_SEC", "1.2"))
TTS_WAIT_SEC = float(os.getenv("TTS_WAIT_SEC", "6.0"))

TRANSLATOR_TARGET = "zh-Hans"          # 普通话简体
TRANSLATOR_BACKEND = os.getenv("TRANSLATOR_BACKEND", "azure").strip().lower()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NORMALIZE_CSV = BASE_DIR / "normalize_map.csv"
NORMALIZE_ADMIN_TOKEN = os.getenv("NORMALIZE_ADMIN_TOKEN", "").strip()

# 全篇回顾修正
REVISION_ENABLE = os.getenv("REVISION_ENABLE", "0").strip().lower() not in ("0", "false", "no")
REVISION_DEBOUNCE_SEC = float(os.getenv("REVISION_DEBOUNCE_SEC", "2.0"))
REVISION_MAX_CHARS = int(os.getenv("REVISION_MAX_CHARS", "8000"))

# 每场字幕文本：SESSION_LOG_DISABLE=1 关闭；SESSION_LOG_DIR 指定目录（默认项目下 session_logs）
SESSION_LOG_DISABLE = os.getenv("SESSION_LOG_DISABLE", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# ─── 全局状态 ──────────────────────────────────────────────────────────────────

clients: set[WebSocket] = set()
event_loop: asyncio.AbstractEventLoop | None = None
audio_queue: queue.Queue = queue.Queue(maxsize=1000)
translator_endpoint: str = ""
translator_key: str = ""
translator_region: str = ""
openai_base_url: str = ""
openai_api_key: str = ""
local_openai_api_key: str = ""
openai_model: str = ""
openai_stt_base_url: str = ""
stream: sd.InputStream | None = None
recent_originals: deque[str] = deque(maxlen=10)
last_original_norm: str = ""
last_translated_norm: str = ""
last_emit_ts: float = 0.0
_normalize_lock = threading.Lock()
_normalize_pairs: list[tuple[str, str]] = []
_session_log_path: Path | None = None
_session_log_lock = threading.Lock()

# 全篇修正状态
all_sentences: list[dict] = []
_all_sentences_lock = threading.Lock()
_revision_event = threading.Event()
_revision_seq: int = 0
_revision_seq_lock = threading.Lock()
_revised_log_path: Path | None = None

# TTS LLM-driven split: pending buffer（上次翻译未完成的尾巴）
_tts_pending: str = ""
_tts_pending_lock = threading.Lock()


def _init_session_log() -> Path | None:
    """启动本场：新建带时间戳的 txt，整场逐条追加；Ctrl+C 结束时文件即完整。"""
    global _session_log_path
    if SESSION_LOG_DISABLE:
        _session_log_path = None
        return None
    raw = os.getenv("SESSION_LOG_DIR", "").strip()
    d = Path(raw) if raw else BASE_DIR / "session_logs"
    if not d.is_absolute():
        d = BASE_DIR / d
    d.mkdir(parents=True, exist_ok=True)
    name = time.strftime("session_%Y%m%d_%H%M%S.txt")
    p = d / name
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    p.write_text(
        f"# 教会同声翻译 · 场次记录\n# 开始: {started}\n"
        f"# 格式：每条两行为 粤/普（与屏幕推送一致，已过词条替换）\n\n",
        encoding="utf-8",
    )
    _session_log_path = p
    return p


def _append_session_log(original: str, translated: str) -> None:
    if not _session_log_path:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    block = f"[{stamp}] 粤: {original}\n[{stamp}] 普: {translated}\n\n"
    with _session_log_lock:
        with open(_session_log_path, "a", encoding="utf-8") as f:
            f.write(block)


def _finalize_session_log() -> None:
    """结束时写入收尾行（文件在拍摄过程中已实时写入）。"""
    global _session_log_path
    if not _session_log_path or not _session_log_path.exists():
        return
    ended = time.strftime("%Y-%m-%d %H:%M:%S")
    with _session_log_lock:
        with open(_session_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n# 结束: {ended}\n")
    print(f"\n📄 本场字幕已保存: {_session_log_path}")


def _normalize_csv_path() -> Path:
    raw = os.getenv("NORMALIZE_MAP_FILE", "").strip()
    p = Path(raw) if raw else DEFAULT_NORMALIZE_CSV
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def reload_normalize_map() -> int:
    """从 CSV 加载词条（Excel 另存为「CSV UTF-8」即可）。返回加载条数。"""
    global _normalize_pairs
    path = _normalize_csv_path()
    pairs: list[tuple[str, str]] = []
    if path.exists():
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        start = 0
        if rows and rows[0] and rows[0][0].strip().lower() in (
            "from",
            "source",
            "错词",
            "原文",
            "原词",
        ):
            start = 1
        for row in rows[start:]:
            if len(row) >= 2 and row[0].strip() and not row[0].strip().startswith("#"):
                pairs.append((row[0].strip(), row[1].strip()))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    with _normalize_lock:
        _normalize_pairs = pairs
    return len(pairs)


def apply_normalize_map(text: str) -> str:
    if not text:
        return text
    with _normalize_lock:
        pairs = list(_normalize_pairs)
    out = text
    for a, b in pairs:
        if a and b is not None:
            out = out.replace(a, b)
    return out


def _check_admin_token(token: str | None) -> bool:
    if not NORMALIZE_ADMIN_TOKEN:
        return True
    return token == NORMALIZE_ADMIN_TOKEN

# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def make_qr_base64(url: str) -> str:
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="white", back_color="#1a1a2e")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── WebSocket 广播 ────────────────────────────────────────────────────────────

async def broadcast(data: dict):
    if not clients:
        return
    msg = json.dumps(data, ensure_ascii=False)
    dead: set[WebSocket] = set()
    for ws in list(clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


# ─── TTS 服务器端合成 + 广播 ──────────────────────────────────────────────────

_tts_buf = ""
_tts_buf_sids: list[int] = []
_tts_buf_lock = threading.Lock()
_tts_timer: threading.Timer | None = None
_SENTENCE_END_RE = re.compile(r"[。？！…?!]$")
_SENTENCE_END_CHAR = set("。？！…?!")


def _tts_enqueue(text: str, sid: int):
    global _tts_buf, _tts_timer
    if not TTS_ENABLE:
        return

    flush_now = False
    with _tts_buf_lock:
        if _tts_buf:
            if _tts_buf[-1] in _SENTENCE_END_CHAR:
                _tts_buf += text
            else:
                _tts_buf += "，" + text
        else:
            _tts_buf = text
        _tts_buf_sids.append(sid)
        buf_len = len(_tts_buf)
        ends_natural = bool(_SENTENCE_END_RE.search(_tts_buf))

        if _tts_timer:
            _tts_timer.cancel()
            _tts_timer = None

        if buf_len >= TTS_MAX_LEN:
            flush_now = True
        elif ends_natural and buf_len >= TTS_FLUSH_LEN:
            flush_now = True
        elif ends_natural:
            _tts_timer = threading.Timer(TTS_PUNCT_WAIT_SEC, _tts_flush)
            _tts_timer.start()
        else:
            _tts_timer = threading.Timer(TTS_WAIT_SEC, _tts_flush)
            _tts_timer.start()

    if flush_now:
        _tts_flush()


def _tts_flush():
    global _tts_buf, _tts_timer
    with _tts_buf_lock:
        if _tts_timer:
            _tts_timer.cancel()
            _tts_timer = None
        if not _tts_buf:
            return
        text = _tts_buf
        sids = list(_tts_buf_sids)
        _tts_buf = ""
        _tts_buf_sids.clear()

    if len(text) > 250:
        text = text[:250]

    print(f"[TTS] 合成: {text[:40]}…（{len(text)}字, sids={sids}）")
    if event_loop:
        asyncio.run_coroutine_threadsafe(_tts_synthesize_and_broadcast(text, sids), event_loop)


async def _tts_synthesize_and_broadcast(text: str, sids: list[int]):
    try:
        audio_bytes = await _synthesize_tts_edge(text)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        n = len(clients)
        await broadcast({
            "type": "tts",
            "audio": audio_b64,
            "sids": sids,
        })
        print(f"[TTS] 广播完成: {len(audio_bytes)}bytes → {n} 客户端")
    except Exception as e:
        print(f"[TTS] 合成/广播失败: {e}")


def _tts_send_ready(text: str, sid: int):
    """LLM-split mode: send ready text directly to TTS without buffering."""
    if not TTS_ENABLE:
        return
    if len(text) > 250:
        text = text[:250]
    print(f"[TTS-ready] 合成: {text[:40]}…（{len(text)}字, sid={sid}）")
    if event_loop:
        asyncio.run_coroutine_threadsafe(
            _tts_synthesize_and_broadcast(text, [sid]), event_loop
        )


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _normalize_for_dedupe(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、,.!?；;:“”\"'（）()\-]", "", text)
    return text.strip().lower()


_stt_language_fallback: str = ""


def _transcribe_audio_openai(audio: np.ndarray) -> str:
    global _stt_language_fallback
    wav_bytes = _audio_to_wav_bytes(audio)
    url = f"{openai_stt_base_url.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {openai_api_key}"}

    lang = _stt_language_fallback or OPENAI_STT_LANGUAGE

    def _do_request(language: str) -> requests.Response:
        files = {"file": ("segment.wav", wav_bytes, "audio/wav")}
        data: dict[str, str] = {
            "model": OPENAI_STT_MODEL,
            "language": language,
        }
        if OPENAI_STT_PROMPT:
            data["prompt"] = OPENAI_STT_PROMPT
        return requests.post(url, headers=headers, files=files, data=data, timeout=12)

    resp = _do_request(lang)

    if resp.status_code == 401:
        raise RuntimeError("OpenAI STT 鉴权失败(401)：请检查 OPENAI_API_KEY 是否有效且未过期。")

    # If 'yue' is not supported, gracefully fall back to 'zh' with Cantonese prompt
    if resp.status_code == 400 and lang == "yue" and not _stt_language_fallback:
        print("[STT] language=yue 不被当前模型支持，回退到 zh（靠 prompt 引导粤语）")
        _stt_language_fallback = "zh"
        resp = _do_request("zh")

    resp.raise_for_status()
    result = resp.json()
    return (result.get("text") or "").strip()


async def _synthesize_tts_edge(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def _translate_text(text: str) -> str:
    if TRANSLATOR_BACKEND == "openai":
        return _translate_text_openai(text)
    return _translate_text_azure(text)


def _translate_text_azure(text: str) -> str:
    url = f"{translator_endpoint}/translate"
    headers = {
        "Ocp-Apim-Subscription-Key": translator_key,
        "Ocp-Apim-Subscription-Region": translator_region,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    body = [{"text": text}]
    # 优先按粤语翻译；若服务端不接受该源语言，再回退到自动检测。
    params_list = [
        {"api-version": "3.0", "from": "yue", "to": TRANSLATOR_TARGET},
        {"api-version": "3.0", "to": TRANSLATOR_TARGET},
    ]
    last_error: Exception | None = None
    for params in params_list:
        try:
            resp = requests.post(url, params=params, headers=headers, json=body, timeout=3)
            resp.raise_for_status()
            data = resp.json()
            return data[0]["translations"][0]["text"].strip()
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Translator 调用失败: {last_error}")


def _translate_text_openai(text: str) -> str:
    context_lines = list(recent_originals)[-5:]
    context_block = ""
    if context_lines:
        context_block = "前文粤语：\n" + "\n".join(context_lines) + "\n\n"

    with _tts_pending_lock:
        pending_snapshot = _tts_pending

    if TTS_ENABLE and TTS_LLM_SPLIT:
        system_prompt = (
            "你是教会讲道粤语转普通话译员。将粤语口语忠实翻译为通顺的简体普通话书面语。\n"
            "要求：保留原意，修正口语冗余使句子通顺，宗教术语使用标准译法。\n"
            "结尾不要加省略号、不要加多余句号；不要以 “。。。“ 这类引号+省略号/重复句号的形式收尾。\n\n"
            "输出格式：用 || 分割可朗读的完整句和未完成的尾部。\n"
            "- || 左边：语义完整的句子（以。？！结尾），可以直接朗读\n"
            "- || 右边：还没说完的部分，留待后续\n"
            "- 如果全部完整：完整句。||\n"
            "- 如果全部未完成：||未完成片段\n"
            "- 只输出一行，不要换行"
        )
        if pending_snapshot:
            user_content = f"{context_block}前文已翻译未完成：{pending_snapshot}\n当前粤语：{text}"
        else:
            user_content = f"{context_block}当前粤语：{text}"
    else:
        system_prompt = (
            "你是教会讲道粤语转普通话译员。将粤语口语忠实翻译为通顺的简体普通话书面语。\n"
            "要求：保留原意，修正口语冗余使句子通顺，宗教术语使用标准译法，只输出译文一行。\n"
            "结尾不要加省略号（…或...）、不要加多余句号；不要以 “。。。“ 这类引号+省略号/重复句号的形式收尾。如果原文未说完就直接在断句处结束即可。"
        )
        user_content = f"{context_block}当前粤语：{text}"

    url = f"{openai_base_url.rstrip('/')}/responses"
    llm_key = local_openai_api_key or openai_api_key
    headers = {
        "Authorization": f"Bearer {llm_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": openai_model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_output_tokens": 300,
        "stream": False,
    }
    def extract_text(resp_json: dict) -> str:
        if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
            return resp_json["output_text"].strip()
        output = resp_json.get("output", [])
        for item in output:
            for content in item.get("content", []):
                txt = content.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
        raise RuntimeError("OpenAI 响应中未找到可用翻译文本")

    def _parse_sse_text(raw: str) -> str:
        """从 SSE 流式响应中拼出完整文本。"""
        chunks: list[str] = []
        for line in raw.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                obj = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            delta = obj.get("delta", "")
            if delta:
                chunks.append(delta)
            txt = obj.get("text", "")
            if txt and obj.get("type", "").endswith(".done"):
                return txt.strip()
        if chunks:
            return "".join(chunks).strip()
        return ""

    def _call_llm(pl: dict, attempt: int = 1) -> str:
        """调 LLM 并处理空响应/SSE 流/非 JSON 重试。"""
        for i in range(attempt, attempt + 2):
            r = requests.post(url, headers=headers, json=pl, timeout=4)
            r.raise_for_status()
            r.encoding = "utf-8"
            body = r.text.strip()
            if not body:
                if i < attempt + 1:
                    time.sleep(0.1)
                    continue
                raise RuntimeError("LLM 返回空响应")
            # SSE 流式响应
            if "event:" in body[:200] and "data:" in body[:500]:
                text = _parse_sse_text(body)
                if text:
                    return text
                if i < attempt + 1:
                    time.sleep(0.1)
                    continue
                raise RuntimeError("SSE 响应中未找到有效文本")
            try:
                return extract_text(r.json())
            except (json.JSONDecodeError, RuntimeError):
                if body and not body.startswith(("<", "{")):
                    return body.strip()
                if i < attempt + 1:
                    time.sleep(0.1)
                    continue
                raise
        raise RuntimeError("LLM 重试后仍失败")

    return _call_llm(payload)


_DANGEROUS_TERMS = re.compile(
    r"东方闪电|全能神[^的]|呼喊派|实际神|女基督|常受主",
    re.IGNORECASE,
)


def _sanitize_dangerous(text: str) -> str | None:
    """Replace dangerous cult names; return None if unrepairable."""
    if not _DANGEROUS_TERMS.search(text):
        return text
    cleaned = text
    cleaned = re.sub(r"东方闪电", "摩西", cleaned)
    cleaned = re.sub(r"全能神(?!的)", "上帝", cleaned)
    cleaned = re.sub(r"呼喊派|实际神|女基督|常受主", "", cleaned)
    if _DANGEROUS_TERMS.search(cleaned):
        return None
    print(f"[安全] 已替换危险词: {text[:40]} → {cleaned[:40]}")
    return cleaned


_GIBBERISH_RE = re.compile(
    r"哈克斯|花的汽水|变质项|菜已[經经][Rr]eady"
)


def _looks_gibberish(text: str) -> bool:
    return bool(_GIBBERISH_RE.search(text))


def _strip_trailing_ellipsis(text: str) -> str:
    """LLM 末尾有时会输出省略号/重复句号或带引号的省略号组合。
    只在末尾清理，避免影响正文内部标点。
    """
    if not text:
        return text

    # 特例：如 `”。。。“`（引号+省略号/重复句号+引号）出现在句末
    text = re.sub(
        r"[”\"]\s*(?:\.{2,}|…+|。。+)\s*[“\"]\s*$",
        "",
        text,
    )
    # 普通尾部省略号/重复句号（不含引号）
    text = re.sub(r"(?:\s*(?:\.{2,}|…+|。。+)\s*)$", "", text)
    return text.strip()


# ─── 全篇回顾修正 ──────────────────────────────────────────────────────────────

def _init_revised_log() -> Path | None:
    global _revised_log_path
    if SESSION_LOG_DISABLE or not _session_log_path:
        _revised_log_path = None
        return None
    p = _session_log_path.with_name(
        _session_log_path.stem + "_revised.txt"
    )
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    p.write_text(
        f"# 教会同声翻译 · 全篇修正版\n# 开始: {started}\n\n",
        encoding="utf-8",
    )
    _revised_log_path = p
    return p


def _revise_full_text(sentences: list[dict]) -> list[str] | None:
    """Send all accumulated translations to LLM for full-text revision.
    Returns a list of revised lines (one per input sentence), or None on failure.
    """
    if not sentences:
        return None

    total_chars = sum(len(s["translated"]) for s in sentences)
    if total_chars > REVISION_MAX_CHARS:
        max_keep = len(sentences)
        running = 0
        for i in range(len(sentences) - 1, -1, -1):
            running += len(sentences[i]["translated"])
            if running > REVISION_MAX_CHARS:
                max_keep = len(sentences) - i - 1
                break
        sentences = sentences[-max_keep:]

    numbered = "\n".join(
        f"{i+1}. {s['translated']}" for i, s in enumerate(sentences)
    )

    system_prompt = (
        "你是教会讲道全文润色编辑。\n"
        "输入是逐句粤语 ASR → 粗译的简体普通话字幕，可能存在：\n"
        "- 残留粤语口语词（佢、咗、喺、嘅、唔、啲、咩、咁、嘢、嚟、畀等）\n"
        "- 前后重复的内容（ASR 重复输出）\n"
        "- 人名/术语前后不一致\n"
        "- ASR 同音误识别（粤语语音→错误汉字）\n"
        "- 断句不自然\n\n"
        "【要求】\n"
        "1. 把所有粤语口语词改写为标准简体普通话。\n"
        "2. 统一圣经人名和术语（使用和合本译名）：摩西、以赛亚、耶利米、大卫、"
        "保罗、彼得、马利亚、亚伯拉罕、约书亚、以利亚、但以理、约拿、"
        "歌利亚、撒母耳、所罗门、以斯帖、路得、以西结。\n"
        "3. 统一经卷名：哥林多前书、出埃及记、以赛亚书、罗马书、创世记、诗篇、箴言等。\n"
        "4. 去除明显重复的句子（保留内容更完整的那一句）。\n"
        "5. 修正 ASR 错误（陈大→上帝、天魔→天鹅等）。\n"
        "6. 保持忠实原意，不可扩写。保持简体中文。\n"
        "7. WhatsApp、Facebook 等品牌名保留原文。\n\n"
        "【输出格式 — 必须严格遵守】\n"
        "- 输出行数必须与输入行数完全一致。\n"
        "- 每行格式为：行号. 修正后的文本\n"
        "- 如果某行不需要修改，原样输出。\n"
        "- 如果某行是重复内容需要删除，输出：行号. [重复]\n"
        "- 不要输出任何额外说明。\n"
    )

    url = f"{openai_base_url.rstrip('/')}/responses"
    llm_key = local_openai_api_key or openai_api_key
    headers = {
        "Authorization": f"Bearer {llm_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": openai_model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"共 {len(sentences)} 行：\n{numbered}"},
        ],
        "temperature": 0.15,
        "stream": False,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        body = resp.text.strip()
        if not body:
            return None

        raw_text = ""
        if "event:" in body[:200] and "data:" in body[:500]:
            raw_text = _parse_sse_text_standalone(body)
        else:
            try:
                rj = resp.json()
                if isinstance(rj.get("output_text"), str) and rj["output_text"].strip():
                    raw_text = rj["output_text"].strip()
                else:
                    for item in rj.get("output", []):
                        for content in item.get("content", []):
                            txt = content.get("text")
                            if isinstance(txt, str) and txt.strip():
                                raw_text = txt.strip()
                                break
                        if raw_text:
                            break
            except (json.JSONDecodeError, RuntimeError):
                if body and not body.startswith(("<", "{")):
                    raw_text = body

        if not raw_text:
            return None

        lines = _parse_revision_lines(raw_text, len(sentences))
        return lines

    except Exception as e:
        print(f"[全篇修正] LLM 调用失败: {e}")
        return None


def _parse_sse_text_standalone(raw: str) -> str:
    """Extract text from SSE streaming response (standalone version for revision)."""
    chunks: list[str] = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        delta = obj.get("delta", "")
        if delta:
            chunks.append(delta)
        txt = obj.get("text", "")
        if txt and obj.get("type", "").endswith(".done"):
            return txt.strip()
    if chunks:
        return "".join(chunks).strip()
    return ""


def _parse_revision_lines(raw_text: str, expected_count: int) -> list[str]:
    """Parse numbered lines from LLM revision output.
    Falls back to splitting by newlines if numbered format isn't found.
    """
    lines: list[str] = []
    for raw_line in raw_text.strip().splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        m = re.match(r"^\d+\.\s*(.+)$", raw_line)
        if m:
            lines.append(m.group(1).strip())
        else:
            lines.append(raw_line)

    if len(lines) == expected_count:
        return lines

    if len(lines) > expected_count:
        return lines[:expected_count]

    while len(lines) < expected_count:
        lines.append("")

    return lines


def _write_revised_log(sentences: list[dict], revised_lines: list[str]) -> None:
    if not _revised_log_path:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    parts = [f"# 最近修正时间: {stamp}\n# 共 {len(revised_lines)} 句\n\n"]
    for i, line in enumerate(revised_lines):
        if line == "[重复]":
            continue
        orig = sentences[i]["original"] if i < len(sentences) else ""
        parts.append(f"[{i+1}] 粤: {orig}\n[{i+1}] 普: {line}\n\n")
    with _session_log_lock:
        _revised_log_path.write_text("".join(parts), encoding="utf-8")


def _trigger_revision():
    """Signal the revision worker that a new sentence is available."""
    global _revision_seq
    if not REVISION_ENABLE:
        return
    with _revision_seq_lock:
        _revision_seq += 1
    _revision_event.set()


def _revision_worker():
    """Background thread: waits for new sentences, debounces, then runs full-text revision."""
    while True:
        _revision_event.wait()
        _revision_event.clear()

        with _revision_seq_lock:
            seq_before = _revision_seq

        time.sleep(REVISION_DEBOUNCE_SEC)

        with _revision_seq_lock:
            seq_after = _revision_seq
        if seq_after != seq_before:
            continue

        with _all_sentences_lock:
            snapshot = list(all_sentences)

        if len(snapshot) < 2:
            continue

        print(f"[全篇修正] 开始修正 {len(snapshot)} 句 ...")
        t0 = time.time()
        revised = _revise_full_text(snapshot)
        elapsed = time.time() - t0

        if revised is None:
            print(f"[全篇修正] 修正失败（耗时 {elapsed:.1f}s）")
            continue

        print(f"[全篇修正] 完成（{len(revised)} 行，耗时 {elapsed:.1f}s）")

        _write_revised_log(snapshot, revised)

        if event_loop:
            event_loop.call_soon_threadsafe(
                lambda r=revised: asyncio.create_task(
                    broadcast({
                        "type": "revision",
                        "lines": r,
                        "ts": time.time(),
                    })
                )
            )


def _is_prompt_echo(text: str) -> bool:
    """Detect when STT hallucinates the prompt back instead of real speech."""
    if not OPENAI_STT_PROMPT:
        return False
    norm = re.sub(r"[\s，。、,.\-""\"'（）()！？!?；;:：]+", "", text)
    prompt_norm = re.sub(r"[\s，。、,.\-""\"'（）()！？!?；;:：]+", "", OPENAI_STT_PROMPT)
    if not norm or not prompt_norm:
        return False
    overlap = sum(1 for c in norm if c in prompt_norm)
    ratio = overlap / len(norm)
    return ratio > 0.6 and len(norm) > 20


def _handle_sentence(audio: np.ndarray):
    global last_original_norm, last_translated_norm, last_emit_ts, _tts_pending

    try:
        original = _transcribe_audio_openai(audio)
    except Exception as e:
        print(f"[识别失败] {e}")
        return
    if not original:
        return

    if _is_prompt_echo(original):
        print(f"[拦截] STT 回吐 prompt，跳过: {original[:40]}…")
        return

    original = apply_normalize_map(original)

    try:
        translated = _translate_text(original)
    except Exception as e:
        print(f"[翻译失败] {e}")
        print(f"  原文: {original[:60]}")
        return

    translated = apply_normalize_map(translated)
    translated = _strip_trailing_ellipsis(translated)

    # ── LLM-driven TTS split: 解析 || 分隔符 ──
    ready_text = ""
    pending_text = ""
    if TTS_ENABLE and TTS_LLM_SPLIT and "||" in translated:
        parts = translated.split("||", 1)
        ready_text = parts[0].strip()
        pending_text = parts[1].strip() if len(parts) > 1 else ""
        ready_text = _strip_trailing_ellipsis(ready_text)
        pending_text = _strip_trailing_ellipsis(pending_text)
        display_text = (ready_text + pending_text) if ready_text or pending_text else translated
        print(f"[TTS-split] ready={len(ready_text)}字 pending={len(pending_text)}字")
    elif TTS_ENABLE and TTS_LLM_SPLIT:
        display_text = translated
        ready_text = translated
        pending_text = ""
    else:
        display_text = translated
        ready_text = translated
        pending_text = ""

    # 更新 pending buffer + 安全阀：pending 过长时强制当作 ready 发送
    if TTS_ENABLE and TTS_LLM_SPLIT:
        with _tts_pending_lock:
            if pending_text and len(pending_text) > TTS_PENDING_MAX:
                print(f"[TTS-split] pending 超限({len(pending_text)}>{TTS_PENDING_MAX})，强制发送")
                ready_text = (ready_text + pending_text) if ready_text else pending_text
                pending_text = ""
            _tts_pending = pending_text

    sanitized = _sanitize_dangerous(display_text)
    if sanitized is None:
        print(f"[拦截] 含不可修复危险词，不推送: {display_text[:50]}")
        if TTS_ENABLE and TTS_LLM_SPLIT:
            with _tts_pending_lock:
                _tts_pending = ""
        return
    display_text = sanitized

    if _looks_gibberish(display_text):
        print(f"[拦截] 疑似乱码，不推送: {display_text[:50]}")
        if TTS_ENABLE and TTS_LLM_SPLIT:
            with _tts_pending_lock:
                _tts_pending = ""
        return

    print(f"[粤] {original}")
    print(f"[普] {display_text}\n")
    recent_originals.append(original)

    now = time.time()
    original_norm = _normalize_for_dedupe(original)
    translated_norm = _normalize_for_dedupe(display_text)
    is_dup = (
        original_norm
        and translated_norm
        and now - last_emit_ts < 8
        and (
            original_norm == last_original_norm
            or translated_norm == last_translated_norm
        )
    )
    if is_dup:
        print("[去重] 跳过重复字幕")
        return
    last_original_norm = original_norm
    last_translated_norm = translated_norm
    last_emit_ts = now

    _append_session_log(original, display_text)

    with _all_sentences_lock:
        sid = len(all_sentences)
        all_sentences.append({"id": sid, "original": original, "translated": display_text})

    if event_loop:
        _sid = sid
        event_loop.call_soon_threadsafe(
            lambda: asyncio.create_task(
                broadcast({
                    "type": "subtitle",
                    "sid": _sid,
                    "original": original,
                    "translated": display_text,
                    "ts": time.time(),
                })
            )
        )

    if TTS_ENABLE and TTS_LLM_SPLIT:
        if ready_text:
            _tts_send_ready(ready_text, sid)
    else:
        _tts_enqueue(display_text, sid)
    _trigger_revision()


def _audio_callback(indata, _frames, _time_info, _status):
    if not audio_queue.full():
        audio_queue.put_nowait(indata[:, 0].astype(np.float32).copy())


def _audio_worker():
    block_duration = BLOCK_MS / 1000
    silence_needed = int(SILENCE_GAP_SEC / block_duration)
    max_frames = int(MAX_SEGMENT_SEC / block_duration)
    min_frames = int(MIN_SEGMENT_SEC / block_duration)

    buf: list[np.ndarray] = []
    speech_frames = 0
    silence_frames = 0

    while True:
        try:
            chunk = audio_queue.get(timeout=0.3)
        except queue.Empty:
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        buf.append(chunk)

        if rms > SILENCE_RMS:
            speech_frames += 1
            silence_frames = 0
        else:
            silence_frames += 1

        sentence_end = silence_frames >= silence_needed and speech_frames >= min_frames
        too_long = len(buf) >= max_frames
        if sentence_end or too_long:
            if speech_frames >= min_frames:
                audio = np.concatenate(buf)
                threading.Thread(target=_handle_sentence, args=(audio,), daemon=True).start()
            buf = []
            speech_frames = 0
            silence_frames = 0


# ─── FastAPI 应用 ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global event_loop, translator_endpoint, translator_key, translator_region
    global openai_base_url, openai_api_key, local_openai_api_key
    global openai_model, openai_stt_base_url, stream

    print("─" * 50)
    print("初始化 OpenAI STT + 文本翻译")
    print(
        f"识别语言: {OPENAI_STT_LANGUAGE}  目标语言: {TRANSLATOR_TARGET}  后端: {TRANSLATOR_BACKEND}"
    )

    openai_api_key = _require_env("OPENAI_API_KEY")
    local_openai_api_key = os.getenv("LOCAL_OPENAI_API_KEY", "").strip()
    openai_stt_base_url = os.getenv("OPENAI_STT_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    if TRANSLATOR_BACKEND == "openai":
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        openai_model = _require_env("OPENAI_MODEL")
    else:
        translator_key = _require_env("AZURE_TRANSLATOR_KEY")
        translator_region = _require_env("AZURE_TRANSLATOR_REGION")
        translator_endpoint = _require_env("AZURE_TRANSLATOR_ENDPOINT").rstrip("/")

    n_terms = reload_normalize_map()
    print(f"✅ 词条表已加载: {_normalize_csv_path()}（{n_terms} 条）")
    if NORMALIZE_ADMIN_TOKEN:
        print("✅ 词条编辑页需 token：/admin/terms?token=…")
    else:
        print("⚠️ 词条编辑页未设 NORMALIZE_ADMIN_TOKEN，同 WiFi 人均可改")

    log_path = _init_session_log()
    if log_path:
        print(f"✅ 本场字幕将写入: {log_path}")
    elif SESSION_LOG_DISABLE:
        print("ℹ️ 场次字幕文件已关闭（SESSION_LOG_DISABLE）")

    if REVISION_ENABLE:
        rev_path = _init_revised_log()
        if rev_path:
            print(f"✅ 全篇修正版将写入: {rev_path}")

    event_loop = asyncio.get_running_loop()
    threading.Thread(target=_audio_worker, daemon=True).start()
    if REVISION_ENABLE:
        threading.Thread(target=_revision_worker, daemon=True).start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_MS / 1000),
        callback=_audio_callback,
    )
    stream.start()

    print(f"✅ OpenAI STT 已启动: {OPENAI_STT_MODEL}  language={OPENAI_STT_LANGUAGE}")
    if OPENAI_STT_PROMPT:
        print(f"   STT prompt: {OPENAI_STT_PROMPT[:60]}…")
    print(f"✅ 翻译后端已就绪: {TRANSLATOR_BACKEND}")
    print(f"✅ 上下文窗口: {recent_originals.maxlen} 句")
    print(f"✅ 句子切分静音阈值: {int(SILENCE_GAP_SEC * 1000)}ms")
    if REVISION_ENABLE:
        print(f"✅ 全篇回顾修正: 开启（防抖 {REVISION_DEBOUNCE_SEC}s，上限 {REVISION_MAX_CHARS} 字）")
    else:
        print("ℹ️ 全篇回顾修正: 关闭")
    if TTS_ENABLE and TTS_LLM_SPLIT:
        print(f"✅ TTS 语音广播: 开启（{EDGE_TTS_VOICE}，LLM智能断句，pending上限{TTS_PENDING_MAX}字）")
    elif TTS_ENABLE:
        print(f"✅ TTS 语音广播: 开启（{EDGE_TTS_VOICE}，传统buffer，句号flush≥{TTS_FLUSH_LEN}字，上限{TTS_MAX_LEN}字）")
    else:
        print("ℹ️ TTS 语音广播: 关闭")

    ip = get_local_ip()
    print(f"\n✅ 教会翻译系统已启动")
    print(f"   投影字幕：http://{ip}:8000")
    print(f"   会众手机：http://{ip}:8000（同一 WiFi 下扫码）")
    print("─" * 50 + "\n")

    yield

    _finalize_session_log()
    if stream:
        stream.stop()
        stream.close()


app = FastAPI(lifespan=lifespan, title="教会同声翻译")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/qr")
def qr_endpoint():
    ip = get_local_ip()
    url = f"http://{ip}:8000/?tts=1"
    return JSONResponse({"url": url, "qr": make_qr_base64(url)})


@app.get("/admin/terms", response_class=HTMLResponse)
def admin_terms_page(token: str | None = None):
    if not _check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)
    path = _normalize_csv_path()
    content = ""
    if path.exists():
        content = path.read_text(encoding="utf-8")
    q = f"?token={token}" if token else ""
    return HTMLResponse(
        f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"/>
<title>词条表</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;background:#0d0d1a;color:#f0f0ff}}
h1{{font-size:18px}}
p,li{{color:#8888aa;font-size:14px;line-height:1.5}}
textarea{{width:100%;min-height:320px;background:#161628;color:#f0f0ff;border:1px solid #2a2a45;border-radius:8px;padding:12px;font-size:14px;box-sizing:border-box}}
button{{margin-top:12px;padding:10px 18px;border-radius:8px;border:1px solid #7c6af7;background:#7c6af7;color:#fff;cursor:pointer;font-size:14px}}
code{{background:#161628;padding:2px 6px;border-radius:4px}}
</style></head><body>
<h1>纠错词条（CSV）</h1>
<p>每行两列，逗号分隔。首行可为表头：<code>from,to</code>。Excel 可「另存为 CSV UTF-8」后把内容粘贴到下方，或直接把 <code>{path.name}</code> 放到程序目录。</p>
<ul>
<li>保存后立即生效，无需重启。</li>
<li>建议设环境变量 <code>NORMALIZE_ADMIN_TOKEN</code>，访问须带 <code>?token=…</code>。</li>
</ul>
<form method="post" action="/admin/terms{q}">
  <textarea name="csv" required>{content}</textarea>
  <br/><button type="submit">保存</button>
</form>
</body></html>"""
    )


@app.post("/admin/terms", response_class=HTMLResponse)
def admin_terms_save(
    csv: str = Form(...),
    token: str | None = Query(None),
):
    if not _check_admin_token(token):
        return HTMLResponse("Unauthorized", status_code=401)
    path = _normalize_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv.replace("\r\n", "\n").strip() + "\n", encoding="utf-8")
    n = reload_normalize_map()
    q = f"?token={token}" if token else ""
    return HTMLResponse(
        f"已保存 {n} 条。<a href=\"/admin/terms{q}\">返回编辑</a>",
        status_code=200,
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    print(f"[连接] {ws.client}  在线: {len(clients)}")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)
        print(f"[断开] {ws.client}  在线: {len(clients)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
