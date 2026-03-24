"""
教会同声翻译系统 — 粤语 → 普通话
M1 Mac 优化版：faster-whisper (CPU int8) + VAD 分句 + Google Translate + WebSocket 字幕
"""

import asyncio
import base64
import io
import json
import queue
import socket
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import qrcode
import sounddevice as sd
from faster_whisper import WhisperModel
from googletrans import Translator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ─── 配置 ──────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
BLOCK_MS = 100               # 音频回调间隔（毫秒）
SILENCE_RMS = 0.008          # 低于此值视为静音
SILENCE_GAP_SEC = 0.7        # 静音持续多久触发处理（秒）
MAX_SEGMENT_SEC = 10         # 超过此长度强制处理（秒）
MIN_SEGMENT_SEC = 0.4        # 低于此长度跳过（秒）

# 模型选择（M1 上 large-v3-turbo 延迟约 0.5s，medium 约 0.2s）
MODEL_SIZE = "large-v3-turbo"
LANGUAGE = "yue"             # 粤语

# ─── 全局状态 ──────────────────────────────────────────────────────────────────

clients: set[WebSocket] = set()
audio_queue: queue.Queue = queue.Queue(maxsize=1000)
event_loop: asyncio.AbstractEventLoop | None = None
whisper_model: WhisperModel | None = None
translator: Translator | None = None

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


# ─── 转录 + 翻译（在后台线程中执行）─────────────────────────────────────────

def transcribe_and_broadcast(audio: np.ndarray):
    global whisper_model, translator, event_loop

    # Whisper 转录粤语
    segments, _ = whisper_model.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 100},
    )
    original = "".join(seg.text for seg in segments).strip()

    if not original or len(original) < 2:
        return

    # Google Translate 粤语 → 普通话（简体）
    try:
        result = translator.translate(original, src="yue", dest="zh-cn")
        translated = result.text
    except Exception as e:
        print(f"[翻译失败] {e}")
        translated = original

    print(f"[粤] {original}")
    print(f"[普] {translated}\n")

    if event_loop:
        asyncio.run_coroutine_threadsafe(
            broadcast({"original": original, "translated": translated, "ts": time.time()}),
            event_loop,
        )


# ─── 音频捕获 + VAD 分句 ───────────────────────────────────────────────────────

def audio_callback(indata, frames, time_info, status):
    """sounddevice 回调，100ms 一个 chunk"""
    if not audio_queue.full():
        audio_queue.put_nowait(indata[:, 0].astype(np.float32).copy())


def audio_worker():
    """
    检测语音活动，自然停顿时触发转录。
    逻辑：音量超阈值 = 说话中；连续沉默 > SILENCE_GAP_SEC = 说完一句。
    """
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
                threading.Thread(
                    target=transcribe_and_broadcast,
                    args=(audio,),
                    daemon=True,
                ).start()
            buf = []
            speech_frames = 0
            silence_frames = 0


# ─── FastAPI 应用 ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, translator, event_loop

    print("─" * 50)
    print("加载 Whisper 模型中（首次运行会从 HuggingFace 下载约 1.5GB）")
    print(f"模型: {MODEL_SIZE}  语言: {LANGUAGE}")
    whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("✅ 模型加载完成")

    translator = Translator()

    event_loop = asyncio.get_running_loop()

    threading.Thread(target=audio_worker, daemon=True).start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_MS / 1000),
        callback=audio_callback,
    )
    stream.start()

    ip = get_local_ip()
    print(f"\n✅ 教会翻译系统已启动")
    print(f"   投影字幕：http://{ip}:8000")
    print(f"   会众手机：http://{ip}:8000（同一 WiFi 下扫码）")
    print("─" * 50 + "\n")

    yield

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
    url = f"http://{ip}:8000"
    return JSONResponse({"url": url, "qr": make_qr_base64(url)})


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
