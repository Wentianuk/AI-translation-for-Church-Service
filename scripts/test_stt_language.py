#!/usr/bin/env python3
"""
快速测试 OpenAI STT API 是否接受 language=yue（纯标准库，无需 requests）。
用法：先 export OPENAI_API_KEY=sk-xxx，然后运行本脚本。
"""
import io
import json
import os
import sys
import uuid
import wave
from urllib.request import Request, urlopen
from urllib.error import HTTPError

key = os.getenv("OPENAI_API_KEY", "")
if not key:
    print("请先 export OPENAI_API_KEY=sk-xxx")
    sys.exit(1)

print(f"API key: {key[:8]}...{key[-4:]}")
print()

buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(b"\x00\x00" * 8000)
wav_bytes = buf.getvalue()

url = "https://api.openai.com/v1/audio/transcriptions"


def multipart_encode(fields: dict, file_name: str, file_bytes: bytes):
    boundary = uuid.uuid4().hex
    lines = []
    for k, v in fields.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{k}"'.encode())
        lines.append(b"")
        lines.append(v.encode() if isinstance(v, str) else v)
    lines.append(f"--{boundary}".encode())
    lines.append(f'Content-Disposition: form-data; name="file"; filename="{file_name}"'.encode())
    lines.append(b"Content-Type: audio/wav")
    lines.append(b"")
    lines.append(file_bytes)
    lines.append(f"--{boundary}--".encode())
    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


for lang in ["yue", "zh"]:
    print(f"--- 测试 language={lang} (gpt-4o-mini-transcribe) ---")
    fields = {"model": "gpt-4o-mini-transcribe", "language": lang}
    body, ct = multipart_encode(fields, "test.wav", wav_bytes)
    req = Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", ct)
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            print(f"  HTTP {resp.status}")
            print(f"  ✅ language={lang} 被接受！响应: {data}")
    except HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")[:300]
        print(f"  HTTP {e.code}")
        if e.code == 400:
            print(f"  ❌ language={lang} 不被支持: {body_text}")
        else:
            print(f"  ⚠️ 错误: {body_text}")
    except Exception as e:
        print(f"  异常: {e}")
    print()
