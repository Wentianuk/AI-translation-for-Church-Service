# AI Translation for Church Service

> 教会同声翻译系统 — 粤语实时转普通话字幕

Real-time Cantonese → Mandarin simultaneous translation for church sermons and events. The congregation follows along on their phones or a projected screen — no headsets, no interpreter booths.

---

## Features

- **Real-time Cantonese (粤語) ASR** via Whisper large-v3-turbo (`language="yue"`)
- **Cantonese → Mandarin translation** via Google Translate (free, no API key)
- **WebSocket live subtitles** — updates instantly on every connected browser
- **Dual display**: large projected screen (`?mode=projector`) + congregation phones
- **QR code** built-in — one tap to share the subtitle link over WiFi
- **History panel** — last 6 sentences shown above current subtitle
- **Auto-reconnect** — survives brief network drops
- Optimised for **Apple M1/M2/M3** (CPU int8 inference, ~0.5 s latency)

---

## Screenshot

```
┌──────────────────────────────────────────────────┐
│ 教会同声翻译 · 粤 → 普              ● 已连接     │
├──────────────────────────────────────────────────┤
│ [history] 神爱世人，甚至将他的独生子赐给他们      │
│ [history] 叫一切信他的，不至灭亡…                │
├══════════════════════════════════════════════════╡
│                                                  │
│  今天我们来看约翰福音三章十六节。                │
│  今日我哋睇约翰福音三章十六节。                  │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Requirements

- Python 3.11+
- macOS with [Homebrew](https://brew.sh) (for PortAudio)
- Same WiFi network for projector + congregation phones

---

## Installation (M1 Mac)

```bash
# 1. Install PortAudio (audio input dependency)
brew install portaudio

# 2. Clone the repo
git clone https://github.com/Wentianuk/AI-translation-for-Church-Service.git
cd AI-translation-for-Church-Service

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

On first run, Whisper large-v3-turbo (~1.5 GB) is downloaded automatically from HuggingFace.

---

## Usage

```
✅ 教会翻译系统已启动
   投影字幕：http://192.168.x.x:8000
   会众手机：http://192.168.x.x:8000
```

| Purpose | URL |
|---|---|
| Projector / big screen | `http://<IP>:8000/?mode=projector` |
| Congregation phones | `http://<IP>:8000` (scan QR in bottom-right corner) |

---

## Configuration

Edit the top of `main.py` to tune for your setup:

```python
MODEL_SIZE     = "large-v3-turbo"  # or "medium" for faster/lower accuracy
LANGUAGE       = "yue"             # Cantonese
SILENCE_RMS    = 0.008             # mic sensitivity
SILENCE_GAP_SEC = 0.7              # pause length to trigger processing (seconds)
MAX_SEGMENT_SEC = 10               # force process after this many seconds
```

---

## Architecture

```
Microphone
   │
   ▼  (sounddevice, 100 ms blocks)
Audio Queue
   │
   ▼  (VAD — energy threshold)
Speech Segments
   │
   ▼  (faster-whisper, language="yue")
Cantonese Text
   │
   ▼  (Google Translate yue → zh-CN)
Mandarin Text
   │
   ▼  (WebSocket broadcast)
Browser Subtitle Page  ←  projector / phones
```

---

## License

MIT
