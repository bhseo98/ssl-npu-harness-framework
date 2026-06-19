"""Browser-based voice assistant demo.

The GPU server runs STT/LLM/TTS; the user's browser supplies the microphone
and speaker over a WebSocket. This avoids requiring /dev/snd inside Docker.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

# Importing the impl modules triggers their @register(...) decorators.
from npu_harness_framework import llm as _llm  # noqa: F401
from npu_harness_framework import stt as _stt  # noqa: F401
from npu_harness_framework import tts as _tts  # noqa: F401
from npu_harness_framework import vp as _vp  # noqa: F401  # on-chip (VP) adapters
from npu_harness_framework.pipeline import VoicePipeline
from npu_harness_framework.registry import build, registered

BROWSER_SAMPLE_RATE = 16000
MAX_RECORDING_BYTES = BROWSER_SAMPLE_RATE * 2 * 60
MIN_INPUT_RMS = 0.001
DEMO_BUILD = "ptt-hold-toss-2026-06-12"


def _input_extension(mime_type: str) -> str:
    if "mp4" in mime_type:
        return ".m4a"
    if "ogg" in mime_type:
        return ".ogg"
    if "wav" in mime_type:
        return ".wav"
    return ".webm"


def _parse_args() -> argparse.Namespace:
    default_cfg = os.environ.get("TARGET_APP_CONFIG", "configs/default.yaml")
    parser = argparse.ArgumentParser(description="Browser mic/speaker demo")
    parser.add_argument("--config", "-c", default=default_cfg)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def _load_config(config_path: str) -> tuple[Path, dict[str, Any]]:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open() as f:
        return cfg_path, yaml.safe_load(f)


def _print_banner(cfg_path: Path, cfg: dict[str, Any], host: str, port: int) -> None:
    print()
    print("━" * 60)
    print("  🎤  NPU Harness Framework — Browser Voice Demo")
    print("━" * 60)
    print(f"  config:    {cfg_path}")
    print(f"  listen:    http://{host}:{port}")
    print(f"  language:  {cfg['pipeline']['language']}")
    print(f"  budget:    {cfg['pipeline']['budget_mb']} MB")
    print(f"  registry:  {registered()}")
    print(f"  demo:      {DEMO_BUILD}")
    print("━" * 60)
    print()


def _build_pipeline(cfg: dict[str, Any]) -> VoicePipeline:
    print("⏳ Loading models (first run downloads weights — be patient)…")
    return VoicePipeline(
        build("stt", cfg["stt"]),
        build("llm", cfg["llm"]),
        build("tts", cfg["tts"]),
        profiler_enabled=cfg["profiler"]["enabled"],
        log_path=cfg["profiler"].get("log_path"),
        budget_mb=cfg["pipeline"].get("budget_mb"),
    )


def _pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        pcm = pcm[:-1]
    audio_i16 = np.frombuffer(pcm, dtype="<i2")
    return (audio_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    out = io.BytesIO()
    sf.write(out, samples, sample_rate, format="WAV")
    return out.getvalue()


def _audio_stats(samples: np.ndarray) -> tuple[float, float, float]:
    if samples.size == 0:
        return 0.0, 0.0, 0.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    duration_s = samples.shape[0] / BROWSER_SAMPLE_RATE
    return duration_s, rms, peak


def _decode_media_to_mono16k(path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-ar",
                str(BROWSER_SAMPLE_RATE),
                "-ac",
                "1",
                str(wav_path),
            ],
            check=True,
        )
        data, _sr = sf.read(wav_path, dtype="float32")
    finally:
        wav_path.unlink(missing_ok=True)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


async def _reject_silent_input(
    websocket: WebSocket,
    input_path: Path,
    duration_s: float,
    rms: float,
    peak: float,
    input_format: str,
) -> bool:
    if rms >= MIN_INPUT_RMS:
        return False
    await websocket.send_json(
        {
            "type": "error",
            "message": (
                f"No microphone signal detected ({input_format}, rms={rms:.6f}). "
                "Rebuild Docker with the latest web_demo.py, open http://localhost:8000 "
                "via SSH tunnel, hard-refresh (Cmd+Shift+R), then try again."
            ),
            "input_path": str(input_path),
            "duration_s": round(duration_s, 2),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
        }
    )
    return True


def _make_app(cfg: dict[str, Any], pipeline: VoicePipeline) -> FastAPI:
    app = FastAPI()
    rec_dir = Path(cfg["audio"]["output_dir"])
    rec_dir.mkdir(parents=True, exist_ok=True)
    turns = count(1)

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "sample_rate": BROWSER_SAMPLE_RATE,
                "registry": registered(),
                "demo_build": DEMO_BUILD,
            }
        )

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        audio_bytes = bytearray()
        input_format = "pcm16"
        mime_type = "audio/pcm;rate=16000"
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return

                chunk = message.get("bytes")
                text = message.get("text")

                if chunk is not None:
                    audio_bytes.extend(chunk)
                    if len(audio_bytes) > MAX_RECORDING_BYTES:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Recording is longer than 60 seconds.",
                            }
                        )
                        audio_bytes.clear()
                    continue

                if text is None:
                    continue

                event = json.loads(text)
                if event.get("type") == "start":
                    audio_bytes.clear()
                    input_format = event.get("format", "pcm16")
                    mime_type = event.get("mime_type", "audio/pcm;rate=16000")
                    await websocket.send_json(
                        {
                            "type": "status",
                            "message": "recording",
                            "format": input_format,
                            "mime_type": mime_type,
                        }
                    )
                elif event.get("type") == "stop":
                    turn = next(turns)
                    await _handle_turn(
                        websocket,
                        pipeline,
                        rec_dir,
                        turn,
                        bytes(audio_bytes),
                        input_format,
                        mime_type,
                    )
                    audio_bytes.clear()
        except (RuntimeError, WebSocketDisconnect) as exc:
            if isinstance(exc, RuntimeError) and "disconnect message" not in str(exc):
                raise
            return

    return app


async def _handle_turn(
    websocket: WebSocket,
    pipeline: VoicePipeline,
    rec_dir: Path,
    turn: int,
    audio_data: bytes,
    input_format: str,
    mime_type: str,
) -> None:
    if not audio_data:
        await websocket.send_json({"type": "error", "message": "No audio captured."})
        return

    input_path: Path
    wav_out = rec_dir / f"out_web_{turn:03d}.wav"

    input_path = rec_dir / f"in_web_{turn:03d}.wav"

    if input_format == "pcm16":
        samples = _pcm16_bytes_to_float32(audio_data)
    else:
        raw_path = rec_dir / f"in_web_{turn:03d}{_input_extension(mime_type)}"
        raw_path.write_bytes(audio_data)
        samples = await asyncio.to_thread(_decode_media_to_mono16k, raw_path)

    duration_s, rms, peak = _audio_stats(samples)
    sf.write(input_path, samples, BROWSER_SAMPLE_RATE)
    if await _reject_silent_input(
        websocket, input_path, duration_s, rms, peak, input_format
    ):
        return

    await websocket.send_json(
        {
            "type": "status",
            "message": "processing",
            "turn": turn,
            "input_path": str(input_path),
            "duration_s": round(duration_s, 2),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "bytes": len(audio_data),
            "mime_type": mime_type,
            "format": input_format,
        }
    )

    user_text, reply, audio_out = await asyncio.to_thread(pipeline.run, input_path)
    sf.write(wav_out, audio_out["samples"], audio_out["sample_rate"])

    audio_b64 = base64.b64encode(
        _wav_bytes(audio_out["samples"], audio_out["sample_rate"])
    ).decode("ascii")
    await websocket.send_json(
        {
            "type": "result",
            "turn": turn,
            "user_text": user_text,
            "reply": reply,
            "input_path": str(input_path),
            "output_path": str(wav_out),
            "audio_wav_base64": audio_b64,
        }
    )


_INDEX_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>음성 어시스턴트 · NPU Harness</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@latest/dist/web/static/pretendard.min.css" />
  <style>
    :root {
      color-scheme: light;
      --bg: #f2f4f6;
      --card: #ffffff;
      --ink: #191f28;
      --sub: #4e5968;
      --muted: #8b95a1;
      --line: #e5e8eb;
      --blue: #3182f6;
      --blue-strong: #1b64da;
      --blue-soft: #e8f3ff;
      --danger: #f04452;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Pretendard, "Apple SD Gothic Neo", ui-sans-serif, system-ui,
        -apple-system, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      -webkit-font-smoothing: antialiased;
    }
    main {
      width: min(480px, calc(100vw - 32px));
      margin: 0 auto;
      min-height: 100vh;
      padding: 24px 0 32px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    header { display: flex; justify-content: space-between; align-items: center; }
    .brand { display: flex; flex-direction: column; gap: 2px; }
    .kicker { font-size: 13px; font-weight: 600; color: var(--muted); }
    h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
    .status {
      font-size: 13px;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: 999px;
      color: var(--muted);
      background: #eef1f4;
      white-space: nowrap;
    }
    .status[data-state="rec"] { color: #fff; background: var(--danger); }
    .status[data-state="busy"] { color: var(--blue-strong); background: var(--blue-soft); }
    .status[data-state="ready"] { color: var(--blue); background: var(--blue-soft); }

    .convo { display: flex; flex-direction: column; gap: 10px; }
    .bubble {
      padding: 14px 16px;
      border-radius: 18px;
      font-size: 16px;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-width: 88%;
    }
    .bubble .role { font-size: 12px; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.02em; }
    .bubble.user {
      align-self: flex-end;
      background: var(--card);
      border: 1px solid var(--line);
      border-bottom-right-radius: 6px;
    }
    .bubble.user .role { color: var(--muted); }
    .bubble.bot {
      align-self: flex-start;
      background: var(--blue);
      color: #fff;
      border-bottom-left-radius: 6px;
    }
    .bubble.bot .role { color: rgba(255, 255, 255, 0.72); }
    audio { width: 100%; }

    .talkwrap {
      margin-top: auto;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      padding-top: 8px;
    }
    .ring { position: relative; width: 124px; height: 124px; display: grid; place-items: center; }
    .ring::before {
      content: "";
      position: absolute;
      width: 124px;
      height: 124px;
      border-radius: 50%;
      background: var(--danger);
      opacity: 0;
    }
    .ring.is-recording::before { animation: pulse 1.4s ease-out infinite; }
    @keyframes pulse {
      0% { transform: scale(1); opacity: 0.35; }
      100% { transform: scale(1.55); opacity: 0; }
    }
    #talkBtn {
      position: relative;
      width: 124px;
      height: 124px;
      border-radius: 50%;
      border: 0;
      cursor: pointer;
      background: var(--blue);
      color: #fff;
      font-size: 16px;
      font-weight: 700;
      line-height: 1.3;
      box-shadow: 0 12px 26px rgba(49, 130, 246, 0.35);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
      touch-action: none;
      user-select: none;
      -webkit-user-select: none;
      transition: transform .12s ease, background .12s ease, box-shadow .12s ease;
    }
    #talkBtn .mic { font-size: 30px; line-height: 1; }
    #talkBtn:active { transform: scale(0.95); }
    .ring.is-recording #talkBtn {
      background: var(--danger);
      box-shadow: 0 12px 30px rgba(240, 68, 82, 0.42);
      transform: scale(1.04);
    }
    #talkBtn:disabled { background: #c9d1d8; box-shadow: none; cursor: not-allowed; }
    .hint { font-size: 14px; font-weight: 600; color: var(--sub); }
    .meter { width: 168px; height: 6px; border-radius: 999px; background: #e5e8eb; overflow: hidden; }
    .meter > div { width: 0%; height: 100%; background: var(--blue); transition: width 80ms linear; }

    .tools { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: center; }
    select {
      height: 40px;
      max-width: 190px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--card);
      color: var(--ink);
      font-size: 14px;
      padding: 0 12px;
    }
    .ghost {
      height: 40px;
      padding: 0 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--card);
      color: var(--sub);
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }
    .ghost:active { background: #f2f4f6; }
    .warn {
      padding: 12px 14px;
      border: 1px solid #ffe08a;
      border-radius: 14px;
      background: #fff8e1;
      color: #8a6d00;
      font-size: 13px;
      line-height: 1.5;
    }
    .warn code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: rgba(0, 0, 0, 0.05);
      padding: 1px 5px;
      border-radius: 5px;
    }
    .logbox { font-size: 12px; color: var(--muted); }
    .logbox summary { cursor: pointer; font-weight: 600; color: var(--sub); }
    .log {
      margin-top: 8px;
      max-height: 150px;
      overflow: auto;
      padding: 10px 12px;
      border-radius: 12px;
      background: #f2f4f6;
      color: var(--sub);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="brand">
        <span class="kicker">NPU Harness</span>
        <h1>음성 어시스턴트</h1>
      </div>
      <div id="status" class="status" data-state="idle">대기 중</div>
    </header>
    <div id="secureWarn" class="warn" hidden></div>
    <div class="convo">
      <div class="bubble user" hidden><div class="role">내 말</div><span id="userText"></span></div>
      <div class="bubble bot" hidden><div class="role">어시스턴트</div><span id="replyText"></span></div>
    </div>
    <audio id="player" controls hidden></audio>
    <div class="talkwrap">
      <div id="ring" class="ring">
        <button id="talkBtn" type="button">
          <span class="mic" aria-hidden="true">🎙️</span>
          <span>꾹 눌러<br>말하기</span>
        </button>
      </div>
      <div class="meter" aria-hidden="true"><div id="level"></div></div>
      <div class="hint">누르고 있는 동안 녹음 · 떼면 전송</div>
      <div class="tools">
        <select id="micSelect" aria-label="마이크"></select>
        <button id="refreshBtn" type="button" class="ghost">새로고침</button>
        <button id="testMicBtn" type="button" class="ghost">마이크 테스트</button>
      </div>
    </div>
    <details class="logbox">
      <summary>로그 보기</summary>
      <div id="log" class="log"></div>
    </details>
  </main>
  <script>
    const DEMO_BUILD = "ptt-hold-toss-2026-06-12";
    const SAMPLE_RATE = 16000;
    const statusEl = document.getElementById("status");
    const talkBtn = document.getElementById("talkBtn");
    const ring = document.getElementById("ring");
    const refreshBtn = document.getElementById("refreshBtn");
    const testMicBtn = document.getElementById("testMicBtn");
    const micSelect = document.getElementById("micSelect");
    const levelEl = document.getElementById("level");
    const secureWarnEl = document.getElementById("secureWarn");
    const logEl = document.getElementById("log");
    const userText = document.getElementById("userText");
    const replyText = document.getElementById("replyText");
    const player = document.getElementById("player");

    let ws;
    let stream;
    let audioCtx;
    let source;
    let captureNode;
    let keepAliveGain;
    let recording = false;
    let holding = false;
    let processing = false;
    let recordedPeak = 0;
    let recordedEnergy = 0;
    let recordedFrames = 0;

    function log(line) {
      const stamp = new Date().toLocaleTimeString();
      logEl.textContent = `[${stamp}] ${line}\\n` + logEl.textContent;
    }

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function ensureSocket() {
      if (ws && ws.readyState === WebSocket.OPEN) return ws;
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${scheme}://${location.host}/ws`);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => log("connected");
      ws.onclose = () => log("disconnected");
      ws.onerror = () => log("socket error");
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "status") {
          setStatus(data.message);
          if (data.input_path) {
            if (data.duration_s === null || data.duration_s === undefined) {
              log(`saved ${data.input_path} (${data.bytes} bytes, ${data.mime_type})`);
            } else {
              log(`saved ${data.input_path} (${data.duration_s}s, rms=${data.rms}, peak=${data.peak})`);
            }
          }
        } else if (data.type === "result") {
          setStatus("준비됨");
          statusEl.dataset.state = "ready";
          processing = false;
          talkBtn.disabled = false;
          userText.textContent = data.user_text || "";
          replyText.textContent = data.reply || "";
          userText.parentElement.hidden = !data.user_text;
          replyText.parentElement.hidden = !data.reply;
          player.src = `data:audio/wav;base64,${data.audio_wav_base64}`;
          player.hidden = false;
          player.play().catch(() => {});
          log(`reply ${data.output_path}`);
        } else if (data.type === "error") {
          setStatus("오류");
          statusEl.dataset.state = "idle";
          processing = false;
          talkBtn.disabled = false;
          log(data.message);
          if (data.input_path) {
            log(`saved ${data.input_path} (${data.duration_s}s, rms=${data.rms}, peak=${data.peak})`);
          }
        }
      };
      return ws;
    }

    async function refreshDevices(requestPermission = false) {
      if (requestPermission) {
        const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
        probe.getTracks().forEach((track) => track.stop());
      }
      const previous = micSelect.value;
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter((device) => device.kind === "audioinput");
      micSelect.innerHTML = "";
      for (const [index, device] of inputs.entries()) {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `Microphone ${index + 1}`;
        micSelect.appendChild(option);
      }
      if (previous && [...micSelect.options].some((option) => option.value === previous)) {
        micSelect.value = previous;
      } else {
        const builtIn = inputs.find((d) =>
          /macbook|built-in|internal|마이크/i.test(d.label || "")
        );
        if (builtIn) micSelect.value = builtIn.deviceId;
      }
      log(inputs.length ? `microphones: ${inputs.length}` : "no microphone found");
    }

    function showSecureContextWarning() {
      const host = location.hostname;
      const local =
        host === "localhost" || host === "127.0.0.1" || host === "[::1]";
      if (window.isSecureContext && local) return;
      secureWarnEl.hidden = false;
      secureWarnEl.innerHTML =
        "마이크는 <b>MacBook에서</b> <code>http://localhost:8000</code> 으로 열 때만 안정적으로 동작합니다. " +
        "GPU 서버 IP(<code>http://" + host + ":" + location.port + "</code>)로 직접 접속하면 " +
        "레벨 미터가 0으로 고정될 수 있습니다.<br><br>" +
        "Mac 터미널: <code>ssh -L 8000:localhost:8000 USER@GPU_SERVER</code> " +
        "→ 브라우저에서 <code>http://localhost:8000</code>";
      log(`insecure or remote host: secure=${window.isSecureContext} host=${host}`);
    }

    function logTrackState(track) {
      if (!track) return;
      log(
        `track: enabled=${track.enabled} muted=${track.muted} ` +
        `ready=${track.readyState} state=${audioCtx ? audioCtx.state : "n/a"}`
      );
      track.onmute = () => log("track muted (browser/OS)");
      track.onunmute = () => log("track unmuted");
    }

    function downsampleTo16k(input, inputRate) {
      if (inputRate === SAMPLE_RATE) return input;
      const ratio = inputRate / SAMPLE_RATE;
      const outputLength = Math.floor(input.length / ratio);
      const output = new Float32Array(outputLength);
      for (let i = 0; i < outputLength; i++) {
        const start = Math.floor(i * ratio);
        const end = Math.min(Math.floor((i + 1) * ratio), input.length);
        let sum = 0;
        for (let j = start; j < end; j++) sum += input[j];
        output[i] = sum / Math.max(1, end - start);
      }
      return output;
    }

    function floatToPcm16(float32) {
      const out = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return out;
    }

    function micConstraints() {
      const audio = {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      };
      if (micSelect.value) audio.deviceId = { ideal: micSelect.value };
      return { audio };
    }

    function mixDownToMono(buffer) {
      const channels = buffer.numberOfChannels;
      const len = buffer.length;
      const mono = new Float32Array(len);
      for (let ch = 0; ch < channels; ch++) {
        const data = buffer.getChannelData(ch);
        for (let i = 0; i < len; i++) mono[i] += data[i];
      }
      if (channels > 1) {
        for (let i = 0; i < len; i++) mono[i] /= channels;
      }
      return mono;
    }

    function meterPeakFromSamples(samples, accumulate) {
      let peak = 0;
      let energy = 0;
      for (let i = 0; i < samples.length; i++) {
        const abs = Math.abs(samples[i]);
        peak = Math.max(peak, abs);
        energy += samples[i] * samples[i];
      }
      if (accumulate) {
        recordedPeak = Math.max(recordedPeak, peak);
        recordedEnergy += energy;
        recordedFrames += samples.length;
      }
      levelEl.style.width = `${Math.min(100, Math.round(peak * 140))}%`;
      return peak;
    }

    async function ensureCaptureWorklet(ctx) {
      if (!ctx.audioWorklet) return false;
      const code = `
        class MicCaptureProcessor extends AudioWorkletProcessor {
          process(inputs) {
            const input = inputs[0];
            if (!input || !input[0]) return true;
            const ch0 = input[0];
            const copy = new Float32Array(ch0.length);
            copy.set(ch0);
            if (input.length > 1) {
              for (let ch = 1; ch < input.length; ch++) {
                const c = input[ch];
                for (let i = 0; i < copy.length; i++) copy[i] += c[i];
              }
              for (let i = 0; i < copy.length; i++) copy[i] /= input.length;
            }
            this.port.postMessage(copy, [copy.buffer]);
            return true;
          }
        }
        registerProcessor("mic-capture", MicCaptureProcessor);
      `;
      const url = URL.createObjectURL(new Blob([code], { type: "application/javascript" }));
      try {
        await ctx.audioWorklet.addModule(url);
      } catch (err) {
        if (!/already been loaded/i.test(String(err))) throw err;
      } finally {
        URL.revokeObjectURL(url);
      }
      return true;
    }

    async function openMicContext() {
      stream = await navigator.mediaDevices.getUserMedia(micConstraints());
      const track = stream.getAudioTracks()[0];
      const settings = track ? track.getSettings() : {};
      log(`using mic: ${track ? track.label : "unknown"}`);
      log(
        `browser audio: sampleRate=${settings.sampleRate || "auto"}, ` +
        `channels=${settings.channelCount || "auto"}, ` +
        `aec=${settings.echoCancellation}`
      );
      logTrackState(track);

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContextClass({ latencyHint: "interactive" });
      if (audioCtx.state === "suspended") await audioCtx.resume();
      source = audioCtx.createMediaStreamSource(stream);
      keepAliveGain = audioCtx.createGain();
      keepAliveGain.gain.value = 0.00001;
      return track;
    }

    async function wireCaptureNode(onSamples) {
      const ok = await ensureCaptureWorklet(audioCtx);
      if (!ok) return wireScriptProcessor(onSamples);
      captureNode = new AudioWorkletNode(audioCtx, "mic-capture", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
      });
      captureNode.port.onmessage = (event) => {
        if (!recording) return;
        onSamples(event.data);
      };
      source.connect(captureNode);
      captureNode.connect(keepAliveGain);
      keepAliveGain.connect(audioCtx.destination);
      return "AudioWorklet";
    }

    function wireScriptProcessor(onSamples) {
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        if (!recording) return;
        onSamples(mixDownToMono(event.inputBuffer));
      };
      source.connect(processor);
      processor.connect(keepAliveGain);
      keepAliveGain.connect(audioCtx.destination);
      captureNode = processor;
      return "ScriptProcessor";
    }

    function onCaptureSamples(float32) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        const pcm = floatToPcm16(downsampleTo16k(float32, audioCtx.sampleRate));
        ws.send(pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength));
      }
      meterPeakFromSamples(float32, true);
    }

    async function testMicrophone() {
      await openMicContext();
      let peak = 0;
      const onLevel = (samples) => {
        peak = Math.max(peak, meterPeakFromSamples(samples, false));
      };
      const mode = audioCtx.audioWorklet
        ? await wireCaptureNode(onLevel)
        : wireScriptProcessor(onLevel);

      const chunks = [];
      const mr = new MediaRecorder(stream);
      mr.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      mr.start(250);
      setStatus("Mic test (3s) — speak now");
      recording = true;
      await new Promise((resolve) => setTimeout(resolve, 3000));
      recording = false;
      await new Promise((resolve) => {
        mr.addEventListener("stop", resolve, { once: true });
        mr.stop();
      });

      const blob = new Blob(chunks, { type: mr.mimeType || "audio/webm" });
      player.src = URL.createObjectURL(blob);
      player.play().catch(() => {});
      log(`test mic (${mode}): peak=${peak.toFixed(6)} — listen to playback below`);
      if (peak < 0.001) {
        log("Web Audio peak=0. If playback is also silent: macOS Sound → Input, or try another mic in the list.");
      }

      if (captureNode) captureNode.disconnect();
      if (keepAliveGain) keepAliveGain.disconnect();
      if (source) source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      await audioCtx.close();
      captureNode = null;
      keepAliveGain = null;
      source = null;
      stream = null;
      audioCtx = null;
      levelEl.style.width = "0%";
      setStatus("Idle");
    }

    async function startRecording() {
      ensureSocket();
      if (ws.readyState !== WebSocket.OPEN) {
        await new Promise((resolve) => ws.addEventListener("open", resolve, { once: true }));
      }

      await refreshDevices(false);
      await openMicContext();
      recordedPeak = 0;
      recordedEnergy = 0;
      recordedFrames = 0;

      ws.send(JSON.stringify({
        type: "start",
        format: "pcm16",
        mime_type: "audio/pcm;rate=16000;encoding=signed-integer;bits=16",
      }));
      await new Promise((resolve) => setTimeout(resolve, 50));

      const mode = await wireCaptureNode(onCaptureSamples);
      recording = true;
      log(`capture: ${mode} @ ${audioCtx.sampleRate}Hz, echoCancellation=off -> ${SAMPLE_RATE}Hz pcm16`);
    }

    async function stopRecording() {
      recording = false;
      levelEl.style.width = "0%";

      if (captureNode) {
        if (captureNode.port) captureNode.port.onmessage = null;
        if (captureNode.onaudioprocess) captureNode.onaudioprocess = null;
        captureNode.disconnect();
      }
      if (keepAliveGain) keepAliveGain.disconnect();
      if (source) source.disconnect();
      if (stream) stream.getTracks().forEach((track) => track.stop());
      if (audioCtx) await audioCtx.close();

      captureNode = null;
      keepAliveGain = null;
      source = null;
      stream = null;
      audioCtx = null;

      const rms = recordedFrames ? Math.sqrt(recordedEnergy / recordedFrames) : 0;
      log(`browser captured: rms=${rms.toFixed(6)}, peak=${recordedPeak.toFixed(6)}`);
      ws.send(JSON.stringify({ type: "stop" }));
    }

    async function finishRecording() {
      processing = true;
      talkBtn.disabled = true;
      statusEl.dataset.state = "busy";
      setStatus("전송 중");
      try { await stopRecording(); }
      catch (err) { log(err.message); }
    }

    async function pressStart(ev) {
      if (ev) ev.preventDefault();
      if (recording || processing || talkBtn.disabled) return;
      holding = true;
      ring.classList.add("is-recording");
      statusEl.dataset.state = "rec";
      setStatus("녹음 중");
      try {
        await startRecording();
      } catch (err) {
        holding = false;
        ring.classList.remove("is-recording");
        statusEl.dataset.state = "idle";
        setStatus("오류");
        log(err.message);
        return;
      }
      if (!holding) await finishRecording();
    }

    function pressEnd() {
      if (!holding) return;
      holding = false;
      ring.classList.remove("is-recording");
      if (recording) finishRecording();
    }

    talkBtn.addEventListener("pointerdown", pressStart);
    talkBtn.addEventListener("pointerup", pressEnd);
    talkBtn.addEventListener("pointerleave", pressEnd);
    talkBtn.addEventListener("pointercancel", pressEnd);
    talkBtn.addEventListener("contextmenu", (ev) => ev.preventDefault());
    refreshBtn.addEventListener("click", () => refreshDevices(true).catch((err) => log(err.message)));
    testMicBtn.addEventListener("click", () => {
      talkBtn.disabled = true;
      testMicrophone().catch((err) => log(err.message)).finally(() => { talkBtn.disabled = false; });
    });
    navigator.mediaDevices.addEventListener("devicechange", () => refreshDevices(false).catch(() => {}));
    log(`demo build: ${DEMO_BUILD}`);
    showSecureContextWarning();
    ensureSocket();
    refreshDevices(false).catch(() => {});
  </script>
</body>
</html>
"""


def main() -> int:
    args = _parse_args()
    try:
        cfg_path, cfg = _load_config(args.config)
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    _print_banner(cfg_path, cfg, args.host, args.port)
    pipeline = _build_pipeline(cfg)
    print("\n✅ Ready. Open the browser UI and press Ctrl+C here to exit.\n")

    app = _make_app(cfg, pipeline)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
