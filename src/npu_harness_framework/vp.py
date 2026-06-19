"""VP (on-chip) pipeline adapters.

Instead of running models locally, these stages drive the RISC-V64 Virtual
Platform guest engines over the serial console via ``vp_console.sh`` (persistent
VP, FIFO stdin, marker-delimited output). Audio is streamed to the guest as
base64 (no block device on this VP). STT/LLM run *on the chip*.

Model swap stays a one-line YAML change: ``type: vp_whisper`` / ``vp_llama``.
TTS voice-out is deferred (VP-C); ``vp_silent`` returns a short silent buffer so
the pipeline completes and the browser shows STT+LLM text.

Env overrides: ``VP_CONSOLE`` (default /tmp/vp_console.sh), ``VP_WHISPER_MODEL``,
``VP_LLM_MODEL``.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import tempfile
import threading
import urllib.request

import numpy as np
import soundfile as sf

from .interfaces import AudioBuffer, AudioInput, BaseLLM, BaseSTT, BaseTTS
from .registry import register

VP_CONSOLE = os.environ.get("VP_CONSOLE", "/tmp/vp_console.sh")
# Host TTS server (MeloTTS Korean): neural TTS won't fit the 2GB chip (no torch on
# riscv64, no model volume), so synthesis runs in a host venv and we POST the reply
# text. The container reaches the host via host.containers.internal.
VP_TTS_URL = os.environ.get("VP_TTS_URL", "http://host.containers.internal:8770/tts")
_LOCK = threading.Lock()  # single VP → serialize turns


def _vp(*args: str, timeout: int = 600) -> str:
    """Invoke a vp_console.sh subcommand; return its stdout."""
    r = subprocess.run(
        ["bash", VP_CONSOLE, *args],
        capture_output=True, text=True, timeout=timeout + 90,
    )
    return r.stdout


def _clean(s: str) -> str:
    """Strip console artifacts: ANSI colors, box-drawing banner chars, shell
    prompts ('# '/'> ' echoed by the guest tty/heredoc)."""
    s = re.sub(r"\x1b\[[0-9;]*m", "", s)              # ANSI colors
    s = re.sub(r"[▀-▟─-╿]", "", s)  # box-drawing (llama banner)
    out = []
    for ln in s.splitlines():
        ln = ln.rstrip("\r")
        ln = re.sub(r"^(?:[#>]\s*)+", "", ln)        # leading '# '/'> ' prompts (repeated)
        if ln.strip():
            out.append(ln)
    return "\n".join(out).strip()


@register("stt", "vp_whisper")
class VpWhisper(BaseSTT):
    """STT on chip: stream wav to guest, run whisper-cli, return transcription."""

    def __init__(self, model: str = "/opt/models/ggml-tiny-q8_0.bin",
                 language: str = "ko", **_: object):
        self._model = os.environ.get("VP_WHISPER_MODEL", model)
        # Force the decode language. Without -l, whisper auto-detects and mishears
        # short Korean as English ("안녕"->"No", "카이스트"->"Caste").
        self._lang = os.environ.get("VP_WHISPER_LANG", language)

    def transcribe(self, audio: AudioInput) -> str:
        # materialize a 16k mono wav on the host side
        if isinstance(audio, dict):
            wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(wav, audio["samples"], audio["sample_rate"])
        else:
            wav = str(audio)
        b64 = tempfile.NamedTemporaryFile(suffix=".b64", delete=False).name
        subprocess.run(f"base64 -w 1000 {wav!r} > {b64!r}", shell=True, check=True)
        with _LOCK:
            _vp("audio", b64, timeout=400)
            out = _vp("run",
                      f"whisper-cli -m {self._model} -l {self._lang} "
                      "-f /tmp/in.wav -nt 2>/dev/null",
                      "300", timeout=360)
        text = _clean(out)
        # whisper may emit a leading newline; collapse to one line
        return " ".join(text.split())


@register("llm", "vp_llama")
class VpLlama(BaseLLM):
    """LLM on chip: run llama-cli single-turn with the user text as prompt."""

    def __init__(self, model: str = "/opt/models/qwen-q8-vp.gguf",
                 max_new_tokens: int = 32, **_: object):
        self._model = os.environ.get("VP_LLM_MODEL", model)
        self._n = int(max_new_tokens)

    def chat(self, user_message: str, history=None) -> str:
        prompt = (user_message or "").replace('"', "'")
        with _LOCK:
            out = _vp("run",
                      f'llama-cli -m {self._model} -p "{prompt}" -st -n {self._n} '
                      f"--simple-io 2>&1", "450", timeout=520)
        text = _clean(out)
        # llama-cli -st output: banner / build: / model: / commands / <prompt> /
        # <REPLY> / "[ Prompt: ...]" / "Exiting...". Keep only the reply: skip the
        # echoed prompt + everything before it, stop at the perf/exit footer.
        reply, seen = [], False
        pl = prompt.strip().lower()
        for ln in text.splitlines():
            low = ln.strip().lower()
            if not seen:
                if low == pl or (pl and pl in low) or low.endswith(pl):
                    seen = True
                continue
            if ln.startswith(("[ Prompt", "Exiting", "common_memory", "llama_",
                              "build", "model ", "modalities")):
                break
            reply.append(ln)
        joined = " ".join(" ".join(reply).split())
        if not joined:  # fallback: drop known noise lines, keep the rest
            noise = ("Loading model", "build", "model", "modalities", "available",
                     "/exit", "/regen", "/clear", "/read", "/glob",
                     "[ Prompt", "Exiting", "common_memory")
            joined = " ".join(" ".join(
                ln for ln in text.splitlines()
                if ln and not ln.startswith(noise)).split())
        return joined


@register("tts", "host_melotts")
class HostMeloTTS(BaseTTS):
    """TTS on the host (server4), not the chip: MeloTTS Korean served over HTTP.

    The chip can't run neural TTS (no torch on riscv64, no model volume for a
    multi-100MB model), so synthesis runs in a host venv (``tts_server.py``) and we
    POST the reply text, getting a 44.1kHz wav back. STT/LLM still run on the chip.
    """

    def __init__(self, url: str = VP_TTS_URL, timeout: int = 120, **_: object):
        self._url = os.environ.get("VP_TTS_URL", url)
        self._timeout = int(timeout)

    def synthesize(self, text: str) -> AudioBuffer:
        body = json.dumps({"text": text or ""}).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            wav = r.read()
        samples, sr = sf.read(io.BytesIO(wav), dtype="float32")
        return {"samples": samples, "sample_rate": int(sr)}


@register("tts", "vp_silent")
class VpSilent(BaseTTS):
    """Placeholder TTS (VP-C voice-out deferred): returns a short silent buffer."""

    def __init__(self, sample_rate: int = 16000, **_: object):
        self._sr = int(sample_rate)

    def synthesize(self, text: str) -> AudioBuffer:
        return {"samples": np.zeros(int(self._sr * 0.2), dtype=np.float32),
                "sample_rate": self._sr}
