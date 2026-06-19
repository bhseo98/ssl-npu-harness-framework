"""Non-interactive smoke test that proves the framework's swap promise.

No microphone, no GPU, no model downloads required — uses synthetic numpy
audio and the lightweight `fixed_text` / `buffer_info` STT stubs +
inline dummy LLM/TTS.

The point: the *same* VoicePipeline runs end-to-end with two different STT
implementations selected by `type:` in a config dict. That is the only
production code path; the demo just makes it visible.

Run:
    PYTHONPATH=src python3 scripts/swap_demo.py
"""
from __future__ import annotations

import numpy as np

from npu_harness_framework import stt as _stt_module  # noqa: F401 (registers built-ins)
from npu_harness_framework.interfaces import AudioBuffer, BaseLLM, BaseTTS
from npu_harness_framework.pipeline import VoicePipeline
from npu_harness_framework.registry import build, registered


class _EchoLLM(BaseLLM):
    def chat(self, user_message: str, history=None) -> str:
        return f"(echo) {user_message}"


class _SilentTTS(BaseTTS):
    def synthesize(self, text: str) -> AudioBuffer:
        return AudioBuffer(
            samples=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
        )


def _synthetic_audio(duration_s: float = 1.0, sample_rate: int = 16000) -> AudioBuffer:
    t = np.linspace(0.0, duration_s, int(duration_s * sample_rate), dtype=np.float32)
    samples = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


def _banner(text: str) -> None:
    print("\n" + "━" * 60)
    print(f"  {text}")
    print("━" * 60)


def run_one(label: str, stt_cfg: dict, audio: AudioBuffer) -> None:
    _banner(f"swap target: STT = {stt_cfg}")
    stt = build("stt", stt_cfg)
    pipeline = VoicePipeline(stt, _EchoLLM(), _SilentTTS(), profiler_enabled=False)
    user_text, reply, audio_out = pipeline.run(audio)
    print(f"   ↳ pipeline class : {type(pipeline).__name__}  (unchanged)")
    print(f"   ↳ stt class      : {type(stt).__name__}")
    print(f"   ↳ transcript     : {user_text!r}")
    print(f"   ↳ reply          : {reply!r}")
    print(f"   ↳ audio samples  : {audio_out['samples'].shape}")


def main() -> int:
    _banner("NPU Harness Framework — model-swap smoke test")
    print(f"  registry: {registered()}")
    print("  (no model downloads — uses fixed_text / buffer_info stubs)")

    audio = _synthetic_audio(duration_s=1.0)

    run_one(
        "A — fixed string transcription",
        {"type": "fixed_text", "text": "안녕하세요, 오늘 날씨 어때요?"},
        audio,
    )
    run_one(
        "B — buffer-info transcription",
        {"type": "buffer_info"},
        audio,
    )
    run_one(
        "C — fixed string with different text (same type, swapped kwargs)",
        {"type": "fixed_text", "text": "다른 입력입니다."},
        audio,
    )

    _banner("✅  Same VoicePipeline, three swappable STTs. Framework OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
