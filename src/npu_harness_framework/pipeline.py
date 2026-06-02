"""End-to-end STT → LLM → TTS orchestration.

Two pipelines:

- ``VoicePipeline`` — eager instantiation. STT/LLM/TTS 가 동시에 메모리에
  상주. GPU 서버 iteration 용 (Phase 1 의 default).

- ``SequentialVoicePipeline`` — per-stage *load → run → unload*. 임베디드
  target 의 자연스러운 단순화 — 한 stage 의 모델만 메모리에 들고 끝나면
  buffer 만 다음 stage 로 넘김. 설계 검토 §2 "2GB 넘지 않는 모델"
  제약을 *동시 load peak* 가 아닌 *stage peak* 로 만족시킨다.

각 stage 는 동일하게 profiler context 로 감싸져 latency / memory 가 JSONL 에 기록.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from .interfaces import AudioBuffer, AudioInput, BaseLLM, BaseSTT, BaseTTS
from .profiler import measure
from .registry import build


def _free_gpu_cache() -> None:
    """Stage 종료 후 GPU memory pool 비움. CPU-only 환경에서는 no-op."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


class VoicePipeline:
    def __init__(
        self,
        stt: BaseSTT,
        llm: BaseLLM,
        tts: BaseTTS,
        profiler_enabled: bool = True,
        log_path: str | Path | None = None,
        budget_mb: int | None = None,
    ):
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._profile = profiler_enabled
        self._log = log_path
        self._budget = budget_mb
        self.history: list[tuple[str, str]] = []

    def run(self, audio_input: AudioInput) -> tuple[str, str, AudioBuffer]:
        log = self._log if self._profile else None
        budget = self._budget if self._profile else None

        with measure("stt", log, budget):
            user_text = self._stt.transcribe(audio_input)
        print(f"📝  STT  →  {user_text!r}")

        with measure("llm", log, budget):
            reply = self._llm.chat(user_text, history=self.history)
        print(f"💬  LLM  →  {reply!r}")
        self.history.append((user_text, reply))

        with measure("tts", log, budget):
            audio_out = self._tts.synthesize(reply)
        duration_s = audio_out["samples"].shape[0] / audio_out["sample_rate"]
        print(f"🔊  TTS  →  {duration_s:.2f}s audio @ {audio_out['sample_rate']}Hz")

        return user_text, reply, audio_out

    def reset_history(self) -> None:
        self.history.clear()


class SequentialVoicePipeline:
    """Per-stage instantiate → run → release. 동시 load peak 가 합쳐지지 않음.

    임베디드 target 의 *2GB DRAM* 제약 (설계 검토 §2) 안에서
    "weight 합" 이 아닌 *실제 peak RSS* 를 fit 시키기 위한 모드. trade-off:
    매 turn 마다 모델을 다시 build 하므로 첫 load 비용이 매 turn 반복된다.
    HF / ESPnet cache 가 disk 에 있다면 두 번째 turn 부터는 부담이 작다.

    API 가 ``VoicePipeline`` 과 같은 ``run(audio_input)`` / ``reset_history()`` /
    ``history`` 를 노출 — drop-in 호환.
    """

    def __init__(
        self,
        stt_config: dict[str, Any],
        llm_config: dict[str, Any],
        tts_config: dict[str, Any],
        profiler_enabled: bool = True,
        log_path: str | Path | None = None,
        budget_mb: int | None = None,
    ):
        self._stt_cfg = stt_config
        self._llm_cfg = llm_config
        self._tts_cfg = tts_config
        self._profile = profiler_enabled
        self._log = log_path
        self._budget = budget_mb
        self.history: list[tuple[str, str]] = []

    def run(self, audio_input: AudioInput) -> tuple[str, str, AudioBuffer]:
        log = self._log if self._profile else None
        budget = self._budget if self._profile else None

        with measure("stt", log, budget):
            stt = build("stt", self._stt_cfg)
            user_text = stt.transcribe(audio_input)
            del stt
            gc.collect()
            _free_gpu_cache()
        print(f"📝  STT  →  {user_text!r}")

        with measure("llm", log, budget):
            llm = build("llm", self._llm_cfg)
            reply = llm.chat(user_text, history=self.history)
            del llm
            gc.collect()
            _free_gpu_cache()
        print(f"💬  LLM  →  {reply!r}")
        self.history.append((user_text, reply))

        with measure("tts", log, budget):
            tts = build("tts", self._tts_cfg)
            audio_out = tts.synthesize(reply)
            del tts
            gc.collect()
            _free_gpu_cache()
        duration_s = audio_out["samples"].shape[0] / audio_out["sample_rate"]
        print(f"🔊  TTS  →  {duration_s:.2f}s audio @ {audio_out['sample_rate']}Hz")

        return user_text, reply, audio_out

    def reset_history(self) -> None:
        self.history.clear()
