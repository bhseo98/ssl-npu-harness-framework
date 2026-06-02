"""Abstract base classes for pipeline stages.

The contract is intentionally tiny — one method per stage — so a new
model = one subclass + one registry call.

Gap B (설계 검토 §5 "하나로 묶이게") 적용 이후 voice-app 의
``BaseSTT``/``BaseLLM``/``BaseTTS`` 는 **main 의 ``BaseStage`` marker ABC 를
직접 상속**. 같은 `@register` registry / 같은 generic ``Pipeline`` 위에서
공존 가능하면서, application 코드 (transcribe/chat/synthesize 호출부) 는
변경 0. ``__call__`` 은 각 ABC 가 자기 메서드로 dispatch 하는 thin wrapper.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict, Union

import numpy as np


class AudioBuffer(TypedDict):
    """Mono float32 PCM audio, with its sample rate."""

    samples: np.ndarray  # shape (N,), dtype float32, range ~[-1, 1]
    sample_rate: int


AudioInput = Union[AudioBuffer, str, Path]
"""STT accepts either an in-memory buffer or a path to a .wav file."""


class BaseStage(ABC):
    """Marker ABC — modality-agnostic stage (matches main / vision-app / torch-mlir-zoo).

    A stage is anything callable that transforms a payload. Voice-app 의
    BaseSTT/BaseLLM/BaseTTS 가 이 ABC 를 상속하므로 voice stage 들이 그대로
    framework 의 generic ``Pipeline`` 위에서 동작한다.
    """

    @abstractmethod
    def __call__(self, payload: Any) -> Any:
        ...


class BaseSTT(BaseStage):
    """Speech-to-text stage. Implements ``__call__`` as a thin dispatcher
    to ``transcribe`` so VoicePipeline (legacy) 와 generic Pipeline (new) 둘 다 동작.
    """

    @abstractmethod
    def transcribe(self, audio: AudioInput) -> str:
        ...

    def __call__(self, audio: AudioInput) -> str:
        return self.transcribe(audio)


class BaseLLM(BaseStage):
    """LLM chat stage.

    ``__call__`` accepts either a bare string (single-turn, no history) or a
    dict ``{"message": ..., "history": [...]}``. VoicePipeline 은 그대로 ``chat()``
    를 호출하므로 application 코드 변경 0.
    """

    @abstractmethod
    def chat(
        self,
        user_message: str,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        ...

    def __call__(self, payload: Any) -> str:
        if isinstance(payload, dict):
            return self.chat(payload["message"], history=payload.get("history"))
        return self.chat(payload, history=None)


class BaseTTS(BaseStage):
    """Text-to-speech stage. ``__call__`` dispatches to ``synthesize``."""

    @abstractmethod
    def synthesize(self, text: str) -> AudioBuffer:
        ...

    def __call__(self, text: str) -> AudioBuffer:
        return self.synthesize(text)
