"""STT implementations.

Built-in:
  - `whisper`         (OpenAI Whisper, default tiny + Korean)
  - `faster_whisper`  (CTranslate2 backend, int8 on GPU/CPU)
Stubs (no heavy deps, useful for tests / framework swap demos):
  - `fixed_text`  — always returns a configured string
  - `buffer_info` — returns a short description of the input buffer
"""
from __future__ import annotations

from pathlib import Path

from .interfaces import AudioInput, BaseSTT
from .registry import register


@register("stt", "fixed_text")
class FixedTextSTT(BaseSTT):
    """Returns a hard-coded string regardless of input. Useful for swap demos
    and pipeline tests that don't want to load a real ASR model."""

    def __init__(self, text: str = "테스트 입력입니다"):
        self._text = text

    def transcribe(self, audio: AudioInput) -> str:
        return self._text


@register("stt", "buffer_info")
class BufferInfoSTT(BaseSTT):
    """Reports the shape of the audio buffer. Useful as a debug stage."""

    def transcribe(self, audio: AudioInput) -> str:
        if isinstance(audio, dict):
            sr = audio["sample_rate"]
            n = audio["samples"].shape[0]
            return f"{n / sr:.2f}s buffer @ {sr}Hz"
        return f"input: {type(audio).__name__}"


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


@register("stt", "whisper")
class WhisperSTT(BaseSTT):
    def __init__(
        self,
        model: str = "tiny",
        language: str = "ko",
        device: str = "auto",
    ):
        import whisper

        self._device = _resolve_device(device)
        self._language = language
        self._fp16 = self._device == "cuda"
        self._model = whisper.load_model(model, device=self._device)

    def transcribe(self, audio: AudioInput) -> str:
        if isinstance(audio, (str, Path)):
            data = str(audio)
        elif isinstance(audio, dict):
            data = audio["samples"]
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio)}")

        result = self._model.transcribe(
            data, language=self._language, fp16=self._fp16
        )
        return result["text"].strip()


@register("stt", "faster_whisper")
class FasterWhisperSTT(BaseSTT):
    def __init__(
        self,
        model: str = "small",
        language: str = "ko",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        from faster_whisper import WhisperModel

        self._device = _resolve_device(device)
        self._language = language
        if compute_type == "auto":
            compute_type = "int8_float16" if self._device == "cuda" else "int8"
        self._model = WhisperModel(model, device=self._device, compute_type=compute_type)

    def transcribe(self, audio: AudioInput) -> str:
        if isinstance(audio, (str, Path)):
            data = str(audio)
        elif isinstance(audio, dict):
            data = audio["samples"]
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio)}")

        segments, _info = self._model.transcribe(data, language=self._language)
        return "".join(seg.text for seg in segments).strip()
