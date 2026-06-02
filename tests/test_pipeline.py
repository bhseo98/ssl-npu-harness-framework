"""Smoke tests using Dummy stages — no heavy model loads, runs in seconds on CPU."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from npu_harness_framework.interfaces import AudioBuffer, BaseLLM, BaseSTT, BaseTTS
from npu_harness_framework.pipeline import VoicePipeline
from npu_harness_framework.registry import build, register, registered


class _DummySTT(BaseSTT):
    def transcribe(self, audio):
        return "안녕하세요"


class _DummyLLM(BaseLLM):
    def chat(self, user_message, history=None):
        prev = len(history or [])
        return f"reply-{prev}: {user_message}"


class _DummyTTS(BaseTTS):
    def synthesize(self, text):
        return AudioBuffer(
            samples=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
        )


def test_pipeline_end_to_end():
    p = VoicePipeline(_DummySTT(), _DummyLLM(), _DummyTTS(), profiler_enabled=False)
    user, reply, audio = p.run("dummy.wav")
    assert user == "안녕하세요"
    assert reply == "reply-0: 안녕하세요"
    assert audio["sample_rate"] == 16000
    assert audio["samples"].shape == (16000,)


def test_history_accumulates_across_turns():
    p = VoicePipeline(_DummySTT(), _DummyLLM(), _DummyTTS(), profiler_enabled=False)
    p.run("a.wav")
    p.run("b.wav")
    _, last_reply, _ = p.run("c.wav")
    assert len(p.history) == 3
    assert last_reply.startswith("reply-2:")


def test_reset_history():
    p = VoicePipeline(_DummySTT(), _DummyLLM(), _DummyTTS(), profiler_enabled=False)
    p.run("a.wav")
    p.reset_history()
    assert p.history == []


def test_registry_decorator_and_build():
    @register("stt", "_test_dummy")
    class _D(BaseSTT):
        def __init__(self, greeting="world"):
            self.greeting = greeting

        def transcribe(self, audio):
            return self.greeting

    obj = build("stt", {"type": "_test_dummy", "greeting": "hi"})
    assert obj.transcribe("x") == "hi"


def test_build_rejects_unknown_type():
    with pytest.raises(KeyError):
        build("stt", {"type": "does_not_exist"})


def test_build_rejects_missing_type():
    with pytest.raises(KeyError):
        build("stt", {"model": "foo"})


def test_build_does_not_mutate_caller_config():
    @register("llm", "_test_noop")
    class _N(BaseLLM):
        def chat(self, user_message, history=None):
            return ""

    cfg = {"type": "_test_noop"}
    build("llm", cfg)
    assert "type" in cfg  # caller's dict is untouched


def test_registered_returns_known_stages():
    out = registered()
    assert isinstance(out, dict)
    assert {"stt", "llm", "tts"}.issubset(out.keys())


def test_model_swap_via_config_changes_behavior_only():
    """Core framework promise: swapping a model = changing one YAML key.

    Same VoicePipeline class, same dummy LLM/TTS, two different STT `type`
    values → two different transcription strings, zero pipeline code change.
    """
    from npu_harness_framework import stt as _stt_module  # noqa: F401 (registers built-ins)

    audio = AudioBuffer(
        samples=np.zeros(16000, dtype=np.float32), sample_rate=16000
    )

    stt_fixed = build("stt", {"type": "fixed_text", "text": "안녕하세요"})
    stt_info = build("stt", {"type": "buffer_info"})

    p_fixed = VoicePipeline(stt_fixed, _DummyLLM(), _DummyTTS(), profiler_enabled=False)
    p_info = VoicePipeline(stt_info, _DummyLLM(), _DummyTTS(), profiler_enabled=False)

    text_fixed, _, _ = p_fixed.run(audio)
    text_info, _, _ = p_info.run(audio)

    assert text_fixed == "안녕하세요"
    assert text_info == "1.00s buffer @ 16000Hz"


def test_model_swap_same_type_different_kwargs():
    """Two instances of the same STT type with different config → different behavior.

    This is what `configs/default.yaml` does when you tweak hyperparameters
    (e.g. `whisper.model: tiny` vs `whisper.model: small`).
    """
    from npu_harness_framework import stt as _stt_module  # noqa: F401

    stt_a = build("stt", {"type": "fixed_text", "text": "first"})
    stt_b = build("stt", {"type": "fixed_text", "text": "second"})

    assert stt_a.transcribe("ignored") == "first"
    assert stt_b.transcribe("ignored") == "second"


def test_default_config_uses_registered_public_kss_tts_model():
    from npu_harness_framework import tts as _tts_module  # noqa: F401

    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["tts"]["type"] in registered("tts")
    assert cfg["tts"]["model"] != "kan-bayashi/kss_fastspeech2"
    assert cfg["tts"]["model"].startswith("imdanboy/kss_tts_train_jets_")
