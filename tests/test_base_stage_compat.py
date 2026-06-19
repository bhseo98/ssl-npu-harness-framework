"""Gap B compat tests — voice-app stages 가 main 의 BaseStage marker 도 만족.

설계 검토 논의 §5 "모든 작업들이 딱 하나로 묶여서 스무스하게 하나의
결과로" 의 정량 검증.

검증 항목:
1. BaseSTT/LLM/TTS 가 BaseStage 의 subclass (정적)
2. concrete WhisperSTT 같은 stage 의 instance 도 isinstance(BaseStage)
3. ``__call__(payload)`` 이 ``transcribe`` / ``chat`` / ``synthesize`` 와 동일 결과
4. ``@register`` + ``build`` 가 BaseStage 인스턴스 반환 — torch-mlir-zoo 의 generic
   ``Pipeline`` 과 plug-in 호환
"""
from __future__ import annotations

import numpy as np
import pytest

from npu_harness_framework import (
    BaseLLM,
    BaseStage,
    BaseSTT,
    BaseTTS,
    build,
    register,
)
from npu_harness_framework.interfaces import AudioBuffer


# --- Dummy concrete stages -------------------------------------------------

class _DummySTT(BaseSTT):
    def transcribe(self, audio):
        return "hello"


class _DummyLLM(BaseLLM):
    def chat(self, user_message, history=None):
        n = len(history or [])
        return f"reply to '{user_message}' (history={n})"


class _DummyTTS(BaseTTS):
    def synthesize(self, text):
        return AudioBuffer(samples=np.zeros(8, dtype=np.float32), sample_rate=16000)


# --- 1. ABC subclass relationship -----------------------------------------

def test_baseSTT_is_subclass_of_baseStage():
    assert issubclass(BaseSTT, BaseStage)


def test_baseLLM_is_subclass_of_baseStage():
    assert issubclass(BaseLLM, BaseStage)


def test_baseTTS_is_subclass_of_baseStage():
    assert issubclass(BaseTTS, BaseStage)


# --- 2. Concrete instance also isinstance(BaseStage) -----------------------

@pytest.mark.parametrize("cls", [_DummySTT, _DummyLLM, _DummyTTS])
def test_concrete_instance_is_basestage(cls):
    obj = cls()
    assert isinstance(obj, BaseStage), f"{cls.__name__} instance must be a BaseStage"


# --- 3. __call__ dispatches to native method -------------------------------

def test_stt_call_dispatches_to_transcribe():
    stt = _DummySTT()
    assert stt("any-audio") == stt.transcribe("any-audio") == "hello"


def test_llm_call_string_dispatches_to_chat_no_history():
    llm = _DummyLLM()
    assert llm("hi") == llm.chat("hi", history=None)


def test_llm_call_dict_dispatches_to_chat_with_history():
    llm = _DummyLLM()
    history = [("prev", "ans")]
    out = llm({"message": "hi", "history": history})
    assert out == llm.chat("hi", history=history)
    assert "(history=1)" in out


def test_tts_call_dispatches_to_synthesize():
    tts = _DummyTTS()
    buf = tts("hello")
    assert buf["sample_rate"] == 16000
    assert buf["samples"].shape == (8,)


# --- 4. registry + build still returns BaseStage instances -----------------

def test_registry_build_returns_basestage_instance():
    @register("stt", "_compat_dummy")
    class _R(_DummySTT):
        pass

    instance = build("stt", {"type": "_compat_dummy"})
    assert isinstance(instance, BaseStage)
    assert isinstance(instance, BaseSTT)
    assert instance("anything") == "hello"
