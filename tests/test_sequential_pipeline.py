"""SequentialVoicePipeline tests — per-stage load/run/unload 동작 검증.

설계 검토 §2 의 *"2GB 넘지 않는 모델"* 을 *동시 load peak* 가 아닌
*stage peak* 로 만족시키는 모드. Dummy stage 로 동작 검증 (real model 다운로드 X).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from npu_harness_framework import BaseLLM, BaseSTT, BaseTTS, register
from npu_harness_framework.interfaces import AudioBuffer
from npu_harness_framework.pipeline import SequentialVoicePipeline


_CALLS: list[str] = []


def setup_module(_):
    _CALLS.clear()


@register("stt", "_seq_dummy")
class _SeqSTT(BaseSTT):
    def __init__(self):
        _CALLS.append("stt-init")

    def transcribe(self, audio):
        _CALLS.append("stt-call")
        return "hello"


@register("llm", "_seq_dummy")
class _SeqLLM(BaseLLM):
    def __init__(self):
        _CALLS.append("llm-init")

    def chat(self, user_message, history=None):
        _CALLS.append("llm-call")
        return f"reply to {user_message}"


@register("tts", "_seq_dummy")
class _SeqTTS(BaseTTS):
    def __init__(self):
        _CALLS.append("tts-init")

    def synthesize(self, text):
        _CALLS.append("tts-call")
        return AudioBuffer(samples=np.zeros(8, dtype=np.float32), sample_rate=16000)


_CFG = {"type": "_seq_dummy"}


def _new_pipeline(tmp_path):
    return SequentialVoicePipeline(
        stt_config=dict(_CFG),
        llm_config=dict(_CFG),
        tts_config=dict(_CFG),
        log_path=str(tmp_path / "profile.jsonl"),
        budget_mb=2048,
    )


def test_sequential_run_returns_expected_tuple(tmp_path):
    p = _new_pipeline(tmp_path)
    user_text, reply, audio = p.run("dummy-audio")
    assert user_text == "hello"
    assert reply == "reply to hello"
    assert audio["sample_rate"] == 16000


def test_sequential_instantiates_each_stage_once_per_turn(tmp_path):
    _CALLS.clear()
    p = _new_pipeline(tmp_path)
    p.run("dummy-audio")
    # Each stage build → init → call, 한 turn 에 stage 별 1회 init.
    assert _CALLS.count("stt-init") == 1
    assert _CALLS.count("llm-init") == 1
    assert _CALLS.count("tts-init") == 1
    assert _CALLS.count("stt-call") == 1
    assert _CALLS.count("llm-call") == 1
    assert _CALLS.count("tts-call") == 1


def test_sequential_rebuilds_models_on_each_turn(tmp_path):
    """Per-turn re-instantiation 동작 — embedded budget mode 의 핵심."""
    _CALLS.clear()
    p = _new_pipeline(tmp_path)
    p.run("turn1")
    p.run("turn2")
    # 2 turn → 각 stage 2 회 init
    assert _CALLS.count("stt-init") == 2
    assert _CALLS.count("llm-init") == 2
    assert _CALLS.count("tts-init") == 2


def test_sequential_history_accumulates(tmp_path):
    p = _new_pipeline(tmp_path)
    p.run("first")
    p.run("second")
    assert len(p.history) == 2
    assert p.history[0] == ("hello", "reply to hello")


def test_sequential_reset_history(tmp_path):
    p = _new_pipeline(tmp_path)
    p.run("x")
    p.reset_history()
    assert p.history == []


def test_sequential_profiler_records_three_stages_per_turn(tmp_path):
    p = _new_pipeline(tmp_path)
    p.run("dummy")
    lines = (tmp_path / "profile.jsonl").read_text().splitlines()
    assert len(lines) == 3
    stages = [json.loads(line)["stage"] for line in lines]
    assert stages == ["stt", "llm", "tts"]
