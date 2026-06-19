"""NPU Harness Framework — modular voice AI evaluation framework.

Pipeline stages (STT → LLM → TTS) are loosely coupled via a registry so a
swap is a one-line config change. Import the impl module of the stage you
need (e.g. `from npu_harness_framework import stt`) to register the built-in
implementations; tests don't pay that import cost.

Gap B (설계 검토 §5 "하나로 묶이게") 이후 ``BaseStage`` marker ABC 가
노출되어 main / vision-app / torch-mlir-zoo 의 `@register` + generic
``Pipeline`` 과 동일 패턴을 따른다. BaseSTT/BaseLLM/BaseTTS 가 모두
BaseStage 를 상속하므로 instance 가 ``isinstance(BaseStage)`` 자동 충족.
"""
from .interfaces import BaseLLM, BaseStage, BaseSTT, BaseTTS  # noqa: F401
from .profiler import measure  # noqa: F401
from .registry import build, register, registered  # noqa: F401

__version__ = "0.2.0"

__all__ = [
    "BaseStage",
    "BaseSTT",
    "BaseLLM",
    "BaseTTS",
    "measure",
    "build",
    "register",
    "registered",
]
