"""SSLNPU CustomOp kernels (design scaffold).

Backend-independent kernel layer for the SSLNPU path: the CustomOp registration,
shape contract (``select``), model-backend call functions, and a *portable*
standard-linalg microkernel. The SSLNPU-specific intrinsic microkernel is a
documented swap-in point pending the SSLNPU Library/Runtime handoff.

See ``docs/kernel-design/01-guideline-and-plan.md``.
"""
from .block_scaled_q8 import block_scaled_q8
from .quantized_linear import (
    BlockScaledQ8Linear,
    quantize_block_scaled_q8,
    quantize_linears_,
    quantized_matmul,
)

__all__ = [
    "block_scaled_q8",
    "quantized_matmul",
    "quantize_block_scaled_q8",
    "BlockScaledQ8Linear",
    "quantize_linears_",
]
