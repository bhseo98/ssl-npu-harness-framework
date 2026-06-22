"""Model-side API for the SSLNPU block-scaled INT8 matmul kernel.

Three ways a model calls the kernel (guideline §4):

  1. direct:  ``block_scaled_q8(a, d, qs)``        — the registered op
  2. helper:  ``quantized_matmul(a, d, qs)``       — named wrapper
  3. module:  ``BlockScaledQ8Linear``              — ``nn.Linear`` drop-in

Plus :func:`quantize_block_scaled_q8`, a symmetric per-block Q8_0 quantizer for
building ``(qs, d)`` from a float weight (used by ``from_linear`` and tests).
"""
from __future__ import annotations

import torch
from torch import nn

from .block_scaled_q8 import block_scaled_q8

__all__ = [
    "quantize_block_scaled_q8",
    "quantized_matmul",
    "BlockScaledQ8Linear",
    "quantize_linears_",
]


def quantize_block_scaled_q8(weight: torch.Tensor, block_size: int = 32):
    """Symmetric per-block Q8_0 quantization of a ``[N, K]`` weight.

    Returns ``(qs, d)`` with ``qs`` int8 ``[N, G, BS]`` and ``d`` float
    ``[N, G, 1]``, where ``K == G * BS``. Levels are ``[-127, 127]`` (Q8_0).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2d [N,K], got {tuple(weight.shape)}")
    n, k = weight.shape
    if k % block_size != 0:
        raise ValueError(f"K={k} not divisible by block_size={block_size}")
    g = k // block_size
    w = weight.reshape(n, g, block_size)
    amax = w.abs().amax(dim=-1, keepdim=True)
    d = (amax / 127.0).clamp(min=1e-12)
    qs = torch.round(w / d).clamp(-127, 127).to(torch.int8)
    return qs, d.to(weight.dtype)


def quantized_matmul(a: torch.Tensor, d: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
    """``a[B,M,K] @ dequant(qs, d)^T -> [B,M,N]`` via the SSLNPU kernel."""
    return block_scaled_q8(a, d, qs)


class BlockScaledQ8Linear(nn.Module):
    """``nn.Linear`` drop-in backed by INT8 block-scaled weight (optional bias).

    The weight is stored as int8 ``qs`` + per-block float scale ``d`` and never
    materialized as a full fp tensor at inference; the kernel fuses dequant into
    the matmul. Accepts inputs of any rank ``[*, K]`` — leading dims are folded
    into the M axis to satisfy the kernel's ``[B, M, K]`` contract.
    """

    def __init__(
        self, in_features: int, out_features: int, block_size: int = 32, bias: bool = False
    ):
        super().__init__()
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by block_size={block_size}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        g = in_features // block_size
        self.register_buffer(
            "qs", torch.zeros(out_features, g, block_size, dtype=torch.int8)
        )
        self.register_buffer("d", torch.ones(out_features, g, 1, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead, k = x.shape[:-1], x.shape[-1]
        out = quantized_matmul(x.reshape(1, -1, k), self.d, self.qs)  # [1, prod(lead), N]
        out = out.reshape(*lead, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = 32) -> "BlockScaledQ8Linear":
        """Build from an existing ``nn.Linear`` (quantizes the weight; keeps bias)."""
        mod = cls(
            linear.in_features,
            linear.out_features,
            block_size,
            bias=linear.bias is not None,
        )
        qs, d = quantize_block_scaled_q8(linear.weight.detach(), block_size)
        mod.qs.copy_(qs)
        mod.d.copy_(d)
        if linear.bias is not None:
            mod.bias.copy_(linear.bias.detach())
        return mod


def _tied_weight_ptrs(module: nn.Module) -> set[int]:
    """Data pointers of weights shared by more than one submodule.

    A tied weight (e.g. ``lm_head``/``proj_out`` sharing the decoder
    ``embed_tokens`` table) appears once per holder when duplicates are kept.
    """
    from collections import Counter

    seen = Counter(
        p.data_ptr() for _, p in module.named_parameters(remove_duplicate=False)
    )
    return {ptr for ptr, count in seen.items() if count > 1}


def quantize_linears_(
    module: nn.Module, block_size: int = 32, _tied: set[int] | None = None
) -> nn.Module:
    """In-place: replace every quantizable ``nn.Linear`` (``K %% block_size == 0``)
    in ``module`` with :class:`BlockScaledQ8Linear`. Returns ``module`` for chaining.

    This is the model-side "apply to the framework" hook — drop any PyTorch model
    in and its Linear layers run on the SSLNPU block-scaled INT8 kernel.

    Linears whose weight is *tied* to another module (e.g. a ``proj_out``/``lm_head``
    sharing the fp32 ``embed_tokens`` table) are skipped: quantizing one half of a
    tie breaks it and stores a redundant INT8 copy of a weight meant to stay full
    precision. ``_tied`` is computed once on the top-level call.
    """
    if _tied is None:
        _tied = _tied_weight_ptrs(module)
    for name, child in list(module.named_children()):
        if (
            isinstance(child, nn.Linear)
            and child.in_features % block_size == 0
            and child.weight.data_ptr() not in _tied
        ):
            setattr(module, name, BlockScaledQ8Linear.from_linear(child, block_size))
        else:
            quantize_linears_(child, block_size, _tied)
    return module
