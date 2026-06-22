#!/usr/bin/env python3
"""Confirm INT8 quantization leaves embeddings / positional / tied weights FP32.

Intent: only pure ``nn.Linear`` weights are quantized to INT8 block-scaled. Token
embeddings, positional embeddings (both ``nn.Embedding``) and any *tied* Linear
(e.g. ``proj_out`` sharing the decoder ``embed_tokens`` table) stay FP32.

Quantizes whisper-tiny via ``quantize_linears_`` and reports exactly which modules
became INT8 (``BlockScaledQ8Linear``) and which stayed FP32, asserting the intent.

Run inside venv-shark:
    PYTHONPATH=src python scripts/verify_int8_fp32_preserved.py
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration

from torch_mlir_zoo.kernels import quantize_linears_
from torch_mlir_zoo.kernels.quantized_linear import BlockScaledQ8Linear


def main() -> int:
    m = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny", attn_implementation="eager"
    ).eval()

    # --- BEFORE ---
    tied = m.proj_out.weight.data_ptr() == m.model.decoder.embed_tokens.weight.data_ptr()
    n_linear_before = sum(1 for _, mod in m.named_modules() if isinstance(mod, nn.Linear))
    embs = [
        (n, tuple(mod.weight.shape), str(mod.weight.dtype))
        for n, mod in m.named_modules() if isinstance(mod, nn.Embedding)
    ]

    quantize_linears_(m)

    # --- AFTER ---
    q8 = [n for n, mod in m.named_modules() if isinstance(mod, BlockScaledQ8Linear)]
    still_fp32_linear = [n for n, mod in m.named_modules() if isinstance(mod, nn.Linear)]
    embs_after = [
        (n, str(mod.weight.dtype))
        for n, mod in m.named_modules() if isinstance(mod, nn.Embedding)
    ]

    print("=" * 70)
    print(f"proj_out tied to decoder.embed_tokens? {tied}")
    print(f"nn.Linear before quantize: {n_linear_before}")
    print(f"  -> INT8 (BlockScaledQ8Linear): {len(q8)}")
    print(f"  -> stayed FP32 nn.Linear (skipped): {len(still_fp32_linear)}")
    print("-" * 70)
    print("EMBEDDINGS (must stay FP32 nn.Embedding) — before:")
    for n, shp, dt in embs:
        print(f"    {n:45s} {str(shp):20s} {dt}")
    print("EMBEDDINGS after quantize (dtype):")
    for n, dt in embs_after:
        print(f"    {n:45s} {dt}")
    print("-" * 70)
    print("Skipped FP32 nn.Linear (tied / non-divisible):")
    modules = dict(m.named_modules())
    for n in still_fp32_linear:
        mod = modules[n]
        print(f"    {n:45s} in={mod.in_features} out={mod.out_features} dtype={mod.weight.dtype}")
    print("=" * 70)

    # assertions encoding the intent
    assert all(dt == "torch.float32" for _, dt in embs_after), "embedding not FP32!"
    if tied:
        assert isinstance(m.proj_out, nn.Linear) and m.proj_out.weight.dtype == torch.float32, \
            "tied proj_out got quantized!"
    print("OK — embeddings/positional FP32 preserved; tied proj_out FP32 preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
