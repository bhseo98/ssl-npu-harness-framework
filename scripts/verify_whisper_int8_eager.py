#!/usr/bin/env python3
"""WM1 — quantize openai/whisper-tiny to INT8 block-scaled and check eager parity.

Swaps every quantizable ``nn.Linear`` in whisper-tiny for ``BlockScaledQ8Linear``
(via ``quantize_linears_``) and compares the INT8 model's forward logits against
the FP32 model on identical inputs. This is the "does INT8 Whisper still compute
the same thing" gate — real output numbers, not string-matching.

Run inside venv-shark:
    python scripts/verify_whisper_int8_eager.py
"""
from __future__ import annotations

import copy

import torch
from transformers import WhisperForConditionalGeneration

from torch_mlir_zoo.kernels import quantize_linears_


def main() -> int:
    torch.manual_seed(0)
    fp32 = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval()
    int8 = quantize_linears_(copy.deepcopy(fp32).eval())

    n_q = sum(1 for m in int8.modules() if type(m).__name__ == "BlockScaledQ8Linear")
    print(f"quantized Linear layers: {n_q}")

    feat = torch.randn(1, 80, 3000)  # mel features (30s)
    dec = torch.tensor([[50258, 50259, 50359, 50363]])  # whisper prompt tokens

    with torch.no_grad():
        kw = dict(input_features=feat, decoder_input_ids=dec, use_cache=False, return_dict=False)
        lf = fp32(**kw)[0]  # [1, T, vocab]
        lq = int8(**kw)[0]

    rel = (lq - lf).norm() / lf.norm()
    agree = (lq.argmax(-1) == lf.argmax(-1)).float().mean()
    top5_f = lf[0, -1].topk(5).indices.tolist()
    top5_q = lq[0, -1].topk(5).indices.tolist()

    print(f"logits shape: {tuple(lf.shape)}")
    print(f"rel error (INT8 vs FP32 logits): {rel:.4e}")
    print(f"argmax token agreement:          {agree:.1%}")
    print(f"last-pos top5 tokens  FP32: {top5_f}")
    print(f"last-pos top5 tokens  INT8: {top5_q}")

    ok = rel < 0.05 and agree > 0.95
    print(f"\nWM1 {'PASS' if ok else 'FAIL'}: INT8 whisper-tiny matches FP32 within quant error.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
