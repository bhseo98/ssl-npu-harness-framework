#!/usr/bin/env python3
"""WM3 — compile INT8 whisper-tiny to a vmfb and RUN it on IREE-CPU.

Extends M1 (single-kernel IREE-CPU execution) to the whole model: quantize every
``nn.Linear`` in whisper-tiny to INT8 block-scaled (``quantize_linears_``), export
+ compile the full forward to a vmfb (``llvm-cpu``), run it via ``iree.runtime`` on
a fixed ``(mel, decoder_ids)`` input, and compare the logits against

  (a) the INT8 *eager* model  -> pure-IREE numeric error (should be ~1e-4), and
  (b) the FP32 *eager* model  -> quantization error (should be <5%).

Real execution end-to-end, not string-matching the IR (that was WM2).

Run inside venv-shark:
    python scripts/verify_whisper_int8_iree.py
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from transformers import WhisperForConditionalGeneration

from torch_mlir_zoo.kernels import quantize_linears_


class WhisperForwardOnly(nn.Module):
    def __init__(self, hf):
        super().__init__()
        self.model = hf

    def forward(self, input_features, decoder_input_ids):
        return self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=False,
        )[0]


def main() -> int:
    torch.manual_seed(0)
    # eager attention so SDPA decomposes (easy7/WM2); quantize all Linears to INT8.
    fp32 = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny", attn_implementation="eager"
    ).eval()
    int8 = quantize_linears_(copy.deepcopy(fp32).eval())

    feat = torch.randn(1, 80, 3000)  # mel features (30s)
    dec = torch.tensor([[50258, 50259, 50359, 50363]])  # whisper prompt tokens

    # eager references (teacher-forced single forward).
    with torch.no_grad():
        kw = dict(use_cache=False, return_dict=False)
        out_fp32 = fp32(input_features=feat, decoder_input_ids=dec, **kw)[0]
        out_int8_eager = int8(input_features=feat, decoder_input_ids=dec, **kw)[0]

    # compile INT8 whisper to a vmfb and run on IREE-CPU.
    from iree.turbine.aot import FxProgramsBuilder, export
    import iree.runtime as rt

    fxb = FxProgramsBuilder(WhisperForwardOnly(int8))

    @fxb.export_program(name="forward", args=(feat, dec), strict=False)
    def _entry(model, *args):
        return model(*args)

    print("compiling INT8 whisper-tiny -> vmfb (llvm-cpu) ... (may take a minute)")
    vmfb = bytes(
        export(fxb).compile(save_to=None, target_backends=("llvm-cpu",)).map_memory()
    )
    print(f"vmfb bytes: {len(vmfb)}")

    config = rt.Config("local-task")
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(rt.VmModule.copy_buffer(config.vm_instance, vmfb))
    out_iree = np.asarray(
        ctx.modules.module.forward(feat.numpy(), dec.numpy())
    ).astype(np.float32)

    o_fp32 = out_fp32.numpy()
    o_int8 = out_int8_eager.numpy()
    rel_iree_vs_int8 = np.linalg.norm(out_iree - o_int8) / np.linalg.norm(o_int8)
    rel_iree_vs_fp32 = np.linalg.norm(out_iree - o_fp32) / np.linalg.norm(o_fp32)
    argmax_agree = (out_iree.argmax(-1) == o_fp32.argmax(-1)).mean()

    print(f"logits shape: {out_iree.shape}")
    print(f"rel (INT8-IREE vs INT8-eager): {rel_iree_vs_int8:.4e}  (pure-IREE numeric)")
    print(f"rel (INT8-IREE vs FP32-eager): {rel_iree_vs_fp32:.4e}  (quant error)")
    print(f"argmax token agreement (vs FP32): {argmax_agree:.1%}")

    ok = rel_iree_vs_fp32 < 0.05 and argmax_agree > 0.95
    print(
        f"\nWM3 {'PASS' if ok else 'FAIL'}: INT8 whisper-tiny compiles + runs on "
        f"IREE-CPU and matches FP32 within quant error."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
