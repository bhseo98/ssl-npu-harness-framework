#!/usr/bin/env python3
"""Measure whisper-tiny INT8 vs FP32: vmfb size + forward latency (one variant).

Run once per variant under /usr/bin/time -v to also capture peak RSS:
    /usr/bin/time -v python scripts/bench_whisper_int8_vs_fp32.py --variant int8
    /usr/bin/time -v python scripts/bench_whisper_int8_vs_fp32.py --variant fp32

Accuracy (INT8 vs FP32 logits) is verified separately in WM3
(scripts/verify_whisper_int8_iree.py): rel 0.8%, argmax 100%.
"""
from __future__ import annotations

import argparse
import statistics
import time

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fp32", "int8"], required=True)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(0)
    m = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny", attn_implementation="eager"
    ).eval()
    if args.variant == "int8":
        quantize_linears_(m)

    feat = torch.randn(1, 80, 3000)
    dec = torch.tensor([[50258, 50259, 50359, 50363]])

    from iree.turbine.aot import FxProgramsBuilder, export
    import iree.runtime as rt

    fxb = FxProgramsBuilder(WhisperForwardOnly(m))

    @fxb.export_program(name="forward", args=(feat, dec), strict=False)
    def _entry(model, *a):  # noqa: ANN001
        return model(*a)

    print(f"[{args.variant}] compiling -> vmfb (llvm-cpu) ...")
    vmfb = bytes(
        export(fxb).compile(save_to=None, target_backends=("llvm-cpu",)).map_memory()
    )
    print(f"[{args.variant}] vmfb bytes: {len(vmfb):,} ({len(vmfb)/1e6:.1f} MB)")

    config = rt.Config("local-task")
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(rt.VmModule.copy_buffer(config.vm_instance, vmfb))
    fn = ctx.modules.module.forward
    fa, da = feat.numpy(), dec.numpy()

    fn(fa, da)  # warmup
    ts = []
    for _ in range(args.iters):
        t = time.perf_counter()
        out = fn(fa, da)
        ts.append((time.perf_counter() - t) * 1e3)
    _ = np.asarray(out)  # realize
    print(f"[{args.variant}] forward median: {statistics.median(ts):.1f} ms "
          f"(min {min(ts):.1f}, n={args.iters})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
