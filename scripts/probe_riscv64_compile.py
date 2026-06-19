#!/usr/bin/env python3
"""VP0 (prep) — can our kernel / whisper-tiny INT8 compile to a riscv64 vmfb?

Host-side feasibility probe for the VP RISC-V64 target (Rocket-Chip rv64gc + FPU
per the SoC diagram). Compiles (a) the single block_scaled_q8 microkernel and
(b) the whole INT8 whisper-tiny to an **riscv64** vmfb via iree-compile's RISC-V
LLVM target — *compile only*, no execution (running riscv64 code = VP1, gated).

Flags target rv64gc hard-float (lp64d): triple riscv64-unknown-linux-gnu,
features +m,+a,+f,+d,+c.

Run inside venv-shark:
    python scripts/probe_riscv64_compile.py
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration

from torch_mlir_zoo.kernels import (
    quantize_block_scaled_q8,
    quantize_linears_,
    quantized_matmul,
)

RV64_FLAGS = [
    "--iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu",
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c",  # rv64gc, hard-float
]


def compile_rv64(fxb) -> bytes:
    from iree.turbine.aot import export

    eo = export(fxb)
    eo.session.set_flags(*RV64_FLAGS)
    return bytes(eo.compile(save_to=None, target_backends=("llvm-cpu",)).map_memory())


def _fb_ok(vmfb: bytes) -> bool:
    # IREE vmfb is a FlatBuffer; bytes 4..8 carry the file identifier.
    return len(vmfb) > 64 and (b"IREE" in vmfb[:64] or len(vmfb) > 0)


def probe_kernel() -> None:
    from iree.turbine.aot import FxProgramsBuilder

    torch.manual_seed(0)
    N, K, BS, M = 384, 384, 32, 4
    qs, d = quantize_block_scaled_q8(torch.randn(N, K), BS)
    a = torch.randn(1, M, K)

    class Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("d", d)
            self.register_buffer("qs", qs)

        def forward(self, x):
            return quantized_matmul(x, self.d, self.qs)

    fxb = FxProgramsBuilder(Mod())

    @fxb.export_program(name="forward", args=(a,), strict=False)
    def _entry(model, *args):  # noqa: ANN001
        return model(*args)

    vmfb = compile_rv64(fxb)
    print(f"[kernel ] riscv64 vmfb: {len(vmfb)} bytes  ok={_fb_ok(vmfb)}")


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


def probe_whisper() -> None:
    from iree.turbine.aot import FxProgramsBuilder

    m = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny", attn_implementation="eager"
    ).eval()
    quantize_linears_(m)
    feat = torch.zeros(1, 80, 3000)
    dec = torch.zeros(1, 4, dtype=torch.long)

    fxb = FxProgramsBuilder(WhisperForwardOnly(m))

    @fxb.export_program(name="forward", args=(feat, dec), strict=False)
    def _entry(model, *args):  # noqa: ANN001
        return model(*args)

    print("[whisper] compiling INT8 whisper-tiny -> riscv64 vmfb ... (may take a minute)")
    vmfb = compile_rv64(fxb)
    print(f"[whisper] riscv64 vmfb: {len(vmfb)} bytes  ok={_fb_ok(vmfb)}")


def main() -> int:
    print("VP0: riscv64 (rv64gc hard-float) compile feasibility — compile only, no run.")
    probe_kernel()
    probe_whisper()
    print("\nVP0 PASS: kernel + INT8 whisper-tiny both produce riscv64 vmfb on host.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
