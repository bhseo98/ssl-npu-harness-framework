#!/usr/bin/env python3
"""Dump a (a, qs, d, out_ref) fixture for the Rust block_scaled_q8 parity test (RK1).

Writes raw little-endian binaries + a one-line ``shape.txt`` into
``rust/block_scaled_q8/tests/fixtures/<name>/`` so the Rust test reads them with no
parser dependency (footprint-minimal). The reference output is the registered
``block_scaled_q8`` op's eager path (pure-torch dequant+matmul).

Run inside venv-shark:
    python scripts/dump_block_scaled_q8_fixture.py
"""
from __future__ import annotations

from pathlib import Path

import torch

from torch_mlir_zoo.kernels import quantize_block_scaled_q8, quantized_matmul

FIX_DIR = Path("rust/block_scaled_q8/tests/fixtures")


def dump(name: str, B: int, M: int, K: int, N: int, BS: int) -> None:
    torch.manual_seed(0)
    a = torch.randn(B, M, K)
    qs, d = quantize_block_scaled_q8(torch.randn(N, K), BS)
    out = quantized_matmul(a, d, qs)  # registered op eager path -> [B, M, N]

    outdir = FIX_DIR / name
    outdir.mkdir(parents=True, exist_ok=True)
    a.contiguous().numpy().astype("<f4").tofile(outdir / "a.bin")
    qs.contiguous().numpy().astype("<i1").tofile(outdir / "qs.bin")
    d.contiguous().numpy().astype("<f4").tofile(outdir / "d.bin")
    out.contiguous().numpy().astype("<f4").tofile(outdir / "out.bin")
    (outdir / "shape.txt").write_text(f"{B} {M} {K} {N} {BS}\n")
    print(
        f"{name}: a{tuple(a.shape)} qs{tuple(qs.shape)} d{tuple(d.shape)} "
        f"out{tuple(out.shape)} -> {outdir}"
    )


if __name__ == "__main__":
    # Dominant whisper-tiny attention-projection shape (N=K=384, BS=32).
    dump("whisper_proj", 1, 4, 384, 384, 32)
