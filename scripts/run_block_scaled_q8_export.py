#!/usr/bin/env python3
"""Dump the SSLNPU block-scaled INT8 matmul kernel as MLIR + IR summary.

Wraps the ``block_scaled_q8`` CustomOp in a tiny forward-only module, exports it
via iree.turbine, and writes the MLIR and a summary JSON. The MLIR should carry
the spliced ``util.func @ssl_mmt_block_scaled_q8_*`` (standard linalg, no
``iree_linalg_ext``) — the portable form that the SSLNPU intrinsic later replaces.

Run inside venv-shark:
    python scripts/run_block_scaled_q8_export.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from torch_mlir_zoo.analysis import summarize
from torch_mlir_zoo.exporters import export_via_iree_turbine
from torch_mlir_zoo.kernels import quantize_block_scaled_q8, quantized_matmul

OUT_DIR = Path("logs/kernels")
B, M, K, N, BS = 1, 8, 128, 32, 32


class BlockScaledQ8Mod(torch.nn.Module):
    def __init__(self, d, qs):
        super().__init__()
        self.register_buffer("d", d)
        self.register_buffer("qs", qs)

    def forward(self, a):
        return quantized_matmul(a, self.d, self.qs)


def main() -> int:
    torch.manual_seed(0)
    qs, d = quantize_block_scaled_q8(torch.randn(N, K), BS)
    mlir = export_via_iree_turbine(BlockScaledQ8Mod(d, qs), (torch.randn(B, M, K),))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "block_scaled_q8.mlir").write_text(mlir)
    summary = summarize(mlir)
    (OUT_DIR / "block_scaled_q8.summary.json").write_text(json.dumps(summary, indent=2))

    spliced = "ssl_mmt_block_scaled_q8" in mlir
    portable = "iree_linalg_ext" not in mlir
    print(f"MLIR lines: {len(mlir.splitlines())}  -> {OUT_DIR/'block_scaled_q8.mlir'}")
    print(f"microkernel spliced: {spliced} | portable (no iree_linalg_ext): {portable}")
    print(f"server_side_op_hits: {summary['server_side_op_hits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
