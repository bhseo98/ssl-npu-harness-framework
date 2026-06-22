#!/usr/bin/env python3
"""WM2 — lower INT8 whisper-tiny through the framework and dump MLIR + summary.

Quantizes whisper-tiny's Linear layers to INT8 block-scaled (``quantize_linears_``)
then exports the whole model via iree.turbine. The MLIR should carry one
``util.call @ssl_mmt_block_scaled_q8_*`` per quantized Linear, with the distinct
kernels deduplicated by shape, no ``iree_linalg_ext`` and no opaque SDPA.

Run inside venv-shark:
    python scripts/run_whisper_int8_export.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration

from torch_mlir_zoo.analysis import summarize
from torch_mlir_zoo.exporters import export_via_iree_turbine
from torch_mlir_zoo.kernels import quantize_linears_

OUT_DIR = Path("logs/whisper-int8")


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
    # eager attention so SDPA decomposes (easy7); quantize all Linears to INT8.
    m = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny", attn_implementation="eager"
    ).eval()
    quantize_linears_(m)

    args = (torch.zeros(1, 80, 3000), torch.zeros(1, 8, dtype=torch.long))
    print("exporting INT8 whisper-tiny ...")
    mlir = export_via_iree_turbine(WhisperForwardOnly(m), args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "whisper-tiny-int8.mlir").write_text(mlir)
    summary = summarize(mlir)
    (OUT_DIR / "whisper-tiny-int8.summary.json").write_text(json.dumps(summary, indent=2))

    n_call = len(re.findall(r"util\.call @ssl_mmt_block_scaled_q8", mlir))
    n_func = len(re.findall(r"util\.func private @ssl_mmt_block_scaled_q8", mlir))
    print(f"MLIR lines: {len(mlir.splitlines())}  -> {OUT_DIR/'whisper-tiny-int8.mlir'}")
    print(f"block_scaled_q8 calls: {n_call} | unique kernels (dedup): {n_func}")
    print(f"iree_linalg_ext: {'iree_linalg_ext' in mlir} | "
          f"opaque SDPA: {'_scaled_dot_product_flash_attention' in mlir}")
    print(f"server_side_op_hits: {summary['server_side_op_hits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
