#!/usr/bin/env python3
"""M1 verification — does the block_scaled_q8 *microkernel* actually run?

Unlike the eager path (which uses the pure-torch ``eager_execute`` reference),
this compiles the generated MLIR through IREE (llvm-cpu) and *executes* it, then
compares the real output against the torch reference. This is the "does it
actually work" gate, not just string-matching the IR.

Run inside venv-shark:
    python scripts/verify_block_scaled_q8_iree.py
"""
from __future__ import annotations

import numpy as np
import torch

from torch_mlir_zoo.kernels import quantize_block_scaled_q8, quantized_matmul

B, M, K, N, BS = 1, 8, 128, 32, 32


class Mod(torch.nn.Module):
    def __init__(self, d, qs):
        super().__init__()
        self.register_buffer("d", d)
        self.register_buffer("qs", qs)

    def forward(self, a):
        return quantized_matmul(a, self.d, self.qs)


def main() -> int:
    torch.manual_seed(0)
    weight = torch.randn(N, K)
    qs, d = quantize_block_scaled_q8(weight, BS)
    a = torch.randn(B, M, K)

    # --- compile the microkernel through IREE (llvm-cpu) ---
    from iree.turbine.aot import FxProgramsBuilder, export

    fxb = FxProgramsBuilder(Mod(d, qs))

    @fxb.export_program(name="forward", args=(a,), strict=False)
    def _entry(model, *args):
        return model(*args)

    output = export(fxb)
    vmfb = bytes(output.compile(save_to=None, target_backends=("llvm-cpu",)).map_memory())
    print(f"compiled vmfb: {len(vmfb)} bytes (llvm-cpu)")

    # --- run it ---
    import iree.runtime as rt

    config = rt.Config("local-task")
    vm_module = rt.VmModule.copy_buffer(config.vm_instance, vmfb)
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    out_iree = np.asarray(ctx.modules.module.forward(a.numpy())).astype(np.float32)

    # --- references ---
    out_eager = quantized_matmul(a, d, qs).numpy()  # pure-torch reference
    out_float = (a @ weight.transpose(-1, -2)).numpy()  # fp32 (no quant)

    d_eager = float(np.abs(out_iree - out_eager).max())
    rel_eager = float(np.linalg.norm(out_iree - out_eager) / np.linalg.norm(out_eager))
    rel_float = float(np.linalg.norm(out_iree - out_float) / np.linalg.norm(out_float))

    print(f"out shape: iree={out_iree.shape} eager={out_eager.shape}")
    print(f"iree  [0,0,:5]: {np.round(out_iree[0,0,:5], 4)}")
    print(f"eager [0,0,:5]: {np.round(out_eager[0,0,:5], 4)}")
    print(f"float [0,0,:5]: {np.round(out_float[0,0,:5], 4)}")
    print(f"max|iree-eager| = {d_eager:.3e}   rel(iree,eager) = {rel_eager:.3e}")
    print(f"rel(iree,float) = {rel_float:.3e}  (quantization error vs fp32)")

    ok = d_eager < 1e-4 and rel_float < 0.05
    print(f"\nM1 {'PASS' if ok else 'FAIL'}: microkernel executes on IREE-CPU and "
          f"matches the torch reference (and tracks fp32 within quant error).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
