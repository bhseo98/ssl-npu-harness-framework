#!/usr/bin/env python3
"""RK2 — footprint/latency benchmark: IREE-lowered microkernel vs Rust-native kernel.

For each whisper-tiny INT8 matmul shape, the *same* block_scaled_q8 kernel is run
two ways and measured for on-device cost:

  (i)  IREE path  — block_scaled_q8 compiled to a vmfb, executed by the **native**
       iree-benchmark-module (the standalone C runtime, NOT python — fair RSS).
  (ii) Rust path  — the dependency-free `bench` binary (rust/block_scaled_q8).

Metrics: peak RSS (`/usr/bin/time -v`), artifact size (vmfb + runtime vs rust bin),
median latency. Numeric parity is already established elsewhere — Rust↔torch (RK1,
rel 3.5e-7) and IREE↔torch (M1, 7.6e-6) — so the two paths compute the same kernel.

Run inside venv-shark (with ~/.cargo/bin built):
    python scripts/bench_kernel_iree_vs_rust.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import torch

from torch_mlir_zoo.kernels import quantize_block_scaled_q8, quantized_matmul

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "logs" / "bench-kernel"
RUST_BIN = REPO / "rust" / "block_scaled_q8" / "target" / "release" / "bench"

# whisper-tiny INT8 unique kernels (from WM2): (N, K). BS=32, M=4, f32.
SHAPES = [(384, 384), (1536, 384), (384, 1536), (51865, 384)]
BS, M, ITERS = 32, 4, 200


def native_tool(name: str) -> Path:
    import iree.runtime as rt

    libdir = Path(rt.__file__).resolve().parents[1] / "_runtime_libs"
    p = libdir / name
    if not p.exists():
        raise FileNotFoundError(f"native {name} not found at {p}")
    return p


def time_v(cmd: list[str]) -> tuple[str, int]:
    """Run cmd under /usr/bin/time -v; return (stdout, peak_rss_kib)."""
    proc = subprocess.run(
        ["/usr/bin/time", "-v", *cmd], capture_output=True, text=True
    )
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", proc.stderr)
    rss = int(m.group(1)) if m else -1
    return proc.stdout, rss


def compile_vmfb(n: int, k: int) -> Path:
    from iree.turbine.aot import FxProgramsBuilder, export

    torch.manual_seed(0)
    qs, d = quantize_block_scaled_q8(torch.randn(n, k), BS)
    a = torch.randn(1, M, k)

    class Mod(torch.nn.Module):
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

    vmfb = bytes(
        export(fxb).compile(save_to=None, target_backends=("llvm-cpu",)).map_memory()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"k_{n}_{k}.vmfb"
    path.write_bytes(vmfb)
    return path


def bench_iree(vmfb: Path, k: int) -> dict:
    tool = native_tool("iree-benchmark-module")
    cmd = [
        str(tool),
        f"--module={vmfb}",
        "--function=forward",
        f"--input=1x{M}x{k}xf32=0.5",
        "--device=local-task",
        "--benchmark_repetitions=5",
        "--benchmark_format=json",
    ]
    out, rss = time_v(cmd)
    median_ms = -1.0
    try:
        data = json.loads(out)
        # google-benchmark: pick the 'median' aggregate real_time (unit in 'time_unit').
        for b in data.get("benchmarks", []):
            if b.get("aggregate_name") == "median":
                unit = b.get("time_unit", "ns")
                t = float(b["real_time"])
                median_ms = t * {"ns": 1e-6, "us": 1e-3, "ms": 1.0}.get(unit, 1e-6)
                break
    except json.JSONDecodeError:
        pass
    return {"median_ms": median_ms, "peak_rss_kib": rss, "vmfb_bytes": vmfb.stat().st_size}


def bench_rust(n: int, k: int) -> dict:
    cmd = [str(RUST_BIN), str(n), str(k), str(BS), str(M), str(ITERS)]
    out, rss = time_v(cmd)
    median_ms = -1.0
    try:
        median_ms = float(json.loads(out.strip().splitlines()[-1])["median_ms"])
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return {"median_ms": median_ms, "peak_rss_kib": rss}


def main() -> int:
    if not RUST_BIN.exists():
        raise SystemExit(f"build the rust bench first: (cd rust/block_scaled_q8 && cargo build --release)\n  missing {RUST_BIN}")

    iree_rt_bytes = native_tool("iree-run-module").stat().st_size  # standalone C runtime
    rust_bin_bytes = RUST_BIN.stat().st_size

    rows = []
    for n, k in SHAPES:
        print(f"--- shape N={n} K={k} (BS={BS}, M={M}) ---")
        vmfb = compile_vmfb(n, k)
        ir = bench_iree(vmfb, k)
        rs = bench_rust(n, k)
        print(f"  IREE: {ir['median_ms']:.4f} ms | RSS {ir['peak_rss_kib']} KiB | vmfb {ir['vmfb_bytes']} B")
        print(f"  Rust: {rs['median_ms']:.4f} ms | RSS {rs['peak_rss_kib']} KiB")
        rows.append({"n": n, "k": k, "iree": ir, "rust": rs})

    summary = {
        "bs": BS, "m": M, "iters_rust": ITERS,
        "artifact_sizes_bytes": {
            "iree_runtime_iree_run_module": iree_rt_bytes,
            "rust_bench_binary": rust_bin_bytes,
        },
        "shapes": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # markdown table
    lines = [
        "# RK2 — kernel footprint/latency: IREE vs Rust",
        "",
        f"BS={BS}, M={M}, f32. IREE = native iree-benchmark-module (local-task, 5 reps median); "
        f"Rust = dependency-free naive scalar kernel ({ITERS} iters median). "
        "Peak RSS via `/usr/bin/time -v`.",
        "",
        f"**Artifact (runtime+code) size:** IREE runtime (`iree-run-module`) = {iree_rt_bytes/1024:.0f} KiB "
        f"+ per-shape vmfb;  Rust self-contained binary = {rust_bin_bytes/1024:.0f} KiB.",
        "",
        "| shape N×K | IREE ms | IREE RSS (MiB) | vmfb (KiB) | Rust ms | Rust RSS (MiB) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        ir, rs = r["iree"], r["rust"]
        lines.append(
            f"| {r['n']}×{r['k']} | {ir['median_ms']:.3f} | {ir['peak_rss_kib']/1024:.1f} | "
            f"{ir['vmfb_bytes']/1024:.0f} | {rs['median_ms']:.3f} | {rs['peak_rss_kib']/1024:.1f} |"
        )
    lines += [
        "",
        "Parity: Rust↔torch rel 3.5e-7 (RK1), IREE↔torch 7.6e-6 (M1) ⇒ same kernel both paths.",
        "Caveat: Rust kernel is naive scalar single-thread; IREE is vectorized/threaded — "
        "latency favours IREE, footprint favours Rust (no compiler-runtime needed).",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT/'summary.json'} and {OUT/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
