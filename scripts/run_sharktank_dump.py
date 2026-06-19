#!/usr/bin/env python3
"""Step 3 — amdsharktank PagedMHAttention top-level MLIR dump (best-effort).

Why best-effort: amdsharktank is an *early preview* (no PyPI wheel; source
build requires iree-turbine + hip kernels + torch-mlir simultaneously). On
hosts without that toolchain, this script exits with a clear message so
the qualitative analysis in `docs/SHARK_AI_ANALYSIS.md` can stand on its
own. When the docker-shell session does have amdsharktank importable, it
attempts a single-layer dump (PagedMHAttention) so the server-side ops
(paged_attention, paged_kv_cache) show up in the IR summary and contrast
against the on-device Llama dump (Step 5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_MLIR = Path("analysis/sharktank_paged_attention.mlir")
OUT_JSON = Path("analysis/sharktank_paged_attention.summary.json")


def _exit(message: str, code: int = 1) -> int:
    print(message)
    return code


def main() -> int:
    try:
        import amdsharktank  # noqa: F401
    except ImportError:
        return _exit(
            "amdsharktank is not installed on this host.\n"
            "  - early-preview, no PyPI wheel yet (build from "
            "github.com/nod-ai/amd-shark-ai).\n"
            "  - the qualitative analysis in docs/SHARK_AI_ANALYSIS.md does\n"
            "    not require importing the package — it reads the source on\n"
            "    GitHub directly.\n"
            "  - on a docker-shell with amdsharktank built, re-run this\n"
            "    script to produce the artifacts referenced by Step 3."
        )

    try:
        from amdsharktank.layers.paged_attention import PagedMHAttention  # noqa: F401
    except ImportError as exc:
        return _exit(
            f"amdsharktank import succeeded but PagedMHAttention failed: {exc}\n"
            "Server-side layer layout may have shifted; pin a specific commit "
            "and update this script."
        )

    # Full instantiation requires Theta + ParallelismConfig + KVCache fixtures
    # which are themselves heavy to build outside docker. Keep this script
    # honest: if you get here, drop into the project's docker-shell where
    # those fixtures are available, then implement the trace below.
    OUT_MLIR.parent.mkdir(parents=True, exist_ok=True)
    OUT_MLIR.write_text(
        "// amdsharktank import OK. Full PagedMHAttention dump pending "
        "docker-shell fixtures (Theta + ParallelismConfig + KVCache).\n"
    )

    from torch_mlir_zoo.analysis import summarize

    OUT_JSON.write_text(json.dumps(summarize(OUT_MLIR.read_text()), indent=2))
    print(f"amdsharktank import OK. Placeholder written to {OUT_MLIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
