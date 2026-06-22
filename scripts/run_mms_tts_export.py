#!/usr/bin/env python3
"""mms-tts (VITS) MLIR lowering — TTS 모달 확장 (STT·LLM에 이어 PDF target 3축).

`facebook/mms-tts-kor`(순수 VITS)는 그대로는 `torch.export`가 막힌다:
  - stochastic duration predictor의 rational-quadratic-spline data-dependent `if`
    (GuardOnDataDependentSymNode) + duration 의존 dynamic 출력 길이.
  - WaveNet `fused_add_tanh_sigmoid_multiply`의 텐서-스칼라 슬라이스 → iree.turbine 세그폴트.
whisper `WhisperForwardOnly` / llama `LlamaOnDevice` 와 같은 forward-only 재작성
(`models/vits_on_device.py`: duration→host 외부화 + noise 입력화 + 2개 export 패치)으로
encoder+flow+decoder 전체를 표준 aten 그래프로 내린다.

검증(실측): `server_side_op_hits == {}`, `dynamic == False` (whisper와 동일 성공 지표).
전처리(uroman 로마자화 + 토크나이저)는 그래프 밖(whisper mel 추출과 동급).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from torch_mlir_zoo.exporters import export_via_iree_turbine
from torch_mlir_zoo.analysis.ir_summary import summarize
from torch_mlir_zoo.models.vits_on_device import build_vits_on_device


def main() -> int:
    out_dir = ROOT / "logs" / "mms-tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = sys.argv[1] if len(sys.argv) > 1 else "facebook/mms-tts-kor"

    print(f"=== {model_id} (VitsOnDevice) ===")
    t0 = time.perf_counter()
    module, args, raw = build_vits_on_device(model_id)
    n_params = sum(p.numel() for p in raw.parameters())
    ids, attn, out_pad, noise = args
    print(f"  loaded+patched ({(time.perf_counter()-t0)*1000:.0f} ms) — {n_params:,} params, "
          f"ids={tuple(ids.shape)}, attn={tuple(attn.shape)}, L_out={out_pad.shape[-1]}")

    # eager sanity (waveform 정상?)
    import torch
    with torch.no_grad():
        wav = module(*args)
    print(f"  eager waveform={tuple(wav.shape)} finite={torch.isfinite(wav).all().item()} "
          f"dur={wav.shape[-1]/raw.config.sampling_rate:.2f}s")

    t1 = time.perf_counter()
    try:
        mlir = export_via_iree_turbine(module, args, func_name="forward")
    except Exception as e:
        print("  EXPORT FAIL:", type(e).__name__, str(e)[:400])
        print("\n".join(traceback.format_exc().splitlines()[-8:]))
        return 1
    export_ms = (time.perf_counter() - t1) * 1000

    (out_dir / "mms-tts-kor.mlir").write_text(mlir)
    s = summarize(mlir)
    print(f"  exported ({export_ms:.0f} ms) — {s['n_lines']} lines, "
          f"{sum(s['op_counts'].values())} aten ops, "
          f"srv_hits={sum(s['server_side_op_hits'].values())}, dynamic={s['has_dynamic_dim']}")

    top = dict(sorted(s["op_counts"].items(), key=lambda kv: -kv[1])[:25])
    # VITS 고유 surface (Conv1d 스택·gated act·HiFiGAN)
    interesting = {k: s["op_counts"].get(k, 0) for k in [
        "convolution", "conv1d", "_convolution",
        "leaky_relu", "tanh", "sigmoid", "bmm", "matmul", "mm",
        "constant_pad_nd", "slice", "cat", "split_with_sizes",
    ] if s["op_counts"].get(k, 0)}

    result = {
        "model_id": model_id, "status": "ok", "n_params": n_params,
        "input_ids": list(ids.shape), "attn": list(attn.shape), "L_out": out_pad.shape[-1],
        "mlir_lines": s["n_lines"], "module_name": s["module_name"],
        "n_aten_ops": sum(s["op_counts"].values()),
        "unique_aten_ops": len(s["op_counts"]),
        "top_ops": top, "vits_surface_ops": interesting,
        "dtypes": s["dtypes"], "has_dynamic_dim": s["has_dynamic_dim"],
        "server_side_op_hits": s["server_side_op_hits"],
        "export_ms": round(export_ms, 1),
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n  -- VITS surface ops --")
    for k, v in interesting.items():
        print(f"     {k:<24} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
