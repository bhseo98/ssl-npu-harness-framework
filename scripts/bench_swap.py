#!/usr/bin/env python3
"""Swap 비교 벤치마크 — Qwen+espnet (baseline) vs Llama+Piper (new).

설계 요점
---------
- 각 config 를 **별도 subprocess** 로 실행 (모델 잔여 메모리가 다음 config 의
  peak 측정을 오염시키지 않도록). parent 는 child 의 JSONL 출력을 모아 집계.
- STT 입력은 mic 없는 환경 가정 → **gTTS 로 합성한 깨끗한 ko wav** 사용. 따라서
  STT 인식률 자체는 제어된 상한값에 가깝다 (변동의 주된 요인은 LLM/TTS 쪽).
- N=5 회 반복, 첫 회는 warm-up 으로 집계 제외.
- pipeline 은 *VoicePipeline* (eager) 사용 — Phase 1 default 와 동일.

사용
----
    python scripts/bench_swap.py \\
        --configs configs/variants/swap-baseline-qwen-espnet.yaml \\
                  configs/variants/swap-new-llama-piper.yaml \\
        --runs 5 --warmup 1 \\
        --out logs/bench-swap/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# 한국어 테스트 문장 — 짧고 흔한 의도 3개.
DEFAULT_SENTENCES = [
    "오늘 서울 날씨 어때?",
    "회의 일정 좀 알려줘.",
    "가까운 카페 추천해줘.",
]


# ============================================================================
# Child entry point — load one config, run N times, dump JSONL.
# ============================================================================

def _child_main(argv: list[str]) -> int:
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wavs", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", required=True, help="JSONL output path for raw measurements")
    args = parser.parse_args(argv)

    # Defer heavy imports.
    from npu_harness_framework import stt as _stt, llm as _llm, tts as _tts  # noqa: F401  (registers @register)
    from npu_harness_framework.pipeline import VoicePipeline
    from npu_harness_framework.profiler import measure
    from npu_harness_framework.registry import build

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_path = Path(args.out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)

    # --- Build (cold load) ---
    t_load = time.perf_counter()
    stt = build("stt", cfg["stt"])
    llm = build("llm", cfg["llm"])
    tts = build("tts", cfg["tts"])
    load_ms = (time.perf_counter() - t_load) * 1000.0

    # Identity dump — verify which model is actually loaded.
    identity = {
        "stage": "identity",
        "stt_type": cfg["stt"].get("type"),
        "stt_model": cfg["stt"].get("model"),
        "llm_type": cfg["llm"].get("type"),
        "llm_model": cfg["llm"].get("model"),
        "tts_type": cfg["tts"].get("type"),
        "tts_model": cfg["tts"].get("model") or cfg["tts"].get("voice"),
        "cold_load_ms": round(load_ms, 1),
    }
    with log_path.open("a") as f:
        f.write(json.dumps(identity) + "\n")
    print(f"[identity] {json.dumps(identity, ensure_ascii=False)}", file=sys.stderr)

    pipeline = VoicePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        profiler_enabled=True,
        log_path=str(log_path),
        budget_mb=cfg["pipeline"].get("budget_mb"),
    )

    # --- Iterate runs ---
    for i in range(args.warmup + args.runs):
        is_warm = i < args.warmup
        for j, wav in enumerate(args.wavs):
            run_idx = i - args.warmup
            tag = "warmup" if is_warm else f"run{run_idx}"
            print(f"\n>>> {tag} sentence#{j}  ({wav})", file=sys.stderr)
            t_e2e = time.perf_counter()
            user_text, reply, audio = pipeline.run(wav)
            e2e_ms = (time.perf_counter() - t_e2e) * 1000.0
            with log_path.open("a") as f:
                f.write(json.dumps({
                    "stage": "e2e",
                    "tag": tag,
                    "warmup": is_warm,
                    "sentence_idx": j,
                    "wav": wav,
                    "stt_text": user_text,
                    "llm_reply": reply,
                    "tts_samples": int(audio["samples"].shape[0]),
                    "tts_sample_rate": int(audio["sample_rate"]),
                    "tts_duration_s": round(audio["samples"].shape[0] / audio["sample_rate"], 3),
                    "elapsed_ms": round(e2e_ms, 1),
                }, ensure_ascii=False) + "\n")
            pipeline.reset_history()
    return 0


# ============================================================================
# Parent — generate wavs, fan out child subprocesses, aggregate.
# ============================================================================

def _ffmpeg_exe() -> str:
    """Prefer pip-bundled ffmpeg (imageio-ffmpeg); fall back to PATH."""
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _generate_wavs(sentences: list[str], out_dir: Path) -> list[str]:
    """Synthesize input wavs with gTTS — clean Korean voice, no mic needed."""
    from gtts import gTTS

    ffmpeg = _ffmpeg_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs: list[str] = []
    for i, text in enumerate(sentences):
        mp3_path = out_dir / f"input_{i:02d}.mp3"
        wav_path = out_dir / f"input_{i:02d}.wav"
        if not wav_path.exists():
            gTTS(text=text, lang="ko").save(str(mp3_path))
            # mp3 → 16k mono wav via ffmpeg.
            subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(mp3_path), "-ac", "1", "-ar", "16000", str(wav_path)],
                check=True,
            )
            mp3_path.unlink(missing_ok=True)
        wavs.append(str(wav_path))
        print(f"  input#{i}: {text!r} → {wav_path}")
    return wavs


def _agg(records: list[dict], stage: str) -> dict:
    """Mean / stdev / min / max for one stage. RAM/GPU optional (e2e records don't have them)."""
    selected = [r for r in records if r["stage"] == stage]
    xs = [r["elapsed_ms"] for r in selected]
    if not xs:
        return {"n": 0}
    out: dict = {
        "n": len(xs),
        "elapsed_ms": {
            "mean": round(statistics.mean(xs), 1),
            "stdev": round(statistics.stdev(xs), 1) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 1),
            "max": round(max(xs), 1),
        },
    }
    rams = [r["ram_after_mb"] for r in selected if "ram_after_mb" in r]
    if rams:
        out["ram_after_mb"] = {"mean": round(statistics.mean(rams), 1), "max": round(max(rams), 1)}
    gpus = [r["gpu_peak_mb"] for r in selected if "gpu_peak_mb" in r]
    if gpus:
        out["gpu_peak_mb"] = {"mean": round(statistics.mean(gpus), 1), "max": round(max(gpus), 1)}
    return out


def _run_child(config_path: Path, wavs: list[str], runs: int, warmup: int,
               log_path: Path) -> dict:
    """Spawn child to bench one config. Returns aggregated stats dict."""
    print(f"\n=== running config: {config_path}  →  log {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, __file__, "__child__",
        "--config", str(config_path),
        "--wavs", *wavs,
        "--runs", str(runs),
        "--warmup", str(warmup),
        "--out", str(log_path),
    ]
    env = dict(os.environ)
    # faster_whisper needs CUDA 12 cublas/cudnn; we install them via pip
    # (nvidia-cublas-cu12 / nvidia-cudnn-cu12). Inject LD_LIBRARY_PATH so the
    # CT2 runtime can find them — system CUDA is 13.x.
    extra_libs = []
    for pkg_path in (
        ".venv-piper/lib/python3.12/site-packages/nvidia/cublas/lib",
        ".venv-piper/lib/python3.12/site-packages/nvidia/cudnn/lib",
    ):
        abs_path = Path(pkg_path).resolve()
        if abs_path.exists():
            extra_libs.append(str(abs_path))
    if extra_libs:
        env["LD_LIBRARY_PATH"] = ":".join(extra_libs + [env.get("LD_LIBRARY_PATH", "")])
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"child failed for {config_path} (exit {proc.returncode})")

    # Load raw JSONL.
    records: list[dict] = []
    with log_path.open() as f:
        for line in f:
            records.append(json.loads(line))

    identity = next((r for r in records if r["stage"] == "identity"), {})
    # Exclude warmup from aggregation.
    e2e_warm = [r for r in records if r["stage"] == "e2e" and not r.get("warmup", False)]
    stage_records = [r for r in records if r["stage"] in ("stt", "llm", "tts")]
    # Tag stage records as warmup by position: first N stages = warmup runs.
    # Simpler: just aggregate all stage records (per-stage measure() doesn't carry the warmup tag).
    # The first (warmup * 3 * len(sentences)) records are warmup.
    n_sent = len({r["sentence_idx"] for r in e2e_warm})
    warmup_skip = warmup * 3 * (n_sent if n_sent else 1)
    stage_records_post = stage_records[warmup_skip:]

    return {
        "config": str(config_path),
        "identity": identity,
        "e2e_runs": len(e2e_warm),
        "e2e": _agg(e2e_warm, "e2e"),
        "stt": _agg(stage_records_post, "stt"),
        "llm": _agg(stage_records_post, "llm"),
        "tts": _agg(stage_records_post, "tts"),
        "raw_records": records,
    }


def _parent_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sentences", nargs="*", default=DEFAULT_SENTENCES)
    parser.add_argument("--wav-dir", default="logs/bench-swap/inputs")
    parser.add_argument("--out", default="logs/bench-swap/summary.json")
    args = parser.parse_args(argv)

    wavs = _generate_wavs(args.sentences, Path(args.wav_dir))

    summaries = []
    for cfg in args.configs:
        cfg_path = Path(cfg)
        log_path = Path("logs/bench-swap") / f"{cfg_path.stem}.jsonl"
        s = _run_child(cfg_path, wavs, args.runs, args.warmup, log_path)
        summaries.append(s)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Don't dump raw_records in summary (kept in per-config JSONL).
    light = []
    for s in summaries:
        c = dict(s)
        c.pop("raw_records", None)
        light.append(c)
    out_path.write_text(json.dumps({
        "sentences": args.sentences,
        "wavs": wavs,
        "runs": args.runs,
        "warmup": args.warmup,
        "configs": light,
    }, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_path}")
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "__child__":
        return _child_main(argv[1:])
    return _parent_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
