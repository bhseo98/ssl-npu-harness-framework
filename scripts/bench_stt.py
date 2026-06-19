#!/usr/bin/env python3
"""STT 후보 벤치마크 — Whisper tiny / base / small.

한국어 짧은 발화 wav 매니페스트를 입력으로, 각 (backend, model) 조합의
CER / WER / cold·warm latency / peak RSS / GPU peak 를 측정해 markdown 표로 출력.

설계 요점
---------
- 모델별로 별도 subprocess 에서 측정 (peak RSS 가 이전 모델의 잔여 메모리로
  오염되지 않도록). parent 가 child 의 stdout JSON 을 모아 표를 렌더링.
- 의존성 추가 없음. 이미 있는 psutil + openai-whisper + faster-whisper 사용.
- CER 은 코드포인트 단위, WER 은 공백 split. 한국어 짧은 발화에서 CER 이 primary.

사용
----
컨테이너에서 (recordings/, docs/ 마운트 추가):

    docker compose run --rm \\
        -v $(pwd)/recordings:/workspace/recordings \\
        -v $(pwd)/docs:/workspace/docs \\
        app python scripts/bench_stt.py \\
        --backend faster_whisper \\
        --models tiny,base,small \\
        --manifest configs/bench/stt_korean_short.yaml \\
        --warmup 1 --runs 3 \\
        --out docs/reports/2026-05-19-stt-bench.md

처음 실행 시 `whisper.load_model("base")` 등이 weight 를 다운로드 (~140MB for base).
GPU 미사용 시 small 은 클립당 30~60초가 걸릴 수 있음.
"""
from __future__ import annotations

import argparse
import json
import statistics
import string
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml


# --- text metrics ----------------------------------------------------------

def _edit_distance(a: list | str, b: list | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[-1]


_PUNCT = set(string.punctuation + "。、，．！？·…「」『』〈〉《》—–")


def _normalize(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace() and ch not in _PUNCT)


def cer(hyp: str, ref: str) -> float:
    ref_n = _normalize(ref)
    if not ref_n:
        return 0.0
    return _edit_distance(_normalize(hyp), ref_n) / len(ref_n)


def wer(hyp: str, ref: str) -> float:
    ref_toks = ref.split()
    if not ref_toks:
        return 0.0
    return _edit_distance(hyp.split(), ref_toks) / len(ref_toks)


# --- memory snapshot -------------------------------------------------------

def _gpu_peak_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return 0.0


def _gpu_reset() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# --- child: single (backend, model) measurement ----------------------------

def run_single(
    backend: str,
    model: str,
    samples: list[dict],
    runs: int,
    warmup: int,
    device: str,
) -> None:
    from npu_harness_framework.registry import build  # type: ignore

    cfg: dict = {"type": backend, "model": model, "device": device, "language": "ko"}
    if backend == "faster_whisper":
        cfg["compute_type"] = "auto"

    _gpu_reset()
    stt = build("stt", cfg)

    per_sample = []
    for s in samples:
        wav, ref = s["wav"], s.get("ref", "") or ""
        cold_ms = None
        warm_times = []
        last_hyp = ""
        for i in range(warmup + max(runs, 1)):
            t0 = time.perf_counter()
            hyp = stt.transcribe(wav)
            elapsed = (time.perf_counter() - t0) * 1000.0
            if i == 0:
                cold_ms = elapsed
            if i >= warmup:
                warm_times.append(elapsed)
            last_hyp = hyp

        warm_median = statistics.median(warm_times) if warm_times else cold_ms

        entry = {
            "wav": wav,
            "ref": ref,
            "hyp": last_hyp,
            "cold_ms": round(cold_ms or 0.0, 1),
            "warm_ms": round(warm_median or 0.0, 1),
        }
        if ref.strip():
            entry["cer"] = round(cer(last_hyp, ref), 4)
            entry["wer"] = round(wer(last_hyp, ref), 4)
        else:
            entry["cer"] = None
            entry["wer"] = None
        per_sample.append(entry)

    rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
    gpu_peak = _gpu_peak_mb()

    cers = [s["cer"] for s in per_sample if s["cer"] is not None]
    wers = [s["wer"] for s in per_sample if s["wer"] is not None]

    out = {
        "backend": backend,
        "model": model,
        "device": device,
        "n_samples": len(per_sample),
        "cer_mean": round(statistics.mean(cers), 4) if cers else None,
        "wer_mean": round(statistics.mean(wers), 4) if wers else None,
        "cold_ms_mean": round(statistics.mean(s["cold_ms"] for s in per_sample), 1),
        "warm_ms_mean": round(statistics.mean(s["warm_ms"] for s in per_sample), 1),
        "peak_rss_mb": round(rss_mb, 1),
        "gpu_peak_mb": round(gpu_peak, 1),
        "fits_2gb": rss_mb <= 2048,
        "samples": per_sample,
    }
    print("\n__BENCH_JSON__" + json.dumps(out, ensure_ascii=False))


# --- parent: orchestration -------------------------------------------------

def _load_samples(args) -> list[dict] | None:
    if args.wav and args.ref is not None:
        return [{"wav": args.wav, "ref": args.ref}]
    p = Path(args.manifest)
    if not p.exists():
        print(f"매니페스트 {p} 가 없습니다. --wav/--ref 로 단일 샘플 지정 가능.", file=sys.stderr)
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    samples = data.get("samples") or []
    if not samples:
        print(f"매니페스트 {p} 에 samples 가 비어있습니다.", file=sys.stderr)
        return None
    return samples


def _render_markdown(rows: list[dict], args) -> str:
    lines = [
        "# STT Benchmark — Whisper tiny / base / small",
        "",
        f"- 실행 일시: {datetime.now().isoformat(timespec='seconds')}",
        f"- device: {args.device}",
        f"- manifest: `{args.manifest}`",
        f"- warmup: {args.warmup}, runs: {args.runs}",
        f"- samples: {rows[0]['n_samples'] if rows else 0}",
        "",
        "| backend | model | CER | WER | cold ms | warm ms | RSS MB | GPU MB | fits 2GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        cer_s = f"{r['cer_mean']:.3f}" if r["cer_mean"] is not None else "—"
        wer_s = f"{r['wer_mean']:.3f}" if r["wer_mean"] is not None else "—"
        fits = "✓" if r["fits_2gb"] else "✗"
        lines.append(
            f"| {r['backend']} | {r['model']} | {cer_s} | {wer_s} | "
            f"{r['cold_ms_mean']:.0f} | {r['warm_ms_mean']:.0f} | "
            f"{r['peak_rss_mb']:.0f} | {r['gpu_peak_mb']:.0f} | {fits} |"
        )

    lines.append("")
    lines.append("## 권장")
    have_cer = [r for r in rows if r["cer_mean"] is not None]
    if have_cer:
        best = min(have_cer, key=lambda r: r["cer_mean"])
        lines.append(f"- 최저 CER: **{best['backend']} × {best['model']}** (CER={best['cer_mean']:.3f})")
    fits = [r for r in rows if r["fits_2gb"]]
    if fits:
        ordered = sorted(fits, key=lambda r: (r["cer_mean"] is None, r["cer_mean"] or 1e9))
        top = ordered[0]
        lines.append(
            f"- 2GB 적합 후보 중 best: **{top['backend']} × {top['model']}** "
            f"(RSS={top['peak_rss_mb']:.0f}MB)"
        )
    else:
        lines.append("- 2GB 적합 모델 없음 (현재 측정 기준).")

    lines.append("")
    lines.append("## 샘플별 결과")
    for r in rows:
        lines.append(f"### {r['backend']} × {r['model']}")
        lines.append("")
        lines.append("| wav | ref | hyp | CER | WER | cold ms | warm ms |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for s in r["samples"]:
            cer_s = f"{s['cer']:.3f}" if s["cer"] is not None else "—"
            wer_s = f"{s['wer']:.3f}" if s["wer"] is not None else "—"
            ref_disp = s["ref"] if s["ref"].strip() else "_(empty)_"
            lines.append(
                f"| `{Path(s['wav']).name}` | {ref_disp} | {s['hyp']} | "
                f"{cer_s} | {wer_s} | {s['cold_ms']:.0f} | {s['warm_ms']:.0f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _spawn_child(args, backend: str, model: str) -> dict | None:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--single",
        "--backend", backend,
        "--model", model,
        "--manifest", args.manifest,
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
        "--device", args.device,
    ]
    if args.wav and args.ref is not None:
        cmd += ["--wav", args.wav, "--ref", args.ref]
    print(f"  ▶ {backend} × {model} 측정 시작 ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ✗ exit={proc.returncode}", flush=True)
        if proc.stderr:
            print("  ── stderr ──", flush=True)
            print(proc.stderr.rstrip(), flush=True)
        return None

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("__BENCH_JSON__"):
            payload = line[len("__BENCH_JSON__"):]
    if payload is None:
        print("  ✗ child JSON 미발견. stdout:", flush=True)
        print(proc.stdout.rstrip()[:500], flush=True)
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse error: {e}", flush=True)
        return None


def run_parent(args) -> int:
    samples = _load_samples(args)
    if samples is None:
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    backends = ["faster_whisper"]
    if args.include_openai_whisper:
        backends.append("whisper")

    rows: list[dict] = []
    for backend in backends:
        for model in models:
            row = _spawn_child(args, backend, model)
            if row is not None:
                rows.append(row)

    if not rows:
        print("결과 행이 없습니다. 위 stderr 를 확인하세요.", file=sys.stderr)
        return 2

    md = _render_markdown(rows, args)
    print()
    print(md)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n결과 기록: {out}", flush=True)

    return 0


# --- entrypoint ------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "STT 후보 벤치마크 (Whisper tiny/base/small). "
            "첫 실행 시 모델 weight 다운로드 발생 (base ≈ 140MB)."
        ),
    )
    p.add_argument("--single", action="store_true",
                   help="내부 child 모드 (한 (backend, model) 측정 후 JSON 출력)")
    p.add_argument("--backend", default="faster_whisper",
                   choices=["faster_whisper", "whisper"])
    p.add_argument("--model", default=None,
                   help="--single 모드에서 단일 모델 지정")
    p.add_argument("--models", default="tiny,base,small",
                   help="parent 모드에서 콤마 구분 모델 리스트")
    p.add_argument("--manifest", default="configs/bench/stt_korean_short.yaml")
    p.add_argument("--wav", default=None, help="매니페스트 대신 단일 wav 경로")
    p.add_argument("--ref", default=None, help="--wav 와 함께 사용할 reference text")
    p.add_argument("--runs", type=int, default=3, help="warmup 이후 timed 반복 횟수")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--include-openai-whisper", action="store_true",
                   help="openai-whisper 백엔드도 비교 매트릭스에 포함")
    p.add_argument("--out", default=None, help="결과 markdown 파일 경로")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    if args.single:
        if not args.model:
            print("--single 은 --model 필수.", file=sys.stderr)
            return 2
        samples = _load_samples(args)
        if samples is None:
            return 1
        run_single(args.backend, args.model, samples,
                   runs=args.runs, warmup=args.warmup, device=args.device)
        return 0

    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
