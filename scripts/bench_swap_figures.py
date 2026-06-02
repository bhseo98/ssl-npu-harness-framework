#!/usr/bin/env python3
"""Generate comparison figures from logs/bench-swap/summary.json.

Outputs:
- figures/swap-latency-stage.png   stage 별 latency mean ± stdev, side-by-side
- figures/swap-latency-stacked.png  e2e stacked bar (stt + llm + tts)
- figures/swap-memory.png           GPU peak / host RAM bar
- figures/swap-cold-load.png        cold load 비교 (cold start cost)
- figures/swap-tts-detail.png       TTS Piper(CPU) vs espnet(GPU) — 의도된 trade-off
- figures/swap-weight-footprint.png 모델 가중치 footprint (MB) 비교
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASELINE_LABEL = "baseline\nQwen 0.5B + espnet_kss"
NEW_LABEL = "new\nLlama 1B + Piper-KSS"
COLORS = {
    "baseline": "#6c757d",   # neutral grey
    "new": "#1f77b4",        # blue
    "stt": "#9ec5fe",
    "llm": "#4dabf7",
    "tts": "#1971c2",
}

# 정적 weight footprint (디스크 기준; HF 카드/모델 메타 기준 값).
WEIGHT_MB = {
    "stt-faster_whisper-small (CT2 int8)": 240,   # 두 config 공통 (control)
    "Qwen 2.5 0.5B (fp16)": 988,
    "Llama 3.2 1B (fp16)": 2469,
    "espnet_kss JETS + vocoder": 150,
    "Piper KSS (ONNX)": 63,
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _stage_means(cfg: dict, stage: str) -> tuple[float, float]:
    s = cfg.get(stage, {}).get("elapsed_ms", {})
    return s.get("mean", 0.0), s.get("stdev", 0.0)


def fig_latency_stage(summary: dict, out: Path) -> None:
    cfgs = summary["configs"]
    baseline, new = cfgs[0], cfgs[1]

    stages = ["stt", "llm", "tts", "e2e"]
    labels = ["STT", "LLM", "TTS", "end-to-end"]

    bm = [_stage_means(baseline, s)[0] for s in stages]
    be = [_stage_means(baseline, s)[1] for s in stages]
    nm = [_stage_means(new, s)[0] for s in stages]
    ne = [_stage_means(new, s)[1] for s in stages]

    x = np.arange(len(stages))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, bm, w, yerr=be, capsize=4, label=BASELINE_LABEL, color=COLORS["baseline"])
    b2 = ax.bar(x + w/2, nm, w, yerr=ne, capsize=4, label=NEW_LABEL, color=COLORS["new"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("latency (ms, mean ± stdev)")
    ax.set_title("Per-stage latency — baseline vs new\n(N=5 runs × 3 sentences = 15 samples; warmup excluded)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.0f}", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_latency_stacked(summary: dict, out: Path) -> None:
    cfgs = summary["configs"]
    labels = [BASELINE_LABEL, NEW_LABEL]
    stt = [_stage_means(c, "stt")[0] for c in cfgs]
    llm = [_stage_means(c, "llm")[0] for c in cfgs]
    tts = [_stage_means(c, "tts")[0] for c in cfgs]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    p1 = ax.bar(x, stt, color=COLORS["stt"], label="STT (faster_whisper-small)")
    p2 = ax.bar(x, llm, bottom=stt, color=COLORS["llm"], label="LLM")
    bottom2 = [a + b for a, b in zip(stt, llm)]
    p3 = ax.bar(x, tts, bottom=bottom2, color=COLORS["tts"], label="TTS")

    totals = [a + b + c for a, b, c in zip(stt, llm, tts)]
    for i, t in enumerate(totals):
        ax.annotate(f"sum {t:.0f} ms",
                    xy=(i, t), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=10, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("latency (ms)")
    ax.set_title("Per-stage latency stack (sum of stage means)\nbaseline vs new")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_memory(summary: dict, out: Path) -> None:
    cfgs = summary["configs"]
    labels = [BASELINE_LABEL, NEW_LABEL]
    # GPU peak: max across stages.
    gpu_peak = [max(c[s]["gpu_peak_mb"]["max"] for s in ("stt", "llm", "tts")) for c in cfgs]
    ram_peak = [max(c[s]["ram_after_mb"]["max"] for s in ("stt", "llm", "tts")) for c in cfgs]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, gpu_peak, w, label="GPU peak (MB)", color=COLORS["new"])
    b2 = ax.bar(x + w/2, ram_peak, w, label="Host RAM peak (MB)", color=COLORS["baseline"])

    budget = 2048
    ax.axhline(budget, color="crimson", linestyle="--", label=f"target budget {budget} MB")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MB")
    ax.set_title("Peak memory — GPU vs host RAM (across stages)\n2 GB embedded budget marked")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.0f}", xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_cold_load(summary: dict, out: Path) -> None:
    cfgs = summary["configs"]
    labels = [BASELINE_LABEL, NEW_LABEL]
    cold = [c["identity"]["cold_load_ms"] / 1000.0 for c in cfgs]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, cold, color=[COLORS["baseline"], COLORS["new"]])
    ax.set_ylabel("seconds")
    ax.set_title("Cold load time (one-time at process start)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.0f} s", xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_tts_detail(summary: dict, out: Path) -> None:
    cfgs = summary["configs"]
    tts_b = cfgs[0]["tts"]["elapsed_ms"]
    tts_n = cfgs[1]["tts"]["elapsed_ms"]

    labels = ["mean", "min", "max", "stdev"]
    base = [tts_b["mean"], tts_b["min"], tts_b["max"], tts_b["stdev"]]
    new = [tts_n["mean"], tts_n["min"], tts_n["max"], tts_n["stdev"]]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, base, w, label="espnet_kss (JETS) — GPU", color=COLORS["baseline"])
    ax.bar(x + w/2, new, w, label="Piper KSS (ONNX) — CPU", color=COLORS["new"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ms")
    ax.set_title("TTS only — espnet (GPU) vs Piper (CPU)\nintended trade-off: 4× smaller weights, no espnet runtime, in exchange for CPU latency")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    for i, (a, b) in enumerate(zip(base, new)):
        ax.annotate(f"{a:.0f}", xy=(i - w/2, a), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{b:.0f}", xy=(i + w/2, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_weight_footprint(out: Path) -> None:
    labels = list(WEIGHT_MB.keys())
    vals = list(WEIGHT_MB.values())
    colors = [
        "#6c757d",      # STT (shared)
        COLORS["baseline"], COLORS["new"],
        COLORS["baseline"], COLORS["new"],
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, vals, color=colors)
    ax.set_xlabel("disk weight footprint (MB)")
    ax.set_title("Model weight footprint — baseline (grey) vs new (blue)\nLlama is 2.5× Qwen; Piper is 2.4× smaller than espnet_kss")
    ax.invert_yaxis()
    for b in bars:
        w = b.get_width()
        ax.annotate(f"{w} MB", xy=(w, b.get_y() + b.get_height()/2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="logs/bench-swap/summary.json")
    ap.add_argument("--out-dir", default="docs/figures/swap")
    args = ap.parse_args(argv)

    summary = _load(Path(args.summary))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig_latency_stage(summary, out / "swap-latency-stage.png")
    fig_latency_stacked(summary, out / "swap-latency-stacked.png")
    fig_memory(summary, out / "swap-memory.png")
    fig_cold_load(summary, out / "swap-cold-load.png")
    fig_tts_detail(summary, out / "swap-tts-detail.png")
    fig_weight_footprint(out / "swap-weight-footprint.png")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
