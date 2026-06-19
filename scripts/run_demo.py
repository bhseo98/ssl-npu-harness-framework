"""Push-to-talk Korean voice assistant demo.

Inside Docker:
    docker compose run --rm app

Bare metal:
    pip install -e ".[dev]"
    python scripts/run_demo.py [-c configs/default.yaml]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf
import yaml

# Importing the impl modules triggers their @register(...) decorators.
from npu_harness_framework import stt as _stt  # noqa: F401
from npu_harness_framework import llm as _llm  # noqa: F401
from npu_harness_framework import tts as _tts  # noqa: F401
from npu_harness_framework.audio import play_buffer, record_push_to_talk
from npu_harness_framework.pipeline import SequentialVoicePipeline, VoicePipeline
from npu_harness_framework.registry import build, registered


def _parse_args() -> argparse.Namespace:
    default_cfg = os.environ.get("TARGET_APP_CONFIG", "configs/default.yaml")
    p = argparse.ArgumentParser(description="Push-to-talk Korean voice assistant")
    p.add_argument("--config", "-c", default=default_cfg)
    return p.parse_args()


def _print_banner(cfg_path: Path, cfg: dict) -> None:
    print()
    print("━" * 60)
    print("  🎤  NPU Harness Framework — Voice Assistant Framework")
    print("━" * 60)
    print(f"  config:    {cfg_path}")
    print(f"  language:  {cfg['pipeline']['language']}")
    print(f"  budget:    {cfg['pipeline']['budget_mb']} MB")
    print(f"  registry:  {registered()}")
    print("━" * 60)
    print()


def main() -> int:
    args = _parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path

    if not cfg_path.exists():
        print(f"❌ Config not found: {cfg_path}", file=sys.stderr)
        return 2

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    _print_banner(cfg_path, cfg)

    sequential = bool(cfg["pipeline"].get("sequential_load", False))
    if sequential:
        print("⏳ Sequential-load mode — models build per stage per turn (budget mode).")
        pipeline = SequentialVoicePipeline(
            stt_config=cfg["stt"],
            llm_config=cfg["llm"],
            tts_config=cfg["tts"],
            profiler_enabled=cfg["profiler"]["enabled"],
            log_path=cfg["profiler"].get("log_path"),
            budget_mb=cfg["pipeline"].get("budget_mb"),
        )
    else:
        print("⏳ Loading models (first run downloads weights — be patient)…")
        stt_obj = build("stt", cfg["stt"])
        llm_obj = build("llm", cfg["llm"])
        tts_obj = build("tts", cfg["tts"])
        pipeline = VoicePipeline(
            stt_obj,
            llm_obj,
            tts_obj,
            profiler_enabled=cfg["profiler"]["enabled"],
            log_path=cfg["profiler"].get("log_path"),
            budget_mb=cfg["pipeline"].get("budget_mb"),
        )

    sr = cfg["pipeline"]["sample_rate"]
    rec_dir = Path(cfg["audio"]["output_dir"])
    max_sec = cfg["audio"]["record_seconds_max"]

    print("\n✅ Ready. Press Ctrl+C to exit.\n")
    turn = 0
    try:
        while True:
            turn += 1
            print(f"━━━━ turn {turn} ━━━━")
            wav_in = record_push_to_talk(sr, max_sec, rec_dir / f"in_{turn:03d}.wav")
            _, _, audio_out = pipeline.run(wav_in)
            wav_out = rec_dir / f"out_{turn:03d}.wav"
            sf.write(wav_out, audio_out["samples"], audio_out["sample_rate"])
            print(f"💾  Saved reply  →  {wav_out}")
            play_buffer(audio_out["samples"], audio_out["sample_rate"])
            print()
    except KeyboardInterrupt:
        print("\n👋  Bye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
