"""Microphone capture (push-to-talk) and .wav playback.

Press Enter to start, Enter again to stop. Hard cap at `max_seconds` so a
forgotten session can't fill the disk.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def record_push_to_talk(
    sample_rate: int,
    max_seconds: int,
    out_path: Path | str,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    input("▶  Press Enter to START recording…")
    print(f"🎙  Recording (max {max_seconds}s). Press Enter to STOP.")

    chunks: list[np.ndarray] = []
    stop = threading.Event()

    def _cb(indata, frames, time_info, status):
        if status:
            # benign overruns — print but keep going
            print(f"  (sounddevice status: {status})")
        chunks.append(indata.copy())
        total_frames = sum(c.shape[0] for c in chunks)
        if total_frames / sample_rate >= max_seconds:
            stop.set()

    def _wait_enter():
        try:
            input()
        except EOFError:
            pass
        stop.set()

    threading.Thread(target=_wait_enter, daemon=True).start()
    with sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="float32", callback=_cb
    ):
        stop.wait()

    if not chunks:
        raise RuntimeError("No audio captured — is the mic device available?")

    samples = np.concatenate(chunks, axis=0)[:, 0]
    sf.write(out, samples, sample_rate)
    print(f"💾 Saved {out}  ({samples.shape[0] / sample_rate:.2f}s)")
    return out


def play_buffer(samples: np.ndarray, sample_rate: int) -> None:
    sd.play(samples, sample_rate)
    sd.wait()
