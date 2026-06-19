"""TTS implementations.

Built-in:
- `espnet_kss` — ESPnet Korean TTS trained on the KSS single-speaker dataset
  (local, JETS). Historical `kan-bayashi/kss_fastspeech2` is not on HF, so the
  default points at a public ESPnet KSS model.
- `edge_tts` — Microsoft Edge Neural TTS over the cloud. Much more natural
  Korean prosody, but requires network access. No model weights or GPU.
- `piper` — VITS-based, ONNX-served (~63 MB Korean weights). CPU-first; the
  intended embedded TTS once we leave ESPnet's heavy framework dependency.
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import numpy as np

from .interfaces import AudioBuffer, BaseTTS
from .registry import register


DEFAULT_KSS_TTS_MODEL = (
    "imdanboy/kss_tts_train_jets_raw_phn_null_g2pk_train.total_count.ave"
)


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


@register("tts", "fastspeech2_kss")  # Backward-compatible alias.
@register("tts", "espnet_kss")
class EspnetKSS(BaseTTS):
    def __init__(
        self,
        model: str = DEFAULT_KSS_TTS_MODEL,
        device: str = "auto",
    ):
        from espnet2.bin.tts_inference import Text2Speech

        self._device = _resolve_device(device)
        self._tts = Text2Speech.from_pretrained(
            model_tag=model, device=self._device
        )

    def synthesize(self, text: str) -> AudioBuffer:
        import torch

        with torch.no_grad():
            out = self._tts(text)
        wav = out["wav"].detach().cpu().numpy().astype(np.float32)
        return {"samples": wav, "sample_rate": int(self._tts.fs)}


FastSpeech2KSS = EspnetKSS


DEFAULT_EDGE_VOICE = "ko-KR-SunHiNeural"  # Korean female; alt: ko-KR-InJoonNeural (male)


@register("tts", "edge_tts")
class EdgeTTS(BaseTTS):
    """Microsoft Edge Neural TTS via the `edge-tts` package.

    Cloud-only — each synthesize() call hits Microsoft's endpoint.
    Decodes the MP3 stream with ffmpeg (already in the image) for portability.
    """

    def __init__(self, voice: str = DEFAULT_EDGE_VOICE, **_kwargs):
        self._voice = voice

    def synthesize(self, text: str) -> AudioBuffer:
        import edge_tts

        async def _stream() -> bytes:
            buf = io.BytesIO()
            comm = edge_tts.Communicate(text, self._voice)
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        mp3_bytes = asyncio.run(_stream())
        wav, sr = _decode_mp3_to_float32(mp3_bytes)
        return {"samples": wav, "sample_rate": sr}


def _decode_mp3_to_float32(mp3: bytes) -> tuple[np.ndarray, int]:
    import subprocess

    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "mp3", "-i", "pipe:0",
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "24000",
            "pipe:1",
        ],
        input=mp3, capture_output=True, check=True,
    )
    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return samples, 24000


@register("tts", "piper")
class PiperTTS(BaseTTS):
    """Piper (VITS) ONNX TTS adapter.

    Loads a Piper voice file pair (``<voice>.onnx`` + ``<voice>.onnx.json``)
    and runs inference via onnxruntime. The KSS Korean voice we ship uses
    the ``pygoruut`` phonemizer (not espeak), so we phonemize ourselves
    and map IPA → phoneme IDs from the model's ``phoneme_id_map``.

    Why this adapter exists alongside ``espnet_kss``: same KSS speaker /
    dataset, but VITS architecture + ONNX serving — strips out the full
    espnet runtime and drops weight footprint by ~2.4×.
    """

    def __init__(
        self,
        voice: str = "models/piper-ko/ko-kss.onnx",
        config: str | None = None,
        device: str = "auto",
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
    ):
        import onnxruntime as ort
        from pygoruut.pygoruut import Pygoruut

        voice_path = Path(voice)
        cfg_path = Path(config) if config else voice_path.with_suffix(voice_path.suffix + ".json")
        with cfg_path.open() as fh:
            self._cfg = json.load(fh)

        providers = ["CPUExecutionProvider"]
        if device in ("auto", "cuda"):
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(voice_path), providers=providers)

        self._sample_rate = int(self._cfg["audio"]["sample_rate"])
        inf = self._cfg.get("inference", {})
        self._noise_scale = float(noise_scale if noise_scale is not None else inf.get("noise_scale", 0.667))
        self._length_scale = float(length_scale if length_scale is not None else inf.get("length_scale", 1.0))
        self._noise_w = float(noise_w if noise_w is not None else inf.get("noise_w", 0.8))

        # Phoneme ID map: each phoneme character → list[int] (usually one ID).
        self._phoneme_ids: dict[str, list[int]] = dict(self._cfg["phoneme_id_map"])
        self._pad_id = self._phoneme_ids.get("_", [0])[0]
        self._bos_id = self._phoneme_ids.get("^", [1])[0]
        self._eos_id = self._phoneme_ids.get("$", [2])[0]
        self._lang = self._cfg.get("language", {}).get("code") or self._cfg.get("espeak", {}).get("voice", "Korean")

        self._phonemizer = Pygoruut()

    def _phonemize(self, text: str) -> str:
        out = self._phonemizer.phonemize(language=self._lang, sentence=text)
        # PhonemeResponse: list[Word(Phonetic=..., PrePunct, PostPunct)].
        pieces: list[str] = []
        for w in out.Words:
            if w.PrePunct:
                pieces.append(w.PrePunct)
            pieces.append(w.Phonetic)
            if w.PostPunct:
                pieces.append(w.PostPunct)
        return " ".join(pieces)

    def _text_to_ids(self, phonemes: str) -> list[int]:
        ids: list[int] = [self._bos_id, self._pad_id]
        for ch in phonemes:
            mapped = self._phoneme_ids.get(ch)
            if mapped is None:
                continue  # unknown char (e.g. dictionary miss) — skip silently
            ids.extend(mapped)
            ids.append(self._pad_id)
        ids.append(self._eos_id)
        return ids

    def synthesize(self, text: str) -> AudioBuffer:
        phonemes = self._phonemize(text)
        ids = self._text_to_ids(phonemes)
        input_ids = np.asarray([ids], dtype=np.int64)
        input_lens = np.asarray([len(ids)], dtype=np.int64)
        scales = np.asarray(
            [self._noise_scale, self._length_scale, self._noise_w],
            dtype=np.float32,
        )
        audio = self._sess.run(
            None,
            {"input": input_ids, "input_lengths": input_lens, "scales": scales},
        )[0]
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        return {"samples": wav, "sample_rate": self._sample_rate}
