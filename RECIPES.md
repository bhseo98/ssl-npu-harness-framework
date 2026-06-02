# 📖 RECIPES

Short, focused "how do I do X?" recipes for the npu-harness-framework voice framework.
Each recipe is self-contained: copy-paste the commands, get the expected output.

Conventions:
- `$` lines are shell commands you run.
- `>>>` lines are Python REPL commands.
- All paths are relative to the repo root.

---

## Table of contents

1. [Run the no-deps smoke test (30 seconds, no model downloads)](#1)
2. [Run the browser demo in Docker (laptop mic + GPU server)](#2)
3. [Run the terminal demo in Docker (server mic + GPU)](#3)
4. [Run the real demo on bare metal (no Docker)](#4)
5. [Run on CPU only (no GPU, slower)](#5)
6. [Add a new STT model](#6)
7. [Add a new LLM model](#7)
8. [Add a new TTS model](#8)
9. [Swap models for a head-to-head benchmark](#9)
10. [Bring your own `.wav` instead of mic input](#10)
11. [Profile latency / memory of a single stage](#11)
12. [Use npu-harness-framework as a library in your own script](#12)
13. [Cross-check with GPT Codex / ChatGPT](#13)
14. [Override the CUDA base image](#14)
15. [Run with Piper TTS instead of ESPnet (smaller, ONNX, MLIR-friendly)](#15)
16. [Reproduce the LLM+TTS swap report (Qwen+espnet vs Llama+Piper)](#16)

---

<a id="1"></a>
## 1) No-deps smoke test (30 seconds)

**Goal:** prove the framework's swap promise without downloading any model.

```
$ python3 -m pip install --user --break-system-packages numpy pytest
$ PYTHONPATH=src python3 -m pytest tests/ -v
$ PYTHONPATH=src python3 scripts/swap_demo.py
```

You should see:
- `10 passed in 0.1?s`
- 3 swap demos with **different transcripts** from 3 different STT configs.

---

<a id="2"></a>
## 2) Browser demo in Docker (laptop mic + GPU server)

**Goal:** run models on the GPU server while using the laptop browser as the
microphone and speaker.

**Preconditions:**
- `docker` installed, `nvidia-container-toolkit` configured
- An NGC-pullable CUDA tag (default `13.2.0-cudnn-devel-ubuntu24.04`)

```
$ docker build -t npu-harness-framework:latest .
$ docker run --rm -it \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/recordings:/workspace/recordings \
    -v hf_cache:/root/.cache/huggingface \
    -v espnet_cache:/root/.cache/espnet \
    npu-harness-framework:latest \
    python scripts/web_demo.py --host 0.0.0.0 --port 8000
```

For a remote GPU server, keep the container running there and open a second
terminal on the Mac:

```
$ ssh -L 8000:localhost:8000 test@gpuserver-HP-Z8-Fury-G5-Workstation-Desktop-PC
$ open http://localhost:8000
```

First run downloads Whisper-tiny (~75 MB), Qwen2.5-0.5B (~1 GB), and the
ESPnet KSS TTS bundle. Subsequent runs are cached in the named
docker volumes (`hf_cache`, `espnet_cache`).

---

<a id="3"></a>
## 3) Terminal demo in Docker (server mic + GPU)

**Goal:** end-to-end Korean voice assistant on a GPU box that has a working
microphone and speaker exposed as `/dev/snd`.

```
$ docker build -t npu-harness-framework:latest .
$ docker run --rm -it \
    --gpus all \
    --device /dev/snd \
    --group-add audio \
    -v $(pwd)/recordings:/workspace/recordings \
    -v hf_cache:/root/.cache/huggingface \
    -v espnet_cache:/root/.cache/espnet \
    npu-harness-framework:latest \
    python scripts/run_demo.py
```

---

<a id="4"></a>
## 4) Real demo on bare metal

**Goal:** same as recipe 3 but without Docker.

```
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -e ".[dev]"
$ python scripts/run_demo.py
```

If you hit `PortAudio not found`, install the system library:
```
$ sudo apt install libportaudio2 portaudio19-dev ffmpeg libsndfile1
```

---

<a id="5"></a>
## 5) Run on CPU only

**Goal:** verify pipeline behavior without a GPU (slower, but matches the
embedded CPU-only virtual-platform stage of the roadmap).

Edit `configs/default.yaml`:
```yaml
stt:
  device: cpu
llm:
  device: cpu
tts:
  device: cpu
```
Then run normally (Docker or bare metal). Expect ~5-10× slower per stage.

---

<a id="6"></a>
## 6) Add a new STT model

**Goal:** plug in, say, Vosk Korean.

Step 1 — add a subclass in [`src/npu_harness_framework/stt.py`](src/npu_harness_framework/stt.py):
```python
@register("stt", "vosk")
class VoskSTT(BaseSTT):
    def __init__(self, model_path: str, language: str = "ko"):
        from vosk import KaldiRecognizer, Model
        self._model = Model(model_path)

    def transcribe(self, audio):
        # ... implement using KaldiRecognizer ...
        return text
```

Step 2 — point config at it in `configs/default.yaml`:
```yaml
stt:
  type: vosk
  model_path: /models/vosk-ko-0.22
```

Step 3 — no other code change needed. Run `pytest` and the demo as before.

The same pattern works for LLM (`@register("llm", "...")`) and TTS
(`@register("tts", "...")`).

---

<a id="7"></a>
## 7) Add a new LLM model

Easiest path — any HuggingFace causal LM with a chat template works with
the existing `hf_causal` impl. Just point config at it:

```yaml
llm:
  type: hf_causal
  model: SomeOrg/SomeModel-1B-Instruct
  max_new_tokens: 256
```

For a non-HF backend (e.g., llama.cpp HTTP server), add a new subclass:
```python
@register("llm", "llamacpp_http")
class LlamaCppHttp(BaseLLM):
    def __init__(self, endpoint: str, system_prompt: str = ""):
        import httpx
        self._endpoint = endpoint
        self._client = httpx.Client(base_url=endpoint, timeout=30)
        self._system_prompt = system_prompt

    def chat(self, user_message, history=None):
        # POST to /v1/chat/completions, parse, return string
        ...
```

---

<a id="8"></a>
## 8) Add a new TTS model

Same pattern. Example — Coqui-TTS Korean VITS as a stable fallback:

```python
@register("tts", "coqui_vits_ko")
class CoquiVitsKo(BaseTTS):
    def __init__(self, model: str = "tts_models/ko/cv/vits", device: str = "auto"):
        from TTS.api import TTS
        self._tts = TTS(model_name=model, progress_bar=False)
        # Coqui's TTS doesn't expose `.to(device)` symmetrically;
        # see their docs for GPU placement.

    def synthesize(self, text):
        wav = self._tts.tts(text=text)
        return {"samples": np.asarray(wav, dtype=np.float32),
                "sample_rate": self._tts.synthesizer.output_sample_rate}
```

Config:
```yaml
tts:
  type: coqui_vits_ko
```

---

<a id="9"></a>
## 9) Head-to-head benchmark of two models

**Goal:** compare `whisper-tiny` vs `whisper-small` on the same input.

Step 1 — make two configs:

`configs/bench_tiny.yaml`:
```yaml
stt: { type: whisper, model: tiny,  language: ko, device: auto }
llm: { ... unchanged ... }
tts: { ... unchanged ... }
profiler: { enabled: true, log_path: logs/bench_tiny.jsonl }
```

`configs/bench_small.yaml`: same but `model: small` and `log_path: logs/bench_small.jsonl`.

Step 2 — feed the same `.wav` to both runs (see recipe 9).

Step 3 — diff the JSONL logs:
```
$ jq '.elapsed_ms' logs/bench_tiny.jsonl
$ jq '.elapsed_ms' logs/bench_small.jsonl
```

---

<a id="10"></a>
## 10) Bring your own `.wav`

**Goal:** skip the microphone and feed a pre-recorded file.

Quick one-off from Python:
```python
from npu_harness_framework import stt as _stt   # registers built-ins
from npu_harness_framework import llm as _llm
from npu_harness_framework import tts as _tts
from npu_harness_framework.registry import build
from npu_harness_framework.pipeline import VoicePipeline
import yaml

cfg = yaml.safe_load(open("configs/default.yaml"))
pipeline = VoicePipeline(
    build("stt", cfg["stt"]),
    build("llm", cfg["llm"]),
    build("tts", cfg["tts"]),
)
user, reply, audio_out = pipeline.run("path/to/my.wav")
print(user, reply)
```

Or via the demo script — replace the `record_push_to_talk(...)` call in
`scripts/run_demo.py` with a hard-coded path.

---

<a id="11"></a>
## 11) Profile a single stage manually

```python
from npu_harness_framework.profiler import measure

with measure("stt", log_path="logs/single.jsonl", budget_mb=2048):
    text = stt.transcribe("input.wav")
```

You'll get a stdout line with elapsed ms, RAM RSS, and GPU peak; and a
JSON line appended to `logs/single.jsonl`. Set `budget_mb` to anything;
the profiler warns when RSS exceeds it.

---

<a id="12"></a>
## 12) Use npu-harness-framework as a library

```python
# my_voice_app.py
from npu_harness_framework import stt, llm, tts                 # registers built-ins
from npu_harness_framework.registry import build
from npu_harness_framework.pipeline import VoicePipeline

pipeline = VoicePipeline(
    stt=build("stt", {"type": "whisper", "model": "tiny", "language": "ko"}),
    llm=build("llm", {"type": "hf_causal", "model": "Qwen/Qwen2.5-0.5B-Instruct"}),
    tts=build("tts", {"type": "espnet_kss"}),
)

user, reply, audio = pipeline.run("greeting.wav")
```

Install as editable from another repo:
```
$ pip install -e /path/to/npu-harness-frameworklication-code
```

---

<a id="13"></a>
## 13) Cross-check with GPT Codex

This repo has **two** Codex-driven review flows. Pick one (or both).

### 12a) Automated, every commit — pre-commit hook

`scripts/codex-review.sh` runs on every `git commit` via `.git/hooks/pre-commit`.
It pipes the **staged diff** + a project-aware prompt to `codex exec` and
emits a 🟢/🟡/🔴 verdict. 🔴 blocks the commit; 🟢/🟡 lets it through.
See the [README's "Automatic Codex cross-check"](README.md#-automatic-codex-cross-check-every-commit)
for setup. Bypass: `git commit --no-verify`.

Verdicts are saved to `.codex-reviews/<UTC-timestamp>.md` (gitignored).

### 12b) Manual full-repo review — (내부 문서)

For big-picture sweeps (e.g. spec-vs-implementation audit), paste
**(내부 문서)** into ChatGPT-with-code, an
interactive `codex` session (`codex` then drop the bundle in), or any
code-aware LLM. The bundle contains:
- the original requirements (verbatim from the (내부 문서) files)
- the implementation file list with one-line summaries
- a structured verdict template the reviewer fills in (Markdown table).

Run it whenever you'd want a second opinion that goes beyond the diff —
e.g. before tagging a release or after a non-trivial refactor.

---

<a id="14"></a>
## 14) Override the CUDA base image

If `nvidia/cuda:13.2.0-…` is not yet on NGC (it was the host driver at the
time of writing, but the container image may lag), build with a known-good
tag:

```
$ docker build --build-arg CUDA_TAG=12.8.0-cudnn-devel-ubuntu24.04 -t npu-harness-framework:latest .
```

Or edit `compose.yaml` `services.app.build.args.CUDA_TAG`.

CUDA 12.x containers run fine on host driver 13.x (drivers are
forward-compatible with older CUDA runtimes).

---

<a id="15"></a>
## 15) Run with Piper TTS instead of ESPnet

**Goal:** swap the default `espnet_kss` (JETS, ~150 MB, full espnet runtime)
for `piper` (VITS, **~63 MB ONNX**, `onnxruntime` + `pygoruut` only). Same
KSS Korean speaker — only the architecture and runtime differ.

**Why bother:** ONNX → MLIR lowering path is much shorter than
ESPnet → MLIR, so this is the TTS we plan to take to the embedded NPU.

Step 1 — download the Korean Piper voice (one-time, ~61 MB):

```
$ mkdir -p models/piper-ko
$ curl -fL -o models/piper-ko/ko-kss.onnx \
    https://huggingface.co/neurlang/piper-onnx-kss-korean/resolve/main/piper-kss-korean.onnx
$ curl -fL -o models/piper-ko/ko-kss.onnx.json \
    https://huggingface.co/neurlang/piper-onnx-kss-korean/resolve/main/piper-kss-korean.onnx.json
```

License: `CC-BY-NC-SA-4.0` (evaluation use; not a production redistribution).

Step 2 — point the config at it:

```yaml
tts:
  type: piper
  voice: models/piper-ko/ko-kss.onnx
  device: auto
```

A ready-made config is at `configs/variants/swap-new-llama-piper.yaml`.

Step 3 — run:

```
$ docker compose run --rm \
    -v $(pwd)/models:/workspace/models \
    app python scripts/run_demo.py -c configs/variants/swap-new-llama-piper.yaml
```

The runtime auto-detects: `onnxruntime` GPU provider when CUDA libs match,
CPU provider otherwise. Piper is CPU-fast by design — CPU fallback is fine
for typical desktop boxes.

---

<a id="16"></a>
## 16) Reproduce the LLM+TTS swap report

**Goal:** regenerate the figures + report comparing baseline
(`Qwen+espnet`) vs new (`Llama+Piper`). Useful as a template for any
future swap (e.g. after quantization).

```
$ python scripts/bench_swap.py \
    --configs configs/variants/swap-baseline-qwen-espnet.yaml \
              configs/variants/swap-new-llama-piper.yaml \
    --runs 5 --warmup 1 \
    --out logs/bench-swap/summary.json
$ python scripts/bench_swap_figures.py
```

Outputs:
- `logs/bench-swap/summary.json` — aggregated mean/stdev/min/max per stage
- `logs/bench-swap/*.jsonl` — raw per-stage records (identity, e2e, stt, llm, tts)
- `docs/figures/swap/*.png` — 6 comparison figures (latency, memory, weight, cold-load)
- (내부 문서) — written report (figure-driven)

The script generates **synthetic Korean wav inputs via gTTS** so no
microphone is needed. Each config runs in a separate subprocess so RAM /
GPU peak measurements are not contaminated by the previous model's
leftover memory.

To compare different model pairs, just point `--configs` at different
yaml files — the figure generator picks whichever two it gets.
