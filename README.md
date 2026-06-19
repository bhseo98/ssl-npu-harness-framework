<div align="center">

# 🎤 NPU Harness Framework — `voice-app`

**Korean voice AI evaluation pipeline** for embedded targets.
*STT (Whisper-small) → LLM (Qwen 2.5-0.5B) → TTS (ESPnet-KSS) · model-swappable · dockerized · 2 GB DRAM budget*

> Applies the [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) / HELM / `inspect-evals` style **AI harness pattern** — uniform contract, swap-any-model, controlled measurement — to the **NPU lowering** domain: same harness profiles candidates *and* prepares the locked configuration for MLIR export.

[![CI](https://github.com/bhseo98/npu-harness-framework/actions/workflows/ci.yml/badge.svg?branch=voice-app)](https://github.com/bhseo98/npu-harness-framework/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/docker-compose-2496ed?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/bhseo98/npu-harness-framework/blob/main/LICENSE)

[**Quick start**](#-quick-start) ·
[**Recipes**](RECIPES.md) ·
[**Architecture**](docs/ARCHITECTURE.md) ·
[**Contributing**](https://github.com/bhseo98/npu-harness-framework/blob/main/CONTRIBUTING.md) ·
[**Security**](https://github.com/bhseo98/npu-harness-framework/blob/main/SECURITY.md)

> **Branch map**: [`main`](../../tree/main) framework core ·
> **`voice-app`** (you are here) — Phase 1 ✅ ·
> [`vision-app`](../../tree/vision-app) ResNet18 classifier ·
> [`torch-mlir-zoo`](../../tree/torch-mlir-zoo) Phase 2 MLIR lowering ✅

</div>

---

## Why this exists

We're building an edge embedded voice assistant on a **custom SoC + NPU**.
The final target ships with **2 GB DRAM**, runs CPU-first, and will later
offload to our own NPU via a custom MLIR lowering compiler.

Flashing an FPGA on every model tweak is too slow when we're still iterating
on **algorithmic** behavior. So we validate everything on a GPU server first,
in Python/PyTorch, under the same memory budget — then lower the locked
configuration down to the chip.

This repo is that test bench. The framework is intentionally minimal:
**8 Python files**, one Docker container, one YAML config.

## 🧩 Pipeline

```
   🎙  .wav        STT              LLM              TTS         🔊
 ┌──────────┐  ┌──────────┐    ┌────────────┐   ┌──────────┐
 │  mic     │→ │  Whisper │ →  │   Qwen 2.5 │ → │  ESPnet  │ → speaker
 │ push-to- │  │  tiny ko │    │ 0.5B-Inst. │   │ ESPnet KO│
 │   talk   │  └──────────┘    └────────────┘   └──────────┘
 └──────────┘        │              │                │
                     └──────────────┴────────────────┘
                                    ↓
                       ┌────────────────────────────┐
                       │  profiler                  │
                       │  · latency (ms)            │
                       │  · RAM RSS (MB)            │
                       │  · GPU peak alloc (MB)     │
                       │  · 2 GB budget warning     │
                       └────────────────────────────┘
```

Every stage is **one class behind one method**. Swap a model = subclass + one YAML line.

## 🚀 Quick start

Four paths, in order of "least to most setup".

### 0) No-deps smoke test (numpy + pytest only — proves swap works)

```bash
python3 -m pip install --user --break-system-packages numpy pytest
PYTHONPATH=src python3 -m pytest tests/ -v
PYTHONPATH=src python3 scripts/swap_demo.py
```
Validates the framework contract without downloading any model. ~150 ms.

### 1) Docker (recommended for real models)

The **`Dockerfile`** is the source of truth. Compose is optional sugar.

```bash
# build
docker build -t npu-harness-framework:latest .

# run pytest in the image
docker run --rm npu-harness-framework:latest pytest -v

# run the browser demo (GPU server models + laptop mic/speaker)
docker run --rm -it \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/recordings:/workspace/recordings \
    -v hf_cache:/root/.cache/huggingface \
    -v espnet_cache:/root/.cache/espnet \
    npu-harness-framework:latest \
    python scripts/web_demo.py --host 0.0.0.0 --port 8000

# optional: run the terminal demo when the server has /dev/snd
docker run --rm -it \
    --gpus all \
    --device /dev/snd \
    --group-add audio \
    -v $(pwd)/recordings:/workspace/recordings \
    -v hf_cache:/root/.cache/huggingface \
    -v espnet_cache:/root/.cache/espnet \
    npu-harness-framework:latest \
    python scripts/run_demo.py
```

If `nvidia/cuda:13.2.0-…` is not yet on NGC, override:
```bash
docker build --build-arg CUDA_TAG=12.8.0-cudnn-devel-ubuntu24.04 -t npu-harness-framework:latest .
```

### 2) Docker Compose (same Dockerfile, less typing)

```bash
docker compose build
docker compose up web                 # browser demo
docker compose run --rm app           # terminal demo with /dev/snd
docker compose run --rm test          # pytest
```

### 2.5) Docker dev shell — 가상환경 접속용 (*harness 권장*)

호스트에 `venv` 셋업 없이 *셋업 끝난 GPU + audio 환경에 진입*. **harness 의
철학에 가장 부합** — uniform contract 아래서 모델 / 실험을 자유롭게 swap.
개발 / 디버깅 / 반복 실험 / 외부 협업자와 동일 환경 보장.

```bash
# 처음 한 번 빌드 (app 이미지와 별도 — :dev tag)
docker compose build dev

# bash 진입 — 전체 repo 가 /workspace 에 마운트, 편집 즉시 반영
docker compose run --rm dev

# 안에서 자유롭게 (faster-whisper, transformers, ESPnet 모두 셋업 완료)
[voice-app:dev] /workspace $ pytest tests/ -v
[voice-app:dev] /workspace $ python scripts/run_demo.py
[voice-app:dev] /workspace $ python scripts/web_demo.py --host 0.0.0.0 --port 8000
[voice-app:dev] /workspace $ ipython              # REPL — 모델 직접 load
[voice-app:dev] /workspace $ vim configs/default.yaml

# 또는 진입 없이 한 줄 명령
HF_TOKEN=$HF_TOKEN docker compose run --rm dev \
    python scripts/web_demo.py --host 0.0.0.0 --port 8000
```

`Dockerfile.dev` 가 추가하는 dev 도구: `vim-tiny`, `jq`, `less`, `procps`,
`bash-completion`, `ipython`. HF/torch/ESPnet 모델 캐시는 named volume 으로
persistent — 재시작해도 모델 재다운로드 없음.

If the GPU server is remote, forward the port and open the UI on the Mac:

```bash
ssh -L 8000:localhost:8000 test@gpuserver-HP-Z8-Fury-G5-Workstation-Desktop-PC
open http://localhost:8000
```

### 3) Bare metal (Linux + CUDA + working microphone)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_demo.py
pytest
```

### Interaction (terminal demo)

| key       | action                                          |
| --------- | ----------------------------------------------- |
| `Enter`   | start recording                                 |
| `Enter`   | stop recording (or it auto-stops at `max_sec`)  |
| —         | reply plays through speaker + saved as `.wav`   |
| `Ctrl+C`  | exit                                            |

For everything else (adding a new STT/LLM/TTS model, profiling, CPU-only mode, using as a library) → see **[RECIPES.md](RECIPES.md)**.

## 🏛 Architecture

설계 의도 + 6개 Mermaid 다이어그램 (system context · module deps · class
hierarchy · runtime sequence · quality gate · lowering roadmap) 은
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** 참조. GitHub web에서 바로
렌더링된다.

### 🤖 Automatic Codex cross-check (every commit)

Every `git commit` runs the staged diff through OpenAI **Codex CLI** for
a project-aware review (model-swap contract, MLIR friendliness, 2 GB
budget). Verdict has 3 states and the wrapper turns them into exit codes:

| verdict   | meaning                                       | exit |
| --------- | --------------------------------------------- | ---- |
| 🟢 LGTM   | safe — commit proceeds                        | 0    |
| 🟡 NIT    | cosmetic only — commit proceeds with note     | 0    |
| 🔴 BLOCK  | real problem — commit is rejected             | 1    |

Set-up (once per clone):
```bash
# 1) Install Codex CLI without npm  (musl prebuilt; ~75 MB)
curl -fsSL -o /tmp/codex.tgz \
    https://github.com/openai/codex/releases/download/rust-v0.130.0/codex-x86_64-unknown-linux-musl.tar.gz
mkdir -p ~/.local/bin && tar -xzf /tmp/codex.tgz -C /tmp/ && \
    mv /tmp/codex-x86_64-unknown-linux-musl ~/.local/bin/codex && chmod +x ~/.local/bin/codex
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 2) Authenticate (browser flow; uses ChatGPT Plus/Pro account)
codex login

# 3) Wire up the pre-commit hook
bash scripts/install-hooks.sh
```

Bypass once: `git commit --no-verify`. Fail-open: if `codex` is missing
or unauthenticated, commits proceed with a warning (offline work is not
blocked). Verdicts pile up in `.codex-reviews/` (gitignored).

## ⚙️ Configuration

All knobs in [`configs/default.yaml`](configs/default.yaml).

```yaml
stt:
  type: whisper                              # registry key
  model: tiny                                # tiny | base | small | medium | large
  language: ko

llm:
  type: hf_causal
  model: Qwen/Qwen2.5-0.5B-Instruct
  max_new_tokens: 128
  temperature: 0.7
  system_prompt: |
    You are a concise Korean voice assistant. Reply in 1-2 sentences.

tts:
  type: espnet_kss
  model: imdanboy/kss_tts_train_jets_raw_phn_null_g2pk_train.total_count.ave
```

Use a custom config: `python scripts/run_demo.py -c configs/my.yaml`.

## 🔄 Adding / swapping a model

```python
# src/npu_harness_framework/stt.py  (or your own module)
from npu_harness_framework.interfaces import BaseSTT
from npu_harness_framework.registry import register

@register("stt", "vosk")
class VoskSTT(BaseSTT):
    def __init__(self, model_path: str, language: str = "ko"):
        ...
    def transcribe(self, audio):
        ...
```

```yaml
# configs/default.yaml
stt:
  type: vosk
  model_path: /models/vosk-ko
```

That's the entire swap. Pipeline / demo / tests don't change.

## 📁 Layout

```
npu-harness-frameworklication/
├── compose.yaml                docker compose (GPU + mic + ESPnet)
├── Dockerfile                  CUDA + python + ffmpeg + portaudio
├── pyproject.toml
├── configs/
│   └── default.yaml            ← model swap lives here
├── src/npu_harness_framework/
│   ├── interfaces.py           BaseSTT / BaseLLM / BaseTTS
│   ├── registry.py             @register / build()
│   ├── pipeline.py             VoicePipeline orchestrator
│   ├── profiler.py             latency + RAM + GPU + budget warning
│   ├── audio.py                push-to-talk + playback (sounddevice)
│   ├── stt.py                  WhisperSTT
│   ├── llm.py                  HFCausalLM (Qwen2.5)
│   └── tts.py                  EspnetKSS
├── scripts/
│   └── run_demo.py             interactive entry point
└── tests/
    └── test_pipeline.py        Dummy-stage smoke tests
```

## 💾 Memory budget (default config)

Target embedded device: **2048 MB DRAM**.

| stage                       | approx fp16 RAM |
| --------------------------- | --------------- |
| whisper-tiny                | ~  75 MB        |
| Qwen 2.5-0.5B-Instruct      | ~1 000 MB       |
| ESPnet KSS TTS              | model-dependent |
| python / torch overhead     | ~ 200 MB        |
| **total**                   | **~1.4 GB** (FastSpeech2 기준; JETS 실측 GPU 검증 별도) |

The profiler emits `⚠` if RSS exceeds `pipeline.budget_mb` at any stage.

## 🗺 Roadmap

| Stage | Goal                                                  | Status |
| ----- | ----------------------------------------------------- | ------ |
| 1     | Functional E2E demo on GPU server                     | ✅ this repo |
| 2     | Model-swap framework + head-to-head benchmarks        | ⏭ next   |
| 3     | PyTorch → MLIR lowering                               | planned  |
| 4     | Virtual platform (CPU-only RISC-V SoC model)          | planned  |
| 5     | NPU integration                                       | planned  |
| 6     | Real chip deployment                                  | planned  |

See [`CLAUDE-properties.md`](CLAUDE-properties.md) for the full staging plan.

## 🐛 Troubleshooting

| symptom                                            | fix                                                                                     |
| -------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `docker pull` fails on `nvidia/cuda:13.2.0-...`    | edit `compose.yaml` → `args.CUDA_TAG` to `12.8.0-cudnn-devel-ubuntu24.04` (or 12.6.3)   |
| `No audio captured`                                | host has no mic, or `/dev/snd` isn't passed through — check `groups` of `docker` user   |
| `OSError: PortAudio library not found` (bare metal)| `apt install libportaudio2 portaudio19-dev`                                             |
| ESPnet model download stalls                       | run once interactively to accept the HF cert / cache; or pre-populate `~/.cache/espnet` |
| pytest can't find `npu_harness_framework`                     | run `pip install -e ".[dev]"` once                                                      |

## Related work (cross-branch)

`torch-mlir-zoo` branch completed Step 5 (2026-05-27) — Llama-3.2-1B FP32
PyTorch on-device model → `iree.turbine.aot.export` → top-level torch dialect
MLIR (12 GB · 6 218 lines · `server_side_op_hits = ∅`) →
`iree-compile --iree-hal-target-backends=llvm-cpu` → `.vmfb` (6.0 GB · 36.86 s).

The on-device LLM implementation lives at [`src/torch_mlir_zoo/models/llama_on_device.py`](https://github.com/bhseo98/npu-harness-framework/blob/torch-mlir-zoo/src/torch_mlir_zoo/models/llama_on_device.py)
(171 LoC, RMSNorm + GQA + SwiGLU + RoPE, real HF gated weights loader). See
[`development.md`](https://github.com/bhseo98/npu-harness-framework/blob/torch-mlir-zoo/development.md)
on `torch-mlir-zoo` for the full PDF↔repo 1:1 status board and
[`docs/RECIPES-zoo.md`](https://github.com/bhseo98/npu-harness-framework/blob/torch-mlir-zoo/docs/RECIPES-zoo.md)
for the lowering recipe.

Budget warning during export (peak RAM 12.3 GB ≫ 2 GB) provides measured
justification for the upcoming Step 6 (Q8_0 GGUF quantization + Whisper-tiny
INT8) per *quantization-last* design principle.

## Citation

```bibtex
@software{npu_harness_framework_voice_app_2026,
  author  = {Seo, BoHyun},
  title   = {voice-app: Korean voice AI evaluation pipeline (NPU Harness Framework)},
  year    = {2026},
  url     = {https://github.com/bhseo98/npu-harness-framework/tree/voice-app}
}
```

## License

MIT — see [`LICENSE`](https://github.com/bhseo98/npu-harness-framework/blob/main/LICENSE) on the `main` branch.
