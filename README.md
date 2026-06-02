<div align="center">

# 🖼️ NPU Harness Framework — `vision-app`

**ResNet18 ImageNet classifier** on the modality-agnostic harness core.
*4-stage Pipeline of `BaseStage` plug-ins — zero framework-core changes.*

[![CI](https://github.com/bhseo98/ssl-npu-harness-framework/actions/workflows/ci.yml/badge.svg?branch=vision-app)](https://github.com/bhseo98/ssl-npu-harness-framework/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.17%2B-ee4c2c)](https://pytorch.org/vision/stable/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/bhseo98/ssl-npu-harness-framework/blob/main/LICENSE)

[**Quick start**](#-quick-start) ·
[**Architecture**](docs/ARCHITECTURE.md) ·
[**Contributing**](https://github.com/bhseo98/ssl-npu-harness-framework/blob/main/CONTRIBUTING.md) ·
[**Security**](https://github.com/bhseo98/ssl-npu-harness-framework/blob/main/SECURITY.md)

> **Branch map**: [`main`](../../tree/main) framework core ·
> [`voice-app`](../../tree/voice-app) STT/LLM/TTS — Phase 1 ✅ ·
> **`vision-app`** (you are here) image classifier ✅ ·
> [`torch-mlir-zoo`](../../tree/torch-mlir-zoo) Phase 2 MLIR lowering ✅

</div>

---

## Why this exists

The voice-app branch proved the harness core (`BaseStage` + `Pipeline` +
`register` + `profiler`) works for STT/LLM/TTS. **`vision-app` proves the same
core works for a completely different modality** — image classification — *without
modifying the framework core a single line*.

```bash
git diff main vision-app -- src/npu_harness_framework/   # empty (framework core invariant)
```

The only changes vs `main` are: `src/vision_app/` plug-ins + `configs/vision/`
+ `scripts/run_vision.py` + `tests/test_vision_pipeline.py` + minimal deps.

## 🧩 Pipeline

```
path/str  →  PIL.Image  →  torch.Tensor (1,3,224,224)
          →  softmax probs (1,1000)  →  list[(label, prob)]
```

| Stage | `@register(...)` | Module |
|---|---|---|
| `load` | `("loader", "image_file")` | `vision_app.stages.LoadImage` |
| `preprocess` | `("preprocess", "imagenet224")` | `vision_app.stages.PreprocessImagenet` |
| `classify` | `("classifier", "resnet18")` | `vision_app.stages.ResNet18Classifier` |
| `topk` | `("postprocess", "topk")` | `vision_app.stages.TopK` |

Same `BaseStage` ABC + same `Pipeline` orchestrator + same `measure(...)`
profiler as voice-app. The only domain-specific code is the four stage
classes (~65 LoC).

## 🚀 Quick start

### 0) No-deps smoke test (numpy + pytest only)

```bash
python3 -m pip install --user --break-system-packages psutil pytest
PYTHONPATH=src python3 -m pytest tests/test_core.py -v
```

### 1) Docker (실행 컨테이너, CI / 외부 데모용 — pulls ~700MB torchvision wheels)

```bash
docker build -t npu-harness-framework-vision:latest .
docker run --rm npu-harness-framework-vision:latest                                  # pytest
docker run --rm npu-harness-framework-vision:latest python scripts/run_vision.py     # demo (offline gradient image)
docker run --rm -v /path/to/img.jpg:/workspace/img.jpg \
       npu-harness-framework-vision:latest python scripts/run_vision.py /workspace/img.jpg
```

또는 `compose.yaml` 의 `test` service:

```bash
docker compose run --rm test    # pytest 한 번
```

### 1.5) Docker dev shell — 가상환경 접속용 (*harness 권장*)

호스트에 `venv` 셋업 없이 *셋업 끝난 환경에 진입*. **harness 의 철학에 가장
부합** — uniform contract 아래서 backbone (ResNet / MobileNet / ViT / ...)
을 자유롭게 swap. 개발 / 디버깅 / 반복 실험 / 외부 협업자와 동일 환경 보장.

```bash
# 처음 한 번 빌드 (test 이미지와 별도 — :dev tag)
docker compose build dev

# bash 진입 — 전체 repo 가 /workspace 에 마운트, 편집 즉시 반영
docker compose run --rm dev

# 안에서 자유롭게 (torch + torchvision + ResNet weights pre-cached)
[vision-app:dev] /workspace $ pytest tests/ -v
[vision-app:dev] /workspace $ python scripts/run_vision.py
[vision-app:dev] /workspace $ python scripts/run_vision.py /workspace/imgs/cat.jpg
[vision-app:dev] /workspace $ ipython                  # REPL — backbone 직접 비교
[vision-app:dev] /workspace $ vim configs/vision/resnet18.yaml

# 또는 진입 없이 한 줄 명령
docker compose run --rm dev \
    python scripts/run_vision.py /workspace/imgs/sample.jpg
```

`Dockerfile.dev` 가 추가하는 dev 도구: `vim-tiny`, `jq`, `less`, `git`,
`procps`, `bash-completion`, `ipython`. torchvision weights cache 는
`torchvision_cache` named volume 으로 persistent.

### 2) Bare metal (Linux + PyTorch CPU/GPU)

```bash
pip install -e .            # installs psutil + torch + torchvision + Pillow
python scripts/run_vision.py                  # offline gradient image
python scripts/run_vision.py /path/to/img.jpg # real image, ImageNet top-5
```

If no image is provided (or the default path doesn't exist), the script
**generates a deterministic gradient image** so the demo runs entirely
offline. Top-5 ImageNet labels print on stdout and `logs/profile.jsonl`
gets one line per stage (latency · RAM · GPU peak).

## ⚙️ Configuration ([`configs/vision/`](configs/vision/))

```yaml
# configs/vision/resnet18.yaml — 4 stages, one line per stage.
stages:
  - name: load
    stage: loader
    config: { type: image_file }
  - name: preprocess
    stage: preprocess
    config: { type: imagenet224 }
  - name: classify
    stage: classifier
    config: { type: resnet18 }
  - name: topk
    stage: postprocess
    config: { type: topk, k: 5 }
```

Swap a stage = change one line in YAML — no Python code touch.

## 🔄 Adding / swapping a model

To add a new backbone (e.g. MobileNetV3, EfficientNet-B0, ViT-Small):

1. Add a new `@register("classifier", "<name>")` class in
   [`src/vision_app/stages.py`](src/vision_app/stages.py).
2. Add a sibling YAML under `configs/vision/` (or change the existing one).
3. (Optional) add a pytest case under `tests/test_vision_pipeline.py`.

No framework-core change, no `Pipeline` change, no profiler change. That's
the entire promise of the harness.

## 🏛 Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full architectural
view (registry · profiler · BaseStage · Pipeline) — same document on every
branch, examples drawn from the host branch's domain.

The `BaseStage` contract is intentionally **narrow** (one `__call__` method)
so new modalities slot in without growing the core. vision-app's plug-ins
are ~65 LoC total; the rest is config + tests + reusable Docker image.

## Related work (cross-branch)

`torch-mlir-zoo` branch completed Step 5 (2026-05-27) — Llama-3.2-1B FP32
PyTorch on-device model → `iree.turbine.aot.export` → top-level torch dialect
MLIR (12 GB · 6 218 lines · `server_side_op_hits = ∅`) →
`iree-compile --iree-hal-target-backends=llvm-cpu` → `.vmfb` (6.0 GB · 36.86 s).
See [`development.md`](https://github.com/bhseo98/ssl-npu-harness-framework/blob/torch-mlir-zoo/development.md)
on `torch-mlir-zoo` branch for the full 1:1 PDF↔repo status board.

This vision-app branch demonstrates the framework's modality-agnostic
contract — same `BaseStage` ABC reused for ResNet18 classification.

## Citation

```bibtex
@software{npu_harness_framework_vision_app_2026,
  author  = {Seo, BoHyun},
  title   = {vision-app: ResNet18 modality-agnostic plugin (NPU Harness Framework)},
  year    = {2026},
  url     = {https://github.com/bhseo98/ssl-npu-harness-framework/tree/vision-app}
}
```

## License

MIT — see [`LICENSE`](https://github.com/bhseo98/ssl-npu-harness-framework/blob/main/LICENSE) on the `main` branch.
