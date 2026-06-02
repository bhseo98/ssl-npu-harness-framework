<div align="center">

# 🦙 NPU Harness Framework — `torch-mlir-zoo`

**PyTorch Model Zoo + Torch-MLIR / IREE-Turbine** export — the top-most box
 of the target NPU compiler stack. *PyTorch-only `nn.Module` 정의,
두 export backend 공존, 모든 IR `server_side_op_hits = {}`.*

[![CI](https://github.com/bhseo98/ssl-npu-harness-framework/actions/workflows/ci.yml/badge.svg?branch=torch-mlir-zoo)](https://github.com/bhseo98/ssl-npu-harness-framework/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torch-MLIR](https://img.shields.io/badge/torch--mlir-LLVM-blueviolet)](https://github.com/llvm/torch-mlir)
[![IREE](https://img.shields.io/badge/IREE-3.11-orange)](https://github.com/iree-org/iree)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/bhseo98/ssl-npu-harness-framework/blob/main/LICENSE)

[**Quick start**](#-quick-start) ·
[**Status board**](development.md) ·
[**Lowering recipe**](docs/RECIPES-zoo.md) ·
[**Architecture**](docs/ARCHITECTURE.md) ·
[**Learning notes**](docs/easy-series/) ·
[**Contributing**](https://github.com/bhseo98/ssl-npu-harness-framework/blob/main/CONTRIBUTING.md) ·
[**Security**](https://github.com/bhseo98/ssl-npu-harness-framework/blob/main/SECURITY.md)

> **Branch map**: [`main`](../../tree/main) framework core ·
> [`voice-app`](../../tree/voice-app) STT/LLM/TTS ·
> [`vision-app`](../../tree/vision-app) ResNet18 classifier ·
> **`torch-mlir-zoo`** (you are here) **Phase 2 MLIR lowering — Step 5 ✅** (5/6 step, 2026-05-27)

</div>

---

## Why this exists

) calls for a
*PyTorch Model Zoo* that feeds the target NPU compiler stack's top-most layer
 — i.e., **PyTorch → Torch-MLIR IR**. The catch (PDF §2):

> *"LLM 모델링을 PyTorch만을 사용한 소스코드로 작성해야 할 필요성"*

HuggingFace Transformers / amdsharktank `PagedLlmModelV1` 둘 다 트레이싱
친화적이지 않거나 server-side primitives (PagedKVCache, ThetaLayer,
DeviceAffinity) 를 강제한다. **이 branch 는 자체 PyTorch `nn.Module` 만으로**
4 단위 op + on-device Llama-3.2-1B 를 재구성하고, **두 export backend**
(`torch_mlir.compile` + `iree.turbine.aot`) 로 MLIR 을 dump 한다.

```bash
git diff main torch-mlir-zoo -- src/npu_harness_framework/   # empty (framework core invariant)
```

## 🧩 Pipeline (4-stage, voice-app/vision-app 와 동일 패턴)

```
config payload  →  loader / tokenizer  →  exporter  →  analyzer  →  summary
                    (module, args)        .mlir text   op_counts +
                                                       server_side_op_hits
```

| Stage | `@register(...)` | 역할 |
|---|---|---|
| `loader` | `("loader", "zoo_op")` | 4 단위 op 중 하나 + dummy input 생성 |
| `tokenizer` / `model` | `("tokenizer", "hf_llama")` + `("model", "llama_on_device")` | Llama-3.2-1B path |
| `exporter` | **`("exporter", "torch_mlir_dialect")`** or **`("exporter", "iree_turbine")`** | 두 backend 공존 (sibling) |
| `analyzer` | `("analyzer", "ir_summary")` | op_counts, dtypes, **`server_side_op_hits` metric** |

## 🦾 Two export backends

| | `torch_mlir_dialect` | `iree_turbine` (2026-05-26 추가, Gap G) |
|---|---|---|
| Library | `torch_mlir.compile(..., OutputType.TORCH)` | `iree.turbine.aot.FxProgramsBuilder` + `aot.export` |
| Dialect | top-level torch only (`torch.aten.*`) | torch + `util.global` + `torch_c.from_builtin_tensor` mix |
| Weight | inlined as `torch.aten.*` args | `util.global.load @__auto.<name>` (external resource) |
| Downstream | target NPU compiler 직접 | IREE `iree-compile` → `.vmfb` 직접 |
| Source ref | torch-mlir LLVM | `nod-ai/amd-shark-ai` 의 export pipeline (transformer layer 정의는 사용 안 함) |

**같은 payload → 두 dialect 형태의 MLIR**. config 한 줄 변경으로 swap:

```yaml
# before
config: { type: torch_mlir_dialect, out_path: artifacts/zoo/attention.mlir }
# after
config: { type: iree_turbine, out_path: artifacts/zoo-iree/attention.mlir }
```

## 📦 Models in the zoo

| 카테고리 | 구현 | 출처 |
|---|---|---|
| 4 단위 op | `ScaledDotProductAttention`, `RMSNorm`, `SwiGLU`, `TopK` | [`src/torch_mlir_zoo/ops/`](src/torch_mlir_zoo/ops/) |
| LLM E2E | `LlamaOnDevice` (Llama-3.2-1B forward-only, no KV cache, no sampling) | [`src/torch_mlir_zoo/models/llama_on_device.py`](src/torch_mlir_zoo/models/llama_on_device.py) |
| HF weight loader | `load_hf_weights` (HF safetensors → on-device state_dict, prefix strip) | 동일 파일 |

amdsharktank 의 **transformer layer 정의** (`PagedLlmModelV1`, `ThetaLayer` 등) 는
**import 0** — server-side ↔ on-device 매핑은 [`docs/SHARK_AI_ANALYSIS.md`](docs/SHARK_AI_ANALYSIS.md) 의
표 reference.

## 🚀 Quick start

### Path A — venv-shark (iree.turbine backend, 권장)

PyPI 가 아닌 nightly index 가 필요해서 별도 격리 venv 권장:

```bash
# 1) uv 설치 (~/.local/bin)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Python 3.11 격리 venv (uv 가 python binary 자동 다운로드)
uv venv --python 3.11 ~/venv-shark
source ~/venv-shark/bin/activate

# 3) 본 repo + [shark] extras (iree-base-compiler/runtime/turbine nightly)
uv pip install -e .[shark] --find-links https://iree.dev/pip-release-links.html

# 4) 검증
pytest tests/test_iree_turbine_export.py -v   # 6 passed
python scripts/run_iree_turbine_export.py     # artifacts/zoo-iree/*.mlir 4개
```

### Path B — Docker dev shell (가상환경 접속용, *harness 권장*)

호스트에 `uv` / venv 셋업 없이 *셋업 끝난 환경에 진입*. **harness 의 철학에
가장 부합** — uniform contract 아래서 모델/실험을 자유롭게 swap. 개발 /
디버깅 / 반복 실험 / 외부 협업자와 동일 환경 보장.

```bash
# 처음 한 번 빌드 (Path C 의 이미지와 별도 — :dev tag)
docker compose build dev

# bash 진입 — 전체 repo 가 /workspace 에 마운트, 편집 즉시 반영
docker compose run --rm dev

# 안에서 자유롭게 명령
[torch-mlir-zoo:dev] /workspace $ pytest tests/test_iree_turbine_export.py -v
[torch-mlir-zoo:dev] /workspace $ python scripts/run_zoo_export.py
[torch-mlir-zoo:dev] /workspace $ ipython         # REPL
[torch-mlir-zoo:dev] /workspace $ jq . artifacts/*.summary.json

# 또는 한 줄로 임의 명령 실행 (bash 진입 안 함)
HF_TOKEN=$HF_TOKEN docker compose run --rm dev \
    python scripts/run_llama_export.py --config configs/zoo/llama_on_device_iree_turbine.yaml
```

`Dockerfile.dev` 가 추가하는 dev 도구: `vim-tiny`, `jq`, `less`, `git`, `curl`,
`procps`, `bash-completion`, `ipython`. HF 모델 캐시는 `hf_cache` named volume
에 persistent.

### Path C — Docker (실행 컨테이너, CI / 외부 데모용)

일회성 pytest / export script 자동 실행. CI / 외부 데모 / 자동화 파이프라인용.

```bash
docker build -t torch-mlir-zoo:latest .
docker run --rm -v "$PWD/artifacts:/workspace/artifacts" torch-mlir-zoo:latest                              # unit-op tests
docker run --rm -v "$PWD/artifacts:/workspace/artifacts" torch-mlir-zoo:latest python scripts/run_zoo_export.py
docker run --rm -e HF_TOKEN="$HF_TOKEN" \
       -v "$PWD/artifacts:/workspace/artifacts" torch-mlir-zoo:latest python scripts/run_llama_export.py "Hello"
```

또는 `compose.yaml` 의 `test` service:

```bash
docker compose run --rm test    # pytest 한 번
```

## 📊 Quantitative result (Gap G, 2026-05-26)

5 모델 모두 `iree.turbine` export 통과, `server_side_op_hits = {}` 일관:

| 모델 | MLIR lines | top op | server-side hits |
|---|---:|---|---:|
| `rmsnorm` (1×32×512) | 31 | pow/mean/add/rsqrt/mul | 0 |
| `topk` (1×32000) | 10 | topk | 0 |
| `mlp` SwiGLU (1×32×512) | 65 | transpose/view/mm/silu/mul | 0 |
| `attention` SDPA+GQA | 204 | view/transpose/mm/bmm/softmax/where/triu | 0 |
| `LlamaOnDevice` TINY (2-layer) | **810** | embedding/view/transpose/mm/add/expand/bmm/silu | **0** |

비교 — amdsharktank `PagedLlmModelV1` 의 toy 3-block IRPA 동일 export:

| | 본 zoo (`LlamaOnDevice` 자체) | amdsharktank (`PagedLlmModelV1`) |
|---|---|---|
| MLIR lines (toy) | **810** (TINY 2-block) | **4859** (3-block) |
| `paged_attention_kv_cache_gather` | **0회** | **7회 출현** |
| `rope_select_concat_*` | 0회 (standard ops 분해) | 출현 |
| Entry function | `forward` 1개 | `prefill_bs4` + `decode_bs4` 2개 |

→ **on-device 친화 IR 의 정량 입증**. 자세히는 (내부 문서).

> real Llama-3.2-1B (toy 아닌 16-layer 전체) 의 **27,269-line** amdsharktank IR 을
> 클래스와 1:1 로 분해한 분석은 [`docs/easy-series/easy4-detail.md`](docs/easy-series/easy4-detail.md),
> 그림은 [`easy5-fig.md`](docs/easy-series/easy5-fig.md) + [`export-vs-lowering.drawio`](docs/diagrams/export-vs-lowering.drawio).

## 🏛 Architecture

전체 architectural view — [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
주요 섹션:

- §1 Overview + 6 단계 매핑
- §3 Compiler stack 
- §4 Framework core invariant
- §5 PyTorch Model Zoo (자체 nn.Module + amdsharktank 분리)
- §6 Two export backends (sibling, additive)
- §9 Gap G 정량 검증 결과
- §10 설계 검토 §1-§5 매핑

## 📚 Learning notes — [`docs/easy-series/`](docs/easy-series/)

PyTorch/HF 모델을 이 stack 으로 lowering 할 때 *무엇이 되고 무엇이 막히는지* 를
실측 기반으로 풀어쓴 학습 노트. 처음이면 [`easy-sum.md`](docs/easy-series/easy-sum.md) (한 장 요약) 부터.

| 문서 | 한 줄 |
|---|---|
| [easy-sum](docs/easy-series/easy-sum.md) | 한 장 요약 — 제약 9종 ↔ 고칠 코드, "modelClass 교체 = 6개 동시 제거" 결론 |
| [easy](docs/easy-series/easy.md) / [easy2](docs/easy-series/easy2.md) | `LlamaOnDevice` MLIR→`.vmfb` 성공 / HF 모델 7종 입고 기준 |
| [easy3](docs/easy-series/easy3.md) | amdsharktank real Llama-3.2-1B export 제약 9종 → 1 근본원인 |
| [easy4](docs/easy-series/easy4.md) + [easy4-detail](docs/easy-series/easy4-detail.md) | export/lowering 에 쓴 클래스 27종 해부 + 실측 IR(27,269줄) op 1:1 매핑 |
| [easy5-fig](docs/easy-series/easy5-fig.md) | 그림판 — [`export-vs-lowering.drawio`](docs/diagrams/export-vs-lowering.drawio) (3페이지) |

## 🔄 Adding / swapping

### 새 단위 op 추가

1. [`src/torch_mlir_zoo/ops/`](src/torch_mlir_zoo/ops/) 에 표준 `nn.Module` 한 클래스 작성.
2. [`stages.py`](src/torch_mlir_zoo/stages.py) 의 `_OP_REGISTRY` dict 에 한 줄 추가.
3. `configs/zoo/<op>.yaml` (torch_mlir backend) + `configs/zoo/<op>_iree_turbine.yaml` (iree.turbine backend) 동시 생성.
4. `tests/test_zoo_export.py` 또는 `tests/test_iree_turbine_export.py` 에 케이스 추가.

### 새 LLM architecture 추가

1. `models/<name>_on_device.py` 작성 (자체 nn.Module + `load_hf_weights`).
2. `stages.py` 에 `@register("model", "<name>")` Loader 한 클래스 추가.
3. `configs/zoo/<name>.yaml` 4-stage pipeline.
4. `scripts/run_<name>_export.py` (선택사항).

framework core / 기존 ops / 기존 backend 모두 변경 0.

## 🗺 Next milestones

| 항목 | 우선순위 | PDF 매핑 |
|---|---|---|
| real Llama-3.2-1B HF download + `LlamaOnDevice.load_hf_weights` + `iree.turbine` E2E | High | Step 5 |
| `dense_resource` inline → `flow.parameter.named` external | Medium | Step 5 후속 |
| Q8_0 GGUF (`unsloth/Llama-3.2-1B-Instruct-GGUF`) 와 자체 path 비교 | Medium | Step 6 |
| Whisper-tiny 자체 PyTorch 구현 + 2-backend export | Medium | Step 6 |
| `iree-compile` → `.vmfb` + IREE runtime 검증 | Medium | PDF IREE 박스 |

## License

MIT — see [`LICENSE`](https://github.com/bhseo98/ssl-npu-harness-framework/blob/main/LICENSE) on the `main` branch.

## Citation

```bibtex
@software{npu_harness_framework_torch_mlir_zoo_2026,
  author  = {Seo, BoHyun},
  title   = {torch-mlir-zoo: PyTorch Model Zoo for the target NPU Compiler Stack},
  year    = {2026},
  url     = {https://github.com/bhseo98/ssl-npu-harness-framework/tree/torch-mlir-zoo}
}
```
