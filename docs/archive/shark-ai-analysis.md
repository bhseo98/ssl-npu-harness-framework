> ⚠ **ARCHIVED (구버전, ~2026-05, superseded).** 최신 현황은 [status](../status.md) · 문서 지도 [docs/README](../README.md). 본문 링크는 원래 위치 기준일 수 있음.

# AMD SHARK-AI (amdsharktank) — Server-side LLM 분석 (Step 3)

> 본 문서는 외부 NPU compiler 측 PDF  의 **Step 3**
> ("AMD-SHARK-AI 분석 — 서버향 LLM 모델링, 이를 On-Device 로 바꿔서 porting
> 하기 위해서 어떤 변화가 필요할지 분석") 의 정량/정성 deliverable.
>
> 분석 대상: `github.com/nod-ai/amd-shark-ai` 의 `amdsharktank` sub-package
> (구 sharktank). 본 zoo (`torch-mlir-zoo` branch) 의 Step 4 (단위 op) 와
> Step 5 (Llama-3.2-1B on-device) 의 선정 근거를 제공한다.

---

## 1. amdsharktank 의 위치 — target NPU compiler stack 상

PDF 다이어그램에서 본 zoo 가 위치하는 영역과 amdsharktank 의 관계:

| Compiler stack 계층 | amdsharktank | 본 `torch-mlir-zoo` |
|---|---|---|
| PyTorch Model Zoo (최상단) | server-side LLM 코드 (PagedAttention + Theta wrapping + sharding) | **on-device** PyTorch-only 모듈 (표준 nn.Module) |
| Torch-MLIR | iree-turbine `aot.export` + Theta-aware export | `torch_mlir.compile(..., OutputType.TORCH)` 직접 |
| IREE / target NPU 하위 | 본 PR scope 밖 (다음 phase) | 본 PR scope 밖 |

> **결론**: amdsharktank 는 본 zoo 의 *그대로 가져다 쓰는* 의존성이 아니라,
> **server-side 의 patterns 를 보여주는 reference**. 본 zoo 는 이를 *제거/단순화*한
> on-device 등가물을 PyTorch 표준 ops 로 직접 구현한다.

---

## 2. amdsharktank 의 핵심 구성요소 (분석)

### 2.1 `amdsharktank.layers.paged_attention` — KV-cache paging

| 클래스 | 역할 | on-device 부적합 사유 |
|---|---|---|
| `PagedKVCache(KVCache, ABC)` | KV 캐시의 추상 base | 페이지 단위 메모리 할당 (vLLM-style) — 임베디드 NPU 의 통합 SRAM/DRAM 모델과 충돌 |
| `DefaultPagedKVCache(PagedKVCache)` | 기본 페이지 KV | 동일 |
| `PipelinedPagedKVCache(PagedKVCache)` | pipeline-parallel KV | 단일 NPU 에서 무의미 |
| `PagedMHAttention(PagedAttention)` | Multi-Head paged | paging 자체 제거 → 표준 SDPA |
| `PagedGQAttention(PagedMHAttention)` | Grouped-Query paged | GQA 자체는 유용, paging 만 제거 |
| `PagedMLAttention(PagedMHAttention)` | Multi-Latent (DeepSeek-V2 류) paged | Llama-3.2 와 무관 — out of scope |

**`from amdsharktank.kernels.mlir_kernel import *`** 의 흔적: server-side
fused/handwritten kernel 사용. on-device 에서는 standard `torch.aten.*` 로 분해.

### 2.2 `amdsharktank.layers.norm.RMSNormLayer`

- Base: `ThetaLayer` (자체 weight 관리 추상화)
- Forward: float32 로 cast → RMS normalize → 원본 dtype 으로 복귀 (dtype 안정성)
- **on-device 변경점**: `nn.Module` 로 단순화, `nn.Parameter` 사용. dtype 보호는
  cast logic 만 유지.

### 2.3 `amdsharktank.layers.ffn_block.FFN`

- Base: `ThetaLayer`
- Forward (gated): `down(activation(gate(h)) * up(h))` — SwiGLU
- Activation: `F.silu` (default)
- **on-device 변경점**: `ThetaLayer` 제거, `nn.Linear * 3` 으로 직접 구성.
  Llama 의 기본 FFN.

### 2.4 `amdsharktank.models.llm.llm.PagedLlmModelV1`

- Base: `BaseCausalLMModel`
- Forward 가 **prefill + decode 2-단계** split (`prefill()` + `decode()`)
- 인자: `tokens, seq_lens, start_positions, seq_block_ids, cache_state` — paged
  KV 의 모든 metadata 노출
- **on-device 변경점**: 단일 `forward(input_ids) -> logits` 로 통합. seq_len 은
  고정 (no KV-cache → re-compute every step, 본 PR 의 forward-only scope).

### 2.5 `amdsharktank.models.llm.export.ServicePagedLlmModelV1`

- `iree.turbine.aot.DeviceAffinity` import — multi-device / sharding
- `prefill()` 내부에 top-k sampling logic — control flow 발생, traceable
  export 어려움
- **on-device 변경점**: sampling/serving 제거. forward 만 trace.

---

## 3. Server-side → on-device 매핑 (본 zoo Step 4-5 가 채울 자리)

| amdsharktank (server-side) | 본 `torch_mlir_zoo` (on-device) | 본 PR 위치 |
|---|---|---|
| `PagedMHAttention` / `PagedGQAttention` (paging) | `ops/attention.py::ScaledDotProductAttention` (no paging, no KV-cache) | Step 4 (commit #2) |
| `RMSNormLayer(ThetaLayer)` | `ops/rmsnorm.py::RMSNorm(nn.Module)` | Step 4 |
| `FFN(ThetaLayer)` (SwiGLU) | `ops/mlp.py::SwiGLU(nn.Module)` | Step 4 |
| (sampling logit) | `ops/topk.py::TopK(nn.Module)` (forward-only, no autoregressive loop) | Step 4 |
| `PagedLlmModelV1.prefill()+decode()` | `models/llama_on_device.py::LlamaOnDevice.forward(input_ids)` | Step 5 (commit #3) |
| `iree.turbine.aot.DeviceAffinity` | (제거 — 단일 NPU) | n/a |
| `ServicePagedLlmModelV1` (sampling control flow) | (제거 — forward only) | n/a |

본 zoo 의 단위 op 는 SHARK 코드를 **직접 import 하지 않음** — Apache-2.0
라이선스 attribution 은 본 문서로 충족하고, 코드는 표준 PyTorch 로 재구현.

---

## 4. "Top-level MLIR 그래프 추출" 의 실행 계획

Step 3 의 "top level 에서 MLIR 그래프 뽑아보고 분석" 은 본 PR 에서 다음
산출물로 실현된다:

| Step | 산출물 | MLIR 추출 대상 |
|---|---|---|
| 3 (호스트) | `scripts/run_sharktank_dump.py` placeholder | amdsharktank `PagedMHAttention` (docker-shell 에서 amdsharktank build 가능 시 실행) |
| 4 (commit #2) | `artifacts/zoo/{attention,rmsnorm,mlp,topk}.mlir` | 본 zoo 단위 op 4종 |
| 5 (commit #3) | `artifacts/llama-3.2-1b-on-device.mlir` | 본 zoo on-device Llama (Llama-3.2-1B weight 주입) |

**대비 (verification)**: Step 5 의 `summary.json` 의 `server_side_op_hits` 는
**0** 이어야 한다. amdsharktank 의 dump (host 미가능 시 reference docs 만) 와
대비하여 on-device 적합성 입증.

---

## 5. amdsharktank install / dump — 2026-05-26 setup 완료

원래 절차 (2026-05-20 시점) 에서는 source build only / hip kernels 등 환경
부담이 컸으나, 2026-05-26 에 **export pipeline 만 격리 사용** path 로 setup:

- venv: `/home/bohyun/venv-shark` (Python 3.11.15, uv 로 구축).
- 설치: `uv pip install -e /home/bohyun/amd-shark-ai/amdsharktank` (editable, sibling clone).
- 핵심 deps: `torch==2.5.1`, `iree-base-compiler==3.11.0rc20260316`,
  `iree-base-runtime==3.11.0rc20260316`, `iree-turbine==3.10.0rc20260316`.
- **사용 layer 분리** (사용자 2026-05-26 지시 반영):
  - **transformer layer 정의** (`amdsharktank/layers/`, `models/llm/llm.py` 의
    `PagedLlmModelV1` 등) — *직접 import 안 함*. 본 zoo 의 자체 `LlamaOnDevice`
    와 `ops/{attention,rmsnorm,mlp,topk}` 로 대체.
  - **export pipeline** (`iree.turbine.aot.FxProgramsBuilder` + `aot.export`) —
    *사용*. 새 backend [`src/torch_mlir_zoo/exporters/iree_turbine_export.py`](iree_turbine_export.py).
- 검증 (2026-05-26):
  - amdsharktank model path: `toy_llama.irpa` (1.5MB) → `export_paged_llm_v1` →
    `toy_llama_export.mlir` (428KB, 4859 lines, `paged_attention_kv_cache_gather`
    custom util.func + prefill_bs4 + decode_bs4) — server-side IR 확인.
  - 자체 model path: TINY `LlamaOnDevice` → `iree_turbine` exporter →
    `llama_tiny.mlir` (810 lines, server_side_op_hits=0, standard torch.aten ops 만) —
    on-device IR 확인.
  - 5.3x line count 차이 + paged_attention custom op 의 유무 가 PDF §3 의
    "server-side → on-device porting" 가설을 IR 수준에서 정량 검증.
- 보고서: [`../REPORT-2026-05-26-amd-shark-integration.md`](../REPORT-2026-05-26-amd-shark-integration.md).

---

## 6. 참고

- amdsharktank source: <https://github.com/nod-ai/amd-shark-ai/tree/main/amdsharktank>
- 분석 대상 파일 (web 으로 직접 읽음):
  - [layers/paged_attention.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/paged_attention.py)
  - [layers/norm.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/norm.py)
  - [layers/ffn_block.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/ffn_block.py)
  - [models/llm/llm.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/llm.py)
  - [models/llm/export.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/export.py)
- License: Apache-2.0 (분석 reference, 코드 copy 없음)
