# Easy Guide 4 — amdsharktank 로 Llama-3.2-1B-Instruct 를 export/lowering 할 때 *쓴 클래스/코드* 해부

> 본인 학습용 노트 — easy3.md 의 짝.
>
> easy3 = "amdsharktank server-side Llama 를 lowering 하면 **무엇이 막히는가** (제약 9종 → 1 근본원인)". 결과(surface)를 셌다.
> **easy4 = 그 surface 를 *만들어내는 클래스/코드* 를, HF safetensors → IRPA → MLIR 로 가는 호출 순서 그대로 해부**. "어떤 클래스가 어떤 줄에서 무엇을 IR 에 박는가" 의 지도.

---

## 0. 범위 못 박기 (중요)

이 문서는 **순수 `meta-llama/Llama-3.2-1B-Instruct` 를 amdsharktank 의 production export 경로로 MLIR 까지 내린 *그 한 경로* 만** 다룬다.

- ✅ 다룸: `import_hf_dataset` → `export_paged_llm_v1` 가 부르는 amdsharktank 클래스 전부 (server-side 정의 그대로)
- ❌ 안 다룸: 우리 `LlamaOnDevice` / `ops/*` / `iree_turbine_export.py` 같은 **on-device 자체 경로** — 그건 easy.md / easy2.md / (내부 문서) 소관. 여기서는 비교/대조도 최소화.

> 즉 이 문서의 모든 클래스는 `/home/bohyun/amd-shark-ai/amdsharktank/amdsharktank/` (sibling clone) 의 **amdsharktank 원본**이다. 우리 레포 코드가 아님. 우리는 이걸 *import 하지 않고* (가드: [`docs/SHARK_AI_ANALYSIS.md`](../SHARK_AI_ANALYSIS.md) §5) 그저 CLI 로 돌려 MLIR 를 뽑아 *읽었다*.

재현 셋업 (easy3 §1 과 동일):

```bash
source /home/bohyun/venv-shark/bin/activate            # Python 3.11.15
SNAP=~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/

# (A) HF safetensors → IRPA
python -m amdsharktank.tools.import_hf_dataset \
  --config-json ${SNAP}config.json --params ${SNAP}model.safetensors \
  --output-irpa-file /tmp/llama32-irpa/llama-3.2-1b.irpa
# (A') tied embedding fix → llama-3.2-1b-tied.irpa  (§1.4)

# (B~E) IRPA → MLIR
python -m amdsharktank.examples.export_paged_llm_v1 \
  --irpa-file /tmp/llama32-irpa/llama-3.2-1b-tied.irpa \
  --output-mlir /tmp/llama32-irpa/llama-3.2-1b.mlir \
  --output-config /tmp/llama32-irpa/llama-3.2-1b.json \
  --bs-prefill 1 --bs-decode 1 \
  --activation-dtype float16 --attention-dtype float16
```

산출물: `llama-3.2-1b.mlir` (27,269 lines, 2.4 MB) — 이 IR 안의 모든 줄이 아래 클래스들이 남긴 흔적이다.

---

## 1. Stage A — HF safetensors → IRPA : *weight 를 그래프 밖으로 빼는* 클래스들

진입점은 도구 모듈 [`amdsharktank/tools/import_hf_dataset.py`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/tools/import_hf_dataset.py). 얇은 CLI 껍데기이고, 실제 일은 `amdsharktank.utils.hf.import_hf_dataset()` 가 한다.

### 1.1 핵심 타입 3종 — `Theta` / `Dataset` / `DatasetMetadata`

| 클래스 | 파일:line | 무엇인가 | export 에서의 의미 |
|---|---|---|---|
| `Theta` | [`types/theta.py:83`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/types/theta.py) | named tensor 트리. `theta("blk", 0, "attn_q", "weight")` 처럼 **이름 경로**로 weight 접근 | 모델 코드가 weight 를 *위치*가 아니라 *이름*으로 집는다 → IR 에 `@__auto.blk.0.attn_q.weight` 같은 symbol 로 박힘 |
| `Dataset` | [`types/theta.py:372`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/types/theta.py) | `root_theta`(텐서) + `properties`(하이퍼파라미터) 묶음. `.save()`/`.load()` 가 IRPA I/O | IRPA 파일 ⇄ 메모리. `Dataset.load(irpa)` 가 export 의 입력 |
| `DatasetMetadata` | [`types/theta.py:434`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/types/theta.py) | IRPA 안의 key 스킴: `__AMD_SHARK_DATASET__` / `__AMD_SHARK_INFERENCE_TENSORS__` / `__AMD_SHARK_SHARD_RANKS__` | "fixed IRPA" 의 실체 = 이 properties 에 모든 runtime 옵션이 고정 저장 |

`Theta.__call__` ([theta.py:206](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/types/theta.py)) 가 sub-Theta 든 leaf tensor 든 이름으로 돌려주는 게 핵심 — 모델 `__init__` 이 전부 이걸로 weight 를 받는다 (§3).

### 1.2 IRPA = IREE/SHARK Runtime Parameter Archive

- **무엇**: amdsharktank 자체 weight 직렬화 포맷. weight + GGUF-style 하이퍼파라미터 메타를 한 파일에.
- **왜**: (1) weight 를 IR 에 inline 하면 IR 가 GB 급으로 폭발 → 외부 파일로 빼고 IR 엔 `util.global` 이름참조만. (2) dtype/shape/quant scheme 을 코드와 분리.
- IR 에서의 모습 (실측):
  ```mlir
  %__auto.token_embd.weight =
    util.global.load @__auto.token_embd.weight : tensor<128256x2048xf16>
  %0 = torch_c.from_builtin_tensor %__auto.token_embd.weight
         : tensor<128256x2048xf16> -> !torch.vtensor<[128256,2048],f16>
  ```
- 런타임 binding: `iree-run-module --parameters=foo.irpa`. → **IRPA 는 weight ↔ MLIR 의 link table.**

### 1.3 Llama-3.2-1B IRPA properties 실측 (= 이 모델의 "설계도")

```
general.architecture=llama,  block_count=16,  embedding_length=2048,
attention.head_count=32,  attention.head_count_kv=8 (GQA),  attn_head_dim=64,
feed_forward_length=8192,  vocab_size=128256,  context_length=2048,
rope.freq_base=500000,  rope.interleave_emb=True,
# export 가 도로 적어넣는 runtime 옵션 (fixed IRPA):
activation_dtype, attention_dtype, attention_kernel, block_seq_stride,
kv_cache_type, fake_quant, tensor_parallelism_size, parallelism_config, use_qk_norm
```

### 1.4 변환 시 실제로 깨졌던 것 — tied embedding (`DefaultPrimitiveTensor`)

Llama-3.2-1B 은 `config.json` 의 `tie_word_embeddings: True` → safetensors 에 `lm_head.weight` 가 *물리적으로 없다*. 그런데 `PagedLlmModelV1.__init__` 은 `theta("output")` (lm_head) 를 **필수**로 찾는다 (§3.1) → `KeyError ['output']`.

수선에 쓴 클래스: [`DefaultPrimitiveTensor`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/types/tensors.py) (= `InferenceTensor` 의 평범한 텐서 구현).

```python
from amdsharktank.types.theta import Dataset, Theta
from amdsharktank.types.tensors import DefaultPrimitiveTensor

ds = Dataset.load("llama-3.2-1b.irpa")
flat = ds.root_theta.flatten()                       # {이름: InferenceTensor}
te = flat["token_embd.weight"].as_torch()            # [128256,2048] bf16
flat["output.weight"] = DefaultPrimitiveTensor(      # tied: 복제해서 lm_head 채움
    name="output.weight", data=te.clone())
Dataset(properties=ds.properties, root_theta=Theta(flat)).save("llama-3.2-1b-tied.irpa")
```

→ `Theta.flatten()` ([theta.py:152](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/types/theta.py)) 로 트리를 평탄화 → dict 수선 → `Theta(flat)` 로 재조립. **IRPA 를 쓴다고 자동이 아님을 보여주는 클래스-레벨 증거.**

---

## 2. Stage B — IRPA → 설정 클래스 : *runtime 옵션을 코드로 굳히는* 클래스들

`export_paged_llm_v1.main()` ([export_paged_llm_v1.py:237](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/examples/export_paged_llm_v1.py)) 가 CLI flag + IRPA properties 를 합쳐 3개의 config 객체를 만든다.

| 클래스 | 파일:line | 역할 | 우리가 준 값 |
|---|---|---|---|
| `LlamaHParams` | [`layers/configs/llm_configs.py:56`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/configs/llm_configs.py) | 순수 아키텍처 하이퍼파라미터 (block_count, head_count, rope_freq_base …) | IRPA properties 에서 자동 로드 |
| `LlamaModelConfig` | [`llm_configs.py:562`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/configs/llm_configs.py) | hp + **lowering 정책** (`kv_cache_type="paged"`, `block_seq_stride=32`, `activation_dtype`, `attention_dtype`, `attention_kernel`, `matmul_kernel`, `tensor_parallelism_size`) | `--activation-dtype float16 --attention-dtype float16` |
| `ExportConfig` | [`models/llm/config.py:45`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/config.py) | export 행위 옵션 (`bs_prefill`, `bs_decode`, `top_k`, `logits_normalization`, `use_linalgext_topk`, `device_block_count=512`) | `--bs-prefill 1 --bs-decode 1` (나머지 default) |
| `ParallelismConfig` | [`layers/configs`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/configs/) | tensor/pipeline 병렬도 | `default_config(tp=1, pp=1)` → single device |

핵심 호출:
```python
dataset      = cli.get_input_dataset(args)            # = Dataset.load(irpa)
llama_config = LlamaModelConfig.from_dataset(dataset, attention_kernel=..., **dtype_flags)
export_config= ExportConfig(bs_prefill=[1], bs_decode=[1], ...)
```

`LlamaModelConfig.from_dataset` ([llm_configs.py:707](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/configs/llm_configs.py)) 가 IRPA properties → 타입화된 config 로 굳히는 곳. **easy3 의 제약 #6(fp16 mixed), #7(dynamic dim), #8(multi-args) 이 다 여기 default 에서 결정**된다.

`if args.use_hf:` ([:281](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/examples/export_paged_llm_v1.py)) 의 `rope_interleave_emb=False` override 가 바로 easy3 §2.3 "fixed IRPA" TODO 의 그 한 줄.

---

## 3. Stage C — 모델 그래프 조립 : `PagedLlmModelV1` 과 그 서브모듈

`export_llm_v1()` ([export_paged_llm_v1.py:50](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/examples/export_paged_llm_v1.py)):
```python
model = PagedLlmModelV1(theta, llama_config)          # ← 진짜 모델
model = ServicePagedLlmModelV1(model=model, config=export_config)  # ← export 래퍼 (§4)
```

### 3.1 `PagedLlmModelV1(BaseCausalLMModel)` — 메인 모델

[`models/llm/llm.py:33`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/llm.py). `__init__` 이 `theta(...)` 로 weight 를 집어 서브모듈을 짠다:

| 줄 | 서브모듈 (클래스) | weight 경로 | 역할 |
|---|---|---|---|
| llm.py:82 | `build_cache_from_config` → **`DefaultPagedKVCache`** | — | KV cache slab (§3.3) → 제약 #1 |
| llm.py:84 | **`TokenEmbeddingLayer`** | `theta("token_embd")` | 토큰 → hidden |
| llm.py:88 | `build_rotary_layer` → **`CachedRotaryLayer`** | — | RoPE (§3.2) → 제약 #4 |
| llm.py:101 | **`RMSNormLayer`** | `theta("output_norm")` | 최종 norm |
| llm.py:107 | **`LinearLayer`** | `theta("output")` | lm_head ← **tied 가 여기서 필요** (§1.4) |
| llm.py:114 | **`AttentionFFNBlock`** × 16 | `theta("blk", n)` | 트랜스포머 블록 |

`forward` 가 아니라 **`prefill()` ([:127](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/llm.py)) 과 `decode()` ([:182](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/llm.py)) 두 메서드**로 갈라져 있다 — 인자가 `tokens, seq_lens, seq_block_ids, cache_state, start_positions`. → **easy3 제약 #2(prefill/decode 분리) + #8(multi-args contract) 의 코드 출처가 정확히 이 두 시그니처.**

### 3.2 RoPE 레이어 체인 — `CachedRotaryLayer` → `RotaryEmbeddingLayer` → `apply_rotary_embedding`

```
build_rotary_layer()  [rotary_embedding.py:72]
   └─ CachedRotaryLayer(BaseLayer)        [rotary_embedding.py:18]  ← sincos table 캐시
        └─ RotaryEmbeddingLayer           [rotary_embedding_hf.py:107]
             └─ (forward 시) apply_rotary_embedding  CustomOp  [kernels/rotary.py:15]  ← §6.2
```

`CachedRotaryLayer.forward` 는 position → sincos table 을 만들고 `RotaryEmbeddingLayer` 에 넘긴다. 이게 IR 에서 `rope_select_concat_*` custom util.func (65 hit) 으로 굳는다.

### 3.3 KV cache — `DefaultPagedKVCache(PagedKVCache(KVCache))`

[`layers/paged_attention.py:148`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/paged_attention.py). `allocate(page_count)` 가 **2D slab** `[page_count, page_slab_flat_dims]` 을 만든다 — 6D 뷰(`[block, K/V, head, seq_stride, head_dim]`)를 flatten 한 것. Llama-3.2-1B 에서 `page_slab_flat_dims = 16×2×8×32×64 = 524288` → easy3 의 그 `[?,524288]` slab.

### 3.4 Attention 블록 — `AttentionFFNBlock` → `create_paged_llama_attention_block` → GQA

`AttentionFFNBlock(ThetaLayer)` ([llm.py:244](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/llm.py)) 가 attn + ffn 을 묶는다. attn 은 factory 로 만든다:

```
create_paged_llama_attention_block()  [paged_llama_attention_block.py:481]
   └─ attn_type_map["llama"] == "gqa"
        └─ PagedLlamaGQAttentionBlock          ← Llama-3.2-1B 은 GQA (head 32 / kv-head 8)
             └─ (내부) PagedGQAttention(PagedMHAttention)  [paged_attention.py:949]
                  └─ paged_attention() → kv_cache_gather (@mlir_kernel)  ← §6.1
```

FFN 쪽은 `n_dense_layers`/MoE 여부로 갈리는데 Llama-3.2-1B 은 비-MoE → **`FFN(ThetaLayer)`** ([ffn_block.py:23](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/ffn_block.py), SwiGLU: `down(silu(gate(h))*up(h))`).

### 3.5 공통 베이스 — `ThetaLayer(BaseLayer)`

[`layers/base.py:347`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/base.py). 위 `RMSNormLayer / LinearLayer / FFN / TokenEmbeddingLayer / AttentionFFNBlock / MoeBlock` 이 전부 `ThetaLayer` 상속 = **weight 를 `Theta` 이름경로로 받는 nn.Module**. 우리 on-device 가 `nn.Module` + `nn.Parameter` 로 *단순화*해 갈아치운 바로 그 추상화 ([SHARK_AI_ANALYSIS](../SHARK_AI_ANALYSIS.md) §2).

---

## 4. Stage D — 서비스 래핑 : `ServicePagedLlmModelV1`

[`models/llm/export.py:47`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/export.py). `PagedLlmModelV1` 을 감싸 **export 가능한 `torch.nn.Module`** 로 만든다. 책임:

| 메서드 | 하는 일 | IR 흔적 / 제약 |
|---|---|---|
| `prefill()` / `decode()` ([:57](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/export.py)/[:100](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/export.py)) | 모델 호출 후 `ops.unshard` + logits_normalization(softmax/log_softmax) + `top_k`/`argmax` 분기 | top_k 켜면 `iree_linalg_ext.topk` 등장 (easy3 §8) |
| `setup_cache()` ([:151](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/export.py)) | `model.cache.allocate(512)` → cache slab + `page` dynamic dim + `device_affinities` | 제약 #1, #7 |
| `setup_arg_devices()` ([:137](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/export.py)) | 각 인자에 `DeviceAffinity` 부여 | 제약 #5 (`iree.abi.affinity`) |

여기서 두 보조 클래스가 IR 에 ABI 메타를 박는다:

- **`CacheAllocation`** ([layers/kv_cache.py:16](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/kv_cache.py)) — slab 텐서 리스트 + 각 텐서의 `DeviceAffinity` 묶음. `cache_state = CacheAllocation(allocation=cs)` 로 export 함수 안에서 재구성.
- **`DeviceAffinity`** (`iree.turbine.aot`) — "이 인자/결과는 `@__device_0` 에 산다"를 IR attribute 로. single-NPU 엔 무의미한 metadata → easy3 #5.

> `build_service_config()` ([export.py:169](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/export.py)) 는 별도로 `ServiceConfig`/`KVCacheConfig` (shortfin 서버용 `config.json`) 를 만든다. MLIR 자체엔 안 들어가지만 "이 IR 를 서버가 어떻게 호출해야 하는가"의 계약서.

---

## 5. Stage E — export : `FxProgramsBuilder` + `aot.export`

여기가 **PyTorch → MLIR lowering 의 실제 엔진**. 전부 `iree.turbine.aot` (= amd-shark 의 export pipeline, 우리가 유일하게 *빌려 쓴* 부분).

```python
from iree.turbine.aot import *                          # export_paged_llm_v1.py:14
fxb = FxProgramsBuilder(model)                           # :54

@fxb.export_program(name="prefill_bs1",
    args=(tokens, seq_lens, seq_block_ids, cache),
    dynamic_shapes=dynamic_shapes, arg_device=arg_devices, strict=False)  # :138
def _(model, tokens, seq_lens, seq_block_ids, cs):
    cache_state = CacheAllocation(allocation=cs)
    return model.prefill(tokens, None, seq_lens, seq_block_ids, cache_state)

# decode_bs1 도 동일 패턴으로 등록 (:184)

output = export(fxb, import_symbolic_shape_expressions=True)  # :232
output.save_mlir(args.output_mlir)                            # :323
```

| 클래스/함수 | 역할 | 이 모델에서 만든 것 |
|---|---|---|
| `FxProgramsBuilder` | 한 nn.Module 에 **여러 entry-point** 를 등록하는 빌더 | `prefill_bs1` + `decode_bs1` 두 함수 → 제약 #2 |
| `@fxb.export_program` | `torch.export` 로 그 함수를 FX 그래프로 trace | 4-args 시그니처가 그대로 함수 인자로 → #8 |
| `torch.export.Dim` ([export_paged_llm_v1.py:65](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/examples/export_paged_llm_v1.py)) | dynamic 차원 선언 (`seq_len_blocks_dim`, `page`) | `vtensor<[1,?]>`, slab `[?,524288]` → #7 |
| `aot.export(...)` | FX 그래프 → MLIR (`ExportOutput`). `import_symbolic_shape_expressions=True` 로 symbolic shape 유지 | torch dialect + util/iree_linalg_ext mix |
| `ExportOutput.save_mlir` | IR 를 텍스트로 저장 | `llama-3.2-1b.mlir` (27,269 lines) |

> 우리 on-device 경로의 [`exporters/iree_turbine_export.py`](../../src/torch_mlir_zoo/exporters/iree_turbine_export.py) 가 쓰는 게 **바로 이 세 줄**(`FxProgramsBuilder`+`export_program`+`export`)의 최소판. 차이는 *어떤 module 을 태우느냐* 뿐 — amdsharktank 는 `ServicePagedLlmModelV1`(서버), 우리는 `LlamaOnDevice`(stateless).

---

## 6. lowering 시 IR 에 *박히는* custom kernel 2종 — 막힘의 코드 출처

표준 `torch.aten.*` 로 안 풀리고 amdsharktank 가 **손수 짠 MLIR 템플릿**을 IR 에 직접 삽입하는 두 지점. easy3 §3·§4 의 "왜 안 되나" 의 코드 원본.

### 6.1 `KVCacheGatherKernel` (`@mlir_kernel`) → `iree_linalg_ext.gather`

[`layers/paged_attention.py:68`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/paged_attention.py). `@mlir_kernel` 데코레이터가 page_ids 기반 KV gather 를 IREE 확장 op 로 inject:

```mlir
util.func private @paged_attention_kv_cache_gather_..._f16(%cache, %page_ids, %t_id, %p_id) -> !result {
  %cache_slice = tensor.extract_slice %cache [...]              // 현재 block/partition
  %result = iree_linalg_ext.gather dimension_map=[0]           // ← 표준 MLIR 아님
            ins(%cache_slice, %page_ids) outs(%empty) -> !result
}
```

→ `iree_linalg_ext.gather` 는 **IREE 전용 extension dialect**. 우리 NPU 컴파일러가 이 dialect 의 lowering rule 을 모르면 거기서 stop. + page_ids data-dependent gather (정적 분석 비친화). real Llama 에서 33 hit.

### 6.2 `apply_rotary_embedding` (`CustomOp`) → `linalg.generic`

[`kernels/rotary.py:15`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/kernels/rotary.py). `CustomOp.register` 로 등록된 RoPE 커널. `generate()` 가 `templates/rotary_embedding.mlir` 를 specialize 해 `linalg.generic` (math.cos/sin + select) 손수 짠 tile 로 emit. IR 에서 `rope_select_concat_*` (65 hit).

```python
@CustomOp.register(library=LIBRARY)
class apply_rotary_embedding(CustomOp):
    signature = "apply_rotary_embedding(Tensor input, Tensor table) -> (Tensor)"
    def generate(self, ksel, kb):
        target_function_name = f"amdsharktank_rotary_embedding_{bs}_{sl}_{heads}_{dims}_{dtype}"
        inline_template_function(kb, "rotary_embedding.mlir", target_function_name, ...)
```

→ amdsharktank library 가 register 한 외부 op → 표준 torch-mlir 패스에 lowering rule 없음. **그 패스를 모르는 NPU 컴파일러는 unknown op 로 reject.** (RoPE *알고리즘*이 안 되는 게 아니라 *이 표현*이 안 되는 것 — easy3 §3.3.)

---

## 7. 클래스 전체 맵 (한 표)

호출 순서 = export 순서. "IR 흔적/제약" 의 #n 은 [easy3 §5](easy3.md) 의 제약 번호.

| # | 클래스 / 함수 | 파일:line | Stage | 역할 | IR 흔적 / 제약 |
|---|---|---|---|---|---|
| 1 | `import_hf_dataset` (tool) | tools/import_hf_dataset.py:31 | A | HF safetensors+config → IRPA | — |
| 2 | `Theta` | types/theta.py:83 | A | named tensor 트리 | `@__auto.*` symbol |
| 3 | `Dataset` | types/theta.py:372 | A | IRPA I/O (`save`/`load`) | IRPA 파일 |
| 4 | `DatasetMetadata` | types/theta.py:434 | A | IRPA key 스킴 | fixed IRPA properties |
| 5 | `DefaultPrimitiveTensor` | types/tensors.py | A' | tied embedding 복제 | `output.weight` #9 |
| 6 | `LlamaHParams` | configs/llm_configs.py:56 | B | 아키텍처 하이퍼파라미터 | — |
| 7 | `LlamaModelConfig` | configs/llm_configs.py:562 | B | hp + lowering 정책 | #6,#7 default |
| 8 | `ExportConfig` | models/llm/config.py:45 | B | export 행위 옵션 | bs/top_k |
| 9 | `ParallelismConfig` | configs/ | B | tp/pp 병렬도 | #5 device map |
| 10 | `PagedLlmModelV1` | models/llm/llm.py:33 | C | 메인 모델 (`prefill`/`decode`) | #2,#8 |
| 11 | `DefaultPagedKVCache` | layers/paged_attention.py:148 | C | KV slab `[?,524288]` | #1 |
| 12 | `TokenEmbeddingLayer` | layers/token_embedding.py:15 | C | 토큰 임베딩 | aten.* |
| 13 | `CachedRotaryLayer`→`RotaryEmbeddingLayer` | layers/rotary_embedding*.py | C | RoPE 레이어 | #4 |
| 14 | `RMSNormLayer` | layers/norm.py:15 | C | RMS norm | aten.* (f32 cast) |
| 15 | `LinearLayer` | layers/linear.py:19 | C | lm_head / proj | aten.mm |
| 16 | `AttentionFFNBlock` | models/llm/llm.py:244 | C | 블록 (attn+ffn) ×16 | — |
| 17 | `PagedLlamaGQAttentionBlock`→`PagedGQAttention` | layers/paged_attention.py:949 | C | GQA paged attention | #3 |
| 18 | `FFN` | layers/ffn_block.py:23 | C | SwiGLU MLP | aten.silu/mul |
| 19 | `ThetaLayer` (base) | layers/base.py:347 | C | weight-by-name nn.Module | — |
| 20 | `ServicePagedLlmModelV1` | models/llm/export.py:47 | D | export 래퍼 + sampling/affinity | #5, top_k |
| 21 | `CacheAllocation` | layers/kv_cache.py:16 | D | slab + device 묶음 | #1,#5 |
| 22 | `DeviceAffinity` | iree.turbine.aot | D | `iree.abi.affinity` attr | #5 |
| 23 | `FxProgramsBuilder` | iree.turbine.aot | E | 멀티 entry 빌더 | #2 |
| 24 | `@fxb.export_program` / `torch.export.Dim` | iree.turbine.aot / torch | E | FX trace + dynamic dim | #7,#8 |
| 25 | `aot.export` / `ExportOutput.save_mlir` | iree.turbine.aot | E | FX → MLIR 텍스트 | 27,269 lines |
| 26 | `KVCacheGatherKernel` (`@mlir_kernel`) | layers/paged_attention.py:68 | (C 내부) | `iree_linalg_ext.gather` inject | #3 (33 hit) |
| 27 | `apply_rotary_embedding` (`CustomOp`) | kernels/rotary.py:15 | (C 내부) | `linalg.generic` RoPE inject | #4 (65 hit) |

---

## 8. 클래스 의존 다이어그램

```
CLI (A) import_hf_dataset ──► Dataset/Theta/DatasetMetadata ──► *.irpa
                                       │  (+ DefaultPrimitiveTensor: tied fix)
                                       ▼
CLI (B) export_paged_llm_v1.main()
   ├─ Dataset.load(irpa)
   ├─ LlamaModelConfig.from_dataset(...)  ◄─ LlamaHParams, ParallelismConfig
   └─ ExportConfig(...)
                                       ▼
(C) PagedLlmModelV1(theta, config)            ── ThetaLayer 계열 ──┐
   ├─ DefaultPagedKVCache (slab[?,524288])                       │  weight: theta("...")
   ├─ TokenEmbeddingLayer                                        │
   ├─ CachedRotaryLayer ─► RotaryEmbeddingLayer ─► apply_rotary_embedding(CustomOp)
   ├─ RMSNormLayer, LinearLayer(lm_head)                         │
   └─ AttentionFFNBlock ×16                                      │
        ├─ PagedGQAttention ─► kv_cache_gather(@mlir_kernel ─► iree_linalg_ext.gather)
        └─ FFN (SwiGLU)                                          ┘
                                       ▼
(D) ServicePagedLlmModelV1(model, export_config)
        prefill()/decode() + logits_norm + top_k
        setup_cache() ─► CacheAllocation ─► DeviceAffinity(@__device_0)
                                       ▼
(E) FxProgramsBuilder(model)
        @fxb.export_program("prefill_bs1", dynamic_shapes=Dim)   ─┐ 두 entry
        @fxb.export_program("decode_bs1",  dynamic_shapes=Dim)   ─┘
        export(fxb) ─► ExportOutput.save_mlir() ─► llama-3.2-1b.mlir (27,269 lines)
```

---

## 9. 의미 한 줄 — 각 클래스는 *서버 invariant 의 운반체*

easy3 §6 의 결론을 **클래스 레벨로** 다시 쓰면:

> amdsharktank 의 export 클래스들은 *모델 수학*만이 아니라 *vLLM 호환 추론 서버의 런타임 계약*을 함께 그래프로 굳힌다.

| 서버 invariant | 운반하는 클래스 |
|---|---|
| "KV cache 는 서버가 page table 로 관리" | `DefaultPagedKVCache`, `CacheAllocation`, `KVCacheGatherKernel` (#1,#3) |
| "prefill 과 decode 는 다른 호출 path" | `PagedLlmModelV1.prefill/decode`, `FxProgramsBuilder` 두 entry (#2,#8) |
| "각 layer 는 미리 짠 high-perf kernel" | `apply_rotary_embedding`, `PagedGQAttention`, fp16 정책 (#3,#4,#6) |
| "weight 는 device id 로 라우팅" | `DeviceAffinity`, `ParallelismConfig` (#5) |
| "seq_len 은 request 마다 가변" | `torch.export.Dim` (#7) |
| "weight 는 IRPA 에서 mmap" | `Theta`/`Dataset`/IRPA (#9 — *유익한* invariant) |

→ on-device 로 옮기려면 이 클래스들을 *제거/단순화*해 `forward(input_ids)->logits` 한 entry 의 standard-aten `nn.Module` 로 만들어야 한다. 그게 우리 `LlamaOnDevice` 가 한 일 (= `PagedLlmModelV1` 의 *재구현*). **fixed IRPA (#9 = `Theta`/`Dataset`) 는 27개 클래스 중 weight 외부화 1개만 해결**한다.

---

## 10. cheat sheet (클래스 추적 재현)

```bash
source /home/bohyun/venv-shark/bin/activate
M=/tmp/llama32-irpa/llama-3.2-1b.mlir

# 어떤 클래스가 남긴 custom util.func 인가
grep -oE "util\.func private @[a-zA-Z0-9_]+" $M | sed -E 's/_(CACHE_SIZE|BS|SL).*//' | sort -u
#  → @paged_attention_kv_cache_gather  (= KVCacheGatherKernel, §6.1)
#  → @rope_select_concat               (= apply_rotary_embedding, §6.2)

grep -c "iree_linalg_ext" $M            # §6.1 gather  (IREE 확장 dialect)
grep -c "hal.device.promise" $M         # §4 DeviceAffinity (#5)
grep -nE "func\.func.*@(prefill|decode)" $M   # §3.1/§5 두 entry (#2)
grep -c "util.global" $M                # §1.2 IRPA weight 참조 (#9)

# amdsharktank 소스에서 클래스 정의 직접 열기 (sibling clone)
ASK=/home/bohyun/amd-shark-ai/amdsharktank/amdsharktank
sed -n '33,180p'  $ASK/models/llm/llm.py          # PagedLlmModelV1
sed -n '47,167p'  $ASK/models/llm/export.py        # ServicePagedLlmModelV1
sed -n '54,135p'  $ASK/layers/paged_attention.py   # KVCacheGatherKernel
sed -n '1,70p'    $ASK/kernels/rotary.py           # apply_rotary_embedding
sed -n '38,234p'  $ASK/examples/export_paged_llm_v1.py  # export 엔진
```

---

## 11. 참고

- 짝 문서: [easy.md](easy.md) (`LlamaOnDevice` 직접 export), [easy2.md](easy2.md) (임의 모델 입고 기준), [easy3.md](easy3.md) (제약 9종 분석 — **이 문서가 코드 근거를 제공**)
- 사전 가드: [`docs/SHARK_AI_ANALYSIS.md`](../SHARK_AI_ANALYSIS.md) (amdsharktank transformer 는 import 안 함), (내부 문서)
- amdsharktank source (분석 대상, Apache-2.0, **import 없이 읽기만**):
  - [examples/export_paged_llm_v1.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/examples/export_paged_llm_v1.py) — export 엔진
  - [models/llm/llm.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/llm.py) — `PagedLlmModelV1`, `AttentionFFNBlock`
  - [models/llm/export.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/export.py) — `ServicePagedLlmModelV1`
  - [models/llm/config.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/config.py) — `ExportConfig`
  - [layers/configs/llm_configs.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/configs/llm_configs.py) — `LlamaModelConfig`, `LlamaHParams`
  - [layers/paged_attention.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/paged_attention.py) — `KVCacheGatherKernel`, `PagedGQAttention`, `DefaultPagedKVCache`
  - [layers/kv_cache.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/kv_cache.py) — `CacheAllocation`, `KVCache`
  - [kernels/rotary.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/kernels/rotary.py) — `apply_rotary_embedding`
  - [types/theta.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/types/theta.py) — `Theta`, `Dataset`, `DatasetMetadata`
  - [tools/import_hf_dataset.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/tools/import_hf_dataset.py) — HF → IRPA
</content>
