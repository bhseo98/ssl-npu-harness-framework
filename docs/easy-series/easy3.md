# Easy Guide 3 — AMD-SHARK-AI 깊이 분해 + 임의 모델 lowering 제약 (PDF Step 3 빡세게)

> 본인 학습용 노트 — easy.md / easy2.md 의 후속편.
>
> 이 문서는 PDF "보라색 점선 최상단" 안의 **Step 3 (AMD-SHARK-AI 분석)** 을 다시 한 번, 이번엔 **실측 MLIR + amdsharktank 소스 코드 기준** 으로 빡세게 파헤친다. 결과를 손에 쥐고 "fixed IRPA 만 적용하면 임의 모델이 다 lowering 되는가?" 에 정직히 답한다.

---

## TL;DR (한 줄 요약)

> amdsharktank 의 `export_paged_llm_v1` 으로 **real `meta-llama/Llama-3.2-1B-Instruct`** (HF safetensors → IRPA 변환 후) 를 진짜 MLIR (`/tmp/llama32-irpa/llama-3.2-1b.mlir`, **27,269 lines, 2.4 MB**) 로 뽑아 surface 를 직접 셌다. 제약은 9개 — *KV cache state argument · prefill/decode 2-함수 split · paged_attention CustomOp (`iree_linalg_ext.gather`) · RoPE `apply_rotary_embedding` CustomOp (`linalg.generic` 템플릿) · `iree.abi.affinity` device promise · fp16 강제 dtype 정책 · dynamic seq_len `?` · 단계마다 다른 entry-point · IRPA 외부 weight 참조* — 인데, **밑에 깔린 1개 원인** 으로 묶인다: *"server-side runtime 의 *런타임 invariant* (KV slab 위치, sampling kernel, device assignment, weight archive 경로) 가 model 의 forward 그래프 안에 *함수 인자/custom op* 로 박혀 있다."* **fixed IRPA 적용은 9개 중 1개만 (weight 외부화)** 해결한다 — 나머지 8개는 별개 작업. 그래서 "fixed IRPA = 만능키" 는 거짓. (동일 9개 surface 가 toy 3-layer Llama 에서도 5,185-line 으로 재현됨 → 제약은 *모델 크기* 가 아니라 *아키텍처/runtime 가정* 의 문제임을 교차 확인.)

> 📊 **그림 요약**: 전체 순서(두 경로 / amdsharktank 순서 / LlamaOnDevice 순서 / 6단계 로드맵 / 임의 모델 입고 순서)는 [`docs/diagrams/lowering-order.drawio`](../diagrams/lowering-order.drawio) 5페이지 다이어그램 참조 (draw.io / VS Code draw.io 확장에서 열기).

---

## 0. 이 문서가 다루는 PDF 박스

[easy2.md §0](easy2.md) 의 그림에서 빨강 `PyTorch Model Zoo` 박스 — 그중에서도 *"server-side 사례를 분석해서 on-device 로 옮길 때 무엇이 막히는지"* 가 PDF §3 의 요구사항.

```
┌─ 보라색 점선 ──────────────────────────────────────┐
│  📦 PyTorch Model Zoo  ◀ easy2 = "어떤 모델이 들어갈 수 있나"  │
│                        ◀ easy3 = "amdsharktank server-side 를   │
│                                   on-device 로 옮길 때의 제약" │
│  Torch-MLIR / Torch-MLIR Compiler                              │
└────────────────────────────────────────────────────┘
```

- easy.md = 우리가 만든 `LlamaOnDevice` 1 모델 lowering 성공
- easy2 = HF 모델 7개 통과/실패 surface
- **easy3 = amdsharktank 서버 모델 lowering surface 측정 + 그 surface 를 만들어내는 unified principle 발견**

---

## 1. 실측 셋업 (반복 가능)

```bash
source /home/bohyun/venv-shark/bin/activate    # Python 3.11.15

# ── real Llama-3.2-1B-Instruct path (이 문서의 main ground truth) ──
SNAP=~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/

# 1) HF safetensors → IRPA 변환
python -m amdsharktank.tools.import_hf_dataset \
  --config-json ${SNAP}config.json \
  --params ${SNAP}model.safetensors \
  --output-irpa-file /tmp/llama32-irpa/llama-3.2-1b.irpa

# 1b) tied embedding fix — Llama-3.2-1B 은 tie_word_embeddings=True 라
#     import 결과에 lm_head(`output.weight`) 가 없음. token_embd.weight 를
#     output.weight 로 복제해 재저장 (안 하면 export 가 KeyError ['output']).
#     (코드: easy3 §2.4 snippet 참조)
#     → /tmp/llama32-irpa/llama-3.2-1b-tied.irpa

# 2) IRPA → MLIR export (amdsharktank 의 production entry-point)
python -m amdsharktank.examples.export_paged_llm_v1 \
  --irpa-file /tmp/llama32-irpa/llama-3.2-1b-tied.irpa \
  --output-mlir /tmp/llama32-irpa/llama-3.2-1b.mlir \
  --output-config /tmp/llama32-irpa/llama-3.2-1b.json \
  --bs-prefill 1 --bs-decode 1 \
  --activation-dtype float16 --attention-dtype float16

# ── (교차확인용) toy 3-layer Llama path — 같은 9개 surface 가 작게 재현 ──
python -m amdsharktank.models.llama.toy_llama -o /tmp/toy_llama.irpa
python -m amdsharktank.examples.export_paged_llm_v1 \
  --irpa-file /tmp/toy_llama.irpa --output-mlir /tmp/toy_llama.mlir \
  --output-config /tmp/toy_llama.json --bs-prefill 1 --bs-decode 1 \
  --activation-dtype float16 --attention-dtype float16
```

산출물 (real Llama-3.2-1B):
- `llama-3.2-1b.irpa` — 2.47 GB (HF import), tied fix 후 2.8 GB. weights + GGUF-style 하이퍼파라미터 메타
- `llama-3.2-1b.mlir` — 2.4 MB, **27,269 lines** ← 이 파일이 §3-§5 의 main ground truth
- `llama-3.2-1b.json` — export config (function signature 메타)

교차확인용 (toy 3-layer): `toy_llama.mlir` — 451 KB, **5,185 lines**. 같은 9개 surface, 5.3× 작음.

---

## 2. IRPA — 정확히 무엇인가?

### 2.1 정의

**IRPA = IREE/SHARK Runtime Parameter Archive.** amdsharktank 의 자체 weight 직렬화 포맷. 파일 시그니처 / scheme:

```python
# amdsharktank/types/theta.py:434
@dataclass
class DatasetMetadata:
    """When saved to an IRPA file, it will be saved with multiple keys:
      * properties:        __AMD_SHARK_DATASET__
      * inference_tensors: __AMD_SHARK_INFERENCE_TENSORS__
      * optional(shard_ranks): __AMD_SHARK_SHARD_RANKS__
    """
```

직접 실측 (`Dataset.load("/tmp/llama32-irpa/llama-3.2-1b-tied.irpa")` — real Llama-3.2-1B):

```python
properties keys:
  'general.architecture', 'llama.attention.head_count',          # 32
  'llama.attention.head_count_kv',                               # 8 (GQA)
  'llama.attention.head_dim', 'llama.attention.layer_norm_rms_epsilon',
  'llama.block_count',                                           # 16
  'llama.context_length', 'llama.embedding_length',              # 2048
  'llama.feed_forward_length',                                   # 8192
  'llama.rope.dimension_count', 'llama.rope.freq_base',          # 500000
  'llama.rope.interleave_emb', 'llama.vocab_size',               # 128256
  # export 가 IRPA 에 도로 적어넣은 runtime 옵션들 (= "fixed IRPA" 의 실체):
  'activation_dtype', 'attention_dtype', 'attention_kernel',
  'block_seq_stride', 'kv_cache_type', 'fake_quant',
  'tensor_parallelism_size', 'parallelism_config', 'use_qk_norm',
  'AMD_SHARK_DATASET_VERSION'

root_theta tensors (sample):
  token_embd.weight           [128256, 2048]  bf16
  blk.0.attn_q.weight         [2048, 2048]    bf16
  blk.0.attn_k.weight         [512, 2048]     bf16   ← GQA: kv 차원 작음
  blk.0.attn_v.weight         [512, 2048]     bf16
  blk.0.attn_output.weight    [2048, 2048]    bf16
  blk.0.attn_norm.weight      [2048]          bf16
  blk.0.ffn_norm.weight       [2048]          bf16
  blk.0.ffn_gate.weight       [8192, 2048]    bf16
  blk.0.ffn_up.weight         [8192, 2048]    bf16
  blk.0.ffn_down.weight       [2048, 8192]    bf16
  output.weight               [128256, 2048]  bf16   ← §2.4 의 tied fix 로 추가
  ...
```

(toy 3-layer 는 같은 키 구조에 숫자만 작다: hidden 256, vocab 256, ffn 23. 또한 toy 는 `expert_count`/`expert_used_count` property 가 있지만 real 1B 는 비-MoE 라 없음.)

### 2.2 IRPA 가 푸는 두 가지 문제

| 문제 | IRPA 해법 |
|---|---|
| **(1) MLIR 안에 weight 가 inline 으로 박히면 IR 가 무지막지하게 커진다** (easy.md §7 의 12 GB 문제) | weight 를 **IRPA 파일에 외부 저장**하고, MLIR 안에는 `util.global` + 이름만 참조 (`@__auto.token_embd.weight`) |
| **(2) 가중치 dtype/shape/quantization scheme 이 코드와 분리돼야 한다** | IRPA properties 에 hyperparameter, root_theta 에 named tensor — model code 는 "이름만 보고" 로드 |

MLIR 안에서 weight 참조는 이렇게 나옴 (real Llama-3.2-1B `llama-3.2-1b.mlir` 발췌):
```mlir
%__auto.token_embd.weight =
  util.global.load @__auto.token_embd.weight : tensor<128256x2048xf16>
%0 = torch_c.from_builtin_tensor %__auto.token_embd.weight
       : tensor<128256x2048xf16> -> !torch.vtensor<[128256,2048],f16>
```

→ 즉 IRPA 는 **weight ↔ MLIR 의 link table**. 런타임에 `iree-run-module --parameters=foo.irpa` 식으로 binding.

### 2.3 "Fixed IRPA" 가 무엇을 뜻하나

amdsharktank 코드 안의 한 줄:

```python
# amdsharktank/examples/export_paged_llm_v1.py:281
# TODO: Remove this flag once we expect values are baked in irpa file
if args.use_hf:
    llama_config.hp.rope_interleave_emb = False
```

→ "fixed IRPA" = **하이퍼파라미터 값들이 IRPA properties 에 모두 *고정 저장* 된 상태**. CLI flag 로 매번 override 할 필요 없는 상태. 즉 **IRPA = 가중치 + 모델 메타 + (이상적으로는) 모든 사용 옵션 까지 한 파일**.

→ "fixed IRPA 만 만들어두면 export 가 그 IRPA 만 가지고 일관되게 돈다" — 라는 의미이지, **MLIR 생성 자체의 제약을 없애주는 것은 아니다** (§7 의 honest answer 참조).

### 2.4 real Llama-3.2-1B IRPA 변환 시 실제로 막혔던 것 — tied embedding

real `meta-llama/Llama-3.2-1B-Instruct` 를 `import_hf_dataset` 으로 IRPA 변환하면 export 가 곧바로 깨진다:

```
KeyError: "Sub-theta ['output'] not found
           (of dict_keys(['token_embd', 'blk', 'output_norm']))"
```

원인: Llama-3.2-1B 은 `config.json` 의 **`tie_word_embeddings: True`** — lm_head 가 token embedding 과 weight 를 공유해서, HF safetensors 에 `lm_head.weight` 가 *물리적으로 없다*. 그런데 amdsharktank 의 `PagedLlmModelV1` 은 `theta("output")` (lm_head) 를 *필수* 로 찾는다. 해결:

```python
from amdsharktank.types.theta import Dataset, Theta
from amdsharktank.types.tensors import DefaultPrimitiveTensor

ds = Dataset.load("llama-3.2-1b.irpa")
flat = ds.root_theta.flatten()
te = flat["token_embd.weight"].as_torch()           # [128256, 2048] bf16
flat["output.weight"] = DefaultPrimitiveTensor(      # tied: 복제
    name="output.weight", data=te.clone())
Dataset(properties=ds.properties, root_theta=Theta(flat)).save("llama-3.2-1b-tied.irpa")
```

→ 이건 "fixed IRPA" 와 다른 *추가* 작업 (IRPA *내용물* 의 수선). 즉 IRPA 를 쓴다고 모든 게 자동이 아니라, **모델별 weight 구조(여기선 tied embedding)까지 IRPA 단계에서 맞춰줘야** 한다 — §7 honest answer 의 또 다른 근거.

---

## 3. RoPE — 무엇인가, 왜 *naive* 로 안 되는가

### 3.1 RoPE (Rotary Position Embedding) 자체

위치 정보를 **(쿼리, 키) 벡터에 회전 행렬을 곱해서 주입**하는 기법. 본질은 이렇다:

```python
# pseudo
for token_idx t in [0..T]:
    angle = inv_freq * t          # (head_dim/2,)
    cos_t, sin_t = cos(angle), sin(angle)
    q[..., t, :] = rotate(q[..., t, :], cos_t, sin_t)
    k[..., t, :] = rotate(k[..., t, :], cos_t, sin_t)
```

`rotate(x, cos, sin)` 의 표준 표현은 두 가지:

- **half-rotation** (HF Llama / 우리 LlamaOnDevice):
  `out = x*cos + rotate_half(x)*sin` where `rotate_half([a,b]) = [-b,a]`
- **interleaved-pair**: `[r0, i0, r1, i1, ...]` → `[r0*cos - i0*sin, i0*cos + r0*sin, ...]`

→ 둘 다 **수학적으로는 같은 회전**, 데이터 레이아웃만 다름. amdsharktank toy IRPA 의 메타에 `llama.rope.interleave_emb: True` → interleaved 레이아웃.

### 3.2 왜 amdsharktank 는 RoPE 를 *CustomOp* 로 만들었나

소스: [`amdsharktank/kernels/rotary.py`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/kernels/rotary.py)

```python
@CustomOp.register(library=LIBRARY)
class apply_rotary_embedding(CustomOp):
    signature = "apply_rotary_embedding(Tensor input, Tensor table) -> (Tensor)"
    def generate(self, ksel, kb):
        template_file = "rotary_embedding.mlir"
        target_function_name = f"amdsharktank_rotary_embedding_{bs}_{sl}_{heads}_{dims}_{input_dtype}"
        ...
```

그리고 그 template 안 ([`kernels/templates/rotary_embedding.mlir`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/kernels/templates/rotary_embedding.mlir)):

```mlir
util.func private @amdsharktank_rotary_embedding_..._f16(
    %input: tensor<...>, %table: tensor<...>) -> tensor<...> {
  %result = linalg.generic {indexing_maps=..., iterator_types=[parallel × 4]}
    ins(%table : ...) outs(%empty : ...) {
    ^bb0(%b0, %b1):
      %a_cosb = math.cos %b0
      %a_sinb = math.sin %b0
      %real_t2 = arith.subf (real*cosb) (imag*sinb)
      %imag_t2 = arith.addf (imag*cosb) (real*sinb)
      %val = arith.select %cmp, %real_t2, %imag_t2
      linalg.yield %val
  }
}
```

→ amdsharktank 의 RoPE 는 *손수 짠 `linalg.generic`* . GPU/NPU 가 이 `linalg.generic` 하나를 **fused tile** 로 컴파일하면 KV write + attention 과 한 번에 메모리 path 가 묶임 (성능 ↑).

### 3.3 그래서 "RoPE 가 안 된다" 는 정확히 무슨 뜻인가

세 가지가 다 섞여 있는데 분리:

| 문장 | 사실 여부 | 무엇이 안 되는가 |
|---|---|---|
| "RoPE 가 PyTorch 에서 안 된다" | ❌ **거짓** | 우리 LlamaOnDevice 는 RoPE 잘 돈다 (`aten.mul / add / cat` 분해) |
| "RoPE 가 torch-mlir 로 lowering 안 된다" | ❌ **거짓** | precomputed cos/sin + `torch.aten.mul/add/cat` 로 쓰면 통과 (easy.md §3.2 의 산출물 증명) |
| "amdsharktank 의 RoPE CustomOp 가 torch-mlir 표준 dialect 로 안 풀린다" | ✅ **참** | `amdsharktank_rotary_embedding_*` 는 amdsharktank library 가 register 한 외부 op → 표준 torch-mlir 패스에 lowering rule 없음. **그 패스를 모르는 NPU 컴파일러는 unknown op 로 reject** |

→ 실측 (real Llama-3.2-1B `llama-3.2-1b.mlir` — 같은 custom util.func 가 65 곳에서 호출됨):
```
util.func private @rope_select_concat_BS_SL_HEADS_HALFDIM_f16_BS_SL_HEADS_HALFDIM_f16_BS_SL_HEADS_TWO_2_HALFDIM_f16(...)
```

이 함수가 IR 안에 그대로 박혀 있어서, 우리 NPU compiler 가 *이 함수 본문 (`linalg.generic` + `tensor.extract` + `math.cos`)* 까지 lower 할 수 있어야 통과. 즉:

- amdsharktank RoPE = **interleaved + custom util.func + 외부 표현** → on-device 적합도 ✗
- 우리 RoPE (LlamaOnDevice) = **half-rotation + standard `aten.*` + 한 forward 안에 인라인** → 적합도 ✓

→ 결론: "RoPE 가 안 된다" 라는 표현은 **server-side 표현 (CustomOp + linalg template) 이 안 된다** 의 줄임말. RoPE 알고리즘 자체는 가능.

---

## 4. Paged Attention — 무엇이고, 왜 못 쓰나

### 4.1 PagedAttention 이 푸는 문제

vLLM 이 도입한 메모리 관리 기법. 핵심 아이디어:

- 일반 attention: 시퀀스마다 KV cache 를 *연속된 큰 텐서* 로 잡음 → max_seq_len 만큼 미리 할당 → **메모리 낭비** (실제로 안 쓴 자리도 OS 가 가져감)
- PagedAttention: KV cache 를 **고정 크기 페이지 (block_seq_stride=32 같은 작은 chunk)** 로 잘라 *page table* 로 indirect 하게 참조 → 가상메모리 paging 과 같은 원리 → 메모리 fragmentation 0, 동시 batch 를 빽빽하게 채울 수 있음

소스: [`amdsharktank/layers/paged_attention.py`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/paged_attention.py) (1000+ 줄).

### 4.2 왜 lowering 이 막히는가 — 세 단계 분해

**(a) Custom MLIR kernel.** PagedAttention 의 KV gather 가 *손수 짠 `iree_linalg_ext.gather`* 로 inject 됨 (source `paged_attention.py:89` 의 `@mlir_kernel` 데코레이터). MLIR 출력 직접 확인:

```mlir
util.func private @paged_attention_kv_cache_gather_CACHE_SIZE_T_BLOCK_3_PART_2_HEAD_COUNT_KV_4_BLOCK_SEQ_STRIDE_32_ATTN_HEAD_DIM_32_f16_..._f16(
    %cache: !cache, %page_ids: !page_ids,
    %transformer_idx: !transformer_idx, %partition_idx: !partition_idx) -> !result {
  %cache_slice = tensor.extract_slice %cache [...] [...] [...]
  %result = iree_linalg_ext.gather dimension_map=[0]
            ins(%cache_slice, %page_ids : ...)
            outs(%empty : ...) -> !result
  util.return %result
}
```

→ `iree_linalg_ext.gather` 는 IREE 전용 extension dialect. **표준 MLIR 가 아님**. 우리 NPU 컴파일러가 `iree_linalg_ext.*` 를 모르면 거기서 stop.

**(b) 데이터 의존적 indirection.**
`page_table[page_ids]` 형태의 **data-dependent gather** — `page_ids` 텐서 값이 어느 페이지를 읽을지 결정. 정적 분석 친화적이지 않음 (NPU compiler 가 access pattern 을 정적으로 못 풀면 DMA 스케줄링 못 함).

**(c) 외부 state 입출력.**
`prefill_bs1` 의 `arg3: torch.tensor<[?,524288],f16>` ↔ `decode_bs1` 의 `arg4: torch.tensor<[?,524288],f16>` 가 바로 *KV cache slab* (real Llama-3.2-1B; toy 는 `[?,24576]`). **mutable buffer 가 함수 인자로 노출** — amdsharktank 의 runtime 이 매 step 마다 채워서 다시 넣어준다는 계약. 우리 forward-only 모델에는 없는 개념.

→ "PagedAttention 을 못 쓴다" = **위 (a) + (b) + (c) 셋 다 NPU compiler 가 받을 수 없는 형태**. 셋 중 하나만 막혀도 import fail.

### 4.3 우리 on-device 대안

LlamaOnDevice = **KV cache 자체 없음** (forward 가 매번 전체 시퀀스 재계산). 작은 모델 + 짧은 컨텍스트 + 임베디드 시나리오에서는 prefill 마다 fresh 가 더 단순하고 메모리도 적게 든다. 이게 PDF §3 의 *"On-Device 로 바꿔서 porting 하기 위해서 어떤 변화가 필요할지"* 에 대한 답.

---

## 5. 9개 제약 surface — 실측 MLIR 에서 직접 셈

**real Llama-3.2-1B** (`/tmp/llama32-irpa/llama-3.2-1b.mlir`) 의 prefill_bs1 시그니처 한 줄에 거의 다 있다:

```mlir
func.func @prefill_bs1(
    %arg0: !torch.vtensor<[1,?],si64>      {iree.abi.affinity = #hal.device.promise<@__device_0>},
    %arg1: !torch.vtensor<[1],si64>        {iree.abi.affinity = #hal.device.promise<@__device_0>},
    %arg2: !torch.vtensor<[1,?],si64>      {iree.abi.affinity = #hal.device.promise<@__device_0>},
    %arg3: !torch.tensor<[?,524288],f16>   {iree.abi.affinity = #hal.device.promise<@__device_0>}
) -> !torch.vtensor<[1,?,128256],f16>
  attributes {torch.assume_strict_symbolic_shapes}
```

(toy 3-layer 는 같은 모양에 숫자만 작다: KV slab `[?,24576]`, vocab `256`.)

여기서 직접 떨어지는 제약 (real Llama-3.2-1B 실측 hit count 포함):

| # | 제약 surface | MLIR 에서의 흔적 (real Llama-3.2-1B) | 의미 |
|---|---|---|---|
| 1 | **KV cache state 가 함수 인자** | `arg3: !torch.tensor<[?,524288],f16>` (mutable, 동적 page count; 524288 = 16 layer × 2 part × 8 kv-head × 32 block-seq × 64 head-dim) — IR 전체 218 hit | forward 가 stateless 가 아니라 runtime buffer 와 강결합 |
| 2 | **prefill / decode 함수 분리** | `@prefill_bs1` (full seq) + `@decode_bs1` (single token) 두 entry | 한 forward 가 아니라 2-step contract — NPU runtime 이 어느 함수를 언제 부를지 알아야 함 |
| 3 | **PagedAttention CustomOp** | `util.func private @paged_attention_kv_cache_gather_*` + `iree_linalg_ext.gather` — 33 hit | §4 |
| 4 | **RoPE CustomOp** | `util.func private @rope_select_concat_*` — 65 hit | §3 |
| 5 | **iree.abi.affinity** | 모든 arg/result 에 `#hal.device.promise<@__device_0>` | 어느 device 에 할당될지 *MLIR 안에서* 결정됨 — single-NPU 환경에서는 무의미한 metadata |
| 6 | **fp16 강제 (mixed)** | activation/attention `f16`, norm weight `f32` | dtype policy 가 server-side 의 mixed-precision kernel 가정에 맞춰져 있음 |
| 7 | **dynamic seq_len `?`** | `vtensor<[1,?],si64>`, `tensor<[?,524288],f16>` — `?` 4,530 hit | NPU compiler 가 dynamic dim 을 못 다루면 specialization 필요 |
| 8 | **여러 entry-point + multi-args contract** | `@prefill_bs1` 의 args: `tokens / seq_lens / seq_block_ids / cache_state` 4개 | "forward(input_ids)" 식 단순 함수가 아님 |
| 9 | **IRPA 외부 weight 참조** | `util.global private @__auto.token_embd.weight = dense_resource<...>` (또는 external IRPA binding) | weight 자체는 export 단계에서 IRPA 또는 dense_resource 로 분리됨 — *이게 IRPA 가 해결한 것* |

### 5.1 real Llama-3.2-1B vs toy 3-layer — 같은 surface, 다른 규모

| 지표 | toy 3-layer | **real Llama-3.2-1B** |
|---|---:|---:|
| MLIR lines | 5,185 | **27,269** (5.3×) |
| MLIR size | 451 KB | **2.4 MB** |
| IRPA size | 1.5 MB | **2.8 GB** (tied fix 후) |
| vocab (output dim) | 256 | **128,256** |
| KV cache slab | `[?,24576]` | `[?,524288]` (21×) |
| 9개 surface | 9/9 출현 | **9/9 출현** |
| `paged_attention_kv_cache_gather` | 1 def | 1 def (33 use) |
| `rope_select_concat` | 1 def | 1 def (65 use) |

→ **제약의 *종류* 는 toy 와 real 이 동일.** 모델을 키워도 9개 surface 의 *집합* 은 변하지 않고 *반복 횟수* 만 늘어난다. 즉 제약은 *스케일* 이 아니라 *amdsharktank 의 server-side 아키텍처/runtime 계약* 에서 온다 — §6 의 unified principle 을 real 모델에서 재확인.

이 9개 surface 가 *모두 함께* server-side runtime 의 가정을 노출.

---

## 6. 9개 제약을 하나로 묶는 원인 — *서버 invariant 의 graph 침투*

위 9개는 따로 보면 잡다하지만, **밑에 깔린 단 하나의 원인** 으로 묶인다:

> **amdsharktank 는 server-side 추론 runtime (vLLM 호환) 의 *런타임 invariant* 를 forward 그래프 안으로 그대로 끌고 들어온다.**

여기서 *server-side runtime invariant* = 추론 서버가 매번 보장해야 하는 외부 상태들:

| Runtime invariant | 어느 surface 로 침투하나 |
|---|---|
| "KV cache slab 은 server 가 page table 로 관리한다" | #1 (cache arg), #3 (paged_attn op), #8 (cache_state / page_ids args) |
| "prefill 과 decode 는 다른 호출 path 다 (batched vs autoregressive)" | #2 (두 entry), #8 (서로 다른 args 구조) |
| "각 layer 는 server 가 미리 짜둔 high-perf kernel 로 실행된다" | #3 (paged_attn), #4 (rope), #6 (fp16 mixed) |
| "여러 device 에 sharding 된 weight 가 다 device id 로 라우팅된다" | #5 (iree.abi.affinity) |
| "seq_len 은 request 마다 가변이고 padding 으로 보정한다" | #7 (dynamic `?`) |
| "weight 는 runtime 에 IRPA 파일에서 mmap 으로 갈아끼운다" | #9 (util.global + external) |

→ **즉 server 의 외부 런타임 책임 (page manager, scheduler, kernel router, sharder, parameter loader) 이 모델 그래프 안에 *함수 인자 + custom op + ABI 메타데이터* 형태로 침투**한 것이 9개 surface 의 근원.

**on-device 로 가려면** = 이 invariant 들을 *모델 그래프 밖으로 추방* 해야 함:
- KV cache → forward 안에서 재계산 또는 외부에서 단순 in/out buffer (mutable 아님)
- prefill/decode → 한 entry-point 로 통합 또는 두 모델로 분리해서 각각 export
- paged_attn / rope → 표준 `torch.aten.*` 로 직접 분해해서 NPU 가 일반 op 로 받음
- iree.abi.affinity → 단일 device 환경에서 제거
- fp16 mixed → NPU 가 지원하는 dtype 로 통일
- dynamic seq_len → fixed `[1, T]` 로 specialize
- 4-args contract → 1-arg `(input_ids,)` 로 정리
- weight → IRPA 그대로 활용 (이건 invariant 가 *유익한* 케이스 — 분리 자체가 NPU 에도 도움)

이 8가지 추방 작업이 바로 `LlamaOnDevice` 가 한 일 (한 줄 `for layer in self.layers: x = layer(x)` 안에 다 들어가도록 재작성).

---

## 7. 그래서 — fixed IRPA 만 적용하면 임의 모델 lowering 다 되는가?

**아니오.** IRPA 는 위 9개 중 **단 1개** (#9 weight 외부 참조) 만 해결한다.

| Surface | "fixed IRPA" 만으로 해결되는가? |
|---|---|
| #1 KV cache state arg | ❌ 모델 forward 자체를 stateless 로 재작성 필요 |
| #2 prefill/decode 분리 | ❌ |
| #3 paged_attention CustomOp | ❌ 표준 `aten.matmul + softmax` 로 재작성 필요 |
| #4 RoPE CustomOp | ❌ standard ops 로 재작성 |
| #5 iree.abi.affinity | ❌ export config 단계에서 single-device 명시 |
| #6 fp16 mixed | ⚠ IRPA 의 dtype 메타 + export flag 같이 조정 (일부 도움) |
| #7 dynamic `?` | ❌ example_args 에서 fixed shape 로 trace |
| #8 multi-args contract | ❌ wrapper module 필요 |
| #9 weight 외부 참조 | ✅ **IRPA 그대로** |

→ "fixed IRPA 만 있으면 임의 모델 다 된다" 는 **거짓**. fixed IRPA 는 *weight 와 모델 메타가 한 파일에 일관되게 잠긴 상태* 일 뿐이고, 나머지 8개 surface 는 **모델 코드 자체** 의 재구성이 필요.

**현실적인 임의 모델 lowering 체크리스트** (필요 조건 — 충분 조건 X):

| # | 항목 | 작업 분량 |
|---|---|---|
| A | forward = `(input_ids,)` → `logits` 단일 entry, stateless | 모델 wrapper 1-50줄 |
| B | KV cache, page table, prefill/decode 분리 모두 제거 | 모델 forward 재작성 (Llama: 100-200줄) |
| C | RoPE / PagedAttn / FlashAttn / vLLM 의 모든 custom kernel 회피 (표준 aten 만) | 위와 같이 forward 재작성에 포함 |
| D | `attn_implementation="eager"` + `use_cache=False` + `return_dict=False` + `eval()` | HF 모델일 때 가드 4줄 (easy2 §7.3) |
| E | example_args 로 fixed shape 강제 (`vtensor<[1,32,...]>`) | export 호출 시 인자 1개 |
| F | dtype 단일화 또는 NPU 가 받을 mixed-policy 명시 | export config |
| G | single-device — iree.abi.affinity 제거 또는 export config 에서 `single_device=True` | export config |
| H | weight 분리 (`aot.externalize_module_parameters`) | export call 보조 — 여기서 IRPA-style 결과 |
| I | (easy2 §5.1 의 `ModuleStackTracer` 이슈 등) torch.fx tracing 호환 | HF 모델별 monkey-patch 또는 재작성 |

→ A–G 7개를 *모두* 통과해야 lowering. **fixed IRPA (= H 한 줄)** 는 그중 1개. 나머지 8개는 모델별로 따로 작업해야 함.

### 7.1 그럼 "Model Zoo" 의 입고 기준은 결국?

easy2 §0.2 의 표 + easy3 §6 의 unified principle 을 합치면:

> **PyTorch Model Zoo 에 들어갈 수 있는 모델 = "server runtime invariant 가 forward 그래프에 침투하지 않은 stateless single-entry PyTorch nn.Module"**.

들어갈 수 있는 형태로 *가져오는 작업* 의 일반화된 비용:
- 잘 짜진 raw nn.Module (ResNet, BERT, GPT-2): 거의 0 (easy2 §4)
- HF transformers Llama-family: 모델 forward 한 함수 재작성 (LlamaOnDevice 패턴, ~170 LoC)
- vLLM / amdsharktank server-side: **모델을 완전히 다시** 작성 (`LlamaOnDevice` 가 sharktank `PagedLlmModelV1` 의 *재구현* 인 이유)

---

## 8. 그리고 빠뜨린 게 더 있나? — 위 9개 외 잠재 surface

amdsharktank source 를 한 번 더 훑어 더 찾을 만한 것:

| 잠재 surface | source 위치 | toy MLIR 에 등장? |
|---|---|---|
| **TopK + temperature sampling** (`ServicePagedLlmModelV1` 의 sampling control flow) | `models/llm/export.py` | toy 셋업 (top_k 비활성) 이라 안 보임 — 켜면 control flow / `iree_linalg_ext.topk` 등장 |
| **Tensor Parallelism / Pipeline Parallelism** | `types/sharding.py`, `ParallelismConfig` | toy=single tensor 라 안 보임. 켜면 `flow.tensor.split / gather` + 여러 `@__device_N` |
| **Quantized layouts** (`mmt_block_scaled_q8`, `einsum_2args_q4`, GGUF Q8_0/Q4_K_M) | `kernels/mmt_*`, `types/layouts.py` | INT8/INT4 IRPA 쓸 때 quant-aware custom kernel 출현 (FP16 toy 라 0) |
| **Wave / ASM-shuffled matmul** | `kernels/wave/`, `kernels/gemm_fp4_asm.py` | `--matmul-kernel amdsharktank.wave` 켜면 raw asm 커널 |
| **MoE expert routing** (Mixtral / DeepSeek) | `models/llm/llm.py` MLA path | toy=non-MoE 라 0. 켜면 data-dependent gather/scatter (NPU 거의 reject) |
| **Logits normalization / softcap** (Gemma 등) | `ExportConfig.logits_normalization` | toy=off |
| **Continuous batching prefill (paged + chunked)** | `--use-extend-attention` | toy=off |

→ 즉 §5 의 9개는 *toy minimum* 에서 본 것. 실제 production IRPA (Llama-3.2-1B INT8, MoE, multi-device) 까지 가면 **추가로 6+개 surface 가 더 침투**. 다만 **§6 의 unified principle ("서버 runtime invariant 의 graph 침투") 는 그대로 적용** — 다 같은 뿌리.

---

## 9. 다음에 까먹었을 때 떠올리는 한 줄

> *"amdsharktank server LLM 을 그대로 lowering 하려고 하면 9+ 개의 surface 가 막힌다 — KV cache state arg / prefill·decode split / PagedAttn CustomOp / RoPE CustomOp / iree.abi.affinity / fp16 mixed / dynamic dim / multi-args contract / IRPA 외부 weight. 이 9개는 **server runtime 의 invariant 가 forward 그래프 안에 함수 인자 + custom op + ABI 메타로 침투** 한 결과다. fixed IRPA 는 그중 weight 외부화 1개만 풀어준다. 임의 모델 zoo 입고 = 8개를 *모델 코드 단계에서 모두 제거* 해서 'stateless single-entry standard-aten nn.Module' 로 만드는 것. 그게 `LlamaOnDevice` 가 한 일."*

---

## 10. cheat sheet (재현)

```bash
# 환경
source /home/bohyun/venv-shark/bin/activate

# (1) real Llama-3.2-1B: HF safetensors → IRPA
SNAP=~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/
python -m amdsharktank.tools.import_hf_dataset \
  --config-json ${SNAP}config.json --params ${SNAP}model.safetensors \
  --output-irpa-file /tmp/llama32-irpa/llama-3.2-1b.irpa
# (1b) tied embedding fix (§2.4 의 python snippet) → llama-3.2-1b-tied.irpa

# (2) amdsharktank export → MLIR + config
python -m amdsharktank.examples.export_paged_llm_v1 \
  --irpa-file /tmp/llama32-irpa/llama-3.2-1b-tied.irpa \
  --output-mlir /tmp/llama32-irpa/llama-3.2-1b.mlir \
  --output-config /tmp/llama32-irpa/llama-3.2-1b.json \
  --bs-prefill 1 --bs-decode 1 \
  --activation-dtype float16 --attention-dtype float16

# (3) 9개 surface 확인
M=/tmp/llama32-irpa/llama-3.2-1b.mlir
wc -l $M                                            # 27,269 lines
grep -oE "util\.func private @[a-zA-Z0-9_]+" $M | sed -E 's/_(CACHE_SIZE|BS|SL).*//' | sort -u
# → @paged_attention_kv_cache_gather, @rope_select_concat
grep -c "hal.device.promise" $M                     # iree.abi.affinity
grep -nE "func\.func.*@(prefill|decode)" $M         # 2개 entry
grep -c "iree_linalg_ext" $M                         # ≥ 1
grep -c "vtensor<\[1,\?" $M                           # dynamic dim

# (교차확인) toy 3-layer — 같은 9개 surface, 5,185 lines
python -m amdsharktank.models.llama.toy_llama -o /tmp/toy_llama.irpa
python -m amdsharktank.examples.export_paged_llm_v1 \
  --irpa-file /tmp/toy_llama.irpa --output-mlir /tmp/toy_llama.mlir \
  --output-config /tmp/toy_llama.json --bs-prefill 1 --bs-decode 1 \
  --activation-dtype float16 --attention-dtype float16
```

소요: real HF→IRPA 변환 ~40 초 + tied fix ~20 초 + export ~90 초 → 총 ~2.5 분 (toy 만 하면 ~30 초).

---

## 11. 참고

- 본 문서 = PDF `torch_mlir_model_zoo.pdf` §3 의 "빡센 버전"
- 사촌 문서: [`easy.md`](easy.md) (Llama 직접 export), [`easy2.md`](easy2.md) (HF 모델 zoo 입고 기준)
- amdsharktank source 분석 대상:
  - [`amdsharktank/types/theta.py`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/types/theta.py) (IRPA, Dataset, DatasetMetadata)
  - [`amdsharktank/kernels/rotary.py`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/kernels/rotary.py) + [`templates/rotary_embedding.mlir`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/kernels/templates/rotary_embedding.mlir)
  - [`amdsharktank/layers/paged_attention.py`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/paged_attention.py) (`@mlir_kernel` + `iree_linalg_ext.gather`)
  - [`amdsharktank/examples/export_paged_llm_v1.py`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/examples/export_paged_llm_v1.py)
  - [`amdsharktank/utils/cli.py`](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/utils/cli.py) (`--irpa-file`, dtype flags)
- 사전 정리: [`docs/SHARK_AI_ANALYSIS.md`](../SHARK_AI_ANALYSIS.md), [`REPORT-2026-05-26-amd-shark-integration.md`](../../REPORT-2026-05-26-amd-shark-integration.md)
