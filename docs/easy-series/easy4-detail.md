# Easy Guide 4-detail — export 에 쓰는 클래스 vs MLIR lowering 에 쓰는 클래스, 그리고 실측 IR 1:1 매핑

> [easy4.md](easy4.md)(클래스 27종 카탈로그) 의 정밀판. 이 문서는 **두 질문에 코드 레벨로 답**한다:
> 1. *export 할 때 어떤 클래스가 쓰이나* (= FX 그래프로 추적돼 `torch.aten.*` 가 되는 nn.Module)
> 2. *amdsharktank 가 MLIR lowering 할 때 어떤 클래스가 쓰이나* (= 추적되지 않고 손수 짠 MLIR `util.func` 를 IR 에 삽입하는 `@mlir_kernel`/`CustomOp`)
>
> 그리고 그 클래스들을 **실측 `llama-3.2-1b.mlir` (27,269 lines)** 의 op 와 1:1 로 맞춰 셈이 맞는지 검증한다. 그림판은 [easy5-fig.md](easy5-fig.md) + [`../diagrams/export-vs-lowering.drawio`](../diagrams/export-vs-lowering.drawio).

---

## 0. 범위 + ground truth

- 대상: 순수 `meta-llama/Llama-3.2-1B-Instruct` → `import_hf_dataset` → `export_paged_llm_v1` (easy3/easy4 와 동일 산출물). on-device 경로(`LlamaOnDevice`)는 §5 의 대조용으로만.
- ground truth 파일 (실재, 재현됨):
  - `/tmp/llama32-irpa/llama-3.2-1b.mlir` — **27,269 lines**, entry `@prefill_bs1`(line 151) + `@decode_bs1`(line 14154)
  - `/tmp/llama32-irpa/llama-3.2-1b.json` — `{block_count:16, attn_head_dim:64, head_count_kv:8, block_seq_stride:32, kv slab elem:524288, max_seq_len:131072, top_k:null, logits_normalization:"none"}`
- amdsharktank 소스: `/home/bohyun/amd-shark-ai/amdsharktank/amdsharktank/` (sibling clone, Apache-2.0, *import 안 하고 읽기만*).

---

## 1. 핵심 개념 — export 에 관여하는 클래스는 *두 계층*이다

amdsharktank export 는 한 종류의 클래스가 아니라, **성격이 다른 두 계층**이 협동한다. 이걸 구분하는 게 이 문서의 골자다.

| | (A) export-time 클래스 | (B) lowering-time 클래스 |
|---|---|---|
| 정체 | `torch.nn.Module` 서브클래스 (모델 정의) | `@mlir_kernel` / `CustomOp` 데코레이트 함수 |
| 처리 | `torch.export` 가 **FX 그래프로 trace** | trace 안 됨. 호출 지점에 **손수 짠 MLIR 템플릿을 직접 emit** |
| IR 결과 | `torch.aten.*` (+ `torch_c`, `util.global`) | 커스텀 `util.func` (`iree_linalg_ext.*`, `linalg.generic`) |
| 예 | `PagedLlmModelV1`, `LinearLayer`, `RMSNormLayer`, `FFN`, `PagedGQAttention`, `CachedRotaryLayer` | `RoPEKernels.rope_select_concat`, `KVCacheGatherKernel` |
| NPU 적합 | 대체로 ◎ (표준 op) | ✗ (NPU 가 모르는 dialect/template) |

> 즉 "export 에 쓰는 클래스"는 (A) 모델 그래프, "MLIR lowering 에 쓰는 클래스"는 (B) 커널 삽입기. **최종 IR 은 (A)의 torch.aten + (B)의 custom util.func 가 섞인 결과물.** NPU porting 의 난점은 거의 (B) 와, (A) 중 SDPA·affinity 같은 불투명 부분에서 온다.

---

## 2. (A) export-time 클래스 — 호출 체인 (코드 레벨)

진입점 `examples/export_paged_llm_v1.py`. 순서대로.

### 2.1 설정 조립 — `main()` (export_paged_llm_v1.py:237)

```python
dataset      = cli.get_input_dataset(args)                       # = Dataset.load(irpa)  → Theta
export_config= ExportConfig(bs_prefill=[1], bs_decode=[1], ...)  # config.py:45
llama_config = LlamaModelConfig.from_dataset(dataset, **dtype_flags)  # llm_configs.py:707
parallelism_config = ParallelismConfig.default_config(tp=1, pp=1)    # single device
```
- `LlamaHParams`(llm_configs.py:56) = 아키텍처 상수 (block_count 16, head 32, kv-head 8, ffn 8192 …) — IRPA properties 에서 자동 로드.
- `LlamaModelConfig`(llm_configs.py:562) = hp + **lowering 정책** (`kv_cache_type="paged"`, `block_seq_stride=32`, `activation/attention_dtype=f16`, `attention_kernel`, `matmul_kernel`). → easy3 제약 #6/#7 의 default 가 여기서 결정.
- `ExportConfig`(config.py:45) = export 행위 (bs, top_k, logits_normalization, skip_prefill/decode).

### 2.2 모델 그래프 조립 — `PagedLlmModelV1.__init__` (llm.py:68)

`theta(...)` 이름경로로 weight 를 집어 서브모듈 트리를 짠다:

```python
self.cache = build_cache_from_config(config)                       # → DefaultPagedKVCache
self.add_module("token_embedding", TokenEmbeddingLayer(theta("token_embd"), ...))
self.attention_embedding = build_rotary_layer(...)                  # → CachedRotaryLayer
self.add_module("output_norm",  RMSNormLayer(theta("output_norm"), ...))
self.add_module("output_lm_head", LinearLayer(theta("output"), ...))   # ← tied embedding 필요 (easy4 §1.4)
self.attn_blocks = nn.ModuleList([AttentionFFNBlock(theta("blk", n), ...) for n in range(16)])
```

`forward` 가 아니라 **`prefill()`(llm.py:127) / `decode()`(llm.py:182) 두 메서드**:
```python
def prefill(self, tokens, *, seq_lens, seq_block_ids, cache_state, start_positions=None):
    h = self.token_embedding(tokens)
    for block in self.attn_blocks:
        h = block(h, embedding=self.attention_embedding, start_positions=..., seq_lens=..., cache_state=..., seq_block_ids=...)
    h = self.output_norm(h); logits = self.output_lm_head(h)
    return logits.to(torch.float16)
```
→ 인자 `tokens, seq_lens, seq_block_ids, cache_state` 가 그대로 `@prefill_bs1` 함수 인자가 됨 (제약 #2/#8 의 코드 출처).

### 2.3 블록 forward — `PagedLlamaAttentionBlock.forward` (paged_llama_attention_block.py:161)

```python
def forward(self, h, *, embedding, seq_block_ids, seq_lens, start_positions, cache_state):
    x = self.attn_norm(h)                                          # RMSNormLayer
    xq, xk, xv = self.pre_process_attention(x, embedding, start_positions)  # §2.4
    is_decode = (h.shape[1] == 1)
    attn_fn = self.paged_attention.forward_decode if is_decode else forward_prefill
    attn_output = attn_fn(q=xq, k=xk, v=xv, cache_state=..., seq_block_ids=..., ...)  # §2.5
    attn_output = attn_output.transpose(1,2).flatten(2,3)
    attn_output = self.attn_output(attn_output)                    # LinearLayer (o-proj)
    h = h + attn_output                                            # residual
    return h
```
그리고 `AttentionFFNBlock.forward`(llm.py:407) 가 이어서 `ffn_norm(h)` + `FFN(SwiGLU)` + residual.

`pre_process_attention`(paged_llama_attention_block.py:365):
```python
xq, xk, xv = self._project_qkv(x)          # LinearLayer q/k/v → view to [bs,sl,heads,head_dim]
if self.use_rope:
    xq = embedding.forward(xt=xq, start_positions=start_positions)   # CachedRotaryLayer
    xk = embedding.forward(xt=xk, start_positions=start_positions)
```

### 2.4 RoPE — `CachedRotaryLayer` / `RotaryEmbeddingLayer` (★ easy3 정정 포인트)

`RotaryEmbeddingLayer.forward`(rotary_embedding_hf.py:300) 의 실제 수식:
```python
x_real = x[..., :head_dim//2];  x_imag = x[..., head_dim//2:]   # (interleaved=False 경로)
x1 = x_real*cos - x_imag*sin                                    # ← 전부 표준 aten.mul/sub
x2 = x_imag*cos + x_real*sin                                    # ← 전부 표준 aten.mul/add
cated = select_concat(x1, x2)                                   # ← (B) 커스텀 커널! §3.2
cated = cated.flatten(-2, -1)
```

> **easy3 정정**: easy3 는 RoPE 커스텀 op 을 `kernels/rotary.py::apply_rotary_embedding`(math.cos/sin 템플릿) 으로 봤는데, **실제 export 된 그래프가 쓰는 건 `rotary_embedding_hf.py::RoPEKernels.rope_select_concat`** 다. cos/sin·곱·합은 *표준 `torch.aten`* (그래서 IR 에 `cos`/`sin` 각 64개 + `bmm` 64개(positions×inv_freq 외적)) 이고, **커스텀은 interleave-pack `concat` 한 조각뿐**. 즉 RoPE 는 easy3 인상보다 *덜* 커스텀하다.

### 2.5 Attention — `PagedGQAttention` (paged_attention.py:949) + KV cache (148)

`forward_prefill`(:835): **write 후 attention** (read 없음, prefill 은 fresh K/V):
```python
self.write(cache_state, cache_partitions=[k,v], ...)    # DefaultPagedKVCache.write → index_copy_
return self.paged_attention(q,k,v, start_positions=None, ...)  # start_positions None → read 생략
```
`forward_decode`(:789): **write_timestep + read(gather) + attention**:
```python
self.write_timestep(cache_state, [k,v], seq_positions=start_positions, ...)  # → index_put_
return self.paged_attention(..., start_positions=start_positions)            # → read() 수행
```
`paged_attention`(:880):
```python
if start_positions is not None:                          # = decode 일 때만
    k, v = self.read(cache_state, ...)                   # DefaultPagedKVCache.read → kv_cache_gather (§3.3)
mask = create_attention_mask(...)                        # create_input_mask + causal
return self.attention(q,k,v, mask=mask, attention_kernel=..., ...)
```
`PagedGQAttention.attention`(:950) = GQA expand(kv-head 8→32) → `super().attention` →
`PagedMHAttention.attention`(:741): `ops.scaled_dot_product_attention(q,k,v,a=mask, impl=attention_kernel)` — **이게 IR 에서 단일 불투명 op** (§4).

### 2.6 서비스 래핑 + export 엔진

```python
model = ServicePagedLlmModelV1(model, export_config)     # export.py:47 — prefill/decode + top_k + setup_cache/arg_devices
fxb = FxProgramsBuilder(model)                            # iree.turbine.aot
@fxb.export_program(name="prefill_bs1", args=(tokens, seq_lens, seq_block_ids, cache),
    dynamic_shapes={"tokens":{1:Dim}, ...}, arg_device=arg_devices, strict=False)  # export_paged_llm_v1.py:138
def _(model, tokens, seq_lens, seq_block_ids, cs):
    return model.prefill(tokens, None, seq_lens, seq_block_ids, CacheAllocation(cs))
# decode_bs1 동일 패턴(:184)
output = export(fxb, import_symbolic_shape_expressions=True)   # FX → MLIR (:232)
output.save_mlir(...)                                          # → 27,269 lines (:323)
```
- `setup_arg_devices`(export.py:137) 가 각 인자에 `DeviceAffinity` → IR 의 `iree.abi.affinity`(제약 #5).
- `torch.export.Dim`(export_paged_llm_v1.py:65) → `vtensor<[1,?]>` / slab `[?,524288]`(제약 #7).

---

## 3. (B) lowering-time 클래스 — `@mlir_kernel` 가 raw MLIR 를 IR 에 박는다

이 클래스들은 모델 forward 에서 *호출*되지만, FX 가 그 내부를 추적하는 게 아니라 **`MLIRSpec` 템플릿을 출력 IR 에 그대로 splice** 한다. (인프라: `kernels/mlir_kernel.py` 의 `@mlir_kernel`, `MLIRSpec`, `KernelBuilder`, `inline_template_function`, `CustomOp`.)

### 3.1 인프라 — `@mlir_kernel`

`@mlir_kernel(inputs=(MLIRTensor[...]), results=(...))` 데코레이터가 붙은 함수는 `MLIRSpec(mlir_text)` 를 반환 → 호출 시 그 텍스트가 specialize 되어 `util.func private @<name>` 로 IR 에 추가되고, 호출부엔 `util.call` 이 박힌다.

### 3.2 `RoPEKernels.rope_select_concat` (rotary_embedding_hf.py:20)

`select_concat(x1,x2)`(§2.4) 가 호출하는 커널. 주석에 *"IREE 에 fusion 가능한 concat op 이 없어서 `linalg.generic` + `arith.select` 로 concat 한다"* 고 명시.

실측 IR (llama-3.2-1b.mlir:27235):
```mlir
util.func private @rope_select_concat_..._f16(%arg0: tensor<?x?x?x?xf16>, %arg1: tensor<?x?x?x?xf16>) -> tensor<?x?x?x2x?xf16> {
  %1 = linalg.generic {indexing_maps=[#map,#map,#map1], iterator_types=[parallel ×5]}
       ins(%arg0,%arg1) outs(%0) {
  ^bb0(%in,%in_3,%out):
    %2 = linalg.index 3 : index
    %3 = arith.cmpi eq, %2, %c0 : index
    %4 = arith.select %3, %in, %in_3 : f16     // two 축 0이면 x1, 1이면 x2
    linalg.yield %4
  }
}
```
- 호출: **prefill 32 + decode 32 = 64** (16블록 × {q,k} × 2함수).
- NPU 관점: `linalg.generic` 자체는 표준이지만 amdsharktank 가 register 한 외부 util.func 형태 → 그 패스를 모르는 컴파일러엔 unknown. (제약 #4)

### 3.3 `KVCacheGatherKernel` (paged_attention.py:55) → `iree_linalg_ext.gather`

`DefaultPagedKVCache.read`(paged_attention.py:209) 가 호출:
```python
key   = kv_cache_gather(page_table, page_ids, t_id, key_p_id=0)
value = kv_cache_gather(page_table, page_ids, t_id, value_p_id=1)
```
실측 IR (llama-3.2-1b.mlir:27254):
```mlir
util.func private @paged_attention_kv_cache_gather_..._HEAD_COUNT_KV_8_BLOCK_SEQ_STRIDE_32_ATTN_HEAD_DIM_64_f16(
    %arg0: tensor<?x16x2x8x32x64xf16>, %arg1: tensor<?x?xi64>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<?x?x8x32x64xf16> {
  %extracted_slice = tensor.extract_slice %arg0[0,%t_id,%p_id,0,0,0] [%dim,1,1,8,32,64] [1,...]  // 현재 block/partition
  %3 = iree_linalg_ext.gather dimension_map=[0] ins(%extracted_slice, %arg1) outs(%2)            // page_ids 로 gather
}
```
- 호출: **prefill 0 + decode 32 = 32** ← **decode 전용!** (prefill 은 read 안 함, §2.5)
- NPU 관점: `iree_linalg_ext.gather` 는 IREE 전용 extension dialect + data-dependent gather → 가장 막히는 지점 (제약 #3).

> **이게 두 계층의 핵심 차이**: (A) `DefaultPagedKVCache.write` 는 `ops.index_copy_/index_put_` 같은 **표준 aten** 으로 내려가지만, (A) `read` 는 (B) `KVCacheGatherKernel` 을 호출해 **커스텀 IREE op** 을 박는다. 같은 cache 클래스 안에서도 write 는 표준, read 는 커스텀.

---

## 4. 실측 1:1 매핑 — count 가 클래스 구조를 증명한다

`@prefill_bs1` / `@decode_bs1` 의 op 를 클래스에 매핑. **count = 16블록 × 함수수 산술로 정확히 분해**된다 (검증: [easy5-fig.md](easy5-fig.md) §부록 명령).

| 클래스 / 메서드 | IR op | line@(prefill) | prefill | decode | 합 | 분해 |
|---|---|---|---:|---:|---:|---|
| `TokenEmbeddingLayer` | `aten.embedding` | 510 | 1 | 1 | **2** | 1×2함수 |
| `RMSNormLayer` | `pow·mean·rsqrt` | 518/524/530 | 33 | 33 | **66** | (16×2+1)×2 |
| `LinearLayer`/`FFN` | `aten.mm` | 553 | 113 | 113 | **226** | (16×7+1)×2 |
| `RotaryEmbeddingLayer` (sincos) | `bmm·cos·sin` | 703/714/721 | 32 | 32 | **64**ea | (16×2)×2 |
| `RoPEKernels.rope_select_concat` 🟨 | `util.call @rope_select_concat` | 773 | 32 | 32 | **64** | (16×2)×2 |
| KV write (`DefaultPagedKVCache.write`) 🟨 | `aten.index_put` | 1015 | 32 | 32 | **64** | (16×2)×2 |
| `PagedGQAttention.attention` 🟪 | `torch.operator _sdpa_flash_attention_for_cpu` | 1243 | 16 | 16 | **32** | 16×2 |
| `FFN` (SwiGLU) | `aten.silu` | 1324 | 16 | 16 | **32** | 16×2 |
| KV read (`DefaultPagedKVCache.read`→`KVCacheGatherKernel`) 🟥 | `util.call @paged_attention_kv_cache_gather`→`iree_linalg_ext.gather` | — | **0** | 32 | **32** | decode 전용 |
| lm_head (`LinearLayer`) | `aten.mm` | 14145 | 1 | 1 | (226에 포함) | |
| weights (`Theta`/IRPA) | `util.global` | — | — | — | **147** | 16×9 + 임베드/노름 |

### 4.1 두 가지 비대칭 (가장 중요한 발견)

1. **gather(KV read)는 decode 전용** (prefill 0 / decode 32). prefill 은 K/V 를 *새로 계산해 write 만* 하고, decode 만 cache 에서 *읽는다(custom gather)*. → "PagedAttention" 은 한 덩어리가 아니라 *write(표준)·read(커스텀, decode)·SDPA(불투명)* 세 조각.
2. **SDPA 는 분해되지 않은 불투명 op**: `torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"` (line 1243). 전 모듈 `_softmax` **0개**, `bmm` 64개는 전부 RoPE 외적(attention 아님). 즉 attention 의 softmax/matmul 이 IR 에 *안 보이고* flash-attention 블랙박스로 남음 → NPU 컴파일러가 이 op 의 의미를 알아야 통과 (제약 추가 surface). 레버 ④ 로 분해 가능.

---

## 5. 커스터마이징 — 어디를 잡고 흔드나 (코드 레벨 seam)

가장 큰 seam: `export_llm_v1(modelClass: BaseCausalLMModel = PagedLlmModelV1)` (export_paged_llm_v1.py:43) — **modelClass 인자만 갈아끼우면 그래프 전체 교체**.

| 레버 | 코드 진입점 | 난이도 | 효과 (easy3 제약) |
|---|---|---|---|
| ① ExportConfig flags | `config.py:45` / CLI (`--bs-prefill`, `--top-k`, `--skip-decode`) | 낮음 | entry 수·sampling 제어 (#2 부분) |
| ② LlamaModelConfig | `llm_configs.py:562` (`kv_cache_type`, dtype, `attention/matmul_kernel`, tp=pp=1) | 낮음 | dtype 통일 #6, single-device #5 |
| ③ `@mlir_kernel` 템플릿 교체 | `rotary_embedding_hf.py:36` / `paged_attention.py:68` | 중 | 커스텀 util.func 제거 #3,#4 |
| ④ attention_kernel impl | `ops.scaled_dot_product_attention(impl=...)` (`"torch"`/`"decomposed"`) | 중 | SDPA 불투명 op 분해 (🟪 surface) |
| ⑤ **modelClass 재작성** | `export_llm_v1(modelClass=...)` = 우리 `LlamaOnDevice` 패턴 | 높음(~170 LoC) | #1,#2,#3,#4,#7,#8 한 번에 추방. IRPA(#9) 재사용 |

### 5.1 레버 ⑤ 의 정체 = 우리가 이미 한 일

`LlamaOnDevice`([../../src/torch_mlir_zoo/models/llama_on_device.py](../../src/torch_mlir_zoo/models/llama_on_device.py)) 가 정확히 레버 ⑤ 다 — `PagedLlmModelV1` 을 stateless single-entry 로 재구현:
- `prefill`/`decode` 두 메서드 → `forward(input_ids)->logits` 하나
- `DefaultPagedKVCache` (slab arg) → KV cache 제거 (forward 마다 재계산)
- `KVCacheGatherKernel`/`RoPEKernels` (custom util.func) → 표준 `torch.aten.*` (precomputed cos/sin + mul/add/cat)
- `torch.export.Dim` dynamic → example_args 로 fixed shape
- `DeviceAffinity` → 없음

그 결과 IR 에서 `server_side_op_hits = {}` (검증: [`../../src/torch_mlir_zoo/analysis/ir_summary.py`](../../src/torch_mlir_zoo/analysis/ir_summary.py)). **IRPA(#9)는 둘 다 동일하게 쓸 수 있다** — weight 외부화는 NPU 에도 유익.

### 5.2 단계적 권장

> ①②④ (옵션 튜닝, 빠름, 부분 개선) → ③ (커널 수술) → ⑤ (modelClass 재작성, 근본 해소). NPU 가 `iree_linalg_ext`/flash-attention 을 어디까지 받는지에 따라 ③④ 로 충분할 수도, ⑤ 까지 가야 할 수도. 우리 프로젝트는 ⑤ 를 택했다(= `LlamaOnDevice`).

---

## 6. cheat sheet — 클래스↔IR 검증

```bash
M=/tmp/llama32-irpa/llama-3.2-1b.mlir
ASK=/home/bohyun/amd-shark-ai/amdsharktank/amdsharktank

# (A) export-time 클래스 정의
sed -n '127,236p' $ASK/models/llm/llm.py                 # PagedLlmModelV1.prefill/decode
sed -n '161,245p'  $ASK/layers/paged_llama_attention_block.py  # block.forward
sed -n '300,397p'  $ASK/layers/rotary_embedding_hf.py    # RoPE forward (x1/x2 + select_concat)
sed -n '741,1001p' $ASK/layers/paged_attention.py        # PagedMHAttention/GQAttention

# (B) lowering-time 커널 정의
sed -n '20,86p'    $ASK/layers/rotary_embedding_hf.py    # RoPEKernels.rope_select_concat
sed -n '55,135p'   $ASK/layers/paged_attention.py        # KVCacheGatherKernel

# 실측 count (§4 표 재현)
grep -c "torch.aten.mm" $M                                # 226
grep -c "_scaled_dot_product_flash_attention_for_cpu" $M  # 32   (그리고 _softmax 0)
grep -nF "util.call @paged_attention_kv_cache_gather" $M | awk -F: '$1<14154{p++}$1>=14154{d++}END{print "gather p/d:",p+0,d+0}'  # 0 32
```

## 7. 참고
- 그림판: [easy5-fig.md](easy5-fig.md), 원본 도식 [`../diagrams/export-vs-lowering.drawio`](../diagrams/export-vs-lowering.drawio)
- 짝: [easy4.md](easy4.md)(클래스 카탈로그), [easy3.md](easy3.md)(제약 9종), [easy.md](easy.md)/[easy2.md](easy2.md)
- amdsharktank source 핵심 파일:
  - export 엔진: [examples/export_paged_llm_v1.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/examples/export_paged_llm_v1.py)
  - 모델: [models/llm/llm.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/models/llm/llm.py), [layers/paged_llama_attention_block.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/paged_llama_attention_block.py)
  - attention/cache: [layers/paged_attention.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/paged_attention.py)
  - RoPE: [layers/rotary_embedding_hf.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/rotary_embedding_hf.py)
  - layers: [norm.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/norm.py), [linear.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/linear.py), [ffn_block.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/ffn_block.py), [token_embedding.py](https://github.com/nod-ai/amd-shark-ai/blob/main/amdsharktank/amdsharktank/layers/token_embedding.py)
</content>
