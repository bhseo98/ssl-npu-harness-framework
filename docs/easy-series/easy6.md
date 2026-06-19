# Easy Guide 6 — input dim(dynamic/static)은 *어느 클래스*가 만드나 + lowering kernel 제약 카탈로그 + "aten만이면 lowering 되나?" 충분조건

> [easy3.md](easy3.md)(제약 9종)·[easy4-detail.md](easy4-detail.md)(export-time vs lowering-time 두 계층)의 후속.
> 이 노트는 그 두 문서를 **코드 레벨로 더 좁혀서 두 질문에 정확히 답**한다.
>
> 1. **input dimension(dynamic / static) 타입은 어느 클래스에서, 어떤 형태로 나오나? "dynamic이라 NPU에 못 쓴다"가 맞나?** → §1
> 2. **lowering 단계에서 제약되는 kernel은 뭐가 있나(RoPE 같은 거)? export 단계의 `nn.Module`은 `torch.aten.*`만 하면 lowering 되나? 제약은 없나?** → §2
>
> ground truth: `/tmp/llama32-irpa/llama-3.2-1b.mlir`(27,269 lines) + amdsharktank 소스 직접 grep
> (`/home/bohyun/amd-shark-ai/amdsharktank/amdsharktank/`, 읽기만).

---

## 1. input dim (dynamic / static) — "모델 클래스"가 아니라 **export 드라이버**가 만든다

### 1.1 핵심 — dim은 모델이 모른다

`PagedLlmModelV1`(모델 nn.Module) 자체는 seq_len이 dynamic인지 static인지 *모른다*. dynamic dim은 **그 위 export 단계**에서 `dynamic_shapes=` 인자로 주입된다. 그래서 "dynamic type이 어느 클래스냐"의 답은 모델 레이어가 아니라 **export 드라이버 + cache 셋업 메서드** 두 군데다.

| dynamic 축 | 만드는 곳 (클래스/함수) | 구문 | IR 흔적 |
|---|---|---|---|
| **seq_len** (`tokens[1]`, `seq_block_ids[1]`) | `examples/export_paged_llm_v1.py:65` `generate_batch_prefill` (드라이버 *함수*, 클래스 아님) | `torch.export.Dim("seq_len_blocks_dim", min=2, max=…)` → `* block_seq_stride` | `vtensor<[1,?],si64>` |
| **page count** (KV slab `[?,524288]`의 0축) | `ServicePagedLlmModelV1.setup_cache()` (`models/llm/export.py:151`) → 내부 `PagedKVCache.allocate()` | `torch.export.Dim("page")` | `tensor<[?,524288],f16>` |

두 `Dim`은 모두 `@fxb.export_program(dynamic_shapes={...})`(`iree.turbine.aot.FxProgramsBuilder`)의 `dynamic_shapes=` kwarg로 전달된다 (export_paged_llm_v1.py:74, 118/141/193).

```python
# export_paged_llm_v1.py:65 — seq_len dynamic dim 생성 (드라이버)
seq_len_blocks_dim = torch.export.Dim("seq_len_blocks_dim", min=2, max=block_dim_max)
seq_len_dim        = seq_len_blocks_dim * llama_config.block_seq_stride
cache, cache_dynamic_shapes, _ = model.setup_cache()    # ← page dim 은 여기서
dynamic_shapes = {
    "tokens":        {1: seq_len_dim},          # axis 1 dynamic
    "seq_lens":      {},                        # static
    "seq_block_ids": {1: seq_len_blocks_dim},   # axis 1 dynamic
    "cs":            cache_dynamic_shapes,       # page dim dynamic
}
```

```python
# models/llm/export.py:151 — ServicePagedLlmModelV1.setup_cache (page dim)
page_dim       = torch.export.Dim("page")
cache_state    = self.model.cache.allocate(page_count=device_block_count)  # PagedKVCache.allocate
dynamic_shapes = [{0: page_dim} for _ in range(len(cache_state.allocation))]
```

### 1.2 타입의 정체 + 형태

- **타입 = `torch.export.Dim`** — PyTorch 표준 심볼릭 dim. NPU가 모르는 *이상한 타입이 아니다*. (특별한 amdsharktank 클래스도 아님.)
- export 시 `import_symbolic_shape_expressions=True`(export_paged_llm_v1.py)로 심볼 식까지 IR에 박힌다 → 함수에 `torch.assume_strict_symbolic_shapes` attribute, 텐서에 `?` 축.
- **dynamic 형태**: `!torch.vtensor<[1,?],si64>`, `!torch.tensor<[?,524288],f16>` — IR 전체 `?` 4,530회.
- **static 형태**: `dynamic_shapes`를 *주지 않고* concrete `example_args`로 trace → `[1,32,…]`처럼 숫자로 고정. (= 우리 `LlamaOnDevice`가 한 것.)

```mlir
// dynamic (amdsharktank export) — ? 가 심볼 축
func.func @prefill_bs1(%arg0: !torch.vtensor<[1,?],si64> ...) -> !torch.vtensor<[1,?,128256],f16>
  attributes {torch.assume_strict_symbolic_shapes}

// static (LlamaOnDevice, fixed example_args) — 숫자로 고정
func.func @main(%arg0: !torch.vtensor<[1,32],si64>) -> !torch.vtensor<[1,32,128256],f16>
```

### 1.3 "dynamic이라 NPU에 못 쓴다"는 *절반만* 맞음

- ❌ `torch.export.Dim`이 NPU가 못 받는 exotic 타입이라서가 **아니다**.
- ✅ **IR이 `?` 심볼 축을 들고 있으면**, static shape를 전제로 DMA/타일링/버퍼 크기를 잡는 NPU 백엔드가 스케줄을 못 짠다 → **specialize(고정) 필요**.
- 그래서 이건 *불가능*이 아니라 **제일 싼 제약**: easy3 #7 / [easy-sum](easy-sum.md) 표에서 ✅ "드라이버 레벨" — `Dim`을 빼고 fixed `example_args`로 trace하면 끝. 백엔드가 dynamic shape를 받으면 그대로도 통과.

> **한 줄**: dynamic dim = export 드라이버(`generate_batch_*`)와 `setup_cache()`가 만든 `torch.export.Dim` → IR의 `?`. 모델 클래스 책임 아님. "dynamic이라 못 쓴다"가 아니라 "static을 요구하는 백엔드면 specialize하라"가 정확. 고치는 비용 최저(설정 1줄).

---

## 2. lowering 단계의 제약 kernel + "aten만이면 lowering 되나?"

### 2.1 제약 kernel = `@mlir_kernel` / `CustomOp.register` (trace 안 되고 raw MLIR splice)

이 함수들은 모델 forward에서 *호출*되지만 FX가 내부를 추적하지 않고 **손으로 짠 MLIR `util.func`를 IR에 그대로 박는다**(easy4-detail §3). 인프라: `kernels/mlir_kernel.py`. 소스 전체 grep으로 모은 **제약 kernel 카탈로그**:

| 그룹 | kernel (정의 위치) | 박히는 IR | default Llama-3.2-1B fp16 export에 등장? |
|---|---|---|---|
| **RoPE** | `RoPEKernels.rope_select_concat` (`layers/rotary_embedding_hf.py:29` `@mlir_kernel`) | `linalg.generic`+`arith.select` (interleave concat) | ✅ **64회** |
| **KV read** | `KVCacheGatherKernel` (`layers/paged_attention.py:68` `@mlir_kernel`) | `iree_linalg_ext.gather` | ✅ **decode 전용 32회** |
| *(불투명, kernel은 아님)* | `aten._scaled_dot_product_flash_attention_for_cpu` | 단일 블랙박스 op (softmax/bmm 안 보임) | ✅ 32회 |
| **quant matmul** | `mmt_block_scaled_q8` · `mmt_block_scaled_offset_q4` · `mmt_super_block_scaled_offset_q4` · `einsum_2args_q4` · `mmtfp` · `batch_matmul_transpose_b` · `gemm_fp4` · `gemm_fp4_asm` (`kernels/`) | custom util.func / asm | ❌ INT8/INT4 켤 때만 |
| **sampling** | `iree_topk` (`kernels/topk.py`) | `iree_linalg_ext.topk` | ❌ `--top-k` 켤 때만 |
| **attention 변종** | `kernels/attention.py`(×2 `@mlir_kernel`), `kernels/wave/attention.py`, `wave/extend_attention.py`, `wave/mxfp4_gemm.py` | wave/asm 커널 | ❌ 해당 kernel flag |
| **vision/기타** | `conv_2d_nchw_fchw` · `pooling_nchw_sum` · `bitcast_to_complex/real` · `kernels/rotary.py::apply_rotary_embedding`(비-HF rope 경로) | custom util.func | ❌ 경로별 |

> **쉽게**: 기본 Llama 경로에서 실제로 NPU를 막는 lowering kernel은 딱 **RoPE concat + KV-gather 2개**(+ SDPA 블랙박스). 나머지(quant·topk·wave·conv·bitcast)는 *해당 기능을 켤 때만* 추가로 튀어나오는 **예비 제약 풀**이다. easy3는 #3(paged gather)·#4(rope) 둘로 요약했는데, 그게 곧 이 2개.

### 2.2 export 단계 `nn.Module`은 `torch.aten.*`만 하면 lowering 되나? — *거의* 맞지만 충분조건 아님

`nn.Module`이 전부 `torch.aten.*`(+`torch_c`,`util.global`)로 trace되면 torch-mlir 표준 파이프라인이 내린다. 그게 `LlamaOnDevice`의 원리. **하지만 "nn.Module이다"만으론 부족** — 세 가지 누수:

| # | 누수 | 무엇이 문제 | 회피 |
|---|---|---|---|
| 1 | **불투명 composite aten** | `F.scaled_dot_product_attention` → `aten._scaled_dot_product_flash_attention_for_cpu` 한 덩어리. `torch.aten.*`이긴 하나 softmax/bmm로 *분해 안 됨* → NPU가 이 op을 알아야 통과 | `attn_implementation="eager"` 또는 decomposed SDPA로 풀어 씀 (easy4-detail 레버 ④) |
| 2 | **forward가 custom op 호출** | RoPE `select_concat`, `kv_cache_gather`, quant matmul 등 `@mlir_kernel`/`CustomOp`를 부르면 trace가 아니라 util.func splice → aten 아님 | forward가 *부르는 op이 전부 순수 aten*이어야 함 (custom kernel 회피) |
| 3 | **aten 밖으로 새는 패턴** | data-dependent shape(`.item()`, page_ids gather) · python 제어흐름 · external mutable buffer in-place(`index_put_` on cache arg) · `iree.abi.affinity` 메타 | stateless 재계산 forward + single device + static-ish shape |

```
"nn.Module이면 lowering" 의 정확한 충분조건 =
   (a) 순수 표준 aten으로만 구성        ← custom kernel 회피(2)
 + (b) 불투명 composite(SDPA) 분해       ← (1)
 + (c) data-dependent/external-state 없음 ← (3)
 + (d) static-ish shape (§1)
```

> **한 줄**: "nn.Module → aten → lowering"은 맞는 방향이지만, **불투명 SDPA 분해 + custom kernel 회피 + 비-aten 누수 제거 + (가능하면) static shape**가 같이 모여야 실제로 NPU까지 내려간다. 이 4개를 한 번에 만족시킨 게 `LlamaOnDevice`(IR `server_side_op_hits = {}`).

---

## 3. cheat sheet (재현)

```bash
ASK=/home/bohyun/amd-shark-ai/amdsharktank/amdsharktank
M=/tmp/llama32-irpa/llama-3.2-1b.mlir

# Q1 — dynamic dim 을 만드는 곳 (모델 아님, 드라이버/cache)
grep -n "torch.export.Dim\|dynamic_shapes\|setup_cache" $ASK/examples/export_paged_llm_v1.py
sed -n '151,165p' $ASK/models/llm/export.py        # ServicePagedLlmModelV1.setup_cache → page_dim
grep -c "vtensor<\[1,\?" $M                          # ? 심볼 축 등장

# Q2 — 제약 lowering kernel 전체 카탈로그
grep -rn "@mlir_kernel\|CustomOp.register" $ASK | grep -v test
# default Llama 경로에서 실제 박힌 2개 + SDPA 블랙박스:
grep -c "util.call @rope_select_concat" $M                       # 64
grep -nF "util.call @paged_attention_kv_cache_gather" $M | \
  awk -F: '$1<14154{p++}$1>=14154{d++}END{print "gather p/d:",p+0,d+0}'  # 0 32 (decode 전용)
grep -c "_scaled_dot_product_flash_attention_for_cpu" $M          # 32 (그리고 _softmax 0)
```

---

## 4. 참고

- 짝 문서: [easy3.md](easy3.md)(제약 9종 → 1 근본원인) · [easy4-detail.md](easy4-detail.md)(export-time vs lowering-time 2계층 + IR 1:1) · [easy-sum.md](easy-sum.md)(한 장 요약)
- 우리 정답 경로: [`../../src/torch_mlir_zoo/models/llama_on_device.py`](../../src/torch_mlir_zoo/models/llama_on_device.py) (§1.3·§2.2의 4조건을 모두 만족), 검증 [`../../src/torch_mlir_zoo/analysis/ir_summary.py`](../../src/torch_mlir_zoo/analysis/ir_summary.py)
- amdsharktank 소스(읽기만): `examples/export_paged_llm_v1.py`(dynamic_shapes), `models/llm/export.py`(setup_cache), `layers/rotary_embedding_hf.py`·`layers/paged_attention.py`(2개 kernel), `kernels/`(예비 제약 풀)
