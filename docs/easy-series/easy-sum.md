# Easy Series 요약 (easy-sum) — 한 장으로 보는 결론

> easy3 / easy4 / easy4-detail / easy5-fig 의 1페이지 압축판.
> 실측: `/tmp/llama32-irpa/llama-3.2-1b.mlir` (27,269 lines) + amdsharktank 소스 직접 대조.

---

## 0. 한 줄 결론

> **amdsharktank 파일을 부분 수정해서는 제약 9개가 다 안 빠진다. `aot.export` 에 넘기는 *최상위 모델 모듈* 을 stateless single-entry 로 교체하는 게 정답** — IRPA/export 파이프라인은 빌려 쓰고 transformer 정의만 우리 `forward` 로 대체. (= 우리 `LlamaOnDevice` 가 이미 한 일.)

---

## 1. export 에 쓰는 클래스는 두 계층

| | (A) export-time | (B) lowering-time |
|---|---|---|
| 정체 | trace 되는 `nn.Module` | `@mlir_kernel` (trace 안 됨) |
| 처리 | `torch.export` → FX 그래프 | 호출 지점에 손수 짠 MLIR `util.func` 삽입 |
| IR | `torch.aten.*` (NPU 친화 ◎) | `iree_linalg_ext.*` / `linalg.generic` (✗) |
| 예 | `PagedLlmModelV1`, `LinearLayer`, `RMSNormLayer`, `FFN`, `PagedGQAttention`, `CachedRotaryLayer` | `RoPEKernels.rope_select_concat`, `KVCacheGatherKernel` |

```
HF safetensors ─import_hf_dataset→ IRPA ─export_paged_llm_v1→ MLIR
  (A) PagedLlmModelV1 → torch.aten.*        ┐ 두 가지가
  (B) @mlir_kernel    → custom util.func    ┘ 섞인 IR
```

IR 에서 NPU 가 막히는 건 딱 4개: `@rope_select_concat`(B) · `@paged_attention_kv_cache_gather`(B, **decode 전용**) · `iree.abi.affinity` · `_scaled_dot_product_flash_attention_for_cpu`(분해 안 된 불투명 SDPA).

---

## 2. 실측이 클래스 구조를 증명 (op count = 16블록 × 함수수)

`aten.mm 226=(16·7+1)·2` · `RMSNorm 66=(16·2+1)·2` · `bmm 64`(RoPE 외적) · `silu 32` · `rope_select_concat 64` · `index_put 64` · `_sdpa 32` · **`gather 0(prefill)+32(decode)`** · `util.global 147`(weights).

핵심 비대칭 2가지:
- **gather(KV read)는 decode 전용** — prefill 은 K/V 새로 계산해 write 만, decode 만 cache 에서 custom gather 로 읽음.
- **SDPA 는 분해 안 된 블랙박스** (`_softmax` 0개).

---

## 3. 제약 9종 → 고칠 코드

| 제약 | 고칠 위치 | 수정 | 종류 |
|---|---|---|---|
| #6 fp16 mixed | `LlamaModelConfig`(llm_configs.py:579) / CLI | dtype 통일 | ✅ 설정 |
| #5 device affinity | `export_paged_llm_v1.py @fxb.export_program(arg_device=)` | `arg_device=` 제거 | ✅ 인자 |
| #7 dynamic seq_len | `export_paged_llm_v1.py:65/161` `torch.export.Dim` | 고정 shape | ✅ 드라이버 |
| #4 RoPE concat | `rotary_embedding_hf.py:390` `select_concat` | `torch.stack(...).flatten` | ⚠️ 커널교체 |
| #3 paged gather | `paged_attention.py:234-235` `kv_cache_gather` | 표준 index_select | ⚠️ 커널교체 |
| #1 KV cache arg | `llm.py:127/182` prefill/decode | cache 인자 제거·재계산 | ❌ forward 재작성 |
| #2 prefill/decode 분리 | `export_paged_llm_v1.py:56/152` | 단일 entry 통합 | ❌ 재작성 |
| #8 multi-args | 위 시그니처 | `(input_ids,)` 하나 | ❌ 재작성 |
| #9 weight 외부참조 | `Theta`/IRPA | — | 🟢 유지(이득) |

→ #3~#7(설정·커널) 5개는 부분 수정으로 빠지지만, **#1·#2·#8 은 `PagedLlmModelV1` 구조 자체**라 한 줄로 안 됨.

---

## 4. 가장 적은 코드로 가장 많이 없애는 한 수

`export_llm_v1(modelClass=PagedLlmModelV1)`(export_paged_llm_v1.py:43) 의 **modelClass 교체**:

```python
class MyOnDeviceLlama(nn.Module):
    def forward(self, input_ids):      # 단일 entry          ← #2,#8
        h = embed(input_ids)
        for blk in self.blocks:        # KV cache 없음(재계산) ← #1
            h = blk(h)                 # 표준 SDPA 분해        ← #3(gather 자체 없음)
        return lm_head(norm(h))        # 표준 cos/sin·mul·cat  ← #4
# example_args = (torch.zeros(1,T,dtype=long),)   ← #7 fixed shape
```

이 한 모듈이 **#1·#2·#3·#4·#7·#8 = 6개를 동시에** 제거. 남는 #5·#6 은 export 설정 2개, #9(IRPA) 는 그대로 재사용.

---

## 5. 우리 프로젝트의 정답 (이미 구현됨)

[`../../src/torch_mlir_zoo/models/llama_on_device.py`](../../src/torch_mlir_zoo/models/llama_on_device.py) 의 `LlamaOnDevice` 가 정확히 그 "교체 모듈". 결과: IR `server_side_op_hits = {}` ([`../../src/torch_mlir_zoo/analysis/ir_summary.py`](../../src/torch_mlir_zoo/analysis/ir_summary.py) 로 검증).

> **고칠 코드 = amdsharktank 파일이 아니라, `aot.export` 에 넘기는 최상위 모델 모듈.**
> amdsharktank 직접 패치는 위 표 ✅/⚠️ 5개가 현실적 한계, ❌ 3개는 어차피 modelClass 재작성이 답.

---

## 6. 문서 지도

| 단계 | 문서 | 한 줄 |
|---|---|---|
| 왜 막히나 | [easy3.md](easy3.md) | 제약 9종 → 1 근본원인 |
| 어떤 클래스로 | [easy4.md](easy4.md) | 클래스 27종 카탈로그 |
| 정밀 (2계층+실측) | [easy4-detail.md](easy4-detail.md) | export-time vs lowering-time + IR 1:1 |
| 그림 | [easy5-fig.md](easy5-fig.md) | [export-vs-lowering.drawio](../diagrams/export-vs-lowering.drawio) |
| 우리 성공 경로 | [easy.md](easy.md) / [easy2.md](easy2.md) | LlamaOnDevice export, 입고 기준 |
</content>
